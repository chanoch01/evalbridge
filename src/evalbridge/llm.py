"""An LLM-as-judge over HTTP, configured from environment variables.

Two wire formats are supported, chosen by ``EVALBRIDGE_PROVIDER``:

* ``openai`` (default): any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI, Azure
  OpenAI's v1 endpoint, OpenRouter, Together, vLLM, Ollama, ...)
* ``anthropic``: the Anthropic Messages API

Every call is cached on disk in SQLite, keyed by prompt version and text hash, provider, model,
question id, turn, the models in the order shown, and a hash of the texts. Re-running a report
therefore costs nothing, and editing the prompt or the data invalidates only what changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import httpx

from evalbridge.judges import ItemRef, Verdict

RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504, 529}
DEFAULT_PROMPT = "pairwise_v1"


class JudgeError(RuntimeError):
    """The judge could not produce a verdict (HTTP failure after retries, bad config, ...)."""


class VerdictParseError(JudgeError):
    """The model replied, but not with exactly one verdict token."""


# --- Prompt --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    user_template: str

    @property
    def sha(self) -> str:
        text = self.system + "\0" + self.user_template
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def render(self, question: str, answer_a: str, answer_b: str) -> str:
        return (
            self.user_template.replace("{{question}}", question)
            .replace("{{answer_a}}", answer_a)
            .replace("{{answer_b}}", answer_b)
        )


def load_prompt(version: str = DEFAULT_PROMPT) -> Prompt:
    """Load ``src/evalbridge/prompts/<version>.md`` (sections marked by HTML comments)."""
    text = (resources.files("evalbridge") / "prompts" / f"{version}.md").read_text("utf-8")
    m = re.fullmatch(r"\s*<!-- system -->\n(.*?)<!-- user -->\n(.*)", text, flags=re.S)
    if not m:
        raise ValueError(f"prompt {version} needs '<!-- system -->' then '<!-- user -->'")
    system, user = m.group(1).strip(), m.group(2).strip()
    for slot in ("{{question}}", "{{answer_a}}", "{{answer_b}}"):
        if slot not in user:
            raise ValueError(f"prompt {version} user section lacks {slot}")
    return Prompt(version=version, system=system, user_template=user)


_TOKEN = re.compile(r"\[\[([ABC])\]\]")


def parse_verdict(text: str) -> Verdict:
    """Strictly parse ``[[A]]``, ``[[B]]`` or ``[[C]]`` (tie).

    The last non-empty line must hold exactly one token, and no different token may appear
    anywhere in the reply. Anything else raises ``VerdictParseError``; guessing is not allowed.
    """
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        raise VerdictParseError("empty reply")
    last = _TOKEN.findall(lines[-1])
    if len(last) != 1:
        raise VerdictParseError(f"last line must hold exactly one verdict token: {lines[-1]!r}")
    if set(_TOKEN.findall(text)) != set(last):
        raise VerdictParseError("reply contains conflicting verdict tokens")
    return {"A": "A", "B": "B", "C": "tie"}[last[0]]


# --- Cache ---------------------------------------------------------------------------------


class JudgmentCache:
    """SQLite cache of raw judge replies. One row per (prompt, model, item, order)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS judgments (
                key TEXT PRIMARY KEY, prompt_version TEXT, model TEXT, question_id INTEGER,
                turn INTEGER, model_a TEXT, model_b TEXT, raw TEXT, created_at REAL)"""
        )
        self._db.commit()

    def get(self, key: str) -> str | None:
        row = self._db.execute("SELECT raw FROM judgments WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, raw: str, *, prompt_version: str, model: str, ref: ItemRef | None):
        r = ref or ItemRef(-1, -1, "", "")
        values = (key, prompt_version, model, r.question_id, r.turn, r.model_a, r.model_b, raw)
        self._db.execute(
            "INSERT OR REPLACE INTO judgments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*values, time.time()),
        )
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]

    def close(self) -> None:
        self._db.close()


# --- Judge ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_key: str
    provider: str = "openai"
    api_base: str | None = None
    max_tokens: int = 1024
    timeout_s: float = 120.0

    @property
    def base(self) -> str:
        default = "https://api.anthropic.com" if self.provider == "anthropic" else None
        default = default or "https://api.openai.com/v1"
        return (self.api_base or default).rstrip("/")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> LLMConfig:
        env = dict(os.environ if env is None else env)
        missing = [k for k in ("EVALBRIDGE_MODEL", "EVALBRIDGE_API_KEY") if not env.get(k)]
        if missing:
            raise JudgeError(f"set {', '.join(missing)} (see .env.example)")
        provider = env.get("EVALBRIDGE_PROVIDER", "openai").lower()
        if provider not in ("openai", "anthropic"):
            raise JudgeError(f"EVALBRIDGE_PROVIDER must be openai or anthropic, not {provider!r}")
        return cls(
            model=env["EVALBRIDGE_MODEL"],
            api_key=env["EVALBRIDGE_API_KEY"],
            provider=provider,
            api_base=env.get("EVALBRIDGE_API_BASE") or None,
            max_tokens=int(env.get("EVALBRIDGE_MAX_TOKENS", 1024)),
            timeout_s=float(env.get("EVALBRIDGE_TIMEOUT_S", 120)),
        )


class LLMJudge:
    def __init__(
        self,
        config: LLMConfig,
        prompt: Prompt | None = None,
        cache: JudgmentCache | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 5,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
    ):
        self.config = config
        self.prompt = prompt or load_prompt()
        self.cache = cache
        self.name = f"llm-{config.model}"
        self.calls = 0  # HTTP calls actually made (cache misses, including retries)
        self._sleep = sleep
        self._max_retries = max_retries
        self._backoff = (backoff_base_s, backoff_max_s)
        self._http = httpx.Client(timeout=config.timeout_s, transport=transport)

    @classmethod
    def from_env(cls, cache_dir: str | Path | None = None, **kwargs) -> LLMJudge:
        cache_dir = Path(cache_dir or os.environ.get("EVALBRIDGE_CACHE_DIR", ".cache/evalbridge"))
        prompt = load_prompt(os.environ.get("EVALBRIDGE_PROMPT", DEFAULT_PROMPT))
        cache = JudgmentCache(cache_dir / "judgments.sqlite")
        return cls(LLMConfig.from_env(), prompt, cache, **kwargs)

    def cache_key(self, question: str, answer_a: str, answer_b: str, ref: ItemRef | None) -> str:
        content = hashlib.sha256("\0".join((question, answer_a, answer_b)).encode()).hexdigest()
        ident = [ref.question_id, ref.turn, ref.model_a, ref.model_b] if ref else None
        parts = [self.prompt.version, self.prompt.sha, self.config.provider, self.config.model]
        return hashlib.sha256(json.dumps([*parts, ident, content]).encode()).hexdigest()

    def judge(self, question, answer_a, answer_b, *, ref: ItemRef | None = None) -> Verdict:
        key = self.cache_key(question, answer_a, answer_b, ref)
        raw = self.cache.get(key) if self.cache is not None else None
        if raw is None:
            raw = self._complete(self.prompt.render(question, answer_a, answer_b))
            if self.cache is not None:
                self.cache.put(
                    key, raw, prompt_version=self.prompt.version, model=self.config.model, ref=ref
                )
        return parse_verdict(raw)

    # -- HTTP --

    def _request(self, user: str) -> tuple[str, dict, dict]:
        c = self.config
        if c.provider == "anthropic":
            headers = {"x-api-key": c.api_key, "anthropic-version": "2023-06-01"}
            body = {
                "model": c.model,
                "max_tokens": c.max_tokens,
                "temperature": 0,
                "system": self.prompt.system,
                "messages": [{"role": "user", "content": user}],
            }
            return f"{c.base}/v1/messages", headers, body
        headers = {"Authorization": f"Bearer {c.api_key}"}
        body = {
            "model": c.model,
            "max_tokens": c.max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": self.prompt.system},
                {"role": "user", "content": user},
            ],
        }
        return f"{c.base}/chat/completions", headers, body

    def _extract(self, payload: dict) -> str:
        try:
            if self.config.provider == "anthropic":
                return "".join(b["text"] for b in payload["content"] if b.get("type") == "text")
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise JudgeError(f"unexpected response shape: {str(payload)[:200]}") from e

    def _delay(self, attempt: int, response: httpx.Response | None) -> float:
        base, cap = self._backoff
        if response is not None:
            try:
                return min(float(response.headers["retry-after"]), cap)
            except (KeyError, ValueError):
                pass
        return min(base * 2**attempt, cap)

    def _complete(self, user: str) -> str:
        url, headers, body = self._request(user)
        for attempt in range(self._max_retries + 1):
            self.calls += 1
            response = None
            try:
                response = self._http.post(url, headers=headers, json=body)
            except httpx.TransportError as e:
                if attempt == self._max_retries:
                    raise JudgeError(f"transport error after {attempt + 1} attempts: {e}") from e
            else:
                if response.status_code == 200:
                    return self._extract(response.json())
                if response.status_code not in RETRY_STATUSES or attempt == self._max_retries:
                    raise JudgeError(f"HTTP {response.status_code}: {response.text[:200]}")
            self._sleep(self._delay(attempt, response))
        raise AssertionError("unreachable")

    def close(self) -> None:
        self._http.close()
        if self.cache is not None:
            self.cache.close()
