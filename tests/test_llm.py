from __future__ import annotations

import json

import httpx
import pytest

from evalbridge.judges import ItemRef, Judge
from evalbridge.llm import (
    JudgeError,
    JudgmentCache,
    LLMConfig,
    LLMJudge,
    Prompt,
    VerdictParseError,
    load_prompt,
    parse_verdict,
)

REF = ItemRef(81, 1, "model-x", "model-y")


# --- Parsing and prompt --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Reply A is more accurate.\n[[A]]", "A"),
        ("B handles the edge case.\n\n[[B]]\n", "B"),
        ("Both are equally good.\n[[C]]", "tie"),
        ("I lean to [[A]] because...\nFinal: [[A]]", "A"),  # same token repeated is fine
    ],
)
def test_parse_verdict_accepts_one_clear_token(text, expected):
    assert parse_verdict(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "A is better.",  # no token
        "[[A]]\nOn reflection, B.",  # token not on the last line
        "[[A]] [[B]]",  # two tokens on the last line
        "Maybe [[B]].\n[[A]]",  # conflicting tokens
        "[[a]]",  # lower case is not accepted
        "[A]",
    ],
)
def test_parse_verdict_rejects_anything_ambiguous(text):
    with pytest.raises(VerdictParseError):
        parse_verdict(text)


def test_versioned_prompt_loads_and_renders():
    p = load_prompt("pairwise_v1")
    assert p.version == "pairwise_v1" and len(p.sha) == 12
    assert "[[A]]" in p.system and "[[C]]" in p.system
    text = p.render("Q?", "ans-a", "ans-b")
    assert "Q?" in text and "ans-a" in text and "{{" not in text


# --- Config --------------------------------------------------------------------------------


def test_config_from_env():
    env = {"EVALBRIDGE_MODEL": "m", "EVALBRIDGE_API_KEY": "k"}
    c = LLMConfig.from_env(env)
    assert (c.provider, c.base) == ("openai", "https://api.openai.com/v1")
    c = LLMConfig.from_env({**env, "EVALBRIDGE_PROVIDER": "anthropic"})
    assert c.base == "https://api.anthropic.com"
    c = LLMConfig.from_env({**env, "EVALBRIDGE_API_BASE": "http://localhost:8000/v1/"})
    assert c.base == "http://localhost:8000/v1"


def test_config_requires_model_and_key():
    with pytest.raises(JudgeError, match="EVALBRIDGE_MODEL"):
        LLMConfig.from_env({"EVALBRIDGE_API_KEY": "k"})
    with pytest.raises(JudgeError, match="PROVIDER"):
        LLMConfig.from_env(
            {"EVALBRIDGE_MODEL": "m", "EVALBRIDGE_API_KEY": "k", "EVALBRIDGE_PROVIDER": "other"}
        )


# --- HTTP ----------------------------------------------------------------------------------


def openai_reply(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def make_judge(handler, tmp_path=None, provider="openai", **kwargs):
    sleeps: list[float] = []
    cache = JudgmentCache(tmp_path / "cache.sqlite") if tmp_path else None
    judge = LLMJudge(
        LLMConfig(model="test-model", api_key="secret", provider=provider),
        cache=cache,
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        **kwargs,
    )
    return judge, sleeps


def test_openai_request_shape():
    seen = []

    def handler(request):
        seen.append(request)
        return openai_reply("ok\n[[B]]")

    judge, _ = make_judge(handler)
    assert isinstance(judge, Judge)
    assert judge.judge("Q?", "a", "b", ref=REF) == "B"
    req = seen[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://api.openai.com/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer secret"
    assert body["model"] == "test-model" and body["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "Q?" in body["messages"][1]["content"]


def test_anthropic_request_shape():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "x\n[[C]]"}]})

    judge, _ = make_judge(handler, provider="anthropic")
    assert judge.judge("Q?", "a", "b") == "tie"
    req = seen[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "secret" and "anthropic-version" in req.headers
    assert body["system"].startswith("You are an impartial judge")
    assert body["messages"][0]["role"] == "user"


def test_retries_429_and_5xx_with_backoff_and_retry_after():
    responses = iter(
        [
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(503),
            httpx.Response(500),
            openai_reply("[[A]]"),
        ]
    )
    judge, sleeps = make_judge(lambda r: next(responses), backoff_base_s=1.0)
    assert judge.judge("q", "a", "b") == "A"
    assert sleeps == [7.0, 2.0, 4.0]  # Retry-After honoured, then 1 * 2**attempt
    assert judge.calls == 4


def test_retries_transport_errors_then_gives_up():
    def handler(request):
        raise httpx.ConnectError("boom")

    judge, sleeps = make_judge(handler, max_retries=2)
    with pytest.raises(JudgeError, match="transport error after 3 attempts"):
        judge.judge("q", "a", "b")
    assert len(sleeps) == 2


def test_does_not_retry_client_errors():
    judge, sleeps = make_judge(lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(JudgeError, match="HTTP 401"):
        judge.judge("q", "a", "b")
    assert sleeps == [] and judge.calls == 1


def test_gives_up_after_max_retries():
    judge, sleeps = make_judge(lambda r: httpx.Response(503), max_retries=3)
    with pytest.raises(JudgeError, match="HTTP 503"):
        judge.judge("q", "a", "b")
    assert judge.calls == 4 and len(sleeps) == 3


def test_unparseable_reply_raises_parse_error():
    judge, _ = make_judge(lambda r: openai_reply("Both fine, I think."))
    with pytest.raises(VerdictParseError):
        judge.judge("q", "a", "b")


# --- Cache ---------------------------------------------------------------------------------


def test_cache_makes_reruns_free_and_survives_restart(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return openai_reply("[[A]]")

    judge, _ = make_judge(handler, tmp_path)
    assert judge.judge("q", "a", "b", ref=REF) == "A"
    assert judge.judge("q", "a", "b", ref=REF) == "A"
    assert len(calls) == 1
    judge.close()

    again, _ = make_judge(handler, tmp_path)  # new process, same cache file
    assert again.judge("q", "a", "b", ref=REF) == "A"
    assert len(calls) == 1 and len(again.cache) == 1


def test_cache_key_separates_order_model_and_prompt(tmp_path):
    judge, _ = make_judge(lambda r: openai_reply("[[A]]"), tmp_path)
    swapped = ItemRef(REF.question_id, REF.turn, REF.model_b, REF.model_a)
    base = judge.cache_key("q", "a", "b", REF)
    assert base != judge.cache_key("q", "b", "a", swapped)  # the other order
    assert base != judge.cache_key("q", "a", "b", ItemRef(82, 1, "model-x", "model-y"))
    assert base != judge.cache_key("q", "a", "b changed", REF)  # content changed

    other_prompt = Prompt("pairwise_v1", judge.prompt.system + " Be brief.", "{{question}}")
    edited = LLMJudge(judge.config, other_prompt, transport=httpx.MockTransport(lambda r: None))
    assert base != edited.cache_key("q", "a", "b", REF)  # same version name, edited text
