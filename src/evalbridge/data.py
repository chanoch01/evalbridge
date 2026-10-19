"""Load the MT-Bench human judgments and turn them into items a judge can score.

The dataset is ``lmsys/mt_bench_human_judgments`` on Hugging Face (CC BY 4.0). It has two splits
with the same schema:

* ``human``: 3,355 pairwise votes by experts and paper authors on 80 questions, 6 models, 2 turns
* ``gpt4_pair``: 2,400 GPT-4 pairwise verdicts on every model pair, already run in both orders

Each row has ``question_id, model_a, model_b, winner, judge, conversation_a, conversation_b,
turn``. Both conversations always hold the full two-turn dialogue (user, assistant, user,
assistant). ``turn`` says which assistant turn the vote is about.

Winner labels are normalised to ``model_a``, ``model_b`` or ``tie``. Any label starting with
``tie`` (``tie``, ``tie (inconsistent)``, Chatbot Arena's ``tie (bothbad)``) counts as a tie, and
the original label is kept in ``winner_raw``.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
import pandas as pd

log = logging.getLogger(__name__)

HF_REPO = "lmsys/mt_bench_human_judgments"
# Pinned dataset revision, so a re-run reads exactly the same rows.
HF_REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
SPLIT_FILES = {
    "human": "data/human-00000-of-00001-25f4910818759289.parquet",
    "gpt4_pair": "data/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet",
}
COLUMNS = [
    "question_id",
    "model_a",
    "model_b",
    "winner",
    "judge",
    "conversation_a",
    "conversation_b",
    "turn",
]
FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
FIXTURES = {
    "human": FIXTURE_DIR / "sample_human_judgments.jsonl.gz",
    "gpt4_pair": FIXTURE_DIR / "sample_gpt4_pair.jsonl.gz",
}


def split_url(split: str, revision: str = HF_REVISION) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/{revision}/{SPLIT_FILES[split]}"


def data_dir() -> Path:
    return Path(os.environ.get("EVALBRIDGE_DATA_DIR", "data/raw"))


def download(split: str = "human", dest: Path | None = None, *, force: bool = False) -> Path:
    """Download one split's parquet file (about 0.7 MB each) and return its local path."""
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_FILES)}")
    dest = dest or data_dir() / "mt_bench" / f"{split}.parquet"
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = split_url(split)
    log.info("Downloading %s", url)
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    tmp.replace(dest)
    return dest


def normalise_winner(label: str) -> str:
    s = str(label).strip().lower()
    if s in ("model_a", "model_b"):
        return s
    if s.startswith("tie"):
        return "tie"
    raise ValueError(f"unrecognised winner label {label!r}")


def _read_jsonl(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def load_judgments(path: str | Path) -> pd.DataFrame:
    """Read a split from parquet or JSON Lines (optionally gzipped) and normalise winners."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else _read_jsonl(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    df = df[COLUMNS].copy()
    for col in ("conversation_a", "conversation_b"):
        df[col] = df[col].map(lambda conv: [dict(m) for m in conv])
    df["question_id"] = df["question_id"].astype(int)
    df["turn"] = df["turn"].astype(int)
    df["winner_raw"] = df["winner"].astype(str)
    df["winner"] = df["winner_raw"].map(normalise_winner)
    return df.reset_index(drop=True)


def load_split(split: str = "human", *, fixture: bool = False) -> pd.DataFrame:
    """Load a split: the synthetic fixture if ``fixture`` is set, else the real (cached) data."""
    return load_judgments(FIXTURES[split] if fixture else download(split))


# --- Items ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairItem:
    """One question and two answers, in the order a judge will see them."""

    question_id: int
    turn: int
    model_a: str
    model_b: str
    question: str
    answer_a: str
    answer_b: str
    len_a: int  # words in the judged (final) answer of model_a
    len_b: int

    def swapped(self) -> PairItem:
        return replace(
            self,
            model_a=self.model_b,
            model_b=self.model_a,
            answer_a=self.answer_b,
            answer_b=self.answer_a,
            len_a=self.len_b,
            len_b=self.len_a,
        )

    @property
    def pair(self) -> frozenset[str]:
        return frozenset((self.model_a, self.model_b))


JUDGED_MARKER = ": judge this reply]\n"


def word_count(text: str) -> int:
    return len(text.split())


def judged_reply(answer: str) -> str:
    """The reply under judgment: for turn 2+, the text after the context marker."""
    head, sep, tail = answer.rpartition(JUDGED_MARKER)
    return tail if sep else answer


def _turns(conv: list[dict], role: str) -> list[str]:
    return [m["content"] for m in conv if m["role"] == role]


def render(conv_a: list[dict], conv_b: list[dict], turn: int) -> tuple[str, str, str, str, str]:
    """Return (question, answer_a, answer_b, final_a, final_b) for the given turn.

    Turn 1 is a plain question and two answers. For turn 2 the judge needs the earlier
    exchange, so the question lists both user messages and each answer shows that model's
    turn-1 reply as context before the turn-2 reply being judged.
    """
    users = _turns(conv_a, "user")
    if users != _turns(conv_b, "user"):
        raise ValueError("conversation_a and conversation_b have different user messages")
    a, b = _turns(conv_a, "assistant"), _turns(conv_b, "assistant")
    if not 1 <= turn <= min(len(users), len(a), len(b)):
        raise ValueError(f"turn {turn} not present in conversation")
    if turn == 1:
        return users[0], a[0], b[0], a[0], b[0]

    question = "\n\n".join(f"[User, turn {i + 1}]\n{u}" for i, u in enumerate(users[:turn]))

    def with_context(answers: list[str]) -> str:
        parts = [f"[Assistant, turn {i + 1}]\n{x}" for i, x in enumerate(answers[: turn - 1])]
        parts.append(f"[Assistant, turn {turn}{JUDGED_MARKER}{answers[turn - 1]}")
        return "\n\n".join(parts)

    return question, with_context(a), with_context(b), a[turn - 1], b[turn - 1]


def item_key(question_id: int, turn: int, model_x: str, model_y: str) -> tuple:
    """Order-free identity of an item: the same pair shown either way round is one item."""
    return (int(question_id), int(turn), *sorted((model_x, model_y)))


def build_items(df: pd.DataFrame, turn: int = 1) -> tuple[list[PairItem], pd.DataFrame]:
    """Split human votes for one turn into unique items and a per-vote table.

    Returns the unique items (in the orientation of the first vote seen for each) and a vote
    table with one row per human vote: ``item_key, question_id, model_a, model_b, judge``,
    ``human`` (``A``, ``B`` or ``tie`` in that vote's own orientation), ``human_winner`` (the
    winning model's name or ``tie``, which does not depend on orientation) and ``len_a, len_b``.
    """
    rows = df[df["turn"] == turn]
    if rows.empty:
        raise ValueError(f"no votes for turn {turn}")
    items: dict[tuple, PairItem] = {}
    votes = []
    for r in rows.itertuples(index=False):
        key = item_key(r.question_id, turn, r.model_a, r.model_b)
        q, ans_a, ans_b, fin_a, fin_b = render(r.conversation_a, r.conversation_b, turn)
        if key not in items:
            items[key] = PairItem(
                question_id=int(r.question_id),
                turn=turn,
                model_a=r.model_a,
                model_b=r.model_b,
                question=q,
                answer_a=ans_a,
                answer_b=ans_b,
                len_a=word_count(fin_a),
                len_b=word_count(fin_b),
            )
        votes.append(
            {
                "item_key": key,
                "question_id": int(r.question_id),
                "model_a": r.model_a,
                "model_b": r.model_b,
                "judge": r.judge,
                "human": {"model_a": "A", "model_b": "B", "tie": "tie"}[r.winner],
                "human_winner": {"model_a": r.model_a, "model_b": r.model_b, "tie": "tie"}[
                    r.winner
                ],
                "len_a": word_count(fin_a),
                "len_b": word_count(fin_b),
            }
        )
    return list(items.values()), pd.DataFrame(votes)


def sample_items(items: Iterable[PairItem], limit: int | None, seed: int = 0) -> list[PairItem]:
    """A seeded random subset of ``limit`` items (all items, in order, if ``limit`` is None)."""
    items = list(items)
    if limit is None or limit >= len(items):
        return items
    order = pd.Series(range(len(items))).sample(frac=1.0, random_state=seed).tolist()
    return [items[i] for i in sorted(order[:limit])]
