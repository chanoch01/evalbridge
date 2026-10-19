"""Pairwise judges.

A judge reads a question and two answers and returns ``"A"``, ``"B"`` or ``"tie"``. Anything
with a matching ``judge`` method works, so a new judge does not need to subclass anything.

Callers may also pass ``ref``, which names the question and the models in the order shown.
Judges that need an identity use it: the LLM judge keys its cache on it, and the recorded judge
looks its verdict up by it. The baseline judges ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

from evalbridge.data import judged_reply, word_count

Verdict = Literal["A", "B", "tie"]
VERDICTS: tuple[Verdict, ...] = ("A", "B", "tie")


@dataclass(frozen=True)
class ItemRef:
    question_id: int
    turn: int
    model_a: str  # model shown in position A
    model_b: str  # model shown in position B


@runtime_checkable
class Judge(Protocol):
    name: str

    def judge(
        self, question: str, answer_a: str, answer_b: str, *, ref: ItemRef | None = None
    ) -> Verdict: ...


class LongerAnswerJudge:
    """Prefers the answer with more words; a tie when the counts are equal.

    In multi-turn items only the reply being judged is counted, not the earlier context.

    A maximally length-biased reference point: whatever agreement it reaches with humans is
    agreement a real judge could get from length alone.
    """

    name = "longer"

    def judge(self, question, answer_a, answer_b, *, ref=None) -> Verdict:
        la, lb = word_count(judged_reply(answer_a)), word_count(judged_reply(answer_b))
        if la == lb:
            return "tie"
        return "A" if la > lb else "B"


class FirstPositionJudge:
    """Always prefers whichever answer it is shown first. A maximally position-biased reference."""

    name = "first"

    def judge(self, question, answer_a, answer_b, *, ref=None) -> Verdict:
        return "A"


class RecordedJudge:
    """Replays verdicts that were recorded earlier, such as the ``gpt4_pair`` split.

    Verdicts are stored by winning model name, so they do not depend on presentation order.
    The ``gpt4_pair`` split was already run in both orders, with disagreements recorded as
    ``tie (inconsistent)``, so its position bias cannot be re-measured here: the report says so.
    """

    order_invariant = True

    def __init__(self, df: pd.DataFrame, name: str = "gpt4-recorded"):
        self.name = name
        self._winners: dict[tuple, str] = {}
        self.raw_labels = df["winner_raw"].value_counts().to_dict()
        for r in df.itertuples(index=False):
            key = (int(r.question_id), int(r.turn), frozenset((r.model_a, r.model_b)))
            winner = {"model_a": r.model_a, "model_b": r.model_b, "tie": "tie"}[r.winner]
            if self._winners.setdefault(key, winner) != winner:
                raise ValueError(f"conflicting recorded verdicts for {key}")

    def judge(self, question, answer_a, answer_b, *, ref=None) -> Verdict:
        if ref is None:
            raise ValueError("RecordedJudge needs ref to look up the verdict")
        key = (ref.question_id, ref.turn, frozenset((ref.model_a, ref.model_b)))
        try:
            winner = self._winners[key]
        except KeyError as e:
            raise LookupError(f"no recorded verdict for {key}") from e
        if winner == "tie":
            return "tie"
        return "A" if winner == ref.model_a else "B"


BASELINES = {"longer": LongerAnswerJudge, "first": FirstPositionJudge}
