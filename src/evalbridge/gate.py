"""Week 5 (not built yet): the ship / kill / A/B-test gate.

For a proposed change, combine the judge's offline score delta, the bridge study's mapping
from judge delta to expected user-preference delta (with its uncertainty), and the cost of
being wrong, then return one decision with its reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Decision = Literal["ship", "kill", "ab_test"]


@dataclass(frozen=True)
class GateResult:
    decision: Decision
    expected_user_delta: float
    ci_low: float
    ci_high: float
    reasons: tuple[str, ...]


def decide(judge_delta: float, bridge: dict, *, min_effect: float = 0.0) -> GateResult:
    """Ship if the whole CI clears ``min_effect``, kill if it is all below 0, else A/B test."""
    raise NotImplementedError("Week 5")
