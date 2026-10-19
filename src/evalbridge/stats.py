"""Week 3 (not built yet): online experiment statistics.

Each function will be tested against published worked examples or simulation before use.
"""

from __future__ import annotations

from collections.abc import Sequence


def srm_check(observed: Sequence[int], expected_ratio: Sequence[float]) -> dict:
    """Sample-ratio mismatch: chi-square test of assignment counts against the planned split."""
    raise NotImplementedError("Week 3")


def cuped_adjust(y: Sequence[float], x_pre: Sequence[float]) -> list[float]:
    """CUPED: subtract theta * (x_pre - mean) to cut variance using a pre-period covariate."""
    raise NotImplementedError("Week 3")


def msprt(successes_a: int, n_a: int, successes_b: int, n_b: int, *, tau: float) -> dict:
    """Mixture sequential probability ratio test: always-valid p-value for a running A/B test."""
    raise NotImplementedError("Week 3")


def required_sample_size(baseline: float, mde: float, *, alpha=0.05, power=0.8) -> int:
    """Per-arm sample size to detect an absolute lift ``mde`` on a ``baseline`` rate."""
    raise NotImplementedError("Week 3")
