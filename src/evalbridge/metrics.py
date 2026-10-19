"""Agreement, bias and uncertainty metrics for pairwise judges.

Verdicts are the strings ``"A"``, ``"B"`` and ``"tie"``. Ties are handled as follows:

* **Agreement with ties**: exact three-way match. A tie only agrees with a tie.
* **Agreement without ties**: only rows where *both* sides picked a winner (MT-Bench setup S2).
* **Cohen's kappa**: three classes (A, B, tie), so it corrects for each side's own tie rate.
* **Position bias**: tie counts as consistent only if both orders give a tie. The first-position
  win rate ignores ties.
* **Length bias**: ties, and pairs whose answers have the same word count, are left out.

Any rate with an empty denominator is ``nan``, never 0.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

LABELS = ("A", "B", "tie")
_SWAP = {"A": "B", "B": "A", "tie": "tie"}


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=object)


def _mean(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


# --- Agreement -----------------------------------------------------------------------------


def agreement(y1: Sequence[str], y2: Sequence[str]) -> float:
    """Share of rows where the two labels match exactly (ties included)."""
    y1, y2 = _arr(y1), _arr(y2)
    return _mean(y1 == y2)


def agreement_without_ties(y1: Sequence[str], y2: Sequence[str]) -> float:
    """Agreement on rows where neither side said tie."""
    y1, y2 = _arr(y1), _arr(y2)
    keep = (y1 != "tie") & (y2 != "tie")
    return _mean(y1[keep] == y2[keep])


def cohen_kappa(y1: Sequence[str], y2: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    """Cohen's kappa: (p_o - p_e) / (1 - p_e), with p_e from each side's own label rates."""
    y1, y2 = _arr(y1), _arr(y2)
    if y1.size == 0:
        return float("nan")
    p_o = float((y1 == y2).mean())
    p_e = sum(float((y1 == k).mean()) * float((y2 == k).mean()) for k in labels)
    if np.isclose(p_e, 1.0):
        return float("nan")  # both sides constant and identical: kappa is undefined
    return (p_o - p_e) / (1 - p_e)


def pairwise_agreement_within_groups(
    labels: Sequence[str], groups: Sequence, raters: Sequence | None = None
) -> dict[str, float]:
    """Agreement between every pair of votes on the same item (e.g. human vs human).

    Pairs from the same rater are skipped. Returns agreement with and without ties and the
    number of vote pairs behind each.
    """
    labels, groups = _arr(labels), np.asarray(groups, dtype=object)
    raters = np.asarray(raters if raters is not None else range(len(labels)), dtype=object)
    order: dict = {}
    for i, g in enumerate(groups):
        order.setdefault(g, []).append(i)
    same, same_nt, n, n_nt = 0, 0, 0, 0
    for idx in order.values():
        for i, j in combinations(idx, 2):
            if raters[i] == raters[j]:
                continue
            n += 1
            same += labels[i] == labels[j]
            if labels[i] != "tie" and labels[j] != "tie":
                n_nt += 1
                same_nt += labels[i] == labels[j]
    return {
        "agreement_with_ties": same / n if n else float("nan"),
        "agreement_without_ties": same_nt / n_nt if n_nt else float("nan"),
        "n_pairs": n,
        "n_pairs_without_ties": n_nt,
    }


# --- Bootstrap -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    estimate: float
    ci_low: float
    ci_high: float
    n: int

    def as_dict(self) -> dict:
        out = asdict(self)
        return {k: None if isinstance(v, float) and np.isnan(v) else v for k, v in out.items()}


def bootstrap_ci(
    stat: Callable[..., float],
    *arrays: Sequence,
    groups: Sequence | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Estimate:
    """Percentile bootstrap confidence interval for ``stat(*arrays)``.

    With ``groups`` it is a cluster bootstrap: whole groups (e.g. questions) are resampled, so
    votes on the same question are not treated as independent. Replicates where the statistic
    is undefined (``nan``) are dropped. The seed makes the interval reproducible.
    """
    cols = [_arr(a) for a in arrays]
    n = len(cols[0])
    if any(len(c) != n for c in cols):
        raise ValueError("arrays must have equal length")
    point = stat(*cols)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    if groups is None:
        blocks = [np.array([i]) for i in range(n)]
    else:
        g = np.asarray(groups, dtype=object)
        if len(g) != n:
            raise ValueError("groups must match the arrays' length")
        index: dict = {}
        for i, key in enumerate(g):
            index.setdefault(key, []).append(i)
        blocks = [np.array(v) for v in index.values()]
    reps = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[k] for k in pick])
        reps[b] = stat(*(c[idx] for c in cols))
    reps = reps[~np.isnan(reps)]
    if reps.size == 0:
        return Estimate(point, float("nan"), float("nan"), n)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(reps, [alpha, 1 - alpha])
    return Estimate(float(point), float(lo), float(hi), n)


# --- Position bias -------------------------------------------------------------------------


def unswap(verdicts: Sequence[str]) -> np.ndarray:
    """Map verdicts given on the swapped presentation (B shown first) back to the original."""
    return np.array([_SWAP[v] for v in verdicts], dtype=object)


def swap_consistency(original: Sequence[str], swapped_raw: Sequence[str]) -> float:
    """Share of items where the verdict survives swapping the order of the two answers."""
    return _mean(_arr(original) == unswap(swapped_raw))


def first_position_win_rate(original: Sequence[str], swapped_raw: Sequence[str]) -> float:
    """Among non-tie verdicts in both presentations, how often the answer shown first won.

    ``swapped_raw`` must be as the judge answered it (position labels, not unswapped).
    0.5 means no position preference.
    """
    raw = np.concatenate([_arr(original), _arr(swapped_raw)])
    decided = raw[raw != "tie"]
    return _mean(decided == "A")


def merge_orders(original: Sequence[str], swapped_raw: Sequence[str]) -> np.ndarray:
    """Final verdict from both orders: kept when consistent, otherwise a tie.

    This is the rule MT-Bench used for its ``gpt4_pair`` split.
    """
    o, s = _arr(original), unswap(swapped_raw)
    return np.where(o == s, o, "tie").astype(object)


# --- Length bias ---------------------------------------------------------------------------


def prefers_longer_rate(verdicts: Sequence[str], len_a: Sequence[int], len_b: Sequence[int]):
    """Among decided verdicts on pairs of unequal length, the share won by the longer answer."""
    v = _arr(verdicts)
    la, lb = np.asarray(len_a), np.asarray(len_b)
    keep = (v != "tie") & (la != lb)
    longer = np.where(la > lb, "A", "B").astype(object)
    return _mean(v[keep] == longer[keep])
