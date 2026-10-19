from __future__ import annotations

import math

import numpy as np
import pytest

from evalbridge.metrics import (
    agreement,
    agreement_without_ties,
    bootstrap_ci,
    cohen_kappa,
    first_position_win_rate,
    merge_orders,
    pairwise_agreement_within_groups,
    prefers_longer_rate,
    swap_consistency,
    unswap,
)


def test_agreement_counts_ties_as_a_class():
    assert agreement(["A", "B", "tie", "tie"], ["A", "A", "tie", "B"]) == 0.5


def test_agreement_without_ties_keeps_rows_where_both_decided():
    # Rows 3 and 4 have a tie on one side and are dropped; 1 of the 2 kept rows match.
    assert agreement_without_ties(["A", "B", "tie", "A"], ["A", "A", "B", "tie"]) == 0.5
    assert math.isnan(agreement_without_ties(["tie"], ["A"]))


def test_kappa_two_class_hand_computed():
    # Classic 2x2 table, 50 items: both A 20, A/B 5, B/A 10, both B 15.
    # p_o = 35/50 = 0.7. Rater 1 says A 25/50, rater 2 says A 30/50.
    # p_e = 0.5*0.6 + 0.5*0.4 = 0.5, so kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4.
    r1 = ["A"] * 20 + ["A"] * 5 + ["B"] * 10 + ["B"] * 15
    r2 = ["A"] * 20 + ["B"] * 5 + ["A"] * 10 + ["B"] * 15
    assert cohen_kappa(r1, r2) == pytest.approx(0.4)


def test_kappa_three_class_hand_computed():
    # 7 of 10 match: p_o = 0.7. Both raters: A 4, B 3, tie 3.
    # p_e = 0.4^2 + 0.3^2 + 0.3^2 = 0.34, kappa = 0.36 / 0.66 = 6/11.
    r1 = ["A", "A", "A", "A", "B", "B", "B", "tie", "tie", "tie"]
    r2 = ["A", "A", "A", "B", "B", "B", "tie", "tie", "tie", "A"]
    assert cohen_kappa(r1, r2) == pytest.approx(6 / 11)


def test_kappa_edge_cases():
    assert cohen_kappa(["A", "B", "A"], ["A", "B", "A"]) == pytest.approx(1.0)
    # A constant judge has p_o == p_e, so kappa is exactly 0 whatever it agrees with.
    assert cohen_kappa(["A", "B", "tie", "A"], ["A"] * 4) == pytest.approx(0.0)
    assert math.isnan(cohen_kappa(["A", "A"], ["A", "A"]))  # undefined: p_e = 1
    assert math.isnan(cohen_kappa([], []))


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    y1 = rng.choice(["A", "B", "tie"], size=300)
    y2 = np.where(rng.random(300) < 0.7, y1, rng.choice(["A", "B", "tie"], size=300))
    e1 = bootstrap_ci(agreement, y1, y2, n_boot=300, seed=42)
    e2 = bootstrap_ci(agreement, y1, y2, n_boot=300, seed=42)
    assert e1 == e2
    assert e1.ci_low < e1.estimate < e1.ci_high
    assert e1.n == 300
    assert bootstrap_ci(agreement, y1, y2, n_boot=300, seed=7) != e1


def test_cluster_bootstrap_is_wider_when_votes_within_a_group_move_together():
    # 20 groups of 10 identical rows: really 20 observations, not 200.
    y1 = np.repeat(np.array(["A", "B"] * 10), 10)
    y2 = np.repeat(np.array(["A"] * 13 + ["B"] * 7), 10)
    groups = np.repeat(np.arange(20), 10)
    rows = bootstrap_ci(agreement, y1, y2, n_boot=500, seed=0)
    clustered = bootstrap_ci(agreement, y1, y2, groups=groups, n_boot=500, seed=0)
    assert rows.estimate == clustered.estimate
    assert clustered.ci_high - clustered.ci_low > 2 * (rows.ci_high - rows.ci_low)


def test_bootstrap_drops_undefined_replicates_and_handles_empty_input():
    e = bootstrap_ci(agreement_without_ties, ["tie", "A"], ["A", "A"], n_boot=200, seed=0)
    assert e.estimate == 1.0 and e.ci_low == 1.0 and e.ci_high == 1.0
    empty = bootstrap_ci(agreement, [], [])
    assert empty.n == 0 and math.isnan(empty.estimate)
    assert empty.as_dict()["estimate"] is None


def test_position_metrics_for_extreme_judges():
    original = ["A", "B", "tie", "A"]
    # A judge that is fully consistent: swapped raw verdicts are the mirror image.
    assert swap_consistency(original, unswap(original)) == 1.0
    # A judge that always picks the first answer.
    always_first = ["A"] * 4
    assert swap_consistency(always_first, always_first) == 0.0
    assert first_position_win_rate(always_first, always_first) == 1.0
    assert list(merge_orders(always_first, always_first)) == ["tie"] * 4


def test_first_position_win_rate_ignores_ties():
    # Decided verdicts across both orders: A, B, A -> first wins 2 of 3.
    assert first_position_win_rate(["A", "tie"], ["B", "A"]) == pytest.approx(2 / 3)
    assert math.isnan(first_position_win_rate(["tie"], ["tie"]))


def test_merge_orders_keeps_consistent_verdicts():
    # Item 1 consistent (B first in swap means original A), item 2 flips with position.
    assert list(merge_orders(["A", "A"], ["B", "A"])) == ["A", "tie"]


def test_prefers_longer_rate_skips_ties_and_equal_lengths():
    verdicts = ["A", "B", "A", "tie", "B"]
    len_a = [100, 100, 10, 50, 30]
    len_b = [10, 10, 100, 10, 30]  # last pair is equal length
    # Kept: rows 1-3. Longer won in row 1 only.
    assert prefers_longer_rate(verdicts, len_a, len_b) == pytest.approx(1 / 3)


def test_pairwise_agreement_within_groups_skips_same_rater():
    labels = ["m1", "m1", "tie", "m2", "m2"]
    groups = ["q1", "q1", "q1", "q2", "q2"]
    raters = ["r1", "r2", "r3", "r1", "r1"]
    out = pairwise_agreement_within_groups(labels, groups, raters)
    # q1 pairs: (m1,m1) agree, (m1,tie) x2 disagree; q2 pair is the same rater -> skipped.
    assert out["n_pairs"] == 3
    assert out["agreement_with_ties"] == pytest.approx(1 / 3)
    assert out["n_pairs_without_ties"] == 1 and out["agreement_without_ties"] == 1.0
