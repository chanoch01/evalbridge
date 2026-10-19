"""Run a judge over items, in both orders, and line its verdicts up with the human votes."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from evalbridge.data import PairItem, item_key
from evalbridge.judges import ItemRef, Judge
from evalbridge.llm import VerdictParseError
from evalbridge.metrics import merge_orders, unswap

log = logging.getLogger(__name__)
_FLIP = {"A": "B", "B": "A", "tie": "tie"}


def is_order_invariant(judge: Judge) -> bool:
    return bool(getattr(judge, "order_invariant", False))


def run_judge(judge: Judge, items: Iterable[PairItem], *, swap: bool = True) -> pd.DataFrame:
    """Judge every item as shown (``v_ab``) and, with ``swap``, with the answers swapped.

    ``v_ba_raw`` is the swapped verdict exactly as the judge gave it (``A`` = the answer it saw
    first). A reply that cannot be parsed is recorded in ``error`` and left out of the scores;
    any other judge failure stops the run (cached calls are kept, so a re-run resumes).
    """
    rows = []
    items = list(items)
    for n, it in enumerate(items, 1):
        rec = {
            "item_key": item_key(it.question_id, it.turn, it.model_a, it.model_b),
            "question_id": it.question_id,
            "turn": it.turn,
            "model_a": it.model_a,
            "model_b": it.model_b,
            "len_a": it.len_a,
            "len_b": it.len_b,
            "v_ab": None,
            "v_ba_raw": None,
            "error": None,
        }
        try:
            ref = ItemRef(it.question_id, it.turn, it.model_a, it.model_b)
            rec["v_ab"] = judge.judge(it.question, it.answer_a, it.answer_b, ref=ref)
            if swap:
                s = it.swapped()
                ref = ItemRef(s.question_id, s.turn, s.model_a, s.model_b)
                rec["v_ba_raw"] = judge.judge(s.question, s.answer_a, s.answer_b, ref=ref)
        except VerdictParseError as e:
            rec["error"] = str(e)
        rows.append(rec)
        if n % 100 == 0:
            log.info("%s: judged %d/%d items", judge.name, n, len(items))
    return pd.DataFrame(rows)


def per_item(results: pd.DataFrame) -> pd.DataFrame:
    """Add ``v_ba`` (swapped verdict mapped back) and ``v_final`` (merged across orders)."""
    out = results[results["error"].isna()].copy()
    swapped = out["v_ba_raw"].notna().all() and len(out) > 0
    if swapped:
        out["v_ba"] = unswap(out["v_ba_raw"])
        out["v_final"] = merge_orders(out["v_ab"], out["v_ba_raw"])
    else:
        out["v_ba"] = None
        out["v_final"] = out["v_ab"]
    return out


def attach_to_votes(results: pd.DataFrame, votes: pd.DataFrame) -> pd.DataFrame:
    """One row per human vote, with the judge's verdicts in that vote's own orientation.

    ``judge_single`` is what the judge said when shown the pair the way this voter saw it.
    ``judge_final`` is the order-merged verdict. Votes on items the judge did not score
    (outside ``--limit`` or unparseable) are dropped.
    """
    items = per_item(results).set_index("item_key")
    v = votes[votes["item_key"].isin(items.index)].copy()
    it = items.loc[v["item_key"]]
    same = (v["model_a"].to_numpy() == it["model_a"].to_numpy()) & (
        v["model_b"].to_numpy() == it["model_b"].to_numpy()
    )
    swapped = it["v_ba_raw"].notna().to_numpy()
    flip = np.vectorize(_FLIP.get, otypes=[object])
    v_ab, v_final = it["v_ab"].to_numpy(dtype=object), it["v_final"].to_numpy(dtype=object)
    # Shown the other way round: use the swapped run's raw verdict if there is one.
    other = np.where(swapped, it["v_ba_raw"].to_numpy(dtype=object), flip(v_ab))
    v["judge_single"] = np.where(same, v_ab, other)
    v["judge_final"] = np.where(same, v_final, flip(v_final))
    return v.reset_index(drop=True)
