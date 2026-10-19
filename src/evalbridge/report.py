"""The judge report card: agreement with expert humans, position bias and length bias."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from evalbridge import __version__
from evalbridge.data import PairItem
from evalbridge.harness import attach_to_votes, is_order_invariant, per_item, run_judge
from evalbridge.judges import BASELINES, Judge
from evalbridge.metrics import (
    agreement,
    agreement_without_ties,
    bootstrap_ci,
    cohen_kappa,
    first_position_win_rate,
    pairwise_agreement_within_groups,
    prefers_longer_rate,
    swap_consistency,
)


def _nan_to_none(x):
    return None if isinstance(x, float) and math.isnan(x) else x


def score(human, judged, groups, *, n_boot: int, seed: int) -> dict:
    """Agreement, non-tie agreement and kappa, each with a question-clustered bootstrap CI."""
    human, judged = list(human), list(judged)
    no_tie = sum(h != "tie" and j != "tie" for h, j in zip(human, judged, strict=True))
    kw = {"groups": groups, "n_boot": n_boot, "seed": seed}
    out = {
        "agreement_with_ties": bootstrap_ci(agreement, human, judged, **kw).as_dict(),
        "agreement_without_ties": bootstrap_ci(
            agreement_without_ties, human, judged, **kw
        ).as_dict(),
        "cohen_kappa": bootstrap_ci(cohen_kappa, human, judged, **kw).as_dict(),
    }
    out["agreement_without_ties"]["n"] = no_tie
    return out


def _evaluate(judge: Judge, items, votes, *, swap: bool, n_boot: int, seed: int) -> dict:
    swap = swap and not is_order_invariant(judge)
    results = run_judge(judge, items, swap=swap)
    v = attach_to_votes(results, votes)
    groups = v["question_id"].to_numpy()
    item_view = per_item(results)
    out: dict = {
        "name": judge.name,
        "n_items_scored": len(item_view),
        "n_parse_errors": int(results["error"].notna().sum()),
        "n_votes_scored": len(v),
        "verdict_used": "merged across both orders" if swap else "single order",
        "agreement": score(v["human"], v["judge_final"], groups, n_boot=n_boot, seed=seed),
    }
    if swap:
        out["agreement_single_order"] = score(
            v["human"], v["judge_single"], groups, n_boot=n_boot, seed=seed
        )
        out["position_bias"] = {
            "swap_consistency": _nan_to_none(
                swap_consistency(item_view["v_ab"], item_view["v_ba_raw"])
            ),
            "first_position_win_rate": _nan_to_none(
                first_position_win_rate(item_view["v_ab"], item_view["v_ba_raw"])
            ),
            "n_items": len(item_view),
        }
    else:
        out["position_bias"] = None
    out["length_bias"] = {
        "judge_prefers_longer": _nan_to_none(
            prefers_longer_rate(v["judge_final"], v["len_a"], v["len_b"])
        ),
        "human_prefers_longer": _nan_to_none(
            prefers_longer_rate(v["human"], v["len_a"], v["len_b"])
        ),
    }
    if getattr(judge, "raw_labels", None):
        out["recorded_labels"] = {str(k): int(n) for k, n in judge.raw_labels.items()}
    return out


def build_report_card(
    judge: Judge,
    items: list[PairItem],
    votes: pd.DataFrame,
    *,
    data_info: dict,
    swap: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    keys = {(i.question_id, i.turn, *sorted((i.model_a, i.model_b))) for i in items}
    votes = votes[votes["item_key"].isin(keys)].reset_index(drop=True)
    main = _evaluate(judge, items, votes, swap=swap, n_boot=n_boot, seed=seed)
    refs = {}
    for name, cls in BASELINES.items():
        refs[name] = (
            main
            if name == judge.name
            else _evaluate(cls(), items, votes, swap=True, n_boot=n_boot, seed=seed)
        )
    human = pairwise_agreement_within_groups(
        votes["human_winner"], votes["item_key"], votes["judge"]
    )
    card = {
        "evalbridge_version": __version__,
        "judge": {
            "name": judge.name,
            "kind": type(judge).__name__,
            "prompt_version": getattr(getattr(judge, "prompt", None), "version", None),
            "prompt_sha": getattr(getattr(judge, "prompt", None), "sha", None),
            "model": getattr(getattr(judge, "config", None), "model", None),
        },
        "data": {
            **data_info,
            "n_items": len(items),
            "n_questions": len({i.question_id for i in items}),
            "n_human_votes": len(votes),
            "bootstrap": {"n_boot": n_boot, "seed": seed, "cluster": "question_id"},
        },
        "result": main,
        "reference": {
            "human_vs_human": {k: _nan_to_none(v) for k, v in human.items()},
            "baselines": {k: _short(v) for k, v in refs.items()},
        },
    }
    return card


def _short(r: dict) -> dict:
    return {
        "agreement_with_ties": r["agreement"]["agreement_with_ties"]["estimate"],
        "agreement_single_order": (r.get("agreement_single_order") or r["agreement"])[
            "agreement_with_ties"
        ]["estimate"],
        "agreement_without_ties": r["agreement"]["agreement_without_ties"]["estimate"],
        "cohen_kappa": r["agreement"]["cohen_kappa"]["estimate"],
        "swap_consistency": (r["position_bias"] or {}).get("swap_consistency"),
        "first_position_win_rate": (r["position_bias"] or {}).get("first_position_win_rate"),
        "judge_prefers_longer": r["length_bias"]["judge_prefers_longer"],
    }


# --- Rendering -----------------------------------------------------------------------------


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _num(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _ci(e: dict, fmt=_pct) -> str:
    if e["estimate"] is None:
        return "n/a"
    if e["ci_low"] is None:
        return fmt(e["estimate"])
    return f"{fmt(e['estimate'])} ({fmt(e['ci_low'])} to {fmt(e['ci_high'])})"


def render_markdown(card: dict) -> str:
    j, d, r = card["judge"], card["data"], card["result"]
    a = r["agreement"]
    lines = [
        f"# Judge report card: `{j['name']}`",
        "",
        f"- Data: {d['source']} (split `{d['split']}`, turn {d['turn']})",
        f"- Items judged: {r['n_items_scored']} of {d['n_items']} "
        f"({d['n_questions']} questions); human votes scored: {r['n_votes_scored']}",
    ]
    if j["model"]:
        lines.append(f"- Model: `{j['model']}`, prompt `{j['prompt_version']}` ({j['prompt_sha']})")
    if r["n_parse_errors"]:
        lines.append(f"- Unparseable replies (excluded): {r['n_parse_errors']}")
    lines += [
        f"- Verdict used: {r['verdict_used']}",
        f"- 95% CIs: bootstrap over questions, {d['bootstrap']['n_boot']} resamples, "
        f"seed {d['bootstrap']['seed']}",
        "",
        "## Agreement with expert humans",
        "",
        "| Measure | Value (95% CI) |",
        "| --- | --- |",
        f"| Agreement, ties count | {_ci(a['agreement_with_ties'])} |",
        f"| Agreement, both sides picked a winner | {_ci(a['agreement_without_ties'])} "
        f"(n={a['agreement_without_ties']['n']}) |",
        f"| Cohen's kappa (A / B / tie) | {_ci(a['cohen_kappa'], _num)} |",
    ]
    if "agreement_single_order" in r:
        s = r["agreement_single_order"]
        lines.append(
            f"| Agreement, single order as voters saw it | {_ci(s['agreement_with_ties'])} |"
        )
    lines += ["", "## Position bias", ""]
    p = r["position_bias"]
    if p is None:
        lines.append(
            "Not measured: this judge's verdicts do not depend on order (recorded verdicts "
            "were already merged across both orders)."
        )
        if r.get("recorded_labels"):
            labels = ", ".join(f"`{k}` {v}" for k, v in sorted(r["recorded_labels"].items()))
            lines.append(f"Recorded labels in the source split: {labels}.")
    else:
        lines += [
            "| Measure | Value | Unbiased judge |",
            "| --- | --- | --- |",
            f"| Same verdict after swapping order | {_pct(p['swap_consistency'])} | 100% |",
            f"| First-shown answer wins (decided verdicts) | "
            f"{_pct(p['first_position_win_rate'])} | 50% |",
        ]
    lb = r["length_bias"]
    lines += [
        "",
        "## Length bias",
        "",
        "Decided verdicts on pairs of unequal length (word count of the judged answer).",
        "",
        "| Who | Prefers the longer answer |",
        "| --- | --- |",
        f"| This judge | {_pct(lb['judge_prefers_longer'])} |",
        f"| Expert humans | {_pct(lb['human_prefers_longer'])} |",
        "",
        "## Reference points (same items)",
        "",
        "| Reference | Agreement | Agreement, single order | Agreement, no ties | Kappa | "
        "Swap-consistent | First wins | Prefers longer |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    hh = card["reference"]["human_vs_human"]
    lines.append(
        f"| Human vs human ({hh['n_pairs']} vote pairs) | {_pct(hh['agreement_with_ties'])} | | "
        f"{_pct(hh['agreement_without_ties'])} | | | | {_pct(lb['human_prefers_longer'])} |"
    )
    for name, b in card["reference"]["baselines"].items():
        lines.append(
            f"| `{name}` baseline | {_pct(b['agreement_with_ties'])} | "
            f"{_pct(b['agreement_single_order'])} | {_pct(b['agreement_without_ties'])} | "
            f"{_num(b['cohen_kappa'])} | "
            f"{_pct(b['swap_consistency'])} | {_pct(b['first_position_win_rate'])} | "
            f"{_pct(b['judge_prefers_longer'])} |"
        )
    lines += [
        "",
        "Ties: a tie agrees only with a tie. Kappa uses three classes. Order-merged verdicts "
        "become a tie when the two orders disagree.",
        "",
    ]
    return "\n".join(lines)


def write_report(card: dict, out_dir: str | Path, stem: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md, js = out / f"{stem}.md", out / f"{stem}.json"
    md.write_text(render_markdown(card), encoding="utf-8")
    js.write_text(json.dumps(card, indent=2, default=str) + "\n", encoding="utf-8")
    return md, js
