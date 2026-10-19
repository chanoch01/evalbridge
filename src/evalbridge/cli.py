"""Command line entry points.

evalbridge download                                   # cache both MT-Bench splits (about 1.4 MB)
evalbridge report --judge longer --out reports/       # length-biased baseline, real data
evalbridge report --judge gpt4-recorded --out reports/  # MT-Bench's own GPT-4 verdicts
evalbridge report --judge llm --limit 50 --out reports/ # your LLM, configured in .env
evalbridge report --judge first --fixture --out /tmp/r  # synthetic fixture, no network
"""

from __future__ import annotations

import argparse
import logging
import re
import sys

from evalbridge import data
from evalbridge.judges import BASELINES, RecordedJudge
from evalbridge.llm import JudgeError
from evalbridge.report import build_report_card, write_report

JUDGES = [*BASELINES, "gpt4-recorded", "llm"]


def _make_judge(name: str, fixture: bool):
    if name in BASELINES:
        return BASELINES[name]()
    if name == "gpt4-recorded":
        return RecordedJudge(data.load_split("gpt4_pair", fixture=fixture))
    from evalbridge.llm import LLMJudge

    return LLMJudge.from_env()


def cmd_download(args: argparse.Namespace) -> int:
    for split in data.SPLIT_FILES if args.split == "all" else [args.split]:
        path = data.download(split, force=args.force)
        print(f"{split}: {len(data.load_judgments(path))} rows -> {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    if args.data:
        df, source = data.load_judgments(args.data), str(args.data)
    elif args.fixture:
        df, source = data.load_split("human", fixture=True), "synthetic fixture (not real data)"
    else:
        df, source = data.load_split("human"), f"{data.HF_REPO}@{data.HF_REVISION[:7]}"
    items, votes = data.build_items(df, turn=args.turn)
    items = data.sample_items(items, args.limit, seed=args.seed)
    judge = _make_judge(args.judge, args.fixture)
    try:
        card = build_report_card(
            judge,
            items,
            votes,
            data_info={
                "source": source,
                "split": "human",
                "turn": args.turn,
                "limit": args.limit,
                "sample_seed": args.seed,
            },
            swap=not args.no_swap,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    finally:
        if hasattr(judge, "close"):
            print(f"LLM calls made: {judge.calls}")
            judge.close()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", judge.name) + f"-turn{args.turn}"
    if args.limit:
        stem += f"-n{args.limit}"
    md, js = write_report(card, args.out, stem)
    a = card["result"]["agreement"]
    print(
        f"{judge.name}: agreement {a['agreement_with_ties']['estimate']:.3f}, "
        f"kappa {a['cohen_kappa']['estimate']:.3f} on {card['result']['n_votes_scored']} votes"
    )
    print(f"Wrote {md} and {js}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="evalbridge")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("download", help="Download MT-Bench human judgments from Hugging Face")
    p.add_argument("--split", choices=[*data.SPLIT_FILES, "all"], default="all")
    p.add_argument("--force", action="store_true", help="Download again even if cached")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("report", help="Write a judge report card (markdown + JSON)")
    p.add_argument("--judge", choices=JUDGES, required=True)
    p.add_argument("--limit", type=int, help="Judge a seeded random sample of N items")
    p.add_argument("--turn", type=int, choices=[1, 2], default=1)
    p.add_argument("--out", default="reports")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--fixture", action="store_true", help="Use the synthetic test fixture")
    src.add_argument("--data", help="A parquet or .jsonl(.gz) file in the MT-Bench schema")
    p.add_argument("--no-swap", action="store_true", help="Judge one order only (half the cost)")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except JudgeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
