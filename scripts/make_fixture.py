"""Generate small SYNTHETIC fixtures in the MT-Bench human-judgments schema for tests and CI.

Nothing here is real: questions, answers, model names and annotators are made up. Only the
field names and value formats follow ``lmsys/mt_bench_human_judgments``.

    python scripts/make_fixture.py  ->  tests/fixtures/sample_human_judgments.jsonl.gz
                                        tests/fixtures/sample_gpt4_pair.jsonl.gz
"""

from __future__ import annotations

import gzip
import json
import random
from itertools import combinations
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

# name -> (quality, typical answer length in words)
MODELS = {
    "synth-terse": (0.2, 12),
    "synth-steady": (0.6, 40),
    "synth-verbose": (0.4, 90),
    "synth-strong": (0.9, 55),
}
TOPICS = [
    "tide pools",
    "sourdough",
    "bus timetables",
    "chess openings",
    "rain gauges",
    "bike gears",
    "library fines",
    "kite design",
]
FILLER = ["this", "synthetic", "answer", "adds", "detail", "about", "the", "topic", "in", "plain"]


def answer(rng: random.Random, model: str, topic: str, turn: int) -> str:
    _, typical = MODELS[model]
    n = max(3, int(rng.gauss(typical, typical * 0.25)))
    words = [rng.choice(FILLER) for _ in range(n)]
    return f"[synthetic {model}, turn {turn}] About {topic}: " + " ".join(words) + "."


def conversation(rng, model, topic, qid):
    return [
        {"content": f"Synthetic question {qid}: explain {topic} to a beginner.", "role": "user"},
        {"content": answer(rng, model, topic, 1), "role": "assistant"},
        {
            "content": f"Synthetic follow-up {qid}: now summarise {topic} in one line.",
            "role": "user",
        },
        {"content": answer(rng, model, topic, 2), "role": "assistant"},
    ]


def winner(rng, ma, mb, conv_a, conv_b, turn, *, length_weight: float, noise: float) -> str:
    la, lb = (len(c[2 * turn - 1]["content"].split()) for c in (conv_a, conv_b))
    score = MODELS[ma][0] - MODELS[mb][0] + length_weight * (la - lb) / 100
    score += rng.gauss(0, noise)
    if abs(score) < 0.12:
        return "tie"
    return "model_a" if score > 0 else "model_b"


def make(seed: int = 11) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    human, gpt4 = [], []
    for k, topic in enumerate(TOPICS):
        qid = 9001 + k
        convs = {m: conversation(rng, m, topic, qid) for m in MODELS}
        for m1, m2 in combinations(MODELS, 2):
            for turn in (1, 2):
                # Recorded "GPT-4" verdict: one per item, canonical order, like gpt4_pair.
                w = winner(rng, m1, m2, convs[m1], convs[m2], turn, length_weight=0.6, noise=0.2)
                if w != "tie" and rng.random() < 0.15:
                    w = "tie (inconsistent)"
                gpt4.append(row(qid, m1, m2, w, "gpt4_pair", convs, turn))
                # Human votes: about half the items, 1-3 annotators, random presentation order.
                if rng.random() < 0.5:
                    continue
                for a in rng.sample(range(6), rng.choice([1, 1, 2, 3])):
                    ma, mb = (m1, m2) if rng.random() < 0.5 else (m2, m1)
                    w = winner(
                        rng, ma, mb, convs[ma], convs[mb], turn, length_weight=0.3, noise=0.3
                    )
                    human.append(row(qid, ma, mb, w, f"synthetic_expert_{a}", convs, turn))
    return human, gpt4


def row(qid, ma, mb, w, judge, convs, turn) -> dict:
    return {
        "question_id": qid,
        "model_a": ma,
        "model_b": mb,
        "winner": w,
        "judge": judge,
        "conversation_a": convs[ma],
        "conversation_b": convs[mb],
        "turn": turn,
    }


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as gz:
        gz.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode())


if __name__ == "__main__":
    human, gpt4 = make()
    write(human, OUT / "sample_human_judgments.jsonl.gz")
    write(gpt4, OUT / "sample_gpt4_pair.jsonl.gz")
    print(f"{len(human)} synthetic human votes, {len(gpt4)} synthetic recorded verdicts")
