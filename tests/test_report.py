from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest

from evalbridge import cli, data
from evalbridge.harness import attach_to_votes, run_judge
from evalbridge.judges import FirstPositionJudge, LongerAnswerJudge, RecordedJudge
from evalbridge.llm import JudgmentCache, LLMConfig, LLMJudge
from evalbridge.report import build_report_card, render_markdown

INFO = {"source": "synthetic fixture", "split": "human", "turn": 1}


@pytest.fixture(scope="module")
def fixture_items():
    return data.build_items(data.load_split("human", fixture=True), turn=1)


def card_for(judge, fixture_items, **kwargs):
    items, votes = fixture_items
    return build_report_card(judge, items, votes, data_info=INFO, n_boot=100, **kwargs)


def test_first_position_judge_is_fully_position_biased(fixture_items):
    r = card_for(FirstPositionJudge(), fixture_items)["result"]
    assert r["position_bias"]["swap_consistency"] == 0.0
    assert r["position_bias"]["first_position_win_rate"] == 1.0
    # Order-merged verdicts are all ties, so kappa is exactly 0.
    assert r["agreement"]["cohen_kappa"]["estimate"] == pytest.approx(0.0)


def test_longer_judge_is_fully_length_biased_and_order_consistent(fixture_items):
    card = card_for(LongerAnswerJudge(), fixture_items)
    r = card["result"]
    assert r["position_bias"]["swap_consistency"] == 1.0
    assert r["length_bias"]["judge_prefers_longer"] == 1.0
    assert 0 <= r["length_bias"]["human_prefers_longer"] <= 1
    a = r["agreement"]["agreement_with_ties"]
    assert a["ci_low"] <= a["estimate"] <= a["ci_high"]
    assert card["reference"]["baselines"]["longer"]["judge_prefers_longer"] == 1.0
    assert card["data"]["n_human_votes"] == r["n_votes_scored"]


def test_recorded_judge_skips_position_bias(fixture_items):
    recorded = RecordedJudge(data.load_split("gpt4_pair", fixture=True))
    card = card_for(recorded, fixture_items)
    assert card["result"]["position_bias"] is None
    assert "tie (inconsistent)" in card["result"]["recorded_labels"]
    md = render_markdown(card)
    assert "Not measured" in md and "## Length bias" in md


def test_votes_get_the_verdict_in_their_own_orientation():
    item = data.PairItem(1, 1, "x", "y", "q", "a a", "b", 2, 1)
    results = run_judge(FirstPositionJudge(), [item], swap=True)
    votes = pd.DataFrame(
        {
            "item_key": [data.item_key(1, 1, "x", "y")] * 2,
            "question_id": [1, 1],
            "model_a": ["x", "y"],  # second voter saw the pair the other way round
            "model_b": ["y", "x"],
            "human": ["A", "A"],
        }
    )
    v = attach_to_votes(results, votes)
    # Shown either way, a first-position judge picks whatever is in position A.
    assert list(v["judge_single"]) == ["A", "A"]
    assert list(v["judge_final"]) == ["tie", "tie"]

    longer = attach_to_votes(run_judge(LongerAnswerJudge(), [item]), votes)
    assert list(longer["judge_final"]) == ["A", "B"]  # x is longer; it is B for voter 2


def test_llm_judge_end_to_end_with_cache(fixture_items, tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        prompt = json.loads(request.content)["messages"][1]["content"]
        a = prompt.split("[Start of reply A]")[1].split("[End of reply A]")[0]
        b = prompt.split("[Start of reply B]")[1].split("[End of reply B]")[0]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "[[A]]" if a > b else "[[B]]"}}]}
        )

    def judge():
        return LLMJudge(
            LLMConfig(model="mock", api_key="k"),
            cache=JudgmentCache(tmp_path / "c.sqlite"),
            transport=httpx.MockTransport(handler),
        )

    items, votes = fixture_items
    first = judge()
    card = build_report_card(first, items[:6], votes, data_info=INFO, n_boot=50)
    assert first.calls == len(calls) == 12  # 6 items x 2 orders
    assert card["judge"]["model"] == "mock" and card["judge"]["prompt_version"] == "pairwise_v1"
    first.close()

    second = judge()
    build_report_card(second, items[:6], votes, data_info=INFO, n_boot=50)
    assert second.calls == 0 and len(calls) == 12


def test_cli_report_writes_markdown_and_json(tmp_path, capsys):
    code = cli.main(
        [
            "report",
            "--judge",
            "longer",
            "--fixture",
            "--limit",
            "8",
            "--n-boot",
            "50",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    md, js = tmp_path / "longer-turn1-n8.md", tmp_path / "longer-turn1-n8.json"
    assert md.read_text().startswith("# Judge report card: `longer`")
    card = json.loads(js.read_text())
    assert card["data"]["n_items"] == 8 and card["data"]["source"].startswith("synthetic")
    assert "agreement" in capsys.readouterr().out
