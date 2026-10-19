from __future__ import annotations

import pandas as pd
import pytest

from evalbridge.judges import (
    FirstPositionJudge,
    ItemRef,
    Judge,
    LongerAnswerJudge,
    RecordedJudge,
)


def test_baselines_satisfy_the_protocol():
    for judge in (LongerAnswerJudge(), FirstPositionJudge()):
        assert isinstance(judge, Judge)


def test_longer_answer_judge():
    j = LongerAnswerJudge()
    assert j.judge("q", "one two three", "one") == "A"
    assert j.judge("q", "one", "one two") == "B"
    assert j.judge("q", "one two", "three four") == "tie"


def test_first_position_judge_ignores_content():
    assert FirstPositionJudge().judge("q", "", "a much longer answer") == "A"


def recorded():
    return pd.DataFrame(
        {
            "question_id": [1, 1, 2],
            "turn": [1, 1, 1],
            "model_a": ["x", "x", "x"],
            "model_b": ["y", "z", "y"],
            "winner": ["model_b", "tie", "model_a"],
            "winner_raw": ["model_b", "tie (inconsistent)", "model_a"],
        }
    )


def test_recorded_judge_follows_the_winning_model_in_either_order():
    j = RecordedJudge(recorded())
    assert j.judge("q", "a", "b", ref=ItemRef(1, 1, "x", "y")) == "B"
    assert j.judge("q", "a", "b", ref=ItemRef(1, 1, "y", "x")) == "A"
    assert j.judge("q", "a", "b", ref=ItemRef(1, 1, "z", "x")) == "tie"
    assert j.raw_labels["tie (inconsistent)"] == 1


def test_recorded_judge_errors():
    j = RecordedJudge(recorded())
    with pytest.raises(ValueError, match="needs ref"):
        j.judge("q", "a", "b")
    with pytest.raises(LookupError):
        j.judge("q", "a", "b", ref=ItemRef(9, 1, "x", "y"))
    conflicting = pd.concat([recorded(), recorded().iloc[[0]].assign(winner="model_a")])
    with pytest.raises(ValueError, match="conflicting"):
        RecordedJudge(conflicting)
