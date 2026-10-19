from __future__ import annotations

import gzip
import json

import pytest

from evalbridge import data


@pytest.fixture(scope="module")
def human():
    return data.load_split("human", fixture=True)


@pytest.mark.parametrize(
    "label,expected",
    [
        ("model_a", "model_a"),
        (" Model_B ", "model_b"),
        ("tie", "tie"),
        ("tie (inconsistent)", "tie"),
        ("tie (bothbad)", "tie"),
    ],
)
def test_normalise_winner(label, expected):
    assert data.normalise_winner(label) == expected


def test_normalise_winner_rejects_unknown_labels():
    with pytest.raises(ValueError):
        data.normalise_winner("model_c")


def test_fixture_has_the_real_schema(human):
    assert list(human.columns) == [*data.COLUMNS, "winner_raw"]
    assert set(human["winner"]) <= {"model_a", "model_b", "tie"}
    assert set(human["turn"]) == {1, 2}
    roles = [m["role"] for m in human.loc[0, "conversation_a"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert human["judge"].str.startswith("synthetic_").all()


def test_load_judgments_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.write(json.dumps({"question_id": 1}) + "\n")
    with pytest.raises(ValueError, match="missing columns"):
        data.load_judgments(path)


def test_build_items_dedupes_pairs_shown_in_either_order(human):
    items, votes = data.build_items(human, turn=1)
    turn1 = human[human["turn"] == 1]
    assert len(votes) == len(turn1)
    keys = {data.item_key(i.question_id, 1, i.model_a, i.model_b) for i in items}
    assert len(keys) == len(items) == votes["item_key"].nunique()
    assert set(votes["human"]) <= {"A", "B", "tie"}
    # human_winner is orientation-free: it names the model.
    row = votes.iloc[0]
    expected = {"A": row["model_a"], "B": row["model_b"], "tie": "tie"}[row["human"]]
    assert row["human_winner"] == expected


def test_turn_one_and_turn_two_rendering(human):
    r = human.iloc[0]
    q, a, b, fa, fb = data.render(r.conversation_a, r.conversation_b, 1)
    assert q == r.conversation_a[0]["content"]
    assert a == fa == r.conversation_a[1]["content"]
    q2, a2, _, fa2, _ = data.render(r.conversation_a, r.conversation_b, 2)
    assert r.conversation_a[2]["content"] in q2 and r.conversation_a[0]["content"] in q2
    assert r.conversation_a[1]["content"] in a2  # turn-1 reply shown as context
    assert data.judged_reply(a2) == fa2 == r.conversation_a[3]["content"]
    assert data.judged_reply(a) == a


def test_render_rejects_mismatched_questions_and_missing_turns():
    a = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "x"}]
    b = [{"role": "user", "content": "other"}, {"role": "assistant", "content": "y"}]
    with pytest.raises(ValueError, match="different user messages"):
        data.render(a, b, 1)
    with pytest.raises(ValueError, match="turn 2"):
        data.render(a, a, 2)


def test_swapped_item_swaps_models_answers_and_lengths(human):
    item = data.build_items(human)[0][0]
    s = item.swapped()
    assert (s.model_a, s.model_b) == (item.model_b, item.model_a)
    assert (s.answer_a, s.len_a) == (item.answer_b, item.len_b)
    assert s.swapped() == item and s.pair == item.pair


def test_sample_items_is_seeded_and_keeps_order(human):
    items, _ = data.build_items(human)
    a = data.sample_items(items, 5, seed=3)
    assert a == data.sample_items(items, 5, seed=3)
    assert a != data.sample_items(items, 5, seed=4)
    assert a == [i for i in items if i in a]
    assert data.sample_items(items, None) == items


def test_split_url_is_pinned_to_a_revision():
    url = data.split_url("human")
    assert data.HF_REVISION in url and url.endswith(".parquet")


@pytest.mark.network
def test_real_dataset_downloads_with_expected_shape(tmp_path):
    df = data.load_judgments(data.download("human", tmp_path / "human.parquet"))
    assert len(df) == 3355 and df["question_id"].nunique() == 80
