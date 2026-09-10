import json

from abcite_core.visual_qa_holdout import (
    negative_index_pool,
    split_by_figure_id,
)


def _item(qid: str, fid: str, **extra) -> dict:
    row = {
        "id": qid,
        "question": f"visual question {qid}",
        "gold_figure_ids": [fid],
        "caption_bm25_should_fail": True,
        "expected_status": "answered",
    }
    row.update(extra)
    return row


def test_split_by_figure_id_no_leak_and_follows_figure_not_query():
    items = [
        _item("q-a1", "fig-a"),
        _item("q-a2", "fig-a"),
        _item("q-b", "fig-b"),
        _item("q-c", "fig-c"),
        _item("q-d", "fig-d"),
        _item("abstain", "fig-z", gold_figure_ids=[], expected_status="abstain"),
        _item("bm25-ok", "fig-e", caption_bm25_should_fail=False),
    ]
    split = split_by_figure_id(items, train_frac=0.5, seed=7)
    train_f = set(split["train_figure_ids"])
    hold_f = set(split["holdout_figure_ids"])
    assert not (train_f & hold_f)
    assert split["n_train_figures"] + split["n_holdout_figures"] == 4
    sides: dict[str, set[str]] = {}
    for it in split["train_items"]:
        sides.setdefault(it["gold_figure_ids"][0], set()).add("train")
    for it in split["holdout_items"]:
        sides.setdefault(it["gold_figure_ids"][0], set()).add("hold")
    assert sides["fig-a"] in ({"train"}, {"hold"})
    assert "fig-e" not in train_f | hold_f
    again = split_by_figure_id(items, train_frac=0.5, seed=7)
    assert split["train_figure_ids"] == again["train_figure_ids"]
    assert split["holdout_figure_ids"] == again["holdout_figure_ids"]


def test_negative_pool_excludes_holdout_gold_and_positive():
    ids = ["a", "b", "c", "d", "e"]
    pool = negative_index_pool(ids, holdout_figure_ids={"d", "e"}, positive_id="b")
    picked = {ids[i] for i in pool}
    assert picked == {"a", "c"}


def test_split_json_roundtrip_ids():
    items = [_item(f"q{i}", f"f{i}") for i in range(10)]
    split = split_by_figure_id(items, train_frac=0.8, seed=7)
    blob = json.dumps(
        {
            "train": split["train_figure_ids"],
            "hold": split["holdout_figure_ids"],
        }
    )
    loaded = json.loads(blob)
    assert loaded["train"] == split["train_figure_ids"]
    assert set(loaded["train"]) & set(loaded["hold"]) == set()
