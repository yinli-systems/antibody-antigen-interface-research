from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "remote_tools" / "abpairfm_distributed_contact_pretraining_v1"
sys.path.insert(0, str(SOURCE))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SOURCE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHARD = _load("pinder_b32k_shard", "materialize_pinder_contact_label_shard_v1.py")
MERGE = _load("pinder_b32k_merge", "merge_pinder_contact_label_shards_v1.py")


def test_b32k_round_robin_shards_are_balanced_and_exhaustive() -> None:
    rows = [{"id": f"id-{index}"} for index in range(SHARD.SYSTEMS)]
    shards = [SHARD.select_shard(rows, index, SHARD.SHARDS) for index in range(SHARD.SHARDS)]
    assert {len(shard) for shard in shards} == {1024}
    restored = [
        shards[index % SHARD.SHARDS][index // SHARD.SHARDS]["id"]
        for index in range(SHARD.SYSTEMS)
    ]
    assert restored == [row["id"] for row in rows]


def test_merge_round_robin_rejects_reordered_shard() -> None:
    expected = ["a", "b", "c", "d"]
    handles = [
        io.StringIO("".join(json.dumps({"example_id": item}) + "\n" for item in ("a", "c"))),
        io.StringIO("".join(json.dumps({"example_id": item}) + "\n" for item in ("b", "d"))),
    ]
    assert [record["example_id"] for record in MERGE.round_robin_records(handles, expected)] == expected

    reordered = [
        io.StringIO(json.dumps({"example_id": "c"}) + "\n"),
        io.StringIO(json.dumps({"example_id": "b"}) + "\n"),
    ]
    try:
        list(MERGE.round_robin_records(reordered, ["a", "b"]))
    except RuntimeError as error:
        assert "order differs" in str(error)
    else:
        raise AssertionError("reordered shard must fail closed")
