import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from antibody_pretraining.dense_shard import (
    DenseShardRecord,
    write_dense_shard,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materialize_antibody_ddp_parity_split.py"
SPEC = importlib.util.spec_from_file_location("parity_split", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SEQUENCE = "ACDEFGHIKLMNPQRSTVWY" * 4


def test_split_is_complete_disjoint_and_permutation_invariant() -> None:
    keys = np.asarray([91, 17, 73, 42, 11, 29, 101], dtype=np.uint64)
    training, evaluation, _ = MODULE.split_indices(
        keys,
        namespace="frozen-split",
        evaluation_count=3,
    )
    permutation = np.asarray([4, 0, 6, 1, 5, 3, 2], dtype=np.int64)
    permuted_training, permuted_evaluation, _ = MODULE.split_indices(
        keys[permutation],
        namespace="frozen-split",
        evaluation_count=3,
    )
    assert not set(training.tolist()) & set(evaluation.tolist())
    assert sorted(np.concatenate((training, evaluation)).tolist()) == list(
        range(len(keys))
    )
    assert set(keys[evaluation].tolist()) == set(
        keys[permutation][permuted_evaluation].tolist()
    )
    assert set(keys[training].tolist()) == set(
        keys[permutation][permuted_training].tolist()
    )


def test_split_rejects_duplicate_keys_and_invalid_count() -> None:
    with pytest.raises(RuntimeError, match="unique"):
        MODULE.split_indices(
            np.asarray([1, 1], dtype=np.uint64),
            namespace="frozen-split",
            evaluation_count=1,
        )
    with pytest.raises(RuntimeError, match="nonempty split"):
        MODULE.split_indices(
            np.asarray([1, 2], dtype=np.uint64),
            namespace="frozen-split",
            evaluation_count=2,
        )


def test_materializer_is_create_only_and_source_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_root = tmp_path / "shard"
    manifest = write_dense_shard(
        shard_root,
        [
            DenseShardRecord("heavy", key, SEQUENCE)
            for key in range(1, 11)
        ],
        maximum_sequence_tokens=96,
        provenance={"pilot": True},
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "format_version": (
                    "antibody_ddp_scientific_parity_gate_v2"
                ),
                "pilot_split": {
                    "dense_manifest_payload_sha256": manifest[
                        "manifest_sha256"
                    ],
                    "evaluation_count": 3,
                    "expected_record_count": 10,
                    "ranking_digest": (
                        "sha256(namespace_ascii_bytes || 0x00 || "
                        "sample_key_uint64_little_endian)"
                    ),
                    "selection": "lowest_ranking_digest_without_ties",
                    "split_namespace": "synthetic-frozen-split",
                    "training_count": 7,
                },
                "test_access": {
                    "downstream_labels_used": False,
                    "sabdab2_heldout_accessed": False,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    output = tmp_path / "split"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--shard",
            str(shard_root),
            "--manifest-sha256",
            manifest["manifest_sha256"],
            "--protocol",
            str(protocol_path),
            "--output",
            str(output),
        ],
    )
    MODULE.main()
    receipt = json.loads(
        (output / "receipt.json").read_text(encoding="ascii")
    )
    assert receipt["accounting"] == {
        "complete": True,
        "disjoint": True,
        "evaluation_count": 3,
        "record_count": 10,
        "training_count": 7,
    }
    assert receipt["source"]["materializer_sha256"] == (
        MODULE.file_sha256(SCRIPT)
    )
    assert np.load(
        output / "training_indices.npy",
        allow_pickle=False,
    ).shape == (7,)
    assert np.load(
        output / "evaluation_indices.npy",
        allow_pickle=False,
    ).shape == (3,)
    with pytest.raises(RuntimeError, match="create-only"):
        MODULE.main()
