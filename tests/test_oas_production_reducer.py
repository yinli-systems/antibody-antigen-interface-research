"""Test exact OAS production reduction and fail-closed evidence checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

import antibody_pretraining.production_materialization as record_module
import antibody_pretraining.tokenizer as tokenizer_module
from antibody_pretraining.production_materialization import (
    ProductionMapRecord,
    pack_map_record,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reduce_oas_production_bucket.py"
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def _load_module():
    spec = importlib.util.spec_from_file_location("oas_production_reducer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _self_hashed(value: dict[str, object]) -> bytes:
    value["receipt_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return _canonical(value)


def _same_bucket_sequences() -> tuple[str, str, int]:
    seen: dict[int, str] = {}
    base = "A" * 78
    for first in RESIDUES:
        for second in RESIDUES:
            sequence = base + first + second
            bucket = bytes.fromhex(_digest(sequence))[0]
            previous = seen.setdefault(bucket, sequence)
            if previous != sequence:
                return previous, sequence, bucket
    raise AssertionError("The deterministic collision search did not find a bucket")


def _record(
    sequence: str,
    *,
    authority_ordinal: int,
    split: str,
    study_id: int = 0,
) -> ProductionMapRecord:
    return ProductionMapRecord(
        authority_ordinal=authority_ordinal,
        chain_type="heavy",
        sequence=sequence,
        sequence_sha256=_digest(sequence),
        split=split,
        status_flags=(),
        stratum="human_heavy",
        study_id=study_id,
    )


def _write_mapper(
    root: Path,
    *,
    mapper_id: int,
    task_count: int,
    bucket: int,
    records: list[ProductionMapRecord],
    study_count: int = 1,
) -> None:
    mapper = root / "map" / f"mapper-{mapper_id:04d}"
    buckets = mapper / "buckets"
    buckets.mkdir(parents=True)
    bucket_path = buckets / f"bucket-{bucket:03d}.bin"
    bucket_path.write_bytes(b"".join(pack_map_record(record) for record in records))
    source_sha256 = {
        "controller": "1" * 64,
        "map_record": "2" * 64,
        "preprocessing": "3" * 64,
    }
    receipt = {
        "audit_bindings": {
            "download_audit_file_sha256": "4" * 64,
            "download_audit_payload_sha256": "5" * 64,
            "filter_policy_file_sha256": "6" * 64,
            "plan_file_sha256": "7" * 64,
            "plan_payload_sha256": "8" * 64,
            "protocol_file_sha256": "9" * 64,
        },
        "bucket_inventory": [
            {
                "bucket": bucket,
                "bytes": bucket_path.stat().st_size,
                "file_sha256": hashlib.sha256(bucket_path.read_bytes()).hexdigest(),
                "records": len(records),
            }
        ],
        "claim_boundary": {
            "global_dedup_complete": False,
            "large_scale_pretraining_authorized": False,
            "map_task_complete": True,
            "pilot_only": False,
            "scientific_claim": False,
        },
        "format_version": "oas_production_map_task_receipt_v1",
        "source_sha256": source_sha256,
        "study_count": study_count,
        "study_vocabulary_sha256": "a" * 64,
        "task_count": task_count,
        "task_id": mapper_id,
        "test_access": {
            "downstream_metrics_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    (mapper / "receipt.json").write_bytes(_self_hashed(receipt))


def _fixture(tmp_path: Path):
    module = _load_module()
    first, second, bucket = _same_bucket_sequences()
    map_root = tmp_path / "materialized"
    _write_mapper(
        map_root,
        mapper_id=0,
        task_count=2,
        bucket=bucket,
        records=[
            _record(first, authority_ordinal=10, split="train"),
            _record(second, authority_ordinal=20, split="train"),
        ],
    )
    _write_mapper(
        map_root,
        mapper_id=1,
        task_count=2,
        bucket=bucket,
        records=[
            _record(first, authority_ordinal=30, split="validation"),
            _record(second, authority_ordinal=5, split="train"),
        ],
    )
    protocol = {
        "execution": {"map_task_count": 2},
        "format_version": "oas_production_materialization_protocol_v1",
        "resource_policy": {"shared_storage_minimum_free_bytes": 0},
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_bytes = _canonical(protocol)
    protocol_path.write_bytes(protocol_bytes)
    output_root = tmp_path / "output"
    output_root.mkdir()
    return {
        "bucket": bucket,
        "first": first,
        "map_root": map_root,
        "module": module,
        "output_root": output_root,
        "protocol": protocol_path,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "second": second,
    }


def _run(fixture) -> dict[str, object]:
    return fixture["module"].reduce_production_bucket(
        bucket=fixture["bucket"],
        map_root=fixture["map_root"],
        protocol_path=fixture["protocol"],
        protocol_file_sha256=fixture["protocol_sha256"],
        output_root=fixture["output_root"],
        expected_controller_source_sha256=hashlib.sha256(
            SCRIPT.read_bytes()
        ).hexdigest(),
        expected_record_source_sha256=hashlib.sha256(
            Path(record_module.__file__).read_bytes()
        ).hexdigest(),
        expected_tokenizer_source_sha256=hashlib.sha256(
            Path(tokenizer_module.__file__).read_bytes()
        ).hexdigest(),
    )


def _make_writable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(stat.S_IRWXU)
    root.chmod(stat.S_IRWXU)


def test_reduce_exactly_applies_validation_precedence_and_minimum_ordinal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    fixture = _fixture(tmp_path)
    receipt = _run(fixture)
    published = fixture["output_root"] / "reduce" / f"bucket-{fixture['bucket']:03d}"
    try:
        assert receipt["counts"] == {
            "cross_split_sequence_digests": 1,
            "input_records": 4,
            "training_records_excluded_by_validation": 1,
            "unique_records": 2,
            "unique_train_records": 1,
            "unique_validation_records": 1,
            "within_split_duplicate_records": 1,
        }
        assert receipt["claim_boundary"]["large_scale_pretraining_authorized"] is False
        assert stat.S_IMODE(published.stat().st_mode) == 0o555
        for split, expected_sequence in (
            ("train", fixture["second"]),
            ("validation", fixture["first"]),
        ):
            shard = published / split
            assert stat.S_IMODE(shard.stat().st_mode) == 0o555
            digests = np.load(shard / "sequence_sha256.npy")
            keys = np.load(shard / "sample_keys.npy")
            assert digests.shape == (1, 32)
            assert bytes(digests[0]).hex() == _digest(expected_sequence)
            assert int(keys[0]) == int.from_bytes(
                bytes.fromhex(_digest(expected_sequence))[:8],
                "big",
            )
        stored = json.loads((published / "receipt.json").read_text("ascii"))
        unsigned = dict(stored)
        claimed = unsigned.pop("receipt_sha256")
        assert hashlib.sha256(_canonical(unsigned)).hexdigest() == claimed
        assert all(
            not os.access(path, os.W_OK)
            for path in published.rglob("*")
            if path.is_file()
        )
    finally:
        _make_writable(published)


def test_equal_digest_with_different_sequence_and_sample_key_collision_fail_closed(
    tmp_path: Path,
) -> None:
    module = _load_module()
    first, second, _ = _same_bucket_sequences()
    collision_digest = "b" * 64
    records = [
        ProductionMapRecord(
            authority_ordinal=0,
            chain_type="heavy",
            sequence=first,
            sequence_sha256=collision_digest,
            split="train",
            status_flags=(),
            stratum="human_heavy",
            study_id=0,
        ),
        ProductionMapRecord(
            authority_ordinal=1,
            chain_type="heavy",
            sequence=second,
            sequence_sha256=collision_digest,
            split="train",
            status_flags=(),
            stratum="human_heavy",
            study_id=0,
        ),
    ]
    with pytest.raises(module.ProductionReduceError, match="different canonical"):
        module._deduplicate_records(records)

    seen: dict[int, str] = {}
    module._register_sample_key(seen, "00" * 8 + "11" * 24)
    with pytest.raises(module.ProductionReduceError, match="uint64 sample key"):
        module._register_sample_key(seen, "00" * 8 + "22" * 24)


def test_mismatched_mapper_identity_and_truncated_bucket_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    module = fixture["module"]
    _, _, identity = module._mapper_bucket_input(
        fixture["map_root"] / "map/mapper-0000",
        mapper_id=0,
        task_count=2,
        bucket=fixture["bucket"],
        common_identity=None,
    )
    mapper_one = fixture["map_root"] / "map/mapper-0001"
    receipt = json.loads((mapper_one / "receipt.json").read_text("ascii"))
    receipt.pop("receipt_sha256")
    receipt["study_count"] = 2
    (mapper_one / "receipt.json").write_bytes(_self_hashed(receipt))
    with pytest.raises(module.ProductionReduceError, match="identities disagree"):
        module._mapper_bucket_input(
            mapper_one,
            mapper_id=1,
            task_count=2,
            bucket=fixture["bucket"],
            common_identity=identity,
        )

    fixture = _fixture(tmp_path / "truncated")
    bucket_path = (
        fixture["map_root"]
        / "map/mapper-0001/buckets"
        / f"bucket-{fixture['bucket']:03d}.bin"
    )
    bucket_path.write_bytes(bucket_path.read_bytes()[:-1])
    with pytest.raises(module.ProductionReduceError, match="differs from its receipt"):
        module._mapper_bucket_input(
            fixture["map_root"] / "map/mapper-0001",
            mapper_id=1,
            task_count=2,
            bucket=fixture["bucket"],
            common_identity=None,
        )
