"""Test complete primary OAS production-corpus finalization."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import stat
import sys
from pathlib import Path

import pytest

import antibody_pretraining.production_materialization as record_module
import antibody_pretraining.tokenizer as tokenizer_module
from antibody_pretraining.production_materialization import (
    ProductionMapRecord,
    pack_map_record,
)

ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "scripts/finalize_oas_production_corpus.py"
REDUCER = ROOT / "scripts/reduce_oas_production_bucket.py"
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
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


def _self_hashed(value: dict[str, object], field: str) -> bytes:
    value[field] = hashlib.sha256(_canonical(value)).hexdigest()
    return _canonical(value)


def _target_bucket_sequences() -> dict[int, str]:
    selected: dict[int, str] = {}
    prefix = "A" * 76
    for suffix in itertools.product(RESIDUES, repeat=4):
        sequence = prefix + "".join(suffix)
        digest = hashlib.sha256(sequence.encode("ascii")).digest()
        if digest[0] in {0, 1}:
            selected.setdefault(digest[0], sequence)
        if set(selected) == {0, 1}:
            return selected
    raise AssertionError("The deterministic search did not cover both buckets")


def _record(sequence: str, *, ordinal: int, split: str) -> ProductionMapRecord:
    return ProductionMapRecord(
        authority_ordinal=ordinal,
        chain_type="heavy",
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        split=split,
        status_flags=(),
        stratum="human_heavy",
        study_id=0,
    )


def _freeze(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(stat.S_IRWXU)
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(stat.S_IRWXU)
        else:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _fixture(tmp_path: Path):
    reducer = _load_module(REDUCER, f"oas_reducer_{tmp_path.name}")
    finalizer = _load_module(FINALIZER, f"oas_finalizer_{tmp_path.name}")
    output = tmp_path / "corpus"
    mapper = output / "map/mapper-0000"
    bucket_root = mapper / "buckets"
    bucket_root.mkdir(parents=True)
    sequences = _target_bucket_sequences()
    records = {
        0: _record(sequences[0], ordinal=0, split="train"),
        1: _record(sequences[1], ordinal=1, split="validation"),
    }
    inventory = []
    for bucket, record in records.items():
        path = bucket_root / f"bucket-{bucket:03d}.bin"
        payload = pack_map_record(record)
        path.write_bytes(payload)
        inventory.append(
            {
                "bucket": bucket,
                "bytes": len(payload),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "records": 1,
            }
        )
    protocol = {
        "execution": {
            "bucket_count": 2,
            "map_task_count": 1,
            "reduce_task_count": 2,
        },
        "format_version": "oas_production_materialization_protocol_v1",
        "inputs": {"source_record_count": 2},
        "resource_policy": {"shared_storage_minimum_free_bytes": 0},
        "test_access": {
            "downstream_metrics_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_bytes = _canonical(protocol)
    protocol_path.write_bytes(protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    map_receipt = {
        "audit_bindings": {
            "download_audit_file_sha256": "1" * 64,
            "download_audit_payload_sha256": "2" * 64,
            "filter_policy_file_sha256": "3" * 64,
            "plan_file_sha256": "4" * 64,
            "plan_payload_sha256": "5" * 64,
            "protocol_file_sha256": protocol_sha256,
        },
        "bucket_inventory": inventory,
        "claim_boundary": {
            "global_dedup_complete": False,
            "large_scale_pretraining_authorized": False,
            "map_task_complete": True,
            "pilot_only": False,
            "scientific_claim": False,
        },
        "counts": {"accepted_records": 2, "source_rows": 2},
        "format_version": "oas_production_map_task_receipt_v1",
        "source_sha256": {
            "controller": "6" * 64,
            "map_record": "7" * 64,
            "preprocessing": "8" * 64,
        },
        "study_count": 1,
        "study_vocabulary_sha256": "9" * 64,
        "task_count": 1,
        "task_id": 0,
        "test_access": {
            "downstream_metrics_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    (mapper / "receipt.json").write_bytes(_self_hashed(map_receipt, "receipt_sha256"))
    _freeze(mapper)
    source_sha256 = {
        "controller": hashlib.sha256(REDUCER.read_bytes()).hexdigest(),
        "record": hashlib.sha256(Path(record_module.__file__).read_bytes()).hexdigest(),
        "tokenizer": hashlib.sha256(
            Path(tokenizer_module.__file__).read_bytes()
        ).hexdigest(),
    }
    for bucket in range(2):
        reducer.reduce_production_bucket(
            bucket=bucket,
            map_root=output,
            protocol_path=protocol_path,
            protocol_file_sha256=protocol_sha256,
            output_root=output,
            expected_controller_source_sha256=source_sha256["controller"],
            expected_record_source_sha256=source_sha256["record"],
            expected_tokenizer_source_sha256=source_sha256["tokenizer"],
        )
    return {
        "finalizer": finalizer,
        "output": output,
        "protocol": protocol_path,
        "protocol_sha256": protocol_sha256,
        "source_sha256": source_sha256,
    }


def _finalize(fixture):
    return fixture["finalizer"].finalize_production_corpus(
        output_root=fixture["output"],
        protocol_path=fixture["protocol"],
        protocol_file_sha256=fixture["protocol_sha256"],
        reducer_source_path=REDUCER,
        reducer_source_sha256=fixture["source_sha256"]["controller"],
        record_source_path=Path(record_module.__file__),
        record_source_sha256=fixture["source_sha256"]["record"],
        tokenizer_source_path=Path(tokenizer_module.__file__),
        tokenizer_source_sha256=fixture["source_sha256"]["tokenizer"],
        controller_source_sha256=hashlib.sha256(FINALIZER.read_bytes()).hexdigest(),
    )


def test_finalizer_authenticates_complete_map_reduce_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    fixture = _fixture(tmp_path)
    try:
        receipt = _finalize(fixture)
        assert receipt["counts"] == {
            "cross_split_sequence_digests": 0,
            "input_records": 2,
            "map_accepted_records": 2,
            "map_source_rows": 2,
            "training_records_excluded_by_validation": 0,
            "unique_records": 2,
            "unique_train_records": 1,
            "unique_validation_records": 1,
            "within_split_duplicate_records": 0,
        }
        assert receipt["authorization"]["large_scale_pretraining"] is False
        manifest_path = fixture["output"] / "corpus_manifest.json"
        manifest = json.loads(manifest_path.read_text("ascii"))
        assert [item["bucket"] for item in manifest["buckets"]] == [0, 1]
        assert manifest["authorization"]["production_corpus_use"] is False
        assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o444
        with pytest.raises(
            fixture["finalizer"].ProductionFinalizeError,
            match="already exists",
        ):
            _finalize(fixture)
    finally:
        _make_writable(fixture["output"])


def test_finalizer_rejects_reducer_identity_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    fixture = _fixture(tmp_path)
    receipt_path = fixture["output"] / "reduce/bucket-001/receipt.json"
    try:
        payload = json.loads(receipt_path.read_text("ascii"))
        payload.pop("receipt_sha256")
        payload["map_identity"]["study_count"] = 2
        receipt_path.chmod(0o644)
        receipt_path.write_bytes(_self_hashed(payload, "receipt_sha256"))
        receipt_path.chmod(0o444)
        with pytest.raises(
            fixture["finalizer"].ProductionFinalizeError,
            match="identity differs",
        ):
            _finalize(fixture)
        assert not (fixture["output"] / "corpus_manifest.json").exists()
    finally:
        _make_writable(fixture["output"])
