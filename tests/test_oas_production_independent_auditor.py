"""Test the independent OAS map-bucket and reduced-shard replay."""

from __future__ import annotations

import ast
import json
import importlib.util
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_oas_production_corpus_independent.py"
FINALIZER_TEST = ROOT / "tests/test_oas_production_finalizer.py"


def _load(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_independent_auditor_replays_complete_bucket_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild representatives without importing mapper, reducer, or record code."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    fixture_module = _load(FINALIZER_TEST, "independent_audit_fixture")
    auditor = _load(AUDITOR, "independent_oas_auditor")
    fixture = fixture_module._fixture(tmp_path)
    try:
        fixture_module._finalize(fixture)
        result = auditor.audit_corpus(
            output_root=fixture["output"],
            protocol_path=fixture["protocol"],
            worker_count=2,
        )
        assert result["authorization"]["independent_replay"] is True
        assert result["authorization"]["production_corpus_use"] is False
        assert result["counts"]["input_records"] == 2
        assert result["counts"]["unique_records"] == 2
        assert [item["bucket"] for item in result["buckets"]] == [0, 1]
    finally:
        fixture_module._make_writable(fixture["output"])


def test_v2_independent_auditor_replays_hidden_gpu_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay the exact scheduler-workaround receipt chain end to end."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", "1")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    fixture_module = _load(FINALIZER_TEST, "independent_v2_fixture")
    auditor = _load(AUDITOR, "independent_v2_auditor")
    fixture = fixture_module._fixture(
        tmp_path,
        protocol_version="v2",
        pretty_protocol=True,
    )
    try:
        fixture_module._finalize(fixture)
        result = auditor.audit_corpus(
            output_root=fixture["output"],
            protocol_path=fixture["protocol"],
            worker_count=2,
        )
        assert result["authorization"]["independent_replay"] is True
        assert result["resource_receipt"]["scheduler_reserved_gpu_count"] == 1
        assert result["resource_receipt"]["gpu_compute_permitted"] is False
    finally:
        fixture_module._make_writable(fixture["output"])


def test_independent_decoder_rejects_digest_tamper() -> None:
    """Detect a sequence payload that no longer matches its full digest."""
    auditor = _load(AUDITOR, "independent_oas_decoder")
    sequence = b"A" * 80
    digest = b"\x00" * 32
    payload = auditor.MAP_RECORD.pack(
        digest,
        0,
        1,
        1,
        0,
        0,
        0,
        len(sequence),
        sequence.ljust(auditor.MAXIMUM_SEQUENCE_LENGTH, b"\0"),
    )
    with pytest.raises(auditor.IndependentCorpusAuditError, match="digest differs"):
        auditor._decode_record(payload, bucket=0, study_count=1)


def test_v3_mapper_accounting_accepts_only_an_inferable_legacy_zero() -> None:
    """Treat an omitted rejected counter as zero only when source equality proves it."""
    auditor = _load(AUDITOR, "independent_v3_mapper_count_compatibility")
    legacy = {
        "accepted_records": 8,
        "plan_fraction_selected": 10,
        "quality_rejected:invalid_sequence": 2,
        "source_rows": 10,
    }
    observed = auditor._validated_mapper_counts(
        legacy,
        protocol_format="oas_production_materialization_protocol_v3",
    )
    assert observed["plan_fraction_rejected"] == 0
    with pytest.raises(
        auditor.IndependentCorpusAuditError,
        match="row accounting differs",
    ):
        auditor._validated_mapper_counts(
            {
                "accepted_records": 8,
                "plan_fraction_selected": 10,
                "quality_rejected:invalid_sequence": 2,
                "source_rows": 11,
            },
            protocol_format="oas_production_materialization_protocol_v3",
        )


def test_v3_mapper_accounting_requires_selected_counter() -> None:
    """Do not generalize the legacy-zero exception to missing selected evidence."""
    auditor = _load(AUDITOR, "independent_v3_mapper_selected_required")
    with pytest.raises(
        auditor.IndependentCorpusAuditError,
        match="row accounting differs",
    ):
        auditor._validated_mapper_counts(
            {
                "accepted_records": 8,
                "plan_fraction_rejected": 2,
                "source_rows": 10,
            },
            protocol_format="oas_production_materialization_protocol_v3",
        )


def test_independent_auditor_rejects_reanchored_manifest_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a self-consistent manifest that no longer names the reducer bytes."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    fixture_module = _load(FINALIZER_TEST, "independent_manifest_tamper_fixture")
    auditor = _load(AUDITOR, "independent_manifest_tamper_auditor")
    fixture = fixture_module._fixture(tmp_path)
    manifest_path = fixture["output"] / "corpus_manifest.json"
    try:
        fixture_module._finalize(fixture)
        fixture_module._make_writable(fixture["output"])
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        manifest.pop("manifest_sha256")
        manifest["buckets"][0]["receipt_file_sha256"] = "0" * 64
        manifest_path.write_bytes(
            fixture_module._self_hashed(manifest, "manifest_sha256")
        )
        fixture_module._freeze(fixture["output"])
        with pytest.raises(
            auditor.IndependentCorpusAuditError,
            match="manifest bucket reference differs",
        ):
            auditor.audit_corpus(
                output_root=fixture["output"],
                protocol_path=fixture["protocol"],
                worker_count=2,
            )
    finally:
        fixture_module._make_writable(fixture["output"])


def test_auditor_source_does_not_import_production_implementations() -> None:
    """Preserve implementation independence from all three primary controllers."""
    source = AUDITOR.read_text(encoding="ascii")
    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    forbidden = {
        "antibody_pretraining.production_materialization",
        "antibody_pretraining.tokenizer",
        "scripts.map_oas_production_records",
        "scripts.reduce_oas_production_bucket",
    }
    assert forbidden.isdisjoint(imported)
    assert stat.S_IMODE(AUDITOR.stat().st_mode) & stat.S_IXUSR
