"""Test authenticated random access over production OAS bucket shards."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from antibody_pretraining.production_corpus import (
    ProductionCorpusError,
    load_production_corpus,
)


ROOT = Path(__file__).resolve().parents[1]
FINALIZER_TEST = ROOT / "tests/test_oas_production_finalizer.py"
AUDITOR = ROOT / "scripts/audit_oas_production_corpus_independent.py"


def _load(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path):
    fixture_module = _load(FINALIZER_TEST, f"loader_fixture_{tmp_path.name}")
    auditor = _load(AUDITOR, f"loader_auditor_{tmp_path.name}")
    fixture = fixture_module._fixture(tmp_path)
    finalization = fixture_module._finalize(fixture)
    audit_result = auditor.audit_corpus(
        output_root=fixture["output"],
        protocol_path=fixture["protocol"],
        worker_count=2,
    )
    audit_path = tmp_path / "independent-audit.json"
    auditor._publish(audit_path, audit_result)
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    return fixture_module, fixture, finalization, audit_path, audit


def _open(fixture, finalization, audit_path, audit, *, split: str):
    return load_production_corpus(
        fixture["output"],
        split=split,
        expected_manifest_file_sha256=finalization["manifest_file_sha256"],
        expected_manifest_payload_sha256=finalization["manifest_payload_sha256"],
        expected_protocol_file_sha256=fixture["protocol_sha256"],
        independent_audit_path=audit_path,
        expected_independent_audit_file_sha256=hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest(),
        expected_independent_audit_payload_sha256=audit["receipt_sha256"],
    )


def test_loader_authenticates_graph_and_gathers_both_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    fixture_module, fixture, finalization, audit_path, audit = _fixture(tmp_path)
    try:
        for split in ("train", "validation"):
            corpus = _open(fixture, finalization, audit_path, audit, split=split)
            assert len(corpus) == 1
            assert corpus.contract.training_authorized is False
            batch = corpus.batch(np.asarray([0, 0], dtype=np.int64))
            assert batch["tokens.npy"].shape == (2, 194)
            assert np.array_equal(batch["tokens.npy"][0], batch["tokens.npy"][1])
            assert batch["sample_keys.npy"].shape == (2,)
            assert batch["sequence_sha256.npy"].shape == (2, 32)
            selected = corpus.batch(
                np.asarray([0, 0], dtype=np.int64),
                fields=("tokens.npy", "sample_keys.npy"),
            )
            assert set(selected) == {"tokens.npy", "sample_keys.npy"}
            assert np.array_equal(
                corpus.residue_lengths(np.asarray([0])),
                batch["lengths.npy"][:1].astype(np.int64) - 2,
            )
            with pytest.raises(ProductionCorpusError, match="fields"):
                corpus.batch(np.asarray([0]), fields=("unknown.npy",))
            with pytest.raises(ProductionCorpusError, match="outside"):
                corpus.batch(np.asarray([1], dtype=np.int64))
            with pytest.raises(ProductionCorpusError, match="one-dimensional"):
                corpus.batch(np.asarray([[0]], dtype=np.int64))
    finally:
        fixture_module._make_writable(fixture["output"])


def test_loader_rejects_file_tamper_before_numpy_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    fixture_module, fixture, finalization, audit_path, audit = _fixture(tmp_path)
    tokens = fixture["output"] / "reduce/bucket-000/train/tokens.npy"
    try:
        tokens.chmod(0o644)
        with tokens.open("ab") as handle:
            handle.write(b"tamper")
        tokens.chmod(0o444)
        with pytest.raises(ProductionCorpusError, match="Shard file differs"):
            _open(fixture, finalization, audit_path, audit, split="train")
    finally:
        fixture_module._make_writable(fixture["output"])


def test_loader_rejects_reanchored_independent_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    fixture_module, fixture, finalization, audit_path, audit = _fixture(tmp_path)
    try:
        with pytest.raises(ProductionCorpusError, match="Audit file digest"):
            load_production_corpus(
                fixture["output"],
                split="train",
                expected_manifest_file_sha256=finalization[
                    "manifest_file_sha256"
                ],
                expected_manifest_payload_sha256=finalization[
                    "manifest_payload_sha256"
                ],
                expected_protocol_file_sha256=fixture["protocol_sha256"],
                independent_audit_path=audit_path,
                expected_independent_audit_file_sha256="not-a-digest",
                expected_independent_audit_payload_sha256=audit["receipt_sha256"],
            )
    finally:
        fixture_module._make_writable(fixture["output"])


def test_schema_only_follower_open_skips_array_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    fixture_module, fixture, finalization, audit_path, audit = _fixture(tmp_path)
    import antibody_pretraining.production_corpus as corpus_module

    original_sha256 = corpus_module._file_sha256

    def reject_array_hash(path: Path) -> str:
        if path.suffix == ".npy":
            raise AssertionError("Follower open rehashed a corpus array")
        return original_sha256(path)

    try:
        monkeypatch.setattr(corpus_module, "_file_sha256", reject_array_hash)
        corpus = load_production_corpus(
            fixture["output"],
            split="train",
            expected_manifest_file_sha256=finalization["manifest_file_sha256"],
            expected_manifest_payload_sha256=finalization[
                "manifest_payload_sha256"
            ],
            expected_protocol_file_sha256=fixture["protocol_sha256"],
            independent_audit_path=audit_path,
            expected_independent_audit_file_sha256=hashlib.sha256(
                audit_path.read_bytes()
            ).hexdigest(),
            expected_independent_audit_payload_sha256=audit["receipt_sha256"],
            verify_array_file_hashes=False,
        )
        assert corpus.contract.array_file_hashes_verified is False
        assert len(corpus) == 1
    finally:
        fixture_module._make_writable(fixture["output"])
