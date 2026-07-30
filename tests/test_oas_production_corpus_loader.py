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
            grouped = corpus.batch_groups(
                (
                    np.asarray([0], dtype=np.int64),
                    np.asarray([0, 0], dtype=np.int64),
                ),
                fields=("tokens.npy", "sample_keys.npy"),
            )
            assert len(grouped) == 2
            assert grouped[0]["tokens.npy"].shape == (1, 194)
            assert grouped[1]["tokens.npy"].shape == (2, 194)
            assert np.array_equal(grouped[1]["tokens.npy"], selected["tokens.npy"])
            assert np.array_equal(
                corpus.residue_lengths(np.asarray([0])),
                batch["lengths.npy"][:1].astype(np.int64) - 2,
            )
            assert corpus.array_cache.open_array_count <= 128
            corpus.close()
            assert corpus.array_cache.open_array_count == 0
            with pytest.raises(ProductionCorpusError, match="fields"):
                corpus.batch(np.asarray([0]), fields=("unknown.npy",))
            with pytest.raises(ProductionCorpusError, match="outside"):
                corpus.batch(np.asarray([1], dtype=np.int64))
            with pytest.raises(ProductionCorpusError, match="one-dimensional"):
                corpus.batch(np.asarray([[0]], dtype=np.int64))
            with pytest.raises(ProductionCorpusError, match="Batch groups"):
                corpus.batch_groups(())
            with pytest.raises(ProductionCorpusError, match="Batch groups"):
                corpus.batch_groups((np.asarray([[0]], dtype=np.int64),))
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


def test_loader_enforces_bounded_array_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    fixture_module, fixture, finalization, audit_path, audit = _fixture(tmp_path)
    try:
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
            maximum_open_arrays=2,
        )
        corpus.batch(
            np.asarray([0], dtype=np.int64),
            fields=("tokens.npy", "lengths.npy", "sample_keys.npy"),
        )
        assert corpus.array_cache.open_array_count == 2
        corpus.close()
        assert corpus.array_cache.open_array_count == 0
        with pytest.raises(ProductionCorpusError, match="cache limit"):
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
                expected_independent_audit_file_sha256=hashlib.sha256(
                    audit_path.read_bytes()
                ).hexdigest(),
                expected_independent_audit_payload_sha256=audit[
                    "receipt_sha256"
                ],
                maximum_open_arrays=0,
            )
    finally:
        fixture_module._make_writable(fixture["output"])


def test_grouped_batch_preserves_cross_shard_order(tmp_path: Path) -> None:
    import antibody_pretraining.production_corpus as corpus_module

    shards = []
    for shard_id, token_offset in enumerate((10, 20)):
        root = tmp_path / f"shard-{shard_id}"
        root.mkdir()
        arrays = {
            "lengths.npy": np.asarray([4, 5], dtype=np.uint16),
            "tokens.npy": np.asarray(
                [
                    [token_offset, token_offset + 1],
                    [token_offset + 2, token_offset + 3],
                ],
                dtype=np.uint8,
            ),
        }
        references = {}
        for name, values in arrays.items():
            path = root / name
            np.save(path, values, allow_pickle=False)
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
            references[name] = corpus_module._ArrayReference(
                dtype=mapped.dtype,
                path=path,
                shape=tuple(mapped.shape),
            )
            corpus_module._close_memmap(mapped)
        shards.append(
            corpus_module._Shard(
                arrays=references,
                bucket=shard_id,
                record_count=2,
                root=root,
            )
        )
    corpus = corpus_module.ProductionCorpus(
        contract=corpus_module.ProductionCorpusContract(
            independent_audit_file_sha256="1" * 64,
            independent_audit_payload_sha256="2" * 64,
            manifest_file_sha256="3" * 64,
            manifest_payload_sha256="4" * 64,
            protocol_file_sha256="5" * 64,
            split="train",
        ),
        manifest={},
        offsets=np.asarray([0, 2, 4], dtype=np.int64),
        shards=tuple(shards),
        array_cache=corpus_module._BoundedArrayCache(2),
    )
    try:
        grouped = corpus.batch_groups(
            (
                np.asarray([3, 0], dtype=np.int64),
                np.asarray([2, 1], dtype=np.int64),
            ),
            fields=("lengths.npy", "tokens.npy"),
        )
        assert grouped[0]["lengths.npy"].tolist() == [5, 4]
        assert grouped[1]["lengths.npy"].tolist() == [4, 5]
        assert grouped[0]["tokens.npy"].tolist() == [[22, 23], [10, 11]]
        assert grouped[1]["tokens.npy"].tolist() == [[20, 21], [12, 13]]
        assert corpus.array_cache.open_array_count == 2
    finally:
        corpus.close()
