"""Adversarial tests for digest-excluded production-corpus shards."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import stat
import sys

import numpy as np
import pytest

from antibody_pretraining.decontaminated_corpus import (
    DecontaminatedCorpusError,
    file_sha256,
    finalize_corpus,
    filter_shard,
    filter_task,
    load_signed_json,
    publish_signed_json,
)
from antibody_pretraining.production_corpus import EXPECTED_ARRAYS, load_production_corpus


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = ROOT / "scripts/audit_oas_decontaminated_corpus_independent.py"
AUDITOR_SPEC = importlib.util.spec_from_file_location(
    "independent_decontamination_auditor_test", AUDITOR_PATH
)
assert AUDITOR_SPEC is not None and AUDITOR_SPEC.loader is not None
AUDITOR_MODULE = importlib.util.module_from_spec(AUDITOR_SPEC)
sys.modules[AUDITOR_SPEC.name] = AUDITOR_MODULE
AUDITOR_SPEC.loader.exec_module(AUDITOR_MODULE)
FREEZER_PATH = ROOT / "scripts/freeze_oas_decontamination_protocol.py"
FREEZER_SPEC = importlib.util.spec_from_file_location(
    "decontamination_protocol_freezer_test", FREEZER_PATH
)
assert FREEZER_SPEC is not None and FREEZER_SPEC.loader is not None
FREEZER_MODULE = importlib.util.module_from_spec(FREEZER_SPEC)
sys.modules[FREEZER_SPEC.name] = FREEZER_MODULE
FREEZER_SPEC.loader.exec_module(FREEZER_MODULE)


def _digest(bucket: int, suffix: int) -> bytes:
    return bytes([bucket]) + suffix.to_bytes(31, "big")


def _freeze(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o400)
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o500)
    root.chmod(0o500)


def _write_source_shard(
    source_root: Path,
    *,
    bucket: int,
    split: str,
    digests: list[bytes],
) -> dict[str, object]:
    shard = source_root / f"reduce/bucket-{bucket:03d}/{split}"
    shard.mkdir(parents=True)
    record_count = len(digests)
    arrays = {
        "chain_types.npy": np.asarray(
            [(index % 3) + 1 for index in range(record_count)], dtype=np.uint8
        ),
        "lengths.npy": np.arange(82, 82 + record_count, dtype=np.uint16),
        "sample_keys.npy": np.arange(10, 10 + record_count, dtype=np.uint64),
        "sequence_lengths.npy": np.arange(80, 80 + record_count, dtype=np.uint16),
        "sequence_sha256.npy": np.frombuffer(b"".join(digests), dtype=np.uint8).reshape(
            record_count, 32
        ),
        "status_flags.npy": np.arange(record_count, dtype=np.uint8),
        "strata.npy": np.asarray(
            [index % 4 for index in range(record_count)], dtype=np.uint8
        ),
        "study_ids.npy": np.arange(4, 4 + record_count, dtype=np.uint16),
        "tokens.npy": np.arange(record_count * 194, dtype=np.uint8).reshape(
            record_count, 194
        ),
    }
    assert set(arrays) == EXPECTED_ARRAYS
    files = {}
    for name, array in arrays.items():
        path = shard / name
        np.save(path, array, allow_pickle=False)
        files[name] = {
            "bytes": path.stat().st_size,
            "dtype": str(array.dtype),
            "sha256": file_sha256(path),
            "shape": list(array.shape),
        }
    publish_signed_json(
        shard / "manifest.json",
        {
            "bucket": bucket,
            "files": files,
            "format_version": "oas_production_dense_bucket_shard_v1",
            "global_order": "sequence_sha256_ascending",
            "maximum_sequence_tokens": 194,
            "record_count": record_count,
            "split": split,
            "test_access": {
                "downstream_metrics_used": False,
                "sabdab2_heldout_accessed": False,
            },
        },
        digest_field="manifest_sha256",
    )
    manifest, manifest_file, manifest_payload = load_signed_json(
        shard / "manifest.json",
        digest_field="manifest_sha256",
        label="fixture shard",
    )
    _freeze(shard)
    reference = {
        "manifest_file_sha256": manifest_file,
        "manifest_payload_sha256": manifest_payload,
        "record_count": record_count,
        "relative_root": f"reduce/bucket-{bucket:03d}/{split}",
    }
    assert manifest["record_count"] == record_count
    return reference


def _source_fixture(
    tmp_path: Path, *, bucket: int = 5, unsorted: bool = False
) -> tuple[Path, dict[str, object], list[bytes]]:
    source_root = tmp_path / "source"
    digests = [_digest(bucket, value) for value in (1, 2, 3, 4)]
    if unsorted:
        digests[1], digests[2] = digests[2], digests[1]
    reference = _write_source_shard(
        source_root,
        bucket=bucket,
        split="train",
        digests=digests,
    )
    return source_root, reference, digests


def _exclusions(tmp_path: Path, digests: list[bytes]) -> Path:
    path = tmp_path / "excluded.bin"
    path.write_bytes(b"".join(sorted(digests)))
    path.chmod(0o400)
    return path


def test_filter_shard_removes_exact_digest_membership_and_keeps_rows_aligned(
    tmp_path: Path,
) -> None:
    source_root, reference, digests = _source_fixture(tmp_path)
    exclusion = _exclusions(
        tmp_path,
        [_digest(1, 9), digests[1], digests[3], _digest(6, 0)],
    )
    output = tmp_path / "candidate/bucket-005/train"
    output.parent.mkdir(parents=True)
    result = filter_shard(
        source_root=source_root,
        source_reference=reference,
        output_root=output,
        exclusion_path=exclusion,
        exclusion_record_count=4,
        bucket=5,
        split="train",
        chunk_records=2,
    )

    assert result["source_record_count"] == 4
    assert result["excluded_records"] == 2
    assert result["record_count"] == 2
    assert result["source_residue_tokens"] == 326
    assert result["excluded_residue_tokens"] == 164
    assert result["residue_token_count"] == 162
    observed_digests = np.load(output / "sequence_sha256.npy", allow_pickle=False)
    assert observed_digests.tobytes() == digests[0] + digests[2]
    assert np.load(output / "sample_keys.npy", allow_pickle=False).tolist() == [10, 12]
    tokens = np.load(output / "tokens.npy", allow_pickle=False)
    assert tokens.shape == (2, 194)
    expected_tokens = np.arange(2 * 194, 3 * 194, dtype=np.int64).astype(np.uint8)
    assert np.array_equal(tokens[1], expected_tokens)
    assert not output.stat().st_mode & stat.S_IWUSR
    assert hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest() == result[
        "manifest_file_sha256"
    ]


def test_filter_shard_accepts_an_empty_exclusion_bucket(tmp_path: Path) -> None:
    source_root, reference, digests = _source_fixture(tmp_path)
    exclusion = _exclusions(tmp_path, [_digest(1, 1), _digest(6, 1)])
    output = tmp_path / "candidate/bucket-005/train"
    output.parent.mkdir(parents=True)
    result = filter_shard(
        source_root=source_root,
        source_reference=reference,
        output_root=output,
        exclusion_path=exclusion,
        exclusion_record_count=2,
        bucket=5,
        split="train",
    )
    assert result["excluded_records"] == 0
    assert np.load(output / "sequence_sha256.npy", allow_pickle=False).tobytes() == b"".join(
        digests
    )


def test_filter_shard_rejects_signed_but_unsorted_source_and_cleans_partial(
    tmp_path: Path,
) -> None:
    source_root, reference, _ = _source_fixture(tmp_path, unsorted=True)
    exclusion = _exclusions(tmp_path, [_digest(5, 2)])
    output = tmp_path / "candidate/bucket-005/train"
    output.parent.mkdir(parents=True)
    with pytest.raises(DecontaminatedCorpusError, match="digest order"):
        filter_shard(
            source_root=source_root,
            source_reference=reference,
            output_root=output,
            exclusion_path=exclusion,
            exclusion_record_count=1,
            bucket=5,
            split="train",
        )
    assert not output.exists()
    assert not list(output.parent.glob("train.partial-*"))


def test_decontamination_launcher_hides_scheduler_mandated_gpu() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "slurm/oas_decontaminated_corpus_array.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --array=0-31%32" in launcher
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "#SBATCH --cpus-per-task=8" in launcher
    assert "export CUDA_VISIBLE_DEVICES=" in launcher
    assert "export OAS_SCHEDULER_RESERVED_GPU_COUNT=1" in launcher
    assert "--task-count 32" in launcher
    assert "--bucket-count 256" in launcher
    assert "OAS_DECONTAMINATION_PROTOCOL_SHA256" in launcher
    finalizer = (
        ROOT / "slurm/oas_decontaminated_corpus_finalize.sbatch"
    ).read_text(encoding="utf-8")
    auditor = (
        ROOT / "slurm/oas_decontaminated_corpus_independent_audit.sbatch"
    ).read_text(encoding="utf-8")
    for script in (finalizer, auditor):
        assert "#SBATCH --gres=gpu:1" in script
        assert "#SBATCH --cpus-per-task=8" in script
        assert "export CUDA_VISIBLE_DEVICES=" in script
        assert "export OAS_SCHEDULER_RESERVED_GPU_COUNT=1" in script
    assert "--finalizer-source-sha256" in finalizer
    assert "--auditor-source-sha256" in auditor
    assert "--worker-count 8" in auditor
    assert "from antibody_pretraining" not in auditor
    assert "import antibody_pretraining" not in auditor


def test_filter_task_authenticates_graph_and_covers_every_protocol_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source-corpus"
    bucket_references = []
    source_counts = {"train": 0, "validation": 0}
    for bucket in range(2):
        shards = {}
        for split in ("train", "validation"):
            digests = [
                _digest(bucket, bucket * 100 + (10 if split == "train" else 20) + index)
                for index in range(3)
            ]
            shards[split] = _write_source_shard(
                source_root,
                bucket=bucket,
                split=split,
                digests=digests,
            )
            source_counts[split] += len(digests)
        bucket_references.append({"bucket": bucket, "shards": shards})
    finalizer_source = ROOT / "scripts/finalize_oas_decontaminated_corpus.py"
    publish_signed_json(
        source_root / "corpus_manifest.json",
        {
            "authorization": {
                "large_scale_pretraining": False,
                "production_corpus_use": False,
                "scientific_claim": False,
            },
            "bucket_count": 2,
            "buckets": bucket_references,
            "counts": {
                "unique_train_records": source_counts["train"],
                "unique_validation_records": source_counts["validation"],
            },
            "format_version": "oas_production_corpus_manifest_v1",
            "global_order": "bucket_ascending_then_sequence_sha256_ascending",
            "map_identity": {"study_count": 8},
            "protocol_file_sha256": "a" * 64,
            "test_access": {
                "downstream_metrics_used": False,
                "sabdab2_heldout_accessed": False,
            },
        },
        digest_field="manifest_sha256",
    )
    source_manifest, source_manifest_file, source_manifest_payload = load_signed_json(
        source_root / "corpus_manifest.json",
        digest_field="manifest_sha256",
        label="source manifest",
    )
    publish_signed_json(
        source_root / "independent_audit_receipt.json",
        {
            "authorization": {
                "independent_replay": True,
                "large_scale_pretraining": False,
                "production_corpus_use": False,
                "scientific_claim": False,
            },
            "format_version": "oas_production_corpus_independent_audit_v1",
            "inputs": {
                "manifest_file_sha256": source_manifest_file,
                "manifest_payload_sha256": source_manifest_payload,
                "protocol_file_sha256": source_manifest["protocol_file_sha256"],
            },
            "test_access": {
                "downstream_metrics_used": False,
                "sabdab2_heldout_accessed": False,
            },
        },
        digest_field="receipt_sha256",
    )
    _, source_audit_file, source_audit_payload = load_signed_json(
        source_root / "independent_audit_receipt.json",
        digest_field="receipt_sha256",
        label="source audit",
    )
    exclusion_root = tmp_path / "exclusions"
    exclusion_root.mkdir()
    excluded = [_digest(0, 10), _digest(1, 120)]
    exclusion_stream = exclusion_root / "excluded_sequence_sha256.bin"
    exclusion_stream.write_bytes(b"".join(excluded))
    exclusion_stream.chmod(0o400)
    publish_signed_json(
        exclusion_root / "receipt.json",
        {
            "claim_boundary": {
                "near_identity_exclusion_list_complete": True,
                "original_training_corpus_filtered": False,
            },
            "format_version": "oas_near_identity_exclusion_compilation_v1",
            "output": {
                "excluded_sequence_sha256_bytes": exclusion_stream.stat().st_size,
                "excluded_sequence_sha256_file_sha256": file_sha256(exclusion_stream),
                "excluded_sequence_sha256_record_bytes": 32,
                "excluded_sequence_sha256_records": len(excluded),
            },
            "test_access": {
                "development_sequence_fields_accessed": True,
                "frozen_official_external_test_accessed": False,
                "model_predictions_or_scores_accessed": False,
                "task_label_values_used": False,
            },
        },
        digest_field="receipt_sha256",
    )
    _, exclusion_file, exclusion_payload = load_signed_json(
        exclusion_root / "receipt.json",
        digest_field="receipt_sha256",
        label="exclusion receipt",
    )
    protocol = tmp_path / "protocol.json"
    filter_controller = ROOT / "scripts/filter_oas_decontaminated_corpus_task.py"
    bad_exclusion_root = tmp_path / "bad-exclusions"
    bad_exclusion_root.mkdir()
    bad_exclusion_stream = bad_exclusion_root / "excluded_sequence_sha256.bin"
    bad_exclusion_stream.write_bytes(exclusion_stream.read_bytes())
    bad_exclusion_stream.chmod(0o400)
    publish_signed_json(
        bad_exclusion_root / "receipt.json",
        {
            "claim_boundary": {
                "near_identity_exclusion_list_complete": True,
                "original_training_corpus_filtered": False,
            },
            "format_version": "oas_near_identity_exclusion_compilation_v1",
            "output": {
                "excluded_sequence_sha256_bytes": bad_exclusion_stream.stat().st_size,
                "excluded_sequence_sha256_file_sha256": file_sha256(
                    bad_exclusion_stream
                ),
                "excluded_sequence_sha256_record_bytes": 32,
                "excluded_sequence_sha256_records": len(excluded),
            },
            "test_access": {
                "downstream_metrics_used": False,
                "sabdab2_heldout_accessed": False,
            },
        },
        digest_field="receipt_sha256",
    )
    with pytest.raises(
        FREEZER_MODULE.DecontaminationProtocolError,
        match="evidence graph",
    ):
        FREEZER_MODULE.freeze_protocol(
            source_root=source_root,
            exclusion_root=bad_exclusion_root,
            output=tmp_path / "bad-protocol.json",
            filter_controller=filter_controller,
            filter_core=ROOT / "src/antibody_pretraining/decontaminated_corpus.py",
            finalizer=finalizer_source,
            independent_auditor=AUDITOR_PATH,
            task_count=1,
            bucket_count=2,
        )
    FREEZER_MODULE.freeze_protocol(
        source_root=source_root,
        exclusion_root=exclusion_root,
        output=protocol,
        filter_controller=filter_controller,
        filter_core=ROOT / "src/antibody_pretraining/decontaminated_corpus.py",
        finalizer=finalizer_source,
        independent_auditor=AUDITOR_PATH,
        task_count=1,
        bucket_count=2,
    )
    protocol_file_sha256 = file_sha256(protocol)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", raising=False)
    output = tmp_path / "filtered-corpus"
    output.mkdir()
    receipt = filter_task(
        task_id=0,
        task_count=1,
        bucket_count=2,
        source_root=source_root,
        exclusion_root=exclusion_root,
        output_root=output,
        protocol_path=protocol,
        expected_protocol_file_sha256=protocol_file_sha256,
        controller_source_sha256=file_sha256(filter_controller),
    )

    assert [item["bucket"] for item in receipt["buckets"]] == [0, 1]
    assert receipt["counts"] == {
        "excluded_train": 1,
        "excluded_train_residue_tokens": 80,
        "excluded_validation": 1,
        "excluded_validation_residue_tokens": 80,
        "retained_train": 5,
        "retained_train_residue_tokens": 406,
        "retained_validation": 5,
        "retained_validation_residue_tokens": 406,
        "source_train": 6,
        "source_train_residue_tokens": 486,
        "source_validation": 6,
        "source_validation_residue_tokens": 486,
    }
    assert receipt["claim_boundary"]["zero_excluded_digest_overlap_observed"]
    assert receipt["claim_boundary"]["independent_full_replay_complete"] is False
    assert not (output / "tasks/task-0000").stat().st_mode & stat.S_IWUSR

    finalized = finalize_corpus(
        output_root=output,
        protocol_path=protocol,
        expected_protocol_file_sha256=protocol_file_sha256,
        finalizer_source_path=finalizer_source,
        finalizer_source_sha256=file_sha256(finalizer_source),
    )
    assert finalized["claim_boundary"][
        "near_identity_decontamination_candidate_complete"
    ]
    assert finalized["claim_boundary"]["independent_full_replay_complete"] is False
    assert finalized["counts"]["unique_train_records"] == 5
    assert finalized["counts"]["unique_validation_records"] == 5
    assert finalized["counts"]["unique_train_residue_tokens"] == 406
    assert finalized["counts"]["unique_validation_residue_tokens"] == 406
    assert finalized["counts"]["decontamination_source_residue_tokens"] == 972
    assert finalized["counts"]["decontamination_excluded_residue_tokens"] == 160
    assert (
        output / "reduce/bucket-000/train/sequence_sha256.npy"
    ).stat().st_ino == (
        output / "tasks/task-0000/bucket-000/train/sequence_sha256.npy"
    ).stat().st_ino

    independent = AUDITOR_MODULE.audit_corpus(
        source_root=source_root,
        exclusion_root=exclusion_root,
        output_root=output,
        protocol_path=protocol,
        expected_protocol_file_sha256=protocol_file_sha256,
        auditor_source_sha256=file_sha256(AUDITOR_PATH),
        worker_count=2,
        chunk_records=2,
    )
    assert independent["authorization"]["independent_replay"] is True
    assert independent["claim_boundary"]["independent_replay_complete"] is True
    assert independent["claim_boundary"][
        "zero_excluded_digest_overlap_observed"
    ] is True
    assert independent["counts"] == finalized["counts"]
    audit_path = output / "independent_audit_receipt.json"
    AUDITOR_MODULE._publish(audit_path, independent)
    _, audit_file, audit_payload = load_signed_json(
        audit_path,
        digest_field="receipt_sha256",
        label="published independent audit",
    )
    _, manifest_file, manifest_payload = load_signed_json(
        output / "corpus_manifest.json",
        digest_field="manifest_sha256",
        label="published decontaminated corpus",
    )
    corpus = load_production_corpus(
        output,
        split="train",
        expected_manifest_file_sha256=manifest_file,
        expected_manifest_payload_sha256=manifest_payload,
        expected_protocol_file_sha256=protocol_file_sha256,
        independent_audit_path=audit_path,
        expected_independent_audit_file_sha256=audit_file,
        expected_independent_audit_payload_sha256=audit_payload,
        verify_array_file_hashes=True,
    )
    assert len(corpus) == 5
    corpus.close()

    tampered = output / "reduce/bucket-000/train/tokens.npy"
    tampered.chmod(0o600)
    payload = bytearray(tampered.read_bytes())
    payload[-1] ^= 1
    tampered.write_bytes(payload)
    tampered.chmod(0o400)
    with pytest.raises(
        AUDITOR_MODULE.IndependentDecontaminationAuditError,
        match="hash differs",
    ):
        AUDITOR_MODULE.audit_corpus(
            source_root=source_root,
            exclusion_root=exclusion_root,
            output_root=output,
            protocol_path=protocol,
            expected_protocol_file_sha256=protocol_file_sha256,
            auditor_source_sha256=file_sha256(AUDITOR_PATH),
            worker_count=2,
            chunk_records=2,
        )
