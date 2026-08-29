"""Adversarial tests for digest-level OAS near-identity exclusions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest

from antibody_pretraining.imgt_search_output import (
    FIXED_IMGT_MATCH_RECORD,
    FixedIMGTOutputMatch,
    pack_fixed_imgt_match,
)
from antibody_pretraining.mmseqs_gap_aware import (
    TEST_ACCESS,
    target_identifier,
)
from antibody_pretraining.near_identity_exclusion import (
    EXCLUSION_DIGEST_RECORD,
    NearIdentityExclusionError,
    compile_exclusions,
)
from antibody_pretraining.near_identity_export import publish_export_receipt
from antibody_pretraining.oas_imgt_export import (
    IMGT_TARGET_RECORD,
    UNUSUAL_TARGET_RECORD,
    classify_oas_target,
    pack_imgt_target,
)
from antibody_pretraining.production_materialization import ProductionMapRecord


TARGET_TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze(path: Path) -> None:
    if path.is_dir():
        for child in path.rglob("*"):
            child.chmod(
                child.lstat().st_mode
                & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            )
    path.chmod(
        path.lstat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    )


def _numbering(sequence: str) -> str:
    return "{" + ", ".join(
        f"'{index} ': '{residue}'"
        for index, residue in enumerate(sequence, start=1)
    ) + "}"


def _target_record(task_id: int) -> tuple[ProductionMapRecord, bytes]:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    sequence = "A" * 98 + alphabet[task_id // 20] + alphabet[task_id % 20]
    ordinal = (task_id << 32) | 17
    record = ProductionMapRecord(
        authority_ordinal=ordinal,
        chain_type="heavy",
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        split="validation" if task_id == 2 else "train",
        status_flags=(),
        stratum="human_heavy",
        study_id=1,
    )
    classified = classify_oas_target(_numbering(sequence), record)
    assert classified.representable is not None
    return record, pack_imgt_target(classified.representable)


def _inventory(path: Path, *, width: int, records: int) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "file_sha256": _sha256(path),
        "record_bytes": width,
        "records": records,
    }


def _build_target_root(tmp_path: Path) -> tuple[Path, dict[int, ProductionMapRecord]]:
    root = tmp_path / "targets"
    records = {}
    tasks = []
    for task_id in range(32):
        record, payload = _target_record(task_id)
        records[task_id] = record
        task_root = root / "imgt-map" / f"mapper-{task_id:04d}"
        task_root.mkdir(parents=True)
        representable = task_root / "representable.bin"
        unusual = task_root / "unusual.bin"
        representable.write_bytes(payload)
        unusual.write_bytes(b"")
        _freeze(representable)
        _freeze(unusual)
        tasks.append(
            {
                "counts": {
                    "accepted_records": 1,
                    "representable_records": 1,
                    "unusual_records": 0,
                },
                "output_inventory": {
                    "representable": _inventory(
                        representable, width=IMGT_TARGET_RECORD.size, records=1
                    ),
                    "unusual": _inventory(
                        unusual, width=UNUSUAL_TARGET_RECORD.size, records=0
                    ),
                },
                "task_id": task_id,
            }
        )
    publish_export_receipt(
        root / "global_receipt.json",
        {
            "claim_boundary": {
                "accepted_oas_domain_partition_complete": True,
                "global_task_set_complete": True,
                "near_identity_search_complete": False,
            },
            "counts": {"accepted_records": 32},
            "format_version": "oas_imgt_target_export_global_v1",
            "task_count": 32,
            "task_receipts": tasks,
            "test_access": TARGET_TEST_ACCESS,
        },
    )
    return root, records


def _build_exact_root(
    tmp_path: Path,
    *,
    records: dict[int, ProductionMapRecord],
    target_receipt: dict[str, object],
    target_file_sha256: str,
) -> Path:
    root = tmp_path / "exact"
    tasks = []
    global_records = 0
    for task_id in range(32):
        task_root = root / "fixed-imgt" / f"mapper-{task_id:04d}"
        task_root.mkdir(parents=True)
        rows = []
        if task_id in {0, 1}:
            target = records[task_id]
            rows.append(
                pack_fixed_imgt_match(
                    FixedIMGTOutputMatch(
                        identical_positions=90,
                        overlapping_positions=100,
                        query_ordinal=task_id,
                        query_positions=100,
                        target_authority_ordinal=target.authority_ordinal,
                        target_positions=100,
                        target_split=target.split,
                        union_positions=100,
                    )
                )
            )
        matches = task_root / "matches.bin"
        matches.write_bytes(b"".join(rows))
        split_counts = {
            "train": int(bool(rows) and records[task_id].split == "train"),
            "validation": int(
                bool(rows) and records[task_id].split == "validation"
            ),
        }
        receipt = {
            "format_version": "oas_imgt_exact_search_task_v1",
            "input": {"global_target_receipt_sha256": target_receipt["receipt_sha256"]},
            "output": {
                "match_record_bytes": FIXED_IMGT_MATCH_RECORD.size,
                "matches_bin_bytes": matches.stat().st_size,
                "matches_bin_sha256": _sha256(matches),
            },
            "task_id": task_id,
        }
        publish_export_receipt(task_root / "receipt.json", receipt)
        task_receipt = json.loads(
            (task_root / "receipt.json").read_text(encoding="ascii")
        )
        tasks.append(
            {
                "match_summary": {
                    "match_records": len(rows),
                    "matched_query_count": len(rows),
                    "matched_target_count": len(rows),
                    "split_match_counts": split_counts,
                },
                "receipt_file_sha256": _sha256(task_root / "receipt.json"),
                "receipt_sha256": task_receipt["receipt_sha256"],
                "task_id": task_id,
            }
        )
        global_records += len(rows)
        _freeze(task_root)
    publish_export_receipt(
        root / "fixed_imgt_global_receipt.json",
        {
            "claim_boundary": {
                "fixed_imgt_representable_target_search_complete": True,
                "gap_aware_search_complete": False,
            },
            "format_version": "oas_imgt_exact_search_global_v1",
            "inputs": {
                "target_receipt_file_sha256": target_file_sha256,
                "target_receipt_sha256": target_receipt["receipt_sha256"],
            },
            "match_summary": {"match_records": global_records},
            "task_count": 32,
            "task_receipts": tasks,
            "test_access": TEST_ACCESS,
        },
    )
    return root


def _build_gap_root(
    tmp_path: Path,
    *,
    records: dict[int, ProductionMapRecord],
    target_receipt: dict[str, object],
    target_file_sha256: str,
) -> Path:
    root = tmp_path / "gap"
    tasks = []
    global_rows = 0
    for task_id in range(32):
        task_root = root / "gap-aware" / f"mapper-{task_id:04d}"
        task_root.mkdir(parents=True)
        rows = []
        if task_id in {0, 2}:
            target = records[task_id]
            identifier = target_identifier(
                task_id=task_id,
                split=target.split,
                chain_type=target.chain_type,
                authority_ordinal=target.authority_ordinal,
            )
            rows.append(
                f"asep_{task_id:064x}\t{identifier}\t0.900000000000\t"
                "1.000000000000\t1.000000000000\n"
            )
        hits = task_root / "hits.tsv"
        hits.write_text("".join(rows), encoding="ascii")
        split_counts = {
            "train": int(bool(rows) and records[task_id].split == "train"),
            "validation": int(
                bool(rows) and records[task_id].split == "validation"
            ),
        }
        receipt = {
            "format_version": "oas_mmseqs_gap_search_task_v1",
            "output": {
                "hits_bytes": hits.stat().st_size,
                "hits_sha256": _sha256(hits),
            },
            "task_id": task_id,
        }
        publish_export_receipt(task_root / "receipt.json", receipt)
        task_receipt = json.loads(
            (task_root / "receipt.json").read_text(encoding="ascii")
        )
        tasks.append(
            {
                "hit_summary": {
                    "alignment_rows": len(rows),
                    "matched_query_count": len(rows),
                    "matched_target_count": len(rows),
                    "split_hit_counts": split_counts,
                },
                "receipt_file_sha256": _sha256(task_root / "receipt.json"),
                "receipt_sha256": task_receipt["receipt_sha256"],
                "task_id": task_id,
            }
        )
        global_rows += len(rows)
        _freeze(task_root)
    publish_export_receipt(
        root / "gap_aware_global_receipt.json",
        {
            "claim_boundary": {
                "gap_aware_search_complete": True,
                "hit_list_truncation_detected": False,
                "treat_mmseqs_as_exhaustive_ground_truth": False,
            },
            "counts": {"hit_pairs": global_rows},
            "format_version": "oas_mmseqs_gap_search_global_v1",
            "input": {
                "global_target_receipt_file_sha256": target_file_sha256,
                "global_target_receipt_sha256": target_receipt["receipt_sha256"],
            },
            "task_count": 32,
            "task_receipts": tasks,
            "test_access": TEST_ACCESS,
        },
    )
    return root


def _fixture(tmp_path: Path) -> dict[str, object]:
    target_root, records = _build_target_root(tmp_path)
    target_file = target_root / "global_receipt.json"
    target_receipt = json.loads(target_file.read_text(encoding="ascii"))
    exact_root = _build_exact_root(
        tmp_path,
        records=records,
        target_receipt=target_receipt,
        target_file_sha256=_sha256(target_file),
    )
    gap_root = _build_gap_root(
        tmp_path,
        records=records,
        target_receipt=target_receipt,
        target_file_sha256=_sha256(target_file),
    )
    exact_receipt = json.loads(
        (exact_root / "fixed_imgt_global_receipt.json").read_text(encoding="ascii")
    )
    gap_receipt = json.loads(
        (gap_root / "gap_aware_global_receipt.json").read_text(encoding="ascii")
    )
    return {
        "exact_hash": exact_receipt["receipt_sha256"],
        "exact_root": exact_root,
        "gap_hash": gap_receipt["receipt_sha256"],
        "gap_root": gap_root,
        "records": records,
        "target_hash": target_receipt["receipt_sha256"],
        "target_root": target_root,
    }


def _compile(tmp_path: Path, fixture: dict[str, object]) -> dict[str, object]:
    return dict(
        compile_exclusions(
            exact_root=fixture["exact_root"],
            gap_root=fixture["gap_root"],
            target_root=fixture["target_root"],
            output_root=tmp_path / "exclusions",
            expected_exact_receipt_sha256=str(fixture["exact_hash"]),
            expected_gap_receipt_sha256=str(fixture["gap_hash"]),
            expected_target_receipt_sha256=str(fixture["target_hash"]),
            controller_source_sha256="a" * 64,
        )
    )


def test_compiler_unions_tracks_maps_ordinals_and_deduplicates_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", raising=False)
    receipt = _compile(tmp_path, fixture)

    assert receipt["counts"] == {
        "raw_matched_both_targets": 1,
        "raw_matched_exact_only_targets": 1,
        "raw_matched_gap_only_targets": 1,
        "raw_matched_target_records": 3,
        "raw_matched_train_targets": 2,
        "raw_matched_validation_targets": 1,
        "unique_both_digests": 1,
        "unique_exact_only_digests": 1,
        "unique_excluded_sequence_sha256": 3,
        "unique_gap_only_digests": 1,
    }
    output = tmp_path / "exclusions/excluded_sequence_sha256.bin"
    raw = output.read_bytes()
    assert len(raw) == 3 * EXCLUSION_DIGEST_RECORD.size
    digests = [raw[index : index + 32] for index in range(0, len(raw), 32)]
    assert digests == sorted(digests)
    assert len(set(digests)) == 3
    assert not output.stat().st_mode & stat.S_IWUSR
    assert receipt["claim_boundary"]["near_identity_exclusion_list_complete"]
    assert receipt["claim_boundary"]["original_training_corpus_filtered"] is False
    assert receipt["claim_boundary"]["zero_overlap_replay_complete"] is False
    assert receipt["claim_boundary"]["large_scale_pretraining_authorized"] is False
    assert receipt["claim_boundary"]["treat_mmseqs_as_exhaustive_ground_truth"] is False


def test_compiler_rejects_mutable_task_evidence_and_cleans_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", raising=False)
    path = Path(fixture["exact_root"]) / "fixed-imgt/mapper-0000/matches.bin"
    path.chmod(0o600)
    with pytest.raises(NearIdentityExclusionError, match="remains writable"):
        _compile(tmp_path, fixture)
    assert not (tmp_path / "exclusions").exists()
    assert not list(tmp_path.glob("exclusions.partial-*"))


def test_compiler_rejects_unfrozen_global_receipt_hash_and_cleans_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("OAS_SCHEDULER_RESERVED_GPU_COUNT", raising=False)
    fixture["exact_hash"] = "0" * 64
    with pytest.raises(NearIdentityExclusionError, match="frozen hash"):
        _compile(tmp_path, fixture)
    assert not (tmp_path / "exclusions").exists()
    assert not list(tmp_path.glob("exclusions.partial-*"))


def test_exclusion_slurm_hides_scheduler_mandated_gpu() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "slurm/oas_near_identity_exclusion_compile.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "#SBATCH --cpus-per-task=8" in launcher
    assert "export CUDA_VISIBLE_DEVICES=" in launcher
    assert "export OAS_SCHEDULER_RESERVED_GPU_COUNT=1" in launcher
    assert "--task-count 32" in launcher
    assert "OAS_EXPECTED_EXACT_RECEIPT_SHA256" in launcher
    assert "OAS_EXPECTED_GAP_RECEIPT_SHA256" in launcher
    assert "OAS_EXPECTED_TARGET_RECEIPT_SHA256" in launcher
