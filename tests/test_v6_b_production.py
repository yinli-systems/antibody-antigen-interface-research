"""Tests for authenticated V6-B production control and durable evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from interaction_first.v6_b_production import (
    JOURNAL_FORMAT,
    ProgressJournal,
    V6BProductionError,
    canonical_json_bytes,
    create_attempt_directory,
    publish_final,
    validate_registration,
)


def _registration() -> dict[str, object]:
    source_files = {
        label: {"path": f"src/{label}.py", "sha256": "1" * 64}
        for label in (
            "score_free_independent_auditor",
            "score_free_split_statistics",
            "v6_b_decomposition",
            "v6_b_independent_core_auditor",
            "v6_cp_sat_formulation",
            "v6_exact_separation_oracle",
            "v6_joint_ir",
            "v6_b_production",
            "production_controller",
            "slurm_launcher",
        )
    }
    return {
        "format_version": "v6_b_production_feasibility_registration_v3",
        "parent_evidence": {
            "v6_b_policy": {"path": "/policy.json", "sha256": "2" * 64},
            "v6_b_protocol": {"path": "/protocol.md", "sha256": "8" * 64},
            "row_shard_telemetry": {
                "path": "/telemetry.json",
                "file_sha256": "9" * 64,
                "receipt_sha256": "a" * 64,
                "terminal_status": "TELEMETRY_HEADROOM_PASSED",
            },
        },
        "authority": {
            "path": "/authority.json",
            "sha256": "3" * 64,
            "scientific_payload_sha256": "4" * 64,
            "counts": {
                "source_rows": 3229,
                "exact_pairs": 1830,
                "antibody_entities": 1383,
                "antigen_entities": 1172,
            },
        },
        "raw_inputs": {
            label: {"path": f"/{label}", "sha256": "5" * 64}
            for label in (
                "train_authority",
                "canonical_manifest",
                "edge_ledger_vh",
                "edge_ledger_vl",
                "edge_ledger_concatenated_antibody",
                "edge_ledger_antigen",
            )
        },
        "runtime": {
            "python_executable": "/runtime/bin/python",
            "python_version": "3.10.12",
            "ortools_version": "9.15.6755",
            "platform": "Linux-test",
            "cp_sat_runtime_receipt_path": "/runtime/receipt.json",
            "cp_sat_runtime_receipt_sha256": "6" * 64,
        },
        "source": {
            "implementation_commit": "7" * 40,
            "source_root": "/source",
            "files": source_files,
        },
        "limits": {
            "maximum_master_iterations": 1000,
            "maximum_combined_candidates": 10000,
            "maximum_time_seconds_per_solve": 21600,
        },
        "slurm": {
            "accelerator_visible_to_solver": False,
            "allocated_gpu_purpose": "site_scheduler_requirement_only",
            "partition": "gpu_5090",
            "nodes": 1,
            "tasks": 1,
            "cpus_per_task": 8,
            "explicit_memory_request": False,
            "memory_allocation_policy": "partition_automatic_per_allocated_gpu",
            "scheduler_memory_gb_per_gpu": 126,
            "wall_time": "48:00:00",
            "gpus_requested": 1,
            "solver_workers": 1,
        },
        "authorization": {
            "split_gate0_authorized": False,
            "training_authorized": False,
            "test_access_authorized": False,
        },
    }


def test_registration_has_no_defaults_and_keeps_authority_false() -> None:
    assert validate_registration(_registration()) == _registration()
    bad = _registration()
    bad["authorization"]["training_authorized"] = True
    with pytest.raises(V6BProductionError, match="forbidden authority"):
        validate_registration(bad)
    bad = _registration()
    bad["limits"].pop("maximum_master_iterations")
    with pytest.raises(V6BProductionError, match="limits schema"):
        validate_registration(bad)


def test_registration_requires_one_hidden_scheduler_only_gpu_and_one_worker() -> None:
    gpu = _registration()
    gpu["slurm"]["gpus_requested"] = 0
    with pytest.raises(V6BProductionError, match="request one GPU"):
        validate_registration(gpu)
    visible = _registration()
    visible["slurm"]["accelerator_visible_to_solver"] = True
    with pytest.raises(V6BProductionError, match="must be hidden"):
        validate_registration(visible)
    workers = _registration()
    workers["slurm"]["solver_workers"] = 8
    with pytest.raises(V6BProductionError, match="one solver worker"):
        validate_registration(workers)
    explicit_memory = _registration()
    explicit_memory["slurm"]["explicit_memory_request"] = True
    with pytest.raises(V6BProductionError, match="must not request memory"):
        validate_registration(explicit_memory)
    wrong_policy = _registration()
    wrong_policy["slurm"]["memory_allocation_policy"] = "manual"
    with pytest.raises(V6BProductionError, match="memory allocation policy"):
        validate_registration(wrong_policy)


def test_progress_journal_is_contiguous_self_hashed_and_hash_chained(
    tmp_path: Path,
) -> None:
    path = tmp_path / "progress.jsonl"
    path.write_bytes(b"")
    journal = ProgressJournal(path, "a" * 64)
    first = journal.append("FIRST", {"value": 1})
    second = journal.append("SECOND", {"value": 2})
    rows = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    assert rows == [first, second]
    assert [row["event_index"] for row in rows] == [1, 2]
    assert rows[0]["format_version"] == JOURNAL_FORMAT
    assert rows[0]["previous_event_sha256"] is None
    assert rows[1]["previous_event_sha256"] == rows[0]["event_sha256"]
    for row in rows:
        claimed = row.pop("event_sha256")
        assert claimed == hashlib.sha256(canonical_json_bytes(row)).hexdigest()


def test_attempt_and_final_publication_are_create_only_and_read_only(
    tmp_path: Path,
) -> None:
    registration = _registration()
    digest = hashlib.sha256(canonical_json_bytes(registration)).hexdigest()
    output = tmp_path / "attempt"
    journal = create_attempt_directory(output, digest, registration)
    journal.append("CHECK", {"passed": True})
    receipt = {
        "format_version": "v6_b_production_feasibility_final_receipt_v1",
        "terminal_status": "PROVEN_INFEASIBLE",
        "decomposition_result": {"terminal_status": "PROVEN_INFEASIBLE"},
        "journal_event_count": journal.event_count,
        "split_gate0_authorized": False,
        "training_authorized": False,
    }
    receipt_path, report_path = publish_final(output, receipt)
    assert receipt_path.is_file() and report_path.is_file()
    assert os.stat(output).st_mode & 0o222 == 0
    assert all(os.stat(path).st_mode & 0o222 == 0 for path in output.iterdir())
    assert "does not authorize Gate 0" in report_path.read_text(encoding="ascii")
    with pytest.raises(V6BProductionError, match="already exists"):
        publish_final(output, dict(receipt))


def test_solver_event_persists_full_exact_oracle_evidence(tmp_path: Path) -> None:
    class Oracle:
        evidence_history = [{"evidence_sha256": "b" * 64, "failure_count": 0}]

    path = tmp_path / "progress.jsonl"
    path.write_bytes(b"")
    journal = ProgressJournal(path, "a" * 64, exact_oracle=Oracle())
    journal.solver_event(
        {
            "event_type": "EXACT_ORACLE_EVALUATED",
            "event_sha256": "c" * 64,
        }
    )
    rows = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    assert [row["event_type"] for row in rows] == [
        "DECOMPOSITION_EVENT",
        "FULL_EXACT_ORACLE_EVIDENCE",
    ]
    assert rows[-1]["evidence"] == Oracle.evidence_history[-1]


def test_submit_script_hides_its_scheduler_required_gpu() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "submit_v6_b_production_feasibility.sbatch"
    ).read_text(encoding="ascii")
    assert "#SBATCH --mem" not in script
    assert "#SBATCH --mem-per-cpu" not in script
    assert "#SBATCH --mem-per-gpu" not in script
    assert "#SBATCH --time=48:00:00" in script
    assert "#SBATCH --signal=B:TERM@120" in script
    assert "#SBATCH --gpus=1" in script
    assert "export CUDA_VISIBLE_DEVICES=" in script
    assert "PYTHONHASHSEED=0" in script
    assert "OMP_NUM_THREADS=1" in script
