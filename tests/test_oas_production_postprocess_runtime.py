"""Validate the fail-closed OAS postprocessing runtime correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "protocols/oas_production_postprocess_runtime_v1.json"
PROTOCOL = ROOT / "protocols/oas_production_postprocess_runtime_v2.json"
MATERIALIZATION = ROOT / "protocols/oas_production_materialization_v3.json"
SLURM_FILES = (
    ROOT / "slurm/oas_production_reduce_batched_v3.sbatch",
    ROOT / "slurm/oas_production_finalize_v3.sbatch",
    ROOT / "slurm/oas_production_independent_audit_v3.sbatch",
)
PYTHON_BIN = "/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python"


def test_protocol_binds_observed_failure_and_parent_semantics() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert protocol["format_version"] == (
        "oas_production_postprocess_runtime_protocol_v2"
    )
    assert protocol["parent_materialization_protocol_file_sha256"] == (
        hashlib.sha256(MATERIALIZATION.read_bytes()).hexdigest()
    )
    failed = protocol["correction"]["failed_attempt"]
    assert protocol["parent_postprocess_protocol_file_sha256"] == hashlib.sha256(
        V1.read_bytes()
    ).hexdigest()
    assert failed["job_id"] == 1342946
    assert failed["terminal_state"] == "FAILED"
    assert failed["exit_code"] == "1:0"
    assert failed["failed_reduce_tasks"] == 256
    assert failed["all_failed_task_stderr_sha256"] == (
        "d9f42fb903258fe0dc42b4cc4ecdcd06ba7337ab9dcd9c6bc19197ab4969f1c4"
    )
    assert protocol["scope"] == {
        "mapped_data_reused_without_mutation": True,
        "mapper_rerun": False,
        "scientific_semantics_changed": False,
        "stages_relaunched": ["reduce", "finalize", "independent_audit"],
    }
    assert protocol["authorization"]["large_scale_pretraining"] is False


def test_protocol_binds_verified_runtime() -> None:
    runtime = json.loads(PROTOCOL.read_text(encoding="ascii"))["runtime"]
    assert runtime == {
        "manifest_content_sha256": (
            "e33841cc19be4e7c3bb5522194f7d34c7cd518ec967949529094567a6004cfcb"
        ),
        "manifest_file_sha256": (
            "8e025279fc1bb795756e6d1acad59d31de981c9ea86bd1a57610dd1dbd90593a"
        ),
        "manifest_path": (
            "/ssd/scxi253/model_deployment/jobs/frontier_fsdp2_runtime_locks/"
            "frontier_torch213_runtime_20260730_attempt_007.json"
        ),
        "manifest_runtime_state_sha256": (
            "13502176182985a33a1a31833bc8f999f16856c6f917daa45c29bf7e83a09192"
        ),
        "numpy_version": "1.26.4",
        "python_executable": PYTHON_BIN,
        "python_version": "3.10.20",
    }


def test_all_postprocessing_launchers_are_runtime_and_correction_bound() -> None:
    for path in SLURM_FILES:
        source = path.read_text(encoding="ascii")
        assert f"PYTHON_BIN={PYTHON_BIN}" in source
        assert "OAS_RUNTIME_MANIFEST" in source
        assert "OAS_RUNTIME_MANIFEST_SHA256" in source
        assert "OAS_POSTPROCESS_PROTOCOL_SHA256" in source
        assert "oas_production_postprocess_runtime_v2.json" in source
        assert "generate_runtime_manifest.py\" verify" in source
        assert "--expected-file-sha256" in source
        assert "exec python3" not in source


def test_reducer_children_use_the_same_locked_interpreter() -> None:
    source = SLURM_FILES[0].read_text(encoding="ascii")
    assert "OAS_MATERIALIZATION_ROOT" in source
    assert "OAS_MAP_ROOT" not in source
    assert '--map-root "${OAS_MATERIALIZATION_ROOT}"' in source
    assert 'test "${OAS_MATERIALIZATION_ROOT}" = "${OAS_OUTPUT_ROOT}"' in source
    assert 'exec "${PYTHON_BIN}"' in source
    reducer_command = (
        '"${PYTHON_BIN}" '
        '"${OAS_SOURCE_ROOT}/scripts/reduce_oas_production_bucket.py"'
    )
    assert reducer_command in source
