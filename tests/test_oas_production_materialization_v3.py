"""Validate the independently authorized OAS materialization v3 correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "protocols/oas_production_materialization_v2.json"
V3 = ROOT / "protocols/oas_production_materialization_v3.json"
NORMALIZATION = ROOT / "protocols/oas_metadata_normalization_v1.json"
MAP = ROOT / "slurm/oas_production_map_batched_v3.sbatch"
REDUCE = ROOT / "slurm/oas_production_reduce_batched_v3.sbatch"
FINALIZE = ROOT / "slurm/oas_production_finalize_v3.sbatch"
AUDIT = ROOT / "slurm/oas_production_independent_audit_v3.sbatch"


def test_v3_preserves_scientific_semantics_and_authorization_boundary() -> None:
    v2 = json.loads(V2.read_text(encoding="ascii"))
    v3 = json.loads(V3.read_text(encoding="ascii"))
    assert v3["format_version"] == "oas_production_materialization_protocol_v3"
    assert v3["parent_protocol_file_sha256"] == hashlib.sha256(
        V2.read_bytes()
    ).hexdigest()
    for field in (
        "authorization",
        "deduplication",
        "map_record",
        "output",
        "resource_policy",
        "test_access",
    ):
        assert v3[field] == v2[field]
    inherited_inputs = {
        key: value
        for key, value in v3["inputs"].items()
        if not key.startswith("metadata_normalization_")
    }
    assert inherited_inputs == v2["inputs"]
    assert v3["authorization"]["production_corpus_use"] is False
    assert v3["authorization"]["large_scale_pretraining"] is False


def test_v3_binds_independent_full_header_inventory() -> None:
    v3 = json.loads(V3.read_text(encoding="ascii"))
    normalization = json.loads(NORMALIZATION.read_text(encoding="ascii"))
    assert v3["inputs"]["metadata_normalization_protocol_file_sha256"] == (
        hashlib.sha256(NORMALIZATION.read_bytes()).hexdigest()
    )
    assert v3["metadata_normalization"] == {
        "affected_unit_count": 9,
        "allowed_constant": "NaN",
        "allowed_json_path": ["Age"],
        "independent_inventory_semantic_sha256": (
            "69e108b1d34c4080a135867d8ae1fccf1dc47546270ad93e60f6871db822e36c"
        ),
        "replacement_json_value": None,
        "scanned_unit_count": 6373,
        "strict_unit_count": 6364,
    }
    assert len(normalization["affected_units"]) == 9
    assert v3["correction"]["failed_attempt"] == {
        "batch_receipt_file_sha256": (
            "9693130b7fd7ff48af17d4ca099192eb5084c2675882bc2e020cbfb2c315f21e"
        ),
        "batch_receipt_payload_sha256": (
            "6193b74cf9b89e9bf62d61dfa9bd40b39d94066e68fd76afe922fabe386e4527"
        ),
        "completed_map_tasks": 23,
        "exit_code": "1:0",
        "failed_map_tasks": 9,
        "job_id": 1342795,
        "root": (
            "/ssd/scxi253/model_deployment/results/pretraining/"
            "oas_production_materialization_v2_fbb2c7c_attempt_001"
        ),
        "terminal_state": "FAILED",
    }


def test_v3_slurm_chain_is_source_bound_and_keeps_cuda_hidden() -> None:
    for path in (MAP, REDUCE, FINALIZE, AUDIT):
        source = path.read_text(encoding="ascii")
        assert "#SBATCH --gres=gpu:1" in source
        assert "#SBATCH --cpus-per-task=8" in source
        assert "export CUDA_VISIBLE_DEVICES=" in source
        assert "export OAS_SCHEDULER_RESERVED_GPU_COUNT=1" in source
        assert "oas_production_materialization_v3.json" in source
        assert "OAS_PROTOCOL_SHA256" in source
    map_source = MAP.read_text(encoding="ascii")
    assert "--metadata-normalization-protocol" in map_source
    assert "--metadata-normalization-audit" in map_source
    assert "OAS_METADATA_AUDIT" in map_source
