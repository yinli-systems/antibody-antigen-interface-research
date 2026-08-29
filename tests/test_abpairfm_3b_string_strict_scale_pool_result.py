"""Audit the compact strict STRING scale-pool result."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "pretraining"
    / "abpairfm_3b_string_strict_scale_pool_job_1458996_pass.json"
)


def test_strict_scale_pool_passes_scale_but_keeps_production_closed() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["job"]["job_id"] == 1_458_996
    assert result["job"]["partition"] == "gpu_4090"
    assert result["job"]["state"] == "COMPLETED"
    assert result["job"]["exit_code"] == "0:0"
    assert result["counts"]["output_cluster_pairs"] == 35_769_362
    assert result["counts"]["development_excluded_clusters"] == 3_137
    assert result["counts"]["missing_endpoint_memberships"] == 0
    assert result["scale_pool_gate"]["passed"] is True
    assert result["authorization"]["strict_scale_pool_sampling"] is True
    assert result["authorization"]["production_pretraining"] is False
    assert result["authorization"]["test_access"] is False
    assert result["string_task_authority"]["contact_supervision"] == (
        "structural_sources_only"
    )
    assert len(result["output"]["sha256"]) == 64
    assert all(len(value) == 64 for value in result["provenance"].values())
