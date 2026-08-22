from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "pretraining"
    / "abpairfm_mint_development_retrieval_job_1459147.json"
)


def test_mint_development_retrieval_result_is_frozen_and_test_closed() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["slurm_array_job_id"] == "1459147"
    assert result["test_access"] is False
    assert len(result["protocol_sha256"]) == 64

    for dataset in ("asep", "sabdab2"):
        evidence = result["datasets"][dataset]
        assert evidence["status"] == "FROZEN_DEVELOPMENT_RETRIEVAL_COMPLETE"
        assert evidence["test_access"] is False
        assert evidence["sacct"] == {
            "elapsed": evidence["sacct"]["elapsed"],
            "exit_code": "0:0",
            "state": "COMPLETED",
            "stderr_bytes": 0,
        }
        assert evidence["predictions"]["row_count"] == 64 * evidence["anchor_instance_count"]
        assert len(evidence["predictions"]["sha256"]) == 64
        assert len(evidence["summary_sha256"]) == 64


def test_mint_result_preserves_primary_macro_unit_and_ci() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    for evidence in result["datasets"].values():
        pair_count = evidence["unique_canonical_exact_pair_count"]
        for metric_name in (
            "primary_bidirectional_unique_endpoint_mrr",
            "primary_bidirectional_unique_endpoint_recall_at_1",
        ):
            metric = evidence[metric_name]
            assert metric["unique_canonical_exact_pair_count"] == pair_count
            estimate = metric["canonical_exact_pair_macro"]
            low, high = metric["canonical_exact_pair_paired_bootstrap_ci95"]
            assert 0.0 <= low <= estimate <= high <= 1.0


def test_mint_result_records_real_4090_efficiency() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    for evidence in result["datasets"].values():
        runtime = evidence["runtime"]
        assert runtime["gpu_name"] == "NVIDIA GeForce RTX 4090"
        assert runtime["gpu_hours"] > 0
        assert runtime["pair_windows_per_second"] > 0
        assert runtime["peak_cuda_bytes"] < 24 * 1024**3
