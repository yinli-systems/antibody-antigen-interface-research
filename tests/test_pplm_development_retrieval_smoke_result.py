from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "pretraining"
    / "abpairfm_pplm_development_retrieval_smoke_job_1459194.json"
)


def test_pplm_smoke_result_preserves_fail_closed_recovery_chain() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert [failure["job_id"] for failure in result["failure_evidence"]] == [
        "1459174",
        "1459176",
        "1459184",
    ]
    assert all(failure["state"] == "FAILED" for failure in result["failure_evidence"])
    assert result["job"]["state"] == "COMPLETED"
    assert result["job"]["stderr_bytes"] == 0


def test_pplm_smoke_result_accepts_probability_equivalence_not_quality() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    equivalence = result["equivalence"]
    assert equivalence["status"] == "EXACT_FEATURE_SEMANTICS_ACCEPTED"
    assert (
        equivalence["clean_room_vs_official_probability_max_abs_delta"]
        <= equivalence["clean_room_vs_official_probability_tolerance"]
    )
    assert result["authorization"]["development_pair_retrieval"] is False
    assert result["test_access"] is False


def test_pplm_smoke_result_records_real_4090_efficiency() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    runtime = result["runtime"]
    assert runtime["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert runtime["pair_windows"] == 125
    assert runtime["pair_windows_per_second"] > 20
    assert runtime["peak_allocated_memory_fraction"] < 0.25
    assert len(result["summary_sha256"]) == 64
