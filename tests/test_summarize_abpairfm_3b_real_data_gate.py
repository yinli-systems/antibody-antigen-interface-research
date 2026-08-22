import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_abpairfm_3b_real_data_gate.py"
SPEC = importlib.util.spec_from_file_location("real_data_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _receipt() -> dict[str, object]:
    updates = [{"update": value} for value in range(1, 201)]
    return {
        "authorization": {
            "production_pretraining": False,
            "real_data_gpu_gate_passed": True,
        },
        "data": {
            "development_decontaminated": True,
            "duplicate_manifest_records": 0,
            "manifest_records_consumed": 25_600,
            "manifest_records_expected": 25_600,
            "manifest_skipped_records": 0,
        },
        "format_version": "abpairfm_3b_real_data_gate_v1",
        "loss": {
            "final_20_update_mean": 2.5,
            "first_20_update_mean": 2.7,
        },
        "optimization": {
            "accumulation_steps": 2,
            "microbatch_pairs_per_rank": 8,
            "updates": 200,
        },
        "precision": "bfloat16_compute_fp32_optimizer",
        "ranks": [
            {
                "device_name": "NVIDIA GeForce RTX 4090",
                "local_manifest_records": 3_200,
                "max_memory_allocated_bytes": 18,
                "rank": rank,
                "total_memory_bytes": 24,
                "updates": updates,
            }
            for rank in range(8)
        ],
        "stability": {"p95_to_median_update_time": 1.1},
        "status": "PASS",
        "test_access": {
            "asep_test_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
        "throughput": {
            "consumed_residue_tokens": 1_000_000,
            "residue_tokens_per_second_mean": 4_000.0,
            "residue_tokens_per_second_median": 3_900.0,
        },
        "world_size": 8,
    }


def test_summarize_promotes_stable_real_data_gate() -> None:
    value = MODULE.summarize(_receipt(), job_id=1458866)
    assert value["decision"] == "PROMOTE_TO_QUALITY_PER_GPU_HOUR_GATE"
    assert value["gates"]["passed"] is True
    assert value["authorization"]["production_pretraining"] is False


def test_summarize_fails_closed_on_loss_regression() -> None:
    receipt = _receipt()
    receipt["loss"]["final_20_update_mean"] = 3.0
    value = MODULE.summarize(receipt, job_id=1458866)
    assert value["decision"] == "FAIL_CLOSED_REAL_DATA_GATE"
    assert value["gates"]["passed"] is False


def test_summarize_rejects_test_access() -> None:
    receipt = _receipt()
    receipt["test_access"]["asep_test_accessed"] = True
    with pytest.raises(ValueError, match="test boundary"):
        MODULE.summarize(receipt, job_id=1458866)
