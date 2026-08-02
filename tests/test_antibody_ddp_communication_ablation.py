import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / (
    "results/pretraining/antibody_ddp_communication_ablation_attempt_012_v1.json"
)
CANDIDATES = (
    "fp32_control",
    "fp32_batched_copy",
    "bf16_compress",
    "bf16_compress_batched_copy",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(name: str) -> dict[str, object]:
    path = ROOT / (
        "results/pretraining/"
        f"antibody_ddp_communication_ablation_attempt_012_{name}_receipt.json"
    )
    return json.loads(path.read_text(encoding="ascii"))


def test_result_binds_every_raw_evidence_file() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert set(result["runs"]) == set(CANDIDATES)
    for name in CANDIDATES:
        run = result["runs"][name]
        prefix = ROOT / (
            "results/pretraining/"
            f"antibody_ddp_communication_ablation_attempt_012_{name}"
        )
        assert _sha256(Path(f"{prefix}_receipt.json")) == run["receipt_sha256"]
        assert _sha256(Path(f"{prefix}_trainer_state.json")) == (
            run["trainer_state_sha256"]
        )
        assert _sha256(Path(f"{prefix}_gpu.csv")) == run["gpu_telemetry_sha256"]
        assert run["slurm_state"] == "COMPLETED"


def test_every_candidate_proves_eight_rank_correctness() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert result["correctness_passers"] == list(CANDIDATES)
    assert result["membership_equal"] is True
    assert result["trace_membership_equal"] is True
    memberships = set()
    for name in CANDIDATES:
        receipt = _receipt(name)
        assert receipt["optimization"]["audit_gradient_rank_parity"] is True
        assert receipt["optimization"]["ddp_static_graph"] is False
        assert len(receipt["trace"]) == 20
        assert len(set(receipt["initial_model_rank_sha256"])) == 1
        assert len(set(receipt["final_model_rank_sha256"])) == 1
        assert len(set(receipt["final_optimizer_rank_sha256"])) == 1
        assert receipt["ddp_logging_data"]["num_buckets_reduced"] == 65
        assert "can_set_static_graph" not in receipt["ddp_logging_data"]
        memberships.add(receipt["membership_sha256"])
    assert len(memberships) == 1


def test_fp32_batched_copy_is_exact_but_not_faster() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    control = _receipt("fp32_control")
    candidate = _receipt("fp32_batched_copy")
    assert control["final_model_sha256"] == candidate["final_model_sha256"]
    assert control["final_optimizer_sha256"] == candidate["final_optimizer_sha256"]
    assert control["parameter_moments"] == candidate["parameter_moments"]
    scientific_keys = tuple(key for key in control["trace"][0] if key != "seconds")
    for left, right in zip(control["trace"], candidate["trace"]):
        assert all(left[key] == right[key] for key in scientific_keys)
    assert (
        result["summaries"]["fp32_batched_copy"][
            "throughput_relative_to_fp32_control"
        ]
        < 1.0
    )


def test_bf16_compression_passes_frozen_numerical_thresholds() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    for name in ("bf16_compress", "bf16_compress_batched_copy"):
        comparison = result["comparisons_to_fp32_control"][name]
        assert comparison["passed"] is True
        assert comparison["maximum_loss_absolute_difference"] < 0.005
        assert comparison["maximum_gradient_norm_relative_difference"] < 0.01
        assert comparison["maximum_parameter_moment_relative_difference"] < 0.00001
        assert result["summaries"][name]["memory_diagnostic_passed"] is True


def test_selection_preserves_short_run_stability_failure() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert result["selected_candidate"] == "bf16_compress_batched_copy"
    assert result["selection_rule"] == (
        "highest_throughput_among_all_correctness_passing_candidates"
    )
    assert (
        result["summaries"]["bf16_compress_batched_copy"]
        ["throughput_relative_to_fp32_control"]
        > 1.27
    )
    assert (
        result["summaries"]["bf16_compress_batched_copy"]
        ["stability_diagnostic_passed"]
        is False
    )
    assert result["authorization"] == {
        "benchmark_500_update_selected_candidate": "bf16_compress_batched_copy",
        "pilot_16_gpu": False,
        "pilot_32_gpu": False,
        "production_pretraining": False,
        "scientific_claim": False,
    }


def test_bf16_batched_copy_preserves_bf16_scientific_trace() -> None:
    left = _receipt("bf16_compress")
    right = _receipt("bf16_compress_batched_copy")
    scientific_keys = tuple(key for key in left["trace"][0] if key != "seconds")
    for left_item, right_item in zip(left["trace"], right["trace"]):
        assert all(left_item[key] == right_item[key] for key in scientific_keys)
    assert left["parameter_moments"] == right["parameter_moments"]
    assert left["final_model_sha256"] == right["final_model_sha256"]
    assert left["final_optimizer_sha256"] == right["final_optimizer_sha256"]
