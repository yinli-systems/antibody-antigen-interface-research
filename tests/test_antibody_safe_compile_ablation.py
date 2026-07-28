import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/pretraining/antibody_safe_compile_ablation_v1.json"
MODES = ("eager", "regional_no_cudagraph", "regional_cudagraph")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(mode: str) -> dict[str, object]:
    path = ROOT / (
        "results/pretraining/"
        f"antibody_safe_compile_ablation_attempt_010_{mode}_receipt.json"
    )
    return json.loads(path.read_text(encoding="ascii"))


def test_every_safe_compile_mode_proves_eight_rank_parity() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert [item["mode"] for item in result["runs"]] == list(MODES)
    memberships = set()
    for run in result["runs"]:
        path = ROOT / (
            "results/pretraining/"
            "antibody_safe_compile_ablation_attempt_010_"
            f"{run['mode']}_receipt.json"
        )
        assert _sha256(path) == run["receipt_sha256"]
        receipt = json.loads(path.read_text(encoding="ascii"))
        assert len(receipt["trace"]) == 20
        assert receipt["optimization"]["ddp_static_graph"] is False
        assert receipt["optimization"]["audit_gradient_rank_parity"] is True
        assert len(set(receipt["initial_model_rank_sha256"])) == 1
        assert len(set(receipt["final_model_rank_sha256"])) == 1
        assert len(set(receipt["final_optimizer_rank_sha256"])) == 1
        assert receipt["ddp_logging_data"]["num_buckets_reduced"] == 65
        assert "can_set_static_graph" not in receipt["ddp_logging_data"]
        memberships.add(receipt["membership_sha256"])
    assert len(memberships) == 1


def test_regional_compile_modes_are_scientifically_exact() -> None:
    left = _receipt("regional_no_cudagraph")
    right = _receipt("regional_cudagraph")
    scientific_keys = tuple(key for key in left["trace"][0] if key != "seconds")
    for left_item, right_item in zip(left["trace"], right["trace"]):
        assert all(left_item[key] == right_item[key] for key in scientific_keys)
    assert left["parameter_moments"] == right["parameter_moments"]
    assert left["final_model_sha256"] == right["final_model_sha256"]
    assert left["final_optimizer_sha256"] == right["final_optimizer_sha256"]


def test_compiled_candidate_stays_within_frozen_correctness_thresholds() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    correctness = result["correctness"]
    assert correctness["maximum_eager_compiled_loss_absolute_difference"] < 0.005
    assert (
        correctness["maximum_eager_compiled_gradient_norm_relative_difference"]
        < 0.01
    )
    assert (
        correctness["maximum_eager_compiled_parameter_moment_relative_difference"]
        < 0.00001
    )
    assert result["efficiency"]["conclusion"] == (
        "provisional_only_requires_500_update_paired_gate"
    )
    assert result["authorization"]["pilot_16_gpu"] is False
    assert result["authorization"]["pilot_32_gpu"] is False
