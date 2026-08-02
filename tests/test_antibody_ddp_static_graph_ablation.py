import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/pretraining/antibody_ddp_static_graph_ablation_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_static_graph_failure_is_reproduced_across_compile_modes() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    failures = result["failed_static_graph_enabled"]
    assert [item["job_id"] for item in failures] == [
        "1336990",
        "1336991",
        "1336992",
    ]
    assert {item["mode"] for item in failures} == {
        "eager",
        "regional_no_cudagraph",
        "regional_cudagraph",
    }
    for item in failures:
        log = ROOT / (
            "results/pretraining/antibody_ddp_static_graph_failure_attempt_008_"
            f"{item['mode']}.err.txt"
        )
        assert _sha256(log) == item["log_sha256"]
        assert result["failure"] in log.read_text(encoding="ascii")


def test_disabling_static_graph_restores_exact_eight_rank_parity() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    receipts = []
    for item, suffix in zip(
        result["passed_static_graph_disabled"],
        ("on", "off"),
    ):
        path = ROOT / (
            "results/pretraining/"
            f"antibody_ddp_static_graph_disabled_attempt_009_view_{suffix}_receipt.json"
        )
        assert _sha256(path) == item["receipt_sha256"]
        receipt = json.loads(path.read_text(encoding="ascii"))
        assert len(receipt["trace"]) == 20
        assert receipt["optimization"]["disable_ddp_static_graph"] is True
        assert (
            not receipt["optimization"]["disable_ddp_gradient_as_bucket_view"]
        ) is item["gradient_as_bucket_view"]
        assert len(set(receipt["initial_model_rank_sha256"])) == 1
        assert len(set(receipt["final_model_rank_sha256"])) == 1
        assert len(set(receipt["final_optimizer_rank_sha256"])) == 1
        receipts.append(receipt)
    assert receipts[0]["membership_sha256"] == receipts[1]["membership_sha256"]
    scientific_trace_keys = (
        "consumed_residue_tokens",
        "gradient_norm",
        "learning_rate",
        "loss",
        "micro_steps",
        "residue_tokens",
        "selected_targets",
        "sequence_count",
        "update",
    )
    for left, right in zip(receipts[0]["trace"], receipts[1]["trace"]):
        assert all(left[key] == right[key] for key in scientific_trace_keys)
    assert receipts[0]["final_model_sha256"] == receipts[1]["final_model_sha256"]
    assert (
        receipts[0]["final_optimizer_sha256"]
        == receipts[1]["final_optimizer_sha256"]
    )
    assert result["authorization"] == {
        "pilot_16_gpu": False,
        "pilot_32_gpu": False,
        "production_pretraining": False,
        "scientific_claim": False,
    }
