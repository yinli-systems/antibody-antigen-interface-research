import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_650m_recovery_is_exact_without_authorizing_larger_scale() -> None:
    receipt = json.loads(
        (
            ROOT / "results/pretraining/antibody_650m_recovery_v1.json"
        ).read_text(encoding="ascii")
    )
    assert receipt["parameter_count"] == 634_513_920
    assert receipt["preset"] == "abf_650m"
    assert receipt["world_size"] == 8
    assert all(receipt["exact_equal"].values())
    assert set(receipt["input_sha256"]) == {
        "continuous_receipt",
        "continuous_state",
        "recovery_receipt",
        "resumed_receipt",
        "resumed_state",
        "stage1_receipt",
        "stage1_state",
    }
    assert receipt["authorization"] == {
        "exact_recovery_backend": "fsdp2",
        "pilot_1_3b": False,
        "production_pretraining": False,
        "scale_32_gpu": False,
        "scale_64_gpu": False,
        "scientific_promotion_required": True,
    }
    assert "no scientific" in receipt["claim_boundary"]
    assert receipt["test_access"] == {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }
