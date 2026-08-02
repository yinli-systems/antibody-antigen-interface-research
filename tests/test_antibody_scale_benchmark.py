import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_300m_benchmark_selects_backend_without_scale_claim() -> None:
    receipt = json.loads(
        (
            ROOT / "results/pretraining/antibody_scale_benchmark_v1.json"
        ).read_text(encoding="ascii")
    )
    assert receipt["parameter_count"] == 308_365_312
    assert receipt["membership_equal"] is True
    assert receipt["headroom_gate"]["passed"] is True
    assert receipt["stability_gate"]["passed"] is True
    assert set(receipt["summaries"]) == {"ddp", "fsdp2", "zero2", "zero3"}
    assert all(
        summary["measured_updates"] == 450
        for summary in receipt["summaries"].values()
    )
    assert receipt["authorization"] == {
        "pilot_650m": True,
        "production_pretraining": False,
        "scale_32_gpu": False,
        "scale_64_gpu": False,
    }
    assert receipt["test_access"]["sabdab2_heldout_accessed"] is False
