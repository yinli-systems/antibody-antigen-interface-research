import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_eight_gpu_benchmark_selects_ddp_without_scale_claim() -> None:
    benchmark = json.loads(
        (
            ROOT / "results/pretraining/antibody_eight_gpu_benchmark_v1.json"
        ).read_text(encoding="ascii")
    )
    registration = json.loads(
        (
            ROOT
            / "results/pretraining/antibody_eight_gpu_benchmark_registration_v1.json"
        ).read_text(encoding="ascii")
    )
    assert benchmark["membership_equal"] is True
    assert benchmark["selected_backend_for_35m"] == "ddp"
    assert benchmark["summaries"]["ddp"]["measured_updates"] == 450
    assert benchmark["summaries"]["fsdp2"]["measured_updates"] == 450
    assert benchmark["summaries"]["ddp"]["p95_over_p50"] <= 1.20
    assert benchmark["summaries"]["fsdp2"]["p95_over_p50"] <= 1.20
    assert registration["authorization"] == {
        "scale_32_gpu": False,
        "scale_64_gpu": False,
    }
    assert registration["claim_boundary"]["strong_scaling_measured"] is False
