import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_eight_gpu_parity_passes_without_efficiency_claim() -> None:
    parity = json.loads(
        (ROOT / "results/pretraining/antibody_eight_gpu_parity_v1.json").read_text(
            encoding="ascii"
        )
    )
    registration = json.loads(
        (
            ROOT
            / "results/pretraining/antibody_eight_gpu_registration_v1.json"
        ).read_text(encoding="ascii")
    )
    assert parity["anchors_equal"] is True
    assert parity["membership_equal"] is True
    assert parity["numerical_pass"] is True
    assert parity["differences"]["loss_absolute_maximum"] <= parity["limits"][
        "loss_absolute"
    ]
    assert registration["claim_boundary"] == {
        "checkpoint_recovery_tested": False,
        "efficiency_gate_passed": False,
        "numerical_correctness_passed": True,
        "scientific_metric_measured": False,
    }
    assert registration["runs"]["ddp"]["slurm_state"] == "COMPLETED"
    assert registration["runs"]["fsdp2"]["slurm_state"] == "COMPLETED"
