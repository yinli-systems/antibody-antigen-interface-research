import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_300m_recovery_distinguishes_exact_and_numerical_evidence() -> None:
    receipt = json.loads(
        (
            ROOT / "results/pretraining/antibody_scale_recovery_v1.json"
        ).read_text(encoding="ascii")
    )
    assert receipt["parameter_count"] == 308_365_312
    assert receipt["ddp"]["numerical_continuation_pass"] is True
    assert receipt["ddp"]["exact_pass"] is False
    assert receipt["ddp"]["optimizer_numerical_equivalence_assessed"] is False
    assert receipt["fsdp2"]["exact_pass"] is True
    assert receipt["authorization"] == {
        "exact_recovery_backend": "fsdp2",
        "production_pretraining": False,
        "scale_32_gpu": False,
        "scale_64_gpu": False,
    }
    assert receipt["test_access"]["sabdab2_heldout_accessed"] is False


def test_300m_recovery_registration_preserves_failed_ddp_job() -> None:
    receipt_path = ROOT / "results/pretraining/antibody_scale_recovery_v1.json"
    registration = json.loads(
        (
            ROOT
            / "results/pretraining/antibody_scale_recovery_registration_v1.json"
        ).read_text(encoding="ascii")
    )
    assert registration["recovery_receipt_sha256"] == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    assert registration["runs"]["ddp"]["slurm_state"] == "FAILED"
    assert registration["runs"]["fsdp2"]["slurm_state"] == "COMPLETED"
    assert registration["checkpoint_bytes"]["ddp"] > 0
    assert registration["checkpoint_bytes"]["fsdp2"] > 0
