import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict[str, object]:
    return json.loads((ROOT / "results/pretraining" / name).read_text(encoding="ascii"))


def test_deepspeed_runtime_v2_supersedes_login_only_probe() -> None:
    receipt = _load("deepspeed_runtime_v2.json")
    assert receipt["deepspeed"]["version"] == "0.19.0"
    assert receipt["platform"]["torch"] == "2.13.0+cu130"
    assert receipt["platform"]["cuda"] == "13.0"
    assert [attempt["outcome"] for attempt in receipt["attempts"]] == [
        "rejected",
        "rejected",
        "passed_gpu_training",
    ]
    assert receipt["zero_gpu_validation"] == [2, 3]
    assert "checkpoint recovery" in receipt["claim_boundary"]


def test_deepspeed_internal_parity_is_narrowly_registered() -> None:
    parity = _load("antibody_deepspeed_parity_v1.json")
    registration = _load("antibody_deepspeed_registration_v1.json")
    assert parity["membership_equal"] is True
    assert parity["numerical_pass"] is True
    assert parity["differences"]["loss_absolute_maximum"] <= parity["limits"][
        "loss_absolute"
    ]
    assert registration["claim_boundary"]["internal_zero2_zero3_parity"] is True
    assert registration["claim_boundary"]["ddp_numerical_parity"] is False
    assert registration["authorization"] == {
        "checkpoint_recovery": False,
        "production_pretraining": False,
        "scale_32_gpu": False,
        "scale_64_gpu": False,
    }
    parity_bytes = (
        ROOT / "results/pretraining/antibody_deepspeed_parity_v1.json"
    ).read_bytes()
    assert registration["parity_receipt_sha256"] == hashlib.sha256(
        parity_bytes
    ).hexdigest()
    assert registration["test_access"]["sabdab2_heldout_accessed"] is False
