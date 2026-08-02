import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_one_gpu_pilot_receipt_has_strict_claim_boundary() -> None:
    receipt = json.loads(
        (ROOT / "results/pretraining/antibody_one_gpu_pilot_v1.json").read_text(
            encoding="ascii"
        )
    )
    assert receipt["slurm_state"] == "COMPLETED"
    assert receipt["device"] == "NVIDIA GeForce RTX 5090"
    assert receipt["checkpoint_format"] == "torch_distributed_checkpoint"
    assert receipt["updates"] == 3
    assert receipt["test_access"] == {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }
    assert "no validation or downstream metric" in receipt["claim_boundary"]
