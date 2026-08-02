import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fsdp2_recovery_is_exact_and_failure_complete() -> None:
    receipt = json.loads(
        (ROOT / "results/pretraining/antibody_fsdp2_recovery_v1.json").read_text(
            encoding="ascii"
        )
    )
    assert receipt["backend"] == "fsdp2"
    assert receipt["world_size"] == 8
    assert receipt["equal"] == {
        "model": True,
        "optimizer": True,
        "trainer_state": True,
    }
    assert receipt["final_state"] == {
        "consumed_residue_tokens": 111564,
        "next_update": 4,
        "sample_cursor": 1024,
    }
    assert [attempt["outcome"] for attempt in receipt["attempts"]] == [
        "failed_closed",
        "training_and_recovery_passed",
        "receipt_superseded",
        "passed",
    ]
    assert receipt["test_access"]["sabdab2_heldout_accessed"] is False
