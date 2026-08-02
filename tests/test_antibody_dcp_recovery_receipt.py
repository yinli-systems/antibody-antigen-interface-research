import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dcp_recovery_is_exact_and_preserves_failure_history() -> None:
    receipt = json.loads(
        (ROOT / "results/pretraining/antibody_dcp_recovery_v1.json").read_text(
            encoding="ascii"
        )
    )
    assert receipt["equal"] == {
        "model": True,
        "optimizer": True,
        "trainer_state": True,
    }
    assert receipt["final_state"] == {
        "consumed_residue_tokens": 27842,
        "next_update": 4,
        "sample_cursor": 256,
    }
    assert [attempt["outcome"] for attempt in receipt["attempts"]] == [
        "failed_closed",
        "passed",
    ]
    assert receipt["test_access"]["sabdab2_heldout_accessed"] is False
