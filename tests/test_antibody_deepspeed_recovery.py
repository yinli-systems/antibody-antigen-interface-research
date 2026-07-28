import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deepspeed_recovery_is_exact_and_failure_complete() -> None:
    receipt = json.loads(
        (
            ROOT / "results/pretraining/antibody_deepspeed_recovery_v1.json"
        ).read_text(encoding="ascii")
    )
    assert receipt["world_size"] == 8
    assert receipt["checkpoint_format"] == "deepspeed_zero_checkpoint"
    assert receipt["final_state"] == {
        "consumed_residue_tokens": 111564,
        "next_update": 4,
        "sample_cursor": 1024,
    }
    assert [attempt["outcome"] for attempt in receipt["attempts"]] == [
        "failed_closed",
        "failed_closed",
        "failed_closed",
        "passed",
    ]
    for backend in ("zero2", "zero3"):
        assert receipt["backends"][backend]["slurm_state"] == "COMPLETED"
        assert receipt["backends"][backend]["equal"] == {
            "model": True,
            "optimizer": True,
            "trainer_state": True,
        }
        assert receipt["backends"][backend]["checkpoint_bytes"] > 0
    assert receipt["authorization"] == {
        "large_model_pilot": True,
        "production_pretraining": False,
        "scale_32_gpu": False,
        "scale_64_gpu": False,
    }
    assert receipt["test_access"]["sabdab2_heldout_accessed"] is False
