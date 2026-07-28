import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deepspeed_runtime_is_isolated_and_version_bound() -> None:
    receipt = json.loads(
        (ROOT / "results/pretraining/deepspeed_runtime_v1.json").read_text(
            encoding="ascii"
        )
    )
    assert receipt["deepspeed"]["version"] == "0.18.4"
    assert receipt["platform"]["torch"] == "2.13.0+cu130"
    assert receipt["platform"]["cuda"] == "13.0"
    assert receipt["zero_config_validation"] == [2, 3]
    assert [attempt["outcome"] for attempt in receipt["attempts"]] == [
        "rejected",
        "passed",
    ]
    assert receipt["test_access"]["sabdab2_heldout_accessed"] is False
