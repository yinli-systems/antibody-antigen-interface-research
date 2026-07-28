import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/pretraining/antibody_bucket144_launch_failure_v1.json"


def test_bucket144_launch_failure_is_preserved_without_training_claim() -> None:
    result = json.loads(RESULT.read_text(encoding="ascii"))
    assert result["model_training_started"] is False
    assert result["failure"]["stage"] == "length_bucket_argument_parsing"
    assert result["failure"]["observed_value_prefix"] == "96\\"
    assert {job["slurm_state"] for job in result["jobs"].values()} == {
        "FAILED"
    }
    assert result["authorization"] == {
        "benchmark_500_updates": False,
        "production_pretraining": False,
        "scale_16_gpu": False,
        "scale_32_gpu": False,
    }
