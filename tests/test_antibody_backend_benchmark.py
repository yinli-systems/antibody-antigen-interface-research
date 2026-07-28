import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_backend_benchmark_preserves_scale_gate() -> None:
    result = json.loads(
        (
            ROOT / "results/pretraining/antibody_backend_benchmark_v2.json"
        ).read_text(encoding="ascii")
    )
    assert result["membership_equal"] is True
    assert result["selected_backend_for_35m"] == "ddp"
    assert set(result["summaries"]) == {"ddp", "fsdp2", "zero2", "zero3"}
    assert result["summaries"]["ddp"]["residue_tokens_per_second"] > result[
        "summaries"
    ]["zero2"]["residue_tokens_per_second"]
    assert result["summaries"]["zero2"]["residue_tokens_per_second"] > result[
        "summaries"
    ]["zero3"]["residue_tokens_per_second"]
    assert all(
        summary["p95_over_p50"] <= 1.20
        for summary in result["summaries"].values()
    )
    assert result["authorization"] == {
        "production_pretraining": False,
        "scale_32_gpu": False,
        "scale_64_gpu": False,
    }
