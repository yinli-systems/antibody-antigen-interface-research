from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_antibody_eight_gpu_pilot_protocol.py"


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _source_protocol() -> dict[str, object]:
    return {
        "format_version": "antibody_production_loader_recovery_gate_v1",
        "authorization": {
            "loader_recovery_gate": True,
            "production_pretraining": False,
            "scale_out": False,
            "scientific_claim": False,
        },
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "claim_boundary": "four GPU gate",
        "execution": {
            "backend": "ddp",
            "data_loader": {
                "grouped_update_read": True,
                "maximum_open_arrays": 1024,
                "minimum_soft_open_file_limit": 4096,
            },
            "length_bucket_boundaries": "96,128,160,194",
            "optimization": {
                "bucketed_padding": False,
                "compile_blocks": False,
                "compile_prewarm_buckets": False,
                "ddp_bucket_cap_mb": 25,
                "deterministic_algorithms": True,
                "disable_compiled_cudagraphs": False,
                "fused_adamw": False,
                "length_bucketing": False,
                "pinned_staging": False,
            },
            "world_size": 4,
        },
        "inputs": {"corpus": "fixed"},
        "implementation": {"runner": "fixed"},
    }


def test_builder_derives_eight_gpu_recovery_protocol(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "pilot.json"
    source.write_bytes(_canonical(_source_protocol()))
    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--source",
            str(source),
            "--output",
            str(output),
            "--continuous-steps",
            "7",
            "--stage1-steps",
            "3",
        ],
        check=True,
    )
    value = json.loads(output.read_text(encoding="ascii"))
    assert value["execution"]["world_size"] == 8
    assert value["pilot"]["continuous_steps"] == 7
    assert value["runs"]["continuous"]["checkpoint"] is False
    assert value["runs"]["stage1"]["checkpoint"] is True
    assert value["runs"]["resumed"]["resume"] is True
    assert value["authorization"]["production_pretraining"] is False


def test_builder_rejects_compiled_source(tmp_path: Path) -> None:
    source = _source_protocol()
    source["execution"]["optimization"]["compile_blocks"] = True  # type: ignore[index]
    source_path = tmp_path / "source.json"
    output = tmp_path / "pilot.json"
    source_path.write_bytes(_canonical(source))
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--source", str(source_path), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not output.exists()
