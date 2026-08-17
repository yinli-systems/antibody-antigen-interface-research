from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/build_antibody_eight_gpu_pilot_protocol.py"
ELIGIBILITY = ROOT / "scripts/evaluate_antibody_eight_gpu_recovery_eligibility.py"
SUBMITTER = ROOT / "slurm/submit_antibody_eight_gpu_pilot_after_4g.sbatch"
PILOT = ROOT / "slurm/antibody_eight_gpu_real_oas_recovery_pilot.sbatch"


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
        "hardware_identity": {
            "gpu_count": 4,
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "node_count": 1,
        },
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
    assert value["hardware_identity"] == {
        "gpu_count": 8,
        "gpu_model": "NVIDIA GeForce RTX 5090",
        "node_count": 1,
    }
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


def test_builder_records_cross_hardware_4090_pilot(tmp_path: Path) -> None:
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
            "--target-gpu-model",
            "NVIDIA GeForce RTX 4090",
        ],
        check=True,
    )
    value = json.loads(output.read_text(encoding="ascii"))
    assert value["hardware_identity"]["gpu_model"] == "NVIDIA GeForce RTX 4090"
    assert value["pilot"]["cross_hardware"] is True
    assert value["pilot"]["source_gpu_model"] == "NVIDIA GeForce RTX 5090"


def test_builder_scales_tokens_and_batch_quantum_for_six_gpus(tmp_path: Path) -> None:
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
            "--target-world-size",
            "6",
            "--target-gpu-model",
            "NVIDIA GeForce RTX 4090",
        ],
        check=True,
    )
    value = json.loads(output.read_text(encoding="ascii"))
    assert value["execution"]["world_size"] == 6
    assert value["hardware_identity"]["gpu_count"] == 6
    assert value["pilot"]["target_residue_tokens_per_update"] == 196608
    assert value["pilot"]["batch_quantum_sequences"] == 384
    assert value["runs"]["continuous"]["target_residue_tokens"] == 196608


def _four_gpu_audit(gates: dict[str, bool]) -> dict[str, object]:
    value: dict[str, object] = {
        "authorization": {
            "configuration_promotion": False,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "format_version": "antibody_four_gpu_efficiency_result_v3",
        "gate_passed": False,
        "gates": gates,
        "test_access": {
            "downstream_labels_used": False,
            "external_test_accessed": False,
            "frozen_test_accessed": False,
            "official_test_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    value["receipt_sha256"] = __import__("hashlib").sha256(_canonical(value)).hexdigest()
    return value


def test_recovery_eligibility_retains_failed_efficiency_gates(tmp_path: Path) -> None:
    gates = {
        "artifact_integrity": True,
        "evaluation_noninferiority": True,
        "membership": True,
        "memory": True,
        "replicate_stability": True,
        "replicate_state": True,
        "short_horizon_numerical": True,
        "step_stability": True,
        "approximate_mfu": False,
        "throughput_each_round": False,
        "throughput_pooled": False,
    }
    audit = tmp_path / "audit.json"
    output = tmp_path / "eligibility.json"
    audit.write_bytes(_canonical(_four_gpu_audit(gates)))
    subprocess.run(
        [sys.executable, str(ELIGIBILITY), "--audit", str(audit), "--output", str(output)],
        check=True,
    )
    value = json.loads(output.read_text(encoding="ascii"))
    assert value["eligible"] is True
    assert value["authorization"]["production_pretraining"] is False
    assert value["failed_efficiency_gates_retained"] == [
        "approximate_mfu",
        "throughput_each_round",
        "throughput_pooled",
    ]


def test_recovery_eligibility_rejects_safety_gate_failure(tmp_path: Path) -> None:
    gates = {name: True for name in (
        "artifact_integrity",
        "evaluation_noninferiority",
        "membership",
        "memory",
        "replicate_stability",
        "replicate_state",
        "short_horizon_numerical",
        "step_stability",
    )}
    gates["membership"] = False
    audit = tmp_path / "audit.json"
    output = tmp_path / "eligibility.json"
    audit.write_bytes(_canonical(_four_gpu_audit(gates)))
    subprocess.run(
        [sys.executable, str(ELIGIBILITY), "--audit", str(audit), "--output", str(output)],
        check=True,
    )
    assert json.loads(output.read_text(encoding="ascii"))["eligible"] is False


def test_submitter_uses_explicit_recovery_eligibility() -> None:
    source = SUBMITTER.read_text(encoding="ascii")
    assert "ABF_ELIGIBILITY_SCRIPT" in source
    assert "recovery_eligibility_failed" in source
    assert "production_pretraining_authorized\":false" in source
    assert "ABF_TARGET_PARTITION" in source
    assert "ABF_TARGET_GPU_MODEL" in source
    assert "ABF_TARGET_GPU_COUNT" in source
    assert "gates\", {}).get(\"safe_eager\"" not in source
    assert "--mem" not in source
    assert "#SBATCH --mem" not in PILOT.read_text(encoding="ascii")
