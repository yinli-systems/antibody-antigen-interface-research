import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_300m_correctness_authorizes_only_the_next_eight_gpu_gate() -> None:
    receipt = json.loads(
        (
            ROOT / "results/pretraining/antibody_scale_correctness_v1.json"
        ).read_text(encoding="ascii")
    )
    assert receipt["parameter_count"] == 308_365_312
    assert receipt["preset"] == "abf_300m"
    assert receipt["membership_equal"] is True
    assert receipt["numerical_pass"] is True
    assert receipt["direct_ddp_zero_numerical_parity"] is False
    assert receipt["authorization"] == {
        "benchmark_300m": True,
        "pilot_650m": False,
        "production_pretraining": False,
        "scale_32_gpu": False,
        "scale_64_gpu": False,
    }
    assert receipt["gpu_telemetry"]["fsdp2"]["maximum_memory_mib"] < receipt[
        "gpu_telemetry"
    ]["ddp"]["maximum_memory_mib"]
    assert receipt["test_access"]["sabdab2_heldout_accessed"] is False


def test_300m_correctness_registration_binds_jobs_and_receipt() -> None:
    receipt_path = ROOT / "results/pretraining/antibody_scale_correctness_v1.json"
    registration = json.loads(
        (
            ROOT
            / "results/pretraining/antibody_scale_correctness_registration_v1.json"
        ).read_text(encoding="ascii")
    )
    assert registration["correctness_receipt_sha256"] == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    assert set(registration["runs"]) == {"ddp", "fsdp2", "zero2", "zero3"}
    assert all(
        run["slurm_state"] == "COMPLETED"
        for run in registration["runs"].values()
    )
