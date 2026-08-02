import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_650m_benchmark_requires_scientific_promotion_before_larger_scale() -> None:
    receipt = json.loads(
        (
            ROOT / "results/pretraining/antibody_650m_benchmark_v1.json"
        ).read_text(encoding="ascii")
    )
    assert receipt["parameter_count"] == 634_513_920
    assert receipt["preset"] == "abf_650m"
    assert receipt["membership_equal"] is True
    assert receipt["headroom_gate"]["passed"] is True
    assert receipt["stability_gate"]["passed"] is True
    assert set(receipt["summaries"]) == {"ddp", "fsdp2", "zero2", "zero3"}
    assert all(
        summary["measured_updates"] == 450
        for summary in receipt["summaries"].values()
    )
    assert receipt["selected_backend_for_650m"] == "ddp"
    throughput = {
        backend: summary["residue_tokens_per_second"]
        for backend, summary in receipt["summaries"].items()
    }
    assert throughput["ddp"] > throughput["fsdp2"]
    assert throughput["fsdp2"] > throughput["zero2"]
    assert throughput["zero2"] > throughput["zero3"]
    assert receipt["authorization"] == {
        "pilot_1_3b": False,
        "production_pretraining": False,
        "scale_32_gpu": False,
        "scale_64_gpu": False,
        "scientific_promotion_required": True,
    }
    assert "repeated pilot data" in receipt["claim_boundary"]
    assert "no scientific" in receipt["claim_boundary"]
    assert receipt["test_access"] == {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }


def test_650m_benchmark_registration_binds_frozen_source() -> None:
    receipt_path = ROOT / "results/pretraining/antibody_650m_benchmark_v1.json"
    registration = json.loads(
        (
            ROOT
            / "results/pretraining/antibody_650m_benchmark_registration_v1.json"
        ).read_text(encoding="ascii")
    )
    assert registration["benchmark_receipt_sha256"] == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    assert registration["selected_backend_for_650m"] == "ddp"
    assert registration["source"] == {
        "model_family_sha256": (
            "1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015"
        ),
        "runner_sha256": (
            "06c837792ae23c0d326f37ad54b8e1ed5756eba77182a48b4eaa70901dbc5b6f"
        ),
        "slurm_sha256": (
            "166c5675864228841f97c0f2f51cc6fbcbd0eb07df36c6eaee77e06b90d0337c"
        ),
        "training_protocol_sha256": (
            "c8974cb06350da46ca49ffff61c149061b7e7f46b43c9e686176539bbe223e6b"
        ),
    }
    source_paths = {
        "model_family_sha256": ROOT / "protocols/antibody_model_family_v1.json",
        "runner_sha256": ROOT / "scripts/run_antibody_pretraining.py",
        "slurm_sha256": ROOT / "slurm/antibody_pretraining_scale_pilot.sbatch",
        "training_protocol_sha256": ROOT / "protocols/antibody_training_v1.json",
    }
    assert registration["source"] == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }
