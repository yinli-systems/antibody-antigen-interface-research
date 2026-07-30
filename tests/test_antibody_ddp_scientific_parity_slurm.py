"""Validate the public-partition nine-run scientific-parity launcher."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/antibody_ddp_scientific_parity.sbatch"


def test_launcher_uses_public_policy_and_one_eight_gpu_allocation() -> None:
    """Keep the full counterbalanced experiment inside one public allocation."""
    source = LAUNCHER.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --gres=gpu:8" in source
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --mem=" not in source
    assert source.count("run_mode seed") == 9
    assert "The scientific parity allocation must expose exactly eight GPUs" in source


def test_launcher_binds_inputs_and_keeps_evaluation_out_of_training() -> None:
    """Require the frozen execution, split, and training-only claims."""
    source = LAUNCHER.read_text(encoding="ascii")
    for required in (
        "ABF_EXECUTION_SHA256",
        "ABF_REGISTRATION_SHA256",
        "ABF_RUNTIME_MANIFEST",
        "ABF_RUNTIME_MANIFEST_SHA256",
        "antibody_ddp_scientific_parity_registration_v3.json",
        "generate_runtime_manifest.py",
        "antibody_ddp_scientific_parity_execution_v2_1.json",
        "antibody_ddp_scientific_parity_gate_v2.json",
        "antibody_ddp_parity_split_v2",
        "--parity-protocol",
        "--parity-split-root",
        "--run-seed",
        "--audit-gradient-rank-parity",
        "--skip-checkpoint",
        "training-excluded dense-pilot MLM systems parity only",
    ):
        assert required in source


def test_launcher_uses_frozen_modes_and_isolates_compile_caches() -> None:
    """Preserve all registered mode differences and per-run cache isolation."""
    source = LAUNCHER.read_text(encoding="ascii")
    assert "abf-ddp-parity-$SLURM_JOB_ID-$label" in source
    assert "safe_eager)" in source
    assert "fp32_compiled)" in source
    assert "bf16_selected)" in source
    assert "--ddp-communication-dtype bfloat16" in source
    assert "--ddp-batched-grad-copy" in source
    assert "--compile-prewarm-buckets" in source
    assert "--deterministic-algorithms" in source
