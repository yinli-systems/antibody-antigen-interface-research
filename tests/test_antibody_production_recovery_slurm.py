"""Validate the same-allocation one-GPU production recovery launcher."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm/antibody_production_loader_recovery_gate.sbatch"
PIPELINE_SLURM = ROOT / "slurm/antibody_production_recovery_pipeline.sbatch"


def test_recovery_launcher_uses_one_gpu_and_public_partition() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=8" in source
    assert "#SBATCH --mem" not in source
    assert "vip_gpu_5090_scxi253" not in source
    assert "--nproc-per-node=1" in source


def test_recovery_pipeline_obeys_public_partition_qos() -> None:
    source = PIPELINE_SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=8" in source
    assert "#SBATCH --mem" not in source
    assert "vip_gpu_5090_scxi253" not in source


def test_recovery_launcher_runs_exact_three_part_comparison() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "run_gate continuous 4" in source
    assert "run_gate stage1 2" in source
    assert "run_gate resumed 2" in source
    assert '--resume-root "$RUN_ROOT/stage1"' in source
    assert source.count("--skip-checkpoint") == 2
    assert "audit_antibody_production_loader_recovery.py" in source
    assert 'chmod -R a-w "$RUN_ROOT"' in source


def test_recovery_launcher_uses_frozen_safe_compiled_candidate() -> None:
    source = SLURM.read_text(encoding="ascii")
    for switch in (
        "--compile-blocks",
        "--disable-compiled-cudagraphs",
        "--compile-prewarm-buckets",
        "--fused-adamw",
        "--pinned-staging",
        "--bucketed-padding",
        "--length-bucketing",
    ):
        assert switch in source
    assert "--ddp-communication-dtype" not in source


def test_recovery_launcher_raises_and_binds_the_open_file_budget() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert 'MAXIMUM_OPEN_ARRAYS=${ABF_MAXIMUM_OPEN_ARRAYS:-1024}' in source
    assert (
        'MINIMUM_SOFT_OPEN_FILE_LIMIT=${ABF_MINIMUM_SOFT_OPEN_FILE_LIMIT:-4096}'
        in source
    )
    assert 'HARD_OPEN_FILE_LIMIT=$(ulimit -Hn)' in source
    assert 'ulimit -Sn "$MINIMUM_SOFT_OPEN_FILE_LIMIT"' in source
    assert '--maximum-open-arrays "$MAXIMUM_OPEN_ARRAYS"' in source
    assert (
        '--minimum-soft-open-file-limit "$MINIMUM_SOFT_OPEN_FILE_LIMIT"'
        in source
    )
