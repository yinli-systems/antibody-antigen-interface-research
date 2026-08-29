"""Test the authenticated, bounded OAS IMGT export launcher."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm/oas_imgt_target_export_batched.sbatch"


def test_imgt_export_reuses_one_idle_gpu_and_runs_all_fixed_tasks() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=8" in source
    assert "export CUDA_VISIBLE_DEVICES=" in source
    assert "export OAS_SCHEDULER_RESERVED_GPU_COUNT=1" in source
    assert "--task-count 32" in source
    assert "--worker-count 8" in source
    assert "--pilot-maximum-data-units" not in source
    assert "mapper-{task_id:04d}/receipt.json" in source
    for variable in (
        "OAS_SOURCE_ROOT",
        "OAS_DOWNLOAD_ROOT",
        "OAS_PLAN",
        "OAS_DOWNLOAD_AUDIT",
        "OAS_METADATA_AUDIT",
        "OAS_ORIGINAL_OUTPUT_ROOT",
        "OAS_OUTPUT_ROOT",
        "OAS_BATCH_LOG_ROOT",
        "OAS_PROTOCOL_SHA256",
        "OAS_SUPERVISOR_SHA256",
        "OAS_CONTROLLER_SHA256",
        "OAS_IMGT_POSITION_SHA256",
        "OAS_IMGT_RECORD_SHA256",
        "OAS_MAPPER_SHA256",
        "OAS_PREPROCESSING_SHA256",
        "OAS_RECORD_SHA256",
    ):
        assert f"${{{variable}:?" in source
