"""Test the CPU-only, concurrency-bounded OAS production map launcher."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm/oas_production_map_array.sbatch"


def test_map_array_is_cpu_only_and_protocol_bound() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --array=0-31%16" in source
    assert "#SBATCH --gres" not in source
    assert "#SBATCH --gpus" not in source
    assert "export CUDA_VISIBLE_DEVICES=" in source
    assert "--task-count 32" in source
    assert "--pilot-maximum-data-units" not in source
    for variable in (
        "OAS_SOURCE_ROOT",
        "OAS_DOWNLOAD_ROOT",
        "OAS_PLAN",
        "OAS_DOWNLOAD_AUDIT",
        "OAS_OUTPUT_ROOT",
        "OAS_PROTOCOL_SHA256",
        "OAS_CONTROLLER_SHA256",
        "OAS_PREPROCESSING_SHA256",
        "OAS_RECORD_SHA256",
    ):
        assert f"${{{variable}:?" in source
