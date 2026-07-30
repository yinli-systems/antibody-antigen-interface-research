"""Test the CPU-only, concurrency-bounded OAS production reducer launcher."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm/oas_production_reduce_array.sbatch"


def test_reduce_array_is_cpu_only_and_protocol_bound() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition" not in source
    assert "#SBATCH --array=0-255%16" in source
    assert "#SBATCH --gres" not in source
    assert "#SBATCH --gpus" not in source
    assert "export CUDA_VISIBLE_DEVICES=" in source
    assert '--bucket "${SLURM_ARRAY_TASK_ID}"' in source
    assert "--pilot" not in source
    for variable in (
        "OAS_SOURCE_ROOT",
        "OAS_MAP_ROOT",
        "OAS_OUTPUT_ROOT",
        "OAS_PROTOCOL_SHA256",
        "OAS_REDUCER_SHA256",
        "OAS_RECORD_SHA256",
        "OAS_TOKENIZER_SHA256",
        "OAS_CPU_PARTITION",
    ):
        assert f"${{{variable}:?" in source
