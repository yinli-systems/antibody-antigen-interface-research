"""Test the full OAS IMGT target finalizer launcher."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm/oas_imgt_target_finalize.sbatch"


def test_imgt_finalizer_is_hash_bound_and_keeps_search_claim_closed() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "export CUDA_VISIBLE_DEVICES=" in source
    assert "--task-count 32" in source
    assert "--expected-accepted-records 234304601" in source
    assert "--pilot" not in source
    for variable in (
        "OAS_SOURCE_ROOT",
        "OAS_OUTPUT_ROOT",
        "OAS_FINALIZER_SHA256",
        "OAS_CONTROLLER_SHA256",
        "OAS_IMGT_POSITION_SHA256",
        "OAS_IMGT_RECORD_SHA256",
        "OAS_MAPPER_SHA256",
        "OAS_PREPROCESSING_SHA256",
        "OAS_RECORD_SHA256",
    ):
        assert f"${{{variable}:?" in source
