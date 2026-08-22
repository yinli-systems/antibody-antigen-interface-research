"""Static fail-closed contracts for the non-finite resume incident probe."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "slurm/antibody_four_gpu_nonfinite_resume_diagnostic.sbatch"
PATCH = ROOT / "patches/antibody_nonfinite_resume_diagnostic.patch"
MICRO_PATCH = ROOT / "patches/antibody_nonfinite_micro_diagnostic.patch"
SUBSET_PATCH = ROOT / "patches/antibody_nonfinite_subset_diagnostic.patch"


def test_diagnostic_is_non_authorizing_and_uses_only_4090() -> None:
    source = SBATCH.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_4090" in source
    assert "#SBATCH --gres=gpu:4" in source
    assert "hp_4090" not in source
    assert "authorization=incident_diagnostic_only" in source
    assert "test_access=false" in source
    assert "--steps 3000" in source
    assert "--skip-checkpoint" not in source
    assert "sbatch " not in source


def test_patch_moves_finite_guard_before_optimizer_step() -> None:
    source = PATCH.read_text(encoding="ascii")
    guard = source.index(
        "if not math.isfinite(loss_sum_value) "
        "or not math.isfinite(gradient_norm):"
    )
    optimizer_step = source.index("+        optimizer.step()", guard)
    assert guard < optimizer_step
    assert "NONFINITE_GATE_DIAGNOSTIC=" in source
    assert "dist.all_gather_object(rank_diagnostics, local_diagnostic)" in source
    assert "ABF_NONFINITE_DIAGNOSTIC_MAXIMUM_UPDATES" in source
    assert "Diagnostic update limit reached without reproduction" in source


def test_micro_patch_finds_first_bad_rank_and_batch_without_raw_sequences() -> None:
    source = MICRO_PATCH.read_text(encoding="ascii")
    assert "NONFINITE_MICRO_DIAGNOSTIC=" in source
    assert "dist.all_reduce(micro_bad_tensor, op=dist.ReduceOp.MAX)" in source
    assert "batch_index_sha256" in source
    assert "local_output_loss_sum" in source
    assert "raw_sequence" not in source


def test_subset_patch_localizes_bad_samples_without_optimizer_step() -> None:
    source = SUBSET_PATCH.read_text(encoding="ascii")
    assert "ABF_NONFINITE_DIAGNOSTIC_SUBSET_SCAN" in source
    assert "for subset_size in (1, 2, 4, 8, 16):" in source
    assert "NONFINITE_SUBSET_DIAGNOSTIC=" in source
    assert "optimizer.step()" not in source
