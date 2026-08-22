"""Validate the isolated production-corpus recovery runner contract."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_antibody_pretraining_production_gate.py"


def test_production_gate_runner_is_ascii_and_syntax_valid() -> None:
    source = RUNNER.read_text(encoding="ascii")
    ast.parse(source)
    assert "load_production_corpus(" in source
    assert "build_indexed_update_plan(" in source


def test_production_gate_is_fail_closed_and_recovery_bound() -> None:
    source = RUNNER.read_text(encoding="ascii")
    assert 'world_size != local_world_size' in source
    assert 'partition_update_indices(' in source
    assert 'output.loss_sum * world_size / target_count' in source
    assert 'verify_array_file_hashes=not elastic_gate' in source
    assert 'verify_array_file_hashes=False' in source
    assert 'dist.all_gather_object(rank_hosts, socket.gethostname())' in source
    assert '"array_file_hashes_verified_by_local_rank_zero": (' in source
    assert '"array_file_hashes_verified_during_gate_freeze": (' in source
    assert '"maximum_open_arrays": args.maximum_open_arrays' in source
    assert '"minimum_soft_open_file_limit": (' in source
    assert "Soft open-file limit is below the frozen minimum" in source
    assert "Open-file limit does not preserve descriptor headroom" in source
    assert '"production_pretraining": False' in source
    assert '"scale_out": False' in source
    assert '"scientific_claim": False' in source
    assert "Gate protocol identity or authorization differs" in source
    assert "Launch input evidence differs from the gate protocol" in source
    assert "Resume provenance differs from this gate" in source
    assert "_load_authenticated_dcp(" in source
    assert "_save_authenticated_dcp(" in source
    assert "_restore_rng_checkpoint(" in source
    assert "_save_rng_checkpoint(" in source
    assert "torch.use_deterministic_algorithms(True, warn_only=False)" in source
    assert "Audited OAS loader and exact-recovery gate only" in source


def test_production_gate_streams_only_training_fields() -> None:
    source = RUNNER.read_text(encoding="ascii")
    for field in (
        '"chain_types.npy"',
        '"lengths.npy"',
        '"sample_keys.npy"',
        '"tokens.npy"',
    ):
        assert field in source
    assert '"split": "train"' in source
    assert '"sabdab2_heldout_accessed": False' in source
    assert '"downstream_labels_used": False' in source
    assert '"grouped_update_read": True' in source


def test_nonfinite_gate_fails_before_optimizer_step_with_rank_evidence() -> None:
    source = RUNNER.read_text(encoding="ascii")
    finite_guard = source.index(
        "if not math.isfinite(loss_sum_value) or not math.isfinite(gradient_norm):"
    )
    optimizer_step = source.index("optimizer.step()", finite_guard)
    assert finite_guard < optimizer_step
    assert "NONFINITE_GATE_DIAGNOSTIC=" in source
    assert "dist.all_gather_object(rank_diagnostics, local_diagnostic)" in source
    assert "nonfinite_gradient_names" in source
