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
    assert 'verify_array_file_hashes=True' in source
    assert 'verify_array_file_hashes=False' in source
    assert 'dist.all_gather_object(rank_hosts, socket.gethostname())' in source
    assert '"array_file_hashes_verified_by_local_rank_zero": True' in source
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
