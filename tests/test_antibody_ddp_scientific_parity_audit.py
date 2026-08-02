"""Test scientific-parity bootstrap and throughput helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from audit_antibody_ddp_scientific_parity import (
    _bootstrap_upper_bounds,
    _expected_optimization,
    _initial_model_digest,
    _throughput,
    _validate_trace,
    _verify_artifact_manifest,
)


def test_paired_bootstrap_is_deterministic_and_detects_noninferiority() -> None:
    """Use fixed PCG64 draws and preserve paired sample effects."""
    reference_ce = np.linspace(1.0, 2.0, 64)
    candidate_ce = reference_ce * 1.002
    reference_accuracy = np.linspace(0.2, 0.8, 64)
    candidate_accuracy = reference_accuracy - 0.001
    first = _bootstrap_upper_bounds(
        reference_ce,
        candidate_ce,
        reference_accuracy,
        candidate_accuracy,
        iterations=1000,
        seed=23,
        confidence=0.95,
    )
    second = _bootstrap_upper_bounds(
        reference_ce,
        candidate_ce,
        reference_accuracy,
        candidate_accuracy,
        iterations=1000,
        seed=23,
        confidence=0.95,
    )
    assert first == second
    assert first["cross_entropy_relative_increase_upper"] < 0.01
    assert first["masked_accuracy_decrease_upper"] < 0.005


def test_paired_bootstrap_rejects_unaligned_arrays() -> None:
    """Fail rather than silently broadcasting unequal sample panels."""
    with np.testing.assert_raises(ValueError):
        _bootstrap_upper_bounds(
            np.ones(4),
            np.ones(3),
            np.ones(4),
            np.ones(4),
            iterations=10,
            seed=7,
            confidence=0.95,
        )


def test_throughput_excludes_exactly_fifty_warmup_updates() -> None:
    """Compute useful residue-token throughput over the frozen window."""
    receipt = {
        "trace": [
            {"residue_tokens": 100 + index, "seconds": 1.0}
            for index in range(500)
        ]
    }
    summary = _throughput(receipt, 50)
    assert summary["measured_updates"] == 450
    assert summary["residue_tokens"] == sum(range(150, 600))
    assert summary["residue_tokens_per_second"] == sum(range(150, 600)) / 450


def test_optimization_modes_are_complete_and_distinct() -> None:
    """Bind every runner switch instead of checking only selected switches."""
    eager = _expected_optimization("safe_eager")
    fp32 = _expected_optimization("fp32_compiled")
    bf16 = _expected_optimization("bf16_selected")
    assert len(eager) == len(fp32) == len(bf16) == 15
    assert eager["compile_blocks"] is False
    assert fp32["compile_blocks"] is True
    assert fp32["ddp_communication_dtype"] == "float32"
    assert bf16["ddp_communication_dtype"] == "bfloat16"
    assert bf16["ddp_batched_grad_copy"] is True
    with pytest.raises(ValueError, match="Unknown optimization mode"):
        _expected_optimization("unknown")


def test_initial_model_digest_requires_exact_rank_parity() -> None:
    """Reject missing ranks and disagreement before comparing modes."""
    digest = "a" * 64
    assert _initial_model_digest({"initial_model_rank_sha256": [digest] * 8}) == digest
    with pytest.raises(ValueError, match="eight-rank parity"):
        _initial_model_digest({"initial_model_rank_sha256": [digest] * 7})
    with pytest.raises(ValueError, match="eight-rank parity"):
        _initial_model_digest(
            {"initial_model_rank_sha256": [digest] * 7 + ["b" * 64]}
        )


def test_trace_requires_contiguous_updates_and_exact_token_accounting() -> None:
    """Fail closed on a discontinuous or internally inconsistent trace."""
    trace = []
    consumed = 0
    for update in range(1, 501):
        consumed += 100
        trace.append(
            {
                "consumed_residue_tokens": consumed,
                "gradient_norm": 1.0,
                "learning_rate": 1e-4,
                "loss": 2.0,
                "micro_steps": 1,
                "residue_tokens": 100,
                "seconds": 0.5,
                "selected_targets": 10,
                "sequence_count": 8,
                "update": update,
            }
        )
    _validate_trace({"trace": trace}, "valid")
    trace[30] = {**trace[30], "consumed_residue_tokens": trace[30]["consumed_residue_tokens"] + 1}
    with pytest.raises(ValueError, match="invalid update"):
        _validate_trace({"trace": trace}, "tampered")


def test_artifact_manifest_replays_hashes_and_mode_bits(tmp_path: Path) -> None:
    """Use permission bits so privileged auditors cannot misclassify artifacts."""
    run_root = tmp_path / "run"
    run_root.mkdir()
    artifact = run_root / "receipt.json"
    artifact.write_bytes(b"{}\n")
    manifest = run_root / "artifact_sha256.txt"
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest.write_text(f"{digest}  {artifact.resolve()}\n", encoding="ascii")
    artifact.chmod(0o444)
    manifest.chmod(0o444)
    audit = _verify_artifact_manifest(run_root)
    assert audit["manifest_verified"] is True
    assert audit["writable_paths"] == []
    artifact.chmod(0o644)
    assert _verify_artifact_manifest(run_root)["writable_paths"] == [str(artifact)]
