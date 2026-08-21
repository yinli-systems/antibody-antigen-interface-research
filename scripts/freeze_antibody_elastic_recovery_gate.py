#!/usr/bin/env python3
"""Freeze the first real-OAS 4-to-2-to-4 elastic recovery gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from antibody_pretraining.production_corpus import load_production_corpus


FORMAT_VERSION = "antibody_elastic_recovery_gate_v1"
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
AUTHORIZATION = {
    "elastic_recovery_gate": True,
    "production_pretraining": False,
    "scientific_claim": False,
    "test_access": False,
}
IMPLEMENTATION_FILES = (
    "scripts/run_antibody_pretraining.py",
    "scripts/run_antibody_pretraining_optimized.py",
    "scripts/run_antibody_pretraining_production_gate.py",
    "src/antibody_pretraining/dense_shard.py",
    "src/antibody_pretraining/elastic_resume.py",
    "src/antibody_pretraining/masking.py",
    "src/antibody_pretraining/model.py",
    "src/antibody_pretraining/production_corpus.py",
    "src/antibody_pretraining/production_update_plan.py",
    "src/antibody_pretraining/tokenizer.py",
    "src/antibody_pretraining/training_contract.py",
)


class ElasticGateFreezeError(RuntimeError):
    """Report an invalid source, corpus, or elastic registration."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _implementation(source_root: Path) -> dict[str, str]:
    result = {}
    for relative in IMPLEMENTATION_FILES:
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise ElasticGateFreezeError(
                f"Elastic implementation input is invalid: {relative}"
            )
        result[relative] = _sha256(path)
    return result


def freeze_gate(
    *,
    source_root: Path,
    corpus_root: Path,
    manifest_file_sha256: str,
    manifest_payload_sha256: str,
    corpus_protocol_file_sha256: str,
    independent_audit_path: Path,
    independent_audit_file_sha256: str,
    independent_audit_payload_sha256: str,
) -> dict[str, Any]:
    """Validate immutable inputs and return the canonical elastic protocol."""
    corpus = load_production_corpus(
        corpus_root,
        split="train",
        expected_manifest_file_sha256=manifest_file_sha256,
        expected_manifest_payload_sha256=manifest_payload_sha256,
        expected_protocol_file_sha256=corpus_protocol_file_sha256,
        independent_audit_path=independent_audit_path,
        expected_independent_audit_file_sha256=independent_audit_file_sha256,
        expected_independent_audit_payload_sha256=(
            independent_audit_payload_sha256
        ),
        maximum_open_arrays=1024,
    )
    if len(corpus) <= 0:
        raise ElasticGateFreezeError("Audited production training split is empty")
    model_family = source_root / "protocols/antibody_model_family_v1.json"
    training_protocol = source_root / "protocols/antibody_training_v1.json"
    efficiency_protocol = source_root / "protocols/antibody_efficiency_gate_v1.json"
    auditor = source_root / "scripts/audit_antibody_elastic_recovery.py"
    optimizer_diagnostic = (
        source_root / "scripts/diagnose_antibody_optimizer_state.py"
    )
    freezer = source_root / "scripts/freeze_antibody_elastic_recovery_gate.py"
    launcher = source_root / "slurm/antibody_elastic_recovery_gate.sbatch"
    for path in (
        model_family,
        training_protocol,
        efficiency_protocol,
        auditor,
        optimizer_diagnostic,
        freezer,
        launcher,
    ):
        if not path.is_file() or path.is_symlink():
            raise ElasticGateFreezeError(f"Frozen input is invalid: {path}")
    optimization = {
        "bucketed_padding": True,
        "compile_blocks": False,
        "compile_prewarm_buckets": False,
        "ddp_bucket_cap_mb": 25,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": False,
        "fused_adamw": True,
        "length_bucketing": True,
        "pinned_staging": True,
    }

    def run(
        *,
        role: str,
        world_size: int,
        source_world_size: int | None,
        steps: int,
    ) -> dict[str, object]:
        return {
            "batch_quantum_sequences": 128,
            "checkpoint": True,
            "micro_batch_per_gpu": 16,
            "preset": "abf_35m",
            "resume": source_world_size is not None,
            "role": role,
            "run_seed": 17,
            "source_world_size": source_world_size,
            "steps": steps,
            "target_residue_tokens": 16_384,
            "world_size": world_size,
        }

    return {
        "auditor_sha256": _sha256(auditor),
        "authorization": AUTHORIZATION,
        "claim_boundary": (
            "Real-OAS 35M systems correctness gate for a 4-to-2-to-4 DDP "
            "DCP transition. It authorizes no production pretraining, "
            "downstream evaluation, test access, biological conclusion, "
            "performance claim, model selection, or SOTA claim."
        ),
        "corpus": {
            "array_file_hashes_verified_during_freeze": (
                corpus.contract.array_file_hashes_verified
            ),
            "stage_revalidation": (
                "immutable_tree_manifest_size_dtype_shape"
            ),
            "train_record_count": len(corpus),
            "validation_record_count": corpus.manifest["counts"][
                "unique_validation_records"
            ],
        },
        "execution": {
            "backend": "ddp",
            "data_loader": {
                "grouped_update_read": True,
                "maximum_open_arrays": 1024,
                "minimum_soft_open_file_limit": 4096,
            },
            "length_bucket_boundaries": "194",
            "optimization": optimization,
        },
        "format_version": FORMAT_VERSION,
        "freezer_sha256": _sha256(freezer),
        "implementation": _implementation(source_root),
        "inputs": {
            "efficiency_protocol_file_sha256": _sha256(efficiency_protocol),
            "independent_audit_file_sha256": independent_audit_file_sha256,
            "independent_audit_payload_sha256": (
                independent_audit_payload_sha256
            ),
            "manifest_file_sha256": manifest_file_sha256,
            "manifest_payload_sha256": manifest_payload_sha256,
            "model_family_file_sha256": _sha256(model_family),
            "protocol_file_sha256": corpus_protocol_file_sha256,
            "split": "train",
            "training_protocol_file_sha256": _sha256(training_protocol),
        },
        "numerical_gate": {
            "absolute_tolerance": 0.005,
            "optimizer_state": {
                "exp_avg": {
                    "maximum_absolute_difference": 2.0**-29,
                    "maximum_relative_difference": 0.001,
                    "maximum_ulp_distance": 4096,
                },
                "exp_avg_sq": {
                    "maximum_absolute_difference": 2.0**-36,
                    "maximum_relative_difference": 1.0e-5,
                    "maximum_ulp_distance": 32,
                },
                "param_groups_exact": True,
                "step_exact": True,
                "terminal_model_exact": True,
            },
            "relative_tolerance": 0.02,
            "same_world_prefix_exact": True,
            "terminal_coordinate_exact": True,
        },
        "randomness": {
            "destination_policy": "authenticate_source_then_topology_reseed",
            "masking": "counter_derived_by_sample_key_update_and_seed",
            "model_dropout": 0.0,
            "rank_rng_scientific_authority": False,
        },
        "optimizer_diagnostic_sha256": _sha256(optimizer_diagnostic),
        "runs": {
            "continuous_4": run(
                role="continuous",
                world_size=4,
                source_world_size=None,
                steps=4,
            ),
            "stage1_4": run(
                role="stage1",
                world_size=4,
                source_world_size=None,
                steps=2,
            ),
            "stage2_2": run(
                role="resumed",
                world_size=2,
                source_world_size=4,
                steps=1,
            ),
            "stage3_4": run(
                role="resumed",
                world_size=4,
                source_world_size=2,
                steps=1,
            ),
        },
        "test_access": TEST_ACCESS,
        "launcher_sha256": _sha256(launcher),
        "topology": {
            "allowed_transitions": [[4, 2], [2, 4]],
            "allowed_world_sizes": [2, 4],
            "gpu_model": "NVIDIA GeForce RTX 4090",
            "node_count": 1,
        },
    }


def _publish(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ElasticGateFreezeError("Elastic protocol output must be create-only")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--manifest-payload-sha256", required=True)
    parser.add_argument("--corpus-protocol-file-sha256", required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--independent-audit-file-sha256", required=True)
    parser.add_argument("--independent-audit-payload-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _publish(
        args.output,
        freeze_gate(
            source_root=args.source_root,
            corpus_root=args.corpus_root,
            manifest_file_sha256=args.manifest_file_sha256,
            manifest_payload_sha256=args.manifest_payload_sha256,
            corpus_protocol_file_sha256=args.corpus_protocol_file_sha256,
            independent_audit_path=args.independent_audit,
            independent_audit_file_sha256=(
                args.independent_audit_file_sha256
            ),
            independent_audit_payload_sha256=(
                args.independent_audit_payload_sha256
            ),
        ),
    )


if __name__ == "__main__":
    main()
