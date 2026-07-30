#!/usr/bin/env python3
"""Freeze the one-GPU recovery gate after independent corpus audit passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from antibody_pretraining.production_corpus import load_production_corpus


FORMAT_VERSION = "antibody_production_loader_recovery_gate_v1"
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
IMPLEMENTATION_FILES = (
    "scripts/run_antibody_pretraining.py",
    "scripts/run_antibody_pretraining_optimized.py",
    "scripts/run_antibody_pretraining_production_gate.py",
    "src/antibody_pretraining/dense_shard.py",
    "src/antibody_pretraining/masking.py",
    "src/antibody_pretraining/model.py",
    "src/antibody_pretraining/production_corpus.py",
    "src/antibody_pretraining/production_update_plan.py",
    "src/antibody_pretraining/tokenizer.py",
    "src/antibody_pretraining/training_contract.py",
)


class GateFreezeError(RuntimeError):
    """Report an invalid corpus audit or source closure."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: object) -> bytes:
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
    """Validate all data and code inputs before constructing the gate protocol."""
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
    )
    if len(corpus) <= 0:
        raise GateFreezeError("Audited production training split is empty")
    implementation = {}
    for relative in IMPLEMENTATION_FILES:
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise GateFreezeError(f"Implementation input is invalid: {relative}")
        implementation[relative] = _file_sha256(path)
    model_family = source_root / "protocols/antibody_model_family_v1.json"
    training_protocol = source_root / "protocols/antibody_training_v1.json"
    efficiency_protocol = source_root / "protocols/antibody_efficiency_gate_v1.json"
    auditor = source_root / "scripts/audit_antibody_production_loader_recovery.py"
    inputs = {
        "efficiency_protocol_file_sha256": _file_sha256(efficiency_protocol),
        "independent_audit_file_sha256": independent_audit_file_sha256,
        "independent_audit_payload_sha256": independent_audit_payload_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_payload_sha256": manifest_payload_sha256,
        "model_family_file_sha256": _file_sha256(model_family),
        "protocol_file_sha256": corpus_protocol_file_sha256,
        "split": "train",
        "training_protocol_file_sha256": _file_sha256(training_protocol),
    }
    optimization = {
        "bucketed_padding": True,
        "compile_blocks": True,
        "compile_prewarm_buckets": True,
        "ddp_bucket_cap_mb": 25,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": True,
        "fused_adamw": True,
        "length_bucketing": True,
        "pinned_staging": True,
    }

    def run(*, steps: int, checkpoint: bool, resume: bool) -> dict[str, object]:
        return {
            "batch_quantum_sequences": 128,
            "checkpoint": checkpoint,
            "micro_batch_per_gpu": 16,
            "preset": "abf_650m",
            "resume": resume,
            "run_seed": 17,
            "steps": steps,
            "target_residue_tokens": 16384,
        }

    return {
        "auditor_sha256": _file_sha256(auditor),
        "authorization": {
            "loader_recovery_gate": True,
            "production_pretraining": False,
            "scale_out": False,
            "scientific_claim": False,
        },
        "claim_boundary": (
            "Authorizes only the frozen one-GPU OAS loader and exact-recovery "
            "gate; no production pretraining, scale, biological, or SOTA claim"
        ),
        "corpus": {
            "train_record_count": len(corpus),
            "validation_record_count": corpus.manifest["counts"][
                "unique_validation_records"
            ],
        },
        "execution": {
            "backend": "ddp",
            "length_bucket_boundaries": "96,128,160,194",
            "optimization": optimization,
            "world_size": 1,
        },
        "format_version": FORMAT_VERSION,
        "implementation": implementation,
        "inputs": inputs,
        "runs": {
            "continuous": run(steps=4, checkpoint=False, resume=False),
            "resumed": run(steps=2, checkpoint=False, resume=True),
            "stage1": run(steps=2, checkpoint=True, resume=False),
        },
        "test_access": TEST_ACCESS,
    }


def _publish(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise GateFreezeError("Gate protocol output must be create-only")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
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
    payload = freeze_gate(
        source_root=args.source_root,
        corpus_root=args.corpus_root,
        manifest_file_sha256=args.manifest_file_sha256,
        manifest_payload_sha256=args.manifest_payload_sha256,
        corpus_protocol_file_sha256=args.corpus_protocol_file_sha256,
        independent_audit_path=args.independent_audit,
        independent_audit_file_sha256=args.independent_audit_file_sha256,
        independent_audit_payload_sha256=args.independent_audit_payload_sha256,
    )
    _publish(args.output, payload)


if __name__ == "__main__":
    main()
