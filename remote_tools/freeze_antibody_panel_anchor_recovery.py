#!/usr/bin/env python3
"""Freeze the narrow recovery contract for the stale validation-panel anchor."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


FORMAT_VERSION = "antibody_panel_anchor_recovery_protocol_v1"
RECEIPT_FORMAT = "antibody_panel_anchor_recovery_protocol_receipt_v1"
STALE_ANCHOR = "cc9219f83bfdaef51e97ee9845f31ba9dd2cbde9863715d6fdfa7a50478197ec"
CORRECTED_ANCHOR = "ba24268e3897320aafb49c8b557d559a73386242e47b666d5fc862fdf453cd38"
SOURCE_FILES = (
    "scripts/audit_antibody_four_gpu_pretraining_segment.py",
    "scripts/evaluate_antibody_production_checkpoint.py",
    "scripts/freeze_antibody_four_gpu_efficiency_gate_v3.py",
    "scripts/freeze_antibody_four_gpu_pretraining_campaign.py",
    "slurm/antibody_four_gpu_pretraining_segment.sbatch",
    "src/antibody_pretraining/training_contract.py",
)
RECOVERY_FILES = (
    "recovery/antibody_panel_anchor_recovery.sbatch",
    "recovery/audit_antibody_panel_anchor_recovery.py",
    "recovery/freeze_antibody_panel_anchor_recovery.py",
)
TEST_ACCESS = {
    "downstream_labels_used": False,
    "external_test_accessed": False,
    "frozen_test_accessed": False,
    "official_test_accessed": False,
    "sabdab2_heldout_accessed": False,
}


class RecoveryFreezeError(RuntimeError):
    """Report a recovery contract or evidence mismatch."""


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise RecoveryFreezeError(f"JSON is not a canonical object: {path}")
    return value


def publish(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def source_hashes(source_root: Path) -> dict[str, str]:
    paths = SOURCE_FILES + RECOVERY_FILES
    result = {}
    for relative in paths:
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise RecoveryFreezeError(f"Recovery source is unavailable: {path}")
        result[relative] = sha256_file(path)
    return result


def v3_anchor(source_root: Path) -> str:
    path = source_root / "scripts/freeze_antibody_four_gpu_efficiency_gate_v3.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PANEL_INDICES_SHA256"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RecoveryFreezeError("The live-v3 panel anchor is unavailable")


def live_anchor(
    source_root: Path, *, record_count: int, sample_count: int, seed: int
) -> str:
    path = source_root / "src/antibody_pretraining/training_contract.py"
    spec = importlib.util.spec_from_file_location("recovery_training_contract", path)
    if spec is None or spec.loader is None:
        raise RecoveryFreezeError("Cannot load the training sampler")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    indices = module.sample_indices(
        record_count=record_count,
        sample_cursor=0,
        sample_count=sample_count,
        seed=seed,
    ).astype("<i8", copy=False)
    if len(np.unique(indices)) != sample_count:
        raise RecoveryFreezeError("Live validation panel contains duplicates")
    return hashlib.sha256(indices.tobytes()).hexdigest()


def verify_artifact_tree(segment_root: Path) -> dict[str, Any]:
    manifest = segment_root / "artifact_sha256.txt"
    if not manifest.is_file() or manifest.is_symlink():
        raise RecoveryFreezeError("Failed segment manifest is unavailable")
    expected = {
        path.resolve()
        for path in segment_root.rglob("*")
        if path.is_file() and path.name != manifest.name
    }
    observed = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, separator, raw_path = line.partition("  ")
        path = Path(raw_path).resolve()
        if (
            not separator
            or path in observed
            or segment_root.resolve() not in path.parents
            or sha256_file(path) != digest
        ):
            raise RecoveryFreezeError("Failed segment artifact manifest differs")
        observed.add(path)
    if observed != expected:
        raise RecoveryFreezeError("Failed segment artifact inventory differs")
    writable = [
        str(path)
        for path in expected | {manifest.resolve()}
        if path.stat().st_mode & 0o222
    ]
    if writable:
        raise RecoveryFreezeError("Failed segment is not immutable")
    return {
        "file_count": len(expected) + 1,
        "manifest_file_sha256": sha256_file(manifest),
    }


def evidence(
    *, source_root: Path, campaign_root: Path, failed_job_id: str
) -> dict[str, Any]:
    gate_path = campaign_root / "campaign_gate.json"
    segment_root = campaign_root / "segment-000000"
    gate = load_json(gate_path)
    panel = gate.get("validation_panel", {})
    record_count = int(gate.get("corpus", {}).get("validation_record_count", 0))
    sample_count = int(panel.get("record_count", 0))
    seed = int(panel.get("seed", -1))
    observed_live = live_anchor(
        source_root,
        record_count=record_count,
        sample_count=sample_count,
        seed=seed,
    )
    if panel.get("indices_sha256") != STALE_ANCHOR:
        raise RecoveryFreezeError("Campaign does not contain the known stale anchor")
    if observed_live != CORRECTED_ANCHOR or v3_anchor(source_root) != CORRECTED_ANCHOR:
        raise RecoveryFreezeError("Live sampler and v3 anchor do not agree")
    evaluation_log = segment_root / "evaluation.log"
    error_text = evaluation_log.read_text(encoding="utf-8", errors="replace")
    if error_text.count("RuntimeError: Validation panel anchor differs") < 4:
        raise RecoveryFreezeError("Original failure is not the four-rank anchor mismatch")
    environment = (segment_root / "environment.txt").read_text(
        encoding="utf-8", errors="strict"
    )
    if f"job_id={failed_job_id}\n" not in environment:
        raise RecoveryFreezeError("Failed job identity differs")
    if (segment_root / "evaluation.json").exists():
        raise RecoveryFreezeError("Original failed segment unexpectedly has an evaluation")
    receipt = load_json(segment_root / "run/receipt.json")
    state = load_json(segment_root / "run/trainer_state.json")
    if (
        receipt.get("campaign", {}).get("executed_steps") != 3000
        or len(receipt.get("trace", [])) != 3000
        or state.get("next_update") != 3000
        or state.get("consumed_residue_tokens") != 431226626
    ):
        raise RecoveryFreezeError("Training did not reach the frozen segment boundary")
    return {
        "corrected_panel_anchor": observed_live,
        "failed_job_id": failed_job_id,
        "failed_segment_artifacts": verify_artifact_tree(segment_root),
        "failed_segment_evaluation_log_sha256": sha256_file(evaluation_log),
        "failed_segment_receipt_sha256": sha256_file(
            segment_root / "run/receipt.json"
        ),
        "failed_segment_root": str(segment_root.resolve()),
        "failed_segment_state_sha256": sha256_file(
            segment_root / "run/trainer_state.json"
        ),
        "gate_protocol_sha256": sha256_file(gate_path),
        "panel_record_count": sample_count,
        "panel_seed": seed,
        "stale_panel_anchor": panel["indices_sha256"],
        "validation_record_count": record_count,
    }


def freeze(
    *,
    source_root: Path,
    campaign_root: Path,
    failed_job_id: str,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    frozen_evidence = evidence(
        source_root=source_root,
        campaign_root=campaign_root,
        failed_job_id=failed_job_id,
    )
    protocol = {
        "authorization": {
            "checkpoint_evaluation": True,
            "production_pretraining_continuation": True,
            "scientific_claim": False,
        },
        "claim_boundary": (
            "Measurement-only correction of a stale frozen OAS validation-panel "
            "index anchor; training state, checkpoint, sampler, optimizer, token "
            "cursor, masking, data, and test authority are unchanged."
        ),
        "evidence": frozen_evidence,
        "format_version": FORMAT_VERSION,
        "implementation": source_hashes(source_root),
        "test_access": TEST_ACCESS,
    }
    output_root.mkdir(parents=True)
    protocol_path = output_root / "PROTOCOL.json"
    publish(protocol_path, protocol)
    receipt = {
        "format_version": RECEIPT_FORMAT,
        "protocol_file_sha256": sha256_file(protocol_path),
        "source_root": str(source_root.resolve()),
        "test_access": TEST_ACCESS,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    publish(output_root / "PROTOCOL_RECEIPT.json", receipt)
    return receipt


def verify(
    *, protocol_path: Path, receipt_path: Path, source_root: Path
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    receipt = load_json(receipt_path)
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    if (
        protocol.get("format_version") != FORMAT_VERSION
        or protocol.get("test_access") != TEST_ACCESS
        or protocol.get("implementation") != source_hashes(source_root)
        or receipt.get("format_version") != RECEIPT_FORMAT
        or receipt.get("protocol_file_sha256") != sha256_file(protocol_path)
        or receipt.get("source_root") != str(source_root.resolve())
        or receipt.get("test_access") != TEST_ACCESS
        or claimed != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    ):
        raise RecoveryFreezeError("Recovery protocol verification failed")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--source-root", type=Path, required=True)
    freeze_parser.add_argument("--campaign-root", type=Path, required=True)
    freeze_parser.add_argument("--failed-job-id", required=True)
    freeze_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--protocol", type=Path, required=True)
    verify_parser.add_argument("--receipt", type=Path, required=True)
    verify_parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        payload = freeze(
            source_root=args.source_root,
            campaign_root=args.campaign_root,
            failed_job_id=args.failed_job_id,
            output_root=args.output_root,
        )
    else:
        payload = verify(
            protocol_path=args.protocol,
            receipt_path=args.receipt,
            source_root=args.source_root,
        )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
