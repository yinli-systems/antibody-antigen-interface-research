#!/usr/bin/env python3
"""Independently audit one PyTorch compiler accuracy-minifier run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FORMAT_VERSION = "antibody_compile_accuracy_minifier_audit_v1"
PROTOCOL_FORMAT = "antibody_compile_accuracy_minifier_protocol_v1"
RANK_FORMAT = "antibody_compile_accuracy_minifier_rank_receipt_v1"
AUTHORIZATION = {
    "aot_minifier": False,
    "configuration_promotion": False,
    "production_pretraining": False,
    "scale_16_gpu": False,
    "scale_32_gpu": False,
    "scientific_claim": False,
}
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}


class AccuracyMinifierAuditError(RuntimeError):
    """Reject an incomplete, mutable, or re-anchored minifier run."""


def _canonical_json_bytes(value: object) -> bytes:
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


def _sha256(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise AccuracyMinifierAuditError(
            "Audit inputs must be regular non-symlink files"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise AccuracyMinifierAuditError(f"{label} file digest differs")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AccuracyMinifierAuditError(f"{label} must be strict ASCII JSON") from error
    if not isinstance(value, dict):
        raise AccuracyMinifierAuditError(f"{label} must be a JSON object")
    return value


def _validate_self_hash(value: dict[str, Any], label: str) -> None:
    payload = dict(value)
    claimed = payload.pop("receipt_sha256", None)
    observed = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if claimed != observed:
        raise AccuracyMinifierAuditError(f"{label} receipt self-hash differs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-file-sha256", required=True)
    parser.add_argument("--predecessor-audit", type=Path, required=True)
    parser.add_argument("--predecessor-audit-file-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--artifact-manifest-file-sha256", required=True)
    parser.add_argument("--auditor-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _verify_artifact_manifest(run_root: Path, expected_sha256: str) -> int:
    manifest = run_root / "artifact_sha256.txt"
    if _sha256(manifest) != expected_sha256:
        raise AccuracyMinifierAuditError("Artifact manifest digest differs")
    listed: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, absolute = line.split("  ", 1)
        path = Path(absolute)
        if not path.is_relative_to(run_root) or path == manifest:
            raise AccuracyMinifierAuditError("Artifact manifest scope differs")
        listed[path.relative_to(run_root).as_posix()] = digest
    observed = {
        path.relative_to(run_root).as_posix(): _sha256(path)
        for path in run_root.rglob("*")
        if path.is_file() and path != manifest
    }
    if listed != observed:
        raise AccuracyMinifierAuditError("Artifact manifest membership differs")
    writable = [
        path.relative_to(run_root).as_posix()
        for path in run_root.rglob("*")
        if path.is_file() and os.access(path, os.W_OK)
    ]
    if writable:
        raise AccuracyMinifierAuditError("The minifier run contains writable files")
    return len(observed)


def _audit_rank_receipts(
    run_root: Path,
    protocol_sha256: str,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    receipts = []
    minifier_artifacts = []
    gpu_count = protocol["hardware"]["gpu_count"]
    for rank in range(gpu_count):
        rank_root = (
            run_root
            / "dynamo_eager"
            / "accuracy_minifier"
            / f"rank-{rank:02d}"
        )
        receipt_path = rank_root / "rank_receipt.json"
        receipt = _load(
            receipt_path,
            _sha256(receipt_path),
            f"Rank {rank} receipt",
        )
        _validate_self_hash(receipt, f"Rank {rank}")
        if (
            receipt.get("format_version") != RANK_FORMAT
            or receipt.get("rank") != rank
            or receipt.get("world_size") != gpu_count
            or receipt.get("boundary") != "dynamo"
            or receipt.get("protocol_file_sha256") != protocol_sha256
            or receipt.get("authorization")
            != {
                "aot_minifier": False,
                "configuration_promotion": False,
                "production_pretraining": False,
                "scale_16_gpu": False,
                "scale_32_gpu": False,
                "scientific_claim": False,
            }
            or receipt.get("test_access") != TEST_ACCESS
        ):
            raise AccuracyMinifierAuditError(f"Rank {rank} receipt differs")
        config = receipt.get("effective_config", {})
        if (
            config.get("repro_after") != "dynamo"
            or config.get("repro_level") != 4
            or config.get("repro_forward_only") is not False
            or config.get("repro_ignore_non_fp") is not False
            or config.get("repro_tolerance") != 0.001
            or config.get("same_two_models_use_fp64") is not True
        ):
            raise AccuracyMinifierAuditError(f"Rank {rank} config differs")
        runtime = receipt.get("runtime", {})
        expected_runtime = protocol.get("runtime", {})
        if (
            runtime.get("torch") != expected_runtime.get("torch")
            or runtime.get("cuda") != expected_runtime.get("cuda")
            or runtime.get("torch_git_version")
            != expected_runtime.get("torch_git_version")
        ):
            raise AccuracyMinifierAuditError(f"Rank {rank} runtime differs")
        receipts.append(receipt)
        for artifact in receipt.get("artifacts", []):
            relative = artifact.get("relative_path")
            if not isinstance(relative, str):
                raise AccuracyMinifierAuditError("A rank artifact path is invalid")
            if Path(relative).name in {"minifier_launcher.py", "repro.py"}:
                minifier_artifacts.append(
                    {
                        "rank": rank,
                        "relative_path": relative,
                        "sha256": artifact["sha256"],
                        "size_bytes": artifact["size_bytes"],
                    }
                )
    return receipts, minifier_artifacts


def main() -> int:
    args = _parser().parse_args()
    if _sha256(Path(__file__)) != args.auditor_sha256:
        raise AccuracyMinifierAuditError("Auditor source digest differs")
    protocol = _load(
        args.protocol,
        args.protocol_file_sha256,
        "Accuracy-minifier protocol",
    )
    predecessor = _load(
        args.predecessor_audit,
        args.predecessor_audit_file_sha256,
        "Predecessor audit",
    )
    if protocol.get("format_version") != PROTOCOL_FORMAT:
        raise AccuracyMinifierAuditError("Protocol version differs")
    if protocol.get("inputs", {}).get("auditor_sha256") != args.auditor_sha256:
        raise AccuracyMinifierAuditError("Protocol-bound auditor digest differs")
    if (
        args.predecessor_audit_file_sha256
        != protocol.get("predecessor", {}).get("audit_file_sha256")
        or predecessor.get("receipt_sha256")
        != protocol.get("predecessor", {}).get("audit_receipt_sha256")
        or predecessor.get("candidate_found") is not False
    ):
        raise AccuracyMinifierAuditError("Predecessor rejection differs")
    if not args.run_root.is_absolute() or args.run_root.is_symlink():
        raise AccuracyMinifierAuditError("Run root must be absolute and non-symlink")
    file_count = _verify_artifact_manifest(
        args.run_root, args.artifact_manifest_file_sha256
    )
    receipts, minifier_artifacts = _audit_rank_receipts(
        args.run_root, args.protocol_file_sha256, protocol
    )
    base_receipt_path = args.run_root / "dynamo_eager/run/receipt.json"
    base_receipt = _load(
        base_receipt_path,
        _sha256(base_receipt_path),
        "Base training receipt",
    )
    expected = protocol["expected_replay"]
    replay_equal = all(
        base_receipt.get(field) == expected[value]
        for field, value in (
            ("membership_sha256", "membership_sha256"),
            ("final_model_sha256", "final_model_sha256"),
            ("final_optimizer_sha256", "final_optimizer_sha256"),
            ("run_seed", "run_seed"),
            ("world_size", "world_size"),
            ("preset", "preset"),
        )
    )
    if not replay_equal:
        raise AccuracyMinifierAuditError("The exact Dynamo replay differs")
    base_runtime = base_receipt.get("runtime", {})
    expected_runtime = protocol.get("runtime", {})
    if (
        base_runtime.get("torch") != expected_runtime.get("torch")
        or base_runtime.get("cuda") != expected_runtime.get("cuda")
        or base_runtime.get("cudnn") != expected_runtime.get("cudnn")
    ):
        raise AccuracyMinifierAuditError("The base receipt runtime differs")
    result = {
        "artifact_audit": {
            "file_count": file_count,
            "manifest_file_sha256": args.artifact_manifest_file_sha256,
            "manifest_verified": True,
            "writable_paths": [],
        },
        "authorization": AUTHORIZATION,
        "boundary": "dynamo",
        "boundary_localized": bool(minifier_artifacts),
        "claim_boundary": (
            "Official PyTorch Dynamo accuracy-minifier diagnostic on an exact "
            "replay of the rejected eight-GPU arm; no repaired candidate, AOT "
            "authorization, configuration promotion, production training, scale, "
            "biological result, or state-of-the-art claim"
        ),
        "exact_predecessor_replay": replay_equal,
        "format_version": FORMAT_VERSION,
        "inputs": {
            "auditor_sha256": args.auditor_sha256,
            "predecessor_audit_file_sha256": args.predecessor_audit_file_sha256,
            "protocol_file_sha256": args.protocol_file_sha256,
            "run_root": str(args.run_root),
        },
        "minifier_artifacts": minifier_artifacts,
        "minifier_rank_receipt_count": len(receipts),
        "next_stage": (
            "external_review_of_dynamo_minified_repro_required"
            if minifier_artifacts
            else "dynamo_boundary_not_localized_manual_bisection_required"
        ),
        "test_access": TEST_ACCESS,
    }
    result["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(result)
    ).hexdigest()
    payload = _canonical_json_bytes(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(hashlib.sha256(payload).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
