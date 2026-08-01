#!/usr/bin/env python3
"""Run one source-bound PyTorch compiler accuracy-minifier diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


FORMAT_VERSION = "antibody_compile_accuracy_minifier_rank_receipt_v1"
PROTOCOL_FORMAT = "antibody_compile_accuracy_minifier_protocol_v1"
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
CONTROLLED_ENVIRONMENT = {
    "TORCHDYNAMO_REPRO_LEVEL": "4",
    "TORCHDYNAMO_REPRO_AFTER": "dynamo",
}
FORBIDDEN_PRESET_ENVIRONMENT = {
    "TORCHDYNAMO_REPRO_FORWARD_ONLY",
    "TORCHDYNAMO_REPRO_IGNORE_NON_FP",
}


class AccuracyMinifierError(RuntimeError):
    """Reject an unbound minifier diagnostic or runtime."""


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
        raise AccuracyMinifierError("Bound inputs must be regular non-symlink files")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_ascii_object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AccuracyMinifierError(f"{label} must be strict ASCII JSON") from error
    if not isinstance(value, dict):
        raise AccuracyMinifierError(f"{label} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _parse(arguments: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--accuracy-minifier-protocol", type=Path, required=True)
    parser.add_argument("--predecessor-audit", type=Path, required=True)
    parser.add_argument("--minifier-boundary", choices=("dynamo", "aot"), required=True)
    return parser.parse_known_args(arguments)


def _one_argument(arguments: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise AccuracyMinifierError(f"Launch requires exactly one {option}")
    return arguments[positions[0] + 1]


def _write_create_only(path: Path, value: object) -> None:
    payload = _canonical_json_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _artifact_manifest(root: Path) -> list[dict[str, object]]:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AccuracyMinifierError("Minifier output contains a symlink")
        if path.is_file() and path.name != "rank_receipt.json":
            artifacts.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return artifacts


def _validate_protocol(
    protocol: dict[str, Any],
    predecessor: dict[str, Any],
    *,
    boundary: str,
    protocol_sha256: str,
    predecessor_sha256: str,
    base_arguments: list[str],
) -> tuple[Path, int, int]:
    if protocol.get("format_version") != PROTOCOL_FORMAT:
        raise AccuracyMinifierError("Accuracy-minifier protocol format differs")
    if (
        protocol.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
    ):
        raise AccuracyMinifierError("Accuracy-minifier authority differs")
    if protocol.get("diagnostic", {}).get("initial_boundary") != boundary:
        raise AccuracyMinifierError("The requested minifier boundary is not authorized")
    if boundary != "dynamo":
        raise AccuracyMinifierError("This numbered protocol authorizes Dynamo first")
    if protocol.get("diagnostic", {}).get("environment") != CONTROLLED_ENVIRONMENT:
        raise AccuracyMinifierError("The registered minifier environment differs")
    if predecessor_sha256 != protocol.get("predecessor", {}).get(
        "audit_file_sha256"
    ):
        raise AccuracyMinifierError("The predecessor audit digest differs")
    if (
        predecessor.get("diagnostic_valid") is not True
        or predecessor.get("candidate_found") is not False
        or predecessor.get("selected_candidate_for_strict_confirmation") is not None
        or predecessor.get("receipt_sha256")
        != protocol.get("predecessor", {}).get("audit_receipt_sha256")
    ):
        raise AccuracyMinifierError("The rejected predecessor state differs")
    bindings = {
        "base_attribution_wrapper_sha256": (
            Path(__file__).parent / "run_antibody_compile_backend_attribution.py"
        ),
        "base_runner_sha256": (
            Path(__file__).parent / "run_antibody_pretraining_optimized.py"
        ),
        "wrapper_sha256": Path(__file__),
    }
    if any(
        _sha256(path) != protocol.get("inputs", {}).get(label)
        for label, path in bindings.items()
    ):
        raise AccuracyMinifierError("A protocol-bound implementation differs")
    if len(protocol_sha256) != 64:
        raise AccuracyMinifierError("The protocol digest is malformed")
    if _one_argument(base_arguments, "--backend-attribution-mode") != (
        "dynamo_eager_gc_off"
    ):
        raise AccuracyMinifierError("The Dynamo minifier requires the Dynamo arm")
    output = Path(_one_argument(base_arguments, "--output"))
    if not output.is_absolute() or not output.parent.is_dir():
        raise AccuracyMinifierError("The rank output parent is unavailable")
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if rank < 0 or world_size != protocol.get("hardware", {}).get("gpu_count"):
        raise AccuracyMinifierError("The distributed rank allocation differs")
    return output, rank, world_size


def main() -> None:
    original_arguments = sys.argv[1:]
    parsed, base_arguments = _parse(original_arguments)
    protocol, protocol_sha256 = _load_ascii_object(
        parsed.accuracy_minifier_protocol, "Accuracy-minifier protocol"
    )
    predecessor, predecessor_sha256 = _load_ascii_object(
        parsed.predecessor_audit, "Predecessor audit"
    )
    output, rank, world_size = _validate_protocol(
        protocol,
        predecessor,
        boundary=parsed.minifier_boundary,
        protocol_sha256=protocol_sha256,
        predecessor_sha256=predecessor_sha256,
        base_arguments=base_arguments,
    )
    controlled_keys = {*CONTROLLED_ENVIRONMENT, "TORCH_COMPILE_DEBUG_DIR"}
    if any(
        key in os.environ
        for key in controlled_keys | FORBIDDEN_PRESET_ENVIRONMENT
    ):
        raise AccuracyMinifierError("A controlled minifier variable was pre-set")
    rank_root = output.parent / "accuracy_minifier" / f"rank-{rank:02d}"
    rank_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.environ.update(CONTROLLED_ENVIRONMENT)
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(rank_root)
    launch_receipt = {
        "boundary": parsed.minifier_boundary,
        "controlled_environment": {
            key: os.environ[key] for key in sorted(controlled_keys)
        },
        "predecessor_audit_file_sha256": predecessor_sha256,
        "protocol_file_sha256": protocol_sha256,
        "rank": rank,
        "test_access": TEST_ACCESS,
        "world_size": world_size,
        "wrapper_sha256": _sha256(Path(__file__)),
    }
    _write_create_only(rank_root / "launch_receipt.json", launch_receipt)

    import run_antibody_compile_backend_attribution as base
    import torch

    sys.argv = [sys.argv[0], *base_arguments]
    base.main()
    import torch._dynamo.config

    if (
        torch._dynamo.config.repro_after != "dynamo"
        or torch._dynamo.config.repro_level != 4
        or torch._dynamo.config.repro_forward_only is not False
        or torch._dynamo.config.repro_ignore_non_fp is not False
        or torch._dynamo.config.same_two_models_use_fp64 is not True
    ):
        raise AccuracyMinifierError("The effective PyTorch minifier config differs")
    receipt = {
        "artifacts": _artifact_manifest(rank_root),
        "authorization": AUTHORIZATION,
        "boundary": parsed.minifier_boundary,
        "effective_config": {
            "debug_dir_root": torch._dynamo.config.debug_dir_root,
            "repro_after": torch._dynamo.config.repro_after,
            "repro_level": torch._dynamo.config.repro_level,
            "repro_forward_only": torch._dynamo.config.repro_forward_only,
            "repro_ignore_non_fp": torch._dynamo.config.repro_ignore_non_fp,
            "repro_tolerance": torch._dynamo.config.repro_tolerance,
            "same_two_models_use_fp64": (
                torch._dynamo.config.same_two_models_use_fp64
            ),
        },
        "format_version": FORMAT_VERSION,
        "protocol_file_sha256": protocol_sha256,
        "rank": rank,
        "runtime": {
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "torch_git_version": torch.version.git_version,
        },
        "test_access": TEST_ACCESS,
        "world_size": world_size,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    _write_create_only(rank_root / "rank_receipt.json", receipt)


if __name__ == "__main__":
    main()
