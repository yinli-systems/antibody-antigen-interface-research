#!/usr/bin/env python3
"""Shared fail-closed contracts for synthetic distributed diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


QOS_MAX_NODES = 8
QOS_MAX_GPUS = 64
NODE_MAX_GPUS = 8
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
SAFE_ENVIRONMENT_KEYS = (
    "NCCL_CUMEM_ENABLE",
    "NCCL_CUMEM_HOST_ENABLE",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_P2P_LEVEL",
    "NCCL_SHM_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "TORCH_DISTRIBUTED_DEBUG",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
)
LAUNCH_MANIFEST_SCHEMA = "frontier.distributed-launch-manifest.v1"
LAUNCH_CLAIM_BOUNDARY = (
    "Synthetic correctness and fault-isolation only; no biological metric, "
    "throughput, scaling-efficiency, or state-of-the-art claim."
)


class LaunchContractError(ValueError):
    """Report an invalid or unverified distributed-launch contract."""


def require_sha256(value: str, *, field: str) -> str:
    """Return a lowercase SHA-256 digest or fail closed."""
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise LaunchContractError(
            f"{field} must be a lowercase 64-character SHA-256 digest"
        )
    return value


def require_safe_label(value: str, *, field: str) -> str:
    """Return a path-safe diagnostic label."""
    if not isinstance(value, str) or SAFE_LABEL_PATTERN.fullmatch(value) is None:
        raise LaunchContractError(
            f"{field} must match {SAFE_LABEL_PATTERN.pattern!r}"
        )
    return value


def sha256_file(path: Path | str) -> str:
    """Hash one regular file through an open descriptor."""
    candidate = Path(path)
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise LaunchContractError(
                    f"Expected a regular file: {candidate}"
                )
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise LaunchContractError(f"Cannot hash file: {candidate}") from error
    return digest.hexdigest()


def verify_file_sha256(
    path: Path | str,
    expected_sha256: str,
    *,
    field: str,
) -> str:
    """Verify one file against a caller-frozen external digest."""
    expected = require_sha256(expected_sha256, field=field)
    observed = sha256_file(path)
    if observed != expected:
        raise LaunchContractError(
            f"{field} mismatch: expected={expected}, observed={observed}"
        )
    return observed


def parse_environment_integer(
    environment: Mapping[str, str],
    name: str,
    *,
    required: bool = True,
) -> int | None:
    """Parse one non-negative scheduler integer without accepting signs."""
    raw = environment.get(name)
    if raw is None:
        if required:
            raise LaunchContractError(f"Missing required environment variable: {name}")
        return None
    if not re.fullmatch(r"[0-9]+", raw):
        raise LaunchContractError(f"{name} must be a non-negative integer")
    return int(raw)


def validate_distributed_identity(
    *,
    rank: int,
    local_rank: int,
    world_size: int,
    visible_device_count: int,
    minimum_world_size: int,
    expected_world_size: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, int]:
    """Validate torchrun rank identity and scheduler topology limits."""
    values = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "visible_device_count": visible_device_count,
        "minimum_world_size": minimum_world_size,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values.values()):
        raise LaunchContractError("Distributed identity values must be integers")
    if minimum_world_size < 1:
        raise LaunchContractError("minimum_world_size must be positive")
    if world_size < minimum_world_size:
        raise LaunchContractError(
            f"This diagnostic requires at least {minimum_world_size} ranks"
        )
    if world_size > QOS_MAX_GPUS:
        raise LaunchContractError(
            f"WORLD_SIZE exceeds the reviewed QoS limit of {QOS_MAX_GPUS}"
        )
    if not 0 <= rank < world_size:
        raise LaunchContractError("RANK must be within [0, WORLD_SIZE)")
    if visible_device_count < 1:
        raise LaunchContractError("At least one visible CUDA device is required")
    if visible_device_count > NODE_MAX_GPUS:
        raise LaunchContractError(
            f"Visible GPU count exceeds the reviewed per-node limit of {NODE_MAX_GPUS}"
        )
    if not 0 <= local_rank < visible_device_count:
        raise LaunchContractError(
            "LOCAL_RANK must identify one of the visible CUDA devices"
        )
    if expected_world_size is not None:
        if (
            not isinstance(expected_world_size, int)
            or isinstance(expected_world_size, bool)
            or expected_world_size < 1
        ):
            raise LaunchContractError("expected_world_size must be a positive integer")
        if world_size != expected_world_size:
            raise LaunchContractError(
                "WORLD_SIZE differs from the externally declared launch size: "
                f"expected={expected_world_size}, observed={world_size}"
            )

    scheduler = environment if environment is not None else os.environ
    node_count = parse_environment_integer(
        scheduler,
        "SLURM_JOB_NUM_NODES",
        required=False,
    )
    local_world_size = parse_environment_integer(
        scheduler,
        "LOCAL_WORLD_SIZE",
        required=False,
    )
    if node_count is not None:
        if not 1 <= node_count <= QOS_MAX_NODES:
            raise LaunchContractError(
                f"Slurm node count must be within [1, {QOS_MAX_NODES}]"
            )
        if node_count * visible_device_count > QOS_MAX_GPUS:
            raise LaunchContractError(
                "Allocated topology exceeds the reviewed 64-GPU QoS limit"
            )
    if local_world_size is not None:
        if not 1 <= local_world_size <= visible_device_count:
            raise LaunchContractError(
                "LOCAL_WORLD_SIZE must be positive and no larger than visible GPUs"
            )
        if node_count is not None and world_size != node_count * local_world_size:
            raise LaunchContractError(
                "WORLD_SIZE must equal Slurm nodes times LOCAL_WORLD_SIZE"
            )
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "visible_device_count": visible_device_count,
        "slurm_node_count": node_count or 0,
        "local_world_size": local_world_size or 0,
        "qos_max_nodes": QOS_MAX_NODES,
        "qos_max_gpus": QOS_MAX_GPUS,
    }


def validate_training_mode(
    *,
    parallelism: str,
    precision: str,
    cuda_available: bool,
    bf16_supported: bool,
    require_cuda: bool,
) -> dict[str, Any]:
    """Require an explicit implemented strategy and supported precision."""
    if parallelism != "ddp":
        raise LaunchContractError(
            "Only DDP is implemented by this diagnostic; FSDP2 is not implied"
        )
    if precision not in {"float32", "bfloat16"}:
        raise LaunchContractError("precision must be float32 or bfloat16")
    if require_cuda and not cuda_available:
        raise LaunchContractError("This launch requires CUDA")
    if precision == "bfloat16" and cuda_available and not bf16_supported:
        raise LaunchContractError("The selected CUDA device does not support bfloat16")
    return {
        "parallelism": parallelism,
        "precision": precision,
        "require_cuda": require_cuda,
        "cuda_available": cuda_available,
        "bf16_supported": bf16_supported if cuda_available else False,
        "gradient_reduction": "ddp_average",
        "metric_claim": "forbidden",
        "efficiency_claim": "forbidden",
    }


def safe_environment_snapshot(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Return only reviewed non-secret distributed configuration keys."""
    source = environment if environment is not None else os.environ
    return {name: source.get(name) for name in SAFE_ENVIRONMENT_KEYS}


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite JSON with stable byte-level ordering."""
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LaunchContractError("Value cannot be represented as finite JSON") from error


def _strict_json_bytes(payload_bytes: bytes, *, source: str) -> dict[str, Any]:
    """Parse finite UTF-8 JSON bytes while rejecting duplicate keys."""
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LaunchContractError(f"Duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise LaunchContractError(f"Non-finite JSON value: {value}")

    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except LaunchContractError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LaunchContractError(f"Cannot read strict JSON: {source}") from error
    if not isinstance(payload, dict):
        raise LaunchContractError("JSON root must be an object")
    canonical_json_bytes(payload)
    return payload


def _read_regular_file_once(path: Path | str) -> tuple[bytes, int]:
    """Read one non-symlink regular file and return bytes plus mode."""
    candidate = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise LaunchContractError(
            f"Cannot open regular file without symlink traversal: {candidate}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise LaunchContractError(f"Expected a regular file: {candidate}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read(), stat.S_IMODE(metadata.st_mode)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _strict_json_object(path: Path | str) -> dict[str, Any]:
    """Read a finite JSON object while rejecting links and duplicate keys."""
    payload_bytes, _ = _read_regular_file_once(path)
    return _strict_json_bytes(payload_bytes, source=str(path))


def publish_create_only_json(path: Path | str, payload: Any) -> None:
    """Atomically publish JSON without a check-then-replace overwrite race."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite result artifact: {output}")
    temporary = output.with_name(
        f".{output.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to overwrite result artifact: {output}"
            ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def build_launch_manifest(
    *,
    source_sha256: str,
    runtime_manifest_sha256: str,
    runtime_verification_sha256: str,
    nodes: int,
    gpus_per_node: int,
    processes_per_node: int,
    world_size: int,
    backend: str,
    parallelism: str,
    precision: str,
    scope: str,
) -> dict[str, Any]:
    """Build an integrity-bound synthetic launch manifest."""
    require_sha256(source_sha256, field="source_sha256")
    require_sha256(runtime_manifest_sha256, field="runtime_manifest_sha256")
    require_sha256(
        runtime_verification_sha256,
        field="runtime_verification_sha256",
    )
    integer_fields = {
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "processes_per_node": processes_per_node,
        "world_size": world_size,
    }
    if any(type(value) is not int for value in integer_fields.values()):
        raise LaunchContractError("Launch topology values must be integers")
    if not 1 <= nodes <= QOS_MAX_NODES:
        raise LaunchContractError(f"nodes must be within [1, {QOS_MAX_NODES}]")
    if not 1 <= gpus_per_node <= NODE_MAX_GPUS:
        raise LaunchContractError(
            f"gpus_per_node must be within [1, {NODE_MAX_GPUS}]"
        )
    if not 1 <= processes_per_node <= gpus_per_node:
        raise LaunchContractError(
            "processes_per_node must be positive and no larger than gpus_per_node"
        )
    if world_size != nodes * processes_per_node:
        raise LaunchContractError(
            "world_size must equal nodes times processes_per_node"
        )
    if world_size > QOS_MAX_GPUS:
        raise LaunchContractError(
            f"world_size exceeds the reviewed QoS limit of {QOS_MAX_GPUS}"
        )
    if backend not in {"nccl", "gloo"}:
        raise LaunchContractError("Unsupported distributed backend")
    if parallelism not in {"ddp", "collective_diagnostic"}:
        raise LaunchContractError("Unsupported launch parallelism")
    if precision not in {"float32", "bfloat16", "mixed_collective_dtypes"}:
        raise LaunchContractError("Unsupported launch precision")
    require_safe_label(scope, field="scope")
    payload: dict[str, Any] = {
        "schema_version": LAUNCH_MANIFEST_SCHEMA,
        "scope": scope,
        "source": {"executed_tree_sha256": source_sha256},
        "runtime": {
            "manifest_file_sha256": runtime_manifest_sha256,
            "verification_receipt_sha256": runtime_verification_sha256,
            "live_state_verified": True,
        },
        "topology": {
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "processes_per_node": processes_per_node,
            "world_size": world_size,
            "qos_max_nodes": QOS_MAX_NODES,
            "qos_max_gpus": QOS_MAX_GPUS,
            "nvlink_assumed": False,
        },
        "execution": {
            "backend": backend,
            "parallelism": parallelism,
            "precision": precision,
        },
        "claim_boundary": LAUNCH_CLAIM_BOUNDARY,
    }
    payload["integrity"] = {
        "algorithm": "sha256",
        "content_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }
    return payload


def verify_launch_manifest(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_source_sha256: str,
    expected_world_size: int,
    expected_backend: str,
    expected_parallelism: str,
    expected_precision: str,
) -> dict[str, Any]:
    """Verify one externally hash-bound launch manifest and exact launch fields."""
    expected_file = require_sha256(
        expected_file_sha256,
        field="expected_launch_manifest_sha256",
    )
    candidate = Path(path)
    payload_bytes, observed_mode = _read_regular_file_once(candidate)
    observed_file = hashlib.sha256(payload_bytes).hexdigest()
    if observed_file != expected_file:
        raise LaunchContractError(
            "expected_launch_manifest_sha256 mismatch: "
            f"expected={expected_file}, observed={observed_file}"
        )
    if observed_mode & 0o222:
        raise LaunchContractError("Launch manifest must be read-only")
    payload = _strict_json_bytes(payload_bytes, source=str(candidate))
    expected_top_level = {
        "schema_version",
        "scope",
        "source",
        "runtime",
        "topology",
        "execution",
        "claim_boundary",
        "integrity",
    }
    if set(payload) != expected_top_level:
        raise LaunchContractError("Launch manifest top-level fields are not exact")
    if payload["schema_version"] != LAUNCH_MANIFEST_SCHEMA:
        raise LaunchContractError("Unsupported launch-manifest schema")
    integrity = payload["integrity"]
    if not isinstance(integrity, dict) or set(integrity) != {
        "algorithm",
        "content_sha256",
    }:
        raise LaunchContractError("Launch-manifest integrity object is invalid")
    if integrity["algorithm"] != "sha256":
        raise LaunchContractError("Unsupported launch-manifest hash algorithm")
    expected_content = require_sha256(
        integrity["content_sha256"],
        field="integrity.content_sha256",
    )
    unsigned = {key: value for key, value in payload.items() if key != "integrity"}
    observed_content = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if observed_content != expected_content:
        raise LaunchContractError("Launch-manifest content SHA-256 mismatch")
    if payload["source"] != {"executed_tree_sha256": expected_source_sha256}:
        raise LaunchContractError("Launch manifest source digest differs")
    topology = payload["topology"]
    execution = payload["execution"]
    if not isinstance(topology, dict) or set(topology) != {
        "nodes",
        "gpus_per_node",
        "processes_per_node",
        "world_size",
        "qos_max_nodes",
        "qos_max_gpus",
        "nvlink_assumed",
    }:
        raise LaunchContractError("Launch manifest topology fields are not exact")
    topology_integers = {
        name: topology.get(name)
        for name in (
            "nodes",
            "gpus_per_node",
            "processes_per_node",
            "world_size",
            "qos_max_nodes",
            "qos_max_gpus",
        )
    }
    if any(type(value) is not int for value in topology_integers.values()):
        raise LaunchContractError("Launch manifest topology values must be integers")
    if (
        topology["world_size"] != expected_world_size
        or not 1 <= topology["nodes"] <= QOS_MAX_NODES
        or not 1 <= topology["gpus_per_node"] <= NODE_MAX_GPUS
        or not (
            1
            <= topology["processes_per_node"]
            <= topology["gpus_per_node"]
        )
        or topology["world_size"]
        != topology["nodes"] * topology["processes_per_node"]
        or topology.get("qos_max_nodes") != QOS_MAX_NODES
        or topology.get("qos_max_gpus") != QOS_MAX_GPUS
        or topology.get("nvlink_assumed") is not False
    ):
        raise LaunchContractError("Launch manifest topology differs")
    if (
        not isinstance(execution, dict)
        or set(execution) != {"backend", "parallelism", "precision"}
        or execution.get("backend") != expected_backend
        or execution.get("parallelism") != expected_parallelism
        or execution.get("precision") != expected_precision
    ):
        raise LaunchContractError("Launch manifest execution mode differs")
    runtime = payload["runtime"]
    if (
        not isinstance(runtime, dict)
        or set(runtime)
        != {
            "manifest_file_sha256",
            "verification_receipt_sha256",
            "live_state_verified",
        }
        or runtime.get("live_state_verified") is not True
    ):
        raise LaunchContractError("Runtime live-state verification is not recorded")
    require_sha256(
        runtime["manifest_file_sha256"],
        field="runtime.manifest_file_sha256",
    )
    require_sha256(
        runtime["verification_receipt_sha256"],
        field="runtime.verification_receipt_sha256",
    )
    require_safe_label(payload["scope"], field="scope")
    if payload["claim_boundary"] != LAUNCH_CLAIM_BOUNDARY:
        raise LaunchContractError("Launch-manifest claim boundary differs")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    """Build the create-only launch-manifest command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--runtime-verification", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--processes-per-node", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--backend", choices=("nccl", "gloo"), required=True)
    parser.add_argument(
        "--parallelism",
        choices=("ddp", "collective_diagnostic"),
        required=True,
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16", "mixed_collective_dtypes"),
        required=True,
    )
    parser.add_argument("--scope", required=True)
    return parser


def main() -> int:
    """Create one immutable launch manifest after runtime verification."""
    arguments = _build_parser().parse_args()
    verification_bytes, _ = _read_regular_file_once(
        arguments.runtime_verification
    )
    verification = _strict_json_bytes(
        verification_bytes,
        source=str(arguments.runtime_verification),
    )
    if verification.get("valid") is not True:
        raise LaunchContractError("Runtime verification receipt is not valid")
    observed_runtime_sha256 = verification.get("file_sha256")
    if observed_runtime_sha256 != arguments.runtime_manifest_sha256:
        raise LaunchContractError(
            "Runtime verification receipt does not bind the expected manifest"
        )
    payload = build_launch_manifest(
        source_sha256=arguments.source_sha256,
        runtime_manifest_sha256=arguments.runtime_manifest_sha256,
        runtime_verification_sha256=hashlib.sha256(
            verification_bytes
        ).hexdigest(),
        nodes=arguments.nodes,
        gpus_per_node=arguments.gpus_per_node,
        processes_per_node=arguments.processes_per_node,
        world_size=arguments.world_size,
        backend=arguments.backend,
        parallelism=arguments.parallelism,
        precision=arguments.precision,
        scope=arguments.scope,
    )
    publish_create_only_json(arguments.output, payload)
    os.chmod(arguments.output, 0o444)
    print(
        json.dumps(
            {
                "valid": True,
                "manifest_path": str(arguments.output),
                "file_sha256": sha256_file(arguments.output),
                "content_sha256": payload["integrity"]["content_sha256"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def require_finite_scalar(value: float, *, field: str) -> float:
    """Return a finite scalar suitable for a diagnostic receipt."""
    observed = float(value)
    if not math.isfinite(observed):
        raise LaunchContractError(f"{field} must be finite")
    return observed


def source_tree_digest(
    repository_root: Path,
    relative_paths: Sequence[str],
) -> str:
    """Hash an exact ordered source boundary without following omitted files."""
    if not repository_root.is_absolute():
        raise LaunchContractError("repository_root must be absolute")
    normalized = sorted(set(relative_paths))
    if not normalized or len(normalized) != len(relative_paths):
        raise LaunchContractError("Source paths must be non-empty and unique")
    digest = hashlib.sha256()
    for relative in normalized:
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            raise LaunchContractError(f"Invalid source-relative path: {relative!r}")
        path = repository_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise LaunchContractError(f"Expected a regular source file: {path}")
        encoded = relative.encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
