"""Immutable, provenance-bound checkpoints for exact training resumption."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CHECKPOINT_FORMAT = "frontier_immutable_checkpoint_v2"
RECEIPT_FORMAT = "frontier_immutable_checkpoint_receipt_v2"
CHECKPOINT_FILENAME = "checkpoint.pt"
RECEIPT_FILENAME = "receipt.json"
_GENERATION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_torch() -> Any:
    """Import PyTorch lazily so receipt inspection does not require it."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for checkpoint operations") from error
    return torch


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value deterministically."""
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    """Reject non-finite JSON constants."""
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _strict_json_object(path: Path) -> dict[str, Any]:
    """Load one finite JSON object with duplicate-key detection."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink JSON file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read strict JSON from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON payload at {path} must be an object")
    return value


def _json_roundtrip(value: Any, name: str) -> Any:
    """Canonicalize and validate a JSON-compatible metadata value."""
    try:
        encoded = _canonical_json_bytes(value)
        canonical = json.loads(
            encoded.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and JSON-compatible") from error
    return canonical


def _sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Stream the SHA-256 digest of a regular file."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_generation(generation: str) -> str:
    """Validate a generation name before using it as a path component."""
    if not isinstance(generation, str) or not _GENERATION_PATTERN.fullmatch(generation):
        raise ValueError(
            "generation must contain 1-128 ASCII letters, digits, dots, dashes, "
            "or underscores and must start with a letter or digit"
        )
    if generation in {".", ".."}:
        raise ValueError("generation cannot be a relative path token")
    return generation


def _validate_hashes(hashes: Mapping[str, str]) -> dict[str, str]:
    """Validate and sort a provenance-hash mapping."""
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("provenance_hashes must be a non-empty mapping")
    validated: dict[str, str] = {}
    for key, value in hashes.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Provenance hash names must be non-empty strings")
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"Invalid SHA-256 digest for provenance field {key!r}")
        validated[key] = value
    return dict(sorted(validated.items()))


def _fsync_directory(path: Path) -> None:
    """Flush directory metadata when the platform supports it."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish bytes through an adjacent create-only temporary file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Atomically publish a PyTorch artifact."""
    torch = _require_torch()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _make_read_only(path: Path) -> None:
    """Remove all write bits after an artifact has been durably published."""
    os.chmod(path, 0o444)
    _fsync_directory(path.parent)


def _safe_torch_load(path: Path, map_location: str | Any = "cpu") -> Any:
    """Load a trusted tensor-state artifact with restricted unpickling."""
    torch = _require_torch()
    try:
        try:
            from torch.torch_version import TorchVersion
        except (AttributeError, ImportError):
            TorchVersion = None
        if TorchVersion is not None and hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([TorchVersion]):
                return torch.load(
                    path,
                    map_location=map_location,
                    weights_only=True,
                )
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "This checkpoint requires a PyTorch release with weights_only loading"
        ) from error


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch CPU/CUDA random-generator states."""
    torch = _require_torch()
    python_version, python_keys, python_gauss = random.getstate()
    numpy_name, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached = (
        np.random.get_state()
    )
    cuda_states = (
        list(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else []
    )
    return {
        "python": {
            "version": int(python_version),
            "keys": [int(value) for value in python_keys],
            "gauss": None if python_gauss is None else float(python_gauss),
        },
        "numpy": {
            "generator": str(numpy_name),
            "keys": [int(value) for value in numpy_keys.tolist()],
            "position": int(numpy_position),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(numpy_cached),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
    }


def _validate_rng_state_schema(state: Mapping[str, Any]) -> None:
    """Validate every RNG component before checkpoint publication or mutation."""
    torch = _require_torch()
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(state, Mapping) or set(state) != required:
        raise ValueError("RNG state is incomplete or has unexpected fields")
    python_state = state["python"]
    numpy_state = state["numpy"]
    if not isinstance(python_state, Mapping) or set(python_state) != {
        "version",
        "keys",
        "gauss",
    }:
        raise ValueError("Python RNG state fields differ from the locked schema")
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("NumPy RNG state fields differ from the locked schema")
    if (
        isinstance(python_state["version"], bool)
        or python_state["version"] != 3
    ):
        raise ValueError("Python RNG version must equal 3")
    python_keys = python_state["keys"]
    if (
        not isinstance(python_keys, list)
        or len(python_keys) != 625
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value > 2**32 - 1
            for value in python_keys[:-1]
        )
        or isinstance(python_keys[-1], bool)
        or not isinstance(python_keys[-1], int)
        or not 0 <= python_keys[-1] <= 624
    ):
        raise ValueError("Python RNG keys do not match the MT19937 state schema")
    gauss = python_state["gauss"]
    if gauss is not None and (
        isinstance(gauss, bool)
        or not isinstance(gauss, (int, float))
        or not math.isfinite(float(gauss))
    ):
        raise ValueError("Python RNG Gaussian cache must be finite or null")
    numpy_keys = numpy_state["keys"]
    if (
        numpy_state["generator"] != "MT19937"
        or not isinstance(numpy_keys, list)
        or len(numpy_keys) != 624
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 2**32 - 1
            for value in numpy_keys
        )
    ):
        raise ValueError("NumPy RNG keys do not match the MT19937 state schema")
    position = numpy_state["position"]
    has_gauss = numpy_state["has_gauss"]
    cached_gaussian = numpy_state["cached_gaussian"]
    if (
        isinstance(position, bool)
        or not isinstance(position, int)
        or not 0 <= position <= 624
    ):
        raise ValueError("NumPy RNG position is invalid")
    if isinstance(has_gauss, bool) or has_gauss not in {0, 1}:
        raise ValueError("NumPy RNG has_gauss must be zero or one")
    if (
        isinstance(cached_gaussian, bool)
        or not isinstance(cached_gaussian, (int, float))
        or not math.isfinite(float(cached_gaussian))
    ):
        raise ValueError("NumPy RNG Gaussian cache must be finite")

    def validate_torch_state(value: Any, name: str) -> None:
        if (
            not torch.is_tensor(value)
            or value.layout != torch.strided
            or value.device.type != "cpu"
            or value.dtype != torch.uint8
            or value.ndim != 1
            or value.numel() == 0
        ):
            raise ValueError(f"{name} must be a nonempty one-dimensional CPU uint8 tensor")

    validate_torch_state(state["torch_cpu"], "torch_cpu")
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, list):
        raise ValueError("torch_cuda must be a rank-local list")
    for index, cuda_state in enumerate(cuda_states):
        validate_torch_state(cuda_state, f"torch_cuda[{index}]")


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore Python, NumPy, and PyTorch CPU/CUDA random-generator states."""
    torch = _require_torch()
    _validate_rng_state_schema(state)
    python_state = state["python"]
    numpy_state = state["numpy"]
    cuda_states = list(state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore captured CUDA RNG state without CUDA")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "Captured CUDA RNG device count does not match the current process"
            )
    python_keys = tuple(int(value) for value in python_state["keys"])
    random.setstate(
        (
            int(python_state["version"]),
            python_keys,
            (
                None
                if python_state["gauss"] is None
                else float(python_state["gauss"])
            ),
        )
    )
    numpy_keys = np.asarray(numpy_state["keys"], dtype=np.uint32)
    np.random.set_state(
        (
            str(numpy_state["generator"]),
            numpy_keys,
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].to(device="cpu", dtype=torch.uint8))
    if cuda_states:
        torch.cuda.set_rng_state_all(
            [value.to(device="cpu", dtype=torch.uint8) for value in cuda_states]
        )


def create_checkpoint_generation(
    root: Path,
    generation: str,
    *,
    model: Any,
    optimizer: Any | None,
    scheduler: Any | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    provenance_hashes: Mapping[str, str],
    config: Mapping[str, Any],
    rng_states: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create one immutable checkpoint generation and its SHA-256 receipt."""
    generation = _validate_generation(generation)
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError("global_step must be a non-negative integer")
    if model is None or not hasattr(model, "state_dict"):
        raise TypeError("model must provide state_dict")
    for name, component in (
        ("optimizer", optimizer),
        ("scheduler", scheduler),
        ("scaler", scaler),
    ):
        if component is not None and not hasattr(component, "state_dict"):
            raise TypeError(f"{name} must provide state_dict when supplied")

    canonical_hashes = _validate_hashes(provenance_hashes)
    canonical_config = _json_roundtrip(config, "config")
    if not isinstance(canonical_config, dict):
        raise ValueError("config must be a JSON object")
    config_bytes = _canonical_json_bytes(canonical_config)
    config_sha256 = _sha256_bytes(config_bytes)
    captured_rng_states = (
        [capture_rng_state()] if rng_states is None else list(rng_states)
    )
    if not captured_rng_states:
        raise ValueError("rng_states must contain at least one rank state")
    for rank_state in captured_rng_states:
        _validate_rng_state_schema(rank_state)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generation_path = root / generation
    generation_path.mkdir(exist_ok=False)
    _fsync_directory(root)

    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "generation": generation,
        "epoch": epoch,
        "global_step": global_step,
        "provenance_hashes": canonical_hashes,
        "config": canonical_config,
        "config_sha256": config_sha256,
        "rng_states": captured_rng_states,
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
    }
    checkpoint_path = generation_path / CHECKPOINT_FILENAME
    _atomic_torch_save(payload, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    receipt = {
        "format_version": RECEIPT_FORMAT,
        "generation": generation,
        "checkpoint_file": CHECKPOINT_FILENAME,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "epoch": epoch,
        "global_step": global_step,
        "rng_world_size": len(captured_rng_states),
        "config_sha256": config_sha256,
        "provenance_hashes": canonical_hashes,
    }
    _atomic_write_bytes(
        generation_path / RECEIPT_FILENAME,
        _canonical_json_bytes(receipt),
    )
    _make_read_only(checkpoint_path)
    _make_read_only(generation_path / RECEIPT_FILENAME)
    return receipt


def load_checkpoint_generation(
    root: Path,
    generation: str,
    *,
    map_location: str | Any = "cpu",
    expected_provenance_hashes: Mapping[str, str] | None = None,
    expected_config_sha256: str | None = None,
    expected_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a generation receipt and safely load its checkpoint payload."""
    generation = _validate_generation(generation)
    generation_path = Path(root) / generation
    if generation_path.is_symlink() or not generation_path.is_dir():
        raise FileNotFoundError(f"Checkpoint generation does not exist: {generation}")
    receipt_path = generation_path / RECEIPT_FILENAME
    checkpoint_path = generation_path / CHECKPOINT_FILENAME
    receipt_sha256 = sha256_file(receipt_path)
    if expected_receipt_sha256 is not None:
        if (
            not isinstance(expected_receipt_sha256, str)
            or not _SHA256_PATTERN.fullmatch(expected_receipt_sha256)
        ):
            raise ValueError("expected_receipt_sha256 must be a lowercase SHA-256")
        if receipt_sha256 != expected_receipt_sha256:
            raise ValueError("Checkpoint receipt SHA-256 does not match its external anchor")
    receipt = _strict_json_object(receipt_path)
    required_receipt = {
        "format_version",
        "generation",
        "checkpoint_file",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "epoch",
        "global_step",
        "rng_world_size",
        "config_sha256",
        "provenance_hashes",
    }
    if set(receipt) != required_receipt:
        raise ValueError("Checkpoint receipt fields do not match the locked schema")
    if receipt["format_version"] != RECEIPT_FORMAT:
        raise ValueError("Unsupported checkpoint receipt format")
    if receipt["generation"] != generation:
        raise ValueError("Checkpoint receipt generation mismatch")
    if receipt["checkpoint_file"] != CHECKPOINT_FILENAME:
        raise ValueError("Checkpoint receipt points to an unexpected file")
    for field in ("checkpoint_size_bytes", "epoch", "global_step", "rng_world_size"):
        value = receipt[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Checkpoint receipt field {field!r} must be non-negative")
    if receipt["checkpoint_size_bytes"] == 0:
        raise ValueError("Checkpoint receipt cannot describe an empty artifact")
    if receipt["rng_world_size"] == 0:
        raise ValueError("Checkpoint receipt must contain at least one RNG rank")
    receipt_hash = receipt["checkpoint_sha256"]
    if not isinstance(receipt_hash, str) or not _SHA256_PATTERN.fullmatch(receipt_hash):
        raise ValueError("Checkpoint receipt contains an invalid SHA-256 digest")
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("Checkpoint artifact must be a regular non-symlink file")
    if checkpoint_path.stat().st_size != receipt["checkpoint_size_bytes"]:
        raise ValueError("Checkpoint size does not match its immutable receipt")
    actual_hash = sha256_file(checkpoint_path)
    if actual_hash != receipt_hash:
        raise ValueError("Checkpoint SHA-256 does not match its immutable receipt")

    receipt_hashes = _validate_hashes(receipt["provenance_hashes"])
    if expected_provenance_hashes is not None:
        expected_hashes = _validate_hashes(expected_provenance_hashes)
        if receipt_hashes != expected_hashes:
            raise ValueError("Checkpoint provenance hashes do not match expectations")
    receipt_config_hash = receipt["config_sha256"]
    if (
        not isinstance(receipt_config_hash, str)
        or not _SHA256_PATTERN.fullmatch(receipt_config_hash)
    ):
        raise ValueError("Checkpoint receipt contains an invalid config digest")
    if expected_config_sha256 is not None:
        if (
            not isinstance(expected_config_sha256, str)
            or not _SHA256_PATTERN.fullmatch(expected_config_sha256)
        ):
            raise ValueError("expected_config_sha256 must be a lowercase SHA-256")
        if receipt_config_hash != expected_config_sha256:
            raise ValueError("Checkpoint config digest does not match expectations")

    payload = _safe_torch_load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping")
    required_payload = {
        "format_version",
        "generation",
        "epoch",
        "global_step",
        "provenance_hashes",
        "config",
        "config_sha256",
        "rng_states",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
    }
    if set(payload) != required_payload:
        raise ValueError("Checkpoint payload fields do not match the locked schema")
    if payload["format_version"] != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported checkpoint payload format")
    if payload["generation"] != generation:
        raise ValueError("Checkpoint payload generation mismatch")
    for field in ("epoch", "global_step"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Checkpoint payload field {field!r} must be non-negative")
    if payload["epoch"] != receipt["epoch"]:
        raise ValueError("Checkpoint epoch does not match its receipt")
    if payload["global_step"] != receipt["global_step"]:
        raise ValueError("Checkpoint global step does not match its receipt")
    rng_states = payload["rng_states"]
    if not isinstance(rng_states, list) or not rng_states:
        raise ValueError("Checkpoint RNG states must be a non-empty rank list")
    if len(rng_states) != receipt["rng_world_size"]:
        raise ValueError("Checkpoint RNG world size does not match its receipt")
    for rank_state in rng_states:
        _validate_rng_state_schema(rank_state)
    payload_hashes = _validate_hashes(payload["provenance_hashes"])
    if payload_hashes != receipt_hashes:
        raise ValueError("Checkpoint provenance does not match its receipt")
    canonical_config = _json_roundtrip(payload["config"], "checkpoint config")
    if not isinstance(canonical_config, dict):
        raise ValueError("Checkpoint config must be a JSON object")
    actual_config_hash = _sha256_bytes(_canonical_json_bytes(canonical_config))
    if payload["config_sha256"] != actual_config_hash:
        raise ValueError("Checkpoint payload config digest is internally inconsistent")
    if actual_config_hash != receipt_config_hash:
        raise ValueError("Checkpoint config digest does not match its receipt")
    payload["config"] = canonical_config
    payload["provenance_hashes"] = payload_hashes
    payload["receipt"] = receipt
    payload["receipt_sha256"] = receipt_sha256
    payload["external_receipt_anchor_verified"] = (
        expected_receipt_sha256 is not None
    )
    payload["checkpoint_sha256"] = actual_hash
    return payload


def _restore_verified_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    model: Any,
    optimizer: Any | None,
    scheduler: Any | None,
    scaler: Any | None,
    expected_rng_world_size: int | None = None,
    restore_random_generators: bool = True,
    rng_rank: int = 0,
) -> dict[str, Any]:
    """Apply one payload returned by :func:`load_checkpoint_generation`."""
    required_verified_fields = {
        "generation",
        "epoch",
        "global_step",
        "rng_states",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "config",
        "config_sha256",
        "provenance_hashes",
        "checkpoint_sha256",
        "receipt_sha256",
        "external_receipt_anchor_verified",
        "receipt",
    }
    if not isinstance(payload, Mapping) or not required_verified_fields.issubset(payload):
        raise ValueError(
            "payload must be the verified output of load_checkpoint_generation"
        )
    if model is None or not hasattr(model, "load_state_dict"):
        raise TypeError("model must provide load_state_dict")
    components = (
        ("optimizer", optimizer, payload["optimizer_state"]),
        ("scheduler", scheduler, payload["scheduler_state"]),
        ("scaler", scaler, payload["scaler_state"]),
    )
    for name, component, saved_state in components:
        if (component is None) != (saved_state is None):
            raise ValueError(
                f"{name} presence must match the immutable checkpoint for exact resume"
            )
        if component is not None and not hasattr(component, "load_state_dict"):
            raise TypeError(f"{name} must provide load_state_dict")

    if isinstance(rng_rank, bool) or not isinstance(rng_rank, int) or rng_rank < 0:
        raise ValueError("rng_rank must be a non-negative integer")
    if rng_rank >= len(payload["rng_states"]):
        raise ValueError("rng_rank exceeds the checkpoint RNG world size")
    if expected_rng_world_size is not None:
        if (
            isinstance(expected_rng_world_size, bool)
            or not isinstance(expected_rng_world_size, int)
            or expected_rng_world_size <= 0
        ):
            raise ValueError("expected_rng_world_size must be a positive integer")
        if len(payload["rng_states"]) != expected_rng_world_size:
            raise ValueError(
                "Checkpoint RNG world size differs from the required runtime"
            )
    if restore_random_generators and payload["rng_states"][rng_rank]["torch_cuda"]:
        torch = _require_torch()
        captured_cuda_count = len(
            payload["rng_states"][rng_rank]["torch_cuda"]
        )
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore captured CUDA RNG state without CUDA")
        if captured_cuda_count != torch.cuda.device_count():
            raise RuntimeError(
                "Captured CUDA RNG device count does not match the current process"
            )

    model.load_state_dict(payload["model_state"], strict=True)
    for _, component, saved_state in components:
        if component is not None:
            component.load_state_dict(saved_state)
    if restore_random_generators:
        restore_rng_state(payload["rng_states"][rng_rank])
    return {
        "generation": payload["generation"],
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "rng_world_size": len(payload["rng_states"]),
        "config": payload["config"],
        "config_sha256": payload["config_sha256"],
        "provenance_hashes": payload["provenance_hashes"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "receipt_sha256": payload["receipt_sha256"],
        "external_receipt_anchor_verified": payload[
            "external_receipt_anchor_verified"
        ],
        "receipt": payload["receipt"],
    }


def restore_checkpoint_generation(
    root: Path,
    generation: str,
    *,
    model: Any,
    optimizer: Any | None,
    scheduler: Any | None,
    scaler: Any | None,
    map_location: str | Any = "cpu",
    expected_provenance_hashes: Mapping[str, str] | None = None,
    expected_config_sha256: str | None = None,
    expected_receipt_sha256: str | None = None,
    expected_rng_world_size: int | None = None,
    restore_random_generators: bool = True,
    rng_rank: int = 0,
) -> dict[str, Any]:
    """Verify and restore all supplied training components and RNG state."""
    payload = load_checkpoint_generation(
        root,
        generation,
        map_location=map_location,
        expected_provenance_hashes=expected_provenance_hashes,
        expected_config_sha256=expected_config_sha256,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    return _restore_verified_checkpoint_payload(
        payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_rng_world_size=expected_rng_world_size,
        restore_random_generators=restore_random_generators,
        rng_rank=rng_rank,
    )
