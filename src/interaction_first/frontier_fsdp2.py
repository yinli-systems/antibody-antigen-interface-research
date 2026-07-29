"""Fail-closed FSDP2 and distributed-checkpoint engineering primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


FSDP2_DCP_FORMAT = "frontier_fsdp2_dcp_generation_v1"
FSDP2_RUNTIME_FORMAT = "frontier_fsdp2_runtime_receipt_v1"
FSDP2_CLAIM_BOUNDARY = (
    "Synthetic train-only systems evidence. This artifact does not authorize "
    "biological training, validation, outer evaluation, frozen-test access, "
    "scaling-efficiency claims, or state-of-the-art claims."
)
_GENERATION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FSDP2ContractError(ValueError):
    """Report an invalid or unverifiable FSDP2 engineering contract."""


def _require_torch() -> Any:
    """Import PyTorch lazily for metadata-only repository workflows."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for FSDP2 engineering") from error
    return torch


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite JSON deterministically."""
    try:
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
    except (TypeError, ValueError, OverflowError) as error:
        raise FSDP2ContractError("Metadata must be finite and JSON-compatible") from error


def _reject_constant(value: str) -> None:
    """Reject non-finite JSON constants."""
    raise FSDP2ContractError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Construct a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FSDP2ContractError(f"Duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _strict_json_object(path: Path) -> dict[str, Any]:
    """Read one finite regular JSON object without following symlinks."""
    if path.is_symlink() or not path.is_file():
        raise FSDP2ContractError(f"Expected a regular JSON file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FSDP2ContractError(f"Cannot read strict JSON from {path}") from error
    if not isinstance(value, dict):
        raise FSDP2ContractError(f"JSON payload at {path} must be an object")
    return value


def _sha256_file(path: Path) -> str:
    """Hash one regular file without accepting a symlink."""
    if path.is_symlink() or not path.is_file():
        raise FSDP2ContractError(f"Expected a regular checkpoint file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise FSDP2ContractError(f"Expected a regular checkpoint file: {path}")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_generation(generation: str) -> str:
    """Validate a checkpoint generation before path construction."""
    if not isinstance(generation, str) or _GENERATION_PATTERN.fullmatch(generation) is None:
        raise FSDP2ContractError(
            "generation must be a 1-128 character path-safe ASCII label"
        )
    if generation in {".", ".."}:
        raise FSDP2ContractError("generation cannot be a relative path token")
    return generation


def _validate_hashes(value: Mapping[str, str]) -> dict[str, str]:
    """Validate and sort externally anchored provenance hashes."""
    if not isinstance(value, Mapping) or not value:
        raise FSDP2ContractError("provenance_hashes must be a non-empty mapping")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str) or not key:
            raise FSDP2ContractError("Provenance names must be non-empty strings")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise FSDP2ContractError(f"Invalid SHA-256 for provenance field {key!r}")
        result[key] = digest
    return dict(sorted(result.items()))


def _distributed_identity(group: Any = None) -> tuple[Any, int, int]:
    """Return distributed module, world size, and rank."""
    torch = _require_torch()
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return distributed, 1, 0
    return (
        distributed,
        int(distributed.get_world_size(group=group)),
        int(distributed.get_rank(group=group)),
    )


def validate_fsdp2_runtime(
    *,
    parallelism: str,
    precision: str,
    cuda_available: bool,
    bf16_supported: bool,
    distributed_initialized: bool,
    world_size: int,
    torch_version: str,
) -> dict[str, Any]:
    """Validate the deliberately narrow FSDP2 engineering runtime."""
    if parallelism != "fsdp2":
        raise FSDP2ContractError("parallelism must be exactly 'fsdp2'")
    if precision not in {"float32", "bfloat16"}:
        raise FSDP2ContractError("precision must be float32 or bfloat16")
    if not cuda_available:
        raise FSDP2ContractError("FSDP2 execution requires CUDA")
    if precision == "bfloat16" and not bf16_supported:
        raise FSDP2ContractError("The selected CUDA device does not support bfloat16")
    if not distributed_initialized:
        raise FSDP2ContractError("FSDP2 execution requires an initialized process group")
    if type(world_size) is not int or world_size < 1:
        raise FSDP2ContractError("world_size must be a positive integer")
    if not isinstance(torch_version, str) or not torch_version:
        raise FSDP2ContractError("torch_version must be a non-empty string")
    return {
        "format": FSDP2_RUNTIME_FORMAT,
        "parallelism": parallelism,
        "precision": precision,
        "cuda_available": True,
        "bf16_supported": bool(bf16_supported),
        "distributed_initialized": True,
        "world_size": world_size,
        "torch_version": torch_version,
        "gradient_reduction": "fsdp2_reduce_scatter_average",
        "checkpoint_backend": "torch_distributed_checkpoint_sync",
        "compile_enabled": False,
        "fp8_enabled": False,
        "claim_boundary": FSDP2_CLAIM_BOUNDARY,
    }


def fully_shard_reranker(
    model: Any,
    *,
    precision: str,
    mesh: Any = None,
    reshard_after_forward: bool = True,
) -> Any:
    """Apply FSDP2 bottom-up to reranker blocks and then the root module."""
    torch = _require_torch()
    if precision not in {"float32", "bfloat16"}:
        raise FSDP2ContractError("precision must be float32 or bfloat16")
    if not isinstance(reshard_after_forward, bool):
        raise FSDP2ContractError("reshard_after_forward must be boolean")
    if not hasattr(model, "blocks") or not isinstance(model.blocks, torch.nn.ModuleList):
        raise FSDP2ContractError("model must expose a ModuleList named blocks")
    if len(model.blocks) == 0:
        raise FSDP2ContractError("model.blocks cannot be empty")
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except (ImportError, AttributeError) as error:
        raise RuntimeError("This PyTorch runtime does not provide public FSDP2") from error
    dtype = torch.bfloat16 if precision == "bfloat16" else None
    policy = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=dtype,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    frozen = {parameter for parameter in model.parameters() if not parameter.requires_grad}
    for block in model.blocks:
        fully_shard(
            block,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=policy,
        )
    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=policy,
        ignored_params=frozen or None,
    )
    return model


def _fsync_directory(path: Path) -> None:
    """Flush directory metadata when supported by the platform."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_create_only_json(path: Path, value: Any) -> None:
    """Publish canonical JSON without overwriting an existing artifact."""
    payload = _canonical_json_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite artifact: {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_files(path: Path) -> list[Path]:
    """Return a deterministic flat inventory of DCP files."""
    files: list[Path] = []
    for candidate in sorted(path.iterdir(), key=lambda item: item.name):
        if candidate.name == "receipt.json":
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise FSDP2ContractError(
                f"DCP generation must contain only regular files: {candidate}"
            )
        files.append(candidate)
    if not files or not any(candidate.name == ".metadata" for candidate in files):
        raise FSDP2ContractError("DCP generation is missing its .metadata file")
    return files


def _maximum_elapsed(elapsed: float, group: Any = None) -> tuple[float, list[float]]:
    """Collect finite rank-local elapsed times and return their maximum."""
    distributed, world_size, _ = _distributed_identity(group)
    observed = float(elapsed)
    if not math.isfinite(observed) or observed < 0.0:
        raise FSDP2ContractError("Checkpoint elapsed time must be finite and non-negative")
    if world_size == 1:
        values = [observed]
    else:
        values = [0.0 for _ in range(world_size)]
        distributed.all_gather_object(values, observed, group=group)
    validated = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in validated):
        raise FSDP2ContractError("A rank reported an invalid checkpoint elapsed time")
    return max(validated), validated


def save_dcp_generation(
    root: Path | str,
    generation: str,
    *,
    model: Any,
    optimizer: Any,
    provenance_hashes: Mapping[str, str],
    training_state: Mapping[str, Any],
    group: Any = None,
) -> dict[str, Any]:
    """Collectively save one immutable, hash-inventoried DCP generation."""
    torch = _require_torch()
    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_state_dict,
        )
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Distributed Checkpoint is unavailable") from error
    selected_generation = _validate_generation(generation)
    provenance = _validate_hashes(provenance_hashes)
    canonical_training_state = json.loads(_canonical_json_bytes(training_state))
    distributed, world_size, rank = _distributed_identity(group)
    output_root = Path(root).resolve()
    final_path = output_root / selected_generation
    preflight: dict[str, str | None] = {"temporary_name": None, "error": None}
    if rank == 0:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            if output_root.is_symlink() or not output_root.is_dir():
                raise FSDP2ContractError(
                    f"Checkpoint root is not a regular directory: {output_root}"
                )
            if final_path.exists() or final_path.is_symlink():
                raise FileExistsError(
                    f"Refusing to overwrite checkpoint generation: {final_path}"
                )
            preflight["temporary_name"] = (
                f".{selected_generation}.tmp-{uuid.uuid4().hex}"
            )
        except BaseException as error:
            preflight["error"] = f"{type(error).__name__}: {error}"
    if world_size > 1:
        values: list[dict[str, str | None]] = [preflight]
        distributed.broadcast_object_list(values, src=0, group=group)
        preflight = values[0]
    if preflight["error"] is not None:
        if preflight["error"].startswith("FileExistsError:"):
            raise FileExistsError(preflight["error"])
        raise FSDP2ContractError(preflight["error"])
    temporary_name = preflight["temporary_name"]
    if not temporary_name:
        raise RuntimeError("Rank zero did not publish a checkpoint staging directory")
    temporary_path = output_root / temporary_name
    if rank == 0:
        temporary_path.mkdir(mode=0o700)
    if world_size > 1:
        distributed.barrier(group=group)
    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    started = time.perf_counter()
    try:
        dcp.save(
            {"model": model_state, "optimizer": optimizer_state},
            checkpoint_id=temporary_path,
            process_group=group,
            no_dist=world_size == 1,
        )
        elapsed = time.perf_counter() - started
        maximum_seconds, rank_seconds = _maximum_elapsed(elapsed, group=group)
        if world_size > 1:
            distributed.barrier(group=group)
        receipt: dict[str, Any] | None = None
        finalize_error: str | None = None
        if rank == 0:
            try:
                files = _checkpoint_files(temporary_path)
                inventory = {
                    candidate.name: {
                        "bytes": candidate.stat().st_size,
                        "sha256": _sha256_file(candidate),
                    }
                    for candidate in files
                }
                unsigned = {
                    "format": FSDP2_DCP_FORMAT,
                    "generation": selected_generation,
                    "status": "complete",
                    "source_world_size": world_size,
                    "torch_version": torch.__version__,
                    "state_dict_options": {
                        "full_state_dict": False,
                        "cpu_offload": False,
                        "strict": True,
                    },
                    "save_seconds_max": maximum_seconds,
                    "save_seconds_by_rank": rank_seconds,
                    "provenance_hashes": provenance,
                    "training_state": canonical_training_state,
                    "files": inventory,
                    "claim_boundary": FSDP2_CLAIM_BOUNDARY,
                }
                receipt = {
                    **unsigned,
                    "content_sha256": hashlib.sha256(
                        _canonical_json_bytes(unsigned)
                    ).hexdigest(),
                }
                _write_create_only_json(temporary_path / "receipt.json", receipt)
                os.replace(temporary_path, final_path)
                for candidate in final_path.iterdir():
                    os.chmod(candidate, 0o444)
                os.chmod(final_path, 0o555)
                _fsync_directory(output_root)
            except BaseException as error:
                finalize_error = f"{type(error).__name__}: {error}"
        if world_size > 1:
            values = [finalize_error]
            distributed.broadcast_object_list(values, src=0, group=group)
            finalize_error = values[0]
        if finalize_error is not None:
            raise FSDP2ContractError(
                f"Rank-zero checkpoint finalization failed: {finalize_error}"
            )
        return verify_dcp_generation(
            output_root,
            selected_generation,
            expected_provenance_hashes=provenance,
        )
    except BaseException:
        if rank == 0 and temporary_path.exists() and temporary_path.is_dir():
            shutil.rmtree(temporary_path)
        raise


def verify_dcp_generation(
    root: Path | str,
    generation: str,
    *,
    expected_provenance_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify one immutable DCP generation without loading tensors."""
    selected_generation = _validate_generation(generation)
    path = Path(root).resolve() / selected_generation
    if path.is_symlink() or not path.is_dir():
        raise FSDP2ContractError(f"Expected a regular checkpoint directory: {path}")
    receipt = _strict_json_object(path / "receipt.json")
    expected_fields = {
        "format",
        "generation",
        "status",
        "source_world_size",
        "torch_version",
        "state_dict_options",
        "save_seconds_max",
        "save_seconds_by_rank",
        "provenance_hashes",
        "training_state",
        "files",
        "claim_boundary",
        "content_sha256",
    }
    if set(receipt) != expected_fields:
        raise FSDP2ContractError("DCP receipt fields differ from the locked schema")
    if (
        receipt["format"] != FSDP2_DCP_FORMAT
        or receipt["generation"] != selected_generation
        or receipt["status"] != "complete"
        or receipt["claim_boundary"] != FSDP2_CLAIM_BOUNDARY
    ):
        raise FSDP2ContractError("DCP receipt identity or status is invalid")
    unsigned = {key: value for key, value in receipt.items() if key != "content_sha256"}
    expected_content = receipt["content_sha256"]
    if (
        not isinstance(expected_content, str)
        or _SHA256_PATTERN.fullmatch(expected_content) is None
    ):
        raise FSDP2ContractError("DCP receipt content SHA-256 is invalid")
    observed_content = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if observed_content != expected_content:
        raise FSDP2ContractError("DCP receipt content SHA-256 mismatch")
    source_world_size = receipt["source_world_size"]
    rank_seconds = receipt["save_seconds_by_rank"]
    if (
        type(source_world_size) is not int
        or source_world_size < 1
        or not isinstance(rank_seconds, list)
        or len(rank_seconds) != source_world_size
    ):
        raise FSDP2ContractError("DCP source-world timing metadata is invalid")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in [receipt["save_seconds_max"], *rank_seconds]
    ):
        raise FSDP2ContractError("DCP timing metadata must be finite and non-negative")
    provenance = _validate_hashes(receipt["provenance_hashes"])
    if expected_provenance_hashes is not None:
        expected = _validate_hashes(expected_provenance_hashes)
        if provenance != expected:
            raise FSDP2ContractError("DCP provenance hashes differ from expectation")
    inventory = receipt["files"]
    if not isinstance(inventory, dict) or not inventory:
        raise FSDP2ContractError("DCP receipt file inventory is empty")
    observed_names = {
        candidate.name
        for candidate in path.iterdir()
        if candidate.name != "receipt.json"
    }
    if observed_names != set(inventory):
        raise FSDP2ContractError("DCP file inventory differs from on-disk contents")
    for name, metadata in inventory.items():
        if Path(name).name != name or name in {".", "..", "receipt.json"}:
            raise FSDP2ContractError(f"Invalid DCP inventory name: {name!r}")
        if not isinstance(metadata, dict) or set(metadata) != {"bytes", "sha256"}:
            raise FSDP2ContractError(f"Invalid DCP inventory metadata for {name!r}")
        candidate = path / name
        if type(metadata["bytes"]) is not int or metadata["bytes"] < 0:
            raise FSDP2ContractError(f"Invalid byte count for {name!r}")
        if candidate.stat().st_size != metadata["bytes"]:
            raise FSDP2ContractError(f"DCP byte count mismatch for {name!r}")
        if _sha256_file(candidate) != metadata["sha256"]:
            raise FSDP2ContractError(f"DCP SHA-256 mismatch for {name!r}")
    return receipt


def load_dcp_generation(
    root: Path | str,
    generation: str,
    *,
    model: Any,
    optimizer: Any,
    expected_provenance_hashes: Mapping[str, str],
    group: Any = None,
) -> dict[str, Any]:
    """Verify and collectively restore model and optimizer state through DCP."""
    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_state_dict,
            set_state_dict,
        )
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Distributed Checkpoint is unavailable") from error
    selected_generation = _validate_generation(generation)
    receipt = verify_dcp_generation(
        root,
        selected_generation,
        expected_provenance_hashes=expected_provenance_hashes,
    )
    distributed, world_size, _ = _distributed_identity(group)
    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    dcp.load(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=Path(root).resolve() / selected_generation,
        process_group=group,
        no_dist=world_size == 1,
    )
    incompatible = set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state,
        optim_state_dict=optimizer_state,
        options=options,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise FSDP2ContractError(
            "DCP restore returned incompatible model keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if world_size > 1:
        distributed.barrier(group=group)
    return receipt


def percentile(values: Sequence[float], quantile: float) -> float:
    """Compute a deterministic linear-interpolated percentile."""
    if not values:
        raise FSDP2ContractError("Percentile input cannot be empty")
    if not 0.0 <= float(quantile) <= 1.0:
        raise FSDP2ContractError("quantile must be within [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in ordered):
        raise FSDP2ContractError("Timing values must be finite and non-negative")
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
