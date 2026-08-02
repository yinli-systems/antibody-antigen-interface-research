#!/usr/bin/env python3
"""Compute protocol-bound useful-token approximate 6N MFU receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping


FORMAT_VERSION = "antibody_approximate_6n_mfu_audit_v1"
RUN_LABEL = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class MfuAuditError(RuntimeError):
    """Report an invalid input receipt or MFU protocol disagreement."""


def _reject_constant(value: str) -> None:
    raise MfuAuditError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise MfuAuditError(f"Duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize one audit receipt with stable ASCII bytes."""
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


def _load_json(
    path: Path, *, require_canonical: bool = True
) -> tuple[dict[str, Any], bytes]:
    """Load strict JSON from a regular non-symlink file."""
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MfuAuditError(f"Input is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MfuAuditError(f"Input is not strict ASCII JSON: {path}") from error
    if not isinstance(value, dict) or (
        require_canonical and raw != _canonical_json_bytes(value)
    ):
        raise MfuAuditError(f"Input is not canonical JSON: {path}")
    return value, raw


def _finite_positive(value: Any, name: str) -> float:
    """Return one finite strictly positive number."""
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise MfuAuditError(f"{name} must be finite and positive")
    return result


def _parameter_count(receipt: Mapping[str, Any]) -> int:
    """Recover the exact counted parameters from the training receipt."""
    moments = receipt.get("parameter_moments")
    if not isinstance(moments, Mapping):
        raise MfuAuditError("Parameter moments are absent")
    count = moments.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise MfuAuditError("Parameter count is invalid")
    logging = receipt.get("ddp_logging_data")
    if isinstance(logging, Mapping) and "total_parameter_size_bytes" in logging:
        if logging["total_parameter_size_bytes"] != count * 4:
            raise MfuAuditError("DDP parameter bytes differ from the exact count")
    return count


def _run_summary(
    receipt: Mapping[str, Any],
    *,
    warmup_updates: int,
    hardware: str,
    peak_tflops: float,
) -> dict[str, Any]:
    """Replay one useful-token throughput window and calculate approximate MFU."""
    if receipt.get("device") != hardware:
        raise MfuAuditError("Training device differs from the scale protocol")
    world_size = receipt.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise MfuAuditError("World size is invalid")
    trace = receipt.get("trace")
    if not isinstance(trace, list) or len(trace) <= warmup_updates:
        raise MfuAuditError("Trace is too short for the frozen warmup window")
    previous_consumed = 0
    for expected_update, item in enumerate(trace, start=1):
        if not isinstance(item, Mapping) or item.get("update") != expected_update:
            raise MfuAuditError("Training updates are not contiguous")
        tokens = item.get("residue_tokens")
        consumed = item.get("consumed_residue_tokens")
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
            or isinstance(consumed, bool)
            or not isinstance(consumed, int)
            or consumed != previous_consumed + tokens
        ):
            raise MfuAuditError("Residue-token accounting is invalid")
        _finite_positive(item.get("seconds"), "Step seconds")
        previous_consumed = consumed
    measured = trace[warmup_updates:]
    measured_tokens = sum(int(item["residue_tokens"]) for item in measured)
    measured_seconds = sum(
        _finite_positive(item["seconds"], "Step seconds") for item in measured
    )
    parameter_count = _parameter_count(receipt)
    useful_tflop = 6.0 * parameter_count * measured_tokens / 1.0e12
    available_tflop = world_size * peak_tflops * measured_seconds
    approximate_mfu = useful_tflop / available_tflop
    if not 0.0 < approximate_mfu <= 1.0:
        raise MfuAuditError("Approximate MFU falls outside the physical domain")
    return {
        "approximate_6n_mfu": approximate_mfu,
        "available_dense_bf16_tflop": available_tflop,
        "measured_seconds": measured_seconds,
        "measured_updates": len(measured),
        "nonpadding_residue_tokens": measured_tokens,
        "parameter_count": parameter_count,
        "residue_tokens_per_second": measured_tokens / measured_seconds,
        "useful_6n_tflop": useful_tflop,
        "warmup_updates_excluded": warmup_updates,
        "world_size": world_size,
    }


def audit_mfu(
    *,
    protocol_path: Path,
    runs: Mapping[str, Path],
    warmup_updates: int,
) -> dict[str, Any]:
    """Audit multiple receipts and pool only identical hardware/model domains."""
    if warmup_updates < 0 or not runs:
        raise MfuAuditError("Warmup and run inventory are invalid")
    protocol, protocol_raw = _load_json(protocol_path, require_canonical=False)
    if protocol.get("format_version") != "antibody_foundation_scale_protocol_v2":
        raise MfuAuditError("Scale protocol format differs")
    mfu = protocol.get("mfu")
    if not isinstance(mfu, Mapping) or mfu != {
        "denominator": (
            "observed_gpu_count_times_dense_bf16_tensor_tflops_with_fp32_accumulate"
        ),
        "estimate_name": "approximate_6n_mfu",
        "flop_estimate": (
            "6_times_exact_parameter_count_times_nonpadding_residue_tokens"
        ),
        "hardware": "NVIDIA GeForce RTX 5090",
        "per_gpu_dense_bf16_tflops": 209.5,
        "profiler_exact_claim": False,
        "sparsity_assumed": False,
        "source": (
            "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/"
            "nvidia-rtx-blackwell-gpu-architecture.pdf"
        ),
    }:
        raise MfuAuditError("Scale protocol MFU semantics differ")
    run_results: dict[str, Any] = {}
    identities: set[tuple[int, int]] = set()
    pooled_tokens = 0
    pooled_seconds = 0.0
    for label, receipt_path in sorted(runs.items()):
        if not RUN_LABEL.fullmatch(label):
            raise MfuAuditError(f"Run label is invalid: {label}")
        receipt, receipt_raw = _load_json(receipt_path)
        summary = _run_summary(
            receipt,
            warmup_updates=warmup_updates,
            hardware=str(mfu["hardware"]),
            peak_tflops=float(mfu["per_gpu_dense_bf16_tflops"]),
        )
        identities.add((summary["parameter_count"], summary["world_size"]))
        pooled_tokens += int(summary["nonpadding_residue_tokens"])
        pooled_seconds += float(summary["measured_seconds"])
        run_results[label] = {
            "input_path": str(receipt_path.resolve()),
            "input_sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "optimization": receipt.get("optimization"),
            "summary": summary,
        }
    pooled = None
    if len(identities) == 1:
        parameter_count, world_size = next(iter(identities))
        throughput = pooled_tokens / pooled_seconds
        pooled = {
            "approximate_6n_mfu": (
                6.0
                * parameter_count
                * throughput
                / (world_size * float(mfu["per_gpu_dense_bf16_tflops"]) * 1.0e12)
            ),
            "measured_seconds": pooled_seconds,
            "nonpadding_residue_tokens": pooled_tokens,
            "parameter_count": parameter_count,
            "residue_tokens_per_second": throughput,
            "world_size": world_size,
        }
    return {
        "authorization": {
            "configuration_promotion": False,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "claim_boundary": (
            "Approximate useful-token 6N MFU only; padded-token compute, attention "
            "quadratic terms, optimizer FLOPs, profiler FLOPs, convergence, and "
            "downstream quality are not claimed"
        ),
        "format_version": FORMAT_VERSION,
        "mfu_semantics": dict(mfu),
        "pooled_identical_domain": pooled,
        "protocol_file_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "runs": run_results,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }


def _parse_runs(values: list[str]) -> dict[str, Path]:
    """Parse unique label-to-receipt command-line bindings."""
    result: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or label in result:
            raise MfuAuditError("Each --run must be a unique label=path binding")
        result[label] = Path(raw_path)
    return result


def _publish_create_only(path: Path, value: dict[str, Any]) -> None:
    """Publish one immutable audit receipt without overwriting evidence."""
    payload = dict(value)
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    """Build the strict MFU auditor command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--warmup-updates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Audit the requested training receipts and publish the result."""
    arguments = _parser().parse_args()
    result = audit_mfu(
        protocol_path=arguments.protocol,
        runs=_parse_runs(arguments.run),
        warmup_updates=arguments.warmup_updates,
    )
    _publish_create_only(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
