#!/usr/bin/env python3
"""Run a source-bound compile numerics and Python-GC diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


FORMAT_VERSION = "antibody_compile_numerics_diagnostic_receipt_v1"
PROTOCOL_FORMAT = "antibody_compile_numerics_diagnostic_protocol_v1"
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
AUTHORIZATION = {
    "configuration_promotion": False,
    "production_pretraining": False,
    "scale_16_gpu": False,
    "scale_32_gpu": False,
    "scientific_claim": False,
}


class CompileNumericsDiagnosticError(RuntimeError):
    """Report an invalid diagnostic protocol, launch, or artifact."""


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompileNumericsDiagnosticError(
            "Diagnostic protocol must be strict ASCII JSON"
        ) from error
    if not isinstance(value, dict) or value.get("format_version") != PROTOCOL_FORMAT:
        raise CompileNumericsDiagnosticError("Diagnostic protocol format differs")
    return value, hashlib.sha256(raw).hexdigest()


def _one_argument(arguments: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise CompileNumericsDiagnosticError(
            f"Launch requires exactly one {option}"
        )
    return arguments[positions[0] + 1]


def _gc_stats() -> list[dict[str, int]]:
    return [
        {
            "collected": int(item["collected"]),
            "collections": int(item["collections"]),
            "uncollectable": int(item["uncollectable"]),
        }
        for item in gc.get_stats()
    ]


def _stats_delta(
    before: list[dict[str, int]], after: list[dict[str, int]]
) -> list[dict[str, int]]:
    return [
        {key: right[key] - left[key] for key in sorted(left)}
        for left, right in zip(before, after)
    ]


def _validate_launch(
    protocol: dict[str, Any],
    *,
    mode: str,
    arguments: list[str],
    scripts_root: Path,
) -> tuple[Path, dict[str, bool]]:
    modes = protocol.get("modes")
    if not isinstance(modes, dict) or mode not in modes:
        raise CompileNumericsDiagnosticError("Diagnostic mode is not registered")
    flags = modes[mode]
    if flags not in (
        {
            "accuracy_first_bf16_reduction": False,
            "compile_blocks": False,
            "disable_python_gc": False,
        },
        {
            "accuracy_first_bf16_reduction": False,
            "compile_blocks": True,
            "disable_python_gc": False,
        },
        {
            "accuracy_first_bf16_reduction": False,
            "compile_blocks": True,
            "disable_python_gc": True,
        },
        {
            "accuracy_first_bf16_reduction": True,
            "compile_blocks": True,
            "disable_python_gc": False,
        },
        {
            "accuracy_first_bf16_reduction": True,
            "compile_blocks": True,
            "disable_python_gc": True,
        },
    ):
        raise CompileNumericsDiagnosticError("Diagnostic mode contract differs")
    compile_arguments = (
        "--compile-blocks" in arguments,
        "--compile-prewarm-buckets" in arguments,
    )
    expected_compile = bool(flags["compile_blocks"])
    if compile_arguments != (expected_compile, expected_compile):
        raise CompileNumericsDiagnosticError("Compile launch flags differ from mode")
    forbidden = (
        "--fused-adamw",
        "--pinned-staging",
        "--bucketed-padding",
        "--length-bucketing",
        "--ddp-batched-grad-copy",
        "--disable-ddp-gradient-as-bucket-view",
    )
    if any(option in arguments for option in forbidden):
        raise CompileNumericsDiagnosticError(
            "Diagnostic launch contains an unregistered optimization"
        )
    output = Path(_one_argument(arguments, "--output"))
    if not output.is_absolute():
        raise CompileNumericsDiagnosticError("Diagnostic output must be absolute")
    bindings = {
        "base_runner_sha256": scripts_root / "run_antibody_pretraining_optimized.py",
        "parity_wrapper_sha256": scripts_root
        / "run_antibody_pretraining_parity_extension.py",
        "diagnostic_wrapper_sha256": Path(__file__),
    }
    if any(_sha256(path) != protocol["inputs"][key] for key, path in bindings.items()):
        raise CompileNumericsDiagnosticError("Diagnostic source binding differs")
    return output, flags


def _publish(
    output: Path,
    *,
    protocol_sha256: str,
    mode: str,
    flags: dict[str, bool],
    original_arguments: list[str],
    precollection_reclaimed: int,
    stats_before: list[dict[str, int]],
    stats_after: list[dict[str, int]],
    gc_enabled_before: bool,
    gc_enabled_after_restore: bool,
    bf16_reduction_before: bool,
    bf16_reduction_after_restore: bool,
) -> None:
    base_receipt = output / "receipt.json"
    trainer_state = output / "trainer_state.json"
    parity_sidecar = output / "parity_extension_receipt.json"
    payload = {
        "authorization": AUTHORIZATION,
        "base_receipt_file_sha256": _sha256(base_receipt),
        "base_trainer_state_file_sha256": _sha256(trainer_state),
        "bf16_reduced_precision_reduction": {
            "after_restore": bf16_reduction_after_restore,
            "before": bf16_reduction_before,
            "during_run": not flags["accuracy_first_bf16_reduction"],
        },
        "claim_boundary": (
            "Training-excluded compile numerics and Python-GC diagnostic only; "
            "no configuration promotion, production training, biological metric, "
            "multi-node scaling, or state-of-the-art claim"
        ),
        "format_version": FORMAT_VERSION,
        "launch_argv": original_arguments,
        "mode": mode,
        "mode_flags": flags,
        "parity_extension_receipt_file_sha256": _sha256(parity_sidecar),
        "protocol_sha256": protocol_sha256,
        "python_gc": {
            "enabled_after_restore": gc_enabled_after_restore,
            "enabled_before": gc_enabled_before,
            "enabled_during_run": not flags["disable_python_gc"],
            "precollection_reclaimed": precollection_reclaimed,
            "stats_after": stats_after,
            "stats_before": stats_before,
            "stats_delta": _stats_delta(stats_before, stats_after),
        },
        "test_access": TEST_ACCESS,
        "wrapper_sha256": _sha256(Path(__file__)),
    }
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    destination = output / "compile_numerics_diagnostic_receipt.json"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--compile-numerics-protocol", type=Path, required=True)
    parser.add_argument("--compile-numerics-mode", required=True)
    parsed, base_arguments = parser.parse_known_args()
    original_arguments = sys.argv[1:]
    protocol, protocol_sha256 = _strict_protocol(parsed.compile_numerics_protocol)
    if protocol.get("authorization") != AUTHORIZATION or protocol.get(
        "test_access"
    ) != TEST_ACCESS:
        raise CompileNumericsDiagnosticError("Diagnostic authority differs")
    scripts_root = Path(__file__).resolve().parent
    output, flags = _validate_launch(
        protocol,
        mode=parsed.compile_numerics_mode,
        arguments=base_arguments,
        scripts_root=scripts_root,
    )

    import torch
    import run_antibody_pretraining_parity_extension as parity

    bf16_reduction_before = bool(
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    if not bf16_reduction_before:
        raise CompileNumericsDiagnosticError(
            "Runtime BF16 reduction default differs from the frozen diagnostic"
        )
    gc_enabled_before = gc.isenabled()
    if not gc_enabled_before:
        raise CompileNumericsDiagnosticError(
            "Python cyclic GC must be enabled before the diagnostic"
        )
    precollection_reclaimed = int(gc.collect())
    stats_before = _gc_stats()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not flags[
        "accuracy_first_bf16_reduction"
    ]
    if flags["disable_python_gc"]:
        gc.disable()
    try:
        sys.argv = [
            str(scripts_root / "run_antibody_pretraining_parity_extension.py"),
            *base_arguments,
        ]
        parity.main()
        stats_after = _gc_stats()
    finally:
        if flags["disable_python_gc"]:
            gc.enable()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            bf16_reduction_before
        )
    rank = int(os.environ.get("RANK", "-1"))
    if rank == 0:
        _publish(
            output,
            protocol_sha256=protocol_sha256,
            mode=parsed.compile_numerics_mode,
            flags=flags,
            original_arguments=original_arguments,
            precollection_reclaimed=precollection_reclaimed,
            stats_before=stats_before,
            stats_after=stats_after,
            gc_enabled_before=gc_enabled_before,
            gc_enabled_after_restore=gc.isenabled(),
            bf16_reduction_before=bf16_reduction_before,
            bf16_reduction_after_restore=bool(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        )


if __name__ == "__main__":
    main()
