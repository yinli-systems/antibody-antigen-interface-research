#!/usr/bin/env python3
"""Run one source-bound Inductor component confirmation arm."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import run_antibody_inductor_component_localization as localization


FORMAT_VERSION = "antibody_inductor_component_confirmation_receipt_v1"
PROTOCOL_FORMAT = "antibody_production_batch_efficiency_protocol_v3"
PROTOCOL_VARIANT = "antibody_inductor_component_confirmation_protocol_v1"
SIDECAR_NAME = "compile_candidate_confirmation_receipt_v2.json"
AUTHORIZATION = localization.AUTHORIZATION
TEST_ACCESS = localization.TEST_ACCESS
CANDIDATES = {
    "inductor_attention_only": "inductor_attention_only_gc_off",
    "inductor_ffn_only": "inductor_ffn_only_gc_off",
}


class ComponentConfirmationError(RuntimeError):
    """Report invalid evidence, launch controls, or runtime state."""


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


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComponentConfirmationError(
            "Confirmation protocol must be strict ASCII JSON"
        ) from error
    if (
        not isinstance(value, dict)
        or value.get("format_version") != PROTOCOL_FORMAT
        or value.get("protocol_variant") != PROTOCOL_VARIANT
    ):
        raise ComponentConfirmationError("Confirmation protocol format differs")
    return value, hashlib.sha256(raw).hexdigest()


def _one_argument(arguments: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ComponentConfirmationError(f"Launch requires exactly one {option}")
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
    before: list[dict[str, int]],
    after: list[dict[str, int]],
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
) -> tuple[Path, dict[str, Any]]:
    selection = protocol.get("component_confirmation_selection", {})
    candidate = selection.get("selected_candidate")
    selected_mode = selection.get("selected_mode")
    if candidate not in CANDIDATES or selected_mode != CANDIDATES[candidate]:
        raise ComponentConfirmationError("Selected component candidate differs")
    if mode == "safe_eager":
        flags = localization.MODES["safe_eager"]
    elif mode == candidate:
        flags = selection.get("selected_mode_flags")
        if flags != localization.MODES[selected_mode]:
            raise ComponentConfirmationError("Selected component controls differ")
    else:
        raise ComponentConfirmationError("Confirmation mode is not frozen")
    compiled = bool(flags["compile_blocks"])
    if (
        ("--compile-blocks" in arguments) is not compiled
        or ("--compile-prewarm-buckets" in arguments) is not compiled
        or ("--disable-compiled-cudagraphs" in arguments)
        is not bool(flags["disable_compiled_cudagraphs"])
    ):
        raise ComponentConfirmationError("Compile launch controls differ")
    forbidden = (
        "--fused-adamw",
        "--pinned-staging",
        "--bucketed-padding",
        "--length-bucketing",
        "--ddp-batched-grad-copy",
        "--disable-ddp-gradient-as-bucket-view",
    )
    if any(option in arguments for option in forbidden):
        raise ComponentConfirmationError(
            "Confirmation contains an unregistered optimization"
        )
    output = Path(_one_argument(arguments, "--output"))
    if not output.is_absolute():
        raise ComponentConfirmationError("Output must be absolute")
    scripts_root = Path(__file__).resolve().parent
    bindings = {
        "confirmation_wrapper_sha256": Path(__file__),
        "runner_sha256": scripts_root / "run_antibody_pretraining_optimized.py",
        "runner_wrapper_sha256": (
            scripts_root / "run_antibody_pretraining_parity_extension.py"
        ),
    }
    if any(
        _sha256(path) != protocol["inputs"][key]
        for key, path in bindings.items()
    ):
        raise ComponentConfirmationError("Protocol-bound source differs")
    return output, dict(flags)


def _publish(
    output: Path,
    *,
    protocol_sha256: str,
    mode: str,
    flags: dict[str, Any],
    original_arguments: list[str],
    stats_before: list[dict[str, int]],
    stats_after: list[dict[str, int]],
) -> None:
    payload = {
        "authorization": AUTHORIZATION,
        "base_receipt_file_sha256": _sha256(output / "receipt.json"),
        "base_trainer_state_file_sha256": _sha256(
            output / "trainer_state.json"
        ),
        "bf16_reduced_precision_reduction": {
            "after_restore": True,
            "before": True,
            "during_run": True,
        },
        "claim_boundary": (
            "Training-excluded eight-GPU confirmation of one independently "
            "selected Inductor component scope; no production, scale, "
            "biological, downstream, or state-of-the-art claim"
        ),
        "format_version": FORMAT_VERSION,
        "launch_argv": original_arguments,
        "mode": mode,
        "mode_flags": flags,
        "parity_extension_receipt_file_sha256": _sha256(
            output / "parity_extension_receipt.json"
        ),
        "protocol_sha256": protocol_sha256,
        "python_gc": {
            "enabled_after_restore": gc.isenabled(),
            "enabled_during_run": not flags["disable_python_gc"],
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
    destination = output / SIDECAR_NAME
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--component-confirmation-protocol", type=Path, required=True)
    parser.add_argument("--component-confirmation-mode", required=True)
    parsed, base_arguments = parser.parse_known_args()
    original_arguments = sys.argv[1:]
    protocol, protocol_sha256 = _load_protocol(
        parsed.component_confirmation_protocol
    )
    if (
        protocol.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
    ):
        raise ComponentConfirmationError("Protocol authority differs")
    output, flags = _validate_launch(
        protocol,
        mode=parsed.component_confirmation_mode,
        arguments=base_arguments,
    )

    import torch
    import run_antibody_pretraining_optimized as base
    import run_antibody_pretraining_parity_extension as parity

    if (
        not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        or not gc.isenabled()
    ):
        raise ComponentConfirmationError("Runtime defaults differ")
    original_compile = base._compile_transformer_blocks

    def compile_selected_scope(model: Any, *, disable_cudagraphs: bool) -> None:
        if disable_cudagraphs is not flags["disable_compiled_cudagraphs"]:
            raise ComponentConfirmationError("Runtime CUDAGraph control differs")
        if flags["compile_backend"] != "inductor" or not disable_cudagraphs:
            raise ComponentConfirmationError("Runtime compiler backend differs")
        localization._compile_scoped(
            model,
            scope=str(flags["compile_scope"]),
        )

    if flags["compile_blocks"]:
        base._compile_transformer_blocks = compile_selected_scope
    gc.collect()
    stats_before = _gc_stats()
    if flags["disable_python_gc"]:
        gc.disable()
    try:
        scripts_root = Path(__file__).resolve().parent
        sys.argv = [
            str(scripts_root / "run_antibody_pretraining_parity_extension.py"),
            *base_arguments,
        ]
        parity.main()
        stats_after = _gc_stats()
    finally:
        base._compile_transformer_blocks = original_compile
        if flags["disable_python_gc"]:
            gc.enable()
    if not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        raise ComponentConfirmationError(
            "BF16 reduction default changed during confirmation"
        )
    if int(os.environ.get("RANK", "-1")) == 0:
        _publish(
            output,
            protocol_sha256=protocol_sha256,
            mode=parsed.component_confirmation_mode,
            flags=flags,
            original_arguments=original_arguments,
            stats_before=stats_before,
            stats_after=stats_after,
        )


if __name__ == "__main__":
    main()
