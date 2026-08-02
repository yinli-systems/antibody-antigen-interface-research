#!/usr/bin/env python3
"""Run one source-bound Inductor component-localization arm."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


FORMAT_VERSION = "antibody_inductor_component_localization_receipt_v1"
PROTOCOL_FORMAT = "antibody_inductor_component_localization_protocol_v1"
SIDECAR_NAME = "compile_backend_attribution_receipt.json"
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
MODES = {
    "safe_eager": {
        "compile_backend": None,
        "compile_blocks": False,
        "compile_scope": "none",
        "disable_compiled_cudagraphs": False,
        "disable_python_gc": False,
    },
    "inductor_attention_only_gc_off": {
        "compile_backend": "inductor",
        "compile_blocks": True,
        "compile_scope": "attention",
        "disable_compiled_cudagraphs": True,
        "disable_python_gc": True,
    },
    "inductor_ffn_only_gc_off": {
        "compile_backend": "inductor",
        "compile_blocks": True,
        "compile_scope": "ffn",
        "disable_compiled_cudagraphs": True,
        "disable_python_gc": True,
    },
    "inductor_full_block_gc_off": {
        "compile_backend": "inductor",
        "compile_blocks": True,
        "compile_scope": "full_block",
        "disable_compiled_cudagraphs": True,
        "disable_python_gc": True,
    },
}


class ComponentLocalizationError(RuntimeError):
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
        raise ComponentLocalizationError(
            "Localization protocol must be strict ASCII JSON"
        ) from error
    if not isinstance(value, dict) or value.get("format_version") != PROTOCOL_FORMAT:
        raise ComponentLocalizationError("Localization protocol format differs")
    return value, hashlib.sha256(raw).hexdigest()


def _one_argument(arguments: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ComponentLocalizationError(
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
    scripts_root: Path,
) -> tuple[Path, dict[str, Any]]:
    flags = protocol.get("modes", {}).get(mode)
    if mode not in MODES or flags != MODES[mode]:
        raise ComponentLocalizationError("Localization mode differs")
    compiled = bool(flags["compile_blocks"])
    if (
        ("--compile-blocks" in arguments) is not compiled
        or ("--compile-prewarm-buckets" in arguments) is not compiled
        or ("--disable-compiled-cudagraphs" in arguments)
        is not bool(flags["disable_compiled_cudagraphs"])
    ):
        raise ComponentLocalizationError("Compile launch controls differ")
    forbidden = (
        "--fused-adamw",
        "--pinned-staging",
        "--bucketed-padding",
        "--length-bucketing",
        "--ddp-batched-grad-copy",
        "--disable-ddp-gradient-as-bucket-view",
    )
    if any(option in arguments for option in forbidden):
        raise ComponentLocalizationError(
            "Localization contains an unregistered optimization"
        )
    output = Path(_one_argument(arguments, "--output"))
    if not output.is_absolute():
        raise ComponentLocalizationError("Output must be absolute")
    bindings = {
        "base_runner_sha256": scripts_root / "run_antibody_pretraining_optimized.py",
        "wrapper_sha256": Path(__file__),
    }
    if any(
        _sha256(path) != protocol["inputs"][key]
        for key, path in bindings.items()
    ):
        raise ComponentLocalizationError("Protocol-bound source differs")
    return output, dict(flags)


def _compile_scoped(model: Any, *, scope: str) -> None:
    options = {"triton.cudagraphs": False}
    for block in model.blocks:
        if scope == "attention":
            region = block.attention
        elif scope == "ffn":
            region = block.ffn
        elif scope == "full_block":
            region = block
        else:
            raise ComponentLocalizationError("Compiled scope differs")
        region.compile(
            backend="inductor",
            fullgraph=True,
            options=options,
        )


def _publish(
    output: Path,
    *,
    protocol_sha256: str,
    mode: str,
    flags: dict[str, Any],
    original_arguments: list[str],
    stats_before: list[dict[str, int]],
    stats_after: list[dict[str, int]],
    gc_enabled_before: bool,
    gc_enabled_after_restore: bool,
) -> None:
    payload = {
        "authorization": AUTHORIZATION,
        "base_receipt_file_sha256": _sha256(output / "receipt.json"),
        "base_trainer_state_file_sha256": _sha256(
            output / "trainer_state.json"
        ),
        "claim_boundary": (
            "Training-excluded Inductor component localization only; no "
            "configuration promotion, production training, biological metric, "
            "multi-node scale, or state-of-the-art claim"
        ),
        "format_version": FORMAT_VERSION,
        "launch_argv": original_arguments,
        "mode": mode,
        "mode_flags": flags,
        "protocol_sha256": protocol_sha256,
        "python_gc": {
            "enabled_after_restore": gc_enabled_after_restore,
            "enabled_before": gc_enabled_before,
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
    parser.add_argument("--component-localization-protocol", type=Path, required=True)
    parser.add_argument("--component-localization-mode", required=True)
    parsed, base_arguments = parser.parse_known_args()
    original_arguments = sys.argv[1:]
    protocol, protocol_sha256 = _load_protocol(
        parsed.component_localization_protocol
    )
    if (
        protocol.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
    ):
        raise ComponentLocalizationError("Protocol authority differs")
    scripts_root = Path(__file__).resolve().parent
    output, flags = _validate_launch(
        protocol,
        mode=parsed.component_localization_mode,
        arguments=base_arguments,
        scripts_root=scripts_root,
    )

    import torch
    import run_antibody_pretraining_optimized as base

    if not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        raise ComponentLocalizationError("Runtime BF16 reduction default differs")
    original_compile = base._compile_transformer_blocks

    def compile_registered_scope(model: Any, *, disable_cudagraphs: bool) -> None:
        if disable_cudagraphs is not flags["disable_compiled_cudagraphs"]:
            raise ComponentLocalizationError("Runtime CUDAGraph control differs")
        if flags["compile_backend"] != "inductor" or not disable_cudagraphs:
            raise ComponentLocalizationError("Runtime compiler backend differs")
        _compile_scoped(model, scope=str(flags["compile_scope"]))

    if flags["compile_blocks"]:
        base._compile_transformer_blocks = compile_registered_scope
    gc_enabled_before = gc.isenabled()
    if not gc_enabled_before:
        raise ComponentLocalizationError("Python cyclic GC must begin enabled")
    gc.collect()
    stats_before = _gc_stats()
    if flags["disable_python_gc"]:
        gc.disable()
    try:
        sys.argv = [
            str(scripts_root / "run_antibody_pretraining_optimized.py"),
            *base_arguments,
        ]
        base.main()
        stats_after = _gc_stats()
    finally:
        base._compile_transformer_blocks = original_compile
        if flags["disable_python_gc"]:
            gc.enable()
    if int(os.environ.get("RANK", "-1")) == 0:
        _publish(
            output,
            protocol_sha256=protocol_sha256,
            mode=parsed.component_localization_mode,
            flags=flags,
            original_arguments=original_arguments,
            stats_before=stats_before,
            stats_after=stats_after,
            gc_enabled_before=gc_enabled_before,
            gc_enabled_after_restore=gc.isenabled(),
        )


if __name__ == "__main__":
    main()
