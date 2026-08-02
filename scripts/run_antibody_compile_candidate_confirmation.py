#!/usr/bin/env python3
"""Run one source-bound arm of the compile-candidate confirmation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


FORMAT_VERSION = "antibody_compile_candidate_confirmation_receipt_v1"
PROTOCOL_FORMAT = "antibody_production_batch_efficiency_protocol_v3"
AUTHORIZATION = {
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


class CompileCandidateConfirmationError(RuntimeError):
    """Report an invalid confirmation protocol or launch."""


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
        raise CompileCandidateConfirmationError(
            "Confirmation protocol must be strict ASCII JSON"
        ) from error
    if not isinstance(value, dict) or value.get("format_version") != PROTOCOL_FORMAT:
        raise CompileCandidateConfirmationError("Confirmation protocol format differs")
    return value, hashlib.sha256(raw).hexdigest()


def _one_argument(arguments: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise CompileCandidateConfirmationError(
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
) -> tuple[Path, dict[str, bool]]:
    selected = protocol.get("confirmation_selection", {}).get(
        "selected_candidate"
    )
    if mode == "safe_eager":
        flags = {
            "accuracy_first_bf16_reduction": False,
            "compile_blocks": False,
            "disable_python_gc": False,
        }
    elif mode == selected:
        flags = protocol["confirmation_selection"]["selected_mode_flags"]
    else:
        raise CompileCandidateConfirmationError("Confirmation mode is not frozen")
    expected_controls = protocol.get("optimization", {}).get(
        "execution_controls"
    )
    if mode != "safe_eager" and expected_controls != {
        "accuracy_first_bf16_reduction": flags[
            "accuracy_first_bf16_reduction"
        ],
        "disable_python_gc": flags["disable_python_gc"],
    }:
        raise CompileCandidateConfirmationError(
            "Candidate execution controls differ from the protocol"
        )
    expected_compile = mode != "safe_eager"
    if (
        bool(flags["compile_blocks"]) is not expected_compile
        or ("--compile-blocks" in arguments) is not expected_compile
        or ("--compile-prewarm-buckets" in arguments) is not expected_compile
    ):
        raise CompileCandidateConfirmationError("Compile launch flags differ")
    forbidden = (
        "--fused-adamw",
        "--pinned-staging",
        "--bucketed-padding",
        "--length-bucketing",
        "--ddp-batched-grad-copy",
        "--disable-ddp-gradient-as-bucket-view",
    )
    if any(option in arguments for option in forbidden):
        raise CompileCandidateConfirmationError(
            "Confirmation contains an unregistered optimization"
        )
    output = Path(_one_argument(arguments, "--output"))
    if not output.is_absolute():
        raise CompileCandidateConfirmationError("Output must be absolute")
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
        raise CompileCandidateConfirmationError("Confirmation source binding differs")
    return output, flags


def _publish(
    output: Path,
    *,
    protocol_sha256: str,
    mode: str,
    flags: dict[str, bool],
    original_arguments: list[str],
    stats_before: list[dict[str, int]],
    stats_after: list[dict[str, int]],
    precollection_reclaimed: int,
    reduction_before: bool,
    reduction_after: bool,
) -> None:
    payload = {
        "authorization": AUTHORIZATION,
        "base_receipt_file_sha256": _sha256(output / "receipt.json"),
        "base_trainer_state_file_sha256": _sha256(
            output / "trainer_state.json"
        ),
        "bf16_reduced_precision_reduction": {
            "after_restore": reduction_after,
            "before": reduction_before,
            "during_run": not flags["accuracy_first_bf16_reduction"],
        },
        "claim_boundary": (
            "Training-excluded eight-GPU compile-candidate confirmation arm; "
            "no production training, scale, biological, or SOTA claim"
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
    destination = output / "compile_candidate_confirmation_receipt.json"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--confirmation-mode", required=True)
    parsed, base_arguments = parser.parse_known_args()
    original_arguments = sys.argv[1:]
    protocol, protocol_sha256 = _load_protocol(parsed.confirmation_protocol)
    if (
        protocol.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
    ):
        raise CompileCandidateConfirmationError("Confirmation authority differs")
    output, flags = _validate_launch(
        protocol,
        mode=parsed.confirmation_mode,
        arguments=base_arguments,
    )

    import torch
    import run_antibody_pretraining_parity_extension as parity

    reduction_before = bool(
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    if not reduction_before or not gc.isenabled():
        raise CompileCandidateConfirmationError("Runtime defaults differ")
    precollection_reclaimed = int(gc.collect())
    stats_before = _gc_stats()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not flags[
        "accuracy_first_bf16_reduction"
    ]
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
        if flags["disable_python_gc"]:
            gc.enable()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            reduction_before
        )
    if int(os.environ.get("RANK", "-1")) == 0:
        _publish(
            output,
            protocol_sha256=protocol_sha256,
            mode=parsed.confirmation_mode,
            flags=flags,
            original_arguments=original_arguments,
            stats_before=stats_before,
            stats_after=stats_after,
            precollection_reclaimed=precollection_reclaimed,
            reduction_before=reduction_before,
            reduction_after=bool(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        )


if __name__ == "__main__":
    main()
