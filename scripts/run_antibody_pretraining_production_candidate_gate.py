#!/usr/bin/env python3
"""Apply a confirmed execution candidate to the real-OAS recovery gate."""

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


FORMAT_VERSION = "antibody_production_candidate_gate_receipt_v1"
GATE_FORMAT = "antibody_production_loader_recovery_gate_v1"
AUTHORIZATION = {
    "loader_recovery_gate": True,
    "production_pretraining": False,
    "scale_out": False,
    "scientific_claim": False,
}
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
BACKEND_CONTROLS = {
    "compile_backend": "inductor",
    "disable_compiled_cudagraphs": True,
    "disable_python_gc": True,
}
COMPONENT_SCOPES = {"attention", "ffn"}
COMPONENT_CANDIDATE_SCOPES = {
    "inductor_attention_only": "attention",
    "inductor_ffn_only": "ffn",
}


class ProductionCandidateGateError(RuntimeError):
    """Report an invalid promoted-candidate recovery launch."""


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


def _one_argument(arguments: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ProductionCandidateGateError(f"Launch requires exactly one {option}")
    return arguments[positions[0] + 1]


def _load_gate(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionCandidateGateError(
            "Candidate gate must be strict ASCII JSON"
        ) from error
    if (
        not isinstance(value, dict)
        or value.get("format_version") != GATE_FORMAT
        or value.get("authorization") != AUTHORIZATION
        or value.get("test_access") != TEST_ACCESS
    ):
        raise ProductionCandidateGateError("Candidate gate identity differs")
    return value, hashlib.sha256(raw).hexdigest()


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


def _component_backend_controls(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "compile_backend",
            "compile_scope",
            "disable_compiled_cudagraphs",
            "disable_python_gc",
        }
        and value["compile_backend"] == "inductor"
        and value["compile_scope"] in COMPONENT_SCOPES
        and value["disable_compiled_cudagraphs"] is True
        and value["disable_python_gc"] is True
    )


def _validate_launch(
    protocol: dict[str, Any], arguments: list[str]
) -> tuple[Path, dict[str, Any]]:
    selection = protocol.get("candidate_selection", {})
    controls = selection.get("execution_controls")
    selected_candidate = selection.get("selected_candidate")
    component_controls_valid = (
        _component_backend_controls(controls)
        and COMPONENT_CANDIDATE_SCOPES.get(selected_candidate)
        == controls["compile_scope"]
    )
    if (
        selection.get("configuration_promotion_verified") is not True
        or not isinstance(selected_candidate, str)
        or (
            controls
            not in (
                {
                    "accuracy_first_bf16_reduction": False,
                    "disable_python_gc": False,
                },
                {
                    "accuracy_first_bf16_reduction": False,
                    "disable_python_gc": True,
                },
                {
                    "accuracy_first_bf16_reduction": True,
                    "disable_python_gc": False,
                },
                {
                    "accuracy_first_bf16_reduction": True,
                    "disable_python_gc": True,
                },
            )
            and controls != BACKEND_CONTROLS
            and not component_controls_valid
        )
        or selection.get("wrapper_sha256") != _sha256(Path(__file__))
    ):
        raise ProductionCandidateGateError("Candidate selection differs")
    gate_argument = Path(_one_argument(arguments, "--gate-protocol"))
    if _sha256(gate_argument) != hashlib.sha256(
        _canonical_json_bytes(protocol)
    ).hexdigest():
        raise ProductionCandidateGateError("Base gate argument differs")
    output = Path(_one_argument(arguments, "--output"))
    if not output.is_absolute():
        raise ProductionCandidateGateError("Output must be absolute")
    if (
        "--compile-blocks" not in arguments
        or "--compile-prewarm-buckets" not in arguments
    ):
        raise ProductionCandidateGateError("Confirmed regional compile is absent")
    expected_cudagraph_control = (
        controls == BACKEND_CONTROLS
        or _component_backend_controls(controls)
    )
    if (
        "--disable-compiled-cudagraphs" in arguments
    ) is not expected_cudagraph_control:
        raise ProductionCandidateGateError("Confirmed CUDAGraph control differs")
    forbidden = (
        "--fused-adamw",
        "--pinned-staging",
        "--bucketed-padding",
        "--length-bucketing",
    )
    if any(option in arguments for option in forbidden):
        raise ProductionCandidateGateError(
            "Recovery launch contains an unconfirmed optimization"
        )
    base = (
        Path(__file__).resolve().parent
        / "run_antibody_pretraining_production_gate.py"
    )
    if _sha256(base) != selection.get("base_runner_sha256"):
        raise ProductionCandidateGateError("Production runner binding differs")
    return output, controls


def _publish(
    output: Path,
    *,
    protocol_sha256: str,
    selected_candidate: str,
    controls: dict[str, Any],
    original_arguments: list[str],
    stats_before: list[dict[str, int]],
    stats_after: list[dict[str, int]],
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
            "during_run": not controls.get(
                "accuracy_first_bf16_reduction",
                False,
            ),
        },
        "claim_boundary": (
            "Confirmed execution controls applied to a real-OAS exact-recovery "
            "gate only; no production-pretraining, scale, biological, or SOTA claim"
        ),
        "execution_controls": controls,
        "format_version": FORMAT_VERSION,
        "gate_protocol_sha256": protocol_sha256,
        "launch_argv": original_arguments,
        "python_gc": {
            "enabled_after_restore": gc.isenabled(),
            "enabled_during_run": not controls["disable_python_gc"],
            "stats_after": stats_after,
            "stats_before": stats_before,
            "stats_delta": _stats_delta(stats_before, stats_after),
        },
        "selected_candidate": selected_candidate,
        "test_access": TEST_ACCESS,
        "wrapper_sha256": _sha256(Path(__file__)),
    }
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    destination = output / "production_candidate_gate_receipt.json"
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate-gate-protocol", type=Path, required=True)
    parsed, base_arguments = parser.parse_known_args()
    original_arguments = sys.argv[1:]
    protocol, protocol_sha256 = _load_gate(parsed.candidate_gate_protocol)
    output, controls = _validate_launch(protocol, base_arguments)

    import torch
    import run_antibody_pretraining_production_gate as base

    reduction_before = bool(
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    if not reduction_before or not gc.isenabled():
        raise ProductionCandidateGateError("Runtime defaults differ")
    gc.collect()
    stats_before = _gc_stats()
    accuracy_first = bool(controls.get("accuracy_first_bf16_reduction", False))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        not accuracy_first
    )
    original_compile = base.engine._compile_transformer_blocks

    def compile_with_frozen_backend(
        model: Any,
        *,
        disable_cudagraphs: bool,
    ) -> None:
        if (
            controls != BACKEND_CONTROLS
            and not _component_backend_controls(controls)
        ) or not disable_cudagraphs:
            raise ProductionCandidateGateError(
                "Runtime compiler backend controls differ"
            )
        if _component_backend_controls(controls):
            localization._compile_scoped(
                model,
                scope=controls["compile_scope"],
            )
        else:
            for block in model.blocks:
                block.compile(
                    backend="inductor",
                    fullgraph=True,
                    options={"triton.cudagraphs": False},
                )

    if controls == BACKEND_CONTROLS or _component_backend_controls(controls):
        base.engine._compile_transformer_blocks = compile_with_frozen_backend
    if controls["disable_python_gc"]:
        gc.disable()
    try:
        sys.argv = [str(Path(base.__file__).resolve()), *base_arguments]
        base.main()
        stats_after = _gc_stats()
    finally:
        base.engine._compile_transformer_blocks = original_compile
        if controls["disable_python_gc"]:
            gc.enable()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            reduction_before
        )
    if int(os.environ.get("RANK", "-1")) == 0:
        _publish(
            output,
            protocol_sha256=protocol_sha256,
            selected_candidate=protocol["candidate_selection"][
                "selected_candidate"
            ],
            controls=controls,
            original_arguments=original_arguments,
            stats_before=stats_before,
            stats_after=stats_after,
            reduction_before=reduction_before,
            reduction_after=bool(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        )


if __name__ == "__main__":
    main()
