#!/usr/bin/env python3
"""Run one source-bound Dynamo Transformer-quarter bisection arm."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import run_antibody_dynamo_region_bisection as parent


FORMAT_VERSION = "antibody_dynamo_quarter_bisection_receipt_v1"
PROTOCOL_FORMAT = "antibody_dynamo_quarter_bisection_protocol_v1"
MODES = {
    "safe_eager": {
        "block_indices": [],
        "compile_backend": None,
        "compile_blocks": False,
        "compile_scope": "none",
        "disable_compiled_cudagraphs": False,
        "disable_python_gc": False,
    },
    "dynamo_q1_gc_off": {
        "block_indices": list(range(0, 8)),
        "compile_backend": "eager",
        "compile_blocks": True,
        "compile_scope": "blocks_00_07",
        "disable_compiled_cudagraphs": False,
        "disable_python_gc": True,
    },
    "dynamo_q2_gc_off": {
        "block_indices": list(range(8, 16)),
        "compile_backend": "eager",
        "compile_blocks": True,
        "compile_scope": "blocks_08_15",
        "disable_compiled_cudagraphs": False,
        "disable_python_gc": True,
    },
    "dynamo_q3_gc_off": {
        "block_indices": list(range(16, 24)),
        "compile_backend": "eager",
        "compile_blocks": True,
        "compile_scope": "blocks_16_23",
        "disable_compiled_cudagraphs": False,
        "disable_python_gc": True,
    },
    "dynamo_q4_gc_off": {
        "block_indices": list(range(24, 32)),
        "compile_backend": "eager",
        "compile_blocks": True,
        "compile_scope": "blocks_24_31",
        "disable_compiled_cudagraphs": False,
        "disable_python_gc": True,
    },
    "dynamo_full_control_gc_off": {
        "block_indices": list(range(32)),
        "compile_backend": "eager",
        "compile_blocks": True,
        "compile_scope": "blocks_00_31_control",
        "disable_compiled_cudagraphs": False,
        "disable_python_gc": True,
    },
}


@contextmanager
def _configured_parent() -> Iterator[None]:
    saved = {
        "FORMAT_VERSION": parent.FORMAT_VERSION,
        "MODES": parent.MODES,
        "PROTOCOL_FORMAT": parent.PROTOCOL_FORMAT,
        "__file__": parent.__file__,
    }
    parent.FORMAT_VERSION = FORMAT_VERSION
    parent.MODES = MODES
    parent.PROTOCOL_FORMAT = PROTOCOL_FORMAT
    parent.__file__ = str(Path(__file__))
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(parent, name, value)


def _validate_launch(
    protocol: dict[str, Any],
    *,
    mode: str,
    arguments: list[str],
    scripts_root: Path,
) -> tuple[Path, dict[str, Any]]:
    parent_path = scripts_root / "run_antibody_dynamo_region_bisection.py"
    if parent._sha256(parent_path) != protocol["inputs"]["parent_wrapper_sha256"]:
        raise parent.DynamoRegionBisectionError("Parent wrapper binding differs")
    with _configured_parent():
        return parent._validate_launch(
            protocol,
            mode=mode,
            arguments=arguments,
            scripts_root=scripts_root,
        )


def _compile_registered_blocks(model: Any, *, block_indices: list[int]) -> None:
    parent._compile_registered_blocks(model, block_indices=block_indices)


def main() -> None:
    with _configured_parent():
        parent.main()


if __name__ == "__main__":
    main()
