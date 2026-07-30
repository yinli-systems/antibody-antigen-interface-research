#!/usr/bin/env python3
"""Run a source-bound systems-only extension of the frozen parity runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


FORMAT_VERSION = "antibody_pretraining_parity_extension_receipt_v1"
EXTENSION_FORMAT = "antibody_ddp_scientific_parity_extension_v1"
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
AUTHORIZATION = {
    "model_selection": False,
    "production_pretraining": False,
    "scientific_claim": False,
}


class ParityExtensionError(RuntimeError):
    """Report an invalid extension, launch, source, or sidecar publication."""


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


def _argument_value(arguments: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(arguments) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ParityExtensionError(f"Launch requires exactly one {option}")
    return arguments[positions[0] + 1]


def _strict_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ParityExtensionError("Extension protocol must be ASCII JSON") from error
    if not isinstance(value, dict):
        raise ParityExtensionError("Extension protocol must contain one object")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_extension(
    extension_path: Path,
    base_arguments: list[str],
) -> tuple[dict[str, Any], str, Path]:
    extension, extension_sha256 = _strict_protocol(extension_path)
    parity_protocol = Path(_argument_value(base_arguments, "--parity-protocol"))
    split_root = Path(_argument_value(base_arguments, "--parity-split-root"))
    output_root = Path(_argument_value(base_arguments, "--output"))
    steps = int(_argument_value(base_arguments, "--steps"))
    run_seed = int(_argument_value(base_arguments, "--run-seed"))
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if "--skip-checkpoint" not in base_arguments or "--resume-root" in base_arguments:
        raise ParityExtensionError("Extension requires no checkpoint and no resume")
    if (
        extension.get("format_version") != EXTENSION_FORMAT
        or extension.get("authorization") != AUTHORIZATION
        or extension.get("test_access") != TEST_ACCESS
        or extension.get("parent_parity_protocol_sha256") != _sha256(parity_protocol)
        or extension.get("execution")
        != {
            "gpu_count": world_size,
            "parent_updates_per_run": 500,
            "run_seed": run_seed,
            "updates_per_run": steps,
        }
        or extension.get("inputs")
        != {
            "evaluation_indices_sha256": _sha256(
                split_root / "evaluation_indices.npy"
            ),
            "parity_split_receipt_file_sha256": _sha256(
                split_root / "receipt.json"
            ),
            "training_indices_sha256": _sha256(
                split_root / "training_indices.npy"
            ),
        }
    ):
        raise ParityExtensionError("Extension protocol or launch bindings differ")
    return extension, extension_sha256, output_root


def _publish_sidecar(
    output_root: Path,
    *,
    extension: dict[str, Any],
    extension_sha256: str,
    original_arguments: list[str],
    base_runner: Path,
) -> None:
    receipt_path = output_root / "receipt.json"
    state_path = output_root / "trainer_state.json"
    payload = {
        "authorization": AUTHORIZATION,
        "base_receipt_file_sha256": _sha256(receipt_path),
        "base_runner_sha256": _sha256(base_runner),
        "base_trainer_state_file_sha256": _sha256(state_path),
        "claim_boundary": extension["claim_boundary"],
        "extension_protocol_sha256": extension_sha256,
        "format_version": FORMAT_VERSION,
        "launch_argv": original_arguments,
        "test_access": TEST_ACCESS,
        "wrapper_sha256": _sha256(Path(__file__)),
    }
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    output = output_root / "parity_extension_receipt.json"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--parity-extension-protocol", type=Path, required=True)
    parsed, base_arguments = parser.parse_known_args()
    original_arguments = sys.argv[1:]
    extension, extension_sha256, output_root = _validate_extension(
        parsed.parity_extension_protocol,
        base_arguments,
    )
    scripts_root = Path(__file__).resolve().parent
    base_runner = scripts_root / "run_antibody_pretraining_optimized.py"
    sys.path.insert(0, str(scripts_root))
    import run_antibody_pretraining_optimized as base

    original_loader = base._load_parity_execution

    def extended_loader(*args: Any, **kwargs: Any):
        if kwargs.get("steps") != extension["execution"]["updates_per_run"]:
            raise ParityExtensionError("Runner step count differs from the extension")
        parent_kwargs = dict(kwargs)
        parent_kwargs["steps"] = extension["execution"]["parent_updates_per_run"]
        return original_loader(*args, **parent_kwargs)

    base._load_parity_execution = extended_loader
    sys.argv = [str(base_runner), *base_arguments]
    base.main()
    if int(os.environ.get("RANK", "-1")) == 0:
        _publish_sidecar(
            output_root,
            extension=extension,
            extension_sha256=extension_sha256,
            original_arguments=original_arguments,
            base_runner=base_runner,
        )


if __name__ == "__main__":
    main()
