#!/usr/bin/env python3
"""Build or audit the registered MMseqs2 GPU sensitivity smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from antibody_pretraining.near_identity_controls import (
    audit_sensitivity_hits,
    build_sensitivity_controls,
)


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


def _strict_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError("Control manifest is not an object")
    return value


def _publish(path: Path, value: dict[str, object]) -> None:
    unsigned = dict(value)
    value = dict(unsigned)
    value["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite result: {path}")
        os.link(temporary, path)
        path.chmod(0o444)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-root", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--expected", type=Path, required=True)
    audit.add_argument("--hits", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        build_sensitivity_controls(arguments.output_root)
        return
    expected = _strict_json(arguments.expected)
    with arguments.hits.open("r", encoding="utf-8") as handle:
        result = audit_sensitivity_hits(handle, expected_manifest=expected)
    result["inputs"] = {
        "expected_file_sha256": hashlib.sha256(arguments.expected.read_bytes()).hexdigest(),
        "hits_file_sha256": hashlib.sha256(arguments.hits.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    _publish(arguments.output, result)
    if result["gate_passed"] is not True:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
