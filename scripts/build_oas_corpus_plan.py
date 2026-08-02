#!/usr/bin/env python3
"""Build one immutable, loss-independent OAS corpus plan."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import antibody_pretraining.corpus_plan as corpus_plan_module
from antibody_pretraining.corpus_plan import (
    build_corpus_plan,
    file_sha256,
    load_plan_spec,
)
from antibody_pretraining.oas_authority import canonical_json_bytes
from antibody_pretraining.oas_download import load_authority_manifest


def _write_create_only(path: Path, payload: bytes) -> None:
    """Create a read-only artifact and reject any existing output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def _parser() -> argparse.ArgumentParser:
    """Build the strict command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--manifest-payload-sha256", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--planner-module-sha256", required=True)
    parser.add_argument("--planner-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    """Load anchored inputs and emit one create-only canonical plan."""
    args = _parser().parse_args()
    observed_source_sha256 = file_sha256(Path(__file__))
    if observed_source_sha256 != args.planner_source_sha256:
        raise ValueError("Planner source digest differs")
    observed_module_sha256 = file_sha256(Path(corpus_plan_module.__file__))
    if observed_module_sha256 != args.planner_module_sha256:
        raise ValueError("Planner module digest differs")
    spec, spec_file_sha256 = load_plan_spec(args.spec)
    if spec["authority"]["manifest_file_sha256"] != args.manifest_file_sha256:
        raise ValueError("Spec and command disagree on manifest file digest")
    if spec["authority"]["manifest_payload_sha256"] != args.manifest_payload_sha256:
        raise ValueError("Spec and command disagree on manifest payload digest")
    manifest = load_authority_manifest(
        args.manifest,
        expected_file_sha256=args.manifest_file_sha256,
        expected_payload_sha256=args.manifest_payload_sha256,
    )
    plan = build_corpus_plan(
        manifest,
        spec,
        spec_file_sha256=spec_file_sha256,
        planner_module_sha256=observed_module_sha256,
        planner_source_sha256=observed_source_sha256,
    )
    payload = canonical_json_bytes(plan)
    _write_create_only(args.output, payload)
    print(hashlib.sha256(payload).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
