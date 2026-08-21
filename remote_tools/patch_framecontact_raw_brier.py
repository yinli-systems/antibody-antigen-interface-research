#!/usr/bin/env python3
"""Add an audit-only raw-logit contact metric to a FrameContact trainer copy."""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import tempfile
from pathlib import Path


RAW_PROBABILITY_ANCHOR = (
    "    contact_probabilities = _sigmoid_numpy(contact_logits / temperature)\n"
)
RAW_PROBABILITY_REPLACEMENT = (
    "    raw_contact_probabilities = _sigmoid_numpy(contact_logits)\n"
    "    contact_probabilities = _sigmoid_numpy(contact_logits / temperature)\n"
)
CONTACT_METRIC_ANCHOR = (
    '        "contact": {\n'
    "            **binary_classification_metrics(\n"
    "                contact_labels, contact_probabilities, threshold=contact_threshold\n"
    "            ),\n"
)
CONTACT_METRIC_REPLACEMENT = (
    '        "contact": {\n'
    '            "fixed_raw_logit_threshold_zero": {\n'
    "                **binary_classification_metrics(\n"
    "                    contact_labels, raw_contact_probabilities, threshold=0.5\n"
    "                ),\n"
    '                "calibration_source": "none",\n'
    '                "logit_threshold": 0.0,\n'
    '                "probability_temperature": 1.0,\n'
    '                "probability_threshold": 0.5,\n'
    "            },\n"
    "            **binary_classification_metrics(\n"
    "                contact_labels, contact_probabilities, threshold=contact_threshold\n"
    "            ),\n"
)
MARKER = '"fixed_raw_logit_threshold_zero"'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instrument_trainer(path: Path) -> str:
    original = path.read_text(encoding="utf-8")
    if MARKER in original:
        ast.parse(original, filename=str(path))
        return sha256_file(path)
    for label, anchor in (
        ("raw probability", RAW_PROBABILITY_ANCHOR),
        ("contact metric", CONTACT_METRIC_ANCHOR),
    ):
        count = original.count(anchor)
        if count != 1:
            raise RuntimeError(f"Expected one {label} anchor in {path}, found {count}")
    instrumented = original.replace(
        RAW_PROBABILITY_ANCHOR,
        RAW_PROBABILITY_REPLACEMENT,
    ).replace(
        CONTACT_METRIC_ANCHOR,
        CONTACT_METRIC_REPLACEMENT,
    )
    ast.parse(instrumented, filename=str(path))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(instrumented)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, path.stat().st_mode)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trainer", type=Path, nargs="+")
    args = parser.parse_args()
    for trainer in args.trainer:
        print(f"{instrument_trainer(trainer)}  {trainer}")


if __name__ == "__main__":
    main()
