from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "freeze_abpairfm_development_sequences",
        ROOT / "scripts/freeze_abpairfm_development_sequences.py",
    )
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_sequence_validation_and_deterministic_dedup() -> None:
    value = module()
    sequences = {}
    value.add_sequence(sequences, "z", "ACDE")
    value.add_sequence(sequences, "a", "ACDE")
    digest = value.sequence_sha256("ACDE")
    assert sequences[digest] == ("a", "ACDE")
    try:
        value.checked_sequence("ACDX", "bad")
    except ValueError as error:
        assert "noncanonical" in str(error)
    else:
        raise AssertionError("noncanonical development sequence was accepted")
    assert value.optional_sequence("", "missing light") == ""
