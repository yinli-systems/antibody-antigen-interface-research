"""Tests for the frozen pretrained-representation format.

These run with no model weights and no GPU: a stub backend stands in for the
real encoders, so the artifact contract, the deduplication, and the failure
accounting are all testable on any machine.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from interaction_first.pretrained_representations import (
    REPRESENTATION_FORMAT,
    RepresentationError,
    RepresentationUnit,
    build_artifact,
    extract,
    read_artifact,
    representation_path,
    sequence_key,
    validate_representation_artifact,
    write_artifact,
    write_receipt,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)


class StubBackend:
    """Deterministic stand-in that needs no weights."""

    name = "stub"
    dim = 4
    revision = "stub-v1"
    requires_backbone = False

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    def encode(self, unit: RepresentationUnit):
        import torch

        self.calls.append(unit.key)
        if unit.sequence in self.fail_on:
            raise ValueError("stub refused this sequence")
        return torch.arange(len(unit.sequence) * self.dim, dtype=torch.float32).reshape(
            len(unit.sequence), self.dim
        )


class BackboneBackend(StubBackend):
    requires_backbone = True


def _unit(sequence: str = "ACDEFGHIK") -> RepresentationUnit:
    return RepresentationUnit(sequence=sequence)


# --- keys and paths ------------------------------------------------------


def test_sequence_key_is_the_sha256_of_the_sequence() -> None:
    import hashlib

    assert sequence_key("ACDE") == hashlib.sha256(b"ACDE").hexdigest()


def test_sequence_key_rejects_empty_and_non_ascii() -> None:
    with pytest.raises(RepresentationError):
        sequence_key("")
    with pytest.raises(RepresentationError):
        sequence_key("ACDEß")


def test_path_is_sharded_by_the_first_byte(tmp_path: Path) -> None:
    key = sequence_key("ACDE")

    assert representation_path(tmp_path, key) == tmp_path / key[:2] / f"{key}.pt"


def test_path_rejects_a_non_digest_key(tmp_path: Path) -> None:
    with pytest.raises(RepresentationError):
        representation_path(tmp_path, "not-a-digest")


# --- artifact contract ---------------------------------------------------


def test_artifact_round_trips_through_disk(tmp_path: Path) -> None:
    unit = _unit()
    backend = StubBackend()
    artifact = build_artifact(
        unit, backend.encode(unit), model_name=backend.name, model_revision=backend.revision
    )

    write_artifact(tmp_path, artifact)
    restored = read_artifact(tmp_path, unit.key, expected_dim=backend.dim)

    assert restored["format_version"] == REPRESENTATION_FORMAT
    assert restored["sequence_length"] == len(unit.sequence)
    assert restored["representation_dim"] == backend.dim
    assert restored["model_name"] == "stub"


def test_chunking_covers_the_sequence_exactly(tmp_path: Path) -> None:
    unit = _unit("A" * 2500)
    backend = StubBackend()

    artifact = build_artifact(
        unit,
        backend.encode(unit),
        model_name=backend.name,
        model_revision=backend.revision,
        chunk_residues=1000,
    )

    assert artifact["starts"] == [0, 1000, 2000]
    assert artifact["ends"] == [1000, 2000, 2500]
    assert sum(int(chunk.shape[0]) for chunk in artifact["chunks"]) == 2500


def test_build_rejects_a_length_mismatch() -> None:
    import torch

    with pytest.raises(RepresentationError, match="does not match sequence"):
        build_artifact(
            _unit("ACDE"),
            torch.zeros(3, 4),
            model_name="stub",
            model_revision="stub-v1",
        )


def test_validation_rejects_a_width_mismatch(tmp_path: Path) -> None:
    unit = _unit()
    backend = StubBackend()
    artifact = build_artifact(
        unit, backend.encode(unit), model_name=backend.name, model_revision=backend.revision
    )

    with pytest.raises(RepresentationError, match="does not match the expected"):
        validate_representation_artifact(artifact, expected_dim=99)


def test_validation_rejects_a_wrong_key(tmp_path: Path) -> None:
    unit = _unit()
    backend = StubBackend()
    artifact = build_artifact(
        unit, backend.encode(unit), model_name=backend.name, model_revision=backend.revision
    )

    with pytest.raises(RepresentationError, match="does not match its requested key"):
        validate_representation_artifact(artifact, expected_key="0" * 64)


def test_validation_rejects_non_finite_values() -> None:
    import torch

    unit = _unit("ACDE")
    representation = torch.zeros(4, 4)
    representation[2, 1] = float("nan")

    with pytest.raises(RepresentationError, match="non-finite"):
        build_artifact(unit, representation, model_name="stub", model_revision="stub-v1")


def test_validation_rejects_incomplete_coverage() -> None:
    import torch

    artifact = {
        "format_version": REPRESENTATION_FORMAT,
        "sequence_sha256": sequence_key("ACDE"),
        "sequence_length": 4,
        "model_name": "stub",
        "representation_dim": 2,
        "starts": [0],
        "ends": [2],
        "chunks": [torch.zeros(2, 2)],
    }

    with pytest.raises(RepresentationError, match="do not cover"):
        validate_representation_artifact(artifact)


def test_validation_reports_missing_keys() -> None:
    with pytest.raises(RepresentationError, match="missing keys"):
        validate_representation_artifact({"format_version": REPRESENTATION_FORMAT})


def test_write_refuses_to_overwrite(tmp_path: Path) -> None:
    unit = _unit()
    backend = StubBackend()
    artifact = build_artifact(
        unit, backend.encode(unit), model_name=backend.name, model_revision=backend.revision
    )
    write_artifact(tmp_path, artifact)

    with pytest.raises(RepresentationError, match="refusing to overwrite"):
        write_artifact(tmp_path, artifact)


# --- extraction ----------------------------------------------------------


def test_extract_deduplicates_by_sequence(tmp_path: Path) -> None:
    backend = StubBackend()
    units = [_unit("ACDE"), _unit("ACDE"), _unit("KLMN")]

    receipt = extract(units, backend, tmp_path)

    assert receipt["requested_unique_sequences"] == 2
    assert receipt["written"] == 2
    assert len(backend.calls) == 2
    assert receipt["complete"] is True


def test_extract_skips_what_already_exists(tmp_path: Path) -> None:
    backend = StubBackend()
    extract([_unit("ACDE")], backend, tmp_path)

    second = extract([_unit("ACDE"), _unit("KLMN")], StubBackend(), tmp_path)

    assert second["skipped_existing"] == 1
    assert second["written"] == 1


def test_extract_records_failures_without_aborting(tmp_path: Path) -> None:
    backend = StubBackend(fail_on={"BAD"})

    receipt = extract([_unit("ACDE"), _unit("BAD"), _unit("KLMN")], backend, tmp_path)

    assert receipt["written"] == 2
    assert len(receipt["failed"]) == 1
    assert receipt["failed"][0]["sequence_sha256"] == sequence_key("BAD")
    assert "ValueError" in receipt["failed"][0]["error"]
    assert receipt["complete"] is False


def test_extract_requires_a_backbone_when_the_backend_does(tmp_path: Path) -> None:
    with pytest.raises(RepresentationError, match="requires a backbone"):
        extract([_unit("ACDE")], BackboneBackend(), tmp_path)


def test_written_key_digest_is_order_independent(tmp_path: Path, tmp_path_factory) -> None:
    other = tmp_path_factory.mktemp("other")
    forward = extract([_unit("ACDE"), _unit("KLMN")], StubBackend(), tmp_path)
    reverse = extract([_unit("KLMN"), _unit("ACDE")], StubBackend(), other)

    assert forward["written_keys_sha256"] == reverse["written_keys_sha256"]


def test_receipt_is_canonical_json(tmp_path: Path) -> None:
    receipt = extract([_unit("ACDE")], StubBackend(), tmp_path)
    target = tmp_path / "receipt.json"

    write_receipt(target, receipt)

    assert json.loads(target.read_text())["backend"] == "stub"
    assert target.read_text().endswith("\n")


def test_backend_registry_rejects_an_unknown_name() -> None:
    from interaction_first.representation_backends import BACKENDS, build_backend

    assert set(BACKENDS) == {"antiberty", "esm2", "esm_if1", "proteinmpnn"}
    with pytest.raises(RepresentationError, match="unknown backend"):
        build_backend("nope")


def test_declared_backend_widths_are_stable() -> None:
    """The width is recorded in every artifact, so a silent change is a defect."""
    from interaction_first.representation_backends import BACKENDS

    assert BACKENDS["antiberty"].dim == 512
    assert BACKENDS["esm2"].dim == 1280
    assert BACKENDS["esm_if1"].dim == 512
    assert BACKENDS["esm_if1"].requires_backbone is True
    assert BACKENDS["antiberty"].requires_backbone is False
