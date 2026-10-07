"""Frozen per-residue representations from external pretrained encoders.

The methods that currently work on this task all import structural or
immunological knowledge the 1,607 retained training pairs cannot supply.
DiscoTope-3.0's advantage comes from ESM-IF1 inverse-folding latents;
DeepAAAssembly's feature-relevance analysis names AntiBERTy embeddings as one of
its two critical inputs. Neither is expensive, and neither touches inter-chain
geometry, so neither disturbs the partner-pose contract.

``interaction_first.runtime`` already defines a frozen-embedding artifact, but it
pins ``EMBEDDING_DIM = 1280`` and is bound by three protocols. This module is a
parallel format that records its own width instead of assuming one, so several
encoders of different dimensions can coexist. The storage layout, the sha256
sequence key, and the chunked-interval representation are deliberately identical
to the existing one, so an artifact is familiar to anyone who has read that code.

Backends are imported lazily and never at module import time. A machine without
``antiberty`` installed can still read, validate, and reason about artifacts that
some other machine produced -- which is what makes the format reviewable in a
test suite that has no GPU and no model weights.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

REPRESENTATION_FORMAT = "pretrained_representation_v1"
MAX_CHUNK_RESIDUES = 1024


class RepresentationError(RuntimeError):
    """Raised when a representation cannot be produced, read, or validated."""


def sequence_key(sequence: str) -> str:
    """Return the sha256 key an artifact is stored under."""
    if not sequence or not sequence.isascii():
        raise RepresentationError("sequence must be non-empty ASCII")
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def representation_path(root: Path, key: str) -> Path:
    """Map a sequence digest to its artifact path, sharded by the first byte."""
    if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
        raise RepresentationError("sequence key must be a lowercase sha256 digest")
    return root / key[:2] / f"{key}.pt"


@dataclasses.dataclass(frozen=True)
class RepresentationUnit:
    """One sequence, optionally with the backbone its encoder needs."""

    sequence: str
    backbone: Any = None

    @property
    def key(self) -> str:
        return sequence_key(self.sequence)


class RepresentationBackend(Protocol):
    """Produces a ``[length, dim]`` float tensor for one unit."""

    name: str
    dim: int
    requires_backbone: bool

    def encode(self, unit: RepresentationUnit) -> Any: ...


def validate_representation_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_key: str | None = None,
    expected_dim: int | None = None,
) -> None:
    """Validate one artifact before it is allowed to reach a model."""
    required = {
        "format_version",
        "sequence_sha256",
        "sequence_length",
        "model_name",
        "representation_dim",
        "starts",
        "ends",
        "chunks",
    }
    missing = required - set(artifact)
    if missing:
        raise RepresentationError(f"artifact is missing keys: {sorted(missing)}")
    if artifact["format_version"] != REPRESENTATION_FORMAT:
        raise RepresentationError(f"unsupported format: {artifact['format_version']!r}")
    if expected_key is not None and artifact["sequence_sha256"] != expected_key:
        raise RepresentationError("artifact digest does not match its requested key")

    dim = int(artifact["representation_dim"])
    if dim <= 0:
        raise RepresentationError("representation_dim must be positive")
    if expected_dim is not None and dim != expected_dim:
        raise RepresentationError(
            f"artifact width {dim} does not match the expected {expected_dim}"
        )

    starts = [int(value) for value in artifact["starts"]]
    ends = [int(value) for value in artifact["ends"]]
    chunks = list(artifact["chunks"])
    if not chunks or len(starts) != len(ends) or len(starts) != len(chunks):
        raise RepresentationError("chunk metadata is empty or inconsistent")

    length = int(artifact["sequence_length"])
    if length <= 0:
        raise RepresentationError("sequence_length must be positive")

    covered = bytearray(length)
    for start, end, chunk in zip(starts, ends, chunks):
        if start < 0 or end <= start or end > length:
            raise RepresentationError(f"invalid interval [{start}, {end})")
        if tuple(chunk.shape) != (end - start, dim):
            raise RepresentationError("chunk shape does not match its interval")
        if not bool(chunk.is_floating_point()) or not bool(chunk.isfinite().all()):
            raise RepresentationError("chunk contains non-finite values")
        for position in range(start, end):
            covered[position] = 1
    if not all(covered):
        raise RepresentationError("chunks do not cover the complete sequence")


def build_artifact(
    unit: RepresentationUnit,
    representation: Any,
    *,
    model_name: str,
    model_revision: str,
    chunk_residues: int = MAX_CHUNK_RESIDUES,
) -> dict[str, Any]:
    """Split a ``[length, dim]`` tensor into the frozen chunked artifact."""
    if chunk_residues <= 0:
        raise RepresentationError("chunk_residues must be positive")
    if representation.ndim != 2:
        raise RepresentationError("representation must be two-dimensional")
    length, dim = int(representation.shape[0]), int(representation.shape[1])
    if length != len(unit.sequence):
        raise RepresentationError(
            f"representation length {length} does not match sequence {len(unit.sequence)}"
        )
    starts, ends, chunks = [], [], []
    for start in range(0, length, chunk_residues):
        end = min(start + chunk_residues, length)
        starts.append(start)
        ends.append(end)
        chunks.append(representation[start:end].contiguous().to("cpu"))
    artifact = {
        "format_version": REPRESENTATION_FORMAT,
        "sequence_sha256": unit.key,
        "sequence_length": length,
        "model_name": model_name,
        "model_revision": model_revision,
        "representation_dim": dim,
        "starts": starts,
        "ends": ends,
        "chunks": chunks,
    }
    validate_representation_artifact(artifact, expected_key=unit.key, expected_dim=dim)
    return artifact


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    """Write an artifact atomically, refusing to overwrite an existing one."""
    import torch

    target = representation_path(root, str(artifact["sequence_sha256"]))
    if target.exists():
        raise RepresentationError(f"{target} already exists; refusing to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".tmp-", suffix=".pt")
    os.close(handle)
    try:
        torch.save(dict(artifact), temporary)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def read_artifact(root: Path, key: str, *, expected_dim: int | None = None) -> dict[str, Any]:
    """Read and validate one artifact."""
    import torch

    path = representation_path(root, key)
    if not path.exists():
        raise RepresentationError(f"{path} does not exist")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    validate_representation_artifact(artifact, expected_key=key, expected_dim=expected_dim)
    return artifact


def extract(
    units: Sequence[RepresentationUnit],
    backend: RepresentationBackend,
    root: Path,
    *,
    skip_existing: bool = True,
    chunk_residues: int = MAX_CHUNK_RESIDUES,
) -> dict[str, Any]:
    """Encode every unit that is not already stored and return a receipt.

    Deduplicates by sequence digest, so a sequence shared by several complexes
    is encoded once.
    """
    seen: dict[str, RepresentationUnit] = {}
    for unit in units:
        if backend.requires_backbone and unit.backbone is None:
            raise RepresentationError(f"{backend.name} requires a backbone for every unit")
        seen.setdefault(unit.key, unit)

    written, skipped, failed = [], [], []
    for key in sorted(seen):
        if skip_existing and representation_path(root, key).exists():
            skipped.append(key)
            continue
        try:
            representation = backend.encode(seen[key])
            artifact = build_artifact(
                seen[key],
                representation,
                model_name=backend.name,
                model_revision=getattr(backend, "revision", "unpinned"),
                chunk_residues=chunk_residues,
            )
            write_artifact(root, artifact)
            written.append(key)
        except Exception as error:  # recorded, not swallowed: the receipt lists every failure
            failed.append({"sequence_sha256": key, "error": f"{type(error).__name__}: {error}"})

    return {
        "format_version": "pretrained_representation_receipt_v1",
        "backend": backend.name,
        "revision": getattr(backend, "revision", "unpinned"),
        "representation_dim": backend.dim,
        "requested_unique_sequences": len(seen),
        "written": len(written),
        "skipped_existing": len(skipped),
        "failed": failed,
        "written_keys_sha256": hashlib.sha256(
            "".join(f"{key}\n" for key in sorted(written)).encode("ascii")
        ).hexdigest(),
        "complete": not failed,
    }


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Write a receipt as canonical JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8")
