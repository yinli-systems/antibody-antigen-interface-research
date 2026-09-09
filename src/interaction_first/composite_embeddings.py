"""Concatenate frozen representations from several encoders into one input.

The model is data-limited, not capacity-limited: 5.5M parameters against 3,229
training instances, and both arms of every seed peak at epoch 1. The lever for a
data-limited model is better frozen features, not more layers. AntiBERTy carries
antibody-specific statistics that 3,229 pairs cannot supply, and ESM-IF1 carries
structure the sequence encoder never sees.

This module joins them to the existing ESM-2 store at the seam the training
script already uses -- ``load(sequence_key, device) -> (chunks, starts, length)``
-- so nothing downstream of that call changes, and ``runtime.EMBEDDING_DIM``
stays pinned at 1280 for the store it describes.

Three facts about this data shape the design.

Keys do not line up by themselves. The model's antibody key is
``sha256(vh + vl)``, but AntiBERTy encodes heavy and light chains separately,
because that is how it was trained -- feeding it a concatenated VH+VL would
invent a junction that does not exist. A source therefore maps one requested key
to an ordered tuple of artifact keys, concatenated along the residue axis, and
the concatenation is checked to cover the sequence exactly.

The two sides get different widths. AntiBERTy has nothing to say about an
antigen, so the antibody side gets three encoders and the antigen side two. The
model already takes ``antibody_input_dim`` and ``antigen_input_dim`` separately.

Missing features must not change the cohort. Some instances have no usable
backbone and so no ESM-IF1 artifact. Dropping them would make this ablation
incomparable to the ten-seed baseline measured on the full cohort, so a missing
artifact is zero-filled and each source carries one extra presence column, which
is 1 where the source spoke and 0 where it did not. A model that is handed zeros
cannot otherwise tell "absent" from "genuinely zero".
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from interaction_first.pretrained_representations import (
    RepresentationError,
    read_artifact,
    representation_path,
)

# One column per source, appended after that source's features.
PRESENCE_COLUMNS = 1


@dataclasses.dataclass(frozen=True)
class RepresentationSource:
    """One encoder's contribution to the concatenated input.

    ``parts_by_key`` maps a key the model asks for to the artifact keys whose
    representations are concatenated, in residue order, to cover it. A key
    absent from the mapping is not covered by this source at all and contributes
    no columns; a key present whose artifacts are missing from disk contributes
    zeros and a presence flag of 0. The distinction matters: the first is a
    property of the encoder, the second is a gap in the data.
    """

    name: str
    root: Path
    dim: int
    parts_by_key: Mapping[str, tuple[str, ...]]

    def covers(self, key: str) -> bool:
        return key in self.parts_by_key

    @property
    def width(self) -> int:
        return self.dim + PRESENCE_COLUMNS


def _concatenate_parts(
    source: RepresentationSource, key: str, length: int, device: Any
) -> Any:
    """Return ``(length, dim)`` for one source, or None when it has no artifact."""
    import torch

    from interaction_first.model import merge_chunk_embeddings

    parts = source.parts_by_key[key]
    if not parts:
        raise RepresentationError(f"{source.name} maps {key[:12]} to no artifact keys")
    if any(not representation_path(source.root, part).exists() for part in parts):
        return None

    pieces = []
    for part in parts:
        artifact = read_artifact(source.root, part, expected_dim=source.dim)
        chunks = [chunk.to(device=device, dtype=torch.float32) for chunk in artifact["chunks"]]
        starts = [int(value) for value in artifact["starts"]]
        pieces.append(
            merge_chunk_embeddings(
                chunks, starts, int(artifact["sequence_length"]), device
            )
        )
    merged = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
    if int(merged.shape[0]) != length:
        raise RepresentationError(
            f"{source.name} covers {int(merged.shape[0])} residues for {key[:12]}, "
            f"but the sequence has {length}"
        )
    return merged


class CompositeEmbeddingStore:
    """Present several representation stores as one wider embedding store.

    The interface is the one ``EmbeddingStore`` already offers, so the training
    and evaluation paths call it without knowing the difference: one chunk
    covering the whole sequence, the base encoder's columns first, then each
    source's columns followed by its presence column, in the order given.
    """

    def __init__(self, base: Any, sources: Sequence[RepresentationSource]) -> None:
        names = [source.name for source in sources]
        if len(set(names)) != len(names):
            raise RepresentationError(f"duplicate source names: {names}")
        self.base = base
        self.sources = tuple(sources)
        self.filled: dict[str, int] = {source.name: 0 for source in self.sources}

    def width_for(self, key: str, base_dim: int) -> int:
        """Return the concatenated width for one key."""
        return base_dim + sum(
            source.width for source in self.sources if source.covers(key)
        )

    def load(self, sequence_key: str, device: Any) -> tuple[list[Any], list[int], int]:
        import torch

        from interaction_first.model import merge_chunk_embeddings

        chunks, starts, length = self.base.load(sequence_key, device)
        merged = merge_chunk_embeddings(chunks, starts, length, device)
        columns = [merged]
        for source in self.sources:
            if not source.covers(sequence_key):
                continue
            features = _concatenate_parts(source, sequence_key, length, device)
            present = features is not None
            if not present:
                self.filled[source.name] += 1
                features = torch.zeros(
                    (length, source.dim), dtype=merged.dtype, device=device
                )
            columns.append(features.to(dtype=merged.dtype))
            columns.append(
                torch.full(
                    (length, PRESENCE_COLUMNS),
                    1.0 if present else 0.0,
                    dtype=merged.dtype,
                    device=device,
                )
            )
        return [torch.cat(columns, dim=1)], [0], length

    def describe(self) -> dict[str, Any]:
        """Report what will be concatenated, without any running counts.

        This is what a receipt written *before* the data is read may record. The
        zero-fill counters belong to ``summary`` and are zero until loads have
        happened, so reporting them here would state a fact that has not been
        measured yet.
        """
        return {
            "sources": [
                {
                    "name": source.name,
                    "root": str(source.root),
                    "representation_dim": source.dim,
                    "presence_columns": PRESENCE_COLUMNS,
                    "keys_covered": len(source.parts_by_key),
                }
                for source in self.sources
            ]
        }

    def summary(self) -> dict[str, Any]:
        """Report what was concatenated and how often a source was absent."""
        described = self.describe()
        for entry in described["sources"]:
            entry["loads_zero_filled"] = self.filled[entry["name"]]
        described["base"] = self.base.summary() if hasattr(self.base, "summary") else {}
        return described


# How each encoder was run, and therefore how its artifacts have to be joined.
# AntiBERTy took heavy and light chains separately and never saw an antigen;
# ESM-IF1 took the concatenated antibody and the antigen, keyed exactly as the
# model's own store is.
ANTIBODY_SIDE = "antibody"
ANTIGEN_SIDE = "antigen"


def _digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def manifest_chain_keys(manifest: Path) -> dict[str, dict[str, Any]]:
    """Return, per instance, the keys the model asks for and their chain parts."""
    rows: dict[str, dict[str, Any]] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for column in ("INSTANCE", "vh_seq", "vl_seq", "ag_seq"):
            if column not in (reader.fieldnames or []):
                raise RepresentationError(f"manifest has no {column!r} column")
        for row in reader:
            heavy = (row["vh_seq"] or "").strip()
            light = (row["vl_seq"] or "").strip()
            antigen = (row["ag_seq"] or "").strip()
            if not heavy or not antigen:
                continue
            rows[row["INSTANCE"]] = {
                "antibody_key": _digest(heavy + light),
                "antigen_key": _digest(antigen),
                # A quarter of this cohort is single-domain, so the light chain
                # is genuinely absent rather than missing.
                "antibody_parts": (
                    (_digest(heavy), _digest(light)) if light else (_digest(heavy),)
                ),
            }
    return rows


def build_sources(
    manifest: Path,
    antiberty_root: Path | None = None,
    antiberty_dim: int = 512,
    esm_if1_root: Path | None = None,
    esm_if1_dim: int = 512,
) -> list[RepresentationSource]:
    """Build the sources for the encoders this project has extracted.

    Coverage is decided by the manifest, not by what happens to be on disk, so
    every key of a side gets the same width whether or not its artifact exists.
    """
    chains = manifest_chain_keys(manifest)
    sources: list[RepresentationSource] = []
    if antiberty_root is not None:
        sources.append(
            RepresentationSource(
                name="antiberty",
                root=antiberty_root,
                dim=antiberty_dim,
                parts_by_key={
                    entry["antibody_key"]: entry["antibody_parts"]
                    for entry in chains.values()
                },
            )
        )
    if esm_if1_root is not None:
        parts: dict[str, tuple[str, ...]] = {}
        for entry in chains.values():
            parts[entry["antibody_key"]] = (entry["antibody_key"],)
            parts[entry["antigen_key"]] = (entry["antigen_key"],)
        sources.append(
            RepresentationSource(
                name="esm_if1", root=esm_if1_root, dim=esm_if1_dim, parts_by_key=parts
            )
        )
    return sources
