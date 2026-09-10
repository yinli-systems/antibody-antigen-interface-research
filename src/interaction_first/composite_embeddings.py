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

    def features(self, key: str, length: int, device: Any, base: Any) -> Any:
        """Return this source's columns, or None when its artifacts are missing."""
        del base  # a stored encoder does not depend on the base embedding
        return _concatenate_parts(self, key, length, device)

    def describe_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "stored_artifacts",
            "root": str(self.root),
            "representation_dim": self.dim,
            "presence_columns": PRESENCE_COLUMNS,
            "keys_covered": len(self.parts_by_key),
        }


class RandomProjectionSource:
    """A width-matched control that carries no information the base lacks.

    The treatment arm is wider than the control and therefore has more
    parameters in the input projections, so part of any gain could come from the
    wider projection rather than from the encoders. This source appends a fixed
    random linear projection of the base embedding: identical width, identical
    parameter count, and nothing the model could not already compute from its
    own input.

    The matrix is drawn once from ``seed`` and reused for every call and every
    process, so an arm built on it is reproducible in the way the encoders are.
    """

    def __init__(self, name: str, dim: int, seed: int, keys: Mapping[str, Any]) -> None:
        self.name = name
        self.dim = dim
        self.seed = seed
        self.parts_by_key = keys
        self._matrix: Any = None

    def covers(self, key: str) -> bool:
        return key in self.parts_by_key

    @property
    def width(self) -> int:
        return self.dim + PRESENCE_COLUMNS

    def _projection(self, base_dim: int, device: Any, dtype: Any) -> Any:
        import torch

        if self._matrix is None or int(self._matrix.shape[0]) != base_dim:
            generator = torch.Generator().manual_seed(self.seed)
            matrix = torch.randn(base_dim, self.dim, generator=generator)
            # Keep the projected columns on the scale of the base ones, so the
            # control differs from the treatment in content and not in size.
            self._matrix = matrix / (base_dim ** 0.5)
        return self._matrix.to(device=device, dtype=dtype)

    def features(self, key: str, length: int, device: Any, base: Any) -> Any:
        del key
        projected = base @ self._projection(int(base.shape[1]), device, base.dtype)
        if int(projected.shape[0]) != length:
            raise RepresentationError(
                f"{self.name} projected {int(projected.shape[0])} residues, "
                f"but the sequence has {length}"
            )
        return projected

    def describe_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "random_projection_of_base",
            "representation_dim": self.dim,
            "presence_columns": PRESENCE_COLUMNS,
            "keys_covered": len(self.parts_by_key),
            "seed": self.seed,
        }


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

    def __init__(
        self,
        base: Any,
        sources: Sequence[RepresentationSource],
        shuffle_seed: int | None = None,
    ) -> None:
        names = [source.name for source in sources]
        if len(set(names)) != len(names):
            raise RepresentationError(f"duplicate source names: {names}")
        self.base = base
        self.sources = tuple(sources)
        self.shuffle_seed = shuffle_seed
        self.filled: dict[str, int] = {source.name: 0 for source in self.sources}

    def _permute(self, features: Any, source_name: str, key: str, device: Any) -> Any:
        """Reorder a source's rows, destroying the residue correspondence.

        The attribution control. A width-matched random projection shows the gain
        is not extra capacity; it cannot show the gain is *positional*. This keeps
        the complex, the length, the vectors themselves and their distribution,
        and destroys only the fact that row i describes residue i.

        Shuffling features between complexes would be the obvious version and is
        not well defined here: these features are per residue and complexes differ
        in length, so donating them requires truncating or padding, which
        fabricates coordinates rather than removing information.

        The permutation is derived from the key, so it is the same in every
        process and every epoch: a fresh permutation each time would be noise
        injection, a different experiment.
        """
        import torch

        digest = hashlib.sha256(
            f"{self.shuffle_seed}:{source_name}:{key}".encode("ascii")
        ).digest()
        generator = torch.Generator()
        generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63))
        order = torch.randperm(int(features.shape[0]), generator=generator)
        return features.index_select(0, order.to(device=features.device))

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
            features = source.features(sequence_key, length, device, merged)
            present = features is not None
            if present and self.shuffle_seed is not None:
                features = self._permute(features, source.name, sequence_key, device)
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

    def describe(self) -> dict[str, Any]:  # noqa: D401
        """Report what will be concatenated, without any running counts.

        This is what a receipt written *before* the data is read may record. The
        zero-fill counters belong to ``summary`` and are zero until loads have
        happened, so reporting them here would state a fact that has not been
        measured yet.
        """
        described = {"sources": [source.describe_entry() for source in self.sources]}
        if self.shuffle_seed is not None:
            described["residue_shuffle_seed"] = self.shuffle_seed
            described["residue_shuffle"] = (
                "rows permuted per key; the residue correspondence is destroyed and "
                "the complex, length and vector distribution are kept"
            )
        return described

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


def build_random_control(
    manifest: Path, dim: int = 512, seed: int = 20260910
) -> list[RandomProjectionSource]:
    """Build the width-matched control for the AntiBERTy + ESM-IF1 arm.

    It mirrors that arm's coverage exactly -- two sources on the antibody side
    and one on the antigen -- so both arms present the model with the same input
    widths and the same parameter count. The difference between them is only
    what the appended columns contain.
    """
    chains = manifest_chain_keys(manifest)
    antibody = {entry["antibody_key"]: () for entry in chains.values()}
    both = {}
    for entry in chains.values():
        both[entry["antibody_key"]] = ()
        both[entry["antigen_key"]] = ()
    return [
        RandomProjectionSource("random_antibody", dim, seed, antibody),
        RandomProjectionSource("random_both", dim, seed + 1, both),
    ]


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
