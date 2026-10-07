"""Authenticated read-only access to sequence-aligned predicted structures."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import gzip
import hashlib
import math
from pathlib import Path
import sys

import numpy as np


THREEDI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
THREEDI_INDEX = {token: index + 3 for index, token in enumerate(THREEDI_ALPHABET)}
FEATURE_DIMENSION = 17


def _allow_large_fields() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class EndpointStructure:
    token_indices: np.ndarray
    features: np.ndarray
    observed_mask: np.ndarray


@dataclass(frozen=True)
class PreparedPairStructure:
    input_token_indices: np.ndarray
    input_features: np.ndarray
    input_feature_mask: np.ndarray
    token_labels: np.ndarray
    feature_targets: np.ndarray
    feature_target_mask: np.ndarray
    dropped_chains: tuple[bool, bool]


@dataclass(frozen=True)
class _Alignment:
    predicted_sequence_sha256: str
    mapping: tuple[int, ...]
    usable: bool


@dataclass(frozen=True)
class _FeatureSlice:
    sequence_sha256: str
    residue_offset: int
    residue_count: int


class PredictedStructureStore:
    """Join 3Di and 17D features to a holo sequence using sequence-only maps."""

    def __init__(
        self,
        *,
        alignment_manifest: Path,
        three_di_manifest: Path,
        feature_index: Path,
        feature_binary: Path,
        minimum_plddt: float = 70.0,
    ) -> None:
        if not 0.0 <= minimum_plddt <= 100.0:
            raise ValueError("minimum_plddt must be within [0, 100]")
        _allow_large_fields()
        self.minimum_confidence = minimum_plddt / 100.0
        self.alignments = self._load_alignments(alignment_manifest)
        self.three_di = self._load_three_di(three_di_manifest)
        self.feature_slices = self._load_feature_index(feature_index)
        expected_values = sum(
            value.residue_count * FEATURE_DIMENSION
            for value in self.feature_slices.values()
        )
        observed_values = feature_binary.stat().st_size // np.dtype("<f4").itemsize
        if feature_binary.stat().st_size % np.dtype("<f4").itemsize or observed_values != expected_values:
            raise ValueError("Predicted-backbone feature binary size differs")
        self.feature_values = np.memmap(feature_binary, mode="r", dtype="<f4")
        members = set(self.three_di)
        if members != set(self.feature_slices):
            raise ValueError("3Di and continuous-feature member sets differ")
        for member in members:
            if self.three_di[member][0] != self.feature_slices[member].sequence_sha256:
                raise ValueError("3Di and continuous-feature sequence authorities differ")

    @staticmethod
    def _load_alignments(path: Path) -> dict[tuple[str, str], _Alignment]:
        result: dict[tuple[str, str], _Alignment] = {}
        with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {
                "member",
                "holo_sequence_sha256",
                "predicted_sequence_sha256",
                "holo_length",
                "ambiguous_optimum",
                "usable_structure_modality",
                "holo_to_predicted_zero_based",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("Sequence-alignment manifest schema differs")
            for row in reader:
                key = (row["member"], row["holo_sequence_sha256"])
                if key in result:
                    raise ValueError("Sequence-alignment association is duplicated")
                usable = row["usable_structure_modality"] == "true"
                ambiguous = row["ambiguous_optimum"] == "true"
                if row["usable_structure_modality"] not in {"true", "false"} or row[
                    "ambiguous_optimum"
                ] not in {"true", "false"}:
                    raise ValueError("Sequence-alignment boolean encoding differs")
                text = row["holo_to_predicted_zero_based"]
                mapping = tuple(int(value) for value in text.split(",")) if text else ()
                if len(mapping) != int(row["holo_length"]):
                    raise ValueError("Sequence-alignment holo axis differs")
                if ambiguous and usable:
                    raise ValueError("Ambiguous sequence alignment cannot be usable")
                result[key] = _Alignment(
                    predicted_sequence_sha256=row["predicted_sequence_sha256"],
                    mapping=mapping,
                    usable=usable,
                )
        if not result:
            raise ValueError("Sequence-alignment manifest is empty")
        return result

    @staticmethod
    def _load_three_di(path: Path) -> dict[str, tuple[str, str]]:
        result: dict[str, tuple[str, str]] = {}
        with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"member", "sequence_sha256", "three_di_tokens"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("3Di manifest schema differs")
            for row in reader:
                member = row["member"]
                tokens = row["three_di_tokens"]
                if member in result:
                    raise ValueError("3Di member is duplicated")
                if not tokens or not set(tokens).issubset(THREEDI_INDEX):
                    raise ValueError("3Di token alphabet differs")
                result[member] = (row["sequence_sha256"], tokens)
        if not result:
            raise ValueError("3Di manifest is empty")
        return result

    @staticmethod
    def _load_feature_index(path: Path) -> dict[str, _FeatureSlice]:
        result: dict[str, _FeatureSlice] = {}
        expected_offset = 0
        with path.open("r", encoding="ascii", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {
                "member",
                "sequence_sha256",
                "residue_offset",
                "residue_count",
                "feature_dimension",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("Continuous-feature index schema differs")
            for row in reader:
                member = row["member"]
                offset = int(row["residue_offset"])
                count = int(row["residue_count"])
                if member in result or offset != expected_offset or count <= 0:
                    raise ValueError("Continuous-feature index order differs")
                if int(row["feature_dimension"]) != FEATURE_DIMENSION:
                    raise ValueError("Continuous-feature dimension differs")
                result[member] = _FeatureSlice(
                    sequence_sha256=row["sequence_sha256"],
                    residue_offset=offset,
                    residue_count=count,
                )
                expected_offset += count
        if not result:
            raise ValueError("Continuous-feature index is empty")
        return result

    def endpoint(self, *, member: str, holo_sequence: str) -> EndpointStructure:
        sequence_sha = _sequence_sha256(holo_sequence)
        key = (member, sequence_sha)
        if key not in self.alignments:
            raise ValueError("Predicted-structure alignment association is absent")
        alignment = self.alignments[key]
        length = len(holo_sequence)
        missing = EndpointStructure(
            token_indices=np.zeros(length, dtype=np.int64),
            features=np.zeros((length, FEATURE_DIMENSION), dtype=np.float32),
            observed_mask=np.zeros(length, dtype=np.bool_),
        )
        if not alignment.usable:
            return missing
        if member not in self.three_di or member not in self.feature_slices:
            raise ValueError("Predicted-structure member is absent")
        predicted_sha, three_di = self.three_di[member]
        feature_slice = self.feature_slices[member]
        if (
            predicted_sha != alignment.predicted_sequence_sha256
            or feature_slice.sequence_sha256 != predicted_sha
            or len(three_di) != feature_slice.residue_count
        ):
            raise ValueError("Predicted-structure sequence authority differs")
        mapped = np.asarray(alignment.mapping, dtype=np.int64)
        if mapped.shape != (length,) or (mapped < -1).any() or (mapped >= len(three_di)).any():
            raise ValueError("Predicted-structure mapping index differs")
        start = feature_slice.residue_offset * FEATURE_DIMENSION
        stop = (feature_slice.residue_offset + feature_slice.residue_count) * FEATURE_DIMENSION
        predicted_features = np.asarray(self.feature_values[start:stop]).reshape(
            feature_slice.residue_count, FEATURE_DIMENSION
        )
        valid = mapped >= 0
        mapped_valid = mapped[valid]
        confidence = predicted_features[mapped_valid, -1]
        confident = confidence >= self.minimum_confidence
        observed_positions = np.flatnonzero(valid)[confident]
        predicted_positions = mapped_valid[confident]
        token_indices = missing.token_indices.copy()
        features = missing.features.copy()
        observed = missing.observed_mask.copy()
        token_indices[observed_positions] = np.fromiter(
            (THREEDI_INDEX[three_di[index]] for index in predicted_positions),
            dtype=np.int64,
            count=len(predicted_positions),
        )
        features[observed_positions] = predicted_features[predicted_positions]
        observed[observed_positions] = True
        if not np.isfinite(features).all():
            raise ValueError("Predicted-structure endpoint contains non-finite features")
        return EndpointStructure(token_indices, features, observed)


def _hash_rank(seed: int, scope: str, index: int = -1) -> bytes:
    return hashlib.sha256(f"{seed}\0{scope}\0{index}".encode("ascii")).digest()


def _event(seed: int, scope: str, probability: float) -> bool:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Structure corruption probability must be within [0, 1]")
    threshold = int(probability * (1 << 64))
    return int.from_bytes(_hash_rank(seed, scope)[:8], "big") < threshold


def prepare_pair_structure(
    left: EndpointStructure,
    right: EndpointStructure,
    *,
    seed: int,
    training: bool,
    masked_residue_probability: float = 0.15,
    chain_dropout_probability: float = 0.15,
    system_dropout_probability: float = 0.10,
) -> PreparedPairStructure:
    """Create deterministic special-token-aligned inputs and targets."""
    for endpoint in (left, right):
        length = len(endpoint.token_indices)
        if (
            endpoint.token_indices.shape != (length,)
            or endpoint.features.shape != (length, FEATURE_DIMENSION)
            or endpoint.observed_mask.shape != (length,)
            or endpoint.token_indices.dtype != np.int64
            or endpoint.observed_mask.dtype != np.bool_
            or not np.issubdtype(endpoint.features.dtype, np.floating)
            or not np.isfinite(endpoint.features).all()
        ):
            raise ValueError("Endpoint predicted-structure tensors differ")
        if (endpoint.token_indices[endpoint.observed_mask] < 3).any():
            raise ValueError("Observed endpoint residues require 3Di tokens")
        if endpoint.token_indices[~endpoint.observed_mask].any() or endpoint.features[
            ~endpoint.observed_mask
        ].any():
            raise ValueError("Missing endpoint residues must be zero")
    if not 0.0 < masked_residue_probability <= 1.0:
        raise ValueError("masked_residue_probability must be within (0, 1]")

    system_dropped = training and _event(seed, "system-dropout", system_dropout_probability)
    dropped_chains = tuple(
        system_dropped
        or (training and _event(seed, f"chain-{chain}-dropout", chain_dropout_probability))
        for chain in range(2)
    )
    token_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    input_mask_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    target_mask_parts: list[np.ndarray] = []
    for chain, endpoint in enumerate((left, right)):
        tokens = endpoint.token_indices.copy()
        features = endpoint.features.astype(np.float32, copy=True)
        input_mask = endpoint.observed_mask.copy()
        labels = np.full(len(tokens), -100, dtype=np.int64)
        targets = np.zeros_like(features)
        target_mask = np.zeros(len(tokens), dtype=np.bool_)
        if dropped_chains[chain]:
            tokens[endpoint.observed_mask] = 2
            features[endpoint.observed_mask] = 0
            input_mask[endpoint.observed_mask] = False
        elif training:
            observed = np.flatnonzero(endpoint.observed_mask)
            if len(observed):
                count = max(1, math.ceil(len(observed) * masked_residue_probability))
                selected = sorted(
                    observed,
                    key=lambda index: _hash_rank(seed, f"chain-{chain}-mask", int(index)),
                )[:count]
                labels[selected] = tokens[selected]
                targets[selected] = features[selected]
                target_mask[selected] = True
                tokens[selected] = 1
                features[selected] = 0
                input_mask[selected] = False
        token_parts.append(np.concatenate([np.zeros(1, dtype=np.int64), tokens, np.zeros(1, dtype=np.int64)]))
        feature_parts.append(
            np.concatenate(
                [
                    np.zeros((1, FEATURE_DIMENSION), dtype=np.float32),
                    features,
                    np.zeros((1, FEATURE_DIMENSION), dtype=np.float32),
                ]
            )
        )
        input_mask_parts.append(np.concatenate([np.zeros(1, dtype=np.bool_), input_mask, np.zeros(1, dtype=np.bool_)]))
        label_parts.append(np.concatenate([np.full(1, -100, dtype=np.int64), labels, np.full(1, -100, dtype=np.int64)]))
        target_parts.append(
            np.concatenate(
                [
                    np.zeros((1, FEATURE_DIMENSION), dtype=np.float32),
                    targets,
                    np.zeros((1, FEATURE_DIMENSION), dtype=np.float32),
                ]
            )
        )
        target_mask_parts.append(np.concatenate([np.zeros(1, dtype=np.bool_), target_mask, np.zeros(1, dtype=np.bool_)]))
    return PreparedPairStructure(
        input_token_indices=np.concatenate(token_parts),
        input_features=np.concatenate(feature_parts),
        input_feature_mask=np.concatenate(input_mask_parts),
        token_labels=np.concatenate(label_parts),
        feature_targets=np.concatenate(target_parts),
        feature_target_mask=np.concatenate(target_mask_parts),
        dropped_chains=(bool(dropped_chains[0]), bool(dropped_chains[1])),
    )
