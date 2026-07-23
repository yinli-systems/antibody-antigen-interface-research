#!/usr/bin/env python3
"""Run development-only inference for asymmetric partner-retrieval scores."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sabdab2_asymmetric_partner_retrieval import (
    CHECKPOINT_SUITE_CLAIM_FORMAT_VERSION,
    CHECKPOINT_SUITE_COMPLETION_FORMAT_VERSION,
    FORMAT_VERSION,
    SCORE_ARTIFACT_FORMAT_VERSION,
    STRICT_NW_CONSTRAINT_MODES,
    _score_array_sha256,
    _expected_final_evaluation_contract,
    canonical_sha256,
    sha256_file,
)


STATISTICS_FORMAT_VERSION = "sabdab2_asymmetric_retrieval_statistics_v2"
CLUSTER_FORMAT_VERSION = "sabdab2_retrieval_anchor_clusters_v1"
AUTHORITY_VALIDATION_BOUNDARY_FORMAT = "sabdab2_authority_validation_nw_boundary_v2"
AUTHORITY_VALIDATION_BOUNDARY_INTENDED_USE = (
    "reused_development_confirmation_only"
)
ONE_SHOT_EVIDENCE_RUN_MODES = frozenset(
    {"final_refit_fixed_epochs", "train_only_control_bundle"}
)
EXPLORATORY_RUN_MODES = frozenset({"fold_selection_with_validation"})
EXPLORATORY_SPLIT_FORMATS = frozenset(
    {
        "sabdab2_interaction_first_v1",
        "sabdab2_identity_isolated_development_v1",
        "sabdab2_antigen_cluster_development_fold_v1",
        "sabdab2_iterative_nw_strict_development_split_v1",
    }
)
EXECUTION_LOCK_SHA256 = (
    "43638fdd17b9ce3e4cac8b83e42c16261ecbf79c2e628d0b494e5dcbe02f9fc9"
)
LOCKED_PERMUTATION_COUNT = 10_000
LOCKED_BOOTSTRAP_COUNT = 10_000
LOCKED_STATISTICS_SEED = 42
LOCKED_ALPHA = 0.05
LOCKED_CANDIDATE_SIZES = (10, 25, 50, 100)
LOCKED_CANDIDATE_SAMPLE_COUNT = 1_000
ONE_SHOT_LEDGER_FORMAT_VERSION = "sabdab2_one_shot_evaluation_ledger_v2"
CONTROL_LEDGER_FORMAT_VERSION = "sabdab2_control_one_shot_ledger_v3"
CONTROL_SUITE_COMPLETION_FORMAT_VERSION = (
    "sabdab2_control_two_boundary_suite_completion_v1"
)
CONTROL_SUITE_VERSION = "partner_retrieval_controls_protocol_v1"
CONTROL_NAMES = (
    "raw_shared_esm_mean_cosine",
    "random_untrained_calm_head",
    "antibody_only_unary",
    "antigen_only_unary",
    "length_composition_logistic",
)
CENTRAL_CONTROL_LEDGER_ROOT = Path(
    "/cluster/project/model_deployment/results/sabdab2/"
    "asymmetric_partner_retrieval/control_evaluation_registry"
)
DIRECTIONS = ("antibody_to_antigen", "antigen_to_antibody")
ALL_DIRECTIONS = (*DIRECTIONS, "bidirectional")
RECALL_KS = (1, 5, 10, 20)
METRIC_NAMES = ("mrr", *(f"recall_at_{value}" for value in RECALL_KS))


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _assert_finite_json(value: Any, location: str = "root") -> None:
    """Reject NaN and infinity anywhere in a decoded JSON document."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite JSON value at {location}")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_finite_json(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_finite_json(nested, f"{location}[{index}]")


def _read_file_bytes_once(path: Path, label: str) -> bytes:
    """Read one stable physical file so hashing and decoding share exact bytes."""
    target = path.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"{label} is missing: {target}")
    with target.open("rb") as handle:
        before = handle.seek(0, 2)
        handle.seek(0)
        descriptor_before = os.fstat(handle.fileno())
        payload = handle.read()
        descriptor_after = os.fstat(handle.fileno())
    current = target.stat()
    identity_before = (
        descriptor_before.st_dev,
        descriptor_before.st_ino,
        descriptor_before.st_size,
        descriptor_before.st_mtime_ns,
    )
    identity_after = (
        descriptor_after.st_dev,
        descriptor_after.st_ino,
        descriptor_after.st_size,
        descriptor_after.st_mtime_ns,
    )
    identity_current = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    )
    if (
        before != len(payload)
        or identity_before != identity_after
        or identity_after != identity_current
    ):
        raise RuntimeError(f"{label} changed while its immutable bytes were acquired")
    return payload


def _decode_strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    """Decode one finite JSON object while rejecting duplicate keys recursively."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains a non-finite JSON value: {value}")

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot decode {label} as strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    _assert_finite_json(value)
    return value


def _strict_json_object(path: Path, label: str) -> dict[str, Any]:
    """Load one finite JSON object while rejecting duplicate keys."""
    try:
        payload = _read_file_bytes_once(path, label)
    except OSError as error:
        raise ValueError(f"Cannot read {label} as strict JSON: {path}") from error
    return _decode_strict_json_bytes(payload, label)


def _physical_hashed_file(value: Any, expected_sha256: Any, label: str) -> Path:
    """Resolve and authenticate one absolute physical file reference."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    digest = str(expected_sha256)
    if not _is_sha256(digest):
        raise ValueError(f"{label} requires an exact SHA-256")
    observed = sha256_file(path)
    if observed != digest:
        raise ValueError(
            f"{label} physical SHA-256 mismatch: expected={digest}, observed={observed}"
        )
    return path


def _identifier_membership_sha256(identifiers: Sequence[str]) -> str:
    """Hash sorted identifiers with the authority-boundary membership contract."""
    return hashlib.sha256(
        ("\n".join(sorted(identifiers)) + "\n").encode("utf-8")
    ).hexdigest()


def _unique_inventory(value: Any, label: str) -> list[str]:
    """Normalize a non-empty unique identifier inventory."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    identifiers = [str(identifier) for identifier in value]
    if any(not identifier for identifier in identifiers):
        raise ValueError(f"{label} contains an empty identifier")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{label} contains duplicate identifiers")
    return identifiers


def _load_instance_sequence_keys(
    instances_path: Path,
    instance_ids: Sequence[str],
) -> dict[str, tuple[str, str]]:
    """Read selected Parquet rows and return authenticated sequence-key pairs."""
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "pandas with a Parquet engine is required for physical library verification"
        ) from error

    requested = sorted(set(str(value) for value in instance_ids))
    if not requested:
        raise ValueError("Physical library verification requires selected identifiers")
    try:
        frame = pd.read_parquet(
            instances_path,
            columns=["instance_id", "ab_sequence_sha256", "ag_sequence_sha256"],
            filters=[("instance_id", "in", requested)],
        )
    except Exception as error:
        raise ValueError(
            f"Cannot read selected rows from the prepared instance table: {instances_path}"
        ) from error
    if frame.empty or frame["instance_id"].isna().any():
        raise ValueError("Prepared instance rows are empty or contain null identifiers")
    frame["instance_id"] = frame["instance_id"].astype(str)
    if frame["instance_id"].duplicated().any():
        raise ValueError("Prepared instance rows contain duplicate identifiers")
    observed = set(frame["instance_id"])
    if observed != set(requested):
        raise ValueError(
            "Prepared instance rows do not match the authorized validation accounting"
        )
    result: dict[str, tuple[str, str]] = {}
    for row in frame.itertuples(index=False):
        antibody_key = str(row.ab_sequence_sha256)
        antigen_key = str(row.ag_sequence_sha256)
        if not _is_sha256(antibody_key) or not _is_sha256(antigen_key):
            raise ValueError(f"Prepared instance {row.instance_id} has an invalid sequence key")
        result[str(row.instance_id)] = (antibody_key, antigen_key)
    return result


def _validate_v2_boundary_and_library(
    dataset: Mapping[str, Any],
    antibody_keys: Sequence[str],
    antigen_keys: Sequence[str],
    positive_mask: np.ndarray,
) -> dict[str, Any]:
    """Reauthenticate v2 accounting and prove the scored library is validation-only."""
    split_path = _physical_hashed_file(
        dataset.get("split_manifest"),
        dataset.get("split_manifest_sha256"),
        "Authority-validation boundary",
    )
    authority_metadata = dataset.get("authority")
    if not isinstance(authority_metadata, Mapping):
        raise ValueError("Evaluation lacks an authority contract")
    authority_path = _physical_hashed_file(
        authority_metadata.get("path"),
        authority_metadata.get("sha256"),
        "Authority split",
    )
    instances_path = _physical_hashed_file(
        dataset.get("instances_path"),
        dataset.get("instances_sha256"),
        "Prepared instance table",
    )
    boundary = _strict_json_object(split_path, "authority-validation boundary")
    authority = _strict_json_object(authority_path, "authority split")
    if boundary.get("format_version") != AUTHORITY_VALIDATION_BOUNDARY_FORMAT:
        raise PermissionError("One-shot evidence requires the approved v2 boundary format")
    if (
        boundary.get("intended_use")
        != AUTHORITY_VALIDATION_BOUNDARY_INTENDED_USE
    ):
        raise PermissionError("V2 boundary has an unapproved intended use")
    if (
        dataset.get("split_format_version") != AUTHORITY_VALIDATION_BOUNDARY_FORMAT
        or dataset.get("split_intended_use")
        != AUTHORITY_VALIDATION_BOUNDARY_INTENDED_USE
    ):
        raise PermissionError("Evaluation does not declare the approved v2 format/use pair")
    if (
        boundary.get("stage") != "build_authority_validation_nw_clean_boundary"
        or boundary.get("valid") is not True
        or boundary.get("selection_influenced_by_labels_predictions_or_model_scores")
        is not False
        or boundary.get("frozen_test_identifiers_read_for_exclusion_only") is not True
        or boundary.get(
            "frozen_test_sequence_fields_accessed_normalized_hashed_or_aligned"
        )
        is not False
    ):
        raise PermissionError("V2 boundary violates its selection and sealed-test firewall")
    constraint_mode = boundary.get("constraint_mode")
    if constraint_mode not in STRICT_NW_CONSTRAINT_MODES:
        raise ValueError("V2 boundary lacks an approved exact-NW constraint mode")
    if dataset.get("strict_nw_constraint_mode") != constraint_mode:
        raise ValueError("Evaluation and physical boundary constraint modes differ")

    raw_splits = boundary.get("splits")
    expected_names = {"train", "validation", "excluded_validation"}
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != expected_names:
        raise ValueError("V2 boundary must contain exactly three accounting splits")
    inventories = {
        name: _unique_inventory(raw_splits[name], f"boundary.{name}")
        for name in sorted(expected_names)
    }
    train = set(inventories["train"])
    validation = set(inventories["validation"])
    excluded = set(inventories["excluded_validation"])
    if train & validation or train & excluded or validation & excluded:
        raise ValueError("V2 boundary accounting inventories overlap")
    membership = boundary.get("membership_sha256")
    expected_membership = {
        name: _identifier_membership_sha256(values)
        for name, values in inventories.items()
    }
    if not isinstance(membership, Mapping) or dict(membership) != expected_membership:
        raise ValueError("V2 boundary membership hash mismatch")
    scientific_hash = boundary.get("scientific_payload_sha256")
    unhashed_boundary = dict(boundary)
    unhashed_boundary.pop("scientific_payload_sha256", None)
    if not _is_sha256(str(scientific_hash)) or canonical_sha256(
        unhashed_boundary
    ) != scientific_hash:
        raise ValueError("V2 boundary scientific payload hash mismatch")

    authority_splits = authority.get("splits")
    if not isinstance(authority_splits, Mapping):
        raise ValueError("Authority split lacks a splits object")
    authority_inventories = {
        name: _unique_inventory(authority_splits.get(name), f"authority.{name}")
        for name in ("train", "validation", "test")
    }
    authority_train = set(authority_inventories["train"])
    authority_validation = set(authority_inventories["validation"])
    authority_test = set(authority_inventories["test"])
    if (
        authority_train & authority_validation
        or authority_train & authority_test
        or authority_validation & authority_test
    ):
        raise ValueError("Authority source splits overlap")
    if train != authority_train:
        raise ValueError("V2 boundary train does not exactly equal authority train")
    if validation | excluded != authority_validation:
        raise ValueError(
            "V2 validation plus excluded_validation does not equal authority validation"
        )
    if (train | validation | excluded) & authority_test:
        raise PermissionError("V2 boundary overlaps the sealed authority test")

    expected_dataset_membership = {
        name: canonical_sha256({"instance_ids": sorted(values)})
        for name, values in inventories.items()
    }
    if dataset.get("split_membership_sha256") != expected_dataset_membership:
        raise ValueError("Evaluation split-membership accounting differs from the boundary")
    if dataset.get("raw_instance_counts") != {
        name: len(values) for name, values in inventories.items()
    }:
        raise ValueError("Evaluation split counts differ from the physical boundary")
    if (
        dataset.get("loaded_splits") != ["train", "validation"]
        or dataset.get("unmaterialized_splits") != []
        or dataset.get("permanently_excluded_splits") != ["excluded_validation"]
        or dataset.get("excluded_validation_rows_materialized") is not False
        or dataset.get("authority_accounting_splits")
        != ["train", "validation", "excluded_validation"]
        or dataset.get("parquet_row_filter") != "train_and_validation_identifiers_only"
        or dataset.get("frozen_test_records_selected") is not False
    ):
        raise PermissionError("Evaluation materialization metadata violates v2 isolation")
    if (
        authority_metadata.get("authorized_source_pool_splits")
        != ["train", "validation"]
        or authority_metadata.get("sealed_source_splits") != ["test"]
        or authority_metadata.get("coverage_mode") != "exact"
        or int(authority_metadata.get("selected_sealed_overlap_count", -1)) != 0
        or int(authority_metadata.get("authorized_source_pool_count", -1))
        != len(authority_train | authority_validation)
        or int(authority_metadata.get("selected_pool_coverage_count", -1))
        != len(train | validation | excluded)
        or authority_metadata.get("sealed_source_counts")
        != {"test": len(authority_test)}
    ):
        raise PermissionError("Evaluation authority accounting violates the v2 contract")

    row_keys = _load_instance_sequence_keys(
        instances_path,
        inventories["validation"],
    )
    validation_pairs = {row_keys[identifier] for identifier in validation}
    expected_antibodies = sorted({pair[0] for pair in validation_pairs})
    expected_antigens = sorted({pair[1] for pair in validation_pairs})
    if list(antibody_keys) != expected_antibodies or list(antigen_keys) != expected_antigens:
        raise PermissionError("Scored libraries are not exactly the authorized validation libraries")
    observed_mask_pairs = {
        (antibody_keys[row], antigen_keys[column])
        for row, column in np.argwhere(np.asarray(positive_mask, dtype=bool))
    }
    if observed_mask_pairs != validation_pairs:
        raise PermissionError("Score positive graph is not exactly the validation pair graph")
    return {
        "format_version": AUTHORITY_VALIDATION_BOUNDARY_FORMAT,
        "intended_use": AUTHORITY_VALIDATION_BOUNDARY_INTENDED_USE,
        "constraint_mode": constraint_mode,
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "authority_split": str(authority_path),
        "authority_split_sha256": sha256_file(authority_path),
        "instances_path": str(instances_path),
        "instances_sha256": sha256_file(instances_path),
        "train_count": len(train),
        "validation_count": len(validation),
        "excluded_validation_count": len(excluded),
        "sealed_test_count": len(authority_test),
        "excluded_validation_materialized": False,
        "excluded_validation_rows_read_for_statistics": False,
        "validation_library_reconstructed_from_physical_rows": True,
        "validation_positive_graph_reconstructed_from_physical_rows": True,
    }


def _stable_rank_matrix(similarities: np.ndarray) -> np.ndarray:
    """Return one-based candidate ranks with stable candidate-index tie breaking."""
    values = np.asarray(similarities)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Similarities must be a finite non-empty matrix")
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int64)
    rows = np.arange(values.shape[0], dtype=np.int64)[:, None]
    ranks[rows, order] = np.arange(1, values.shape[1] + 1, dtype=np.int64)
    return ranks


def _best_positive_ranks(rank_matrix: np.ndarray, positive_mask: np.ndarray) -> np.ndarray:
    """Return the best known-positive rank for every anchor."""
    ranks = np.asarray(rank_matrix, dtype=np.int64)
    mask = np.asarray(positive_mask, dtype=bool)
    if ranks.ndim != 2 or ranks.shape != mask.shape or not mask.any(axis=1).all():
        raise ValueError("Rank and positive matrices must align with a positive per anchor")
    return np.where(mask, ranks, ranks.shape[1] + 1).min(axis=1)


def _metric_vectors(ranks: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a best-rank vector to per-anchor MRR and recall contributions."""
    values = np.asarray(ranks, dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0):
        raise ValueError("Ranks must be a non-empty positive vector")
    return {
        "mrr": 1.0 / values.astype(np.float64),
        **{
            f"recall_at_{cutoff}": (values <= cutoff).astype(np.float64)
            for cutoff in RECALL_KS
        },
    }


def _point_metrics(vectors: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {
        metric: _finite_float(np.asarray(vector, dtype=np.float64).mean(), metric)
        for metric, vector in vectors.items()
    }


def _survival_probability(candidate_count: int, degree: int, cutoff: int) -> float:
    """Return P(best positive rank > cutoff) for a random finite-library order."""
    if candidate_count <= 0 or not 1 <= degree <= candidate_count:
        raise ValueError("Positive degree must fit the finite candidate library")
    if cutoff <= 0:
        return 1.0
    if cutoff >= candidate_count or candidate_count - cutoff < degree:
        return 0.0
    probability = 1.0
    for offset in range(cutoff):
        probability *= (candidate_count - degree - offset) / (candidate_count - offset)
    return max(0.0, min(1.0, probability))


def exact_random_anchor_vectors(
    candidate_count: int,
    positive_degrees: Sequence[int],
) -> dict[str, np.ndarray]:
    """Return exact anchor-specific metric expectations under random ordering."""
    degrees = np.asarray(positive_degrees, dtype=np.int64)
    if degrees.ndim != 1 or degrees.size == 0:
        raise ValueError("At least one positive degree is required")
    if np.any(degrees <= 0) or np.any(degrees > candidate_count):
        raise ValueError("Every positive degree must fit the candidate library")
    expected = {
        metric: np.empty(degrees.size, dtype=np.float64) for metric in METRIC_NAMES
    }
    for index, degree_value in enumerate(degrees):
        degree = int(degree_value)
        survival_before = 1.0
        expected_reciprocal_rank = 0.0
        for rank in range(1, candidate_count - degree + 2):
            survival_after = _survival_probability(candidate_count, degree, rank)
            expected_reciprocal_rank += (survival_before - survival_after) / rank
            survival_before = survival_after
        expected["mrr"][index] = expected_reciprocal_rank
        for cutoff in RECALL_KS:
            expected[f"recall_at_{cutoff}"][index] = 1.0 - _survival_probability(
                candidate_count, degree, cutoff
            )
    return expected


def exact_random_direction(
    candidate_count: int,
    positive_degrees: Sequence[int],
) -> dict[str, Any]:
    """Summarize exact random expectations for heterogeneous positive degrees."""
    degrees = [int(value) for value in positive_degrees]
    vectors = exact_random_anchor_vectors(candidate_count, degrees)
    histogram = Counter(degrees)
    return {
        "anchor_count": len(degrees),
        "candidate_count": int(candidate_count),
        "positive_degree_histogram": {
            str(degree): histogram[degree] for degree in sorted(histogram)
        },
        "mean_positives_per_anchor": _finite_float(np.mean(degrees), "mean degree"),
        "mean_rank": _finite_float(
            np.mean(
                [
                    (candidate_count + 1.0) / (degree + 1.0)
                    for degree in degrees
                ]
            ),
            "expected mean rank",
        ),
        **_point_metrics(vectors),
    }


def exact_random_expectations(positive_mask: np.ndarray) -> dict[str, Any]:
    """Compute exact finite-library chance in both retrieval directions."""
    mask = np.asarray(positive_mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0 or not mask.any(axis=1).all() or not mask.any(axis=0).all():
        raise ValueError("The positive graph must contain a positive for every anchor")
    directional = {
        "antibody_to_antigen": exact_random_direction(
            mask.shape[1], mask.sum(axis=1).astype(int).tolist()
        ),
        "antigen_to_antibody": exact_random_direction(
            mask.shape[0], mask.sum(axis=0).astype(int).tolist()
        ),
    }
    directional["bidirectional"] = {
        metric: 0.5
        * (
            directional["antibody_to_antigen"][metric]
            + directional["antigen_to_antibody"][metric]
        )
        for metric in METRIC_NAMES
    }
    return {
        "protocol": "uniform_finite_candidate_order_with_anchor_specific_positive_degree",
        "formula": {
            "survival": "P(R>k)=C(N-k,d)/C(N,d)",
            "mean_rank": "E[R]=(N+1)/(d+1)",
        },
        **directional,
    }


def _decode_digest_keys(array: np.ndarray, name: str) -> list[str]:
    values = np.asarray(array)
    if values.ndim != 1 or values.dtype != np.dtype("S64") or values.size == 0:
        raise ValueError(f"{name} must be a non-empty fixed-width S64 vector")
    try:
        decoded = [bytes(value).decode("ascii") for value in values]
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} contains non-ASCII bytes") from error
    if not all(_is_sha256(value) for value in decoded):
        raise ValueError(f"{name} contains a non-SHA-256 identifier")
    if len(decoded) != len(set(decoded)) or decoded != sorted(decoded):
        raise ValueError(f"{name} must be unique and lexicographically ordered")
    return decoded


def _raw_npz_member_names(payload: bytes, label: str) -> tuple[str, ...]:
    """Authenticate the raw ZIP member inventory before NumPy normalization."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            raw_names = tuple(info.filename for info in archive.infolist())
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"{label} is not a valid NPZ archive") from error
    if not raw_names:
        raise ValueError(f"{label} contains no array members")
    if len(raw_names) != len(set(raw_names)):
        raise ValueError(f"{label} contains duplicate raw ZIP members")
    if any(
        not name.endswith(".npy")
        or not name[:-4]
        or "/" in name
        or "\\" in name
        for name in raw_names
    ):
        raise ValueError(f"{label} contains a non-canonical NPZ member name")
    logical_names = tuple(name[:-4] for name in raw_names)
    if len(logical_names) != len(set(logical_names)):
        raise ValueError(f"{label} contains duplicate logical array members")
    return logical_names


def load_score_artifact(path: Path) -> dict[str, Any]:
    """Load and internally authenticate one deterministic score artifact."""
    artifact_path = path.expanduser().resolve()
    artifact_bytes = _read_file_bytes_once(artifact_path, "score artifact")
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    required = {
        "similarity_matrix",
        "positive_mask",
        "antibody_keys",
        "antigen_keys",
        "metadata_json_utf8",
    }
    raw_members = _raw_npz_member_names(artifact_bytes, "score artifact")
    if set(raw_members) != required:
        raise ValueError(
            f"Score artifact raw members differ from the exact contract: "
            f"{sorted(raw_members)}"
        )
    with np.load(io.BytesIO(artifact_bytes), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != required:
            raise ValueError(
                f"Score artifact members differ from the exact contract: {sorted(archive.files)}"
            )
        arrays = {name: np.array(archive[name], copy=True) for name in required}
    if arrays["similarity_matrix"].dtype != np.float32:
        raise ValueError("Similarity matrix must be float32")
    if arrays["positive_mask"].dtype != np.bool_:
        raise ValueError("Positive mask must be boolean")
    if arrays["metadata_json_utf8"].dtype != np.uint8 or arrays[
        "metadata_json_utf8"
    ].ndim != 1:
        raise ValueError("Score metadata must be a uint8 UTF-8 vector")
    metadata = _decode_strict_json_bytes(
        arrays["metadata_json_utf8"].tobytes(), "score metadata"
    )
    if metadata.get("format_version") != SCORE_ARTIFACT_FORMAT_VERSION:
        raise ValueError("Score metadata format is incompatible")
    contract = metadata.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Score metadata lacks a contract")
    if metadata.get("contract_sha256") != canonical_sha256(contract):
        raise ValueError("Score metadata contract hash mismatch")
    scope = contract.get("scientific_scope")
    if scope != {
        "split": "validation",
        "intended_use": "development_statistics_only",
        "frozen_test_used": False,
        "contains_raw_sequences": False,
    }:
        raise PermissionError("Score artifact is not restricted to development validation")
    similarities = arrays["similarity_matrix"]
    mask = arrays["positive_mask"]
    antibody_keys = _decode_digest_keys(arrays["antibody_keys"], "antibody_keys")
    antigen_keys = _decode_digest_keys(arrays["antigen_keys"], "antigen_keys")
    if similarities.ndim != 2 or similarities.size == 0 or not np.isfinite(similarities).all():
        raise ValueError("Similarity matrix is empty, malformed, or non-finite")
    if mask.shape != similarities.shape:
        raise ValueError("Positive mask and similarities have different shapes")
    if len(antibody_keys) != similarities.shape[0] or len(antigen_keys) != similarities.shape[1]:
        raise ValueError("Score keys do not match matrix axes")
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        raise ValueError("Every score anchor must have a known positive")
    hashed_arrays = {
        name: arrays[name]
        for name in (
            "similarity_matrix",
            "positive_mask",
            "antibody_keys",
            "antigen_keys",
        )
    }
    observed_hashes = {
        name: _score_array_sha256(value) for name, value in hashed_arrays.items()
    }
    if contract.get("array_sha256") != observed_hashes:
        raise ValueError("Score array hash mismatch")
    if contract.get("array_dtypes") != {
        name: str(value.dtype) for name, value in hashed_arrays.items()
    }:
        raise ValueError("Score array dtype contract mismatch")
    if contract.get("matrix_shape") != list(similarities.shape):
        raise ValueError("Score matrix shape contract mismatch")
    if int(contract.get("positive_edge_count", -1)) != int(mask.sum()):
        raise ValueError("Score positive-edge count mismatch")
    return {
        "path": artifact_path,
        "sha256": artifact_sha256,
        "metadata": metadata,
        "similarities": similarities,
        "positive_mask": mask,
        "antibody_keys": antibody_keys,
        "antigen_keys": antigen_keys,
        "array_sha256": observed_hashes,
    }


def _expected_evaluation_context(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    dataset = evaluation["dataset"]
    authority = dataset["authority"]
    context = {
        "evaluation_format_version": evaluation["format_version"],
        "stage": evaluation["stage"],
        "split": evaluation["split"],
        "frozen_test_used": evaluation["frozen_test_used"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "config_sha256": evaluation["config_sha256"],
        "instances_sha256": dataset["instances_sha256"],
        "split_manifest_sha256": dataset["split_manifest_sha256"],
        "split_format_version": dataset["split_format_version"],
        "split_intended_use": dataset["split_intended_use"],
        "strict_nw_constraint_mode": dataset.get("strict_nw_constraint_mode"),
        "authority_split_sha256": authority["sha256"],
        "authorized_source_pool_splits": authority["authorized_source_pool_splits"],
        "authority_sealed_splits": authority["sealed_source_splits"],
        "authority_selected_sealed_overlap_count": authority[
            "selected_sealed_overlap_count"
        ],
        "evaluation_embedding_corpora_sha256": canonical_sha256(
            evaluation["embedding_corpora"]
        ),
    }
    ledger = evaluation.get("one_shot_ledger")
    if isinstance(ledger, Mapping):
        context["one_shot_ledger_key"] = ledger["canonical_key"]
        context["one_shot_ledger_path"] = ledger["canonical_path"]
    if evaluation.get("checkpoint_run_mode") == "final_refit_fixed_epochs":
        evidence = evaluation.get("locked_submission_evidence")
        claim = evaluation.get("checkpoint_suite_claim")
        expected_evidence_keys = {
            "submission_manifest",
            "protocol_v1_5_execution_lock",
            "protocol_v1_6_execution_hardening",
            "boundary_verification_v2",
            "fixed_evaluation",
            "selected_boundary",
        }
        if not isinstance(evidence, Mapping) or set(evidence) != expected_evidence_keys:
            raise ValueError("Final evaluation lacks exact locked submission evidence")
        if not isinstance(claim, Mapping) or set(claim) != {"path", "sha256"}:
            raise ValueError("Final evaluation lacks an exact checkpoint-suite claim")
        for name in (
            "submission_manifest",
            "protocol_v1_5_execution_lock",
            "protocol_v1_6_execution_hardening",
            "boundary_verification_v2",
        ):
            reference = evidence.get(name)
            if not isinstance(reference, Mapping) or set(reference) != {
                "path",
                "sha256",
            }:
                raise ValueError(f"Locked submission reference {name!r} is malformed")
        context.update(
            {
                "submission_manifest_sha256": evidence["submission_manifest"][
                    "sha256"
                ],
                "protocol_v1_5_execution_lock_sha256": evidence[
                    "protocol_v1_5_execution_lock"
                ]["sha256"],
                "protocol_v1_6_execution_hardening_sha256": evidence[
                    "protocol_v1_6_execution_hardening"
                ]["sha256"],
                "boundary_verification_v2_sha256": evidence[
                    "boundary_verification_v2"
                ]["sha256"],
                "fixed_evaluation_sha256": canonical_sha256(
                    evidence["fixed_evaluation"]
                ),
                "selected_boundary": evidence["selected_boundary"],
                "checkpoint_suite_claim_sha256": claim["sha256"],
            }
        )
    elif evaluation.get("checkpoint_run_mode") == "train_only_control_bundle":
        evidence = evaluation.get("locked_submission_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("Control evaluation lacks locked submission evidence")
        protocol = evidence.get("execution_protocol_v1_6")
        control_protocol = evidence.get("control_protocol_v1")
        canonical = evidence.get("canonical_sequence_manifest")
        embedding_manifests = evidence.get(
            "training_embedding_content_manifest_sha256"
        )
        required_modes = evidence.get("required_boundary_modes")
        if (
            not isinstance(protocol, Mapping)
            or not isinstance(control_protocol, Mapping)
            or not isinstance(canonical, Mapping)
            or not isinstance(embedding_manifests, Mapping)
            or not isinstance(required_modes, list)
            or evidence.get("control_suite_completion_required") is not True
        ):
            raise ValueError("Control locked submission evidence is malformed")
        context.update(
            {
                "control_bundle_sha256": evaluation["control_bundle_sha256"],
                "control_evaluation_identity_sha256": ledger["canonical_key"],
                "locked_submission_evidence_sha256": canonical_sha256(evidence),
                "control_protocol_v1": dict(control_protocol),
                "control_protocol_v1_sha256": control_protocol["sha256"],
                "execution_protocol_v1_6_sha256": protocol["sha256"],
                "canonical_sequence_manifest_sha256": canonical["sha256"],
                "training_embedding_content_manifest_sha256": canonical_sha256(
                    embedding_manifests
                ),
                "required_boundary_modes": list(required_modes),
                "control_suite_completion_required": True,
            }
        )
    return context


def _classify_evaluation_scope(evaluation: Mapping[str, Any]) -> str:
    """Classify an evaluation while failing closed for final/control evidence."""
    run_mode = evaluation.get("checkpoint_run_mode")
    one_shot = evaluation.get("one_shot_evaluation")
    if run_mode in ONE_SHOT_EVIDENCE_RUN_MODES:
        if one_shot is not True:
            raise PermissionError("Final-refit/control evidence must be explicitly one-shot")
        return "reused_development_confirmation"
    if run_mode in EXPLORATORY_RUN_MODES:
        if one_shot is not False:
            raise PermissionError("Exploratory compatibility requires one_shot_evaluation=false")
        return "exploratory_development"
    raise PermissionError("Evaluation run mode is not approved for statistical evidence")


def _validate_locked_confirmation_parameters(args: argparse.Namespace) -> dict[str, Any]:
    """Enforce the immutable v1.5 statistical settings for one-shot evidence."""
    observed = {
        "permutations": int(args.permutations),
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
        "alpha": float(args.alpha),
        "candidate_sizes": list(args.candidate_sizes),
        "candidate_samples": int(args.candidate_samples),
    }
    expected = {
        "permutations": LOCKED_PERMUTATION_COUNT,
        "bootstrap_samples": LOCKED_BOOTSTRAP_COUNT,
        "seed": LOCKED_STATISTICS_SEED,
        "alpha": LOCKED_ALPHA,
        "candidate_sizes": list(LOCKED_CANDIDATE_SIZES),
        "candidate_samples": LOCKED_CANDIDATE_SAMPLE_COUNT,
    }
    if observed != expected:
        raise PermissionError(
            "One-shot statistics arguments differ from the locked v1.5 execution contract: "
            f"expected={expected}, observed={observed}"
        )
    if args.anchor_clusters_json is not None or args.expected_anchor_clusters_sha256 is not None:
        raise PermissionError(
            "The locked confirmation analysis permits anchor and joint bipartite-component "
            "bootstraps only; optional cluster/subgroup analysis is post hoc"
        )
    return {
        "protocol": "PARTNER_RETRIEVAL_PROTOCOL_V1_5_EXECUTION_LOCK",
        "protocol_sha256": EXECUTION_LOCK_SHA256,
        **expected,
        "oversized_candidate_sizes": "omit_per_direction",
        "native_complete_library_is_primary": True,
    }


def _verify_control_suite_completion(
    evaluation: Mapping[str, Any],
    evaluation_path: Path,
    evaluation_sha256: str,
    artifact: Mapping[str, Any],
    ledger_path: Path,
    ledger_sha256: str,
    control_name: str,
) -> dict[str, Any]:
    """Authenticate the canonical two-boundary completion for a control bundle."""
    bundle_sha256 = str(evaluation.get("control_bundle_sha256", ""))
    if bundle_sha256 != evaluation.get("checkpoint_sha256") or not _is_sha256(
        bundle_sha256
    ):
        raise ValueError("Control evaluation lacks its exact control-bundle SHA-256")
    receipt_path = (
        ledger_path.parent / "suite-completions" / f"{bundle_sha256}.json"
    ).resolve()
    receipt = _strict_json_object(receipt_path, "control suite completion")
    expected_top_keys = {
        "format_version",
        "status",
        "control_suite_version",
        "completion_identity_sha256",
        "control_bundle",
        "control_protocol_v1",
        "execution_protocol_v1_6",
        "boundary_verification_v2",
        "authority_split",
        "immutable_inputs",
        "required_boundary_modes",
        "boundaries",
        "frozen_test_used",
    }
    if set(receipt) != expected_top_keys:
        raise ValueError("Control suite completion has unexpected fields")
    scientific_payload = dict(receipt)
    completion_identity = scientific_payload.pop("completion_identity_sha256", None)
    if (
        receipt.get("format_version") != CONTROL_SUITE_COMPLETION_FORMAT_VERSION
        or receipt.get("status") != "completed"
        or receipt.get("control_suite_version") != CONTROL_SUITE_VERSION
        or receipt.get("frozen_test_used") is not False
        or completion_identity != canonical_sha256(scientific_payload)
    ):
        raise ValueError("Control suite completion identity or scope is invalid")
    bundle = receipt.get("control_bundle")
    if not isinstance(bundle, Mapping) or set(bundle) != {
        "path",
        "sha256",
        "config_sha256",
    }:
        raise ValueError("Control suite completion bundle reference is malformed")
    bundle_path = _physical_hashed_file(
        bundle["path"], bundle["sha256"], "control bundle"
    )
    bundle_bytes = _read_file_bytes_once(bundle_path, "control bundle")
    if hashlib.sha256(bundle_bytes).hexdigest() != bundle["sha256"]:
        raise ValueError("Control bundle changed during physical authentication")
    raw_bundle_members = _raw_npz_member_names(bundle_bytes, "control bundle")
    if "metadata_json_utf8" not in raw_bundle_members:
        raise ValueError("Control bundle metadata member is missing")
    with np.load(io.BytesIO(bundle_bytes), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Control bundle contains duplicate array members")
        metadata_array = np.array(archive["metadata_json_utf8"], copy=True)
    if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
        raise ValueError("Control bundle metadata is malformed")
    bundle_metadata = _decode_strict_json_bytes(
        metadata_array.tobytes(), "control bundle metadata"
    )
    bundle_contract = bundle_metadata.get("contract")
    if (
        not isinstance(bundle_contract, Mapping)
        or bundle_metadata.get("contract_sha256")
        != canonical_sha256(bundle_contract)
        or not isinstance(bundle_contract.get("config"), Mapping)
    ):
        raise ValueError("Control bundle metadata contract is invalid")
    bundle_config = bundle_contract["config"]
    unhashed_bundle_config = {
        key: value for key, value in bundle_config.items() if key != "config_sha256"
    }
    if bundle_config.get("config_sha256") != canonical_sha256(
        unhashed_bundle_config
    ):
        raise ValueError("Control bundle config SHA-256 is invalid")
    if (
        bundle["sha256"] != bundle_sha256
        or bundle.get("config_sha256")
        != evaluation.get("control", {}).get("bundle_config_sha256")
        or bundle.get("config_sha256") != bundle_config.get("config_sha256")
    ):
        raise ValueError("Control suite completion binds another control bundle")
    evidence = evaluation.get("locked_submission_evidence")
    expected_evidence_keys = {
        "format_version",
        "control_protocol_v1",
        "execution_protocol_v1_6",
        "prepared_instances",
        "canonical_sequence_manifest",
        "training_embedding_content_manifest_sha256",
        "boundary_verification_v2",
        "authority_split",
        "selected_boundary",
        "required_boundary_modes",
        "control_suite_completion_required",
        "frozen_test_used",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_evidence_keys:
        raise ValueError("Control locked submission evidence schema is not exact")
    if (
        evaluation.get("control_protocol_v1") != evidence["control_protocol_v1"]
        or evaluation.get("execution_protocol_v1_6")
        != evidence["execution_protocol_v1_6"]
        or bundle_config.get("control_protocol_v1")
        != evidence["control_protocol_v1"]
        or bundle_config.get("execution_protocol_v1_6")
        != evidence["execution_protocol_v1_6"]
    ):
        raise ValueError("Control protocols differ across evaluation and bundle")
    physical_references = {}
    for field in (
        "control_protocol_v1",
        "execution_protocol_v1_6",
        "boundary_verification_v2",
        "authority_split",
    ):
        physical_references[field] = _exact_hashed_reference(
            receipt[field], f"control suite completion {field}"
        )
        if physical_references[field] != evidence[field]:
            raise ValueError(f"Control suite completion {field} differs from evaluation")
    authority = evaluation.get("dataset", {}).get("authority", {})
    if physical_references["authority_split"] != {
        "path": str(Path(str(authority.get("path", ""))).expanduser().resolve()),
        "sha256": authority.get("sha256"),
    }:
        raise ValueError("Control suite completion authority differs from evaluation")
    immutable = receipt.get("immutable_inputs")
    if not isinstance(immutable, Mapping) or set(immutable) != {
        "prepared_instances",
        "canonical_sequence_manifest",
        "training_embedding_content_manifest_sha256",
    }:
        raise ValueError("Control suite completion immutable inputs are malformed")
    prepared = _exact_hashed_reference(
        immutable["prepared_instances"], "control prepared instances"
    )
    canonical = _exact_hashed_reference(
        immutable["canonical_sequence_manifest"],
        "control canonical sequence manifest",
    )
    embedding_manifests = immutable.get(
        "training_embedding_content_manifest_sha256"
    )
    if (
        not isinstance(embedding_manifests, Mapping)
        or set(embedding_manifests)
        != {"primary_antibody", "shared_esm_antibody", "esm_antigen"}
        or not all(_is_sha256(str(value)) for value in embedding_manifests.values())
    ):
        raise ValueError("Control suite completion embedding manifests are invalid")
    if (
        prepared != evidence["prepared_instances"]
        or canonical != evidence["canonical_sequence_manifest"]
        or dict(embedding_manifests)
        != evidence["training_embedding_content_manifest_sha256"]
        or bundle_config.get("locked_training_embedding_content_manifest_sha256")
        != dict(embedding_manifests)
    ):
        raise ValueError("Control immutable inputs differ across receipt/evaluation/bundle")
    dataset = evaluation["dataset"]
    if prepared != {
        "path": str(Path(str(dataset.get("instances_path", ""))).expanduser().resolve()),
        "sha256": dataset.get("instances_sha256"),
    }:
        raise ValueError("Control suite completion prepared instances differ")
    dataset_canonical = dataset.get("canonical_sequence_manifest")
    if isinstance(dataset_canonical, Mapping) and canonical != dict(dataset_canonical):
        raise ValueError("Control suite completion canonical manifest differs")
    required_modes = list(STRICT_NW_CONSTRAINT_MODES)
    if (
        receipt.get("required_boundary_modes") != required_modes
        or evidence.get("required_boundary_modes") != required_modes
        or evidence.get("control_suite_completion_required") is not True
        or evidence.get("frozen_test_used") is not False
    ):
        raise ValueError("Control suite completion boundary modes are not locked")
    boundaries = receipt.get("boundaries")
    if not isinstance(boundaries, Mapping) or set(boundaries) != set(required_modes):
        raise ValueError("Control suite completion lacks both mandatory boundaries")
    authenticated_boundaries: dict[str, Any] = {}
    for mode in required_modes:
        boundary = boundaries[mode]
        if not isinstance(boundary, Mapping) or set(boundary) != {
            "split_manifest",
            "boundary_scientific_payload_sha256",
            "ledger",
            "output_dir",
            "controls",
        }:
            raise ValueError("Control suite completion boundary is malformed")
        split_reference = _exact_hashed_reference(
            boundary["split_manifest"], f"control boundary {mode} split"
        )
        if not _is_sha256(str(boundary["boundary_scientific_payload_sha256"])):
            raise ValueError("Control boundary scientific payload digest is invalid")
        ledger_reference = boundary["ledger"]
        if not isinstance(ledger_reference, Mapping) or set(ledger_reference) != {
            "path",
            "sha256",
            "canonical_key",
        }:
            raise ValueError("Control suite completion ledger reference is malformed")
        boundary_ledger_path = _physical_hashed_file(
            ledger_reference["path"],
            ledger_reference["sha256"],
            f"control boundary {mode} ledger",
        )
        output_dir_raw = boundary.get("output_dir")
        if not isinstance(output_dir_raw, str) or not Path(
            output_dir_raw
        ).expanduser().is_absolute():
            raise ValueError("Control suite completion output_dir must be absolute")
        output_dir = Path(output_dir_raw).expanduser().resolve()
        controls = boundary.get("controls")
        if not isinstance(controls, Mapping) or set(controls) != set(CONTROL_NAMES):
            raise ValueError("Control suite completion does not contain all controls")
        authenticated_controls = {}
        for name in CONTROL_NAMES:
            record = controls[name]
            if not isinstance(record, Mapping) or set(record) != {
                "evaluation_json",
                "score_artifact",
            }:
                raise ValueError("Control suite completion output record is malformed")
            evaluation_reference = _exact_hashed_reference(
                record["evaluation_json"], f"control suite {mode}.{name}.evaluation"
            )
            score_reference = record["score_artifact"]
            if not isinstance(score_reference, Mapping) or set(score_reference) != {
                "path",
                "sha256",
                "contract_sha256",
            }:
                raise ValueError("Control suite score reference is malformed")
            score_path = _physical_hashed_file(
                score_reference["path"],
                score_reference["sha256"],
                f"control suite {mode}.{name}.score",
            )
            if not _is_sha256(str(score_reference["contract_sha256"])):
                raise ValueError("Control suite score contract digest is invalid")
            if (
                Path(evaluation_reference["path"]).parent != output_dir
                or score_path.parent != output_dir
            ):
                raise ValueError("Control suite output escapes its declared directory")
            authenticated_controls[name] = {
                "evaluation_json": evaluation_reference,
                "score_artifact": {
                    "path": str(score_path),
                    "sha256": str(score_reference["sha256"]),
                    "contract_sha256": str(score_reference["contract_sha256"]),
                },
            }
        authenticated_boundaries[mode] = {
            "split_manifest": split_reference,
            "ledger": {
                "path": str(boundary_ledger_path),
                "sha256": str(ledger_reference["sha256"]),
                "canonical_key": str(ledger_reference["canonical_key"]),
            },
            "controls": authenticated_controls,
        }
    selected_mode = evaluation["dataset"].get("strict_nw_constraint_mode")
    if selected_mode not in authenticated_boundaries:
        raise ValueError("Control evaluation boundary is absent from suite completion")
    selected = authenticated_boundaries[selected_mode]
    selected_control = selected["controls"][control_name]
    if (
        selected["ledger"]["path"] != str(ledger_path)
        or selected["ledger"]["sha256"] != ledger_sha256
        or selected_control["evaluation_json"]
        != {"path": str(evaluation_path), "sha256": evaluation_sha256}
        or selected_control["score_artifact"]["path"] != str(artifact["path"])
        or selected_control["score_artifact"]["sha256"] != artifact["sha256"]
        or selected_control["score_artifact"]["contract_sha256"]
        != artifact["metadata"]["contract_sha256"]
    ):
        raise ValueError("Control suite completion does not bind analyzed artifacts")
    return {
        "required": True,
        "status": "completed",
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "control_bundle": {
            "path": str(bundle_path),
            "sha256": bundle_sha256,
            "config_sha256": bundle["config_sha256"],
        },
        "boundaries": authenticated_boundaries,
    }


def _verify_final_refit_one_shot_ledger(
    evaluation: Mapping[str, Any],
    evaluation_path: Path,
    evaluation_sha256: str,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the canonical completed ledger for a final-refit evaluation."""
    if evaluation.get("checkpoint_run_mode") != "final_refit_fixed_epochs":
        if evaluation.get("checkpoint_run_mode") == "train_only_control_bundle":
            control = evaluation.get("control")
            if not isinstance(control, Mapping) or not isinstance(
                control.get("name"), str
            ):
                raise PermissionError("One-shot control evaluation lacks a control identity")
            control_name = control["name"]
            if control_name not in CONTROL_NAMES:
                raise PermissionError("One-shot control identity is not in the locked suite")
            reference = evaluation.get("one_shot_ledger")
            if not isinstance(reference, Mapping) or set(reference) != {
                "format_version",
                "canonical_key",
                "canonical_path",
                "registry_dir",
            }:
                raise PermissionError(
                    "One-shot control evaluation lacks its exact central-ledger reference"
                )
            if reference.get("format_version") != CONTROL_LEDGER_FORMAT_VERSION:
                raise ValueError("Control one-shot ledger reference has the wrong format")
            dataset = evaluation["dataset"]
            checkpoint_sha256 = str(evaluation.get("checkpoint_sha256", ""))
            control_bundle_sha256 = str(
                evaluation.get("control_bundle_sha256", "")
            )
            split_sha256 = str(dataset.get("split_manifest_sha256", ""))
            authority_sha256 = str(dataset["authority"].get("sha256", ""))
            evidence = evaluation.get("locked_submission_evidence")
            if not isinstance(evidence, Mapping):
                raise PermissionError("Control evaluation lacks locked v1.6 evidence")
            protocol_reference = evidence.get("execution_protocol_v1_6")
            control_protocol_reference = evidence.get("control_protocol_v1")
            prepared_reference = evidence.get("prepared_instances")
            canonical_reference = evidence.get("canonical_sequence_manifest")
            embedding_manifests = evidence.get(
                "training_embedding_content_manifest_sha256"
            )
            if (
                control_bundle_sha256 != checkpoint_sha256
                or not isinstance(protocol_reference, Mapping)
                or not isinstance(control_protocol_reference, Mapping)
                or not isinstance(prepared_reference, Mapping)
                or not isinstance(canonical_reference, Mapping)
                or not isinstance(embedding_manifests, Mapping)
            ):
                raise ValueError("Control evaluation v1.6 identity evidence is malformed")
            ledger_contract = {
                "format_version": CONTROL_LEDGER_FORMAT_VERSION,
                "control_suite_version": CONTROL_SUITE_VERSION,
                "control_bundle_sha256": control_bundle_sha256,
                "split_manifest_sha256": split_sha256,
                "authority_split_sha256": authority_sha256,
                "control_protocol_v1_sha256": control_protocol_reference.get(
                    "sha256"
                ),
                "execution_protocol_v1_6_sha256": protocol_reference.get(
                    "sha256"
                ),
                "prepared_instances_sha256": prepared_reference.get("sha256"),
                "canonical_sequence_manifest_sha256": canonical_reference.get(
                    "sha256"
                ),
                "training_embedding_content_manifest_sha256": dict(
                    embedding_manifests
                ),
                "split": "validation",
                "controls": list(CONTROL_NAMES),
                "frozen_test_used": False,
            }
            expected_key = canonical_sha256(ledger_contract)
            if reference.get("canonical_key") != expected_key:
                raise ValueError("Control one-shot ledger key is not canonical")
            registry = Path(str(reference.get("registry_dir", ""))).expanduser().resolve()
            ledger_path = Path(str(reference.get("canonical_path", ""))).expanduser().resolve()
            if registry != CENTRAL_CONTROL_LEDGER_ROOT.resolve():
                raise ValueError("Control evaluation used an unapproved ledger registry")
            if ledger_path != (registry / f"{expected_key}.json").resolve():
                raise ValueError("Control one-shot ledger path is not canonical")
            if not ledger_path.is_file():
                raise FileNotFoundError(
                    f"Completed control one-shot ledger is missing: {ledger_path}"
                )
            ledger = _strict_json_object(ledger_path, "completed control one-shot ledger")
            expected_ledger_keys = {
                *ledger_contract,
                "evaluation_identity_sha256",
                "status",
                "requested_output_dir",
                "control_protocol_v1",
                "execution_protocol_v1_6",
                "locked_submission_evidence_sha256",
                "control_outputs",
            }
            if set(ledger) != expected_ledger_keys:
                raise ValueError("Completed control ledger schema is not exact")
            for name, expected in {
                **ledger_contract,
                "evaluation_identity_sha256": expected_key,
                "status": "completed",
            }.items():
                if ledger.get(name) != expected:
                    raise ValueError(
                        f"Completed control ledger field {name!r} is inconsistent"
                    )
            if (
                ledger.get("control_protocol_v1") != control_protocol_reference
                or evaluation.get("control_protocol_v1")
                != control_protocol_reference
                or ledger.get("execution_protocol_v1_6") != protocol_reference
                or ledger.get("locked_submission_evidence_sha256")
                != canonical_sha256(evidence)
            ):
                raise ValueError("Completed control ledger v1.6 evidence is inconsistent")
            requested_output = Path(
                str(ledger.get("requested_output_dir", ""))
            ).expanduser().resolve()
            if requested_output != evaluation_path.parent.resolve():
                raise ValueError("Completed control ledger points to a different output suite")
            outputs = ledger.get("control_outputs")
            if not isinstance(outputs, Mapping) or set(outputs) != set(CONTROL_NAMES):
                raise ValueError("Completed control ledger does not contain the full locked suite")
            for name in CONTROL_NAMES:
                record = outputs[name]
                if not isinstance(record, Mapping) or set(record) != {
                    "evaluation_json",
                    "evaluation_json_sha256",
                    "score_artifact",
                    "score_artifact_sha256",
                    "score_contract_sha256",
                    "bidirectional_mrr",
                }:
                    raise ValueError(f"Control ledger output {name!r} is malformed")
                evaluation_file = Path(
                    str(record.get("evaluation_json", ""))
                ).expanduser().resolve()
                score_file = Path(
                    str(record.get("score_artifact", ""))
                ).expanduser().resolve()
                if (
                    evaluation_file.parent != requested_output
                    or score_file.parent != requested_output
                    or not evaluation_file.is_file()
                    or not score_file.is_file()
                    or sha256_file(evaluation_file)
                    != record.get("evaluation_json_sha256")
                    or sha256_file(score_file) != record.get("score_artifact_sha256")
                    or not _is_sha256(str(record.get("score_contract_sha256", "")))
                ):
                    raise ValueError(f"Control ledger output {name!r} failed physical authentication")
            selected = outputs[control_name]
            if (
                Path(str(selected.get("evaluation_json", ""))).expanduser().resolve()
                != evaluation_path
                or selected.get("evaluation_json_sha256") != evaluation_sha256
                or Path(str(selected.get("score_artifact", ""))).expanduser().resolve()
                != artifact["path"]
                or selected.get("score_artifact_sha256") != artifact["sha256"]
                or selected.get("score_contract_sha256")
                != artifact["metadata"]["contract_sha256"]
            ):
                raise ValueError("Control ledger entry does not bind the analyzed artifacts")
            ledger_sha256 = sha256_file(ledger_path)
            suite_completion = _verify_control_suite_completion(
                evaluation,
                evaluation_path,
                evaluation_sha256,
                artifact,
                ledger_path,
                ledger_sha256,
                control_name,
            )
            return {
                "required": True,
                "mode": "authenticated_completed_central_control_suite_ledger",
                "format_version": CONTROL_LEDGER_FORMAT_VERSION,
                "status": "completed",
                "canonical_key": expected_key,
                "path": str(ledger_path),
                "sha256": ledger_sha256,
                "control_name": control_name,
                "checkpoint_suite_completion": suite_completion,
            }
        raise PermissionError("One-shot ledger verification received an unknown run mode")

    reference = evaluation.get("one_shot_ledger")
    expected_reference_keys = {
        "format_version",
        "canonical_key",
        "canonical_path",
        "registry_dir",
    }
    if not isinstance(reference, Mapping) or set(reference) != expected_reference_keys:
        raise PermissionError("Final-refit evaluation lacks its exact one-shot ledger reference")
    if reference.get("format_version") != ONE_SHOT_LEDGER_FORMAT_VERSION:
        raise ValueError("Final-refit one-shot ledger reference has the wrong format")
    checkpoint_sha256 = str(evaluation.get("checkpoint_sha256", ""))
    dataset = evaluation["dataset"]
    split_sha256 = str(dataset.get("split_manifest_sha256", ""))
    authority_sha256 = str(dataset["authority"].get("sha256", ""))
    expected_key = canonical_sha256(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "split_manifest_sha256": split_sha256,
            "authority_split_sha256": authority_sha256,
            "split": "validation",
        }
    )
    if reference.get("canonical_key") != expected_key:
        raise ValueError("Final-refit one-shot ledger key is not canonical")
    registry_raw = reference.get("registry_dir")
    path_raw = reference.get("canonical_path")
    if not isinstance(registry_raw, str) or not Path(registry_raw).expanduser().is_absolute():
        raise ValueError("Final-refit one-shot registry path must be absolute")
    if not isinstance(path_raw, str) or not Path(path_raw).expanduser().is_absolute():
        raise ValueError("Final-refit one-shot ledger path must be absolute")
    registry = Path(registry_raw).expanduser().resolve()
    ledger_path = Path(path_raw).expanduser().resolve()
    if ledger_path != (registry / f"validation-{expected_key}.json").resolve():
        raise ValueError("Final-refit one-shot ledger path is not canonical")
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Completed one-shot ledger is missing: {ledger_path}")
    ledger = _strict_json_object(ledger_path, "completed one-shot ledger")
    expected_ledger_keys = {
        "format_version",
        "status",
        "canonical_key",
        "canonical_path",
        "registry_dir",
        "checkpoint_suite_claim",
        "locked_submission_evidence",
        "boundary_name",
        "strict_nw_constraint_mode",
        "checkpoint_sha256",
        "config_sha256",
        "split_manifest_sha256",
        "authority_split_sha256",
        "evaluation_json",
        "evaluation_json_sha256",
        "score_artifact",
        "score_artifact_sha256",
        "split",
        "frozen_test_used",
    }
    if set(ledger) != expected_ledger_keys:
        raise ValueError("Completed one-shot ledger has unexpected fields")
    expected_values = {
        "format_version": ONE_SHOT_LEDGER_FORMAT_VERSION,
        "status": "completed",
        "canonical_key": expected_key,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": evaluation.get("config_sha256"),
        "split_manifest_sha256": split_sha256,
        "authority_split_sha256": authority_sha256,
        "checkpoint_suite_claim": evaluation.get("checkpoint_suite_claim"),
        "locked_submission_evidence": evaluation.get(
            "locked_submission_evidence"
        ),
        "boundary_name": evaluation.get("predeclared_evaluation_boundary"),
        "strict_nw_constraint_mode": dataset.get("strict_nw_constraint_mode"),
        "evaluation_json_sha256": evaluation_sha256,
        "score_artifact_sha256": artifact["sha256"],
        "split": "validation",
        "frozen_test_used": False,
    }
    mismatches = {
        name: {"expected": expected, "observed": ledger.get(name)}
        for name, expected in expected_values.items()
        if ledger.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"Completed one-shot ledger is inconsistent: {mismatches}")
    if Path(str(ledger.get("canonical_path", ""))).expanduser().resolve() != ledger_path:
        raise ValueError("Completed ledger repeats a noncanonical path")
    if Path(str(ledger.get("registry_dir", ""))).expanduser().resolve() != registry:
        raise ValueError("Completed ledger repeats a different registry")
    if Path(str(ledger.get("evaluation_json", ""))).expanduser().resolve() != evaluation_path:
        raise ValueError("Completed ledger points to a different evaluation JSON")
    if Path(str(ledger.get("score_artifact", ""))).expanduser().resolve() != artifact["path"]:
        raise ValueError("Completed ledger points to a different score artifact")
    return {
        "required": True,
        "format_version": ONE_SHOT_LEDGER_FORMAT_VERSION,
        "status": "completed",
        "canonical_key": expected_key,
        "path": str(ledger_path),
        "sha256": sha256_file(ledger_path),
    }


def _exact_hashed_reference(value: Any, label: str) -> dict[str, str]:
    """Authenticate one exact ``{path, sha256}`` reference."""
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = _physical_hashed_file(value["path"], value["sha256"], label)
    return {"path": str(path), "sha256": str(value["sha256"])}


def _verify_final_checkpoint_suite(
    evaluation: Mapping[str, Any],
    evaluation_path: Path,
    evaluation_sha256: str,
    artifact: Mapping[str, Any],
    one_shot_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Physically authenticate locked evidence, claim, and two-boundary receipt."""
    if evaluation.get("checkpoint_run_mode") != "final_refit_fixed_epochs":
        return {"required": False, "mode": "not_a_final_refit_checkpoint"}
    evidence = evaluation.get("locked_submission_evidence")
    expected_evidence_keys = {
        "submission_manifest",
        "protocol_v1_5_execution_lock",
        "protocol_v1_6_execution_hardening",
        "boundary_verification_v2",
        "fixed_evaluation",
        "selected_boundary",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_evidence_keys:
        raise ValueError("Final evaluation lacks exact locked submission evidence")
    authenticated_evidence = {
        name: _exact_hashed_reference(evidence[name], f"locked evidence {name}")
        for name in (
            "submission_manifest",
            "protocol_v1_5_execution_lock",
            "protocol_v1_6_execution_hardening",
            "boundary_verification_v2",
        )
    }
    if evidence.get("fixed_evaluation") != _expected_final_evaluation_contract():
        raise ValueError("Locked fixed_evaluation differs from the approved contract")
    selected_boundary = evaluation.get("predeclared_evaluation_boundary")
    if (
        not isinstance(selected_boundary, str)
        or evidence.get("selected_boundary") != selected_boundary
    ):
        raise ValueError("Locked selected_boundary differs from the evaluation")
    checkpoint_path = _physical_hashed_file(
        evaluation.get("checkpoint"),
        evaluation.get("checkpoint_sha256"),
        "final checkpoint",
    )
    claim_reference = _exact_hashed_reference(
        evaluation.get("checkpoint_suite_claim"), "checkpoint-suite claim"
    )
    claim_path = Path(claim_reference["path"])
    claim = _strict_json_object(claim_path, "checkpoint-suite claim")
    expected_claim_keys = {
        "format_version",
        "stage",
        "intended_use",
        "status",
        "submission_manifest",
        "checkpoint",
        "one_shot_ledger_registry_dir",
        "boundaries",
    }
    if not isinstance(claim, Mapping) or set(claim) != expected_claim_keys:
        raise ValueError("Checkpoint-suite claim has unexpected fields")
    registry_raw = claim.get("one_shot_ledger_registry_dir")
    if not isinstance(registry_raw, str) or not Path(registry_raw).expanduser().is_absolute():
        raise ValueError("Checkpoint-suite claim registry must be absolute")
    registry = Path(registry_raw).expanduser().resolve()
    expected_claim_path = (
        registry
        / "checkpoint-suite-claims"
        / f"{authenticated_evidence['submission_manifest']['sha256']}.json"
    ).resolve()
    claim_boundaries = claim.get("boundaries")
    if (
        claim.get("format_version") != CHECKPOINT_SUITE_CLAIM_FORMAT_VERSION
        or claim.get("stage") != "final_checkpoint_suite_claim"
        or claim.get("intended_use")
        != "two_boundary_one_shot_development_confirmation"
        or claim.get("status") != "immutable"
        or claim.get("submission_manifest")
        != authenticated_evidence["submission_manifest"]
        or claim.get("checkpoint")
        != {"path": str(checkpoint_path), "sha256": evaluation["checkpoint_sha256"]}
        or claim_path != expected_claim_path
        or not isinstance(claim_boundaries, Mapping)
        or len(claim_boundaries) != 2
    ):
        raise ValueError("Checkpoint-suite claim is not canonical")
    for name, boundary in claim_boundaries.items():
        if not isinstance(name, str) or not isinstance(boundary, Mapping) or set(
            boundary
        ) != {"split_manifest_sha256", "strict_nw_constraint_mode"}:
            raise ValueError("Checkpoint-suite claim boundary is malformed")
        if (
            not _is_sha256(str(boundary.get("split_manifest_sha256", "")))
            or boundary.get("strict_nw_constraint_mode")
            not in STRICT_NW_CONSTRAINT_MODES
        ):
            raise ValueError("Checkpoint-suite claim boundary is invalid")
    selected_claim_boundary = claim_boundaries.get(selected_boundary)
    if (
        not isinstance(selected_claim_boundary, Mapping)
        or selected_claim_boundary.get("split_manifest_sha256")
        != evaluation["dataset"]["split_manifest_sha256"]
        or selected_claim_boundary.get("strict_nw_constraint_mode")
        != evaluation["dataset"]["strict_nw_constraint_mode"]
    ):
        raise ValueError("Checkpoint-suite claim does not bind the selected boundary")
    completion_path = (
        registry
        / "checkpoint-suite-completions"
        / f"{authenticated_evidence['submission_manifest']['sha256']}.json"
    ).resolve()
    completion = _strict_json_object(
        completion_path, "checkpoint-suite completion"
    )
    expected_completion_keys = {
        "format_version",
        "stage",
        "intended_use",
        "status",
        "checkpoint_suite_claim",
        "submission_manifest",
        "checkpoint",
        "one_shot_ledger_registry_dir",
        "boundaries",
    }
    if set(completion) != expected_completion_keys:
        raise ValueError("Checkpoint-suite completion has unexpected fields")
    completion_boundaries = completion.get("boundaries")
    if (
        completion.get("format_version")
        != CHECKPOINT_SUITE_COMPLETION_FORMAT_VERSION
        or completion.get("stage") != "final_checkpoint_suite_completion"
        or completion.get("intended_use")
        != "two_boundary_one_shot_development_confirmation"
        or completion.get("status") != "completed"
        or completion.get("checkpoint_suite_claim") != claim_reference
        or completion.get("submission_manifest")
        != authenticated_evidence["submission_manifest"]
        or completion.get("checkpoint") != claim["checkpoint"]
        or completion.get("one_shot_ledger_registry_dir") != str(registry)
        or not isinstance(completion_boundaries, Mapping)
        or set(completion_boundaries) != set(claim_boundaries)
    ):
        raise ValueError("Checkpoint-suite completion is inconsistent")
    authenticated_boundaries: dict[str, Any] = {}
    for name, boundary in completion_boundaries.items():
        if not isinstance(boundary, Mapping) or set(boundary) != {
            "split_manifest_sha256",
            "strict_nw_constraint_mode",
            "one_shot_ledger",
            "evaluation_json",
            "score_artifact",
        }:
            raise ValueError("Checkpoint-suite completion boundary is malformed")
        if (
            boundary.get("split_manifest_sha256")
            != claim_boundaries[name]["split_manifest_sha256"]
            or boundary.get("strict_nw_constraint_mode")
            != claim_boundaries[name]["strict_nw_constraint_mode"]
        ):
            raise ValueError("Checkpoint-suite completion boundary differs from its claim")
        authenticated_boundaries[name] = {
            field: _exact_hashed_reference(
                boundary[field], f"checkpoint-suite completion {name}.{field}"
            )
            for field in ("one_shot_ledger", "evaluation_json", "score_artifact")
        }
    selected = authenticated_boundaries[selected_boundary]
    if (
        selected["evaluation_json"]
        != {"path": str(evaluation_path), "sha256": evaluation_sha256}
        or selected["score_artifact"]
        != {"path": str(artifact["path"]), "sha256": artifact["sha256"]}
        or selected["one_shot_ledger"].get("path")
        != one_shot_verification.get("path")
        or selected["one_shot_ledger"].get("sha256")
        != one_shot_verification.get("sha256")
    ):
        raise ValueError("Checkpoint-suite completion does not bind analyzed artifacts")
    return {
        "required": True,
        "status": "completed",
        "locked_submission_evidence": {
            **authenticated_evidence,
            "fixed_evaluation_sha256": canonical_sha256(
                evidence["fixed_evaluation"]
            ),
            "selected_boundary": selected_boundary,
        },
        "checkpoint_suite_claim": claim_reference,
        "checkpoint_suite_completion": {
            "path": str(completion_path),
            "sha256": sha256_file(completion_path),
        },
        "boundaries": authenticated_boundaries,
    }


def _verify_reported_metrics(
    evaluation: Mapping[str, Any],
    similarities: np.ndarray,
    positive_mask: np.ndarray,
    antibody_keys: Sequence[str],
    antigen_keys: Sequence[str],
) -> None:
    """Recompute every rank-based value needed for statistical analysis."""
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Evaluation JSON lacks metrics")
    libraries = metrics.get("library_keys")
    if not isinstance(libraries, Mapping):
        raise ValueError("Evaluation JSON lacks ordered library keys")
    if libraries.get("antibody") != list(antibody_keys) or libraries.get(
        "antigen"
    ) != list(antigen_keys):
        raise ValueError("Evaluation and score-artifact library keys differ")
    directional_inputs = {
        "antibody_to_antigen": (
            similarities,
            positive_mask,
            antibody_keys,
            antigen_keys,
        ),
        "antigen_to_antibody": (
            similarities.T,
            positive_mask.T,
            antigen_keys,
            antibody_keys,
        ),
    }
    observed_mrrs = []
    for direction, (scores, mask, anchors, candidates) in directional_inputs.items():
        ranks = _best_positive_ranks(_stable_rank_matrix(scores), mask)
        point = _point_metrics(_metric_vectors(ranks))
        reported = metrics.get(direction)
        if not isinstance(reported, Mapping):
            raise ValueError(f"Evaluation JSON lacks {direction} metrics")
        for metric, expected in point.items():
            if not math.isclose(
                _finite_float(reported.get(metric), f"reported {direction} {metric}"),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"Reported {direction} {metric} does not match scores")
        per_anchor = metrics.get("per_anchor_ranks", {}).get(direction)
        if not isinstance(per_anchor, list) or len(per_anchor) != len(anchors):
            raise ValueError(f"Reported {direction} per-anchor ranks are incomplete")
        for index, (anchor_key, rank) in enumerate(zip(anchors, ranks)):
            record = per_anchor[index]
            positive_indices = np.flatnonzero(mask[index])
            positive_keys = [candidates[int(candidate)] for candidate in positive_indices]
            if (
                record.get("anchor_key") != anchor_key
                or int(record.get("best_known_positive_rank", 0)) != int(rank)
                or record.get("known_positive_candidate_keys") != positive_keys
                or int(record.get("known_positive_candidate_count", 0))
                != len(positive_keys)
            ):
                raise ValueError(f"Reported {direction} per-anchor record is inconsistent")
        observed_mrrs.append(point["mrr"])
    expected_bidirectional = sum(observed_mrrs) / 2.0
    if not math.isclose(
        _finite_float(metrics.get("bidirectional_mrr"), "bidirectional MRR"),
        expected_bidirectional,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Reported bidirectional MRR does not match score matrices")
    if int(metrics.get("exact_pair_count", -1)) != int(positive_mask.sum()):
        raise ValueError("Reported exact-pair count does not match the positive mask")


def verify_evaluation_and_artifact(
    evaluation_path: Path,
    score_artifact_path: Path,
    expected_evaluation_sha256: str,
    expected_score_artifact_sha256: str,
) -> dict[str, Any]:
    """Authenticate a development evaluation and its bound score artifact."""
    if not _is_sha256(expected_evaluation_sha256) or not _is_sha256(
        expected_score_artifact_sha256
    ):
        raise ValueError("Exact evaluation and score-artifact SHA-256 values are required")
    evaluation_path = evaluation_path.expanduser().resolve()
    evaluation_bytes = _read_file_bytes_once(evaluation_path, "evaluation JSON")
    observed_evaluation_sha256 = hashlib.sha256(evaluation_bytes).hexdigest()
    if observed_evaluation_sha256 != expected_evaluation_sha256:
        raise ValueError("Evaluation JSON SHA-256 mismatch")
    evaluation = _decode_strict_json_bytes(evaluation_bytes, "evaluation JSON")
    if evaluation.get("format_version") != FORMAT_VERSION:
        raise ValueError("Evaluation format is incompatible")
    if evaluation.get("stage") != "full_library_development_validation":
        raise PermissionError("Only full-library development validation is authorized")
    if evaluation.get("split") != "validation" or evaluation.get("frozen_test_used") is not False:
        raise PermissionError("Frozen or non-validation evaluation is forbidden")
    evidence_scope = _classify_evaluation_scope(evaluation)
    dataset = evaluation.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("Evaluation JSON lacks a dataset contract")
    if dataset.get("loaded_splits") != ["train", "validation"]:
        raise PermissionError("Evaluation materialized an unauthorized split")
    if dataset.get("frozen_test_records_selected") is not False:
        raise PermissionError("Evaluation selected frozen-test rows")
    if not _is_sha256(str(dataset.get("split_manifest_sha256", ""))):
        raise ValueError("Evaluation lacks an exact split-manifest hash")
    authority = dataset.get("authority")
    if not isinstance(authority, Mapping) or not _is_sha256(str(authority.get("sha256", ""))):
        raise ValueError("Evaluation lacks an exact authority-split hash")
    if not authority.get("sealed_source_splits"):
        raise PermissionError("Evaluation did not declare a sealed authority split")
    if int(authority.get("selected_sealed_overlap_count", -1)) != 0:
        raise PermissionError("Evaluation overlaps a sealed authority split")
    strict_mode = dataset.get("strict_nw_constraint_mode")
    if evidence_scope == "reused_development_confirmation":
        if (
            dataset.get("split_format_version")
            != AUTHORITY_VALIDATION_BOUNDARY_FORMAT
            or dataset.get("split_intended_use")
            != AUTHORITY_VALIDATION_BOUNDARY_INTENDED_USE
            or strict_mode not in STRICT_NW_CONSTRAINT_MODES
        ):
            raise PermissionError(
                "Final-refit/control evidence requires the approved v2 format/use pair"
            )
    else:
        if dataset.get("split_intended_use") != "development_only":
            raise PermissionError("Exploratory evaluation split is not development-only")
        if dataset.get("split_format_version") not in EXPLORATORY_SPLIT_FORMATS:
            raise PermissionError("Exploratory evaluation split format is not approved")
        if (
            dataset.get("split_format_version")
            == "sabdab2_iterative_nw_strict_development_split_v1"
        ):
            if strict_mode not in STRICT_NW_CONSTRAINT_MODES:
                raise PermissionError("Strict NW evaluation lacks an explicit constraint mode")
        elif strict_mode is not None:
            raise ValueError("A non-strict split declares a strict NW constraint mode")
    artifact = load_score_artifact(score_artifact_path)
    if artifact["sha256"] != expected_score_artifact_sha256:
        raise ValueError("Score artifact SHA-256 mismatch")
    reference = evaluation.get("score_artifact")
    if not isinstance(reference, Mapping):
        raise ValueError("Evaluation JSON is not bound to a score artifact")
    if reference.get("format_version") != SCORE_ARTIFACT_FORMAT_VERSION:
        raise ValueError("Evaluation score-artifact reference has the wrong format")
    if reference.get("sha256") != artifact["sha256"]:
        raise ValueError("Evaluation JSON references a different score artifact")
    metadata = artifact["metadata"]
    if reference.get("contract_sha256") != metadata["contract_sha256"]:
        raise ValueError("Evaluation and artifact contract hashes differ")
    if reference.get("array_sha256") != artifact["array_sha256"]:
        raise ValueError("Evaluation and artifact array hashes differ")
    if reference.get("matrix_shape") != list(artifact["similarities"].shape):
        raise ValueError("Evaluation and artifact matrix shapes differ")
    if reference.get("contains_raw_sequences") is not False:
        raise PermissionError("Score-artifact sequence-free declaration is missing")
    expected_context = _expected_evaluation_context(evaluation)
    if metadata["contract"].get("evaluation_context") != expected_context:
        raise ValueError("Score artifact was produced for a different evaluation context")
    if evidence_scope == "reused_development_confirmation":
        boundary_verification = _validate_v2_boundary_and_library(
            dataset,
            artifact["antibody_keys"],
            artifact["antigen_keys"],
            artifact["positive_mask"],
        )
        one_shot_verification = _verify_final_refit_one_shot_ledger(
            evaluation,
            evaluation_path,
            observed_evaluation_sha256,
            artifact,
        )
        checkpoint_suite_verification = _verify_final_checkpoint_suite(
            evaluation,
            evaluation_path,
            observed_evaluation_sha256,
            artifact,
            one_shot_verification,
        )
    else:
        boundary_verification = {
            "format_version": dataset["split_format_version"],
            "intended_use": dataset["split_intended_use"],
            "scope": "explicit_non_one_shot_exploratory_compatibility",
        }
        one_shot_verification = {
            "required": False,
            "mode": "explicit_non_one_shot_exploratory_compatibility",
        }
        checkpoint_suite_verification = {
            "required": False,
            "mode": "explicit_non_one_shot_exploratory_compatibility",
        }
    _verify_reported_metrics(
        evaluation,
        artifact["similarities"],
        artifact["positive_mask"],
        artifact["antibody_keys"],
        artifact["antigen_keys"],
    )
    return {
        "evaluation": evaluation,
        "evaluation_path": evaluation_path,
        "evaluation_sha256": observed_evaluation_sha256,
        "artifact": artifact,
        "evidence_scope": evidence_scope,
        "boundary_verification": boundary_verification,
        "one_shot_verification": one_shot_verification,
        "checkpoint_suite_verification": checkpoint_suite_verification,
    }


def _permuted_positive_ranks(
    antibody_rank_matrix: np.ndarray,
    antigen_rank_matrix: np.ndarray,
    positive_mask: np.ndarray,
    antibody_permutation: np.ndarray,
    antigen_permutation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel both score axes independently while preserving graph degrees."""
    mask = np.asarray(positive_mask, dtype=bool)
    antibody_order = np.asarray(antibody_permutation, dtype=np.int64)
    antigen_order = np.asarray(antigen_permutation, dtype=np.int64)
    if not np.array_equal(np.sort(antibody_order), np.arange(mask.shape[0])):
        raise ValueError("Antibody label permutation is invalid")
    if not np.array_equal(np.sort(antigen_order), np.arange(mask.shape[1])):
        raise ValueError("Antigen label permutation is invalid")
    antibody_ranks = np.empty(mask.shape[0], dtype=np.int64)
    for anchor, positive_labels in enumerate(mask):
        antibody_ranks[anchor] = antibody_rank_matrix[
            antibody_order[anchor], antigen_order[positive_labels]
        ].min()
    antigen_ranks = np.empty(mask.shape[1], dtype=np.int64)
    for anchor in range(mask.shape[1]):
        antigen_ranks[anchor] = antigen_rank_matrix[
            antigen_order[anchor], antibody_order[mask[:, anchor]]
        ].min()
    return antibody_ranks, antigen_ranks


def _null_summary(values: np.ndarray, observed: float) -> dict[str, Any]:
    null = np.asarray(values, dtype=np.float64)
    if null.ndim != 1 or null.size == 0 or not np.isfinite(null).all():
        raise ValueError("Permutation null must be a finite non-empty vector")
    exceedances = int(np.count_nonzero(null >= observed - 1e-15))
    return {
        "observed": _finite_float(observed, "observed metric"),
        "null_mean": _finite_float(null.mean(), "null mean"),
        "null_standard_deviation": _finite_float(
            null.std(ddof=1) if null.size > 1 else 0.0, "null standard deviation"
        ),
        "null_percentile_1": _finite_float(np.quantile(null, 0.01), "null percentile"),
        "null_percentile_50": _finite_float(np.quantile(null, 0.50), "null percentile"),
        "null_percentile_95": _finite_float(np.quantile(null, 0.95), "null percentile"),
        "null_percentile_99": _finite_float(np.quantile(null, 0.99), "null percentile"),
        "upper_tail_exceedance_count": exceedances,
        "p_value_plus_one": (exceedances + 1.0) / (null.size + 1.0),
    }


def fixed_degree_label_permutation_test(
    similarities: np.ndarray,
    positive_mask: np.ndarray,
    permutation_count: int,
    seed: int,
) -> dict[str, Any]:
    """Run independent two-axis fixed-degree relabeling against fixed scores."""
    scores = np.asarray(similarities)
    mask = np.asarray(positive_mask, dtype=bool)
    if scores.ndim != 2 or scores.shape != mask.shape or not np.isfinite(scores).all():
        raise ValueError("Similarity and positive matrices must be finite and aligned")
    if permutation_count <= 0 or not mask.any(axis=1).all() or not mask.any(axis=0).all():
        raise ValueError("Permutation settings or positive graph are invalid")
    antibody_rank_matrix = _stable_rank_matrix(scores)
    antigen_rank_matrix = _stable_rank_matrix(scores.T)
    observed_vectors = {
        "antibody_to_antigen": _metric_vectors(
            _best_positive_ranks(antibody_rank_matrix, mask)
        ),
        "antigen_to_antibody": _metric_vectors(
            _best_positive_ranks(antigen_rank_matrix, mask.T)
        ),
    }
    observed = {
        direction: _point_metrics(vectors)
        for direction, vectors in observed_vectors.items()
    }
    observed["bidirectional"] = {
        metric: 0.5
        * (
            observed["antibody_to_antigen"][metric]
            + observed["antigen_to_antibody"][metric]
        )
        for metric in METRIC_NAMES
    }
    null = {
        direction: {
            metric: np.empty(permutation_count, dtype=np.float64)
            for metric in METRIC_NAMES
        }
        for direction in ALL_DIRECTIONS
    }
    per_anchor_null_sums = {
        direction: {
            metric: np.zeros_like(vector, dtype=np.float64)
            for metric, vector in observed_vectors[direction].items()
        }
        for direction in DIRECTIONS
    }
    generator = np.random.default_rng(seed)
    for permutation_index in range(permutation_count):
        antibody_permutation = generator.permutation(mask.shape[0])
        antigen_permutation = generator.permutation(mask.shape[1])
        antibody_ranks, antigen_ranks = _permuted_positive_ranks(
            antibody_rank_matrix,
            antigen_rank_matrix,
            mask,
            antibody_permutation,
            antigen_permutation,
        )
        permuted_vectors = {
            "antibody_to_antigen": _metric_vectors(antibody_ranks),
            "antigen_to_antibody": _metric_vectors(antigen_ranks),
        }
        for direction in DIRECTIONS:
            for metric in METRIC_NAMES:
                vector = permuted_vectors[direction][metric]
                per_anchor_null_sums[direction][metric] += vector
                null[direction][metric][permutation_index] = vector.mean()
        for metric in METRIC_NAMES:
            null["bidirectional"][metric][permutation_index] = 0.5 * (
                null["antibody_to_antigen"][metric][permutation_index]
                + null["antigen_to_antibody"][metric][permutation_index]
            )
    mean_null_vectors = {
        direction: {
            metric: vector / permutation_count
            for metric, vector in per_anchor_null_sums[direction].items()
        }
        for direction in DIRECTIONS
    }
    return {
        "protocol": "independent_two_axis_embedding_label_permutation_against_fixed_positive_graph",
        "degree_constraint": "The positive graph is fixed, preserving every antibody and antigen degree.",
        "alternative": "trained_metric_is_greater_than_or_equal_to_the_relabeling_null",
        "p_value_estimator": "(1 + upper-tail exceedances) / (1 + permutations)",
        "permutation_count": int(permutation_count),
        "seed": int(seed),
        "minimum_attainable_p_value": 1.0 / (permutation_count + 1.0),
        "results": {
            direction: {
                metric: _null_summary(null[direction][metric], observed[direction][metric])
                for metric in METRIC_NAMES
            }
            for direction in ALL_DIRECTIONS
        },
        "observed_vectors": observed_vectors,
        "mean_null_vectors": mean_null_vectors,
    }


def _bootstrap_direction(
    effect_vectors: Mapping[str, np.ndarray],
    bootstrap_count: int,
    alpha: float,
    generator: np.random.Generator,
    cluster_ids: Sequence[str] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Bootstrap paired per-anchor effects by anchors or complete clusters."""
    anchor_count = len(next(iter(effect_vectors.values())))
    if any(len(vector) != anchor_count for vector in effect_vectors.values()):
        raise ValueError("Paired effect vectors have inconsistent lengths")
    if cluster_ids is None:
        indices = generator.integers(
            0, anchor_count, size=(bootstrap_count, anchor_count), endpoint=False
        )
        distributions = {
            metric: np.asarray(vector, dtype=np.float64)[indices].mean(axis=1)
            for metric, vector in effect_vectors.items()
        }
        resampling_unit_count = anchor_count
    else:
        labels = np.asarray([str(value) for value in cluster_ids])
        if labels.shape != (anchor_count,) or any(not value for value in labels):
            raise ValueError("Cluster labels must cover every anchor exactly once")
        unique_labels = sorted(set(labels.tolist()))
        cluster_indices = [np.flatnonzero(labels == label) for label in unique_labels]
        cluster_counts = np.asarray([len(indices) for indices in cluster_indices], dtype=np.float64)
        draws = generator.integers(
            0,
            len(cluster_indices),
            size=(bootstrap_count, len(cluster_indices)),
            endpoint=False,
        )
        denominators = cluster_counts[draws].sum(axis=1)
        distributions = {}
        for metric, vector in effect_vectors.items():
            values = np.asarray(vector, dtype=np.float64)
            cluster_sums = np.asarray([values[indices].sum() for indices in cluster_indices])
            distributions[metric] = cluster_sums[draws].sum(axis=1) / denominators
        resampling_unit_count = len(cluster_indices)
    intervals = {}
    for metric, vector in effect_vectors.items():
        distribution = distributions[metric]
        intervals[metric] = {
            "estimate": _finite_float(np.mean(vector), "paired effect estimate"),
            "lower": _finite_float(np.quantile(distribution, alpha / 2.0), "CI lower"),
            "upper": _finite_float(
                np.quantile(distribution, 1.0 - alpha / 2.0), "CI upper"
            ),
            "bootstrap_standard_error": _finite_float(
                distribution.std(ddof=1) if bootstrap_count > 1 else 0.0,
                "bootstrap standard error",
            ),
        }
    return intervals, distributions


def paired_effect_bootstrap(
    directional_effect_vectors: Mapping[str, Mapping[str, np.ndarray]],
    bootstrap_count: int,
    seed: int,
    alpha: float,
    cluster_ids: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Return directional and bidirectional percentile intervals for paired effects."""
    if bootstrap_count <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("Bootstrap settings are invalid")
    seed_sequence = np.random.SeedSequence(seed)
    direction_seeds = seed_sequence.spawn(2)
    intervals: dict[str, Any] = {}
    distributions: dict[str, dict[str, np.ndarray]] = {}
    for direction, child_seed in zip(DIRECTIONS, direction_seeds):
        direction_clusters = None if cluster_ids is None else cluster_ids[direction]
        intervals[direction], distributions[direction] = _bootstrap_direction(
            directional_effect_vectors[direction],
            bootstrap_count,
            alpha,
            np.random.default_rng(child_seed),
            direction_clusters,
        )
    intervals["bidirectional"] = {}
    for metric in METRIC_NAMES:
        distribution = 0.5 * (
            distributions["antibody_to_antigen"][metric]
            + distributions["antigen_to_antibody"][metric]
        )
        estimate = 0.5 * (
            intervals["antibody_to_antigen"][metric]["estimate"]
            + intervals["antigen_to_antibody"][metric]["estimate"]
        )
        intervals["bidirectional"][metric] = {
            "estimate": estimate,
            "lower": _finite_float(np.quantile(distribution, alpha / 2.0), "CI lower"),
            "upper": _finite_float(
                np.quantile(distribution, 1.0 - alpha / 2.0), "CI upper"
            ),
            "bootstrap_standard_error": _finite_float(
                distribution.std(ddof=1) if bootstrap_count > 1 else 0.0,
                "bootstrap standard error",
            ),
        }
    return {
        "protocol": (
            "paired_cluster_bootstrap" if cluster_ids is not None else "paired_anchor_bootstrap"
        ),
        "directional_resampling": "independent_within_each_retrieval_direction",
        "interval": "two-sided_percentile",
        "confidence_level": 1.0 - alpha,
        "bootstrap_count": int(bootstrap_count),
        "seed": int(seed),
        **intervals,
    }


def _load_cluster_ids(
    path: Path | None,
    expected_sha256: str | None,
    evaluation_sha256: str,
    artifact_sha256: str,
    antibody_keys: Sequence[str],
    antigen_keys: Sequence[str],
) -> tuple[dict[str, list[str]] | None, dict[str, Any]]:
    if path is None and expected_sha256 is None:
        return None, {
            "available": False,
            "reason": "No independently defined anchor-cluster artifact was supplied.",
        }
    if path is None or expected_sha256 is None or not _is_sha256(expected_sha256):
        raise ValueError("Cluster path and exact SHA-256 must be supplied together")
    resolved = path.expanduser().resolve()
    observed = sha256_file(resolved)
    if observed != expected_sha256:
        raise ValueError("Anchor-cluster artifact SHA-256 mismatch")
    payload = _strict_json_object(resolved, "anchor-cluster artifact")
    if payload.get("format_version") != CLUSTER_FORMAT_VERSION:
        raise ValueError("Anchor-cluster artifact format is incompatible")
    if payload.get("intended_use") != "development_statistics_only":
        raise PermissionError("Anchor clusters are not development-statistics-only")
    if payload.get("evaluation_json_sha256") != evaluation_sha256 or payload.get(
        "score_artifact_sha256"
    ) != artifact_sha256:
        raise ValueError("Anchor clusters are bound to different score inputs")
    direction_keys = {
        "antibody_to_antigen": list(antibody_keys),
        "antigen_to_antibody": list(antigen_keys),
    }
    raw = payload.get("directions")
    if not isinstance(raw, Mapping):
        raise ValueError("Anchor-cluster artifact lacks direction mappings")
    clusters: dict[str, list[str]] = {}
    counts = {}
    for direction, keys in direction_keys.items():
        mapping = raw.get(direction)
        if not isinstance(mapping, Mapping) or set(mapping) != set(keys):
            raise ValueError(f"Anchor clusters do not exactly cover {direction} keys")
        clusters[direction] = [str(mapping[key]) for key in keys]
        if any(not value for value in clusters[direction]):
            raise ValueError("Anchor cluster identifiers cannot be empty")
        counts[direction] = len(set(clusters[direction]))
    return clusters, {
        "available": True,
        "path": str(resolved),
        "sha256": observed,
        "cluster_counts": counts,
    }


def joint_bipartite_component_bootstrap(
    directional_effect_vectors: Mapping[str, Mapping[str, np.ndarray]],
    positive_mask: np.ndarray,
    bootstrap_count: int,
    seed: int,
    alpha: float,
) -> dict[str, Any]:
    """Jointly resample connected positive-graph components across both directions."""
    mask = np.asarray(positive_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any(axis=1).all() or not mask.any(axis=0).all():
        raise ValueError("Joint bootstrap requires a complete bipartite anchor graph")
    if bootstrap_count <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("Joint bootstrap settings are invalid")
    node_count = mask.shape[0] + mask.shape[1]
    parent = np.arange(node_count, dtype=np.int64)

    def find(node: int) -> int:
        while int(parent[node]) != node:
            parent[node] = parent[int(parent[node])]
            node = int(parent[node])
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for antibody, antigen in np.argwhere(mask):
        union(int(antibody), mask.shape[0] + int(antigen))
    components: dict[int, dict[str, list[int]]] = {}
    for antibody in range(mask.shape[0]):
        components.setdefault(find(antibody), {"antibody": [], "antigen": []})[
            "antibody"
        ].append(antibody)
    for antigen in range(mask.shape[1]):
        components.setdefault(
            find(mask.shape[0] + antigen), {"antibody": [], "antigen": []}
        )["antigen"].append(antigen)
    ordered = [
        components[key]
        for key in sorted(
            components,
            key=lambda value: (
                min(components[value]["antibody"] or [mask.shape[0]]),
                min(components[value]["antigen"] or [mask.shape[1]]),
            ),
        )
    ]
    if any(not component["antibody"] or not component["antigen"] for component in ordered):
        raise AssertionError("Every positive-graph component must contain both node types")
    if len(ordered) < 2:
        return {
            "available": False,
            "reason": "The positive graph has fewer than two connected components.",
            "component_count": len(ordered),
        }
    generator = np.random.default_rng(seed)
    draws = generator.integers(
        0,
        len(ordered),
        size=(bootstrap_count, len(ordered)),
        endpoint=False,
    )
    distributions: dict[str, dict[str, np.ndarray]] = {
        direction: {} for direction in ALL_DIRECTIONS
    }
    intervals: dict[str, dict[str, Any]] = {
        direction: {} for direction in ALL_DIRECTIONS
    }
    for metric in METRIC_NAMES:
        directional_distributions = {}
        for direction, node_type in (
            ("antibody_to_antigen", "antibody"),
            ("antigen_to_antibody", "antigen"),
        ):
            vector = np.asarray(directional_effect_vectors[direction][metric], dtype=np.float64)
            expected_length = mask.shape[0] if node_type == "antibody" else mask.shape[1]
            if vector.shape != (expected_length,):
                raise ValueError("Joint-bootstrap effect vectors do not align with graph nodes")
            counts = np.asarray(
                [len(component[node_type]) for component in ordered], dtype=np.float64
            )
            sums = np.asarray(
                [vector[component[node_type]].sum() for component in ordered],
                dtype=np.float64,
            )
            distribution = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            directional_distributions[direction] = distribution
            distributions[direction][metric] = distribution
        distributions["bidirectional"][metric] = 0.5 * (
            directional_distributions["antibody_to_antigen"]
            + directional_distributions["antigen_to_antibody"]
        )
    for direction in ALL_DIRECTIONS:
        for metric in METRIC_NAMES:
            if direction == "bidirectional":
                estimate = 0.5 * (
                    np.mean(directional_effect_vectors["antibody_to_antigen"][metric])
                    + np.mean(directional_effect_vectors["antigen_to_antibody"][metric])
                )
            else:
                estimate = np.mean(directional_effect_vectors[direction][metric])
            distribution = distributions[direction][metric]
            intervals[direction][metric] = {
                "estimate": _finite_float(estimate, "joint effect estimate"),
                "lower": _finite_float(
                    np.quantile(distribution, alpha / 2.0), "joint CI lower"
                ),
                "upper": _finite_float(
                    np.quantile(distribution, 1.0 - alpha / 2.0), "joint CI upper"
                ),
                "bootstrap_standard_error": _finite_float(
                    distribution.std(ddof=1) if bootstrap_count > 1 else 0.0,
                    "joint bootstrap standard error",
                ),
            }
    return {
        "available": True,
        "protocol": "joint_connected_component_bootstrap_of_the_positive_bipartite_graph",
        "component_count": len(ordered),
        "component_sizes": [
            {
                "antibody_count": len(component["antibody"]),
                "antigen_count": len(component["antigen"]),
            }
            for component in ordered
        ],
        "interval": "two-sided_percentile",
        "confidence_level": 1.0 - alpha,
        "bootstrap_count": bootstrap_count,
        "seed": seed,
        **intervals,
    }


def candidate_size_curves(
    similarities: np.ndarray,
    positive_mask: np.ndarray,
    candidate_sizes: Sequence[int],
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate all-positive-retaining retrieval curves over sampled negatives."""
    if sample_count <= 0:
        raise ValueError("Candidate-curve sample count must be positive")
    sizes = sorted(set(int(value) for value in candidate_sizes))
    if not sizes or sizes[0] <= 0:
        raise ValueError("Candidate sizes must be positive")
    directional_data = {
        "antibody_to_antigen": (np.asarray(similarities), np.asarray(positive_mask, dtype=bool)),
        "antigen_to_antibody": (
            np.asarray(similarities).T,
            np.asarray(positive_mask, dtype=bool).T,
        ),
    }
    seeds = np.random.SeedSequence(seed).spawn(2)
    output: dict[str, Any] = {
        "protocol": "retain_all_known_positives_and_uniformly_sample_negatives_per_anchor",
        "scientific_role": "secondary_candidate_library_size_sensitivity_only",
        "requested_candidate_sizes": sizes,
        "oversized_size_policy": "omit_per_direction",
        "sample_count_per_nonfull_size": int(sample_count),
        "seed": int(seed),
    }
    for (direction, (scores, mask)), child_seed in zip(directional_data.items(), seeds):
        generator = np.random.default_rng(child_seed)
        rank_matrix = _stable_rank_matrix(scores)
        degrees = mask.sum(axis=1).astype(int)
        direction_output: dict[str, Any] = {}
        omitted_sizes: list[int] = []
        for requested_size in sizes:
            if requested_size > scores.shape[1]:
                omitted_sizes.append(requested_size)
                continue
            if requested_size < int(degrees.max()):
                direction_output[str(requested_size)] = {
                    "available": False,
                    "reason": "Requested size cannot retain every positive for every anchor.",
                }
                continue
            repetitions = 1 if requested_size == scores.shape[1] else sample_count
            distributions = {
                metric: np.empty(repetitions, dtype=np.float64) for metric in METRIC_NAMES
            }
            for repetition in range(repetitions):
                sampled_ranks = np.empty(scores.shape[0], dtype=np.int64)
                for anchor in range(scores.shape[0]):
                    positives = np.flatnonzero(mask[anchor])
                    negatives = np.flatnonzero(~mask[anchor])
                    required_negatives = requested_size - len(positives)
                    selected_negatives = (
                        negatives
                        if required_negatives == len(negatives)
                        else generator.choice(negatives, size=required_negatives, replace=False)
                    )
                    best_positive_rank = rank_matrix[anchor, positives].min()
                    sampled_ranks[anchor] = 1 + int(
                        np.count_nonzero(
                            rank_matrix[anchor, selected_negatives] < best_positive_rank
                        )
                    )
                vectors = _metric_vectors(sampled_ranks)
                for metric in METRIC_NAMES:
                    distributions[metric][repetition] = vectors[metric].mean()
            chance = exact_random_direction(requested_size, degrees.tolist())
            direction_output[str(requested_size)] = {
                "available": True,
                "candidate_count": requested_size,
                "replicate_count": repetitions,
                "model": {
                    metric: {
                        "mean": _finite_float(values.mean(), "curve mean"),
                        "percentile_2_5": _finite_float(
                            np.quantile(values, 0.025), "curve percentile"
                        ),
                        "percentile_97_5": _finite_float(
                            np.quantile(values, 0.975), "curve percentile"
                        ),
                    }
                    for metric, values in distributions.items()
                },
                "exact_random": {metric: chance[metric] for metric in METRIC_NAMES},
            }
        output[direction] = {
            "native_candidate_count": int(scores.shape[1]),
            "omitted_sizes_larger_than_native_library": omitted_sizes,
            "curves": direction_output,
        }
    return output


def evaluate_preregistered_gates(
    observed: Mapping[str, Mapping[str, float]],
    exact_random: Mapping[str, Any],
    permutation: Mapping[str, Any],
    partner_effect: Mapping[str, Any],
    partner_effect_interval_source: str = "paired_anchor_bootstrap",
) -> dict[str, Any]:
    """Evaluate every preregistered gate available from one model score artifact."""
    checks: list[dict[str, Any]] = []
    directional_mrr_pass = all(
        observed[direction]["mrr"]
        > permutation["results"][direction]["mrr"]["null_percentile_99"]
        for direction in DIRECTIONS
    )
    checks.append(
        {
            "gate": "both_directional_mrr_above_permutation_99th_percentile",
            "status": "passed" if directional_mrr_pass else "failed",
        }
    )
    bidirectional_mrr = observed["bidirectional"]["mrr"]
    bidirectional_chance = exact_random["bidirectional"]["mrr"]
    bidirectional_ratio = bidirectional_mrr / bidirectional_chance
    checks.append(
        {
            "gate": "bidirectional_mrr_at_least_two_times_exact_random",
            "observed_to_random_ratio": bidirectional_ratio,
            "status": "passed" if bidirectional_ratio >= 2.0 else "failed",
        }
    )
    recall_ratios = {
        direction: observed[direction]["recall_at_1"]
        / exact_random[direction]["recall_at_1"]
        for direction in DIRECTIONS
    }
    checks.append(
        {
            "gate": "both_directional_recall_at_1_at_least_two_times_exact_random",
            "observed_to_random_ratios": recall_ratios,
            "status": "passed"
            if all(value >= 2.0 for value in recall_ratios.values())
            else "failed",
        }
    )
    drop = partner_effect["bidirectional"]["mrr"]["estimate"]
    excess = bidirectional_mrr - bidirectional_chance
    removal_fraction = drop / excess if excess > 0.0 else None
    partner_pass = (
        removal_fraction is not None
        and removal_fraction >= 0.5
        and partner_effect["bidirectional"]["mrr"]["lower"] > 0.0
    )
    checks.append(
        {
            "gate": "partner_relabeling_removes_half_excess_with_positive_paired_ci",
            "partner_drop": drop,
            "excess_above_exact_random": excess,
            "fraction_of_excess_removed": removal_fraction,
            "paired_ci_lower": partner_effect["bidirectional"]["mrr"]["lower"],
            "paired_ci_source": partner_effect_interval_source,
            "status": "passed" if partner_pass else "failed",
        }
    )
    checks.append(
        {
            "gate": "raw_random_head_length_composition_and_unary_controls_lower",
            "status": "pending",
            "reason": "Control runs require separately pinned score artifacts.",
        }
    )
    statuses = [check["status"] for check in checks]
    overall = "failed" if "failed" in statuses else "pending" if "pending" in statuses else "passed"
    return {
        "protocol": "PARTNER_RETRIEVAL_PROTOCOL_V1_go_no_go_gates",
        "checks": checks,
        "overall_status": overall,
        "interpretation": (
            "A pending or failed status does not authorize seed expansion or frozen-test evaluation."
        ),
    }


def run_statistics(args: argparse.Namespace) -> dict[str, Any]:
    """Verify inputs, run all deterministic analyses, and return a JSON-safe report."""
    if args.permutations <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("Permutation and bootstrap counts must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    verified = verify_evaluation_and_artifact(
        args.evaluation_json,
        args.score_artifact,
        args.expected_evaluation_sha256,
        args.expected_score_artifact_sha256,
    )
    execution_lock = (
        _validate_locked_confirmation_parameters(args)
        if verified["evidence_scope"] == "reused_development_confirmation"
        else None
    )
    artifact = verified["artifact"]
    similarities = artifact["similarities"]
    mask = artifact["positive_mask"]
    rank_matrices = {
        "antibody_to_antigen": (_stable_rank_matrix(similarities), mask),
        "antigen_to_antibody": (_stable_rank_matrix(similarities.T), mask.T),
    }
    observed_vectors = {
        direction: _metric_vectors(_best_positive_ranks(ranks, direction_mask))
        for direction, (ranks, direction_mask) in rank_matrices.items()
    }
    observed = {
        direction: _point_metrics(vectors)
        for direction, vectors in observed_vectors.items()
    }
    observed["bidirectional"] = {
        metric: 0.5
        * (
            observed["antibody_to_antigen"][metric]
            + observed["antigen_to_antibody"][metric]
        )
        for metric in METRIC_NAMES
    }
    exact_random = exact_random_expectations(mask)
    exact_vectors = {
        "antibody_to_antigen": exact_random_anchor_vectors(
            mask.shape[1], mask.sum(axis=1).astype(int)
        ),
        "antigen_to_antibody": exact_random_anchor_vectors(
            mask.shape[0], mask.sum(axis=0).astype(int)
        ),
    }
    model_minus_random_vectors = {
        direction: {
            metric: observed_vectors[direction][metric] - exact_vectors[direction][metric]
            for metric in METRIC_NAMES
        }
        for direction in DIRECTIONS
    }
    permutation_internal = fixed_degree_label_permutation_test(
        similarities,
        mask,
        args.permutations,
        args.seed,
    )
    partner_drop_vectors = {
        direction: {
            metric: observed_vectors[direction][metric]
            - permutation_internal["mean_null_vectors"][direction][metric]
            for metric in METRIC_NAMES
        }
        for direction in DIRECTIONS
    }
    cluster_ids, cluster_provenance = _load_cluster_ids(
        args.anchor_clusters_json,
        args.expected_anchor_clusters_sha256,
        verified["evaluation_sha256"],
        artifact["sha256"],
        artifact["antibody_keys"],
        artifact["antigen_keys"],
    )
    exact_effect_anchor = paired_effect_bootstrap(
        model_minus_random_vectors,
        args.bootstrap_samples,
        args.seed + 1,
        args.alpha,
    )
    partner_effect_anchor = paired_effect_bootstrap(
        partner_drop_vectors,
        args.bootstrap_samples,
        args.seed + 2,
        args.alpha,
    )
    joint_exact_effect = joint_bipartite_component_bootstrap(
        model_minus_random_vectors,
        mask,
        args.bootstrap_samples,
        args.seed + 6,
        args.alpha,
    )
    joint_partner_effect = joint_bipartite_component_bootstrap(
        partner_drop_vectors,
        mask,
        args.bootstrap_samples,
        args.seed + 7,
        args.alpha,
    )
    if cluster_ids is None:
        cluster_bootstrap: dict[str, Any] = {
            "available": False,
            "reason": cluster_provenance["reason"],
        }
    else:
        cluster_bootstrap = {
            "available": True,
            "model_minus_exact_random": paired_effect_bootstrap(
                model_minus_random_vectors,
                args.bootstrap_samples,
                args.seed + 3,
                args.alpha,
                cluster_ids,
            ),
            "model_minus_partner_relabeling": paired_effect_bootstrap(
                partner_drop_vectors,
                args.bootstrap_samples,
                args.seed + 4,
                args.alpha,
                cluster_ids,
            ),
        }
    permutation = {
        key: value
        for key, value in permutation_internal.items()
        if key not in {"observed_vectors", "mean_null_vectors"}
    }
    effects = {
        "anchor_bootstrap": {
            "model_minus_exact_random": exact_effect_anchor,
            "model_minus_partner_relabeling": partner_effect_anchor,
        },
        "cluster_bootstrap": cluster_bootstrap,
        "joint_bipartite_component_bootstrap": {
            "model_minus_exact_random": joint_exact_effect,
            "model_minus_partner_relabeling": joint_partner_effect,
        },
        "independent_direction_bootstrap_note": (
            "Independent directional resampling is an approximation; the joint "
            "positive-graph component bootstrap is preferred when at least two "
            "components are available."
        ),
    }
    gate_partner_effect = (
        joint_partner_effect if joint_partner_effect.get("available") else partner_effect_anchor
    )
    gate_interval_source = (
        "joint_bipartite_component_bootstrap"
        if joint_partner_effect.get("available")
        else "paired_anchor_bootstrap_independent_directions"
    )
    gate = evaluate_preregistered_gates(
        observed,
        exact_random,
        permutation,
        gate_partner_effect,
        gate_interval_source,
    )
    curves = None
    if args.candidate_sizes:
        curves = candidate_size_curves(
            similarities,
            mask,
            args.candidate_sizes,
            args.candidate_samples,
            args.seed + 5,
        )
    evaluation = verified["evaluation"]
    confirmation_scope = verified["evidence_scope"] == "reused_development_confirmation"
    return {
        "format_version": STATISTICS_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": {
            "split": "validation",
            "intended_use": (
                "reused_development_confirmation_statistics_only"
                if confirmation_scope
                else "exploratory_development_statistics_only"
            ),
            "frozen_test_used": False,
            "claims_allowed": (
                "reused development confirmation evidence only; not an independent test"
                if confirmation_scope
                else "exploratory development partner-retrieval evidence only"
            ),
        },
        "inputs": {
            "evaluation_json": str(verified["evaluation_path"]),
            "evaluation_json_sha256": verified["evaluation_sha256"],
            "score_artifact": str(artifact["path"]),
            "score_artifact_sha256": artifact["sha256"],
            "score_contract_sha256": artifact["metadata"]["contract_sha256"],
            "checkpoint_sha256": evaluation["checkpoint_sha256"],
            "config_sha256": evaluation["config_sha256"],
            "split_manifest_sha256": evaluation["dataset"]["split_manifest_sha256"],
            "authority_split_sha256": evaluation["dataset"]["authority"]["sha256"],
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "anchor_clusters": cluster_provenance,
            "boundary_verification": verified["boundary_verification"],
            "one_shot_verification": verified["one_shot_verification"],
            "checkpoint_suite_verification": verified[
                "checkpoint_suite_verification"
            ],
        },
        "parameters": {
            "permutations": args.permutations,
            "bootstrap_samples": args.bootstrap_samples,
            "confidence_level": 1.0 - args.alpha,
            "seed": args.seed,
            "recall_ks": list(RECALL_KS),
            "execution_lock": execution_lock,
        },
        "library": {
            "antibody_count": int(mask.shape[0]),
            "antigen_count": int(mask.shape[1]),
            "positive_edge_count": int(mask.sum()),
            "matrix_sha256": artifact["array_sha256"]["similarity_matrix"],
            "positive_mask_sha256": artifact["array_sha256"]["positive_mask"],
        },
        "observed": observed,
        "exact_random": exact_random,
        "fixed_degree_label_permutation": permutation,
        "paired_effect_intervals": effects,
        "candidate_size_curves": curves,
        "preregistered_gate_evaluation": gate,
        "limitations": [
            "Unobserved SAbDab pairings are not experimentally verified non-binders.",
            "Frozen protein-language-model pretraining exposure cannot be fully audited here.",
            "Anchor bootstrap is not a substitute for cluster bootstrap when clusters are unavailable.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-json", type=Path, required=True)
    parser.add_argument("--score-artifact", type=Path, required=True)
    parser.add_argument("--expected-evaluation-sha256", required=True)
    parser.add_argument("--expected-score-artifact-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-clusters-json", type=Path)
    parser.add_argument("--expected-anchor-clusters-sha256")
    parser.add_argument("--candidate-sizes", type=int, nargs="*", default=[])
    parser.add_argument("--candidate-samples", type=int, default=1_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    protected = {
        args.evaluation_json.expanduser().resolve(),
        args.score_artifact.expanduser().resolve(),
    }
    if output in protected:
        raise ValueError("Statistics output cannot overwrite a verified input")
    report = run_statistics(args)
    from sabdab2_asymmetric_partner_retrieval import atomic_write_json

    atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
