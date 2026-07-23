#!/usr/bin/env python3
"""Run validation-only statistical analysis for a partner-retrieval checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sabdab2_partner_retrieval import (
    FORMAT_VERSION,
    SPLIT_FORMAT_VERSION,
    _encode_library,
    _model_from_config,
    _pad_sequence_features,
    _pair_set,
    _records_from_frame,
    _source_hashes,
    atomic_write_json,
    audit_double_remote_isolation,
    build_positive_mask,
    canonical_sha256,
    deduplicate_exact_pairs,
    embedding_corpus_fingerprint,
    full_library_metrics,
    sequence_bucket_batches,
    sha256_file,
    unique_sequences,
    validate_config_integrity,
)


STATISTICS_FORMAT_VERSION = "sabdab2_partner_retrieval_statistics_v1"
RECALL_KS = (1, 5, 10)
METRIC_NAMES = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10")


def _strict_float(value: Any, name: str) -> float:
    """Convert a scalar to a finite built-in float for strict JSON output."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _array_sha256(array: np.ndarray) -> str:
    """Hash an array together with its shape and dtype."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _stable_rank_matrix(similarities: np.ndarray) -> np.ndarray:
    """Return one-based candidate ranks with deterministic index tie-breaking."""
    values = np.asarray(similarities, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Similarities must be a finite non-empty matrix")
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int64)
    row_indices = np.arange(values.shape[0], dtype=np.int64)[:, None]
    ranks[row_indices, order] = np.arange(1, values.shape[1] + 1, dtype=np.int64)
    return ranks


def _best_positive_ranks(
    rank_matrix: np.ndarray, positive_mask: np.ndarray
) -> np.ndarray:
    """Extract the best known-positive rank for every anchor."""
    ranks = np.asarray(rank_matrix, dtype=np.int64)
    mask = np.asarray(positive_mask, dtype=bool)
    if ranks.shape != mask.shape or ranks.ndim != 2:
        raise ValueError("Rank and positive matrices must be aligned")
    if not mask.any(axis=1).all():
        raise ValueError("Every anchor must have at least one positive candidate")
    sentinel = ranks.shape[1] + 1
    return np.where(mask, ranks, sentinel).min(axis=1)


def _metric_vectors(ranks: np.ndarray) -> dict[str, np.ndarray]:
    """Convert positive ranks into per-anchor retrieval metric values."""
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


def _point_metrics(ranks: np.ndarray) -> dict[str, float]:
    """Average per-anchor metric vectors into directional point estimates."""
    return {
        name: _strict_float(vector.mean(), name)
        for name, vector in _metric_vectors(ranks).items()
    }


def _survival_probability(candidate_count: int, positive_count: int, cutoff: int) -> float:
    """Return P(best positive rank > cutoff) under a uniform random ordering."""
    if candidate_count <= 0 or not 1 <= positive_count <= candidate_count:
        raise ValueError("Invalid finite-library degree")
    if cutoff <= 0:
        return 1.0
    if cutoff >= candidate_count or candidate_count - cutoff < positive_count:
        return 0.0
    probability = 1.0
    for offset in range(cutoff):
        probability *= (candidate_count - positive_count - offset) / (
            candidate_count - offset
        )
    return max(0.0, min(1.0, probability))


def exact_random_direction(
    candidate_count: int,
    positive_degrees: Sequence[int],
) -> dict[str, Any]:
    """Compute exact finite-library expectations for heterogeneous positive degrees."""
    degrees = [int(value) for value in positive_degrees]
    if not degrees:
        raise ValueError("At least one anchor degree is required")
    if any(value <= 0 or value > candidate_count for value in degrees):
        raise ValueError("Every positive degree must fit the candidate library")

    expected_mrr = []
    expected_mean_rank = []
    expected_recalls = {cutoff: [] for cutoff in RECALL_KS}
    for degree in degrees:
        survival_before = 1.0
        reciprocal_rank = 0.0
        for rank in range(1, candidate_count - degree + 2):
            survival_after = _survival_probability(candidate_count, degree, rank)
            probability_at_rank = survival_before - survival_after
            reciprocal_rank += probability_at_rank / rank
            survival_before = survival_after
        expected_mrr.append(reciprocal_rank)
        expected_mean_rank.append((candidate_count + 1.0) / (degree + 1.0))
        for cutoff in RECALL_KS:
            expected_recalls[cutoff].append(
                1.0 - _survival_probability(candidate_count, degree, cutoff)
            )

    histogram = Counter(degrees)
    return {
        "anchor_count": len(degrees),
        "candidate_count": candidate_count,
        "positive_degree_histogram": {
            str(key): histogram[key] for key in sorted(histogram)
        },
        "mean_positives_per_anchor": _strict_float(np.mean(degrees), "mean degree"),
        "mrr": _strict_float(np.mean(expected_mrr), "expected MRR"),
        "mean_rank": _strict_float(np.mean(expected_mean_rank), "expected rank"),
        **{
            f"recall_at_{cutoff}": _strict_float(
                np.mean(expected_recalls[cutoff]), f"expected recall@{cutoff}"
            )
            for cutoff in RECALL_KS
        },
    }


def exact_random_expectations(positive_mask: np.ndarray) -> dict[str, Any]:
    """Compute exact random-order expectations in both retrieval directions."""
    mask = np.asarray(positive_mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("Positive mask must be a non-empty matrix")
    antibody_to_antigen = exact_random_direction(
        mask.shape[1], mask.sum(axis=1).astype(int).tolist()
    )
    antigen_to_antibody = exact_random_direction(
        mask.shape[0], mask.sum(axis=0).astype(int).tolist()
    )
    bidirectional = {
        metric: 0.5
        * (antibody_to_antigen[metric] + antigen_to_antibody[metric])
        for metric in METRIC_NAMES
    }
    return {
        "protocol": "uniform_candidate_order_with_anchor_specific_positive_degree",
        "formula": {
            "survival": "P(R>k)=C(N-k,d)/C(N,d)",
            "mean_rank": "E[R]=(N+1)/(d+1)",
        },
        "antibody_to_antigen": antibody_to_antigen,
        "antigen_to_antibody": antigen_to_antibody,
        "bidirectional": bidirectional,
    }


def _permuted_positive_ranks(
    antibody_rank_matrix: np.ndarray,
    antigen_rank_matrix: np.ndarray,
    positive_mask: np.ndarray,
    antibody_permutation: np.ndarray,
    antigen_permutation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute both sequence-label axes while keeping graph degrees fixed."""
    mask = np.asarray(positive_mask, dtype=bool)
    antibody_order = np.asarray(antibody_permutation, dtype=np.int64)
    antigen_order = np.asarray(antigen_permutation, dtype=np.int64)
    if antibody_order.shape != (mask.shape[0],) or not np.array_equal(
        np.sort(antibody_order), np.arange(mask.shape[0])
    ):
        raise ValueError("Antibody permutation is invalid")
    if antigen_order.shape != (mask.shape[1],) or not np.array_equal(
        np.sort(antigen_order), np.arange(mask.shape[1])
    ):
        raise ValueError("Antigen permutation is invalid")

    antibody_ranks = np.empty(mask.shape[0], dtype=np.int64)
    for anchor_index, positive_labels in enumerate(mask):
        antibody_ranks[anchor_index] = antibody_rank_matrix[
            antibody_order[anchor_index], antigen_order[positive_labels]
        ].min()

    antigen_ranks = np.empty(mask.shape[1], dtype=np.int64)
    for anchor_index in range(mask.shape[1]):
        antigen_ranks[anchor_index] = antigen_rank_matrix[
            antigen_order[anchor_index], antibody_order[mask[:, anchor_index]]
        ].min()
    return antibody_ranks, antigen_ranks


def _null_metric_summary(
    null_values: np.ndarray, observed: float
) -> dict[str, Any]:
    """Summarize a Monte Carlo null and compute an add-one upper-tail p-value."""
    values = np.asarray(null_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Null statistics must be a finite non-empty vector")
    exceedances = int(np.count_nonzero(values >= observed - 1e-15))
    return {
        "observed": _strict_float(observed, "observed statistic"),
        "null_mean": _strict_float(values.mean(), "null mean"),
        "null_standard_deviation": _strict_float(
            values.std(ddof=1) if values.size > 1 else 0.0, "null standard deviation"
        ),
        "null_percentile_2_5": _strict_float(
            np.quantile(values, 0.025), "null percentile"
        ),
        "null_percentile_97_5": _strict_float(
            np.quantile(values, 0.975), "null percentile"
        ),
        "upper_tail_exceedance_count": exceedances,
        "p_value": (exceedances + 1.0) / (values.size + 1.0),
    }


def fixed_degree_label_permutation_test(
    similarities: np.ndarray,
    positive_mask: np.ndarray,
    permutation_count: int,
    seed: int,
) -> dict[str, Any]:
    """Test partner alignment by permuting antigen labels with graph degrees fixed."""
    values = np.asarray(similarities, dtype=np.float64)
    mask = np.asarray(positive_mask, dtype=bool)
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("Similarity and positive matrices must be aligned")
    if permutation_count <= 0:
        raise ValueError("permutation_count must be positive")
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        raise ValueError("Every anchor must have at least one positive")

    antibody_rank_matrix = _stable_rank_matrix(values)
    antigen_rank_matrix = _stable_rank_matrix(values.transpose())
    observed_antibody_ranks = _best_positive_ranks(antibody_rank_matrix, mask)
    observed_antigen_ranks = _best_positive_ranks(
        antigen_rank_matrix, mask.transpose()
    )
    observed_vectors = {
        "antibody_to_antigen": _metric_vectors(observed_antibody_ranks),
        "antigen_to_antibody": _metric_vectors(observed_antigen_ranks),
    }
    observed = {
        direction: {
            metric: _strict_float(vector.mean(), f"observed {metric}")
            for metric, vector in vectors.items()
        }
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
        for direction in (
            "antibody_to_antigen",
            "antigen_to_antibody",
            "bidirectional",
        )
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
        directional = {
            "antibody_to_antigen": _point_metrics(antibody_ranks),
            "antigen_to_antibody": _point_metrics(antigen_ranks),
        }
        for metric in METRIC_NAMES:
            null["antibody_to_antigen"][metric][permutation_index] = directional[
                "antibody_to_antigen"
            ][metric]
            null["antigen_to_antibody"][metric][permutation_index] = directional[
                "antigen_to_antibody"
            ][metric]
            null["bidirectional"][metric][permutation_index] = 0.5 * (
                directional["antibody_to_antigen"][metric]
                + directional["antigen_to_antibody"][metric]
            )

    results = {
        direction: {
            metric: _null_metric_summary(
                null[direction][metric], observed[direction][metric]
            )
            for metric in METRIC_NAMES
        }
        for direction in null
    }
    return {
        "protocol": (
            "independently_permute_antibody_and_antigen_embedding_labels_against_"
            "the_fixed_positive_graph"
        ),
        "degree_constraint": (
            "The positive mask is unchanged, so every antibody and antigen degree "
            "is fixed in every permutation."
        ),
        "alternative": "trained_retrieval_metric_is_greater_than_or_equal_to_null",
        "p_value_estimator": "(1 + upper-tail exceedances) / (1 + permutations)",
        "permutation_count": permutation_count,
        "seed": int(seed),
        "minimum_attainable_p_value": 1.0 / (permutation_count + 1.0),
        "results": results,
    }


def _bootstrap_metric_intervals(
    vectors: Mapping[str, np.ndarray],
    bootstrap_count: int,
    alpha: float,
    generator: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Bootstrap one direction by resampling anchors with replacement."""
    anchor_count = len(next(iter(vectors.values())))
    indices = generator.integers(
        0, anchor_count, size=(bootstrap_count, anchor_count), endpoint=False
    )
    distributions = {
        metric: np.asarray(vector, dtype=np.float64)[indices].mean(axis=1)
        for metric, vector in vectors.items()
    }
    intervals = {
        metric: {
            "estimate": _strict_float(np.mean(vector), f"{metric} estimate"),
            "lower": _strict_float(
                np.quantile(distributions[metric], alpha / 2.0), "bootstrap lower"
            ),
            "upper": _strict_float(
                np.quantile(distributions[metric], 1.0 - alpha / 2.0),
                "bootstrap upper",
            ),
            "bootstrap_standard_error": _strict_float(
                distributions[metric].std(ddof=1)
                if bootstrap_count > 1
                else 0.0,
                "bootstrap standard error",
            ),
        }
        for metric, vector in vectors.items()
    }
    return intervals, distributions


def anchor_bootstrap_intervals(
    similarities: np.ndarray,
    positive_mask: np.ndarray,
    bootstrap_count: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute deterministic percentile CIs by independently resampling anchors."""
    values = np.asarray(similarities, dtype=np.float64)
    mask = np.asarray(positive_mask, dtype=bool)
    if values.shape != mask.shape or values.ndim != 2:
        raise ValueError("Similarity and positive matrices must be aligned")
    if bootstrap_count <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("Bootstrap settings are invalid")

    antibody_ranks = _best_positive_ranks(_stable_rank_matrix(values), mask)
    antigen_ranks = _best_positive_ranks(
        _stable_rank_matrix(values.transpose()), mask.transpose()
    )
    seed_sequence = np.random.SeedSequence(seed)
    antibody_seed, antigen_seed = seed_sequence.spawn(2)
    antibody_intervals, antibody_distributions = _bootstrap_metric_intervals(
        _metric_vectors(antibody_ranks),
        bootstrap_count,
        alpha,
        np.random.default_rng(antibody_seed),
    )
    antigen_intervals, antigen_distributions = _bootstrap_metric_intervals(
        _metric_vectors(antigen_ranks),
        bootstrap_count,
        alpha,
        np.random.default_rng(antigen_seed),
    )
    bidirectional = {}
    for metric in METRIC_NAMES:
        distribution = 0.5 * (
            antibody_distributions[metric] + antigen_distributions[metric]
        )
        estimate = 0.5 * (
            antibody_intervals[metric]["estimate"]
            + antigen_intervals[metric]["estimate"]
        )
        bidirectional[metric] = {
            "estimate": estimate,
            "lower": _strict_float(
                np.quantile(distribution, alpha / 2.0), "bootstrap lower"
            ),
            "upper": _strict_float(
                np.quantile(distribution, 1.0 - alpha / 2.0), "bootstrap upper"
            ),
            "bootstrap_standard_error": _strict_float(
                distribution.std(ddof=1) if bootstrap_count > 1 else 0.0,
                "bootstrap standard error",
            ),
        }
    return {
        "protocol": "nonparametric_anchor_bootstrap",
        "directional_resampling": "independent_within_each_retrieval_direction",
        "interval": "two-sided_percentile",
        "confidence_level": 1.0 - alpha,
        "bootstrap_count": bootstrap_count,
        "seed": int(seed),
        "antibody_to_antigen": antibody_intervals,
        "antigen_to_antibody": antigen_intervals,
        "bidirectional": bidirectional,
    }


def _encode_raw_masked_mean_library(
    records: Sequence[Any],
    embedding_store: Any,
    input_dim: int,
    batch_size: int,
    bucket_width: int,
    device: Any,
) -> tuple[list[str], Any]:
    """Encode a library with parameter-free masked mean pooling of frozen ESM residues."""
    import torch
    import torch.nn.functional as functional

    encoded: dict[str, Any] = {}
    for batch in sequence_bucket_batches(records, batch_size, bucket_width):
        features, mask = _pad_sequence_features(
            batch, embedding_store, input_dim, device
        )
        weights = mask.unsqueeze(-1).to(dtype=features.dtype)
        pooled = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = functional.normalize(pooled, p=2.0, dim=-1, eps=1e-8)
        for record, embedding in zip(batch, pooled):
            encoded[record.key] = embedding.detach().to(device="cpu")
    keys = sorted(encoded)
    if keys != sorted(record.key for record in records):
        raise AssertionError("Raw library encoding did not preserve every sequence")
    return keys, torch.stack([encoded[key] for key in keys], dim=0)


def _load_development_dataset(
    prepare_dir: Path,
    split_manifest: Path,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Load only train and validation rows while leaving frozen-test rows unmaterialized."""
    import pandas as pd
    import pyarrow.parquet as parquet

    prepare_root = prepare_dir.expanduser().resolve()
    instances_path = prepare_root / "instances.parquet"
    manifest_path = split_manifest.expanduser().resolve()
    if not instances_path.is_file():
        raise FileNotFoundError(f"Prepared instances are missing: {instances_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest is missing: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != SPLIT_FORMAT_VERSION:
        raise ValueError("Split manifest is not a compatible interaction-first v1 split")
    if payload.get("intended_use") != "development_only":
        raise ValueError("Statistical analysis requires a development-only split")
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("Split manifest does not contain split lists")

    split_ids: dict[str, list[str]] = {}
    assigned_ids: set[str] = set()
    for split_name in ("train", "validation", "test"):
        identifiers = [str(value) for value in raw_splits.get(split_name, [])]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Split {split_name!r} must contain unique instance IDs")
        overlap = assigned_ids.intersection(identifiers)
        if overlap:
            raise ValueError(f"Split membership overlaps at {sorted(overlap)[:5]}")
        split_ids[split_name] = identifiers
        assigned_ids.update(identifiers)

    parquet_row_count = int(parquet.ParquetFile(instances_path).metadata.num_rows)
    if parquet_row_count != len(assigned_ids):
        raise ValueError(
            "Split membership count differs from the immutable Parquet row count"
        )

    development_ids = [*split_ids["train"], *split_ids["validation"]]
    columns = [
        "instance_id",
        "ab_sequence_sha256",
        "ag_sequence_sha256",
        "ab_length",
        "ag_length",
    ]
    frame = pd.read_parquet(
        instances_path,
        columns=columns,
        filters=[("instance_id", "in", development_ids)],
    )
    if frame["instance_id"].isna().any():
        raise ValueError("Development instances contain null IDs")
    frame["instance_id"] = frame["instance_id"].astype(str)
    if frame["instance_id"].duplicated().any():
        raise ValueError("Development instances contain duplicate IDs")
    if set(frame["instance_id"]) != set(development_ids):
        raise ValueError("Development rows do not match train/validation membership")

    assignment = {
        instance_id: split_name
        for split_name in ("train", "validation")
        for instance_id in split_ids[split_name]
    }
    frame["retrieval_split"] = frame["instance_id"].map(assignment)
    records = {
        split_name: _records_from_frame(
            frame.loc[frame["retrieval_split"] == split_name, columns].sort_values(
                "instance_id", kind="stable"
            )
        )
        for split_name in ("train", "validation")
    }
    metadata = {
        "instances_path": str(instances_path),
        "instances_sha256": sha256_file(instances_path),
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "split_intended_use": payload["intended_use"],
        "raw_instance_counts": {
            name: len(split_ids[name]) for name in ("train", "validation", "test")
        },
    }
    return records, metadata


def _verify_checkpoint_and_data(args: argparse.Namespace, device: Any) -> dict[str, Any]:
    """Load validation inputs and enforce all checkpoint provenance fingerprints."""
    from interaction_first.runtime import safe_torch_load

    split_records, dataset_metadata = _load_development_dataset(
        args.prepare_dir, args.split_manifest
    )
    train_records, train_duplicates = deduplicate_exact_pairs(split_records["train"])
    validation_records, validation_duplicates = deduplicate_exact_pairs(
        split_records["validation"]
    )
    isolation = audit_double_remote_isolation(
        train_records, validation_records, "validation"
    )

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint_path}")
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Checkpoint format is incompatible")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint configuration is missing")
    validate_config_integrity(config)
    if config.get("source_sha256") != _source_hashes():
        raise ValueError("Analysis source dependencies differ from checkpoint provenance")
    if config["dataset"]["instances_sha256"] != dataset_metadata["instances_sha256"]:
        raise ValueError("Prepared instances changed after checkpoint training")
    if (
        config["dataset"]["split_manifest_sha256"]
        != dataset_metadata["split_manifest_sha256"]
    ):
        raise ValueError("Split manifest changed after checkpoint training")

    validation_pair_hash = canonical_sha256(
        {"pairs": sorted([list(value) for value in _pair_set(validation_records)])}
    )
    train_pair_hash = canonical_sha256(
        {"pairs": sorted([list(value) for value in _pair_set(train_records)])}
    )
    if config["validation"]["exact_pair_sha256"] != validation_pair_hash:
        raise ValueError("Validation exact-pair set differs from checkpoint provenance")
    if config["validation"]["pair_count"] != len(validation_records):
        raise ValueError("Validation pair count differs from checkpoint provenance")
    if config["training_data"]["exact_pair_sha256"] != train_pair_hash:
        raise ValueError("Training exact-pair set differs from checkpoint provenance")
    if (
        config["training_data"]["pair_count_after_exact_deduplication"]
        != len(train_records)
    ):
        raise ValueError("Training pair count differs from checkpoint provenance")

    embedding_root = args.embedding_dir.expanduser().resolve()
    embedding_corpus = embedding_corpus_fingerprint(
        embedding_root,
        [*train_records, *validation_records],
        "deduplicated_train_and_validation_sequences",
    )
    if config.get("embedding_corpus") != embedding_corpus:
        raise ValueError("Frozen embedding corpus changed after checkpoint training")
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "config": config,
        "train_records": train_records,
        "validation_records": validation_records,
        "dataset_metadata": dataset_metadata,
        "embedding_corpus": embedding_corpus,
        "embedding_root": embedding_root,
        "verification": {
            "checkpoint_format": True,
            "checkpoint_config_sha256": True,
            "source_sha256": True,
            "prepared_instances_sha256": True,
            "split_manifest_sha256": True,
            "train_exact_pair_sha256": True,
            "validation_exact_pair_sha256": True,
            "embedding_content_manifest_sha256": True,
            "train_validation_double_remote_isolation": isolation,
            "exact_duplicate_pairs_removed": {
                "train": train_duplicates,
                "validation": validation_duplicates,
            },
        },
    }


def _metrics_with_counts(
    antibody_embeddings: Any,
    antigen_embeddings: Any,
    positive_mask: Any,
    temperature: Any,
    exact_pair_count: int,
) -> dict[str, Any]:
    """Compute the common full-library metric schema for any embedding baseline."""
    metrics = full_library_metrics(
        antibody_embeddings,
        antigen_embeddings,
        positive_mask,
        temperature,
        RECALL_KS,
    )
    metrics.update(
        {
            "exact_pair_count": exact_pair_count,
            "unique_antibody_count": int(positive_mask.shape[0]),
            "unique_antigen_count": int(positive_mask.shape[1]),
        }
    )
    return metrics


def command_analyze(args: argparse.Namespace) -> None:
    """Run the sealed-test-safe validation analysis and write strict JSON."""
    import torch

    from interaction_first.retrieval import cosine_similarity_matrix
    from interaction_first.runtime import EmbeddingStore, set_reproducible_seed

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.permutations <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("Permutation and bootstrap counts must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if args.embedding_cache_gib < 0.0:
        raise ValueError("embedding_cache_gib cannot be negative")

    device = torch.device(args.device)
    verified = _verify_checkpoint_and_data(args, device)
    checkpoint = verified["checkpoint"]
    config = verified["config"]
    validation_records = verified["validation_records"]
    model_batch_size = args.batch_size or int(
        config["validation"]["validation_batch_size"]
    )
    bucket_width = args.bucket_width or int(config["optimization"]["bucket_width"])
    if model_batch_size <= 0 or bucket_width <= 0:
        raise ValueError("Batch and bucket settings must be positive")
    input_dim = int(config["model"]["input_dim"])
    embedding_store = EmbeddingStore(
        verified["embedding_root"], args.embedding_cache_gib
    )
    antibodies = unique_sequences(validation_records, "antibody")
    antigens = unique_sequences(validation_records, "antigen")

    trained_model = _model_from_config(config, device)
    trained_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    antibody_keys, trained_antibody_embeddings = _encode_library(
        trained_model,
        antibodies,
        "antibody",
        embedding_store,
        input_dim,
        model_batch_size,
        bucket_width,
        device,
    )
    antigen_keys, trained_antigen_embeddings = _encode_library(
        trained_model,
        antigens,
        "antigen",
        embedding_store,
        input_dim,
        model_batch_size,
        bucket_width,
        device,
    )
    positive_mask = build_positive_mask(
        antibody_keys,
        antigen_keys,
        _pair_set(validation_records),
        device="cpu",
    )
    trained_similarities = cosine_similarity_matrix(
        trained_antibody_embeddings.to(dtype=torch.float32, device="cpu"),
        trained_antigen_embeddings.to(dtype=torch.float32, device="cpu"),
    )
    trained_metrics = _metrics_with_counts(
        trained_antibody_embeddings,
        trained_antigen_embeddings,
        positive_mask,
        trained_model.temperature_head(),
        len(_pair_set(validation_records)),
    )

    initialization_seed = int(config["optimization"]["seed"])
    set_reproducible_seed(initialization_seed, deterministic=True)
    random_model = _model_from_config(config, device)
    random_antibody_keys, random_antibody_embeddings = _encode_library(
        random_model,
        antibodies,
        "antibody",
        embedding_store,
        input_dim,
        model_batch_size,
        bucket_width,
        device,
    )
    random_antigen_keys, random_antigen_embeddings = _encode_library(
        random_model,
        antigens,
        "antigen",
        embedding_store,
        input_dim,
        model_batch_size,
        bucket_width,
        device,
    )
    if random_antibody_keys != antibody_keys or random_antigen_keys != antigen_keys:
        raise AssertionError("Random baseline library ordering changed")
    random_metrics = _metrics_with_counts(
        random_antibody_embeddings,
        random_antigen_embeddings,
        positive_mask,
        random_model.temperature_head(),
        len(_pair_set(validation_records)),
    )

    raw_antibody_keys, raw_antibody_embeddings = _encode_raw_masked_mean_library(
        antibodies,
        embedding_store,
        input_dim,
        model_batch_size,
        bucket_width,
        device,
    )
    raw_antigen_keys, raw_antigen_embeddings = _encode_raw_masked_mean_library(
        antigens,
        embedding_store,
        input_dim,
        model_batch_size,
        bucket_width,
        device,
    )
    if raw_antibody_keys != antibody_keys or raw_antigen_keys != antigen_keys:
        raise AssertionError("Raw baseline library ordering changed")
    raw_metrics = _metrics_with_counts(
        raw_antibody_embeddings,
        raw_antigen_embeddings,
        positive_mask,
        torch.tensor(1.0, dtype=torch.float32),
        len(_pair_set(validation_records)),
    )

    similarity_array = trained_similarities.detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    mask_array = positive_mask.detach().cpu().numpy().astype(bool, copy=False)
    exact_null = exact_random_expectations(mask_array)
    permutation = fixed_degree_label_permutation_test(
        similarity_array,
        mask_array,
        args.permutations,
        args.analysis_seed,
    )
    bootstrap = anchor_bootstrap_intervals(
        similarity_array,
        mask_array,
        args.bootstrap_samples,
        args.analysis_seed,
        args.alpha,
    )

    output = {
        "format_version": STATISTICS_FORMAT_VERSION,
        "stage": "validation_only_partner_retrieval_statistics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": {
            "analysis_split": "validation",
            "frozen_test_scored": False,
            "frozen_test_embeddings_loaded": False,
            "frozen_test_pair_rows_materialized": False,
            "model_selection_used_this_validation_split": True,
            "inference_status": (
                "Post-selection validation analysis. Confidence intervals and "
                "permutation p-values do not correct for checkpoint selection on "
                "the same validation split and are not confirmatory test evidence."
            ),
        },
        "checkpoint": {
            "path": str(verified["checkpoint_path"]),
            "sha256": sha256_file(verified["checkpoint_path"]),
            "format_version": checkpoint["format_version"],
            "config_sha256": config["config_sha256"],
            "training_seed": initialization_seed,
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_validation_mrr_stored": _strict_float(
                checkpoint["best_validation_mrr"], "stored validation MRR"
            ),
        },
        "verification": verified["verification"],
        "dataset": verified["dataset_metadata"],
        "embedding_corpus": verified["embedding_corpus"],
        "validation_library": {
            "exact_pair_count": len(_pair_set(validation_records)),
            "unique_antibody_count": len(antibody_keys),
            "unique_antigen_count": len(antigen_keys),
            "antibody_key_order_sha256": hashlib.sha256(
                ("\n".join(antibody_keys) + "\n").encode("ascii")
            ).hexdigest(),
            "antigen_key_order_sha256": hashlib.sha256(
                ("\n".join(antigen_keys) + "\n").encode("ascii")
            ).hexdigest(),
            "positive_mask_sha256": _array_sha256(mask_array.astype(np.uint8)),
            "trained_cosine_similarity_sha256": _array_sha256(
                similarity_array.astype("<f8", copy=False)
            ),
            "trained_cosine_similarity_minimum": _strict_float(
                similarity_array.min(), "minimum similarity"
            ),
            "trained_cosine_similarity_maximum": _strict_float(
                similarity_array.max(), "maximum similarity"
            ),
        },
        "observed": {
            "trained_checkpoint": trained_metrics,
        },
        "baselines": {
            "same_architecture_random_initialization": {
                "protocol": (
                    "Same checkpoint architecture, deterministic training seed, "
                    "and no parameter updates."
                ),
                "seed": initialization_seed,
                "metrics": random_metrics,
            },
            "raw_frozen_esm_masked_mean_cosine": {
                "protocol": (
                    "Parameter-free masked mean of frozen residue embeddings followed "
                    "by L2 normalization and cosine similarity."
                ),
                "loss_temperature": 1.0,
                "metrics": raw_metrics,
            },
        },
        "exact_finite_library_random_expectation": exact_null,
        "fixed_degree_label_permutation": permutation,
        "anchor_bootstrap_confidence_intervals": bootstrap,
        "runtime": {
            "device": str(device),
            "encoding_batch_size": model_batch_size,
            "bucket_width": bucket_width,
            "embedding_cache": embedding_store.summary(),
        },
    }
    output_path = args.output.expanduser().resolve()
    atomic_write_json(output_path, output)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    """Build the validation-only statistical analysis interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--bucket-width", type=int, default=0)
    parser.add_argument("--embedding-cache-gib", type=float, default=24.0)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--analysis-seed", type=int, default=20260723)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser


def main() -> None:
    """Run the validation-only statistical analysis."""
    command_analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
