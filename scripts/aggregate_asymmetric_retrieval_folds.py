#!/usr/bin/env python3
"""Aggregate the predeclared three-fold asymmetric retrieval screen.

The input manifest is itself SHA-256 pinned on the command line. Every JSON
file read by this program must also have an explicit path and SHA-256 entry in
that manifest. The accepted manifest format is documented by
``asymmetric_retrieval_fold_aggregation_manifest.schema.json``.

Example:

    python aggregate_asymmetric_retrieval_folds.py \
      --manifest aggregation_manifest.json \
      --manifest-sha256 <reviewed-sha256> \
      --output fold_selection.json

The program never reads a checkpoint or a frozen test artifact. It selects a
variant by unweighted fold-macro bidirectional MRR, then worst-fold MRR, then
the exact trainable parameter count, and finally the lexical variant name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_FORMAT_VERSION = "sabdab2_asymmetric_fold_aggregation_manifest_v1"
OUTPUT_FORMAT_VERSION = "sabdab2_asymmetric_fold_aggregation_v1"
HISTORY_FORMAT_VERSION = "sabdab2_asymmetric_partner_retrieval_v1"
FOLD_FORMAT_VERSION = "sabdab2_antigen_cluster_development_fold_v1"
REQUIRED_VARIANTS = frozenset(
    {
        "esm2_calm_mean",
        "antiberty_calm_mean",
        "antiberty_chain_balanced",
    }
)
REQUIRED_FOLDS = frozenset({0, 1, 2})
SHA256_LENGTH = 64


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the aggregation command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Accept an existing output only when it is byte-identical.",
    )
    args = parser.parse_args(argv)
    args.manifest_sha256 = normalize_sha256(
        args.manifest_sha256, "--manifest-sha256"
    )
    return args


def normalize_sha256(value: Any, label: str) -> str:
    """Return a normalized SHA-256 digest or reject the value."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a SHA-256 string")
    return normalized


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the bounded-memory SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"JSON contains a duplicate key: {key!r}")
        output[key] = value
    return output


def strict_json_loads(value: bytes, label: str) -> Any:
    """Parse strict finite JSON with duplicate-key rejection."""
    try:
        return json.loads(
            value.decode("utf-8-sig"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid strict JSON in {label}: {error}") from error


def load_pinned_json(path: Path, expected_sha256: str, label: str) -> tuple[Any, str]:
    """Read and parse one file only after verifying its exact byte digest."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    expected = normalize_sha256(expected_sha256, f"{label} SHA-256")
    encoded = resolved.read_bytes()
    observed = sha256_bytes(encoded)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected}, observed={observed}"
        )
    return strict_json_loads(encoded, label), observed


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash one JSON-compatible mapping with checkpoint-compatible encoding."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256_bytes(encoded)


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Return a mapping or raise a focused error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    """Return a list or raise a focused error."""
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def require_exact_keys(
    value: Mapping[str, Any], required: set[str], optional: set[str], label: str
) -> None:
    """Reject missing or undocumented mapping keys."""
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise ValueError(
            f"{label} keys differ from the manifest contract: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def require_integer(value: Any, label: str, minimum: int | None = None) -> int:
    """Return a non-boolean integer subject to an optional lower bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def require_finite_float(value: Any, label: str) -> float:
    """Return one finite numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def resolve_reference(
    manifest_path: Path, reference: Any, label: str
) -> tuple[Path, str]:
    """Resolve an exact path-and-hash reference relative to the manifest."""
    payload = require_mapping(reference, label)
    require_exact_keys(payload, {"path", "sha256"}, set(), label)
    raw_path = payload["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path must be a non-empty string")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve(), normalize_sha256(payload["sha256"], f"{label}.sha256")


def assert_subset_contract(actual: Any, expected: Any, label: str) -> None:
    """Recursively require every manifest contract value in an actual object."""
    if isinstance(expected, Mapping):
        actual_mapping = require_mapping(actual, label)
        for key, expected_value in expected.items():
            if key not in actual_mapping:
                raise ValueError(f"{label} is missing contract key {key!r}")
            assert_subset_contract(
                actual_mapping[key], expected_value, f"{label}.{key}"
            )
        return
    if actual != expected:
        raise ValueError(
            f"{label} differs from the pinned contract: "
            f"expected={expected!r}, observed={actual!r}"
        )


def reject_true_frozen_flags(value: Any, label: str = "payload") -> None:
    """Reject an affirmative frozen-test access flag anywhere in an input."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_label = f"{label}.{key}"
            if key in {
                "frozen_test_used",
                "frozen_test_unlocked",
                "frozen_test_records_selected",
                "frozen_test_labels_or_predictions_read",
                "frozen_test_sequences_aligned",
            } and nested is not False:
                raise PermissionError(f"Frozen test access is forbidden: {nested_label}")
            reject_true_frozen_flags(nested, nested_label)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            reject_true_frozen_flags(nested, f"{label}[{index}]")


def validate_config_integrity(config: Mapping[str, Any], label: str) -> None:
    """Verify the canonical configuration digest stored by training."""
    stored = normalize_sha256(config.get("config_sha256"), f"{label}.config_sha256")
    unhashed = {key: value for key, value in config.items() if key != "config_sha256"}
    observed = canonical_sha256(unhashed)
    if observed != stored:
        raise ValueError(
            f"{label} canonical hash mismatch: expected={stored}, observed={observed}"
        )


def exact_pooler_parameter_count(
    input_dim: int, hidden_dim: int, embedding_dim: int, pooling: str
) -> int:
    """Count trainable scalars in one CALMMeanPooler exactly."""
    input_dim = require_integer(input_dim, "input_dim", 1)
    hidden_dim = require_integer(hidden_dim, "hidden_dim", 1)
    embedding_dim = require_integer(embedding_dim, "embedding_dim", 1)
    if pooling not in {"mean", "attention"}:
        raise ValueError(f"Unsupported pooler for exact parameter counting: {pooling!r}")
    layer_norm = 2 * input_dim
    first_linear = input_dim * hidden_dim + hidden_dim
    second_linear = hidden_dim * embedding_dim + embedding_dim
    attention = embedding_dim if pooling == "attention" else 0
    return layer_norm + first_linear + second_linear + attention


def exact_trainable_parameter_count(config: Mapping[str, Any]) -> int:
    """Count every trainable scalar in the saved asymmetric model contract."""
    model = require_mapping(config.get("model"), "config.model")
    if model.get("architecture") != "AsymmetricPartnerRetrievalModel":
        raise ValueError("Only AsymmetricPartnerRetrievalModel can be counted exactly")
    if (
        model.get("projection_order")
        != "per_residue_ffn_then_length_pool_then_l2_normalize"
    ):
        raise ValueError("Unexpected projection order in saved model configuration")
    embedding_dim = require_integer(model.get("embedding_dim"), "embedding_dim", 1)
    antibody = exact_pooler_parameter_count(
        model.get("antibody_input_dim"),
        model.get("antibody_hidden_dim"),
        embedding_dim,
        model.get("antibody_pooling"),
    )
    antigen = exact_pooler_parameter_count(
        model.get("antigen_input_dim"),
        model.get("antigen_hidden_dim"),
        embedding_dim,
        model.get("antigen_pooling"),
    )
    return antibody + antigen + 1


def exact_random_best_positive_mrr(candidate_count: int, positive_count: int) -> Fraction:
    """Return exact random E[1/minimum positive rank] for one anchor."""
    candidate_count = require_integer(candidate_count, "candidate_count", 1)
    positive_count = require_integer(positive_count, "positive_count", 1)
    if positive_count > candidate_count:
        raise ValueError("positive_count cannot exceed candidate_count")
    denominator = comb(candidate_count, positive_count)
    expectation = Fraction(0, 1)
    for rank in range(1, candidate_count - positive_count + 2):
        arrangements = comb(candidate_count - rank, positive_count - 1)
        expectation += Fraction(arrangements, denominator * rank)
    return expectation


def fraction_payload(value: Fraction) -> dict[str, Any]:
    """Represent an exact rational and its finite floating approximation."""
    return {
        "exact_fraction": f"{value.numerator}/{value.denominator}",
        "value": float(value),
    }


def _validate_direction(
    direction: Mapping[str, Any],
    per_anchor: list[Any],
    anchor_keys: list[Any],
    candidate_keys: list[Any],
    label: str,
) -> tuple[float, Fraction, set[tuple[str, str]], dict[str, Any]]:
    """Validate one complete-library direction and compute its exact chance."""
    if not anchor_keys or len(anchor_keys) != len(set(anchor_keys)):
        raise ValueError(f"{label} anchor library must be non-empty and unique")
    if not candidate_keys or len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError(f"{label} candidate library must be non-empty and unique")
    if not all(isinstance(value, str) and value for value in anchor_keys + candidate_keys):
        raise ValueError(f"{label} library keys must be non-empty strings")
    anchor_count = require_integer(direction.get("anchor_count"), f"{label}.anchor_count", 1)
    candidate_count = require_integer(
        direction.get("candidate_count"), f"{label}.candidate_count", 1
    )
    if anchor_count != len(anchor_keys) or candidate_count != len(candidate_keys):
        raise ValueError(f"{label} reported library dimensions are inconsistent")
    if len(per_anchor) != anchor_count:
        raise ValueError(f"{label} per-anchor record count is inconsistent")

    candidate_set = set(candidate_keys)
    reciprocal_ranks: list[float] = []
    degrees: list[int] = []
    pairs: set[tuple[str, str]] = set()
    for index, raw_record in enumerate(per_anchor):
        record = require_mapping(raw_record, f"{label}.per_anchor[{index}]")
        anchor_key = record.get("anchor_key")
        if anchor_key != anchor_keys[index]:
            raise ValueError(f"{label} per-anchor key order differs from the library")
        rank = require_integer(
            record.get("best_known_positive_rank"),
            f"{label}.per_anchor[{index}].best_known_positive_rank",
            1,
        )
        if rank > candidate_count:
            raise ValueError(f"{label} contains an impossible positive rank")
        raw_positive_keys = require_list(
            record.get("known_positive_candidate_keys"),
            f"{label}.per_anchor[{index}].known_positive_candidate_keys",
        )
        positive_keys = [str(value) for value in raw_positive_keys]
        if not positive_keys or len(positive_keys) != len(set(positive_keys)):
            raise ValueError(f"{label} positive candidate keys must be non-empty and unique")
        if not set(positive_keys).issubset(candidate_set):
            raise ValueError(f"{label} positive keys are absent from the candidate library")
        degree = require_integer(
            record.get("known_positive_candidate_count"),
            f"{label}.per_anchor[{index}].known_positive_candidate_count",
            1,
        )
        if degree != len(positive_keys):
            raise ValueError(f"{label} positive degree differs from its candidate-key list")
        if "reciprocal_rank" in record:
            stored_rr = require_finite_float(
                record["reciprocal_rank"],
                f"{label}.per_anchor[{index}].reciprocal_rank",
            )
            if not math.isclose(stored_rr, 1.0 / rank, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{label} per-anchor reciprocal rank is inconsistent")
        reciprocal_ranks.append(1.0 / rank)
        degrees.append(degree)
        pairs.update((str(anchor_key), candidate) for candidate in positive_keys)

    observed_mrr = require_finite_float(direction.get("mrr"), f"{label}.mrr")
    recomputed_mrr = math.fsum(reciprocal_ranks) / anchor_count
    if not math.isclose(observed_mrr, recomputed_mrr, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{label} MRR differs from per-anchor ranks: "
            f"reported={observed_mrr}, recomputed={recomputed_mrr}"
        )
    reported_positive_count = require_integer(
        direction.get("known_positive_count"), f"{label}.known_positive_count", 1
    )
    if reported_positive_count != sum(degrees):
        raise ValueError(f"{label} known-positive count differs from per-anchor degrees")

    cached: dict[int, Fraction] = {}
    chance_terms: list[Fraction] = []
    for degree in degrees:
        if degree not in cached:
            cached[degree] = exact_random_best_positive_mrr(candidate_count, degree)
        chance_terms.append(cached[degree])
    exact_chance = sum(chance_terms, Fraction(0, 1)) / anchor_count
    degree_histogram = {str(key): value for key, value in sorted(Counter(degrees).items())}
    chance_payload = {
        **fraction_payload(exact_chance),
        "method": "finite_library_uniform_random_ranking_multi_positive",
        "anchor_count": anchor_count,
        "candidate_count": candidate_count,
        "positive_degree_histogram": degree_histogram,
        "positive_degree_sha256": sha256_bytes(
            ("".join(f"{value}\n" for value in degrees)).encode("ascii")
        ),
    }
    return observed_mrr, exact_chance, pairs, chance_payload


def validate_metrics(metrics: Any, label: str) -> dict[str, Any]:
    """Validate best-epoch full-library metrics and derive exact random chance."""
    payload = require_mapping(metrics, label)
    library_keys = require_mapping(payload.get("library_keys"), f"{label}.library_keys")
    antibody_keys = require_list(
        library_keys.get("antibody"), f"{label}.library_keys.antibody"
    )
    antigen_keys = require_list(
        library_keys.get("antigen"), f"{label}.library_keys.antigen"
    )
    ranks = require_mapping(payload.get("per_anchor_ranks"), f"{label}.per_anchor_ranks")
    antibody_to_antigen = require_mapping(
        payload.get("antibody_to_antigen"), f"{label}.antibody_to_antigen"
    )
    antigen_to_antibody = require_mapping(
        payload.get("antigen_to_antibody"), f"{label}.antigen_to_antibody"
    )
    ab_mrr, ab_chance, ab_pairs, ab_payload = _validate_direction(
        antibody_to_antigen,
        require_list(
            ranks.get("antibody_to_antigen"),
            f"{label}.per_anchor_ranks.antibody_to_antigen",
        ),
        antibody_keys,
        antigen_keys,
        f"{label}.antibody_to_antigen",
    )
    ag_mrr, ag_chance, reversed_ag_pairs, ag_payload = _validate_direction(
        antigen_to_antibody,
        require_list(
            ranks.get("antigen_to_antibody"),
            f"{label}.per_anchor_ranks.antigen_to_antibody",
        ),
        antigen_keys,
        antibody_keys,
        f"{label}.antigen_to_antibody",
    )
    ag_pairs = {(antibody, antigen) for antigen, antibody in reversed_ag_pairs}
    if ab_pairs != ag_pairs:
        raise ValueError(f"{label} directional positive-pair graphs differ")
    reported = require_finite_float(
        payload.get("bidirectional_mrr"), f"{label}.bidirectional_mrr"
    )
    recomputed = 0.5 * (ab_mrr + ag_mrr)
    if not math.isclose(reported, recomputed, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} bidirectional MRR is inconsistent")
    bidirectional_chance = (ab_chance + ag_chance) / 2
    return {
        "bidirectional_mrr": reported,
        "directional_mrr": {
            "antibody_to_antigen": ab_mrr,
            "antigen_to_antibody": ag_mrr,
        },
        "exact_random_chance": {
            "antibody_to_antigen": ab_payload,
            "antigen_to_antibody": ag_payload,
            "bidirectional": fraction_payload(bidirectional_chance),
        },
        "_chance_fraction": bidirectional_chance,
    }


def validate_fold_manifest(
    fold: Any,
    fold_index: int,
    fold_count: int,
    seed: int,
    authority: Mapping[str, Any],
    label: str,
) -> None:
    """Validate one pinned train-only fold contract."""
    payload = require_mapping(fold, label)
    reject_true_frozen_flags(payload, label)
    if payload.get("format_version") != FOLD_FORMAT_VERSION:
        raise ValueError(f"{label} has an unexpected format version")
    if payload.get("intended_use") != "development_only":
        raise PermissionError(f"{label} is not development-only")
    if payload.get("stage") != "train_only_antigen_cluster_model_selection":
        raise PermissionError(f"{label} is not a train-only selection fold")
    if require_integer(payload.get("fold_index"), f"{label}.fold_index") != fold_index:
        raise ValueError(f"{label} fold index differs from its manifest entry")
    if require_integer(payload.get("fold_count"), f"{label}.fold_count", 1) != fold_count:
        raise ValueError(f"{label} fold count differs from the aggregation manifest")
    if require_integer(payload.get("seed"), f"{label}.seed") != seed:
        raise ValueError(f"{label} seed differs from the aggregation manifest")
    if payload.get("source_pool_split") != "train":
        raise PermissionError(f"{label} uses a source split other than train")
    if payload.get("coverage_mode") != authority["coverage_mode"]:
        raise ValueError(f"{label} authority coverage mode differs")
    if payload.get("isolation_policy") != "antigen_exact":
        raise ValueError(f"{label} isolation policy is not antigen_exact")
    audit = require_mapping(payload.get("audit"), f"{label}.audit")
    if audit.get("valid") is not True:
        raise ValueError(f"{label} does not carry a valid audit")
    coverage = require_mapping(audit.get("coverage"), f"{label}.audit.coverage")
    if require_integer(
        coverage.get("sealed_id_intersection_count"),
        f"{label}.audit.coverage.sealed_id_intersection_count",
        0,
    ) != 0:
        raise PermissionError(f"{label} contains sealed source identifiers")
    provenance = require_mapping(payload.get("provenance"), f"{label}.provenance")
    fold_authority = require_mapping(
        provenance.get("authority"), f"{label}.provenance.authority"
    )
    if fold_authority.get("sha256") != authority["sha256"]:
        raise ValueError(f"{label} authority SHA-256 differs")
    if fold_authority.get("source_pool_split") != "train":
        raise PermissionError(f"{label} provenance is not source-train-only")
    if fold_authority.get("sealed_splits") != authority["sealed_source_splits"]:
        raise ValueError(f"{label} sealed source splits differ")
    if provenance.get("labels_contacts_predictions_or_checkpoints_read") is not False:
        raise PermissionError(f"{label} reports target or checkpoint access")
    if provenance.get("sealed_sequences_read") is not False:
        raise PermissionError(f"{label} reports sealed sequence access")


def _validate_history_config(
    config: Mapping[str, Any],
    fold_sha256: str,
    seed: int,
    authority: Mapping[str, Any],
    expected_source_sha256: Mapping[str, Any],
    shared_contract: Mapping[str, Any],
    model_contract: Mapping[str, Any],
    label: str,
) -> int:
    """Validate one run configuration and return its exact parameter count."""
    validate_config_integrity(config, label)
    reject_true_frozen_flags(config, label)
    if config.get("format_version") != HISTORY_FORMAT_VERSION:
        raise ValueError(f"{label} has an unexpected format version")
    assert_subset_contract(config, shared_contract, f"{label}.shared_contract")
    model = require_mapping(config.get("model"), f"{label}.model")
    assert_subset_contract(model, model_contract, f"{label}.model_contract")
    optimization = require_mapping(config.get("optimization"), f"{label}.optimization")
    if require_integer(optimization.get("seed"), f"{label}.optimization.seed") != seed:
        raise ValueError(f"{label} seed differs from the aggregation manifest")
    if config.get("source_sha256") != expected_source_sha256:
        raise ValueError(f"{label} source SHA-256 contract differs")
    dataset = require_mapping(config.get("dataset"), f"{label}.dataset")
    if dataset.get("split_manifest_sha256") != fold_sha256:
        raise ValueError(f"{label} split SHA-256 differs from its pinned fold")
    if dataset.get("split_format_version") != FOLD_FORMAT_VERSION:
        raise ValueError(f"{label} split format differs from the pinned fold")
    if dataset.get("split_intended_use") != "development_only":
        raise PermissionError(f"{label} split is not development-only")
    if dataset.get("loaded_splits") != ["train", "validation"]:
        raise PermissionError(f"{label} loaded unexpected split memberships")
    if dataset.get("frozen_test_records_selected") is not False:
        raise PermissionError(f"{label} selected frozen test records")
    dataset_authority = require_mapping(
        dataset.get("authority"), f"{label}.dataset.authority"
    )
    if dataset_authority.get("sha256") != authority["sha256"]:
        raise ValueError(f"{label} authority SHA-256 differs")
    if dataset_authority.get("authorized_source_pool_splits") != authority[
        "source_pool_splits"
    ]:
        raise PermissionError(f"{label} authorized source pool differs")
    if dataset_authority.get("sealed_source_splits") != authority[
        "sealed_source_splits"
    ]:
        raise PermissionError(f"{label} sealed source split contract differs")
    if dataset_authority.get("coverage_mode") != authority["coverage_mode"]:
        raise ValueError(f"{label} authority coverage mode differs")
    if require_integer(
        dataset_authority.get("selected_sealed_overlap_count"),
        f"{label}.dataset.authority.selected_sealed_overlap_count",
        0,
    ) != 0:
        raise PermissionError(f"{label} contains sealed source identifiers")
    scope = require_mapping(config.get("scientific_scope"), f"{label}.scientific_scope")
    if scope.get("frozen_test_used") is not False:
        raise PermissionError(f"{label} reports frozen test use")
    if scope.get("model_selection_data") != "development validation only":
        raise PermissionError(f"{label} is outside the development selection scope")
    return exact_trainable_parameter_count(config)


def validate_run(
    manifest_path: Path,
    run: Any,
    variant_name: str,
    model_contract: Mapping[str, Any],
    fold_count: int,
    seed: int,
    authority: Mapping[str, Any],
    expected_source_sha256: Mapping[str, Any],
    shared_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and validate one fold history, summary, and fold definition."""
    entry = require_mapping(run, f"variant {variant_name} run")
    require_exact_keys(
        entry,
        {"fold_index", "fold_manifest", "history", "training_summary"},
        set(),
        f"variant {variant_name} run",
    )
    fold_index = require_integer(entry["fold_index"], "fold_index", 0)
    if fold_index not in REQUIRED_FOLDS:
        raise ValueError(f"Unexpected fold index for {variant_name}: {fold_index}")
    references: dict[str, tuple[Path, str]] = {}
    for key in ("fold_manifest", "history", "training_summary"):
        references[key] = resolve_reference(
            manifest_path, entry[key], f"{variant_name}.fold{fold_index}.{key}"
        )
    fold, fold_sha256 = load_pinned_json(
        *references["fold_manifest"],
        label=f"{variant_name} fold {fold_index} manifest",
    )
    validate_fold_manifest(
        fold,
        fold_index,
        fold_count,
        seed,
        authority,
        f"{variant_name}.fold{fold_index}.fold_manifest",
    )
    history_payload, history_sha256 = load_pinned_json(
        *references["history"], label=f"{variant_name} fold {fold_index} history"
    )
    history_object = require_mapping(
        history_payload, f"{variant_name}.fold{fold_index}.history"
    )
    reject_true_frozen_flags(history_object, f"{variant_name}.fold{fold_index}.history")
    if history_object.get("format_version") != HISTORY_FORMAT_VERSION:
        raise ValueError("History format version is incompatible")
    config = require_mapping(
        history_object.get("config"), f"{variant_name}.fold{fold_index}.history.config"
    )
    parameter_count = _validate_history_config(
        config,
        fold_sha256,
        seed,
        authority,
        expected_source_sha256,
        shared_contract,
        model_contract,
        f"{variant_name}.fold{fold_index}.history.config",
    )
    summary_payload, summary_sha256 = load_pinned_json(
        *references["training_summary"],
        label=f"{variant_name} fold {fold_index} training summary",
    )
    summary = require_mapping(
        summary_payload, f"{variant_name}.fold{fold_index}.training_summary"
    )
    reject_true_frozen_flags(summary, f"{variant_name}.fold{fold_index}.training_summary")
    if summary.get("format_version") != HISTORY_FORMAT_VERSION:
        raise ValueError("Training summary format version is incompatible")
    if summary.get("stage") != "train_complete":
        raise ValueError("Training summary does not report a completed training stage")
    if summary.get("frozen_test_used") is not False:
        raise PermissionError("Training summary reports frozen test use")
    if summary.get("config_sha256") != config.get("config_sha256"):
        raise ValueError("Training summary configuration differs from history")
    best_epoch = require_integer(summary.get("best_epoch"), "best_epoch", 1)
    raw_history = require_list(
        history_object.get("history"), f"{variant_name}.fold{fold_index}.history.history"
    )
    history_records = [
        require_mapping(record, f"history epoch record {index}")
        for index, record in enumerate(raw_history)
    ]
    epoch_numbers = [
        require_integer(record.get("epoch"), f"history epoch record {index}.epoch", 1)
        for index, record in enumerate(history_records)
    ]
    if not epoch_numbers or epoch_numbers != sorted(set(epoch_numbers)):
        raise ValueError("History epochs must be non-empty, unique, and increasing")
    matches = [record for record in history_records if record.get("epoch") == best_epoch]
    if len(matches) != 1:
        raise ValueError("Saved best epoch must occur exactly once in history")
    best_record = matches[0]
    if best_record.get("selection_metric") != "bidirectional_validation_mrr":
        raise ValueError("Best epoch used an unexpected selection metric")
    if require_integer(best_record.get("best_epoch"), "record.best_epoch", 1) != best_epoch:
        raise ValueError("Best history record disagrees with the summary best epoch")
    if best_record.get("improved") is not True:
        raise ValueError("Saved best epoch is not marked as an improved checkpoint")
    validated_metrics = validate_metrics(
        best_record.get("validation"),
        f"{variant_name}.fold{fold_index}.best_epoch_validation",
    )
    summary_mrr = require_finite_float(
        summary.get("best_validation_mrr"), "summary.best_validation_mrr"
    )
    record_best_mrr = require_finite_float(
        best_record.get("best_validation_mrr"), "record.best_validation_mrr"
    )
    observed_mrr = validated_metrics["bidirectional_mrr"]
    if not math.isclose(summary_mrr, observed_mrr, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Training summary MRR differs from best-epoch full-library MRR")
    if not math.isclose(record_best_mrr, observed_mrr, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Best-record running MRR differs from its validation metrics")
    final_record = history_records[-1]
    if require_integer(final_record.get("best_epoch"), "final_record.best_epoch", 1) != best_epoch:
        raise ValueError("Final history state disagrees with the summary best epoch")
    final_best_mrr = require_finite_float(
        final_record.get("best_validation_mrr"), "final_record.best_validation_mrr"
    )
    if not math.isclose(final_best_mrr, summary_mrr, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Final history state disagrees with the summary best MRR")
    completed_epoch = require_integer(
        summary.get("completed_epoch"), "summary.completed_epoch", 1
    )
    if completed_epoch != epoch_numbers[-1]:
        raise ValueError("Training summary completed epoch differs from history")
    chance_fraction = validated_metrics.pop("_chance_fraction")
    return {
        "fold_index": fold_index,
        "best_epoch": best_epoch,
        "validation": validated_metrics,
        "exact_trainable_parameter_count": parameter_count,
        "input_sha256": {
            "fold_manifest": fold_sha256,
            "history": history_sha256,
            "training_summary": summary_sha256,
        },
        "_chance_fraction": chance_fraction,
    }


def _selection_trace(variants: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply and report the fully deterministic predeclared tiebreak sequence."""
    macro_best = max(value["macro_bidirectional_mrr"] for value in variants)
    macro_tied = sorted(
        value["name"]
        for value in variants
        if value["macro_bidirectional_mrr"] == macro_best
    )
    stage = [value for value in variants if value["name"] in macro_tied]
    worst_best = max(value["worst_fold_bidirectional_mrr"] for value in stage)
    worst_tied = sorted(
        value["name"]
        for value in stage
        if value["worst_fold_bidirectional_mrr"] == worst_best
    )
    stage = [value for value in stage if value["name"] in worst_tied]
    smallest = min(value["exact_trainable_parameter_count"] for value in stage)
    parameter_tied = sorted(
        value["name"]
        for value in stage
        if value["exact_trainable_parameter_count"] == smallest
    )
    winner = parameter_tied[0]
    return {
        "selected_variant": winner,
        "criteria_in_order": [
            "highest_unweighted_fold_macro_bidirectional_mrr",
            "highest_worst_fold_bidirectional_mrr",
            "lowest_exact_trainable_parameter_count",
            "lexicographically_smallest_variant_name_on_exact_tie",
        ],
        "tied_after_primary": macro_tied,
        "tied_after_worst_fold": worst_tied,
        "tied_after_parameter_count": parameter_tied,
        "variant_name_tiebreak_used": len(parameter_tied) > 1,
    }


def aggregate_manifest(manifest_path: Path, expected_sha256: str) -> dict[str, Any]:
    """Validate all pinned inputs, aggregate folds, and select one variant."""
    manifest_path = manifest_path.expanduser().resolve()
    manifest_raw, manifest_sha256 = load_pinned_json(
        manifest_path, expected_sha256, "aggregation manifest"
    )
    manifest = require_mapping(manifest_raw, "aggregation manifest")
    require_exact_keys(
        manifest,
        {
            "format_version",
            "intended_use",
            "frozen_test_used",
            "seed",
            "fold_count",
            "authority",
            "expected_source_sha256",
            "shared_config_contract",
            "variants",
        },
        set(),
        "aggregation manifest",
    )
    reject_true_frozen_flags(manifest, "aggregation_manifest")
    if manifest.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError("Aggregation manifest format version is incompatible")
    if manifest.get("intended_use") != "development_only":
        raise PermissionError("Aggregation manifest is not development-only")
    if manifest.get("frozen_test_used") is not False:
        raise PermissionError("Aggregation manifest reports frozen test use")
    seed = require_integer(manifest.get("seed"), "manifest.seed")
    fold_count = require_integer(manifest.get("fold_count"), "manifest.fold_count", 1)
    if fold_count != len(REQUIRED_FOLDS):
        raise ValueError("The locked protocol requires exactly three folds")
    authority = require_mapping(manifest.get("authority"), "manifest.authority")
    require_exact_keys(
        authority,
        {"sha256", "source_pool_splits", "sealed_source_splits", "coverage_mode"},
        set(),
        "manifest.authority",
    )
    authority = dict(authority)
    authority["sha256"] = normalize_sha256(
        authority["sha256"], "manifest.authority.sha256"
    )
    if authority["source_pool_splits"] != ["train"]:
        raise PermissionError("Only the source train split may be used for fold selection")
    if authority["sealed_source_splits"] != ["validation", "test"]:
        raise PermissionError("Source validation and test must remain sealed")
    if authority["coverage_mode"] != "exact":
        raise ValueError("Fold aggregation requires exact source-train coverage")
    expected_source_sha256 = require_mapping(
        manifest.get("expected_source_sha256"), "manifest.expected_source_sha256"
    )
    if not expected_source_sha256:
        raise ValueError("expected_source_sha256 cannot be empty")
    normalized_sources = {
        str(key): normalize_sha256(value, f"expected_source_sha256.{key}")
        for key, value in expected_source_sha256.items()
    }
    shared_contract = require_mapping(
        manifest.get("shared_config_contract"), "manifest.shared_config_contract"
    )
    if not shared_contract:
        raise ValueError("shared_config_contract cannot be empty")
    variants_raw = require_list(manifest.get("variants"), "manifest.variants")
    if len(variants_raw) != len(REQUIRED_VARIANTS):
        raise ValueError("The locked protocol requires exactly three variants")

    variant_results: list[dict[str, Any]] = []
    observed_names: set[str] = set()
    for raw_variant in variants_raw:
        variant = require_mapping(raw_variant, "manifest variant")
        require_exact_keys(
            variant,
            {"name", "model_contract", "runs"},
            set(),
            "manifest variant",
        )
        name = variant.get("name")
        if not isinstance(name, str) or name not in REQUIRED_VARIANTS:
            raise ValueError(f"Unexpected variant name: {name!r}")
        if name in observed_names:
            raise ValueError(f"Duplicate variant name: {name}")
        observed_names.add(name)
        model_contract = require_mapping(
            variant.get("model_contract"), f"variant {name}.model_contract"
        )
        if not model_contract:
            raise ValueError(f"Variant {name} model_contract cannot be empty")
        runs = require_list(variant.get("runs"), f"variant {name}.runs")
        if len(runs) != fold_count:
            raise ValueError(f"Variant {name} must contain exactly three runs")
        validated_runs = [
            validate_run(
                manifest_path,
                run,
                name,
                model_contract,
                fold_count,
                seed,
                authority,
                normalized_sources,
                shared_contract,
            )
            for run in runs
        ]
        fold_indices = {value["fold_index"] for value in validated_runs}
        if fold_indices != REQUIRED_FOLDS:
            raise ValueError(f"Variant {name} does not cover folds 0, 1, and 2 exactly")
        validated_runs.sort(key=lambda value: value["fold_index"])
        parameter_counts = {
            value["exact_trainable_parameter_count"] for value in validated_runs
        }
        if len(parameter_counts) != 1:
            raise ValueError(f"Variant {name} changes trainable parameter count across folds")
        mrrs = [value["validation"]["bidirectional_mrr"] for value in validated_runs]
        chance_fractions = [value.pop("_chance_fraction") for value in validated_runs]
        macro_mrr = math.fsum(mrrs) / fold_count
        macro_chance = sum(chance_fractions, Fraction(0, 1)) / fold_count
        variant_results.append(
            {
                "name": name,
                "folds": validated_runs,
                "macro_bidirectional_mrr": macro_mrr,
                "worst_fold_bidirectional_mrr": min(mrrs),
                "exact_trainable_parameter_count": next(iter(parameter_counts)),
                "macro_exact_random_chance": fraction_payload(macro_chance),
                "macro_excess_over_exact_random_chance": macro_mrr
                - float(macro_chance),
                "macro_multiple_over_exact_random_chance": macro_mrr
                / float(macro_chance),
            }
        )
    if observed_names != REQUIRED_VARIANTS:
        raise ValueError("Aggregation manifest does not contain the locked variant set")
    variant_results.sort(key=lambda value: value["name"])
    selection = _selection_trace(variant_results)
    return {
        "format_version": OUTPUT_FORMAT_VERSION,
        "intended_use": "development_model_selection_only",
        "frozen_test_used": False,
        "selection_statistic": "unweighted_macro_mean_over_three_native_full_validation_libraries",
        "random_chance_method": "exact_finite_library_multi_positive_expectation_from_best_epoch_per_anchor_degrees",
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "format_version": MANIFEST_FORMAT_VERSION,
        },
        "seed": seed,
        "fold_count": fold_count,
        "authority": authority,
        "expected_source_sha256": normalized_sources,
        "variants": variant_results,
        "selection": selection,
        "aggregator_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def canonical_output_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize one deterministic strict JSON output."""
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def atomic_create_or_validate(path: Path, payload: bytes, resume: bool) -> None:
    """Create an immutable output or validate a byte-identical resumed result."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        if path.read_bytes() != payload:
            raise ValueError("Existing output differs from the deterministic aggregation")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line aggregator."""
    args = parse_args(argv)
    result = aggregate_manifest(args.manifest, args.manifest_sha256)
    encoded = canonical_output_bytes(result)
    atomic_create_or_validate(args.output, encoded, args.resume)
    print(encoded.decode("utf-8"), end="", flush=True)


if __name__ == "__main__":
    main()
