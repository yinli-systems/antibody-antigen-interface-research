#!/usr/bin/env python3
"""Publication-grade audits and uncertainty analysis for SPPIDER-seq results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa

from sabdab2_sppider_seq import (
    atomic_write_json,
    atomic_write_parquet,
    binary_metrics,
    mean_optional,
    sha256_file,
)


def load_frozen_layout(benchmark_root: Path) -> tuple[Path, dict[str, list[str]]]:
    """Resolve the frozen table root and fixed SAbDab2 memberships."""
    frozen = (
        benchmark_root.expanduser().resolve()
        / "data/nfs/yangwuyu/baseline-eval/frozen/sabdab2"
    )
    splits = json.loads((frozen / "splits.json").read_text(encoding="utf-8-sig"))[
        "sabdab2"
    ]
    return frozen, {
        "train": [str(value) for value in splits["train"]],
        "test": [str(value) for value in splits["test"]],
    }


def command_audit_split(args: argparse.Namespace) -> None:
    """Audit exact sequence overlap, cluster membership, and source reuse."""
    started = time.time()
    frozen, splits = load_frozen_layout(args.benchmark_root)
    instances = pd.read_parquet(
        frozen / "instances.parquet",
        columns=["instance_id", "ab_seq", "ag_seq", "train_provenance"],
    ).set_index("instance_id", verify_integrity=True)
    manifest = pd.read_csv(args.canonical_manifest).set_index("INSTANCE", verify_integrity=True)
    if set(instances.index) != set(manifest.index):
        raise ValueError("Canonical manifest and frozen instance memberships differ")
    for split_name, identifiers in splits.items():
        if not set(identifiers) <= set(instances.index):
            raise ValueError(f"Frozen {split_name} contains unknown instance IDs")

    train = instances.loc[splits["train"]]
    test = instances.loc[splits["test"]]
    sequence_audits: dict[str, Any] = {}
    for role, column in (("antibody", "ab_seq"), ("antigen", "ag_seq")):
        train_sequences = set(train[column].astype(str))
        test_sequences = set(test[column].astype(str))
        overlap = train_sequences & test_sequences
        sequence_audits[role] = {
            "train_unique_sequence_count": len(train_sequences),
            "test_unique_sequence_count": len(test_sequences),
            "exact_train_test_overlap_count": len(overlap),
            "test_instance_count_with_exact_train_sequence": int(
                test[column].isin(overlap).sum()
            ),
        }

    train_pairs = set(zip(train["ab_seq"], train["ag_seq"], strict=True))
    test_pairs = set(zip(test["ab_seq"], test["ag_seq"], strict=True))
    train_manifest = manifest.loc[splits["train"]]
    test_manifest = manifest.loc[splits["test"]]
    train_clusters = set(train_manifest["ab_ag_cluster"].astype(str))
    test_clusters = set(test_manifest["ab_ag_cluster"].astype(str))
    overlapping_clusters = train_clusters & test_clusters
    result = {
        "stage": "audit_split",
        "train_instance_count": len(train),
        "test_instance_count": len(test),
        "test_source_pdb_count": int(test["train_provenance"].nunique()),
        "train_cluster_count": len(train_clusters),
        "test_cluster_count": len(test_clusters),
        "train_test_cluster_overlap_count": len(overlapping_clusters),
        "test_instance_count_in_overlapping_cluster": int(
            test_manifest["ab_ag_cluster"].astype(str).isin(overlapping_clusters).sum()
        ),
        "sequence_audits": sequence_audits,
        "exact_antibody_antigen_pair_overlap_count": len(train_pairs & test_pairs),
        "test_cluster_size": {
            "minimum": int(test_manifest["ab_ag_cluster"].value_counts().min()),
            "median": float(test_manifest["ab_ag_cluster"].value_counts().median()),
            "maximum": int(test_manifest["ab_ag_cluster"].value_counts().max()),
        },
        "limitations": [
            "Exact sequence disjointness does not establish disjointness at a lower sequence-identity threshold.",
            "A separate MMseqs2 audit is required for thresholded train-test similarity.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def sequence_sha256(sequence: str) -> str:
    """Return a stable sequence identifier without exposing sequence text in reports."""
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def write_unique_fasta(path: Path, sequences: Iterable[str]) -> dict[str, str]:
    """Write one normalized FASTA record per unique protein sequence."""
    normalized = sorted({str(sequence).strip().upper() for sequence in sequences})
    if not normalized or any(not sequence for sequence in normalized):
        raise ValueError("FASTA export requires non-empty protein sequences")
    mapping = {sequence_sha256(sequence): sequence for sequence in normalized}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="ascii") as handle:
        for identifier, sequence in sorted(mapping.items()):
            handle.write(f">{identifier}\n{sequence}\n")
    temporary.replace(path)
    return mapping


def summarize_mmseqs_matches(
    matches: pd.DataFrame,
    test_sequence_ids: set[str],
    test_instance_sequence_ids: pd.Series,
    identity_thresholds: list[float],
    coverage_thresholds: list[float],
) -> dict[str, Any]:
    """Summarize thresholded nearest-neighbor similarity at sequence and instance level."""
    observed_queries = set(matches["query"].astype(str)) if not matches.empty else set()
    if not observed_queries <= test_sequence_ids:
        raise ValueError("MMseqs2 results contain unknown test query identifiers")
    nearest_records: list[dict[str, Any]] = []
    for query in sorted(test_sequence_ids):
        query_matches = matches.loc[matches["query"].astype(str) == query]
        if query_matches.empty:
            nearest_records.append(
                {
                    "query": query,
                    "target": None,
                    "identity_fraction": None,
                    "query_coverage": None,
                    "target_coverage": None,
                    "evalue": None,
                    "bits": None,
                }
            )
            continue
        best_index = query_matches.sort_values(
            ["bits", "fident", "qcov", "tcov"],
            ascending=False,
            kind="stable",
        ).index[0]
        best = query_matches.loc[best_index]
        nearest_records.append(
            {
                "query": query,
                "target": str(best["target"]),
                "identity_fraction": float(best["fident"]),
                "query_coverage": float(best["qcov"]),
                "target_coverage": float(best["tcov"]),
                "evalue": float(best["evalue"]),
                "bits": float(best["bits"]),
            }
        )

    threshold_results: dict[str, Any] = {}
    for coverage in coverage_thresholds:
        coverage_modes = {
            "bidirectional": (matches["qcov"] >= coverage)
            & (matches["tcov"] >= coverage),
            "query": matches["qcov"] >= coverage,
        }
        for coverage_mode, coverage_mask in coverage_modes.items():
            coverage_matches = matches.loc[coverage_mask]
            for identity in identity_thresholds:
                qualifying_queries = set(
                    coverage_matches.loc[
                        coverage_matches["fident"] >= identity,
                        "query",
                    ].astype(str)
                )
                qualifying_instances = int(
                    test_instance_sequence_ids.astype(str).isin(qualifying_queries).sum()
                )
                key = (
                    f"identity_ge_{identity:g}_{coverage_mode}_coverage_ge_{coverage:g}"
                )
                threshold_results[key] = {
                    "unique_test_sequence_count": len(qualifying_queries),
                    "unique_test_sequence_fraction": len(qualifying_queries)
                    / len(test_sequence_ids),
                    "test_instance_count": qualifying_instances,
                    "test_instance_fraction": qualifying_instances
                    / len(test_instance_sequence_ids),
                }
    nearest_identities = [
        record["identity_fraction"]
        for record in nearest_records
        if record["identity_fraction"] is not None
    ]
    return {
        "unique_test_sequence_count": len(test_sequence_ids),
        "test_instance_count": len(test_instance_sequence_ids),
        "query_with_any_match_count": len(observed_queries),
        "nearest_hit_identity_fraction": (
            {
                "minimum": float(np.min(nearest_identities)),
                "median": float(np.median(nearest_identities)),
                "p90": float(np.quantile(nearest_identities, 0.90)),
                "p95": float(np.quantile(nearest_identities, 0.95)),
                "maximum": float(np.max(nearest_identities)),
            }
            if nearest_identities
            else None
        ),
        "thresholds": threshold_results,
        "nearest_hits": nearest_records,
    }


def command_mmseqs_audit(args: argparse.Namespace) -> None:
    """Search every unique test sequence against frozen training sequences."""
    started = time.time()
    frozen, splits = load_frozen_layout(args.benchmark_root)
    instances = pd.read_parquet(
        frozen / "instances.parquet",
        columns=["instance_id", "ab_seq", "ag_seq"],
    ).set_index("instance_id", verify_integrity=True)
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    mmseqs = args.mmseqs.expanduser().resolve()
    if not mmseqs.is_file():
        raise FileNotFoundError(f"MMseqs2 binary is missing: {mmseqs}")
    version = subprocess.run(
        [str(mmseqs), "version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    role_results: dict[str, Any] = {}
    commands: dict[str, list[str]] = {}
    for role, column in (("antibody", "ab_seq"), ("antigen", "ag_seq")):
        train_sequences = instances.loc[splits["train"], column].astype(str)
        test_sequences = instances.loc[splits["test"], column].astype(str)
        train_fasta = work_dir / f"{role}_train.fasta"
        test_fasta = work_dir / f"{role}_test.fasta"
        train_mapping = write_unique_fasta(train_fasta, train_sequences)
        test_mapping = write_unique_fasta(test_fasta, test_sequences)
        result_tsv = work_dir / f"{role}_test_vs_train.tsv"
        temporary_dir = work_dir / f"{role}_tmp"
        command = [
            str(mmseqs),
            "easy-search",
            str(test_fasta),
            str(train_fasta),
            str(result_tsv),
            str(temporary_dir),
            "--format-output",
            "query,target,fident,qcov,tcov,alnlen,qlen,tlen,evalue,bits",
            "--max-seqs",
            str(args.max_hits),
            "-s",
            str(args.sensitivity),
            "-e",
            str(args.evalue),
            "--cov-mode",
            "0",
            "-c",
            "0.0",
            "--threads",
            str(args.threads),
        ]
        subprocess.run(command, check=True)
        commands[role] = command
        columns = [
            "query",
            "target",
            "fident",
            "qcov",
            "tcov",
            "alnlen",
            "qlen",
            "tlen",
            "evalue",
            "bits",
        ]
        if result_tsv.stat().st_size == 0:
            matches = pd.DataFrame(columns=columns)
        else:
            matches = pd.read_csv(result_tsv, sep="\t", names=columns)
        for numeric in columns[2:]:
            matches[numeric] = pd.to_numeric(matches[numeric], errors="raise")
        instance_sequence_ids = test_sequences.map(sequence_sha256)
        role_result = summarize_mmseqs_matches(
            matches,
            set(test_mapping),
            instance_sequence_ids,
            args.identity_thresholds,
            args.coverage_thresholds,
        )
        role_result.update(
            {
                "unique_train_sequence_count": len(train_mapping),
                "alignment_count": len(matches),
            }
        )
        role_results[role] = role_result

    result = {
        "stage": "mmseqs_train_test_similarity_audit",
        "mmseqs_version": version,
        "identity_unit": "fraction",
        "coverage_unit": "fraction",
        "coverage_modes": {
            "bidirectional": "minimum of query and target coverage",
            "query": "test-query coverage only",
        },
        "identity_thresholds": args.identity_thresholds,
        "coverage_thresholds": args.coverage_thresholds,
        "roles": role_results,
        "commands": commands,
        "limitations": [
            "Identity is MMseqs2 alignment identity rather than antibody-numbering-aware CDR identity.",
            "A separate CDR-H3 audit is required after antibody numbering is available.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def numeric_distribution(values: Iterable[float]) -> dict[str, float]:
    """Summarize a finite numeric distribution with robust quantiles."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Distribution values must be non-empty and finite")
    return {
        "minimum": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def command_audit_prepared(args: argparse.Namespace) -> None:
    """Audit grouped validation isolation and split-level label distributions."""
    started = time.time()
    prepared = pd.read_parquet(args.prepared_instances).set_index(
        "instance_id", verify_integrity=True
    )
    manifest = pd.read_csv(
        args.canonical_manifest,
        usecols=["INSTANCE", "ab_ag_cluster"],
    ).set_index("INSTANCE", verify_integrity=True)
    if set(prepared.index) != set(manifest.index):
        raise ValueError("Prepared and canonical manifest memberships differ")
    cluster_by_instance = manifest["ab_ag_cluster"].astype(str)

    roles: dict[str, Any] = {}
    for role, query_key, context_key, length_column, positive_column in (
        (
            "paratope",
            "ab_sequence_sha256",
            "ag_sequence_sha256",
            "ab_length",
            "paratope_positive_count",
        ),
        (
            "epitope",
            "ag_sequence_sha256",
            "ab_sequence_sha256",
            "ag_length",
            "epitope_positive_count",
        ),
    ):
        split_column = f"{role}_split"
        split_results: dict[str, Any] = {}
        split_queries: dict[str, set[str]] = {}
        split_clusters: dict[str, set[str]] = {}
        for split_name in ("train", "validation", "test"):
            subset = prepared.loc[prepared[split_column] == split_name].copy()
            if subset.empty:
                raise ValueError(f"Prepared {role} split {split_name} is empty")
            query_values = set(subset[query_key].astype(str))
            cluster_values = set(cluster_by_instance.loc[subset.index])
            split_queries[split_name] = query_values
            split_clusters[split_name] = cluster_values
            positive_counts = subset[positive_column].to_numpy(dtype=np.int64)
            residue_counts = subset[length_column].to_numpy(dtype=np.int64)
            instance_fractions = positive_counts / residue_counts
            split_results[split_name] = {
                "instance_count": len(subset),
                "cluster_count": len(cluster_values),
                "unique_query_sequence_count": len(query_values),
                "unique_context_sequence_count": int(subset[context_key].nunique()),
                "residue_count": int(residue_counts.sum()),
                "positive_residue_count": int(positive_counts.sum()),
                "micro_positive_fraction": float(positive_counts.sum() / residue_counts.sum()),
                "instance_positive_fraction": numeric_distribution(instance_fractions),
                "sequence_length": numeric_distribution(residue_counts),
            }
        overlaps = {}
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlaps[f"{left}_{right}"] = {
                "query_sequence_overlap_count": len(
                    split_queries[left] & split_queries[right]
                ),
                "cluster_overlap_count": len(
                    split_clusters[left] & split_clusters[right]
                ),
            }
        roles[role] = {"splits": split_results, "overlaps": overlaps}

    result = {
        "stage": "audit_prepared_splits",
        "prepared_instances": str(args.prepared_instances.expanduser().resolve()),
        "roles": roles,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def command_audit_components(args: argparse.Namespace) -> None:
    """Describe train components induced by canonical clusters and focal queries."""
    frozen, splits = load_frozen_layout(args.benchmark_root)
    instances = pd.read_parquet(
        frozen / "instances.parquet",
        columns=["instance_id", "ab_seq", "ag_seq"],
    ).set_index("instance_id", verify_integrity=True)
    manifest = pd.read_csv(
        args.canonical_manifest,
        usecols=["INSTANCE", "ab_ag_cluster"],
    ).set_index("INSTANCE", verify_integrity=True)
    train_ids = [str(value) for value in splits["train"]]
    if not set(train_ids) <= set(manifest.index):
        raise ValueError("Canonical manifest is missing frozen train instances")

    role_results: dict[str, Any] = {}
    for role, sequence_column in (("paratope", "ab_seq"), ("epitope", "ag_seq")):
        parent = {identifier: identifier for identifier in train_ids}

        def find(identifier: str) -> str:
            while parent[identifier] != identifier:
                parent[identifier] = parent[parent[identifier]]
                identifier = parent[identifier]
            return identifier

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        first_by_sequence: dict[str, str] = {}
        first_by_cluster: dict[str, str] = {}
        for identifier in train_ids:
            sequence = str(instances.at[identifier, sequence_column])
            cluster = str(manifest.at[identifier, "ab_ag_cluster"])
            union(identifier, first_by_sequence.setdefault(sequence, identifier))
            union(identifier, first_by_cluster.setdefault(cluster, identifier))
        members: dict[str, list[str]] = {}
        for identifier in train_ids:
            members.setdefault(find(identifier), []).append(identifier)
        records = []
        for component in members.values():
            clusters = set(manifest.loc[component, "ab_ag_cluster"].astype(str))
            records.append(
                {
                    "instance_count": len(component),
                    "cluster_count": len(clusters),
                    "query_sequence_count": int(
                        instances.loc[component, sequence_column].nunique()
                    ),
                    "member_digest": hashlib.sha256(
                        ",".join(sorted(component)).encode("utf-8")
                    ).hexdigest(),
                }
            )
        records.sort(
            key=lambda record: (
                -record["instance_count"],
                -record["cluster_count"],
                record["member_digest"],
            )
        )
        role_results[role] = {
            "component_count": len(records),
            "instance_count": len(train_ids),
            "canonical_cluster_count": int(
                manifest.loc[train_ids, "ab_ag_cluster"].nunique()
            ),
            "component_instance_count": numeric_distribution(
                record["instance_count"] for record in records
            ),
            "component_cluster_count": numeric_distribution(
                record["cluster_count"] for record in records
            ),
            "largest_components": records[:20],
        }

    result = {
        "stage": "audit_validation_components",
        "roles": role_results,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def exact_threshold_optimum(
    labels: np.ndarray,
    scores: np.ndarray,
    objective_name: str,
) -> tuple[float, float]:
    """Find the exact observed-score threshold maximizing MCC or F1."""
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if objective_name not in {"mcc", "f1"}:
        raise ValueError(f"Unsupported threshold objective: {objective_name}")
    if len(labels) == 0 or len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Threshold selection requires aligned finite labels and scores")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positive_total = int(labels.sum())
    negative_total = len(labels) - positive_total
    true_positive = 0
    false_positive = 0
    best_objective = -math.inf
    best_threshold = 0.5
    position = 0
    while position < len(labels):
        end = position + 1
        while end < len(labels) and sorted_scores[end] == sorted_scores[position]:
            end += 1
        group = sorted_labels[position:end]
        true_positive += int(group.sum())
        false_positive += len(group) - int(group.sum())
        false_negative = positive_total - true_positive
        true_negative = negative_total - false_positive
        if objective_name == "f1":
            denominator = 2 * true_positive + false_positive + false_negative
            objective = 2 * true_positive / denominator if denominator else 0.0
        else:
            denominator = math.sqrt(
                (true_positive + false_positive)
                * (true_positive + false_negative)
                * (true_negative + false_positive)
                * (true_negative + false_negative)
            )
            numerator = true_positive * true_negative - false_positive * false_negative
            objective = numerator / denominator if denominator else 0.0
        threshold = float(sorted_scores[position])
        if (
            objective > best_objective
            or (
                objective == best_objective
                and abs(threshold - 0.5) < abs(best_threshold - 0.5)
            )
        ):
            best_objective = float(objective)
            best_threshold = threshold
        position = end
    return best_threshold, best_objective


def command_select_threshold(args: argparse.Namespace) -> None:
    """Select decision thresholds exclusively from validation predictions."""
    predictions = pd.read_parquet(args.residue_predictions)
    required = {"instance_id", "role", "res_idx", "label", "score"}
    if set(predictions.columns) != required:
        raise ValueError("Validation prediction columns differ from the required contract")
    if predictions["role"].nunique() != 1 or predictions["role"].iloc[0] != args.role:
        raise ValueError("Validation prediction role does not match the requested role")
    labels = predictions["label"].to_numpy(dtype=np.uint8)
    scores = predictions["score"].to_numpy(dtype=np.float64)
    optimized = {}
    for objective_name in ("mcc", "f1"):
        threshold, objective_value = exact_threshold_optimum(
            labels,
            scores,
            objective_name,
        )
        optimized[objective_name] = {
            "threshold": threshold,
            "objective_value": objective_value,
            "metrics": binary_metrics(labels, scores, threshold),
        }
    result = {
        "stage": "select_validation_threshold",
        "role": args.role,
        "selection_split": "validation",
        "residue_predictions": str(args.residue_predictions.expanduser().resolve()),
        "fixed_0_5_metrics": binary_metrics(labels, scores, 0.5),
        "optimized": optimized,
        "policy": (
            "Use the MCC-optimal validation threshold for the primary frozen-test "
            "thresholded metrics; retain 0.5 as the official fixed-threshold comparison."
        ),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def command_ensemble_predictions(args: argparse.Namespace) -> None:
    """Create a fixed weighted ensemble from aligned residue predictions."""
    paths = [path.expanduser().resolve() for path in args.inputs]
    if len(paths) < 2:
        raise ValueError("A prediction ensemble requires at least two inputs")
    if args.weights is None:
        weights = np.full(len(paths), 1.0 / len(paths), dtype=np.float64)
    else:
        weights = np.asarray(args.weights, dtype=np.float64)
        if weights.shape != (len(paths),):
            raise ValueError("Ensemble weight count does not match input count")
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Ensemble weights must be finite, non-negative, and non-zero")
        weights = weights / weights.sum()

    key_columns = ["instance_id", "role", "res_idx"]
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        required = {"instance_id", "role", "res_idx", "label", "score"}
        if set(frame.columns) != required:
            raise ValueError(f"Prediction columns differ from the contract in {path}")
        frame = frame.sort_values(key_columns, kind="stable").reset_index(drop=True)
        if frame["role"].nunique() != 1 or frame["role"].iloc[0] != args.role:
            raise ValueError(f"Prediction role mismatch in {path}")
        if not np.isfinite(frame["score"]).all():
            raise ValueError(f"Non-finite prediction score in {path}")
        frames.append(frame)
    reference = frames[0]
    for frame in frames[1:]:
        if not reference[key_columns].equals(frame[key_columns]):
            raise ValueError("Ensemble prediction keys are not exactly aligned")
        if not reference["label"].equals(frame["label"]):
            raise ValueError("Ensemble prediction labels disagree")

    score_matrix = np.stack(
        [frame["score"].to_numpy(dtype=np.float64) for frame in frames],
        axis=0,
    )
    output_frame = reference.copy()
    output_frame["score"] = np.average(score_matrix, axis=0, weights=weights).astype(
        np.float32
    )
    output_path = args.output_predictions.expanduser().resolve()
    atomic_write_parquet(
        pa.Table.from_pandas(output_frame, preserve_index=False),
        output_path,
    )
    labels = output_frame["label"].to_numpy(dtype=np.uint8)
    scores = output_frame["score"].to_numpy(dtype=np.float64)
    result = {
        "stage": "ensemble_residue_predictions",
        "role": args.role,
        "input_count": len(paths),
        "inputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "weight": float(weight),
            }
            for path, weight in zip(paths, weights, strict=True)
        ],
        "output_predictions": str(output_path),
        "output_predictions_sha256": sha256_file(output_path),
        "evaluation_threshold": args.threshold,
        "thresholded_metrics": binary_metrics(labels, scores, args.threshold),
    }
    atomic_write_json(args.output_summary, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def command_similarity_performance(args: argparse.Namespace) -> None:
    """Relate frozen-test performance to nearest training-sequence identity."""
    from scipy.stats import spearmanr

    predictions = pd.read_parquet(args.residue_predictions)
    required = {"instance_id", "role", "res_idx", "label", "score"}
    if set(predictions.columns) != required:
        raise ValueError("Residue prediction columns differ from the required contract")
    if predictions["role"].nunique() != 1 or predictions["role"].iloc[0] != args.role:
        raise ValueError("Residue prediction role does not match the requested role")
    frozen, splits = load_frozen_layout(args.benchmark_root)
    sequence_column = "ab_seq" if args.role == "paratope" else "ag_seq"
    similarity_role = "antibody" if args.role == "paratope" else "antigen"
    instances = pd.read_parquet(
        frozen / "instances.parquet",
        columns=["instance_id", sequence_column],
    ).set_index("instance_id", verify_integrity=True)
    if set(predictions["instance_id"].astype(str)) != set(splits["test"]):
        raise ValueError("Residue predictions do not match frozen test membership")
    similarity = json.loads(args.similarity_audit.read_text(encoding="utf-8"))
    nearest_by_query = {
        str(record["query"]): record
        for record in similarity["roles"][similarity_role]["nearest_hits"]
    }
    qualifying_identity_by_query = None
    if args.matches_tsv is not None:
        match_columns = [
            "query",
            "target",
            "fident",
            "qcov",
            "tcov",
            "alnlen",
            "qlen",
            "tlen",
            "evalue",
            "bits",
        ]
        matches = pd.read_csv(args.matches_tsv, sep="\t", names=match_columns)
        for numeric in match_columns[2:]:
            matches[numeric] = pd.to_numeric(matches[numeric], errors="raise")
        if args.coverage_mode == "bidirectional":
            coverage_mask = (matches["qcov"] >= args.coverage_threshold) & (
                matches["tcov"] >= args.coverage_threshold
            )
        else:
            coverage_mask = matches["qcov"] >= args.coverage_threshold
        qualifying_identity_by_query = (
            matches.loc[coverage_mask]
            .groupby("query", sort=False)["fident"]
            .max()
            .astype(float)
            .to_dict()
        )

    instance_records = []
    for instance_id, group in predictions.groupby("instance_id", sort=True):
        metrics = binary_metrics(
            group["label"].to_numpy(dtype=np.uint8),
            group["score"].to_numpy(dtype=np.float64),
            args.threshold,
        )
        query_key = sequence_sha256(str(instances.at[str(instance_id), sequence_column]))
        identity = None
        if qualifying_identity_by_query is not None:
            if query_key in qualifying_identity_by_query:
                identity = float(qualifying_identity_by_query[query_key])
        else:
            nearest = nearest_by_query.get(query_key)
            if nearest is not None and nearest["identity_fraction"] is not None:
                if args.coverage_mode == "bidirectional":
                    coverage = min(
                        float(nearest["query_coverage"]),
                        float(nearest["target_coverage"]),
                    )
                else:
                    coverage = float(nearest["query_coverage"])
                if coverage >= args.coverage_threshold:
                    identity = float(nearest["identity_fraction"])
        instance_records.append(
            {
                "instance_id": str(instance_id),
                "nearest_identity_fraction": identity,
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
                "positive_fraction": metrics["positive_fraction"],
            }
        )
    instance_frame = pd.DataFrame(instance_records)
    finite = instance_frame.dropna(subset=["nearest_identity_fraction"]).copy()
    correlations = {}
    for metric_name in ("auroc", "auprc", "f1", "mcc"):
        paired = finite.dropna(subset=[metric_name])
        if len(paired) < 3:
            correlations[metric_name] = None
            continue
        statistic = spearmanr(
            paired["nearest_identity_fraction"],
            paired[metric_name],
        )
        if not np.isfinite(statistic.statistic) or not np.isfinite(statistic.pvalue):
            correlations[metric_name] = None
        else:
            correlations[metric_name] = {
                "rho": float(statistic.statistic),
                "pvalue": float(statistic.pvalue),
                "instance_count": len(paired),
            }

    bins = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.000001]
    labels = ["[0.0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0]"]
    finite["identity_bin"] = pd.cut(
        finite["nearest_identity_fraction"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    stratified = {}
    for label in labels:
        instance_ids = set(finite.loc[finite["identity_bin"] == label, "instance_id"])
        if not instance_ids:
            continue
        subset = predictions.loc[predictions["instance_id"].isin(instance_ids)]
        stratified[label] = {
            "instance_count": len(instance_ids),
            "residue_metrics": binary_metrics(
                subset["label"].to_numpy(dtype=np.uint8),
                subset["score"].to_numpy(dtype=np.float64),
                args.threshold,
            ),
        }
    result = {
        "stage": "similarity_performance_audit",
        "role": args.role,
        "coverage_mode": args.coverage_mode,
        "coverage_threshold": args.coverage_threshold,
        "identity_source": (
            "maximum_identity_among_coverage_qualifying_mmseqs_hits"
            if qualifying_identity_by_query is not None
            else "bit_score_top_hit_from_similarity_audit"
        ),
        "test_instance_count": predictions["instance_id"].nunique(),
        "instance_with_qualifying_neighbor_count": len(finite),
        "spearman_correlations": correlations,
        "identity_strata": stratified,
        "instance_records": instance_records,
        "interpretation_guardrail": (
            "Association with nearest-neighbor identity is descriptive and does not establish "
            "that the model memorized training labels."
        ),
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def calibration_metrics(labels: np.ndarray, scores: np.ndarray, bins: int) -> dict[str, Any]:
    """Compute Brier score and equal-mass expected calibration error."""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if bins < 2 or bins > len(labels):
        raise ValueError("Calibration bins must fall within [2, sample_count]")
    order = np.argsort(scores, kind="stable")
    groups = np.array_split(order, bins)
    records = []
    expected_calibration_error = 0.0
    for group in groups:
        mean_score = float(scores[group].mean())
        positive_fraction = float(labels[group].mean())
        weight = len(group) / len(labels)
        expected_calibration_error += weight * abs(mean_score - positive_fraction)
        records.append(
            {
                "count": len(group),
                "score_mean": mean_score,
                "positive_fraction": positive_fraction,
            }
        )
    return {
        "brier_score": float(np.mean(np.square(scores - labels))),
        "equal_mass_bin_count": bins,
        "expected_calibration_error": float(expected_calibration_error),
        "bins": records,
    }


def finite_metric_mean(records: Iterable[dict[str, Any]], name: str) -> float | None:
    """Average one optional metric across instance records."""
    return mean_optional(record[name] for record in records)


def summarize_distribution(values: list[float]) -> dict[str, float]:
    """Return percentile confidence limits and bootstrap dispersion."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap metric distribution is empty or non-finite")
    return {
        "mean": float(array.mean()),
        "standard_error": float(array.std(ddof=1)),
        "ci95_lower": float(np.quantile(array, 0.025)),
        "ci95_upper": float(np.quantile(array, 0.975)),
    }


def command_cluster_bootstrap(args: argparse.Namespace) -> None:
    """Estimate cluster-resampled confidence intervals for residue metrics."""
    started = time.time()
    predictions = pd.read_parquet(args.residue_predictions)
    required = {"instance_id", "role", "res_idx", "label", "score"}
    if set(predictions.columns) != required:
        raise ValueError(
            f"Residue prediction columns differ from the required contract: {predictions.columns}"
        )
    if predictions["role"].nunique() != 1 or predictions["role"].iloc[0] != args.role:
        raise ValueError("Residue prediction role does not match the requested bootstrap role")
    if predictions["score"].isna().any() or not np.isfinite(predictions["score"]).all():
        raise ValueError("Residue predictions contain null or non-finite scores")

    manifest = pd.read_csv(
        args.canonical_manifest,
        usecols=["INSTANCE", "split", "ab_ag_cluster"],
    )
    test_manifest = manifest.loc[manifest["split"] == "test"].copy()
    cluster_by_instance = dict(
        zip(
            test_manifest["INSTANCE"].astype(str),
            test_manifest["ab_ag_cluster"].astype(str),
            strict=True,
        )
    )
    prediction_ids = set(predictions["instance_id"].astype(str))
    if prediction_ids != set(cluster_by_instance):
        raise ValueError("Prediction and canonical test instance memberships differ")
    predictions["cluster"] = predictions["instance_id"].map(cluster_by_instance)

    labels = predictions["label"].to_numpy(dtype=np.uint8)
    scores = predictions["score"].to_numpy(dtype=np.float64)
    point_micro = binary_metrics(labels, scores, args.threshold)
    instance_records = []
    for instance_id, group in predictions.groupby("instance_id", sort=True):
        record = binary_metrics(
            group["label"].to_numpy(dtype=np.uint8),
            group["score"].to_numpy(dtype=np.float64),
            args.threshold,
        )
        record["instance_id"] = str(instance_id)
        record["cluster"] = cluster_by_instance[str(instance_id)]
        instance_records.append(record)
    point_macro = {
        name: finite_metric_mean(instance_records, name)
        for name in ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "mcc")
    }

    cluster_row_indices = {
        str(cluster): group.index.to_numpy(dtype=np.int64)
        for cluster, group in predictions.groupby("cluster", sort=True)
    }
    instance_frame = pd.DataFrame(instance_records)
    cluster_instance_indices = {
        str(cluster): group.index.to_numpy(dtype=np.int64)
        for cluster, group in instance_frame.groupby("cluster", sort=True)
    }
    clusters = np.asarray(sorted(cluster_row_indices), dtype=object)
    rng = np.random.default_rng(args.seed)
    metric_names = ("auroc", "auprc", "f1", "mcc")
    micro_distributions: dict[str, list[float]] = {name: [] for name in metric_names}
    macro_distributions: dict[str, list[float]] = {name: [] for name in metric_names}
    for replicate in range(1, args.replicates + 1):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        row_indices = np.concatenate([cluster_row_indices[str(cluster)] for cluster in sampled])
        sampled_micro = binary_metrics(labels[row_indices], scores[row_indices], args.threshold)
        instance_indices = np.concatenate(
            [cluster_instance_indices[str(cluster)] for cluster in sampled]
        )
        sampled_instances = instance_frame.iloc[instance_indices]
        for name in metric_names:
            micro_value = sampled_micro[name]
            macro_values = pd.to_numeric(sampled_instances[name], errors="coerce")
            if micro_value is None or not math.isfinite(float(micro_value)):
                raise ValueError(f"Undefined bootstrap micro metric {name}")
            micro_distributions[name].append(float(micro_value))
            macro_distributions[name].append(float(macro_values.dropna().mean()))
        if replicate % 100 == 0 or replicate == args.replicates:
            print(f"Bootstrap {replicate}/{args.replicates}", flush=True)

    result = {
        "stage": "cluster_bootstrap",
        "role": args.role,
        "residue_prediction_path": str(args.residue_predictions.resolve()),
        "residue_count": len(predictions),
        "instance_count": predictions["instance_id"].nunique(),
        "cluster_count": len(clusters),
        "replicates": args.replicates,
        "seed": args.seed,
        "threshold": args.threshold,
        "point_micro": point_micro,
        "point_macro_mean": point_macro,
        "cluster_bootstrap_micro": {
            name: summarize_distribution(values)
            for name, values in micro_distributions.items()
        },
        "cluster_bootstrap_macro_mean": {
            name: summarize_distribution(values)
            for name, values in macro_distributions.items()
        },
        "calibration": calibration_metrics(labels, scores, args.calibration_bins),
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def command_paired_cluster_bootstrap(args: argparse.Namespace) -> None:
    """Compare two aligned prediction sets with paired cluster resampling."""
    started = time.time()
    key_columns = ["instance_id", "role", "res_idx"]
    frames: dict[str, pd.DataFrame] = {}
    for name, path in (
        ("candidate", args.candidate_predictions),
        ("reference", args.reference_predictions),
    ):
        frame = pd.read_parquet(path)
        required = {"instance_id", "role", "res_idx", "label", "score"}
        if set(frame.columns) != required:
            raise ValueError(f"{name.capitalize()} prediction columns violate the contract")
        frame = frame.sort_values(key_columns, kind="stable").reset_index(drop=True)
        if frame["role"].nunique() != 1 or frame["role"].iloc[0] != args.role:
            raise ValueError(f"{name.capitalize()} prediction role does not match {args.role}")
        if frame["score"].isna().any() or not np.isfinite(frame["score"]).all():
            raise ValueError(f"{name.capitalize()} predictions contain non-finite scores")
        frames[name] = frame

    candidate = frames["candidate"]
    reference = frames["reference"]
    if not candidate[key_columns].equals(reference[key_columns]):
        raise ValueError("Candidate and reference prediction keys are not exactly aligned")
    if not np.array_equal(
        candidate["label"].to_numpy(dtype=np.uint8),
        reference["label"].to_numpy(dtype=np.uint8),
    ):
        raise ValueError("Candidate and reference prediction labels disagree")

    manifest = pd.read_csv(
        args.canonical_manifest,
        usecols=["INSTANCE", "split", "ab_ag_cluster"],
    )
    test_manifest = manifest.loc[manifest["split"] == "test"].copy()
    cluster_by_instance = dict(
        zip(
            test_manifest["INSTANCE"].astype(str),
            test_manifest["ab_ag_cluster"].astype(str),
            strict=True,
        )
    )
    prediction_ids = set(candidate["instance_id"].astype(str))
    if prediction_ids != set(cluster_by_instance):
        raise ValueError("Prediction and canonical test instance memberships differ")
    candidate["cluster"] = candidate["instance_id"].map(cluster_by_instance)

    labels = candidate["label"].to_numpy(dtype=np.uint8)
    candidate_scores = candidate["score"].to_numpy(dtype=np.float64)
    reference_scores = reference["score"].to_numpy(dtype=np.float64)
    candidate_point = binary_metrics(labels, candidate_scores, args.candidate_threshold)
    reference_point = binary_metrics(labels, reference_scores, args.reference_threshold)
    metric_names = ("auroc", "auprc", "f1", "mcc")
    point_differences = {
        name: float(candidate_point[name]) - float(reference_point[name])
        for name in metric_names
    }

    cluster_row_indices = {
        str(cluster): group.index.to_numpy(dtype=np.int64)
        for cluster, group in candidate.groupby("cluster", sort=True)
    }
    clusters = np.asarray(sorted(cluster_row_indices), dtype=object)
    rng = np.random.default_rng(args.seed)
    differences: dict[str, list[float]] = {name: [] for name in metric_names}
    for replicate in range(1, args.replicates + 1):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        row_indices = np.concatenate(
            [cluster_row_indices[str(cluster)] for cluster in sampled]
        )
        sampled_labels = labels[row_indices]
        candidate_metrics = binary_metrics(
            sampled_labels,
            candidate_scores[row_indices],
            args.candidate_threshold,
        )
        reference_metrics = binary_metrics(
            sampled_labels,
            reference_scores[row_indices],
            args.reference_threshold,
        )
        for name in metric_names:
            candidate_value = candidate_metrics[name]
            reference_value = reference_metrics[name]
            if candidate_value is None or reference_value is None:
                raise ValueError(f"Undefined paired bootstrap metric {name}")
            differences[name].append(float(candidate_value) - float(reference_value))
        if replicate % 100 == 0 or replicate == args.replicates:
            print(f"Paired bootstrap {replicate}/{args.replicates}", flush=True)

    difference_summary = {}
    for name, values in differences.items():
        summary = summarize_distribution(values)
        array = np.asarray(values, dtype=np.float64)
        summary.update(
            {
                "probability_greater_than_zero": float(np.mean(array > 0.0)),
                "probability_less_than_zero": float(np.mean(array < 0.0)),
            }
        )
        difference_summary[name] = summary
    result = {
        "stage": "paired_cluster_bootstrap",
        "role": args.role,
        "candidate_name": args.candidate_name,
        "reference_name": args.reference_name,
        "candidate_prediction_path": str(args.candidate_predictions.resolve()),
        "reference_prediction_path": str(args.reference_predictions.resolve()),
        "candidate_prediction_sha256": sha256_file(args.candidate_predictions),
        "reference_prediction_sha256": sha256_file(args.reference_predictions),
        "residue_count": len(candidate),
        "instance_count": candidate["instance_id"].nunique(),
        "cluster_count": len(clusters),
        "replicates": args.replicates,
        "seed": args.seed,
        "candidate_threshold": args.candidate_threshold,
        "reference_threshold": args.reference_threshold,
        "candidate_point_metrics": candidate_point,
        "reference_point_metrics": reference_point,
        "point_differences_candidate_minus_reference": point_differences,
        "paired_cluster_bootstrap_differences": difference_summary,
        "interpretation": (
            "Positive differences favor the candidate. Confidence intervals preserve "
            "the pairing of predictions and resample canonical test clusters rather "
            "than individual residues."
        ),
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Build the research analysis command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-split")
    audit.add_argument("--benchmark-root", type=Path, required=True)
    audit.add_argument("--canonical-manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(handler=command_audit_split)

    similarity = subparsers.add_parser("mmseqs-audit")
    similarity.add_argument("--benchmark-root", type=Path, required=True)
    similarity.add_argument("--mmseqs", type=Path, required=True)
    similarity.add_argument("--work-dir", type=Path, required=True)
    similarity.add_argument("--output", type=Path, required=True)
    similarity.add_argument(
        "--identity-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.8, 0.9, 0.95],
    )
    similarity.add_argument(
        "--coverage-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 0.9],
    )
    similarity.add_argument("--sensitivity", type=float, default=7.5)
    similarity.add_argument("--evalue", type=float, default=1000.0)
    similarity.add_argument("--max-hits", type=int, default=10000)
    similarity.add_argument("--threads", type=int, default=8)
    similarity.set_defaults(handler=command_mmseqs_audit)

    prepared = subparsers.add_parser("audit-prepared")
    prepared.add_argument("--prepared-instances", type=Path, required=True)
    prepared.add_argument("--canonical-manifest", type=Path, required=True)
    prepared.add_argument("--output", type=Path, required=True)
    prepared.set_defaults(handler=command_audit_prepared)

    components = subparsers.add_parser("audit-components")
    components.add_argument("--benchmark-root", type=Path, required=True)
    components.add_argument("--canonical-manifest", type=Path, required=True)
    components.add_argument("--output", type=Path, required=True)
    components.set_defaults(handler=command_audit_components)

    threshold = subparsers.add_parser("select-threshold")
    threshold.add_argument("--residue-predictions", type=Path, required=True)
    threshold.add_argument("--role", choices=("paratope", "epitope"), required=True)
    threshold.add_argument("--output", type=Path, required=True)
    threshold.set_defaults(handler=command_select_threshold)

    ensemble = subparsers.add_parser("ensemble-predictions")
    ensemble.add_argument("--inputs", type=Path, nargs="+", required=True)
    ensemble.add_argument("--weights", type=float, nargs="+")
    ensemble.add_argument("--role", choices=("paratope", "epitope"), required=True)
    ensemble.add_argument("--threshold", type=float, default=0.5)
    ensemble.add_argument("--output-predictions", type=Path, required=True)
    ensemble.add_argument("--output-summary", type=Path, required=True)
    ensemble.set_defaults(handler=command_ensemble_predictions)

    similarity_performance = subparsers.add_parser("similarity-performance")
    similarity_performance.add_argument("--residue-predictions", type=Path, required=True)
    similarity_performance.add_argument("--benchmark-root", type=Path, required=True)
    similarity_performance.add_argument("--similarity-audit", type=Path, required=True)
    similarity_performance.add_argument("--matches-tsv", type=Path)
    similarity_performance.add_argument(
        "--role", choices=("paratope", "epitope"), required=True
    )
    similarity_performance.add_argument(
        "--coverage-mode", choices=("bidirectional", "query"), default="bidirectional"
    )
    similarity_performance.add_argument("--coverage-threshold", type=float, default=0.8)
    similarity_performance.add_argument("--threshold", type=float, required=True)
    similarity_performance.add_argument("--output", type=Path, required=True)
    similarity_performance.set_defaults(handler=command_similarity_performance)

    bootstrap = subparsers.add_parser("cluster-bootstrap")
    bootstrap.add_argument("--residue-predictions", type=Path, required=True)
    bootstrap.add_argument("--canonical-manifest", type=Path, required=True)
    bootstrap.add_argument("--role", choices=("paratope", "epitope"), required=True)
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument("--replicates", type=int, default=1000)
    bootstrap.add_argument("--seed", type=int, default=20260722)
    bootstrap.add_argument("--threshold", type=float, default=0.50)
    bootstrap.add_argument("--calibration-bins", type=int, default=15)
    bootstrap.set_defaults(handler=command_cluster_bootstrap)

    paired = subparsers.add_parser("paired-cluster-bootstrap")
    paired.add_argument("--candidate-predictions", type=Path, required=True)
    paired.add_argument("--reference-predictions", type=Path, required=True)
    paired.add_argument("--canonical-manifest", type=Path, required=True)
    paired.add_argument("--role", choices=("paratope", "epitope"), required=True)
    paired.add_argument("--candidate-name", required=True)
    paired.add_argument("--reference-name", required=True)
    paired.add_argument("--candidate-threshold", type=float, required=True)
    paired.add_argument("--reference-threshold", type=float, required=True)
    paired.add_argument("--replicates", type=int, default=1000)
    paired.add_argument("--seed", type=int, default=20260723)
    paired.add_argument("--output", type=Path, required=True)
    paired.set_defaults(handler=command_paired_cluster_bootstrap)
    return parser


def main() -> None:
    """Parse arguments and execute one research analysis stage."""
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
