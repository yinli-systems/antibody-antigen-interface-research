#!/usr/bin/env python3
"""Exhaustively audit train-only antigen folds with Needleman-Wunsch.

The authority validation and test IDs are used only as an exclusion set. This
program parses antigen sequences exclusively for authority-train identifiers,
audits every unique train-by-validation antigen pair in each fold, and exports
all threshold-crossing digest edges for deterministic component rebuilding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_train_only_antigen_folds as builder
from needleman_wunsch_development_audit import (
    AuditPolicy,
    _dependency_versions,
    _pair_payload,
    audit_role,
)


FORMAT_VERSION = "sabdab2_train_only_antigen_nw_audit_v1"
EDGE_FORMAT_VERSION = "sabdab2_train_only_antigen_nw_edges_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the antigen-only audit command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-split", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--fold-index", type=Path, required=True)
    parser.add_argument("--expected-fold-index-sha256", required=True)
    parser.add_argument(
        "--expected-authority-sha256",
        default=builder.EXPECTED_AUTHORITY_SHA256,
    )
    parser.add_argument("--prior-edge-tsv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.80)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--gap-open", type=float, default=10.0)
    parser.add_argument("--gap-extend", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-pair-count", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for name in ("expected_fold_index_sha256", "expected_authority_sha256"):
        digest = str(getattr(args, name)).lower()
        if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
            parser.error(f"--{name.replace('_', '-')} must be a SHA-256 digest")
        setattr(args, name, digest)
    for name in ("identity_threshold", "coverage_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be within [0, 1]")
    if args.gap_open <= 0 or args.gap_extend <= 0:
        parser.error("Gap penalties must be positive magnitudes")
    if args.workers <= 0 or args.chunk_size <= 0 or args.top_pair_count <= 0:
        parser.error("Workers, chunk size, and top-pair count must be positive")
    return args


def load_source_train_antigens(
    path: Path,
    allowed_ids: set[str],
) -> dict[str, str]:
    """Parse antigen sequences only after source-train ID authorization."""
    records: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError("Canonical manifest is empty")
        header = next(csv.reader([header_line]))
        required = {"INSTANCE", "ag_seq"}
        missing = sorted(required - set(header))
        if missing:
            raise ValueError(f"Canonical manifest is missing columns: {missing}")
        if header[0] != "INSTANCE":
            raise ValueError("INSTANCE must be the first canonical-manifest column")
        antigen_index = header.index("ag_seq")
        for line_number, line in enumerate(handle, start=2):
            instance_id = builder._leading_csv_identifier(line)
            if instance_id not in allowed_ids:
                continue
            fields = next(csv.reader([line]))
            if len(fields) != len(header):
                raise ValueError(f"Malformed authorized canonical row at line {line_number}")
            if instance_id in records:
                raise ValueError(f"Duplicate authorized canonical row for {instance_id}")
            records[instance_id] = builder.normalize_sequence(
                fields[antigen_index],
                f"{instance_id}:AG",
            )
    missing_ids = sorted(allowed_ids - set(records))
    if missing_ids:
        raise ValueError(
            f"Canonical manifest is missing {len(missing_ids)} source-train IDs"
        )
    return records


def load_verified_folds(
    index_path: Path,
    expected_index_sha256: str,
    allowed_ids: set[str],
    sealed_ids: set[str],
) -> tuple[list[tuple[int, Path, list[str], list[str]]], dict[str, Any]]:
    """Verify the index, every fold hash, exact coverage, and sealed exclusion."""
    observed_index_sha256 = builder.sha256_file(index_path)
    if observed_index_sha256 != expected_index_sha256:
        raise ValueError(
            "Fold index digest mismatch: "
            f"expected={expected_index_sha256}, observed={observed_index_sha256}"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("format_version") != builder.INDEX_FORMAT_VERSION:
        raise ValueError("Unexpected fold index format")
    raw_entries = index.get("folds")
    if not isinstance(raw_entries, list) or len(raw_entries) != 3:
        raise ValueError("Fold index must contain exactly three entries")

    folds: list[tuple[int, Path, list[str], list[str]]] = []
    validation_counts: dict[str, int] = {}
    for expected_fold_index, entry in enumerate(raw_entries):
        if set(entry) != {"fold_index", "path", "sha256"}:
            raise ValueError("Fold index entry schema differs from the pinned contract")
        fold_index = int(entry["fold_index"])
        if fold_index != expected_fold_index:
            raise ValueError("Fold entries are not ordered by contiguous fold index")
        relative_path = Path(str(entry["path"]))
        if relative_path.name != str(entry["path"]) or relative_path.suffix != ".json":
            raise ValueError("Fold index contains an unsafe relative path")
        fold_path = index_path.parent / relative_path
        if builder.sha256_file(fold_path) != str(entry["sha256"]):
            raise ValueError(f"Fold {fold_index} digest mismatch")
        payload = json.loads(fold_path.read_text(encoding="utf-8"))
        if payload.get("format_version") != builder.FOLD_FORMAT_VERSION:
            raise ValueError(f"Fold {fold_index} format version differs")
        split_payload = payload.get("splits")
        if not isinstance(split_payload, dict) or set(split_payload) != {
            "train",
            "validation",
        }:
            raise ValueError("Fold split schema must contain only train and validation")
        train = [str(value) for value in split_payload["train"]]
        validation = [str(value) for value in split_payload["validation"]]
        if len(train) != len(set(train)) or len(validation) != len(set(validation)):
            raise ValueError(f"Fold {fold_index} contains duplicate identifiers")
        observed = set(train) | set(validation)
        sealed_crossing = observed & sealed_ids
        if sealed_crossing:
            raise PermissionError(
                f"Fold {fold_index} contains {len(sealed_crossing)} sealed identifiers"
            )
        if set(train) & set(validation) or observed != allowed_ids:
            raise ValueError(f"Fold {fold_index} does not exactly partition source train")
        for value in validation:
            validation_counts[value] = validation_counts.get(value, 0) + 1
        folds.append((fold_index, fold_path, train, validation))

    if set(validation_counts) != allowed_ids or any(
        value != 1 for value in validation_counts.values()
    ):
        raise ValueError("Three validation folds do not partition source train exactly once")
    summary = {
        "path": str(index_path),
        "sha256": observed_index_sha256,
        "fold_count": len(folds),
        "validation_union_instance_count": len(validation_counts),
        "validation_membership_exactly_once": True,
    }
    return folds, summary


def unique_antigens(
    antigen_by_id: Mapping[str, str],
    identifiers: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Return digest-keyed unique antigens for one fold side."""
    sequences = {antigen_by_id[value] for value in identifiers}
    return tuple(
        sorted((builder.sequence_digest(sequence), sequence) for sequence in sequences)
    )


def edge_from_pair_result(result: Any) -> builder.SimilarityHit:
    """Convert one directed audit result to a canonical undirected edge."""
    metrics = result.metrics
    train_digest = str(result.train_digest)
    validation_digest = str(result.validation_digest)
    left, right = sorted((train_digest, validation_digest))
    if train_digest == left:
        left_coverage = metrics.left_reciprocal_coverage
        right_coverage = metrics.right_reciprocal_coverage
    else:
        left_coverage = metrics.right_reciprocal_coverage
        right_coverage = metrics.left_reciprocal_coverage
    return builder.SimilarityHit(
        left=left,
        right=right,
        identity=metrics.aligned_region_identity,
        left_coverage=left_coverage,
        right_coverage=right_coverage,
        source="exhaustive_needleman_wunsch_antigen",
    )


def edge_rank(edge: builder.SimilarityHit) -> tuple[float, float, float, str]:
    """Rank duplicate evidence deterministically by its strongest alignment."""
    return (
        edge.identity,
        min(edge.left_coverage, edge.right_coverage),
        edge.left_coverage + edge.right_coverage,
        edge.source,
    )


def merge_edges(
    prior: Sequence[builder.SimilarityHit],
    current: Sequence[builder.SimilarityHit],
) -> list[builder.SimilarityHit]:
    """Merge cumulative undirected edges and retain the strongest evidence."""
    result: dict[tuple[str, str], builder.SimilarityHit] = {}
    for edge in [*prior, *current]:
        key = (edge.left, edge.right)
        previous = result.get(key)
        if previous is None or edge_rank(edge) > edge_rank(previous):
            result[key] = edge
    return [result[key] for key in sorted(result)]


def edge_tsv_bytes(edges: Sequence[builder.SimilarityHit]) -> bytes:
    """Serialize digest-only edges in the builder's five-column format."""
    rows = [
        "\t".join(
            (
                edge.left,
                edge.right,
                f"{edge.identity:.17g}",
                f"{edge.left_coverage:.17g}",
                f"{edge.right_coverage:.17g}",
            )
        )
        for edge in edges
    ]
    return (("\n".join(rows) + "\n") if rows else "").encode("ascii")


def jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize deterministic JSON Lines."""
    return (
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":"))
            + "\n"
            for row in rows
        )
    ).encode("utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run all three exhaustive antigen audits and publish crossing edges."""
    args.authority_split = args.authority_split.expanduser().resolve()
    args.canonical_manifest = args.canonical_manifest.expanduser().resolve()
    args.fold_index = args.fold_index.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.prior_edge_tsv is not None:
        args.prior_edge_tsv = args.prior_edge_tsv.expanduser().resolve()
    for path in (
        args.authority_split,
        args.canonical_manifest,
        args.fold_index,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    if args.prior_edge_tsv is not None and not args.prior_edge_tsv.is_file():
        raise FileNotFoundError(f"Missing prior edge artifact: {args.prior_edge_tsv}")

    allowed_ids, sealed_ids, authority_summary = builder.load_authority(
        args.authority_split,
        args.expected_authority_sha256,
    )
    folds, fold_index_summary = load_verified_folds(
        args.fold_index,
        args.expected_fold_index_sha256,
        allowed_ids,
        sealed_ids,
    )
    antigen_by_id = load_source_train_antigens(
        args.canonical_manifest,
        allowed_ids,
    )
    allowed_digests = {
        builder.sequence_digest(sequence) for sequence in antigen_by_id.values()
    }
    policy = AuditPolicy(
        identity_threshold=args.identity_threshold,
        coverage_threshold=args.coverage_threshold,
        calm_scaled_threshold=args.identity_threshold,
        gap_open=abs(float(args.gap_open)),
        gap_extend=abs(float(args.gap_extend)),
        top_pair_count=args.top_pair_count,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict[str, Any]] = []
    all_current_edges: list[builder.SimilarityHit] = []
    for fold_index, fold_path, train_ids, validation_ids in folds:
        train_antigens = unique_antigens(antigen_by_id, train_ids)
        validation_antigens = unique_antigens(antigen_by_id, validation_ids)
        role_result = audit_role(
            train_antigens,
            validation_antigens,
            policy,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
        raw_edges = role_result.pop("_primary_crossing_edges")
        current_edges = [edge_from_pair_result(value) for value in raw_edges]
        all_current_edges.extend(current_edges)
        edge_rows = [_pair_payload(value) for value in raw_edges]
        edge_path = args.output_dir / (
            f"fold_{fold_index}.antigen.identity_coverage_edges.jsonl"
        )
        encoded_edges = jsonl_bytes(edge_rows)
        builder.atomic_create_or_validate(edge_path, encoded_edges, args.resume)
        role_result["identity_and_reciprocal_coverage_edge_artifact"] = {
            "format": "JSON Lines; digest-only crossing edges",
            "path": str(edge_path),
            "sha256": builder.sha256_bytes(encoded_edges),
            "edge_count": len(edge_rows),
            "raw_sequences_or_instance_ids_included": False,
        }
        fold_results.append(
            {
                "fold_index": fold_index,
                "fold_path": str(fold_path),
                "fold_sha256": builder.sha256_file(fold_path),
                "train_instance_count": len(train_ids),
                "validation_instance_count": len(validation_ids),
                "valid_identity_and_reciprocal_coverage_rule": role_result[
                    "valid_identity_and_reciprocal_coverage_rule"
                ],
                "antigen": role_result,
            }
        )

    current_edges = merge_edges([], all_current_edges)
    if args.prior_edge_tsv is None:
        prior_edges: list[builder.SimilarityHit] = []
        prior_summary = {"provided": False, "edge_count": 0}
    else:
        prior_edges, parsed_summary = builder.load_similarity_edges(
            args.prior_edge_tsv,
            allowed_digests,
            args.identity_threshold,
            args.coverage_threshold,
            source="prior_needleman_wunsch_antigen",
            require_complete_inventory=False,
        )
        prior_summary = {
            "provided": True,
            "path": str(args.prior_edge_tsv),
            "sha256": builder.sha256_file(args.prior_edge_tsv),
            "edge_count": len(prior_edges),
            "parsed_summary": parsed_summary,
        }
    cumulative_edges = merge_edges(prior_edges, current_edges)
    cumulative_path = args.output_dir / "cumulative_antigen_identity80_edges.tsv"
    cumulative_encoded = edge_tsv_bytes(cumulative_edges)
    builder.atomic_create_or_validate(cumulative_path, cumulative_encoded, args.resume)
    cumulative_artifact = {
        "format_version": EDGE_FORMAT_VERSION,
        "path": str(cumulative_path),
        "sha256": builder.sha256_bytes(cumulative_encoded),
        "edge_count": len(cumulative_edges),
        "current_iteration_unique_edge_count": len(current_edges),
        "prior": prior_summary,
        "raw_sequences_or_instance_ids_included": False,
    }

    generalized_source = Path(
        sys.modules[audit_role.__module__].__file__
    ).resolve()
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "stage": "exhaustive_train_only_antigen_needleman_wunsch_audit",
        "intended_use": "development_only",
        "valid": all(
            value["valid_identity_and_reciprocal_coverage_rule"]
            for value in fold_results
        ),
        "policy": {
            "role": "antigen_only",
            "algorithm": "exact global dynamic programming (Needleman-Wunsch family)",
            "backend": "Biopython PairwiseAligner",
            "substitution_matrix": "BLOSUM62",
            "gap_model": "affine with penalized terminal gaps",
            "gap_open_penalty": policy.gap_open,
            "gap_extend_penalty": policy.gap_extend,
            "identity_threshold": policy.identity_threshold,
            "reciprocal_coverage_threshold": policy.coverage_threshold,
            "validity_rule": (
                "zero cross-fold pairs with aligned-region identity >= threshold "
                "and both reciprocal coverages >= threshold"
            ),
            "enumeration": "all unique train x validation antigens in every fold",
        },
        "authority": authority_summary,
        "fold_index": fold_index_summary,
        "folds": fold_results,
        "crossing_edges": cumulative_artifact,
        "provenance": {
            "canonical_manifest": {
                "path": str(args.canonical_manifest),
                "sha256": builder.sha256_file(args.canonical_manifest),
                "sequence_rows_parsed": "source_train_only",
            },
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": builder.sha256_file(Path(__file__).resolve()),
            },
            "generalized_alignment_source": {
                "path": str(generalized_source),
                "sha256": builder.sha256_file(generalized_source),
            },
            "runtime": {
                **_dependency_versions(),
                "workers": args.workers,
                "chunk_size": args.chunk_size,
            },
        },
        "source_validation_or_test_sequences_read": False,
        "labels_contacts_predictions_or_checkpoints_read": False,
    }
    payload["scientific_payload_sha256"] = builder.sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    report_path = args.output_dir / "audit.json"
    builder.atomic_create_or_validate(
        report_path,
        builder.canonical_json_bytes(payload),
        args.resume,
    )
    receipt = {
        "report_path": str(report_path),
        "report_sha256": builder.sha256_file(report_path),
        "scientific_payload_sha256": payload["scientific_payload_sha256"],
        "cumulative_edge_artifact_sha256": cumulative_artifact["sha256"],
        "cumulative_edge_count": cumulative_artifact["edge_count"],
        "valid": payload["valid"],
    }
    receipt_path = args.output_dir / "receipt.json"
    builder.atomic_create_or_validate(
        receipt_path,
        builder.canonical_json_bytes(receipt),
        args.resume,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    """Run the audit without printing source sequences."""
    args = parse_args(argv)
    try:
        run(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
