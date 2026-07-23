#!/usr/bin/env python3
"""Reconcile exact Needleman-Wunsch crossing edges with MMseqs2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from build_identity_isolated_development_split import (
    atomic_write_json,
    normalize_fraction,
    sha256_file,
)


FORMAT_VERSION = "sabdab2_mmseqs_nw_reconciliation_v1"


@dataclass(frozen=True)
class MmseqsRow:
    """One directed MMseqs2 alignment row."""

    query: str
    target: str
    identity: float
    query_coverage: float
    target_coverage: float


def _canonical_sha256(payload: object) -> str:
    """Hash a JSON-compatible object deterministically."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_fasta_digests(path: Path) -> set[str]:
    """Load digest identifiers without retaining development sequences."""
    identifiers: set[str] = set()
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(">"):
                identifier = line[1:].strip().split()[0]
                if not identifier or identifier in identifiers:
                    raise ValueError(f"Invalid or duplicate FASTA identifier: {identifier!r}")
                identifiers.add(identifier)
    if not identifiers:
        raise ValueError("FASTA contains no identifiers")
    return identifiers


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load one strict JSON object per non-empty line."""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(payload)
    return rows


def load_relevant_mmseqs_rows(
    path: Path, relevant_pairs: set[tuple[str, str]]
) -> tuple[dict[tuple[str, str], list[MmseqsRow]], dict[str, object]]:
    """Index only MMseqs2 rows relevant to exact-alignment crossing edges."""
    result: dict[tuple[str, str], list[MmseqsRow]] = defaultdict(list)
    parsed_count = 0
    relevant_row_count = 0
    malformed_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                malformed_count += 1
                continue
            parsed_count += 1
            query, target = fields[:2]
            key = tuple(sorted((query, target)))
            if query == target or key not in relevant_pairs:
                continue
            row = MmseqsRow(
                query=query,
                target=target,
                identity=normalize_fraction(fields[2]),
                query_coverage=normalize_fraction(fields[3]),
                target_coverage=normalize_fraction(fields[4]),
            )
            result[key].append(row)
            relevant_row_count += 1
    if malformed_count:
        raise ValueError(f"MMseqs2 artifact contains {malformed_count} malformed rows")
    return dict(result), {
        "path": str(path.expanduser().resolve()),
        "sha256": sha256_file(path),
        "parsed_row_count": parsed_count,
        "relevant_nonself_row_count": relevant_row_count,
    }


def _row_payload(row: MmseqsRow) -> dict[str, object]:
    """Serialize one MMseqs2 row."""
    return {
        "query": row.query,
        "target": row.target,
        "identity": row.identity,
        "query_coverage": row.query_coverage,
        "target_coverage": row.target_coverage,
    }


def reconcile_edge(
    edge: Mapping[str, object],
    fasta_digests: set[str],
    rows: Sequence[MmseqsRow],
    identity_threshold: float,
    coverage_threshold: float,
) -> dict[str, object]:
    """Classify why one exact-alignment edge was absent from the MMseqs graph."""
    train_digest = str(edge["train_digest"])
    validation_digest = str(edge["validation_digest"])
    missing_digests = sorted(
        {train_digest, validation_digest} - fasta_digests
    )
    tolerance = 1e-12
    qualified = [
        row
        for row in rows
        if row.identity + tolerance >= identity_threshold
        and row.query_coverage + tolerance >= coverage_threshold
        and row.target_coverage + tolerance >= coverage_threshold
    ]
    if missing_digests:
        category = "digest_mapping_failure"
    elif not rows:
        category = "no_nonself_mmseqs_row"
    elif qualified:
        category = "threshold_qualified_row_present_inconsistency"
    elif max(row.identity for row in rows) + tolerance < identity_threshold:
        category = "mmseqs_identity_below_threshold"
    else:
        category = "mmseqs_reciprocal_coverage_below_threshold"
    directions = {(row.query, row.target) for row in rows}
    forward = (train_digest, validation_digest) in directions
    reverse = (validation_digest, train_digest) in directions
    if forward and reverse:
        direction_status = "both_directions_present"
    elif forward:
        direction_status = "train_to_validation_only"
    elif reverse:
        direction_status = "validation_to_train_only"
    else:
        direction_status = "no_direction_present"
    best_rows = sorted(
        rows,
        key=lambda row: (
            row.identity,
            min(row.query_coverage, row.target_coverage),
            row.query,
            row.target,
        ),
        reverse=True,
    )[:4]
    return {
        "train_digest": train_digest,
        "validation_digest": validation_digest,
        "reconciliation_category": category,
        "direction_status": direction_status,
        "missing_fasta_digest_count": len(missing_digests),
        "mmseqs_row_count": len(rows),
        "threshold_qualified_mmseqs_row_count": len(qualified),
        "needleman_wunsch_metrics": edge["metrics"],
        "best_mmseqs_rows": [_row_payload(row) for row in best_rows],
    }


def _edge_rank(edge: Mapping[str, object]) -> tuple[object, ...]:
    """Rank exact-alignment edges identically to the audit report."""
    metrics = edge["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("Edge metrics are not an object")
    return (
        float(metrics["calm_scaled_similarity"]),
        float(metrics["aligned_region_identity"]),
        float(metrics["minimum_reciprocal_coverage"]),
        str(edge["train_digest"]),
        str(edge["validation_digest"]),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    """Reconcile every concat-primary edge referenced by one NW report."""
    report = json.loads(args.nw_report.read_text(encoding="utf-8"))
    if report.get("intended_use") != "development_only":
        raise ValueError("Needleman-Wunsch report is not development-only")
    if report.get("frozen_test_sequences_aligned") is not False:
        raise ValueError("Needleman-Wunsch report does not prove the test stayed sealed")
    role_inputs = {
        "antibody": (args.antibody_mmseqs, args.antibody_fasta),
        "antigen": (args.antigen_mmseqs, args.antigen_fasta),
    }
    identity_threshold = float(args.identity_threshold)
    coverage_threshold = float(args.coverage_threshold)
    output_cases: dict[str, object] = {}
    input_provenance: dict[str, object] = {}
    for role, (mmseqs_path, fasta_path) in role_inputs.items():
        cases: dict[str, list[dict[str, object]]] = {}
        relevant_pairs: set[tuple[str, str]] = set()
        for split_name, split_result in report["results"].items():
            artifact = split_result["roles"][role][
                "identity_and_reciprocal_coverage_edge_artifact"
            ]
            edge_path = Path(artifact["path"])
            if sha256_file(edge_path) != artifact["sha256"]:
                raise ValueError(f"NW edge artifact hash changed: {edge_path}")
            edges = load_jsonl(edge_path)
            if len(edges) != artifact["edge_count"]:
                raise ValueError(f"NW edge artifact count changed: {edge_path}")
            cases[str(split_name)] = edges
            relevant_pairs.update(
                tuple(sorted((str(edge["train_digest"]), str(edge["validation_digest"]))))
                for edge in edges
            )
        fasta_digests = load_fasta_digests(fasta_path)
        indexed_rows, mmseqs_summary = load_relevant_mmseqs_rows(
            mmseqs_path, relevant_pairs
        )
        input_provenance[role] = {
            "fasta": {
                "path": str(fasta_path.expanduser().resolve()),
                "sha256": sha256_file(fasta_path),
                "digest_count": len(fasta_digests),
            },
            "mmseqs": mmseqs_summary,
        }
        for split_name, edges in cases.items():
            reconciled = [
                reconcile_edge(
                    edge,
                    fasta_digests,
                    indexed_rows.get(
                        tuple(sorted((str(edge["train_digest"]), str(edge["validation_digest"])))),
                        [],
                    ),
                    identity_threshold,
                    coverage_threshold,
                )
                for edge in edges
            ]
            category_counts = Counter(
                str(item["reconciliation_category"]) for item in reconciled
            )
            direction_counts = Counter(str(item["direction_status"]) for item in reconciled)
            top_edges = sorted(edges, key=_edge_rank, reverse=True)[:20]
            top_keys = {
                (str(edge["train_digest"]), str(edge["validation_digest"]))
                for edge in top_edges
            }
            top_reconciled = [
                item
                for item in reconciled
                if (str(item["train_digest"]), str(item["validation_digest"])) in top_keys
            ]
            top_reconciled.sort(
                key=lambda item: _edge_rank({
                    "train_digest": item["train_digest"],
                    "validation_digest": item["validation_digest"],
                    "metrics": item["needleman_wunsch_metrics"],
                }),
                reverse=True,
            )
            output_cases[f"{split_name}.{role}"] = {
                "edge_count": len(edges),
                "reconciliation_category_counts": dict(sorted(category_counts.items())),
                "direction_status_counts": dict(sorted(direction_counts.items())),
                "top_20_edges": top_reconciled,
            }
    source_path = Path(__file__).resolve()
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "stage": "reconcile_mmseqs_with_exact_nw_crossing_edges",
        "intended_use": "development_only",
        "frozen_test_labels_or_predictions_read": False,
        "frozen_test_sequences_read": False,
        "identity_threshold": identity_threshold,
        "reciprocal_coverage_threshold": coverage_threshold,
        "nw_report": {
            "path": str(args.nw_report.expanduser().resolve()),
            "sha256": sha256_file(args.nw_report),
        },
        "inputs": input_provenance,
        "cases": output_cases,
        "interpretation_guardrail": (
            "MMseqs2 fident and the declared NW aligned-region identity use different "
            "denominators. A category explains artifact reconciliation, not biological "
            "homology or causal model behavior."
        ),
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
    }
    payload["scientific_payload_sha256"] = _canonical_sha256(payload)
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output.expanduser().resolve()),
        "cases": {
            name: case["reconciliation_category_counts"]
            for name, case in output_cases.items()
        },
        "scientific_payload_sha256": payload["scientific_payload_sha256"],
    }, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nw-report", type=Path, required=True)
    parser.add_argument("--antibody-mmseqs", type=Path, required=True)
    parser.add_argument("--antigen-mmseqs", type=Path, required=True)
    parser.add_argument("--antibody-fasta", type=Path, required=True)
    parser.add_argument("--antigen-fasta", type=Path, required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.80)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Run the command-line reconciliation."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
