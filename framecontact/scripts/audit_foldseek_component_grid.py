#!/usr/bin/env python3
"""Audit label-free Foldseek graph connectivity across symmetric score rules."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_split_matrix import (  # noqa: E402
    _UnionFind,
    _base_rows,
    _components,
    _greedy_component_roles,
    _load_csv,
    _load_prepared,
    _role_counts,
    _union_equal_values,
    structure_ids_sha256,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)
from interaction_first.structural_novelty import FOLDSEEK_RECEIPT_FORMAT  # noqa: E402


FORMAT_VERSION = "framecontact_foldseek_component_grid_v1"
CONFIGURATIONS = (
    ("reciprocal_min", 0.50),
    ("reciprocal_min", 0.60),
    ("reciprocal_min", 0.70),
    ("reciprocal_min", 0.80),
    ("geometric_mean", 0.50),
    ("geometric_mean", 0.60),
    ("geometric_mean", 0.70),
    ("reciprocal_max", 0.70),
    ("reciprocal_max", 0.80),
    ("reciprocal_max", 0.90),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--foldseek-alignments", type=Path, required=True)
    parser.add_argument("--foldseek-alignments-sha256", required=True)
    parser.add_argument("--foldseek-receipt", type=Path, required=True)
    parser.add_argument("--foldseek-receipt-sha256", required=True)
    parser.add_argument("--assignment-seed", type=int, default=20_260_730)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def symmetric_score(query_tm: float, target_tm: float, rule: str) -> float:
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (query_tm, target_tm)):
        raise ValueError("Foldseek TM scores must be finite and lie in [0, 1]")
    if rule == "reciprocal_min":
        return min(query_tm, target_tm)
    if rule == "geometric_mean":
        return math.sqrt(query_tm * target_tm)
    if rule == "reciprocal_max":
        return max(query_tm, target_tm)
    raise ValueError(f"Unknown symmetric score rule: {rule}")


def component_grid(
    rows: list[dict[str, Any]],
    alignment_path: Path,
    *,
    configurations: tuple[tuple[str, float], ...] = CONFIGURATIONS,
    assignment_seed: int = 20_260_730,
) -> tuple[list[dict[str, Any]], int]:
    identifiers = {str(row["instance_id"]) for row in rows}
    if len(configurations) == 0 or len(set(configurations)) != len(configurations):
        raise ValueError("Connectivity configurations must be nonempty and unique")
    union_finds = {configuration: _UnionFind(identifiers) for configuration in configurations}
    for union_find in union_finds.values():
        _union_equal_values(rows, union_find)
    edge_counts = {configuration: 0 for configuration in configurations}
    ordered_rows = 0
    with alignment_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"query", "target", "qtmscore", "ttmscore"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Foldseek table lacks reciprocal TM-score columns")
        for record in reader:
            ordered_rows += 1
            query, target = str(record["query"]), str(record["target"])
            if query not in identifiers or target not in identifiers:
                raise ValueError("Foldseek table references an unknown structure")
            if query >= target:
                continue
            query_tm, target_tm = float(record["qtmscore"]), float(record["ttmscore"])
            scores = {
                rule: symmetric_score(query_tm, target_tm, rule)
                for rule in {value[0] for value in configurations}
            }
            for configuration, union_find in union_finds.items():
                rule, threshold = configuration
                if scores[rule] >= threshold:
                    union_find.union(query, target)
                    edge_counts[configuration] += 1
    result = []
    cohort_size = len(rows)
    for configuration in configurations:
        rule, threshold = configuration
        components = _components(rows, union_finds[configuration])
        roles = _greedy_component_roles(
            components,
            seed=assignment_seed,
            fractions=(0.70, 0.15, 0.15),
        )
        sizes = sorted((len(component) for component in components), reverse=True)
        counts = _role_counts(roles)
        result.append(
            {
                "score_rule": rule,
                "threshold": threshold,
                "hard_undirected_edge_count": edge_counts[configuration],
                "component_count": len(components),
                "largest_component_size": sizes[0],
                "largest_component_fraction": sizes[0] / cohort_size,
                "components_at_least_10": sum(value >= 10 for value in sizes),
                "suggested_role_counts": counts,
                "minimum_holdout_count": min(counts["validation"], counts["test"]),
                "eligible_for_primary_split": (
                    sizes[0] <= 0.50 * cohort_size
                    and counts["validation"] >= 0.10 * cohort_size
                    and counts["test"] >= 0.10 * cohort_size
                ),
            }
        )
    return result, ordered_rows


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError("Refusing to overwrite component-grid audit")
    authority = {
        "metadata_sha256": authenticate_file(
            arguments.metadata, arguments.metadata_sha256, "metadata"
        ),
        "prepared_sha256": authenticate_file(
            arguments.prepared, arguments.prepared_sha256, "prepared cohort"
        ),
        "foldseek_alignments_sha256": authenticate_file(
            arguments.foldseek_alignments,
            arguments.foldseek_alignments_sha256,
            "Foldseek alignments",
        ),
        "foldseek_receipt_sha256": authenticate_file(
            arguments.foldseek_receipt,
            arguments.foldseek_receipt_sha256,
            "Foldseek receipt",
        ),
        "task_labels_read": False,
        "predictions_read": False,
    }
    metadata = _load_csv(arguments.metadata, "INSTANCE")
    prepared = _load_prepared(arguments.prepared)
    rows = _base_rows(metadata, prepared)
    receipt = json.loads(arguments.foldseek_receipt.read_text(encoding="utf-8"))
    identifiers = {str(row["instance_id"]) for row in rows}
    if (
        receipt.get("schema_version") != FOLDSEEK_RECEIPT_FORMAT
        or receipt.get("complete_ordered_cartesian_grid") is not True
        or receipt.get("query_target_same_database") is not True
        or receipt.get("test_labels_read") is not False
        or int(receipt.get("raw_row_count", -1)) != len(rows) ** 2
        or receipt.get("structure_ids_sha256") != structure_ids_sha256(identifiers)
    ):
        raise ValueError("Foldseek receipt does not authenticate this cohort")
    grid, ordered_rows = component_grid(
        rows,
        arguments.foldseek_alignments,
        assignment_seed=arguments.assignment_seed,
    )
    if ordered_rows != len(rows) ** 2:
        raise ValueError("Foldseek audit did not observe the complete ordered grid")
    eligible = [row for row in grid if row["eligible_for_primary_split"]]
    recommendation = (
        sorted(
            eligible,
            key=lambda row: (
                ("reciprocal_min", "geometric_mean", "reciprocal_max").index(
                    row["score_rule"]
                ),
                row["threshold"],
            ),
        )[0]
        if eligible
        else None
    )
    payload = {
        "format_version": FORMAT_VERSION,
        "authority": authority,
        "cohort": {
            "instance_count": len(rows),
            "instance_ids_sha256": structure_ids_sha256(identifiers),
            "ordered_alignment_rows": ordered_rows,
        },
        "predeclared_eligibility_rule": {
            "largest_component_fraction_max": 0.50,
            "validation_fraction_min": 0.10,
            "test_fraction_min": 0.10,
            "priority": "lowest-threshold reciprocal_min, then geometric_mean, then reciprocal_max",
        },
        "grid": grid,
        "recommended_configuration": recommendation,
        "claim_boundary": (
            "Label-free graph-topology audit only. The recommendation did not read task "
            "labels, model predictions, or performance metrics."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
                "recommended_configuration": recommendation,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
