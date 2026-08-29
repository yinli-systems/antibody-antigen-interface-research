#!/usr/bin/env python3
"""Analyze when spatial topology helps or harms frozen CHIMERA epitope scores."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from interaction_first.chimera_cache import load_cache_manifest  # noqa: E402
from interaction_first.chimera_case_study import (  # noqa: E402
    analyze_instances,
    flatten_case_rows,
    load_prediction_matrix,
)
from interaction_first.sabdab2_structure import sha256_file  # noqa: E402


def _format(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _case_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Complex | Length | Epi n | 8Å components | Rg (Å) | Dense AP | Spatial AP | Delta AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['instance_id']} | {row['antigen_length']} | {row['epitope_count']} "
            f"| {row['epitope_components_8a']} | {_format(row['epitope_radius_of_gyration_a'], 2)} "
            f"| {_format(row['sequence_dense_auprc'])} | {_format(row['spatial_topology_auprc'])} "
            f"| {row['delta_auprc_spatial_minus_dense']:+.4f} |"
        )
    return lines


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# CHIMERA frozen epitope case study",
        "",
        result["claim_boundary"],
        "",
        "## Design",
        "",
        f"- Split: `{result['split']}`; complexes: {result['summary']['complex_count']}.",
        "- Comparison: three-seed ensemble spatial-topology minus three-seed ensemble dense sequence-only.",
        "- Spatial discontinuity: connected components among labeled epitope residues in a Euclidean CA graph; 8 Å is the main descriptive threshold, with 6/10/12 Å sensitivity fields in the CSV.",
        "- All correlations and quartile analyses are exploratory and do not alter the frozen benchmark decision.",
        "- Sensitivity cohorts remove peptide-like antigens shorter than 50 residues and/or label prevalence above 0.5; these cutoffs are descriptive robustness checks, not tuned model-selection rules.",
        "",
        "## Spearman association with per-complex epitope AUPRC delta",
        "",
        "| Feature | n | rho | 95% antigen-group bootstrap CI | nominal p |",
        "|---|---:|---:|---:|---:|",
    ]
    for feature, record in result["summary"]["correlations"].items():
        interval = f"[{_format(record['ci_low'])}, {_format(record['ci_high'])}]"
        lines.append(
            f"| `{feature}` | {record['n']} | {_format(record['spearman_rho'])} "
            f"| {interval} | {_format(record['p_value'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity cohorts",
            "",
            "| Cohort | n | Dense mean AP | Spatial mean AP | Mean delta | 95% antigen-group bootstrap CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, record in result["summary"]["sensitivity_cohorts"].items():
        lines.append(
            f"| `{name}` | {record['n']} | {_format(record['dense_mean_auprc'])} "
            f"| {_format(record['spatial_mean_auprc'])} | {_format(record['mean_delta_auprc'])} "
            f"| [{_format(record['ci_low'])}, {_format(record['ci_high'])}] |"
        )
    lines.extend(["", "## Most harmed cases", ""])
    lines.extend(_case_table(result["summary"]["most_harmed_cases"]))
    lines.extend(["", "## Most helped cases", ""])
    lines.extend(_case_table(result["summary"]["most_helped_cases"]))
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This analysis can identify failure patterns and candidate hypotheses. It cannot show that the model learned transferable local structural motifs from training data, because it does not compare test motifs against a training-structure reference bank and it is post hoc. The antigen-fold split supplies a genuine held-out evaluation axis, but causal or mechanistic claims require a separate, training-only-fitted representation-similarity analysis. Very short or near-all-positive antigen axes can make per-complex average precision nearly degenerate, so the full-cohort results must be read together with the sensitivity table.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-manifest-sha256", required=True)
    parser.add_argument("--split", choices=("antigen_fold.test", "temporal.test"), required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("bootstrap-replicates must be at least 100")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise SystemExit("output-root already exists")
    cache_manifest = load_cache_manifest(
        args.cache_root, expected_sha256=args.expected_cache_manifest_sha256
    )
    prediction_manifest_path = args.prediction_root / "PREDICTION_MANIFEST.json"
    if sha256_file(prediction_manifest_path) != args.expected_prediction_manifest_sha256:
        raise ValueError("prediction manifest SHA-256 differs")
    prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    if (
        prediction_manifest.get("schema_version") != "chimera_frozen_external_predictions_v1"
        or prediction_manifest.get("cache_manifest_sha256") != args.expected_cache_manifest_sha256
        or prediction_manifest.get("metrics_computed") is not False
    ):
        raise ValueError("prediction manifest authority differs")
    matrix = load_prediction_matrix(args.prediction_root, prediction_manifest)
    rows, summary = analyze_instances(
        cache_root=args.cache_root,
        cache_manifest=cache_manifest,
        prediction_matrix=matrix,
        split_name=args.split,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    result = {
        "schema_version": "chimera_frozen_epitope_case_study_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Post-hoc descriptive analysis of real frozen external predictions. "
            "It is exploratory, not a new confirmatory benchmark result and not the "
            "canonical CHIMERA CDR-design task."
        ),
        "split": args.split,
        "comparison": "spatial_topology minus sequence_dense",
        "seed_ensemble": [17, 42, 73],
        "fixed_classification_threshold": 0.5,
        "bootstrap": {
            "unit": "normalized antigen sequence SHA-256 group",
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "digests": {
            "cache_manifest_sha256": args.expected_cache_manifest_sha256,
            "prediction_manifest_sha256": args.expected_prediction_manifest_sha256,
            "case_study_source_sha256": sha256_file(Path(__file__)),
        },
        "summary": summary,
    }
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_root.name}.tmp-", dir=args.output_root.parent))
    try:
        csv_path = temporary / "PER_COMPLEX_CASES.csv"
        flat = flatten_case_rows(rows)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        json_path = temporary / "CASE_STUDY.json"
        md_path = temporary / "CASE_STUDY.md"
        json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(markdown_report(result), encoding="utf-8")
        receipt = {
            "schema_version": 1,
            "files": {
                path.name: sha256_file(path)
                for path in (csv_path, json_path, md_path)
            },
        }
        (temporary / "CASE_STUDY_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.chmod(temporary, 0o755)
        os.replace(temporary, args.output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"case_study={args.output_root / 'CASE_STUDY.md'}")
    print(f"case_study_sha256={sha256_file(args.output_root / 'CASE_STUDY.md')}")


if __name__ == "__main__":
    main()
