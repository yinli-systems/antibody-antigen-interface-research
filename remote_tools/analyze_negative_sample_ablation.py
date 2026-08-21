#!/usr/bin/env python3
"""Audit matched partner-negative ablations without opening test roles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


SEEDS = (17, 42, 73, 101, 137)
ARMS = ("with_negative", "no_negative")
SHARED_CONFIGURATION = (
    "asep_cache_manifest_sha256", "asep_feature_mode", "batch_size",
    "calibration_max_pairs", "contact_loss_weight", "contact_manifest_sha256",
    "d_model", "dataset_backend", "dropout", "early_stopping_patience",
    "embedding_manifest_sha256", "epitope_loss_weight", "epochs",
    "evaluation_batch_size", "experiment", "learning_rate", "neighbor_k",
    "neighbor_radius", "num_heads", "num_rbf", "num_self_layers",
    "offload_saved_tensors", "paratope_loss_weight", "precision",
    "score_fraction", "seed", "sequence_window", "source_manifest_sha256",
    "split_manifest_sha256", "structure_manifest_sha256",
    "training_partner_panel_sha256", "validation_partner_panel_sha256",
    "weight_decay", "workers",
)
SELECTED_METRICS = (
    ("contact_auprc", ("contact", "auprc")),
    ("contact_macro_auprc", ("contact", "mean_per_complex_auprc")),
    ("contact_auroc", ("contact", "auroc")),
    ("contact_f1", ("contact", "f1")),
    ("contact_mcc", ("contact", "mcc")),
    ("contact_brier", ("contact", "brier")),
    ("contact_raw_brier", ("contact", "fixed_raw_logit_threshold_zero", "brier")),
    ("contact_ece_15", ("contact", "ece_15")),
    ("epitope_auprc", ("epitope", "auprc")),
    ("epitope_auroc", ("epitope", "auroc")),
    ("paratope_auprc", ("paratope", "auprc")),
    ("paratope_auroc", ("paratope", "auroc")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def nested(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        current = current[key]
    return current


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def paired_bootstrap(deltas: list[float]) -> dict[str, float | int]:
    generator = random.Random(20_260_821)
    samples = sorted(
        statistics.fmean(generator.choice(deltas) for _ in deltas)
        for _ in range(10_000)
    )
    return {
        "estimate": statistics.fmean(deltas),
        "ci95_lower": samples[249],
        "ci95_upper": samples[9749],
        "replicates": 10_000,
        "seed": 20_260_821,
    }


def load_member(root: Path, seed: int, expected_weight: float) -> dict[str, Any]:
    run_root = root / f"seed-{seed}"
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    receipt_path = run_root / "RUN_RECEIPT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    for path in (result_path, receipt_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    result_sha = sha256_file(result_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    receipt = load_json(receipt_path)
    if (
        receipt.get("seed") != seed
        or receipt.get("result_sha256") != result_sha
        or receipt.get("selected_checkpoint_sha256") != checkpoint_sha
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
    ):
        raise ValueError(f"Run receipt failed: {run_root}")
    result = load_json(result_path)
    authority = result["authority"]
    if (
        authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
    ):
        raise ValueError(f"Test authority opened: {run_root}")
    configuration = result["configuration"]
    observed_weight = finite(
        configuration["specificity_loss_weight"], "specificity_loss_weight"
    )
    if not math.isclose(observed_weight, expected_weight, abs_tol=1e-12):
        raise ValueError(f"Wrong arm weight: {run_root}")
    curves = []
    for item in result["history"]:
        train = item["train"]
        validation = item["validation"]
        curves.append(
            {
                "epoch": int(item["epoch"]),
                "train_contact_bce": finite(train["contact_loss"], "train contact"),
                "train_shared_objective": (
                    finite(train["contact_loss"], "train contact")
                    + 0.25 * finite(train["epitope_loss"], "train epitope")
                    + 0.25 * finite(train["paratope_loss"], "train paratope")
                ),
                "train_total_objective": finite(train["loss"], "train total"),
                "validation_contact_raw_brier": finite(
                    nested(
                        validation,
                        ("contact", "fixed_raw_logit_threshold_zero", "brier"),
                    ),
                    "validation raw brier",
                ),
            }
        )
    selected = result["selected_checkpoint_validation"]
    metrics = {
        name: finite(nested(selected, path), name)
        for name, path in SELECTED_METRICS
    }
    return {
        "seed": seed,
        "run_root": str(run_root),
        "result_sha256": result_sha,
        "receipt_sha256": sha256_file(receipt_path),
        "checkpoint_sha256": checkpoint_sha,
        "configuration": configuration,
        "data": result["data"],
        "authority": authority,
        "curves": curves,
        "selected_epoch": int(result["selection"]["best_epoch"]),
        "selected_metrics": metrics,
    }


def verify_pair(positive: dict[str, Any], negative: dict[str, Any]) -> None:
    if positive["data"] != negative["data"]:
        raise ValueError("Dataset accounting differs between arms")
    for key in SHARED_CONFIGURATION:
        if positive["configuration"].get(key) != negative["configuration"].get(key):
            raise ValueError(f"Configuration differs beyond ablation: {key}")


def curve_rows(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for member in members:
        for row in member["curves"]:
            for name, value in row.items():
                if name != "epoch":
                    grouped[row["epoch"]][name].append(value)
    output = []
    for epoch in sorted(grouped):
        row: dict[str, Any] = {"epoch": epoch}
        for name, values in grouped[epoch].items():
            summary = summarize(values)
            row[f"{name}_n"] = summary["count"]
            row[f"{name}_mean"] = summary["mean"]
            row[f"{name}_sd"] = summary["sample_sd"]
        output.append(row)
    return output


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Partner-negative ablation: development-only audit",
        "",
        "Test and external-test roles remained sealed. Total objectives differ by "
        "definition, so curve ranking uses shared train contact BCE/shared cognate "
        "objective and fixed raw-logit validation Brier loss.",
        "",
    ]
    for dataset in ("asep", "sabdab2"):
        study = payload["datasets"][dataset]
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| Selected validation metric | With K=1 negative | No negative | No-minus-with delta (95% paired-seed CI) |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric in (
            "contact_auprc", "contact_macro_auprc", "contact_auroc",
            "contact_raw_brier", "epitope_auprc", "paratope_auprc",
        ):
            positive = study["arms"]["with_negative"]["selected_summary"][metric]
            negative = study["arms"]["no_negative"]["selected_summary"][metric]
            delta = study["paired_deltas"][metric]
            lines.append(
                f"| {metric} | {positive['mean']:.6f} +/- {positive['sample_sd']:.6f} "
                f"| {negative['mean']:.6f} +/- {negative['sample_sd']:.6f} "
                f"| {delta['estimate']:+.6f} [{delta['ci95_lower']:+.6f}, {delta['ci95_upper']:+.6f}] |"
            )
        lines.extend(
            [
                "",
                f"- With-negative root: {study['roots']['with_negative']}",
                f"- No-negative root: {study['roots']['no_negative']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def plot_curves(payload: dict[str, Any], output: Path) -> None:
    """Render a dependency-free SVG so the locked training env is untouched."""
    width, height = 1200, 800
    panel_width, panel_height = 560, 340
    colors = {"with_negative": "#2166ac", "no_negative": "#b2182b"}
    specs = (
        ("train_shared_objective", "Train shared cognate objective"),
        ("validation_contact_raw_brier", "Validation raw Brier loss"),
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:17px;font-weight:600}.axis{font-size:12px}.legend{font-size:12px}</style>',
    ]
    for row_index, dataset in enumerate(("asep", "sabdab2")):
        for column_index, (metric, title) in enumerate(specs):
            left = 55 + column_index * 595
            top = 45 + row_index * 380
            plot_left, plot_top = left + 65, top + 35
            plot_width, plot_height = panel_width - 90, panel_height - 85
            arm_rows = {
                arm: payload["datasets"][dataset]["arms"][arm]["curve_summary"]
                for arm in ARMS
            }
            all_rows = [row for rows in arm_rows.values() for row in rows]
            xmax = max(row["epoch"] for row in all_rows)
            values = [row[f"{metric}_mean"] for row in all_rows]
            ymin, ymax = min(values), max(values)
            padding = max((ymax - ymin) * 0.08, 1e-9)
            ymin -= padding
            ymax += padding
            svg.extend(
                [
                    f'<text class="title" x="{left + panel_width / 2}" y="{top + 18}" text-anchor="middle">{dataset}: {title}</text>',
                    f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#333"/>',
                    f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#333"/>',
                    f'<text class="axis" x="{plot_left + plot_width / 2}" y="{plot_top + plot_height + 35}" text-anchor="middle">epoch</text>',
                    f'<text class="axis" x="{plot_left - 45}" y="{plot_top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 {plot_left - 45} {plot_top + plot_height / 2})">loss</text>',
                    f'<text class="axis" x="{plot_left - 8}" y="{plot_top + 5}" text-anchor="end">{ymax:.4g}</text>',
                    f'<text class="axis" x="{plot_left - 8}" y="{plot_top + plot_height}" text-anchor="end">{ymin:.4g}</text>',
                    f'<text class="axis" x="{plot_left}" y="{plot_top + plot_height + 18}" text-anchor="middle">1</text>',
                    f'<text class="axis" x="{plot_left + plot_width}" y="{plot_top + plot_height + 18}" text-anchor="middle">{xmax}</text>',
                ]
            )
            for arm in ARMS:
                points = " ".join(
                    f"{plot_left + (row['epoch'] - 1) / max(1, xmax - 1) * plot_width:.2f},"
                    f"{plot_top + (ymax - row[f'{metric}_mean']) / (ymax - ymin) * plot_height:.2f}"
                    for row in arm_rows[arm]
                )
                svg.append(
                    f'<polyline points="{points}" fill="none" stroke="{colors[arm]}" stroke-width="2.5"/>'
                )
            svg.extend(
                [
                    f'<line x1="{plot_left + 10}" y1="{plot_top + 14}" x2="{plot_left + 35}" y2="{plot_top + 14}" stroke="{colors["with_negative"]}" stroke-width="3"/>',
                    f'<text class="legend" x="{plot_left + 42}" y="{plot_top + 18}">with K=1 negative</text>',
                    f'<line x1="{plot_left + 165}" y1="{plot_top + 14}" x2="{plot_left + 190}" y2="{plot_top + 14}" stroke="{colors["no_negative"]}" stroke-width="3"/>',
                    f'<text class="legend" x="{plot_left + 197}" y="{plot_top + 18}">no negative</text>',
                ]
            )
    svg.append("</svg>")
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    roots = {
        "asep": {
            "with_negative": args.data_root
            / "asep/development-runs/epitope-group-v1/framecontact",
            "no_negative": args.data_root
            / "asep/development-runs/negative-sample-ablation-v1/no_partner_negative/framecontact",
        },
        "sabdab2": {
            "with_negative": args.data_root
            / "sabdab2/development-runs/antigen-fold-v3_1-v5/framecontact",
            "no_negative": args.data_root
            / "sabdab2/development-runs/negative-sample-ablation-v1/no_partner_negative/framecontact",
        },
    }
    payload: dict[str, Any] = {
        "format_version": "framecontact_partner_negative_ablation_audit_v1",
        "claim_boundary": "Development train/validation only; no test claim.",
        "seeds": list(SEEDS),
        "datasets": {},
    }
    for dataset, dataset_roots in roots.items():
        arms = {
            arm: [
                load_member(
                    dataset_roots[arm], seed,
                    0.25 if arm == "with_negative" else 0.0,
                )
                for seed in SEEDS
            ]
            for arm in ARMS
        }
        for positive, negative in zip(
            arms["with_negative"], arms["no_negative"], strict=True
        ):
            verify_pair(positive, negative)
        summaries = {
            arm: {
                metric: summarize(
                    [member["selected_metrics"][metric] for member in arms[arm]]
                )
                for metric, _ in SELECTED_METRICS
            }
            for arm in ARMS
        }
        paired_deltas = {}
        for metric, _ in SELECTED_METRICS:
            deltas = [
                no["selected_metrics"][metric] - yes["selected_metrics"][metric]
                for yes, no in zip(
                    arms["with_negative"], arms["no_negative"], strict=True
                )
            ]
            paired_deltas[metric] = paired_bootstrap(deltas)
            paired_deltas[metric]["by_seed"] = dict(zip(map(str, SEEDS), deltas))
        payload["datasets"][dataset] = {
            "roots": {key: str(value) for key, value in dataset_roots.items()},
            "arms": {
                arm: {
                    "members": arms[arm],
                    "selected_summary": summaries[arm],
                    "curve_summary": curve_rows(arms[arm]),
                }
                for arm in ARMS
            },
            "paired_deltas": paired_deltas,
        }
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.tmp-", dir=args.output_root.parent
        )
    )
    try:
        audit = temporary / "NEGATIVE_SAMPLE_ABLATION_AUDIT.json"
        report = temporary / "NEGATIVE_SAMPLE_ABLATION_REPORT.md"
        curves = temporary / "NEGATIVE_SAMPLE_LOSS_CURVES.csv"
        figure = temporary / "NEGATIVE_SAMPLE_LOSS_CURVES.svg"
        audit.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report.write_text(render_report(payload), encoding="utf-8")
        fieldnames = (
            "dataset", "arm", "epoch", "train_contact_bce_n",
            "train_contact_bce_mean", "train_contact_bce_sd",
            "train_shared_objective_n", "train_shared_objective_mean",
            "train_shared_objective_sd", "validation_contact_raw_brier_n",
            "validation_contact_raw_brier_mean",
            "validation_contact_raw_brier_sd",
        )
        with curves.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for dataset, study in payload["datasets"].items():
                for arm in ARMS:
                    for row in study["arms"][arm]["curve_summary"]:
                        writer.writerow(
                            {
                                "dataset": dataset,
                                "arm": arm,
                                **{key: row.get(key) for key in fieldnames[2:]},
                            }
                        )
        plot_curves(payload, figure)
        receipt = {
            "format_version": "framecontact_partner_negative_ablation_receipt_v1",
            "audit_sha256": sha256_file(audit),
            "report_sha256": sha256_file(report),
            "curve_csv_sha256": sha256_file(curves),
            "curve_svg_sha256": sha256_file(figure),
            "test_loaded": False,
            "external_test_loaded": False,
        }
        (temporary / "RUN_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output_root)
    except Exception:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise
    print(args.output_root)


if __name__ == "__main__":
    main()
