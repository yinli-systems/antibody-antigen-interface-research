#!/usr/bin/env python3
"""Generate reproducible publication-quality SPPIDER-seq test figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from sabdab2_sppider_seq import atomic_write_json, sha256_file


ROLE_LABELS = {"paratope": "Paratope", "epitope": "Epitope"}
METHOD_STYLES = {
    "Final ensemble": {"color": "#0072B2", "linestyle": "-", "linewidth": 2.4},
    "Previous run": {"color": "#D55E00", "linestyle": "--", "linewidth": 2.0},
    "Partner-permuted": {"color": "#009E73", "linestyle": ":", "linewidth": 2.2},
}


def load_predictions(path: Path, role: str) -> pd.DataFrame:
    """Load one residue prediction file and enforce the analysis contract."""
    frame = pd.read_parquet(path)
    required = {"instance_id", "role", "res_idx", "label", "score"}
    if set(frame.columns) != required:
        raise ValueError(f"Prediction columns violate the contract in {path}")
    if frame["role"].nunique() != 1 or frame["role"].iloc[0] != role:
        raise ValueError(f"Prediction role mismatch in {path}")
    if frame["score"].isna().any() or not np.isfinite(frame["score"]).all():
        raise ValueError(f"Prediction scores are not finite in {path}")
    return frame.sort_values(["instance_id", "role", "res_idx"], kind="stable").reset_index(
        drop=True
    )


def build_figure(args: argparse.Namespace) -> None:
    """Render ROC and precision-recall panels for both residue roles."""
    paths = {
        "paratope": {
            "Final ensemble": args.paratope_final,
            "Previous run": args.paratope_previous,
            "Partner-permuted": args.paratope_permuted,
        },
        "epitope": {
            "Final ensemble": args.epitope_final,
            "Previous run": args.epitope_previous,
            "Partner-permuted": args.epitope_permuted,
        },
    }
    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    summary: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for role, method_paths in paths.items():
        loaded[role] = {
            method: load_predictions(path.expanduser().resolve(), role)
            for method, path in method_paths.items()
        }
        reference = loaded[role]["Final ensemble"]
        key_columns = ["instance_id", "role", "res_idx"]
        for method, frame in loaded[role].items():
            if not reference[key_columns].equals(frame[key_columns]):
                raise ValueError(f"Prediction keys are not aligned for {role}/{method}")
            if not np.array_equal(
                reference["label"].to_numpy(dtype=np.uint8),
                frame["label"].to_numpy(dtype=np.uint8),
            ):
                raise ValueError(f"Prediction labels disagree for {role}/{method}")
        labels = reference["label"].to_numpy(dtype=np.uint8)
        summary[role] = {}
        for method, frame in loaded[role].items():
            scores = frame["score"].to_numpy(dtype=np.float64)
            summary[role][method] = {
                "path": str(method_paths[method].expanduser().resolve()),
                "sha256": sha256_file(method_paths[method]),
                "residue_count": len(frame),
                "instance_count": frame["instance_id"].nunique(),
                "positive_fraction": float(labels.mean()),
                "auroc": float(roc_auc_score(labels, scores)),
                "auprc": float(average_precision_score(labels, scores)),
            }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 8.2), constrained_layout=True)
    panel_labels = iter("ABCD")
    for row, role in enumerate(("paratope", "epitope")):
        reference = loaded[role]["Final ensemble"]
        labels = reference["label"].to_numpy(dtype=np.uint8)
        roc_axis = axes[row, 0]
        pr_axis = axes[row, 1]
        for method, frame in loaded[role].items():
            scores = frame["score"].to_numpy(dtype=np.float64)
            false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            style = METHOD_STYLES[method]
            roc_axis.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{method} ({summary[role][method]['auroc']:.3f})",
                **style,
            )
            pr_axis.plot(
                recall,
                precision,
                label=f"{method} ({summary[role][method]['auprc']:.3f})",
                **style,
            )
        roc_axis.plot([0, 1], [0, 1], color="#666666", linewidth=1.0, alpha=0.7)
        pr_axis.axhline(
            float(labels.mean()), color="#666666", linewidth=1.0, alpha=0.7
        )
        for axis in (roc_axis, pr_axis):
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.7)
            axis.legend(loc="lower right" if axis is roc_axis else "upper right", frameon=False)
            axis.text(
                -0.13,
                1.04,
                next(panel_labels),
                transform=axis.transAxes,
                fontsize=13,
                fontweight="bold",
            )
        roc_axis.set_title(f"{ROLE_LABELS[role]} ROC")
        roc_axis.set_xlabel("False positive rate")
        roc_axis.set_ylabel("True positive rate")
        pr_axis.set_title(f"{ROLE_LABELS[role]} precision-recall")
        pr_axis.set_xlabel("Recall")
        pr_axis.set_ylabel("Precision")

    figure.suptitle(
        "SPPIDER-seq on the frozen SAbDab2 test set",
        fontsize=14,
        fontweight="bold",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "final_test_curves.png"
    pdf_path = args.output_dir / "final_test_curves.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    manifest = {
        "stage": "final_test_figures",
        "description": (
            "ROC and precision-recall curves for aligned residue predictions on the "
            "frozen SAbDab2 test set. The partner-permuted control uses the same focal "
            "sequences with deterministically reassigned partner sequences."
        ),
        "roles": summary,
        "outputs": {
            "png": str(png_path.resolve()),
            "png_sha256": sha256_file(png_path),
            "pdf": str(pdf_path.resolve()),
            "pdf_sha256": sha256_file(pdf_path),
        },
    }
    atomic_write_json(args.output_dir / "figure_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paratope-final", type=Path, required=True)
    parser.add_argument("--paratope-previous", type=Path, required=True)
    parser.add_argument("--paratope-permuted", type=Path, required=True)
    parser.add_argument("--epitope-final", type=Path, required=True)
    parser.add_argument("--epitope-previous", type=Path, required=True)
    parser.add_argument("--epitope-permuted", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    """Parse arguments and generate final figures."""
    build_figure(build_parser().parse_args())


if __name__ == "__main__":
    main()
