#!/usr/bin/env python3
"""Plot audited epoch curves and the 2x2 task-competition screen."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def median_curve(runs, value_key, maximum_epoch=12):
    rows = []
    for epoch in range(1, maximum_epoch + 1):
        values = [
            row[value_key]
            for run in runs
            for row in run["curve"]
            if row["epoch"] == epoch
        ]
        if values:
            rows.append((epoch, statistics.median(values)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-audit", type=Path, required=True)
    parser.add_argument("--factorial-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()
    curves = json.loads(args.curve_audit.read_text())
    factorial = json.loads(args.factorial_result.read_text())

    colors = {
        "FrameContact": "#4C78A8",
        "EpiFormer-style": "#F58518",
        "epitope_only": "#4C78A8",
        "epitope_paratope": "#54A24B",
        "epitope_contact": "#E45756",
        "joint_current": "#B279A2",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    groups = {
        "FrameContact": curves["framecontact"]["runs"],
        "EpiFormer-style": curves["epiformer_style"]["runs"],
    }
    csv_rows = []
    for label, runs in groups.items():
        loss_curve = median_curve(runs, "train_epitope_loss_relative_to_epoch1")
        val_curve = median_curve(runs, "validation_epitope_mean_per_complex_auprc")
        axes[0, 0].plot(*zip(*loss_curve), marker="o", label=label, color=colors[label])
        axes[0, 1].plot(*zip(*val_curve), marker="o", label=label, color=colors[label])
        for epoch, value in loss_curve:
            csv_rows.append(["full_development", label, "median_normalized_train_epitope_loss", epoch, value])
        for epoch, value in val_curve:
            csv_rows.append(["full_development", label, "median_validation_epitope_auprc", epoch, value])

    axes[0, 0].set(title="Normalized epitope training loss", xlabel="Epoch", ylabel="Loss / epoch-1 loss")
    axes[0, 1].set(title="Validation epitope AUPRC", xlabel="Epoch", ylabel="Mean-per-complex AUPRC")
    axes[0, 0].legend()
    axes[0, 1].legend()

    arm_order = ["epitope_only", "epitope_paratope", "epitope_contact", "joint_current"]
    for arm in arm_order:
        runs = list(factorial["arms"][arm]["runs"].values())
        curve = median_curve(runs, "validation_epitope_mean_per_complex_auprc")
        axes[1, 0].plot(*zip(*curve), marker="o", label=arm, color=colors[arm])
        for epoch, value in curve:
            csv_rows.append(["factorial_screen", arm, "median_validation_epitope_auprc", epoch, value])
    axes[1, 0].set(title="2x2 task ablation: validation curves", xlabel="Epoch", ylabel="Mean-per-complex AUPRC")
    axes[1, 0].legend(fontsize=8)

    for index, arm in enumerate(arm_order):
        values = [factorial["arms"][arm]["runs"][str(seed)]["epitope_mean_per_complex_auprc"] for seed in (17, 29, 43)]
        axes[1, 1].scatter([index - 0.12, index, index + 0.12], values, color=colors[arm], s=42)
        axes[1, 1].plot([index - 0.18, index + 0.18], [statistics.median(values)] * 2, color="black", linewidth=2)
        for seed, value in zip((17, 29, 43), values):
            csv_rows.append(["factorial_screen", arm, f"selected_epitope_auprc_seed_{seed}", "selected", value])
    axes[1, 1].set_xticks(range(len(arm_order)), ["epi", "epi+para", "epi+contact", "joint"], rotation=15)
    axes[1, 1].set(title="Frozen best checkpoint by seed", ylabel="Mean-per-complex AUPRC")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle("SAbDab2 epitope learning dynamics and task interaction", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    with args.csv_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study", "model_or_arm", "metric", "epoch", "value"])
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
