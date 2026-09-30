#!/usr/bin/env python3
"""Generate the compact paper overview from frozen, paper-safe metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
METRICS = json.loads((ROOT / "data/frozen_metrics.json").read_text())


def box(ax, xy, width, height, text, color, fontsize=8):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=color,
        facecolor=color + "18",
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1F2937",
    )


def arrow(ax, start, end, color="#64748B", style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            linestyle=style,
            color=color,
        )
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(10.2, 3.45))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.25)

    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    blue, teal, orange, purple = "#275DAD", "#1F8A70", "#E57A44", "#6D5BD0"

    box(ax, (0.02, 0.63), 0.17, 0.22, "ESM-2 residue\nembeddings", blue)
    box(ax, (0.02, 0.20), 0.17, 0.22, "Sequence-predicted\nN/CA/C frames", teal)
    box(ax, (0.27, 0.42), 0.19, 0.25, "Separate Ab/Ag\nintra-chain\nencoders", purple)
    box(ax, (0.54, 0.42), 0.17, 0.25, "Explicit pair\ncontact map", orange)
    box(ax, (0.78, 0.64), 0.19, 0.20, "Joint baseline\ncontact + paratope", blue)
    box(ax, (0.78, 0.20), 0.19, 0.20, "TopK2 expert\nepitope only", orange)
    arrow(ax, (0.19, 0.74), (0.27, 0.58))
    arrow(ax, (0.19, 0.31), (0.27, 0.50))
    arrow(ax, (0.46, 0.545), (0.54, 0.545))
    arrow(ax, (0.71, 0.57), (0.78, 0.72), blue)
    arrow(ax, (0.71, 0.50), (0.78, 0.31), orange)
    ax.text(
        0.365,
        0.31,
        "interleaved Ab/Ag\ncontent cross-attention",
        ha="center",
        va="center",
        fontsize=6.7,
        color="#475569",
    )
    ax.text(0.5, 0.94, "(a) FrameContact and deterministic task routing", ha="center", fontsize=9, fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    delta = METRICS["fresh156"]["epitope_delta_by_seed"]
    bootstrap = METRICS["fresh156"]["paired_antigen_cluster_bootstrap"]
    seeds = ["17", "29", "43"]
    values = [delta[s] for s in seeds]
    bars = ax.bar([0, 1, 2], values, color=[blue, teal, orange], width=0.62)
    point = bootstrap["point_estimate"]
    ci_low, ci_high = bootstrap["ci95"]
    ax.errorbar(
        3,
        point,
        yerr=[[point - ci_low], [ci_high - point]],
        fmt="D",
        color="#111827",
        capsize=4,
        markersize=5,
        linewidth=1.2,
        label="cluster-weighted, seed-averaged",
    )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_ylabel("Epitope MPC-AP difference")
    ax.set_xticks([0, 1, 2, 3], ["Seed 17", "Seed 29", "Seed 43", "Cluster-\nweighted"])
    ax.set_title("(b) Earlier fresh156 route confirmation", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ymax = max(ci_high, max(values)) * 1.16
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.5, 3.5)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.025,
            f"+{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.4,
        )
    ax.text(
        3.07,
        point + ymax * 0.012,
        "+0.0295",
        ha="left",
        va="bottom",
        fontsize=7.4,
    )
    ax.text(
        0.02,
        0.97,
        "Bars: complex-weighted per seed\nDiamond: 49-cluster estimate (95% CI)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="#334155",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.savefig(ROOT / "figures/overview.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(ROOT / "figures/overview.png", dpi=220, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    main()
