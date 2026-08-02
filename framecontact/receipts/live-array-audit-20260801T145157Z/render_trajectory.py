#!/usr/bin/env python3
"""Render a bounded development-only trajectory figure from audited summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "matched-development-trajectories.png"


def load_series(path: Path, condition: str, metric: str) -> tuple[np.ndarray, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectory = payload["conditions"][condition]["complete_epoch_trajectory"]
    epochs = np.asarray([entry["epoch"] for entry in trajectory], dtype=float)
    means = np.asarray(
        [entry["metrics"][metric]["mean"] for entry in trajectory], dtype=float
    )
    standard_deviations = np.asarray(
        [entry["metrics"][metric]["sample_sd"] for entry in trajectory],
        dtype=float,
    )
    return epochs, means, standard_deviations


def main() -> None:
    conditions = (
        (
            ROOT / "residue-head-full.json",
            "framecontact_unary_only",
            "Unary-only",
            "#5965a8",
        ),
        (
            ROOT / "distogram-full.json",
            "framecontact_distogram",
            "Distogram",
            "#e45756",
        ),
    )
    panels = (
        ("contact_fixed_raw_precision", "Pair contact precision at logit 0"),
        ("contact_fixed_raw_recall", "Pair contact recall at logit 0"),
        ("contact_pooled_auprc", "Pooled pair contact AUPRC"),
        ("epitope_pooled_auprc", "Pooled epitope AUPRC"),
        ("paratope_pooled_auprc", "Pooled paratope AUPRC"),
        ("retrieval_worst_directional_margin", "Worst-direction partner margin"),
    )

    width, height = 2100, 1160
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = ImageFont.load_default(size=28)
    panel_font = ImageFont.load_default(size=20)
    tick_font = ImageFont.load_default(size=15)
    draw.text(
        (width // 2, 25),
        "FrameContact in-flight development validation (mean +/- sample SD, 5 seeds)",
        font=title_font,
        fill="#222222",
        anchor="ma",
    )
    draw.text(
        (width // 2, 62),
        "Descriptive only: no checkpoint selection and no frozen/external test access",
        font=panel_font,
        fill="#555555",
        anchor="ma",
    )

    outer_x, outer_y = 55, 115
    gap_x, gap_y = 40, 55
    panel_width = (width - 2 * outer_x - 2 * gap_x) // 3
    panel_height = (height - outer_y - 60 - gap_y) // 2

    for panel_index, (metric, title) in enumerate(panels):
        column, row = panel_index % 3, panel_index // 3
        left = outer_x + column * (panel_width + gap_x)
        top = outer_y + row * (panel_height + gap_y)
        right, bottom = left + panel_width, top + panel_height
        plot_left, plot_top = left + 78, top + 52
        plot_right, plot_bottom = right - 25, bottom - 58

        loaded = []
        for path, condition, label, color in conditions:
            epochs, means, deviations = load_series(path, condition, metric)
            loaded.append((epochs, means, deviations, label, color))
        x_min = min(float(epochs.min()) for epochs, *_ in loaded)
        x_max = max(float(epochs.max()) for epochs, *_ in loaded)
        y_min = min(float((means - deviations).min()) for _, means, deviations, *_ in loaded)
        y_max = max(float((means + deviations).max()) for _, means, deviations, *_ in loaded)
        y_min = min(y_min, 0.0) if metric.endswith("margin") else y_min
        y_max = max(y_max, 0.0) if metric.endswith("margin") else y_max
        y_pad = max((y_max - y_min) * 0.08, 1e-6)
        y_min, y_max = y_min - y_pad, y_max + y_pad

        def xy(epoch: float, value: float) -> tuple[float, float]:
            x = plot_left + (epoch - x_min) / max(x_max - x_min, 1.0) * (
                plot_right - plot_left
            )
            y = plot_bottom - (value - y_min) / max(y_max - y_min, 1e-12) * (
                plot_bottom - plot_top
            )
            return x, y

        draw.text(
            ((left + right) // 2, top + 8),
            title,
            font=panel_font,
            fill="#222222",
            anchor="ma",
        )
        for fraction in np.linspace(0.0, 1.0, 5):
            y_value = y_min + fraction * (y_max - y_min)
            _, y = xy(x_min, y_value)
            draw.line((plot_left, y, plot_right, y), fill="#d8d8d8", width=1)
            draw.text(
                (plot_left - 8, y),
                f"{y_value:.3g}",
                font=tick_font,
                fill="#555555",
                anchor="rm",
            )
        for epoch in np.linspace(x_min, x_max, 5):
            x, _ = xy(epoch, y_min)
            draw.line((x, plot_top, x, plot_bottom), fill="#eeeeee", width=1)
            draw.text(
                (x, plot_bottom + 10),
                f"{epoch:.0f}",
                font=tick_font,
                fill="#555555",
                anchor="ma",
            )
        if y_min <= 0.0 <= y_max:
            _, zero_y = xy(x_min, 0.0)
            draw.line((plot_left, zero_y, plot_right, zero_y), fill="#777777", width=2)
        draw.rectangle(
            (plot_left, plot_top, plot_right, plot_bottom), outline="#888888", width=2
        )
        draw.text(
            ((plot_left + plot_right) // 2, bottom - 20),
            "Epoch",
            font=tick_font,
            fill="#444444",
            anchor="ma",
        )

        for epochs, means, deviations, label, color in loaded:
            upper = [xy(float(x), float(y)) for x, y in zip(epochs, means + deviations)]
            lower = [xy(float(x), float(y)) for x, y in zip(epochs[::-1], (means - deviations)[::-1])]
            draw.polygon(upper + lower, fill=color + "28")
            points = [xy(float(x), float(y)) for x, y in zip(epochs, means)]
            draw.line(points, fill=color, width=5, joint="curve")

        if panel_index == 0:
            legend_x, legend_y = plot_left + 15, plot_top + 15
            for offset, (_, _, _, label, color) in enumerate(loaded):
                y = legend_y + offset * 28
                draw.line((legend_x, y, legend_x + 34, y), fill=color, width=5)
                draw.text(
                    (legend_x + 44, y),
                    label,
                    font=tick_font,
                    fill="#222222",
                    anchor="lm",
                )

    image.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
