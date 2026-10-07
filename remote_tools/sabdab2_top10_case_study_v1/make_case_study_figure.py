#!/usr/bin/env python3
"""Render the frozen SAbDab2 top-10 contact case-study evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BLUE = "#31688E"
ORANGE = "#E6863B"
GREEN = "#2A9D8F"
DARK = "#263238"
LIGHT = "#E9EEF2"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def blend(low: str, high: str, fraction: float) -> tuple[int, int, int]:
    def channels(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))

    start, end = channels(low), channels(high)
    fraction = min(1.0, max(0.0, fraction))
    return tuple(round(a + fraction * (b - a)) for a, b in zip(start, end, strict=True))


def short_identifier(identifier: str) -> str:
    fields = identifier.split("_")
    if len(fields) >= 4 and fields[0] == "pdb":
        return f"{fields[1].lstrip('0') or '0'} {fields[2]}:{fields[3]}"
    return identifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--pair-csv", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("status") != "PASS_SABDAB2_TOP10_CONTACT_CASE_STUDY":
        raise ValueError("Case-study result is not a verified pass")
    cases = result["cases"]
    if len(cases) != 20 or any(len(case["top_pairs"]) != 10 for case in cases):
        raise ValueError("Expected exactly 20 cases with 10 ranked pairs each")

    recommended = {
        row["instance_id"] for row in result["recommended_case_studies"]
    }
    args.case_csv.parent.mkdir(parents=True, exist_ok=True)
    case_fields = [
        "instance_id",
        "antibody_length",
        "antigen_length",
        "candidate_pair_count",
        "positive_pair_count",
        "top10_tn",
        "top10_fp",
        "top10_fn",
        "top10_tp",
        "top10_precision",
        "top10_recall",
        "first_true_contact_rank",
        "average_precision",
        "recommended_case_study",
    ]
    with args.case_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=case_fields)
        writer.writeheader()
        for case in cases:
            confusion = case["top10_confusion"]
            writer.writerow(
                {
                    "instance_id": case["instance_id"],
                    "antibody_length": case["antibody_length"],
                    "antigen_length": case["antigen_length"],
                    "candidate_pair_count": case["candidate_pair_count"],
                    "positive_pair_count": case["positive_pair_count"],
                    "top10_tn": confusion["tn"],
                    "top10_fp": confusion["fp"],
                    "top10_fn": confusion["fn"],
                    "top10_tp": confusion["tp"],
                    "top10_precision": case["top10_precision"],
                    "top10_recall": case["top10_recall"],
                    "first_true_contact_rank": case["first_true_contact_rank"],
                    "average_precision": case["average_precision"],
                    "recommended_case_study": case["instance_id"] in recommended,
                }
            )

    pair_fields = [
        "instance_id",
        "rank",
        "antibody_position_1based",
        "antigen_position_1based",
        "probability",
        "label",
    ]
    with args.pair_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        for case in cases:
            for pair in case["top_pairs"]:
                writer.writerow(
                    {
                        "instance_id": case["instance_id"],
                        **{field: pair[field] for field in pair_fields[1:]},
                    }
                )

    canvas = Image.new("RGB", (3200, 1850), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (1600, 65),
        "Frozen SAbDab2 Top-10 Contact Case Study",
        fill=DARK,
        font=font(60, bold=True),
        anchor="mm",
    )
    aggregate = result["aggregate_top10_confusion"]
    matrix = aggregate["matrix"]
    draw.text((100, 165), "a  Aggregate confusion matrix (20 complexes)", fill=DARK, font=font(34, bold=True))
    grid_x, grid_y = 240, 300
    cell_width, cell_height = 330, 215
    maximum_log = math.log10(max(max(row) for row in matrix) + 1)
    for row in range(2):
        for column in range(2):
            value = int(matrix[row][column])
            intensity = math.log10(value + 1) / maximum_log
            box = (
                grid_x + column * cell_width,
                grid_y + row * cell_height,
                grid_x + (column + 1) * cell_width,
                grid_y + (row + 1) * cell_height,
            )
            draw.rectangle(box, fill=blend("#F4F8FB", BLUE, intensity), outline="white", width=5)
            draw.text(
                ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                f"{value:,}",
                fill="white" if intensity > 0.62 else DARK,
                font=font(36, bold=True),
                anchor="mm",
            )
    for column, label in enumerate(("Not in Top-10", "In Top-10")):
        draw.text(
            (grid_x + (column + 0.5) * cell_width, grid_y - 28),
            label,
            fill=DARK,
            font=font(24),
            anchor="ms",
        )
    for row, label in enumerate(("Non-contact", "Contact")):
        draw.text(
            (grid_x - 25, grid_y + (row + 0.5) * cell_height),
            label,
            fill=DARK,
            font=font(24),
            anchor="rm",
        )
    draw.text(
        (grid_x + cell_width, grid_y + 2 * cell_height + 38),
        "Prediction by per-complex rank",
        fill=DARK,
        font=font(25),
        anchor="mt",
    )
    draw.text(
        (grid_x + cell_width, grid_y + 2 * cell_height + 100),
        f"Top-10 precision = {aggregate['precision']:.1%}   |   Top-10 recall = {aggregate['recall']:.1%}",
        fill=DARK,
        font=font(27, bold=True),
        anchor="mt",
    )

    ordered = sorted(
        cases,
        key=lambda case: (
            int(case["top10_confusion"]["tp"]),
            float(case["average_precision"]),
            case["instance_id"],
        ),
    )
    labels = [short_identifier(case["instance_id"]) for case in ordered]
    true_positive = [case["top10_confusion"]["tp"] for case in ordered]
    panel_x, panel_y = 1220, 165
    draw.text((panel_x, panel_y), "b  Per-complex Top-10 composition", fill=DARK, font=font(34, bold=True))
    label_right = panel_x + 255
    bar_left, bar_width = label_right + 30, 1250
    row_height, bar_height = 68, 42
    first_row = panel_y + 95
    for tick in range(0, 11, 2):
        x_value = bar_left + bar_width * tick / 10
        draw.line((x_value, first_row - 25, x_value, first_row + 19 * row_height + bar_height), fill="#D7E0E5", width=2)
        draw.text((x_value, first_row - 35), str(tick), fill="#546E7A", font=font(20), anchor="ms")
    for index, case in enumerate(ordered):
        y_value = first_row + index * row_height
        draw.text((label_right, y_value + bar_height / 2), labels[index], fill=DARK, font=font(22), anchor="rm")
        draw.rectangle((bar_left, y_value, bar_left + bar_width, y_value + bar_height), fill=LIGHT)
        false_width = bar_width * (10 - true_positive[index]) / 10
        if true_positive[index]:
            draw.rectangle(
                (bar_left + false_width, y_value, bar_left + bar_width, y_value + bar_height),
                fill=ORANGE,
            )
            draw.text(
                (bar_left + false_width + (bar_width - false_width) / 2, y_value + bar_height / 2),
                str(true_positive[index]),
                fill="white",
                font=font(21, bold=True),
                anchor="mm",
            )
        if case["instance_id"] in recommended:
            draw.text(
                (bar_left + bar_width + 25, y_value + bar_height / 2),
                "case study",
                fill=ORANGE,
                font=font(21, bold=True),
                anchor="lm",
            )
    footer_y = first_row + 20 * row_height + 20
    draw.text((bar_left + bar_width / 2, footer_y), "Pairs among the ten highest model scores", fill=DARK, font=font(25), anchor="mt")
    draw.rectangle((bar_left, footer_y + 58, bar_left + 38, footer_y + 86), fill=LIGHT)
    draw.text((bar_left + 50, footer_y + 72), "False contact", fill=DARK, font=font(21), anchor="lm")
    draw.rectangle((bar_left + 260, footer_y + 58, bar_left + 298, footer_y + 86), fill=ORANGE)
    draw.text((bar_left + 310, footer_y + 72), "True contact", fill=DARK, font=font(21), anchor="lm")

    draw.text((100, 985), "c  Candidate case-study quality", fill=DARK, font=font(34, bold=True))
    plot_left, plot_top, plot_width, plot_height = 200, 1090, 780, 500
    draw.line((plot_left, plot_top, plot_left, plot_top + plot_height), fill=DARK, width=3)
    draw.line((plot_left, plot_top + plot_height, plot_left + plot_width, plot_top + plot_height), fill=DARK, width=3)
    maximum_rank = max(case["first_true_contact_rank"] for case in cases)
    maximum_ap = max(case["average_precision"] for case in cases) * 1.15
    for tick in (1, 10, 100, 1000):
        x_value = plot_left + plot_width * math.log10(tick) / math.log10(maximum_rank)
        draw.line((x_value, plot_top, x_value, plot_top + plot_height), fill="#D7E0E5", width=2)
        draw.text((x_value, plot_top + plot_height + 18), str(tick), fill="#546E7A", font=font(20), anchor="mt")
    for tick_index in range(5):
        tick = maximum_ap * tick_index / 4
        y_value = plot_top + plot_height * (1 - tick / maximum_ap)
        draw.line((plot_left, y_value, plot_left + plot_width, y_value), fill="#D7E0E5", width=2)
        draw.text((plot_left - 15, y_value), f"{tick:.2f}", fill="#546E7A", font=font(20), anchor="rm")
    annotation_offsets = {
        "pdb_00008ol9_B_A": (24, -26),
        "pdb_00006vmk_H_G": (24, -28),
        "pdb_00006vmk_i_h": (24, 26),
    }
    for case in cases:
        x_value = plot_left + plot_width * math.log10(case["first_true_contact_rank"]) / math.log10(maximum_rank)
        y_value = plot_top + plot_height * (1 - case["average_precision"] / maximum_ap)
        hits = int(case["top10_confusion"]["tp"])
        radius = 11 + 5 * hits
        color = ORANGE if hits else BLUE
        draw.ellipse((x_value - radius, y_value - radius, x_value + radius, y_value + radius), fill=color, outline="white", width=3)
        if case["instance_id"] in recommended:
            offset_x, offset_y = annotation_offsets.get(case["instance_id"], (20, -20))
            draw.text(
                (x_value + radius + offset_x, y_value + offset_y),
                short_identifier(case["instance_id"]),
                fill=DARK,
                font=font(18, bold=True),
                anchor="lm",
            )
    draw.text((plot_left + plot_width / 2, plot_top + plot_height + 65), "Rank of first true contact (log scale)", fill=DARK, font=font(23), anchor="mt")
    draw.text((plot_left, plot_top - 32), "Full-grid AP", fill=DARK, font=font(23), anchor="ls")
    draw.ellipse((205, 1685, 227, 1707), fill=BLUE)
    draw.text((240, 1696), "No true contact in Top-10", fill=DARK, font=font(20), anchor="lm")
    draw.ellipse((535, 1685, 563, 1713), fill=ORANGE)
    draw.text((578, 1699), "At least one true contact in Top-10", fill=DARK, font=font(20), anchor="lm")

    draw.text(
        (1600, 1795),
        "Fixed random sample (n=20); checkpoint selected on development only. Historical exposed test data are descriptive, not a fresh blind estimate.",
        fill="#546E7A",
        font=font(22),
        anchor="mm",
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output_prefix.with_suffix(".png"), dpi=(300, 300), optimize=True)
    canvas.save(args.output_prefix.with_suffix(".pdf"), "PDF", resolution=300.0)


if __name__ == "__main__":
    main()
