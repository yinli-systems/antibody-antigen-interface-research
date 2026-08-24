#!/usr/bin/env python3
"""Select development checkpoints from JSONL stdout using the frozen rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def records(path: Path) -> list[dict[str, object]]:
    output = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("{"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in value:
                output.append(value)
    if not output:
        raise ValueError(f"no epoch records: {path}")
    return output


def compact(name: str, path: Path) -> dict[str, object]:
    history = records(path)
    best = max(
        history,
        key=lambda row: (
            row["validation"]["epitope"]["mean_per_complex_auprc"],
            row["validation"]["contact"]["mean_per_complex_auprc"],
            -row["epoch"],
        ),
    )
    return {
        "name": name,
        "epochs_observed": len(history),
        "selected_epoch": best["epoch"],
        "epitope_mean_per_complex_auprc": best["validation"]["epitope"][
            "mean_per_complex_auprc"
        ],
        "contact_mean_per_complex_auprc": best["validation"]["contact"][
            "mean_per_complex_auprc"
        ],
        "paratope_mean_per_complex_auprc": best["validation"]["paratope"][
            "mean_per_complex_auprc"
        ],
        "elapsed_seconds_at_selected_epoch": best["elapsed_seconds"],
        "train_loss_at_selected_epoch": best["train"]["loss"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "LOG"))
    parser.add_argument("--control", nargs=2, metavar=("NAME", "LOG"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not arguments.run:
        raise ValueError("at least one --run is required")
    candidates = [compact(name, Path(path)) for name, path in arguments.run]
    winner = max(
        candidates,
        key=lambda row: (
            row["epitope_mean_per_complex_auprc"],
            row["contact_mean_per_complex_auprc"],
            -row["selected_epoch"],
        ),
    )
    result: dict[str, object] = {
        "format_version": "sabdab2_frozen_development_screen_summary_v1",
        "selection_rule": "epitope macro AUPRC; contact macro AUPRC; earlier epoch",
        "runs": candidates,
        "winner": winner["name"],
        "test_loaded": False,
    }
    if arguments.control:
        control = compact(arguments.control[0], Path(arguments.control[1]))
        result["control"] = control
        result["winner_minus_control"] = {
            key: winner[key] - control[key]
            for key in (
                "epitope_mean_per_complex_auprc",
                "contact_mean_per_complex_auprc",
                "paratope_mean_per_complex_auprc",
            )
        }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if arguments.output:
        if arguments.output.exists():
            raise FileExistsError(arguments.output)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(encoded)
    print(encoded.decode(), end="")


if __name__ == "__main__":
    main()
