#!/usr/bin/env python3
"""Render receipt-verified AsEP development learning curves.

The v1 FrameContact result schema stores the training objective and validation
metrics at every epoch, but it does not store a validation objective.  This
script therefore refuses to label a validation metric as ``validation loss``.
It emits a training-loss figure, a validation-metric figure, and a compact JSON
receipt that records this limitation explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


DEFAULT_SEEDS = (17, 42, 73, 101, 137)
GROUPS = ("contact", "epitope", "paratope")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Missing numeric {label}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {label}")
    return value


def load_run(run: Path) -> dict[str, Any]:
    result_path = run / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    receipt_path = run / "RUN_RECEIPT.json"
    if not result_path.is_file() or not receipt_path.is_file():
        raise ValueError(f"Incomplete run: {run}")
    if result_path.is_symlink() or receipt_path.is_symlink():
        raise ValueError(f"Symlinked authority is not accepted: {run}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("result_sha256") != _sha256(result_path):
        raise ValueError(f"Result hash mismatch: {run}")
    if receipt.get("test_loaded") is not False or receipt.get("external_test_loaded") is not False:
        raise ValueError(f"Non-development receipt: {run}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    authority = result.get("authority") or {}
    if authority.get("test_loaded") is not False or authority.get("external_test_loaded") is not False:
        raise ValueError(f"Non-development result: {run}")
    history = result.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Missing epoch history: {run}")
    rows: list[dict[str, Any]] = []
    for row in history:
        epoch = row.get("epoch")
        train = row.get("train") or {}
        validation = row.get("validation") or {}
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
            raise ValueError(f"Invalid epoch in {run}")
        metrics = {}
        for group in GROUPS:
            values = validation.get(group) or {}
            metrics[group] = {
                "auprc": _number(values.get("auprc"), f"{group}.auprc"),
                "mcc": _number(values.get("mcc"), f"{group}.mcc"),
                "f1": _number(values.get("f1"), f"{group}.f1"),
            }
        rows.append(
            {
                "epoch": epoch,
                "train_loss": _number(train.get("loss"), "train.loss"),
                "validation": metrics,
            }
        )
    epochs = [row["epoch"] for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError(f"Epoch history is not contiguous: {run}")
    if "loss" in (rows[0].get("validation") or {}):
        raise ValueError("Unexpected ambiguous validation loss field")
    return {
        "seed": int(result.get("configuration", {}).get("seed", receipt.get("seed"))),
        "experiment": str(result.get("experiment")),
        "selected_epoch": int(result.get("selection", {}).get("best_epoch")),
        "result_sha256": receipt["result_sha256"],
        "history": rows,
    }


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one run is required")
    experiments = {run["experiment"] for run in runs}
    if len(experiments) != 1:
        raise ValueError("All runs must use the same condition")
    matched_epochs = min(len(run["history"]) for run in runs)
    trajectory = []
    for index in range(matched_epochs):
        train = [run["history"][index]["train_loss"] for run in runs]
        row: dict[str, Any] = {
            "epoch": index + 1,
            "seed_count": len(runs),
            "train_loss": {
                "mean": statistics.mean(train),
                "sample_sd": statistics.stdev(train) if len(train) > 1 else 0.0,
            },
            "validation": {},
        }
        for group in GROUPS:
            values = [run["history"][index]["validation"][group]["auprc"] for run in runs]
            row["validation"][group] = {
                "auprc_mean": statistics.mean(values),
                "auprc_sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        trajectory.append(row)
    return {
        "format_version": "framecontact_asep_learning_curves_v1",
        "claim_boundary": (
            "Receipt-verified AsEP development train/validation history only; "
            "official, frozen, and external test roles were not loaded."
        ),
        "experiment": runs[0]["experiment"],
        "seeds": sorted(run["seed"] for run in runs),
        "members": [
            {
                "seed": run["seed"],
                "selected_epoch": run["selected_epoch"],
                "result_sha256": run["result_sha256"],
            }
            for run in sorted(runs, key=lambda item: item["seed"])
        ],
        "matched_epoch_count": matched_epochs,
        "validation_objective_available": False,
        "validation_objective_note": (
            "The v1 result schema did not record validation objective terms. "
            "Validation AUPRC trajectories are plotted separately and are not called loss."
        ),
        "trajectory": trajectory,
    }


def render(runs: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"contact": "#0072B2", "epitope": "#D55E00", "paratope": "#009E73"}

    figure, axis = pyplot.subplots(figsize=(7.2, 4.6))
    for run in sorted(runs, key=lambda item: item["seed"]):
        rows = run["history"][: summary["matched_epoch_count"]]
        axis.plot(
            [row["epoch"] for row in rows],
            [row["train_loss"] for row in rows],
            alpha=0.22,
            linewidth=0.9,
            color="#666666",
        )
    trajectory = summary["trajectory"]
    epochs = [row["epoch"] for row in trajectory]
    means = [row["train_loss"]["mean"] for row in trajectory]
    sds = [row["train_loss"]["sample_sd"] for row in trajectory]
    axis.plot(epochs, means, linewidth=2.2, color="#111111", label="5-seed mean")
    axis.fill_between(epochs, [m - s for m, s in zip(means, sds)], [m + s for m, s in zip(means, sds)], color="#999999", alpha=0.25, label="±1 sample SD")
    axis.set(xlabel="Epoch", ylabel="Training objective", title=f"{summary['experiment']} — development training loss")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "training_loss_curve.png", dpi=180)
    pyplot.close(figure)

    figure, axis = pyplot.subplots(figsize=(7.2, 4.6))
    for group in GROUPS:
        means = [row["validation"][group]["auprc_mean"] for row in trajectory]
        sds = [row["validation"][group]["auprc_sample_sd"] for row in trajectory]
        axis.plot(epochs, means, linewidth=2.0, color=colors[group], label=f"{group} AUPRC")
        axis.fill_between(epochs, [m - s for m, s in zip(means, sds)], [m + s for m, s in zip(means, sds)], color=colors[group], alpha=0.12)
    axis.set(xlabel="Epoch", ylabel="Validation AUPRC", title=f"{summary['experiment']} — development validation trajectories")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "validation_auprc_curves.png", dpi=180)
    pyplot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Write the compact receipt without importing matplotlib.",
    )
    args = parser.parse_args()
    runs = [load_run(args.runs_root / args.condition / f"seed-{seed}") for seed in args.seeds]
    if sorted(run["seed"] for run in runs) != sorted(args.seeds):
        raise ValueError("Run seeds do not match requested seeds")
    summary = summarize(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output_dir / "LEARNING_CURVE_RECEIPT.json"
    receipt_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.skip_render:
        render(runs, summary, args.output_dir)
    print(receipt_path)


if __name__ == "__main__":
    main()
