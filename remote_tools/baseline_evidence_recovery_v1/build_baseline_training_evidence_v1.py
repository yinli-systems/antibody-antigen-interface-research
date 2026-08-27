#!/usr/bin/env python3
"""Recover auditable baseline training curves from immutable archived runs.

This script is intentionally read-only with respect to the source runs.  It
copies only scalar histories and hashes into a new evidence directory.  It
does not read or score any held-out labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EPI_EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s+\|.*?LR:\s+([0-9.eE+-]+)")
EPI_TOTAL_RE = re.compile(
    r"Train Loss:\s+([0-9.eE+-]+)\s+\|\s+Val Loss:\s+([0-9.eE+-]+)"
)
EPI_COMPONENT_RE = re.compile(r"([A-Za-z0-9_]+):\s+([0-9.eE+-]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{location}: expected a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{location}: non-finite number")
    return value


def svg_curve(
    path: Path,
    title: str,
    x: Sequence[int],
    series: Sequence[tuple[str, Sequence[float], str, str]],
    selected_epochs: Iterable[int] = (),
) -> None:
    """Write a dependency-free SVG line chart.

    Each series tuple is (label, values, colour, dash-array).  The SVG embeds
    fonts as text rather than a bitmap and therefore does not create Type-3
    PDF fonts when included by modern LaTeX engines.
    """
    if not x or any(len(values) != len(x) for _, values, _, _ in series):
        raise ValueError(f"invalid curve data for {path}")
    all_values = [finite_number(v, str(path)) for _, values, _, _ in series for v in values]
    lo, hi = min(all_values), max(all_values)
    if math.isclose(lo, hi):
        pad = abs(lo or 1.0) * 0.05
        lo, hi = lo - pad, hi + pad
    width, height = 960, 520
    left, right, top, bottom = 84, 28, 58, 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    xmin, xmax = min(x), max(x)
    xspan = max(1, xmax - xmin)

    def sx(value: int) -> float:
        return left + (value - xmin) / xspan * plot_w

    def sy(value: float) -> float:
        return top + (hi - value) / (hi - lo) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="32" font-family="Arial,Helvetica,sans-serif" font-size="22" fill="#263238">{html.escape(title)}</text>',
    ]
    for index in range(6):
        y = top + index / 5 * plot_h
        value = hi - index / 5 * (hi - lo)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#dfe5e8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial,Helvetica,sans-serif" font-size="12" fill="#607078">{value:.4g}</text>'
        )
    for epoch in sorted(set(selected_epochs)):
        if xmin <= epoch <= xmax:
            xx = sx(epoch)
            parts.append(
                f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top+plot_h}" stroke="#9b59b6" stroke-width="1.5" stroke-dasharray="3 4"/>'
            )
            parts.append(
                f'<text x="{xx+4:.2f}" y="{top+14}" font-family="Arial,Helvetica,sans-serif" font-size="11" fill="#7d3c98">selected {epoch}</text>'
            )
    for label, values, colour, dash in series:
        points = " ".join(f"{sx(epoch):.2f},{sy(float(value)):.2f}" for epoch, value in zip(x, values))
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2.2"{dash_attribute}/>'
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#607078"/>',
            f'<text x="{left}" y="{height-35}" font-family="Arial,Helvetica,sans-serif" font-size="12" fill="#607078">epoch {xmin}</text>',
            f'<text x="{width-right}" y="{height-35}" text-anchor="end" font-family="Arial,Helvetica,sans-serif" font-size="12" fill="#607078">epoch {xmax}</text>',
        ]
    )
    legend_x = left
    for label, _, colour, dash in series:
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{legend_x}" y1="{height-15}" x2="{legend_x+28}" y2="{height-15}" stroke="{colour}" stroke-width="2.2"{dash_attribute}/>'
        )
        parts.append(
            f'<text x="{legend_x+34}" y="{height-11}" font-family="Arial,Helvetica,sans-serif" font-size="12" fill="#263238">{html.escape(label)}</text>'
        )
        legend_x += 42 + 8 * len(label)
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_epiformer_log(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    records: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        epoch_match = EPI_EPOCH_RE.search(lines[index])
        if epoch_match is None:
            index += 1
            continue
        epoch = int(epoch_match.group(1))
        record: dict[str, Any] = {"epoch": epoch, "learning_rate": float(epoch_match.group(2))}
        for offset in range(1, min(7, len(lines) - index)):
            line = lines[index + offset]
            total_match = EPI_TOTAL_RE.search(line)
            if total_match:
                record["train_loss"] = float(total_match.group(1))
                record["validation_loss"] = float(total_match.group(2))
            elif line.startswith("Train:"):
                record["train_components"] = {
                    key: float(value) for key, value in EPI_COMPONENT_RE.findall(line)
                }
            elif line.startswith("Val:"):
                record["validation_components"] = {
                    key: float(value) for key, value in EPI_COMPONENT_RE.findall(line)
                }
        required = {"train_loss", "validation_loss", "train_components", "validation_components"}
        if not required.issubset(record):
            raise ValueError(f"{path}: incomplete EpiFormer record for epoch {epoch}")
        records.append(record)
        index += 1
    return records


def recover_epiformer(output: Path) -> dict[str, Any]:
    log_epoch1 = Path("/ssd/scxi253/model_deployment/logs/epiformer-train-1260704.out")
    log_resume = Path("/ssd/scxi253/model_deployment/logs/epiformer-train-1260772.out")
    launcher = Path("/ssd/scxi253/model_deployment/jobs/epiformer_train.sbatch")
    trainer = Path("/ssd/scxi253/model_deployment/src/epiformer/code/trainer.py")
    checkpoint_root = Path(
        "/ssd/scxi253/model_deployment/data/results/epiformer/checkpoints/formal_epi_group_seed42"
    )
    epoch1_records = parse_epiformer_log(log_epoch1)
    resumed_records = parse_epiformer_log(log_resume)
    if [item["epoch"] for item in epoch1_records] != [1]:
        raise ValueError("EpiFormer predecessor log must contain exactly epoch 1")
    records = epoch1_records + resumed_records
    epochs = [item["epoch"] for item in records]
    if epochs != list(range(1, 58)):
        raise ValueError(f"EpiFormer epochs are not contiguous 1..57: {epochs}")
    best = min(records, key=lambda item: item["validation_loss"])
    write_json(output / "history.json", records)
    flat_rows = []
    component_names = sorted(records[0]["train_components"])
    for item in records:
        row: dict[str, Any] = {
            "epoch": item["epoch"],
            "learning_rate": item["learning_rate"],
            "train_loss": item["train_loss"],
            "validation_loss": item["validation_loss"],
        }
        for name in component_names:
            row[f"train_{name}"] = item["train_components"][name]
            row[f"validation_{name}"] = item["validation_components"][name]
        flat_rows.append(row)
    write_csv(output / "history.csv", flat_rows)
    svg_curve(
        output / "loss_curves.svg",
        "EpiFormer epitope-group training loss (seed 42)",
        epochs,
        [
            ("train total", [item["train_loss"] for item in records], "#21618c", ""),
            ("validation total", [item["validation_loss"] for item in records], "#c0392b", "7 5"),
        ],
        [int(best["epoch"])],
    )
    svg_curve(
        output / "epitope_loss_components.svg",
        "EpiFormer epitope loss components (validation)",
        epochs,
        [
            ("BCE", [item["validation_components"]["epitope_bce_loss"] for item in records], "#21618c", ""),
            ("Dice", [item["validation_components"]["epitope_dice_loss"] for item in records], "#1e8449", ""),
            ("count", [item["validation_components"]["epitope_count_loss"] for item in records], "#c0392b", "7 5"),
        ],
        [int(best["epoch"])],
    )
    source_paths = [log_epoch1, log_resume, launcher, trainer]
    checkpoints = {}
    for name in ("epiformer_best.pt", "epiformer_last.pt"):
        path = checkpoint_root / name
        if path.is_file():
            checkpoints[name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    return {
        "status": "PASS_ARCHIVED_COMPLETE_HISTORY_RECOVERED",
        "method": "EpiFormer",
        "training_kind": "project_internal_retraining",
        "output_granularity": "antigen_residue_epitope_score; pair values are projection, not native pair scores",
        "epochs": len(records),
        "seed": 42,
        "selected_epoch_by_minimum_validation_loss": int(best["epoch"]),
        "selected_validation_loss": best["validation_loss"],
        "early_stopping_epoch": 57,
        "history_source_policy": "epoch 1 from immediate predecessor job 1260704; epochs 2-57 from resumed job 1260772",
        "source_files": {
            str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in source_paths
        },
        "checkpoints": checkpoints,
        "held_out_access_performed_by_this_recovery": False,
    }


def recover_mipe_track(output: Path, root: Path) -> dict[str, Any]:
    track_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    track_id = track_manifest["track_id"]
    histories: list[list[dict[str, Any]]] = []
    selections = []
    source_files: dict[str, Any] = {}
    for fold in range(5):
        history_path = root / f"fold_{fold}/state/epochs/epoch_0800/history.json"
        selection_path = root / f"fold_{fold}/state/SELECTION_FROZEN.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if len(history) != 800 or [item["epoch"] for item in history] != list(range(1, 801)):
            raise ValueError(f"MIPE fold {fold}: incomplete 800-epoch history")
        for item in history:
            finite_number(item["train"]["loss"], f"MIPE fold {fold} train")
            finite_number(item["validation"]["loss"], f"MIPE fold {fold} validation")
        histories.append(history)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selections.append(selection)
        source_files[str(history_path)] = {"sha256": sha256(history_path), "bytes": history_path.stat().st_size}
        source_files[str(selection_path)] = {"sha256": sha256(selection_path), "bytes": selection_path.stat().st_size}
        rows = [
            {
                "epoch": item["epoch"],
                "train_loss": item["train"]["loss"],
                "validation_loss": item["validation"]["loss"],
                "validation_joint_auprc": item["validation_joint_auprc"],
                "train_antigen_auprc": item["train"]["antigen"]["auprc"],
                "train_antibody_auprc": item["train"]["antibody"]["auprc"],
                "validation_antigen_auprc": item["validation"]["antigen"]["auprc"],
                "validation_antibody_auprc": item["validation"]["antibody"]["auprc"],
            }
            for item in history
        ]
        write_csv(output / f"fold_{fold}_history.csv", rows)
        selected_epochs = [selection["primary"]["epoch"], selection["secondary"]["epoch"]]
        svg_curve(
            output / f"fold_{fold}_loss_curves.svg",
            f"MIPE fold {fold} loss (seed 42, 800 epochs)",
            list(range(1, 801)),
            [
                ("train", [item["train"]["loss"] for item in history], "#21618c", ""),
                ("validation", [item["validation"]["loss"] for item in history], "#c0392b", "7 5"),
            ],
            selected_epochs,
        )
    mean_rows = []
    for index in range(800):
        mean_rows.append(
            {
                "epoch": index + 1,
                "mean_train_loss": statistics.fmean(history[index]["train"]["loss"] for history in histories),
                "mean_validation_loss": statistics.fmean(history[index]["validation"]["loss"] for history in histories),
                "mean_validation_joint_auprc": statistics.fmean(history[index]["validation_joint_auprc"] for history in histories),
            }
        )
    write_csv(output / "five_fold_mean_history.csv", mean_rows)
    svg_curve(
        output / "five_fold_mean_loss_curves.svg",
        "MIPE five-fold mean loss (seed 42, 800 epochs)",
        list(range(1, 801)),
        [
            ("mean train", [item["mean_train_loss"] for item in mean_rows], "#21618c", ""),
            ("mean validation", [item["mean_validation_loss"] for item in mean_rows], "#c0392b", "7 5"),
        ],
        [selection["secondary"]["epoch"] for selection in selections],
    )
    manifest_path = root / "run_manifest.json"
    source_files[str(manifest_path)] = {"sha256": sha256(manifest_path), "bytes": manifest_path.stat().st_size}
    return {
        "status": "PASS_ARCHIVED_COMPLETE_HISTORY_RECOVERED",
        "method": "MIPE",
        "track_id": track_id,
        "training_kind": "five_fold_project_internal_retraining",
        "output_granularity": "native interaction matrix plus antibody/antigen marginals",
        "fold_count": 5,
        "epochs_per_fold": 800,
        "seed": 42,
        "primary_checkpoint_rule": "epoch_800_endpoint",
        "secondary_checkpoint_rule": "validation_best_joint_auprc",
        "secondary_selected_epochs": [selection["secondary"]["epoch"] for selection in selections],
        "source_files": source_files,
        "held_out_access_performed_by_this_recovery": False,
    }


def recover_mipe(output: Path) -> dict[str, Any]:
    tracks = {
        "paper_appendix_lr1e4": Path(
            "/ssd/scxi253/model_deployment/results/mipe/paper_appendix_lr1e4_seed42_fivefold_v3"
        ),
        "released_code_lr1e3": Path(
            "/ssd/scxi253/model_deployment/results/mipe/released_code_lr1e3_seed42_hardened_v3"
        ),
    }
    recovered = {
        name: recover_mipe_track(output / name, root) for name, root in tracks.items()
    }
    return {
        "status": "PASS_TWO_AUTHENTICATED_800_EPOCH_TRACKS_RECOVERED",
        "method": "MIPE",
        "track_relationship": {
            "paper_appendix_lr1e4": "paper-hyperparameter track; no held-out result is bound here",
            "released_code_lr1e3": "released-code-compatible track; existing held-out evaluation binds only to this track",
        },
        "tracks": recovered,
        "held_out_access_performed_by_this_recovery": False,
    }


def recover_sppider(output: Path) -> dict[str, Any]:
    root = Path("/ssd/scxi253/model_deployment/results/sabdab2/sppider_seq")
    selected_members = {
        17: {
            "epitope": ["cluster_val_v3_base_seed17", "cluster_val_v3_lr3e5_seed17"],
            "paratope": ["cluster_val_v3_base_seed17", "cluster_val_v3_lr1e5_seed17"],
        },
        42: {
            "epitope": ["cluster_val_v3_scratch_seed42", "cluster_val_v3_lr3e5_seed42"],
            "paratope": ["cluster_val_v3_scratch_seed42", "cluster_val_v3_lr1e5_seed42"],
        },
        73: {
            "epitope": ["cluster_val_v3_base_seed73", "cluster_val_v3_lr3e5_seed73"],
            "paratope": ["cluster_val_v3_base_seed73", "cluster_val_v3_lr1e5_seed73"],
        },
    }
    members = []
    source_files: dict[str, Any] = {}
    for seed, roles in selected_members.items():
        for role, run_names in roles.items():
            for run_name in run_names:
                history_path = root / run_name / role / "checkpoints/history.json"
                payload = json.loads(history_path.read_text(encoding="utf-8"))
                history = payload["history"]
                epochs = [int(item["epoch"]) for item in history]
                if epochs != list(range(1, epochs[-1] + 1)):
                    raise ValueError(f"SPPIDER-seq non-contiguous history: {history_path}")
                best = min(history, key=lambda item: item["validation_loss"])
                rows = [
                    {
                        "epoch": item["epoch"],
                        "training_loss": item["training_loss"],
                        "validation_loss": item["validation_loss"],
                        "best_validation_loss": item["best_validation_loss"],
                        "improved": item["improved"],
                    }
                    for item in history
                ]
                member_slug = f"seed{seed}_{role}_{run_name}"
                write_csv(output / f"{member_slug}.csv", rows)
                svg_curve(
                    output / f"{member_slug}.svg",
                    f"SPPIDER-seq {role}, seed {seed}, {run_name}",
                    epochs,
                    [
                        ("train", [item["training_loss"] for item in history], "#21618c", ""),
                        ("validation", [item["validation_loss"] for item in history], "#c0392b", "7 5"),
                    ],
                    [int(best["epoch"])],
                )
                source_files[str(history_path)] = {
                    "sha256": sha256(history_path),
                    "bytes": history_path.stat().st_size,
                }
                members.append(
                    {
                        "seed": seed,
                        "role": role,
                        "run_name": run_name,
                        "epochs": len(history),
                        "selected_epoch_by_minimum_validation_loss": int(best["epoch"]),
                        "selected_validation_loss": best["validation_loss"],
                    }
                )
    split_audit = root / "research/cluster_validation_v3_split_audit.json"
    source_files[str(split_audit)] = {"sha256": sha256(split_audit), "bytes": split_audit.stat().st_size}
    return {
        "status": "PASS_ARCHIVED_COMPLETE_HISTORY_RECOVERED",
        "method": "SPPIDER-seq",
        "training_kind": "cluster_aware_internal_retraining_and_validation_selected_ensembles",
        "output_granularity": "antigen-residue epitope and antibody-residue paratope scores",
        "seeds": [17, 42, 73],
        "split_counts": {"train": 2727, "validation": 481, "test": 455},
        "selected_training_members": members,
        "source_files": source_files,
        "held_out_access_performed_by_this_recovery": False,
    }


def discotope_record() -> dict[str, Any]:
    return {
        "status": "TRAINING_CURVE_NOT_APPLICABLE_RELEASED_ENSEMBLE",
        "method": "DiscoTope-3.0",
        "training_kind": "released_pretrained_100_model_xgboost_ensemble_reproduction",
        "output_granularity": "antigen-residue epitope score; pair-domain value is projection",
        "train_loss_curve_applicable": False,
        "validation_loss_curve_applicable": False,
        "reason": "The public baseline distributes fitted XGBoost ensemble members. This project reproduces inference and does not retrain the ensemble, so no per-epoch neural optimization trajectory exists.",
        "required_reproduction_evidence": [
            "official source revision",
            "models.zip SHA256 and 100-member count",
            "ESM-IF1/structure feature provenance",
            "per-residue raw prediction snapshot",
            "held-out report-only scoring receipt",
        ],
        "held_out_access_performed_by_this_recovery": False,
    }


def build_receipt(output: Path, result_path: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "BASELINE_TRAINING_EVIDENCE_V1_RECEIPT.json":
            files[str(path.relative_to(output))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return {
        "schema_version": "baseline-training-evidence-receipt-v1",
        "status": "PASS_RECOVERED_TRAINING_EVIDENCE",
        "result": {"path": str(result_path), "sha256": sha256(result_path)},
        "files": files,
        "test_or_temporal_data_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": "baseline-training-evidence-v1",
        "status": "PASS_RECOVERED_TRAINING_EVIDENCE_WITH_DISCOTOPE_NOT_APPLICABLE",
        "methods": {
            "epiformer": recover_epiformer(output / "epiformer"),
            "mipe": recover_mipe(output / "mipe"),
            "sppider_seq": recover_sppider(output / "sppider_seq"),
            "discotope_3": discotope_record(),
        },
        "scope": "training-history recovery only; no held-out scoring",
        "test_or_temporal_data_read": False,
    }
    result_path = output / "BASELINE_TRAINING_EVIDENCE_V1.json"
    write_json(result_path, result)
    receipt = build_receipt(output, result_path)
    receipt_path = output / "BASELINE_TRAINING_EVIDENCE_V1_RECEIPT.json"
    write_json(receipt_path, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
