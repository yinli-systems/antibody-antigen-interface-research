#!/usr/bin/env python3
"""Aggregate fixed-final-epoch SAbDab2 geometry ablations across seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


FORMAT_VERSION = "sabdab2_self_attention_geometry_multiseed_v1"
EXPECTED_RUN_FORMAT = "sabdab2_self_attention_geometry_ablation_v2"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Input must be a regular non-symlink file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != EXPECTED_RUN_FORMAT:
        raise ValueError(f"Unexpected run format in {path}")
    if payload.get("claim_boundary") is None:
        raise ValueError(f"Run lacks a claim boundary: {path}")
    return payload


def _scientific_contract(payload: dict[str, Any]) -> dict[str, Any]:
    configuration = dict(payload["configuration"])
    configuration.pop("seed", None)
    runtime = dict(payload["runtime"])
    runtime.pop("device", None)
    return {
        "data": payload["data"],
        "configuration_without_seed": configuration,
        "runtime_contract": runtime,
        "model_parameter_count": payload["model_parameter_count"],
        "mode_order": [item["mode"] for item in payload["modes"]],
    }


def _summary(values: list[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Metrics must be non-empty and finite")
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    arguments = _arguments()
    if len(arguments.inputs) < 2:
        raise ValueError("At least two seed runs are required")
    if arguments.output.exists() or arguments.output.is_symlink():
        raise ValueError("Output already exists")
    paths = [path.expanduser().resolve() for path in arguments.inputs]
    if len(set(paths)) != len(paths):
        raise ValueError("Input paths must be unique")
    runs = [_load(path) for path in paths]
    contracts = [_scientific_contract(run) for run in runs]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Seed runs do not share one scientific contract")
    seeds = [int(run["configuration"]["seed"]) for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed values must be unique")

    mode_names = contracts[0]["mode_order"]
    metrics: dict[str, Any] = {}
    for mode_index, mode in enumerate(mode_names):
        final_validations = [
            run["modes"][mode_index]["history"][-1]["validation"]
            for run in runs
        ]
        if any(run["modes"][mode_index]["mode"] != mode for run in runs):
            raise ValueError("Mode order differs across seed runs")
        metrics[mode] = {
            "macro_auprc": _summary(
                [float(value["macro_auprc"]) for value in final_validations]
            ),
            "paratope_auprc": _summary(
                [float(value["paratope"]["auprc"]) for value in final_validations]
            ),
            "epitope_auprc": _summary(
                [float(value["epitope"]["auprc"]) for value in final_validations]
            ),
            "macro_auroc": _summary(
                [float(value["macro_auroc"]) for value in final_validations]
            ),
        }

    payload = {
        "format_version": FORMAT_VERSION,
        "claim_boundary": (
            "Development-only multi-seed aggregation of fixed-final-epoch validation. "
            "No frozen test, external, clinical, or benchmark-improvement claim."
        ),
        "seed_count": len(seeds),
        "seeds": seeds,
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in paths
        ],
        "scientific_contract": contracts[0],
        "metrics": metrics,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical_bytes(payload))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
