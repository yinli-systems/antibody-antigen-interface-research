#!/usr/bin/env python3
"""Aggregate the frozen binary-head, distogram, and triangle development runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_structure import canonical_json_bytes, sha256_file  # noqa: E402


RUN_FORMAT = "framecontact_development_training_v1"
SUMMARY_FORMAT = "framecontact_head_distogram_multiseed_summary_v1"
SEEDS = (17, 42, 73, 101, 137)
CONDITIONS = (
    "framecontact_contact_hybrid",
    "framecontact_distogram",
    "framecontact_distogram_triangle",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-run-root", type=Path, required=True)
    parser.add_argument("--distogram-run-root", type=Path, required=True)
    parser.add_argument("--binary-protocol", type=Path, required=True)
    parser.add_argument("--binary-protocol-receipt", type=Path, required=True)
    parser.add_argument("--distogram-protocol", type=Path, required=True)
    parser.add_argument("--distogram-protocol-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mean_sd(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("aggregate metric values must be finite and nonempty")
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def _protocol_authority(protocol: Path, receipt: Path) -> tuple[dict, dict, str]:
    value = json.loads(protocol.read_text(encoding="utf-8"))
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    protocol_sha256 = sha256_file(protocol)
    if receipt_value.get("protocol_file_sha256") != protocol_sha256:
        raise ValueError(f"Protocol receipt differs: {protocol}")
    return value, receipt_value, protocol_sha256


def _validation_metrics(validation: Mapping[str, Any]) -> dict[str, float]:
    contact = validation["contact"]
    fixed_contact = contact["fixed_raw_logit_threshold_zero"]
    metrics = {
        "selection_margin": float(validation["selection_metric"]),
        "contact_mean_per_complex_auprc": float(contact["mean_per_complex_auprc"]),
        "contact_pooled_auprc": float(contact["auprc"]),
        "contact_pooled_brier": float(contact["brier"]),
        "contact_pooled_ece_15": float(contact["ece_15"]),
        "contact_validation_selected_threshold": float(
            contact["threshold_selected_on_validation"]
        ),
        "contact_validation_selected_precision": float(contact["precision"]),
        "contact_validation_selected_recall": float(contact["recall"]),
        "contact_validation_selected_f1": float(contact["f1"]),
        "contact_validation_selected_mcc": float(contact["mcc"]),
        "contact_validation_selected_predicted_positive_ratio": float(
            contact["predicted_positive_ratio"]
        ),
        "contact_fixed_raw_precision": float(fixed_contact["precision"]),
        "contact_fixed_raw_recall": float(fixed_contact["recall"]),
        "contact_fixed_raw_f1": float(fixed_contact["f1"]),
        "contact_fixed_raw_mcc": float(fixed_contact["mcc"]),
        "contact_fixed_raw_predicted_positive_ratio": float(
            fixed_contact["predicted_positive_ratio"]
        ),
    }
    for cutoff, values in contact["ranked_pair_macro"].items():
        metrics[f"contact_p_at_{cutoff}"] = float(values["mean_precision"])
        metrics[f"contact_recall_at_{cutoff}"] = float(values["mean_recall"])
    for residue_name in ("paratope", "epitope"):
        residue = validation[residue_name]
        fixed = residue["fixed_raw_logit_threshold_zero"]
        metrics[f"{residue_name}_mean_per_complex_auprc"] = float(
            residue["mean_per_complex_auprc"]
        )
        metrics[f"{residue_name}_pooled_auprc"] = float(residue["auprc"])
        metrics[f"{residue_name}_validation_selected_threshold"] = float(
            residue["threshold_selected_on_validation"]
        )
        for metric in (
            "precision",
            "recall",
            "f1",
            "mcc",
            "predicted_positive_ratio",
        ):
            metrics[f"{residue_name}_validation_selected_{metric}"] = float(
                residue[metric]
            )
        for metric in (
            "precision",
            "recall",
            "f1",
            "mcc",
            "predicted_positive_ratio",
        ):
            metrics[f"{residue_name}_fixed_raw_{metric}"] = float(fixed[metric])
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("selected-checkpoint metrics must all be finite")
    return metrics


def _metrics(result: Mapping[str, Any]) -> dict[str, float]:
    return _validation_metrics(result["selected_checkpoint_validation"])


def _epoch_trajectory(results: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Keep seed-level curves and summarize only epochs shared by all five seeds."""
    per_seed: dict[str, list[dict[str, Any]]] = {}
    epoch_metrics: dict[int, dict[int, dict[str, float]]] = {}
    for seed in SEEDS:
        history = results[seed].get("history")
        if not isinstance(history, list) or not history:
            raise ValueError(f"Run history is empty for seed {seed}")
        expected_epochs = list(range(1, len(history) + 1))
        observed_epochs = [int(record.get("epoch", -1)) for record in history]
        if observed_epochs != expected_epochs:
            raise ValueError(f"Run history epochs are not contiguous for seed {seed}")
        seed_curve = []
        for record in history:
            epoch = int(record["epoch"])
            local = _validation_metrics(record["validation"])
            epoch_metrics.setdefault(epoch, {})[seed] = local
            seed_curve.append({"epoch": epoch, **local})
        per_seed[str(seed)] = seed_curve
    complete_epochs = sorted(
        epoch for epoch, values in epoch_metrics.items() if set(values) == set(SEEDS)
    )
    complete_five_seed = {
        str(epoch): {
            metric: _mean_sd(
                [epoch_metrics[epoch][seed][metric] for seed in SEEDS]
            )
            for metric in epoch_metrics[epoch][SEEDS[0]]
        }
        for epoch in complete_epochs
    }
    return {
        "threshold_contract": {
            "fixed_raw": "raw logit zero, unchanged across epochs and conditions",
            "selected": "threshold selected on the same validation epoch; descriptive only",
        },
        "per_seed": per_seed,
        "complete_five_seed_epochs": complete_five_seed,
        "last_complete_five_seed_epoch": (
            complete_epochs[-1] if complete_epochs else None
        ),
    }


def _load_run(root: Path, condition: str, seed: int, protocol_sha256: str) -> dict:
    run = root / condition / f"seed-{seed}"
    result_path = run / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run / "RUN_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("format_version") != RUN_FORMAT
        or receipt.get("seed") != seed
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or receipt.get("result_sha256") != sha256_file(result_path)
        or receipt.get("selected_checkpoint_sha256") != sha256_file(checkpoint_path)
    ):
        raise ValueError(f"Run receipt validation failed: {condition}, seed {seed}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("format_version") != RUN_FORMAT
        or result.get("experiment") != condition
        or result.get("configuration", {}).get("seed") != seed
        or result.get("authority", {}).get("protocol_file_sha256") != protocol_sha256
        or result.get("authority", {}).get("test_loaded") is not False
        or result.get("authority", {}).get("external_test_loaded") is not False
    ):
        raise ValueError(f"Run metadata validation failed: {condition}, seed {seed}")
    return result


def _paired_delta(
    target: Mapping[int, dict[str, float]], reference: Mapping[int, dict[str, float]]
) -> dict[str, Any]:
    per_seed = {
        str(seed): {
            metric: target[seed][metric] - reference[seed][metric]
            for metric in target[seed]
        }
        for seed in SEEDS
    }
    return {
        "per_seed": per_seed,
        "five_seed": {
            metric: _mean_sd([per_seed[str(seed)][metric] for seed in SEEDS])
            for metric in next(iter(per_seed.values()))
        },
    }


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite summary: {arguments.output}")
    binary_protocol, _, binary_protocol_sha256 = _protocol_authority(
        arguments.binary_protocol, arguments.binary_protocol_receipt
    )
    distogram_protocol, _, distogram_protocol_sha256 = _protocol_authority(
        arguments.distogram_protocol, arguments.distogram_protocol_receipt
    )
    if binary_protocol.get("development_matrix") != [
        "framecontact_unary_only",
        "framecontact_contact_marginal",
        "framecontact_contact_hybrid",
    ]:
        raise ValueError("Binary reference protocol matrix differs")
    if distogram_protocol.get("development_matrix") != list(CONDITIONS[1:]):
        raise ValueError("Distogram protocol matrix differs")
    if distogram_protocol.get("evidence_boundary_at_freeze", {}).get(
        "current_sabdab2_test_is_confirmatory"
    ) is not False:
        raise ValueError("Distogram protocol misstates the test boundary")

    results: dict[str, dict[int, dict]] = {condition: {} for condition in CONDITIONS}
    metrics: dict[str, dict[int, dict[str, float]]] = {
        condition: {} for condition in CONDITIONS
    }
    shared_authority: dict[str, Any] | None = None
    for condition in CONDITIONS:
        run_root = (
            arguments.binary_run_root
            if condition == "framecontact_contact_hybrid"
            else arguments.distogram_run_root
        )
        protocol_sha256 = (
            binary_protocol_sha256
            if condition == "framecontact_contact_hybrid"
            else distogram_protocol_sha256
        )
        for seed in SEEDS:
            result = _load_run(run_root, condition, seed, protocol_sha256)
            authority = result["authority"]
            comparable = {
                key: authority.get(key)
                for key in (
                    "dataset_backend",
                    "structure_manifest_sha256",
                    "contact_manifest_sha256",
                    "embedding_manifest_sha256",
                    "training_partner_panel_sha256",
                    "validation_partner_panel_sha256",
                    "split_manifest_sha256",
                )
            }
            if shared_authority is None:
                shared_authority = comparable
            elif comparable != shared_authority:
                raise ValueError("Comparable data authorities differ across runs")
            expected_pair_geometry = (
                None
                if condition == "framecontact_contact_hybrid"
                else distogram_protocol["data"]["pair_geometry_manifest_sha256"]
            )
            if authority.get("pair_geometry_manifest_sha256") != expected_pair_geometry:
                raise ValueError("Pair geometry authority differs from the protocol")
            results[condition][seed] = result
            metrics[condition][seed] = _metrics(result)

    summaries = {
        condition: {
            "per_seed": {str(seed): metrics[condition][seed] for seed in SEEDS},
            "five_seed": {
                metric: _mean_sd([metrics[condition][seed][metric] for seed in SEEDS])
                for metric in metrics[condition][SEEDS[0]]
            },
            "selected_epoch_by_seed": {
                str(seed): int(results[condition][seed]["selection"]["best_epoch"])
                for seed in SEEDS
            },
            "epoch_trajectory": _epoch_trajectory(results[condition]),
        }
        for condition in CONDITIONS
    }
    distogram_vs_binary = _paired_delta(
        metrics["framecontact_distogram"],
        metrics["framecontact_contact_hybrid"],
    )
    triangle_vs_distogram = _paired_delta(
        metrics["framecontact_distogram_triangle"],
        metrics["framecontact_distogram"],
    )
    recall_ratios = [
        metrics["framecontact_distogram_triangle"][seed][
            "contact_fixed_raw_recall"
        ]
        / max(
            metrics["framecontact_distogram"][seed]["contact_fixed_raw_recall"],
            1e-12,
        )
        for seed in SEEDS
    ]
    if len(recall_ratios) != len(SEEDS):
        raise ValueError("Recall-retention ratio count differs from the seed registry")
    triangle_gate = {
        "mean_contact_mean_per_complex_auprc_delta_above_zero": (
            triangle_vs_distogram["five_seed"][
                "contact_mean_per_complex_auprc"
            ]["mean"]
            > 0.0
        ),
        "mean_contact_p_at_lmin_over_5_delta_above_zero": (
            triangle_vs_distogram["five_seed"]["contact_p_at_lmin_over_5"][
                "mean"
            ]
            > 0.0
        ),
        "mean_fixed_raw_precision_delta_above_zero": (
            triangle_vs_distogram["five_seed"]["contact_fixed_raw_precision"][
                "mean"
            ]
            > 0.0
        ),
        "mean_fixed_raw_recall_retention_ratio_at_least_0_95": (
            statistics.fmean(recall_ratios) >= 0.95
        ),
        "mean_selection_margin_delta_at_least_minus_0_01": (
            triangle_vs_distogram["five_seed"]["selection_margin"]["mean"]
            >= -0.01
        ),
    }
    distogram_gate = {
        "mean_contact_mean_per_complex_auprc_delta_above_zero": (
            distogram_vs_binary["five_seed"][
                "contact_mean_per_complex_auprc"
            ]["mean"]
            > 0.0
        ),
        "mean_contact_pooled_brier_delta_at_most_zero": (
            distogram_vs_binary["five_seed"]["contact_pooled_brier"]["mean"]
            <= 0.0
        ),
        "mean_selection_margin_delta_at_least_minus_0_01": (
            distogram_vs_binary["five_seed"]["selection_margin"]["mean"]
            >= -0.01
        ),
    }
    payload = {
        "format_version": SUMMARY_FORMAT,
        "claim_boundary": (
            "Five-seed train/validation aggregation only. The current SAbDab2 test is "
            "not confirmatory; no external, benchmark-improvement, or SOTA claim."
        ),
        "matrix_complete": True,
        "seeds": list(SEEDS),
        "authority": {
            "shared_data": shared_authority,
            "binary_protocol_sha256": binary_protocol_sha256,
            "distogram_protocol_sha256": distogram_protocol_sha256,
            "pair_geometry_manifest_sha256": distogram_protocol["data"][
                "pair_geometry_manifest_sha256"
            ],
        },
        "conditions": summaries,
        "paired_delta": {
            "distogram_vs_binary_hybrid": distogram_vs_binary,
            "triangle_vs_distogram": triangle_vs_distogram,
        },
        "recall_retention_ratio_triangle_vs_distogram": {
            "per_seed": {
                str(seed): value for seed, value in zip(SEEDS, recall_ratios)
            },
            "five_seed": _mean_sd(recall_ratios),
        },
        "promotion_gates": {
            "distogram_vs_binary_hybrid": {
                **distogram_gate,
                "passed": all(distogram_gate.values()),
            },
            "triangle_vs_distogram": {
                **triangle_gate,
                "passed": all(triangle_gate.values()),
            },
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, arguments.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
                "runs": len(CONDITIONS) * len(SEEDS),
                "promotion_gates": payload["promotion_gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
