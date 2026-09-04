#!/usr/bin/env python3
"""Run a frozen 20-case top-10 contact analysis on predicted geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


FORMAT = "sabdab2_top10_contact_case_study_result_v1"
PROTOCOL_FORMAT = "sabdab2_top10_contact_case_study_protocol_v1"


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 differs for {path}: {observed}")


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.lexsort((np.arange(scores.size), -scores))
    ranked = labels[order].astype(np.float64, copy=False)
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(precision[ranked == 1].mean())


def confusion_from_topk(labels: np.ndarray, order: np.ndarray, k: int) -> dict[str, int]:
    selected = np.zeros(labels.size, dtype=bool)
    selected[order[:k]] = True
    truth = labels.astype(bool, copy=False)
    return {
        "tn": int((~truth & ~selected).sum()),
        "fp": int((~truth & selected).sum()),
        "fn": int((truth & ~selected).sum()),
        "tp": int((truth & selected).sum()),
    }


def finite_tree(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError("Non-finite output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("gpu_4090 is required")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("One NVIDIA GeForce RTX 4090 is required")
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    require_sha(args.protocol, args.protocol_sha256)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("format_version") != PROTOCOL_FORMAT
        or protocol.get("status") != "FROZEN_BEFORE_TOP10_CONTACT_LABEL_READ"
        or protocol.get("selection", {}).get("selected_seed") != 43
        or protocol.get("claim_boundary", {}).get("case_study_only") is not True
        or protocol.get("claim_boundary", {}).get("fresh_blind_test") is not False
        or protocol.get("execution", {}).get("evaluator_sha256")
        != sha256_file(Path(__file__))
    ):
        raise ValueError("Frozen protocol differs")
    data = protocol["data"]
    for key in (
        "structure_manifest",
        "structure_receipt",
        "structure_binding",
        "contact_manifest",
        "contact_receipt",
        "embedding_manifest",
        "test_split",
    ):
        path = Path(data[key])
        require_sha(path, data[f"{key}_sha256"])
    if (args.structure_root / "STRUCTURE_CACHE_MANIFEST.json").resolve() != Path(
        data["structure_manifest"]
    ).resolve():
        raise ValueError("Structure root differs")
    if (args.contact_root / "CONTACT_CACHE_MANIFEST.json").resolve() != Path(
        data["contact_manifest"]
    ).resolve():
        raise ValueError("Contact root differs")
    if (args.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json").resolve() != Path(
        data["embedding_manifest"]
    ).resolve():
        raise ValueError("Embedding root differs")

    selected_root = Path(protocol["selection"]["selected_run_root"])
    result_path = selected_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = selected_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = selected_root / "RUN_RECEIPT.json"
    selected = next(
        row
        for row in protocol["selection"]["candidates"]
        if int(row["seed"]) == 43
    )
    for path, expected in (
        (result_path, selected["result_sha256"]),
        (checkpoint_path, selected["checkpoint_sha256"]),
        (receipt_path, selected["receipt_sha256"]),
    ):
        require_sha(path, expected)
    development = json.loads(result_path.read_text(encoding="utf-8"))
    configuration = development["configuration"]
    if (
        configuration.get("distance_loss_weight") != 0.0
        or configuration.get("distance_label_manifest_sha256") is not None
        or development.get("authority", {}).get("test_loaded") is not False
    ):
        raise ValueError("Selected development run is not no-privileged-distance")

    source_root = Path(protocol["execution"]["source_root"])
    require_sha(
        Path(protocol["execution"]["source_manifest"]),
        protocol["execution"]["source_manifest_sha256"],
    )
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root / "scripts"))
    import train_framecontact as training  # noqa: PLC0415
    from interaction_first.sabdab2_contacts import (  # noqa: PLC0415
        Sabdab2ContactBatchCollator,
        Sabdab2ContactDataset,
    )
    from interaction_first.sabdab2_structure import (  # noqa: PLC0415
        Sabdab2StructureEmbeddingDataset,
    )

    split = Path(data["test_split"])
    base = Sabdab2StructureEmbeddingDataset(
        args.structure_root,
        args.embedding_root,
        None,
        split_role="test",
        split_path=split,
        split_sha256=data["test_split_sha256"],
    )
    dataset = Sabdab2ContactDataset(base, args.contact_root)
    index_by_id = {
        str(row["instance_id"]): index for index, row in enumerate(dataset.records)
    }
    sample_ids = protocol["sampling"]["sampled_instance_ids"]
    if len(dataset) != 549 or any(identifier not in index_by_id for identifier in sample_ids):
        raise ValueError("Frozen sample and authenticated test dataset differ")

    training._set_seed(43)
    first = dataset[index_by_id[sample_ids[0]]]
    model = training._model_from_configuration(
        first, configuration, development["experiment"]
    )
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    state_before = training._state_sha256(model)
    collator = Sabdab2ContactBatchCollator()
    temperature = float(
        development["selected_checkpoint_validation"]["validation_temperature"]
    )
    top_k = int(protocol["sampling"]["top_k"])
    cases: list[dict[str, Any]] = []
    aggregate = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    with torch.inference_mode():
        for instance_id in sample_ids:
            raw = collator([dataset[index_by_id[instance_id]]])
            batch = training._move_batch(raw, device)
            with training._autocast(device, configuration["precision"] == "bf16"):
                output = model(batch, return_attention=False)
            ab_length = int(output["antibody_mask"][0].sum().item())
            ag_length = int(output["antigen_mask"][0].sum().item())
            pair_mask = output["pair_candidate_mask"][0, :ab_length, :ag_length]
            logits_grid = output["contact_logits"][0, :ab_length, :ag_length].float()
            labels_grid = batch["pair_labels"][0, :ab_length, :ag_length]
            coordinates = pair_mask.nonzero(as_tuple=False).cpu().numpy()
            logits = logits_grid[pair_mask].cpu().numpy().astype(np.float64, copy=False)
            labels = labels_grid[pair_mask].cpu().numpy().astype(np.int8, copy=False)
            flat_tiebreak = coordinates[:, 0] * ag_length + coordinates[:, 1]
            order = np.lexsort((flat_tiebreak, -logits))
            effective_k = min(top_k, labels.size)
            confusion = confusion_from_topk(labels, order, effective_k)
            for key in aggregate:
                aggregate[key] += confusion[key]
            scaled_logits = np.clip(logits / temperature, -80.0, 80.0)
            probabilities = 1.0 / (1.0 + np.exp(-scaled_logits))
            positive_ranks = np.flatnonzero(labels[order] == 1)
            top_pairs = []
            for rank, local_index in enumerate(order[:effective_k], start=1):
                ab_index, ag_index = map(int, coordinates[local_index])
                top_pairs.append(
                    {
                        "rank": rank,
                        "antibody_index_0based": ab_index,
                        "antibody_position_1based": ab_index + 1,
                        "antigen_index_0based": ag_index,
                        "antigen_position_1based": ag_index + 1,
                        "probability": float(probabilities[local_index]),
                        "label": int(labels[local_index]),
                    }
                )
            cases.append(
                {
                    "instance_id": instance_id,
                    "antibody_length": ab_length,
                    "antigen_length": ag_length,
                    "candidate_pair_count": int(labels.size),
                    "positive_pair_count": int(labels.sum()),
                    "top10_confusion": confusion,
                    "top10_precision": confusion["tp"] / effective_k,
                    "top10_recall": (
                        confusion["tp"] / int(labels.sum()) if int(labels.sum()) else 0.0
                    ),
                    "first_true_contact_rank": (
                        int(positive_ranks[0] + 1) if positive_ranks.size else None
                    ),
                    "average_precision": average_precision(labels, probabilities),
                    "top_pairs": top_pairs,
                }
            )
    state_after = training._state_sha256(model)
    if state_after != state_before:
        raise RuntimeError("Model parameters changed during inference")
    recommended = sorted(
        cases,
        key=lambda row: (
            -int(row["top10_confusion"]["tp"]),
            int(row["first_true_contact_rank"] or 10**12),
            -float(row["average_precision"] or 0.0),
            str(row["instance_id"]),
        ),
    )[:3]
    result = {
        "format_version": FORMAT,
        "status": "PASS_SABDAB2_TOP10_CONTACT_CASE_STUDY",
        "claim_boundary": protocol["claim_boundary"],
        "model": {
            "seed": 43,
            "selected_epoch": int(development["selection"]["best_epoch"]),
            "selection_rule": protocol["selection"]["rule"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "contact_temperature_frozen_on_development": temperature,
            "model_state_sha256": state_after,
        },
        "sampling": protocol["sampling"],
        "confusion_matrix_definition": {
            "rows": ["true_non_contact", "true_contact"],
            "columns": ["not_in_top10", "in_top10"],
            "layout": "[[TN, FP], [FN, TP]]",
        },
        "aggregate_top10_confusion": {
            **aggregate,
            "matrix": [
                [aggregate["tn"], aggregate["fp"]],
                [aggregate["fn"], aggregate["tp"]],
            ],
            "precision": aggregate["tp"] / (aggregate["tp"] + aggregate["fp"]),
            "recall": aggregate["tp"] / (aggregate["tp"] + aggregate["fn"]),
        },
        "cases": cases,
        "recommended_case_studies": [
            {
                "instance_id": row["instance_id"],
                "top10_true_contacts": row["top10_confusion"]["tp"],
                "first_true_contact_rank": row["first_true_contact_rank"],
                "average_precision": row["average_precision"],
            }
            for row in recommended
        ],
        "case_selection_rule": (
            "Within the pre-frozen random 20 only: descending top-10 TP, then first "
            "true-contact rank, then AP. This is descriptive label-aware case selection."
        ),
        "audit": {
            "protocol_sha256": args.protocol_sha256,
            "historical_test_previously_exposed": True,
            "fresh_blind_test": False,
            "model_or_checkpoint_selected_from_test": False,
            "calibration_or_threshold_refit": False,
            "model_parameters_updated": False,
            "experimental_bound_coordinates_as_model_input": False,
            "predicted_monomer_geometry_as_model_input": True,
            "experimental_contact_labels_used_for_case_study_measurement": True,
        },
    }
    finite_tree(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(result))
    receipt = {
        "format_version": FORMAT,
        "status": result["status"],
        "result_sha256": sha256_file(args.output),
        "protocol_sha256": args.protocol_sha256,
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "sampled_instance_ids_sha256": protocol["sampling"][
            "sampled_instance_ids_sha256"
        ],
        "model_or_checkpoint_selected_from_test": False,
        "model_parameters_updated": False,
    }
    args.receipt.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                "status": result["status"],
                "aggregate_top10_confusion": result["aggregate_top10_confusion"],
                "recommended_case_studies": result["recommended_case_studies"],
                "result_sha256": receipt["result_sha256"],
                "receipt_sha256": sha256_file(args.receipt),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
