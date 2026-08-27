#!/usr/bin/env python3
"""Evaluate deterministic epitope readouts from one frozen pair-logit checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


EXPECTED_PROTOCOL_SHA256 = (
    "b361bcf437fc73e9611b1564cdec06fbc1f4da633066753cef6276755b3eb928"
)


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, name: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{name} SHA differs: {observed} != {expected}")


def readout_logits(
    contact_logits: torch.Tensor,
    pair_mask: torch.Tensor,
    framecontact: Any,
) -> dict[str, torch.Tensor]:
    return {
        "topk1_max": framecontact.masked_topk_mean_probability_logit(
            contact_logits, pair_mask, dim=1, k=1
        ),
        "topk2": framecontact.masked_topk_mean_probability_logit(
            contact_logits, pair_mask, dim=1, k=2
        ),
        "topk4": framecontact.masked_topk_mean_probability_logit(
            contact_logits, pair_mask, dim=1, k=4
        ),
        "topk8": framecontact.masked_topk_mean_probability_logit(
            contact_logits, pair_mask, dim=1, k=8
        ),
        "noisy_or": framecontact.masked_noisy_or_probability_logit(
            contact_logits, pair_mask, dim=1
        ),
        "logmeanexp": framecontact.masked_logmeanexp(
            contact_logits, pair_mask, dim=1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("gpu_4090 is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    require_sha(args.protocol, EXPECTED_PROTOCOL_SHA256, "protocol")
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("format_version")
        != "sabdab2_fixed_checkpoint_epitope_readout_protocol_v1"
        or protocol.get("status") != "FROZEN_BEFORE_MEASUREMENT"
        or protocol.get("development_only") is not True
        or protocol.get("model_selection_performed") is not False
        or protocol.get("test_loaded") is not False
        or protocol.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("protocol claim boundary differs")

    paths = {key: Path(value) for key, value in protocol["paths"].items()}
    for name, expected in protocol["sha256"].items():
        require_sha(paths[name], expected, name)

    repository_root = paths["repository_root"].resolve()
    sys.path.insert(0, str(repository_root / "src"))
    sys.path.insert(0, str(repository_root / "scripts"))
    import train_framecontact as training  # noqa: PLC0415
    from interaction_first import framecontact  # noqa: PLC0415
    from interaction_first.sabdab2_contacts import Sabdab2ContactDataset  # noqa: PLC0415
    from interaction_first.sabdab2_structure import (  # noqa: PLC0415
        Sabdab2StructureEmbeddingDataset,
    )

    training.verify_source_manifest(
        repository_root,
        paths["source_manifest"],
        protocol["sha256"]["source_manifest"],
    )

    selected_result = json.loads(paths["selected_result"].read_text())
    configuration = dict(selected_result["configuration"])
    if (
        selected_result.get("experiment") != "epiformer_style_same_protocol"
        or configuration.get("seed") != 17
        or configuration.get("split_manifest_sha256")
        != protocol["sha256"]["split_manifest"]
        or selected_result.get("authority", {}).get("test_loaded") is not False
    ):
        raise ValueError("selected checkpoint result authority differs")
    configuration["epitope_pooling"] = "topk_mean"
    configuration["epitope_pooling_k"] = 2

    training._set_seed(17)
    base = Sabdab2StructureEmbeddingDataset(
        paths["structure_cache_root"],
        paths["embedding_root"],
        None,
        split_role="validation",
        split_path=paths["split_manifest"],
        split_sha256=protocol["sha256"]["split_manifest"],
    )
    dataset = Sabdab2ContactDataset(base, paths["contact_cache_root"])
    if len(dataset) != 549:
        raise ValueError(f"validation cohort differs: {len(dataset)} != 549")
    loader_arguments = {
        "batch_size": int(configuration["evaluation_batch_size"]),
        "workers": args.workers,
        "shuffle": False,
        "seed": 17,
        "pin_memory": True,
    }
    if "persistent_workers" in inspect.signature(training._loader).parameters:
        loader_arguments["persistent_workers"] = args.workers > 0
    loader = training._loader(dataset, **loader_arguments)

    model = training._model_from_configuration(
        dataset[0], configuration, selected_result["experiment"]
    )
    device = torch.device("cuda")
    model.to(device)
    checkpoint = torch.load(paths["selected_checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    state_before = training._state_sha256(model)

    from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: PLC0415

    arms = tuple(protocol["readouts"])
    pooled_labels: list[np.ndarray] = []
    pooled_probabilities: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
    per_complex: dict[str, dict[str, float]] = {}
    max_topk2_logit_error = 0.0
    with torch.no_grad():
        for raw_batch in loader:
            batch = training._move_batch(raw_batch, device)
            with training._autocast(device, configuration["precision"] == "bf16"):
                output = model(batch, return_attention=False)
            logits_by_arm = readout_logits(
                output["contact_logits"], output["pair_candidate_mask"], framecontact
            )
            max_topk2_logit_error = max(
                max_topk2_logit_error,
                float((logits_by_arm["topk2"] - output["epitope_logits"]).abs().max()),
            )
            for batch_index, instance_id in enumerate(raw_batch["instance_ids"]):
                mask = output["antigen_mask"][batch_index]
                labels = (
                    batch["antigen_labels"][batch_index][mask]
                    .cpu()
                    .numpy()
                    .astype(np.int8, copy=False)
                )
                pooled_labels.append(labels)
                local = {}
                for arm in arms:
                    probabilities = (
                        torch.sigmoid(logits_by_arm[arm][batch_index][mask].float())
                        .cpu()
                        .numpy()
                    )
                    pooled_probabilities[arm].append(probabilities)
                    local[arm] = float(average_precision_score(labels, probabilities))
                identifier = str(instance_id)
                if identifier in per_complex:
                    raise ValueError(f"duplicate validation instance: {identifier}")
                per_complex[identifier] = local

    if max_topk2_logit_error > 1e-6:
        raise ValueError(f"topk2 contract differs: {max_topk2_logit_error}")
    state_after = training._state_sha256(model)
    if state_after != state_before:
        raise RuntimeError("model parameters changed during readout evaluation")

    all_labels = np.concatenate(pooled_labels)
    metrics = {}
    for arm in arms:
        all_probabilities = np.concatenate(pooled_probabilities[arm])
        values = np.asarray([row[arm] for row in per_complex.values()], dtype=np.float64)
        metrics[arm] = {
            "pooled_average_precision": float(
                average_precision_score(all_labels, all_probabilities)
            ),
            "pooled_auroc": float(roc_auc_score(all_labels, all_probabilities)),
            "mean_per_complex_average_precision": float(values.mean()),
            "median_per_complex_average_precision": float(np.median(values)),
            "complex_count": int(values.size),
        }

    payload = {
        "format_version": "sabdab2_fixed_checkpoint_epitope_readout_result_v1",
        "status": "PASS_FIXED_CHECKPOINT_DETERMINISTIC_READOUT_DIAGNOSTIC",
        "metrics": metrics,
        "per_complex_average_precision": per_complex,
        "audit": {
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "selected_checkpoint_sha256": protocol["sha256"]["selected_checkpoint"],
            "selected_epoch": int(selected_result["selection"]["best_epoch"]),
            "seed": 17,
            "validation_complexes": len(dataset),
            "max_topk2_reference_logit_absolute_error": max_topk2_logit_error,
            "model_state_sha256": state_after,
            "model_parameters_updated": False,
            "model_selection_performed": False,
            "learned_pooling_evaluated": False,
            "full_validation_is_checkpoint_selection_data": True,
            "development_only": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    result_sha256 = sha256_file(args.output)
    receipt = {
        "format_version": "sabdab2_fixed_checkpoint_epitope_readout_receipt_v1",
        "status": payload["status"],
        "result_sha256": result_sha256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "model_parameters_updated": False,
        "model_selection_performed": False,
        "development_only": True,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.receipt.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "metrics": metrics,
                "result_sha256": result_sha256,
                "receipt_sha256": sha256_file(args.receipt),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
