#!/usr/bin/env python3
"""Evaluate frozen ABF-30B contact components on development validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score


SCORE_NAMES = (
    "full_contact",
    "unary_only",
    "residue_head_outer_product",
    "raw_interaction",
    "gauge_fixed_residual",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def require_sha(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority SHA differs: {path}")


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64, copy=False), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def rank_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    positive_count = int(labels.sum())
    if positive_count <= 0:
        raise ValueError("ranking metrics require a positive contact")
    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]
    first_positive = int(np.flatnonzero(ranked_labels)[0]) + 1

    def recall_at(k: int) -> float:
        effective = min(len(labels), max(1, k))
        return float(ranked_labels[:effective].sum() / positive_count)

    return {
        "reciprocal_rank_first_positive": 1.0 / first_positive,
        "hits_at_1": float(ranked_labels[0]),
        "recall_at_P": recall_at(positive_count),
        "recall_at_ceil_P_over_2": recall_at(math.ceil(positive_count / 2)),
        "recall_at_ceil_P_over_5": recall_at(math.ceil(positive_count / 5)),
    }


def gauge_decomposition(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    grand = float(values.mean())
    row = values.mean(axis=1) - grand
    column = values.mean(axis=0) - grand
    residual = values - grand - row[:, None] - column[None, :]
    reconstruction = grand + row[:, None] + column[None, :] + residual
    reconstruction_error = float(np.max(np.abs(reconstruction - values)))
    centering_error = max(
        float(np.max(np.abs(residual.mean(axis=1)))),
        float(np.max(np.abs(residual.mean(axis=0)))),
    )
    centered = values - grand
    antibody_ss = float(values.shape[1] * np.square(row).sum())
    antigen_ss = float(values.shape[0] * np.square(column).sum())
    interaction_ss = float(np.square(residual).sum())
    observed_ss = float(np.square(centered).sum())
    component_ss = antibody_ss + antigen_ss + interaction_ss
    partition_error = abs(component_ss - observed_ss) / max(observed_ss, 1e-30)
    if reconstruction_error > 2e-5 or centering_error > 2e-5 or partition_error > 2e-5:
        raise RuntimeError("gauge decomposition numerical invariant failed")
    denominator = max(component_ss, np.finfo(np.float64).tiny)
    return residual, {
        "partner_specific_variance_share": interaction_ss / denominator,
        "antibody_propensity_variance_share": antibody_ss / denominator,
        "antigen_propensity_variance_share": antigen_ss / denominator,
        "maximum_reconstruction_error": reconstruction_error,
        "maximum_centering_error": centering_error,
        "variance_partition_relative_error": partition_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("gpu_4090 is required")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("one RTX 4090 is required")
    for path in (args.output, args.receipt_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    protocol_sha = sha256_file(args.protocol)
    protocol = json.loads(args.protocol.read_text())
    receipt = json.loads(args.protocol_receipt.read_text())
    if (
        protocol.get("status") != "FROZEN_BEFORE_READ_ONLY_DEVELOPMENT_COMPONENT_AUDIT"
        or receipt.get("protocol_sha256") != protocol_sha
        or receipt.get("status") != protocol["status"]
        or protocol.get("seeds") != [17, 29, 43]
        or protocol.get("tools", {}).get("evaluator_sha256") != sha256_file(Path(__file__).resolve())
        or protocol.get("evidence_boundary", {}).get("development_only") is not True
        or protocol.get("evidence_boundary", {}).get("model_training") is not False
        or protocol.get("evidence_boundary", {}).get("old_sabdab2_test_loaded") is not False
        or protocol.get("evidence_boundary", {}).get("temporal_holdout_loaded") is not False
        or protocol.get("evidence_boundary", {}).get("sable_external_labels_loaded") is not False
    ):
        raise ValueError("frozen audit protocol differs")

    binding = protocol["run_bindings"][str(args.seed)]
    result_path = Path(binding["result_path"])
    checkpoint_path = Path(binding["checkpoint_path"])
    run_receipt_path = Path(binding["receipt_path"])
    for path, expected in (
        (result_path, binding["result_sha256"]),
        (checkpoint_path, binding["checkpoint_sha256"]),
        (run_receipt_path, binding["receipt_sha256"]),
    ):
        require_sha(path, expected)
    development = json.loads(result_path.read_text())
    configuration = development["configuration"]
    authority = development["authority"]
    if (
        development.get("experiment") != "epiformer_style_same_protocol"
        or configuration.get("seed") != args.seed
        or configuration.get("distance_loss_weight") != 0.0
        or configuration.get("specificity_loss_weight") != 0.0
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
        or authority.get("distance_label_manifest_sha256") is not None
    ):
        raise ValueError("development checkpoint authority differs")

    data = protocol["data"]
    structure_root = Path(data["structure_root"])
    contact_root = Path(data["contact_root"])
    embedding_root = Path(data["embedding_root"])
    split_path = Path(data["split_path"])
    for path, expected in (
        (structure_root / "STRUCTURE_CACHE_MANIFEST.json", data["structure_manifest_sha256"]),
        (contact_root / "CONTACT_CACHE_MANIFEST.json", data["contact_manifest_sha256"]),
        (embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json", data["embedding_manifest_sha256"]),
        (split_path, data["split_sha256"]),
    ):
        require_sha(path, expected)
    source = protocol["source"]
    source_manifest_path = Path(source["source_manifest_path"])
    require_sha(source_manifest_path, source["source_manifest_sha256"])
    source_manifest = json.loads(source_manifest_path.read_text())
    if (
        source_manifest.get("tree_sha256") != source["source_tree_sha256"]
        or Path(source_manifest["source_root"]) != Path(source["source_root"])
    ):
        raise ValueError("source tree authority differs")
    source_root = Path(source["source_root"])
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root / "scripts"))
    import train_framecontact as training  # type: ignore

    base = training.Sabdab2StructureEmbeddingDataset(
        structure_root,
        embedding_root,
        None,
        split_role="validation",
        split_path=split_path,
        split_sha256=data["split_sha256"],
    )
    dataset = training.Sabdab2ContactDataset(
        base,
        contact_root,
        distance_label_cache_root=None,
        distance_label_manifest_sha256=None,
    )
    if len(dataset) != data["instance_count"]:
        raise ValueError("validation dataset cardinality differs")
    with split_path.open(newline="") as handle:
        metadata = {
            row["instance_id"]: row
            for row in csv.DictReader(handle)
            if row["role"] == "validation"
        }
    if set(metadata) != {record["instance_id"] for record in dataset.records}:
        raise ValueError("validation split metadata differs from dataset")

    loader = training._loader(
        dataset,
        batch_size=1,
        workers=2,
        shuffle=False,
        seed=args.seed,
        pin_memory=True,
    )
    device = torch.device("cuda")
    training._set_seed(args.seed)
    model = training._model_from_configuration(
        dataset[0], configuration, "epiformer_style_same_protocol"
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    state_before = training._state_sha256(model)

    pooled_labels: list[np.ndarray] = []
    pooled_scores: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_NAMES}
    per_instance: dict[str, Any] = {}
    zero_positive = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = training._move_batch(raw_batch, device)
            with training._autocast(device, True):
                output = model(batch, return_attention=False)
            if len(raw_batch["instance_ids"]) != 1:
                raise RuntimeError("audit requires batch size one")
            instance_id = str(raw_batch["instance_ids"][0])
            antibody_mask = output["antibody_mask"][0]
            antigen_mask = output["antigen_mask"][0]
            pair_mask = output["pair_candidate_mask"][0]
            expected_mask = antibody_mask[:, None] & antigen_mask[None, :]
            if not torch.equal(pair_mask, expected_mask):
                raise RuntimeError("pair candidate mask is not a complete rectangular grid")
            antibody_length = int(antibody_mask.sum())
            antigen_length = int(antigen_mask.sum())
            labels_matrix = (
                batch["pair_labels"][0, :antibody_length, :antigen_length]
                .cpu()
                .numpy()
                .astype(np.int8, copy=False)
            )
            full = (
                output["contact_logits"][0, :antibody_length, :antigen_length]
                .float()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            interaction = (
                output["contact_interaction_logits"][0, :antibody_length, :antigen_length]
                .float()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            unary_only = full - interaction
            paratope = (
                output["paratope_logits"][0, :antibody_length].float().cpu().numpy()
            )
            epitope = output["epitope_logits"][0, :antigen_length].float().cpu().numpy()
            residue_outer = sigmoid(paratope)[:, None] * sigmoid(epitope)[None, :]
            residual, gauge = gauge_decomposition(full)
            matrices = {
                "full_contact": full,
                "unary_only": unary_only,
                "residue_head_outer_product": residue_outer,
                "raw_interaction": interaction,
                "gauge_fixed_residual": residual,
            }
            labels = labels_matrix.reshape(-1)
            positive_count = int(labels.sum())
            if positive_count == 0:
                zero_positive += 1
            metrics_by_score: dict[str, Any] = {}
            for name, matrix in matrices.items():
                scores = matrix.reshape(-1)
                if not np.isfinite(scores).all():
                    raise RuntimeError(f"non-finite score for {instance_id}: {name}")
                pooled_scores[name].append(scores.astype(np.float32, copy=False))
                if positive_count > 0:
                    metrics_by_score[name] = {
                        "average_precision": float(average_precision_score(labels, scores)),
                        **rank_metrics(labels, scores),
                    }
                else:
                    metrics_by_score[name] = None
            pooled_labels.append(labels)
            row = metadata[instance_id]
            per_instance[instance_id] = {
                "instance_id": instance_id,
                "antigen_cluster_id": row["antigen_cluster_id"],
                "pdb_assembly_id": row["pdb_assembly_id"],
                "antibody_length": antibody_length,
                "antigen_length": antigen_length,
                "pair_count": int(labels.size),
                "positive_pair_count": positive_count,
                "metrics_by_score": metrics_by_score,
                "gauge": gauge,
            }

    state_after = training._state_sha256(model)
    if state_before != state_after:
        raise RuntimeError("model parameters changed during read-only audit")
    all_labels = np.concatenate(pooled_labels)
    score_summary: dict[str, Any] = {}
    for name in SCORE_NAMES:
        all_scores = np.concatenate(pooled_scores[name])
        eligible = [
            row["metrics_by_score"][name]["average_precision"]
            for row in per_instance.values()
            if row["metrics_by_score"][name] is not None
        ]
        score_summary[name] = {
            "pooled_average_precision": float(average_precision_score(all_labels, all_scores)),
            "mean_per_complex_average_precision": float(np.mean(eligible)),
            "eligible_complex_count": len(eligible),
        }
    gauge_rows = [row["gauge"] for row in per_instance.values()]
    result = {
        "format_version": "abf30b_beyond_marginals_component_v1",
        "status": "PASS_READ_ONLY_DEVELOPMENT_COMPONENT_MEASUREMENT",
        "seed": args.seed,
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "selected_epoch": binding["selected_epoch"],
        "score_summary": score_summary,
        "gauge_summary": {
            "mean_partner_specific_variance_share": float(
                np.mean([row["partner_specific_variance_share"] for row in gauge_rows])
            ),
            "median_partner_specific_variance_share": float(
                np.median([row["partner_specific_variance_share"] for row in gauge_rows])
            ),
            "maximum_reconstruction_error": float(
                max(row["maximum_reconstruction_error"] for row in gauge_rows)
            ),
            "maximum_centering_error": float(
                max(row["maximum_centering_error"] for row in gauge_rows)
            ),
            "maximum_variance_partition_relative_error": float(
                max(row["variance_partition_relative_error"] for row in gauge_rows)
            ),
        },
        "counts": {
            "complexes": len(per_instance),
            "zero_positive_complexes": zero_positive,
            "valid_pairs": int(len(all_labels)),
            "positive_pairs": int(all_labels.sum()),
        },
        "per_instance": per_instance,
        "authority": {
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_training": False,
            "checkpoint_selection": False,
            "calibration_or_threshold_fitting": False,
            "raw_pair_predictions_persisted": False,
            "development_only": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
            "sable_external_labels_loaded": False,
            "distance_label_manifest_sha256": None,
            "experimental_bound_coordinates_used_as_model_input": False,
        },
    }
    result_sha = publish(args.output, result)
    receipt_sha = publish(
        args.receipt_output,
        {
            "format_version": "abf30b_beyond_marginals_component_receipt_v1",
            "status": result["status"],
            "seed": args.seed,
            "result_sha256": result_sha,
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "protocol_sha256": protocol_sha,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
            "sable_external_labels_loaded": False,
        },
    )
    print(json.dumps({"result_sha256": result_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
