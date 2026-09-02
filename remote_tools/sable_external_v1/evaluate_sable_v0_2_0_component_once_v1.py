#!/usr/bin/env python3
"""Evaluate one frozen no-distance checkpoint on once-opened SABLE labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


EXPECTED_EXTERNAL_PLAN_SHA256 = (
    "d97f0b94863fe9227d80c5675e944091809245b8fab22a89cda81148a8de172a"
)
ARMS = ("baseline", "edge_topk2_supervised")
SEEDS = (17, 29, 43)


class EvaluationError(ValueError):
    """Raised when an external-evaluation authority differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON root is not an object: {path}")
    return value


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise EvaluationError(f"{label} SHA256 differs: {observed}")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def load_embedding(
    path: Path, expected_sha256: str, expected_length: int
) -> torch.Tensor:
    if not path.is_file() or path.is_symlink():
        raise EvaluationError(f"missing ESM embedding: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != "sppider_seq_embedding_v1"
        or payload.get("sequence_sha256") != expected_sha256
        or int(payload.get("sequence_length", -1)) != expected_length
    ):
        raise EvaluationError("ESM embedding metadata differs")
    chunks = payload.get("chunks")
    starts = payload.get("starts")
    ends = payload.get("ends")
    if not isinstance(chunks, list) or not chunks or not (
        isinstance(starts, list)
        and isinstance(ends, list)
        and len(chunks) == len(starts) == len(ends)
    ):
        raise EvaluationError("ESM embedding chunks are malformed")
    dimension = int(chunks[0].shape[-1])
    output = torch.zeros(expected_length, dimension, dtype=torch.float32)
    counts = torch.zeros(expected_length, 1, dtype=torch.float32)
    for chunk, start, end in zip(chunks, starts, ends):
        start, end = int(start), int(end)
        if (
            not torch.is_tensor(chunk)
            or tuple(chunk.shape) != (end - start, dimension)
            or start < 0
            or end > expected_length
            or start >= end
        ):
            raise EvaluationError("ESM embedding chunk bounds differ")
        output[start:end] += chunk.to(torch.float32)
        counts[start:end] += 1.0
    if bool((counts == 0).any()):
        raise EvaluationError("ESM embedding does not cover the residue axis")
    return output / counts


class SableExternalDataset(Dataset):
    """Join predicted geometry, ESM features, and masked bound labels."""

    def __init__(
        self,
        structure_root: Path,
        embedding_root: Path,
        label_root: Path,
        structure_manifest: dict[str, Any],
        embedding_manifest: dict[str, Any],
        label_manifest: dict[str, Any],
    ) -> None:
        self.structure_root = structure_root
        self.embedding_root = embedding_root
        self.label_root = label_root
        structure_by_id = {
            str(row["entry_id"]): row for row in structure_manifest.get("records", [])
        }
        label_by_id = {
            str(row["entry_id"]): row for row in label_manifest.get("records", [])
        }
        embedding_by_sha = {
            str(row["sequence_sha256"]): row
            for row in embedding_manifest.get("records", [])
        }
        if set(label_by_id) - set(structure_by_id):
            raise EvaluationError("label records lack predicted structures")
        self.records = tuple(label_by_id[key] for key in sorted(label_by_id))
        self.structure_by_id = structure_by_id
        self.embedding_by_sha = embedding_by_sha
        self.evaluation: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        label_record = self.records[index]
        entry_id = str(label_record["entry_id"])
        structure_record = self.structure_by_id[entry_id]
        if (
            structure_record["antibody_sequence_sha256"]
            != label_record["antibody_sequence_sha256"]
            or structure_record["antigen_sequence_sha256"]
            != label_record["antigen_sequence_sha256"]
            or int(structure_record["antibody_length"])
            != int(label_record["antibody_length"])
            or int(structure_record["antigen_length"])
            != int(label_record["antigen_length"])
        ):
            raise EvaluationError(f"structure/label residue axes differ: {entry_id}")
        structure_path = self.structure_root / structure_record["artifact_path"]
        require_sha(structure_path, structure_record["artifact_sha256"], "structure artifact")
        structure = torch.load(structure_path, map_location="cpu", weights_only=True)
        if (
            structure.get("format_version")
            != "sable_v0_2_0_sequence_predicted_structure_v1"
            or structure.get("entry_id") != entry_id
            or structure.get("geometry_source") != "independent_sequence_predicted_af3"
        ):
            raise EvaluationError(f"predicted structure artifact differs: {entry_id}")

        antibody_sha = str(label_record["antibody_sequence_sha256"])
        antigen_sha = str(label_record["antigen_sequence_sha256"])
        antibody_embedding_record = self.embedding_by_sha.get(antibody_sha)
        antigen_embedding_record = self.embedding_by_sha.get(antigen_sha)
        if antibody_embedding_record is None or antigen_embedding_record is None:
            raise EvaluationError(f"ESM embedding record is missing: {entry_id}")
        antibody_embedding_path = (
            self.embedding_root / antibody_embedding_record["artifact_path"]
        )
        antigen_embedding_path = self.embedding_root / antigen_embedding_record["artifact_path"]
        require_sha(
            antibody_embedding_path,
            antibody_embedding_record["artifact_sha256"],
            "antibody ESM artifact",
        )
        require_sha(
            antigen_embedding_path,
            antigen_embedding_record["artifact_sha256"],
            "antigen ESM artifact",
        )
        antibody_length = int(label_record["antibody_length"])
        antigen_length = int(label_record["antigen_length"])
        antibody_features = load_embedding(
            antibody_embedding_path, antibody_sha, antibody_length
        )
        antigen_features = load_embedding(antigen_embedding_path, antigen_sha, antigen_length)

        label_path = self.label_root / label_record["artifact_path"]
        require_sha(label_path, label_record["artifact_sha256"], "bound-label artifact")
        with np.load(label_path, allow_pickle=False) as labels:
            required = {
                "format_version",
                "entry_id",
                "antibody_length",
                "antigen_length",
                "antibody_observed_label_mask",
                "antigen_observed_label_mask",
                "positive_flat_indices",
            }
            if set(labels.files) != required:
                raise EvaluationError(f"bound-label artifact schema differs: {entry_id}")
            if (
                str(labels["format_version"].item())
                != "sable_v0_2_0_bound_contact_labels_v1"
                or str(labels["entry_id"].item()) != entry_id
                or int(labels["antibody_length"].item()) != antibody_length
                or int(labels["antigen_length"].item()) != antigen_length
            ):
                raise EvaluationError(f"bound-label artifact metadata differs: {entry_id}")
            antibody_observed = labels["antibody_observed_label_mask"].astype(
                np.bool_, copy=True
            )
            antigen_observed = labels["antigen_observed_label_mask"].astype(
                np.bool_, copy=True
            )
            positive_flat = labels["positive_flat_indices"].astype(np.int64, copy=True)
        if (
            antibody_observed.shape != (antibody_length,)
            or antigen_observed.shape != (antigen_length,)
            or not bool(antibody_observed.any())
            or not bool(antigen_observed.any())
        ):
            raise EvaluationError(f"bound-label masks differ: {entry_id}")
        pair_labels = torch.zeros(antibody_length * antigen_length, dtype=torch.float32)
        if len(positive_flat):
            if (
                positive_flat[0] < 0
                or positive_flat[-1] >= pair_labels.numel()
                or len(np.unique(positive_flat)) != len(positive_flat)
            ):
                raise EvaluationError(f"positive contact indices differ: {entry_id}")
            pair_labels[torch.from_numpy(positive_flat)] = 1.0
        pair_labels = pair_labels.view(antibody_length, antigen_length)
        observed_pair = np.outer(antibody_observed, antigen_observed)
        if bool(np.any(pair_labels.numpy().astype(bool) & ~observed_pair)):
            raise EvaluationError(f"positive contact is outside observed labels: {entry_id}")
        self.evaluation[entry_id] = {
            "pair_labels": pair_labels.numpy().astype(np.int8, copy=True),
            "antibody_labels": pair_labels.bool().any(dim=1).numpy().astype(np.int8),
            "antigen_labels": pair_labels.bool().any(dim=0).numpy().astype(np.int8),
            "antibody_observed": antibody_observed,
            "antigen_observed": antigen_observed,
        }
        return {
            "instance_id": entry_id,
            "antibody_features": antibody_features,
            "antigen_features": antigen_features,
            "antibody_backbone": structure["antibody_backbone"],
            "antigen_backbone": structure["antigen_backbone"],
            "antibody_chain_ids": structure["antibody_chain_ids"],
            "antigen_chain_ids": structure["antigen_chain_ids"],
            "antibody_frame_valid_mask": structure["antibody_frame_valid_mask"],
            "antigen_frame_valid_mask": structure["antigen_frame_valid_mask"],
            "antibody_missing_frame_mask": structure["antibody_missing_frame_mask"],
            "antigen_missing_frame_mask": structure["antigen_missing_frame_mask"],
            "antibody_labels": pair_labels.bool().any(dim=1).to(torch.float32),
            "antigen_labels": pair_labels.bool().any(dim=0).to(torch.float32),
            "pair_labels": pair_labels,
            "pair_candidate_mask": torch.ones(
                antibody_length, antigen_length, dtype=torch.bool
            ),
        }


def calibration(selected: dict[str, Any]) -> dict[str, float]:
    return {
        "contact_temperature": float(selected["validation_temperature"]),
        "contact_threshold": float(selected["contact"]["threshold_selected_on_validation"]),
        "epitope_temperature": float(selected["epitope"]["validation_temperature"]),
        "epitope_threshold": float(selected["epitope"]["threshold_selected_on_validation"]),
        "paratope_temperature": float(selected["paratope"]["validation_temperature"]),
        "paratope_threshold": float(selected["paratope"]["threshold_selected_on_validation"]),
    }


def summarize(
    labels_by_entry: dict[str, np.ndarray],
    logits_by_entry: dict[str, np.ndarray],
    *,
    temperature: float,
    threshold: float,
) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if set(labels_by_entry) != set(logits_by_entry):
        raise EvaluationError("metric entry IDs differ")
    probabilities_by_entry = {
        entry_id: sigmoid(logits_by_entry[entry_id] / temperature)
        for entry_id in sorted(labels_by_entry)
    }
    per_entry_ap = {
        entry_id: float(average_precision_score(labels_by_entry[entry_id], probabilities))
        for entry_id, probabilities in probabilities_by_entry.items()
        if bool(labels_by_entry[entry_id].any())
    }
    labels = np.concatenate([labels_by_entry[key] for key in sorted(labels_by_entry)])
    probabilities = np.concatenate(
        [probabilities_by_entry[key] for key in sorted(probabilities_by_entry)]
    )
    predictions = (probabilities >= threshold).astype(np.int8)
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "auroc": (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) == 2
            else None
        ),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "threshold_frozen_on_development": threshold,
        "temperature_frozen_on_development": temperature,
        "mean_per_eligible_entry_average_precision": float(np.mean(list(per_entry_ap.values()))),
        "eligible_entry_count": len(per_entry_ap),
        "total_entry_count": len(labels_by_entry),
        "observed_label_count": int(len(labels)),
        "positive_label_count": int(labels.sum()),
        "per_entry_average_precision": per_entry_ap,
        "zero_positive_entry_ids": sorted(set(labels_by_entry) - set(per_entry_ap)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-plan", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-binding", type=Path, required=True)
    parser.add_argument("--checkpoint-binding-receipt", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
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
    require_sha(args.external_plan, EXPECTED_EXTERNAL_PLAN_SHA256, "external plan")
    external_plan = load_json(args.external_plan)
    authorization = load_json(args.authorization)
    authorization_sha256 = sha256_file(args.authorization)
    authorization_receipt = load_json(args.authorization_receipt)
    binding = load_json(args.checkpoint_binding)
    binding_sha256 = sha256_file(args.checkpoint_binding)
    binding_receipt = load_json(args.checkpoint_binding_receipt)
    if (
        external_plan.get("status") != "FROZEN_POST_GATE_BEFORE_BOUND_LABEL_OPENING"
        or authorization.get("status") != "AUTHORIZED_ONE_TIME_BOUND_LABEL_OPENING"
        or authorization.get("external_execution_plan_sha256")
        != EXPECTED_EXTERNAL_PLAN_SHA256
        or authorization.get("one_time_evaluation_authorized") is not True
        or authorization.get("selection_or_tuning") is not False
        or authorization_receipt.get("authorization_sha256") != authorization_sha256
        or binding.get("status") != "PASS_EXACT_NO_DISTANCE_CHECKPOINT_BINDING"
        or binding.get("distance_labels_loaded") is not False
        or binding.get("experimental_structure_required_at_inference") is not False
        or binding.get("selection_or_tuning") is not False
        or binding_receipt.get("binding_sha256") != binding_sha256
    ):
        raise EvaluationError("external evaluation authority differs")

    structure_manifest_path = args.structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    embedding_manifest_path = args.embedding_root / "SABLE_ESM_EMBEDDING_MANIFEST.json"
    label_manifest_path = args.label_root / "SABLE_BOUND_LABEL_MANIFEST.json"
    label_receipt_path = args.label_root / "SABLE_BOUND_LABEL_RECEIPT.json"
    structure_manifest = load_json(structure_manifest_path)
    embedding_manifest = load_json(embedding_manifest_path)
    label_manifest = load_json(label_manifest_path)
    label_receipt = load_json(label_receipt_path)
    if (
        sha256_file(structure_manifest_path)
        != authorization.get("predicted_structure_manifest_sha256")
        or sha256_file(embedding_manifest_path)
        != authorization.get("esm_embedding_manifest_sha256")
        or label_manifest.get("status")
        != "PASS_SINGLE_BOUND_LABEL_OPENING_AND_MAPPING_GATE"
        or label_manifest.get("authority", {}).get("authorization_sha256")
        != authorization_sha256
        or label_manifest.get("authority", {}).get("opening_count") != 1
        or label_manifest.get("authority", {}).get(
            "experimental_bound_coordinates_used_as_model_input"
        )
        is not False
        or label_manifest.get("authority", {}).get("selection_or_tuning") is not False
        or label_manifest.get("mapping_gate", {}).get("passed") is not True
        or label_receipt.get("manifest_sha256") != sha256_file(label_manifest_path)
        or label_receipt.get("mapping_gate_passed") is not True
        or label_receipt.get("performance_read") is not False
    ):
        raise EvaluationError("external data authority differs")

    run_binding = binding["runs"][args.arm][str(args.seed)]
    result_path = Path(run_binding["result_path"])
    checkpoint_path = Path(run_binding["checkpoint_path"])
    run_receipt_path = Path(run_binding["receipt_path"])
    require_sha(result_path, run_binding["result_sha256"], "development result")
    require_sha(checkpoint_path, run_binding["checkpoint_sha256"], "development checkpoint")
    require_sha(run_receipt_path, run_binding["receipt_sha256"], "development receipt")
    development = load_json(result_path)
    if (
        development.get("configuration", {}).get("seed") != args.seed
        or float(development.get("configuration", {}).get("distance_loss_weight", -1.0))
        != 0.0
        or development.get("authority", {}).get("distance_label_manifest_sha256")
        is not None
        or development.get("authority", {}).get("test_loaded") is not False
    ):
        raise EvaluationError("development result authority differs")
    source_manifest_path = Path(development["configuration"]["source_manifest"])
    require_sha(
        source_manifest_path,
        development["configuration"]["source_manifest_sha256"],
        "source manifest",
    )
    source_manifest = load_json(source_manifest_path)
    source_root = Path(source_manifest["source_root"])
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root / "scripts"))
    import train_framecontact as training  # type: ignore

    dataset = SableExternalDataset(
        args.structure_root,
        args.embedding_root,
        args.label_root,
        structure_manifest,
        embedding_manifest,
        label_manifest,
    )
    if len(dataset) != int(label_manifest["counts"]["analyzable_entries"]):
        raise EvaluationError("external dataset cardinality differs")
    first = dataset[0]
    loader = training._loader(
        dataset,
        batch_size=1,
        workers=0,
        shuffle=False,
        seed=args.seed,
        pin_memory=True,
    )
    device = torch.device("cuda")
    training._set_seed(args.seed)
    model = training._model_from_configuration(
        first, development["configuration"], "epiformer_style_same_protocol"
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    state_before = training._state_sha256(model)
    labels_by_task: dict[str, dict[str, np.ndarray]] = {
        "contact": {},
        "epitope": {},
        "paratope": {},
    }
    logits_by_task: dict[str, dict[str, np.ndarray]] = {
        "contact": {},
        "epitope": {},
        "paratope": {},
    }
    with torch.no_grad():
        for raw_batch in loader:
            batch = training._move_batch(raw_batch, device)
            with training._autocast(device, True):
                output = model(batch, return_attention=False)
            for batch_index, raw_entry_id in enumerate(raw_batch["instance_ids"]):
                entry_id = str(raw_entry_id)
                evaluation = dataset.evaluation[entry_id]
                antibody_observed = evaluation["antibody_observed"]
                antigen_observed = evaluation["antigen_observed"]
                pair_observed = np.outer(antibody_observed, antigen_observed)
                pair_logits = (
                    output["contact_logits"][batch_index]
                    .float()
                    .cpu()
                    .numpy()[: len(antibody_observed), : len(antigen_observed)]
                )
                labels_by_task["contact"][entry_id] = evaluation["pair_labels"][
                    pair_observed
                ]
                logits_by_task["contact"][entry_id] = pair_logits[pair_observed]
                labels_by_task["paratope"][entry_id] = evaluation["antibody_labels"][
                    antibody_observed
                ]
                logits_by_task["paratope"][entry_id] = (
                    output["paratope_logits"][batch_index]
                    .float()
                    .cpu()
                    .numpy()[: len(antibody_observed)][antibody_observed]
                )
                labels_by_task["epitope"][entry_id] = evaluation["antigen_labels"][
                    antigen_observed
                ]
                logits_by_task["epitope"][entry_id] = (
                    output["epitope_logits"][batch_index]
                    .float()
                    .cpu()
                    .numpy()[: len(antigen_observed)][antigen_observed]
                )
    state_after = training._state_sha256(model)
    if state_after != state_before:
        raise RuntimeError("model parameters changed during SABLE measurement")
    frozen = calibration(development["selected_checkpoint_validation"])
    metrics = {
        task: summarize(
            labels_by_task[task],
            logits_by_task[task],
            temperature=frozen[f"{task}_temperature"],
            threshold=frozen[f"{task}_threshold"],
        )
        for task in ("contact", "epitope", "paratope")
    }
    payload = {
        "format_version": "sable_v0_2_0_framecontact_component_result_v1",
        "status": "PASS_SINGLE_OPENING_SABLE_COMPONENT_MEASUREMENT",
        "arm": args.arm,
        "seed": args.seed,
        "metrics": metrics,
        "audit": {
            "external_execution_plan_sha256": EXPECTED_EXTERNAL_PLAN_SHA256,
            "authorization_sha256": authorization_sha256,
            "checkpoint_binding_sha256": binding_sha256,
            "checkpoint_sha256": run_binding["checkpoint_sha256"],
            "development_result_sha256": run_binding["result_sha256"],
            "predicted_structure_manifest_sha256": sha256_file(structure_manifest_path),
            "esm_embedding_manifest_sha256": sha256_file(embedding_manifest_path),
            "bound_label_manifest_sha256": sha256_file(label_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "model_state_sha256": state_after,
            "entry_count": len(dataset),
            "opening_count": 1,
            "experimental_bound_coordinates_used_as_model_input": False,
            "cross_partner_geometry_used_as_model_input": False,
            "distance_labels_loaded": False,
            "calibration_refit": False,
            "model_parameters_updated": False,
            "selection_performed": False,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "sable_v0_2_0_framecontact_component_receipt_v1",
        "status": payload["status"],
        "arm": args.arm,
        "seed": args.seed,
        "result_sha256": sha256_file(args.output),
        "checkpoint_sha256": run_binding["checkpoint_sha256"],
        "opening_count": 1,
        "selection_performed": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.receipt_output.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {**receipt, "receipt_sha256": sha256_file(args.receipt_output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
