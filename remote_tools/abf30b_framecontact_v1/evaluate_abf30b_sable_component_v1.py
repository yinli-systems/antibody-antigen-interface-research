#!/usr/bin/env python3
"""Measure one fixed ABF-30B FrameContact checkpoint on exposed SABLE labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

SEEDS = (17, 29, 43)
ARMS = ("baseline", "edge_topk2_supervised")
FINAL_MODEL_SHA256 = "c36e0b1d812f97317109f1f8ab085a349beb2a782a6f4092b3074bfc1eaab3ea"
SHARED_EVALUATOR_SHA256 = "4e203b58d9d54c2559aaf27c1d3684e8b49bec9f9c9e322e88bf615ec82f9677"


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root differs: {path}")
    return value


def require_sha(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority SHA differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-plan", type=Path, required=True)
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

    shared_path = (
        Path(__file__).resolve().parent.parent
        / "sable_external_v1/evaluate_sable_v0_2_0_component_once_v1.py"
    )
    require_sha(shared_path, SHARED_EVALUATOR_SHA256)
    spec = importlib.util.spec_from_file_location("abf30b_sable_shared_evaluator", shared_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned shared SABLE evaluator")
    shared = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = shared
    spec.loader.exec_module(shared)

    plan = load_json(args.measurement_plan)
    plan_sha = sha256_file(args.measurement_plan)
    authorization = load_json(args.authorization)
    authorization_sha = sha256_file(args.authorization)
    authorization_receipt = load_json(args.authorization_receipt)
    binding = load_json(args.checkpoint_binding)
    binding_sha = sha256_file(args.checkpoint_binding)
    binding_receipt = load_json(args.checkpoint_binding_receipt)
    if (
        plan.get("status")
        != "FROZEN_BEFORE_MODEL_SPECIFIC_ANALYST_EXPOSED_SABLE_MEASUREMENT"
        or plan.get("evidence_boundary", {}).get("fresh_blind_test") is not False
        or plan.get("evidence_boundary", {}).get("measurement_only") is not True
        or authorization.get("status")
        != "AUTHORIZED_MODEL_SPECIFIC_ANALYST_EXPOSED_MEASUREMENT_ONLY"
        or authorization.get("measurement_plan_sha256") != plan_sha
        or authorization.get("checkpoint_binding_sha256") != binding_sha
        or authorization.get("model_specific_measurement_authorized") is not True
        or authorization.get("fresh_blind_test") is not False
        or authorization.get("measurement_only") is not True
        or authorization.get("model_unseen_before_this_measurement") is not True
        or authorization.get("analyst_has_seen_prior_model_results_on_dataset") is not True
        or authorization.get("selection_or_tuning") is not False
        or authorization_receipt.get("authorization_sha256") != authorization_sha
        or binding.get("status") != "PASS_EXACT_ABF30B_DEVELOPMENT_CHECKPOINT_BINDING"
        or binding.get("abf30b_checkpoint_model_sha256") != FINAL_MODEL_SHA256
        or binding.get("sable_used_for_checkpoint_selection") is not False
        or binding.get("selection_or_tuning") is not False
        or binding_receipt.get("binding_sha256") != binding_sha
    ):
        raise ValueError("model-specific measurement authority differs")

    structure_manifest_path = args.structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    embedding_manifest_path = args.embedding_root / "SABLE_ESM_EMBEDDING_MANIFEST.json"
    label_manifest_path = args.label_root / "SABLE_BOUND_LABEL_MANIFEST.json"
    label_receipt_path = args.label_root / "SABLE_BOUND_LABEL_RECEIPT.json"
    for path, expected in (
        (structure_manifest_path, authorization["predicted_structure_manifest_sha256"]),
        (embedding_manifest_path, authorization["mixed_embedding_manifest_sha256"]),
        (label_manifest_path, authorization["bound_label_manifest_sha256"]),
        (label_receipt_path, authorization["bound_label_receipt_sha256"]),
    ):
        require_sha(path, expected)
    structure_manifest = load_json(structure_manifest_path)
    embedding_manifest = load_json(embedding_manifest_path)
    label_manifest = load_json(label_manifest_path)
    label_receipt = load_json(label_receipt_path)
    if (
        structure_manifest.get("status") != "PASS_LABEL_FREE_SEQUENCE_PREDICTED_STRUCTURE_CACHE"
        or structure_manifest.get("authority", {}).get("task_labels_read") is not False
        or embedding_manifest.get("status") != "PASS_LABEL_FREE_SABLE_MIXED_PARTNER_EMBEDDINGS"
        or embedding_manifest.get("antibody_model", {}).get("checkpoint_model_sha256")
        != FINAL_MODEL_SHA256
        or embedding_manifest.get("authority", {}).get("task_labels_read") is not False
        or label_manifest.get("status") != "PASS_SINGLE_BOUND_LABEL_OPENING_AND_MAPPING_GATE"
        or label_manifest.get("mapping_gate", {}).get("passed") is not True
        or label_manifest.get("authority", {}).get("opening_count") != 1
        or label_manifest.get("authority", {}).get(
            "experimental_bound_coordinates_used_as_model_input"
        )
        is not False
        or label_receipt.get("manifest_sha256") != sha256_file(label_manifest_path)
        or label_receipt.get("mapping_gate_passed") is not True
    ):
        raise ValueError("SABLE data authority differs")

    run_binding = binding["runs"][args.arm][str(args.seed)]
    result_path = Path(run_binding["result_path"])
    checkpoint_path = Path(run_binding["checkpoint_path"])
    run_receipt_path = Path(run_binding["receipt_path"])
    for path, expected in (
        (result_path, run_binding["result_sha256"]),
        (checkpoint_path, run_binding["checkpoint_sha256"]),
        (run_receipt_path, run_binding["receipt_sha256"]),
    ):
        require_sha(path, expected)
    development = load_json(result_path)
    if (
        development.get("configuration", {}).get("seed") != args.seed
        or float(development.get("configuration", {}).get("distance_loss_weight", -1.0))
        != 0.0
        or development.get("authority", {}).get("distance_label_manifest_sha256")
        is not None
        or development.get("authority", {}).get("test_loaded") is not False
        or development.get("authority", {}).get("external_test_loaded") is not False
    ):
        raise ValueError("development result authority differs")
    source_manifest_path = Path(development["configuration"]["source_manifest"])
    require_sha(
        source_manifest_path,
        development["configuration"]["source_manifest_sha256"],
    )
    source_manifest = load_json(source_manifest_path)
    source_root = Path(source_manifest["source_root"])
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root / "scripts"))
    import train_framecontact as training  # type: ignore

    dataset = shared.SableExternalDataset(
        args.structure_root,
        args.embedding_root,
        args.label_root,
        structure_manifest,
        embedding_manifest,
        label_manifest,
    )
    if len(dataset) != int(label_manifest["counts"]["analyzable_entries"]):
        raise ValueError("SABLE dataset cardinality differs")
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
        task: {} for task in ("contact", "epitope", "paratope")
    }
    logits_by_task: dict[str, dict[str, np.ndarray]] = {
        task: {} for task in ("contact", "epitope", "paratope")
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
                labels_by_task["contact"][entry_id] = evaluation["pair_labels"][pair_observed]
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
    frozen = shared.calibration(development["selected_checkpoint_validation"])
    metrics = {
        task: shared.summarize(
            labels_by_task[task],
            logits_by_task[task],
            temperature=frozen[f"{task}_temperature"],
            threshold=frozen[f"{task}_threshold"],
        )
        for task in ("contact", "epitope", "paratope")
    }
    payload = {
        "format_version": "abf30b_sable_component_measurement_v1",
        "status": "PASS_MODEL_SPECIFIC_ANALYST_EXPOSED_SABLE_COMPONENT_MEASUREMENT",
        "arm": args.arm,
        "seed": args.seed,
        "metrics": metrics,
        "audit": {
            "measurement_plan_sha256": plan_sha,
            "authorization_sha256": authorization_sha,
            "checkpoint_binding_sha256": binding_sha,
            "checkpoint_sha256": run_binding["checkpoint_sha256"],
            "development_result_sha256": run_binding["result_sha256"],
            "predicted_structure_manifest_sha256": sha256_file(structure_manifest_path),
            "mixed_embedding_manifest_sha256": sha256_file(embedding_manifest_path),
            "bound_label_manifest_sha256": sha256_file(label_manifest_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "model_state_sha256": state_after,
            "entry_count": len(dataset),
            "bound_label_opening_count": 1,
            "sable_external_labels_loaded": True,
            "fresh_blind_test": False,
            "analyst_has_seen_prior_model_results_on_dataset": True,
            "model_unseen_before_this_measurement": True,
            "measurement_only": True,
            "experimental_bound_coordinates_used_as_model_input": False,
            "experimental_bound_coordinates_used_for_labels_only": True,
            "cross_partner_geometry_used_as_model_input": False,
            "distance_labels_loaded": False,
            "calibration_refit": False,
            "model_parameters_updated": False,
            "selection_performed": False,
            "feedback_to_model_or_protocol": False,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "abf30b_sable_component_measurement_receipt_v1",
        "status": payload["status"],
        "arm": args.arm,
        "seed": args.seed,
        "result_sha256": sha256_file(args.output),
        "checkpoint_sha256": run_binding["checkpoint_sha256"],
        "fresh_blind_test": False,
        "model_unseen_before_this_measurement": True,
        "measurement_only": True,
        "selection_performed": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.receipt_output.write_bytes(canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.receipt_output)}, sort_keys=True))


if __name__ == "__main__":
    main()
