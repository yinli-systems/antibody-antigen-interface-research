#!/usr/bin/env python3
"""Measure one frozen component checkpoint on the untouched fresh156 holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


CHECKPOINT_BINDING_SHA256 = "accf813011342102fa2572aa27c2bde6796636cde3ac872756472c32b66f4ae1"
SOURCE_MANIFEST_SHA256 = {
    "baseline_current": "f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81",
    "edge_topk2_supervised": "f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55",
}
FRAMECONTACT_SHA256 = {
    "baseline_current": "a85584f288b32e66e6ab34bcd93b0001ffcd2b173bb1b0fc9e3ab48bccd6171c",
    "edge_topk2_supervised": "121b3e06e24db0eeb79db100a3870d19b299b44f77355b590c97ab45cc8f38fe",
}


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


@torch.no_grad()
def collect_per_complex_auprc(
    model: torch.nn.Module,
    loader: object,
    device: torch.device,
    *,
    use_bf16: bool,
    training: object,
) -> dict[str, dict[str, float]]:
    from sklearn.metrics import average_precision_score

    by_instance: dict[str, dict[str, float]] = {}
    for raw_batch in loader:
        batch = training._move_batch(raw_batch, device)
        with training._autocast(device, use_bf16):
            output = model(batch, return_attention=False)
        for batch_index, instance_id in enumerate(raw_batch["instance_ids"]):
            identifier = str(instance_id)
            if identifier in by_instance:
                raise ValueError(f"Duplicate holdout instance: {identifier}")
            pair_mask = output["pair_candidate_mask"][batch_index]
            pair_labels = (
                batch["pair_labels"][batch_index][pair_mask]
                .cpu()
                .numpy()
                .astype(np.int8, copy=False)
            )
            pair_logits = (
                output["contact_logits"][batch_index][pair_mask].float().cpu().numpy()
            )
            metrics = {
                "contact": float(
                    average_precision_score(
                        pair_labels, training._sigmoid_numpy(pair_logits)
                    )
                )
            }
            for name, labels_key, logits_key, mask_key in (
                ("paratope", "antibody_labels", "paratope_logits", "antibody_mask"),
                ("epitope", "antigen_labels", "epitope_logits", "antigen_mask"),
            ):
                mask = output[mask_key][batch_index]
                labels = (
                    batch[labels_key][batch_index][mask]
                    .cpu()
                    .numpy()
                    .astype(np.int8, copy=False)
                )
                logits = output[logits_key][batch_index][mask].float().cpu().numpy()
                metrics[name] = float(
                    average_precision_score(labels, training._sigmoid_numpy(logits))
                )
            by_instance[identifier] = metrics
    return by_instance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--checkpoint-binding", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(SOURCE_MANIFEST_SHA256), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
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

    authority = args.campaign_root / "authority"
    protocol_path = authority / "PROTOCOL_TASK_ROUTED_FRESH156_V1.json"
    split_path = authority / "split-fresh156-measurement-only.csv"
    panel_path = authority / "validation-fresh156-reciprocal-7.json"
    if (
        sha256_file(protocol_path) != args.protocol_sha256
        or sha256_file(args.checkpoint_binding) != CHECKPOINT_BINDING_SHA256
        or sha256_file(args.source_manifest) != SOURCE_MANIFEST_SHA256[args.arm]
        or sha256_file(args.repository_root / "src/interaction_first/framecontact.py")
        != FRAMECONTACT_SHA256[args.arm]
    ):
        raise ValueError("frozen source authority differs")
    protocol = json.loads(protocol_path.read_text())
    binding = json.loads(args.checkpoint_binding.read_text())
    holdout = protocol.get("holdout", {})
    if (
        protocol.get("status") != "FROZEN_BEFORE_FRESH_HOLDOUT_MEASUREMENT"
        or protocol.get("fresh_holdout_selection_performed") is not False
        or protocol.get("test_loaded") is not False
        or protocol.get("temporal_holdout_loaded") is not False
        or protocol.get("checkpoint_binding_sha256") != CHECKPOINT_BINDING_SHA256
        or protocol.get("execution_sources", {}).get("evaluator_sha256")
        != sha256_file(Path(__file__))
        or sha256_file(split_path) != holdout.get("split_sha256")
        or sha256_file(panel_path) != holdout.get("partner_panel_sha256")
        or binding.get("status") != "FROZEN_BEFORE_CONFIRMATION_MEASUREMENT"
        or binding.get("test_loaded") is not False
    ):
        raise ValueError("fresh holdout authorization differs")
    with split_path.open(newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    expected_ids = {
        str(row["instance_id"])
        for row in split_rows
        if row["role"] == "validation"
        and row.get("fresh_holdout_included") == "true"
    }
    if len(expected_ids) != 156:
        raise ValueError("fresh holdout identifiers differ")

    bound = binding["runs"][args.arm][str(args.seed)]
    run_root = Path(bound["run_root"])
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
    if (
        sha256_file(result_path) != bound["result_sha256"]
        or sha256_file(checkpoint_path) != bound["checkpoint_sha256"]
        or sha256_file(receipt_path) != bound["receipt_sha256"]
    ):
        raise ValueError("bound checkpoint differs")
    selected_result = json.loads(result_path.read_text())
    configuration = selected_result["configuration"]
    if (
        selected_result.get("experiment") != "epiformer_style_same_protocol"
        or configuration.get("seed") != args.seed
        or selected_result.get("authority", {}).get("test_loaded") is not False
        or selected_result.get("authority", {}).get("external_test_loaded") is not False
    ):
        raise ValueError("bound checkpoint authority differs")

    repository_root = args.repository_root.resolve()
    sys.path.insert(0, str(repository_root / "src"))
    sys.path.insert(0, str(repository_root / "scripts"))
    import train_framecontact as training  # noqa: PLC0415
    from interaction_first.sabdab2_contacts import Sabdab2ContactDataset  # noqa: PLC0415
    from interaction_first.sabdab2_structure import (  # noqa: PLC0415
        Sabdab2StructureEmbeddingDataset,
    )

    training._set_seed(args.seed)
    base = Sabdab2StructureEmbeddingDataset(
        args.structure_cache_root,
        args.embedding_root,
        None,
        split_role="validation",
        split_path=split_path,
        split_sha256=holdout["split_sha256"],
    )
    dataset = Sabdab2ContactDataset(base, args.contact_cache_root)
    if {str(row["instance_id"]) for row in dataset.records} != expected_ids:
        raise ValueError("fresh holdout dataset differs")
    loader_arguments = {
        "batch_size": int(configuration["evaluation_batch_size"]),
        "workers": args.workers,
        "shuffle": False,
        "seed": args.seed,
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    state_before = training._state_sha256(model)
    per_complex_auprc = collect_per_complex_auprc(
        model,
        loader,
        device,
        use_bf16=configuration["precision"] == "bf16",
        training=training,
    )
    if set(per_complex_auprc) != expected_ids:
        raise ValueError("per-complex fresh holdout identifiers differ")
    metrics = training._evaluate(
        model,
        loader,
        dataset,
        json.loads(panel_path.read_text()),
        device,
        use_bf16=configuration["precision"] == "bf16",
        evaluation_batch_size=int(configuration["evaluation_batch_size"]),
        score_fraction=float(configuration["score_fraction"]),
        seed=args.seed,
        calibration_max_pairs=int(configuration["calibration_max_pairs"]),
        specificity_score_source=bound["specificity_score_source"],
        evaluate_partner_retrieval=True,
        collect_attention=False,
        frozen_calibration=bound["frozen_calibration"],
    )
    state_after = training._state_sha256(model)
    if state_after != state_before:
        raise RuntimeError("model parameters changed during fresh holdout measurement")
    if (
        metrics.get("calibration_source") != "frozen_validation"
        or metrics["epitope"].get("calibration_source") != "frozen_validation"
        or metrics["paratope"].get("calibration_source") != "frozen_validation"
    ):
        raise ValueError("fresh holdout calibration was refit")

    payload = {
        "format_version": "sabdab2_edge_topk2_task_routed_component_result_v1",
        "status": "PASS_MEASUREMENT_ONLY_FRESH156",
        "arm": args.arm,
        "seed": args.seed,
        "metrics": metrics,
        "per_complex_auprc": per_complex_auprc,
        "audit": {
            "protocol_sha256": args.protocol_sha256,
            "checkpoint_binding_sha256": CHECKPOINT_BINDING_SHA256,
            "checkpoint_sha256": bound["checkpoint_sha256"],
            "model_state_sha256": state_after,
            "fresh_holdout_complexes": len(expected_ids),
            "fresh_holdout_selection_performed": False,
            "calibration_refit_on_fresh_holdout": False,
            "model_parameters_updated": False,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    result_sha256 = sha256_file(args.output)
    receipt = {
        "format_version": "sabdab2_edge_topk2_task_routed_component_receipt_v1",
        "status": payload["status"],
        "arm": args.arm,
        "seed": args.seed,
        "result_sha256": result_sha256,
        "protocol_sha256": args.protocol_sha256,
        "checkpoint_binding_sha256": CHECKPOINT_BINDING_SHA256,
        "fresh_holdout_selection_performed": False,
        "model_parameters_updated": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.receipt.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "arm": args.arm,
                "seed": args.seed,
                "contact": metrics["contact"]["mean_per_complex_auprc"],
                "epitope": metrics["epitope"]["mean_per_complex_auprc"],
                "paratope": metrics["paratope"]["mean_per_complex_auprc"],
                "result_sha256": result_sha256,
                "receipt_sha256": sha256_file(args.receipt),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
