#!/usr/bin/env python3
"""Measure one frozen seed/arm across four bound/unbound geometry conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch


ARMS = ("baseline_current", "edge_topk2_supervised")
CONDITIONS = (
    "bound_bound",
    "unbound_ab_bound_ag",
    "bound_ab_unbound_ag",
    "unbound_unbound",
)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def calibration(selected: dict[str, object]) -> dict[str, float]:
    return {
        "contact_temperature": float(selected["validation_temperature"]),
        "contact_threshold": float(selected["contact"]["threshold_selected_on_validation"]),
        "epitope_temperature": float(selected["epitope"]["validation_temperature"]),
        "epitope_threshold": float(selected["epitope"]["threshold_selected_on_validation"]),
        "paratope_temperature": float(selected["paratope"]["validation_temperature"]),
        "paratope_threshold": float(selected["paratope"]["threshold_selected_on_validation"]),
    }


def summarize(
    labels_by_case: dict[str, np.ndarray],
    logits_by_case: dict[str, np.ndarray],
    *,
    temperature: float,
    threshold: float,
) -> dict[str, object]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if set(labels_by_case) != set(logits_by_case):
        raise ValueError("case keys differ")
    probabilities_by_case = {
        case: sigmoid(logits_by_case[case] / temperature)
        for case in sorted(labels_by_case)
    }
    eligible = {
        case: float(average_precision_score(labels_by_case[case], probabilities))
        for case, probabilities in probabilities_by_case.items()
        if bool(labels_by_case[case].any())
    }
    labels = np.concatenate([labels_by_case[case] for case in sorted(labels_by_case)])
    probabilities = np.concatenate(
        [probabilities_by_case[case] for case in sorted(probabilities_by_case)]
    )
    predictions = (probabilities >= threshold).astype(np.int8)
    bin_ids = np.minimum((probabilities * 15).astype(np.int64), 14)
    ece = 0.0
    for bin_index in range(15):
        in_bin = bin_ids == bin_index
        if bool(in_bin.any()):
            ece += float(in_bin.mean()) * abs(
                float(probabilities[in_bin].mean()) - float(labels[in_bin].mean())
            )
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece_15": ece,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "threshold_frozen_on_development": threshold,
        "temperature_frozen_on_development": temperature,
        "mean_per_eligible_complex_average_precision": float(np.mean(list(eligible.values()))),
        "eligible_complex_count": len(eligible),
        "total_complex_count": len(labels_by_case),
        "per_complex_average_precision": eligible,
        "zero_positive_complex_ids": sorted(set(labels_by_case) - set(eligible)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("gpu_4090 is required")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("one RTX 4090 is required")
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    require_sha(args.protocol, args.protocol_sha256, "protocol")
    data_manifest_path = args.data_root / "BOUND_UNBOUND_MEASUREMENT_DATA_MANIFEST_V1.json"
    require_sha(data_manifest_path, args.data_manifest_sha256, "data manifest")
    protocol = json.loads(args.protocol.read_text())
    data_manifest = json.loads(data_manifest_path.read_text())
    if (
        protocol.get("format_version") != "framecontact_bound_unbound_measurement_protocol_v1"
        or protocol.get("status") != "FROZEN_AFTER_LABEL_FREE_GATE_BEFORE_CONTACTS"
        or protocol.get("execution", {}).get("evaluator_sha256") != sha256_file(Path(__file__))
        or data_manifest.get("status") != "PASS_BOUND_UNBOUND_MEASUREMENT_DATA_MATERIALIZED"
        or data_manifest.get("protocol_sha256") != args.protocol_sha256
    ):
        raise ValueError("measurement authority differs")
    binding = protocol["model_bindings"][args.arm][str(args.seed)]
    if args.source_root.resolve() != Path(binding["source_root"]).resolve():
        raise ValueError("source root differs")
    for path, expected, label in (
        (Path(binding["checkpoint"]), binding["checkpoint_sha256"], "checkpoint"),
        (Path(binding["result"]), binding["result_sha256"], "development result"),
        (Path(binding["receipt"]), binding["receipt_sha256"], "development receipt"),
        (Path(binding["source_manifest"]), binding["source_manifest_sha256"], "source manifest"),
    ):
        require_sha(path, expected, label)
    development = json.loads(Path(binding["result"]).read_text())
    if (
        development.get("authority", {}).get("test_loaded") is not False
        or int(development.get("configuration", {}).get("seed", -1)) != args.seed
    ):
        raise ValueError("development result differs")
    frozen = calibration(development["selected_checkpoint_validation"])

    sys.path.insert(0, str(args.source_root / "src"))
    sys.path.insert(0, str(args.source_root / "scripts"))
    import train_framecontact as training  # type: ignore
    from interaction_first.sabdab2_contacts import Sabdab2ContactDataset  # type: ignore
    from interaction_first.sabdab2_structure import Sabdab2StructureEmbeddingDataset  # type: ignore

    device = torch.device("cuda")
    training._set_seed(args.seed)
    condition_results: dict[str, object] = {}
    model_state_sha256 = None
    for condition in CONDITIONS:
        structure_root = args.data_root / "structures" / condition
        contact_root = args.data_root / "contacts" / condition
        require_sha(
            structure_root / "STRUCTURE_CACHE_MANIFEST.json",
            data_manifest["conditions"][condition]["structure_manifest_sha256"],
            f"{condition} structure manifest",
        )
        require_sha(
            contact_root / "CONTACT_CACHE_MANIFEST.json",
            data_manifest["conditions"][condition]["contact_manifest_sha256"],
            f"{condition} contact manifest",
        )
        base = Sabdab2StructureEmbeddingDataset(
            structure_root,
            args.data_root / "embeddings",
            None,
            split_role="test",
            split_path=args.data_root / "BOUND_UNBOUND_SPLIT.csv",
            split_sha256=data_manifest["split_sha256"],
        )
        dataset = Sabdab2ContactDataset(base, contact_root)
        loader = training._loader(
            dataset,
            batch_size=1,
            workers=4,
            shuffle=False,
            seed=args.seed,
            pin_memory=True,
        )
        model = training._model_from_configuration(
            dataset[0], development["configuration"], "epiformer_style_same_protocol"
        ).to(device)
        checkpoint = torch.load(Path(binding["checkpoint"]), map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        before = training._state_sha256(model)
        labels_by_task: dict[str, dict[str, np.ndarray]] = {
            "contact": {}, "epitope": {}, "paratope": {}
        }
        logits_by_task: dict[str, dict[str, np.ndarray]] = {
            "contact": {}, "epitope": {}, "paratope": {}
        }
        with torch.no_grad():
            for raw_batch in loader:
                batch = training._move_batch(raw_batch, device)
                with training._autocast(device, True):
                    output = model(batch, return_attention=False)
                for batch_index, case_id in enumerate(raw_batch["instance_ids"]):
                    case = str(case_id)
                    pair_mask = output["pair_candidate_mask"][batch_index]
                    labels_by_task["contact"][case] = batch["pair_labels"][batch_index][pair_mask].cpu().numpy().astype(np.int8)
                    logits_by_task["contact"][case] = output["contact_logits"][batch_index][pair_mask].float().cpu().numpy()
                    for task, label_key, logit_key, mask_key in (
                        ("paratope", "antibody_labels", "paratope_logits", "antibody_mask"),
                        ("epitope", "antigen_labels", "epitope_logits", "antigen_mask"),
                    ):
                        mask = output[mask_key][batch_index]
                        labels_by_task[task][case] = batch[label_key][batch_index][mask].cpu().numpy().astype(np.int8)
                        logits_by_task[task][case] = output[logit_key][batch_index][mask].float().cpu().numpy()
        after = training._state_sha256(model)
        if after != before:
            raise RuntimeError("model parameters changed during measurement")
        if model_state_sha256 is None:
            model_state_sha256 = after
        elif model_state_sha256 != after:
            raise RuntimeError("reloaded model state differs across conditions")
        condition_results[condition] = {
            task: summarize(
                labels_by_task[task],
                logits_by_task[task],
                temperature=frozen[f"{task}_temperature"],
                threshold=frozen[f"{task}_threshold"],
            )
            for task in ("contact", "epitope", "paratope")
        }
        print(json.dumps({"arm": args.arm, "seed": args.seed, "condition": condition, "case_count": len(dataset)}), flush=True)
    payload = {
        "format_version": "framecontact_bound_unbound_component_result_v1",
        "status": "PASS_BOUND_UNBOUND_COMPONENT_MEASUREMENT",
        "arm": args.arm,
        "seed": args.seed,
        "conditions": condition_results,
        "audit": {
            "protocol_sha256": args.protocol_sha256,
            "data_manifest_sha256": args.data_manifest_sha256,
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "model_state_sha256": model_state_sha256,
            "case_count": 52,
            "calibration_refit": False,
            "model_parameters_updated": False,
            "selection_performed": False,
            "old_exposed_test_access": False,
            "temporal_holdout_access": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "framecontact_bound_unbound_component_receipt_v1",
        "status": payload["status"],
        "arm": args.arm,
        "seed": args.seed,
        "result_sha256": sha256_file(args.output),
        "protocol_sha256": args.protocol_sha256,
        "selection_performed": False,
    }
    args.receipt.write_bytes(canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
