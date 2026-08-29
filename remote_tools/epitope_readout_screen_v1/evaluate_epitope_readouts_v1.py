#!/usr/bin/env python3
"""Replay one frozen FrameContact checkpoint with task-local epitope readouts."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score


READOUT_ORDER = (
    "direct_total",
    "paper_mean_total_contact",
    "head_plus_paper_mean_a100",
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def update_tensor_digest(
    digest: "hashlib._Hash", label: str, tensor: torch.Tensor
) -> None:
    local = tensor.detach().contiguous().cpu()
    digest.update(
        canonical_bytes(
            {"label": label, "dtype": str(local.dtype), "shape": list(local.shape)}
        )
    )
    digest.update(local.view(torch.uint8).numpy().tobytes(order="C"))


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def stable_logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(values) - np.log1p(-values)


def masked_column_reduce(
    probabilities: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str,
    k: int | None = None,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    observed = np.asarray(mask, dtype=np.bool_)
    if values.shape != observed.shape or values.ndim != 2:
        raise ValueError("pair probabilities and mask must be aligned matrices")
    outputs = []
    for column in range(values.shape[1]):
        local = values[:, column][observed[:, column]]
        if local.size == 0:
            raise ValueError("every antigen residue must have an observed antibody partner")
        if mode == "mean":
            reduced = float(local.mean())
        elif mode == "max":
            reduced = float(local.max())
        elif mode == "topk_mean":
            if k is None or k <= 0:
                raise ValueError("top-k mean requires positive k")
            count = min(int(k), int(local.size))
            reduced = float(np.partition(local, -count)[-count:].mean())
        elif mode in {"raw_noisy_or", "length_normalized_noisy_or"}:
            log_survival = np.log1p(-np.clip(local, 0.0, 1.0 - 1e-12))
            exponent = log_survival.sum() if mode == "raw_noisy_or" else log_survival.mean()
            reduced = float(-np.expm1(exponent))
        else:
            raise ValueError(f"unsupported reduction: {mode}")
        outputs.append(reduced)
    return np.asarray(outputs, dtype=np.float64)


def candidate_readouts(
    *,
    total_contact_logits: np.ndarray,
    interaction_contact_logits: np.ndarray,
    pair_mask: np.ndarray,
    epitope_total_logits: np.ndarray,
    epitope_contact_logits: np.ndarray,
    antigen_unary_logits: np.ndarray,
    direct_head_logits: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    total_probability = stable_sigmoid(total_contact_logits)
    head_logits = (
        np.asarray(direct_head_logits, dtype=np.float64)
        if direct_head_logits is not None
        else np.asarray(epitope_total_logits, dtype=np.float64)
        - np.asarray(epitope_contact_logits, dtype=np.float64)
    )
    paper_mean = masked_column_reduce(total_probability, pair_mask, mode="mean")
    paper_logit = stable_logit(paper_mean)
    readouts = {
        "direct_total": stable_sigmoid(epitope_total_logits),
        "paper_mean_total_contact": paper_mean,
        "head_plus_paper_mean_a100": stable_sigmoid(head_logits + paper_logit),
    }
    if tuple(readouts) != READOUT_ORDER:
        raise RuntimeError("readout implementation order differs from frozen contract")
    for name, scores in readouts.items():
        if scores.shape != head_logits.shape or not np.isfinite(scores).all():
            raise ValueError(f"invalid readout {name}")
    return readouts


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    truth = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, int(truth.size))
    value = 0.0
    for index in range(bins):
        if index + 1 == bins:
            local = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            local = (scores >= edges[index]) & (scores < edges[index + 1])
        if local.any():
            value += (int(local.sum()) / total) * abs(
                float(scores[local].mean()) - float(truth[local].mean())
            )
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--source-split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("gpu_4090 is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise ValueError("protocol SHA-256 differs")
    protocol = read_json(args.protocol)
    repository_root = args.repository_root.resolve()
    phase = protocol.get("phase")
    active_readouts = protocol.get("readout_order")
    if phase == "selection":
        valid_readout_contract = active_readouts == list(READOUT_ORDER)
    elif phase == "confirmation":
        valid_readout_contract = (
            isinstance(active_readouts, list)
            and len(active_readouts) == 2
            and active_readouts[0] == "direct_total"
            and active_readouts[1] in READOUT_ORDER[1:]
        )
    else:
        valid_readout_contract = False
    if (
        protocol.get("format_version") != "sabdab2_epitope_readout_protocol_v1"
        or protocol.get("status") != "frozen_before_measurement"
        or not valid_readout_contract
        or protocol.get("authority", {}).get("test_loaded") is not False
        or protocol.get("authority", {}).get("temporal_holdout_loaded") is not False
        or sha256_file(args.source_split_manifest)
        != protocol["data"]["source_split_sha256"]
        or sha256_file(args.split_manifest) != protocol["data"]["role_split_sha256"]
        or sha256_file(args.partition_manifest)
        != protocol["data"]["partition_manifest_sha256"]
        or sha256_file(args.structure_manifest)
        != protocol["data"]["structure_manifest_sha256"]
        or sha256_file(args.contact_manifest)
        != protocol["data"]["contact_manifest_sha256"]
        or sha256_file(args.embedding_manifest)
        != protocol["data"]["embedding_manifest_sha256"]
        or sha256_file(repository_root / "src/interaction_first/framecontact.py")
        != protocol["source"]["framecontact_sha256"]
        or sha256_file(repository_root / "scripts/train_framecontact.py")
        != protocol["source"]["trainer_sha256"]
    ):
        raise ValueError("frozen protocol or source authority differs")
    partition = read_json(args.partition_manifest)
    role = protocol["data"]["panel_role"]
    expected_entries = partition.get("roles", {}).get(role)
    if not isinstance(expected_entries, list) or not expected_entries:
        raise ValueError("partition role is missing")
    expected_instances = {str(entry["instance_id"]): entry for entry in expected_entries}

    sys.path.insert(0, str(repository_root / "src"))
    sys.path.insert(0, str(repository_root / "scripts"))
    import train_framecontact as training  # noqa: PLC0415
    from interaction_first.sabdab2_contacts import Sabdab2ContactDataset  # noqa: PLC0415
    from interaction_first.sabdab2_structure import (  # noqa: PLC0415
        Sabdab2StructureEmbeddingDataset,
    )

    run_root = args.run_root.resolve()
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
    result = read_json(result_path)
    receipt = read_json(receipt_path)
    seed = int(result["configuration"]["seed"])
    expected = protocol["runs"][str(seed)]
    if (
        sha256_file(result_path) != expected["result_sha256"]
        or sha256_file(checkpoint_path) != expected["checkpoint_sha256"]
        or sha256_file(receipt_path) != expected["receipt_sha256"]
        or receipt.get("result_sha256") != expected["result_sha256"]
        or receipt.get("selected_checkpoint_sha256") != expected["checkpoint_sha256"]
        or result.get("authority", {}).get("test_loaded") is not False
        or result.get("authority", {}).get("external_test_loaded") is not False
        or result.get("authority", {}).get("split_manifest_sha256")
        != protocol["data"]["source_split_sha256"]
        or result.get("authority", {}).get("structure_manifest_sha256")
        != protocol["data"]["structure_manifest_sha256"]
        or result.get("authority", {}).get("contact_manifest_sha256")
        != protocol["data"]["contact_manifest_sha256"]
        or result.get("authority", {}).get("embedding_manifest_sha256")
        != protocol["data"]["embedding_manifest_sha256"]
        or result.get("experiment") != "epiformer_style_same_protocol"
    ):
        raise ValueError("source run authority differs")

    configuration = result["configuration"]
    training._set_seed(seed)
    base = Sabdab2StructureEmbeddingDataset(
        args.structure_cache_root,
        args.embedding_root,
        None,
        split_role="validation",
        split_path=args.split_manifest,
        split_sha256=protocol["data"]["role_split_sha256"],
    )
    dataset = Sabdab2ContactDataset(base, args.contact_cache_root)
    loader_arguments = {
        "batch_size": int(configuration["evaluation_batch_size"]),
        "workers": args.workers,
        "shuffle": False,
        "seed": seed,
        "pin_memory": True,
    }
    if "persistent_workers" in inspect.signature(training._loader).parameters:
        loader_arguments["persistent_workers"] = args.workers > 0
    loader = training._loader(dataset, **loader_arguments)
    model = training._model_from_configuration(dataset[0], configuration, result["experiment"])
    device = torch.device("cuda")
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    parameter_versions_before = tuple(parameter._version for parameter in model.parameters())

    captured_epitope_head: list[torch.Tensor] = []

    def capture_epitope_head(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if captured_epitope_head:
            raise RuntimeError("epitope head called more than once per forward")
        captured_epitope_head.append(output.detach())

    hook = model.epitope_head.register_forward_hook(capture_epitope_head)

    per_instance = {name: {} for name in active_readouts}
    pooled_labels: list[np.ndarray] = []
    pooled_scores = {name: [] for name in active_readouts}
    instance_ids: list[str] = []
    label_manifest: list[dict[str, object]] = []
    decomposition_max_absolute_error = 0.0
    non_epitope_output_digest = hashlib.sha256()
    try:
        with torch.inference_mode():
            for raw_batch in loader:
                batch = training._move_batch(raw_batch, device)
                captured_epitope_head.clear()
                with training._autocast(device, True):
                    output = model(batch)
                if len(captured_epitope_head) != 1:
                    raise RuntimeError("epitope head hook count differs")
                direct_head_logits = captured_epitope_head[0].squeeze(-1)
                for index, raw_identifier in enumerate(raw_batch["instance_ids"]):
                    identifier = str(raw_identifier)
                    if identifier not in expected_instances:
                        raise ValueError(f"{identifier}: outside frozen {role} role")
                    antigen_mask = output["antigen_mask"][index]
                    antibody_mask = output["antibody_mask"][index]
                    pair_mask = output["pair_candidate_mask"][index][antibody_mask][
                        :, antigen_mask
                    ]
                    if not bool(pair_mask.all()) or int(pair_mask.sum()) != int(
                        antibody_mask.sum() * antigen_mask.sum()
                    ):
                        raise ValueError(f"{identifier}: candidate region is not complete A x G")
                    labels = (
                        batch["antigen_labels"][index][antigen_mask]
                        .cpu()
                        .numpy()
                        .astype(np.uint8, copy=False)
                    )
                    if labels.sum() <= 0 or not np.isin(labels, [0, 1]).all():
                        raise ValueError(f"{identifier}: invalid epitope labels")
                    antigen_count = int(antigen_mask.sum())
                    pair_mask_np = pair_mask.cpu().numpy().astype(np.bool_)
                    head_logits = direct_head_logits[index][antigen_mask]
                    total_logits = output["epitope_logits"][index][antigen_mask]
                    contact_marginal_logits = output["epitope_contact_logits"][index][
                        antigen_mask
                    ]
                    contact_logits_for_hash = output["contact_logits"][index][
                        antibody_mask
                    ][:, antigen_mask]
                    paratope_logits_for_hash = output["paratope_logits"][index][
                        antibody_mask
                    ]
                    non_epitope_output_digest.update(identifier.encode("utf-8"))
                    update_tensor_digest(
                        non_epitope_output_digest,
                        "contact_logits",
                        contact_logits_for_hash,
                    )
                    update_tensor_digest(
                        non_epitope_output_digest,
                        "contact_probabilities",
                        torch.sigmoid(contact_logits_for_hash.float()),
                    )
                    update_tensor_digest(
                        non_epitope_output_digest, "pair_candidate_mask", pair_mask
                    )
                    update_tensor_digest(
                        non_epitope_output_digest,
                        "paratope_logits",
                        paratope_logits_for_hash,
                    )
                    update_tensor_digest(
                        non_epitope_output_digest,
                        "paratope_probabilities",
                        torch.sigmoid(paratope_logits_for_hash.float()),
                    )
                    local_error = float(
                        (total_logits - (head_logits + contact_marginal_logits))
                        .abs()
                        .max()
                        .float()
                        .cpu()
                    )
                    decomposition_max_absolute_error = max(
                        decomposition_max_absolute_error, local_error
                    )
                    if local_error > 1e-3:
                        raise ValueError(
                            f"{identifier}: epitope logit decomposition differs ({local_error})"
                        )
                    readouts = candidate_readouts(
                        total_contact_logits=output["contact_logits"][index][antibody_mask][
                            :, antigen_mask
                        ].float().cpu().numpy(),
                        interaction_contact_logits=output["contact_interaction_logits"][index][
                            antibody_mask
                        ][:, antigen_mask].float().cpu().numpy(),
                        pair_mask=pair_mask_np,
                        epitope_total_logits=total_logits.float().cpu().numpy(),
                        epitope_contact_logits=contact_marginal_logits.float().cpu().numpy(),
                        antigen_unary_logits=output[
                            "antigen_contact_propensity_logits"
                        ][index][antigen_mask].float().cpu().numpy(),
                        direct_head_logits=head_logits.float().cpu().numpy(),
                    )
                    if labels.size != antigen_count:
                        raise RuntimeError("antigen label length differs")
                    instance_ids.append(identifier)
                    label_manifest.append(
                        {
                            "instance_id": identifier,
                            "antibody_length": int(antibody_mask.sum()),
                            "antigen_length": antigen_count,
                            "positive_count": int(labels.sum()),
                            "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
                        }
                    )
                    pooled_labels.append(labels)
                    for name in active_readouts:
                        scores = readouts[name]
                        per_instance[name][identifier] = float(
                            average_precision_score(labels, scores)
                        )
                        pooled_scores[name].append(scores)
    finally:
        hook.remove()
    if tuple(parameter._version for parameter in model.parameters()) != parameter_versions_before:
        raise RuntimeError("model parameter version changed during replay")

    all_labels = np.concatenate(pooled_labels)
    metrics = {}
    for name in active_readouts:
        all_scores = np.concatenate(pooled_scores[name])
        metrics[name] = {
            "mean_per_complex_auprc": float(np.mean(list(per_instance[name].values()))),
            "pooled_auprc": float(average_precision_score(all_labels, all_scores)),
            "raw_brier": float(np.mean(np.square(all_scores - all_labels))),
            "raw_ece_15": ece(all_labels, all_scores),
            "per_complex_auprc_by_instance": per_instance[name],
        }
    if set(instance_ids) != set(expected_instances) or len(instance_ids) != len(
        expected_instances
    ):
        raise ValueError("measured instance set differs from frozen role")
    payload = {
        "format_version": "sabdab2_epitope_readout_seed_result_v1",
        "status": "PASS_MEASUREMENT_ONLY_REPLAY",
        "phase": phase,
        "panel_role": role,
        "seed": seed,
        "readout_order": active_readouts,
        "metrics": metrics,
        "unchanged_outputs": {
            "contact_mean_per_complex_auprc": result["selected_checkpoint_validation"][
                "contact"
            ]["mean_per_complex_auprc"],
            "paratope_mean_per_complex_auprc": result["selected_checkpoint_validation"][
                "paratope"
            ]["mean_per_complex_auprc"],
            "reason": "frozen checkpoint; readout-only epitope replay",
        },
        "audit": {
            "baseline_is_direct_model_output": True,
            "instance_count": len(instance_ids),
            "antigen_cluster_count": len(
                {str(entry["antigen_cluster_id"]) for entry in expected_entries}
            ),
            "ordered_instance_ids_sha256": hashlib.sha256(
                canonical_bytes(instance_ids)
            ).hexdigest(),
            "ordered_label_manifest_sha256": hashlib.sha256(
                canonical_bytes(label_manifest)
            ).hexdigest(),
            "epitope_logit_decomposition_max_absolute_error": decomposition_max_absolute_error,
            "non_epitope_output_sha256": non_epitope_output_digest.hexdigest(),
            "candidate_sidecar_only": True,
            "model_parameter_versions_unchanged": True,
            "result_sha256": expected["result_sha256"],
            "checkpoint_sha256": expected["checkpoint_sha256"],
            "receipt_sha256": expected["receipt_sha256"],
            "protocol_sha256": args.protocol_sha256,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
            "model_parameters_updated": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "seed": seed,
                "best_readout": max(
                    active_readouts,
                    key=lambda name: metrics[name]["mean_per_complex_auprc"],
                ),
                "best_epitope_auprc": max(
                    metrics[name]["mean_per_complex_auprc"] for name in active_readouts
                ),
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
