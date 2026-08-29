#!/usr/bin/env python3
"""Run a factorized, equal-parameter SAbDab2 self-attention ablation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_structure import (  # noqa: E402
    IntraChainResidueClassifier,
    Sabdab2StructureBatchCollator,
    Sabdab2StructureEmbeddingDataset,
    canonical_json_bytes,
    sha256_file,
)
from interaction_first.attention_diagnostics import (  # noqa: E402
    AttentionDiagnosticsAccumulator,
)


@dataclass(frozen=True)
class ExperimentSpec:
    geometry_feature_mode: str
    attention_graph_mode: str


EXPERIMENTS = {
    "sequence_dense": ExperimentSpec("none", "dense"),
    "sequence_local": ExperimentSpec("none", "sequence"),
    "spatial_topology": ExperimentSpec("none", "spatial"),
    "distance": ExperimentSpec("distance", "spatial"),
    "distance_direction": ExperimentSpec("distance_direction", "spatial"),
    "full_pose": ExperimentSpec("full_pose", "spatial"),
}
RESULT_FORMAT = "sabdab2_self_attention_factorized_ablation_v2"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-antigen-length", type=int, default=384)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument("--neighbor-k", type=int, default=32)
    parser.add_argument("--neighbor-radius", type=float, default=16.0)
    parser.add_argument("--sequence-window", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "batch_size",
        "max_antigen_length",
        "d_model",
        "num_heads",
        "num_layers",
        "num_rbf",
        "neighbor_k",
        "sequence_window",
    ):
        value = getattr(arguments, name)
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if arguments.workers < 0:
        raise ValueError("workers must be nonnegative")
    for name in ("neighbor_radius", "learning_rate"):
        if not math.isfinite(getattr(arguments, name)) or getattr(arguments, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0.0 <= arguments.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not math.isfinite(arguments.weight_decay) or arguments.weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    if len(set(arguments.experiments)) != len(arguments.experiments):
        raise ValueError("experiments must be unique")


def _device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    paratope_pos_weight: torch.Tensor,
    epitope_pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    antibody_mask = output["antibody_mask"]
    antigen_mask = output["antigen_mask"]
    paratope = F.binary_cross_entropy_with_logits(
        output["paratope_logits"][antibody_mask],
        batch["antibody_labels"][antibody_mask],
        pos_weight=paratope_pos_weight,
    )
    epitope = F.binary_cross_entropy_with_logits(
        output["epitope_logits"][antigen_mask],
        batch["antigen_labels"][antigen_mask],
        pos_weight=epitope_pos_weight,
    )
    return 0.5 * (paratope + epitope), paratope, epitope


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = probabilities >= 0.5
    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "precision_at_0_5": float(precision_score(labels, predictions, zero_division=0)),
        "recall_at_0_5": float(recall_score(labels, predictions, zero_division=0)),
        "f1_at_0_5": float(f1_score(labels, predictions, zero_division=0)),
        "mcc_at_0_5": float(matthews_corrcoef(labels, predictions)),
        "positive_count": int(labels.sum()),
        "residue_count": int(labels.size),
    }


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    paratope_pos_weight: torch.Tensor,
    epitope_pos_weight: torch.Tensor,
    *,
    use_bf16: bool,
    collect_attention: bool = False,
    prediction_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    streams: dict[str, dict[str, list[torch.Tensor]]] = {
        "paratope": {"labels": [], "probabilities": []},
        "epitope": {"labels": [], "probabilities": []},
    }
    per_complex: dict[str, dict[str, list[np.ndarray]]] = {
        "paratope": {"labels": [], "probabilities": []},
        "epitope": {"labels": [], "probabilities": []},
    }
    instance_ids: list[str] = []
    diagnostics = AttentionDiagnosticsAccumulator() if collect_attention else None
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )
        with context:
            output = model(batch, return_attention=collect_attention)
            loss, _, _ = _loss(
                output,
                batch,
                paratope_pos_weight,
                epitope_pos_weight,
            )
        total_loss += float(loss.detach().cpu())
        total_batches += 1
        if diagnostics is not None:
            attention_details = output["attention_details"]
            if attention_details is None:
                raise RuntimeError("Model did not return requested attention diagnostics")
            diagnostics.update(
                "antibody", attention_details["antibody"], output["antibody_mask"]
            )
            diagnostics.update(
                "antigen", attention_details["antigen"], output["antigen_mask"]
            )
        batch_instance_ids = list(batch["instance_ids"])
        instance_ids.extend(batch_instance_ids)
        for name, logits_key, labels_key, mask_key in (
            ("paratope", "paratope_logits", "antibody_labels", "antibody_mask"),
            ("epitope", "epitope_logits", "antigen_labels", "antigen_mask"),
        ):
            mask = output[mask_key]
            streams[name]["labels"].append(batch[labels_key][mask].detach().cpu())
            streams[name]["probabilities"].append(
                output[logits_key][mask].float().sigmoid().detach().cpu()
            )
            probabilities = output[logits_key].float().sigmoid().detach().cpu()
            labels = batch[labels_key].detach().cpu()
            mask_cpu = mask.detach().cpu()
            for batch_index in range(len(batch_instance_ids)):
                per_complex[name]["labels"].append(
                    labels[batch_index][mask_cpu[batch_index]].numpy()
                )
                per_complex[name]["probabilities"].append(
                    probabilities[batch_index][mask_cpu[batch_index]].numpy()
                )
    result: dict[str, Any] = {"weighted_bce": total_loss / total_batches}
    for name, values in streams.items():
        labels = torch.cat(values["labels"]).numpy().astype(np.int64, copy=False)
        probabilities = torch.cat(values["probabilities"]).numpy()
        result[name] = _binary_metrics(labels, probabilities)
    result["macro_auprc"] = 0.5 * (
        result["paratope"]["auprc"] + result["epitope"]["auprc"]
    )
    result["macro_auroc"] = 0.5 * (
        result["paratope"]["auroc"] + result["epitope"]["auroc"]
    )
    if diagnostics is not None:
        result["attention_diagnostics"] = diagnostics.result()
    if prediction_path is not None:
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "instance_ids": np.asarray(instance_ids, dtype=np.str_),
        }
        for name, values in per_complex.items():
            lengths = np.asarray([len(value) for value in values["labels"]], dtype=np.int64)
            offsets = np.concatenate((np.zeros(1, dtype=np.int64), lengths.cumsum()))
            arrays[f"{name}_offsets"] = offsets
            arrays[f"{name}_labels"] = np.concatenate(values["labels"]).astype(
                np.float32, copy=False
            )
            arrays[f"{name}_probabilities"] = np.concatenate(
                values["probabilities"]
            ).astype(np.float32, copy=False)
        np.savez_compressed(prediction_path, **arrays)
    return result


def _positive_weights(dataset: Sabdab2StructureEmbeddingDataset) -> tuple[float, float]:
    ab_total = sum(int(record["antibody_length"]) for record in dataset.records)
    ag_total = sum(int(record["antigen_length"]) for record in dataset.records)
    ab_positive = sum(int(record["paratope_positive_count"]) for record in dataset.records)
    ag_positive = sum(int(record["epitope_positive_count"]) for record in dataset.records)
    if not 0 < ab_positive < ab_total or not 0 < ag_positive < ag_total:
        raise ValueError("Training labels must contain both classes for each partner")
    return (ab_total - ab_positive) / ab_positive, (ag_total - ag_positive) / ag_positive


def _loader(
    dataset: Sabdab2StructureEmbeddingDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=Sabdab2StructureBatchCollator(),
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def _train_experiment(
    experiment: str,
    arguments: argparse.Namespace,
    train_dataset: Sabdab2StructureEmbeddingDataset,
    validation_dataset: Sabdab2StructureEmbeddingDataset,
    device: torch.device,
    temporary_root: Path,
    expected_initial_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    specification = EXPERIMENTS[experiment]
    _set_seed(arguments.seed)
    train_loader = _loader(
        train_dataset,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
        shuffle=True,
        seed=arguments.seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
        shuffle=False,
        seed=arguments.seed,
        pin_memory=device.type == "cuda",
    )
    first_example = train_dataset[0]
    model = IntraChainResidueClassifier(
        antibody_input_dim=int(first_example["antibody_features"].shape[-1]),
        antigen_input_dim=int(first_example["antigen_features"].shape[-1]),
        d_model=arguments.d_model,
        num_heads=arguments.num_heads,
        num_layers=arguments.num_layers,
        geometry_feature_mode=specification.geometry_feature_mode,
        attention_graph_mode=specification.attention_graph_mode,
        num_rbf=arguments.num_rbf,
        neighbor_k=arguments.neighbor_k,
        neighbor_radius=arguments.neighbor_radius,
        sequence_window=arguments.sequence_window,
        dropout=arguments.dropout,
    )
    initial_sha256 = _state_sha256(model)
    if expected_initial_sha256 is not None and initial_sha256 != expected_initial_sha256:
        raise RuntimeError("Ablation modes do not share an identical initialization")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    ab_weight, ag_weight = _positive_weights(train_dataset)
    paratope_pos_weight = torch.tensor(ab_weight, device=device)
    epitope_pos_weight = torch.tensor(ag_weight, device=device)
    use_bf16 = arguments.precision == "bf16" and device.type == "cuda"
    history: list[dict[str, Any]] = []
    best_macro_auprc = -math.inf
    best_epoch = 0
    best_checkpoint_path = temporary_root / f"{experiment}_best_dev.pt"
    started = time.monotonic()
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        epoch_loss = 0.0
        paratope_loss = 0.0
        epitope_loss = 0.0
        steps = 0
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with context:
                output = model(batch)
                loss, ab_loss, ag_loss = _loss(
                    output,
                    batch,
                    paratope_pos_weight,
                    epitope_pos_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            paratope_loss += float(ab_loss.detach().cpu())
            epitope_loss += float(ag_loss.detach().cpu())
            steps += 1
        validation = _evaluate(
            model,
            validation_loader,
            device,
            paratope_pos_weight,
            epitope_pos_weight,
            use_bf16=use_bf16,
        )
        record = {
            "epoch": epoch,
            "train": {
                "weighted_bce": epoch_loss / steps,
                "paratope_weighted_bce": paratope_loss / steps,
                "epitope_weighted_bce": epitope_loss / steps,
            },
            "validation": validation,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(record)
        if validation["macro_auprc"] > best_macro_auprc:
            best_macro_auprc = float(validation["macro_auprc"])
            best_epoch = epoch
            torch.save(
                {
                    "format_version": RESULT_FORMAT,
                    "experiment": experiment,
                    "experiment_spec": asdict(specification),
                    "seed": arguments.seed,
                    "selected_epoch": epoch,
                    "selection_rule": "maximum development macro AUPRC",
                    "model_state_dict": model.state_dict(),
                },
                best_checkpoint_path,
            )
        print(json.dumps({"experiment": experiment, **record}, sort_keys=True), flush=True)

    prediction_path = temporary_root / f"{experiment}_validation_predictions.npz"
    final_validation = _evaluate(
        model,
        validation_loader,
        device,
        paratope_pos_weight,
        epitope_pos_weight,
        use_bf16=use_bf16,
        collect_attention=True,
        prediction_path=prediction_path,
    )
    for metric in ("weighted_bce", "macro_auprc", "macro_auroc"):
        if not math.isclose(
            float(final_validation[metric]),
            float(history[-1]["validation"][metric]),
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise RuntimeError(f"Repeated final validation changed for {metric}")

    checkpoint_path = temporary_root / f"{experiment}_final.pt"
    torch.save(
        {
            "format_version": RESULT_FORMAT,
            "experiment": experiment,
            "experiment_spec": asdict(specification),
            "seed": arguments.seed,
            "final_epoch": arguments.epochs,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return (
        {
            "experiment": experiment,
            "experiment_spec": asdict(specification),
            "parameter_count": parameter_count,
            "initial_state_sha256": initial_sha256,
            "final_checkpoint": checkpoint_path.name,
            "final_checkpoint_sha256": sha256_file(checkpoint_path),
            "history": history,
            "final_attention_diagnostics": final_validation["attention_diagnostics"],
            "validation_predictions": prediction_path.name,
            "validation_predictions_sha256": sha256_file(prediction_path),
            "development_best_checkpoint": best_checkpoint_path.name,
            "development_best_checkpoint_sha256": sha256_file(best_checkpoint_path),
            "development_best_epoch": best_epoch,
            "development_best_macro_auprc": best_macro_auprc,
            "reported_epoch": arguments.epochs,
            "primary_reporting_rule": "fixed final epoch",
            "secondary_checkpoint_rule": (
                "maximum development macro AUPRC; never selected from frozen test"
            ),
        },
        initial_sha256,
    )


def main() -> None:
    arguments = _arguments()
    _validate_arguments(arguments)
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise ValueError("output-root already exists")
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = arguments.cache_root / "STRUCTURE_CACHE_MANIFEST.json"
    observed_manifest_sha256 = sha256_file(manifest_path)
    if observed_manifest_sha256 != arguments.expected_cache_manifest_sha256:
        raise ValueError("Structure cache manifest SHA-256 differs")
    train_dataset = Sabdab2StructureEmbeddingDataset(
        arguments.cache_root,
        arguments.embedding_root,
        arguments.label_root,
        split_role="train",
        max_antigen_length=arguments.max_antigen_length,
    )
    validation_dataset = Sabdab2StructureEmbeddingDataset(
        arguments.cache_root,
        arguments.embedding_root,
        arguments.label_root,
        split_role="validation",
        max_antigen_length=arguments.max_antigen_length,
    )
    device = _device(arguments.device)
    if arguments.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 mode is currently supported only on CUDA")
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device.type == "cuda" and cublas_workspace_config not in {":4096:8", ":16:8"}:
        raise ValueError(
            "CUDA ablations require CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.tmp-",
            dir=arguments.output_root.parent,
        )
    )
    try:
        configuration = {
            "seed": arguments.seed,
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "workers": arguments.workers,
            "max_antigen_length": arguments.max_antigen_length,
            "d_model": arguments.d_model,
            "num_heads": arguments.num_heads,
            "num_layers": arguments.num_layers,
            "num_rbf": arguments.num_rbf,
            "neighbor_k": arguments.neighbor_k,
            "neighbor_radius_angstrom": arguments.neighbor_radius,
            "sequence_window": arguments.sequence_window,
            "dropout": arguments.dropout,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
            "precision": arguments.precision,
            "experiments": arguments.experiments,
            "experiment_specs": {
                name: asdict(EXPERIMENTS[name]) for name in arguments.experiments
            },
            "factorization": (
                "sequence_dense/sequence_local/spatial_topology isolate graph choice; "
                "spatial_topology/distance/distance_direction/full_pose isolate geometry "
                "feature content on one fixed spatial graph"
            ),
        }
        experiment_results = []
        initial_sha256: str | None = None
        for experiment in arguments.experiments:
            result, observed_initial_sha256 = _train_experiment(
                experiment,
                arguments,
                train_dataset,
                validation_dataset,
                device,
                temporary_root,
                initial_sha256,
            )
            initial_sha256 = observed_initial_sha256
            experiment_results.append(result)
        parameter_counts = {result["parameter_count"] for result in experiment_results}
        if len(parameter_counts) != 1:
            raise RuntimeError("Ablation modes have unequal parameter counts")
        train_ids = [record["instance_id"] for record in train_dataset.records]
        validation_ids = [record["instance_id"] for record in validation_dataset.records]
        result_payload = {
            "format_version": RESULT_FORMAT,
            "claim_boundary": (
                "Development-only locked-split ablation. No frozen test was read and "
                "no confirmatory, external, clinical, or benchmark-improvement claim is made."
            ),
            "data": {
                "structure_cache_manifest_sha256": observed_manifest_sha256,
                "train_instance_count": len(train_ids),
                "validation_instance_count": len(validation_ids),
                "train_instance_ids_sha256": hashlib.sha256(
                    canonical_json_bytes(train_ids)
                ).hexdigest(),
                "validation_instance_ids_sha256": hashlib.sha256(
                    canonical_json_bytes(validation_ids)
                ).hexdigest(),
                "native_release_test_cif_read": False,
            },
            "configuration": configuration,
            "runtime": {
                "device": str(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cublas_workspace_config": cublas_workspace_config,
                "deterministic_algorithms": True,
            },
            "model_parameter_count": next(iter(parameter_counts)),
            "identical_initial_state_sha256": initial_sha256,
            "experiments": experiment_results,
        }
        result_path = temporary_root / "SABDAB2_SELF_ATTENTION_ABLATION.json"
        result_path.write_bytes(canonical_json_bytes(result_payload))
        receipt = {
            "format_version": RESULT_FORMAT,
            "result_sha256": sha256_file(result_path),
            "model_parameter_count": next(iter(parameter_counts)),
            "experiment_count": len(experiment_results),
            "train_instance_count": len(train_ids),
            "validation_instance_count": len(validation_ids),
        }
        (temporary_root / "RUN_RECEIPT.json").write_bytes(canonical_json_bytes(receipt))
        os.replace(temporary_root, arguments.output_root)
        print(json.dumps({**receipt, "output_root": str(arguments.output_root)}, sort_keys=True))
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


if __name__ == "__main__":
    main()
