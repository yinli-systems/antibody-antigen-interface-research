#!/usr/bin/env python3
"""Create per-complex statistics from immutable SAbDab2 geometry checkpoints."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.geometry_ablation_statistics import (  # noqa: E402
    PREDICTION_FORMAT,
    analyze_geometry_predictions,
    canonical_json_bytes,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    IntraChainResidueClassifier,
    Sabdab2StructureBatchCollator,
    Sabdab2StructureEmbeddingDataset,
    sha256_file,
)


AUDIT_FORMAT = "sabdab2_geometry_checkpoint_audit_v1"
MODE_CONFIGS = {
    "dense_sequence": {
        "geometry_feature_mode": "none",
        "attention_topology": "dense_sequence",
    },
    "topology_only": {
        "geometry_feature_mode": "none",
        "attention_topology": "spatial",
    },
    "none": {
        "geometry_feature_mode": "none",
        "attention_topology": "spatial",
    },
    "distance": {
        "geometry_feature_mode": "distance",
        "attention_topology": "spatial",
    },
    "distance_direction": {
        "geometry_feature_mode": "distance_direction",
        "attention_topology": "spatial",
    },
    "full_pose": {
        "geometry_feature_mode": "full_pose",
        "attention_topology": "spatial",
    },
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--expected-run-receipt-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--metric-tolerance", type=float, default=1e-6)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_729)
    parser.add_argument("--permutation-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-seed", type=int, default=20_260_730)
    return parser.parse_args()


def _validate_arguments(arguments: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "bootstrap_iterations",
        "permutation_iterations",
    ):
        value = getattr(arguments, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if arguments.workers < 0:
        raise ValueError("workers must be nonnegative")
    if not math.isfinite(arguments.metric_tolerance) or arguments.metric_tolerance < 0:
        raise ValueError("metric-tolerance must be finite and nonnegative")


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _authenticate(path: Path, expected: str, name: str) -> str:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"{name} expected digest is invalid")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{name} SHA-256 differs: expected {expected}, observed {observed}"
        )
    return observed


def _device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_path(root: Path, filename: str) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("Checkpoint filename must be a safe basename")
    candidates = [root / filename, root / "checkpoints" / filename]
    existing = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(existing) != 1:
        raise ValueError(
            f"Expected exactly one regular checkpoint named {filename}, found {len(existing)}"
        )
    return existing[0]


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _loader(
    dataset: Sabdab2StructureEmbeddingDataset,
    *,
    batch_size: int,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=Sabdab2StructureBatchCollator(),
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def _configuration(result: Mapping[str, Any], mode: str) -> dict[str, Any]:
    configuration = result.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("Ablation result is missing configuration")
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unsupported geometry mode: {mode}")
    required_positive = (
        "d_model",
        "num_heads",
        "num_layers",
        "num_rbf",
        "neighbor_k",
    )
    for name in required_positive:
        value = configuration.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Ablation configuration {name} is invalid")
    radius = configuration.get("neighbor_radius_angstrom")
    if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Ablation neighbor radius is invalid")
    dropout = configuration.get("dropout")
    if (
        not isinstance(dropout, (int, float))
        or not math.isfinite(dropout)
        or not 0.0 <= dropout < 1.0
    ):
        raise ValueError("Ablation dropout is invalid")
    return {
        **MODE_CONFIGS[mode],
        "d_model": int(configuration["d_model"]),
        "num_heads": int(configuration["num_heads"]),
        "num_layers": int(configuration["num_layers"]),
        "num_rbf": int(configuration["num_rbf"]),
        "neighbor_k": int(configuration["neighbor_k"]),
        "neighbor_radius": float(configuration["neighbor_radius_angstrom"]),
        "dropout": float(configuration["dropout"]),
    }


def _manifest_endpoints(
    dataset: Sabdab2StructureEmbeddingDataset,
) -> dict[str, dict[str, str]]:
    endpoints: dict[str, dict[str, str]] = {}
    for record in dataset.records:
        identifier = str(record["instance_id"])
        if identifier in endpoints:
            raise ValueError(f"Duplicate validation instance: {identifier}")
        endpoints[identifier] = {
            "antibody_sequence_sha256": str(record["antibody_sequence_sha256"]),
            "antigen_sequence_sha256": str(record["antigen_sequence_sha256"]),
        }
    return endpoints


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    endpoints: Mapping[str, Mapping[str, str]],
    device: torch.device,
    *,
    use_bf16: bool,
) -> list[dict[str, Any]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    encountered: set[str] = set()
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )
        with context:
            output = model(batch)
        for batch_index, identifier in enumerate(raw_batch["instance_ids"]):
            if identifier in encountered or identifier not in endpoints:
                raise ValueError("Prediction membership is duplicated or unknown")
            encountered.add(identifier)
            instance: dict[str, Any] = {
                "instance_id": identifier,
                **endpoints[identifier],
            }
            for side, logits_key, labels_key, mask_key in (
                ("paratope", "paratope_logits", "antibody_labels", "antibody_mask"),
                ("epitope", "epitope_logits", "antigen_labels", "antigen_mask"),
            ):
                mask = output[mask_key][batch_index]
                labels = batch[labels_key][batch_index][mask].detach().cpu()
                probabilities = (
                    output[logits_key][batch_index][mask]
                    .float()
                    .sigmoid()
                    .detach()
                    .cpu()
                )
                instance[side] = {
                    "labels": labels.to(torch.uint8).tolist(),
                    "probabilities": probabilities.tolist(),
                }
            predictions.append(instance)
    if encountered != set(endpoints):
        raise ValueError("Prediction pass did not cover the exact validation universe")
    return predictions


def _mode_records(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    modes = result.get("modes")
    if not isinstance(modes, list) or not modes:
        raise ValueError("Ablation result has no mode records")
    observed: set[str] = set()
    records: list[Mapping[str, Any]] = []
    for raw in modes:
        if not isinstance(raw, Mapping):
            raise ValueError("Ablation mode records must be objects")
        mode = raw.get("mode")
        if not isinstance(mode, str) or mode in observed:
            raise ValueError("Ablation mode names are invalid or duplicated")
        observed.add(mode)
        records.append(raw)
    return records


def _historical_metric_audit(
    result: Mapping[str, Any],
    statistics: Mapping[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    statistics_modes = statistics["modes"]
    for mode_record in _mode_records(result):
        raw_mode = str(mode_record["mode"])
        normalized_mode = "topology_only" if raw_mode == "none" else raw_mode
        if normalized_mode not in statistics_modes:
            raise ValueError(f"Statistics are missing mode {normalized_mode}")
        history = mode_record.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError(f"Mode {raw_mode} has no training history")
        historical = history[-1].get("validation")
        if not isinstance(historical, Mapping):
            raise ValueError(f"Mode {raw_mode} has no final validation metrics")
        observed = statistics_modes[normalized_mode]["pooled_residue_descriptive"]
        metric_pairs = (
            ("macro_auprc", historical["macro_auprc"], observed["macro_auprc"]),
            (
                "paratope_auprc",
                historical["paratope"]["auprc"],
                observed["paratope"]["auprc"],
            ),
            (
                "epitope_auprc",
                historical["epitope"]["auprc"],
                observed["epitope"]["auprc"],
            ),
            (
                "paratope_auroc",
                historical["paratope"]["auroc"],
                observed["paratope"]["auroc"],
            ),
            (
                "epitope_auroc",
                historical["epitope"]["auroc"],
                observed["epitope"]["auroc"],
            ),
        )
        for metric, expected, actual in metric_pairs:
            delta = abs(float(expected) - float(actual))
            checks.append(
                {
                    "mode": raw_mode,
                    "metric": metric,
                    "historical": float(expected),
                    "replayed": float(actual),
                    "absolute_difference": delta,
                    "tolerance": tolerance,
                    "passed": delta <= tolerance,
                }
            )
    return {
        "check_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def main() -> None:
    arguments = _arguments()
    _validate_arguments(arguments)
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise ValueError("output-root already exists")
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = arguments.cache_root / "STRUCTURE_CACHE_MANIFEST.json"
    result_path = arguments.ablation_root / "SABDAB2_SELF_ATTENTION_ABLATION.json"
    receipt_path = arguments.ablation_root / "RUN_RECEIPT.json"
    manifest_sha256 = _authenticate(
        manifest_path,
        arguments.expected_cache_manifest_sha256,
        "structure-cache manifest",
    )
    result_sha256 = _authenticate(
        result_path,
        arguments.expected_result_sha256,
        "ablation result",
    )
    receipt_sha256 = _authenticate(
        receipt_path,
        arguments.expected_run_receipt_sha256,
        "ablation run receipt",
    )
    result = _load_json(result_path, "ablation result")
    receipt = _load_json(receipt_path, "ablation run receipt")
    if receipt.get("result_sha256") != result_sha256:
        raise ValueError("Run receipt does not authenticate the ablation result")
    data = result.get("data")
    if (
        not isinstance(data, Mapping)
        or data.get("structure_cache_manifest_sha256") != manifest_sha256
        or data.get("native_release_test_cif_read") is not False
    ):
        raise ValueError("Ablation result does not bind the sealed cache authority")

    configuration = result.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("Ablation result is missing configuration")
    max_antigen_length = configuration.get("max_antigen_length")
    if (
        isinstance(max_antigen_length, bool)
        or not isinstance(max_antigen_length, int)
        or max_antigen_length <= 0
    ):
        raise ValueError("Ablation max antigen length is invalid")
    dataset = Sabdab2StructureEmbeddingDataset(
        arguments.cache_root,
        arguments.embedding_root,
        arguments.label_root,
        split_role="validation",
        max_antigen_length=max_antigen_length,
    )
    validation_ids = [str(record["instance_id"]) for record in dataset.records]
    expected_ids_sha256 = hashlib.sha256(
        canonical_json_bytes(validation_ids)
    ).hexdigest()
    if (
        len(validation_ids) != data.get("validation_instance_count")
        or expected_ids_sha256 != data.get("validation_instance_ids_sha256")
    ):
        raise ValueError("Validation dataset does not match the ablation result")

    selected_device = _device(arguments.device)
    use_bf16 = arguments.precision == "bf16" and selected_device.type == "cuda"
    if arguments.precision == "bf16" and selected_device.type != "cuda":
        raise ValueError("bf16 checkpoint audit requires CUDA")
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if selected_device.type == "cuda" and cublas_workspace_config not in {
        ":4096:8",
        ":16:8",
    }:
        raise ValueError(
            "CUDA checkpoint audits require CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    loader = _loader(
        dataset,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
        pin_memory=selected_device.type == "cuda",
    )
    endpoints = _manifest_endpoints(dataset)
    prediction_modes: list[dict[str, Any]] = []
    checkpoint_receipts: list[dict[str, Any]] = []
    first_example = dataset[0]
    for mode_record in _mode_records(result):
        mode = str(mode_record["mode"])
        settings = _configuration(result, mode)
        checkpoint_filename = mode_record.get("final_checkpoint")
        checkpoint_path = _checkpoint_path(
            arguments.ablation_root, str(checkpoint_filename)
        )
        checkpoint_sha256 = _authenticate(
            checkpoint_path,
            str(mode_record.get("final_checkpoint_sha256")),
            f"{mode} checkpoint",
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("mode") != mode
            or checkpoint.get("seed") != configuration.get("seed")
            or checkpoint.get("final_epoch") != configuration.get("epochs")
            or not isinstance(checkpoint.get("model_state_dict"), Mapping)
        ):
            raise ValueError(f"{mode} checkpoint metadata differs from the result")
        model = IntraChainResidueClassifier(
            antibody_input_dim=int(first_example["antibody_features"].shape[-1]),
            antigen_input_dim=int(first_example["antigen_features"].shape[-1]),
            d_model=settings["d_model"],
            num_heads=settings["num_heads"],
            num_layers=settings["num_layers"],
            geometry_feature_mode=settings["geometry_feature_mode"],
            num_rbf=settings["num_rbf"],
            neighbor_k=settings["neighbor_k"],
            neighbor_radius=settings["neighbor_radius"],
            attention_topology=settings["attention_topology"],
            dropout=settings["dropout"],
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != mode_record.get("parameter_count"):
            raise ValueError(f"{mode} parameter count differs from the result")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(selected_device)
        instances = _predict(
            model,
            loader,
            endpoints,
            selected_device,
            use_bf16=use_bf16,
        )
        prediction_modes.append({"mode": mode, "instances": instances})
        checkpoint_receipts.append(
            {
                "mode": mode,
                "relative_path": str(checkpoint_path.relative_to(arguments.ablation_root)),
                "sha256": checkpoint_sha256,
                "parameter_count": parameter_count,
            }
        )
        del model
        if selected_device.type == "cuda":
            torch.cuda.empty_cache()

    prediction_payload = {
        "format_version": PREDICTION_FORMAT,
        "claim_boundary": (
            "Development-only fixed-final-checkpoint predictions. The native and "
            "repository frozen test sets were not read. This is not partner-specific "
            "or SOTA evidence."
        ),
        "authority": {
            "structure_cache_manifest_sha256": manifest_sha256,
            "ablation_result_sha256": result_sha256,
            "ablation_run_receipt_sha256": receipt_sha256,
            "native_release_test_cif_read": False,
            "validation_instance_ids_sha256": expected_ids_sha256,
        },
        "runtime": {
            "device": str(selected_device),
            "precision": arguments.precision,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(selected_device)
                if selected_device.type == "cuda"
                else None
            ),
            "cublas_workspace_config": cublas_workspace_config,
        },
        "checkpoints": checkpoint_receipts,
        "modes": prediction_modes,
    }
    statistics = analyze_geometry_predictions(
        prediction_payload,
        bootstrap_iterations=arguments.bootstrap_iterations,
        bootstrap_seed=arguments.bootstrap_seed,
        permutation_iterations=arguments.permutation_iterations,
        permutation_seed=arguments.permutation_seed,
    )
    historical_audit = _historical_metric_audit(
        result,
        statistics,
        tolerance=arguments.metric_tolerance,
    )
    if not historical_audit["passed"]:
        raise RuntimeError("Checkpoint inference does not reproduce historical metrics")
    statistics["historical_metric_audit"] = historical_audit

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.tmp-",
            dir=arguments.output_root.parent,
        )
    )
    try:
        predictions_path = temporary_root / "PER_COMPLEX_PREDICTIONS.json"
        statistics_path = temporary_root / "GEOMETRY_ABLATION_STATISTICS.json"
        predictions_path.write_bytes(canonical_json_bytes(prediction_payload))
        statistics_path.write_bytes(canonical_json_bytes(statistics))
        receipt_payload = {
            "format_version": AUDIT_FORMAT,
            "status": "passed",
            "predictions_sha256": sha256_file(predictions_path),
            "statistics_sha256": sha256_file(statistics_path),
            "source": {
                "audit_script_sha256": sha256_file(Path(__file__)),
                "statistics_module_sha256": sha256_file(
                    SOURCE_ROOT
                    / "interaction_first"
                    / "geometry_ablation_statistics.py"
                ),
            },
            "validation_instance_count": len(validation_ids),
            "checkpoint_count": len(checkpoint_receipts),
            "historical_metric_check_count": historical_audit["check_count"],
            "historical_metric_checks_passed": historical_audit["passed_count"],
            "confirmatory_superiority_authorized": statistics["primary_gate"][
                "confirmatory_superiority_authorized"
            ],
        }
        (temporary_root / "AUDIT_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt_payload)
        )
        os.replace(temporary_root, arguments.output_root)
        print(
            json.dumps(
                {**receipt_payload, "output_root": str(arguments.output_root)},
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


if __name__ == "__main__":
    main()

