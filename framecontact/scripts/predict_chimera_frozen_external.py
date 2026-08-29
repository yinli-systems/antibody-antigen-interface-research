#!/usr/bin/env python3
"""Run six frozen checkpoint inferences without computing external metrics."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from interaction_first.chimera_cache import ChimeraAdapterCacheDataset  # noqa: E402
from interaction_first.chimera_external import verify_frozen_authority  # noqa: E402
from interaction_first.sabdab2_structure import (  # noqa: E402
    IntraChainResidueClassifier,
    Sabdab2StructureBatchCollator,
    sha256_file,
)


MODEL_CONFIG = {
    "antibody_input_dim": 1280,
    "antigen_input_dim": 1280,
    "d_model": 64,
    "num_heads": 4,
    "num_layers": 1,
    "num_rbf": 16,
    "neighbor_k": 32,
    "neighbor_radius": 16.0,
    "sequence_window": 32,
    "dropout": 0.1,
}
MODEL_SPECS = {
    "sequence_dense": {"geometry_feature_mode": "none", "attention_graph_mode": "dense"},
    "spatial_topology": {
        "geometry_feature_mode": "none",
        "attention_graph_mode": "spatial",
    },
}
EXPECTED_PARAMETER_COUNT = 271670


def _seed() -> None:
    random.seed(20260730)
    np.random.seed(20260730)
    torch.manual_seed(20260730)
    torch.cuda.manual_seed_all(20260730)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


@torch.inference_mode()
def _predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, use_bf16: bool
) -> dict[str, np.ndarray]:
    instance_ids: list[str] = []
    data: dict[str, list[np.ndarray]] = {
        "antibody_labels": [],
        "antigen_labels": [],
        "paratope_probabilities": [],
        "epitope_probabilities": [],
    }
    for index, raw in enumerate(loader, start=1):
        batch = _move(raw, device)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )
        with context:
            output = model(batch)
        instance_ids.extend(batch["instance_ids"])
        ab_mask = output["antibody_mask"][0]
        ag_mask = output["antigen_mask"][0]
        data["antibody_labels"].append(
            batch["antibody_labels"][0, ab_mask].detach().cpu().numpy()
        )
        data["antigen_labels"].append(
            batch["antigen_labels"][0, ag_mask].detach().cpu().numpy()
        )
        data["paratope_probabilities"].append(
            output["paratope_logits"][0, ab_mask].float().sigmoid().cpu().numpy()
        )
        data["epitope_probabilities"].append(
            output["epitope_logits"][0, ag_mask].float().sigmoid().cpu().numpy()
        )
        if index % 50 == 0 or index == len(loader):
            print(f"predicted {index}/{len(loader)} complexes", flush=True)
    arrays: dict[str, np.ndarray] = {"instance_ids": np.asarray(instance_ids, dtype=np.str_)}
    for partner, label_key, probability_key in (
        ("antibody", "antibody_labels", "paratope_probabilities"),
        ("antigen", "antigen_labels", "epitope_probabilities"),
    ):
        lengths = np.asarray([len(value) for value in data[label_key]], dtype=np.int64)
        arrays[f"{partner}_offsets"] = np.concatenate(
            (np.zeros(1, dtype=np.int64), lengths.cumsum())
        )
        arrays[label_key] = np.concatenate(data[label_key]).astype(np.float32, copy=False)
        arrays[probability_key] = np.concatenate(data[probability_key]).astype(
            np.float32, copy=False
        )
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise SystemExit("output-root already exists")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if args.precision == "bf16" and device.type != "cuda":
        raise SystemExit("bf16 requires CUDA")
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if device.type == "cuda" and cublas_workspace_config not in {":4096:8", ":16:8"}:
        raise SystemExit(
            "CUDA determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    protocol = verify_frozen_authority(args.protocol, args.freeze_receipt)
    dataset = ChimeraAdapterCacheDataset(
        args.cache_root,
        args.embedding_root,
        expected_manifest_sha256=args.expected_cache_manifest_sha256,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=Sabdab2StructureBatchCollator(),
        pin_memory=device.type == "cuda",
    )
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.tmp-", dir=args.output_root.parent)
    )
    records = []
    reference_labels: tuple[np.ndarray, ...] | None = None
    started = time.monotonic()
    try:
        for candidate in ("sequence_dense", "spatial_topology"):
            specification = MODEL_SPECS[candidate]
            for seed in (17, 42, 73):
                _seed()
                checkpoint_record = protocol["model"]["candidates"][candidate][str(seed)]
                checkpoint_path = Path(checkpoint_record["path"])
                observed_checkpoint_sha = sha256_file(checkpoint_path)
                if observed_checkpoint_sha != checkpoint_record["checkpoint_sha256"]:
                    raise ValueError("checkpoint SHA-256 differs")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if (
                    checkpoint.get("format_version")
                    != "sabdab2_self_attention_factorized_ablation_v2"
                    or checkpoint.get("experiment") != candidate
                    or checkpoint.get("experiment_spec") != specification
                    or int(checkpoint.get("seed", -1)) != seed
                    or int(checkpoint.get("final_epoch", -1)) != 20
                ):
                    raise ValueError("checkpoint metadata differs from the frozen protocol")
                model = IntraChainResidueClassifier(**MODEL_CONFIG, **specification)
                if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
                    raise ValueError("model parameter count differs")
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.to(device).eval()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                run_started = time.monotonic()
                arrays = _predict(
                    model, loader, device, args.precision == "bf16" and device.type == "cuda"
                )
                labels = (
                    arrays["instance_ids"],
                    arrays["antibody_offsets"],
                    arrays["antigen_offsets"],
                    arrays["antibody_labels"],
                    arrays["antigen_labels"],
                )
                if reference_labels is None:
                    reference_labels = tuple(value.copy() for value in labels)
                elif any(not np.array_equal(left, right) for left, right in zip(reference_labels, labels)):
                    raise ValueError("label or instance ordering changed between checkpoints")
                output_path = temporary_root / f"{candidate}_seed{seed}_predictions.npz"
                with output_path.open("wb") as handle:
                    np.savez_compressed(handle, **arrays)
                    handle.flush()
                    os.fsync(handle.fileno())
                records.append(
                    {
                        "candidate": candidate,
                        "seed": seed,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": observed_checkpoint_sha,
                        "prediction_path": output_path.name,
                        "prediction_sha256": sha256_file(output_path),
                        "elapsed_seconds": time.monotonic() - run_started,
                        "cuda_peak_memory_bytes": (
                            int(torch.cuda.max_memory_allocated(device))
                            if device.type == "cuda"
                            else None
                        ),
                    }
                )
                del model, checkpoint, arrays
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        manifest = {
            "schema_version": "chimera_frozen_external_predictions_v1",
            "claim_boundary": (
                "Model scores on a manifest-bound CHIMERA adapter cache. Prediction "
                "does not determine whether confirmatory interpretation is permitted; "
                "that boundary is inherited from the bound adapter audit."
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(args.protocol),
            "freeze_receipt_sha256": sha256_file(args.freeze_receipt),
            "cache_manifest_sha256": args.expected_cache_manifest_sha256,
            "instance_count": len(dataset),
            "batch_size": 1,
            "precision": args.precision,
            "metrics_computed": False,
            "runtime": {
                "device": str(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cublas_workspace_config": cublas_workspace_config,
                "elapsed_seconds": time.monotonic() - started,
            },
            "source_sha256": {
                "predictor": sha256_file(Path(__file__)),
                "model": sha256_file(
                    PROJECT_ROOT / "src" / "interaction_first" / "sabdab2_structure.py"
                ),
                "geometry": sha256_file(
                    PROJECT_ROOT / "src" / "interaction_first" / "intra_chain_geometry.py"
                ),
                "cache_dataset": sha256_file(
                    PROJECT_ROOT / "src" / "interaction_first" / "chimera_cache.py"
                ),
            },
            "predictions": records,
        }
        manifest_path = temporary_root / "PREDICTION_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.chmod(temporary_root, 0o755)
        os.replace(temporary_root, args.output_root)
        print(f"manifest={args.output_root / 'PREDICTION_MANIFEST.json'}")
        print(f"manifest_sha256={sha256_file(args.output_root / 'PREDICTION_MANIFEST.json')}")
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
