#!/usr/bin/env python3
"""Run one real AsEP development batch through all frozen model conditions."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from interaction_first.asep_external import AsepDevelopmentDataset  # noqa: E402
from interaction_first.sabdab2_contacts import Sabdab2ContactBatchCollator  # noqa: E402
from interaction_first.sabdab2_structure import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)
from train_framecontact import EXPERIMENTS, _model_from_configuration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--experiments", nargs="+", choices=tuple(EXPERIMENTS), default=list(EXPERIMENTS)
    )
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite smoke receipt: {arguments.output}")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if arguments.precision == "bf16" and arguments.device != "cuda":
        raise ValueError("BF16 smoke requires CUDA")
    if arguments.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise ValueError("CUDA smoke requires deterministic CUBLAS_WORKSPACE_CONFIG")
    device = torch.device(arguments.device)
    dataset = AsepDevelopmentDataset(
        arguments.cache_root,
        manifest_sha256=arguments.cache_manifest_sha256,
        split_role="train",
        feature_mode="official_esm2_35m",
    )
    example = dataset[0]
    batch = Sabdab2ContactBatchCollator()([example])
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    configuration = {
        "d_model": 64,
        "num_heads": 4,
        "num_self_layers": 1,
        "num_rbf": 8,
        "neighbor_k": 16,
        "neighbor_radius": 16.0,
        "sequence_window": 16,
        "dropout": 0.0,
    }
    results = []
    for experiment in arguments.experiments:
        torch.manual_seed(17)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(17)
            torch.cuda.reset_peak_memory_stats(device)
        model = _model_from_configuration(example, configuration, experiment).to(device)
        model.train()
        started = time.monotonic()
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if arguments.precision == "bf16"
            else nullcontext()
        )
        with context:
            output = model(batch)
            pair_mask = output["pair_candidate_mask"]
            contact_loss = F.binary_cross_entropy_with_logits(
                output["contact_logits"][pair_mask], batch["pair_labels"][pair_mask]
            )
            antibody_mask = output["antibody_mask"]
            antigen_mask = output["antigen_mask"]
            paratope_loss = F.binary_cross_entropy_with_logits(
                output["paratope_logits"][antibody_mask],
                batch["antibody_labels"][antibody_mask],
            )
            epitope_loss = F.binary_cross_entropy_with_logits(
                output["epitope_logits"][antigen_mask],
                batch["antigen_labels"][antigen_mask],
            )
            loss = contact_loss + 0.25 * (paratope_loss + epitope_loss)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{experiment} produced a non-finite loss")
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise RuntimeError(f"{experiment} produced a non-finite gradient")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        results.append(
            {
                "experiment": experiment,
                "loss": float(loss.detach().cpu()),
                "contact_loss": float(contact_loss.detach().cpu()),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else None
                ),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    receipt = {
        "format_version": "framecontact_asep_gpu_smoke_v1",
        "authority": {
            "cache_manifest_sha256": arguments.cache_manifest_sha256,
            "model_source_sha256": sha256_file(
                REPOSITORY_ROOT / "src/interaction_first/framecontact.py"
            ),
            "geometry_source_sha256": sha256_file(
                REPOSITORY_ROOT / "src/interaction_first/intra_chain_geometry.py"
            ),
            "trainer_source_sha256": sha256_file(
                REPOSITORY_ROOT / "scripts/train_framecontact.py"
            ),
        },
        "runtime": {
            "device": str(device),
            "precision": arguments.precision,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "instance_id": example["instance_id"],
        "antibody_length": int(example["pair_labels"].shape[0]),
        "antigen_length": int(example["pair_labels"].shape[1]),
        "results": results,
        "checks": {
            "forward_backward_finite": True,
            "optimizer_step_performed": False,
            "official_test_loaded": False,
            "performance_metric_computed": False,
        },
        "claim_boundary": (
            "One development-batch device forward/backward smoke only. Loss values are "
            "not benchmark metrics and no optimizer step or test access occurred."
        ),
    }
    if any(not math.isfinite(float(result["loss"])) for result in results):
        raise RuntimeError("Smoke receipt contains a non-finite loss")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
