#!/usr/bin/env python3
"""Stress one exact three-way SAbDab2 training loss without an optimizer step."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from interaction_first.framecontact_objective import (  # noqa: E402
    FrameContactLossWeights,
    FrameContactObjective,
)
from interaction_first.sabdab2_contacts import Sabdab2ContactDataset  # noqa: E402
from interaction_first.sabdab2_structure import (  # noqa: E402
    Sabdab2StructureEmbeddingDataset,
    canonical_json_bytes,
    sha256_file,
)
from train_framecontact import (  # noqa: E402
    _FrozenPanelTrainingCollator,
    _FrozenPanelTrainingDataset,
    _checkpointed_wrong_training_view,
    _cognate_training_view,
    _model_from_configuration,
    _move_batch,
    _saved_tensor_context,
    _wrong_training_view,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--training-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", default="frozen_frame_ipa")
    parser.add_argument("--offload-saved-tensors", action="store_true")
    parser.add_argument("--specificity-negative-count", type=int, default=1)
    parser.add_argument(
        "--specificity-loss-mode",
        choices=("pairwise_softplus", "listwise_contrastive"),
        default="pairwise_softplus",
    )
    parser.add_argument("--specificity-temperature", type=float, default=1.0)
    parser.add_argument(
        "--specificity-direction-aggregation",
        choices=("mean", "smooth_max"),
        default="mean",
    )
    parser.add_argument(
        "--specificity-direction-temperature", type=float, default=0.1
    )
    parser.add_argument("--checkpoint-wrong-partners", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError("Refusing to overwrite stress receipt")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("Deterministic CUDA workspace is required")
    device = torch.device("cuda")
    base = Sabdab2StructureEmbeddingDataset(
        arguments.structure_root,
        arguments.embedding_root,
        None,
        split_role="train",
        split_path=arguments.split_manifest,
        split_sha256=arguments.split_manifest_sha256,
    )
    dataset = Sabdab2ContactDataset(base, arguments.contact_root)
    panel = json.loads(arguments.training_panel.read_text(encoding="utf-8"))
    paired = _FrozenPanelTrainingDataset(
        dataset,
        panel,
        seed=17,
        negative_count=arguments.specificity_negative_count,
    )
    paired.set_epoch(1)
    index = max(
        range(len(dataset)),
        key=lambda value: int(dataset.records[value]["antigen_length"]),
    )
    examples = _FrozenPanelTrainingCollator()([paired[index]])
    batch = _move_batch(examples[0], device)
    wrong_batches = {
        direction: (
            tuple(_move_batch(item, device) for item in value)
            if isinstance(value, (tuple, list))
            else _move_batch(value, device)
        )
        for direction, value in examples[1].items()
    }
    first = dataset[index]
    configuration = {
        "d_model": 128,
        "num_heads": 8,
        "num_self_layers": 2,
        "num_rbf": 16,
        "neighbor_k": 32,
        "neighbor_radius": 16.0,
        "sequence_window": 32,
        "dropout": 0.1,
    }
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.cuda.reset_peak_memory_stats(device)
    model = _model_from_configuration(first, configuration, arguments.experiment).to(device)
    objective = FrameContactObjective(
        weights=FrameContactLossWeights(
            contact=1.0, paratope=0.25, epitope=0.25, specificity=0.25
        ),
        specificity_margin=0.05,
        score_fraction=0.01,
        specificity_score_source="interaction_only",
        specificity_loss_mode=arguments.specificity_loss_mode,
        specificity_temperature=arguments.specificity_temperature,
        specificity_direction_aggregation=(
            arguments.specificity_direction_aggregation
        ),
        specificity_direction_temperature=(
            arguments.specificity_direction_temperature
        ),
    )
    started = time.monotonic()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), _saved_tensor_context(
        arguments.offload_saved_tensors
    ):
        output = _cognate_training_view(model(batch))
        wrong_outputs = {
            direction: (
                tuple(
                    (
                        _checkpointed_wrong_training_view(model, item)
                        if arguments.checkpoint_wrong_partners
                        else _wrong_training_view(model(item))
                    )
                    for item in wrong_batch
                )
                if isinstance(wrong_batch, tuple)
                else (
                    _checkpointed_wrong_training_view(model, wrong_batch)
                    if arguments.checkpoint_wrong_partners
                    else _wrong_training_view(model(wrong_batch))
                )
            )
            for direction, wrong_batch in wrong_batches.items()
        }
        losses = objective(output, batch, wrong_partner_outputs=wrong_outputs)
    losses["loss"].backward()
    torch.cuda.synchronize(device)
    receipt = {
        "format_version": "framecontact_sabdab2_multi_negative_stress_v2",
        "experiment": arguments.experiment,
        "instance_id": first["instance_id"],
        "antigen_length": int(first["antigen_features"].shape[0]),
        "antibody_length": int(first["antibody_features"].shape[0]),
        "wrong_directions": sorted(wrong_batches),
        "specificity_negative_count": arguments.specificity_negative_count,
        "specificity_loss_mode": arguments.specificity_loss_mode,
        "specificity_temperature": arguments.specificity_temperature,
        "specificity_direction_aggregation": (
            arguments.specificity_direction_aggregation
        ),
        "specificity_direction_temperature": (
            arguments.specificity_direction_temperature
        ),
        "offload_saved_tensors": arguments.offload_saved_tensors,
        "checkpoint_wrong_partners": arguments.checkpoint_wrong_partners,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.monotonic() - started,
        "runtime": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "authority": {
            "split_manifest_sha256": arguments.split_manifest_sha256,
            "training_panel_sha256": sha256_file(arguments.training_panel),
            "trainer_source_sha256": sha256_file(
                REPOSITORY_ROOT / "scripts/train_framecontact.py"
            ),
            "objective_source_sha256": sha256_file(
                REPOSITORY_ROOT
                / "src/interaction_first/framecontact_objective.py"
            ),
            "model_source_sha256": sha256_file(
                REPOSITORY_ROOT / "src/interaction_first/framecontact.py"
            ),
        },
        "checks": {
            "forward_backward_finite": bool(torch.isfinite(losses["loss"])),
            "optimizer_constructed": False,
            "optimizer_step_performed": False,
            "validation_loaded": False,
            "test_loaded": False,
            "performance_metric_computed": False,
        },
        "claim_boundary": (
            "Largest-train-complex three-way loss memory stress only. No optimizer, "
            "validation, test, prediction report, or benchmark metric is produced."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
