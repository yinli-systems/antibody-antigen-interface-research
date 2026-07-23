#!/usr/bin/env python3
"""Run a small, explicitly non-benchmark MIPE training/validation smoke test."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def edge_targets(sample, ag_count, ab_count, device):
    target = torch.zeros((ag_count, ab_count), dtype=torch.float32, device=device)
    indices = sample["edge_AGAB"]
    target[torch.as_tensor(indices[0], dtype=torch.long), torch.as_tensor(indices[1], dtype=torch.long)] = 1.0
    return target


def prepare(sample, device, create_graph):
    ag_target = torch.as_tensor(sample["label_AG"], dtype=torch.float32, device=device)
    ab_target = torch.as_tensor(sample["label_AB"], dtype=torch.float32, device=device)
    ag_sequence = torch.as_tensor(sample["ESM1b_AG"], device=device).t().unsqueeze(0)
    ab_sequence = torch.as_tensor(sample["AbLang_AB"], device=device).t().unsqueeze(0)
    ag_nodes = torch.as_tensor(sample["vertex_AG"], dtype=torch.float32, device=device).unsqueeze(0)
    ab_nodes = torch.as_tensor(sample["vertex_AB"], dtype=torch.float32, device=device).unsqueeze(0)
    ag_graph, ab_graph = create_graph(sample)
    inputs = [ag_nodes, ag_graph, ab_nodes, ab_graph, ag_sequence, ab_sequence, False, ag_target, ab_target, 0]
    return inputs, ag_target, ab_target


def full_loss(outputs, ag_target, ab_target, interaction_target, bce, contrastive, cosine_inter):
    ag_sequence, ag_structure = outputs[4], outputs[8]
    ab_sequence, ab_structure = outputs[6], outputs[9]
    ag_cosine = cosine_inter(ag_sequence, ag_structure)
    ab_cosine = cosine_inter(ab_sequence, ab_structure)
    multimodal = 0.06 * (
        (ag_target * ag_cosine).sum() / (ag_cosine.sum() + 1e-8)
        + (ab_target * ab_cosine).sum() / (ab_cosine.sum() + 1e-8)
        + contrastive(ag_sequence, ag_sequence)
        + contrastive(ag_structure, ag_structure)
        + contrastive(ab_sequence, ab_sequence)
        + contrastive(ab_structure, ab_structure)
        + contrastive(ag_sequence, ag_structure)
        + contrastive(ab_sequence, ab_structure)
    )
    return (
        6 * bce(outputs[0].squeeze(1), ag_target)
        + 6 * bce(outputs[1].squeeze(1), ab_target)
        + multimodal
        + 10 * bce(outputs[2], interaction_target)
        + 10 * bce(outputs[3].t(), interaction_target)
    )


def binary_metrics(prediction, target):
    prediction = prediction.detach().float().cpu()
    target = target.detach().long().cpu()
    from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryMatthewsCorrCoef
    return {
        "auroc": float(BinaryAUROC()(prediction, target)),
        "auprc": float(BinaryAveragePrecision()(prediction, target)),
        "mcc": float(BinaryMatthewsCorrCoef()(prediction, target)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=2)
    parser.add_argument("--val-samples", type=int, default=2)
    args = parser.parse_args()

    started = time.time()
    source_code = args.source_root.resolve() / "code"
    sys.path.insert(0, str(source_code))
    from model import MIPE
    from NTXentLoss import NTXentLoss
    from utils import CreateGearnetGraph, consine_inter

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this smoke test")
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    print(f"device={torch.cuda.get_device_name(device)}", flush=True)
    print(f"loading={args.data}", flush=True)
    dataset = pd.read_pickle(args.data)
    required = {"vertex_AG", "vertex_AB", "label_AG", "label_AB", "ESM1b_AG", "AbLang_AB", "edge_AGAB", "coord_AG", "coord_AB"}
    usable = [sample for sample in dataset if required.issubset(sample.keys())]
    needed = args.train_samples + args.val_samples
    if len(usable) < needed:
        raise RuntimeError(f"Only {len(usable)} usable samples, need {needed}")

    model = MIPE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bce = torch.nn.BCELoss()
    contrastive = NTXentLoss()
    train_losses = []
    model.train()
    for index, sample in enumerate(usable[:args.train_samples]):
        optimizer.zero_grad(set_to_none=True)
        inputs, ag_target, ab_target = prepare(sample, device, CreateGearnetGraph)
        outputs = model(*inputs)
        interaction = edge_targets(sample, len(ag_target), len(ab_target), device)
        loss = full_loss(outputs, ag_target, ab_target, interaction, bce, contrastive, consine_inter)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss for training sample {index}: {loss}")
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))
        print(f"train_sample={index} loss={train_losses[-1]:.6f}", flush=True)

    ag_predictions, ag_targets, ab_predictions, ab_targets = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for index, sample in enumerate(usable[args.train_samples:needed]):
            inputs, ag_target, ab_target = prepare(sample, device, CreateGearnetGraph)
            outputs = model(*inputs)
            ag_predictions.append(outputs[0].flatten())
            ab_predictions.append(outputs[1].flatten())
            ag_targets.append(ag_target)
            ab_targets.append(ab_target)
            print(f"val_sample={index} ag_residues={len(ag_target)} ab_residues={len(ab_target)}", flush=True)

    ag_prediction = torch.cat(ag_predictions)
    ab_prediction = torch.cat(ab_predictions)
    ag_target = torch.cat(ag_targets)
    ab_target = torch.cat(ab_targets)
    metrics = {"antigen": binary_metrics(ag_prediction, ag_target), "antibody": binary_metrics(ab_prediction, ab_target)}
    torch.cuda.synchronize()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "mipe_smoke_checkpoint.pt"
    report_path = args.output_dir / "mipe_smoke_report.json"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "smoke_test": True,
        "seed": 42,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "train_losses": train_losses,
        "metrics": metrics,
        "source_commit": "a77ece1b39e2b91166387080bfcb92b8ab5d6621",
    }
    torch.save(checkpoint, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not reloaded.get("smoke_test") or not reloaded.get("model_state_dict"):
        raise RuntimeError("Saved checkpoint failed reload validation")

    report = {
        "status": "complete",
        "scope": "smoke_test_only_not_comparable_to_paper",
        "device": torch.cuda.get_device_name(0),
        "dataset_records": len(dataset),
        "usable_records": len(usable),
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "train_losses": train_losses,
        "validation_metrics": metrics,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_reload_validated": True,
        "known_upstream_observation": "CreateGearnetGraph builds no sequential antibody edges because it appends them to edge_AG_seq; smoke test preserves upstream behavior.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
