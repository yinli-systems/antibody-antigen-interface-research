#!/usr/bin/env python3
"""Recoverable MIPE fold training using the repository's joint objective."""

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
    target[
        torch.as_tensor(indices[0], dtype=torch.long),
        torch.as_tensor(indices[1], dtype=torch.long),
    ] = 1.0
    return target


def prepare(sample, device, create_graph, sample_index, inference=False):
    ag_target = torch.as_tensor(sample["label_AG"], dtype=torch.float32, device=device)
    ab_target = torch.as_tensor(sample["label_AB"], dtype=torch.float32, device=device)
    ag_sequence = torch.as_tensor(sample["ESM1b_AG"], device=device).t().unsqueeze(0)
    ab_sequence = torch.as_tensor(sample["AbLang_AB"], device=device).t().unsqueeze(0)
    ag_nodes = torch.as_tensor(
        sample["vertex_AG"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    ab_nodes = torch.as_tensor(
        sample["vertex_AB"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    ag_graph, ab_graph = create_graph(sample)
    inputs = [
        ag_nodes,
        ag_graph,
        ab_nodes,
        ab_graph,
        ag_sequence,
        ab_sequence,
        inference,
        ag_target,
        ab_target,
        sample_index,
    ]
    return inputs, ag_target, ab_target


def joint_loss(outputs, ag_target, ab_target, interaction_target, bce, ntxent, cosine):
    ag_sequence, ag_structure = outputs[4], outputs[8]
    ab_sequence, ab_structure = outputs[6], outputs[9]
    ag_cosine = cosine(ag_sequence, ag_structure)
    ab_cosine = cosine(ab_sequence, ab_structure)
    multimodal = 0.06 * (
        (ag_target * ag_cosine).sum() / (ag_cosine.sum() + 1e-8)
        + (ab_target * ab_cosine).sum() / (ab_cosine.sum() + 1e-8)
        + ntxent(ag_sequence, ag_sequence)
        + ntxent(ag_structure, ag_structure)
        + ntxent(ab_sequence, ab_sequence)
        + ntxent(ab_structure, ab_structure)
        + ntxent(ag_sequence, ag_structure)
        + ntxent(ab_sequence, ab_structure)
    )
    return (
        6 * bce(outputs[0].squeeze(1), ag_target)
        + 6 * bce(outputs[1].squeeze(1), ab_target)
        + multimodal
        + 10 * bce(outputs[2], interaction_target)
        + 10 * bce(outputs[3].t(), interaction_target)
    )


def binary_metrics(prediction, target):
    from torchmetrics.classification import (
        BinaryAUROC,
        BinaryAveragePrecision,
        BinaryMatthewsCorrCoef,
    )

    prediction = torch.cat(prediction).float()
    target = torch.cat(target).long()
    return {
        "auroc": float(BinaryAUROC()(prediction, target)),
        "auprc": float(BinaryAveragePrecision()(prediction, target)),
        "mcc": float(BinaryMatthewsCorrCoef()(prediction, target)),
        "residues": int(target.numel()),
        "positives": int(target.sum()),
    }


def run_evaluation(model, samples, device, create_graph, bce, ntxent, cosine, label):
    losses = []
    ag_predictions, ag_targets = [], []
    ab_predictions, ab_targets = [], []
    model.eval()
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            inputs, ag_target, ab_target = prepare(
                sample, device, create_graph, index, inference=False
            )
            outputs = model(*inputs)
            interaction = edge_targets(
                sample, len(ag_target), len(ab_target), device
            )
            loss = joint_loss(
                outputs, ag_target, ab_target, interaction, bce, ntxent, cosine
            )
            losses.append(float(loss.cpu()))
            ag_predictions.append(outputs[0].flatten().float().cpu())
            ab_predictions.append(outputs[1].flatten().float().cpu())
            ag_targets.append(ag_target.long().cpu())
            ab_targets.append(ab_target.long().cpu())
            if (index + 1) % 25 == 0 or index + 1 == len(samples):
                print(f"{label}_samples={index + 1}/{len(samples)}", flush=True)
    return {
        "loss": float(np.mean(losses)),
        "antigen": binary_metrics(ag_predictions, ag_targets),
        "antibody": binary_metrics(ab_predictions, ab_targets),
    }


def atomic_save(payload, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    sys.path.insert(0, str(args.source_root.resolve() / "code"))
    from model import MIPE
    from NTXentLoss import NTXentLoss
    from utils import CreateGearnetGraph, consine_inter, get_k_fold_data

    if not 0 <= args.fold < 5:
        raise ValueError("fold must be in [0, 4]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"loading={args.data}", flush=True)
    dataset = pd.read_pickle(args.data)
    train_data, val_data, test_data = get_k_fold_data(5, args.fold, dataset)
    print(
        f"fold={args.fold} train={len(train_data)} val={len(val_data)} "
        f"test={len(test_data)} target_epochs={args.epochs}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / f"fold_{args.fold}_last.pt"
    best_path = args.output_dir / f"fold_{args.fold}_best.pt"
    history_path = args.output_dir / f"fold_{args.fold}_history.json"
    report_path = args.output_dir / f"fold_{args.fold}_report.json"

    model = MIPE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_epoch = 1
    history = []
    best_score = float("-inf")
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = checkpoint.get("history", [])
        best_score = float(checkpoint.get("best_val_joint_auprc", best_score))
        print(f"resuming_from_epoch={start_epoch - 1}", flush=True)

    bce = torch.nn.BCELoss()
    ntxent = NTXentLoss()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.time()
        train_losses = []
        ag_predictions, ag_targets = [], []
        ab_predictions, ab_targets = [], []
        optimizer.zero_grad(set_to_none=True)
        model.train()
        for index, sample in enumerate(train_data):
            inputs, ag_target, ab_target = prepare(
                sample, device, CreateGearnetGraph, index, inference=False
            )
            outputs = model(*inputs)
            interaction = edge_targets(
                sample, len(ag_target), len(ab_target), device
            )
            loss = joint_loss(
                outputs, ag_target, ab_target, interaction, bce, ntxent, consine_inter
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, sample {index}: {loss}"
                )
            loss.backward()
            if (index + 1) % args.accumulation == 0 or index + 1 == len(train_data):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach().cpu()))
            ag_predictions.append(outputs[0].detach().flatten().float().cpu())
            ab_predictions.append(outputs[1].detach().flatten().float().cpu())
            ag_targets.append(ag_target.detach().long().cpu())
            ab_targets.append(ab_target.detach().long().cpu())
            if (index + 1) % 25 == 0 or index + 1 == len(train_data):
                print(
                    f"epoch={epoch} train_samples={index + 1}/{len(train_data)} "
                    f"loss={train_losses[-1]:.6f}",
                    flush=True,
                )

        train_result = {
            "loss": float(np.mean(train_losses)),
            "antigen": binary_metrics(ag_predictions, ag_targets),
            "antibody": binary_metrics(ab_predictions, ab_targets),
        }
        val_result = run_evaluation(
            model,
            val_data,
            device,
            CreateGearnetGraph,
            bce,
            ntxent,
            consine_inter,
            "val",
        )
        joint_auprc = (
            val_result["antigen"]["auprc"]
            + val_result["antibody"]["auprc"]
        ) / 2
        epoch_record = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - epoch_started,
            "train": train_result,
            "validation": val_result,
            "validation_joint_auprc": joint_auprc,
        }
        history.append(epoch_record)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "fold": args.fold,
            "seed": args.seed,
            "history": history,
            "best_val_joint_auprc": max(best_score, joint_auprc),
            "source_commit": "a77ece1b39e2b91166387080bfcb92b8ab5d6621",
        }
        atomic_save(checkpoint, last_path)
        if joint_auprc > best_score:
            best_score = joint_auprc
            atomic_save(checkpoint, best_path)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(epoch_record, indent=2), flush=True)

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_result = run_evaluation(
        model,
        test_data,
        device,
        CreateGearnetGraph,
        bce,
        ntxent,
        consine_inter,
        "test",
    )
    torch.cuda.synchronize()
    report = {
        "status": "complete",
        "scope": "single_fold_recoverable_training",
        "fold": args.fold,
        "epochs_completed": int(best_checkpoint["epoch"] if start_epoch > args.epochs else args.epochs),
        "target_epochs": args.epochs,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "test_samples": len(test_data),
        "best_validation_joint_auprc": best_score,
        "test": test_result,
        "elapsed_seconds_this_invocation": time.time() - started,
        "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path),
        "known_upstream_observation": "CreateGearnetGraph antibody sequential-edge bug is preserved for repository-comparable behavior.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
