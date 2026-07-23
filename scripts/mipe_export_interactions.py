#!/usr/bin/env python3
"""Export MIPE residue-pair interaction maps on every held-out fold."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def prepare(sample, device, create_graph, sample_index):
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
    return [
        ag_nodes,
        ag_graph,
        ab_nodes,
        ab_graph,
        ag_sequence,
        ab_sequence,
        True,
        ag_target,
        ab_target,
        sample_index,
    ]


def interaction_target(sample, ag_count, ab_count):
    target = np.zeros((ag_count, ab_count), dtype=np.uint8)
    indices = sample["edge_AGAB"]
    target[np.asarray(indices[0], dtype=np.int64), np.asarray(indices[1], dtype=np.int64)] = 1
    return target


def ranking_metrics(scores, targets):
    from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

    prediction = torch.from_numpy(np.concatenate(scores)).float()
    target = torch.from_numpy(np.concatenate(targets)).long()
    return {
        "auroc": float(BinaryAUROC()(prediction, target)),
        "auprc": float(BinaryAveragePrecision()(prediction, target)),
        "pairs": int(target.numel()),
        "positive_pairs": int(target.sum()),
        "positive_fraction": float(target.float().mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    sys.path.insert(0, str(args.source_root.resolve() / "code"))
    from model import MIPE
    from utils import CreateGearnetGraph, get_k_fold_data

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = pd.read_pickle(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_scores = []
    all_ag_to_ab_scores = []
    all_ab_to_ag_scores = []
    all_targets = []
    fold_summaries = []

    for fold in range(5):
        _, _, test_data = get_k_fold_data(5, fold, dataset)
        checkpoint_path = args.checkpoint_dir / f"fold_{fold}_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = MIPE().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        fold_dir = args.output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_scores = []
        fold_ag_to_ab_scores = []
        fold_ab_to_ag_scores = []
        fold_targets = []
        instances = []

        with torch.inference_mode():
            for index, sample in enumerate(test_data):
                outputs = model(*prepare(sample, device, CreateGearnetGraph, index))
                ag_to_ab = outputs[2].detach().float().cpu().numpy()
                ab_to_ag = outputs[3].detach().float().cpu().numpy()
                if ag_to_ab.shape != ab_to_ag.T.shape:
                    raise ValueError(
                        f"directional map mismatch: {ag_to_ab.shape} vs {ab_to_ag.shape}"
                    )
                mean_score = (ag_to_ab + ab_to_ag.T) / 2.0
                target = interaction_target(sample, *mean_score.shape)
                complex_code = str(sample.get("complex_code", f"sample_{index:04d}"))
                filename = f"{index:04d}_{complex_code}.npz"
                np.savez_compressed(
                    fold_dir / filename,
                    complex_code=np.asarray(complex_code),
                    ag_to_ab=ag_to_ab.astype(np.float32),
                    ab_to_ag=ab_to_ag.astype(np.float32),
                    interaction_score_mean=mean_score.astype(np.float32),
                    interaction_target=target,
                    antigen_score=outputs[0].detach().flatten().float().cpu().numpy(),
                    antibody_score=outputs[1].detach().flatten().float().cpu().numpy(),
                    antigen_target=np.asarray(sample["label_AG"], dtype=np.uint8),
                    antibody_target=np.asarray(sample["label_AB"], dtype=np.uint8),
                )
                flat_score = mean_score.reshape(-1)
                flat_target = target.reshape(-1)
                fold_scores.append(flat_score)
                fold_ag_to_ab_scores.append(ag_to_ab.reshape(-1))
                fold_ab_to_ag_scores.append(ab_to_ag.T.reshape(-1))
                fold_targets.append(flat_target)
                instances.append(
                    {
                        "complex_code": complex_code,
                        "file": filename,
                        "antigen_residues": int(mean_score.shape[0]),
                        "antibody_residues": int(mean_score.shape[1]),
                        "candidate_pairs": int(mean_score.size),
                        "positive_pairs": int(target.sum()),
                    }
                )
                if (index + 1) % 20 == 0 or index + 1 == len(test_data):
                    print(f"fold={fold} exported={index + 1}/{len(test_data)}", flush=True)

        metrics = ranking_metrics(fold_scores, fold_targets)
        fold_summary = {
            "fold": fold,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "best_validation_joint_auprc": float(checkpoint["best_val_joint_auprc"]),
            "interaction_score": "mean(ag_to_ab, transpose(ab_to_ag))",
            "interaction_metrics": {
                "mean_bidirectional": metrics,
                "ag_to_ab": ranking_metrics(fold_ag_to_ab_scores, fold_targets),
                "ab_to_ag_transposed": ranking_metrics(fold_ab_to_ag_scores, fold_targets),
            },
            "instances": instances,
        }
        (fold_dir / "manifest.json").write_text(
            json.dumps(fold_summary, indent=2), encoding="utf-8"
        )
        fold_summaries.append({k: v for k, v in fold_summary.items() if k != "instances"})
        all_scores.extend(fold_scores)
        all_ag_to_ab_scores.extend(fold_ag_to_ab_scores)
        all_ab_to_ag_scores.extend(fold_ab_to_ag_scores)
        all_targets.extend(fold_targets)
        del model, checkpoint
        torch.cuda.empty_cache()

    summary = {
        "status": "complete",
        "scope": "five_fold_held_out_interaction_export",
        "matrix_axes": {
            "ag_to_ab": "[antigen residue, antibody residue]",
            "ab_to_ag": "[antibody residue, antigen residue]",
            "interaction_score_mean": "[antigen residue, antibody residue]",
        },
        "notes": [
            "Both directional matrices are 16-head-averaged cross-attention weights.",
            "Both directions were explicitly supervised against edge_AGAB with BCE during MIPE training.",
            "Ranking metrics use every residue pair in each held-out fold.",
        ],
        "folds": fold_summaries,
        "overall_interaction_metrics": {
            "mean_bidirectional": ranking_metrics(all_scores, all_targets),
            "ag_to_ab": ranking_metrics(all_ag_to_ab_scores, all_targets),
            "ab_to_ag_transposed": ranking_metrics(all_ab_to_ag_scores, all_targets),
        },
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# MIPE held-out interaction matrices\n\n"
        "Each NPZ contains the two native directional cross-attention maps, their "
        "mean score, the binary edge_AGAB target, and residue-level predictions. "
        "Rows of interaction_score_mean are antigen residues and columns are antibody "
        "residues. See summary.json and each fold manifest for metrics and provenance.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
