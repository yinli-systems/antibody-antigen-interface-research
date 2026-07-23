#!/usr/bin/env python3
"""Independently reload and validate the final EpiFormer graph dataset."""

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    errors = []
    complex_ids = []
    total_ag_residues = 0
    total_ab_residues = 0

    for index, graph in enumerate(dataset):
        complex_id = str(graph.complex_id)
        complex_ids.append(complex_id)
        for node_type in ("ag_res", "ab_res"):
            node = graph[node_type]
            row_counts = {
                "x": int(node.x.shape[0]),
                "plm": int(node.plm.shape[0]),
                "y": int(node.y.numel()),
                "pos": int(node.pos.shape[0]),
            }
            if len(set(row_counts.values())) != 1:
                errors.append(
                    {
                        "complex_id": complex_id,
                        "node_type": node_type,
                        "row_counts": row_counts,
                    }
                )
            if not torch.isfinite(node.x).all() or not torch.isfinite(node.plm).all():
                errors.append(
                    {"complex_id": complex_id, "node_type": node_type, "nonfinite": True}
                )
        total_ag_residues += int(graph["ag_res"].x.shape[0])
        total_ab_residues += int(graph["ab_res"].x.shape[0])
        if (index + 1) % 250 == 0:
            print(f"validated={index + 1}/{len(dataset)}", flush=True)

    shard_ids = {path.stem for path in args.shard_dir.glob("*.pt")}
    id_set = set(complex_ids)
    report = {
        "status": "complete" if not errors else "failed",
        "dataset_path": str(args.dataset),
        "dataset_bytes": args.dataset.stat().st_size,
        "complexes": len(dataset),
        "unique_complex_ids": len(id_set),
        "graph_shards": len(shard_ids),
        "dataset_ids_equal_shard_ids": id_set == shard_ids,
        "total_antigen_residues": total_ag_residues,
        "total_antibody_residues": total_ab_residues,
        "shape_or_finiteness_errors": errors,
        "elapsed_seconds": time.time() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    if (
        errors
        or len(dataset) != 1721
        or len(id_set) != len(dataset)
        or id_set != shard_ids
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
