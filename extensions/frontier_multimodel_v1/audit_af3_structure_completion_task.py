#!/usr/bin/env python3
"""Audit one AF3 missing-chain task against the frozen completion plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--project-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    if sha256(plan_path) != args.plan_sha256:
        raise RuntimeError("plan digest mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tasks = {int(task["task_id"]): task for task in plan["tasks"]}
    if args.task_id not in tasks:
        raise ValueError(f"unknown task {args.task_id}")
    task = tasks[args.task_id]
    input_path = Path(task["input_path"])
    if sha256(input_path) != task["input_sha256"]:
        raise RuntimeError("AF3 input changed")

    project_source = args.project_source.resolve()
    if str(project_source) not in sys.path:
        sys.path.insert(0, str(project_source))
    from extensions.frontier_multimodel_v1.materialize_af3_structure_completion import (
        selected_top_sample,
    )
    from extensions.frontier_multimodel_v1.structure_units import extract_protein_chains

    result_dir = Path(task["batch_dir"]) / "results" / task["name"]
    ranking_path = result_dir / "ranking_scores.csv"
    top_model_path = result_dir / f"{task['name']}_model.cif"
    if not ranking_path.is_file() or not top_model_path.is_file():
        raise FileNotFoundError(f"incomplete AF3 result {result_dir}")
    score, seed, sample, source_model = selected_top_sample(
        result_dir, ranking_path, top_model_path
    )
    chains = extract_protein_chains(top_model_path)
    if len(chains) != 1:
        raise ValueError(f"expected one protein chain, got {len(chains)}")
    chain = chains[0]
    if chain.sequence != task["sequence"]:
        raise ValueError("AF3 selected CIF sequence mismatch")
    for residue in chain.residues:
        if set(residue.atoms) != {"N", "CA", "C", "O"}:
            raise ValueError(f"incomplete backbone at residue {residue.name}")
        if not all(math.isfinite(value) for atom in residue.atoms.values() for value in atom):
            raise ValueError("non-finite AF3 coordinates")

    payload = {
        "format_version": "af3_structure_completion_task_audit_v1",
        "status": "PASS_AF3_STRUCTURE_TASK_AUDIT",
        "task_id": args.task_id,
        "name": task["name"],
        "sequence_sha256": task["sequence_sha256"],
        "sequence_length": task["length"],
        "roles": task["roles"],
        "instances": task["instances"],
        "input_path": str(input_path),
        "input_sha256": task["input_sha256"],
        "ranking_scores_path": str(ranking_path),
        "ranking_scores_sha256": sha256(ranking_path),
        "ranking_score": score,
        "selected_seed": seed,
        "selected_sample": sample,
        "selected_sample_model_path": str(source_model),
        "selected_sample_model_sha256": sha256(source_model),
        "top_level_model_path": str(top_model_path),
        "top_level_model_sha256": sha256(top_model_path),
        "top_level_matches_selected_sample": sha256(top_model_path) == sha256(source_model),
        "chain_count": 1,
        "residue_count": len(chain.residues),
        "backbone_atoms_per_residue": ["N", "CA", "C", "O"],
        "contact_labels_read": False,
        "test_labels_read": False,
        "ood_labels_read": False,
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
