#!/usr/bin/env python3
"""Materialise all selected AF3 missing-chain structures and merge the authority."""
from __future__ import annotations

import argparse
import csv
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


def ranking_rows(path: Path) -> list[tuple[float, int, int]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"seed", "sample", "ranking_score"}:
            raise ValueError(f"unexpected ranking columns in {path}")
        for row in reader:
            score = float(row["ranking_score"])
            if not math.isfinite(score):
                raise ValueError(f"non-finite ranking score in {path}")
            rows.append((score, int(row["seed"]), int(row["sample"])))
    if len(rows) != 25:
        raise ValueError(f"expected 25 AF3 samples in {path}, got {len(rows)}")
    return rows


def selected_top_sample(result_dir: Path, ranking_path: Path, top_model_path: Path) -> tuple[float, int, int, Path]:
    rows = ranking_rows(ranking_path)
    maximum = max(score for score, _seed, _sample in rows)
    top_digest = sha256(top_model_path)
    matches = []
    for score, seed, sample in rows:
        if score != maximum:
            continue
        source = result_dir / f"seed-{seed}_sample-{sample}" / "model.cif"
        if not source.is_file():
            raise FileNotFoundError(source)
        if sha256(source) == top_digest:
            matches.append((score, seed, sample, source))
    if not matches:
        raise ValueError(
            f"top-level AF3 model does not match any maximum-score sample: {top_model_path}"
        )
    return min(matches, key=lambda item: (item[1], item[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--base-structure-manifest", type=Path, required=True)
    parser.add_argument("--project-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    if sha256(plan_path) != args.plan_sha256:
        raise RuntimeError("plan digest mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    base_path = args.base_structure_manifest.resolve()
    if sha256(base_path) != plan["base_structure_manifest_sha256"]:
        raise RuntimeError("base structure manifest changed")
    base = json.loads(base_path.read_text(encoding="utf-8"))

    source_root = args.project_source.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from extensions.frontier_multimodel_v1.structure_units import (
        extract_protein_chains,
        write_pdb,
    )

    output_root = args.output_root.resolve()
    units = output_root / "units"
    units.mkdir(parents=True, exist_ok=False)
    extension_records = []
    for task in plan["tasks"]:
        batch = Path(task["batch_dir"])
        result_dir = batch / "results" / task["name"]
        ranking_path = result_dir / "ranking_scores.csv"
        model_path = result_dir / f"{task['name']}_model.cif"
        if not ranking_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"incomplete AF3 task {task['task_id']}: {result_dir}")
        top_score, top_seed, top_sample, top_source_path = selected_top_sample(
            result_dir, ranking_path, model_path
        )
        chains = extract_protein_chains(model_path)
        if len(chains) != 1:
            raise ValueError(f"expected one chain in {model_path}, got {len(chains)}")
        chain = chains[0]
        if chain.sequence != task["sequence"]:
            raise ValueError(f"selected CIF sequence mismatch task {task['task_id']}")
        if any(not all(math.isfinite(value) for atom in residue.atoms.values() for value in atom) for residue in chain.residues):
            raise ValueError(f"non-finite coordinates task {task['task_id']}")
        pdb_path = units / task["sequence_sha256"][:2] / f"{task['sequence_sha256']}.pdb"
        write_pdb(pdb_path, chain)
        extension_records.append({
            "sequence_sha256": task["sequence_sha256"],
            "sequence": task["sequence"],
            "side": "antigen" if task["roles"] == ["antigen"] else "antibody",
            "roles": task["roles"],
            "source": "af3_completion_v1",
            "chosen": {
                "source": "af3_completion_v1",
                "source_key": task["name"],
                "score": top_score,
                "seed": top_seed,
                "sample": top_sample,
                "model_path": str(model_path),
                "model_sha256": sha256(model_path),
                "selected_sample_model_path": str(top_source_path),
                "selected_sample_model_sha256": sha256(top_source_path),
                "ranking_scores_path": str(ranking_path),
                "ranking_scores_sha256": sha256(ranking_path),
                "input_path": task["input_path"],
                "input_sha256": task["input_sha256"],
            },
            "pdb_path": str(pdb_path),
            "pdb_sha256": sha256(pdb_path),
            "instances": task["instances"],
        })

    if len(extension_records) != 40:
        raise AssertionError(len(extension_records))
    existing = {record["sequence_sha256"] for record in base["records"]}
    if existing & {record["sequence_sha256"] for record in extension_records}:
        raise ValueError("extension duplicates base structure sequence")

    combined_records = []
    for record in base["records"]:
        copied = json.loads(json.dumps(record))
        pdb = Path(copied["pdb_path"])
        if not pdb.is_absolute():
            copied["pdb_path"] = str((base_path.parent / pdb).resolve())
        combined_records.append(copied)
    combined_records.extend(extension_records)
    sides: dict[str, int] = {}
    for record in combined_records:
        sides[record["side"]] = sides.get(record["side"], 0) + 1
    combined = {
        "format_version": "frontier_structure_units_v2",
        "authority_generation": "base_v2_plus_af3_completion_v1",
        "base_manifest": str(base_path),
        "base_manifest_sha256": sha256(base_path),
        "extension_plan": str(plan_path),
        "extension_plan_sha256": args.plan_sha256,
        "selection_policy": base["selection_policy"],
        "coordinate_policy": base["coordinate_policy"],
        "counts_by_side": sides,
        "total_units": len(combined_records),
        "base_unit_count": len(base["records"]),
        "extension_unit_count": len(extension_records),
        "contact_labels_loaded": False,
        "test_loaded": False,
        "records": sorted(combined_records, key=lambda record: record["sequence_sha256"]),
    }
    write_json(output_root / "EXTENSION_MANIFEST.json", {
        "format_version": "frontier_structure_units_extension_v1",
        "plan_sha256": args.plan_sha256,
        "record_count": len(extension_records),
        "records": sorted(extension_records, key=lambda record: record["sequence_sha256"]),
        "contact_labels_loaded": False,
        "test_loaded": False,
    })
    write_json(output_root / "STRUCTURE_UNITS_MANIFEST.json", combined)
    print(json.dumps({
        "status": "PASS_COMPLETE_STRUCTURE_AUTHORITY",
        "base_units": len(base["records"]),
        "extension_units": len(extension_records),
        "combined_units": len(combined_records),
        "combined_manifest": str(output_root / "STRUCTURE_UNITS_MANIFEST.json"),
        "combined_manifest_sha256": sha256(output_root / "STRUCTURE_UNITS_MANIFEST.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
