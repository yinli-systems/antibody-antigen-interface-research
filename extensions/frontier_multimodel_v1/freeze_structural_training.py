#!/usr/bin/env python3
"""Freeze the complete SaProt/ProSST development-training task grid.

This program reads only metadata, source specifications and filesystem stats.
It never reads contact labels or model performance metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

REAL_LAYERS = {"saprot": (0, 8, 16, 24, 33), "prosst": (0, 3, 6, 9, 12)}
SEEDS = (1, 2, 3)
SOURCE_FILES = (
    "scripts/sabdab2_interaction_first.py",
    "src/interaction_first/composite_embeddings.py",
    "src/interaction_first/pretrained_representations.py",
    "src/interaction_first/runtime.py",
    "src/interaction_first/model.py",
    "src/interaction_first/metrics.py",
    "src/interaction_first/partners.py",
    "src/interaction_first/multitask_balancers.py",
    "extensions/frontier_multimodel_v1/STRUCTURAL_PLM_TRAINING_PROTOCOL.md",
    "extensions/frontier_multimodel_v1/freeze_structural_training.py",
    "extensions/frontier_multimodel_v1/run_structural_training.py",
    "extensions/frontier_multimodel_v1/aggregate_structural_training.py",
    "extensions/frontier_multimodel_v1/structural_training.sbatch",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def inventory(root: Path) -> dict[str, Any]:
    """Fast immutable-tree check over names, sizes and nanosecond mtimes.

    This is explicitly not a content digest. The large historical base stores
    are also constrained by their exact paths and counts and are unchanged from
    the registered historical controls. New structural artifacts are bound by
    the extraction audit's content manifest.
    """
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        count += 1
        total += stat.st_size
    return {
        "path": str(root.resolve()),
        "file_count": count,
        "total_bytes": total,
        "path_size_mtime_inventory_sha256": digest.hexdigest(),
        "content_digest": False,
    }


def discover_specs(spec_root: Path) -> dict[tuple[str, int], list[Path]]:
    groups: dict[tuple[str, int], list[Path]] = {}
    pattern = re.compile(r"^(saprot|prosst)_layer_(\d+)_(antibody|antigen)\.json$")
    for path in sorted(spec_root.glob("*.json")):
        match = pattern.match(path.name)
        if not match:
            raise ValueError(f"unexpected source specification: {path.name}")
        family, layer, _side = match.groups()
        groups.setdefault((family, int(layer)), []).append(path.resolve())
    expected = {(family, layer) for family, layers in REAL_LAYERS.items() for layer in layers}
    if set(groups) != expected:
        raise ValueError(f"source-spec groups differ: got {sorted(groups)}, expected {sorted(expected)}")
    for key, paths in groups.items():
        if len(paths) != 2:
            raise ValueError(f"{key} requires antibody and antigen specs, got {paths}")
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        sides = {tuple(payload["sides"]) for payload in payloads}
        if sides != {("antibody",), ("antigen",)}:
            raise ValueError(f"{key} has invalid sides: {sides}")
        if len({payload["dim"] for payload in payloads}) != 1:
            raise ValueError(f"{key} has inconsistent dimensions")
        if len({payload["model_revision"] for payload in payloads}) != 1:
            raise ValueError(f"{key} has inconsistent revisions")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--contact-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    extraction_root = args.extraction_root.resolve()
    audit_path = extraction_root / "COVERAGE_AUDIT.json"
    extraction_plan_path = extraction_root / "PLAN.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    extraction_plan = json.loads(extraction_plan_path.read_text(encoding="utf-8"))
    if audit.get("feature_artifact_count") != 33450 or audit.get("feature_artifacts_expected") != 33450:
        raise ValueError("structural feature coverage is not complete")
    if audit.get("contact_labels_loaded") is not False:
        raise ValueError("extraction audit reports contact-label access")
    if audit.get("plan_sha256") != sha256(extraction_plan_path):
        raise ValueError("coverage audit does not bind the extraction plan")
    if extraction_plan.get("test_authorized") is not False:
        raise ValueError("extraction plan unexpectedly authorizes test access")

    spec_groups = discover_specs(extraction_root / "output/source_specs")
    input_hashes: dict[str, str] = {}
    for path in (
        audit_path,
        extraction_plan_path,
        args.prepare_dir.resolve() / "instances.parquet",
        args.split_manifest.resolve(),
        args.canonical_manifest.resolve(),
    ):
        input_hashes[str(path)] = sha256(path)
    spec_metadata: dict[str, dict[str, Any]] = {}
    for paths in spec_groups.values():
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            input_hashes[str(path)] = sha256(path)
            spec_metadata[str(path)] = {
                "name": payload["name"],
                "dim": payload["dim"],
                "sides": payload["sides"],
                "model_name": payload["model_name"],
                "model_revision": payload["model_revision"],
                "root": payload["root"],
                "specification_sha256": sha256(path),
            }

    source_hashes = {}
    for relative in SOURCE_FILES:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[relative] = sha256(path)

    tasks = []
    arms = []
    for family, layers in REAL_LAYERS.items():
        for layer in layers:
            arm = f"{family}_layer_{layer:02d}"
            specs = [str(path) for path in sorted(spec_groups[(family, layer)])]
            arms.append({"arm": arm, "family": family, "layer": layer, "kind": "real", "specs": specs})
            for seed in SEEDS:
                tasks.append({
                    "task_id": len(tasks), "arm": arm, "family": family,
                    "layer": layer, "kind": "real", "seed": seed,
                    "specs": specs, "random_projection_control": False,
                })
    for family in ("saprot", "prosst"):
        layer = REAL_LAYERS[family][0]
        arm = f"{family}_random_width_control"
        specs = [str(path) for path in sorted(spec_groups[(family, layer)])]
        arms.append({"arm": arm, "family": family, "layer": None, "kind": "control", "specs": specs})
        for seed in SEEDS:
            tasks.append({
                "task_id": len(tasks), "arm": arm, "family": family,
                "layer": None, "kind": "control", "seed": seed,
                "specs": specs, "random_projection_control": True,
            })
    if len(tasks) != 36:
        raise AssertionError(len(tasks))

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if len(split.get("splits", {}).get("train", [])) != 3229:
        raise ValueError("train size differs from the registered historical authority")
    if len(split.get("splits", {}).get("validation", [])) != 217:
        raise ValueError("validation size differs from the registered historical authority")
    split_sets = {
        name: set(map(str, split.get("splits", {}).get(name, [])))
        for name in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_sets[left] & split_sets[right]
        if overlap:
            raise ValueError(f"split overlap detected for {left}/{right}: {sorted(overlap)[:10]}")

    payload = {
        "format_version": "structural_plm_training_plan_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_role": "development-only SaProt/ProSST layer-sweep training",
        "source_root": str(source_root),
        "source_commit": args.source_commit,
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "generic_source_metadata": spec_metadata,
        "extraction_audit_sha256": sha256(audit_path),
        "feature_artifact_manifest_sha256": audit["artifact_manifest_sha256"],
        "large_base_store_inventory": {
            "contact_dir": inventory(args.contact_dir.resolve()),
            "embedding_dir": inventory(args.embedding_dir.resolve()),
        },
        "paths": {
            "prepare_dir": str(args.prepare_dir.resolve()),
            "contact_dir": str(args.contact_dir.resolve()),
            "embedding_dir": str(args.embedding_dir.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "canonical_manifest": str(args.canonical_manifest.resolve()),
            "python_bin": str(args.python_bin.resolve()),
        },
        "cohort": {"train_complexes": 3229, "validation_complexes": 217},
        "training": {
            "d_model": 128, "pair_dim": 64, "num_heads": 8, "num_layers": 1,
            "dropout": 0.2, "marginal_pooling": "logmeanexp", "score_pooling": "logmeanexp",
            "negative_count": 4, "negative_directions": "both", "hard_pool_size": 64,
            "partner_rank_weight": 0.5, "learning_rate": 5e-5,
            "minimum_learning_rate": 1e-7, "weight_decay": 5e-4,
            "gradient_clip": 1.0, "max_epochs": 12, "patience": 10,
            "scheduler_factor": 0.5, "scheduler_patience": 3,
            "selection_metric": "partner_rank", "pair_decoder": "bilinear",
            "random_projection_seed": 20260930,
        },
        "smoke": {
            "task_id": next(task["task_id"] for task in tasks if task["arm"] == "saprot_layer_16" and task["seed"] == 1),
            "max_train_instances": 48, "max_validation_instances": 24,
            "max_epochs": 1, "scientific_result": False,
        },
        "arms": arms,
        "tasks": tasks,
        "primary_endpoint": "validation macro mean-per-complex pair AP from best.pt",
        "selection_rule": "within family highest three-seed mean primary AP; <=0.001 tie selects shallower layer",
        "test_authorized": False,
        "ood_authorized": False,
        "blind_claim_authorized": False,
        "sota_claim_authorized": False,
        "training_authorized": True,
    }
    canonical_json(args.output.resolve(), payload)
    print(json.dumps({
        "plan": str(args.output.resolve()), "sha256": sha256(args.output.resolve()),
        "task_count": len(tasks), "arm_count": len(arms),
        "source_specs": len(spec_metadata), "feature_artifacts": audit["feature_artifact_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
