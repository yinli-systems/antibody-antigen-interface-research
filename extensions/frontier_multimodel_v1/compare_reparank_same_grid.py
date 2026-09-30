#!/usr/bin/env python3
"""Compare RePaRank with existing baselines after exact pair-grid validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PRIMARY_SEEDS = tuple(range(1, 11))
ESM2_SENSITIVITY_SEEDS = tuple(range(11, 21))
BASELINE_ARMS = ("esm2", "esm_if1_only", "proteinmpnn_aligned", "antiberty_only")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_key_digest(
    antigen_index: np.ndarray, antibody_index: np.ndarray, label: np.ndarray
) -> str:
    if not (len(antigen_index) == len(antibody_index) == len(label)):
        raise ValueError("pair-key arrays differ in length")
    order = np.lexsort((antibody_index, antigen_index))
    table = np.empty(
        len(order),
        dtype=[("ag", "<i8"), ("ab", "<i8"), ("label", "u1")],
    )
    table["ag"] = antigen_index[order].astype(np.int64)
    table["ab"] = antibody_index[order].astype(np.int64)
    table["label"] = label[order].astype(np.uint8)
    return hashlib.sha256(table.tobytes()).hexdigest()


def reparank_rows(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    evaluation = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    if evaluation.get("instance_count") != 217:
        raise ValueError("RePaRank result is not the complete 217-complex evaluation")
    if evaluation.get("test_labels_read") is not False:
        raise ValueError("RePaRank result reports test-label access")
    rows: dict[str, dict[str, Any]] = {}
    aps: dict[str, float] = {}
    for path in sorted((root / "pair_predictions").glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        instance = str(data["instance_id"].item())
        if instance in rows:
            raise ValueError(f"duplicate RePaRank instance {instance}")
        label = data["label"].astype(np.uint8)
        score = data["score"].astype(np.float64)
        if not label.any() or label.all():
            raise ValueError(f"undefined binary problem for {instance}")
        rows[instance] = {
            "digest": pair_key_digest(
                data["antigen_residue_index"],
                data["antibody_residue_index"],
                label,
            ),
            "pair_count": int(len(label)),
            "positive_count": int(label.sum()),
        }
        aps[instance] = float(average_precision_score(label, score))
    if len(rows) != 217:
        raise ValueError(f"expected 217 RePaRank raw files, got {len(rows)}")
    return rows, aps


def baseline_per_complex(
    path: Path, expected: dict[str, dict[str, Any]]
) -> dict[str, float]:
    frame = pd.read_parquet(
        path,
        columns=["instance_id", "ag_res_idx", "ab_res_idx", "label", "score"],
    )
    output: dict[str, float] = {}
    if frame["instance_id"].nunique() != len(expected):
        raise ValueError(f"baseline instance count mismatch in {path}")
    for instance, group in frame.groupby("instance_id", sort=True):
        instance = str(instance)
        if instance not in expected:
            raise ValueError(f"unexpected baseline instance {instance}")
        label = group["label"].to_numpy(dtype=np.uint8)
        digest = pair_key_digest(
            group["ag_res_idx"].to_numpy(dtype=np.int64),
            group["ab_res_idx"].to_numpy(dtype=np.int64),
            label,
        )
        authority = expected[instance]
        if digest != authority["digest"]:
            raise ValueError(f"candidate/label digest mismatch {path}: {instance}")
        if len(group) != authority["pair_count"] or int(label.sum()) != authority["positive_count"]:
            raise ValueError(f"candidate/positive count mismatch {path}: {instance}")
        output[instance] = float(
            average_precision_score(label, group["score"].to_numpy(dtype=np.float64))
        )
    if set(output) != set(expected):
        raise ValueError(f"baseline coverage mismatch in {path}")
    return output


def crossed_bootstrap(
    differences: dict[int, dict[str, float]],
    component_by_instance: dict[str, str],
    *,
    seed: int = 20260930,
    repeats: int = 10000,
) -> dict[str, Any]:
    seeds = sorted(differences)
    shared = sorted(set.intersection(*(set(differences[value]) for value in seeds)))
    if set(shared) != set(component_by_instance):
        raise ValueError("paired effect coverage differs from component authority")
    components = sorted(set(component_by_instance.values()))
    matrix = np.empty((len(seeds), len(components)), dtype=np.float64)
    for seed_index, training_seed in enumerate(seeds):
        for component_index, component in enumerate(components):
            values = [
                differences[training_seed][instance]
                for instance in shared
                if component_by_instance[instance] == component
            ]
            matrix[seed_index, component_index] = float(np.mean(values))
    rng = np.random.default_rng(seed)
    crossed = np.empty(repeats, dtype=np.float64)
    conditional = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled_seeds = rng.integers(matrix.shape[0], size=matrix.shape[0])
        sampled_components = rng.integers(matrix.shape[1], size=matrix.shape[1])
        crossed[index] = float(matrix[sampled_seeds][:, sampled_components].mean())
        conditional[index] = float(matrix[:, sampled_components].mean())
    raw = np.asarray(
        [differences[training_seed][instance] for training_seed in seeds for instance in shared]
    )
    return {
        "estimand": "equal_component_equal_baseline_seed_mean_paired_AP_difference",
        "mean_difference": float(matrix.mean()),
        "complex_weighted_mean_difference": float(raw.mean()),
        "crossed_baseline_seed_component_ci95": [
            float(value) for value in np.quantile(crossed, [0.025, 0.975])
        ],
        "component_ci95_conditional_on_baseline_seeds": [
            float(value) for value in np.quantile(conditional, [0.025, 0.975])
        ],
        "positive_component_fraction": float(np.mean(matrix.mean(axis=0) > 0.0)),
        "negative_component_fraction": float(np.mean(matrix.mean(axis=0) < 0.0)),
        "component_count": len(components),
        "complex_count": len(shared),
        "baseline_seeds": seeds,
        "bootstrap_repeats": repeats,
        "scope": "development-only exploratory paired comparison",
    }


def compare_one(
    name: str,
    reparank_ap: dict[str, float],
    authority: dict[str, dict[str, Any]],
    baseline_root: Path,
    arm: str,
    seeds: tuple[int, ...],
    components: dict[str, str],
) -> dict[str, Any]:
    effects: dict[int, dict[str, float]] = {}
    seed_rows = []
    input_sha256 = {}
    for seed in seeds:
        path = baseline_root / arm / f"seed_{seed}" / "validation" / "pair_predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        baseline = baseline_per_complex(path, authority)
        input_sha256[str(path)] = sha256(path)
        effect = {instance: reparank_ap[instance] - baseline[instance] for instance in reparank_ap}
        effects[seed] = effect
        seed_rows.append({
            "seed": seed,
            "reparank_macro_AP": float(np.mean(list(reparank_ap.values()))),
            "baseline_macro_AP": float(np.mean(list(baseline.values()))),
            "complex_weighted_paired_difference": float(np.mean(list(effect.values()))),
        })
    return {
        "comparison": name,
        "baseline_arm": arm,
        "baseline_seeds": list(seeds),
        "per_seed": seed_rows,
        "paired_analysis": crossed_bootstrap(effects, components),
        "baseline_input_sha256": input_sha256,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reparank-sequence-root", type=Path, required=True)
    parser.add_argument("--reparank-structure-root", type=Path)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    validation = set(map(str, split["splits"]["validation"]))
    components = {
        instance: str(split["component_id_by_instance"][instance]) for instance in validation
    }
    sequence_authority, sequence_ap = reparank_rows(args.reparank_sequence_root.resolve())
    if set(sequence_ap) != validation:
        raise ValueError("RePaRank sequence coverage differs from validation authority")

    checkpoint_results = {
        "reparank_sequence": {
            "root": str(args.reparank_sequence_root.resolve()),
            "evaluation_sha256": sha256(args.reparank_sequence_root.resolve() / "evaluation.json"),
            "macro_AP": float(np.mean(list(sequence_ap.values()))),
            "analysis_role": "post-hoc because aggregate AP was known before protocol freeze",
        }
    }
    comparisons = []
    for arm in BASELINE_ARMS:
        comparisons.append(
            compare_one(
                f"reparank_sequence_minus_{arm}",
                sequence_ap,
                sequence_authority,
                args.baseline_root.resolve(),
                arm,
                PRIMARY_SEEDS,
                components,
            )
        )
    comparisons.append(
        compare_one(
            "reparank_sequence_minus_esm2_seeds_11_20_sensitivity",
            sequence_ap,
            sequence_authority,
            args.baseline_root.resolve(),
            "esm2",
            ESM2_SENSITIVITY_SEEDS,
            components,
        )
    )

    if args.reparank_structure_root is not None:
        structure_authority, structure_ap = reparank_rows(args.reparank_structure_root.resolve())
        if structure_authority != sequence_authority:
            raise ValueError("sequence and structure RePaRank candidate/label authorities differ")
        checkpoint_results["reparank_sequence_structure"] = {
            "root": str(args.reparank_structure_root.resolve()),
            "evaluation_sha256": sha256(args.reparank_structure_root.resolve() / "evaluation.json"),
            "macro_AP": float(np.mean(list(structure_ap.values()))),
            "analysis_role": "predeclared before full structure result was opened",
        }
        for arm in BASELINE_ARMS:
            comparisons.append(
                compare_one(
                    f"reparank_sequence_structure_minus_{arm}",
                    structure_ap,
                    sequence_authority,
                    args.baseline_root.resolve(),
                    arm,
                    PRIMARY_SEEDS,
                    components,
                )
            )

    payload = {
        "format_version": "reparank_same_grid_baseline_comparison_v1",
        "status": "COMPLETE_DEVELOPMENT_COMPARISON",
        "protocol": "REPARANK_BASELINE_COMPARISON_PROTOCOL.md",
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256(args.split_manifest.resolve()),
        "validation_complex_count": len(validation),
        "component_count": len(set(components.values())),
        "checkpoint_results": checkpoint_results,
        "comparisons": comparisons,
        "test_labels_read": False,
        "ood_labels_read": False,
        "checkpoint_training_overlap_audited": False,
        "blind_claim_authorized": False,
        "sota_claim_authorized": False,
        "limitations": [
            "Sequence-checkpoint comparison is post-hoc; its aggregate AP was already known.",
            "Official checkpoint training overlap with this cohort is unknown.",
            "Models differ in pretraining, parameter count, inputs and training budget.",
            "Baseline training seeds quantify optimisation variation, not RePaRank uncertainty.",
            "Intervals are pointwise exploratory and not multiplicity-adjusted.",
        ],
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps({
        "status": payload["status"],
        "checkpoint_results": checkpoint_results,
        "comparison_count": len(comparisons),
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
