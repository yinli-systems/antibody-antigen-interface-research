#!/usr/bin/env python3
"""Relate sequence-predicted geometry error to development contact ranking changes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


PREDICTED_STRUCTURE_MANIFEST_SHA256 = (
    "bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614"
)
BOUND_STRUCTURE_MANIFEST_SHA256 = (
    "edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0"
)
SPLIT_SHA256 = (
    "ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700"
)
PREDICTED_ANALYSIS_SHA256 = (
    "a4168a919e97b1344d718946fdd51305803773ae3355a3014ee2f05583a077b7"
)
BOUND_ANALYSIS_SHA256 = (
    "f4368275e38d5134e667e23470702e987bb1a301b1c89aee5e82718d39911824"
)
SEEDS = (17, 29, 43)
CHAIN_NAMES = ("antibody", "antigen")
GRAPH_K = 32
GRAPH_RADIUS = 16.0
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260829


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite value")
    return number


def _records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {str(row["instance_id"]): row for row in manifest["records"]}
    if len(rows) != len(manifest["records"]):
        raise ValueError("duplicate structure manifest instance")
    return rows


def _load_artifact(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = root / str(record["artifact_path"])
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != int(record["artifact_size_bytes"])
        or sha256_file(path) != record["artifact_sha256"]
    ):
        raise ValueError(f"structure artifact differs: {record['instance_id']}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("instance_id") != record["instance_id"]:
        raise ValueError(f"artifact instance differs: {record['instance_id']}")
    return payload


def _kabsch_errors(
    predicted: np.ndarray,
    bound: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_center = predicted[:, 1, :].mean(axis=0)
    bound_center = bound[:, 1, :].mean(axis=0)
    p_ca = predicted[:, 1, :] - predicted_center
    q_ca = bound[:, 1, :] - bound_center
    u, _, vt = np.linalg.svd(p_ca.T @ q_ca, full_matrices=False)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = (predicted - predicted_center) @ rotation + bound_center
    ca_error = np.linalg.norm(aligned[:, 1, :] - bound[:, 1, :], axis=-1)
    backbone_error = np.linalg.norm(aligned - bound, axis=-1).reshape(-1)
    return ca_error, backbone_error


def _chain_geometry_metrics(
    predicted: dict[str, Any],
    bound: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    predicted_backbone = predicted[f"{name}_backbone"].numpy().astype(np.float64)
    bound_backbone = bound[f"{name}_backbone"].numpy().astype(np.float64)
    predicted_chain_ids = predicted[f"{name}_chain_ids"].numpy()
    bound_chain_ids = bound[f"{name}_chain_ids"].numpy()
    predicted_valid = predicted[f"{name}_frame_valid_mask"].numpy().astype(bool)
    bound_valid = bound[f"{name}_frame_valid_mask"].numpy().astype(bool)
    if (
        predicted_backbone.shape != bound_backbone.shape
        or not np.array_equal(predicted_chain_ids, bound_chain_ids)
    ):
        raise ValueError(f"{name} residue axis differs")
    joint_valid = predicted_valid & bound_valid
    ca_errors: list[np.ndarray] = []
    backbone_errors: list[np.ndarray] = []
    chain_rows: list[dict[str, Any]] = []
    for chain_id in np.unique(bound_chain_ids):
        indices = np.flatnonzero(joint_valid & (bound_chain_ids == chain_id))
        if len(indices) < 3:
            raise ValueError(f"{name} chain has fewer than three jointly valid frames")
        local_ca, local_backbone = _kabsch_errors(
            predicted_backbone[indices], bound_backbone[indices]
        )
        ca_errors.append(local_ca)
        backbone_errors.append(local_backbone)
        chain_rows.append(
            {
                "chain_id": int(chain_id),
                "joint_valid_residues": int(len(indices)),
                "ca_rmsd_angstrom": float(np.sqrt(np.mean(local_ca**2))),
                "backbone_rmsd_angstrom": float(
                    np.sqrt(np.mean(local_backbone**2))
                ),
            }
        )
    ca_error = np.concatenate(ca_errors)
    backbone_error = np.concatenate(backbone_errors)
    return {
        "residue_count": int(len(bound_chain_ids)),
        "joint_valid_residue_count": int(joint_valid.sum()),
        "predicted_frame_coverage": float(predicted_valid.mean()),
        "bound_frame_coverage": float(bound_valid.mean()),
        "joint_frame_coverage": float(joint_valid.mean()),
        "ca_rmsd_angstrom": float(np.sqrt(np.mean(ca_error**2))),
        "backbone_rmsd_angstrom": float(np.sqrt(np.mean(backbone_error**2))),
        "ca_gdt_fraction": {
            str(cutoff): float(np.mean(ca_error <= cutoff))
            for cutoff in (1.0, 2.0, 4.0, 8.0)
        },
        "chains": chain_rows,
    }


def _neighbor_graph(payload: dict[str, Any], name: str) -> np.ndarray:
    backbone = payload[f"{name}_backbone"].numpy().astype(np.float64)
    chain_ids = payload[f"{name}_chain_ids"].numpy()
    valid = payload[f"{name}_frame_valid_mask"].numpy().astype(bool)
    residue_count = len(chain_ids)
    graph = np.zeros((residue_count, residue_count), dtype=bool)
    ca = backbone[:, 1, :]
    for chain_id in np.unique(chain_ids):
        chain_indices = np.flatnonzero(chain_ids == chain_id)
        valid_indices = chain_indices[valid[chain_indices]]
        if len(valid_indices):
            coords = ca[valid_indices]
            distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            for local_row, global_row in enumerate(valid_indices):
                candidates = np.flatnonzero(distances[local_row] <= GRAPH_RADIUS)
                if len(candidates) > GRAPH_K:
                    order = np.argsort(distances[local_row, candidates], kind="stable")
                    candidates = candidates[order[:GRAPH_K]]
                graph[global_row, valid_indices[candidates]] = True
        graph[chain_indices, chain_indices] = True
    return graph


def _graph_metrics(
    predicted: dict[str, Any], bound: dict[str, Any], name: str
) -> dict[str, float | int]:
    predicted_graph = _neighbor_graph(predicted, name)
    bound_graph = _neighbor_graph(bound, name)
    identity = np.eye(len(predicted_graph), dtype=bool)
    predicted_edges = predicted_graph & ~identity
    bound_edges = bound_graph & ~identity
    intersection = int(np.logical_and(predicted_edges, bound_edges).sum())
    union = int(np.logical_or(predicted_edges, bound_edges).sum())
    predicted_count = int(predicted_edges.sum())
    bound_count = int(bound_edges.sum())
    return {
        "predicted_directed_edges_excluding_self": predicted_count,
        "bound_directed_edges_excluding_self": bound_count,
        "intersection_directed_edges_excluding_self": intersection,
        "union_directed_edges_excluding_self": union,
        "jaccard_excluding_self": float(intersection / union) if union else 1.0,
        "predicted_edge_precision_vs_bound": (
            float(intersection / predicted_count) if predicted_count else 1.0
        ),
        "predicted_edge_recall_vs_bound": (
            float(intersection / bound_count) if bound_count else 1.0
        ),
    }


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rank_average(x)
    ry = _rank_average(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denominator)


def _cluster_bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[str, Any]:
    clusters = np.unique(cluster_ids)
    members = {cluster: np.flatnonzero(cluster_ids == cluster) for cluster in clusters}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        indices = np.concatenate([members[cluster] for cluster in chosen])
        samples[replicate] = _spearman(x[indices], y[indices])
    samples = samples[np.isfinite(samples)]
    if len(samples) < int(0.95 * BOOTSTRAP_REPLICATES):
        raise ValueError("too many undefined cluster-bootstrap correlations")
    coefficient = _spearman(x, y)
    if not math.isfinite(coefficient):
        raise ValueError("undefined point Spearman correlation")
    return {
        "coefficient": coefficient,
        "bootstrap_unit": "antigen_cluster",
        "cluster_count": int(len(clusters)),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "finite_bootstrap_replicates": int(len(samples)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "percentile_95_ci": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def _quartiles(x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(x, kind="mergesort")
    groups = np.array_split(order, 4)
    return [
        {
            "quartile": index + 1,
            "count": int(len(indices)),
            "predictor_min": float(x[indices].min()),
            "predictor_max": float(x[indices].max()),
            "outcome_mean": float(y[indices].mean()),
            "outcome_median": float(np.median(y[indices])),
        }
        for index, indices in enumerate(groups)
    ]


def _ranked_contact_rows(
    analysis: dict[str, Any], arm: str
) -> dict[int, dict[str, dict[str, Any]]]:
    output: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in SEEDS:
        binding = analysis["run_bindings"][arm][str(seed)]
        result_path = Path(binding["run_root"]) / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        if sha256_file(result_path) != binding["result_sha256"]:
            raise ValueError(f"result SHA differs: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        authority = result["authority"]
        if (
            authority.get("test_loaded") is not False
            or authority.get("external_test_loaded") is not False
        ):
            raise ValueError(f"test access differs: {result_path}")
        output[seed] = result["selected_checkpoint_validation"]["contact"][
            "ranked_pair_by_instance"
        ]
    return output


def _contact_delta(
    instance_id: str,
    predicted: dict[int, dict[str, dict[str, Any]]],
    bound: dict[int, dict[str, dict[str, Any]]],
    cutoff: str,
    field: str,
) -> tuple[dict[str, float], float]:
    by_seed = {
        str(seed): finite(predicted[seed][instance_id]["cutoffs"][cutoff][field])
        - finite(bound[seed][instance_id]["cutoffs"][cutoff][field])
        for seed in SEEDS
    }
    return by_seed, float(np.mean(list(by_seed.values())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted-structure-root", type=Path, required=True)
    parser.add_argument("--bound-structure-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--predicted-analysis", type=Path, required=True)
    parser.add_argument("--bound-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    predicted_manifest_path = args.predicted_structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    bound_manifest_path = args.bound_structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    expected = {
        predicted_manifest_path: PREDICTED_STRUCTURE_MANIFEST_SHA256,
        bound_manifest_path: BOUND_STRUCTURE_MANIFEST_SHA256,
        args.split: SPLIT_SHA256,
        args.predicted_analysis: PREDICTED_ANALYSIS_SHA256,
        args.bound_analysis: BOUND_ANALYSIS_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"authority SHA differs: {path}")

    predicted_manifest = json.loads(predicted_manifest_path.read_text(encoding="utf-8"))
    bound_manifest = json.loads(bound_manifest_path.read_text(encoding="utf-8"))
    predicted_records = _records(predicted_manifest)
    bound_records = _records(bound_manifest)
    validation: list[dict[str, str]] = []
    with args.split.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["role"] == "validation":
                validation.append(row)
    if len(validation) != 549 or len({row["instance_id"] for row in validation}) != 549:
        raise ValueError("validation split differs")

    predicted_analysis = json.loads(args.predicted_analysis.read_text(encoding="utf-8"))
    bound_analysis = json.loads(args.bound_analysis.read_text(encoding="utf-8"))
    if (
        predicted_analysis.get("old_sabdab2_test_loaded") is not False
        or predicted_analysis.get("temporal_holdout_loaded") is not False
        or bound_analysis.get("old_test_loaded") is not False
        or bound_analysis.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("analysis test-access boundary differs")
    predicted_ranked = _ranked_contact_rows(predicted_analysis, "baseline")
    bound_ranked = _ranked_contact_rows(bound_analysis, "baseline_current")

    rows: list[dict[str, Any]] = []
    for split_row in validation:
        identifier = split_row["instance_id"]
        predicted_record = predicted_records[identifier]
        bound_record = bound_records[identifier]
        for key in (
            "antibody_length",
            "antigen_length",
            "antibody_sequence_sha256",
            "antigen_sequence_sha256",
        ):
            if predicted_record[key] != bound_record[key]:
                raise ValueError(f"{identifier}: structure axes differ")
        predicted_artifact = _load_artifact(
            args.predicted_structure_root, predicted_record
        )
        bound_artifact = _load_artifact(args.bound_structure_root, bound_record)
        chain_geometry = {
            name: _chain_geometry_metrics(predicted_artifact, bound_artifact, name)
            for name in CHAIN_NAMES
        }
        graph = {
            name: _graph_metrics(predicted_artifact, bound_artifact, name)
            for name in CHAIN_NAMES
        }
        total_residues = sum(chain_geometry[name]["joint_valid_residue_count"] for name in CHAIN_NAMES)
        weighted_ca_rmsd = math.sqrt(
            sum(
                chain_geometry[name]["joint_valid_residue_count"]
                * chain_geometry[name]["ca_rmsd_angstrom"] ** 2
                for name in CHAIN_NAMES
            )
            / total_residues
        )
        total_union = sum(graph[name]["union_directed_edges_excluding_self"] for name in CHAIN_NAMES)
        total_intersection = sum(
            graph[name]["intersection_directed_edges_excluding_self"]
            for name in CHAIN_NAMES
        )
        precision_by_seed, precision_delta = _contact_delta(
            identifier,
            predicted_ranked,
            bound_ranked,
            "lmin_over_5",
            "precision",
        )
        recall_by_seed, recall_delta = _contact_delta(
            identifier,
            predicted_ranked,
            bound_ranked,
            "lmin_over_5",
            "recall",
        )
        rows.append(
            {
                "instance_id": identifier,
                "antigen_cluster_id": split_row["antigen_cluster_id"],
                "antibody_cluster_id": split_row["antibody_cluster_id"],
                "geometry": chain_geometry,
                "graph": graph,
                "weighted_ca_rmsd_angstrom": float(weighted_ca_rmsd),
                "combined_graph_jaccard_excluding_self": (
                    float(total_intersection / total_union) if total_union else 1.0
                ),
                "af3_ranking_score": {
                    "antibody": finite(
                        predicted_record["antibody_prediction"]["selected_ranking_score"]
                    ),
                    "antigen": finite(
                        predicted_record["antigen_prediction"]["selected_ranking_score"]
                    ),
                },
                "contact_precision_lmin_over_5_predicted_minus_bound_by_seed": precision_by_seed,
                "contact_precision_lmin_over_5_predicted_minus_bound_seed_mean": precision_delta,
                "contact_recall_lmin_over_5_predicted_minus_bound_by_seed": recall_by_seed,
                "contact_recall_lmin_over_5_predicted_minus_bound_seed_mean": recall_delta,
            }
        )

    cluster_ids = np.asarray([row["antigen_cluster_id"] for row in rows])
    outcomes = {
        "contact_precision_lmin_over_5_delta_seed_mean": np.asarray(
            [
                row[
                    "contact_precision_lmin_over_5_predicted_minus_bound_seed_mean"
                ]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "contact_recall_lmin_over_5_delta_seed_mean": np.asarray(
            [
                row["contact_recall_lmin_over_5_predicted_minus_bound_seed_mean"]
                for row in rows
            ],
            dtype=np.float64,
        ),
    }
    predictors = {
        "weighted_ca_rmsd_angstrom": np.asarray(
            [row["weighted_ca_rmsd_angstrom"] for row in rows], dtype=np.float64
        ),
        "combined_graph_jaccard_excluding_self": np.asarray(
            [row["combined_graph_jaccard_excluding_self"] for row in rows],
            dtype=np.float64,
        ),
        "antibody_ca_rmsd_angstrom": np.asarray(
            [row["geometry"]["antibody"]["ca_rmsd_angstrom"] for row in rows],
            dtype=np.float64,
        ),
        "antigen_ca_rmsd_angstrom": np.asarray(
            [row["geometry"]["antigen"]["ca_rmsd_angstrom"] for row in rows],
            dtype=np.float64,
        ),
        "antibody_af3_ranking_score": np.asarray(
            [row["af3_ranking_score"]["antibody"] for row in rows],
            dtype=np.float64,
        ),
        "antigen_af3_ranking_score": np.asarray(
            [row["af3_ranking_score"]["antigen"] for row in rows],
            dtype=np.float64,
        ),
    }
    relationships = {
        predictor_name: {
            outcome_name: _cluster_bootstrap_spearman(
                predictor_values, outcome_values, cluster_ids
            )
            for outcome_name, outcome_values in outcomes.items()
        }
        for predictor_name, predictor_values in predictors.items()
    }
    quartile_summaries = {
        predictor_name: {
            outcome_name: _quartiles(predictor_values, outcome_values)
            for outcome_name, outcome_values in outcomes.items()
        }
        for predictor_name, predictor_values in predictors.items()
    }
    payload = {
        "format_version": "sabdab2_predicted_geometry_quality_mechanism_analysis_v1",
        "status": "PASS_DEVELOPMENT_ONLY_GEOMETRY_QUALITY_CONTACT_RANKING_ANALYSIS",
        "definition": {
            "alignment": "Kabsch alignment independently within each physical chain",
            "ca_atom_index": 1,
            "graph": {
                "same_physical_chain_only": True,
                "radius_angstrom": GRAPH_RADIUS,
                "maximum_neighbors": GRAPH_K,
                "self_edges_excluded_from_comparison": True,
            },
            "contact_outcome": (
                "seed-mean predicted-minus-bound precision or recall at Lmin/5 "
                "from seed-matched baseline checkpoints"
            ),
        },
        "authority": {
            "predicted_structure_manifest_sha256": PREDICTED_STRUCTURE_MANIFEST_SHA256,
            "bound_structure_manifest_sha256": BOUND_STRUCTURE_MANIFEST_SHA256,
            "development_split_sha256": SPLIT_SHA256,
            "predicted_analysis_sha256": PREDICTED_ANALYSIS_SHA256,
            "bound_analysis_sha256": BOUND_ANALYSIS_SHA256,
        },
        "counts": {
            "validation_complexes": len(rows),
            "antigen_clusters": int(len(np.unique(cluster_ids))),
            "seeds": len(SEEDS),
        },
        "relationships": relationships,
        "quartile_summaries": quartile_summaries,
        "per_complex": rows,
        "development_only": True,
        "full_validation_is_checkpoint_selection_data": True,
        "independent_confirmation": False,
        "experimental_bound_geometry_used_for_quality_reference": True,
        "experimental_bound_geometry_used_as_predicted_model_input": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
        "selection_or_tuning_from_this_analysis": False,
    }
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "sabdab2_predicted_geometry_quality_mechanism_receipt_v1",
        "status": payload["status"],
        "analysis_sha256": sha256_file(args.output),
        "all_structure_artifact_sha256_verified": True,
        "all_result_sha256_verified": True,
        "validation_complexes": len(rows),
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.receipt_output.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "analysis_sha256": receipt["analysis_sha256"],
                "receipt_sha256": sha256_file(args.receipt_output),
                "validation_complexes": len(rows),
                "relationships": relationships,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
