#!/usr/bin/env python3
"""Audit multi-scale epitope topology on authenticated AsEP validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.asep_external import AsepDevelopmentDataset  # noqa: E402
from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)
from interaction_first.topology_signal import (  # noqa: E402
    paired_relation_delta,
    relation_label_signal,
    sequence_relation_mask,
    spatial_relation_mask,
    summarize_metric,
)


FORMAT = "asep_validation_topology_signal_v1"
PROTOCOL_FORMAT = "framecontact_topology_signal_exploratory_protocol_v1"
RECEIPT_FORMAT = "framecontact_topology_signal_exploratory_freeze_receipt_v1"
RADII = (6.0, 8.0, 12.0, 16.0)
SEQUENCE_WINDOWS = (2, 8, 32)
METRICS = (
    "loo_neighbor_auprc",
    "positive_neighbor_rate",
    "negative_neighbor_rate",
    "positive_neighbor_lift",
    "positive_vs_negative_neighbor_rate_gap",
    "edge_homophily",
    "positive_component_count",
    "positive_largest_component_fraction",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--protocol-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def _authority(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol_sha256 = authenticate_file(
        arguments.protocol, arguments.protocol_sha256, "topology protocol"
    )
    receipt_sha256 = authenticate_file(
        arguments.protocol_receipt,
        arguments.protocol_receipt_sha256,
        "topology protocol receipt",
    )
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    receipt = json.loads(arguments.protocol_receipt.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != PROTOCOL_FORMAT
        or receipt.get("schema_version") != RECEIPT_FORMAT
        or receipt.get("protocol_file_sha256") != protocol_sha256
        or receipt.get("frozen_test_authorized") is not False
    ):
        raise ValueError("topology protocol authority is not eligible")
    manifest = arguments.cache_root / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"
    cache_sha256 = authenticate_file(
        manifest, arguments.cache_manifest_sha256, "AsEP cache manifest"
    )
    cache = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        cache.get("checks", {}).get("official_test_members_materialized") is not False
        or any(record.get("role") == "test" for record in cache.get("records", []))
    ):
        raise ValueError("AsEP development cache contains a test role")
    return {
        "cache_manifest_sha256": cache_sha256,
        "protocol_file_sha256": protocol_sha256,
        "protocol_receipt_sha256": receipt_sha256,
        "official_test_members_materialized": False,
        "model_predictions_accessed": False,
        "checkpoint_selection_performed": False,
    }


def _stratum(row: dict[str, Any]) -> dict[str, bool]:
    length = int(row["antigen_length"])
    prevalence = float(row["epitope_prevalence"])
    components = row["relations"]["spatial_radius_8a"]["positive_component_count"]
    return {
        "length_lt_200": length < 200,
        "length_200_to_500": 200 <= length <= 500,
        "length_gt_500": length > 500,
        "prevalence_le_0_05": prevalence <= 0.05,
        "prevalence_gt_0_05": prevalence > 0.05,
        "spatial_8a_single_component": components == 1,
        "spatial_8a_multiple_components": components > 1,
    }


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {arguments.output}")
    if arguments.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    authority = _authority(arguments)
    dataset = AsepDevelopmentDataset(
        arguments.cache_root,
        manifest_sha256=arguments.cache_manifest_sha256,
        split_role="val",
        feature_mode="official_esm2_35m",
    )
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        labels = item["antigen_labels"].numpy().astype(np.float64, copy=False)
        backbone = item["antigen_backbone"].numpy()
        chain_ids = item["antigen_chain_ids"].numpy()
        frame_valid = item["antigen_frame_valid_mask"].numpy().astype(
            np.bool_, copy=False
        )
        residue_mask = np.ones(len(labels), dtype=np.bool_)
        relations: dict[str, Any] = {}
        for radius in RADII:
            name = f"spatial_radius_{int(radius)}a"
            relations[name] = relation_label_signal(
                spatial_relation_mask(
                    backbone, chain_ids, frame_valid, radius=radius
                ),
                labels,
            )
        for window in SEQUENCE_WINDOWS:
            name = f"sequence_window_{window}"
            relations[name] = relation_label_signal(
                sequence_relation_mask(chain_ids, residue_mask, window=window),
                labels,
            )
        rows.append(
            {
                "instance_id": item["instance_id"],
                "antigen_length": int(len(labels)),
                "epitope_count": int(labels.sum()),
                "epitope_prevalence": float(labels.mean()),
                "valid_frame_fraction": float(frame_valid.mean()),
                "relations": relations,
            }
        )
    relation_names = tuple(rows[0]["relations"])
    aggregate = {
        relation: {
            metric: summarize_metric(
                [row["relations"][relation][metric] for row in rows],
                bootstrap_replicates=arguments.bootstrap_replicates,
                seed=arguments.seed + relation_index * 101 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
        for relation_index, relation in enumerate(relation_names)
    }
    comparisons = {
        "spatial_8a_minus_sequence_window_8": paired_relation_delta(
            rows,
            left="spatial_radius_8a",
            right="sequence_window_8",
            metric="loo_neighbor_auprc",
            bootstrap_replicates=arguments.bootstrap_replicates,
            seed=arguments.seed + 70_001,
        ),
        "spatial_16a_minus_spatial_8a": paired_relation_delta(
            rows,
            left="spatial_radius_16a",
            right="spatial_radius_8a",
            metric="loo_neighbor_auprc",
            bootstrap_replicates=arguments.bootstrap_replicates,
            seed=arguments.seed + 70_002,
        ),
    }
    stratum_members = {
        name: [row for row in rows if _stratum(row)[name]]
        for name in _stratum(rows[0])
    }
    strata = {
        name: {
            "complex_count": len(members),
            "spatial_8a_loo_neighbor_auprc": summarize_metric(
                [
                    row["relations"]["spatial_radius_8a"]["loo_neighbor_auprc"]
                    for row in members
                ],
                bootstrap_replicates=arguments.bootstrap_replicates,
                seed=arguments.seed + 80_000 + index,
            ),
            "spatial_8a_positive_neighbor_lift": summarize_metric(
                [
                    row["relations"]["spatial_radius_8a"]["positive_neighbor_lift"]
                    for row in members
                ],
                bootstrap_replicates=arguments.bootstrap_replicates,
                seed=arguments.seed + 90_000 + index,
            ),
        }
        for index, (name, members) in enumerate(stratum_members.items())
    }
    payload = {
        "format_version": FORMAT,
        "authority": authority,
        "split_role": "val",
        "complex_count": len(rows),
        "registered_relations": {
            "spatial_radii_angstrom": list(RADII),
            "sequence_windows": list(SEQUENCE_WINDOWS),
            "same_chain_only": True,
            "self_edges_excluded": True,
        },
        "metric_semantics": {
            "loo_neighbor_auprc": (
                "Descriptive label-autocorrelation statistic using neighboring "
                "validation labels; it is not prediction performance."
            ),
            "positive_neighbor_lift": (
                "Mean positive-neighbor fraction around positive queries divided "
                "by query prevalence."
            ),
        },
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "strata": strata,
        "per_complex": rows,
        "claim_boundary": (
            "Authenticated AsEP validation-label topology diagnostic only. No "
            "official test member, model prediction, checkpoint, external score, "
            "or benchmark-improvement claim was accessed or produced."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
                "complex_count": len(rows),
                "official_test_members_materialized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
