#!/usr/bin/env python3
"""Measurement-only recovery for the frozen SAbDab2 ensemble test.

The frozen evaluator's cluster bootstrap accidentally reads cluster identifiers
from structure-cache records.  Those identifiers live in the already
authenticated reciprocal partner panel.  This wrapper validates and imports the
original evaluator, intercepts that authenticated panel, and replaces only the
cluster-group lookup used by bootstrap confidence intervals.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any


EXPECTED_ORIGINAL_SHA256 = (
    "122650985338b5cedd0b452958aceaf20e9ce43d45546f72ac9ddd7a4d1e6b3a"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


original_path = Path(os.environ["FRAMECONTACT_ORIGINAL_EVALUATOR"])
if _sha256(original_path) != EXPECTED_ORIGINAL_SHA256:
    raise RuntimeError("Original frozen evaluator SHA-256 differs")
if str(original_path.parent) not in sys.path:
    sys.path.insert(0, str(original_path.parent))

spec = importlib.util.spec_from_file_location(
    "framecontact_frozen_test_original", original_path
)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load original frozen evaluator")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

active_panel: dict[str, Any] | None = None
original_authenticated_panel = module._authenticated_panel


def _authenticated_panel(*args: Any, **kwargs: Any) -> dict[str, Any]:
    global active_panel
    panel = original_authenticated_panel(*args, **kwargs)
    active_panel = panel
    return panel


def _add_bootstrap_recovery(
    metrics: dict[str, Any],
    dataset: Any,
    *,
    replicates: int,
    seed: int,
) -> None:
    if active_panel is None:
        raise RuntimeError("Authenticated reciprocal partner panel is unavailable")

    for index, direction_metrics in enumerate(
        metrics["partner_retrieval_by_direction"].values()
    ):
        direction_metrics["cluster_bootstrap_ci"] = (
            module.bootstrap_cluster_weighted_margin(
                direction_metrics["cluster_margins"],
                replicates=replicates,
                seed=seed + index,
            )
        )

    dataset_ids = {str(record["instance_id"]) for record in dataset.records}
    panel_records = {
        str(entry["instance_id"]): entry for entry in active_panel["entries"]
    }
    if dataset_ids != set(panel_records):
        raise ValueError("Authenticated panel and test dataset identifiers differ")

    for index, (metric_name, group_field) in enumerate(
        (
            ("contact", "antigen_cluster_id"),
            ("epitope", "antigen_cluster_id"),
            ("paratope", "antibody_cluster_id"),
        )
    ):
        group_by_instance = {
            identifier: str(record[group_field])
            for identifier, record in panel_records.items()
        }
        if any(not value for value in group_by_instance.values()):
            raise ValueError(f"Authenticated panel has an empty {group_field}")
        metrics[metric_name]["mean_per_complex_auprc_cluster_bootstrap_ci"] = (
            module.bootstrap_group_weighted_values(
                metrics[metric_name]["per_complex_auprc_by_instance"],
                group_by_instance,
                replicates=replicates,
                seed=seed + 1_000 * (index + 1),
            )
        )
        metrics[metric_name]["mean_per_complex_auprc_cluster_definition"] = (
            f"equal-weight mean over {group_field} means; zero-positive complexes "
            "are ineligible for positive-required per-complex AP but remain in "
            "pooled metrics"
        )


module._authenticated_panel = _authenticated_panel
module._add_bootstrap = _add_bootstrap_recovery

if __name__ == "__main__":
    module.main()
