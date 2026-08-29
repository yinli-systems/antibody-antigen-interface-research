#!/usr/bin/env python3
"""Open frozen SAbDab2 test once for five seeds and their frozen ensemble."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.framecontact_ensemble import (  # noqa: E402
    CalibratedProbabilityEnsemble,
)
from interaction_first.framecontact_metrics import (  # noqa: E402
    bootstrap_cluster_weighted_margin,
    bootstrap_group_weighted_values,
)
from interaction_first.framecontact_interventions import (  # noqa: E402
    corrupt_local_frames,
    degree_matched_graph_rewiring,
    geometry_shuffle,
    independent_monomer_rigid_transform,
    perturb_coordinates,
)
from interaction_first.sabdab2_contacts import Sabdab2ContactDataset  # noqa: E402
from interaction_first.sabdab2_frozen_test import (  # noqa: E402
    authenticate_sabdab2_development_selection,
    authenticate_sabdab2_test_opening,
    consume_sabdab2_score_once,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    Sabdab2StructureBatchCollator,
    Sabdab2StructureEmbeddingDataset,
    canonical_json_bytes,
    sha256_file,
)
from interaction_first.same_antigen_challenge import (  # noqa: E402
    LABEL_FREE_SAME_ANTIGEN_CHALLENGE_FORMAT,
    aggregate_same_antigen_pair_metrics,
    bootstrap_same_antigen_group_metrics,
    score_same_antigen_pair,
)
from freeze_framecontact_development_ensemble import (  # noqa: E402
    ENSEMBLE_FORMAT,
    FROZEN_SEEDS,
    _authenticated_run,
    _member_calibration,
)
from train_framecontact import (  # noqa: E402
    RECIPROCAL_PARTNER_PANEL_FORMAT,
    _autocast,
    _authenticated_panel,
    _device,
    _evaluate,
    _loader,
    _model_from_configuration,
    _move_batch,
    _wrong_antibody_pair_item,
)


TEST_FORMAT = "framecontact_sabdab2_frozen_ensemble_test_v2"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-receipt", type=Path, required=True)
    parser.add_argument("--opening-receipt-sha256", required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--selection-receipt-sha256", required=True)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--run-roots", type=Path, nargs=5, required=True)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    parser.add_argument("--contact-manifest-sha256", required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--embedding-manifest-sha256", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--test-partner-panel", type=Path, required=True)
    parser.add_argument("--test-partner-panel-sha256", required=True)
    parser.add_argument("--same-antigen-challenge", type=Path, required=True)
    parser.add_argument("--same-antigen-challenge-sha256", required=True)
    parser.add_argument(
        "--same-antigen-maximum-epitope-jaccard", type=float, default=0.50
    )
    parser.add_argument("--panel-size", type=int, default=31)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_730)
    parser.add_argument(
        "--coordinate-noise",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50),
        help="Sequence-preserving backbone noise levels in Angstrom.",
    )
    parser.add_argument(
        "--skip-interventions",
        action="store_true",
        help="Diagnostic escape hatch; outputs are marked incomplete for the frozen protocol.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def _authenticated_ensemble(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = root / "FRAMECONTACT_DEVELOPMENT_ENSEMBLE.json"
    receipt_path = root / "ENSEMBLE_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("format_version") != ENSEMBLE_FORMAT
        or receipt.get("result_sha256") != sha256_file(result_path)
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
    ):
        raise ValueError("Development ensemble receipt is not test eligible")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("format_version") != ENSEMBLE_FORMAT
        or result.get("authority", {}).get("test_loaded") is not False
        or result.get("authority", {}).get("external_test_loaded") is not False
        or tuple(result.get("seeds", ())) != FROZEN_SEEDS
    ):
        raise ValueError("Development ensemble result is not test eligible")
    return result, receipt


def _frozen_calibration(validation: Mapping[str, Any]) -> dict[str, float]:
    result = {
        "temperature": float(validation["validation_temperature"]),
        "paratope_temperature": float(
            validation["paratope"]["validation_temperature"]
        ),
        "epitope_temperature": float(
            validation["epitope"]["validation_temperature"]
        ),
        "contact_threshold": float(
            validation["contact"]["threshold_selected_on_validation"]
        ),
        "paratope_threshold": float(
            validation["paratope"]["threshold_selected_on_validation"]
        ),
        "epitope_threshold": float(
            validation["epitope"]["threshold_selected_on_validation"]
        ),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Frozen calibration contains non-finite values")
    if any(result[key] <= 0 for key in (
        "temperature", "paratope_temperature", "epitope_temperature"
    )):
        raise ValueError("Frozen temperatures must be positive")
    return result


def _add_bootstrap(
    metrics: dict[str, Any],
    dataset: Any,
    *,
    replicates: int,
    seed: int,
) -> None:
    for index, direction_metrics in enumerate(
        metrics["partner_retrieval_by_direction"].values()
    ):
        direction_metrics["cluster_bootstrap_ci"] = (
            bootstrap_cluster_weighted_margin(
                direction_metrics["cluster_margins"],
                replicates=replicates,
                seed=seed + index,
            )
        )
    records = {
        str(record["instance_id"]): record for record in dataset.records
    }
    for index, (metric_name, group_field) in enumerate(
        (
            ("contact", "antigen_cluster_id"),
            ("epitope", "antigen_cluster_id"),
            ("paratope", "antibody_cluster_id"),
        )
    ):
        group_by_instance = {
            identifier: str(record[group_field])
            for identifier, record in records.items()
        }
        metrics[metric_name]["mean_per_complex_auprc_cluster_bootstrap_ci"] = (
            bootstrap_group_weighted_values(
                metrics[metric_name]["per_complex_auprc_by_instance"],
                group_by_instance,
                replicates=replicates,
                seed=seed + 1_000 * (index + 1),
            )
        )
        metrics[metric_name]["mean_per_complex_auprc_cluster_definition"] = (
            f"equal-weight mean over {group_field} means; zero-positive complexes "
            "are ineligible for positive-required per-complex AP but remain in pooled metrics"
        )


def _generator(device: torch.device, seed: int) -> torch.Generator:
    location = "cuda" if device.type == "cuda" else "cpu"
    return torch.Generator(device=location).manual_seed(seed)


def _metric_snapshot(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {
        "selection_metric": float(metrics["selection_metric"]),
        "contact_pooled_auprc": float(metrics["contact"]["auprc"]),
        "contact_mean_per_complex_auprc": float(
            metrics["contact"]["mean_per_complex_auprc"]
        ),
        "paratope_pooled_auprc": float(metrics["paratope"]["auprc"]),
        "epitope_pooled_auprc": float(metrics["epitope"]["auprc"]),
    }
    for direction, values in metrics["partner_retrieval_by_direction"].items():
        result[f"{direction}_margin"] = float(
            values["cluster_weighted_cognate_compatibility_margin"]
        )
        result[f"{direction}_top1"] = float(values["top1_retrieval_accuracy"])
        result[f"{direction}_mrr"] = float(values["mean_reciprocal_rank"])
    return result


def _intervention_delta(
    clean: Mapping[str, Any], intervened: Mapping[str, Any]
) -> dict[str, float]:
    clean_values = _metric_snapshot(clean)
    intervention_values = _metric_snapshot(intervened)
    if set(clean_values) != set(intervention_values):
        raise ValueError("Intervention and clean metric schemas differ")
    return {
        key: intervention_values[key] - clean_values[key]
        for key in sorted(clean_values)
    }


def _authenticated_same_antigen_challenge(
    path: Path,
    expected_sha256: str,
    *,
    split_sha256: str,
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Same-antigen challenge SHA-256 differs")
    panel = json.loads(path.read_text(encoding="utf-8"))
    authority = panel.get("authority", {})
    if (
        panel.get("format_version") != LABEL_FREE_SAME_ANTIGEN_CHALLENGE_FORMAT
        or panel.get("role") != "test"
        or authority.get("split_sha256") != split_sha256
        or authority.get("task_labels_read") is not False
        or authority.get("model_predictions_read") is not False
        or authority.get("same_antigen_rule") != "exact antigen_sequence_sha256"
        or not panel.get("entries")
    ):
        raise ValueError("Same-antigen challenge is not a test-eligible authority")
    focal_ids = [entry.get("focal_instance_id") for entry in panel["entries"]]
    if len(focal_ids) != len(set(focal_ids)):
        raise ValueError("Same-antigen challenge contains duplicate focal instances")
    for entry in panel["entries"]:
        alternates = entry.get("alternates", [])
        if not alternates or any(
            not alternate.get("instance_id")
            or alternate["instance_id"] == entry["focal_instance_id"]
            for alternate in alternates
        ):
            raise ValueError("Same-antigen challenge contains an invalid swap")
    return panel


@torch.no_grad()
def _evaluate_same_antigen_challenge(
    model: torch.nn.Module,
    dataset: Any,
    panel: Mapping[str, Any],
    device: torch.device,
    *,
    use_bf16: bool,
    maximum_epitope_jaccard: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if (
        not math.isfinite(maximum_epitope_jaccard)
        or not 0 <= maximum_epitope_jaccard < 1
    ):
        raise ValueError("Same-antigen maximum epitope Jaccard must lie in [0, 1)")
    index_by_id = {
        str(record["instance_id"]): index
        for index, record in enumerate(dataset.records)
    }
    referenced_ids = {
        str(entry["focal_instance_id"])
        for entry in panel["entries"]
    }
    referenced_ids.update(
        str(alternate["instance_id"])
        for entry in panel["entries"]
        for alternate in entry["alternates"]
    )
    if not referenced_ids.issubset(index_by_id):
        raise ValueError("Same-antigen challenge references a non-test instance")
    item_by_id = {
        identifier: dataset[index_by_id[identifier]]
        for identifier in sorted(referenced_ids)
    }
    collator = Sabdab2StructureBatchCollator()

    def epitope_probabilities(item: Mapping[str, Any]) -> np.ndarray:
        raw_batch = collator([item])
        batch = _move_batch(raw_batch, device)
        with _autocast(device, use_bf16):
            output = model(batch, return_attention=False)
        mask = output["antigen_mask"][0]
        return (
            torch.sigmoid(output["epitope_logits"][0][mask].float())
            .cpu()
            .numpy()
        )

    cognate_probabilities = {
        str(entry["focal_instance_id"]): epitope_probabilities(
            item_by_id[str(entry["focal_instance_id"])]
        )
        for entry in panel["entries"]
    }
    rows = []
    for entry in panel["entries"]:
        focal_id = str(entry["focal_instance_id"])
        focal = item_by_id[focal_id]
        focal_labels = focal["antigen_labels"].detach().cpu().numpy()
        for alternate in entry["alternates"]:
            alternate_id = str(alternate["instance_id"])
            alternate_item = item_by_id[alternate_id]
            if (
                focal["antigen_features"].shape
                != alternate_item["antigen_features"].shape
                or not torch.equal(
                    focal["antigen_features"],
                    alternate_item["antigen_features"],
                )
                or not torch.equal(
                    focal["antigen_chain_ids"],
                    alternate_item["antigen_chain_ids"],
                )
            ):
                rows.append(
                    {
                        "antigen_sequence_sha256": str(
                            entry["antigen_sequence_sha256"]
                        ),
                        "focal_instance_id": focal_id,
                        "alternate_instance_id": alternate_id,
                        "eligible": False,
                        "ineligible_reason": "antigen_residue_axis_mismatch",
                    }
                )
                continue
            alternate_labels = (
                alternate_item["antigen_labels"].detach().cpu().numpy()
            )
            swapped = _wrong_antibody_pair_item(focal, alternate_item)
            swapped_probabilities = epitope_probabilities(swapped)
            scores = score_same_antigen_pair(
                cognate_probabilities[focal_id],
                swapped_probabilities,
                focal_labels,
                alternate_labels,
            )
            rows.append(
                {
                    "antigen_sequence_sha256": str(
                        entry["antigen_sequence_sha256"]
                    ),
                    "focal_instance_id": focal_id,
                    "alternate_instance_id": alternate_id,
                    **scores,
                }
            )

    eligible = [row for row in rows if row["eligible"]]
    primary = [
        row
        for row in eligible
        if float(row["epitope_jaccard"]) <= maximum_epitope_jaccard
    ]
    high_overlap = [
        row
        for row in eligible
        if float(row["epitope_jaccard"]) > maximum_epitope_jaccard
    ]
    ineligible = [row for row in rows if not row["eligible"]]
    if not primary:
        raise ValueError("Same-antigen primary stratum is empty")
    return {
        "definition": {
            "fixed_input": "focal antigen sequence, ESM features, and monomer geometry",
            "intervention": "replace only the antibody with an exact-antigen alternate",
            "primary_stratum": (
                "both complexes have at least one epitope residue and true epitope "
                f"Jaccard <= {maximum_epitope_jaccard:g}"
            ),
            "directed_assignment_margin": (
                "0.5 * ([AP(cognate map,focal target)-AP(cognate map,alternate "
                "target)] + [AP(swapped map,alternate target)-AP(swapped map,focal "
                "target)])"
            ),
            "success_requires": (
                "both cognate and swapped assignment margins are positive; map change "
                "alone is insufficient"
            ),
            "aggregation": "one equal vote per exact-antigen sequence group",
        },
        "authority_counts": {
            "focal_instance_count": int(panel["focal_instance_count"]),
            "ordered_pair_count": int(panel["ordered_pair_count"]),
        },
        "scored_ordered_pair_count": len(rows),
        "primary_ordered_pair_count": len(primary),
        "high_overlap_secondary_ordered_pair_count": len(high_overlap),
        "ineligible_ordered_pair_count": len(ineligible),
        "ineligible_pairs": [
            {
                "focal_instance_id": row["focal_instance_id"],
                "alternate_instance_id": row["alternate_instance_id"],
                "reason": row["ineligible_reason"],
            }
            for row in ineligible
        ],
        "primary_metrics": aggregate_same_antigen_pair_metrics(primary),
        "primary_group_bootstrap_ci": bootstrap_same_antigen_group_metrics(
            primary,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "all_positive_eligible_secondary_metrics": (
            aggregate_same_antigen_pair_metrics(eligible) if eligible else None
        ),
        "ordered_pair_rows": rows,
    }


@torch.no_grad()
def _independent_se3_prediction_invariance(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    use_bf16: bool,
    seed: int,
    probability_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Compare every clean prediction against independently transformed monomers."""
    if not math.isfinite(probability_tolerance) or probability_tolerance <= 0:
        raise ValueError("Invariance tolerance must be finite and positive")
    generator = _generator(device, seed)
    accumulators = {
        name: {"absolute_sum": 0.0, "maximum": 0.0, "count": 0}
        for name in (
            "contact_probability",
            "interaction_only_probability",
            "paratope_probability",
            "epitope_probability",
        )
    }
    for raw_batch in loader:
        clean_batch = _move_batch(raw_batch, device)
        transformed_batch = independent_monomer_rigid_transform(
            clean_batch, generator=generator
        )
        with _autocast(device, use_bf16):
            clean = model(clean_batch, return_attention=False)
            transformed = model(transformed_batch, return_attention=False)
        for mask_name in ("pair_candidate_mask", "antibody_mask", "antigen_mask"):
            if not torch.equal(clean[mask_name], transformed[mask_name]):
                raise ValueError("Independent SE(3) transform changed an evaluation mask")
        fields = (
            ("contact_probability", "contact_logits", "pair_candidate_mask"),
            (
                "interaction_only_probability",
                "contact_interaction_logits",
                "pair_candidate_mask",
            ),
            ("paratope_probability", "paratope_logits", "antibody_mask"),
            ("epitope_probability", "epitope_logits", "antigen_mask"),
        )
        for name, logit_name, mask_name in fields:
            mask = clean[mask_name]
            clean_probability = torch.sigmoid(clean[logit_name].float())[mask]
            transformed_probability = torch.sigmoid(
                transformed[logit_name].float()
            )[mask]
            absolute = (clean_probability - transformed_probability).abs()
            accumulators[name]["absolute_sum"] += float(absolute.sum().cpu())
            accumulators[name]["maximum"] = max(
                accumulators[name]["maximum"],
                float(absolute.max().cpu()) if absolute.numel() else 0.0,
            )
            accumulators[name]["count"] += int(absolute.numel())
    metrics = {
        name: {
            "max_absolute_probability_difference": values["maximum"],
            "mean_absolute_probability_difference": (
                values["absolute_sum"] / values["count"]
            ),
            "prediction_count": values["count"],
        }
        for name, values in accumulators.items()
    }
    maximum = max(
        value["max_absolute_probability_difference"] for value in metrics.values()
    )
    return {
        "definition": (
            "Apply independent random proper rotations and translations to the "
            "antibody and antigen monomer coordinates while preserving all labels "
            "and sequence features; compare every clean probability elementwise."
        ),
        "seed": seed,
        "probability_tolerance": probability_tolerance,
        "overall_max_absolute_probability_difference": maximum,
        "passed": maximum <= probability_tolerance,
        "metrics": metrics,
    }


def main() -> None:
    arguments = _arguments()
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError("Refusing to overwrite frozen-test output")
    if arguments.panel_size != 31:
        raise ValueError("SAbDab2 frozen test requires exactly 31 wrong partners")
    if (
        arguments.bootstrap_replicates != 10_000
        or arguments.bootstrap_seed != 20_260_730
        or arguments.skip_interventions
    ):
        raise ValueError("SAbDab2 frozen-test confirmatory diagnostics are immutable")
    if (
        tuple(arguments.coordinate_noise) != (0.10, 0.25, 0.50)
        or not math.isclose(
            arguments.same_antigen_maximum_epitope_jaccard,
            0.50,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("SAbDab2 frozen-test intervention values are immutable")
    opening = authenticate_sabdab2_test_opening(
        arguments.opening_receipt,
        arguments.opening_receipt_sha256,
    )
    selection = authenticate_sabdab2_development_selection(
        arguments.selection_root,
        selection_receipt_sha256=arguments.selection_receipt_sha256,
    )
    if (
        opening.get("selection_receipt_sha256")
        != arguments.selection_receipt_sha256
        or Path(selection["selected_ensemble_root"]).resolve()
        != arguments.ensemble_root.resolve()
        or opening.get("selected_ensemble_result_sha256")
        != selection["selected_ensemble_result_sha256"]
    ):
        raise ValueError("SAbDab2 opening differs from the frozen selection")
    for path, expected, name in (
        (
            arguments.structure_cache_root / "STRUCTURE_CACHE_MANIFEST.json",
            arguments.structure_manifest_sha256,
            "structure manifest",
        ),
        (
            arguments.contact_cache_root / "CONTACT_CACHE_MANIFEST.json",
            arguments.contact_manifest_sha256,
            "contact manifest",
        ),
        (
            arguments.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json",
            arguments.embedding_manifest_sha256,
            "ESM embedding manifest",
        ),
        (arguments.split_manifest, arguments.split_manifest_sha256, "split manifest"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{name} SHA-256 differs")
    panel = _authenticated_panel(
        arguments.test_partner_panel,
        arguments.test_partner_panel_sha256,
        role="test",
        expected_size=arguments.panel_size,
        expected_format=RECIPROCAL_PARTNER_PANEL_FORMAT,
    )
    same_antigen_panel = _authenticated_same_antigen_challenge(
        arguments.same_antigen_challenge,
        arguments.same_antigen_challenge_sha256,
        split_sha256=arguments.split_manifest_sha256,
    )
    ensemble_result, ensemble_receipt = _authenticated_ensemble(
        arguments.ensemble_root
    )
    if ensemble_receipt.get("result_sha256") != selection[
        "selected_ensemble_result_sha256"
    ]:
        raise ValueError("SAbDab2 ensemble differs from the frozen selection")
    members = [_authenticated_run(path) for path in arguments.run_roots]
    members.sort(key=lambda member: int(member["checkpoint"]["seed"]))
    seeds = tuple(int(member["checkpoint"]["seed"]) for member in members)
    if seeds != FROZEN_SEEDS:
        raise ValueError("Test scorer requires the five pre-registered seeds")
    expected_members = {
        int(member["seed"]): member
        for member in ensemble_result["authority"]["members"]
    }
    for member in members:
        seed = int(member["checkpoint"]["seed"])
        expected = expected_members.get(seed)
        if (
            expected is None
            or expected["development_result_sha256"] != member["result_sha256"]
            or expected["selected_checkpoint_sha256"]
            != member["checkpoint_sha256"]
        ):
            raise ValueError("Test member differs from frozen ensemble authority")
        opening_member = {
            int(value["seed"]): value for value in opening.get("members", [])
        }.get(seed)
        if (
            opening_member is None
            or opening_member.get("development_result_sha256")
            != member["result_sha256"]
            or opening_member.get("selected_checkpoint_sha256")
            != member["checkpoint_sha256"]
        ):
            raise ValueError("Test member differs from the opening receipt")
    authority = ensemble_result["authority"]
    if (
        authority["structure_manifest_sha256"]
        != arguments.structure_manifest_sha256
        or authority["contact_manifest_sha256"]
        != arguments.contact_manifest_sha256
        or authority["embedding_manifest_sha256"]
        != arguments.embedding_manifest_sha256
        or authority["split_manifest_sha256"]
        != arguments.split_manifest_sha256
    ):
        raise ValueError("Frozen ensemble and requested test authorities differ")
    for key, observed in (
        ("structure_manifest_sha256", arguments.structure_manifest_sha256),
        ("contact_manifest_sha256", arguments.contact_manifest_sha256),
        ("embedding_manifest_sha256", arguments.embedding_manifest_sha256),
        ("split_manifest_sha256", arguments.split_manifest_sha256),
        ("test_partner_panel_sha256", arguments.test_partner_panel_sha256),
        ("same_antigen_challenge_sha256", arguments.same_antigen_challenge_sha256),
    ):
        if opening.get(key) != observed:
            raise ValueError("SAbDab2 evaluator authority differs from the opening")
    experiment = str(ensemble_result["experiment"])
    if any(member["result"]["experiment"] != experiment for member in members):
        raise ValueError("Test member experiment differs from frozen ensemble")

    device = _device(arguments.device)
    use_bf16 = arguments.precision == "bf16" and device.type == "cuda"
    if arguments.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 requires CUDA")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise ValueError("CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG")
    random.seed(arguments.bootstrap_seed)
    np.random.seed(arguments.bootstrap_seed)
    torch.manual_seed(arguments.bootstrap_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(arguments.bootstrap_seed)

    # Consume the only scoring session before the first test-role item is loaded.
    score_consumption_token = consume_sabdab2_score_once(
        opening,
        opening_sha256=arguments.opening_receipt_sha256,
        output_root=arguments.output_root,
    )
    # This is the only point at which the authenticated test role is opened.
    base = Sabdab2StructureEmbeddingDataset(
        arguments.structure_cache_root,
        arguments.embedding_root,
        None,
        split_role="test",
        split_path=arguments.split_manifest,
        split_sha256=arguments.split_manifest_sha256,
    )
    dataset = Sabdab2ContactDataset(base, arguments.contact_cache_root)
    if len(dataset) != int(opening["test_instance_count"]):
        raise ValueError(
            "Authenticated SAbDab2 test instance count differs from the opening"
        )
    loader = _loader(
        dataset,
        batch_size=arguments.evaluation_batch_size,
        workers=arguments.workers,
        shuffle=False,
        seed=arguments.bootstrap_seed,
        pin_memory=device.type == "cuda",
    )
    first = dataset[0]
    models = []
    member_calibrations = []
    member_metrics = {}
    specificity_source = None
    for member in members:
        seed = int(member["checkpoint"]["seed"])
        model = _model_from_configuration(
            first,
            member["result"]["configuration"],
            experiment,
        )
        model.load_state_dict(member["checkpoint"]["model_state_dict"])
        model.to(device).eval()
        local_source = str(member["checkpoint"]["specificity_score_source"])
        if specificity_source is None:
            specificity_source = local_source
        elif local_source != specificity_source:
            raise ValueError("Member specificity score sources differ")
        configuration = member["result"]["configuration"]
        metrics = _evaluate(
            model,
            loader,
            dataset,
            panel,
            device,
            use_bf16=use_bf16,
            evaluation_batch_size=arguments.evaluation_batch_size,
            score_fraction=float(configuration["score_fraction"]),
            seed=seed,
            calibration_max_pairs=int(configuration["calibration_max_pairs"]),
            specificity_score_source=local_source,
            collect_attention=True,
            frozen_calibration=_frozen_calibration(
                member["result"]["selected_checkpoint_validation"]
            ),
            compute_calibration_regression=True,
        )
        _add_bootstrap(
            metrics,
            dataset,
            replicates=arguments.bootstrap_replicates,
            seed=arguments.bootstrap_seed + 100 * seed,
        )
        member_metrics[str(seed)] = metrics
        models.append(model)
        member_calibrations.append(_member_calibration(member["result"]))
    assert specificity_source is not None
    ensemble = CalibratedProbabilityEnsemble(
        models, member_calibrations
    ).to(device).eval()
    reference_configuration = members[0]["result"]["configuration"]
    ensemble_metrics = _evaluate(
        ensemble,
        loader,
        dataset,
        panel,
        device,
        use_bf16=use_bf16,
        evaluation_batch_size=arguments.evaluation_batch_size,
        score_fraction=float(reference_configuration["score_fraction"]),
        seed=arguments.bootstrap_seed,
        calibration_max_pairs=int(reference_configuration["calibration_max_pairs"]),
        specificity_score_source=specificity_source,
        collect_attention=False,
        frozen_calibration=_frozen_calibration(ensemble_result["validation"]),
        compute_calibration_regression=True,
    )
    _add_bootstrap(
        ensemble_metrics,
        dataset,
        replicates=arguments.bootstrap_replicates,
        seed=arguments.bootstrap_seed,
    )
    same_antigen_metrics = _evaluate_same_antigen_challenge(
        ensemble,
        dataset,
        same_antigen_panel,
        device,
        use_bf16=use_bf16,
        maximum_epitope_jaccard=(
            arguments.same_antigen_maximum_epitope_jaccard
        ),
        bootstrap_replicates=arguments.bootstrap_replicates,
        bootstrap_seed=arguments.bootstrap_seed + 50_000,
    )
    independent_se3_invariance = _independent_se3_prediction_invariance(
        ensemble,
        loader,
        device,
        use_bf16=use_bf16,
        seed=arguments.bootstrap_seed + 60_000,
    )
    intervention_metrics: dict[str, Any] = {}
    if not arguments.skip_interventions:
        intervention_factories = [
            (
                "geometry_shuffle",
                lambda generator: lambda batch: geometry_shuffle(
                    batch, generator=generator
                ),
            ),
            (
                "local_frame_corruption",
                lambda generator: lambda batch: corrupt_local_frames(
                    batch, generator=generator
                ),
            ),
            (
                "degree_matched_graph_rewiring",
                lambda generator: lambda batch: degree_matched_graph_rewiring(
                    batch,
                    generator=generator,
                    neighbor_k=int(reference_configuration["neighbor_k"]),
                    neighbor_radius=float(reference_configuration["neighbor_radius"]),
                ),
            ),
        ]
        intervention_factories.extend(
            (
                f"coordinate_noise_{standard_deviation:g}_angstrom",
                lambda generator, standard_deviation=standard_deviation: (
                    lambda batch: perturb_coordinates(
                        batch,
                        generator=generator,
                        standard_deviation=standard_deviation,
                    )
                ),
            )
            for standard_deviation in arguments.coordinate_noise
        )
        for index, (name, factory) in enumerate(intervention_factories):
            local_seed = arguments.bootstrap_seed + 10_000 + index
            metrics = _evaluate(
                ensemble,
                loader,
                dataset,
                panel,
                device,
                use_bf16=use_bf16,
                evaluation_batch_size=arguments.evaluation_batch_size,
                score_fraction=float(reference_configuration["score_fraction"]),
                seed=local_seed,
                calibration_max_pairs=int(
                    reference_configuration["calibration_max_pairs"]
                ),
                specificity_score_source=specificity_source,
                collect_attention=False,
                frozen_calibration=_frozen_calibration(ensemble_result["validation"]),
                batch_intervention=factory(_generator(device, local_seed)),
            )
            _add_bootstrap(
                metrics,
                dataset,
                replicates=arguments.bootstrap_replicates,
                seed=local_seed,
            )
            intervention_metrics[name] = {
                "metrics": metrics,
                "delta_vs_clean": _intervention_delta(ensemble_metrics, metrics),
            }
    payload = {
        "format_version": TEST_FORMAT,
        "claim_boundary": (
            "Single frozen SAbDab2 test opening for all five validation-selected "
            "members and their validation-frozen ensemble; no test selection or fitting."
        ),
        "experiment": experiment,
        "seeds": list(seeds),
        "authority": {
            "opening_receipt_sha256": arguments.opening_receipt_sha256,
            "selection_receipt_sha256": arguments.selection_receipt_sha256,
            "development_ensemble_result_sha256": ensemble_receipt["result_sha256"],
            "structure_manifest_sha256": arguments.structure_manifest_sha256,
            "contact_manifest_sha256": arguments.contact_manifest_sha256,
            "embedding_manifest_sha256": arguments.embedding_manifest_sha256,
            "split_manifest_sha256": arguments.split_manifest_sha256,
            "test_partner_panel_sha256": arguments.test_partner_panel_sha256,
            "same_antigen_challenge_sha256": (
                arguments.same_antigen_challenge_sha256
            ),
            "validation_calibration_frozen": True,
            "selection_performed_on_test": False,
            "score_consumption_token_sha256": sha256_file(
                score_consumption_token
            ),
            "test_score_session_count": 1,
            "required_interventions_complete": True,
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
        },
        "test_complexes": len(dataset),
        "per_seed_metrics": member_metrics,
        "ensemble_metrics": ensemble_metrics,
        "same_antigen_different_antibody": same_antigen_metrics,
        "independent_monomer_se3_prediction_invariance": (
            independent_se3_invariance
        ),
        "intervention_metrics": intervention_metrics,
        "intervention_definition": {
            "partner_counterfactual": (
                "the two frozen reciprocal 31-wrong-partner panels in ensemble_metrics"
            ),
            "geometry_shuffle": "permute residue frames while preserving sequence and labels",
            "local_frame_corruption": "independently rotate N/C offsets around fixed CA",
            "degree_matched_graph_rewiring": (
                "random same-chain off-diagonal neighbors with every query degree preserved"
            ),
            "coordinate_noise_angstrom": list(arguments.coordinate_noise),
            "independent_monomer_se3_transform": (
                "elementwise exact-invariance control on every clean test prediction; "
                "antibody and antigen receive separate rigid transforms"
            ),
        },
    }
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.tmp-",
            dir=arguments.output_root.parent,
        )
    )
    try:
        result_path = temporary / "FRAMECONTACT_FROZEN_ENSEMBLE_TEST_RESULT.json"
        result_path.write_bytes(canonical_json_bytes(payload))
        receipt = {
            "format_version": TEST_FORMAT,
            "result_sha256": sha256_file(result_path),
            "test_loaded": True,
            "opening_receipt_sha256": arguments.opening_receipt_sha256,
            "selection_receipt_sha256": arguments.selection_receipt_sha256,
            "test_score_session_count": 1,
            "selection_performed_on_test": False,
            "same_antigen_challenge_sha256": (
                arguments.same_antigen_challenge_sha256
            ),
            "required_interventions_complete": True,
        }
        (temporary / "FROZEN_ENSEMBLE_TEST_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, arguments.output_root)
        print(json.dumps({**receipt, "output_root": str(arguments.output_root)}, sort_keys=True))
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
