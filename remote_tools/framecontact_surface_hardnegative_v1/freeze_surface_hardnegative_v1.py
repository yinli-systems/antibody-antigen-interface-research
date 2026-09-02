#!/usr/bin/env python3
"""Freeze the development-only surface-prior and hard-negative study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


REFERENCE_CAMPAIGN_SHA256 = (
    "9eddf93aca94c1018287620e3720f873ece3730d9e3fe85d340b372a9ef0b4d0"
)
REFERENCE_ANALYSIS_SHA256 = (
    "1382c6fc18624b53f50657c0c0d61408fa03f0acc5aa3a812ffa3f84bd23b78c"
)
STRUCTURE_MANIFEST_SHA256 = (
    "bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614"
)
CONTACT_MANIFEST_SHA256 = (
    "d772a8dce8181d161318135a1bf4947506210363bb8ed3527bfa41dc7131ad88"
)
CONTACT_RECEIPT_SHA256 = (
    "426975feebdb5294b0fdbc45d31422fb3adbadc4d97b032958a38d19b050f1d9"
)
SPLIT_SHA256 = (
    "ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700"
)
SEEDS = (17, 29, 43)
INTERVENTIONS = {
    "surface": {
        "surface_prior_mode": "ca_density",
        "surface_radius_angstrom": 10.0,
        "contact_hard_negative": 0.0,
    },
    "hard_negative": {
        "surface_prior_mode": "none",
        "surface_radius_angstrom": 10.0,
        "contact_hard_negative": 0.25,
    },
    "combined": {
        "surface_prior_mode": "ca_density",
        "surface_radius_angstrom": 10.0,
        "contact_hard_negative": 0.25,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> str:
    path.write_bytes(canonical_bytes(payload))
    return sha256_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-source-root", type=Path, required=True)
    parser.add_argument("--baseline-source-manifest", type=Path, required=True)
    parser.add_argument("--topk2-source-root", type=Path, required=True)
    parser.add_argument("--topk2-source-manifest", type=Path, required=True)
    parser.add_argument("--training-launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--canary-launcher", type=Path, required=True)
    parser.add_argument("--canary-test", type=Path, required=True)
    parser.add_argument("--reused-canary-job-id")
    return parser.parse_args()


def verify_reference(reference: Path) -> tuple[Path, Path, dict[str, Any]]:
    authority = reference / "authority" / "no_privileged_distance_v1"
    campaign_path = authority / "CAMPAIGN_NO_PRIVILEGED_DISTANCE_V1.json"
    analysis_path = reference / "PREDICTED_GEOMETRY_NO_PRIVILEGED_DISTANCE_ANALYSIS_V1.json"
    if sha256_file(campaign_path) != REFERENCE_CAMPAIGN_SHA256:
        raise ValueError("no-distance reference campaign differs")
    if sha256_file(analysis_path) != REFERENCE_ANALYSIS_SHA256:
        raise ValueError("no-distance reference analysis differs")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if (
        campaign.get("status") != "FROZEN_BEFORE_NO_PRIVILEGED_DISTANCE_TRAINING_V1"
        or campaign.get("model_input_structure_manifest_sha256")
        != STRUCTURE_MANIFEST_SHA256
        or campaign.get("contact_manifest_sha256") != CONTACT_MANIFEST_SHA256
        or campaign.get("development_split_sha256") != SPLIT_SHA256
        or campaign.get("distance_labels_loaded") is not False
        or campaign.get("test_loaded") is not False
        or campaign.get("temporal_holdout_loaded") is not False
        or analysis.get("status")
        != "PASS_PREDICTED_GEOMETRY_NO_BOUND_DISTANCE_AUXILIARY_DEVELOPMENT"
        or analysis.get("old_sabdab2_test_loaded") is not False
        or analysis.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("no-distance reference authority differs")
    return authority, analysis_path, campaign


def protocol_for(
    template: dict[str, Any],
    *,
    checkpoint_arm: str,
    intervention: str,
    source_manifest_sha256: str,
    execution_sha256: dict[str, str],
) -> dict[str, Any]:
    settings = INTERVENTIONS[intervention]
    protocol = json.loads(json.dumps(template))
    protocol["schema_version"] = (
        "sabdab2_predicted_geometry_surface_hardnegative_development_v1_"
        f"{checkpoint_arm}_{intervention}"
    )
    protocol["protocol_status"] = "FROZEN_BEFORE_SURFACE_HARDNEGATIVE_TRAINING_V1"
    protocol["arm"] = checkpoint_arm
    protocol["development_intervention"] = intervention

    implementation = protocol["implementation"]
    implementation.update(
        {
            "surface_prior_mode": settings["surface_prior_mode"],
            "surface_radius_angstrom": settings["surface_radius_angstrom"],
            "contact_hard_negative_fraction": 0.01,
            "contact_hard_negative_maximum_pairs": 64,
            "contact_hard_negative_margin": 0.5,
        }
    )
    protocol["training_objective"]["loss_weights"]["contact_hard_negative"] = (
        settings["contact_hard_negative"]
    )
    protocol["training_objective"]["contact_hard_negative"] = {
        "scope": "within_complex_valid_pair_candidates",
        "negative_selection": "highest_logit_top_one_percent_capped_at_64",
        "positive_selection": "lowest_logit_capped_at_64",
        "pairwise_penalty": "softplus(margin_plus_negative_logit_minus_positive_logit)",
        "margin": 0.5,
        "zero_positive_or_zero_negative_complex": "zero_contribution",
    }
    protocol["surface_prior"] = {
        "mode": settings["surface_prior_mode"],
        "input": "antigen_predicted_CA_coordinates_and_physical_chain_ids_only",
        "chain_scope": "same_physical_chain_only",
        "density_kernel": "exp(-(distance/10A)^2)_excluding_self",
        "normalization": "per_chain_mean_density_minus_density_over_population_std_clip_3",
        "injection": "zero_initialized_scalar_gate_added_to_pair_contact_logits_by_antigen_residue",
        "cross_partner_geometry_used": False,
        "task_labels_used_to_construct_prior": False,
    }
    protocol["claim_boundary"].update(
        {
            "claim": (
                "Adaptive SAbDab2 development-only comparison of a predicted-"
                "surface prior and within-complex hard-negative contact ranking."
            ),
            "external_sable_v0_2_0_was_exposed_before_design": True,
            "external_sable_read_or_evaluated_by_this_study": False,
            "external_sable_used_for_checkpoint_or_hyperparameter_selection": False,
            "requires_new_prospective_external_confirmation": True,
            "independent_confirmation": False,
            "selection_or_tuning_from_this_study": True,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
            "test_loaded": False,
            "experimental_bound_geometry_used_as_model_input": False,
            "experimental_bound_geometry_used_for_auxiliary_training_supervision": False,
            "experimental_bound_structure_used_to_define_task_contact_labels": True,
        }
    )
    protocol.pop("no_privileged_distance_v1", None)
    protocol["surface_hardnegative_v1"] = {
        **execution_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "reference_campaign_sha256": REFERENCE_CAMPAIGN_SHA256,
        "reference_analysis_sha256": REFERENCE_ANALYSIS_SHA256,
        "intervention": intervention,
        "checkpoint_arm": checkpoint_arm,
        "training_data_or_split_changed": False,
        "pair_contact_labels_changed": False,
        "optimizer_or_schedule_changed": False,
        "distance_label_cache_supplied": False,
        "distance_labels_loaded": False,
    }
    return protocol


def main() -> None:
    args = parse_args()
    reference_authority, analysis_path, reference_campaign = verify_reference(
        args.reference_campaign_root
    )
    authority = args.output_root / "authority"
    if authority.exists() or authority.is_symlink():
        raise FileExistsError(authority)

    source_rows = {
        "baseline": (args.baseline_source_root, args.baseline_source_manifest),
        "edge_topk2_supervised": (args.topk2_source_root, args.topk2_source_manifest),
    }
    source_sha256: dict[str, str] = {}
    for arm, (source_root, manifest_path) in source_rows.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_root") != str(source_root):
            raise ValueError(f"source manifest root differs for {arm}")
        source_sha256[arm] = sha256_file(manifest_path)

    execution_sha256 = {
        "freezer_sha256": sha256_file(Path(__file__)),
        "training_launcher_sha256": sha256_file(args.training_launcher),
        "analyzer_sha256": sha256_file(args.analyzer),
        "canary_launcher_sha256": sha256_file(args.canary_launcher),
        "canary_test_sha256": sha256_file(args.canary_test),
    }
    authority.mkdir(parents=True)
    copied_source: dict[str, Path] = {}
    for arm, (_, manifest_path) in source_rows.items():
        destination = authority / f"SOURCE_MANIFEST_{arm.upper()}_V1.json"
        shutil.copyfile(manifest_path, destination)
        copied_source[arm] = destination

    protocol_rows: dict[str, dict[str, str]] = {}
    for checkpoint_arm in ("baseline", "edge_topk2_supervised"):
        reference_name = (
            "PROTOCOL_PREDICTED_GEOMETRY_BASELINE_NO_PRIVILEGED_DISTANCE_V1.json"
            if checkpoint_arm == "baseline"
            else "PROTOCOL_PREDICTED_GEOMETRY_TOPK2_NO_PRIVILEGED_DISTANCE_V1.json"
        )
        template = json.loads(
            (reference_authority / reference_name).read_text(encoding="utf-8")
        )
        for intervention in INTERVENTIONS:
            key = f"{checkpoint_arm}:{intervention}"
            protocol_path = authority / (
                f"PROTOCOL_{checkpoint_arm.upper()}_{intervention.upper()}_V1.json"
            )
            protocol_sha = write_json(
                protocol_path,
                protocol_for(
                    template,
                    checkpoint_arm=checkpoint_arm,
                    intervention=intervention,
                    source_manifest_sha256=source_sha256[checkpoint_arm],
                    execution_sha256=execution_sha256,
                ),
            )
            receipt_path = protocol_path.with_name(protocol_path.stem + "_RECEIPT.json")
            receipt_sha = write_json(
                receipt_path,
                {
                    "format_version": "framecontact_surface_hardnegative_protocol_receipt_v1",
                    "checkpoint_arm": checkpoint_arm,
                    "intervention": intervention,
                    "protocol_file_sha256": protocol_sha,
                    "source_manifest_sha256": source_sha256[checkpoint_arm],
                    "structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
                    "contact_manifest_sha256": CONTACT_MANIFEST_SHA256,
                    "development_split_sha256": SPLIT_SHA256,
                    "distance_labels_loaded": False,
                    "sable_v0_2_0_accessed": False,
                    "test_loaded": False,
                    "temporal_holdout_loaded": False,
                    "frozen_before_training": True,
                },
            )
            protocol_rows[key] = {
                "protocol_path": str(protocol_path),
                "protocol_sha256": protocol_sha,
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt_sha,
                "source_root": str(source_rows[checkpoint_arm][0]),
                "source_manifest_path": str(copied_source[checkpoint_arm]),
                "source_manifest_sha256": source_sha256[checkpoint_arm],
            }

    promotion_gates = {
        "contact_route": {
            "contact_median_delta_minimum": 0.003,
            "epitope_median_delta_minimum": 0.0,
            "paratope_median_delta_minimum": -0.005,
        },
        "epitope_route": {
            "contact_median_delta_minimum": -0.002,
            "epitope_median_delta_minimum": 0.01,
            "paratope_median_delta_minimum": -0.005,
        },
        "development_selection_priority": [
            "combined",
            "surface",
            "hard_negative",
        ],
        "status_if_gate_passes": "ELIGIBLE_FOR_NEW_PROSPECTIVE_EXTERNAL_CONFIRMATION_ONLY",
    }
    campaign_path = authority / "CAMPAIGN_SURFACE_HARDNEGATIVE_V1.json"
    campaign_sha = write_json(
        campaign_path,
        {
            "format_version": "framecontact_surface_hardnegative_campaign_v1",
            "status": "FROZEN_BEFORE_SURFACE_HARDNEGATIVE_TRAINING_V1",
            "output_root": str(args.output_root),
            "reference_campaign_root": str(args.reference_campaign_root),
            "reference_campaign_sha256": REFERENCE_CAMPAIGN_SHA256,
            "reference_analysis_path": str(analysis_path),
            "reference_analysis_sha256": REFERENCE_ANALYSIS_SHA256,
            "model_input_structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
            "contact_manifest_sha256": CONTACT_MANIFEST_SHA256,
            "contact_receipt_sha256": CONTACT_RECEIPT_SHA256,
            "development_split_sha256": SPLIT_SHA256,
            "seeds": list(SEEDS),
            "interventions": INTERVENTIONS,
            "protocols": protocol_rows,
            "execution_sources": execution_sha256,
            "reused_canary_job_id": args.reused_canary_job_id,
            "reused_canary_requires_all_tasks_completed_zero": (
                args.reused_canary_job_id is not None
            ),
            "promotion_gates": promotion_gates,
            "development_only": True,
            "full_validation_is_checkpoint_selection_data": True,
            "independent_confirmation": False,
            "external_sable_v0_2_0_was_exposed_before_design": True,
            "external_sable_read_or_evaluated_by_this_study": False,
            "external_sable_used_for_selection": False,
            "requires_new_prospective_external_confirmation": True,
            "distance_label_cache_supplied": False,
            "distance_labels_loaded": False,
            "experimental_bound_geometry_used_as_model_input": False,
            "experimental_bound_geometry_used_for_auxiliary_training_supervision": False,
            "experimental_bound_structure_used_to_define_task_contact_labels": True,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
            "test_loaded": False,
        },
    )
    checksum_paths = sorted(
        path for path in authority.iterdir() if path.name != "AUTHORITY_SHA256SUMS.txt"
    )
    (authority / "AUTHORITY_SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_SURFACE_HARDNEGATIVE_TRAINING_V1",
                "campaign_sha256": campaign_sha,
                "protocol_count": len(protocol_rows),
                "source_manifest_sha256": source_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
