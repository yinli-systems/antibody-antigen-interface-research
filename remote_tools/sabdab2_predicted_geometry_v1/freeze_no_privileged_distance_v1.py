#!/usr/bin/env python3
"""Freeze a development-only predicted-geometry study with no distance labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


UPSTREAM_CAMPAIGN_SHA256 = (
    "1117a6ac45bd0ab57063af52e55ad14948447513d461a959f5d984bc41f99099"
)
UPSTREAM_ANALYSIS_SHA256 = (
    "a4168a919e97b1344d718946fdd51305803773ae3355a3014ee2f05583a077b7"
)
BASELINE_PROTOCOL_SHA256 = (
    "1490cc0a68551e83e9a7837e7eb94a4ed061a84f856fb8566ab31d860b1f7020"
)
TOPK2_PROTOCOL_SHA256 = (
    "a781272a270d1189dd4cf81e90ba3b23c33b646e992ecdd608a10682c569f26b"
)
BASELINE_SOURCE_SHA256 = (
    "a320a5f60794ea386d355e8ab5ec895130d3b714596bc817e303eee589dbc7e8"
)
TOPK2_SOURCE_SHA256 = (
    "885449525b7f41b3b00511245ca6c0b0859336db959becc7680ad9cb6e872332"
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
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--training-launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    return parser.parse_args()


def _receipt(
    path: Path,
    *,
    arm: str,
    protocol_sha256: str,
    source_sha256: str,
) -> str:
    return write_json(
        path,
        {
            "format_version": "sabdab2_predicted_geometry_no_privileged_distance_protocol_receipt_v1",
            "arm": arm,
            "protocol_file_sha256": protocol_sha256,
            "source_manifest_sha256": source_sha256,
            "structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
            "contact_manifest_sha256": CONTACT_MANIFEST_SHA256,
            "development_split_sha256": SPLIT_SHA256,
            "distance_label_authority_supplied": False,
            "distance_loss_weight": 0.0,
            "frozen_before_training": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    )


def _new_protocol(
    template: dict[str, Any],
    *,
    arm: str,
    source_sha256: str,
    execution: dict[str, Any],
) -> dict[str, Any]:
    protocol = json.loads(json.dumps(template))
    protocol["schema_version"] = (
        "sabdab2_predicted_geometry_no_privileged_distance_development_v1"
        f"_{arm}"
    )
    protocol["protocol_status"] = "frozen_before_training_no_privileged_distance_v1"

    data = protocol["data"]
    for key in (
        "distance_label_manifest_sha256",
        "distance_label_receipt_sha256",
        "distance_label_source_bound_structure_manifest_sha256",
    ):
        data.pop(key, None)

    objective = protocol["training_objective"]
    objective["loss_weights"]["distance"] = 0.0
    objective.pop("distance_supervision", None)
    objective["distance_supervision_disabled"] = {
        "distance_label_cache_supplied": False,
        "distance_labels_loaded": False,
        "distance_loss_weight": 0.0,
    }

    winner = protocol.get("tasks", {}).get("winner", {})
    if "distance_loss_weight" in winner:
        winner["distance_loss_weight"] = 0.0
    if "factorial_design" in protocol:
        protocol["factorial_design"]["distance_weight_held_fixed"] = 0.0

    protocol["claim_boundary"].update(
        {
            "claim": (
                "Development-only sensitivity of the fixed FrameContact recipe to "
                "sequence-predicted chain geometry without bound-derived distance "
                "auxiliary supervision."
            ),
            "experimental_bound_geometry_used_as_model_input": False,
            "experimental_bound_geometry_used_for_auxiliary_training_supervision": False,
            "experimental_structure_required_at_inference": False,
            "experimental_bound_structure_used_to_define_task_contact_labels": True,
            "model_architecture_and_optimization_unchanged": True,
            "training_objective_changed_only_by_removing_distance_auxiliary_loss": True,
            "model_architecture_loss_and_optimization_unchanged": False,
            "selection_or_tuning_from_this_study": False,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
            "test_loaded": False,
        }
    )

    for key in (
        "distance_label_recovery_v2",
        "contact_authority_recovery",
        "source_authority",
        "execution_sources",
    ):
        protocol.pop(key, None)
    protocol["no_privileged_distance_v1"] = {
        **execution,
        "source_manifest_sha256": source_sha256,
        "upstream_campaign_sha256": UPSTREAM_CAMPAIGN_SHA256,
        "upstream_analysis_sha256": UPSTREAM_ANALYSIS_SHA256,
        "single_factor_change": "distance_loss_weight_0.1_to_0.0_and_no_distance_cache",
        "model_architecture_changed": False,
        "optimizer_or_schedule_changed": False,
        "training_data_or_split_changed": False,
        "pair_contact_labels_changed": False,
        "distance_labels_loaded": False,
    }
    return protocol


def main() -> None:
    args = parse_args()
    upstream = args.campaign_root / "authority" / "distance_label_recovery_v2"
    authority = args.campaign_root / "authority" / "no_privileged_distance_v1"
    if authority.exists() or authority.is_symlink():
        raise FileExistsError(authority)

    upstream_campaign = (
        upstream / "CAMPAIGN_PREDICTED_GEOMETRY_DISTANCE_LABEL_RECOVERY_V2.json"
    )
    upstream_analysis = (
        args.campaign_root
        / "PREDICTED_GEOMETRY_DEVELOPMENT_ANALYSIS_DISTANCE_LABEL_RECOVERY_V2.json"
    )
    baseline_template = (
        upstream / "PROTOCOL_PREDICTED_GEOMETRY_BASELINE_DISTANCE_LABEL_RECOVERY_V2.json"
    )
    topk2_template = (
        upstream / "PROTOCOL_PREDICTED_GEOMETRY_TOPK2_DISTANCE_LABEL_RECOVERY_V2.json"
    )
    baseline_source = upstream / "SOURCE_MANIFEST_BASELINE_DISTANCE_LABELS_V4.json"
    topk2_source = upstream / "SOURCE_MANIFEST_EDGE_TOPK2_DISTANCE_LABELS_V2.json"

    expected = {
        upstream_campaign: UPSTREAM_CAMPAIGN_SHA256,
        upstream_analysis: UPSTREAM_ANALYSIS_SHA256,
        baseline_template: BASELINE_PROTOCOL_SHA256,
        topk2_template: TOPK2_PROTOCOL_SHA256,
        baseline_source: BASELINE_SOURCE_SHA256,
        topk2_source: TOPK2_SOURCE_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"upstream authority differs: {path}")

    upstream_payload = json.loads(upstream_campaign.read_text(encoding="utf-8"))
    upstream_result = json.loads(upstream_analysis.read_text(encoding="utf-8"))
    if (
        upstream_payload.get("status")
        != "FROZEN_BEFORE_DISTANCE_LABEL_RECOVERY_TRAINING_V2"
        or upstream_payload.get("model_input_structure_manifest_sha256")
        != STRUCTURE_MANIFEST_SHA256
        or upstream_payload.get("contact_manifest_sha256") != CONTACT_MANIFEST_SHA256
        or upstream_payload.get("development_split_sha256") != SPLIT_SHA256
        or upstream_payload.get("test_loaded") is not False
        or upstream_payload.get("temporal_holdout_loaded") is not False
        or upstream_result.get("status")
        != "PASS_PREDICTED_GEOMETRY_DEVELOPMENT_SENSITIVITY_CORRECTED_DISTANCE_LABELS"
        or upstream_result.get("old_sabdab2_test_loaded") is not False
        or upstream_result.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("upstream campaign content differs")

    execution = {
        "freezer_sha256": sha256_file(Path(__file__)),
        "training_launcher_sha256": sha256_file(args.training_launcher),
        "analyzer_sha256": sha256_file(args.analyzer),
    }
    authority.mkdir(parents=True)
    baseline_source_copy = authority / "SOURCE_MANIFEST_BASELINE_NO_DISTANCE_V1.json"
    topk2_source_copy = authority / "SOURCE_MANIFEST_TOPK2_NO_DISTANCE_V1.json"
    shutil.copyfile(baseline_source, baseline_source_copy)
    shutil.copyfile(topk2_source, topk2_source_copy)

    protocol_rows: dict[str, dict[str, str]] = {}
    protocol_paths: list[Path] = []
    for arm, template_path, source_sha256 in (
        ("baseline", baseline_template, BASELINE_SOURCE_SHA256),
        ("edge_topk2_supervised", topk2_template, TOPK2_SOURCE_SHA256),
    ):
        protocol = _new_protocol(
            json.loads(template_path.read_text(encoding="utf-8")),
            arm=arm,
            source_sha256=source_sha256,
            execution=execution,
        )
        stem = (
            "PROTOCOL_PREDICTED_GEOMETRY_BASELINE_NO_PRIVILEGED_DISTANCE_V1"
            if arm == "baseline"
            else "PROTOCOL_PREDICTED_GEOMETRY_TOPK2_NO_PRIVILEGED_DISTANCE_V1"
        )
        protocol_path = authority / f"{stem}.json"
        protocol_sha256 = write_json(protocol_path, protocol)
        receipt_path = authority / f"{stem}_RECEIPT.json"
        receipt_sha256 = _receipt(
            receipt_path,
            arm=arm,
            protocol_sha256=protocol_sha256,
            source_sha256=source_sha256,
        )
        protocol_rows[arm] = {
            "protocol_sha256": protocol_sha256,
            "receipt_sha256": receipt_sha256,
            "source_manifest_sha256": source_sha256,
        }
        protocol_paths.extend((protocol_path, receipt_path))

    campaign = {
        "format_version": "sabdab2_predicted_geometry_no_privileged_distance_campaign_v1",
        "status": "FROZEN_BEFORE_NO_PRIVILEGED_DISTANCE_TRAINING_V1",
        "campaign_root": str(args.campaign_root),
        "upstream_campaign_sha256": UPSTREAM_CAMPAIGN_SHA256,
        "upstream_analysis_sha256": UPSTREAM_ANALYSIS_SHA256,
        "model_input_structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
        "contact_manifest_sha256": CONTACT_MANIFEST_SHA256,
        "contact_receipt_sha256": CONTACT_RECEIPT_SHA256,
        "development_split_sha256": SPLIT_SHA256,
        "arms": protocol_rows,
        "seeds": [17, 29, 43],
        "single_factor_change": "remove_bound_derived_distance_auxiliary_supervision",
        "distance_loss_weight": 0.0,
        "distance_label_cache_supplied": False,
        "distance_labels_loaded": False,
        "model_architecture_changed": False,
        "optimizer_or_schedule_changed": False,
        "pair_contact_labels_changed": False,
        "split_changed": False,
        "experimental_bound_geometry_used_as_model_input": False,
        "experimental_bound_geometry_used_for_auxiliary_training_supervision": False,
        "experimental_bound_structure_used_to_define_task_contact_labels": True,
        "experimental_structure_required_at_inference": False,
        "selection_or_tuning": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
        "execution_sources": execution,
    }
    campaign_path = authority / "CAMPAIGN_NO_PRIVILEGED_DISTANCE_V1.json"
    campaign_sha256 = write_json(campaign_path, campaign)

    checksum_paths = [
        upstream_campaign,
        upstream_analysis,
        baseline_source_copy,
        topk2_source_copy,
        *protocol_paths,
        campaign_path,
    ]
    (authority / "AUTHORITY_SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "campaign_sha256": campaign_sha256,
                "baseline_protocol_sha256": protocol_rows["baseline"][
                    "protocol_sha256"
                ],
                "topk2_protocol_sha256": protocol_rows["edge_topk2_supervised"][
                    "protocol_sha256"
                ],
                "distance_labels_loaded": False,
                "test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
