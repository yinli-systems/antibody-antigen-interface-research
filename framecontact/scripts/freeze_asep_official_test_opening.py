#!/usr/bin/env python3
"""Freeze every AsEP official-test decision before graph labels are opened."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from interaction_first.asep_authority import INNER_AUDIT_FORMAT  # noqa: E402
from interaction_first.asep_decontamination import (  # noqa: E402
    DECONTAMINATION_FORMAT,
)
from interaction_first.asep_external import _load_split_roles  # noqa: E402
from interaction_first.asep_frozen_test import (  # noqa: E402
    LABEL_FREE_TEST_FORMAT,
    OPENING_FORMAT,
    asep_evaluation_source_hashes,
    authenticate_asep_development_selection,
)
from interaction_first.reciprocal_partner_panel import (  # noqa: E402
    RECIPROCAL_PARTNER_PANEL_FORMAT,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)
from freeze_asep_development_ensemble import (  # noqa: E402
    FROZEN_SEEDS,
    authenticated_development_run,
)


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot parse {name}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--selection-receipt-sha256", required=True)
    parser.add_argument("--run-roots", type=Path, nargs=5, required=True)
    parser.add_argument("--inner-audit", type=Path, required=True)
    parser.add_argument("--inner-audit-sha256", required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split-receipt-sha256", required=True)
    parser.add_argument("--split-method", default="epitope_group")
    parser.add_argument("--label-free-test-root", type=Path, required=True)
    parser.add_argument("--label-free-test-receipt-sha256", required=True)
    parser.add_argument("--decontamination-root", type=Path, required=True)
    parser.add_argument("--decontamination-receipt-sha256", required=True)
    parser.add_argument("--full-test-panel", type=Path, required=True)
    parser.add_argument("--full-test-panel-sha256", required=True)
    parser.add_argument("--clean-test-panel", type=Path, required=True)
    parser.add_argument("--clean-test-panel-sha256", required=True)
    parser.add_argument("--panel-size", type=int, default=31)
    parser.add_argument("--evaluator-source", type=Path, required=True)
    parser.add_argument("--test-state-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_730)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _metadata_ids(root: Path, receipt_sha256: str) -> tuple[set[str], dict[str, Any]]:
    receipt_path = root / "ASEP_LABEL_FREE_TEST_METADATA_RECEIPT.json"
    manifest_path = root / "ASEP_LABEL_FREE_TEST_METADATA.csv"
    authenticate_file(receipt_path, receipt_sha256, "AsEP label-free test receipt")
    receipt = _json(receipt_path, "AsEP label-free test receipt")
    authenticate_file(
        manifest_path,
        str(receipt.get("manifest_sha256", "")),
        "AsEP label-free test manifest",
    )
    if (
        receipt.get("format_version") != LABEL_FREE_TEST_FORMAT
        or receipt.get("label_free_test_feature_open_count") != 1
        or receipt.get("test_label_open_count") != 0
        or receipt.get("checks", {}).get("official_test_graph_tensors_opened")
        is not False
        or receipt.get("checks", {}).get(
            "official_test_contact_or_residue_labels_read"
        )
        is not False
        or receipt.get("checks", {}).get("model_predictions_or_scores_read")
        is not False
    ):
        raise ValueError("Label-free AsEP test authority is ineligible")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    identifiers = {str(row.get("instance_id", "")) for row in rows}
    if (
        "" in identifiers
        or len(identifiers) != len(rows)
        or len(rows) != int(receipt.get("test_instance_count", -1))
        or any(row.get("role") != "test" for row in rows)
    ):
        raise ValueError("Label-free AsEP test instance axis is malformed")
    return identifiers, receipt


def _panel(
    path: Path,
    expected_sha256: str,
    *,
    expected_ids: set[str],
    expected_scope: str,
    expected_size: int,
) -> dict[str, Any]:
    authenticate_file(path, expected_sha256, f"AsEP {expected_scope} panel")
    panel = _json(path, f"AsEP {expected_scope} panel")
    authority = panel.get("authority", {})
    entries = panel.get("entries", [])
    identifiers = {
        str(entry.get("instance_id", ""))
        for entry in entries
        if isinstance(entry, dict)
    }
    if (
        panel.get("format_version") != RECIPROCAL_PARTNER_PANEL_FORMAT
        or panel.get("role") != "test"
        or panel.get("panel_size_per_direction") != expected_size
        or authority.get("panel_scope") != expected_scope
        or authority.get("task_label_columns_read") is not False
        or authority.get("official_test_graph_tensors_opened") is not False
        or authority.get("model_predictions_or_scores_read") is not False
        or identifiers != expected_ids
        or len(entries) != len(expected_ids)
    ):
        raise ValueError(f"AsEP {expected_scope} panel is ineligible")
    for entry in entries:
        focal = str(entry["instance_id"])
        for key in ("wrong_antigen_instance_ids", "wrong_antibody_instance_ids"):
            candidates = entry.get(key, [])
            if (
                len(candidates) != expected_size
                or len(set(candidates)) != expected_size
                or focal in candidates
                or not set(candidates).issubset(expected_ids)
            ):
                raise ValueError(f"AsEP {expected_scope} panel candidates differ")
    return panel


def freeze_opening(arguments: argparse.Namespace) -> dict[str, Any]:
    """Validate all pre-opening authorities and write one immutable receipt."""
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite opening receipt: {arguments.output}")
    if arguments.panel_size != 31:
        raise ValueError("AsEP official opening requires exactly 31 wrong partners")
    if (
        arguments.bootstrap_replicates != 10_000
        or arguments.bootstrap_seed != 20_260_730
    ):
        raise ValueError("AsEP official opening requires the frozen bootstrap protocol")
    if arguments.split_method != "epitope_group":
        raise ValueError("AsEP official opening requires the epitope_group split")
    if (
        not arguments.test_state_root.is_absolute()
        or arguments.test_state_root.is_symlink()
        or not arguments.test_state_root.is_dir()
    ):
        raise ValueError("test-state-root must be an existing absolute regular directory")
    graph_open_token = (
        arguments.test_state_root / "ASEP_TEST_GRAPH_OPEN_CONSUMED.json"
    )
    score_token = arguments.test_state_root / "ASEP_TEST_SCORE_CONSUMED.json"
    if any(path.exists() or path.is_symlink() for path in (graph_open_token, score_token)):
        raise ValueError("AsEP official graph opening or scoring was already consumed")

    selection = authenticate_asep_development_selection(
        arguments.selection_root,
        selection_receipt_sha256=arguments.selection_receipt_sha256,
    )
    members = [authenticated_development_run(root) for root in arguments.run_roots]
    members.sort(key=lambda member: int(member["checkpoint"]["seed"]))
    seeds = tuple(int(member["checkpoint"]["seed"]) for member in members)
    if seeds != FROZEN_SEEDS:
        raise ValueError("Opening requires the five pre-registered seeds")
    ensemble_result_path = (
        Path(selection["selected_ensemble_root"])
        / "FRAMECONTACT_ASEP_DEVELOPMENT_ENSEMBLE.json"
    )
    ensemble = _json(ensemble_result_path, "selected AsEP ensemble")
    expected_members = {
        int(member["seed"]): member
        for member in ensemble.get("authority", {}).get("members", [])
    }
    for member in members:
        seed = int(member["checkpoint"]["seed"])
        expected = expected_members.get(seed)
        if (
            expected is None
            or expected.get("development_result_sha256") != member["result_sha256"]
            or expected.get("selected_checkpoint_sha256")
            != member["checkpoint_sha256"]
            or member["result"].get("experiment")
            != selection["selection_result"].get("selected_experiment")
        ):
            raise ValueError("Opening member differs from selected ensemble authority")

    authenticate_file(arguments.inner_audit, arguments.inner_audit_sha256, "AsEP inner audit")
    inner = _json(arguments.inner_audit, "AsEP inner audit")
    if (
        inner.get("format_version") != INNER_AUDIT_FORMAT
        or inner.get("authority", {}).get("task_labels_read") is not False
        or inner.get("authority", {}).get("model_predictions_or_scores_read")
        is not False
        or inner.get("checks", {}).get("split_payload_opened") is not False
    ):
        raise ValueError("AsEP inner audit is not pre-opening eligible")
    roles, split_receipt, split_manifest_sha256 = _load_split_roles(
        arguments.split_root,
        arguments.split_receipt_sha256,
        arguments.split_method,
    )
    official_test_ids = {
        identifier for identifier, role in roles.items() if role == "test"
    }
    expected_count = int(
        split_receipt.get("manifests", {})
        .get(arguments.split_method, {})
        .get("role_counts", {})
        .get("test", -1)
    )
    if not official_test_ids or len(official_test_ids) != expected_count:
        raise ValueError("AsEP official test role is incomplete")

    metadata_ids, metadata_receipt = _metadata_ids(
        arguments.label_free_test_root,
        arguments.label_free_test_receipt_sha256,
    )
    metadata_authority = metadata_receipt.get("authority", {})
    if (
        metadata_ids != official_test_ids
        or metadata_authority.get("inner_audit_sha256")
        != arguments.inner_audit_sha256
        or metadata_authority.get("split_receipt_sha256")
        != arguments.split_receipt_sha256
        or metadata_authority.get("split_manifest_sha256")
        != split_manifest_sha256
        or metadata_authority.get("selection_receipt_sha256")
        != arguments.selection_receipt_sha256
        or metadata_authority.get("selected_ensemble_result_sha256")
        != selection["selected_ensemble_result_sha256"]
    ):
        raise ValueError("Label-free metadata differs from the opening authorities")

    decon_receipt_path = arguments.decontamination_root / "DECONTAMINATION_RECEIPT.json"
    decon_result_path = (
        arguments.decontamination_root / "ASEP_SABDAB2_DECONTAMINATION.json"
    )
    authenticate_file(
        decon_receipt_path,
        arguments.decontamination_receipt_sha256,
        "AsEP decontamination receipt",
    )
    decon_receipt = _json(decon_receipt_path, "AsEP decontamination receipt")
    authenticate_file(
        decon_result_path,
        str(decon_receipt.get("result_sha256", "")),
        "AsEP decontamination result",
    )
    decon = _json(decon_result_path, "AsEP decontamination result")
    clean_ids = set(decon.get("decontaminated_primary_instance_ids", []))
    decon_records = decon.get("records", [])
    decon_record_ids = {
        str(record.get("instance_id", ""))
        for record in decon_records
        if isinstance(record, dict)
    }
    expected_flags = {
        "exact_pdb_identity",
        "exact_antigen_pdb_seqres",
        "exact_antibody_pdb_seqres",
        "antigen_structural_similarity",
    }
    if (
        decon_receipt.get("format_version") != DECONTAMINATION_FORMAT
        or decon.get("format_version") != DECONTAMINATION_FORMAT
        or decon_receipt.get("test_labels_read") is not False
        or decon_receipt.get("model_predictions_or_scores_read") is not False
        or decon_receipt.get("decontaminated_primary_instance_count")
        != len(clean_ids)
        or decon.get("authority", {}).get("test_labels_read") is not False
        or decon.get("authority", {}).get("model_predictions_or_scores_read")
        is not False
        or decon.get("authority", {}).get("test_graph_tensors_opened") is not False
        or decon.get("authority", {}).get("label_free_test_receipt_sha256")
        != arguments.label_free_test_receipt_sha256
        or decon.get("authority", {}).get("label_free_test_manifest_sha256")
        != metadata_receipt["manifest_sha256"]
        or decon.get("policy", {}).get("antigen_structural_rule")
        != "reciprocal_min(query_tm,target_tm) >= 0.50"
        or int(decon.get("counts", {}).get("asep_test_instances", -1))
        != len(official_test_ids)
        or int(decon.get("counts", {}).get("cross_structure_pairs", -1)) <= 0
        or not clean_ids
        or not clean_ids.issubset(official_test_ids)
        or len(clean_ids)
        != int(decon.get("counts", {}).get("decontaminated_primary_instances", -1))
        or decon_record_ids != official_test_ids
        or len(decon_records) != len(official_test_ids)
        or any(
            set(record.get("flags", {})) != expected_flags
            or bool(record.get("eligible_for_decontaminated_primary"))
            != (str(record.get("instance_id")) in clean_ids)
            or bool(record.get("contaminated"))
            == bool(record.get("eligible_for_decontaminated_primary"))
            for record in decon_records
        )
    ):
        raise ValueError("AsEP decontamination result is ineligible")

    _panel(
        arguments.full_test_panel,
        arguments.full_test_panel_sha256,
        expected_ids=official_test_ids,
        expected_scope="full_official_test",
        expected_size=arguments.panel_size,
    )
    _panel(
        arguments.clean_test_panel,
        arguments.clean_test_panel_sha256,
        expected_ids=clean_ids,
        expected_scope="sabdab2_decontaminated_primary",
        expected_size=arguments.panel_size,
    )
    expected_evaluator = REPOSITORY_ROOT / "scripts/evaluate_asep_frozen_test.py"
    if arguments.evaluator_source.resolve() != expected_evaluator.resolve():
        raise ValueError("Unexpected AsEP frozen-test evaluator source path")
    if not arguments.evaluator_source.is_file() or arguments.evaluator_source.is_symlink():
        raise ValueError("AsEP frozen-test evaluator must be a regular file")
    evaluation_dependencies = asep_evaluation_source_hashes(REPOSITORY_ROOT)

    payload = {
        "format_version": OPENING_FORMAT,
        "eligible_to_open": True,
        "test_feature_open_count_before": 1,
        "test_graph_or_label_open_count_before": 0,
        "test_labels_read_at_freeze": False,
        "model_predictions_or_scores_read_at_freeze": False,
        "selection_receipt_sha256": arguments.selection_receipt_sha256,
        "selection_result_sha256": selection["selection_result_sha256"],
        "selected_experiment": selection["selection_result"]["selected_experiment"],
        "selected_ensemble_result_sha256": selection[
            "selected_ensemble_result_sha256"
        ],
        "selected_ensemble_receipt_sha256": selection[
            "selected_ensemble_receipt_sha256"
        ],
        "members": [
            {
                "seed": int(member["checkpoint"]["seed"]),
                "development_result_sha256": member["result_sha256"],
                "selected_checkpoint_sha256": member["checkpoint_sha256"],
            }
            for member in members
        ],
        "inner_audit_sha256": arguments.inner_audit_sha256,
        "split_receipt_sha256": arguments.split_receipt_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "split_method": arguments.split_method,
        "graph_archive_sha256": inner["graph_archive"]["sha256"],
        "structure_archive_sha256": inner["structure_archive"]["sha256"],
        "official_test_instance_count": len(official_test_ids),
        "label_free_test_receipt_sha256": arguments.label_free_test_receipt_sha256,
        "label_free_test_manifest_sha256": metadata_receipt["manifest_sha256"],
        "decontamination_receipt_sha256": arguments.decontamination_receipt_sha256,
        "decontamination_result_sha256": decon_receipt["result_sha256"],
        "decontaminated_primary_instance_count": len(clean_ids),
        "full_test_panel_sha256": arguments.full_test_panel_sha256,
        "clean_test_panel_sha256": arguments.clean_test_panel_sha256,
        "test_partner_panel_size": arguments.panel_size,
        "test_evaluator_source_sha256": sha256_file(arguments.evaluator_source),
        "evaluation_source_dependencies": evaluation_dependencies,
        "test_adapter_source_sha256": sha256_file(
            REPOSITORY_ROOT / "src/interaction_first/asep_frozen_test.py"
        ),
        "test_state_root": str(arguments.test_state_root.resolve()),
        "graph_open_consumption_token_name": (
            "ASEP_TEST_GRAPH_OPEN_CONSUMED.json"
        ),
        "score_consumption_token_name": "ASEP_TEST_SCORE_CONSUMED.json",
        "primary_endpoints": [
            "contact_pooled_auprc",
            "epitope_pooled_auprc",
            "paratope_pooled_auprc",
            "minimum_direction_cluster_weighted_partner_margin",
        ],
        "reporting_scopes": [
            "full_official_test",
            "sabdab2_decontaminated_primary",
        ],
        "calibration_rule": "reuse validation-only ensemble temperatures and thresholds",
        "bootstrap": {
            "unit": "test panel antigen or antibody PDB-SEQRES cluster",
            "replicates": arguments.bootstrap_replicates,
            "seed": arguments.bootstrap_seed,
        },
        "claim_boundary": (
            "Hash-bound pre-opening receipt. Development selection, checkpoints, "
            "label-free contamination controls, panels, calibration rule, metrics, "
            "and evaluator source were frozen before official graph tensors or labels "
            "were opened. Counterfactual partners are not asserted non-binders."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload))
    return {"output": str(arguments.output), "sha256": sha256_file(arguments.output)}


def main() -> None:
    result = freeze_opening(_arguments())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
