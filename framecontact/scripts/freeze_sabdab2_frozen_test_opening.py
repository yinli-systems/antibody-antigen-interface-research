#!/usr/bin/env python3
"""Freeze the one authorized SAbDab2 frozen-test scoring session."""

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

from interaction_first.reciprocal_partner_panel import (  # noqa: E402
    RECIPROCAL_PARTNER_PANEL_FORMAT,
)
from interaction_first.sabdab2_frozen_test import (  # noqa: E402
    FROZEN_BOOTSTRAP,
    FROZEN_COORDINATE_NOISE_ANGSTROM,
    FROZEN_REQUIRED_INTERVENTIONS,
    FROZEN_SAME_ANTIGEN_MAXIMUM_EPITOPE_JACCARD,
    FROZEN_TEST_PARTNER_PANEL_SIZE,
    OPENING_FORMAT,
    authenticate_sabdab2_development_selection,
    sabdab2_evaluation_source_hashes,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)
from interaction_first.same_antigen_challenge import (  # noqa: E402
    LABEL_FREE_SAME_ANTIGEN_CHALLENGE_FORMAT,
)
from freeze_framecontact_development_ensemble import (  # noqa: E402
    FROZEN_SEEDS,
    _authenticated_run,
)
from train_framecontact import _authenticated_panel  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--selection-receipt-sha256", required=True)
    parser.add_argument("--run-roots", type=Path, nargs=5, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--contact-manifest-sha256", required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest-sha256", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--test-partner-panel", type=Path, required=True)
    parser.add_argument("--test-partner-panel-sha256", required=True)
    parser.add_argument("--same-antigen-challenge", type=Path, required=True)
    parser.add_argument("--same-antigen-challenge-sha256", required=True)
    parser.add_argument("--panel-size", type=int, default=31)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_730)
    parser.add_argument(
        "--coordinate-noise", type=float, nargs="+", default=(0.10, 0.25, 0.50)
    )
    parser.add_argument("--evaluator-source", type=Path, required=True)
    parser.add_argument("--test-state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot parse {name}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def freeze_opening(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite SAbDab2 opening: {arguments.output}")
    if arguments.panel_size != FROZEN_TEST_PARTNER_PANEL_SIZE:
        raise ValueError("SAbDab2 frozen test requires exactly 31 wrong partners")
    if (
        {
            "replicates": arguments.bootstrap_replicates,
            "seed": arguments.bootstrap_seed,
        }
        != FROZEN_BOOTSTRAP
        or tuple(arguments.coordinate_noise) != FROZEN_COORDINATE_NOISE_ANGSTROM
    ):
        raise ValueError("SAbDab2 frozen test diagnostics differ from the protocol")
    if (
        not arguments.test_state_root.is_absolute()
        or arguments.test_state_root.is_symlink()
        or not arguments.test_state_root.is_dir()
    ):
        raise ValueError("test-state-root must be an existing absolute regular directory")
    score_token = arguments.test_state_root / "SABDAB2_TEST_SCORE_CONSUMED.json"
    if score_token.exists() or score_token.is_symlink():
        raise ValueError("SAbDab2 frozen-test scoring was already consumed")

    selection = authenticate_sabdab2_development_selection(
        arguments.selection_root,
        selection_receipt_sha256=arguments.selection_receipt_sha256,
    )
    ensemble_root = Path(selection["selected_ensemble_root"])
    ensemble_result_path = ensemble_root / "FRAMECONTACT_DEVELOPMENT_ENSEMBLE.json"
    ensemble = _json(ensemble_result_path, "selected SAbDab2 ensemble")
    members = [_authenticated_run(root) for root in arguments.run_roots]
    members.sort(key=lambda member: int(member["checkpoint"]["seed"]))
    if tuple(int(member["checkpoint"]["seed"]) for member in members) != FROZEN_SEEDS:
        raise ValueError("SAbDab2 opening requires five pre-registered seeds")
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
            raise ValueError("SAbDab2 opening member differs from selected ensemble")

    for path, expected, name in (
        (arguments.structure_manifest, arguments.structure_manifest_sha256, "structure manifest"),
        (arguments.contact_manifest, arguments.contact_manifest_sha256, "contact manifest"),
        (arguments.embedding_manifest, arguments.embedding_manifest_sha256, "embedding manifest"),
        (arguments.split_manifest, arguments.split_manifest_sha256, "split manifest"),
    ):
        authenticate_file(path, expected, name)
    ensemble_authority = ensemble["authority"]
    for key, expected in (
        ("structure_manifest_sha256", arguments.structure_manifest_sha256),
        ("contact_manifest_sha256", arguments.contact_manifest_sha256),
        ("embedding_manifest_sha256", arguments.embedding_manifest_sha256),
        ("split_manifest_sha256", arguments.split_manifest_sha256),
    ):
        if ensemble_authority.get(key) != expected:
            raise ValueError("SAbDab2 ensemble and test data authorities differ")

    panel = _authenticated_panel(
        arguments.test_partner_panel,
        arguments.test_partner_panel_sha256,
        role="test",
        expected_size=FROZEN_TEST_PARTNER_PANEL_SIZE,
        expected_format=RECIPROCAL_PARTNER_PANEL_FORMAT,
    )
    with arguments.split_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    test_ids = {str(row.get("instance_id", "")) for row in rows if row.get("role") == "test"}
    panel_ids = {str(entry.get("instance_id", "")) for entry in panel.get("entries", [])}
    if not test_ids or panel_ids != test_ids or len(panel.get("entries", [])) != len(test_ids):
        raise ValueError("SAbDab2 test panel and split axes differ")

    authenticate_file(
        arguments.same_antigen_challenge,
        arguments.same_antigen_challenge_sha256,
        "SAbDab2 same-antigen challenge",
    )
    same_antigen = _json(arguments.same_antigen_challenge, "same-antigen challenge")
    challenge_authority = same_antigen.get("authority", {})
    if (
        same_antigen.get("format_version")
        != LABEL_FREE_SAME_ANTIGEN_CHALLENGE_FORMAT
        or same_antigen.get("role") != "test"
        or challenge_authority.get("split_sha256")
        != arguments.split_manifest_sha256
        or challenge_authority.get("task_labels_read") is not False
        or challenge_authority.get("model_predictions_read") is not False
        or challenge_authority.get("same_antigen_rule")
        != "exact antigen_sequence_sha256"
        or not same_antigen.get("entries")
    ):
        raise ValueError("SAbDab2 same-antigen challenge is ineligible")
    challenge_ids = {
        str(entry.get("focal_instance_id", ""))
        for entry in same_antigen["entries"]
    }
    challenge_ids.update(
        str(alternate.get("instance_id", ""))
        for entry in same_antigen["entries"]
        for alternate in entry.get("alternates", [])
    )
    if not challenge_ids.issubset(test_ids):
        raise ValueError("Same-antigen challenge references a non-test instance")

    expected_evaluator = REPOSITORY_ROOT / "scripts/evaluate_framecontact_sabdab2_ensemble_test.py"
    if arguments.evaluator_source.resolve() != expected_evaluator.resolve():
        raise ValueError("Unexpected SAbDab2 evaluator source path")
    sources = sabdab2_evaluation_source_hashes(REPOSITORY_ROOT)
    payload = {
        "format_version": OPENING_FORMAT,
        "eligible_to_open": True,
        "test_score_session_count_before": 0,
        "test_predictions_or_scores_read_at_freeze": False,
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
        "structure_manifest_sha256": arguments.structure_manifest_sha256,
        "contact_manifest_sha256": arguments.contact_manifest_sha256,
        "embedding_manifest_sha256": arguments.embedding_manifest_sha256,
        "split_manifest_sha256": arguments.split_manifest_sha256,
        "test_partner_panel_sha256": arguments.test_partner_panel_sha256,
        "same_antigen_challenge_sha256": arguments.same_antigen_challenge_sha256,
        "test_instance_count": len(test_ids),
        "test_partner_panel_size": FROZEN_TEST_PARTNER_PANEL_SIZE,
        "same_antigen_maximum_epitope_jaccard": (
            FROZEN_SAME_ANTIGEN_MAXIMUM_EPITOPE_JACCARD
        ),
        "coordinate_noise_angstrom": list(FROZEN_COORDINATE_NOISE_ANGSTROM),
        "bootstrap": FROZEN_BOOTSTRAP,
        "required_interventions": list(FROZEN_REQUIRED_INTERVENTIONS),
        "test_state_root": str(arguments.test_state_root.resolve()),
        "score_consumption_token_name": "SABDAB2_TEST_SCORE_CONSUMED.json",
        "test_evaluator_source_sha256": sources[
            "scripts/evaluate_framecontact_sabdab2_ensemble_test.py"
        ],
        "evaluation_source_dependencies": sources,
        "claim_boundary": (
            "One hash-bound SAbDab2 frozen-test scoring session, frozen after "
            "validation-only architecture and checkpoint selection. No test "
            "prediction, score, fitting, or intervention result was read."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(canonical_json_bytes(payload))
    return {"output": str(arguments.output), "sha256": sha256_file(arguments.output)}


def main() -> None:
    print(json.dumps(freeze_opening(_arguments()), sort_keys=True))


if __name__ == "__main__":
    main()
