#!/usr/bin/env python3
"""Re-evaluate one frozen E1/E2 checkpoint with the V2 contact audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import interaction_first.pairscale_e1_e2_evaluation as evaluation_module
import interaction_first.pairscale_e1_e2_posthoc_audit as posthoc_audit_module
from interaction_first.development_contact_dataset import DevelopmentContactDataset
from interaction_first.development_holdout import parse_component_holdout_assignment
from interaction_first.frontier_checkpoint import (
    restore_checkpoint_generation,
    sha256_file,
)
from interaction_first.pairscale_e1_e2_evaluation import evaluate_e1_e2
from interaction_first.pairscale_e1_e2_posthoc_audit import (
    POSTHOC_AUDIT_FORMAT,
    validate_frozen_report,
    validate_posthoc_source_boundary,
    verify_v2_reproduces_v1,
)
from interaction_first.pairscale_e1_e2_step import build_e1_e2_model
from interaction_first.pairscale_e1_e2_training import (
    SCREENING_SEEDS,
    e1_e2_source_hashes,
    locked_e1_e2_config,
)
from interaction_first.pairscale_holdout_training import split_dataset_indices
from interaction_first.pairscale_overfit import configure_determinism
from interaction_first.pairscale_unary_training import _create_json, _load_examples


ACCESS_FORMAT = "pairscale_e1_e2_posthoc_holdout_access_receipt_v1"


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _regular_bytes(path: Path) -> bytes:
    target = path.expanduser().absolute()
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {target}")
    return target.read_bytes()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            _regular_bytes(path).decode("utf-8", errors="strict"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot parse strict JSON from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON payload at {path} must be an object")
    return value


def _require_sha256(path: Path, expected: str, name: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"{name} SHA-256 differs from its external anchor")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SCREENING_SEEDS, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--cache-receipt-file-sha256", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--subset-manifest-sha256", required=True)
    parser.add_argument("--authorization-manifest-sha256", required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-file-sha256", required=True)
    parser.add_argument("--assignment-content-sha256", required=True)
    parser.add_argument("--assignment-receipt", type=Path, required=True)
    parser.add_argument("--assignment-receipt-file-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--original-access-receipt", type=Path, required=True)
    parser.add_argument("--original-access-receipt-sha256", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--created-at-utc")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    _require_sha256(arguments.report, arguments.report_sha256, "report")
    report = _strict_json(arguments.report)
    checkpoint = validate_frozen_report(
        report,
        seed=arguments.seed,
        expected_holdout_count=64,
    )
    _require_sha256(
        arguments.original_access_receipt,
        arguments.original_access_receipt_sha256,
        "original access receipt",
    )
    original_access = _strict_json(arguments.original_access_receipt)
    if (
        original_access.get("frozen_checkpoint_epoch") != 260
        or original_access.get("frozen_checkpoint_sha256")
        != checkpoint["checkpoint_sha256"]
        or original_access.get("frozen_checkpoint_receipt_file_sha256")
        != checkpoint["receipt_file_sha256"]
        or original_access.get("development_holdout_is_untouched_final_test")
        is not False
        or original_access.get("new_final_test_accessed") is not False
        or original_access.get("scaling_authorized") is not False
    ):
        raise ValueError("Original holdout access receipt differs")

    assignment_bytes = _regular_bytes(arguments.assignment)
    assignment_receipt_bytes = _regular_bytes(arguments.assignment_receipt)
    if hashlib.sha256(assignment_bytes).hexdigest() != (
        arguments.assignment_file_sha256
    ):
        raise ValueError("Assignment file SHA-256 differs")
    if hashlib.sha256(assignment_receipt_bytes).hexdigest() != (
        arguments.assignment_receipt_file_sha256
    ):
        raise ValueError("Assignment receipt SHA-256 differs")
    assignment = parse_component_holdout_assignment(
        assignment_bytes,
        assignment_receipt_bytes,
        expected_assignment_file_sha256=arguments.assignment_file_sha256,
        expected_assignment_content_sha256=arguments.assignment_content_sha256,
        expected_receipt_file_sha256=(
            arguments.assignment_receipt_file_sha256
        ),
        expected_source_subset_sha256=arguments.subset_manifest_sha256,
        expected_policy_sha256=arguments.policy_sha256,
    )
    dataset = DevelopmentContactDataset(
        arguments.cache_root,
        expected_manifest_sha256=arguments.manifest_sha256,
        expected_receipt_file_sha256=arguments.cache_receipt_file_sha256,
        expected_policy_sha256=arguments.policy_sha256,
        expected_decision_sha256=arguments.decision_sha256,
        expected_subset_manifest_sha256=arguments.subset_manifest_sha256,
        expected_authorization_manifest_sha256=(
            arguments.authorization_manifest_sha256
        ),
    )
    if len(dataset) != 256:
        raise ValueError("Dataset source count differs from the execution lock")
    _, holdout_indices = split_dataset_indices(dataset, assignment)
    if len(holdout_indices) != 64:
        raise ValueError("Development holdout count differs")
    report_identifiers = {
        row["instance_id"]
        for row in report["development_holdout_evaluation"]["per_complex"]
    }
    assignment_identifiers = {
        dataset.records[index].instance_id for index in holdout_indices
    }
    if report_identifiers != assignment_identifiers:
        raise ValueError("Report and assignment holdout inventories differ")

    current_sources = e1_e2_source_hashes()
    source_boundary = validate_posthoc_source_boundary(
        checkpoint["provenance_hashes"],
        current_sources,
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Posthoc E1/E2 checkpoint audit requires CUDA")
    configure_determinism(arguments.seed)
    device = torch.device("cuda", 0)
    config = locked_e1_e2_config(arguments.seed)
    model = build_e1_e2_model(dataset, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    restored = restore_checkpoint_generation(
        arguments.checkpoint_root,
        checkpoint["generation"],
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        map_location=device,
        expected_provenance_hashes=checkpoint["provenance_hashes"],
        expected_receipt_sha256=checkpoint["receipt_file_sha256"],
        expected_rng_world_size=1,
        restore_random_generators=False,
    )
    if (
        restored["epoch"] != 260
        or restored["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]
        or restored["receipt_sha256"] != checkpoint["receipt_file_sha256"]
        or restored["generation"] != checkpoint["generation"]
        or restored["config"].get("base_config", {}).get("training")
        != config.canonical_payload()
    ):
        raise ValueError("Restored checkpoint evidence differs from the report")

    output_root = arguments.output_root.expanduser().absolute()
    if output_root.is_symlink() or output_root.exists():
        raise ValueError("Posthoc output root must be create-only")
    output_root.mkdir(parents=True, exist_ok=False)
    created_at = arguments.created_at_utc or (
        dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    )
    access_receipt = {
        "format_version": ACCESS_FORMAT,
        "created_at_utc": created_at,
        "evaluation_role": "repeated_development_holdout_contact_audit",
        "development_holdout_is_untouched_final_test": False,
        "new_final_test_accessed": False,
        "model_selection_authorized": False,
        "checkpoint_promotion_authorized": False,
        "confirmatory_claim_authorized": False,
        "scaling_authorized": False,
        "frozen_checkpoint_epoch": restored["epoch"],
        "frozen_checkpoint_sha256": restored["checkpoint_sha256"],
        "frozen_checkpoint_receipt_file_sha256": restored["receipt_sha256"],
        "frozen_report_sha256": arguments.report_sha256,
        "original_holdout_access_receipt_sha256": (
            arguments.original_access_receipt_sha256
        ),
        "holdout_instance_count": len(holdout_indices),
        "shared_v1_metrics_must_reproduce_exactly": True,
    }
    access_path = output_root / "POSTHOC_HOLDOUT_ACCESS_RECEIPT.json"
    _create_json(access_path, access_receipt)

    holdout_examples = _load_examples(dataset, holdout_indices)
    evaluation = evaluate_e1_e2(
        model=model,
        examples=holdout_examples,
        device=device,
        decoy_count=config.decoy_count_per_direction,
        seed=config.seed,
        include_partner_panel=True,
        include_exhaustive_retrieval=True,
    )
    reproduction = verify_v2_reproduces_v1(
        report["development_holdout_evaluation"],
        evaluation,
    )
    audit = {
        "format_version": POSTHOC_AUDIT_FORMAT,
        "created_at_utc": created_at,
        "evaluation_role": "repeated_development_holdout_contact_audit",
        "claim_boundary": (
            "The reused development holdout is not an untouched final test. "
            "This audit cannot authorize selection, promotion, confirmation, "
            "scaling, SOTA, or world-leading claims."
        ),
        "screening_seed": arguments.seed,
        "model_selection_authorized": False,
        "checkpoint_promotion_authorized": False,
        "confirmatory_claim_authorized": False,
        "scaling_authorized": False,
        "new_final_test_accessed": False,
        "frozen_report_sha256": arguments.report_sha256,
        "frozen_checkpoint": {
            "generation": restored["generation"],
            "epoch": restored["epoch"],
            "checkpoint_sha256": restored["checkpoint_sha256"],
            "receipt_file_sha256": restored["receipt_sha256"],
        },
        "source_boundary": source_boundary,
        "posthoc_implementation_source_sha256": {
            "audit_pairscale_e1_e2_checkpoint.py": sha256_file(Path(__file__)),
            "pairscale_e1_e2_evaluation.py": sha256_file(
                Path(evaluation_module.__file__)
            ),
            "pairscale_e1_e2_posthoc_audit.py": sha256_file(
                Path(posthoc_audit_module.__file__)
            ),
        },
        "posthoc_access_receipt_sha256": sha256_file(access_path),
        "v1_exact_reproduction": reproduction,
        "development_holdout_evaluation_v2": evaluation,
    }
    _create_json(output_root / "POSTHOC_CONTACT_AUDIT_V2.json", audit)


if __name__ == "__main__":
    main()
