#!/usr/bin/env python3
"""Train one preregistered PAIRScale E1/E2 development seed."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from interaction_first.development_contact_dataset import DevelopmentContactDataset
from interaction_first.development_holdout import parse_component_holdout_assignment
from interaction_first.pairscale_e1_e2_training import (
    SCREENING_SEEDS,
    locked_e1_e2_config,
    run_e1_e2_screen,
)


def _regular_bytes(path: Path) -> bytes:
    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")
    return path.read_bytes()


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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-file-sha256", required=True)
    parser.add_argument("--expected-controller-sha256", required=True)
    parser.add_argument("--scheduler-script-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint-root", type=Path)
    parser.add_argument("--resume-generation")
    parser.add_argument("--resume-receipt-sha256")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    controller_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if controller_sha256 != arguments.expected_controller_sha256:
        raise ValueError("Controller source digest differs")
    protocol_bytes = _regular_bytes(arguments.protocol)
    if hashlib.sha256(protocol_bytes).hexdigest() != arguments.protocol_file_sha256:
        raise ValueError("Protocol file digest differs")
    assignment_bytes = _regular_bytes(arguments.assignment)
    receipt_bytes = _regular_bytes(arguments.assignment_receipt)
    if hashlib.sha256(assignment_bytes).hexdigest() != (
        arguments.assignment_file_sha256
    ):
        raise ValueError("Assignment file digest differs")
    if hashlib.sha256(receipt_bytes).hexdigest() != (
        arguments.assignment_receipt_file_sha256
    ):
        raise ValueError("Assignment receipt file digest differs")
    assignment = parse_component_holdout_assignment(
        assignment_bytes,
        receipt_bytes,
        expected_assignment_file_sha256=arguments.assignment_file_sha256,
        expected_assignment_content_sha256=arguments.assignment_content_sha256,
        expected_receipt_file_sha256=arguments.assignment_receipt_file_sha256,
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
    run_e1_e2_screen(
        dataset=dataset,
        assignment=assignment,
        assignment_receipt_file_sha256=(
            arguments.assignment_receipt_file_sha256
        ),
        protocol_file_sha256=arguments.protocol_file_sha256,
        controller_sha256=controller_sha256,
        scheduler_script_sha256=arguments.scheduler_script_sha256,
        output_root=arguments.output_root,
        config=locked_e1_e2_config(arguments.seed),
        resume_checkpoint_root=arguments.resume_checkpoint_root,
        resume_generation=arguments.resume_generation,
        expected_resume_receipt_sha256=arguments.resume_receipt_sha256,
    )


if __name__ == "__main__":
    main()
