#!/usr/bin/env python3
"""Train PAIRFormer on a frozen train-only component-disjoint holdout."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from interaction_first.development_contact_dataset import DevelopmentContactDataset
from interaction_first.development_holdout import parse_component_holdout_assignment
from interaction_first.pairscale_holdout_training import (
    run_development_holdout_training,
)
from interaction_first.pairscale_overfit import PairscaleOverfitConfig


def _regular_bytes(path: Path) -> bytes:
    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")
    return path.read_bytes()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--expected-controller-sha256", required=True)
    parser.add_argument("--scheduler-script-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--positive-weight", type=float, default=1024.0)
    parser.add_argument("--contact-ranking-weight", type=float, default=1.0)
    parser.add_argument("--hard-negative-ratio", type=int, default=8)
    parser.add_argument("--partner-ranking-weight", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--expected-source-count", type=int, default=256)
    parser.add_argument("--expected-train-count", type=int, default=192)
    parser.add_argument("--expected-holdout-count", type=int, default=64)
    parser.add_argument("--resume-checkpoint-root", type=Path)
    parser.add_argument("--resume-generation")
    parser.add_argument("--resume-receipt-sha256")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    controller_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if controller_sha256 != arguments.expected_controller_sha256:
        raise ValueError("Controller source digest mismatch")
    assignment_bytes = _regular_bytes(arguments.assignment)
    receipt_bytes = _regular_bytes(arguments.assignment_receipt)
    if hashlib.sha256(assignment_bytes).hexdigest() != (
        arguments.assignment_file_sha256
    ):
        raise ValueError("Assignment file digest mismatch")
    if hashlib.sha256(receipt_bytes).hexdigest() != (
        arguments.assignment_receipt_file_sha256
    ):
        raise ValueError("Assignment receipt file digest mismatch")
    assignment = parse_component_holdout_assignment(
        assignment_bytes,
        receipt_bytes,
        expected_assignment_file_sha256=arguments.assignment_file_sha256,
        expected_assignment_content_sha256=arguments.assignment_content_sha256,
        expected_receipt_file_sha256=arguments.assignment_receipt_file_sha256,
        expected_source_subset_sha256=arguments.subset_manifest_sha256,
        expected_policy_sha256=arguments.policy_sha256,
    )
    if len(assignment.train_instances) != arguments.expected_train_count:
        raise ValueError("Assignment train count differs")
    if len(assignment.holdout_instances) != arguments.expected_holdout_count:
        raise ValueError("Assignment holdout count differs")
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
    if len(dataset) != arguments.expected_source_count:
        raise ValueError("Dataset source count differs")
    config = PairscaleOverfitConfig(
        seed=arguments.seed,
        expected_complex_count=arguments.expected_train_count,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        positive_weight=arguments.positive_weight,
        contact_ranking_weight=arguments.contact_ranking_weight,
        hard_negative_ratio=arguments.hard_negative_ratio,
        partner_ranking_weight=arguments.partner_ranking_weight,
        eval_every_epochs=arguments.eval_every,
        checkpoint_every_epochs=arguments.checkpoint_every,
        precision=arguments.precision,
    )
    run_development_holdout_training(
        dataset=dataset,
        assignment=assignment,
        assignment_receipt_file_sha256=(
            arguments.assignment_receipt_file_sha256
        ),
        controller_sha256=controller_sha256,
        scheduler_script_sha256=arguments.scheduler_script_sha256,
        output_root=arguments.output_root,
        config=config,
        resume_checkpoint_root=arguments.resume_checkpoint_root,
        resume_generation=arguments.resume_generation,
        expected_resume_receipt_sha256=arguments.resume_receipt_sha256,
    )


if __name__ == "__main__":
    main()
