#!/usr/bin/env python3
"""Run the authenticated 32-complex PAIRFormer contact-only overfit gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from interaction_first.development_contact_dataset import DevelopmentContactDataset
from interaction_first.pairscale_overfit import PairscaleOverfitConfig, run_overfit_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--receipt-file-sha256", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--subset-manifest-sha256", required=True)
    parser.add_argument("--authorization-manifest-sha256", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--positive-weight", type=float, default=1024.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume-checkpoint-root", type=Path)
    parser.add_argument("--resume-generation")
    parser.add_argument("--resume-receipt-sha256")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    dataset = DevelopmentContactDataset(
        arguments.cache_root,
        expected_manifest_sha256=arguments.manifest_sha256,
        expected_receipt_file_sha256=arguments.receipt_file_sha256,
        expected_policy_sha256=arguments.policy_sha256,
        expected_decision_sha256=arguments.decision_sha256,
        expected_subset_manifest_sha256=arguments.subset_manifest_sha256,
        expected_authorization_manifest_sha256=(
            arguments.authorization_manifest_sha256
        ),
    )
    config = PairscaleOverfitConfig(
        seed=arguments.seed,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        positive_weight=arguments.positive_weight,
        eval_every_epochs=arguments.eval_every,
        checkpoint_every_epochs=arguments.checkpoint_every,
        precision=arguments.precision,
    )
    run_overfit_training(
        dataset=dataset,
        output_root=arguments.output_root,
        config=config,
        resume_checkpoint_root=arguments.resume_checkpoint_root,
        resume_generation=arguments.resume_generation,
        expected_resume_receipt_sha256=arguments.resume_receipt_sha256,
    )


if __name__ == "__main__":
    main()
