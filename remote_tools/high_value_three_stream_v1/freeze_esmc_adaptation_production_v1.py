#!/usr/bin/env python3
"""Freeze equal-compute four-epoch direct ESM-C adaptation after batch-16 sizing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizing-protocol", type=Path, required=True)
    parser.add_argument("--sizing-receipt", type=Path, required=True)
    parser.add_argument("--sizing-analysis", type=Path, required=True)
    parser.add_argument("--sizing-analysis-receipt", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    expected = {
        "sizing_protocol": "1672791821459f23719e62cac165d61bc8a9655e917b8f3b71222b1e77b119e6",
        "sizing_receipt": "c2583bb88e70ed591a0e7ea734d0dd4d2c34d6233f4976c7396976be7788de93",
        "sizing_analysis": "e591ee4e7ac647e8327fd9ae409e4a9c58cb321b6a6d10a91f208ece03fb049c",
        "sizing_analysis_receipt": "d75689159a633e703a5e7530cddaa0b5f77a8a3449fd4eb0837165ec3fde6a09",
        "parent_protocol": "a7d32158d3ab27ca93d4e730abdb7c05d9c7d4e930091a0505bde67ca6c448ea",
    }
    for name, path in {
        "sizing_protocol": args.sizing_protocol,
        "sizing_receipt": args.sizing_receipt,
        "sizing_analysis": args.sizing_analysis,
        "sizing_analysis_receipt": args.sizing_analysis_receipt,
        "parent_protocol": args.parent_protocol,
    }.items():
        if sha256_file(path) != expected[name]:
            raise RuntimeError(f"{name} SHA differs")
    sizing = json.loads(args.sizing_analysis.read_text())
    sizing_receipt = json.loads(args.sizing_analysis_receipt.read_text())
    parent = json.loads(args.parent_protocol.read_text())
    if (
        sizing.get("status") != "PASS_WHOLE_CHAIN_AND_CDR_FOUR_GPU_CANARIES"
        or sizing_receipt.get("production_protocol_freeze_authorized") is not True
        or sizing_receipt.get("production_training_authorized") is not False
        or sizing_receipt.get("sabdab2_test_access") is not False
    ):
        raise RuntimeError("batch-16 sizing did not authorize production freeze")
    sizing_protocol = json.loads(args.sizing_protocol.read_text())
    train_rows = 757_640
    validation_rows = 15_415
    world_size = 4
    microbatch = 16
    accumulation = 4
    global_batch = world_size * microbatch * accumulation
    updates_per_epoch = train_rows // global_batch
    epochs = parent["optimization_after_model_binding"]["epochs_maximum"]
    if (epochs, parent["optimization_after_model_binding"]["checkpoint_rule"]) != (
        4,
        "pre-frozen epoch 4 for the adapted arms, matching the authors' validation-selected direct-CDR ESM-C checkpoint; no downstream metric selection",
    ):
        raise RuntimeError("parent fixed-epoch authority differs")
    protocol = {
        "format_version": "paired_esmc_adaptation_production_protocol_v1",
        "status": "FROZEN_EQUAL_COMPUTE_DIRECT_ADAPTATION_BEFORE_TRAINING",
        "research_question": "Does direct CDR-50% adaptation improve fixed-protocol antibody-interface transfer relative to equal-compute WC-15% adaptation from the same ESM-C 600M base?",
        "authority_chain": {
            "parent_protocol_sha256": expected["parent_protocol"],
            "batch16_sizing_protocol_sha256": expected["sizing_protocol"],
            "batch16_sizing_analysis_sha256": expected["sizing_analysis"],
            "batch16_sizing_analysis_receipt_sha256": expected["sizing_analysis_receipt"],
        },
        "model": sizing_protocol["model"],
        "data": {
            **sizing_protocol["data"],
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "official_abcdr_validation_or_test_rows_used": False,
            "sabdab2_development_decontaminated": True,
        },
        "arms": {
            "whole_chain_control": {
                "initialization": "frozen base ESM-C 600M",
                "eligible_tokens": "all VH and VL residue tokens",
                "mask_probability": 0.15,
            },
            "cdr_focused": {
                "initialization": "same frozen base ESM-C 600M",
                "eligible_tokens": "exact V5 AIRR-query-coordinate CDR1-3 spans on VH and VL",
                "mask_probability": 0.50,
            },
        },
        "optimization": {
            **sizing_protocol["optimization"],
            "scheduler": "linear warmup then linear decay over all four epochs",
        },
        "training": {
            "seed": 1701,
            "epochs": epochs,
            "fixed_checkpoint_epoch": 4,
            "checkpoint_selection": "fixed before training; MLM and downstream metrics cannot change it",
            "world_size": world_size,
            "per_rank_microbatch": microbatch,
            "gradient_accumulation_steps": accumulation,
            "global_batch_size": global_batch,
            "updates_per_epoch": updates_per_epoch,
            "total_updates": updates_per_epoch * epochs,
            "examples_dropped_per_epoch_after_deterministic_shuffle": train_rows - updates_per_epoch * global_batch,
            "log_every_updates": 100,
            "save_each_epoch": True,
        },
        "validation": {
            "split_rows": validation_rows,
            "whole_chain_seed": 910_017,
            "cdr_seed": 910_043,
            "fixed_corruption_masks_across_epochs_and_arms": True,
            "schemes": ["whole_chain_15pct", "cdr_50pct"],
            "used_for_checkpoint_selection": False,
        },
        "implementation": {
            "minimal_runtime_sha256": sizing_protocol["implementation"]["minimal_runtime_sha256"],
            "mlm_runtime_sha256": sizing_protocol["implementation"]["mlm_runtime_sha256"],
            "runner_sha256": sha256_file(args.runner),
            "launcher_sha256": sha256_file(args.launcher),
            "analyzer_sha256": sha256_file(args.analyzer),
        },
        "frozen_completion_gate": {
            "both_arms_complete": True,
            "nonfinite": 0,
            "all_epoch_checkpoint_sha_verified": True,
            "fixed_epoch4_checkpoint_sha_verified": True,
            "identical_base_state_probe": True,
            "all_validation_metrics_finite": True,
            "forbidden_access": False,
        },
        "claim_boundary": {
            "not_an_exact_ten_epoch_reproduction": True,
            "literature_informed_resource_bounded_four_epoch_study": True,
            "equal_compute_direct_from_base_comparison": True,
            "mlm_metrics_are_diagnostic_not_selection": True,
            "no_representation_gain_claim_before_fixed_downstream_development": True,
            "no_downstream_test_authorized": True,
            "abcdr_validation_or_test_rows_used": False,
            "task_labels_read_before_training": False,
            "sabdab2_test_access": False,
        },
    }
    atomic_write(args.output, protocol)
    receipt = {
        "format_version": "paired_esmc_adaptation_production_protocol_receipt_v1",
        "status": "FROZEN_AND_AUTHORIZED_EQUAL_COMPUTE_DIRECT_ADAPTATION",
        "protocol_sha256": sha256_file(args.output),
        "batch16_sizing_analysis_sha256": expected["sizing_analysis"],
        "production_training_authorized": True,
        "downstream_test_authorized": False,
        "task_labels_read": False,
        "sabdab2_test_access": False,
    }
    atomic_write(args.receipt, receipt)
    print(json.dumps({"protocol_sha256": receipt["protocol_sha256"], "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
