#!/usr/bin/env python3
"""Freeze an official-per-device-batch sizing canary after the V2 correctness pass."""

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


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-protocol", type=Path, required=True)
    parser.add_argument("--v2-receipt", type=Path, required=True)
    parser.add_argument("--v2-analysis", type=Path, required=True)
    parser.add_argument("--v2-analysis-receipt", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--official-training-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    expected = {
        "v2_protocol": "2bf8d8c1b73e9a9eff265404510606445d6f649dac3503bd43ed3a4797ff5276",
        "v2_receipt": "97761338f17102a6f0a1f960f95007741246efed72b5c65c559fbe3ddcd7a4ed",
        "v2_analysis": "b33a9e9ead4881d1ed3dcd8d9e4be22e0a373982e54c8374a312d55d98bb5e75",
        "v2_analysis_receipt": "ec65d6397158bb058aa4a76886b4a4c86e8f12d136c45af8bbd680ccfe6c5ade",
    }
    for name, path in {
        "v2_protocol": args.v2_protocol,
        "v2_receipt": args.v2_receipt,
        "v2_analysis": args.v2_analysis,
        "v2_analysis_receipt": args.v2_analysis_receipt,
    }.items():
        if sha256_file(path) != expected[name]:
            raise RuntimeError(f"{name} SHA differs")
    analysis = json.loads(args.v2_analysis.read_text())
    analysis_receipt = json.loads(args.v2_analysis_receipt.read_text())
    if (
        analysis.get("status") != "PASS_WHOLE_CHAIN_AND_CDR_FOUR_GPU_CANARIES"
        or analysis_receipt.get("production_protocol_freeze_authorized") is not True
        or analysis_receipt.get("production_training_authorized") is not False
        or analysis_receipt.get("sabdab2_test_access") is not False
    ):
        raise RuntimeError("V2 correctness gate did not authorize protocol sizing")

    protocol = json.loads(args.v2_protocol.read_text())
    protocol["format_version"] = "paired_esmc_adaptation_batch16_canary_protocol_v3"
    protocol["protocol_status"] = "frozen_hardware_sizing_after_v2_correctness_pass"
    protocol["research_question"] = (
        "Can both frozen masking arms use the released training source's per-device batch "
        "size of 16 on four RTX 4090 GPUs without changing model, data, or objectives?"
    )
    protocol["supersedes_v2_correctness_gate"] = {
        "protocol_sha256": expected["v2_protocol"],
        "analysis_sha256": expected["v2_analysis"],
        "analysis_receipt_sha256": expected["v2_analysis_receipt"],
    }
    protocol["official_source_binding"] = {
        "training_script_uri": str(args.official_training_script),
        "training_script_sha256": sha256_file(args.official_training_script),
        "released_per_device_train_batch_size": 16,
        "released_optimizer": "adamw_torch",
        "released_learning_rate": 0.00002,
        "released_warmup_ratio": 0.05,
        "released_weight_decay": 0.01,
        "released_precision": "bf16",
    }
    protocol["canary"].update({
        "sample_rows": 512,
        "updates": 3,
        "world_size": 4,
        "per_rank_microbatch": 16,
        "gradient_accumulation_steps": 1,
        "global_batch_size": 64,
        "checkpoint_saved": False,
    })
    protocol["frozen_gate"]["maximum_peak_allocated_fraction"] = 0.92
    protocol["implementation"]["runner_sha256"] = sha256_file(args.runner)
    protocol["implementation"]["launcher_sha256"] = sha256_file(args.launcher)
    protocol["implementation"]["analyzer_sha256"] = sha256_file(args.analyzer)
    protocol["claim_boundary"].update({
        "hardware_sizing_only": True,
        "production_training_authorized_before_sizing": False,
        "production_global_batch_if_passed": 256,
        "production_gradient_accumulation_if_passed": 4,
    })
    atomic_write(args.output, protocol)
    receipt = {
        "format_version": "paired_esmc_adaptation_batch16_canary_protocol_receipt_v3",
        "status": "FROZEN_HARDWARE_SIZING_AFTER_V2_CORRECTNESS_PASS",
        "protocol_sha256": sha256_file(args.output),
        "v2_analysis_sha256": expected["v2_analysis"],
        "production_training_authorized": False,
        "task_labels_read": False,
        "sabdab2_test_access": False,
    }
    atomic_write(args.receipt, receipt)
    print(json.dumps({"protocol_sha256": receipt["protocol_sha256"], "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
