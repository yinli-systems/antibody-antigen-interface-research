#!/usr/bin/env python3
"""Freeze a four-GPU whole-chain versus CDR-focused ESM-C training canary."""

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
    parser.add_argument("--span-result", type=Path, required=True)
    parser.add_argument("--span-receipt", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--source-access-receipt", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--mlm-runtime", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise FileExistsError("canary authority already exists")

    expected = {
        "base_checkpoint": "e4232c30fd35fe2f57051ec88a703996ac94520580b4b836894207a3d45d9ff8",
        "source": "d1628da898286d1de6833bcaa1998ade7b5715da815cbe247b59d68c79ec1b68",
        "parent_protocol": "a7d32158d3ab27ca93d4e730abdb7c05d9c7d4e930091a0505bde67ca6c448ea",
        "source_access_receipt": "f9c268ce42f66e9d45e23077f9cf8cebaaf35fa207f3fed3d909936ef2181518",
        "runtime": "4c2e07f89b9922f56fbbb020f66c0e88778cc2f02a97005fbf089119a658ef2b",
    }
    for name, path in {
        "base_checkpoint": args.base_checkpoint,
        "source": args.source,
        "parent_protocol": args.parent_protocol,
        "source_access_receipt": args.source_access_receipt,
        "runtime": args.runtime,
    }.items():
        if sha256_file(path) != expected[name]:
            raise RuntimeError(f"{name} SHA differs")
    span = json.loads(args.span_result.read_text())
    span_receipt = json.loads(args.span_receipt.read_text())
    if (
        span.get("status") != "PASS_EXACT_AIRR_QUERY_COORDINATE_GATE"
        or not all(span.get("checks", {}).values())
        or span.get("authorization", {}).get("local_whole_chain_vs_cdr_canary_authorized") is not True
        or span_receipt.get("result_sha256") != sha256_file(args.span_result)
        or span_receipt.get("span_artifact_sha256") != span["span_artifact"]["sha256"]
        or span_receipt.get("abcdr_validation_or_test_rows_used") is not False
        or span_receipt.get("sabdab2_test_access") is not False
    ):
        raise RuntimeError("exact CDR span gate did not authorize canary")

    protocol = {
        "format_version": "paired_esmc_adaptation_canary_protocol_v1",
        "protocol_status": "frozen_after_exact_span_gate_before_training",
        "research_question": "Can the exact base ESM-C 600M weights be stably adapted on the same paired corpus under whole-chain 15% and CDR-only 50% masking on four RTX 4090 GPUs?",
        "model": {
            "model_id": "biohub/ESMC-600M",
            "revision": "a7e82012c83126b9eedb055fea9fa84b6c02f094",
            "weight_sha256": expected["base_checkpoint"],
            "parameter_count_body": 573_632_640,
            "embedding_dimension": 1152,
            "layers": 36,
            "heads": 18,
        },
        "data": {
            "source_uri": str(args.source),
            "source_sha256": expected["source"],
            "source_rows": 776_634,
            "span_uri": span["span_artifact"]["path"],
            "span_sha256": span["span_artifact"]["sha256"],
            "eligible_rows": span["counts"]["eligible_rows"],
            "selection": "smallest sha256('esmc-canary-v1|' + row_hash) among V5 train rows",
            "paired_format": "BOS + VH + literal '-' + VL + EOS",
            "full_sequence_used_without_truncation": True,
        },
        "source_code_correction": {
            "parent_protocol_sha256": expected["parent_protocol"],
            "correction": "The released dataset constructor accepts max_length=164 but never passes it to the tokenizer and performs no truncation; this canary therefore preserves each full quality-filtered pair rather than imposing an unsupported 164-token truncation.",
            "maximum_quality_filtered_pair_residues_including_separator": 301,
        },
        "arms": {
            "whole_chain_control": {"eligible_tokens": "all VH and VL residue tokens", "mask_probability": 0.15},
            "cdr_focused": {"eligible_tokens": "exact V5 AIRR-query-coordinate CDR1-3 spans on VH and VL", "mask_probability": 0.50},
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": 0.00002,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "scheduler": "linear_decay",
            "precision": "bf16",
            "gradient_clip_norm": 1.0,
            "corruption": "BERT 80/10/10; random token sampled from the 33-token ESM-C tokenizer vocabulary",
        },
        "canary": {
            "seed": 1701,
            "sample_rows": 512,
            "updates": 12,
            "world_size": 4,
            "per_rank_microbatch": 1,
            "gradient_accumulation_steps": 4,
            "global_batch_size": 16,
            "checkpoint_saved": False,
        },
        "frozen_gate": {
            "world_size": 4,
            "nonfinite": 0,
            "maximum_peak_allocated_fraction": 0.95,
            "whole_chain_masked_fraction_range": [0.12, 0.18],
            "cdr_masked_fraction_range": [0.40, 0.60],
            "require_identical_initial_state_probe": True,
            "require_changed_final_state_probe": True,
            "require_finite_positive_loss_and_gradient": True,
        },
        "implementation": {
            "minimal_runtime_sha256": expected["runtime"],
            "mlm_runtime_sha256": sha256_file(args.mlm_runtime),
            "runner_sha256": sha256_file(args.runner),
            "launcher_sha256": sha256_file(args.launcher),
            "analyzer_sha256": sha256_file(args.analyzer),
        },
        "claim_boundary": {
            "system_correctness_canary_only": True,
            "published_abcdr_checkpoint_used_for_training": False,
            "production_training_authorized_before_canary": False,
            "downstream_gain_claim": False,
            "abcdr_validation_or_test_rows_used": False,
            "task_labels_read": False,
            "sabdab2_old_test_accessed": False,
            "sabdab2_temporal_holdout_accessed": False,
            "source_access_receipt_sha256": expected["source_access_receipt"],
        },
    }
    atomic_write(args.output, protocol)
    receipt = {
        "format_version": "paired_esmc_adaptation_canary_protocol_receipt_v1",
        "status": "FROZEN_AFTER_EXACT_SPAN_GATE_BEFORE_TRAINING",
        "protocol_sha256": sha256_file(args.output),
        "span_result_sha256": sha256_file(args.span_result),
        "span_receipt_sha256": sha256_file(args.span_receipt),
        "production_training_authorized": False,
        "task_labels_read": False,
        "sabdab2_test_access": False,
    }
    atomic_write(args.receipt, receipt)
    print(json.dumps({"protocol_sha256": receipt["protocol_sha256"], "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
