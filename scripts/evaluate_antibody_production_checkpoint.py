#!/usr/bin/env python3
"""Evaluate one authenticated DDP checkpoint on a frozen OAS validation panel."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from antibody_pretraining.masking import apply_uniform_mlm
from antibody_pretraining.model import (
    AntibodyMaskedLanguageModel,
    config_from_preset,
    load_model_family,
)
from antibody_pretraining.production_corpus import load_production_corpus
from antibody_pretraining.tokenizer import PAD_ID
from antibody_pretraining.training_contract import (
    load_training_protocol,
    sample_indices,
)
import run_antibody_pretraining_optimized as engine


FORMAT_VERSION = "antibody_production_checkpoint_evaluation_v1"
RECOVERY_GATE_FORMAT = "antibody_production_loader_recovery_gate_v1"
RECOVERY_RUN_FORMAT = "antibody_production_loader_recovery_run_v1"
PRETRAINING_GATE_FORMAT = "antibody_four_gpu_pretraining_segment_gate_v1"
PRETRAINING_RUN_FORMAT = "antibody_four_gpu_pretraining_segment_run_v1"
RECOVERY_RUN_AUTHORIZATION = {
    "loader_recovery_gate": True,
    "production_pretraining": False,
    "scale_out": False,
    "scientific_claim": False,
}
PRETRAINING_RUN_AUTHORIZATION = {
    "configuration_promotion": True,
    "downstream_transfer": False,
    "production_pretraining": True,
    "scientific_claim": False,
}
GATE_FORMAT = RECOVERY_GATE_FORMAT
RUN_FORMAT = RECOVERY_RUN_FORMAT
AUTHORIZATION = {
    "checkpoint_evaluation": True,
    "configuration_promotion": False,
    "production_pretraining": False,
    "scientific_claim": False,
}
TEST_ACCESS = {
    "downstream_labels_used": False,
    "external_test_accessed": False,
    "frozen_test_accessed": False,
    "official_test_accessed": False,
    "sabdab2_heldout_accessed": False,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-corpus-root", type=Path, required=True)
    parser.add_argument("--production-corpus-manifest-file-sha256", required=True)
    parser.add_argument("--production-corpus-manifest-payload-sha256", required=True)
    parser.add_argument("--production-corpus-protocol-file-sha256", required=True)
    parser.add_argument(
        "--production-corpus-independent-audit", type=Path, required=True
    )
    parser.add_argument(
        "--production-corpus-independent-audit-file-sha256", required=True
    )
    parser.add_argument(
        "--production-corpus-independent-audit-payload-sha256", required=True
    )
    parser.add_argument("--gate-protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-family", type=Path, required=True)
    parser.add_argument("--training-protocol", type=Path, required=True)
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("abf_300m", "abf_650m"), required=True)
    parser.add_argument("--panel-sample-count", type=int, required=True)
    parser.add_argument("--panel-seed", type=int, required=True)
    parser.add_argument("--panel-indices-sha256", required=True)
    parser.add_argument("--mask-seed", type=int, required=True)
    parser.add_argument("--mask-update", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, required=True)
    parser.add_argument("--maximum-open-arrays", type=int, required=True)
    parser.add_argument(
        "--authorization-mode",
        choices=("recovery_gate", "production_pretraining"),
        default="recovery_gate",
    )
    args = parser.parse_args()
    positive = (
        args.panel_sample_count,
        args.micro_batch,
        args.maximum_open_arrays,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Evaluation dimensions must be positive")
    if min(args.panel_seed, args.mask_seed, args.mask_update) < 0:
        raise ValueError("Evaluation seeds and update must be non-negative")
    if not _is_sha256(args.panel_indices_sha256):
        raise ValueError("Panel index anchor is not a SHA-256")
    return args


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _load_json(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON input is not an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _mode_contract(authorization_mode: str) -> tuple[str, str, dict[str, bool]]:
    if authorization_mode == "recovery_gate":
        return (
            RECOVERY_GATE_FORMAT,
            RECOVERY_RUN_FORMAT,
            RECOVERY_RUN_AUTHORIZATION,
        )
    if authorization_mode == "production_pretraining":
        return (
            PRETRAINING_GATE_FORMAT,
            PRETRAINING_RUN_FORMAT,
            PRETRAINING_RUN_AUTHORIZATION,
        )
    raise ValueError("Unknown authorization mode")


def _publish(path: Path, value: dict[str, object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _panel_indices(
    *, record_count: int, sample_count: int, seed: int
) -> np.ndarray:
    if sample_count > record_count:
        raise RuntimeError("Validation panel exceeds the audited split")
    indices = sample_indices(
        record_count=record_count,
        sample_cursor=0,
        sample_count=sample_count,
        seed=seed,
    ).astype("<i8", copy=False)
    if len(np.unique(indices)) != sample_count:
        raise RuntimeError("Validation panel contains duplicate indices")
    return indices


def _engine_arguments(optimization: dict[str, object]) -> argparse.Namespace:
    return argparse.Namespace(
        audit_gradient_rank_parity=False,
        backend="ddp",
        compile_blocks=optimization["compile_blocks"],
        compile_prewarm_buckets=optimization["compile_prewarm_buckets"],
        ddp_batched_grad_copy=False,
        ddp_bucket_cap_mb=optimization["ddp_bucket_cap_mb"],
        ddp_communication_dtype="float32",
        disable_compiled_cudagraphs=optimization[
            "disable_compiled_cudagraphs"
        ],
        disable_ddp_gradient_as_bucket_view=False,
        fsdp_output_dtype="bfloat16",
    )


def main() -> None:
    args = _arguments()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size))):
        raise RuntimeError("Evaluation requires one local GPU node")
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        raise RuntimeError("CUDA rank identity is invalid")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(minutes=10),
        device_id=device,
    )
    gate, gate_sha256 = _load_json(args.gate_protocol)
    run, run_file_sha256 = _load_json(args.run_root / "receipt.json")
    state, state_file_sha256 = _load_json(args.run_root / "trainer_state.json")
    optimization = gate.get("execution", {}).get("optimization")
    gate_format, run_format, run_authorization = _mode_contract(
        args.authorization_mode
    )
    if (
        gate.get("format_version") != gate_format
        or gate.get("execution", {}).get("world_size") != world_size
        or not isinstance(optimization, dict)
        or run.get("format_version") != run_format
        or run.get("authorization") != run_authorization
        or run.get("gate_protocol_sha256") != gate_sha256
        or run.get("world_size") != world_size
        or state.get("world_size") != world_size
        or run.get("checkpoint_model_sha256") != run.get("final_model_sha256")
        or run.get("checkpoint_optimizer_sha256") is None
        or run.get("test_access")
        != {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        }
    ):
        raise RuntimeError("Checkpoint, gate, or evaluation identity differs")
    family = load_model_family(args.model_family)
    training = load_training_protocol(args.training_protocol)
    model_family_sha256 = engine._sha256(args.model_family)
    training_protocol_sha256 = engine._sha256(args.training_protocol)
    _, efficiency_protocol_sha256 = engine._load_efficiency_protocol(
        args.efficiency_protocol
    )
    if (
        run.get("model_family_sha256") != model_family_sha256
        or run.get("training_protocol_sha256") != training_protocol_sha256
        or run.get("efficiency_protocol_sha256") != efficiency_protocol_sha256
        or run.get("optimization") != optimization
        or run.get("preset") != args.preset
    ):
        raise RuntimeError("Checkpoint source closure differs")

    corpus_arguments = {
        "split": "validation",
        "expected_manifest_file_sha256": (
            args.production_corpus_manifest_file_sha256
        ),
        "expected_manifest_payload_sha256": (
            args.production_corpus_manifest_payload_sha256
        ),
        "expected_protocol_file_sha256": (
            args.production_corpus_protocol_file_sha256
        ),
        "independent_audit_path": args.production_corpus_independent_audit,
        "expected_independent_audit_file_sha256": (
            args.production_corpus_independent_audit_file_sha256
        ),
        "expected_independent_audit_payload_sha256": (
            args.production_corpus_independent_audit_payload_sha256
        ),
        "maximum_open_arrays": args.maximum_open_arrays,
    }
    corpus = None
    if local_rank == 0:
        corpus = load_production_corpus(
            args.production_corpus_root,
            **corpus_arguments,
            verify_array_file_hashes=True,
        )
    dist.barrier()
    if local_rank != 0:
        corpus = load_production_corpus(
            args.production_corpus_root,
            **corpus_arguments,
            verify_array_file_hashes=False,
        )
    if corpus is None:
        raise RuntimeError("Validation corpus did not open")
    expected_hash_mode = local_rank == 0
    if corpus.contract.array_file_hashes_verified is not expected_hash_mode:
        raise RuntimeError("Distributed validation hash-verification mode differs")
    corpus_counts: list[int | None] = [None] * world_size
    dist.all_gather_object(corpus_counts, len(corpus))
    if len(set(corpus_counts)) != 1:
        raise RuntimeError("Distributed validation record counts differ")
    dist.barrier()
    panel = _panel_indices(
        record_count=len(corpus),
        sample_count=args.panel_sample_count,
        seed=args.panel_seed,
    )
    panel_sha256 = hashlib.sha256(panel.tobytes()).hexdigest()
    if panel_sha256 != args.panel_indices_sha256:
        raise RuntimeError("Validation panel anchor differs")
    local_indices = panel[rank::world_size]
    if len(local_indices) % args.micro_batch:
        raise RuntimeError("Panel is not divisible by world size and microbatch")

    model = AntibodyMaskedLanguageModel(
        config_from_preset(family, args.preset)
    ).to(device)
    model = engine._wrap_model(
        model,
        args=_engine_arguments(optimization),
        device=device,
        local_rank=local_rank,
        world_size=world_size,
    )
    optimizer = engine._optimizer(
        model,
        training,
        args.preset,
        bool(optimization["fused_adamw"]),
    )
    engine._load_authenticated_dcp(
        model,
        optimizer,
        args.run_root / "checkpoint",
        backend="ddp",
        rank=rank,
    )
    module = model.module if hasattr(model, "module") else model
    observed_model_sha256 = engine._replicated_rank_digests(
        module.state_dict(), "evaluation model"
    )[0]
    if observed_model_sha256 != run["final_model_sha256"]:
        raise RuntimeError("Loaded evaluation model differs from checkpoint")

    boundaries = engine._parse_boundaries(
        gate["execution"]["length_bucket_boundaries"],
        family["maximum_sequence_tokens"],
    )
    model.eval()
    local_records: list[dict[str, object]] = []
    with torch.no_grad():
        for start in range(0, len(local_indices), args.micro_batch):
            indices = local_indices[start : start + args.micro_batch]
            batch = corpus.batch(
                indices,
                fields=(
                    "chain_types.npy",
                    "lengths.npy",
                    "sample_keys.npy",
                    "tokens.npy",
                ),
            )
            masked = apply_uniform_mlm(
                batch["tokens.npy"].astype(np.int64, copy=False),
                batch["sample_keys.npy"],
                update=args.mask_update,
                seed=args.mask_seed,
                mask_fraction=training["masking"]["fraction"],
            )
            maximum_length = int(batch["lengths.npy"].max())
            width = engine._bucket_width(maximum_length, boundaries)
            input_ids = torch.from_numpy(
                np.ascontiguousarray(masked.input_ids[:, :width])
            ).to(device)
            labels = torch.from_numpy(
                np.ascontiguousarray(masked.labels[:, :width])
            ).to(device)
            chain = torch.from_numpy(
                batch["chain_types.npy"].astype(np.int64, copy=False)
            ).to(device)
            attention = input_ids.ne(PAD_ID)
            type_ids = chain[:, None].expand_as(input_ids).masked_fill(~attention, 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids, type_ids, attention)
            selected = labels.ne(-100)
            losses = F.cross_entropy(
                output.logits.float().flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="none",
            ).view_as(labels)
            correct = (output.logits.argmax(dim=-1).eq(labels) & selected).sum(dim=1)
            loss_sums = losses.sum(dim=1)
            target_counts = selected.sum(dim=1)
            for offset, index in enumerate(indices):
                targets = int(target_counts[offset])
                if targets <= 0:
                    raise RuntimeError("Validation sample has no MLM targets")
                local_records.append(
                    {
                        "correct": int(correct[offset]),
                        "index": int(index),
                        "loss_sum": float(loss_sums[offset]),
                        "masked_accuracy": int(correct[offset]) / targets,
                        "masked_cross_entropy": float(loss_sums[offset]) / targets,
                        "sample_key": int(batch["sample_keys.npy"][offset]),
                        "selected_targets": targets,
                    }
                )
    gathered: list[list[dict[str, object]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_records)
    if rank == 0:
        records = sorted(
            (record for group in gathered if group for record in group),
            key=lambda item: item["index"],
        )
        if [record["index"] for record in records] != sorted(map(int, panel)):
            raise RuntimeError("Gathered validation membership differs")
        total_targets = sum(int(record["selected_targets"]) for record in records)
        total_correct = sum(int(record["correct"]) for record in records)
        total_loss = sum(float(record["loss_sum"]) for record in records)
        _publish(
            args.output,
            {
                "aggregate": {
                    "masked_accuracy": total_correct / total_targets,
                    "masked_cross_entropy": total_loss / total_targets,
                    "selected_targets": total_targets,
                },
                "allocation": {
                    "hostname": socket.gethostname(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                },
                "authorization": AUTHORIZATION,
                "checkpoint": {
                    "final_model_sha256": run["final_model_sha256"],
                    "receipt_file_sha256": run_file_sha256,
                    "root": str(args.run_root),
                    "trainer_state_file_sha256": state_file_sha256,
                },
                "evaluated_run_authorization": run_authorization,
                "claim_boundary": (
                    "Frozen OAS validation-panel checkpoint evaluation only; "
                    "no configuration promotion, production, downstream, "
                    "official-test, biological, or SOTA claim"
                ),
                "format_version": FORMAT_VERSION,
                "gate_protocol_sha256": gate_sha256,
                "panel": {
                    "indices_sha256": panel_sha256,
                    "mask_seed": args.mask_seed,
                    "mask_update": args.mask_update,
                    "record_count": len(records),
                    "seed": args.panel_seed,
                    "split": "validation",
                },
                "records": records,
                "test_access": TEST_ACCESS,
                "world_size": world_size,
            },
        )
    corpus.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
