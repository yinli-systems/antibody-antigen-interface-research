#!/usr/bin/env python3
"""Audit exact DCP recovery of the optimized antibody training candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PARAMETER_COUNT = 634_513_920
EXPECTED_MANIFEST_SHA256 = (
    "0f73ccf7f537d223ca6d94b83f08b40a34c23e557f2a747cc32d798c974450f6"
)
EXPECTED_MODEL_FAMILY_SHA256 = (
    "1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015"
)
EXPECTED_TRAINING_PROTOCOL_SHA256 = (
    "c8974cb06350da46ca49ffff61c149061b7e7f46b43c9e686176539bbe223e6b"
)
EXPECTED_BUCKETS = [96, 128, 144, 194]
LEGACY_OPTIMIZATION = {
    "bucketed_padding": True,
    "compile_blocks": True,
    "compile_prewarm_buckets": True,
    "ddp_bucket_cap_mb": 25,
    "disable_compiled_cudagraphs": True,
    "fsdp_output_dtype": "bfloat16",
    "fused_adamw": True,
    "length_bucketing": False,
    "pinned_staging": True,
}
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
TRACE_KEYS = (
    "consumed_residue_tokens",
    "gradient_norm",
    "learning_rate",
    "loss",
    "micro_steps",
    "residue_tokens",
    "selected_targets",
    "sequence_count",
    "update",
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_equal(left: list[dict[str, object]], right: list[dict[str, object]]) -> bool:
    return len(left) == len(right) and all(
        all(left_item[key] == right_item[key] for key in TRACE_KEYS)
        for left_item, right_item in zip(left, right)
    )


def _validate_receipt(
    receipt: dict[str, object],
    *,
    resume_used: bool,
    updates: list[int],
    checkpoint_format: str | None,
    checkpoint_payload_digests_required: bool,
    efficiency_protocol_sha256: str,
    expected_optimization: dict[str, object],
    checkpoint_rng_manifest_sha256: str | None,
    resume_loaded_digests_required: bool,
    resume_payload_digests_required: bool,
    resume_rng_manifest_sha256: str | None,
) -> None:
    expected = {
        "backend": "ddp",
        "batch_quantum": 128,
        "checkpoint_format": checkpoint_format,
        "checkpoint_rng_manifest_sha256": checkpoint_rng_manifest_sha256,
        "efficiency_protocol_sha256": efficiency_protocol_sha256,
        "format_version": "antibody_pretraining_optimized_pilot_v1",
        "length_bucket_boundaries": EXPECTED_BUCKETS,
        "micro_batch_per_gpu": 16,
        "model_family_sha256": EXPECTED_MODEL_FAMILY_SHA256,
        "optimization": expected_optimization,
        "preset": "abf_650m",
        "resume_used": resume_used,
        "resume_rng_manifest_sha256": resume_rng_manifest_sha256,
        "shard_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "target_residue_tokens": 16384,
        "test_access": TEST_ACCESS,
        "training_protocol_sha256": EXPECTED_TRAINING_PROTOCOL_SHA256,
        "world_size": 8,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Optimized recovery receipt failed validation")
    loaded_digests = (
        receipt.get("resume_loaded_model_sha256"),
        receipt.get("resume_loaded_optimizer_sha256"),
    )
    if resume_loaded_digests_required:
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in loaded_digests
        ):
            raise RuntimeError("Loaded checkpoint digests are invalid")
    elif loaded_digests != (None, None):
        raise RuntimeError("Non-resumed run reports loaded checkpoint digests")
    payload_groups = {
        "checkpoint": (
            receipt.get("checkpoint_payload_model_sha256"),
            receipt.get("checkpoint_payload_optimizer_sha256"),
            checkpoint_payload_digests_required,
        ),
        "resume": (
            receipt.get("resume_payload_model_sha256"),
            receipt.get("resume_payload_optimizer_sha256"),
            resume_payload_digests_required,
        ),
    }
    for label, (model_digest, optimizer_digest, required) in payload_groups.items():
        digests = (model_digest, optimizer_digest)
        if required:
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            ):
                raise RuntimeError(f"{label} DCP payload digests are invalid")
        elif digests != (None, None):
            raise RuntimeError(f"Unexpected {label} DCP payload digests")
    if receipt.get("parameter_moments", {}).get("count") != PARAMETER_COUNT:
        raise RuntimeError("Optimized recovery parameter count differs")
    if [item["update"] for item in receipt.get("trace", [])] != updates:
        raise RuntimeError("Optimized recovery update coordinates differ")
    arguments = receipt.get("launch_argv")
    if not isinstance(arguments, list):
        raise RuntimeError("Optimized recovery launch arguments are absent")
    if ("--resume-root" in arguments) != resume_used:
        raise RuntimeError("Optimized recovery resume launch switch differs")


def _validate_state(
    state: dict[str, object],
    *,
    next_update: int,
    consumed_residue_tokens: int,
    expected_optimization: dict[str, object],
    checkpoint_rng_manifest_sha256: str | None,
) -> None:
    expected = {
        "backend": "ddp",
        "batch_quantum": 128,
        "checkpoint_rng_manifest_sha256": checkpoint_rng_manifest_sha256,
        "consumed_residue_tokens": consumed_residue_tokens,
        "length_bucket_boundaries": EXPECTED_BUCKETS,
        "micro_batch_per_gpu": 16,
        "model_family_sha256": EXPECTED_MODEL_FAMILY_SHA256,
        "next_update": next_update,
        "optimization": expected_optimization,
        "preset": "abf_650m",
        "shard_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "target_residue_tokens": 16384,
        "training_protocol_sha256": EXPECTED_TRAINING_PROTOCOL_SHA256,
        "world_size": 8,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Optimized recovery trainer state failed validation")
    if int(state.get("sample_cursor", -1)) != next_update * 256:
        raise RuntimeError("Optimized recovery sampler cursor differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--rng-manifest", type=Path)
    parser.add_argument("--efficiency-protocol", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--require-enabled-compiled-cudagraphs", action="store_true")
    parser.add_argument("--require-deterministic-algorithms", action="store_true")
    parser.add_argument("--require-same-hostname", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.job_id) != 3 or len(set(args.job_id)) != 3:
        raise RuntimeError("Recovery audit requires three distinct job IDs")
    expected_optimization = dict(LEGACY_OPTIMIZATION)
    if args.require_enabled_compiled_cudagraphs:
        expected_optimization["disable_compiled_cudagraphs"] = False
    if args.require_deterministic_algorithms:
        expected_optimization["deterministic_algorithms"] = True
    if args.require_deterministic_algorithms and args.rng_manifest is None:
        raise RuntimeError("Deterministic recovery requires an RNG manifest")
    if not args.require_deterministic_algorithms and args.rng_manifest is not None:
        raise RuntimeError("Legacy recovery cannot attach an RNG manifest")

    paths = {
        "continuous_receipt": args.continuous / "receipt.json",
        "continuous_state": args.continuous / "trainer_state.json",
        "stage1_receipt": args.stage1 / "receipt.json",
        "stage1_state": args.stage1 / "trainer_state.json",
        "resumed_receipt": args.resumed / "receipt.json",
        "resumed_state": args.resumed / "trainer_state.json",
        "checkpoint_manifest": args.checkpoint_manifest,
    }
    if args.rng_manifest is not None:
        paths["rng_manifest"] = args.rng_manifest
    values = {name: _load(path) for name, path in paths.items() if name != "checkpoint_manifest"}
    efficiency_protocol_sha256 = _sha256(args.efficiency_protocol)
    if _load(args.efficiency_protocol).get("format_version") != "antibody_efficiency_gate_v1":
        raise RuntimeError("Unexpected efficiency protocol")
    continuous = values["continuous_receipt"]
    stage1 = values["stage1_receipt"]
    resumed = values["resumed_receipt"]
    continuous_state = values["continuous_state"]
    stage1_state = values["stage1_state"]
    resumed_state = values["resumed_state"]
    rng_manifest_sha256 = None
    if args.rng_manifest is not None:
        rng_manifest = _load(args.rng_manifest)
        unsigned_rng_manifest = dict(rng_manifest)
        rng_manifest_sha256 = unsigned_rng_manifest.pop("manifest_sha256", None)
        observed_rng_manifest_sha256 = hashlib.sha256(
            (
                json.dumps(
                    unsigned_rng_manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("ascii")
        ).hexdigest()
        if (
            rng_manifest_sha256 != observed_rng_manifest_sha256
            or rng_manifest.get("format_version")
            != "antibody_rng_checkpoint_manifest_v1"
            or rng_manifest.get("world_size") != 8
            or len(rng_manifest.get("files", [])) != 8
        ):
            raise RuntimeError("Recovery RNG manifest failed validation")
        for expected_rank, entry in enumerate(rng_manifest["files"]):
            expected_path = f"rng/rank-{expected_rank:05d}.json"
            if (
                entry.get("rank") != expected_rank
                or entry.get("path") != expected_path
            ):
                raise RuntimeError("Recovery RNG rank mapping differs")
            state_path = args.rng_manifest.parent / expected_path
            if (
                not state_path.is_file()
                or state_path.is_symlink()
                or _sha256(state_path) != entry.get("file_sha256")
            ):
                raise RuntimeError("Recovery RNG rank file differs")
            state = _load(state_path)
            unsigned_state = dict(state)
            recorded_state_sha256 = unsigned_state.pop("state_sha256", None)
            observed_state_sha256 = hashlib.sha256(
                (
                    json.dumps(
                        unsigned_state,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("ascii")
            ).hexdigest()
            if (
                state.get("format_version") != "antibody_rank_rng_state_v1"
                or state.get("rank") != expected_rank
                or recorded_state_sha256 != observed_state_sha256
                or recorded_state_sha256 != entry.get("state_sha256")
            ):
                raise RuntimeError("Recovery RNG rank state differs")
    _validate_receipt(
        continuous,
        resume_used=False,
        updates=list(range(1, 21)),
        checkpoint_format=None,
        checkpoint_payload_digests_required=False,
        efficiency_protocol_sha256=efficiency_protocol_sha256,
        expected_optimization=expected_optimization,
        checkpoint_rng_manifest_sha256=None,
        resume_loaded_digests_required=False,
        resume_payload_digests_required=False,
        resume_rng_manifest_sha256=None,
    )
    _validate_receipt(
        stage1,
        resume_used=False,
        updates=list(range(1, 11)),
        checkpoint_format="torch_distributed_checkpoint",
        checkpoint_payload_digests_required=True,
        efficiency_protocol_sha256=efficiency_protocol_sha256,
        expected_optimization=expected_optimization,
        checkpoint_rng_manifest_sha256=rng_manifest_sha256,
        resume_loaded_digests_required=False,
        resume_payload_digests_required=False,
        resume_rng_manifest_sha256=None,
    )
    _validate_receipt(
        resumed,
        resume_used=True,
        updates=list(range(11, 21)),
        checkpoint_format=None,
        checkpoint_payload_digests_required=False,
        efficiency_protocol_sha256=efficiency_protocol_sha256,
        expected_optimization=expected_optimization,
        checkpoint_rng_manifest_sha256=None,
        resume_loaded_digests_required=True,
        resume_payload_digests_required=True,
        resume_rng_manifest_sha256=rng_manifest_sha256,
    )
    _validate_state(
        continuous_state,
        next_update=20,
        consumed_residue_tokens=int(continuous["trace"][-1]["consumed_residue_tokens"]),
        expected_optimization=expected_optimization,
        checkpoint_rng_manifest_sha256=None,
    )
    _validate_state(
        stage1_state,
        next_update=10,
        consumed_residue_tokens=int(stage1["trace"][-1]["consumed_residue_tokens"]),
        expected_optimization=expected_optimization,
        checkpoint_rng_manifest_sha256=rng_manifest_sha256,
    )
    _validate_state(
        resumed_state,
        next_update=20,
        consumed_residue_tokens=int(resumed["trace"][-1]["consumed_residue_tokens"]),
        expected_optimization=expected_optimization,
        checkpoint_rng_manifest_sha256=None,
    )
    hostnames = {
        receipt.get("allocation", {}).get("hostname")
        for receipt in (continuous, stage1, resumed)
    }
    if args.require_same_hostname and (
        len(hostnames) != 1 or None in hostnames or "" in hostnames
    ):
        raise RuntimeError("Recovery runs do not share one authenticated hostname")
    manifest_lines = args.checkpoint_manifest.read_text(encoding="ascii").splitlines()
    if not manifest_lines or not any(".metadata" in line for line in manifest_lines):
        raise RuntimeError("DCP checkpoint manifest is incomplete")
    exact = {
        "final_model": (
            continuous["final_model_sha256"] == resumed["final_model_sha256"]
        ),
        "final_optimizer": (
            continuous["final_optimizer_sha256"]
            == resumed["final_optimizer_sha256"]
        ),
        "loaded_model": (
            resumed["resume_loaded_model_sha256"]
            == stage1["final_model_sha256"]
        ),
        "loaded_optimizer": (
            resumed["resume_loaded_optimizer_sha256"]
            == stage1["final_optimizer_sha256"]
        ),
        "saved_model_payload": (
            stage1["checkpoint_payload_model_sha256"]
            == stage1["final_model_sha256"]
        ),
        "serialized_model_payload": (
            resumed["resume_payload_model_sha256"]
            == stage1["checkpoint_payload_model_sha256"]
        ),
        "serialized_optimizer_payload": (
            resumed["resume_payload_optimizer_sha256"]
            == stage1["checkpoint_payload_optimizer_sha256"]
        ),
        "prefix_trace": _trace_equal(continuous["trace"][:10], stage1["trace"]),
        "resumed_trace": _trace_equal(continuous["trace"][10:], resumed["trace"]),
        "trainer_state": continuous_state == resumed_state,
        "rng_manifest": (
            rng_manifest_sha256 is None
            or (
                stage1.get("checkpoint_rng_manifest_sha256")
                == rng_manifest_sha256
                and resumed.get("resume_rng_manifest_sha256")
                == rng_manifest_sha256
            )
        ),
    }
    passed = all(exact.values())
    trace_differences = {
        "prefix_maximum_gradient_norm_absolute_difference": max(
            abs(float(left["gradient_norm"]) - float(right["gradient_norm"]))
            for left, right in zip(continuous["trace"][:10], stage1["trace"])
        ),
        "prefix_maximum_loss_absolute_difference": max(
            abs(float(left["loss"]) - float(right["loss"]))
            for left, right in zip(continuous["trace"][:10], stage1["trace"])
        ),
        "resumed_maximum_gradient_norm_absolute_difference": max(
            abs(float(left["gradient_norm"]) - float(right["gradient_norm"]))
            for left, right in zip(continuous["trace"][10:], resumed["trace"])
        ),
        "resumed_maximum_loss_absolute_difference": max(
            abs(float(left["loss"]) - float(right["loss"]))
            for left, right in zip(continuous["trace"][10:], resumed["trace"])
        ),
    }
    payload = {
        "auditor_source_sha256": _sha256(Path(__file__)),
        "authorization": {
            "pilot_16_gpu": passed,
            "pilot_32_gpu": False,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "checkpoint_manifest_sha256": _sha256(args.checkpoint_manifest),
        "claim_boundary": "Eight-GPU repeated-shard DDP checkpoint recovery; no production-corpus, biological, strong-scaling, or state-of-the-art claim",
        "exact_equal": exact,
        "format_version": "antibody_optimized_recovery_audit_v1",
        "input_sha256": {name: _sha256(path) for name, path in paths.items()},
        "job_ids": args.job_id,
        "required_deterministic_algorithms": (
            args.require_deterministic_algorithms
        ),
        "required_enabled_compiled_cudagraphs": (
            args.require_enabled_compiled_cudagraphs
        ),
        "required_same_hostname": args.require_same_hostname,
        "run_hostname": next(iter(hostnames)) if len(hostnames) == 1 else None,
        "recovery_passed": passed,
        "source": {
            "archive_sha256": args.source_archive_sha256,
            "runner_sha256": args.runner_sha256,
        },
        "test_access": TEST_ACCESS,
        "trace_differences": trace_differences,
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


if __name__ == "__main__":
    main()
