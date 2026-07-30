#!/usr/bin/env python3
"""Prove FSDP2 DCP continuation across different GPU world sizes."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from distributed_launch_contract import (
    LaunchContractError,
    publish_create_only_json,
    require_sha256,
    sha256_file,
    source_tree_digest,
    verify_file_sha256,
    verify_launch_manifest,
)
from frontier_fsdp2_dcp_smoke import (
    _assert_difference,
    _build_model,
    _full_model_state,
    _global_mean_scalar,
    _initialize_distributed,
    _optimizer,
    _sha256_json,
    _state_difference,
    _synthetic_batch,
    _train_step,
)

from interaction_first.frontier_fsdp2 import (
    FSDP2_CLAIM_BOUNDARY,
    fully_shard_reranker,
    load_dcp_generation,
    promote_scalar_parameters_for_fsdp2,
    save_dcp_generation,
    validate_fsdp2_runtime,
)

SOURCE_PATHS = (
    "scripts/distributed_launch_contract.py",
    "scripts/frontier_fsdp2_dcp_smoke.py",
    "scripts/frontier_fsdp2_reshard.py",
    "scripts/generate_runtime_manifest.py",
    "slurm/frontier_fsdp2_reshard.sbatch",
    "slurm/frontier_fsdp2_multinode_gate.sbatch",
    "src/interaction_first/__init__.py",
    "src/interaction_first/asymmetric_retrieval.py",
    "src/interaction_first/contacts.py",
    "src/interaction_first/frontier_fsdp2.py",
    "src/interaction_first/frontier_model.py",
    "src/interaction_first/intra_chain_geometry.py",
    "src/interaction_first/partners.py",
    "src/interaction_first/runtime.py",
    "src/interaction_first/sabdab2_structure.py",
    "src/interaction_first/splits.py",
)
RESULT_FORMAT = "frontier_fsdp2_cross_world_reshard_result_v1"
GLOBAL_BATCH_SIZE = 8
GENERATION = "step_000003"
SEED = 20260730


def _emit_stage(*, rank: int, stage: str) -> None:
    """Emit a flush-safe rank-local stage marker for hang diagnosis."""
    print(
        json.dumps(
            {
                "event": "frontier_fsdp2_reshard_stage",
                "rank": rank,
                "stage": stage,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _source_digest(repository_root: Path) -> str:
    """Bind every executable source file used by the cross-world replay."""
    return source_tree_digest(repository_root, SOURCE_PATHS)


def _atomic_torch_save(path: Path, state: Mapping[str, Any]) -> str:
    """Publish a weights-only reference file without overwriting a path."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite reference state: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(state), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _load_reference(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify and load one tensor-only state reference."""
    verify_file_sha256(path, expected_sha256, field="reference_state_sha256")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not value:
        raise LaunchContractError("Reference state must be a non-empty dictionary")
    if any(
        not isinstance(name, str) or not torch.is_tensor(tensor)
        for name, tensor in value.items()
    ):
        raise LaunchContractError("Reference state contains an invalid entry")
    return value


def _strict_save_result(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load a source-world receipt through an externally supplied file hash."""
    verify_file_sha256(path, expected_sha256, field="save_result_sha256")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "format",
        "status",
        "mode",
        "claim_boundary",
        "source_sha256",
        "launch_manifest_sha256",
        "precision",
        "source_world_size",
        "global_batch_size",
        "steps_before_checkpoint",
        "generation",
        "checkpoint_root",
        "checkpoint_receipt_content_sha256",
        "provenance_hashes",
        "synthetic_manifest",
        "synthetic_manifest_sha256",
        "checkpoint_reference",
        "continuation_reference",
        "continuation_global_loss",
        "outer_or_test_access",
        "biological_training_authorized",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise LaunchContractError("Save result fields differ from the locked schema")
    if (
        payload["format"] != RESULT_FORMAT
        or payload["status"] != "passed"
        or payload["mode"] != "save"
        or payload["outer_or_test_access"] is not False
        or payload["biological_training_authorized"] is not False
    ):
        raise LaunchContractError("Save result identity or authorization is invalid")
    return payload


def _fixture(world_size: int, batch_size_per_rank: int) -> dict[str, Any]:
    """Return a world-size-independent fixture identity."""
    return {
        "format": "frontier_fsdp2_cross_world_fixture_v1",
        "seed": SEED,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "global_indices": list(range(GLOBAL_BATCH_SIZE)),
        "antibody_length": 7,
        "antigen_length": 9,
        "source_world_size_or_load_world_size": world_size,
        "batch_size_per_rank": batch_size_per_rank,
        "biological_data_access": "forbidden",
    }


def _stable_fixture(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Remove topology fields before computing the shared data identity."""
    return {
        key: value
        for key, value in fixture.items()
        if key not in {"source_world_size_or_load_world_size", "batch_size_per_rank"}
    }


def _validate_common(args: argparse.Namespace) -> tuple[Path, str]:
    """Validate source and fixed-global-batch requirements."""
    if not args.synthetic_only:
        raise ValueError("--synthetic-only is mandatory")
    if GLOBAL_BATCH_SIZE % args.expected_world_size != 0:
        raise ValueError("The locked global batch must divide evenly across ranks")
    if args.steps_before_checkpoint != 3:
        raise ValueError("Cross-world replay requires exactly three source steps")
    if args.absolute_tolerance <= 0.0 or args.relative_tolerance <= 0.0:
        raise ValueError("Parity tolerances must be positive")
    repository_root = Path(__file__).resolve().parents[1]
    observed_source = _source_digest(repository_root)
    expected_source = require_sha256(
        args.expected_source_sha256,
        field="expected_source_sha256",
    )
    if observed_source != expected_source:
        raise LaunchContractError(
            "Executed source boundary differs from the frozen digest: "
            f"expected={expected_source}, observed={observed_source}"
        )
    return repository_root, observed_source


def _build_sharded(
    *,
    precision: str,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Build the frozen fixture and apply the reviewed FSDP2 order."""
    model = _build_model(SEED, device)
    promoted = promote_scalar_parameters_for_fsdp2(model)
    if promoted != ("contact_head.bias",):
        raise RuntimeError(f"Unexpected scalar promotion set: {promoted}")
    fully_shard_reranker(model, precision=precision)
    return model, _optimizer(model)


def _local_batch(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Partition the same global examples under every tested topology."""
    batch_size = GLOBAL_BATCH_SIZE // world_size
    start = rank * batch_size
    return _synthetic_batch(
        range(start, start + batch_size),
        seed=SEED,
        device=device,
    )


def _save_mode(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    observed_source: str,
    output_root: Path,
    runtime_receipt: Mapping[str, Any],
) -> None:
    """Save a sharded checkpoint and sealed pre/post-continuation states."""
    batch_size = GLOBAL_BATCH_SIZE // world_size
    fixture = _fixture(world_size, batch_size)
    stable_fixture = _stable_fixture(fixture)
    provenance = {
        "launch_manifest": args.expected_launch_manifest_sha256,
        "source_tree": observed_source,
        "synthetic_fixture": _sha256_json(stable_fixture),
    }
    batch = _local_batch(rank=rank, world_size=world_size, device=device)
    model, optimizer = _build_sharded(precision=args.precision, device=device)
    for _ in range(args.steps_before_checkpoint):
        local_loss, _ = _train_step(model, optimizer, batch, precision=args.precision)
        _global_mean_scalar(local_loss, device)
    checkpoint_state = _full_model_state(model)
    if rank == 0:
        checkpoint_reference = output_root / "checkpoint_reference.pt"
        checkpoint_reference_sha256 = _atomic_torch_save(
            checkpoint_reference,
            checkpoint_state,
        )
    else:
        checkpoint_reference = output_root / "checkpoint_reference.pt"
        checkpoint_reference_sha256 = ""
    checkpoint_receipt = save_dcp_generation(
        output_root / "checkpoints",
        GENERATION,
        model=model,
        optimizer=optimizer,
        provenance_hashes=provenance,
        training_state={
            "global_step": args.steps_before_checkpoint,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "precision": args.precision,
            "synthetic_only": True,
        },
    )
    local_continuation_loss, _ = _train_step(
        model,
        optimizer,
        batch,
        precision=args.precision,
    )
    continuation_global_loss = _global_mean_scalar(local_continuation_loss, device)
    continuation_state = _full_model_state(model)
    if rank == 0:
        continuation_reference = output_root / "continuation_reference.pt"
        continuation_reference_sha256 = _atomic_torch_save(
            continuation_reference,
            continuation_state,
        )
        result = {
            "format": RESULT_FORMAT,
            "status": "passed",
            "mode": "save",
            "claim_boundary": FSDP2_CLAIM_BOUNDARY,
            "source_sha256": observed_source,
            "launch_manifest_sha256": args.expected_launch_manifest_sha256,
            "precision": args.precision,
            "source_world_size": world_size,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "steps_before_checkpoint": args.steps_before_checkpoint,
            "generation": GENERATION,
            "checkpoint_root": str((output_root / "checkpoints").resolve()),
            "checkpoint_receipt_content_sha256": checkpoint_receipt["content_sha256"],
            "provenance_hashes": provenance,
            "synthetic_manifest": fixture,
            "synthetic_manifest_sha256": provenance["synthetic_fixture"],
            "checkpoint_reference": {
                "path": str(checkpoint_reference.resolve()),
                "sha256": checkpoint_reference_sha256,
            },
            "continuation_reference": {
                "path": str(continuation_reference.resolve()),
                "sha256": continuation_reference_sha256,
            },
            "continuation_global_loss": continuation_global_loss,
            "outer_or_test_access": False,
            "biological_training_authorized": False,
        }
        publish_create_only_json(output_root / "result.json", result)
        os.chmod(output_root / "result.json", 0o444)
        print(
            json.dumps(
                {"status": "passed", "mode": "save", "world_size": world_size},
                sort_keys=True,
            ),
            flush=True,
        )


def _load_mode(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    observed_source: str,
    output_root: Path,
    runtime_receipt: Mapping[str, Any],
) -> None:
    """Restore under a new topology and compare state before and after one step."""
    if args.source_save_result is None or args.expected_save_result_sha256 is None:
        raise ValueError("Load mode requires the source save result and its SHA-256")
    save_result = _strict_save_result(
        args.source_save_result.resolve(),
        require_sha256(
            args.expected_save_result_sha256, field="expected_save_result_sha256"
        ),
    )
    if save_result["source_sha256"] != observed_source:
        raise LaunchContractError("Save and load source digests differ")
    if save_result["precision"] != args.precision:
        raise LaunchContractError("Save and load precision differ")
    if save_result["source_world_size"] == world_size:
        raise LaunchContractError("Cross-world replay requires different world sizes")
    batch_size = GLOBAL_BATCH_SIZE // world_size
    fixture = _fixture(world_size, batch_size)
    if (
        _sha256_json(_stable_fixture(fixture))
        != save_result["synthetic_manifest_sha256"]
    ):
        raise LaunchContractError("Save and load synthetic fixture identities differ")
    _emit_stage(rank=rank, stage="before_build_sharded")
    model, optimizer = _build_sharded(precision=args.precision, device=device)
    _emit_stage(rank=rank, stage="after_build_sharded")
    _emit_stage(rank=rank, stage="before_load_dcp_generation")
    loaded_receipt = load_dcp_generation(
        save_result["checkpoint_root"],
        save_result["generation"],
        model=model,
        optimizer=optimizer,
        expected_provenance_hashes=save_result["provenance_hashes"],
    )
    _emit_stage(rank=rank, stage="after_load_dcp_generation")
    if (
        loaded_receipt["content_sha256"]
        != save_result["checkpoint_receipt_content_sha256"]
    ):
        raise LaunchContractError("Loaded checkpoint receipt differs from save result")
    _emit_stage(rank=rank, stage="before_checkpoint_full_model_state")
    loaded_state = _full_model_state(model)
    _emit_stage(rank=rank, stage="after_checkpoint_full_model_state")
    if rank == 0:
        checkpoint_reference = _load_reference(
            Path(save_result["checkpoint_reference"]["path"]),
            save_result["checkpoint_reference"]["sha256"],
        )
        checkpoint_difference = _state_difference(
            loaded_state,
            checkpoint_reference,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        )
        _assert_difference(
            checkpoint_difference,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            label="Cross-world checkpoint load",
        )
    else:
        checkpoint_difference = {}
    batch = _local_batch(rank=rank, world_size=world_size, device=device)
    _emit_stage(rank=rank, stage="before_continuation_train_step")
    local_loss, _ = _train_step(model, optimizer, batch, precision=args.precision)
    _emit_stage(rank=rank, stage="after_continuation_train_step")
    _emit_stage(rank=rank, stage="before_continuation_global_loss")
    continuation_global_loss = _global_mean_scalar(local_loss, device)
    _emit_stage(rank=rank, stage="after_continuation_global_loss")
    _emit_stage(rank=rank, stage="before_continuation_full_model_state")
    continued_state = _full_model_state(model)
    _emit_stage(rank=rank, stage="after_continuation_full_model_state")
    if rank == 0:
        continuation_reference = _load_reference(
            Path(save_result["continuation_reference"]["path"]),
            save_result["continuation_reference"]["sha256"],
        )
        continuation_difference = _state_difference(
            continued_state,
            continuation_reference,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
        )
        _assert_difference(
            continuation_difference,
            absolute_tolerance=args.absolute_tolerance,
            relative_tolerance=args.relative_tolerance,
            label="Cross-world optimizer continuation",
        )
        expected_loss = float(save_result["continuation_global_loss"])
        allowed_loss = args.absolute_tolerance + args.relative_tolerance * abs(
            expected_loss
        )
        loss_absolute_difference = abs(continuation_global_loss - expected_loss)
        if (
            not math.isfinite(loss_absolute_difference)
            or loss_absolute_difference > allowed_loss
        ):
            raise RuntimeError("Cross-world continuation loss exceeds tolerance")
        result = {
            "format": RESULT_FORMAT,
            "status": "passed",
            "mode": "load",
            "claim_boundary": FSDP2_CLAIM_BOUNDARY,
            "source_sha256": observed_source,
            "launch_manifest_sha256": args.expected_launch_manifest_sha256,
            "source_save_result": str(args.source_save_result.resolve()),
            "source_save_result_sha256": args.expected_save_result_sha256,
            "precision": args.precision,
            "source_world_size": save_result["source_world_size"],
            "load_world_size": world_size,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "checkpoint_receipt_content_sha256": loaded_receipt["content_sha256"],
            "checkpoint_state_difference": checkpoint_difference,
            "continuation_state_difference": continuation_difference,
            "expected_continuation_global_loss": expected_loss,
            "observed_continuation_global_loss": continuation_global_loss,
            "continuation_loss_absolute_difference": loss_absolute_difference,
            "absolute_tolerance": args.absolute_tolerance,
            "relative_tolerance": args.relative_tolerance,
            "runtime": dict(runtime_receipt),
            "outer_or_test_access": False,
            "biological_training_authorized": False,
        }
        publish_create_only_json(output_root / "result.json", result)
        os.chmod(output_root / "result.json", 0o444)
        print(
            json.dumps(
                {
                    "status": "passed",
                    "mode": "load",
                    "source_world_size": save_result["source_world_size"],
                    "load_world_size": world_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    """Execute one source-save or target-load half of a cross-world replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("save", "load"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps-before-checkpoint", type=int, default=3)
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--parallelism", choices=("fsdp2",), required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--expected-launch-manifest-sha256", required=True)
    parser.add_argument("--source-save-result", type=Path)
    parser.add_argument("--expected-save-result-sha256")
    parser.add_argument("--absolute-tolerance", type=float, default=2e-5)
    parser.add_argument("--relative-tolerance", type=float, default=2e-4)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    _, observed_source = _validate_common(args)
    rank, _, device = _initialize_distributed(args.expected_world_size)
    world_size = dist.get_world_size()
    try:
        verify_launch_manifest(
            args.launch_manifest,
            expected_file_sha256=args.expected_launch_manifest_sha256,
            expected_source_sha256=observed_source,
            expected_world_size=world_size,
            expected_backend="nccl",
            expected_parallelism="fsdp2",
            expected_precision=args.precision,
        )
        runtime_receipt = validate_fsdp2_runtime(
            parallelism=args.parallelism,
            precision=args.precision,
            cuda_available=True,
            bf16_supported=bool(torch.cuda.is_bf16_supported()),
            distributed_initialized=dist.is_initialized(),
            world_size=world_size,
            torch_version=torch.__version__,
        )
        output_root = args.output_root.resolve()
        if rank == 0:
            if output_root.exists() or output_root.is_symlink():
                raise FileExistsError(
                    f"Refusing to overwrite output root: {output_root}"
                )
            output_root.mkdir(parents=True)
        dist.barrier()
        if args.mode == "save":
            _save_mode(
                args,
                rank=rank,
                world_size=world_size,
                device=device,
                observed_source=observed_source,
                output_root=output_root,
                runtime_receipt=runtime_receipt,
            )
        else:
            _load_mode(
                args,
                rank=rank,
                world_size=world_size,
                device=device,
                observed_source=observed_source,
                output_root=output_root,
                runtime_receipt=runtime_receipt,
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
