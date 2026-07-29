#!/usr/bin/env python3
"""Run a train-only real-data D0 engineering gate under torchrun."""

from __future__ import annotations

import argparse
import contextlib
from datetime import timedelta
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
from typing import Any, Mapping
import uuid

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.frontier_checkpoint import sha256_file  # noqa: E402
from interaction_first.frontier_data import FoldLocalFrontierDataset  # noqa: E402
from interaction_first.frontier_distributed import (  # noqa: E402
    LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND,
    SUPPORTED_GATHER_GRADIENT_BACKENDS,
)
from interaction_first.frontier_model import ProtocolD0Retriever  # noqa: E402
from interaction_first.frontier_training import (  # noqa: E402
    ExactDistributedBatchSampler,
    assert_parameter_state_agreement,
    create_training_dataloader,
    d0_forward_loss,
    make_training_cursor,
    restore_training_checkpoint,
    save_training_checkpoint,
)


MODEL_CONTRACT_RELATIVE_PATH = Path("protocols/frontier_model_contract_v1.json")
MODEL_CONTRACT_PATH = REPOSITORY_ROOT / MODEL_CONTRACT_RELATIVE_PATH
ENGINEERING_DIAGNOSTIC_PROFILE = "engineering_diagnostic"
CONTRACT_D0_PROFILE = "contract_d0"
SUPPORTED_RUN_PROFILES = frozenset(
    {
        ENGINEERING_DIAGNOSTIC_PROFILE,
        CONTRACT_D0_PROFILE,
    }
)
LINEAR_WARMUP_COSINE_SCHEDULER = "linear_warmup_cosine"
COSINE_WARM_RESTARTS_SCHEDULER = "cosine_annealing_warm_restarts"
SUPPORTED_SCHEDULERS = frozenset(
    {
        LINEAR_WARMUP_COSINE_SCHEDULER,
        COSINE_WARM_RESTARTS_SCHEDULER,
    }
)
CONTRACT_D0_MODEL_ID = "D0_dual_encoder_locked"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite JSON deterministically."""
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    """Return a deterministic SHA-256 for JSON metadata."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _required_sha256(value: Any, name: str) -> str:
    """Return one externally supplied lowercase SHA-256 digest."""
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _validate_external_anchors(args: argparse.Namespace) -> None:
    """Require external data anchors and an exact receipt anchor on resume."""
    _required_sha256(
        args.expected_authorization_manifest_sha256,
        "expected_authorization_manifest_sha256",
    )
    _required_sha256(
        args.expected_cache_manifest_sha256,
        "expected_cache_manifest_sha256",
    )
    resuming = args.resume_generation is not None
    receipt_anchor = args.expected_resume_receipt_sha256
    if resuming:
        if receipt_anchor is None:
            raise ValueError(
                "Resume requires expected_resume_receipt_sha256"
            )
        _required_sha256(
            receipt_anchor,
            "expected_resume_receipt_sha256",
        )
    elif receipt_anchor is not None:
        raise ValueError(
            "expected_resume_receipt_sha256 is only valid with a resume generation"
        )


def _load_model_contract(
    path: Path = MODEL_CONTRACT_PATH,
) -> tuple[dict[str, Any], str]:
    """Load and minimally validate the prospective model contract."""
    path = path.expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError("The model contract must be a regular non-symlink file")
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Unable to load the model contract") from error
    if not isinstance(contract, dict):
        raise ValueError("The model contract must be a JSON object")
    if contract.get("format_version") != "frontier_model_contract_v1":
        raise ValueError("Unsupported frontier model contract format")
    models = contract.get("models")
    if not isinstance(models, dict):
        raise ValueError("The model contract lacks the models mapping")
    d0_contract = models.get(CONTRACT_D0_MODEL_ID)
    if not isinstance(d0_contract, dict):
        raise ValueError("The model contract lacks the locked D0 model")
    scheduler = d0_contract.get("scheduler")
    required_scheduler = {
        "eta_min": 0.0,
        "name": "CosineAnnealingWarmRestarts",
        "period_epochs": 10,
        "period_multiplier": 2,
        "period_steps_derivation": "period_epochs * steps_per_epoch",
        "step_unit": "optimizer_step",
    }
    if scheduler != required_scheduler:
        raise ValueError("The locked D0 scheduler contract is unsupported or malformed")
    return contract, sha256_file(path)


def _atomic_json(path: Path, value: Any) -> None:
    """Publish one JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_rank_event(path: Path, event: Mapping[str, Any]) -> None:
    """Append and flush one finite rank-local JSON event."""
    payload = _canonical_json_bytes(dict(event))
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _distributed_context(device_argument: str) -> tuple[int, int, int, torch.device]:
    """Initialize torch.distributed from the torchrun environment."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("Invalid torchrun rank environment")
    if device_argument == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = device_argument
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError("LOCAL_RANK exceeds the visible CUDA device count")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        init_kwargs: dict[str, Any] = {
            "backend": backend,
            "timeout": timedelta(minutes=10),
        }
        if (
            device.type == "cuda"
            and "device_id"
            in inspect.signature(torch.distributed.init_process_group).parameters
        ):
            init_kwargs["device_id"] = device
        torch.distributed.init_process_group(**init_kwargs)
    return rank, world_size, local_rank, device


def _seed_process(seed: int) -> None:
    """Seed every process identically before model construction."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def _source_manifest() -> tuple[dict[str, str], str]:
    """Hash the exact source boundary used by this gate."""
    static_paths = (
        "pyproject.toml",
        "protocols/frontier_model_contract_v1.json",
        "scripts/frontier_real_data_gate.py",
        "slurm/frontier_real_data_gate.sbatch",
    )
    package_paths = tuple(
        sorted(
            str(path.relative_to(REPOSITORY_ROOT))
            for path in (REPOSITORY_ROOT / "src" / "interaction_first").glob("*.py")
        )
    )
    relative_paths = static_paths + package_paths
    hashes = {
        relative: sha256_file(REPOSITORY_ROOT / relative)
        for relative in relative_paths
    }
    return hashes, _sha256_json(hashes)


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when DDP is active."""
    return module.module if hasattr(module, "module") else module


def _wrap_ddp(
    model: torch.nn.Module,
    *,
    device: torch.device,
    local_rank: int,
    world_size: int,
    init_sync: bool,
    static_graph: bool,
    gradient_as_bucket_view: bool,
) -> torch.nn.Module:
    """Wrap a verified identical model with efficient fixed-graph DDP settings."""
    if world_size == 1:
        return model
    kwargs: dict[str, Any] = {
        "broadcast_buffers": False,
        "gradient_as_bucket_view": gradient_as_bucket_view,
        "static_graph": static_graph,
    }
    signature = inspect.signature(torch.nn.parallel.DistributedDataParallel)
    if "init_sync" in signature.parameters:
        kwargs["init_sync"] = init_sync
    elif not init_sync:
        raise RuntimeError("This PyTorch release cannot disable DDP initialization sync")
    if device.type == "cuda":
        kwargs["device_ids"] = [local_rank]
        kwargs["output_device"] = local_rank
    return torch.nn.parallel.DistributedDataParallel(model, **kwargs)


def _autocast_context(device: torch.device, precision: str) -> Any:
    """Return the selected forward precision context."""
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision != "bf16":
        raise ValueError("Unsupported precision")
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=True,
    )


def _learning_rate_lambda(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
) -> float:
    """Apply linear warmup followed by cosine decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-12, (step + 1) / warmup_steps)
    remaining = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _scheduler_specification(
    args: argparse.Namespace,
    *,
    steps_per_epoch: int,
) -> dict[str, Any]:
    """Return the complete optimizer-step scheduler identity."""
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    total_steps = args.epochs * steps_per_epoch
    if args.scheduler == LINEAR_WARMUP_COSINE_SCHEDULER:
        warmup_steps = int(round(args.warmup_fraction * total_steps))
        return {
            "id": LINEAR_WARMUP_COSINE_SCHEDULER,
            "implementation": "torch.optim.lr_scheduler.LambdaLR",
            "step_unit": "optimizer_step",
            "total_steps": total_steps,
            "warmup_fraction": args.warmup_fraction,
            "warmup_steps": warmup_steps,
            "cosine_decay_to_zero": True,
        }
    if args.scheduler == COSINE_WARM_RESTARTS_SCHEDULER:
        period_steps = args.restart_period_epochs * steps_per_epoch
        return {
            "id": COSINE_WARM_RESTARTS_SCHEDULER,
            "implementation": (
                "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
            ),
            "step_unit": "optimizer_step",
            "period_epochs": args.restart_period_epochs,
            "period_steps": period_steps,
            "period_multiplier": args.restart_period_multiplier,
            "eta_min": 0.0,
            "warmup_fraction": 0.0,
            "warmup_steps": 0,
        }
    raise ValueError("Unsupported scheduler")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    specification: Mapping[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler:
    """Construct the scheduler described by a validated specification."""
    scheduler_id = specification.get("id")
    if scheduler_id == LINEAR_WARMUP_COSINE_SCHEDULER:
        total_steps = int(specification["total_steps"])
        warmup_steps = int(specification["warmup_steps"])
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _learning_rate_lambda(
                step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            ),
        )
    if scheduler_id == COSINE_WARM_RESTARTS_SCHEDULER:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(specification["period_steps"]),
            T_mult=int(specification["period_multiplier"]),
            eta_min=float(specification["eta_min"]),
        )
    raise ValueError("Unsupported scheduler specification")


def _contract_d0_reference(
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    """Return the immutable D0 contract reference stored in every run."""
    d0_contract = contract["models"][CONTRACT_D0_MODEL_ID]
    return {
        "model_id": CONTRACT_D0_MODEL_ID,
        "contract_format_version": contract["format_version"],
        "contract_path": MODEL_CONTRACT_RELATIVE_PATH.as_posix(),
        "contract_file_sha256": contract_sha256,
        "scheduler": dict(d0_contract["scheduler"]),
    }


def _validate_contract_d0_profile(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    *,
    world_size: int | None = None,
) -> None:
    """Reject any runtime that could mislabel a non-contract run as D0."""
    if args.run_profile != CONTRACT_D0_PROFILE:
        return
    d0_contract = contract["models"][CONTRACT_D0_MODEL_ID]
    scheduler_contract = d0_contract["scheduler"]
    expected_values = {
        "scheduler": COSINE_WARM_RESTARTS_SCHEDULER,
        "epochs": d0_contract["epochs"],
        "embedding_dim": d0_contract["embedding_dim"],
        "projection_hidden_dim": d0_contract["projection_hidden_dim"],
        "learning_rate": d0_contract["optimizer"]["learning_rate"],
        "weight_decay": d0_contract["optimizer"]["weight_decay"],
        "restart_period_epochs": scheduler_contract["period_epochs"],
        "restart_period_multiplier": scheduler_contract["period_multiplier"],
        "precision": "bf16",
        "gather_gradient_backend": LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND,
        "ddp_init_sync": True,
        "ddp_static_graph": True,
        "ddp_gradient_as_bucket_view": True,
        "warmup_fraction": 0.0,
    }
    mismatches = {
        name: {
            "expected": expected,
            "observed": getattr(args, name),
        }
        for name, expected in expected_values.items()
        if getattr(args, name) != expected
    }
    if args.max_steps is not None:
        mismatches["max_steps"] = {
            "expected": None,
            "observed": args.max_steps,
        }
    if args.verify_only:
        mismatches["verify_only"] = {
            "expected": False,
            "observed": True,
        }
    if getattr(args, "contract_preflight_only", False) and args.run_profile != (
        CONTRACT_D0_PROFILE
    ):
        mismatches["contract_preflight_only"] = {
            "expected": False,
            "observed": True,
        }
    if world_size is not None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        observed_global_batch = args.local_batch_size * world_size
        expected_global_batch = d0_contract["global_batch_exact_pairs"]
        if observed_global_batch != expected_global_batch:
            mismatches["global_batch_exact_pairs"] = {
                "expected": expected_global_batch,
                "observed": observed_global_batch,
            }
    if mismatches:
        raise ValueError(
            "contract_d0 profile differs from the locked D0 contract: "
            + json.dumps(
                mismatches,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )


def _runtime_version_string(value: Any) -> str:
    """Normalize one runtime-library version without guessing its semantics."""
    if isinstance(value, (tuple, list)) and value:
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ValueError("Runtime version components must be integers")
        return ".".join(str(item) for item in value)
    if isinstance(value, str) and value:
        return value
    raise ValueError("Runtime version is unavailable or malformed")


def _validate_contract_d0_runtime(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Bind a contract run to the reviewed isolated PyTorch runtime."""
    if args.run_profile != CONTRACT_D0_PROFILE:
        return {
            "validated": False,
            "reason": "engineering_diagnostic profile does not claim the D0 contract",
        }
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("contract_d0 requires an available CUDA runtime")
    runtime_lock = contract["distributed_execution"]["runtime_lock"]
    manifest_path = Path(runtime_lock["manifest_path"]).expanduser()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("The locked runtime manifest must be a regular non-symlink file")
    manifest_sha256 = sha256_file(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Unable to load the locked runtime manifest") from error
    runtime_state = manifest.get("runtime_state")
    integrity = manifest.get("integrity")
    if not isinstance(runtime_state, dict) or not isinstance(integrity, dict):
        raise ValueError("The locked runtime manifest has an unsupported schema")
    observed = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_build_version": str(torch.version.cuda),
        "cudnn_version_integer": int(torch.backends.cudnn.version()),
        "nccl_version": _runtime_version_string(torch.cuda.nccl.version()),
        "manifest_file_sha256": manifest_sha256,
        "manifest_runtime_state_sha256": integrity.get("runtime_state_sha256"),
        "scope_classification": runtime_state.get("environment", {}).get(
            "scope_classification"
        ),
        "pip_check_passed": runtime_state.get("package_health", {})
        .get("pip_check", {})
        .get("ok"),
    }
    expected = {
        name: runtime_lock[name]
        for name in (
            "python_version",
            "torch_version",
            "cuda_build_version",
            "cudnn_version_integer",
            "nccl_version",
            "manifest_file_sha256",
            "manifest_runtime_state_sha256",
            "scope_classification",
            "pip_check_passed",
        )
    }
    mismatches = {
        name: {"expected": expected[name], "observed": observed[name]}
        for name in expected
        if observed[name] != expected[name]
    }
    if mismatches:
        raise ValueError(
            "contract_d0 runtime differs from the locked runtime: "
            + json.dumps(
                mismatches,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return {
        "validated": True,
        **observed,
    }


def _instance_membership_sha256(dataset: FoldLocalFrontierDataset) -> str:
    """Hash the exact sorted instance membership used by the cache builder."""
    payload = "".join(
        f"{record.instance_id}\n"
        for record in sorted(
            dataset.contract.records,
            key=lambda item: item.instance_id,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_contract_d0_dataset(
    args: argparse.Namespace,
    contract: Mapping[str, Any],
    dataset: FoldLocalFrontierDataset,
) -> dict[str, Any]:
    """Bind a contract run to the complete reviewed authority-train cache."""
    if args.run_profile != CONTRACT_D0_PROFILE:
        return {
            "validated": False,
            "reason": "engineering_diagnostic profile does not claim the D0 contract",
        }
    cache_lock = contract["data"]["engineering_cache"]
    observed = {
        "authorization_manifest_sha256": (
            dataset.contract.authorization_manifest_sha256
        ),
        "cache_manifest_sha256": dataset.contract.manifest_sha256,
        "exact_pair_count": len(dataset.contract.positive_pairs),
        "selected_instance_count": len(dataset.contract.records),
        "selected_antibody_count": len(
            {record.antibody_id for record in dataset.contract.records}
        ),
        "selected_antigen_count": len(
            {record.antigen_id for record in dataset.contract.records}
        ),
        "selected_instance_membership_sha256": _instance_membership_sha256(
            dataset
        ),
        "excluded_pair_count": len(dataset.contract.excluded_pairs),
    }
    expected = {
        "authorization_manifest_sha256": cache_lock[
            "authorization_manifest_sha256"
        ],
        "cache_manifest_sha256": cache_lock["cache_manifest_sha256"],
        "exact_pair_count": cache_lock["exact_pair_count"],
        "selected_instance_count": cache_lock["selected_instance_count"],
        "selected_antibody_count": cache_lock["selected_antibody_count"],
        "selected_antigen_count": cache_lock["selected_antigen_count"],
        "selected_instance_membership_sha256": cache_lock[
            "selected_instance_membership_sha256"
        ],
        "excluded_pair_count": cache_lock["excluded_pair_count"],
    }
    mismatches = {
        name: {"expected": expected[name], "observed": observed[name]}
        for name in expected
        if observed[name] != expected[name]
    }
    if mismatches:
        raise ValueError(
            "contract_d0 data differs from the locked engineering cache: "
            + json.dumps(
                mismatches,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return {
        "validated": True,
        **observed,
    }


def _generation_name(epoch: int, next_step: int, global_step: int) -> str:
    """Return a sortable immutable checkpoint generation name."""
    return (
        f"epoch_{epoch:04d}_next_{next_step:06d}_"
        f"global_{global_step:09d}"
    )


def _validate_output_boundary(output_root: Path, cache_root: Path) -> Path:
    """Keep run artifacts outside the immutable fold-local cache."""
    output_root = output_root.expanduser().resolve()
    try:
        output_root.relative_to(cache_root)
    except ValueError:
        pass
    else:
        raise ValueError("output_root cannot be inside the fold-local cache")
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _create_run_root_collectively(
    run_root: Path,
    *,
    rank: int,
    world_size: int,
) -> None:
    """Create one run directory without stranding nonzero ranks at a barrier."""
    outcome: list[str | None] = [None]
    if rank == 0:
        try:
            run_root.mkdir(parents=False, exist_ok=False)
        except BaseException as error:
            outcome[0] = f"{type(error).__name__}: {error}"
    if world_size > 1:
        torch.distributed.broadcast_object_list(outcome, src=0)
    if outcome[0] is not None:
        raise FileExistsError(
            f"Unable to create the immutable run root {run_root}: {outcome[0]}"
        )


def parse_args() -> argparse.Namespace:
    """Parse the strict engineering-gate command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--authorization-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-authorization-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-cache-manifest-sha256",
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--run-profile",
        choices=tuple(sorted(SUPPORTED_RUN_PROFILES)),
        default=ENGINEERING_DIAGNOSTIC_PROFILE,
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--local-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--memory-cache-size", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--projection-hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument(
        "--scheduler",
        choices=tuple(sorted(SUPPORTED_SCHEDULERS)),
        default=LINEAR_WARMUP_COSINE_SCHEDULER,
    )
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--restart-period-epochs", type=int, default=10)
    parser.add_argument("--restart-period-multiplier", type=int, default=2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-generation")
    parser.add_argument("--resume-checkpoint-root", type=Path)
    parser.add_argument("--expected-resume-receipt-sha256")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument(
        "--gather-gradient-backend",
        choices=tuple(sorted(SUPPORTED_GATHER_GRADIENT_BACKENDS)),
        default=LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--ddp-init-sync",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ddp-static-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ddp-gradient-as-bucket-view",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--diagnostic-sync-boundaries", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--contract-preflight-only", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Reject ambiguous or scientifically unsafe runtime settings."""
    for name in (
        "epochs",
        "local_batch_size",
        "embedding_dim",
        "projection_hidden_dim",
        "restart_period_epochs",
        "restart_period_multiplier",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("num_workers", "memory_cache_size", "checkpoint_every_steps"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Optimizer settings are invalid")
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if args.gradient_clip <= 0.0:
        raise ValueError("gradient_clip must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive when supplied")
    if (args.resume_generation is None) != (args.resume_checkpoint_root is None):
        raise ValueError(
            "resume_generation and resume_checkpoint_root must be supplied together"
        )
    _validate_external_anchors(args)
    if args.verify_only and args.resume_generation is not None:
        raise ValueError("verify_only cannot restore a training checkpoint")
    if args.contract_preflight_only and args.run_profile != CONTRACT_D0_PROFILE:
        raise ValueError("contract_preflight_only requires run_profile=contract_d0")
    if args.contract_preflight_only and (
        args.verify_only or args.resume_generation is not None
    ):
        raise ValueError(
            "contract_preflight_only cannot combine with verification or resume"
        )
    if not args.run_id or args.run_id != args.run_id.strip():
        raise ValueError("run_id must be a nonempty trimmed string")
    if any(character in args.run_id for character in ("/", "\\", "\0")):
        raise ValueError("run_id cannot contain path separators")


def main() -> None:
    """Verify the cache and execute a resumable train-only engineering run."""
    args = parse_args()
    _validate_args(args)
    model_contract, model_contract_sha256 = _load_model_contract()
    _validate_contract_d0_profile(args, model_contract)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rank, world_size, local_rank, device = _distributed_context(args.device)
    try:
        _validate_contract_d0_profile(
            args,
            model_contract,
            world_size=world_size,
        )
        runtime_contract_validation = _validate_contract_d0_runtime(
            args,
            model_contract,
            device=device,
        )
        _seed_process(args.seed)
        dataset = FoldLocalFrontierDataset(
            args.cache_root,
            args.authorization_manifest,
            verify_artifact_hashes=rank == 0,
            memory_cache_size=args.memory_cache_size,
            expected_authorization_manifest_sha256=(
                args.expected_authorization_manifest_sha256
            ),
            expected_cache_manifest_sha256=args.expected_cache_manifest_sha256,
        )
        if (
            not dataset.contract.authorization_external_anchor_verified
            or not dataset.contract.cache_external_anchor_verified
        ):
            raise RuntimeError(
                "Real-data execution requires both external data anchors"
            )
        data_contract_validation = _validate_contract_d0_dataset(
            args,
            model_contract,
            dataset,
        )
        output_root = _validate_output_boundary(
            args.output_root,
            dataset.contract.cache_root,
        )
        run_root = output_root / args.run_id
        _create_run_root_collectively(
            run_root,
            rank=rank,
            world_size=world_size,
        )
        if world_size > 1:
            torch.distributed.barrier()
        rank_log = run_root / f"rank_{rank:04d}.jsonl"
        source_hashes, source_manifest_sha256 = _source_manifest()
        verification_payload: list[Any] = [None]
        if rank == 0:
            inventory_summary = dataset.verify_all_examples()
            verification_payload[0] = {
                "inventory": inventory_summary,
                "collective_receipt": dataset.collective_verification_receipt(),
            }
        if world_size > 1:
            torch.distributed.broadcast_object_list(verification_payload, src=0)
        verification = verification_payload[0]
        if (
            not isinstance(verification, dict)
            or not isinstance(verification.get("inventory"), dict)
            or not isinstance(verification.get("collective_receipt"), dict)
        ):
            raise RuntimeError("Rank-zero data verification returned an invalid summary")
        inventory_summary = verification["inventory"]
        collectively_verified_artifacts = (
            dataset.accept_collective_verification_receipt(
                verification["collective_receipt"]
            )
        )
        cleared_memory_cache_entries = dataset.clear_memory_cache()
        _append_rank_event(
            rank_log,
            {
                "event": "data_contract_verified",
                "rank": rank,
                "world_size": world_size,
                "device": str(device),
                "manifest_sha256": dataset.contract.manifest_sha256,
                "authorization_manifest_sha256": (
                    dataset.contract.authorization_manifest_sha256
                ),
                "collective_verification": verification["collective_receipt"],
                "collectively_verified_artifacts": collectively_verified_artifacts,
                "cleared_memory_cache_entries": cleared_memory_cache_entries,
                **inventory_summary,
            },
        )
        if args.verify_only:
            if rank == 0:
                _atomic_json(
                    run_root / "VERIFY_ONLY_RECEIPT.json",
                    {
                        "format_version": "frontier_real_data_verify_receipt_v1",
                        "run_id": args.run_id,
                        "run_profile": args.run_profile,
                        "claim_boundary": (
                            "Train-only data engineering verification; no model "
                            "selection, validation, test access, or performance claim."
                        ),
                        "data": inventory_summary,
                        "manifest_sha256": dataset.contract.manifest_sha256,
                        "authorization_manifest_sha256": (
                            dataset.contract.authorization_manifest_sha256
                        ),
                        "source_manifest_sha256": source_manifest_sha256,
                        "model_contract_sha256": model_contract_sha256,
                        "source_hashes": source_hashes,
                    },
                )
            if world_size > 1:
                torch.distributed.barrier()
            return
        if args.contract_preflight_only:
            preflight_sampler = ExactDistributedBatchSampler(
                len(dataset),
                local_batch_size=args.local_batch_size,
                world_size=world_size,
                rank=rank,
                seed=args.seed,
                shuffle=True,
            )
            scheduler_specification = _scheduler_specification(
                args,
                steps_per_epoch=len(preflight_sampler),
            )
            if rank == 0:
                _atomic_json(
                    run_root / "CONTRACT_PREFLIGHT_RECEIPT.json",
                    {
                        "format_version": "frontier_contract_d0_preflight_v1",
                        "run_id": args.run_id,
                        "run_profile": args.run_profile,
                        "claim_boundary": (
                            "Train-only contract, runtime, cache, and scheduler "
                            "preflight. No optimizer step, validation, model "
                            "selection, held-out access, or performance claim."
                        ),
                        "world_size": world_size,
                        "local_batch_size": args.local_batch_size,
                        "global_batch_size": args.local_batch_size * world_size,
                        "steps_per_epoch": len(preflight_sampler),
                        "scheduler": scheduler_specification,
                        "runtime_contract_validation": runtime_contract_validation,
                        "data_contract_validation": data_contract_validation,
                        "data": inventory_summary,
                        "model_contract_sha256": model_contract_sha256,
                        "source_manifest_sha256": source_manifest_sha256,
                        "source_hashes": source_hashes,
                        "optimizer_steps_performed": 0,
                    },
                )
            if world_size > 1:
                torch.distributed.barrier()
            return

        sampler = ExactDistributedBatchSampler(
            len(dataset),
            local_batch_size=args.local_batch_size,
            world_size=world_size,
            rank=rank,
            seed=args.seed,
            shuffle=True,
        )
        loader = create_training_dataloader(
            dataset,
            sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=args.prefetch_factor,
            materialize_contact_tensors=False,
        )
        total_steps = args.epochs * len(sampler)
        scheduler_specification = _scheduler_specification(
            args,
            steps_per_epoch=len(sampler),
        )
        run_profile = {
            "name": args.run_profile,
            "contract_d0_settings_validated": (
                args.run_profile == CONTRACT_D0_PROFILE
            ),
            "claim_boundary": (
                "Train-only optimization and systems evidence; this profile does "
                "not authorize validation, model selection, test access, or a "
                "biological performance claim."
            ),
        }
        base_config = {
            "run_id": args.run_id,
            "run_profile": run_profile,
            "model_contract_reference": _contract_d0_reference(
                model_contract,
                model_contract_sha256,
            ),
            "runtime_contract_validation": runtime_contract_validation,
            "data_contract_validation": data_contract_validation,
            "fold_id": dataset.contract.fold_id,
            "epochs": args.epochs,
            "world_size": world_size,
            "local_batch_size": args.local_batch_size,
            "global_batch_size": args.local_batch_size * world_size,
            "steps_per_epoch": len(sampler),
            "seed": args.seed,
            "precision": args.precision,
            "antibody_input_dim": dataset.contract.antibody_input_dim,
            "antigen_input_dim": dataset.contract.antigen_input_dim,
            "embedding_dim": args.embedding_dim,
            "projection_hidden_dim": args.projection_hidden_dim,
            "dropout": 0.0,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "betas": [0.9, 0.98],
                "epsilon": 1e-8,
            },
            "scheduler": scheduler_specification,
            "warmup_steps": scheduler_specification["warmup_steps"],
            "total_scheduler_steps": total_steps,
            "gradient_clip": args.gradient_clip,
            "gather_gradient_backend": args.gather_gradient_backend,
            "ddp_init_sync": args.ddp_init_sync,
            "ddp_static_graph": args.ddp_static_graph,
            "ddp_gradient_as_bucket_view": args.ddp_gradient_as_bucket_view,
            "data_access_scope": "train_only_fold_local",
            "batch_payload": "d0_retrieval_only_without_pair_tensors",
            "input_pipeline": {
                "num_workers": args.num_workers,
                "memory_cache_size_per_worker": args.memory_cache_size,
                "prefetch_factor": args.prefetch_factor,
                "persistent_workers": args.num_workers > 0,
                "pin_memory": device.type == "cuda",
                "collective_rank_zero_artifact_verification": True,
            },
            "external_data_anchors": {
                "authorization_manifest_sha256": (
                    args.expected_authorization_manifest_sha256
                ),
                "cache_manifest_sha256": args.expected_cache_manifest_sha256,
                "authorization_external_anchor_verified": (
                    dataset.contract.authorization_external_anchor_verified
                ),
                "cache_external_anchor_verified": (
                    dataset.contract.cache_external_anchor_verified
                ),
            },
        }
        provenance_hashes = {
            "authorization_manifest": (
                dataset.contract.authorization_manifest_sha256
            ),
            "data_manifest": dataset.contract.manifest_sha256,
            "model_contract": model_contract_sha256,
            "source_manifest": source_manifest_sha256,
        }
        model = ProtocolD0Retriever(
            antibody_input_dim=dataset.contract.antibody_input_dim,
            antigen_input_dim=dataset.contract.antigen_input_dim,
            embedding_dim=args.embedding_dim,
            projection_hidden_dim=args.projection_hidden_dim,
            dropout=0.0,
        ).to(device)
        initial_parameter_sha256 = assert_parameter_state_agreement(model)
        _append_rank_event(
            rank_log,
            {
                "event": "initial_parameter_state_verified",
                "parameter_sha256": initial_parameter_sha256,
                "rank": rank,
            },
        )
        distributed_model = _wrap_ddp(
            model,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
            init_sync=args.ddp_init_sync,
            static_graph=args.ddp_static_graph,
            gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
        )
        _append_rank_event(
            rank_log,
            {
                "event": "ddp_wrapper_constructed",
                "init_sync": args.ddp_init_sync,
                "static_graph": args.ddp_static_graph,
                "gradient_as_bucket_view": args.ddp_gradient_as_bucket_view,
                "rank": rank,
            },
        )
        optimizer = torch.optim.AdamW(
            distributed_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        scheduler = _build_scheduler(
            optimizer,
            scheduler_specification,
        )
        _append_rank_event(
            rank_log,
            {
                "event": "scheduler_constructed",
                "rank": rank,
                "run_profile": args.run_profile,
                "scheduler": scheduler_specification,
                "model_contract_sha256": model_contract_sha256,
            },
        )
        checkpoints_root = run_root / "checkpoints"
        if rank == 0:
            checkpoints_root.mkdir(parents=True, exist_ok=True)
        if world_size > 1:
            torch.distributed.barrier()

        if args.resume_generation:
            resume_checkpoint_root = (
                args.resume_checkpoint_root.expanduser().resolve()
            )
            if (
                resume_checkpoint_root.is_symlink()
                or not resume_checkpoint_root.is_dir()
            ):
                raise ValueError(
                    "resume_checkpoint_root must be a regular checkpoint directory"
                )
            cursor, resume_metadata = restore_training_checkpoint(
                resume_checkpoint_root,
                args.resume_generation,
                model=_unwrap(distributed_model),
                optimizer=optimizer,
                scheduler=scheduler,
                expected_base_config=base_config,
                dataset=dataset,
                sampler=sampler,
                provenance_hashes=provenance_hashes,
                map_location=device,
                rank=rank,
                expected_receipt_sha256=(
                    args.expected_resume_receipt_sha256
                ),
            )
            _append_rank_event(
                rank_log,
                {
                    "event": "checkpoint_restored",
                    "generation": resume_metadata["generation"],
                    "epoch": cursor.epoch,
                    "next_step": cursor.next_step,
                    "global_step": cursor.global_step,
                    "checkpoint_sha256": resume_metadata["checkpoint_sha256"],
                    "receipt_sha256": resume_metadata["receipt_sha256"],
                    "external_receipt_anchor_verified": resume_metadata[
                        "external_receipt_anchor_verified"
                    ],
                },
            )
        else:
            cursor = make_training_cursor(
                dataset,
                sampler,
                epoch=0,
                next_step=0,
                global_step=0,
            )

        last_loss: float | None = None
        last_checkpoint_generation: str | None = None
        stop_requested = False
        completed_valid_examples = 0
        completed_valid_residues = 0
        training_start = time.monotonic()
        for epoch in range(cursor.epoch, args.epochs):
            sampler.set_epoch(epoch)
            resume_step = cursor.next_step if epoch == cursor.epoch else 0
            for batch in loader:
                if batch.step < resume_step:
                    continue
                if args.max_steps is not None and cursor.global_step >= args.max_steps:
                    stop_requested = True
                    break
                batch = batch.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(device, args.precision):
                    details = d0_forward_loss(
                        distributed_model,
                        batch,
                        gather_gradient_backend=args.gather_gradient_backend,
                    )
                    loss = details["loss"]
                if args.diagnostic_sync_boundaries and device.type == "cuda":
                    torch.cuda.synchronize(device)
                    _append_rank_event(
                        rank_log,
                        {
                            "event": "forward_synchronized",
                            "epoch": epoch,
                            "step": batch.step,
                            "rank": rank,
                        },
                    )
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Training loss became non-finite")
                loss.backward()
                if args.diagnostic_sync_boundaries and device.type == "cuda":
                    torch.cuda.synchronize(device)
                    _append_rank_event(
                        rank_log,
                        {
                            "event": "backward_synchronized",
                            "epoch": epoch,
                            "step": batch.step,
                            "rank": rank,
                        },
                    )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    distributed_model.parameters(),
                    max_norm=args.gradient_clip,
                )
                if not bool(torch.isfinite(gradient_norm).item()):
                    raise FloatingPointError("Gradient norm became non-finite")
                optimizer.step()
                scheduler.step()
                local_valid_residues = (
                    (
                        batch.frontier.antibody_lengths
                        + batch.frontier.antigen_lengths
                    )
                    * batch.local_valid_mask.to(
                        dtype=batch.frontier.antibody_lengths.dtype
                    )
                ).sum()
                if world_size > 1:
                    torch.distributed.all_reduce(
                        local_valid_residues,
                        op=torch.distributed.ReduceOp.SUM,
                    )
                completed_valid_examples += int(details["global_antibody_count"])
                completed_valid_residues += int(local_valid_residues.item())
                next_cursor = make_training_cursor(
                    dataset,
                    sampler,
                    epoch=epoch,
                    next_step=batch.step + 1,
                    global_step=cursor.global_step + 1,
                )
                last_loss = float(loss.detach().cpu())
                _append_rank_event(
                    rank_log,
                    {
                        "event": "train_step",
                        "epoch": epoch,
                        "step": batch.step,
                        "global_step": next_cursor.global_step,
                        "valid_global_examples": details["global_antibody_count"],
                        "padded_global_examples": details["global_padded_count"],
                        "loss": last_loss,
                        "antibody_to_antigen_loss": float(
                            details["antibody_to_antigen"].detach().cpu()
                        ),
                        "antigen_to_antibody_loss": float(
                            details["antigen_to_antibody"].detach().cpu()
                        ),
                        "temperature": float(
                            details["temperature"].detach().float().cpu()
                        ),
                        "gradient_norm_before_clip": float(
                            gradient_norm.detach().cpu()
                        ),
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "gather_gradient_backend": details[
                            "gather_gradient_backend"
                        ],
                        "similarity_dtype": details["similarity_dtype"],
                    },
                )
                cursor = next_cursor
                if (
                    args.checkpoint_every_steps > 0
                    and cursor.global_step % args.checkpoint_every_steps == 0
                ):
                    generation = _generation_name(
                        cursor.epoch,
                        cursor.next_step,
                        cursor.global_step,
                    )
                    save_training_checkpoint(
                        checkpoints_root,
                        generation,
                        model=_unwrap(distributed_model),
                        optimizer=optimizer,
                        scheduler=scheduler,
                        base_config=base_config,
                        cursor=cursor,
                        provenance_hashes=provenance_hashes,
                    )
                    last_checkpoint_generation = generation
            if stop_requested:
                break
            cursor = make_training_cursor(
                dataset,
                sampler,
                epoch=epoch,
                next_step=len(sampler),
                global_step=cursor.global_step,
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_loop_elapsed_seconds = time.monotonic() - training_start
        final_generation = _generation_name(
            cursor.epoch,
            cursor.next_step,
            cursor.global_step,
        )
        if final_generation != last_checkpoint_generation:
            save_training_checkpoint(
                checkpoints_root,
                final_generation,
                model=_unwrap(distributed_model),
                optimizer=optimizer,
                scheduler=scheduler,
                base_config=base_config,
                cursor=cursor,
                provenance_hashes=provenance_hashes,
            )
        final_parameter_sha256 = assert_parameter_state_agreement(
            _unwrap(distributed_model)
        )
        elapsed_seconds = time.monotonic() - training_start
        elapsed_tensor = torch.tensor(
            training_loop_elapsed_seconds,
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            torch.distributed.all_reduce(
                elapsed_tensor,
                op=torch.distributed.ReduceOp.MAX,
            )
        conservative_elapsed_seconds = float(elapsed_tensor.item())
        throughput = {
            "timing_scope": (
                "Training loop including data loading, optimizer steps, and any "
                "configured periodic checkpoints; excluding initial cache "
                "verification and the final checkpoint write."
            ),
            "max_rank_elapsed_seconds": conservative_elapsed_seconds,
            "completed_valid_examples": completed_valid_examples,
            "completed_valid_residues": completed_valid_residues,
            "valid_examples_per_second": (
                completed_valid_examples / conservative_elapsed_seconds
                if conservative_elapsed_seconds > 0.0
                else None
            ),
            "valid_residues_per_second": (
                completed_valid_residues / conservative_elapsed_seconds
                if conservative_elapsed_seconds > 0.0
                else None
            ),
        }
        if rank == 0:
            _atomic_json(
                run_root / "ENGINEERING_GATE_SUMMARY.json",
                {
                    "format_version": "frontier_real_data_engineering_gate_v1",
                    "run_id": args.run_id,
                    "run_profile": run_profile,
                    "scheduler": scheduler_specification,
                    "status": "completed",
                    "claim_boundary": (
                        "Train-only optimization and systems evidence only. This "
                        "artifact contains no validation, held-out, external-test, "
                        "or biological performance result."
                    ),
                    "fold_id": dataset.contract.fold_id,
                    "world_size": world_size,
                    "device_type": device.type,
                    "precision": args.precision,
                    "completed_global_steps": cursor.global_step,
                    "resume_epoch": cursor.epoch,
                    "resume_next_step": cursor.next_step,
                    "last_train_loss": last_loss,
                    "elapsed_seconds": elapsed_seconds,
                    "throughput": throughput,
                    "peak_cuda_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0
                    ),
                    "operational_max_steps": args.max_steps,
                    "operational_resume": (
                        None
                        if args.resume_generation is None
                        else {
                            "checkpoint_root": str(
                                args.resume_checkpoint_root.expanduser().resolve()
                            ),
                            "generation": args.resume_generation,
                            "receipt_sha256": (
                                args.expected_resume_receipt_sha256
                            ),
                        }
                    ),
                    "final_generation": final_generation,
                    "initial_parameter_sha256": initial_parameter_sha256,
                    "final_parameter_sha256": final_parameter_sha256,
                    "data": inventory_summary,
                    "manifest_sha256": dataset.contract.manifest_sha256,
                    "authorization_manifest_sha256": (
                        dataset.contract.authorization_manifest_sha256
                    ),
                    "source_manifest_sha256": source_manifest_sha256,
                    "source_hashes": source_hashes,
                    "base_config": base_config,
                },
            )
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
