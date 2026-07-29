"""Tests for scheduler and profile locking in the real-data D0 gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")

from frontier_test_cache import build_fold_local_cache  # noqa: E402
from interaction_first.frontier_data import FoldLocalFrontierDataset  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "frontier_real_data_gate.py"
SPEC = importlib.util.spec_from_file_location("frontier_real_data_gate", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _contract_d0_args(**updates: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "run_profile": MODULE.CONTRACT_D0_PROFILE,
        "scheduler": MODULE.COSINE_WARM_RESTARTS_SCHEDULER,
        "epochs": 58,
        "local_batch_size": 128,
        "embedding_dim": 128,
        "projection_hidden_dim": 256,
        "learning_rate": 1e-3,
        "weight_decay": 0.2,
        "warmup_fraction": 0.0,
        "restart_period_epochs": 10,
        "restart_period_multiplier": 2,
        "precision": "bf16",
        "gather_gradient_backend": (
            MODULE.LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND
        ),
        "ddp_init_sync": True,
        "ddp_static_graph": True,
        "ddp_gradient_as_bucket_view": True,
        "max_steps": None,
        "verify_only": False,
        "contract_preflight_only": False,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_contract_d0_builds_epoch_derived_warm_restarts() -> None:
    contract, contract_sha256 = MODULE._load_model_contract()
    args = _contract_d0_args()
    MODULE._validate_contract_d0_profile(args, contract, world_size=8)

    specification = MODULE._scheduler_specification(
        args,
        steps_per_epoch=7,
    )
    assert specification == {
        "id": MODULE.COSINE_WARM_RESTARTS_SCHEDULER,
        "implementation": (
            "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
        ),
        "step_unit": "optimizer_step",
        "period_epochs": 10,
        "period_steps": 70,
        "period_multiplier": 2,
        "eta_min": 0.0,
        "warmup_fraction": 0.0,
        "warmup_steps": 0,
    }
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = MODULE._build_scheduler(optimizer, specification)
    assert isinstance(
        scheduler,
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    )
    assert scheduler.T_0 == 70
    assert scheduler.T_mult == 2

    reference = MODULE._contract_d0_reference(contract, contract_sha256)
    assert reference["contract_file_sha256"] == hashlib.sha256(
        MODULE.MODEL_CONTRACT_PATH.read_bytes()
    ).hexdigest()
    assert reference["scheduler"] == contract["models"][
        MODULE.CONTRACT_D0_MODEL_ID
    ]["scheduler"]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("scheduler", "linear_warmup_cosine"),
        ("epochs", 57),
        ("warmup_fraction", 0.05),
        ("restart_period_epochs", 9),
        ("restart_period_multiplier", 1),
        ("precision", "fp32"),
        ("ddp_static_graph", False),
        ("ddp_gradient_as_bucket_view", False),
        ("max_steps", 10),
        ("verify_only", True),
    ],
)
def test_contract_d0_rejects_runtime_drift(
    field: str,
    invalid_value: object,
) -> None:
    contract, _ = MODULE._load_model_contract()
    args = _contract_d0_args(**{field: invalid_value})
    with pytest.raises(ValueError, match="differs from the locked D0 contract"):
        MODULE._validate_contract_d0_profile(args, contract, world_size=8)


def test_contract_d0_rejects_wrong_global_batch() -> None:
    contract, _ = MODULE._load_model_contract()
    args = _contract_d0_args(local_batch_size=64)
    with pytest.raises(ValueError, match="global_batch_exact_pairs"):
        MODULE._validate_contract_d0_profile(args, contract, world_size=8)


def test_engineering_profile_preserves_linear_warmup_cosine() -> None:
    contract, _ = MODULE._load_model_contract()
    args = _contract_d0_args(
        run_profile=MODULE.ENGINEERING_DIAGNOSTIC_PROFILE,
        scheduler=MODULE.LINEAR_WARMUP_COSINE_SCHEDULER,
        epochs=3,
        warmup_fraction=0.25,
    )
    MODULE._validate_contract_d0_profile(args, contract, world_size=2)
    specification = MODULE._scheduler_specification(
        args,
        steps_per_epoch=4,
    )
    assert specification["id"] == MODULE.LINEAR_WARMUP_COSINE_SCHEDULER
    assert specification["total_steps"] == 12
    assert specification["warmup_steps"] == 3
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    assert isinstance(
        MODULE._build_scheduler(optimizer, specification),
        torch.optim.lr_scheduler.LambdaLR,
    )


def test_warm_restart_scheduler_state_resume_matches_uninterrupted() -> None:
    args = _contract_d0_args()
    specification = MODULE._scheduler_specification(
        args,
        steps_per_epoch=3,
    )
    reference_parameter = torch.nn.Parameter(torch.tensor(1.0))
    reference_optimizer = torch.optim.AdamW([reference_parameter], lr=1e-3)
    reference_scheduler = MODULE._build_scheduler(
        reference_optimizer,
        specification,
    )
    observed_reference = []
    for _ in range(23):
        reference_optimizer.step()
        reference_scheduler.step()
        observed_reference.append(reference_optimizer.param_groups[0]["lr"])

    resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=1e-3)
    resumed_scheduler = MODULE._build_scheduler(
        resumed_optimizer,
        specification,
    )
    observed_resumed = []
    for _ in range(11):
        resumed_optimizer.step()
        resumed_scheduler.step()
        observed_resumed.append(resumed_optimizer.param_groups[0]["lr"])
    optimizer_state = copy.deepcopy(resumed_optimizer.state_dict())
    scheduler_state = copy.deepcopy(resumed_scheduler.state_dict())

    restored_parameter = torch.nn.Parameter(torch.tensor(1.0))
    restored_optimizer = torch.optim.AdamW([restored_parameter], lr=1e-3)
    restored_scheduler = MODULE._build_scheduler(
        restored_optimizer,
        specification,
    )
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler_state)
    for _ in range(12):
        restored_optimizer.step()
        restored_scheduler.step()
        observed_resumed.append(restored_optimizer.param_groups[0]["lr"])

    assert observed_resumed == pytest.approx(observed_reference, abs=0.0, rel=0.0)


def test_model_contract_scheduler_drift_fails_closed(tmp_path: Path) -> None:
    contract, _ = MODULE._load_model_contract()
    changed = copy.deepcopy(contract)
    changed["models"][MODULE.CONTRACT_D0_MODEL_ID]["scheduler"][
        "period_epochs"
    ] = 11
    path = tmp_path / "changed_contract.json"
    path.write_text(
        json.dumps(changed, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="scheduler contract"):
        MODULE._load_model_contract(path)


def test_contract_d0_dataset_lock_accepts_only_exact_cache(
    tmp_path: Path,
) -> None:
    cache_root, authorization_path, _, _ = build_fold_local_cache(
        tmp_path,
        instance_count=3,
    )
    dataset = FoldLocalFrontierDataset(cache_root, authorization_path)
    contract, _ = MODULE._load_model_contract()
    changed = copy.deepcopy(contract)
    lock = changed["data"]["engineering_cache"]
    lock.update(
        {
            "authorization_manifest_sha256": (
                dataset.contract.authorization_manifest_sha256
            ),
            "cache_manifest_sha256": dataset.contract.manifest_sha256,
            "exact_pair_count": len(dataset.contract.positive_pairs),
            "excluded_pair_count": len(dataset.contract.excluded_pairs),
            "selected_instance_count": len(dataset.contract.records),
            "selected_antibody_count": len(
                {record.antibody_id for record in dataset.contract.records}
            ),
            "selected_antigen_count": len(
                {record.antigen_id for record in dataset.contract.records}
            ),
            "selected_instance_membership_sha256": (
                MODULE._instance_membership_sha256(dataset)
            ),
        }
    )
    args = _contract_d0_args()

    proof = MODULE._validate_contract_d0_dataset(args, changed, dataset)
    assert proof["validated"] is True
    assert proof["selected_instance_count"] == 3

    changed["data"]["engineering_cache"]["cache_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="data differs"):
        MODULE._validate_contract_d0_dataset(args, changed, dataset)


def test_real_gate_requires_external_data_and_resume_anchors() -> None:
    digest = "a" * 64
    base = {
        "expected_authorization_manifest_sha256": digest,
        "expected_cache_manifest_sha256": digest,
        "resume_generation": None,
        "expected_resume_receipt_sha256": None,
    }
    MODULE._validate_external_anchors(argparse.Namespace(**base))

    with pytest.raises(ValueError, match="authorization"):
        MODULE._validate_external_anchors(
            argparse.Namespace(
                **{
                    **base,
                    "expected_authorization_manifest_sha256": "not-a-digest",
                }
            )
        )
    with pytest.raises(ValueError, match="requires"):
        MODULE._validate_external_anchors(
            argparse.Namespace(
                **{
                    **base,
                    "resume_generation": "generation-000001",
                }
            )
        )
    with pytest.raises(ValueError, match="only valid"):
        MODULE._validate_external_anchors(
            argparse.Namespace(
                **{
                    **base,
                    "expected_resume_receipt_sha256": digest,
                }
            )
        )
    MODULE._validate_external_anchors(
        argparse.Namespace(
            **{
                **base,
                "resume_generation": "generation-000001",
                "expected_resume_receipt_sha256": digest,
            }
        )
    )


def test_slurm_wrapper_propagates_profile_and_scheduler() -> None:
    wrapper = (
        REPOSITORY_ROOT / "slurm" / "frontier_real_data_gate.sbatch"
    ).read_text(encoding="utf-8")
    required_fragments = (
        'RUN_PROFILE=${FRONTIER_RUN_PROFILE:-engineering_diagnostic}',
        'SCHEDULER=${FRONTIER_SCHEDULER:-}',
        "FRONTIER_SCHEDULER must be explicitly set for contract_d0",
        "contract_d0 requires cosine_annealing_warm_restarts",
        '--run-profile "$RUN_PROFILE"',
        '--expected-authorization-manifest-sha256 "$EXPECTED_AUTHORIZATION_SHA256"',
        '--expected-cache-manifest-sha256 "$EXPECTED_CACHE_SHA256"',
        '--expected-resume-receipt-sha256 "$RESUME_RECEIPT_SHA256"',
        "FRONTIER_EXPECTED_AUTHORIZATION_MANIFEST_SHA256",
        "FRONTIER_EXPECTED_CACHE_MANIFEST_SHA256",
        "FRONTIER_EXPECTED_RESUME_RECEIPT_SHA256",
        '--scheduler "$SCHEDULER"',
        '--restart-period-epochs "$RESTART_PERIOD_EPOCHS"',
        '--restart-period-multiplier "$RESTART_PERIOD_MULTIPLIER"',
        'if [[ -n "$MAX_STEPS" ]]; then',
        'COMMAND+=(--contract-preflight-only)',
        'NUM_WORKERS=${FRONTIER_NUM_WORKERS:-2}',
        'MEMORY_CACHE_SIZE=${FRONTIER_MEMORY_CACHE_SIZE:-1830}',
        'PREFETCH_FACTOR=${FRONTIER_PREFETCH_FACTOR:-1}',
        'STAGE_CACHE=${FRONTIER_STAGE_CACHE:-true}',
        'LOCAL_ROOT=$(mktemp -d "$LOCAL_BASE/frontier-real-gate-$SLURM_JOB_ID.XXXXXX")',
        'chmod -R u+w "$LOCAL_ROOT" 2>/dev/null || true',
        "Insufficient node-local capacity for fail-closed cache staging",
        '--memory-cache-size "$MEMORY_CACHE_SIZE"',
        'chmod -R a-w "$RUN_ROOT"',
        "torch.cuda.device_count()",
    )
    for fragment in required_fragments:
        assert fragment in wrapper
    stop_index = wrapper.rindex("stop_monitor")
    hash_index = wrapper.index(
        'find "$RUN_ROOT" -type f ! -name artifact_sha256.txt'
    )
    assert stop_index < hash_index
