"""Fixed-epoch training runner for PAIRScale architecture-level unary controls."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

from .development_authority import canonical_json_bytes
from .development_contact_dataset import (
    DevelopmentContactDataset,
    DevelopmentContactExample,
)
from .development_holdout import ComponentHoldoutAssignment
from .frontier_checkpoint import (
    create_checkpoint_generation,
    restore_checkpoint_generation,
    sha256_file,
)
from .pairscale_holdout_training import split_dataset_indices
from .pairscale_overfit import (
    PairscaleOverfitConfig,
    configure_determinism,
    evaluate_overfit,
    module_source_hashes as overfit_source_hashes,
    train_one_epoch,
)
from .pairscale_unary_controls import ControlSide, build_unary_control_model


REPORT_FORMAT = "pairscale_fixed_epoch_unary_control_report_v1"
METRICS_FORMAT = "pairscale_fixed_epoch_unary_epoch_metrics_v1"
ACCESS_FORMAT = "pairscale_fixed_epoch_unary_holdout_access_receipt_v1"
RUN_MANIFEST_FORMAT = "pairscale_fixed_epoch_unary_run_manifest_v1"
LOCKED_EPOCHS = 260
LOCKED_SEED = 17
LOCKED_TRAIN_COUNT = 192
LOCKED_HOLDOUT_COUNT = 64


def _validate_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one canonical JSON object without replacing prior evidence."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing evidence: {path}")
    raw = canonical_json_bytes(dict(payload))
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(dict(payload)))
        handle.flush()
        os.fsync(handle.fileno())


def _load_examples(
    dataset: DevelopmentContactDataset,
    indices: Sequence[int],
) -> tuple[DevelopmentContactExample, ...]:
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Example indices must be nonempty and unique")
    return tuple(dataset[index] for index in indices)


def unary_training_source_hashes() -> dict[str, str]:
    """Bind the complete directly executed scientific source set."""

    result = overfit_source_hashes()
    for module_name in (
        "interaction_first.development_holdout",
        "interaction_first.pairscale_holdout_training",
        "interaction_first.pairscale_unary_controls",
        "interaction_first.pairscale_unary_training",
    ):
        module = importlib.import_module(module_name)
        path = Path(str(module.__file__)).resolve()
        if path.name in result:
            raise RuntimeError("Executed scientific module basenames are not unique")
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def locked_unary_config() -> PairscaleOverfitConfig:
    """Return the frozen optimization configuration for both unary controls."""

    return PairscaleOverfitConfig(
        seed=LOCKED_SEED,
        expected_complex_count=LOCKED_TRAIN_COUNT,
        epochs=LOCKED_EPOCHS,
        learning_rate=0.001,
        minimum_learning_rate_fraction=0.05,
        weight_decay=0.0,
        positive_weight=1024.0,
        contact_ranking_weight=1.0,
        hard_negative_ratio=8,
        contact_ranking_temperature=1.0,
        partner_ranking_weight=0.0,
        gradient_clip_norm=5.0,
        d_model=192,
        num_fusion_layers=2,
        ffn_dim=384,
        dropout=0.0,
        eval_every_epochs=20,
        checkpoint_every_epochs=20,
        precision="bf16",
    )


def _validate_locked_config(config: PairscaleOverfitConfig) -> None:
    config.validate()
    if config.canonical_payload() != locked_unary_config().canonical_payload():
        raise ValueError("Unary training configuration differs from the frozen protocol")


def _evaluation_payload(
    evaluation: Mapping[str, Any],
    control_side: ControlSide,
) -> dict[str, Any]:
    payload = dict(evaluation)
    payload.update(
        {
            "scope": "authority_train_only_development_double_cold_unary_control",
            "control_side": control_side,
            "architecture_reads_partner_feature_values": False,
            "evaluation_role": "single_fixed_checkpoint_development_holdout",
            "held_out_evaluation_performed": True,
            "model_selection_authorized": False,
            "checkpoint_promotion_authorized": False,
            "confirmatory_claim_authorized": False,
            "scaling_authorized": False,
            "gate_checks": {},
            "gate_passed": False,
            "interpretation": (
                "Development-only one-sided control. Pair-model comparison requires "
                "the preregistered connected-component paired bootstrap."
            ),
        }
    )
    return payload


def run_fixed_epoch_unary_control(
    *,
    dataset: DevelopmentContactDataset,
    assignment: ComponentHoldoutAssignment,
    assignment_receipt_file_sha256: str,
    protocol_file_sha256: str,
    controller_sha256: str,
    scheduler_script_sha256: str,
    output_root: Path,
    control_side: ControlSide,
    config: PairscaleOverfitConfig,
    resume_checkpoint_root: Path | None = None,
    resume_generation: str | None = None,
    expected_resume_receipt_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Train exactly 260 epochs, freeze, then deserialize holdout exactly once."""

    import torch

    _validate_locked_config(config)
    for digest, name in (
        (assignment_receipt_file_sha256, "assignment_receipt_file_sha256"),
        (protocol_file_sha256, "protocol_file_sha256"),
        (controller_sha256, "controller_sha256"),
        (scheduler_script_sha256, "scheduler_script_sha256"),
    ):
        _validate_digest(digest, name)
    if len(assignment.train_instances) != LOCKED_TRAIN_COUNT:
        raise ValueError("Assignment train count differs from the protocol")
    if len(assignment.holdout_instances) != LOCKED_HOLDOUT_COUNT:
        raise ValueError("Assignment holdout count differs from the protocol")
    train_indices, holdout_indices = split_dataset_indices(dataset, assignment)
    train_examples = _load_examples(dataset, train_indices)
    if len(train_examples) != LOCKED_TRAIN_COUNT:
        raise AssertionError("Loaded train count differs from the protocol")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    metrics_path = output_root / "train_epoch_metrics.jsonl"
    checkpoints_root = output_root / "checkpoints"
    configure_determinism(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Real-data unary-control training requires CUDA")
    device = torch.device("cuda", 0)
    model = build_unary_control_model(dataset, config, control_side).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * config.minimum_learning_rate_fraction,
    )
    sources = unary_training_source_hashes()
    provenance = {
        "cache_manifest_sha256": dataset.manifest_sha256,
        "cache_receipt_file_sha256": dataset.receipt_file_sha256,
        "assignment_file_sha256": assignment.assignment_file_sha256,
        "assignment_content_sha256": assignment.assignment_content_sha256,
        "assignment_receipt_file_sha256": assignment_receipt_file_sha256,
        "protocol_file_sha256": protocol_file_sha256,
        "controller_sha256": controller_sha256,
        "scheduler_script_sha256": scheduler_script_sha256,
        **{f"source_{name}": digest for name, digest in sources.items()},
    }
    config_payload = {
        "control_side": control_side,
        "training": config.canonical_payload(),
        "source_complex_count": len(dataset),
        "train_complex_count": len(train_indices),
        "holdout_complex_count": len(holdout_indices),
        "epoch_selection": "none_fixed_260",
        "checkpoint_selection": "none_epoch_260_only",
        "holdout_access_policy": "once_after_immutable_epoch_260_checkpoint",
    }

    start_epoch = 1
    global_step = 0
    resume_evidence: Mapping[str, Any] | None = None
    resume_values = (
        resume_checkpoint_root,
        resume_generation,
        expected_resume_receipt_sha256,
    )
    if any(value is not None for value in resume_values):
        if not all(value is not None for value in resume_values):
            raise ValueError("Resume root, generation, and receipt are inseparable")
        resume_evidence = restore_checkpoint_generation(
            Path(resume_checkpoint_root),
            str(resume_generation),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            map_location="cpu",
            expected_provenance_hashes=provenance,
            expected_receipt_sha256=str(expected_resume_receipt_sha256),
            expected_rng_world_size=1,
        )
        if resume_evidence["config"] != config_payload:
            raise ValueError("Resume checkpoint configuration differs")
        start_epoch = int(resume_evidence["epoch"]) + 1
        global_step = int(resume_evidence["global_step"])
        if start_epoch > LOCKED_EPOCHS:
            raise ValueError("Resume checkpoint is already at or beyond epoch 260")

    manifest = {
        "format_version": RUN_MANIFEST_FORMAT,
        "scope": "authority_train_only_development_double_cold_unary_control",
        "claim_boundary": (
            "Development-only endpoint-disjoint unary-control evidence. No model "
            "selection, checkpoint promotion, confirmatory, scaling, comparative "
            "superiority, or state-of-the-art claim is authorized."
        ),
        "control_side": control_side,
        "forbidden_partner_feature_values_read": False,
        "validation_or_test_data_accessed": False,
        "holdout_tensor_accessed_before_training_freeze": False,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_receipt_file_sha256": dataset.receipt_file_sha256,
        "assignment_file_sha256": assignment.assignment_file_sha256,
        "assignment_content_sha256": assignment.assignment_content_sha256,
        "assignment_receipt_file_sha256": assignment_receipt_file_sha256,
        "protocol_file_sha256": protocol_file_sha256,
        "controller_sha256": controller_sha256,
        "scheduler_script_sha256": scheduler_script_sha256,
        "source_instance_count": len(dataset),
        "train_instance_count": len(train_indices),
        "holdout_instance_count": len(holdout_indices),
        "train_candidate_pair_count": sum(
            int(value.contact_mask.sum()) for value in train_examples
        ),
        "train_positive_pair_count": sum(
            value.positive_pair_count for value in train_examples
        ),
        "config": config_payload,
        "source_sha256": sources,
        "resume": None if resume_evidence is None else dict(resume_evidence),
    }
    _create_json(output_root / "RUN_MANIFEST.json", manifest)

    last_train_evaluation: Mapping[str, Any] | None = None
    last_receipt: Mapping[str, Any] | None = None
    for epoch in range(start_epoch, LOCKED_EPOCHS + 1):
        losses, steps = train_one_epoch(
            model=model,
            optimizer=optimizer,
            examples=train_examples,
            device=device,
            config=config,
            epoch=epoch,
        )
        global_step += steps
        scheduler.step()
        should_evaluate = (
            epoch == 1
            or epoch % config.eval_every_epochs == 0
            or epoch == LOCKED_EPOCHS
        )
        if should_evaluate:
            last_train_evaluation = evaluate_overfit(
                model=model,
                examples=train_examples,
                device=device,
                config=config,
            )
        epoch_payload = {
            "format_version": METRICS_FORMAT,
            "control_side": control_side,
            "epoch": epoch,
            "global_step": global_step,
            "mean_training_loss": losses["total"],
            "mean_loss_components": dict(losses),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train_evaluation": last_train_evaluation if should_evaluate else None,
            "holdout_evaluation": None,
        }
        _append_jsonl(metrics_path, epoch_payload)
        print(json.dumps(epoch_payload, sort_keys=True), flush=True)
        if epoch % config.checkpoint_every_epochs == 0 or epoch == LOCKED_EPOCHS:
            generation = f"epoch-{epoch:04d}-step-{global_step:08d}"
            receipt = create_checkpoint_generation(
                checkpoints_root,
                generation,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=epoch,
                global_step=global_step,
                provenance_hashes=provenance,
                config=config_payload,
            )
            last_receipt = {
                **dict(receipt),
                "receipt_file_sha256": sha256_file(
                    checkpoints_root / generation / "receipt.json"
                ),
            }

    if last_train_evaluation is None or last_receipt is None:
        raise RuntimeError("Fixed-epoch training ended without final evidence")
    if int(last_receipt["epoch"]) != LOCKED_EPOCHS:
        raise RuntimeError("Final immutable checkpoint is not epoch 260")

    holdout_examples = _load_examples(dataset, holdout_indices)
    access_receipt = {
        "format_version": ACCESS_FORMAT,
        "control_side": control_side,
        "assignment_file_sha256": assignment.assignment_file_sha256,
        "protocol_file_sha256": protocol_file_sha256,
        "frozen_checkpoint_epoch": int(last_receipt["epoch"]),
        "frozen_checkpoint_sha256": last_receipt["checkpoint_sha256"],
        "frozen_checkpoint_receipt_file_sha256": last_receipt[
            "receipt_file_sha256"
        ],
        "holdout_instance_count": len(holdout_examples),
        "holdout_deserialization_count_within_runner": 1,
        "holdout_access_condition": "immutable_epoch_260_checkpoint_exists",
        "model_selection_authorized": False,
        "checkpoint_promotion_authorized": False,
        "confirmatory_claim_authorized": False,
        "scaling_authorized": False,
    }
    _create_json(output_root / "HOLDOUT_ACCESS_RECEIPT.json", access_receipt)
    holdout_evaluation = _evaluation_payload(
        evaluate_overfit(
            model=model,
            examples=holdout_examples,
            device=device,
            config=config,
        ),
        control_side,
    )
    _create_json(output_root / "HOLDOUT_EVALUATION.json", holdout_evaluation)
    report = {
        "format_version": REPORT_FORMAT,
        "scope": "authority_train_only_development_double_cold_unary_control",
        "control_side": control_side,
        "completed_epoch": LOCKED_EPOCHS,
        "global_step": global_step,
        "fixed_epoch_rule_satisfied": True,
        "early_stopping_used": False,
        "holdout_evaluation_performed": True,
        "holdout_deserialization_count_within_runner": 1,
        "model_selection_authorized": False,
        "checkpoint_promotion_authorized": False,
        "confirmatory_claim_authorized": False,
        "scaling_authorized": False,
        "train_evaluation": dict(last_train_evaluation),
        "holdout_evaluation": holdout_evaluation,
        "frozen_checkpoint": dict(last_receipt),
        "gate_l1_status": "pending_pairwise_component_bootstrap_comparison",
    }
    _create_json(output_root / "FINAL_REPORT.json", report)
    return report


__all__ = [
    "ACCESS_FORMAT",
    "LOCKED_EPOCHS",
    "LOCKED_HOLDOUT_COUNT",
    "LOCKED_SEED",
    "LOCKED_TRAIN_COUNT",
    "METRICS_FORMAT",
    "REPORT_FORMAT",
    "RUN_MANIFEST_FORMAT",
    "locked_unary_config",
    "run_fixed_epoch_unary_control",
    "unary_training_source_hashes",
]
