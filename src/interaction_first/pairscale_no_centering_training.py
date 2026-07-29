"""Fixed-epoch PAIRScale ablation without double-centered training scores."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from .pair_geometry_losses import masked_contact_bce_with_logits
from .pairscale_holdout_training import split_dataset_indices
from .pairscale_overfit import (
    PairscaleOverfitConfig,
    _forward_contact,
    _forward_cross_pair,
    _select_cyclic_decoys,
    _top_k_mean,
    build_contact_only_model,
    configure_determinism,
    deterministic_epoch_order,
    evaluate_overfit,
    hard_negative_contact_ranking_loss,
    module_source_hashes as overfit_source_hashes,
)
from .pairscale_unary_training import _append_jsonl, _create_json, _load_examples


REPORT_FORMAT = "pairscale_no_centering_ablation_report_v1"
METRICS_FORMAT = "pairscale_no_centering_ablation_epoch_metrics_v1"
ACCESS_FORMAT = "pairscale_no_centering_ablation_holdout_access_receipt_v1"
RUN_MANIFEST_FORMAT = "pairscale_no_centering_ablation_run_manifest_v1"
FIXED_STOP_EPOCH = 260
SCHEDULER_HORIZON_EPOCHS = 400


def locked_no_centering_config() -> PairscaleOverfitConfig:
    """Return the reference model configuration with its 400-epoch LR horizon."""

    return PairscaleOverfitConfig(
        seed=17,
        expected_complex_count=192,
        epochs=SCHEDULER_HORIZON_EPOCHS,
        learning_rate=0.001,
        minimum_learning_rate_fraction=0.05,
        weight_decay=0.0,
        positive_weight=1024.0,
        contact_ranking_weight=1.0,
        hard_negative_ratio=8,
        contact_ranking_temperature=1.0,
        partner_ranking_weight=0.25,
        partner_ranking_margin=0.10,
        partner_ranking_temperature=1.0,
        gradient_clip_norm=5.0,
        d_model=192,
        pair_dim=96,
        num_heads=6,
        num_fusion_layers=2,
        ffn_dim=384,
        pair_chunk_size=64,
        dropout=0.0,
        eval_every_epochs=10,
        checkpoint_every_epochs=10,
        precision="bf16",
    )


def raw_topk_partner_ranking_loss(
    matched_outputs: Mapping[str, Any],
    decoy_outputs: Mapping[str, Any],
    *,
    positive_pair_count: int,
    margin: float,
    temperature: float,
) -> Any:
    """Rank a cognate pair using raw top-k logits without any centering."""

    import torch

    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be positive and finite")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    matched_top = _top_k_mean(
        matched_outputs["contact_logits"].float()[0],
        positive_pair_count,
    )
    decoy_top = _top_k_mean(
        decoy_outputs["contact_logits"].float()[0],
        positive_pair_count,
    )
    return torch.nn.functional.softplus(
        (margin - matched_top + decoy_top) / temperature
    ) * temperature


def train_one_epoch_without_centering(
    *,
    model: Any,
    optimizer: Any,
    examples: Sequence[DevelopmentContactExample],
    device: Any,
    config: PairscaleOverfitConfig,
    epoch: int,
) -> tuple[Mapping[str, float], int]:
    """Train once over all candidates using only raw top-k partner ranking."""

    import torch

    model.train()
    loss_sums = {
        "total": 0.0,
        "contact_bce": 0.0,
        "contact_ranking": 0.0,
        "partner_ranking_raw_topk": 0.0,
    }
    step_count = 0
    use_bf16 = config.precision == "bf16" and device.type == "cuda"
    decoy_indices = _select_cyclic_decoys(examples)
    for index in deterministic_epoch_order(len(examples), config.seed, epoch):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
            enabled=use_bf16,
        ):
            outputs, targets = _forward_contact(model, examples[index], device)
            contact_bce = masked_contact_bce_with_logits(
                outputs["contact_logits"],
                targets,
                outputs["candidate_pair_mask"],
                positive_weight=config.positive_weight,
            )
            contact_ranking = hard_negative_contact_ranking_loss(
                outputs["contact_logits"],
                targets,
                outputs["candidate_pair_mask"],
                negative_ratio=config.hard_negative_ratio,
                temperature=config.contact_ranking_temperature,
            )
            decoy_outputs = _forward_cross_pair(
                model,
                examples[index],
                examples[decoy_indices[index]],
                device,
            )
            partner_ranking = raw_topk_partner_ranking_loss(
                outputs,
                decoy_outputs,
                positive_pair_count=examples[index].positive_pair_count,
                margin=config.partner_ranking_margin,
                temperature=config.partner_ranking_temperature,
            )
            loss = (
                contact_bce
                + config.contact_ranking_weight * contact_ranking
                + config.partner_ranking_weight * partner_ranking
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, example {index}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("Non-finite gradient norm")
        optimizer.step()
        for name, value in (
            ("total", loss),
            ("contact_bce", contact_bce),
            ("contact_ranking", contact_ranking),
            ("partner_ranking_raw_topk", partner_ranking),
        ):
            loss_sums[name] += float(value.detach().cpu())
        step_count += 1
    return {
        name: value / step_count for name, value in loss_sums.items()
    }, step_count


def no_centering_source_hashes() -> dict[str, str]:
    """Bind all directly executed scientific modules to exact bytes."""

    result = overfit_source_hashes()
    for module_name in (
        "interaction_first.development_holdout",
        "interaction_first.pairscale_holdout_training",
        "interaction_first.pairscale_no_centering_training",
        "interaction_first.pairscale_unary_training",
    ):
        module = importlib.import_module(module_name)
        path = Path(str(module.__file__)).resolve()
        if path.name in result:
            raise RuntimeError("Executed scientific module basenames are not unique")
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def _validate_locked_config(config: PairscaleOverfitConfig) -> None:
    config.validate()
    if config.canonical_payload() != locked_no_centering_config().canonical_payload():
        raise ValueError("No-centering configuration differs from the protocol")


def run_no_centering_ablation(
    *,
    dataset: DevelopmentContactDataset,
    assignment: ComponentHoldoutAssignment,
    assignment_receipt_file_sha256: str,
    protocol_file_sha256: str,
    controller_sha256: str,
    scheduler_script_sha256: str,
    output_root: Path,
    config: PairscaleOverfitConfig,
    resume_checkpoint_root: Path | None = None,
    resume_generation: str | None = None,
    expected_resume_receipt_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Run the fixed no-centering training ablation and one holdout evaluation."""

    import torch

    _validate_locked_config(config)
    for value, name in (
        (assignment_receipt_file_sha256, "assignment_receipt_file_sha256"),
        (protocol_file_sha256, "protocol_file_sha256"),
        (controller_sha256, "controller_sha256"),
        (scheduler_script_sha256, "scheduler_script_sha256"),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if len(assignment.train_instances) != 192 or len(assignment.holdout_instances) != 64:
        raise ValueError("Assignment size differs from the protocol")
    train_indices, holdout_indices = split_dataset_indices(dataset, assignment)
    train_examples = _load_examples(dataset, train_indices)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    metrics_path = output_root / "train_epoch_metrics.jsonl"
    checkpoints_root = output_root / "checkpoints"
    configure_determinism(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Real-data no-centering training requires CUDA")
    device = torch.device("cuda", 0)
    model = build_contact_only_model(dataset, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SCHEDULER_HORIZON_EPOCHS,
        eta_min=config.learning_rate * config.minimum_learning_rate_fraction,
    )
    sources = no_centering_source_hashes()
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
        "training": config.canonical_payload(),
        "fixed_training_stop_epoch": FIXED_STOP_EPOCH,
        "scheduler_horizon_epochs": SCHEDULER_HORIZON_EPOCHS,
        "partner_ranking_score": "raw_topk_only_no_centering",
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
        if start_epoch > FIXED_STOP_EPOCH:
            raise ValueError("Resume checkpoint is already at or beyond epoch 260")

    manifest = {
        "format_version": RUN_MANIFEST_FORMAT,
        "scope": "authority_train_only_development_double_cold_no_centering",
        "claim_boundary": (
            "Development-only endpoint-disjoint centering ablation. No model "
            "selection, checkpoint promotion, confirmatory, scaling, comparative "
            "superiority, or state-of-the-art claim is authorized."
        ),
        "training_partner_score": "raw_topk_only_no_centering",
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
        "config": config_payload,
        "source_sha256": sources,
        "resume": None if resume_evidence is None else dict(resume_evidence),
    }
    _create_json(output_root / "RUN_MANIFEST.json", manifest)

    last_train_evaluation: Mapping[str, Any] | None = None
    last_receipt: Mapping[str, Any] | None = None
    for epoch in range(start_epoch, FIXED_STOP_EPOCH + 1):
        losses, steps = train_one_epoch_without_centering(
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
            or epoch == FIXED_STOP_EPOCH
        )
        if should_evaluate:
            last_train_evaluation = evaluate_overfit(
                model=model,
                examples=train_examples,
                device=device,
                config=config,
            )
        payload = {
            "format_version": METRICS_FORMAT,
            "epoch": epoch,
            "global_step": global_step,
            "mean_training_loss": losses["total"],
            "mean_loss_components": dict(losses),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train_evaluation": last_train_evaluation if should_evaluate else None,
            "holdout_evaluation": None,
        }
        _append_jsonl(metrics_path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        if epoch % config.checkpoint_every_epochs == 0 or epoch == FIXED_STOP_EPOCH:
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
        raise RuntimeError("No-centering run ended without final evidence")
    if int(last_receipt["epoch"]) != FIXED_STOP_EPOCH:
        raise RuntimeError("Final immutable checkpoint is not epoch 260")

    holdout_examples = _load_examples(dataset, holdout_indices)
    access_receipt = {
        "format_version": ACCESS_FORMAT,
        "assignment_file_sha256": assignment.assignment_file_sha256,
        "protocol_file_sha256": protocol_file_sha256,
        "training_partner_score": "raw_topk_only_no_centering",
        "frozen_checkpoint_epoch": int(last_receipt["epoch"]),
        "frozen_checkpoint_sha256": last_receipt["checkpoint_sha256"],
        "frozen_checkpoint_receipt_file_sha256": last_receipt[
            "receipt_file_sha256"
        ],
        "holdout_instance_count": len(holdout_examples),
        "holdout_deserialization_count_within_runner": 1,
        "model_selection_authorized": False,
        "checkpoint_promotion_authorized": False,
        "confirmatory_claim_authorized": False,
        "scaling_authorized": False,
    }
    _create_json(output_root / "HOLDOUT_ACCESS_RECEIPT.json", access_receipt)
    holdout_evaluation = dict(
        evaluate_overfit(
            model=model,
            examples=holdout_examples,
            device=device,
            config=config,
        )
    )
    holdout_evaluation.update(
        {
            "scope": "authority_train_only_development_double_cold_no_centering",
            "training_partner_score": "raw_topk_only_no_centering",
            "evaluation_role": "single_fixed_checkpoint_development_holdout",
            "held_out_evaluation_performed": True,
            "model_selection_authorized": False,
            "checkpoint_promotion_authorized": False,
            "confirmatory_claim_authorized": False,
            "scaling_authorized": False,
            "gate_checks": {},
            "gate_passed": False,
        }
    )
    _create_json(output_root / "HOLDOUT_EVALUATION.json", holdout_evaluation)
    report = {
        "format_version": REPORT_FORMAT,
        "scope": "authority_train_only_development_double_cold_no_centering",
        "training_partner_score": "raw_topk_only_no_centering",
        "completed_epoch": FIXED_STOP_EPOCH,
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
        "gate_l1_status": "pending_paired_centering_ablation_statistics",
    }
    _create_json(output_root / "FINAL_REPORT.json", report)
    return report


__all__ = [
    "ACCESS_FORMAT",
    "FIXED_STOP_EPOCH",
    "METRICS_FORMAT",
    "REPORT_FORMAT",
    "RUN_MANIFEST_FORMAT",
    "SCHEDULER_HORIZON_EPOCHS",
    "locked_no_centering_config",
    "no_centering_source_hashes",
    "raw_topk_partner_ranking_loss",
    "run_no_centering_ablation",
    "train_one_epoch_without_centering",
]
