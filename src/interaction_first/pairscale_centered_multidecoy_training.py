"""Fixed-epoch PAIRScale training with a centered multi-decoy objective."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from .development_authority import canonical_json_bytes
from .development_contact_dataset import DevelopmentContactDataset
from .development_holdout import ComponentHoldoutAssignment
from .frontier_checkpoint import (
    create_checkpoint_generation,
    restore_checkpoint_generation,
    sha256_file,
)
from .pairscale_centered_multidecoy import (
    DECOY_COUNT,
    DECOY_POLICY,
    PARTNER_OBJECTIVE,
    locked_centered_multidecoy_config,
    select_score_free_length_matched_decoys,
    train_one_epoch_centered_multidecoy,
)
from .pairscale_holdout_training import split_dataset_indices
from .pairscale_no_centering_training import (
    FIXED_STOP_EPOCH,
    SCHEDULER_HORIZON_EPOCHS,
    no_centering_source_hashes,
)
from .pairscale_overfit import (
    PairscaleOverfitConfig,
    build_contact_only_model,
    configure_determinism,
    evaluate_overfit,
)
from .pairscale_unary_training import _append_jsonl, _create_json, _load_examples


REPORT_FORMAT = "pairscale_centered_multidecoy_repair_report_v1"
METRICS_FORMAT = "pairscale_centered_multidecoy_repair_epoch_metrics_v1"
ACCESS_FORMAT = "pairscale_centered_multidecoy_holdout_access_receipt_v1"
RUN_MANIFEST_FORMAT = "pairscale_centered_multidecoy_run_manifest_v1"
DECOY_ASSIGNMENT_FORMAT = "pairscale_score_free_multidecoy_assignment_v1"


def centered_multidecoy_source_hashes() -> dict[str, str]:
    """Bind all directly executed scientific modules to exact bytes."""

    result = no_centering_source_hashes()
    for module_name in (
        "interaction_first.pairscale_centered_multidecoy",
        "interaction_first.pairscale_centered_multidecoy_training",
    ):
        module = importlib.import_module(module_name)
        path = Path(str(module.__file__)).resolve()
        if path.name in result:
            raise RuntimeError("Executed scientific module basenames are not unique")
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def _validate_locked_config(config: PairscaleOverfitConfig) -> None:
    config.validate()
    expected = locked_centered_multidecoy_config().canonical_payload()
    if config.canonical_payload() != expected:
        raise ValueError("Centered multi-decoy configuration differs from the protocol")


def _decoy_assignment_payload(
    examples: Any,
    decoy_indices: Any,
    *,
    seed: int,
) -> Mapping[str, Any]:
    rows = []
    for matched_index, indices in enumerate(decoy_indices):
        matched = examples[matched_index]
        rows.append(
            {
                "matched_antigen_id": matched.antigen_id,
                "matched_antigen_length": int(matched.antigen_embeddings.shape[0]),
                "matched_instance_id": matched.instance_id,
                "decoys": [
                    {
                        "antigen_id": examples[index].antigen_id,
                        "antigen_length": int(examples[index].antigen_embeddings.shape[0]),
                        "instance_id": examples[index].instance_id,
                    }
                    for index in indices
                ],
            }
        )
    return {
        "format_version": DECOY_ASSIGNMENT_FORMAT,
        "decoy_count_per_matched_complex": DECOY_COUNT,
        "decoy_policy": DECOY_POLICY,
        "selection_inputs": [
            "antigen_id",
            "antigen_length",
            "candidate_instance_id",
            "matched_instance_id",
            "seed",
        ],
        "score_or_label_fields_accessed": False,
        "embedding_values_accessed": False,
        "seed": seed,
        "rows": rows,
    }


def run_centered_multidecoy_repair(
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
    """Run the frozen centered multi-decoy repair and one holdout evaluation."""

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
    decoy_indices = select_score_free_length_matched_decoys(
        train_examples,
        count=DECOY_COUNT,
        seed=config.seed,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    decoy_assignment = _decoy_assignment_payload(
        train_examples,
        decoy_indices,
        seed=config.seed,
    )
    decoy_assignment_sha256 = hashlib.sha256(
        canonical_json_bytes(dict(decoy_assignment))
    ).hexdigest()
    _create_json(output_root / "DECOY_ASSIGNMENT.json", decoy_assignment)
    if sha256_file(output_root / "DECOY_ASSIGNMENT.json") != decoy_assignment_sha256:
        raise RuntimeError("Decoy assignment serialization differs")
    metrics_path = output_root / "train_epoch_metrics.jsonl"
    checkpoints_root = output_root / "checkpoints"
    configure_determinism(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Real-data centered multi-decoy training requires CUDA")
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
    sources = centered_multidecoy_source_hashes()
    provenance = {
        "cache_manifest_sha256": dataset.manifest_sha256,
        "cache_receipt_file_sha256": dataset.receipt_file_sha256,
        "assignment_file_sha256": assignment.assignment_file_sha256,
        "assignment_content_sha256": assignment.assignment_content_sha256,
        "assignment_receipt_file_sha256": assignment_receipt_file_sha256,
        "protocol_file_sha256": protocol_file_sha256,
        "controller_sha256": controller_sha256,
        "scheduler_script_sha256": scheduler_script_sha256,
        "decoy_assignment_sha256": decoy_assignment_sha256,
        **{f"source_{name}": digest for name, digest in sources.items()},
    }
    config_payload = {
        "training": config.canonical_payload(),
        "fixed_training_stop_epoch": FIXED_STOP_EPOCH,
        "scheduler_horizon_epochs": SCHEDULER_HORIZON_EPOCHS,
        "partner_objective": PARTNER_OBJECTIVE,
        "decoy_policy": DECOY_POLICY,
        "decoy_count_per_matched_complex": DECOY_COUNT,
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
        "scope": "authority_train_only_development_double_cold_centered_multidecoy",
        "claim_boundary": (
            "Development-only endpoint-disjoint partner-objective repair. Unknown "
            "cross-pairs are ranking controls, not measured non-binders. No model "
            "selection, checkpoint promotion, confirmatory, scaling, comparative "
            "superiority, or state-of-the-art claim is authorized."
        ),
        "partner_objective": PARTNER_OBJECTIVE,
        "decoy_policy": DECOY_POLICY,
        "decoy_assignment_sha256": decoy_assignment_sha256,
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
        losses, steps = train_one_epoch_centered_multidecoy(
            model=model,
            optimizer=optimizer,
            examples=train_examples,
            device=device,
            config=config,
            epoch=epoch,
            decoy_indices=decoy_indices,
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
        raise RuntimeError("Centered multi-decoy run ended without final evidence")
    if int(last_receipt["epoch"]) != FIXED_STOP_EPOCH:
        raise RuntimeError("Final immutable checkpoint is not epoch 260")

    holdout_examples = _load_examples(dataset, holdout_indices)
    access_receipt = {
        "format_version": ACCESS_FORMAT,
        "assignment_file_sha256": assignment.assignment_file_sha256,
        "protocol_file_sha256": protocol_file_sha256,
        "partner_objective": PARTNER_OBJECTIVE,
        "decoy_assignment_sha256": decoy_assignment_sha256,
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
            "scope": (
                "authority_train_only_development_double_cold_centered_multidecoy"
            ),
            "partner_objective": PARTNER_OBJECTIVE,
            "training_decoy_policy": DECOY_POLICY,
            "training_decoy_count": DECOY_COUNT,
            "evaluation_role": "single_fixed_checkpoint_development_holdout",
            "held_out_evaluation_performed": True,
            "model_selection_authorized": False,
            "checkpoint_promotion_authorized": False,
            "confirmatory_claim_authorized": False,
            "scaling_authorized": False,
            "interpretation": (
                "Development-only repair evaluation. Comparative inference "
                "requires the preregistered connected-component bootstrap."
            ),
            "gate_checks": {},
            "gate_passed": False,
        }
    )
    _create_json(output_root / "HOLDOUT_EVALUATION.json", holdout_evaluation)
    report = {
        "format_version": REPORT_FORMAT,
        "scope": "authority_train_only_development_double_cold_centered_multidecoy",
        "partner_objective": PARTNER_OBJECTIVE,
        "decoy_policy": DECOY_POLICY,
        "decoy_assignment_sha256": decoy_assignment_sha256,
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
        "gate_l1_status": "pending_paired_component_bootstrap_statistics",
    }
    _create_json(output_root / "FINAL_REPORT.json", report)
    return report


__all__ = [
    "ACCESS_FORMAT",
    "DECOY_ASSIGNMENT_FORMAT",
    "METRICS_FORMAT",
    "REPORT_FORMAT",
    "RUN_MANIFEST_FORMAT",
    "centered_multidecoy_source_hashes",
    "run_centered_multidecoy_repair",
]
