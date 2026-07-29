"""Train-only PAIRScale development holdout runner.

The runner trains only on one authenticated endpoint-disjoint partition. It
does not deserialize holdout tensors until the training-only L0 gate has passed
and the final training checkpoint has been frozen. The holdout is then evaluated
exactly once and cannot control epochs, checkpoints, thresholds, or promotion.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .development_authority import CLAIM_BOUNDARY, canonical_json_bytes
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
from .pairscale_overfit import (
    PairscaleOverfitConfig,
    build_contact_only_model,
    configure_determinism,
    evaluate_overfit,
    module_source_hashes as overfit_source_hashes,
    train_one_epoch,
)


REPORT_FORMAT = "pairscale_development_double_cold_report_v1"
METRICS_FORMAT = "pairscale_development_train_epoch_metrics_v1"
HOLDOUT_ACCESS_FORMAT = "pairscale_development_holdout_access_receipt_v1"


def split_dataset_indices(
    dataset: DevelopmentContactDataset,
    assignment: ComponentHoldoutAssignment,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Map assignment identities without deserializing any tensor artifact."""

    identity_to_index: dict[str, int] = {}
    identities: dict[str, tuple[str, str]] = {}
    for index in range(len(dataset)):
        record = dataset.identity(index)
        if record.instance_id in identity_to_index:
            raise ValueError("Dataset contains duplicate instance identities")
        identity_to_index[record.instance_id] = index
        identities[record.instance_id] = (record.antibody_id, record.antigen_id)
    train_ids = tuple(value.instance_id for value in assignment.train_instances)
    holdout_ids = tuple(value.instance_id for value in assignment.holdout_instances)
    assigned_ids = set(train_ids).union(holdout_ids)
    if set(identity_to_index) != assigned_ids:
        raise ValueError("Dataset and assignment instance inventories differ")
    for value in (*assignment.train_instances, *assignment.holdout_instances):
        if identities[value.instance_id] != (value.antibody_id, value.antigen_id):
            raise ValueError("Dataset endpoint identity differs from assignment")
    train_indices = tuple(identity_to_index[value] for value in train_ids)
    holdout_indices = tuple(identity_to_index[value] for value in holdout_ids)
    if set(train_indices).intersection(holdout_indices):
        raise AssertionError("Dataset indices cross the assignment")
    return train_indices, holdout_indices


def load_examples(
    dataset: DevelopmentContactDataset,
    indices: Sequence[int],
) -> tuple[DevelopmentContactExample, ...]:
    """Deserialize exactly the requested authenticated dataset partition."""

    if not indices:
        raise ValueError("Example indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ValueError("Example indices contain duplicates")
    return tuple(dataset[index] for index in indices)


def holdout_training_source_hashes() -> dict[str, str]:
    """Bind all executed scientific modules to their exact source bytes."""

    result = overfit_source_hashes()
    for module_name in (
        "interaction_first.development_holdout",
        "interaction_first.pairscale_holdout_training",
    ):
        module = importlib.import_module(module_name)
        path = Path(str(module.__file__)).resolve()
        if path.name in result:
            raise RuntimeError("Executed module basenames are not unique")
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(dict(payload))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(dict(payload)))
        handle.flush()


def _holdout_evaluation_payload(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Relabel generic contact metrics with the strict holdout claim boundary."""

    payload = dict(evaluation)
    payload.update(
        {
            "scope": "authority_train_only_development_double_cold_holdout",
            "evaluation_role": "single_frozen_development_holdout_evaluation",
            "held_out_evaluation_performed": True,
            "model_selection_authorized": False,
            "checkpoint_promotion_authorized": False,
            "confirmatory_claim_authorized": False,
            "gate_checks": {},
            "gate_passed": False,
            "interpretation": (
                "Development evidence only. The label-derived marginal outer score "
                "is an oracle diagnostic, not a deployable baseline. Gate L1 remains "
                "unresolved until independent unary controls and cluster-bootstrap "
                "comparisons are complete."
            ),
        }
    )
    return payload


def run_development_holdout_training(
    *,
    dataset: DevelopmentContactDataset,
    assignment: ComponentHoldoutAssignment,
    assignment_receipt_file_sha256: str,
    controller_sha256: str,
    scheduler_script_sha256: str,
    output_root: Path,
    config: PairscaleOverfitConfig,
    resume_checkpoint_root: Path | None = None,
    resume_generation: str | None = None,
    expected_resume_receipt_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Train on the frozen train partition and evaluate holdout at most once."""

    import torch

    config.validate()
    for digest, name in (
        (assignment_receipt_file_sha256, "assignment_receipt_file_sha256"),
        (controller_sha256, "controller_sha256"),
        (scheduler_script_sha256, "scheduler_script_sha256"),
    ):
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if config.expected_complex_count != len(assignment.train_instances):
        raise ValueError("Config train count differs from assignment")
    train_indices, holdout_indices = split_dataset_indices(dataset, assignment)
    if len(holdout_indices) != len(assignment.holdout_instances):
        raise AssertionError("Holdout index count differs from assignment")
    train_examples = load_examples(dataset, train_indices)
    if len(train_examples) != config.expected_complex_count:
        raise AssertionError("Train tensor count differs from configuration")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    metrics_path = output_root / "train_epoch_metrics.jsonl"
    checkpoints_root = output_root / "checkpoints"
    configure_determinism(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This real-data holdout run requires CUDA")
    device = torch.device("cuda", 0)
    model = build_contact_only_model(dataset, config).to(device)
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
    sources = holdout_training_source_hashes()
    provenance = {
        "cache_manifest_sha256": dataset.manifest_sha256,
        "cache_receipt_file_sha256": dataset.receipt_file_sha256,
        "assignment_file_sha256": assignment.assignment_file_sha256,
        "assignment_content_sha256": assignment.assignment_content_sha256,
        "assignment_receipt_file_sha256": assignment_receipt_file_sha256,
        "controller_sha256": controller_sha256,
        "scheduler_script_sha256": scheduler_script_sha256,
        **{f"source_{name}": digest for name, digest in sources.items()},
    }
    config_payload: dict[str, Any] = {
        "training": config.canonical_payload(),
        "source_complex_count": len(dataset),
        "train_complex_count": len(train_indices),
        "holdout_complex_count": len(holdout_indices),
        "holdout_access_policy": (
            "deserialize_once_after_training_gate_and_checkpoint_freeze"
        ),
    }
    start_epoch = 1
    global_step = 0
    resume_evidence: Mapping[str, Any] | None = None
    resume_requested = any(
        value is not None
        for value in (
            resume_checkpoint_root,
            resume_generation,
            expected_resume_receipt_sha256,
        )
    )
    if resume_requested:
        if not all(
            value is not None
            for value in (
                resume_checkpoint_root,
                resume_generation,
                expected_resume_receipt_sha256,
            )
        ):
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
            raise ValueError("Resume checkpoint configuration mismatch")
        start_epoch = int(resume_evidence["epoch"]) + 1
        global_step = int(resume_evidence["global_step"])
    run_manifest = {
        "format_version": "pairscale_development_double_cold_run_manifest_v1",
        "scope": "authority_train_only_development_double_cold",
        "claim_boundary": CLAIM_BOUNDARY,
        "model_selection_authorized": False,
        "checkpoint_promotion_authorized": False,
        "confirmatory_claim_authorized": False,
        "validation_or_test_data_accessed": False,
        "holdout_tensor_accessed_before_training_freeze": False,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_receipt_file_sha256": dataset.receipt_file_sha256,
        "assignment_file_sha256": assignment.assignment_file_sha256,
        "assignment_content_sha256": assignment.assignment_content_sha256,
        "assignment_receipt_file_sha256": assignment_receipt_file_sha256,
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
    _atomic_json(output_root / "RUN_MANIFEST.json", run_manifest)
    last_train_evaluation: Mapping[str, Any] | None = None
    last_receipt: Mapping[str, Any] | None = None
    if resume_evidence is not None:
        last_train_evaluation = evaluate_overfit(
            model=model,
            examples=train_examples,
            device=device,
            config=config,
        )
    for epoch in range(start_epoch, config.epochs + 1):
        if last_train_evaluation is not None and bool(
            last_train_evaluation["gate_passed"]
        ):
            break
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
            or epoch == config.epochs
        )
        if should_evaluate:
            last_train_evaluation = evaluate_overfit(
                model=model,
                examples=train_examples,
                device=device,
                config=config,
            )
        epoch_metrics = {
            "format_version": METRICS_FORMAT,
            "epoch": epoch,
            "global_step": global_step,
            "mean_training_loss": losses["total"],
            "mean_loss_components": dict(losses),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train_evaluation": (
                last_train_evaluation if should_evaluate else None
            ),
            "holdout_evaluation": None,
        }
        _append_jsonl(metrics_path, epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
        should_checkpoint = (
            epoch % config.checkpoint_every_epochs == 0
            or epoch == config.epochs
            or (
                last_train_evaluation is not None
                and bool(last_train_evaluation["gate_passed"])
            )
        )
        if should_checkpoint:
            generation = f"epoch-{epoch:04d}-step-{global_step:08d}"
            last_receipt = create_checkpoint_generation(
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
                **dict(last_receipt),
                "receipt_file_sha256": sha256_file(
                    checkpoints_root / generation / "receipt.json"
                ),
            }
    if last_train_evaluation is None:
        raise RuntimeError("Training ended without a train-only evaluation")
    if last_receipt is None:
        if resume_evidence is None:
            raise RuntimeError("Training ended without checkpoint evidence")
        generation = f"epoch-{int(resume_evidence['epoch']):04d}-step-{global_step:08d}"
        last_receipt = create_checkpoint_generation(
            checkpoints_root,
            generation,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            epoch=int(resume_evidence["epoch"]),
            global_step=global_step,
            provenance_hashes=provenance,
            config=config_payload,
        )
        last_receipt = {
            **dict(last_receipt),
            "receipt_file_sha256": sha256_file(
                checkpoints_root / generation / "receipt.json"
            ),
        }
    train_gate_passed = bool(last_train_evaluation["gate_passed"])
    holdout_evaluation: Mapping[str, Any] | None = None
    if train_gate_passed:
        holdout_examples = load_examples(dataset, holdout_indices)
        access_receipt = {
            "format_version": HOLDOUT_ACCESS_FORMAT,
            "assignment_file_sha256": assignment.assignment_file_sha256,
            "frozen_checkpoint_sha256": last_receipt["checkpoint_sha256"],
            "frozen_checkpoint_receipt_file_sha256": last_receipt[
                "receipt_file_sha256"
            ],
            "holdout_instance_count": len(holdout_examples),
            "holdout_access_count_within_runner": 1,
            "holdout_access_condition": "train_only_l0_gate_passed",
            "model_selection_authorized": False,
            "checkpoint_promotion_authorized": False,
            "confirmatory_claim_authorized": False,
        }
        _atomic_json(output_root / "HOLDOUT_ACCESS_RECEIPT.json", access_receipt)
        holdout_evaluation = _holdout_evaluation_payload(
            evaluate_overfit(
                model=model,
                examples=holdout_examples,
                device=device,
                config=config,
            )
        )
        _atomic_json(
            output_root / "HOLDOUT_EVALUATION.json",
            holdout_evaluation,
        )
    report = {
        "format_version": REPORT_FORMAT,
        "scope": "authority_train_only_development_double_cold",
        "claim_boundary": (
            "Development-only endpoint-disjoint evidence. No validation, test, "
            "confirmatory, checkpoint-promotion, biological-generalization, "
            "comparative, or state-of-the-art claim is authorized."
        ),
        "completed_epoch": int(last_receipt["epoch"]),
        "global_step": int(last_receipt["global_step"]),
        "train_l0_gate_passed": train_gate_passed,
        "holdout_evaluation_performed": holdout_evaluation is not None,
        "holdout_access_count_within_runner": 1 if holdout_evaluation else 0,
        "train_evaluation": dict(last_train_evaluation),
        "holdout_evaluation": (
            None if holdout_evaluation is None else dict(holdout_evaluation)
        ),
        "frozen_checkpoint": dict(last_receipt),
        "gate_l1_status": "unresolved_pending_unary_controls_and_bootstrap",
    }
    _atomic_json(output_root / "FINAL_REPORT.json", report)
    return report
