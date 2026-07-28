"""Create-only fixed-horizon controller for PAIRScale E1/E2 screening."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from .pairscale_e1_e2_evaluation import evaluate_e1_e2
from .pairscale_e1_e2_step import (
    PairscaleE1E2Config,
    build_e1_e2_model,
    parameter_path_state_sha256,
    train_one_epoch_e1_e2,
)
from .pairscale_holdout_training import split_dataset_indices
from .pairscale_overfit import configure_determinism, module_source_hashes
from .pairscale_partner_specificity import (
    BidirectionalDecoyPanel,
    DECOY_POLICY,
    select_score_free_bidirectional_decoys,
)
from .pairscale_unary_training import _append_jsonl, _create_json, _load_examples


REPORT_FORMAT = "pairscale_e1_e2_screen_report_v1"
METRICS_FORMAT = "pairscale_e1_e2_epoch_metrics_v1"
ACCESS_FORMAT = "pairscale_e1_e2_development_holdout_access_receipt_v1"
RUN_MANIFEST_FORMAT = "pairscale_e1_e2_run_manifest_v1"
DECOY_ASSIGNMENT_FORMAT = "pairscale_e1_e2_bidirectional_decoy_assignment_v1"
SCREENING_SEEDS = (17, 42, 73)


def locked_e1_e2_config(seed: int) -> PairscaleE1E2Config:
    """Return one of the three preregistered fixed-horizon screening configs."""
    if seed not in SCREENING_SEEDS:
        raise ValueError("seed is not in the preregistered screening set")
    config = PairscaleE1E2Config(seed=seed)
    config.validate()
    return config


def e1_e2_source_hashes() -> dict[str, str]:
    """Bind every directly executed scientific module to exact source bytes."""
    result = module_source_hashes()
    for module_name in (
        "interaction_first.development_holdout",
        "interaction_first.pairformer_evaluator",
        "interaction_first.pairscale_e1_e2_evaluation",
        "interaction_first.pairscale_e1_e2_step",
        "interaction_first.pairscale_e1_e2_training",
        "interaction_first.pairscale_holdout_training",
        "interaction_first.pairscale_partner_specificity",
        "interaction_first.pairscale_unary_training",
    ):
        module = importlib.import_module(module_name)
        path = Path(str(module.__file__)).resolve()
        if path.name in result:
            raise RuntimeError("Executed scientific module basenames are not unique")
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def _lower_sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_locked_config(config: PairscaleE1E2Config) -> None:
    config.validate()
    expected = locked_e1_e2_config(config.seed).canonical_payload()
    if config.canonical_payload() != expected:
        raise ValueError("E1/E2 configuration differs from the execution lock")


def _endpoint_length(example: DevelopmentContactExample, axis: str) -> int:
    tensor = (
        example.antibody_embeddings
        if axis == "antibody"
        else example.antigen_embeddings
    )
    return int(tensor.shape[0])


def _decoy_assignment_payload(
    examples: Sequence[DevelopmentContactExample],
    panel: BidirectionalDecoyPanel,
) -> Mapping[str, Any]:
    panel.validate(len(examples))
    rows = []
    for index, example in enumerate(examples):
        rows.append(
            {
                "matched": {
                    "instance_id": example.instance_id,
                    "antibody_id": example.antibody_id,
                    "antigen_id": example.antigen_id,
                    "antibody_length": _endpoint_length(example, "antibody"),
                    "antigen_length": _endpoint_length(example, "antigen"),
                },
                "antibody_to_antigen": [
                    {
                        "instance_id": examples[decoy_index].instance_id,
                        "antigen_id": examples[decoy_index].antigen_id,
                        "antigen_length": _endpoint_length(
                            examples[decoy_index],
                            "antigen",
                        ),
                    }
                    for decoy_index in panel.antibody_to_antigen[index]
                ],
                "antigen_to_antibody": [
                    {
                        "instance_id": examples[decoy_index].instance_id,
                        "antibody_id": examples[decoy_index].antibody_id,
                        "antibody_length": _endpoint_length(
                            examples[decoy_index],
                            "antibody",
                        ),
                    }
                    for decoy_index in panel.antigen_to_antibody[index]
                ],
            }
        )
    return {
        "format_version": DECOY_ASSIGNMENT_FORMAT,
        "policy": DECOY_POLICY,
        "seed": panel.seed,
        "count_per_direction": panel.count_per_direction,
        "selection_inputs": [
            "antibody_id",
            "antigen_id",
            "antibody_length",
            "antigen_length",
            "candidate_instance_id",
            "matched_instance_id",
            "observed_pair_incidence",
            "seed",
        ],
        "forbidden_fields_accessed": False,
        "embedding_values_accessed": False,
        "unknown_cross_pair_semantics": (
            "unlabeled_ranking_control_not_measured_nonbinder"
        ),
        "rows": rows,
    }


def _compact_evaluation(evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value
        for key, value in evaluation.items()
        if key
        not in {
            "per_complex",
            "partner_controls",
            "exhaustive_bidirectional_retrieval",
        }
    }


def run_e1_e2_screen(
    *,
    dataset: DevelopmentContactDataset,
    assignment: ComponentHoldoutAssignment,
    assignment_receipt_file_sha256: str,
    protocol_file_sha256: str,
    controller_sha256: str,
    scheduler_script_sha256: str,
    output_root: Path,
    config: PairscaleE1E2Config,
    resume_checkpoint_root: Path | None = None,
    resume_generation: str | None = None,
    expected_resume_receipt_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Train one seed, freeze epoch 260, then evaluate development data once."""
    import torch

    _validate_locked_config(config)
    for value, name in (
        (assignment_receipt_file_sha256, "assignment_receipt_file_sha256"),
        (protocol_file_sha256, "protocol_file_sha256"),
        (controller_sha256, "controller_sha256"),
        (scheduler_script_sha256, "scheduler_script_sha256"),
    ):
        _lower_sha256(value, name)
    if len(assignment.train_instances) != 192:
        raise ValueError("Training assignment must contain exactly 192 complexes")
    if len(assignment.holdout_instances) != 64:
        raise ValueError("Development holdout must contain exactly 64 complexes")
    train_indices, holdout_indices = split_dataset_indices(dataset, assignment)
    train_examples = _load_examples(dataset, train_indices)
    decoy_panel = select_score_free_bidirectional_decoys(
        train_examples,
        count=config.decoy_count_per_direction,
        seed=config.seed,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    assignment_payload = _decoy_assignment_payload(train_examples, decoy_panel)
    assignment_sha256 = hashlib.sha256(
        canonical_json_bytes(dict(assignment_payload))
    ).hexdigest()
    assignment_path = output_root / "BIDIRECTIONAL_DECOY_ASSIGNMENT.json"
    _create_json(assignment_path, assignment_payload)
    if sha256_file(assignment_path) != assignment_sha256:
        raise RuntimeError("Decoy assignment serialization differs")

    configure_determinism(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Real-data E1/E2 screening requires CUDA")
    device = torch.device("cuda", 0)
    model = build_e1_e2_model(dataset, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    sources = e1_e2_source_hashes()
    provenance = {
        "cache_manifest_sha256": dataset.manifest_sha256,
        "cache_receipt_file_sha256": dataset.receipt_file_sha256,
        "assignment_file_sha256": assignment.assignment_file_sha256,
        "assignment_content_sha256": assignment.assignment_content_sha256,
        "assignment_receipt_file_sha256": assignment_receipt_file_sha256,
        "protocol_file_sha256": protocol_file_sha256,
        "controller_sha256": controller_sha256,
        "scheduler_script_sha256": scheduler_script_sha256,
        "decoy_assignment_sha256": assignment_sha256,
        **{f"source_{name}": digest for name, digest in sources.items()},
    }
    config_payload = {
        "training": dict(config.canonical_payload()),
        "contact_logit_formula": (
            "antibody_unary_plus_antigen_unary_plus_interaction_residual"
        ),
        "specificity_score": (
            "double_centered_independent_specificity_interaction_residual"
        ),
        "training_decoy_policy": DECOY_POLICY,
        "source_complex_count": len(dataset),
        "train_complex_count": len(train_indices),
        "development_holdout_complex_count": len(holdout_indices),
        "checkpoint_selection": "none_fixed_epoch_260_only",
        "early_stopping": False,
        "phase_2_parameter_policy": "complete_unary_path_bitwise_frozen",
        "unused_geometry_head_policy": "bitwise_frozen_all_phases",
    }
    start_epoch = 1
    global_step = 0
    resume_evidence: Mapping[str, Any] | None = None
    unary_phase_2_reference_sha256: str | None = None
    unary_epoch_220_sha256: str | None = None
    unary_epoch_260_sha256: str | None = None
    unused_path_reference_sha256 = parameter_path_state_sha256(model, "unused")
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
            scheduler=None,
            scaler=None,
            map_location="cpu",
            expected_provenance_hashes=provenance,
            expected_receipt_sha256=str(expected_resume_receipt_sha256),
            expected_rng_world_size=1,
        )
        checkpoint_config = resume_evidence["config"]
        if (
            not isinstance(checkpoint_config, dict)
            or checkpoint_config.get("base_config") != config_payload
        ):
            raise ValueError("Resume checkpoint configuration differs")
        integrity = checkpoint_config.get("integrity")
        if not isinstance(integrity, dict):
            raise ValueError("Resume checkpoint integrity state is missing")
        restored_unused = integrity.get("unused_path_reference_sha256")
        if restored_unused != unused_path_reference_sha256:
            raise ValueError("Resume checkpoint unused-path state differs")
        restored_unary = integrity.get("unary_phase_2_reference_sha256")
        if restored_unary is not None:
            _lower_sha256(restored_unary, "unary_phase_2_reference_sha256")
            unary_phase_2_reference_sha256 = restored_unary
        restored_epoch_220 = integrity.get("unary_epoch_220_sha256")
        if restored_epoch_220 is not None:
            _lower_sha256(restored_epoch_220, "unary_epoch_220_sha256")
            unary_epoch_220_sha256 = restored_epoch_220
        start_epoch = int(resume_evidence["epoch"]) + 1
        global_step = int(resume_evidence["global_step"])
        if start_epoch > config.epochs:
            raise ValueError("Resume checkpoint already reached the fixed horizon")

    metrics_path = output_root / "train_epoch_metrics.jsonl"
    checkpoints_root = output_root / "checkpoints"
    manifest = {
        "format_version": RUN_MANIFEST_FORMAT,
        "scope": "development_only_endpoint_disjoint_e1_e2_screen",
        "claim_boundary": (
            "Development-only fixed-seed repair. The reused holdout is not an "
            "untouched final test. No confirmatory, promotion, scaling, SOTA, "
            "or world-leading claim is authorized."
        ),
        "screening_seed": config.seed,
        "screening_seed_set": list(SCREENING_SEEDS),
        "development_holdout_is_untouched_final_test": False,
        "new_final_test_accessed": False,
        "unknown_cross_pairs_assigned_binary_labels": False,
        "decoy_assignment_sha256": assignment_sha256,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_receipt_file_sha256": dataset.receipt_file_sha256,
        "assignment_file_sha256": assignment.assignment_file_sha256,
        "assignment_content_sha256": assignment.assignment_content_sha256,
        "assignment_receipt_file_sha256": assignment_receipt_file_sha256,
        "protocol_file_sha256": protocol_file_sha256,
        "controller_sha256": controller_sha256,
        "scheduler_script_sha256": scheduler_script_sha256,
        "source_sha256": sources,
        "config": config_payload,
        "resume": None if resume_evidence is None else dict(resume_evidence),
    }
    _create_json(output_root / "RUN_MANIFEST.json", manifest)

    last_train_evaluation: Mapping[str, Any] | None = None
    last_receipt: Mapping[str, Any] | None = None
    for epoch in range(start_epoch, config.epochs + 1):
        losses, steps = train_one_epoch_e1_e2(
            model=model,
            optimizer=optimizer,
            examples=train_examples,
            device=device,
            config=config,
            epoch=epoch,
            decoy_panel=decoy_panel,
        )
        global_step += steps
        unary_path_sha256 = parameter_path_state_sha256(model, "unary")
        unused_path_sha256 = parameter_path_state_sha256(model, "unused")
        if unused_path_sha256 != unused_path_reference_sha256:
            raise RuntimeError("Unused geometry-head parameters changed")
        if epoch == config.unary_warmup_end_epoch:
            unary_phase_2_reference_sha256 = unary_path_sha256
        elif config.unary_warmup_end_epoch < epoch <= config.residual_phase_end_epoch:
            if unary_phase_2_reference_sha256 is None:
                raise RuntimeError("Phase-2 unary reference is unavailable")
            if unary_path_sha256 != unary_phase_2_reference_sha256:
                raise RuntimeError("Complete unary path changed during phase 2")
        if epoch == config.residual_phase_end_epoch:
            unary_epoch_220_sha256 = unary_path_sha256
        if epoch == config.epochs:
            unary_epoch_260_sha256 = unary_path_sha256
        should_evaluate = (
            epoch == 1
            or epoch % config.eval_every_epochs == 0
            or epoch == config.epochs
        )
        if should_evaluate:
            last_train_evaluation = evaluate_e1_e2(
                model=model,
                examples=train_examples,
                device=device,
                decoy_count=config.decoy_count_per_direction,
                seed=config.seed,
                include_partner_panel=False,
                include_exhaustive_retrieval=False,
            )
        payload = {
            "format_version": METRICS_FORMAT,
            "epoch": epoch,
            "global_step": global_step,
            "mean_training_loss": losses["total"],
            "mean_loss_components": dict(losses),
            "learning_rate": losses["learning_rate"],
            "train_evaluation": (
                _compact_evaluation(last_train_evaluation)
                if should_evaluate and last_train_evaluation is not None
                else None
            ),
            "development_holdout_evaluation": None,
            "unary_path_state_sha256": unary_path_sha256,
            "unused_geometry_path_state_sha256": unused_path_sha256,
        }
        _append_jsonl(metrics_path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
        if (
            epoch % config.checkpoint_every_epochs == 0
            or epoch == config.epochs
        ):
            checkpoint_config = {
                "base_config": config_payload,
                "integrity": {
                    "unary_phase_2_reference_sha256": (
                        unary_phase_2_reference_sha256
                    ),
                    "unary_epoch_220_sha256": unary_epoch_220_sha256,
                    "unused_path_reference_sha256": (
                        unused_path_reference_sha256
                    ),
                },
            }
            generation = f"epoch-{epoch:04d}-step-{global_step:08d}"
            receipt = create_checkpoint_generation(
                checkpoints_root,
                generation,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                scaler=None,
                epoch=epoch,
                global_step=global_step,
                provenance_hashes=provenance,
                config=checkpoint_config,
            )
            last_receipt = {
                **dict(receipt),
                "receipt_file_sha256": sha256_file(
                    checkpoints_root / generation / "receipt.json"
                ),
            }
    if last_train_evaluation is None or last_receipt is None:
        raise RuntimeError("E1/E2 run ended without final evidence")
    if int(last_receipt["epoch"]) != config.epochs:
        raise RuntimeError("Final immutable checkpoint is not epoch 260")
    if (
        unary_phase_2_reference_sha256 is None
        or unary_epoch_220_sha256 != unary_phase_2_reference_sha256
        or unary_epoch_260_sha256 is None
    ):
        raise RuntimeError("Unary-path integrity evidence is incomplete")
    if parameter_path_state_sha256(model, "unused") != (
        unused_path_reference_sha256
    ):
        raise RuntimeError("Unused geometry-head integrity failed")

    train_final_evaluation = evaluate_e1_e2(
        model=model,
        examples=train_examples,
        device=device,
        decoy_count=config.decoy_count_per_direction,
        seed=config.seed,
        include_partner_panel=True,
        include_exhaustive_retrieval=False,
    )
    access_receipt = {
        "format_version": ACCESS_FORMAT,
        "development_holdout_is_untouched_final_test": False,
        "new_final_test_accessed": False,
        "frozen_checkpoint_epoch": int(last_receipt["epoch"]),
        "frozen_checkpoint_sha256": last_receipt["checkpoint_sha256"],
        "frozen_checkpoint_receipt_file_sha256": last_receipt[
            "receipt_file_sha256"
        ],
        "holdout_instance_count": len(holdout_indices),
        "holdout_deserialization_count_within_runner": 1,
        "model_selection_authorized": False,
        "checkpoint_promotion_authorized": False,
        "confirmatory_claim_authorized": False,
        "scaling_authorized": False,
        "phase_2_unary_path_bitwise_identical": True,
        "unary_epoch_20_sha256": unary_phase_2_reference_sha256,
        "unary_epoch_220_sha256": unary_epoch_220_sha256,
        "unused_geometry_path_bitwise_identical": True,
        "unused_geometry_path_sha256": unused_path_reference_sha256,
    }
    _create_json(output_root / "DEVELOPMENT_HOLDOUT_ACCESS_RECEIPT.json", access_receipt)
    holdout_examples = _load_examples(dataset, holdout_indices)
    if len(holdout_examples) != len(holdout_indices):
        raise RuntimeError("Development holdout deserialization count differs")
    holdout_evaluation = evaluate_e1_e2(
        model=model,
        examples=holdout_examples,
        device=device,
        decoy_count=config.decoy_count_per_direction,
        seed=config.seed,
        include_partner_panel=True,
        include_exhaustive_retrieval=True,
    )
    _create_json(output_root / "TRAIN_FINAL_EVALUATION.json", train_final_evaluation)
    _create_json(
        output_root / "DEVELOPMENT_HOLDOUT_EVALUATION.json",
        holdout_evaluation,
    )
    report = {
        "format_version": REPORT_FORMAT,
        "scope": "development_only_endpoint_disjoint_e1_e2_screen",
        "screening_seed": config.seed,
        "completed_epoch": config.epochs,
        "global_step": global_step,
        "fixed_epoch_rule_satisfied": True,
        "early_stopping_used": False,
        "development_holdout_evaluation_performed": True,
        "development_holdout_is_untouched_final_test": False,
        "new_final_test_accessed": False,
        "model_selection_authorized": False,
        "checkpoint_promotion_authorized": False,
        "confirmatory_claim_authorized": False,
        "scaling_authorized": False,
        "train_evaluation": train_final_evaluation,
        "development_holdout_evaluation": holdout_evaluation,
        "frozen_checkpoint": dict(last_receipt),
        "gate_status": "pending_cross_seed_component_bootstrap_and_permutation",
        "parameter_path_integrity": {
            "phase_2_unary_path_bitwise_identical": True,
            "unary_epoch_20_sha256": unary_phase_2_reference_sha256,
            "unary_epoch_220_sha256": unary_epoch_220_sha256,
            "unary_epoch_260_sha256": unary_epoch_260_sha256,
            "unused_geometry_path_bitwise_identical": True,
            "unused_geometry_path_sha256": unused_path_reference_sha256,
        },
    }
    _create_json(output_root / "FINAL_REPORT.json", report)
    return report


__all__ = [
    "ACCESS_FORMAT",
    "DECOY_ASSIGNMENT_FORMAT",
    "METRICS_FORMAT",
    "REPORT_FORMAT",
    "RUN_MANIFEST_FORMAT",
    "SCREENING_SEEDS",
    "e1_e2_source_hashes",
    "locked_e1_e2_config",
    "run_e1_e2_screen",
]
