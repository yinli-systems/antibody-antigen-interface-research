#!/usr/bin/env python3
"""Run a real-cache three-phase and exact-resume PAIRScale E1/E2 smoke."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from interaction_first.development_authority import canonical_json_bytes
from interaction_first.development_contact_dataset import DevelopmentContactDataset
from interaction_first.development_holdout import parse_component_holdout_assignment
from interaction_first.frontier_checkpoint import (
    create_checkpoint_generation,
    restore_checkpoint_generation,
    sha256_file,
)
from interaction_first.pairscale_e1_e2_evaluation import evaluate_e1_e2
from interaction_first.pairscale_e1_e2_step import (
    build_e1_e2_model,
    train_one_epoch_e1_e2,
)
from interaction_first.pairscale_e1_e2_training import (
    e1_e2_source_hashes,
    locked_e1_e2_config,
)
from interaction_first.pairscale_holdout_training import split_dataset_indices
from interaction_first.pairscale_overfit import configure_determinism
from interaction_first.pairscale_partner_specificity import (
    select_score_free_bidirectional_decoys,
)
from interaction_first.pairscale_unary_training import _create_json, _load_examples


SMOKE_FORMAT = "pairscale_e1_e2_real_cache_gpu_smoke_v1"
SMOKE_COMPLEX_COUNT = 8


def _regular_bytes(path: Path) -> bytes:
    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")
    return path.read_bytes()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--cache-receipt-file-sha256", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--subset-manifest-sha256", required=True)
    parser.add_argument("--authorization-manifest-sha256", required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-file-sha256", required=True)
    parser.add_argument("--assignment-content-sha256", required=True)
    parser.add_argument("--assignment-receipt", type=Path, required=True)
    parser.add_argument("--assignment-receipt-file-sha256", required=True)
    parser.add_argument("--expected-controller-sha256", required=True)
    parser.add_argument("--scheduler-script-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _assert_exact(reference: Any, replay: Any, path: str = "root") -> None:
    import torch

    if torch.is_tensor(reference):
        if not torch.is_tensor(replay) or not torch.equal(reference, replay):
            raise RuntimeError(f"Tensor replay differs at {path}")
        return
    if isinstance(reference, Mapping):
        if not isinstance(replay, Mapping) or set(reference) != set(replay):
            raise RuntimeError(f"Mapping replay differs at {path}")
        for key in reference:
            _assert_exact(reference[key], replay[key], f"{path}.{key}")
        return
    if isinstance(reference, (tuple, list)):
        if not isinstance(replay, type(reference)) or len(reference) != len(replay):
            raise RuntimeError(f"Sequence replay differs at {path}")
        for index, (left, right) in enumerate(zip(reference, replay, strict=True)):
            _assert_exact(left, right, f"{path}[{index}]")
        return
    if reference != replay:
        raise RuntimeError(f"Scalar replay differs at {path}")


def _cpu_clone(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return value


def main() -> None:
    import torch

    arguments = _parser().parse_args()
    controller_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if controller_sha256 != arguments.expected_controller_sha256:
        raise ValueError("Smoke controller source digest differs")
    assignment_bytes = _regular_bytes(arguments.assignment)
    receipt_bytes = _regular_bytes(arguments.assignment_receipt)
    if hashlib.sha256(assignment_bytes).hexdigest() != (
        arguments.assignment_file_sha256
    ):
        raise ValueError("Assignment file digest differs")
    if hashlib.sha256(receipt_bytes).hexdigest() != (
        arguments.assignment_receipt_file_sha256
    ):
        raise ValueError("Assignment receipt file digest differs")
    assignment = parse_component_holdout_assignment(
        assignment_bytes,
        receipt_bytes,
        expected_assignment_file_sha256=arguments.assignment_file_sha256,
        expected_assignment_content_sha256=arguments.assignment_content_sha256,
        expected_receipt_file_sha256=arguments.assignment_receipt_file_sha256,
        expected_source_subset_sha256=arguments.subset_manifest_sha256,
        expected_policy_sha256=arguments.policy_sha256,
    )
    dataset = DevelopmentContactDataset(
        arguments.cache_root,
        expected_manifest_sha256=arguments.manifest_sha256,
        expected_receipt_file_sha256=arguments.cache_receipt_file_sha256,
        expected_policy_sha256=arguments.policy_sha256,
        expected_decision_sha256=arguments.decision_sha256,
        expected_subset_manifest_sha256=arguments.subset_manifest_sha256,
        expected_authorization_manifest_sha256=(
            arguments.authorization_manifest_sha256
        ),
    )
    train_indices, _ = split_dataset_indices(dataset, assignment)
    smoke_indices = tuple(
        sorted(
            train_indices,
            key=lambda index: dataset.records[index].instance_id,
        )[:SMOKE_COMPLEX_COUNT]
    )
    examples = _load_examples(dataset, smoke_indices)
    if len(examples) != SMOKE_COMPLEX_COUNT:
        raise RuntimeError("Smoke subset size differs")
    output_root = arguments.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("E1/E2 real-cache smoke requires CUDA")
    device = torch.device("cuda", 0)
    config = replace(
        locked_e1_e2_config(17),
        epochs=3,
        unary_warmup_end_epoch=1,
        residual_phase_end_epoch=2,
        eval_every_epochs=1,
        checkpoint_every_epochs=1,
        precision="bf16",
    )
    config.validate()
    panel = select_score_free_bidirectional_decoys(
        examples,
        count=config.decoy_count_per_direction,
        seed=config.seed,
    )
    source_hashes = e1_e2_source_hashes()
    provenance = {
        "cache_manifest_sha256": dataset.manifest_sha256,
        "cache_receipt_file_sha256": dataset.receipt_file_sha256,
        "assignment_file_sha256": assignment.assignment_file_sha256,
        "assignment_content_sha256": assignment.assignment_content_sha256,
        "assignment_receipt_file_sha256": (
            arguments.assignment_receipt_file_sha256
        ),
        "controller_sha256": controller_sha256,
        "scheduler_script_sha256": arguments.scheduler_script_sha256,
        **{f"source_{name}": digest for name, digest in source_hashes.items()},
    }
    config_payload = {
        "purpose": "real_cache_gpu_three_phase_exact_resume_smoke",
        "training": dict(config.canonical_payload()),
        "instance_ids": [example.instance_id for example in examples],
        "development_holdout_tensor_accessed": False,
        "new_final_test_accessed": False,
    }
    configure_determinism(config.seed)
    model = build_e1_e2_model(dataset, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    started = time.monotonic()
    epoch_1, steps_1 = train_one_epoch_e1_e2(
        model=model,
        optimizer=optimizer,
        examples=examples,
        device=device,
        config=config,
        epoch=1,
        decoy_panel=panel,
    )
    checkpoint_root = output_root / "checkpoint"
    generation = "epoch-0001"
    receipt = create_checkpoint_generation(
        checkpoint_root,
        generation,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=1,
        global_step=steps_1,
        provenance_hashes=provenance,
        config=config_payload,
    )
    receipt_sha256 = sha256_file(checkpoint_root / generation / "receipt.json")
    epoch_2_reference, steps_2_reference = train_one_epoch_e1_e2(
        model=model,
        optimizer=optimizer,
        examples=examples,
        device=device,
        config=config,
        epoch=2,
        decoy_panel=panel,
    )
    reference_model_state = _cpu_clone(model.state_dict())
    reference_optimizer_state = _cpu_clone(optimizer.state_dict())

    configure_determinism(config.seed)
    replay_model = build_e1_e2_model(dataset, config).to(device)
    replay_optimizer = torch.optim.AdamW(
        replay_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    replay_evidence = restore_checkpoint_generation(
        checkpoint_root,
        generation,
        model=replay_model,
        optimizer=replay_optimizer,
        scheduler=None,
        scaler=None,
        map_location="cpu",
        expected_provenance_hashes=provenance,
        expected_receipt_sha256=receipt_sha256,
        expected_rng_world_size=1,
    )
    if replay_evidence["config"] != config_payload:
        raise RuntimeError("Replayed smoke config differs")
    epoch_2_replay, steps_2_replay = train_one_epoch_e1_e2(
        model=replay_model,
        optimizer=replay_optimizer,
        examples=examples,
        device=device,
        config=config,
        epoch=2,
        decoy_panel=panel,
    )
    _assert_exact(reference_model_state, _cpu_clone(replay_model.state_dict()))
    _assert_exact(
        reference_optimizer_state,
        _cpu_clone(replay_optimizer.state_dict()),
    )
    epoch_3, steps_3 = train_one_epoch_e1_e2(
        model=replay_model,
        optimizer=replay_optimizer,
        examples=examples,
        device=device,
        config=config,
        epoch=3,
        decoy_panel=panel,
    )
    evaluation = evaluate_e1_e2(
        model=replay_model,
        examples=examples,
        device=device,
        decoy_count=config.decoy_count_per_direction,
        seed=config.seed,
        include_partner_panel=True,
        include_exhaustive_retrieval=False,
    )
    report = {
        "format_version": SMOKE_FORMAT,
        "passed": True,
        "scope": "authority_train_only_real_cache_engineering_smoke",
        "claim_boundary": (
            "Engineering evidence only; no held-out, model-selection, "
            "confirmatory, scaling, or performance claim."
        ),
        "source_complex_count": len(dataset),
        "smoke_complex_count": len(examples),
        "smoke_instance_ids": [example.instance_id for example in examples],
        "development_holdout_tensor_accessed": False,
        "new_final_test_accessed": False,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "epoch_metrics": [epoch_1, epoch_2_reference, epoch_2_replay, epoch_3],
        "step_counts": [steps_1, steps_2_reference, steps_2_replay, steps_3],
        "checkpoint_receipt": {
            **dict(receipt),
            "receipt_file_sha256": receipt_sha256,
        },
        "resume_model_state_exact": True,
        "resume_optimizer_state_exact": True,
        "evaluation": evaluation,
        "production_training_authorized_by_smoke": True,
        "production_scientific_gate_passed": False,
    }
    output_path = output_root / "SMOKE_REPORT.json"
    _create_json(output_path, report)
    if output_path.read_bytes() != canonical_json_bytes(report):
        raise RuntimeError("Smoke report serialization differs")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
