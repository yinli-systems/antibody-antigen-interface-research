"""Tests for deterministic sharding, tail masking, and resume contracts."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from frontier_test_cache import build_fold_local_cache  # noqa: E402
from interaction_first.frontier_data import FoldLocalFrontierDataset  # noqa: E402
from interaction_first.frontier_distributed import (  # noqa: E402
    symmetric_multi_positive_info_nce,
)
from interaction_first.frontier_model import ProtocolD0Retriever  # noqa: E402
from interaction_first.frontier_training import (  # noqa: E402
    D0FrontierBatch,
    DistributedSampleReference,
    DistributedTrainingDataset,
    ExactDistributedBatchSampler,
    MATERIALIZED_PERMUTATION_INDEX_BYTES,
    MATERIALIZED_PERMUTATION_STRATEGY,
    MAX_MATERIALIZED_PERMUTATION_INDICES,
    SAMPLER_SCALE_SCOPE,
    UNSUPPORTED_SCALE_ACTION,
    TrainingBatchCollator,
    build_training_checkpoint_config,
    create_training_dataloader,
    d0_forward_loss,
    make_training_cursor,
    parameter_state_sha256,
    restore_training_checkpoint,
    save_training_checkpoint,
    training_cursor_from_checkpoint_config,
)


def _digest(value: str) -> str:
    """Return one deterministic synthetic SHA-256."""
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _dataset(tmp_path: Path, instance_count: int = 5) -> FoldLocalFrontierDataset:
    cache_root, authorization_path, _, _ = build_fold_local_cache(
        tmp_path,
        instance_count=instance_count,
    )
    return FoldLocalFrontierDataset(cache_root, authorization_path)


def test_sampler_covers_every_example_once_and_masks_only_the_tail() -> None:
    samplers = [
        ExactDistributedBatchSampler(
            5,
            local_batch_size=2,
            world_size=2,
            rank=rank,
            seed=17,
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        sampler.set_epoch(3)

    valid_indices = []
    for step in range(len(samplers[0])):
        rank_batches = [sampler.batch_for_step(step) for sampler in samplers]
        assert rank_batches[0][0].global_indices == rank_batches[1][0].global_indices
        assert (
            rank_batches[0][0].global_valid_mask
            == rank_batches[1][0].global_valid_mask
        )
        global_indices = rank_batches[0][0].global_indices
        global_valid = rank_batches[0][0].global_valid_mask
        valid_indices.extend(
            index
            for index, is_valid in zip(global_indices, global_valid)
            if is_valid
        )
    assert sorted(valid_indices) == list(range(5))
    assert len(valid_indices) == 5
    assert samplers[0].batch_for_step(1)[0].global_valid_mask == (
        True,
        False,
        False,
        False,
    )

    repeated = ExactDistributedBatchSampler(
        5,
        local_batch_size=2,
        world_size=2,
        rank=0,
        seed=17,
    )
    repeated.set_epoch(3)
    assert repeated.batch_for_step(0) == samplers[0].batch_for_step(0)
    repeated.set_epoch(4)
    assert (
        repeated.batch_for_step(0)[0].global_indices
        != samplers[0].batch_for_step(0)[0].global_indices
    )


def test_sampler_materializes_only_one_permutation_per_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = torch.randperm

    def counted_randperm(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "randperm", counted_randperm)
    sampler = ExactDistributedBatchSampler(
        101,
        local_batch_size=4,
        world_size=2,
        rank=0,
        seed=53,
    )
    for step in range(len(sampler)):
        sampler.batch_for_step(step)
    assert calls == 1
    sampler.set_epoch(1)
    sampler.batch_for_step(0)
    assert calls == 2


def test_sampler_contract_fails_closed_outside_finite_cache_scope() -> None:
    sampler = ExactDistributedBatchSampler(
        101,
        local_batch_size=4,
        world_size=2,
        rank=0,
        seed=53,
    )
    contract = sampler.contract()
    assert contract["permutation_strategy"] == MATERIALIZED_PERMUTATION_STRATEGY
    assert (
        contract["permutation_memory_bytes_per_rank"]
        == 101 * MATERIALIZED_PERMUTATION_INDEX_BYTES
    )
    assert (
        contract["maximum_materialized_permutation_indices"]
        == MAX_MATERIALIZED_PERMUTATION_INDICES
    )
    assert contract["scale_scope"] == SAMPLER_SCALE_SCOPE
    assert contract["billion_scale_supported"] is False
    assert contract["unsupported_scale_action"] == UNSUPPORTED_SCALE_ACTION

    with pytest.raises(AttributeError):
        sampler.dataset_size = 10**9
    with pytest.raises(ValueError, match="billion-scale pretraining"):
        ExactDistributedBatchSampler(
            MAX_MATERIALIZED_PERMUTATION_INDICES + 1,
            local_batch_size=1,
        )


def test_tail_fillers_receive_unique_neutral_identities_and_exact_masks(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    sampler = ExactDistributedBatchSampler(
        len(dataset),
        local_batch_size=2,
        world_size=2,
        rank=1,
        seed=7,
        shuffle=False,
    )
    sampler.set_epoch(0)
    references = sampler.batch_for_step(1)
    adapter = DistributedTrainingDataset(dataset)
    batch = TrainingBatchCollator(dataset)([adapter[item] for item in references])

    assert batch.local_valid_mask.tolist() == [False, False]
    assert len(set(batch.frontier.instance_ids)) == 2
    assert all(
        value.startswith("__frontier_padding__")
        for value in batch.frontier.instance_ids
    )
    assert not bool(batch.frontier.contact_mask.any())
    assert batch.global_valid_mask == (True, False, False, False)
    assert batch.global_positive_mask[0, 0]
    assert not bool(batch.global_positive_mask[1:].any())
    assert bool(batch.global_exclusion_mask[1:, :].all())
    assert bool(batch.global_exclusion_mask[:, 1:].all())


def test_tail_safe_loss_matches_valid_submatrix_and_uses_fp32_logits() -> None:
    antibody = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
            [0.2, 0.2],
        ],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    antigen = torch.tensor(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.7, 0.7],
            [0.3, 0.3],
        ],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    positive = torch.eye(4, dtype=torch.bool)
    excluded = torch.zeros(4, 4, dtype=torch.bool)
    positive[3, 3] = False
    excluded[3, :] = True
    excluded[:, 3] = True
    valid = torch.tensor([True, True, True, False])

    details = symmetric_multi_positive_info_nce(
        antibody,
        antigen,
        positive,
        excluded,
        temperature=0.2,
        local_valid_mask=valid,
        return_details=True,
    )
    expected = symmetric_multi_positive_info_nce(
        antibody[:3],
        antigen[:3],
        torch.eye(3, dtype=torch.bool),
        temperature=0.2,
    )

    assert details["loss"].dtype == torch.float32
    assert details["similarity_dtype"] == "torch.float32"
    assert details["global_antibody_count"] == 3
    assert details["global_padded_count"] == 1
    torch.testing.assert_close(details["loss"], expected)
    details["loss"].backward()
    assert torch.count_nonzero(antibody.grad[3]) == 0
    assert torch.count_nonzero(antigen.grad[3]) == 0
    assert torch.isfinite(antibody.grad[:3]).all()


def test_similarity_matmul_remains_fp32_inside_outer_autocast() -> None:
    antibody = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    antigen = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    positive = torch.eye(4, dtype=torch.bool)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        details = symmetric_multi_positive_info_nce(
            antibody,
            antigen,
            positive,
            return_details=True,
        )

    assert details["similarity_dtype"] == "torch.float32"
    assert details["loss"].dtype == torch.float32


def test_real_batch_runs_d0_without_tail_entering_the_denominator(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, instance_count=3)
    sampler = ExactDistributedBatchSampler(
        len(dataset),
        local_batch_size=4,
        seed=19,
        shuffle=False,
    )
    loader = create_training_dataloader(dataset, sampler, num_workers=0)
    batch = next(iter(loader))
    model = ProtocolD0Retriever(
        antibody_input_dim=7,
        antigen_input_dim=5,
        embedding_dim=4,
        projection_hidden_dim=8,
        dropout=0.0,
    )

    details = d0_forward_loss(model, batch)

    assert details["global_antibody_count"] == 3
    assert details["global_padded_count"] == 1
    assert details["similarity_dtype"] == "torch.float32"
    assert torch.isfinite(details["loss"])
    details["loss"].backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_compact_d0_batch_matches_full_batch_loss_and_gradients(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, instance_count=4)
    sampler = ExactDistributedBatchSampler(
        len(dataset),
        local_batch_size=4,
        seed=31,
        shuffle=False,
    )
    references = sampler.batch_for_step(0)
    adapter = DistributedTrainingDataset(dataset)
    sampled = [adapter[reference] for reference in references]
    full_batch = TrainingBatchCollator(dataset)(sampled)
    compact_batch = TrainingBatchCollator(
        dataset,
        materialize_contact_tensors=False,
    )(sampled)

    assert isinstance(compact_batch.frontier, D0FrontierBatch)
    assert not hasattr(compact_batch.frontier, "pair_mask")
    assert not hasattr(compact_batch.frontier, "contact_target")
    torch.testing.assert_close(
        compact_batch.frontier.antibody_embeddings,
        full_batch.frontier.antibody_embeddings,
    )
    torch.testing.assert_close(
        compact_batch.frontier.antigen_embeddings,
        full_batch.frontier.antigen_embeddings,
    )

    reference_model = ProtocolD0Retriever(
        antibody_input_dim=7,
        antigen_input_dim=5,
        embedding_dim=4,
        projection_hidden_dim=8,
        dropout=0.0,
    )
    compact_model = ProtocolD0Retriever(
        antibody_input_dim=7,
        antigen_input_dim=5,
        embedding_dim=4,
        projection_hidden_dim=8,
        dropout=0.0,
    )
    compact_model.load_state_dict(reference_model.state_dict())
    full_loss = d0_forward_loss(reference_model, full_batch)["loss"]
    compact_loss = d0_forward_loss(compact_model, compact_batch)["loss"]
    torch.testing.assert_close(compact_loss, full_loss, rtol=0.0, atol=0.0)
    full_loss.backward()
    compact_loss.backward()
    for reference_parameter, compact_parameter in zip(
        reference_model.parameters(),
        compact_model.parameters(),
    ):
        torch.testing.assert_close(
            compact_parameter.grad,
            reference_parameter.grad,
            rtol=0.0,
            atol=0.0,
        )


def test_compact_d0_loader_is_explicit_and_tail_safe(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, instance_count=3)
    sampler = ExactDistributedBatchSampler(
        len(dataset),
        local_batch_size=4,
        seed=37,
        shuffle=False,
    )
    loader = create_training_dataloader(
        dataset,
        sampler,
        num_workers=0,
        materialize_contact_tensors=False,
    )
    batch = next(iter(loader))

    assert isinstance(batch.frontier, D0FrontierBatch)
    assert batch.local_valid_mask.tolist() == [True, True, True, False]
    assert batch.global_valid_mask == (True, True, True, False)


def test_parameter_state_hash_supports_scalar_and_non_scalar_tensors() -> None:
    class ScalarStateModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.register_buffer("temperature", torch.tensor(0.07))

    module = ScalarStateModule()
    initial = parameter_state_sha256(module)
    repeated = parameter_state_sha256(module)
    assert initial == repeated

    with torch.no_grad():
        module.temperature.add_(0.01)
    assert parameter_state_sha256(module) != initial


def test_checkpoint_cursor_rejects_data_sampler_and_config_drift(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    sampler = ExactDistributedBatchSampler(
        len(dataset),
        local_batch_size=2,
        world_size=1,
        rank=0,
        seed=23,
    )
    sampler.set_epoch(1)
    cursor = make_training_cursor(
        dataset,
        sampler,
        epoch=1,
        next_step=1,
        global_step=4,
    )
    base_config = {"model": "d0", "world_size": 1}
    checkpoint_config = build_training_checkpoint_config(base_config, cursor)

    restored = training_cursor_from_checkpoint_config(
        checkpoint_config,
        expected_base_config=base_config,
        dataset=dataset,
        sampler=sampler,
    )
    assert restored == cursor

    with pytest.raises(ValueError, match="base configuration"):
        training_cursor_from_checkpoint_config(
            checkpoint_config,
            expected_base_config={"model": "changed", "world_size": 1},
            dataset=dataset,
            sampler=sampler,
        )
    changed_sampler = ExactDistributedBatchSampler(
        len(dataset),
        local_batch_size=1,
        world_size=1,
        rank=0,
        seed=23,
    )
    with pytest.raises(ValueError, match="sampler contract"):
        training_cursor_from_checkpoint_config(
            checkpoint_config,
            expected_base_config=base_config,
            dataset=dataset,
            sampler=changed_sampler,
        )


def test_training_checkpoint_restores_model_optimizer_and_cursor(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "data")
    sampler = ExactDistributedBatchSampler(
        len(dataset),
        local_batch_size=2,
        seed=29,
    )
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    expected_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    cursor = make_training_cursor(
        dataset,
        sampler,
        epoch=0,
        next_step=1,
        global_step=1,
    )
    base_config = {"model": "tiny", "world_size": 1}
    provenance = {"source": _digest("source")}
    checkpoint_root = tmp_path / "checkpoints"
    receipt = save_training_checkpoint(
        checkpoint_root,
        "generation-1",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        base_config=base_config,
        cursor=cursor,
        provenance_hashes=provenance,
    )
    assert receipt is not None
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)

    restored_cursor, metadata = restore_training_checkpoint(
        checkpoint_root,
        "generation-1",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_base_config=base_config,
        dataset=dataset,
        sampler=sampler,
        provenance_hashes=provenance,
        map_location="cpu",
        rank=0,
    )

    assert restored_cursor == cursor
    assert metadata["global_step"] == 1
    for key, expected in expected_state.items():
        torch.testing.assert_close(model.state_dict()[key], expected)


def test_sampler_reference_and_collator_reject_identity_or_plan_forgery(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, instance_count=4)
    sampler = ExactDistributedBatchSampler(
        len(dataset),
        local_batch_size=2,
        seed=41,
        shuffle=False,
    )
    adapter = DistributedTrainingDataset(dataset)
    references = sampler.batch_for_step(0)
    sampled = [adapter[reference] for reference in references]
    sampled[0] = replace(sampled[0], example=dataset[1])
    with pytest.raises(ValueError, match="identity differs"):
        TrainingBatchCollator(dataset)(sampled)

    duplicate_plan = tuple(
        replace(
            reference,
            index=0,
            global_indices=(0, 0),
            global_valid_mask=(True, True),
        )
        for reference in references
    )
    duplicate_sampled = [adapter[reference] for reference in duplicate_plan]
    with pytest.raises(ValueError, match="must be unique"):
        TrainingBatchCollator(dataset)(duplicate_sampled)

    with pytest.raises(ValueError, match="global_indices"):
        DistributedSampleReference(
            index=0,
            valid=True,
            epoch=0,
            step=0,
            global_position=0,
            global_indices=(0, -1),
            global_valid_mask=(True, True),
        )


def test_cursor_rejects_sampler_epoch_drift_and_skipped_global_steps(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, instance_count=4)
    sampler = ExactDistributedBatchSampler(
        len(dataset),
        local_batch_size=2,
        seed=43,
        shuffle=False,
    )
    with pytest.raises(ValueError, match="without skips or replays"):
        make_training_cursor(
            dataset,
            sampler,
            epoch=0,
            next_step=1,
            global_step=2,
        )
    with pytest.raises(ValueError, match="sampler epoch"):
        make_training_cursor(
            dataset,
            sampler,
            epoch=1,
            next_step=0,
            global_step=len(sampler),
        )


def test_checkpoint_world_size_and_preflight_config_fail_before_mutation(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "data", instance_count=4)
    sampler = ExactDistributedBatchSampler(
        len(dataset),
        local_batch_size=2,
        seed=47,
    )
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    cursor = make_training_cursor(
        dataset,
        sampler,
        epoch=0,
        next_step=0,
        global_step=0,
    )
    provenance = {"source": _digest("preflight-source")}
    checkpoint_root = tmp_path / "checkpoints"
    receipt = save_training_checkpoint(
        checkpoint_root,
        "preflight",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        base_config={"model": "tiny", "world_size": 1},
        cursor=cursor,
        provenance_hashes=provenance,
    )
    assert receipt is not None
    with torch.no_grad():
        model.weight.add_(10.0)
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="base configuration"):
        restore_training_checkpoint(
            checkpoint_root,
            "preflight",
            model=model,
            optimizer=optimizer,
            scheduler=None,
            expected_base_config={"model": "changed", "world_size": 1},
            dataset=dataset,
            sampler=sampler,
            provenance_hashes=provenance,
            map_location="cpu",
            rank=0,
        )
    torch.testing.assert_close(model.weight, before)

    distributed_sampler = ExactDistributedBatchSampler(
        len(dataset),
        local_batch_size=1,
        world_size=2,
        rank=0,
        seed=47,
    )
    distributed_cursor = make_training_cursor(
        dataset,
        distributed_sampler,
        epoch=0,
        next_step=0,
        global_step=0,
    )
    with pytest.raises(ValueError, match="world size"):
        save_training_checkpoint(
            tmp_path / "invalid-world",
            "generation",
            model=model,
            optimizer=optimizer,
            scheduler=None,
            base_config={"world_size": 2},
            cursor=distributed_cursor,
            provenance_hashes=provenance,
        )


def test_training_checkpoint_restores_amp_scaler_state(tmp_path: Path) -> None:
    class SyntheticScaler:
        """Minimal tensor-only scaler state for a CPU checkpoint test."""

        def __init__(self, scale: float) -> None:
            self.scale = torch.tensor(scale)

        def state_dict(self) -> dict[str, torch.Tensor]:
            return {"scale": self.scale.clone()}

        def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
            self.scale = state["scale"].clone()

    dataset = _dataset(tmp_path / "data", instance_count=2)
    sampler = ExactDistributedBatchSampler(
        len(dataset),
        local_batch_size=2,
        seed=59,
    )
    cursor = make_training_cursor(
        dataset,
        sampler,
        epoch=0,
        next_step=0,
        global_step=0,
    )
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = SyntheticScaler(1024.0)
    root = tmp_path / "checkpoints"
    save_training_checkpoint(
        root,
        "with-scaler",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        base_config={"precision": "fp16", "world_size": 1},
        cursor=cursor,
        provenance_hashes={"source": _digest("amp-source")},
    )
    scaler.scale.fill_(1.0)
    restore_training_checkpoint(
        root,
        "with-scaler",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        expected_base_config={"precision": "fp16", "world_size": 1},
        dataset=dataset,
        sampler=sampler,
        provenance_hashes={"source": _digest("amp-source")},
        map_location="cpu",
        rank=0,
    )
    torch.testing.assert_close(scaler.scale, torch.tensor(1024.0))
