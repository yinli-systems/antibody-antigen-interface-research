"""Tests for FSDP2 and distributed-checkpoint engineering primitives."""

from __future__ import annotations

import os
import multiprocessing
from pathlib import Path
import tempfile

import pytest
import torch

from interaction_first.frontier_fsdp2 import (
    FSDP2_CLAIM_BOUNDARY,
    FSDP2ContractError,
    load_dcp_generation,
    percentile,
    save_dcp_generation,
    validate_fsdp2_runtime,
    verify_dcp_generation,
)


PROVENANCE = {
    "protocol": "1" * 64,
    "source": "2" * 64,
    "synthetic_fixture": "3" * 64,
}


def _trained_linear() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Return a deterministic model with populated AdamW state."""
    torch.manual_seed(17)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 20.0
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    return model, optimizer


def _make_writable(path: Path) -> None:
    """Restore test artifact permissions so pytest can remove its temp root."""
    os.chmod(path, 0o755)
    for candidate in path.iterdir():
        os.chmod(candidate, 0o644)


def _distributed_dcp_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    root: str,
) -> None:
    """Exercise collective DCP save and restore on a real Gloo group."""
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
    )
    try:
        model, optimizer = _trained_linear()
        expected = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        save_dcp_generation(
            Path(root) / "checkpoints",
            "distributed-step",
            model=model,
            optimizer=optimizer,
            provenance_hashes=PROVENANCE,
            training_state={"global_step": 1, "synthetic_only": True},
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(float(rank + 1))
        load_dcp_generation(
            Path(root) / "checkpoints",
            "distributed-step",
            model=model,
            optimizer=optimizer,
            expected_provenance_hashes=PROVENANCE,
        )
        exact = all(
            torch.equal(tensor, expected[name])
            for name, tensor in model.state_dict().items()
        )
        (Path(root) / f"rank-{rank}.txt").write_text(
            "pass\n" if exact else "fail\n",
            encoding="utf-8",
        )
    finally:
        torch.distributed.destroy_process_group()


def test_runtime_contract_accepts_only_explicit_cuda_fsdp2() -> None:
    receipt = validate_fsdp2_runtime(
        parallelism="fsdp2",
        precision="bfloat16",
        cuda_available=True,
        bf16_supported=True,
        distributed_initialized=True,
        world_size=8,
        torch_version="2.13.0+cu130",
    )
    assert receipt["gradient_reduction"] == "fsdp2_reduce_scatter_average"
    assert receipt["checkpoint_backend"] == "torch_distributed_checkpoint_sync"
    assert receipt["compile_enabled"] is False
    assert receipt["fp8_enabled"] is False
    assert receipt["claim_boundary"] == FSDP2_CLAIM_BOUNDARY


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parallelism": "ddp"}, "parallelism"),
        ({"precision": "float16"}, "precision"),
        ({"cuda_available": False}, "requires CUDA"),
        ({"bf16_supported": False}, "does not support"),
        ({"distributed_initialized": False}, "process group"),
        ({"world_size": 0}, "positive integer"),
    ],
)
def test_runtime_contract_fails_closed(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "parallelism": "fsdp2",
        "precision": "bfloat16",
        "cuda_available": True,
        "bf16_supported": True,
        "distributed_initialized": True,
        "world_size": 8,
        "torch_version": "2.13.0+cu130",
    }
    arguments.update(overrides)
    with pytest.raises(FSDP2ContractError, match=message):
        validate_fsdp2_runtime(**arguments)  # type: ignore[arg-type]


def test_percentile_uses_locked_linear_interpolation() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 0.5) == 2.5
    assert percentile(values, 0.95) == pytest.approx(3.85)
    assert percentile(values, 1.0) == 4.0
    with pytest.raises(FSDP2ContractError, match="cannot be empty"):
        percentile([], 0.5)


def test_dcp_generation_round_trip_restores_model_and_optimizer(
    tmp_path: Path,
) -> None:
    source_model, source_optimizer = _trained_linear()
    expected_model = {
        name: tensor.detach().clone()
        for name, tensor in source_model.state_dict().items()
    }
    receipt = save_dcp_generation(
        tmp_path / "checkpoints",
        "step_000001",
        model=source_model,
        optimizer=source_optimizer,
        provenance_hashes=PROVENANCE,
        training_state={"global_step": 1, "synthetic_only": True},
    )
    generation = tmp_path / "checkpoints" / "step_000001"
    try:
        assert receipt["source_world_size"] == 1
        assert receipt["status"] == "complete"
        assert receipt["training_state"]["synthetic_only"] is True
        assert ".metadata" in receipt["files"]
        assert not bool(generation.stat().st_mode & 0o222)

        restored_model, restored_optimizer = _trained_linear()
        with torch.no_grad():
            for parameter in restored_model.parameters():
                parameter.add_(10.0)
        load_receipt = load_dcp_generation(
            tmp_path / "checkpoints",
            "step_000001",
            model=restored_model,
            optimizer=restored_optimizer,
            expected_provenance_hashes=PROVENANCE,
        )
        assert load_receipt["content_sha256"] == receipt["content_sha256"]
        for name, tensor in restored_model.state_dict().items():
            torch.testing.assert_close(tensor, expected_model[name], rtol=0.0, atol=0.0)
        assert restored_optimizer.state_dict()["state"]
    finally:
        _make_writable(generation)


def test_dcp_generation_is_create_only_and_provenance_bound(
    tmp_path: Path,
) -> None:
    model, optimizer = _trained_linear()
    save_dcp_generation(
        tmp_path,
        "generation-a",
        model=model,
        optimizer=optimizer,
        provenance_hashes=PROVENANCE,
        training_state={"global_step": 1},
    )
    generation = tmp_path / "generation-a"
    try:
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            save_dcp_generation(
                tmp_path,
                "generation-a",
                model=model,
                optimizer=optimizer,
                provenance_hashes=PROVENANCE,
                training_state={"global_step": 2},
            )
        with pytest.raises(FSDP2ContractError, match="provenance"):
            verify_dcp_generation(
                tmp_path,
                "generation-a",
                expected_provenance_hashes={**PROVENANCE, "source": "4" * 64},
            )
    finally:
        _make_writable(generation)


def test_dcp_verifier_rejects_tampered_shard(tmp_path: Path) -> None:
    model, optimizer = _trained_linear()
    receipt = save_dcp_generation(
        tmp_path,
        "generation-b",
        model=model,
        optimizer=optimizer,
        provenance_hashes=PROVENANCE,
        training_state={"global_step": 1},
    )
    generation = tmp_path / "generation-b"
    shard_name = next(name for name in receipt["files"] if name.endswith(".distcp"))
    shard = generation / shard_name
    try:
        os.chmod(generation, 0o755)
        os.chmod(shard, 0o644)
        with shard.open("ab") as handle:
            handle.write(b"tamper")
        with pytest.raises(FSDP2ContractError, match="byte count mismatch"):
            verify_dcp_generation(tmp_path, "generation-b")
    finally:
        _make_writable(generation)


def test_two_rank_gloo_dcp_round_trip() -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("The Gloo backend is unavailable")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rendezvous = f"file://{root / 'rendezvous'}"
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(
                target=_distributed_dcp_worker,
                args=(rank, 2, rendezvous, str(root)),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                pytest.fail("Distributed DCP worker timed out")
            assert process.exitcode == 0
        for rank in range(2):
            assert (root / f"rank-{rank}.txt").read_text(encoding="utf-8") == "pass\n"
        generation = root / "checkpoints" / "distributed-step"
        receipt = verify_dcp_generation(
            root / "checkpoints",
            "distributed-step",
            expected_provenance_hashes=PROVENANCE,
        )
        assert receipt["source_world_size"] == 2
        one_process_model, one_process_optimizer = _trained_linear()
        with torch.no_grad():
            for parameter in one_process_model.parameters():
                parameter.zero_()
        load_dcp_generation(
            root / "checkpoints",
            "distributed-step",
            model=one_process_model,
            optimizer=one_process_optimizer,
            expected_provenance_hashes=PROVENANCE,
        )
        expected_model, _ = _trained_linear()
        for name, tensor in one_process_model.state_dict().items():
            torch.testing.assert_close(
                tensor,
                expected_model.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )
        _make_writable(generation)
