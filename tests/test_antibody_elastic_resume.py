"""Test topology-independent antibody pretraining recovery contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from antibody_pretraining.elastic_resume import (
    ElasticResumeError,
    authenticate_source_rng_manifest,
    destination_rank_seed,
    semantic_optimizer_state,
    semantic_trainer_state_sha256,
    validate_randomness_contract,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def _trainer_state(world_size: int, rng_sha256: str) -> dict[str, object]:
    return {
        "checkpoint_rng_manifest_sha256": rng_sha256,
        "consumed_residue_tokens": 262_400,
        "model_family_sha256": "a" * 64,
        "next_update": 2,
        "run_seed": 17,
        "sample_cursor": 2048,
        "target_residue_tokens": 131_072,
        "training_protocol_sha256": "b" * 64,
        "training_universe_count": 224_887_833,
        "world_size": world_size,
    }


def _write_rng_checkpoint(root: Path, world_size: int) -> str:
    rng = root / "rng"
    rng.mkdir(parents=True)
    files = []
    for rank in range(world_size):
        state = {
            "format_version": "antibody_rank_rng_state_v1",
            "rank": rank,
            "torch_cpu_state_hex": "00",
            "torch_cuda_state_hex": "01",
        }
        state["state_sha256"] = hashlib.sha256(_canonical(state)).hexdigest()
        path = rng / f"rank-{rank:05d}.json"
        path.write_bytes(_canonical(state))
        files.append(
            {
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "path": f"rng/{path.name}",
                "rank": rank,
                "state_sha256": state["state_sha256"],
            }
        )
    manifest = {
        "files": files,
        "format_version": "antibody_rng_checkpoint_manifest_v1",
        "world_size": world_size,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    (root / "rng_manifest.json").write_bytes(_canonical(manifest))
    return str(manifest["manifest_sha256"])


def test_semantic_state_ignores_only_topology_and_rank_rng() -> None:
    first = _trainer_state(4, "c" * 64)
    second = _trainer_state(2, "d" * 64)
    assert semantic_trainer_state_sha256(first) == semantic_trainer_state_sha256(
        second
    )
    second["sample_cursor"] = 2049
    assert semantic_trainer_state_sha256(first) != semantic_trainer_state_sha256(
        second
    )


def test_rng_manifest_is_fully_authenticated_without_rank_restore(
    tmp_path: Path,
) -> None:
    expected = _write_rng_checkpoint(tmp_path, 4)
    result = authenticate_source_rng_manifest(
        tmp_path,
        expected_manifest_sha256=expected,
    )
    assert result["source_world_size"] == 4
    assert len(result["source_rank_state_sha256"]) == 4
    state = tmp_path / "rng/rank-00002.json"
    state.write_bytes(state.read_bytes() + b" ")
    with pytest.raises(ElasticResumeError, match="file hash"):
        authenticate_source_rng_manifest(
            tmp_path,
            expected_manifest_sha256=expected,
        )


def test_destination_rank_seed_is_stable_and_topology_bound() -> None:
    seed = destination_rank_seed(run_seed=17, next_update=2, rank=0, world_size=2)
    assert seed == destination_rank_seed(
        run_seed=17,
        next_update=2,
        rank=0,
        world_size=2,
    )
    assert seed != destination_rank_seed(
        run_seed=17,
        next_update=2,
        rank=1,
        world_size=2,
    )
    assert seed != destination_rank_seed(
        run_seed=17,
        next_update=2,
        rank=0,
        world_size=4,
    )


def test_semantic_optimizer_state_uses_parameter_names() -> None:
    first_model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.LayerNorm(4),
    )
    first_optimizer = torch.optim.AdamW(first_model.parameters(), lr=1.0e-3)
    first_model(torch.ones(2, 3)).sum().backward()
    first_optimizer.step()
    first = semantic_optimizer_state(first_model, first_optimizer)

    second_model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.LayerNorm(4),
    )
    second_optimizer = torch.optim.AdamW(second_model.parameters(), lr=1.0e-3)
    second_model.load_state_dict(first_model.state_dict())
    second_optimizer.load_state_dict(first_optimizer.state_dict())
    second = semantic_optimizer_state(second_model, second_optimizer)

    assert first["param_groups"][0]["parameters"] == [
        "0.weight",
        "0.bias",
        "1.weight",
        "1.bias",
    ]
    for name in first["state"]:
        for key in first["state"][name]:
            assert torch.equal(first["state"][name][key], second["state"][name][key])


def test_elastic_randomness_contract_requires_stateless_training() -> None:
    family = {
        "architecture": {
            "attention_dropout": 0.0,
            "residual_dropout": 0.0,
        }
    }
    validate_randomness_contract(
        family,
        masking_implementation="counter_derived_by_sample_key_update_and_seed",
    )
    family["architecture"]["attention_dropout"] = 0.1
    with pytest.raises(ElasticResumeError, match="zero dropout"):
        validate_randomness_contract(
            family,
            masking_implementation="counter_derived_by_sample_key_update_and_seed",
        )
