from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from antibody_pretraining.stage_a1_batch import build_stage_a1_microbatch
from scripts import run_abpairfm_stage_a1_multitask_pilot as runner


def _record(index: int) -> dict:
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    rotated = sequence[index:] + sequence[:index]
    return {
        "contact_pairs": [[0, 0], [1, 1]],
        "contact_supervision": True,
        "example_id": f"record-{index}",
        "known_positive_group": f"group-{index}",
        "retrieval_supervision": True,
        "sequence1": rotated[:9],
        "sequence2": rotated[9:18],
        "source": "test",
    }


def test_global_retrieval_mask_uses_remote_cluster_ids(monkeypatch) -> None:
    batch = build_stage_a1_microbatch(
        [_record(0), _record(1)], seed=17, maximum_tokens=64
    )
    remote_metadata = (
        batch.retrieval_left_exact_ids + 100,
        batch.retrieval_right_exact_ids + 200,
        torch.tensor(
            [
                int(batch.retrieval_left_cluster_ids[0]),
                int(batch.retrieval_left_cluster_ids[1]) + 300,
            ]
        ),
        batch.retrieval_right_cluster_ids + 400,
        batch.retrieval_positive_group_ids + 500,
    )
    responses = iter(
        [
            torch.tensor([2, 2]),
            *[
                torch.cat([local, remote])
                for local, remote in zip(
                    (
                        batch.retrieval_left_exact_ids,
                        batch.retrieval_right_exact_ids,
                        batch.retrieval_left_cluster_ids,
                        batch.retrieval_right_cluster_ids,
                        batch.retrieval_positive_group_ids,
                    ),
                    remote_metadata,
                )
            ],
        ]
    )
    monkeypatch.setattr(
        runner,
        "_all_gather_equal_shape",
        lambda _tensor, _world_size: next(responses),
    )
    allowed, targets = runner._global_retrieval_mask(batch, rank=0, world_size=2)
    assert allowed.shape == (2, 4)
    assert torch.equal(targets, torch.tensor([0, 1]))
    assert not bool(allowed[0, 2])
    assert bool(allowed[0, 3])
    assert bool(allowed[0, 0])
    assert bool(allowed[1, 1])


def test_global_candidate_gather_detaches_remote_negatives(monkeypatch) -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    chain_ids = torch.tensor([[0, 1, 1], [0, 0, 1]])
    heads = runner.StageA1Heads(embedding_dimension=4, projection_dimension=2)

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    calls = []

    def fake_all_gather(outputs, local) -> None:
        calls.append(local)
        outputs[0].copy_(torch.full_like(local, 0.25))
        outputs[1].copy_(local)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    left, right = runner._global_retrieval_candidates(heads, hidden, chain_ids)
    assert len(calls) == 1
    assert left.shape == (4, 2)
    assert right.shape == (4, 2)
    assert torch.equal(left[:2], torch.full_like(left[:2], 0.25))
    assert torch.equal(right[:2], torch.full_like(right[:2], 0.25))
    (left.sum() + right.sum()).backward()
    retrieval_parameters = [
        parameter
        for name, parameter in heads.named_parameters()
        if name.startswith("retrieval_")
    ]
    assert len(retrieval_parameters) == 2
    assert all(parameter.grad is not None for parameter in retrieval_parameters)


def test_candidate_minimum_uses_supported_non_inplace_torch_api() -> None:
    current = torch.tensor(float("inf"), dtype=torch.float64)
    first = torch.tensor(
        [[True, True, False], [True, True, True]], dtype=torch.bool
    )
    second = torch.tensor(
        [[True, True, True, True], [True, True, True, False]], dtype=torch.bool
    )
    current = runner._update_minimum_allowed_candidates(current, first)
    current = runner._update_minimum_allowed_candidates(current, second)
    assert current.dtype == torch.float64
    assert current.item() == 2.0


def test_rank_phase_receipt_is_atomic_and_test_sealed(tmp_path) -> None:
    runner._publish_rank_phase(
        tmp_path,
        rank=6,
        update=2,
        accumulation=1,
        phase="forward_and_loss_complete",
    )
    receipt = json.loads((tmp_path / "rank-6.json").read_text(encoding="utf-8"))
    assert receipt["format_version"] == "abpairfm_stage_a1_rank_phase_v1"
    assert receipt["rank"] == 6
    assert receipt["update"] == 2
    assert receipt["accumulation"] == 1
    assert receipt["phase"] == "forward_and_loss_complete"
    assert receipt["test_access"] is False


def test_runner_requests_independent_hidden_only_retrieval_view() -> None:
    batch = build_stage_a1_microbatch(
        [_record(0), _record(1)], seed=17, maximum_tokens=64
    )
    calls = []

    class FakeBackbone:
        def __call__(self, tokens, chain_ids, **kwargs):
            calls.append((tokens.clone(), chain_ids.clone(), dict(kwargs)))
            hidden = tokens.to(torch.float32).unsqueeze(-1).expand(-1, -1, 4)
            return SimpleNamespace(
                balanced_chain_loss_sum=(
                    torch.tensor(2.0) if kwargs.get("labels") is not None else None
                ),
                contributing_chain_count=torch.tensor(2),
                hidden_states=hidden,
            )

    _, joint_hidden, retrieval_hidden, retrieval_chain_ids = (
        runner._stage_a1_backbone_views(
            FakeBackbone(),
            batch,
            independent_retrieval_view=True,
        )
    )
    assert len(calls) == 2
    assert calls[0][2]["return_hidden_states"] is True
    assert calls[1][2] == {
        "allow_cross_chain_attention": False,
        "return_hidden_states": True,
        "compute_logits": False,
    }
    assert calls[1][0].shape[0] == len(batch.retrieval_batch_indices)
    assert joint_hidden.shape == retrieval_hidden.shape
    assert torch.equal(retrieval_chain_ids, calls[1][1])


def test_v5_protocol_requires_counterfactual_and_representation_gates(
    tmp_path,
) -> None:
    protocol = json.loads(
        (
            runner.REPOSITORY_ROOT
            / "protocols"
            / "abpairfm_stage_a1_multitask_pilot_v4.json"
        ).read_text(encoding="utf-8")
    )
    protocol["format_version"] = "abpairfm_stage_a1_multitask_pilot_v5"
    protocol["optimization"]["updates"] = 10
    protocol["systems_gate_only"] = True
    protocol["loss"]["masked_language_modeling"]["representation_view"] = (
        "joint_cross_chain_attention"
    )
    protocol["loss"]["contact"]["representation_view"] = (
        "joint_cross_chain_attention"
    )
    protocol["loss"]["retrieval"]["representation_view"] = (
        "chain_independent_no_cross_attention"
    )
    receipt = {
        "materialization": {
            "coverage": 0.99675,
            "mapping_sha256": "a" * 64,
        },
        "status": "PASS_TRAIN_ONLY_MATERIALIZATION",
        "test_access": False,
    }
    receipt_path = tmp_path / "counterfactual.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    protocol["counterfactual_gate"] = {
        "mapping_sha256": "a" * 64,
        "minimum_coverage": 0.995,
        "receipt_path": str(receipt_path),
        "receipt_sha256": runner._sha256(receipt_path),
        "status": "PASS_TRAIN_ONLY_MATERIALIZATION",
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    arguments = SimpleNamespace(
        protocol=path,
        expected_protocol_sha256=runner._sha256(path),
        expected_world_size=8,
        input_generation=protocol["checkpoint"]["generation"],
        expected_input_content_sha256=protocol["checkpoint"]["content_sha256"],
    )
    assert runner._load_protocol(arguments)["format_version"].endswith("_v5")

    protocol["format_version"] = "abpairfm_stage_a1_multitask_pilot_v6"
    protocol["pilot_manifest"]["source_protocol_sha256"] = "b" * 64
    protocol["pilot_manifest"]["manifest_sha256"] = "c" * 64
    protocol["pilot_manifest"]["receipt_sha256"] = "d" * 64
    path.write_text(json.dumps(protocol), encoding="utf-8")
    arguments.expected_protocol_sha256 = runner._sha256(path)
    assert runner._load_protocol(arguments)["format_version"].endswith("_v6")

    protocol["counterfactual_gate"]["status"] = "NOT_RUN"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    arguments.expected_protocol_sha256 = runner._sha256(path)
    with pytest.raises(RuntimeError, match="representation-view gate"):
        runner._load_protocol(arguments)
