from __future__ import annotations

import json
from pathlib import Path
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


def test_replicate_shard_mesh_shape_supports_four_and_eight_gpu_runs() -> None:
    assert runner._replicate_shard_mesh_shape(4, 4) == (1, 4)
    assert runner._replicate_shard_mesh_shape(8, 4) == (2, 4)
    with pytest.raises(ValueError, match="divide"):
        runner._replicate_shard_mesh_shape(6, 4)


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


def test_v7_protocol_freezes_alternating_dynamic_padding_efficiency_gate(
    tmp_path,
) -> None:
    protocol = json.loads(
        (
            runner.REPOSITORY_ROOT
            / "protocols"
            / "abpairfm_stage_a1_multitask_pilot_v6.json"
        ).read_text(encoding="utf-8")
    )
    receipt_path = Path(protocol["counterfactual_gate"]["receipt_path"])
    receipt = {
        "materialization": {
            "coverage": 0.99675,
            "mapping_sha256": protocol["counterfactual_gate"]["mapping_sha256"],
        },
        "status": "PASS_TRAIN_ONLY_MATERIALIZATION",
        "test_access": False,
    }
    local_receipt = tmp_path / "counterfactual.json"
    local_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    assert receipt_path != local_receipt
    protocol["counterfactual_gate"]["receipt_path"] = str(local_receipt)
    protocol["counterfactual_gate"]["receipt_sha256"] = runner._sha256(
        local_receipt
    )
    protocol["format_version"] = "abpairfm_stage_a1_efficiency_gate_v7"
    protocol["efficiency_gate_only"] = True
    protocol["efficiency"] = {
        "expected_bucket_records": 10380,
        "gpu_peak_bfloat16_flops": 82580000000000,
        "maximum_combined_tokens": 64,
        "padding_multiple": 64,
        "parameter_count": 2839005633,
        "phase_schedule": "alternating_joint_and_retrieval_single_backbone_pass",
        "selection_seed": 1701,
    }
    protocol["optimization"]["accumulation_steps"] = 1
    protocol["optimization"].pop("microbatch_pairs_per_rank")
    protocol["optimization"]["microbatch_pairs_per_rank_candidates"] = [64, 96, 128]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    arguments = SimpleNamespace(
        protocol=path,
        expected_protocol_sha256=runner._sha256(path),
        expected_world_size=8,
        efficiency_microbatch_pairs_per_rank=96,
        input_generation=protocol["checkpoint"]["generation"],
        expected_input_content_sha256=protocol["checkpoint"]["content_sha256"],
    )
    loaded = runner._load_protocol(arguments)
    assert loaded["efficiency"]["padding_multiple"] == 64

    protocol["format_version"] = "abpairfm_stage_a1_efficiency_ac_gate_v8"
    protocol["efficiency"]["activation_checkpointing_candidates"] = [
        "selective_attention",
        "off",
    ]
    protocol["optimization"]["microbatch_pairs_per_rank_candidates"] = [64]
    path.write_text(json.dumps(protocol), encoding="utf-8")
    arguments.expected_protocol_sha256 = runner._sha256(path)
    arguments.efficiency_microbatch_pairs_per_rank = 64
    arguments.efficiency_activation_checkpoint_mode = "off"
    assert runner._load_protocol(arguments)["format_version"].endswith("_v8")

    protocol["format_version"] = (
        "abpairfm_stage_a1_representative_efficiency_gate_v9"
    )
    protocol["efficiency"].pop("expected_bucket_records")
    protocol["efficiency"]["expected_source_records"] = 19200
    protocol["efficiency"]["maximum_combined_tokens"] = 1024
    protocol["efficiency"]["activation_checkpointing_candidates"] = [
        "full",
        "off",
    ]
    protocol["efficiency"]["trial_matrix"] = [
        {
            "activation_checkpointing": "full",
            "microbatch_pairs_per_rank": 6,
        },
        {
            "activation_checkpointing": "full",
            "microbatch_pairs_per_rank": 7,
        },
        {
            "activation_checkpointing": "full",
            "microbatch_pairs_per_rank": 8,
        },
        {
            "activation_checkpointing": "off",
            "microbatch_pairs_per_rank": 6,
        },
    ]
    protocol["optimization"]["microbatch_pairs_per_rank_candidates"] = [6, 7, 8]
    path.write_text(json.dumps(protocol), encoding="utf-8")
    arguments.expected_protocol_sha256 = runner._sha256(path)
    arguments.efficiency_microbatch_pairs_per_rank = 7
    arguments.efficiency_activation_checkpoint_mode = "full"
    assert runner._load_protocol(arguments)["format_version"].endswith("_v9")

    protocol["format_version"] = "abpairfm_stage_a1_length_bucket_efficiency_gate_v10"
    protocol["efficiency"]["activation_checkpointing_candidates"] = ["full"]
    protocol["efficiency"]["length_bucket_block_microbatches"] = 16
    protocol["efficiency"]["ppiref_records_per_group"] = 5
    protocol["efficiency"]["selection_seed"] = 1701
    protocol["efficiency"]["string_records_per_group"] = 1
    protocol["efficiency"]["trial_matrix"] = [
        {
            "activation_checkpointing": "full",
            "microbatch_pairs_per_rank": microbatch,
        }
        for microbatch in (6, 7, 8)
    ]
    path.write_text(json.dumps(protocol), encoding="utf-8")
    arguments.expected_protocol_sha256 = runner._sha256(path)
    assert runner._load_protocol(arguments)["format_version"].endswith("_v10")

    protocol["format_version"] = (
        "abpairfm_stage_a1_structural_bucket_efficiency_gate_v11"
    )
    for field in (
        "length_bucket_block_microbatches",
        "ppiref_records_per_group",
        "string_records_per_group",
    ):
        protocol["efficiency"].pop(field)
    protocol["efficiency"]["expected_structural_records"] = 16000
    protocol["efficiency"]["length_bucket_block_records"] = 128
    path.write_text(json.dumps(protocol), encoding="utf-8")
    arguments.expected_protocol_sha256 = runner._sha256(path)
    assert runner._load_protocol(arguments)["format_version"].endswith("_v11")

    protocol["format_version"] = "abpairfm_stage_a1_length_band_efficiency_gate_v13"
    protocol["efficiency"].pop("expected_structural_records")
    protocol["efficiency"].pop("length_bucket_block_records")
    protocol["efficiency"]["expected_bucket_records"] = 4967
    protocol["efficiency"]["minimum_combined_tokens"] = 65
    protocol["efficiency"]["maximum_combined_tokens"] = 128
    protocol["efficiency"]["trial_matrix"] = [
        {
            "activation_checkpointing": "full",
            "microbatch_pairs_per_rank": microbatch,
        }
        for microbatch in (24, 32, 48)
    ]
    protocol["optimization"]["microbatch_pairs_per_rank_candidates"] = [24, 32, 48]
    path.write_text(json.dumps(protocol), encoding="utf-8")
    arguments.expected_protocol_sha256 = runner._sha256(path)
    arguments.efficiency_microbatch_pairs_per_rank = 32
    assert runner._load_protocol(arguments)["format_version"].endswith("_v13")

    protocol["format_version"] = "abpairfm_stage_a1_length_band_promotion_v14"
    protocol["authorization"]["pilot_training"] = False
    protocol["authorization"]["production_training"] = True
    protocol["checkpoint"]["generation"] = "segment-000031"
    protocol["checkpoint"]["content_sha256"] = "7" * 64
    protocol["efficiency_gate_only"] = False
    protocol["systems_gate_only"] = False
    protocol["optimization"]["updates"] = 200
    protocol["promotion_selection"] = {
        "source_gate": "abpairfm_stage_a1_length_band_efficiency_gate_v13",
        "rule": "highest_mean_mfu_among_pass_then_highest_throughput",
        "required_status": "PASS_STAGE_A1_EFFICIENCY_GATE_CHECKPOINT_VERIFIED",
    }
    protocol["status"] = "FROZEN_BEFORE_STAGE_A1_PROMOTION"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    arguments.expected_protocol_sha256 = runner._sha256(path)
    arguments.input_generation = "segment-000031"
    arguments.expected_input_content_sha256 = "7" * 64
    assert runner._load_protocol(arguments)["format_version"].endswith("_v14")


def test_efficiency_bucket_is_deterministic_and_checks_capacity() -> None:
    records = []
    for index in range(8):
        value = _record(index % 4)
        value["example_id"] = f"record-{index}"
        value["pilot_record_index"] = index
        value["source"] = "ppiref50k"
        records.append(value)
    protocol = {
        "efficiency": {
            "expected_bucket_records": 8,
            "maximum_combined_tokens": 64,
            "selection_seed": 1701,
        },
        "format_version": "abpairfm_stage_a1_efficiency_gate_v7",
        "optimization": {"microbatch_pairs_per_rank": 2, "updates": 2},
        "systems": {"world_size": 2},
    }
    first = runner._efficiency_bucket_records(
        records, protocol, microbatch_pairs_per_rank=2
    )
    second = runner._efficiency_bucket_records(
        list(reversed(records)), protocol, microbatch_pairs_per_rank=2
    )
    assert [value["pilot_record_index"] for value in first] == [
        value["pilot_record_index"] for value in second
    ]

    representative_protocol = {
        "efficiency": {"expected_source_records": len(records)},
        "format_version": "abpairfm_stage_a1_representative_efficiency_gate_v9",
        "optimization": {"updates": 1},
        "systems": {"world_size": 2},
    }
    assert runner._efficiency_bucket_records(
        records, representative_protocol, microbatch_pairs_per_rank=2
    ) == records


def test_length_band_efficiency_bucket_applies_both_bounds() -> None:
    records = []
    for index, combined_length in enumerate((64, 65, 96, 128, 129)):
        record = _record(index)
        record["example_id"] = f"record-{index}"
        record["pilot_record_index"] = index
        record["sequence1"] = "A" * 20
        record["sequence2"] = "C" * (combined_length - 24)
        record["source"] = "ppiref50k"
        records.append(record)
    protocol = {
        "efficiency": {
            "expected_bucket_records": 3,
            "minimum_combined_tokens": 65,
            "maximum_combined_tokens": 128,
            "selection_seed": 1701,
        },
        "format_version": "abpairfm_stage_a1_length_band_efficiency_gate_v13",
        "optimization": {"updates": 1},
        "systems": {"world_size": 1},
    }
    selected = runner._efficiency_bucket_records(
        records, protocol, microbatch_pairs_per_rank=2
    )
    lengths = sorted(
        len(record["sequence1"]) + len(record["sequence2"]) + 4
        for record in selected
    )
    assert lengths == [65, 96, 128]

    protocol["format_version"] = "abpairfm_stage_a1_length_band_promotion_v14"
    protocol["optimization"]["updates"] = 2
    repeated = runner._efficiency_bucket_records(
        records, protocol, microbatch_pairs_per_rank=2
    )
    assert len(repeated) == 4
    assert len({record["pilot_record_index"] for record in repeated}) == 3


def test_length_bucket_preserves_membership_and_five_to_one_source_ratio() -> None:
    records = []
    for index in range(12):
        record = _record(index)
        record["example_id"] = f"record-{index}"
        record["pilot_record_index"] = index
        record["sequence1"] = "A" * (10 + index)
        record["sequence2"] = "C" * (12 + index)
        record["source"] = (
            "string_v12_physical" if index in {5, 11} else "ppiref50k"
        )
        records.append(record)
    protocol = {
        "efficiency": {
            "expected_source_records": 12,
            "length_bucket_block_microbatches": 1,
            "ppiref_records_per_group": 5,
            "selection_seed": 1701,
            "string_records_per_group": 1,
        },
        "format_version": "abpairfm_stage_a1_length_bucket_efficiency_gate_v10",
        "optimization": {"updates": 1},
        "systems": {"world_size": 1},
    }
    ordered = runner._efficiency_bucket_records(
        records, protocol, microbatch_pairs_per_rank=6
    )
    assert {record["pilot_record_index"] for record in ordered} == set(range(12))
    for start in range(0, len(ordered), 6):
        sources = [record["source"] for record in ordered[start : start + 6]]
        assert sources.count("ppiref50k") == 5
        assert sources.count("string_v12_physical") == 1


def test_structural_bucket_excludes_sequence_only_padding() -> None:
    records = []
    for index in range(12):
        record = _record(index)
        record["example_id"] = f"record-{index}"
        record["pilot_record_index"] = index
        record["sequence1"] = "A" * (10 + index)
        record["sequence2"] = "C" * (12 + index)
        if index in {5, 11}:
            record["source"] = "string_v12_physical"
            record["contact_supervision"] = False
            record["retrieval_supervision"] = False
        else:
            record["source"] = "ppiref50k"
        records.append(record)
    protocol = {
        "efficiency": {
            "expected_source_records": 12,
            "expected_structural_records": 10,
            "length_bucket_block_records": 4,
            "selection_seed": 1701,
        },
        "format_version": "abpairfm_stage_a1_structural_bucket_efficiency_gate_v11",
        "optimization": {"updates": 1},
        "systems": {"world_size": 1},
    }
    ordered = runner._efficiency_bucket_records(
        records, protocol, microbatch_pairs_per_rank=6
    )
    assert len(ordered) == 10
    assert all(record["source"] == "ppiref50k" for record in ordered)
    assert {record["pilot_record_index"] for record in ordered} == (
        set(range(12)) - {5, 11}
    )


def test_rank_phase_telemetry_can_be_disabled(tmp_path: Path) -> None:
    runner._publish_rank_phase(
        None,
        rank=0,
        update=1,
        accumulation=None,
        phase="optimizer_step_complete",
    )
    assert list(tmp_path.iterdir()) == []
