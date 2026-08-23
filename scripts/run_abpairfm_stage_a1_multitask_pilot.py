#!/usr/bin/env python3
"""Run the frozen 200-update ABPairFM Stage-A1 multitask pilot."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time
from typing import Any, Mapping

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SCRIPT_ROOT = REPOSITORY_ROOT / "scripts"
for path in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from antibody_pretraining.pair_aware_distributed import (  # noqa: E402
    fully_shard_pair_aware_esm2,
    load_authenticated_esm2_into_fsdp2,
    reshard_pair_aware_esm2,
)
from antibody_pretraining.pair_aware_esm2 import (  # noqa: E402
    PairAwareESM2,
    PairAwareESM2Config,
)
from antibody_pretraining.stage_a1_batch import (  # noqa: E402
    StageA1Microbatch,
    build_stage_a1_microbatch,
    build_stage_a1_retrieval_allowed,
)
from antibody_pretraining.stage_a1_objective import (  # noqa: E402
    StageA1Heads,
    StageA1Loss,
)
from interaction_first.frontier_fsdp2 import verify_dcp_generation  # noqa: E402
from run_abpairfm_3b_real_data_gate_v2 import (  # noqa: E402
    _collect,
    _nearest_rank_p95,
    _publish_create_only,
    _publish_progress,
    _sha256,
    _zstd_lines,
)


EXPECTED_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_multitask_pilot_v3",
    "abpairfm_stage_a1_multitask_pilot_v4",
    "abpairfm_stage_a1_multitask_pilot_v5",
    "abpairfm_stage_a1_multitask_pilot_v6",
    "abpairfm_stage_a1_efficiency_gate_v7",
    "abpairfm_stage_a1_efficiency_ac_gate_v8",
    "abpairfm_stage_a1_representative_efficiency_gate_v9",
}
GLOBAL_RETRIEVAL_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_multitask_pilot_v4",
    "abpairfm_stage_a1_multitask_pilot_v5",
    "abpairfm_stage_a1_multitask_pilot_v6",
    "abpairfm_stage_a1_efficiency_gate_v7",
    "abpairfm_stage_a1_efficiency_ac_gate_v8",
    "abpairfm_stage_a1_representative_efficiency_gate_v9",
}
SPLIT_VIEW_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_multitask_pilot_v5",
    "abpairfm_stage_a1_multitask_pilot_v6",
    "abpairfm_stage_a1_efficiency_gate_v7",
    "abpairfm_stage_a1_efficiency_ac_gate_v8",
    "abpairfm_stage_a1_representative_efficiency_gate_v9",
}
ALTERNATING_EFFICIENCY_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_efficiency_gate_v7",
    "abpairfm_stage_a1_efficiency_ac_gate_v8",
    "abpairfm_stage_a1_representative_efficiency_gate_v9",
}
REPRESENTATIVE_EFFICIENCY_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_representative_efficiency_gate_v9",
}
SOURCE_PATHS = (
    "scripts/run_abpairfm_stage_a1_multitask_pilot.py",
    "scripts/run_abpairfm_3b_real_data_gate_v2.py",
    "src/antibody_pretraining/pair_aware_attention.py",
    "src/antibody_pretraining/pair_aware_distributed.py",
    "src/antibody_pretraining/pair_aware_esm2.py",
    "src/antibody_pretraining/stage_a1_batch.py",
    "src/antibody_pretraining/stage_a1_objective.py",
    "src/interaction_first/frontier_fsdp2.py",
)


class StageA1System(nn.Module):
    """Register the sharded backbone and replicated heads under one DCP root."""

    def __init__(self, backbone: PairAwareESM2, heads: StageA1Heads) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = heads


def _frontier_module() -> Any:
    path = SOURCE_ROOT / "interaction_first" / "frontier_fsdp2.py"
    specification = importlib.util.spec_from_file_location("stage_a1_dcp", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load DCP implementation: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


_DCP = _frontier_module()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-initial-checkpoint-size", type=int, required=True)
    parser.add_argument("--expected-initial-checkpoint-sha256", required=True)
    parser.add_argument("--input-checkpoint-root", type=Path, required=True)
    parser.add_argument("--input-generation", required=True)
    parser.add_argument("--expected-input-content-sha256", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--shard-group-size", type=int, default=4)
    parser.add_argument("--maximum-tokens", type=int, default=1024)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--efficiency-microbatch-pairs-per-rank", type=int)
    parser.add_argument(
        "--efficiency-activation-checkpoint-mode",
        choices=("full", "selective_attention", "off"),
        default="full",
    )
    return parser.parse_args()


def source_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Reviewed Stage-A1 source is missing: {relative}")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _load_protocol(arguments: argparse.Namespace) -> dict[str, Any]:
    if _sha256(arguments.protocol) != arguments.expected_protocol_sha256:
        raise RuntimeError("Stage-A1 protocol SHA-256 differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    authorization = protocol.get("authorization", {})
    systems = protocol.get("systems", {})
    optimization = protocol.get("optimization", {})
    format_version = protocol.get("format_version")
    expected_updates = 10 if format_version in SPLIT_VIEW_PROTOCOL_FORMATS else 200
    alternating_efficiency = format_version in ALTERNATING_EFFICIENCY_PROTOCOL_FORMATS
    selected_efficiency_microbatch = getattr(
        arguments, "efficiency_microbatch_pairs_per_rank", None
    )
    selected_activation_checkpoint_mode = getattr(
        arguments, "efficiency_activation_checkpoint_mode", "full"
    )
    if format_version == "abpairfm_stage_a1_efficiency_gate_v7":
        expected_efficiency_microbatches = [64, 96, 128]
        expected_activation_checkpoint_modes = ["full"]
    elif format_version == "abpairfm_stage_a1_efficiency_ac_gate_v8":
        expected_efficiency_microbatches = [64]
        expected_activation_checkpoint_modes = ["selective_attention", "off"]
    else:
        expected_efficiency_microbatches = [6, 7, 8]
        expected_activation_checkpoint_modes = ["full", "off"]
    selected_trial = {
        "activation_checkpointing": selected_activation_checkpoint_mode,
        "microbatch_pairs_per_rank": selected_efficiency_microbatch,
    }
    valid_shape = (
        optimization.get("updates") == expected_updates
        and (
            (
                optimization.get("microbatch_pairs_per_rank_candidates")
                == expected_efficiency_microbatches
                and selected_efficiency_microbatch
                in expected_efficiency_microbatches
                and (
                    selected_activation_checkpoint_mode == "full"
                    if format_version == "abpairfm_stage_a1_efficiency_gate_v7"
                    else (
                        protocol.get("efficiency", {}).get(
                            "activation_checkpointing_candidates"
                        )
                        == expected_activation_checkpoint_modes
                        and selected_activation_checkpoint_mode
                        in expected_activation_checkpoint_modes
                        and (
                            format_version
                            != "abpairfm_stage_a1_representative_efficiency_gate_v9"
                            or selected_trial
                            in protocol.get("efficiency", {}).get("trial_matrix", [])
                        )
                    )
                )
                and optimization.get("accumulation_steps") == 1
            )
            if alternating_efficiency
            else (
                optimization.get("microbatch_pairs_per_rank") in {6, 7}
                and optimization.get("accumulation_steps") == 2
            )
        )
    )
    if (
        format_version not in EXPECTED_PROTOCOL_FORMATS
        or protocol.get("status") != "FROZEN_BEFORE_PILOT_MANIFEST_AND_GPU_TRAINING"
        or authorization.get("pilot_training") is not True
        or authorization.get("production_training") is not False
        or authorization.get("test_access") is not False
        or protocol.get("test_access") is not False
        or systems.get("partition") != "gpu_4090"
        or systems.get("world_size") != arguments.expected_world_size
        or not valid_shape
    ):
        raise RuntimeError("Stage-A1 pilot authorization or shape differs")
    if format_version in GLOBAL_RETRIEVAL_PROTOCOL_FORMATS and (
        (not alternating_efficiency and optimization.get("microbatch_pairs_per_rank") != 6)
        or protocol.get("loss", {}).get("retrieval", {}).get("candidate_scope")
        != "same_node_all_ranks"
        or protocol.get("pilot_manifest", {}).get("string_retrieval_supervision")
        is not False
    ):
        raise RuntimeError("Stage-A1 global-retrieval gate differs")
    if format_version in SPLIT_VIEW_PROTOCOL_FORMATS and (
        protocol.get("loss", {}).get("retrieval", {}).get("representation_view")
        != "chain_independent_no_cross_attention"
        or protocol.get("loss", {}).get("contact", {}).get("representation_view")
        != "joint_cross_chain_attention"
        or protocol.get("loss", {})
        .get("masked_language_modeling", {})
        .get("representation_view")
        != "joint_cross_chain_attention"
        or protocol.get("systems_gate_only") is not True
        or protocol.get("counterfactual_gate", {}).get("status")
        != "PASS_TRAIN_ONLY_MATERIALIZATION"
    ):
        raise RuntimeError("Stage-A1 v5 representation-view gate differs")
    if alternating_efficiency:
        efficiency = protocol.get("efficiency", {})
        common_efficiency_valid = (
            protocol.get("efficiency_gate_only") is True
            and protocol.get("systems_gate_only") is True
            and efficiency.get("padding_multiple") == 64
            and efficiency.get("parameter_count") == 2839005633
            and efficiency.get("gpu_peak_bfloat16_flops") == 82580000000000
            and efficiency.get("phase_schedule")
            == "alternating_joint_and_retrieval_single_backbone_pass"
        )
        representative_valid = (
            efficiency.get("maximum_combined_tokens") == 1024
            and efficiency.get("expected_source_records") == 19200
        )
        short_bucket_valid = (
            efficiency.get("maximum_combined_tokens") == 64
            and efficiency.get("expected_bucket_records") == 10380
        )
        selected_efficiency_shape_valid = (
            representative_valid
            if format_version in REPRESENTATIVE_EFFICIENCY_PROTOCOL_FORMATS
            else short_bucket_valid
        )
        if not common_efficiency_valid or not selected_efficiency_shape_valid:
            raise RuntimeError("Stage-A1 efficiency gate differs")
    if format_version in SPLIT_VIEW_PROTOCOL_FORMATS:
        gate = protocol["counterfactual_gate"]
        receipt_path = Path(str(gate.get("receipt_path", "")))
        if (
            receipt_path.is_symlink()
            or not receipt_path.is_file()
            or _sha256(receipt_path) != gate.get("receipt_sha256")
        ):
            raise RuntimeError("Stage-A1 v5 counterfactual receipt authentication failed")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "PASS_TRAIN_ONLY_MATERIALIZATION"
            or receipt.get("test_access") is not False
            or receipt.get("materialization", {}).get("coverage")
            < gate.get("minimum_coverage", 1.0)
            or receipt.get("materialization", {}).get("mapping_sha256")
            != gate.get("mapping_sha256")
        ):
            raise RuntimeError("Stage-A1 v5 counterfactual receipt differs")
    if format_version in {
        "abpairfm_stage_a1_multitask_pilot_v6",
        "abpairfm_stage_a1_efficiency_gate_v7",
        "abpairfm_stage_a1_efficiency_ac_gate_v8",
        "abpairfm_stage_a1_representative_efficiency_gate_v9",
    } and (
        len(protocol.get("pilot_manifest", {}).get("source_protocol_sha256", ""))
        != 64
        or len(protocol.get("pilot_manifest", {}).get("manifest_sha256", ""))
        != 64
        or len(protocol.get("pilot_manifest", {}).get("receipt_sha256", ""))
        != 64
    ):
        raise RuntimeError("Stage-A1 v6 manifest anchors differ")
    if protocol["checkpoint"]["generation"] != arguments.input_generation:
        raise RuntimeError("Stage-A1 input generation differs from protocol")
    if (
        protocol["checkpoint"]["content_sha256"]
        != arguments.expected_input_content_sha256
    ):
        raise RuntimeError("Stage-A1 input checkpoint anchor differs from protocol")
    return protocol


def _load_manifest(
    arguments: argparse.Namespace, protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str, str]:
    receipt = json.loads(arguments.manifest_receipt.read_text(encoding="utf-8"))
    manifest_sha256 = _sha256(arguments.manifest)
    receipt_sha256 = _sha256(arguments.manifest_receipt)
    if protocol.get("format_version") in {
        "abpairfm_stage_a1_multitask_pilot_v6",
        "abpairfm_stage_a1_efficiency_gate_v7",
        "abpairfm_stage_a1_efficiency_ac_gate_v8",
    } and (
        manifest_sha256 != protocol["pilot_manifest"]["manifest_sha256"]
        or receipt_sha256 != protocol["pilot_manifest"]["receipt_sha256"]
    ):
        raise RuntimeError("Stage-A1 v6 manifest files differ from frozen anchors")
    if (
        receipt.get("status") != "PASS_STAGE_A1_PILOT_SOURCE_ADMISSION"
        or receipt.get("test_access") is not False
        or receipt.get("inputs", {}).get("protocol_sha256")
        != protocol.get("pilot_manifest", {}).get(
            "source_protocol_sha256", arguments.expected_protocol_sha256
        )
        or receipt.get("output", {}).get("sha256") != manifest_sha256
        or receipt.get("counts", {}).get("records")
        != protocol["pilot_manifest"]["total_records"]
        or receipt.get("cross_source_exact_pair_overlap_after_filter") != 0
    ):
        raise RuntimeError("Stage-A1 manifest receipt differs")
    if protocol.get("format_version") in GLOBAL_RETRIEVAL_PROTOCOL_FORMATS and (
        receipt.get("counts", {}).get("retrieval_supervised_records")
        != protocol["pilot_manifest"]["expected_retrieval_supervised_records"]
        or receipt.get("counts", {}).get(
            "retrieval_records_with_endpoint_cluster_ids"
        )
        != protocol["pilot_manifest"]["expected_retrieval_supervised_records"]
    ):
        raise RuntimeError("Stage-A1 v4 retrieval-cluster coverage differs")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(_zstd_lines(arguments.manifest), start=1):
        value = json.loads(line)
        if (
            value.get("pilot_record_index") != line_number - 1
            or value.get("source") not in {"ppiref50k", "string_v12_physical"}
            or value.get("contact_supervision")
            != (value.get("source") == "ppiref50k")
            or not isinstance(value.get("retrieval_supervision"), bool)
            or min(len(value.get("sequence1", "")), len(value.get("sequence2", "")))
            < 3
            or (
                protocol.get("format_version") in GLOBAL_RETRIEVAL_PROTOCOL_FORMATS
                and (
                    not value.get("endpoint_cluster1")
                    or not value.get("endpoint_cluster2")
                    or not value.get("endpoint_exact1_sha256")
                    or not value.get("endpoint_exact2_sha256")
                    or value.get("retrieval_supervision")
                    != (value.get("source") == "ppiref50k")
                )
            )
        ):
            raise RuntimeError("Stage-A1 manifest record differs")
        records.append(value)
    if len(records) != protocol["pilot_manifest"]["total_records"]:
        raise RuntimeError("Stage-A1 manifest count differs")
    return records, manifest_sha256, receipt_sha256


def _efficiency_bucket_records(
    records: list[dict[str, Any]],
    protocol: Mapping[str, Any],
    *,
    microbatch_pairs_per_rank: int,
) -> list[dict[str, Any]]:
    """Select one immutable short-structure bucket for comparable MFU trials."""

    format_version = protocol.get("format_version")
    if format_version not in ALTERNATING_EFFICIENCY_PROTOCOL_FORMATS:
        return records
    if format_version in REPRESENTATIVE_EFFICIENCY_PROTOCOL_FORMATS:
        if len(records) != protocol["efficiency"]["expected_source_records"]:
            raise RuntimeError("Stage-A1 representative source count differs")
        required = (
            protocol["optimization"]["updates"]
            * protocol["systems"]["world_size"]
            * microbatch_pairs_per_rank
        )
        if len(records) < required:
            raise RuntimeError("Stage-A1 representative source is too small")
        return records
    efficiency = protocol["efficiency"]
    maximum_combined_tokens = int(efficiency["maximum_combined_tokens"])
    selected = [
        record
        for record in records
        if record.get("source") == "ppiref50k"
        and record.get("contact_supervision") is True
        and record.get("retrieval_supervision") is True
        and len(str(record["sequence1"]))
        + len(str(record["sequence2"]))
        + 4
        <= maximum_combined_tokens
    ]
    if len(selected) != efficiency["expected_bucket_records"]:
        raise RuntimeError("Stage-A1 efficiency bucket count differs")
    selected.sort(
        key=lambda record: hashlib.sha256(
            (
                f"{efficiency['selection_seed']}\0"
                f"{record['pilot_record_index']}\0{record['example_id']}"
            ).encode("utf-8")
        ).digest()
    )
    required = (
        protocol["optimization"]["updates"]
        * protocol["systems"]["world_size"]
        * microbatch_pairs_per_rank
    )
    if len(selected) < required:
        raise RuntimeError("Stage-A1 efficiency bucket is too small for the frozen trial")
    return selected


def _verify_input_dcp_collectively(
    arguments: argparse.Namespace, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rank = torch.distributed.get_rank()
    payload: list[dict[str, Any]] = [{"error": None, "receipt": None}]
    if rank == 0:
        try:
            receipt = verify_dcp_generation(
                arguments.input_checkpoint_root, arguments.input_generation
            )
            if receipt.get("content_sha256") != arguments.expected_input_content_sha256:
                raise RuntimeError("Stage-A1 immutable DCP content SHA-256 differs")
            if (
                receipt.get("provenance_hashes", {}).get("initial_checkpoint")
                != arguments.expected_initial_checkpoint_sha256
            ):
                raise RuntimeError("Stage-A1 DCP initial-checkpoint provenance differs")
            consumed = receipt.get("training_state", {}).get(
                "consumed_residue_tokens"
            )
            if (
                not isinstance(consumed, int)
                or consumed < protocol["checkpoint"]["minimum_consumed_residue_tokens"]
                or receipt.get("training_state", {}).get("test_access") is not False
            ):
                raise RuntimeError("Stage-A1 input DCP training cursor differs")
            payload[0]["receipt"] = receipt
        except BaseException as error:
            payload[0]["error"] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0]["error"] is not None:
        raise RuntimeError(f"Stage-A1 DCP verification failed: {payload[0]['error']}")
    receipt = payload[0]["receipt"]
    if not isinstance(receipt, dict):
        raise RuntimeError("Stage-A1 DCP receipt was not broadcast")
    return receipt


def _load_model_only_dcp(
    model: PairAwareESM2, *, root: Path, generation: str
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
    state = get_model_state_dict(model, options=options)
    dcp.load(
        {"model": state},
        checkpoint_id=root.resolve() / generation,
        no_dist=False,
    )
    incompatible = set_model_state_dict(model, state, options=options)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Stage-A1 model-only DCP restore returned incompatible keys")
    torch.distributed.barrier()


def _to_device(batch: StageA1Microbatch, device: torch.device) -> StageA1Microbatch:
    values: dict[str, Any] = {}
    for field in batch.__dataclass_fields__:
        value = getattr(batch, field)
        values[field] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return StageA1Microbatch(**values)


def _all_gather_equal_shape(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def _global_retrieval_mask(
    batch: StageA1Microbatch,
    *,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_count = torch.tensor(
        [len(batch.retrieval_batch_indices)],
        device=batch.retrieval_batch_indices.device,
        dtype=torch.long,
    )
    counts = _all_gather_equal_shape(local_count, world_size)
    if not bool(counts.eq(local_count[0]).all()):
        raise RuntimeError("Global retrieval requires equal candidate counts per rank")

    local_metadata = (
        batch.retrieval_left_exact_ids,
        batch.retrieval_right_exact_ids,
        batch.retrieval_left_cluster_ids,
        batch.retrieval_right_cluster_ids,
        batch.retrieval_positive_group_ids,
    )
    global_metadata = tuple(
        _all_gather_equal_shape(value, world_size) for value in local_metadata
    )
    targets = rank * int(local_count[0]) + torch.arange(
        int(local_count[0]), device=batch.retrieval_batch_indices.device
    )
    allowed = build_stage_a1_retrieval_allowed(
        local_metadata, global_metadata, targets
    )
    return allowed, targets


def _global_retrieval_candidates(
    heads: StageA1Heads,
    hidden: torch.Tensor,
    chain_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.distributed.nn.functional import all_gather

    left, right = heads.retrieval_embeddings(hidden, chain_ids)
    return torch.cat(all_gather(left), dim=0), torch.cat(all_gather(right), dim=0)


def _stage_a1_backbone_views(
    backbone: PairAwareESM2,
    batch: StageA1Microbatch,
    *,
    independent_retrieval_view: bool,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    output = backbone(
        batch.tokens,
        batch.chain_ids,
        labels=batch.labels,
        return_hidden_states=True,
    )
    if output.balanced_chain_loss_sum is None or output.hidden_states is None:
        raise RuntimeError("Stage-A1 backbone did not expose MLM and hidden states")
    retrieval_indices = batch.retrieval_batch_indices
    joint_hidden = output.hidden_states[retrieval_indices]
    retrieval_chain_ids = batch.chain_ids[retrieval_indices]
    retrieval_hidden = joint_hidden
    if independent_retrieval_view:
        independent_output = backbone(
            batch.tokens[retrieval_indices],
            retrieval_chain_ids,
            allow_cross_chain_attention=False,
            return_hidden_states=True,
            compute_logits=False,
        )
        if independent_output.hidden_states is None:
            raise RuntimeError(
                "Stage-A1 independent retrieval hidden states are missing"
            )
        retrieval_hidden = independent_output.hidden_states
    return output, joint_hidden, retrieval_hidden, retrieval_chain_ids


def _update_minimum_allowed_candidates(
    current: torch.Tensor, allowed: torch.Tensor
) -> torch.Tensor:
    if current.ndim != 0 or allowed.ndim != 2 or allowed.dtype is not torch.bool:
        raise ValueError("Stage-A1 candidate minimum inputs are malformed")
    observed = allowed.sum(dim=1).min().to(device=current.device, dtype=current.dtype)
    return torch.minimum(current, observed)


def _publish_rank_phase(
    root: Path,
    *,
    rank: int,
    update: int,
    accumulation: int | None,
    phase: str,
) -> None:
    _publish_progress(
        root / f"rank-{rank}.json",
        {
            "accumulation": accumulation,
            "format_version": "abpairfm_stage_a1_rank_phase_v1",
            "observed_at_unix_seconds": time.time(),
            "phase": phase,
            "rank": rank,
            "test_access": False,
            "update": update,
        },
    )


def _sync_replicated_head_gradients(
    heads: StageA1Heads,
    world_size: int,
    *,
    required_prefixes: tuple[str, ...] | None = None,
) -> None:
    for name, parameter in heads.named_parameters():
        required = required_prefixes is None or name.startswith(required_prefixes)
        if parameter.grad is None:
            if required:
                raise RuntimeError(
                    f"Stage-A1 required head parameter did not receive a gradient: {name}"
                )
            continue
        torch.distributed.all_reduce(parameter.grad)
        parameter.grad.div_(world_size)


def _clip_gradients(
    backbone: PairAwareESM2, heads: StageA1Heads, maximum_norm: float
) -> float:
    backbone_norm = torch.nn.utils.clip_grad_norm_(
        backbone.parameters(), float("inf"), foreach=False
    )
    head_norm = torch.nn.utils.clip_grad_norm_(
        heads.parameters(), float("inf"), foreach=False
    )
    total = math.sqrt(float(backbone_norm) ** 2 + float(head_norm) ** 2)
    if not math.isfinite(total):
        raise RuntimeError("Stage-A1 gradient norm is nonfinite")
    scale = min(1.0, maximum_norm / (total + 1e-6))
    if scale < 1.0:
        for parameter in [*backbone.parameters(), *heads.parameters()]:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    return total


def _mean_records(records: list[dict[str, float]], key: str, count: int) -> float:
    return statistics.mean(record[key] for record in records[-count:])


def main() -> None:
    arguments = _arguments()
    protocol = _load_protocol(arguments)
    project_root = arguments.project_root.resolve()
    observed_source_sha256 = source_sha256(project_root)
    if observed_source_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("Reviewed Stage-A1 source SHA-256 differs")
    records, manifest_sha256, manifest_receipt_sha256 = _load_manifest(
        arguments, protocol
    )
    alternating_efficiency = (
        protocol.get("format_version") in ALTERNATING_EFFICIENCY_PROTOCOL_FORMATS
    )
    microbatch_pairs_per_rank = (
        arguments.efficiency_microbatch_pairs_per_rank
        if alternating_efficiency
        else protocol["optimization"]["microbatch_pairs_per_rank"]
    )
    assert microbatch_pairs_per_rank is not None
    records = _efficiency_bucket_records(
        records,
        protocol,
        microbatch_pairs_per_rank=microbatch_pairs_per_rank,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    process_group_timeout = timedelta(
        minutes=3 if protocol.get("systems_gate_only") is True else 15
    )
    torch.distributed.init_process_group(
        backend="nccl", device_id=device, timeout=process_group_timeout
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    try:
        if world_size != arguments.expected_world_size:
            raise RuntimeError("Stage-A1 world size differs")
        properties = torch.cuda.get_device_properties(device)
        if properties.name != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"Forbidden GPU model: {properties.name}")
        input_receipt = _verify_input_dcp_collectively(arguments, protocol)
        torch.manual_seed(1701)
        torch.cuda.manual_seed(1701)
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(
            "cuda", (2, arguments.shard_group_size), mesh_dim_names=("replicate", "shard")
        )
        activation_checkpoint_mode = (
            arguments.efficiency_activation_checkpoint_mode
            if alternating_efficiency
            else "full"
        )
        with torch.device("meta"):
            backbone = PairAwareESM2(
                PairAwareESM2Config(
                    activation_checkpointing=activation_checkpoint_mode != "off",
                    activation_checkpoint_policy=(
                        "selective_attention"
                        if activation_checkpoint_mode == "selective_attention"
                        else "full"
                    ),
                )
            )
        fully_shard_pair_aware_esm2(
            backbone, mesh=mesh, precision="bfloat16", reshard_after_forward=False
        )
        initialization = load_authenticated_esm2_into_fsdp2(
            backbone,
            arguments.initial_checkpoint,
            expected_size=arguments.expected_initial_checkpoint_size,
            expected_sha256=arguments.expected_initial_checkpoint_sha256,
        )
        _load_model_only_dcp(
            backbone,
            root=arguments.input_checkpoint_root,
            generation=arguments.input_generation,
        )
        heads = StageA1Heads(temperature=protocol["loss"]["retrieval"]["temperature"]).to(
            device
        )
        system = StageA1System(backbone, heads)
        optimization = protocol["optimization"]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(backbone.parameters()),
                    "lr": optimization["backbone_learning_rate"],
                },
                {
                    "params": list(heads.parameters()),
                    "lr": optimization["head_learning_rate"],
                },
            ],
            betas=tuple(optimization["betas"]),
            weight_decay=optimization["weight_decay"],
            fused=True,
        )
        if not hasattr(backbone, "set_requires_gradient_sync") or not hasattr(
            backbone, "set_is_last_backward"
        ):
            raise RuntimeError("Stage-A1 FSDP2 accumulation controls are unavailable")
        torch.manual_seed(1701 + rank)
        torch.cuda.manual_seed(1701 + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        backbone.train()
        heads.train()
        torch.cuda.reset_peak_memory_stats(device)
        independent_retrieval_view = (
            protocol.get("format_version") in SPLIT_VIEW_PROTOCOL_FORMATS
        )
        update_records: list[dict[str, float]] = []
        from torch.nn.attention import SDPBackend, sdpa_kernel

        phase_root = arguments.result.parent / "rank_phases"
        for update in range(optimization["updates"]):
            optimizer.zero_grad(set_to_none=True)
            _publish_rank_phase(
                phase_root,
                rank=rank,
                update=update + 1,
                accumulation=None,
                phase="update_barrier_pending",
            )
            torch.distributed.barrier()
            _publish_rank_phase(
                phase_root,
                rank=rank,
                update=update + 1,
                accumulation=None,
                phase="update_barrier_complete",
            )
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            totals = torch.zeros(9, device=device, dtype=torch.float64)
            minimum_allowed_candidates = torch.tensor(
                float("inf"), device=device, dtype=torch.float64
            )
            active_phase = (
                "joint" if alternating_efficiency and update % 2 == 0 else "retrieval"
            )
            active_head_prefixes: tuple[str, ...] | None = None
            for accumulation in range(optimization["accumulation_steps"]):
                microstep = update * optimization["accumulation_steps"] + accumulation
                start = (
                    microstep * world_size + rank
                ) * microbatch_pairs_per_rank
                cpu_batch = build_stage_a1_microbatch(
                    records[start : start + microbatch_pairs_per_rank],
                    seed=1701 + microstep * world_size + rank,
                    maximum_tokens=arguments.maximum_tokens,
                    padding_multiple=(
                        protocol.get("efficiency", {}).get("padding_multiple")
                        if alternating_efficiency
                        else None
                    ),
                    maximum_positive_pairs_per_complex=protocol["loss"]["contact"][
                        "maximum_positive_pairs_per_complex"
                    ],
                    maximum_sampled_pairs_per_complex=protocol["loss"]["contact"][
                        "maximum_sampled_pairs_per_complex"
                    ],
                    maximum_negative_to_positive_ratio=protocol["loss"]["contact"][
                        "maximum_negative_to_positive_ratio"
                    ],
                )
                _publish_rank_phase(
                    phase_root,
                    rank=rank,
                    update=update + 1,
                    accumulation=accumulation + 1,
                    phase="microbatch_build_complete",
                )
                batch = _to_device(cpu_batch, device)
                synchronize = accumulation == optimization["accumulation_steps"] - 1
                retrieval_indices = batch.retrieval_batch_indices
                retrieval_chain_ids = batch.chain_ids[retrieval_indices]
                retrieval_allowed = batch.retrieval_allowed
                retrieval_target_indices = None
                if (
                    protocol["loss"]["retrieval"].get("candidate_scope")
                    == "same_node_all_ranks"
                    and (not alternating_efficiency or active_phase == "retrieval")
                ):
                    retrieval_allowed, retrieval_target_indices = (
                        _global_retrieval_mask(
                            batch,
                            rank=rank,
                            world_size=world_size,
                        )
                    )
                if alternating_efficiency:
                    backbone.set_requires_gradient_sync(synchronize)
                    backbone.set_is_last_backward(synchronize)
                    if active_phase == "joint":
                        with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16
                        ):
                            output = backbone(
                                batch.tokens,
                                batch.chain_ids,
                                labels=batch.labels,
                                return_hidden_states=True,
                            )
                            if (
                                output.balanced_chain_loss_sum is None
                                or output.hidden_states is None
                            ):
                                raise RuntimeError(
                                    "Stage-A1 efficiency joint view is incomplete"
                                )
                            mlm_loss = (
                                output.balanced_chain_loss_sum
                                / output.contributing_chain_count
                            )
                            contact_loss, contact_count, contact_positive_count = (
                                heads.contact_loss(
                                    hidden=output.hidden_states,
                                    pair_batch_indices=batch.contact_batch_indices,
                                    pair_left_token_indices=(
                                        batch.contact_left_token_indices
                                    ),
                                    pair_right_token_indices=(
                                        batch.contact_right_token_indices
                                    ),
                                    pair_labels=batch.contact_labels,
                                )
                            )
                            joint_loss = (
                                protocol["loss"]["masked_language_modeling"]["weight"]
                                * mlm_loss
                                + protocol["loss"]["contact"]["weight"]
                                * contact_loss
                            )
                        if not bool(torch.isfinite(joint_loss)):
                            raise RuntimeError("Stage-A1 efficiency joint loss is nonfinite")
                        joint_loss.backward()
                        zero = joint_loss.detach().new_zeros(())
                        multitask = StageA1Loss(
                            total=joint_loss,
                            mlm=mlm_loss,
                            retrieval=zero,
                            contact=contact_loss,
                            retrieval_accuracy_left_to_right=zero,
                            retrieval_accuracy_right_to_left=zero,
                            contact_count=contact_count,
                            contact_positive_count=contact_positive_count,
                        )
                        active_head_prefixes = ("contact_",)
                    else:
                        with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16
                        ):
                            retrieval_output = backbone(
                                batch.tokens[retrieval_indices],
                                retrieval_chain_ids,
                                allow_cross_chain_attention=False,
                                return_hidden_states=True,
                                compute_logits=False,
                            )
                            if retrieval_output.hidden_states is None:
                                raise RuntimeError(
                                    "Stage-A1 efficiency retrieval view is incomplete"
                                )
                            (
                                retrieval_candidate_left,
                                retrieval_candidate_right,
                            ) = _global_retrieval_candidates(
                                heads,
                                retrieval_output.hidden_states,
                                retrieval_chain_ids,
                            )
                            (
                                retrieval_loss,
                                retrieval_accuracy_left_to_right,
                                retrieval_accuracy_right_to_left,
                            ) = heads.retrieval_loss(
                                hidden=retrieval_output.hidden_states,
                                chain_ids=retrieval_chain_ids,
                                retrieval_allowed=retrieval_allowed,
                                retrieval_candidate_left=retrieval_candidate_left,
                                retrieval_candidate_right=retrieval_candidate_right,
                                retrieval_target_indices=retrieval_target_indices,
                            )
                            retrieval_total = (
                                protocol["loss"]["retrieval"]["weight"]
                                * retrieval_loss
                            )
                        if not bool(torch.isfinite(retrieval_total)):
                            raise RuntimeError(
                                "Stage-A1 efficiency retrieval loss is nonfinite"
                            )
                        retrieval_total.backward()
                        zero = retrieval_loss.detach().new_zeros(())
                        multitask = StageA1Loss(
                            total=retrieval_total,
                            mlm=zero,
                            retrieval=retrieval_loss,
                            contact=zero,
                            retrieval_accuracy_left_to_right=(
                                retrieval_accuracy_left_to_right
                            ),
                            retrieval_accuracy_right_to_left=(
                                retrieval_accuracy_right_to_left
                            ),
                            contact_count=0,
                            contact_positive_count=0,
                        )
                        active_head_prefixes = ("retrieval_",)
                elif independent_retrieval_view:
                    backbone.set_requires_gradient_sync(False)
                    backbone.set_is_last_backward(False)
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        output = backbone(
                            batch.tokens,
                            batch.chain_ids,
                            labels=batch.labels,
                            return_hidden_states=True,
                        )
                        if (
                            output.balanced_chain_loss_sum is None
                            or output.hidden_states is None
                        ):
                            raise RuntimeError(
                                "Stage-A1 joint backbone view is incomplete"
                            )
                        mlm_loss = (
                            output.balanced_chain_loss_sum
                            / output.contributing_chain_count
                        )
                        joint_hidden = output.hidden_states[retrieval_indices]
                        contact_loss, contact_count, contact_positive_count = (
                            heads.contact_loss(
                                hidden=joint_hidden,
                                pair_batch_indices=torch.searchsorted(
                                    retrieval_indices,
                                    batch.contact_batch_indices,
                                ),
                                pair_left_token_indices=(
                                    batch.contact_left_token_indices
                                ),
                                pair_right_token_indices=(
                                    batch.contact_right_token_indices
                                ),
                                pair_labels=batch.contact_labels,
                            )
                        )
                        joint_loss = (
                            protocol["loss"]["masked_language_modeling"]["weight"]
                            * mlm_loss
                            + protocol["loss"]["contact"]["weight"]
                            * contact_loss
                        )
                    _publish_rank_phase(
                        phase_root,
                        rank=rank,
                        update=update + 1,
                        accumulation=accumulation + 1,
                        phase="joint_forward_and_loss_complete",
                    )
                    joint_scaled = joint_loss / optimization["accumulation_steps"]
                    if not bool(torch.isfinite(joint_scaled)):
                        raise RuntimeError("Stage-A1 joint loss is nonfinite")
                    joint_scaled.backward()
                    _publish_rank_phase(
                        phase_root,
                        rank=rank,
                        update=update + 1,
                        accumulation=accumulation + 1,
                        phase="joint_backward_complete",
                    )

                    backbone.set_requires_gradient_sync(synchronize)
                    backbone.set_is_last_backward(synchronize)
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        retrieval_output = backbone(
                            batch.tokens[retrieval_indices],
                            retrieval_chain_ids,
                            allow_cross_chain_attention=False,
                            return_hidden_states=True,
                            compute_logits=False,
                        )
                        if retrieval_output.hidden_states is None:
                            raise RuntimeError(
                                "Stage-A1 independent retrieval view is missing"
                            )
                        retrieval_hidden = retrieval_output.hidden_states
                        retrieval_candidate_left = None
                        retrieval_candidate_right = None
                        if (
                            protocol["loss"]["retrieval"].get("candidate_scope")
                            == "same_node_all_ranks"
                        ):
                            (
                                retrieval_candidate_left,
                                retrieval_candidate_right,
                            ) = _global_retrieval_candidates(
                                heads,
                                retrieval_hidden,
                                retrieval_chain_ids,
                            )
                        (
                            retrieval_loss,
                            retrieval_accuracy_left_to_right,
                            retrieval_accuracy_right_to_left,
                        ) = heads.retrieval_loss(
                            hidden=retrieval_hidden,
                            chain_ids=retrieval_chain_ids,
                            retrieval_allowed=retrieval_allowed,
                            retrieval_candidate_left=retrieval_candidate_left,
                            retrieval_candidate_right=retrieval_candidate_right,
                            retrieval_target_indices=retrieval_target_indices,
                        )
                        retrieval_scaled = (
                            protocol["loss"]["retrieval"]["weight"]
                            * retrieval_loss
                            / optimization["accumulation_steps"]
                        )
                    _publish_rank_phase(
                        phase_root,
                        rank=rank,
                        update=update + 1,
                        accumulation=accumulation + 1,
                        phase="retrieval_forward_and_loss_complete",
                    )
                    if not bool(torch.isfinite(retrieval_scaled)):
                        raise RuntimeError("Stage-A1 retrieval loss is nonfinite")
                    retrieval_scaled.backward()
                    multitask = StageA1Loss(
                        total=(
                            joint_loss
                            + protocol["loss"]["retrieval"]["weight"]
                            * retrieval_loss
                        ),
                        mlm=mlm_loss,
                        retrieval=retrieval_loss,
                        contact=contact_loss,
                        retrieval_accuracy_left_to_right=(
                            retrieval_accuracy_left_to_right
                        ),
                        retrieval_accuracy_right_to_left=(
                            retrieval_accuracy_right_to_left
                        ),
                        contact_count=contact_count,
                        contact_positive_count=contact_positive_count,
                    )
                else:
                    backbone.set_requires_gradient_sync(synchronize)
                    backbone.set_is_last_backward(synchronize)
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        (
                            output,
                            joint_hidden,
                            retrieval_hidden,
                            retrieval_chain_ids,
                        ) = _stage_a1_backbone_views(
                            backbone,
                            batch,
                            independent_retrieval_view=False,
                        )
                        assert output.balanced_chain_loss_sum is not None
                        mlm_loss = (
                            output.balanced_chain_loss_sum
                            / output.contributing_chain_count
                        )
                        retrieval_candidate_left = None
                        retrieval_candidate_right = None
                        if (
                            protocol["loss"]["retrieval"].get("candidate_scope")
                            == "same_node_all_ranks"
                        ):
                            (
                                retrieval_candidate_left,
                                retrieval_candidate_right,
                            ) = _global_retrieval_candidates(
                                heads,
                                retrieval_hidden,
                                retrieval_chain_ids,
                            )
                        multitask = heads.loss(
                            hidden=joint_hidden,
                            chain_ids=retrieval_chain_ids,
                            retrieval_hidden=retrieval_hidden,
                            mlm_loss=mlm_loss,
                            retrieval_allowed=retrieval_allowed,
                            pair_batch_indices=torch.searchsorted(
                                retrieval_indices, batch.contact_batch_indices
                            ),
                            pair_left_token_indices=batch.contact_left_token_indices,
                            pair_right_token_indices=batch.contact_right_token_indices,
                            pair_labels=batch.contact_labels,
                            retrieval_candidate_left=retrieval_candidate_left,
                            retrieval_candidate_right=retrieval_candidate_right,
                            retrieval_target_indices=retrieval_target_indices,
                            mlm_weight=protocol["loss"][
                                "masked_language_modeling"
                            ]["weight"],
                            retrieval_weight=protocol["loss"]["retrieval"]["weight"],
                            contact_weight=protocol["loss"]["contact"]["weight"],
                        )
                        loss = multitask.total / optimization["accumulation_steps"]
                    _publish_rank_phase(
                        phase_root,
                        rank=rank,
                        update=update + 1,
                        accumulation=accumulation + 1,
                        phase="forward_and_loss_complete",
                    )
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Stage-A1 multitask loss is nonfinite")
                    loss.backward()
                minimum_allowed_candidates = _update_minimum_allowed_candidates(
                    minimum_allowed_candidates,
                    retrieval_allowed,
                )
                _publish_rank_phase(
                    phase_root,
                    rank=rank,
                    update=update + 1,
                    accumulation=accumulation + 1,
                    phase="backward_complete",
                )
                totals += torch.tensor(
                    [
                        float(multitask.total.detach()),
                        float(multitask.mlm.detach()),
                        float(multitask.retrieval.detach()),
                        float(multitask.contact.detach()),
                        float(multitask.retrieval_accuracy_left_to_right.detach()),
                        float(multitask.retrieval_accuracy_right_to_left.detach()),
                        float(batch.residue_count),
                        float(retrieval_allowed.shape[1]),
                        float(batch.tokens.numel()),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
            _sync_replicated_head_gradients(
                heads,
                world_size,
                required_prefixes=active_head_prefixes,
            )
            _publish_rank_phase(
                phase_root,
                rank=rank,
                update=update + 1,
                accumulation=None,
                phase="head_gradient_sync_complete",
            )
            gradient_norm = _clip_gradients(
                backbone, heads, optimization["gradient_clip_norm"]
            )
            optimizer.step()
            _publish_rank_phase(
                phase_root,
                rank=rank,
                update=update + 1,
                accumulation=None,
                phase="optimizer_step_complete",
            )
            torch.distributed.all_reduce(totals)
            torch.distributed.all_reduce(
                minimum_allowed_candidates, op=torch.distributed.ReduceOp.MIN
            )
            totals[:6] /= world_size * optimization["accumulation_steps"]
            totals[7] /= world_size * optimization["accumulation_steps"]
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            approximate_useful_token_mfu = None
            if alternating_efficiency:
                approximate_useful_token_mfu = (
                    6.0
                    * protocol["efficiency"]["parameter_count"]
                    * float(totals[6])
                    / elapsed
                    / (
                        world_size
                        * protocol["efficiency"]["gpu_peak_bfloat16_flops"]
                    )
                )
            record = {
                "active_phase": active_phase if alternating_efficiency else "multitask",
                "approximate_useful_token_6n_mfu": approximate_useful_token_mfu,
                "contact_loss": float(totals[3]),
                "elapsed_seconds": elapsed,
                "gradient_norm": gradient_norm,
                "mlm_loss": float(totals[1]),
                "packing_efficiency": float(totals[6] / totals[8]),
                "padded_tokens": float(totals[8]),
                "minimum_allowed_candidates_per_query": float(
                    minimum_allowed_candidates
                ),
                "residue_tokens": float(totals[6]),
                "retrieval_candidate_matrix_width": float(totals[7]),
                "retrieval_accuracy_left_to_right": float(totals[4]),
                "retrieval_accuracy_right_to_left": float(totals[5]),
                "retrieval_loss": float(totals[2]),
                "total_loss": float(totals[0]),
            }
            update_records.append(record)
            if rank == 0 and (
                (update + 1) % arguments.progress_interval == 0
                or update + 1 == optimization["updates"]
            ):
                _publish_progress(
                    arguments.progress,
                    {
                        "format_version": "abpairfm_stage_a1_pilot_progress_v1",
                        "latest": record,
                        "status": (
                            "RUNNING"
                            if update + 1 < optimization["updates"]
                            else "TRAINING_COMPLETE_CHECKPOINT_PENDING"
                        ),
                        "test_access": False,
                        "update": update + 1,
                    },
                )
        rank_record = {
            "device_name": properties.name,
            "hostname": socket.gethostname(),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "rank": rank,
            "total_memory_bytes": properties.total_memory,
        }
        ranks = _collect(rank_record, world_size)
        reshard_pair_aware_esm2(backbone)
        provenance = {
            "input_checkpoint": arguments.expected_input_content_sha256,
            "manifest": manifest_sha256,
            "manifest_receipt": manifest_receipt_sha256,
            "protocol": arguments.expected_protocol_sha256,
            "source": observed_source_sha256,
        }
        checkpoint_receipt = _DCP.save_dcp_generation(
            arguments.checkpoint_root,
            "pilot-000000",
            model=system,
            optimizer=optimizer,
            provenance_hashes=provenance,
            training_state={
                "input_consumed_residue_tokens": input_receipt["training_state"][
                    "consumed_residue_tokens"
                ],
                "pilot_updates": optimization["updates"],
                "test_access": False,
            },
        )
        restore_receipt = _DCP.load_dcp_generation(
            arguments.checkpoint_root,
            "pilot-000000",
            model=system,
            optimizer=optimizer,
            expected_provenance_hashes=provenance,
        )
        peak_fraction = max(
            rank_value["max_memory_allocated_bytes"] / rank_value["total_memory_bytes"]
            for rank_value in ranks
        )
        candidate_floor = min(
            record["minimum_allowed_candidates_per_query"]
            for record in update_records
        )
        required_candidate_floor = protocol["promotion_gates"].get(
            "minimum_retrieval_candidates_per_query", 2
        )
        if peak_fraction > protocol["promotion_gates"]["maximum_peak_memory_fraction"]:
            status = "FAIL_STAGE_A1_MULTITASK_PILOT_MEMORY_GATE"
        elif candidate_floor < required_candidate_floor:
            status = "FAIL_STAGE_A1_RETRIEVAL_CANDIDATE_GATE"
        elif protocol.get("format_version") in ALTERNATING_EFFICIENCY_PROTOCOL_FORMATS:
            status = "PASS_STAGE_A1_EFFICIENCY_GATE_CHECKPOINT_VERIFIED"
        elif protocol.get("format_version") in GLOBAL_RETRIEVAL_PROTOCOL_FORMATS:
            status = "PASS_STAGE_A1_GLOBAL_RETRIEVAL_PILOT_CHECKPOINT_VERIFIED"
        else:
            status = "PASS_STAGE_A1_MULTITASK_PILOT_CHECKPOINT_VERIFIED"
        if rank == 0:
            elapsed_values = [record["elapsed_seconds"] for record in update_records]
            result = {
                "authorization": protocol["authorization"],
                "checkpoint": {
                    "content_sha256": checkpoint_receipt["content_sha256"],
                    "generation": "pilot-000000",
                    "restore_content_sha256": restore_receipt["content_sha256"],
                    "root": str(arguments.checkpoint_root.resolve()),
                },
                "claim_boundary": protocol["claim_boundary"],
                "format_version": "abpairfm_stage_a1_multitask_pilot_result_v1",
                "input_checkpoint": {
                    "consumed_residue_tokens": input_receipt["training_state"][
                        "consumed_residue_tokens"
                    ],
                    "content_sha256": arguments.expected_input_content_sha256,
                    "generation": arguments.input_generation,
                },
                "initialization": initialization,
                "loss": {
                    name: {
                        "final_20_update_mean": _mean_records(update_records, name, 20),
                        "first_20_update_mean": statistics.mean(
                            record[name] for record in update_records[:20]
                        ),
                    }
                    for name in ("total_loss", "mlm_loss", "retrieval_loss", "contact_loss")
                },
                "manifest": {
                    "records_consumed": (
                        optimization["updates"]
                        * optimization["accumulation_steps"]
                        * world_size
                        * microbatch_pairs_per_rank
                    ),
                    "receipt_sha256": manifest_receipt_sha256,
                    "sha256": manifest_sha256,
                    "total_records_available": len(records),
                },
                "memory": {
                    "maximum_allocated_fraction": peak_fraction,
                    "required_maximum_fraction": protocol["promotion_gates"][
                        "maximum_peak_memory_fraction"
                    ],
                },
                "optimization": {
                    **optimization,
                    "selected_activation_checkpoint_mode": (
                        activation_checkpoint_mode
                    ),
                    "selected_microbatch_pairs_per_rank": microbatch_pairs_per_rank,
                },
                "ranks": ranks,
                "retrieval": {
                    "candidate_scope": protocol["loss"]["retrieval"].get(
                        "candidate_scope", "local_rank"
                    ),
                    "final_20_accuracy_left_to_right": _mean_records(
                        update_records, "retrieval_accuracy_left_to_right", 20
                    ),
                    "final_20_accuracy_right_to_left": _mean_records(
                        update_records, "retrieval_accuracy_right_to_left", 20
                    ),
                    "candidate_matrix_width": min(
                        record["retrieval_candidate_matrix_width"]
                        for record in update_records
                    ),
                    "minimum_allowed_candidates_per_query": candidate_floor,
                    "representation_view": protocol["loss"]["retrieval"].get(
                        "representation_view", "joint_cross_chain_attention"
                    ),
                    "required_minimum_candidates_per_query": required_candidate_floor,
                },
                "source_sha256": observed_source_sha256,
                "stability": {
                    "maximum_gradient_norm_before_clipping": max(
                        record["gradient_norm"] for record in update_records
                    ),
                    "p95_to_median_update_time": _nearest_rank_p95(elapsed_values)
                    / statistics.median(elapsed_values),
                },
                "status": status,
                "test_access": False,
                "throughput": {
                    "approximate_useful_token_6n_mfu_mean": (
                        statistics.mean(
                            record["approximate_useful_token_6n_mfu"]
                            for record in update_records
                            if record["approximate_useful_token_6n_mfu"] is not None
                        )
                        if alternating_efficiency
                        else None
                    ),
                    "approximate_useful_token_6n_mfu_by_phase": (
                        {
                            phase: statistics.mean(
                                record["approximate_useful_token_6n_mfu"]
                                for record in update_records
                                if record["active_phase"] == phase
                            )
                            for phase in ("joint", "retrieval")
                        }
                        if alternating_efficiency
                        else None
                    ),
                    "mean_residue_tokens_per_second": statistics.mean(
                        record["residue_tokens"] / record["elapsed_seconds"]
                        for record in update_records
                    ),
                    "mean_packing_efficiency": statistics.mean(
                        record["packing_efficiency"] for record in update_records
                    ),
                    "pilot_residue_tokens": sum(
                        record["residue_tokens"] for record in update_records
                    ),
                },
            }
            _publish_create_only(arguments.result, result)
            _publish_progress(
                arguments.progress,
                {
                    "checkpoint_content_sha256": checkpoint_receipt["content_sha256"],
                    "format_version": "abpairfm_stage_a1_pilot_progress_v1",
                    "latest": update_records[-1],
                    "status": status,
                    "test_access": False,
                    "update": optimization["updates"],
                },
            )
        torch.distributed.barrier()
        if status.startswith("FAIL_"):
            raise RuntimeError("Stage-A1 pilot failed its frozen memory gate")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
