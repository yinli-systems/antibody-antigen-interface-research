#!/usr/bin/env python3
"""Run the frozen 80-update ABF-PairContact residual mechanism canary."""

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
from antibody_pretraining.stage_a1_geometry_objective import (  # noqa: E402
    StageA1GeometryHeads,
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
    "abpairfm_stage_a1_paircontact_residual_canary_v18",
    "abpairfm_stage_a1_paircontact_zero_sum_canary_v19",
    "abpairfm_stage_a1_paircontact_zero_sum_fp32_canary_v20",
}
GEOMETRY_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_geometry_canary_v15",
    "abpairfm_stage_a1_geometry_head_only_canary_v16",
    "abpairfm_stage_a1_geometry_ranking_canary_v17",
}
HEAD_ONLY_GEOMETRY_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_geometry_head_only_canary_v16",
    "abpairfm_stage_a1_geometry_ranking_canary_v17",
}
RANKING_GEOMETRY_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_geometry_ranking_canary_v17",
}
PAIRCONTACT_RESIDUAL_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_paircontact_residual_canary_v18",
    "abpairfm_stage_a1_paircontact_zero_sum_canary_v19",
    "abpairfm_stage_a1_paircontact_zero_sum_fp32_canary_v20",
}
PAIRCONTACT_ZERO_SUM_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_paircontact_zero_sum_canary_v19",
    "abpairfm_stage_a1_paircontact_zero_sum_fp32_canary_v20",
}
HEAD_ONLY_PROTOCOL_FORMATS = (
    HEAD_ONLY_GEOMETRY_PROTOCOL_FORMATS | PAIRCONTACT_RESIDUAL_PROTOCOL_FORMATS
)
CONTACT_RANKING_PROTOCOL_FORMATS = (
    RANKING_GEOMETRY_PROTOCOL_FORMATS | PAIRCONTACT_RESIDUAL_PROTOCOL_FORMATS
)
GLOBAL_RETRIEVAL_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_multitask_pilot_v4",
    "abpairfm_stage_a1_multitask_pilot_v5",
    "abpairfm_stage_a1_multitask_pilot_v6",
    "abpairfm_stage_a1_efficiency_gate_v7",
    "abpairfm_stage_a1_efficiency_ac_gate_v8",
    "abpairfm_stage_a1_representative_efficiency_gate_v9",
    "abpairfm_stage_a1_length_bucket_efficiency_gate_v10",
    "abpairfm_stage_a1_structural_bucket_efficiency_gate_v11",
    "abpairfm_stage_a1_length_band_efficiency_gate_v13",
    "abpairfm_stage_a1_length_band_promotion_v14",
    "abpairfm_stage_a1_geometry_canary_v15",
}
SPLIT_VIEW_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_multitask_pilot_v5",
    "abpairfm_stage_a1_multitask_pilot_v6",
    "abpairfm_stage_a1_efficiency_gate_v7",
    "abpairfm_stage_a1_efficiency_ac_gate_v8",
    "abpairfm_stage_a1_representative_efficiency_gate_v9",
    "abpairfm_stage_a1_length_bucket_efficiency_gate_v10",
    "abpairfm_stage_a1_structural_bucket_efficiency_gate_v11",
    "abpairfm_stage_a1_length_band_efficiency_gate_v13",
    "abpairfm_stage_a1_length_band_promotion_v14",
    "abpairfm_stage_a1_geometry_canary_v15",
}
ALTERNATING_EFFICIENCY_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_efficiency_gate_v7",
    "abpairfm_stage_a1_efficiency_ac_gate_v8",
    "abpairfm_stage_a1_representative_efficiency_gate_v9",
    "abpairfm_stage_a1_length_bucket_efficiency_gate_v10",
    "abpairfm_stage_a1_structural_bucket_efficiency_gate_v11",
    "abpairfm_stage_a1_length_band_efficiency_gate_v13",
    "abpairfm_stage_a1_length_band_promotion_v14",
    "abpairfm_stage_a1_geometry_canary_v15",
    "abpairfm_stage_a1_geometry_head_only_canary_v16",
    "abpairfm_stage_a1_geometry_ranking_canary_v17",
    "abpairfm_stage_a1_paircontact_residual_canary_v18",
    "abpairfm_stage_a1_paircontact_zero_sum_canary_v19",
    "abpairfm_stage_a1_paircontact_zero_sum_fp32_canary_v20",
}
REPRESENTATIVE_EFFICIENCY_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_representative_efficiency_gate_v9",
    "abpairfm_stage_a1_length_bucket_efficiency_gate_v10",
    "abpairfm_stage_a1_structural_bucket_efficiency_gate_v11",
}
LENGTH_BUCKET_EFFICIENCY_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_length_bucket_efficiency_gate_v10",
}
STRUCTURAL_BUCKET_EFFICIENCY_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_structural_bucket_efficiency_gate_v11",
}
LENGTH_BAND_EFFICIENCY_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_length_band_efficiency_gate_v13",
    "abpairfm_stage_a1_length_band_promotion_v14",
    "abpairfm_stage_a1_geometry_canary_v15",
    "abpairfm_stage_a1_geometry_head_only_canary_v16",
    "abpairfm_stage_a1_geometry_ranking_canary_v17",
    "abpairfm_stage_a1_paircontact_residual_canary_v18",
    "abpairfm_stage_a1_paircontact_zero_sum_canary_v19",
    "abpairfm_stage_a1_paircontact_zero_sum_fp32_canary_v20",
}
PROMOTION_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_length_band_promotion_v14",
}
SOURCE_PATHS = (
    "scripts/run_abf_paircontact_residual_canary.py",
    "scripts/run_abpairfm_3b_real_data_gate_v2.py",
    "src/antibody_pretraining/pair_aware_attention.py",
    "src/antibody_pretraining/pair_aware_distributed.py",
    "src/antibody_pretraining/pair_aware_esm2.py",
    "src/antibody_pretraining/stage_a1_batch.py",
    "src/antibody_pretraining/stage_a1_objective.py",
    "src/antibody_pretraining/stage_a1_geometry_objective.py",
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
    parser.add_argument(
        "--rank-phase-telemetry",
        choices=("full", "off"),
        default="full",
        help="Disable shared-filesystem phase writes only for isolated systems gates.",
    )
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


def _replicate_shard_mesh_shape(
    world_size: int, shard_group_size: int
) -> tuple[int, int]:
    if shard_group_size < 1 or shard_group_size > world_size:
        raise ValueError("shard-group-size must be in [1, world-size]")
    if world_size % shard_group_size:
        raise ValueError("shard-group-size must divide world-size")
    return world_size // shard_group_size, shard_group_size


def _load_protocol(arguments: argparse.Namespace) -> dict[str, Any]:
    if _sha256(arguments.protocol) != arguments.expected_protocol_sha256:
        raise RuntimeError("Stage-A1 protocol SHA-256 differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    authorization = protocol.get("authorization", {})
    systems = protocol.get("systems", {})
    optimization = protocol.get("optimization", {})
    format_version = protocol.get("format_version")
    promotion = format_version in PROMOTION_PROTOCOL_FORMATS
    expected_updates = (
        160
        if format_version == "abpairfm_stage_a1_paircontact_zero_sum_fp32_canary_v20"
        else 80
        if format_version in HEAD_ONLY_PROTOCOL_FORMATS
        else 40
        if format_version in GEOMETRY_PROTOCOL_FORMATS
        else 200
        if promotion
        else 10
        if format_version in SPLIT_VIEW_PROTOCOL_FORMATS
        else 200
    )
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
    elif format_version in HEAD_ONLY_PROTOCOL_FORMATS:
        expected_efficiency_microbatches = [48]
        expected_activation_checkpoint_modes = ["off"]
    elif format_version in LENGTH_BAND_EFFICIENCY_PROTOCOL_FORMATS:
        expected_efficiency_microbatches = [24, 32, 48]
        expected_activation_checkpoint_modes = ["full"]
    else:
        expected_efficiency_microbatches = [6, 7, 8]
        expected_activation_checkpoint_modes = (
            ["full"]
            if format_version
            in (
                LENGTH_BUCKET_EFFICIENCY_PROTOCOL_FORMATS
                | STRUCTURAL_BUCKET_EFFICIENCY_PROTOCOL_FORMATS
            )
            else ["full", "off"]
        )
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
                            not in REPRESENTATIVE_EFFICIENCY_PROTOCOL_FORMATS
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
    authorized_training = (
        (
            protocol.get("status") == "FROZEN_BEFORE_STAGE_A1_PROMOTION"
            and authorization.get("pilot_training") is False
            and authorization.get("production_training") is True
        )
        if promotion
        else (
            protocol.get("status")
            == "FROZEN_BEFORE_PILOT_MANIFEST_AND_GPU_TRAINING"
            and authorization.get("pilot_training") is True
            and authorization.get("production_training") is False
        )
    )
    if (
        format_version not in EXPECTED_PROTOCOL_FORMATS
        or not authorized_training
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
        or protocol.get("systems_gate_only") != (not promotion)
        or protocol.get("counterfactual_gate", {}).get("status")
        != "PASS_TRAIN_ONLY_MATERIALIZATION"
    ):
        raise RuntimeError("Stage-A1 v5 representation-view gate differs")
    if alternating_efficiency:
        efficiency = protocol.get("efficiency", {})
        expected_phase_schedule = (
            "paircontact_zero_sum_head_only_frozen_backbone"
            if format_version in PAIRCONTACT_ZERO_SUM_PROTOCOL_FORMATS
            else (
                "paircontact_residual_head_only_frozen_backbone"
                if format_version in PAIRCONTACT_RESIDUAL_PROTOCOL_FORMATS
                else (
                    "geometry_ranking_head_only_frozen_backbone"
                    if format_version in RANKING_GEOMETRY_PROTOCOL_FORMATS
                    else (
                        "geometry_head_only_frozen_backbone"
                        if format_version in HEAD_ONLY_GEOMETRY_PROTOCOL_FORMATS
                        else "alternating_joint_and_retrieval_single_backbone_pass"
                    )
                )
            )
        )
        common_efficiency_valid = (
            protocol.get("efficiency_gate_only") == (not promotion)
            and protocol.get("systems_gate_only") == (not promotion)
            and efficiency.get("padding_multiple") == 64
            and efficiency.get("parameter_count") == 2839005633
            and efficiency.get("gpu_peak_bfloat16_flops") == 82580000000000
            and efficiency.get("phase_schedule") == expected_phase_schedule
        )
        representative_valid = (
            efficiency.get("maximum_combined_tokens") == 1024
            and efficiency.get("expected_source_records") == 19200
        )
        if format_version in LENGTH_BUCKET_EFFICIENCY_PROTOCOL_FORMATS:
            representative_valid = representative_valid and (
                efficiency.get("length_bucket_block_microbatches") == 16
                and efficiency.get("ppiref_records_per_group") == 5
                and efficiency.get("string_records_per_group") == 1
                and efficiency.get("selection_seed") == 1701
            )
        if format_version in STRUCTURAL_BUCKET_EFFICIENCY_PROTOCOL_FORMATS:
            representative_valid = representative_valid and (
                efficiency.get("expected_structural_records") == 16000
                and efficiency.get("length_bucket_block_records") == 128
                and efficiency.get("selection_seed") == 1701
            )
        short_bucket_valid = (
            efficiency.get("maximum_combined_tokens") == 64
            and efficiency.get("expected_bucket_records") == 10380
        )
        length_band_valid = (
            efficiency.get("minimum_combined_tokens") == 65
            and efficiency.get("maximum_combined_tokens") == 128
            and efficiency.get("expected_bucket_records") == 4967
            and efficiency.get("expected_source_records") == 19200
            and efficiency.get("selection_seed") == 1701
        )
        selected_efficiency_shape_valid = (
            representative_valid
            if format_version in REPRESENTATIVE_EFFICIENCY_PROTOCOL_FORMATS
            else (
                length_band_valid
                if format_version in LENGTH_BAND_EFFICIENCY_PROTOCOL_FORMATS
                else short_bucket_valid
            )
        )
        if not common_efficiency_valid or not selected_efficiency_shape_valid:
            raise RuntimeError("Stage-A1 efficiency gate differs")
        if format_version in CONTACT_RANKING_PROTOCOL_FORMATS:
            holdout = protocol.get("internal_holdout", {})
            promotion_gates = protocol.get("promotion_gates", {})
            common_holdout_valid = (
                holdout.get("target_records") == 512
                and holdout.get("grouping") == "lowercase_pdb_prefix"
                and holdout.get("test_access") is False
            )
            legacy_holdout_valid = (
                holdout.get("selection_seed") == 1701
                and holdout.get("ordering_salt", "v17_holdout") == "v17_holdout"
                and holdout.get("expected_pdb_count") == 511
                and holdout.get("pdb_inventory_sha256")
                == "0166ecee25b61f333fe36197ca9625ed3a134221946a38e023f6022b4cc49028"
                and "excluded_pdb_set" not in holdout
            )
            excluded = holdout.get("excluded_pdb_set", {})
            zero_sum_v19_holdout_valid = (
                holdout.get("selection_seed") == 2909
                and holdout.get("ordering_salt") == "v19_confirmatory_holdout"
                and holdout.get("expected_pdb_count") == 511
                and holdout.get("pdb_inventory_sha256")
                == "305e868ed39fcc9794ebe06b48b28228e1831a2b29e568b9a8b80835652b119c"
                and excluded
                == {
                    "expected_pdb_count": 511,
                    "ordering_salt": "v17_holdout",
                    "pdb_inventory_sha256": "0166ecee25b61f333fe36197ca9625ed3a134221946a38e023f6022b4cc49028",
                    "selection_seed": 1701,
                    "target_records": 512,
                }
            )
            zero_sum_v20_holdout_valid = (
                holdout.get("selection_seed") == 3109
                and holdout.get("ordering_salt")
                == "v20_fp32_confirmatory_holdout"
                and holdout.get("expected_pdb_count") == 508
                and holdout.get("pdb_inventory_sha256")
                == "9bd3afebb2ab4b21db19ae4c564179ead91b9fa93f673bd09fcaa56fa223e17d"
                and holdout.get("excluded_pdb_sets")
                == [
                    {
                        "expected_pdb_count": 511,
                        "ordering_salt": "v17_holdout",
                        "pdb_inventory_sha256": "0166ecee25b61f333fe36197ca9625ed3a134221946a38e023f6022b4cc49028",
                        "selection_seed": 1701,
                        "target_records": 512,
                    },
                    {
                        "expected_pdb_count": 511,
                        "ordering_salt": "v19_confirmatory_holdout",
                        "pdb_inventory_sha256": "305e868ed39fcc9794ebe06b48b28228e1831a2b29e568b9a8b80835652b119c",
                        "selection_seed": 2909,
                        "target_records": 512,
                    },
                ]
                and "excluded_pdb_set" not in holdout
            )
            holdout_valid = (
                zero_sum_v20_holdout_valid
                if format_version
                == "abpairfm_stage_a1_paircontact_zero_sum_fp32_canary_v20"
                else zero_sum_v19_holdout_valid
                if format_version == "abpairfm_stage_a1_paircontact_zero_sum_canary_v19"
                else legacy_holdout_valid
            )
            if not common_holdout_valid or not holdout_valid:
                raise RuntimeError("Stage-A1 geometry-ranking holdout differs")
            if format_version in PAIRCONTACT_ZERO_SUM_PROTOCOL_FORMATS and (
                protocol["loss"]["contact"].get("interaction_residualization")
                != "endpoint_zero_sum"
                or protocol["loss"]["contact"].get("ranking_weight") != 0.5
                or protocol["loss"]["contact"].get("centering_weight") != 0.0
                or protocol["loss"]["contact"].get("ranking_margin") != 0.5
                or protocol["loss"]["contact"].get("distance_bin_weight") != 0.0
                or protocol["loss"]["contact"].get(
                    "distance_label_manifest_permitted"
                )
                is not False
                or promotion_gates.get("maximum_endpoint_centering_error")
                != 1e-8
                or promotion_gates.get(
                    "minimum_internal_holdout_full_macro_ap_absolute_gain"
                )
                != 0.01
                or promotion_gates.get(
                    "minimum_internal_holdout_interaction_macro_ap_absolute_gain"
                )
                != 0.01
                or promotion_gates.get(
                    "minimum_internal_holdout_row_matched_accuracy_absolute_gain"
                )
                != 0.02
                or promotion_gates.get(
                    "minimum_internal_holdout_column_matched_accuracy_absolute_gain"
                )
                != 0.02
                or promotion_gates.get("minimum_train_contact_relative_reduction")
                != 0.01
                or promotion_gates.get("minimum_train_ranking_relative_reduction")
                != 0.02
            ):
                raise RuntimeError("Stage-A1 PairContact zero-sum gate differs")
            if (
                format_version in PAIRCONTACT_RESIDUAL_PROTOCOL_FORMATS
                and format_version not in PAIRCONTACT_ZERO_SUM_PROTOCOL_FORMATS
                and (
                protocol["loss"]["contact"].get("ranking_weight") != 0.5
                or protocol["loss"]["contact"].get("centering_weight") != 0.1
                or protocol["loss"]["contact"].get("ranking_margin") != 0.5
                or protocol["loss"]["contact"].get("distance_bin_weight") != 0.0
                or protocol["loss"]["contact"].get(
                    "distance_label_manifest_permitted"
                )
                is not False
                or promotion_gates.get(
                    "minimum_internal_holdout_full_macro_ap_absolute_gain"
                )
                != 0.02
                or promotion_gates.get(
                    "minimum_internal_holdout_interaction_macro_ap_absolute_gain"
                )
                != 0.01
                or promotion_gates.get(
                    "minimum_internal_holdout_row_matched_accuracy_absolute_gain"
                )
                != 0.02
                or promotion_gates.get(
                    "minimum_internal_holdout_column_matched_accuracy_absolute_gain"
                )
                != 0.02
                or promotion_gates.get("minimum_train_contact_relative_reduction")
                != 0.02
                or promotion_gates.get("minimum_train_ranking_relative_reduction")
                != 0.02
                )
            ):
                raise RuntimeError("Stage-A1 PairContact residual gate differs")
            if format_version in RANKING_GEOMETRY_PROTOCOL_FORMATS and (
                protocol["loss"]["contact"].get("ranking_weight") != 1.0
                or promotion_gates.get(
                    "minimum_internal_holdout_macro_auprc_absolute_gain"
                )
                != 0.02
                or promotion_gates.get(
                    "minimum_internal_holdout_pairwise_accuracy_absolute_gain"
                )
                != 0.02
            ):
                raise RuntimeError("Stage-A1 geometry-ranking metric gate differs")
        if promotion and protocol.get("promotion_selection") != {
            "source_gate": "abpairfm_stage_a1_length_band_efficiency_gate_v13",
            "rule": "highest_mean_mfu_among_pass_then_highest_throughput",
            "required_status": "PASS_STAGE_A1_EFFICIENCY_GATE_CHECKPOINT_VERIFIED",
        }:
            raise RuntimeError("Stage-A1 promotion selection rule differs")
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
        "abpairfm_stage_a1_paircontact_residual_canary_v18",
        "abpairfm_stage_a1_paircontact_zero_sum_canary_v19",
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
    if protocol.get("format_version") in GEOMETRY_PROTOCOL_FORMATS:
        if (
            receipt.get("status")
            != "PASS_STAGE_A1_DISTANCE_BIN_TRAIN_ONLY_MATERIALIZATION"
            or receipt.get("test_access") is not False
            or receipt.get("output", {}).get("sha256") != manifest_sha256
            or receipt.get("counts", {}).get("records")
            != protocol["pilot_manifest"]["total_records"]
            or receipt.get("counts", {}).get("distance_bin_records") != 4967
            or receipt.get("distance_bin_edges_angstrom")
            != protocol["loss"]["contact"]["distance_bin_edges_angstrom"]
            or receipt.get("cross_source_exact_pair_overlap_after_filter") != 0
        ):
            raise RuntimeError("Stage-A1 geometry manifest receipt differs")
    elif (
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
    if (
        protocol.get("format_version") in GLOBAL_RETRIEVAL_PROTOCOL_FORMATS
        and protocol.get("format_version") not in GEOMETRY_PROTOCOL_FORMATS
        and (
        receipt.get("counts", {}).get("retrieval_supervised_records")
        != protocol["pilot_manifest"]["expected_retrieval_supervised_records"]
        or receipt.get("counts", {}).get(
            "retrieval_records_with_endpoint_cluster_ids"
        )
        != protocol["pilot_manifest"]["expected_retrieval_supervised_records"]
        )
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
        if format_version in LENGTH_BUCKET_EFFICIENCY_PROTOCOL_FORMATS:
            return _length_bucket_stage_a1_records(records, protocol)
        if format_version in STRUCTURAL_BUCKET_EFFICIENCY_PROTOCOL_FORMATS:
            return _structural_bucket_stage_a1_records(records, protocol)
        return records
    efficiency = protocol["efficiency"]
    minimum_combined_tokens = int(efficiency.get("minimum_combined_tokens", 0))
    maximum_combined_tokens = int(efficiency["maximum_combined_tokens"])
    selected = [
        record
        for record in records
        if record.get("source") == "ppiref50k"
        and record.get("contact_supervision") is True
        and record.get("retrieval_supervision") is True
        and minimum_combined_tokens
        <= len(str(record["sequence1"]))
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
    if format_version in CONTACT_RANKING_PROTOCOL_FORMATS:
        return selected
    required = (
        protocol["optimization"]["updates"]
        * protocol["systems"]["world_size"]
        * microbatch_pairs_per_rank
    )
    if len(selected) < required:
        if format_version not in (PROMOTION_PROTOCOL_FORMATS | GEOMETRY_PROTOCOL_FORMATS):
            raise RuntimeError("Stage-A1 efficiency bucket is too small for the frozen trial")
        repeats = math.ceil(required / len(selected))
        selected = (selected * repeats)[:required]
    return selected


def _geometry_ranking_holdout(
    records: list[dict[str, Any]], protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Freeze a PDB-disjoint internal holdout before repeating training records."""

    holdout = protocol["internal_holdout"]
    pdb_to_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        example_id = str(record["example_id"])
        pdb_id = example_id.split("_", 1)[0].lower()
        if len(pdb_id) != 4:
            raise RuntimeError("Stage-A1 internal holdout PDB identity is malformed")
        pdb_to_records.setdefault(pdb_id, []).append(record)
    excluded_pdbs: set[str] = set()
    excluded_specs = holdout.get("excluded_pdb_sets")
    if excluded_specs is None:
        excluded = holdout.get("excluded_pdb_set")
        excluded_specs = [] if excluded is None else [excluded]
    if not isinstance(excluded_specs, list):
        raise RuntimeError("Stage-A1 excluded holdout sets are malformed")
    for excluded in excluded_specs:
        if not isinstance(excluded, dict):
            raise RuntimeError("Stage-A1 excluded holdout set is malformed")
        excluded_order = sorted(
            (pdb_id for pdb_id in pdb_to_records if pdb_id not in excluded_pdbs),
            key=lambda pdb_id: hashlib.sha256(
                (
                    f"{excluded['selection_seed']}\0{excluded['ordering_salt']}\0"
                    f"{pdb_id}"
                ).encode("utf-8")
            ).digest(),
        )
        excluded_count = 0
        excluded_ordered_selection: list[str] = []
        for pdb_id in excluded_order:
            if excluded_count >= excluded["target_records"]:
                break
            excluded_ordered_selection.append(pdb_id)
            excluded_count += len(pdb_to_records[pdb_id])
        selected_exclusion = set(excluded_ordered_selection)
        excluded_inventory_sha256 = hashlib.sha256(
            ("\n".join(excluded_ordered_selection) + "\n").encode("utf-8")
        ).hexdigest()
        if (
            excluded_count != excluded["target_records"]
            or len(selected_exclusion) != excluded["expected_pdb_count"]
            or excluded_inventory_sha256 != excluded["pdb_inventory_sha256"]
            or bool(selected_exclusion & excluded_pdbs)
        ):
            raise RuntimeError("Stage-A1 excluded holdout inventory differs")
        excluded_pdbs.update(selected_exclusion)
    ordering_salt = holdout.get("ordering_salt", "v17_holdout")
    ordered_pdbs = sorted(
        (pdb_id for pdb_id in pdb_to_records if pdb_id not in excluded_pdbs),
        key=lambda pdb_id: hashlib.sha256(
            (
                f"{holdout['selection_seed']}\0{ordering_salt}\0{pdb_id}"
            ).encode("utf-8")
        ).digest(),
    )
    selected_pdbs: list[str] = []
    selected_count = 0
    for pdb_id in ordered_pdbs:
        if selected_count >= holdout["target_records"]:
            break
        selected_pdbs.append(pdb_id)
        selected_count += len(pdb_to_records[pdb_id])
    inventory_sha256 = hashlib.sha256(
        ("\n".join(selected_pdbs) + "\n").encode("utf-8")
    ).hexdigest()
    if (
        selected_count != holdout["target_records"]
        or len(selected_pdbs) != holdout["expected_pdb_count"]
        or inventory_sha256 != holdout["pdb_inventory_sha256"]
    ):
        raise RuntimeError("Stage-A1 internal holdout inventory differs")
    selected_set = set(selected_pdbs)
    holdout_records = [
        record
        for record in records
        if str(record["example_id"]).split("_", 1)[0].lower() in selected_set
    ]
    training_records = [
        record
        for record in records
        if str(record["example_id"]).split("_", 1)[0].lower() not in selected_set
    ]
    if (
        not training_records
        or len(training_records) + len(holdout_records) != len(records)
    ):
        raise RuntimeError("Stage-A1 internal holdout partition is incomplete")
    return training_records, holdout_records, {
        "grouping": holdout["grouping"],
        "holdout_pdb_count": len(selected_pdbs),
        "holdout_record_count": len(holdout_records),
        "pdb_inventory_sha256": inventory_sha256,
        "excluded_pdb_count": len(excluded_pdbs),
        "ordering_salt": ordering_salt,
        "selection_seed": holdout["selection_seed"],
        "test_access": False,
        "training_record_count": len(training_records),
    }


def _length_bucket_stage_a1_records(
    records: list[dict[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Preserve the source ratio while shuffling contiguous length buckets."""

    efficiency = protocol["efficiency"]
    ppiref_per_group = int(efficiency["ppiref_records_per_group"])
    string_per_group = int(efficiency["string_records_per_group"])
    if string_per_group != 1:
        raise RuntimeError("Stage-A1 length bucketing requires one STRING record")

    def order_key(record: Mapping[str, Any]) -> tuple[int, str]:
        return (
            len(str(record["sequence1"])) + len(str(record["sequence2"])) + 4,
            str(record["example_id"]),
        )

    ppiref = sorted(
        (record for record in records if record["source"] == "ppiref50k"),
        key=order_key,
    )
    string = sorted(
        (
            record
            for record in records
            if record["source"] == "string_v12_physical"
        ),
        key=order_key,
    )
    if len(ppiref) != len(string) * ppiref_per_group:
        raise RuntimeError("Stage-A1 source ratio differs from the length-bucket gate")
    groups = [
        [
            *ppiref[index * ppiref_per_group : (index + 1) * ppiref_per_group],
            string[index],
        ]
        for index in range(len(string))
    ]
    groups.sort(key=lambda group: max(order_key(record) for record in group))
    block_size = int(efficiency["length_bucket_block_microbatches"])
    blocks = [groups[index : index + block_size] for index in range(0, len(groups), block_size)]
    seed = int(efficiency["selection_seed"])
    blocks.sort(
        key=lambda block: hashlib.sha256(
            (
                f"{seed}\0{block[0][0]['example_id']}\0"
                f"{block[-1][-1]['example_id']}"
            ).encode("utf-8")
        ).digest()
    )
    ordered = [record for block in blocks for group in block for record in group]
    if len(ordered) != len(records) or {
        record["pilot_record_index"] for record in ordered
    } != {record["pilot_record_index"] for record in records}:
        raise RuntimeError("Stage-A1 length bucketing changed manifest membership")
    return ordered


def _structural_bucket_stage_a1_records(
    records: list[dict[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Bucket structural interface pairs without long sequence-only padding."""

    efficiency = protocol["efficiency"]

    def order_key(record: Mapping[str, Any]) -> tuple[int, str]:
        return (
            len(str(record["sequence1"])) + len(str(record["sequence2"])) + 4,
            str(record["example_id"]),
        )

    structural = sorted(
        (
            record
            for record in records
            if record["source"] == "ppiref50k"
            and record["contact_supervision"] is True
            and record["retrieval_supervision"] is True
        ),
        key=order_key,
    )
    if len(structural) != efficiency["expected_structural_records"]:
        raise RuntimeError("Stage-A1 structural bucket count differs")
    block_size = int(efficiency["length_bucket_block_records"])
    blocks = [
        structural[index : index + block_size]
        for index in range(0, len(structural), block_size)
    ]
    seed = int(efficiency["selection_seed"])
    blocks.sort(
        key=lambda block: hashlib.sha256(
            (
                f"{seed}\0{block[0]['example_id']}\0"
                f"{block[-1]['example_id']}"
            ).encode("utf-8")
        ).digest()
    )
    ordered = [record for block in blocks for record in block]
    if len(ordered) != len(structural):
        raise RuntimeError("Stage-A1 structural bucketing changed membership")
    return ordered


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
    root: Path | None,
    *,
    rank: int,
    update: int,
    accumulation: int | None,
    phase: str,
) -> None:
    if root is None:
        return
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


def _head_only_geometry_parameters(heads: StageA1Heads) -> list[nn.Parameter]:
    parameters = [
        parameter
        for name, parameter in heads.named_parameters()
        if name.startswith("contact_")
    ]
    if not parameters:
        raise RuntimeError("Stage-A1 head has no contact parameters")
    return parameters


def _binary_average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    if (
        scores.ndim != 1
        or labels.shape != scores.shape
        or labels.dtype is not torch.bool
        or not bool(labels.any())
    ):
        raise ValueError("Stage-A1 average-precision inputs are malformed")
    order = torch.argsort(scores, descending=True, stable=True)
    ordered_scores = scores[order]
    ranked = labels[order].to(dtype=torch.float64)
    cumulative_positive = ranked.cumsum(0)
    rank = torch.arange(
        1, len(ranked) + 1, device=ranked.device, dtype=torch.float64
    )
    group_end = torch.ones(len(ranked), device=ranked.device, dtype=torch.bool)
    if len(ranked) > 1:
        group_end[:-1] = ordered_scores[:-1] != ordered_scores[1:]
    positive_at_threshold = cumulative_positive[group_end]
    previous_positive = torch.cat(
        [positive_at_threshold.new_zeros(1), positive_at_threshold[:-1]]
    )
    recall_increment = (positive_at_threshold - previous_positive) / ranked.sum()
    precision_at_threshold = positive_at_threshold / rank[group_end]
    return float((recall_increment * precision_at_threshold).sum())


def _pairwise_ranking_accuracy(
    positive_scores: torch.Tensor, negative_scores: torch.Tensor
) -> float:
    if positive_scores.ndim != 1 or negative_scores.ndim != 1:
        raise ValueError("Stage-A1 ranking scores must be one-dimensional")
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        raise ValueError("Stage-A1 ranking accuracy requires both classes")
    differences = positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0)
    return float(
        (differences.gt(0) + 0.5 * differences.eq(0)).to(torch.float64).mean()
    )


def _evaluate_geometry_ranking_holdout(
    *,
    backbone: PairAwareESM2,
    heads: StageA1GeometryHeads,
    holdout_records: list[dict[str, Any]],
    protocol: Mapping[str, Any],
    arguments: argparse.Namespace,
    device: torch.device,
    rank: int,
    world_size: int,
    microbatch_pairs_per_rank: int,
) -> dict[str, float | int]:
    """Evaluate aggregate sampled-contact metrics without publishing predictions."""

    local_records = holdout_records[rank::world_size]
    if len(local_records) < 2:
        raise RuntimeError("Stage-A1 internal holdout shard is too small")
    totals = torch.zeros(12, device=device, dtype=torch.float64)
    heads_was_training = heads.training
    heads.eval()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with torch.no_grad():
        for start in range(0, len(local_records), microbatch_pairs_per_rank):
            selected = local_records[start : start + microbatch_pairs_per_rank]
            if len(selected) < 2:
                raise RuntimeError("Stage-A1 internal holdout tail is too small")
            cpu_batch = build_stage_a1_microbatch(
                selected,
                seed=1701 + rank * 100_000 + start,
                maximum_tokens=arguments.maximum_tokens,
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
            batch = _to_device(cpu_batch, device)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output = backbone(
                    batch.tokens,
                    batch.chain_ids,
                    return_hidden_states=True,
                    compute_logits=False,
                )
                if output.hidden_states is None or batch.contact_distance_bins is None:
                    raise RuntimeError("Stage-A1 holdout geometry view is incomplete")
                (
                    contact_loss,
                    ranking_loss,
                    distance_loss,
                    pair_count,
                    positive_count,
                    complex_count,
                ) = heads.balanced_contact_distance_and_ranking_loss(
                    hidden=output.hidden_states,
                    pair_batch_indices=batch.contact_batch_indices,
                    pair_left_token_indices=batch.contact_left_token_indices,
                    pair_right_token_indices=batch.contact_right_token_indices,
                    distance_bin_targets=batch.contact_distance_bins,
                )
                scores = heads.contact_logits(
                    output.hidden_states,
                    batch.contact_batch_indices,
                    batch.contact_left_token_indices,
                    batch.contact_right_token_indices,
                ).float()
            labels = batch.contact_distance_bins < heads.contact_distance_bin_count
            ap_sum = 0.0
            accuracy_sum = 0.0
            prevalence_sum = 0.0
            for batch_index in torch.unique(batch.contact_batch_indices, sorted=True):
                mask = batch.contact_batch_indices == batch_index
                local_labels = labels[mask]
                local_scores = scores[mask]
                positives = local_scores[local_labels]
                negatives = local_scores[~local_labels]
                ap_sum += _binary_average_precision(local_scores, local_labels)
                accuracy_sum += _pairwise_ranking_accuracy(positives, negatives)
                prevalence_sum += float(local_labels.float().mean())
            totals += torch.tensor(
                [
                    float(contact_loss) * complex_count,
                    float(ranking_loss) * complex_count,
                    float(distance_loss) * pair_count,
                    float(pair_count),
                    float(complex_count),
                    ap_sum,
                    accuracy_sum,
                    prevalence_sum,
                    float(scores[labels].sum()),
                    float(positive_count),
                    float(scores[~labels].sum()),
                    float(pair_count - positive_count),
                ],
                device=device,
                dtype=torch.float64,
            )
    torch.distributed.all_reduce(totals)
    if heads_was_training:
        heads.train()
    pair_count = int(totals[3])
    complex_count = int(totals[4])
    if pair_count < 1 or complex_count != len(holdout_records):
        raise RuntimeError("Stage-A1 internal holdout evaluation count differs")
    return {
        "balanced_contact_bce": float(totals[0] / totals[4]),
        "complex_count": complex_count,
        "distance_cross_entropy": float(totals[2] / totals[3]),
        "macro_sampled_contact_auprc": float(totals[5] / totals[4]),
        "mean_negative_logit": float(totals[10] / totals[11]),
        "mean_positive_logit": float(totals[8] / totals[9]),
        "mean_sampled_positive_prevalence": float(totals[7] / totals[4]),
        "pair_count": pair_count,
        "pairwise_positive_over_negative_accuracy": float(totals[6] / totals[4]),
        "ranking_softplus": float(totals[1] / totals[4]),
    }


def _matched_axis_accuracy(
    scores: torch.Tensor,
    labels: torch.Tensor,
    endpoints: torch.Tensor,
) -> tuple[float, int]:
    """Return accuracy over positive-negative pairs sharing one endpoint."""

    positive = labels
    negative = ~labels
    matched = endpoints[positive, None].eq(endpoints[None, negative])
    if not bool(matched.any()):
        raise RuntimeError("PairContact holdout lacks one matched negative axis")
    differences = scores[positive, None] - scores[None, negative]
    correct = differences.gt(0) + 0.5 * differences.eq(0)
    return float(correct[matched].to(torch.float64).mean()), int(matched.sum())


def _evaluate_paircontact_residual_holdout(
    *,
    backbone: PairAwareESM2,
    heads: StageA1Heads,
    holdout_records: list[dict[str, Any]],
    protocol: Mapping[str, Any],
    arguments: argparse.Namespace,
    device: torch.device,
    rank: int,
    world_size: int,
    microbatch_pairs_per_rank: int,
) -> dict[str, float | int]:
    """Score full, unary, and partner-specific residual views on one frozen holdout."""

    local_records = holdout_records[rank::world_size]
    if len(local_records) < 2:
        raise RuntimeError("PairContact internal holdout shard is too small")
    totals = torch.zeros(17, device=device, dtype=torch.float64)
    heads_was_training = heads.training
    heads.eval()
    contact_protocol = protocol["loss"]["contact"]
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with torch.no_grad():
        for start in range(0, len(local_records), microbatch_pairs_per_rank):
            selected = local_records[start : start + microbatch_pairs_per_rank]
            if len(selected) < 2:
                raise RuntimeError("PairContact internal holdout tail is too small")
            cpu_batch = build_stage_a1_microbatch(
                selected,
                seed=1701 + rank * 100_000 + start,
                maximum_tokens=arguments.maximum_tokens,
                maximum_positive_pairs_per_complex=contact_protocol[
                    "maximum_positive_pairs_per_complex"
                ],
                maximum_sampled_pairs_per_complex=contact_protocol[
                    "maximum_sampled_pairs_per_complex"
                ],
                maximum_negative_to_positive_ratio=contact_protocol[
                    "maximum_negative_to_positive_ratio"
                ],
            )
            batch = _to_device(cpu_batch, device)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output = backbone(
                    batch.tokens,
                    batch.chain_ids,
                    return_hidden_states=True,
                    compute_logits=False,
                )
                if output.hidden_states is None:
                    raise RuntimeError("PairContact holdout backbone view is incomplete")
                objective = heads.contact_objective(
                    hidden=output.hidden_states,
                    pair_batch_indices=batch.contact_batch_indices,
                    pair_left_token_indices=batch.contact_left_token_indices,
                    pair_right_token_indices=batch.contact_right_token_indices,
                    pair_labels=batch.contact_labels,
                    ranking_weight=contact_protocol["ranking_weight"],
                    centering_weight=contact_protocol["centering_weight"],
                    ranking_margin=contact_protocol["ranking_margin"],
                )
                components = heads.contact_components(
                    output.hidden_states,
                    batch.contact_batch_indices,
                    batch.contact_left_token_indices,
                    batch.contact_right_token_indices,
                )
            labels = batch.contact_labels.bool()
            full_scores = components.full_logits.float()
            interaction_scores = components.interaction_logits.float()
            unary_scores = (
                components.left_unary_logits
                + components.right_unary_logits
                + heads.contact_bias
            ).float()
            full_ap_sum = 0.0
            interaction_ap_sum = 0.0
            unary_ap_sum = 0.0
            prevalence_sum = 0.0
            row_accuracy_sum = 0.0
            column_accuracy_sum = 0.0
            row_comparisons = 0
            column_comparisons = 0
            complex_count = 0
            for batch_index in torch.unique(batch.contact_batch_indices, sorted=True):
                mask = batch.contact_batch_indices == batch_index
                local_labels = labels[mask]
                local_full = full_scores[mask]
                local_interaction = interaction_scores[mask]
                full_ap_sum += _binary_average_precision(local_full, local_labels)
                interaction_ap_sum += _binary_average_precision(
                    local_interaction, local_labels
                )
                unary_ap_sum += _binary_average_precision(unary_scores[mask], local_labels)
                prevalence_sum += float(local_labels.float().mean())
                row_accuracy, row_count = _matched_axis_accuracy(
                    local_interaction,
                    local_labels,
                    batch.contact_left_token_indices[mask],
                )
                column_accuracy, column_count = _matched_axis_accuracy(
                    local_interaction,
                    local_labels,
                    batch.contact_right_token_indices[mask],
                )
                row_accuracy_sum += row_accuracy
                column_accuracy_sum += column_accuracy
                row_comparisons += row_count
                column_comparisons += column_count
                complex_count += 1
            pair_count = len(labels)
            positive_count = int(labels.sum())
            totals += torch.tensor(
                [
                    float(objective.bce) * complex_count,
                    float(objective.ranking) * complex_count,
                    float(objective.centering) * complex_count,
                    float(pair_count),
                    float(complex_count),
                    full_ap_sum,
                    interaction_ap_sum,
                    unary_ap_sum,
                    prevalence_sum,
                    row_accuracy_sum,
                    column_accuracy_sum,
                    float(row_comparisons),
                    float(column_comparisons),
                    float(full_scores[labels].sum()),
                    float(positive_count),
                    float(full_scores[~labels].sum()),
                    float(pair_count - positive_count),
                ],
                device=device,
                dtype=torch.float64,
            )
    torch.distributed.all_reduce(totals)
    if heads_was_training:
        heads.train()
    pair_count = int(totals[3])
    complex_count = int(totals[4])
    if pair_count < 1 or complex_count != len(holdout_records):
        raise RuntimeError("PairContact internal holdout evaluation count differs")
    return {
        "centering_penalty": float(totals[2] / totals[4]),
        "column_matched_accuracy": float(totals[10] / totals[4]),
        "column_matched_comparison_count": int(totals[12]),
        "complex_count": complex_count,
        "full_macro_sampled_contact_ap": float(totals[5] / totals[4]),
        "interaction_macro_sampled_contact_ap": float(totals[6] / totals[4]),
        "mean_negative_full_logit": float(totals[15] / totals[16]),
        "mean_positive_full_logit": float(totals[13] / totals[14]),
        "mean_sampled_positive_prevalence": float(totals[8] / totals[4]),
        "pair_count": pair_count,
        "positive_weighted_bce": float(totals[0] / totals[4]),
        "ranking_softplus": float(totals[1] / totals[4]),
        "row_matched_accuracy": float(totals[9] / totals[4]),
        "row_matched_comparison_count": int(totals[11]),
        "unary_macro_sampled_contact_ap": float(totals[7] / totals[4]),
    }


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


def _loss_summary(
    records: list[dict[str, float]], key: str, *, allow_inactive: bool = False
) -> dict[str, float | str | None]:
    phase = {
        "contact_loss": "joint",
        "centering_loss": "joint",
        "distance_loss": "joint",
        "mlm_loss": "joint",
        "ranking_loss": "joint",
        "retrieval_loss": "retrieval",
    }.get(key)
    summary: dict[str, float | str | None] = {
        "final_20_update_mean": _mean_records(records, key, 20),
        "first_20_update_mean": statistics.mean(record[key] for record in records[:20]),
        "active_phase": phase,
    }
    if phase is None:
        summary["active_update_count"] = None
        summary["final_10_active_update_mean"] = None
        summary["final_20_active_update_mean"] = None
        summary["first_10_active_update_mean"] = None
        summary["first_20_active_update_mean"] = None
        return summary
    active = [record for record in records if record.get("active_phase") == phase]
    if not active:
        if allow_inactive and all(record[key] == 0.0 for record in records):
            summary["active_update_count"] = 0
            summary["final_10_active_update_mean"] = None
            summary["final_20_active_update_mean"] = None
            summary["first_10_active_update_mean"] = None
            summary["first_20_active_update_mean"] = None
            return summary
        raise RuntimeError(f"Stage-A1 loss phase is absent: {phase}")
    summary["active_update_count"] = len(active)
    summary["final_10_active_update_mean"] = statistics.mean(
        record[key] for record in active[-10:]
    )
    summary["final_20_active_update_mean"] = statistics.mean(
        record[key] for record in active[-20:]
    )
    summary["first_10_active_update_mean"] = statistics.mean(
        record[key] for record in active[:10]
    )
    summary["first_20_active_update_mean"] = statistics.mean(
        record[key] for record in active[:20]
    )
    return summary


def main() -> None:
    arguments = _arguments()
    protocol = _load_protocol(arguments)
    promotion = protocol.get("format_version") in PROMOTION_PROTOCOL_FORMATS
    head_only_geometry = protocol.get("format_version") in HEAD_ONLY_PROTOCOL_FORMATS
    contact_ranking = protocol.get("format_version") in CONTACT_RANKING_PROTOCOL_FORMATS
    paircontact_residual = (
        protocol.get("format_version") in PAIRCONTACT_RESIDUAL_PROTOCOL_FORMATS
    )
    output_generation = "stage-a1-000000" if promotion else "pilot-000000"
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
    holdout_records: list[dict[str, Any]] = []
    internal_holdout_receipt: dict[str, Any] | None = None
    if contact_ranking:
        records, holdout_records, internal_holdout_receipt = (
            _geometry_ranking_holdout(records, protocol)
        )
        required = (
            protocol["optimization"]["updates"]
            * protocol["systems"]["world_size"]
            * microbatch_pairs_per_rank
        )
        repeats = math.ceil(required / len(records))
        records = (records * repeats)[:required]
    unique_selected_records = len(
        {record["pilot_record_index"] for record in records}
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
            "cuda",
            _replicate_shard_mesh_shape(world_size, arguments.shard_group_size),
            mesh_dim_names=("replicate", "shard"),
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
            backbone,
            mesh=mesh,
            precision="bfloat16",
            reshard_after_forward=head_only_geometry,
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
        if protocol.get("format_version") in GEOMETRY_PROTOCOL_FORMATS:
            contact_geometry = protocol["loss"]["contact"]["geometry_head"]
            heads = StageA1GeometryHeads(
                pair_dimension=contact_geometry["pair_dimension"],
                num_distance_bins=len(
                    protocol["loss"]["contact"]["distance_bin_edges_angstrom"]
                )
                + 1,
                contact_distance_bin_count=contact_geometry[
                    "contact_distance_bin_count"
                ],
                temperature=protocol["loss"]["retrieval"]["temperature"],
            ).to(device)
        else:
            heads = StageA1Heads(
                temperature=protocol["loss"]["retrieval"]["temperature"],
                interaction_residualization=protocol["loss"]["contact"].get(
                    "interaction_residualization", "none"
                ),
            ).to(device)
        system = StageA1System(backbone, heads)
        optimization = protocol["optimization"]
        if head_only_geometry:
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            (
                [
                    {
                        "params": _head_only_geometry_parameters(heads),
                        "lr": optimization["head_learning_rate"],
                    }
                ]
                if head_only_geometry
                else [
                    {
                        "params": list(backbone.parameters()),
                        "lr": optimization["backbone_learning_rate"],
                    },
                    {
                        "params": list(heads.parameters()),
                        "lr": optimization["head_learning_rate"],
                    },
                ]
            ),
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
        if head_only_geometry:
            backbone.eval()
        else:
            backbone.train()
        heads.train()
        torch.cuda.reset_peak_memory_stats(device)
        independent_retrieval_view = (
            protocol.get("format_version") in SPLIT_VIEW_PROTOCOL_FORMATS
        )
        update_records: list[dict[str, float]] = []
        from torch.nn.attention import SDPBackend, sdpa_kernel

        phase_root = (
            None
            if arguments.rank_phase_telemetry == "off"
            else arguments.result.parent / "rank_phases"
        )
        internal_holdout_before = (
            (
                _evaluate_paircontact_residual_holdout
                if paircontact_residual
                else _evaluate_geometry_ranking_holdout
            )(
                backbone=backbone,
                heads=heads,
                holdout_records=holdout_records,
                protocol=protocol,
                arguments=arguments,
                device=device,
                rank=rank,
                world_size=world_size,
                microbatch_pairs_per_rank=microbatch_pairs_per_rank,
            )
            if contact_ranking
            else None
        )
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
            totals = torch.zeros(12, device=device, dtype=torch.float64)
            minimum_allowed_candidates = torch.tensor(
                0.0 if head_only_geometry else float("inf"),
                device=device,
                dtype=torch.float64,
            )
            active_phase = (
                "joint"
                if head_only_geometry or (alternating_efficiency and update % 2 == 0)
                else "retrieval"
            )
            active_head_prefixes: tuple[str, ...] | None = None
            for accumulation in range(optimization["accumulation_steps"]):
                distance_loss_for_record = torch.zeros((), device=device)
                ranking_loss_for_record = torch.zeros((), device=device)
                centering_loss_for_record = torch.zeros((), device=device)
                microstep = update * optimization["accumulation_steps"] + accumulation
                start = (
                    microstep * world_size + rank
                ) * microbatch_pairs_per_rank
                cpu_batch = build_stage_a1_microbatch(
                    records[start : start + microbatch_pairs_per_rank],
                    seed=1701 + microstep * world_size + rank,
                    maximum_tokens=arguments.maximum_tokens,
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
                            if head_only_geometry:
                                with torch.no_grad():
                                    output = backbone(
                                        batch.tokens,
                                        batch.chain_ids,
                                        return_hidden_states=True,
                                        compute_logits=False,
                                    )
                                if output.hidden_states is None:
                                    raise RuntimeError(
                                        "Stage-A1 frozen geometry view is incomplete"
                                    )
                                mlm_loss = output.hidden_states.new_zeros(())
                            else:
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
                            assert output.hidden_states is not None
                            if paircontact_residual:
                                contact_objective = heads.contact_objective(
                                    hidden=output.hidden_states,
                                    pair_batch_indices=batch.contact_batch_indices,
                                    pair_left_token_indices=(
                                        batch.contact_left_token_indices
                                    ),
                                    pair_right_token_indices=(
                                        batch.contact_right_token_indices
                                    ),
                                    pair_labels=batch.contact_labels,
                                    ranking_weight=protocol["loss"]["contact"][
                                        "ranking_weight"
                                    ],
                                    centering_weight=protocol["loss"]["contact"][
                                        "centering_weight"
                                    ],
                                    ranking_margin=protocol["loss"]["contact"][
                                        "ranking_margin"
                                    ],
                                )
                                contact_loss = contact_objective.bce
                                ranking_loss_for_record = contact_objective.ranking
                                centering_loss_for_record = contact_objective.centering
                                contact_count = contact_objective.pair_count
                                contact_positive_count = (
                                    contact_objective.positive_count
                                )
                            elif protocol.get("format_version") in GEOMETRY_PROTOCOL_FORMATS:
                                if (
                                    not isinstance(heads, StageA1GeometryHeads)
                                    or batch.contact_distance_bins is None
                                ):
                                    raise RuntimeError(
                                        "Stage-A1 geometry head or labels are missing"
                                    )
                                if contact_ranking:
                                    (
                                        contact_loss,
                                        ranking_loss_for_record,
                                        distance_loss_for_record,
                                        contact_count,
                                        contact_positive_count,
                                        _,
                                    ) = heads.balanced_contact_distance_and_ranking_loss(
                                        hidden=output.hidden_states,
                                        pair_batch_indices=(
                                            batch.contact_batch_indices
                                        ),
                                        pair_left_token_indices=(
                                            batch.contact_left_token_indices
                                        ),
                                        pair_right_token_indices=(
                                            batch.contact_right_token_indices
                                        ),
                                        distance_bin_targets=(
                                            batch.contact_distance_bins
                                        ),
                                    )
                                else:
                                    (
                                        contact_loss,
                                        distance_loss_for_record,
                                        contact_count,
                                        contact_positive_count,
                                    ) = heads.contact_and_distance_loss(
                                        hidden=output.hidden_states,
                                        pair_batch_indices=(
                                            batch.contact_batch_indices
                                        ),
                                        pair_left_token_indices=(
                                            batch.contact_left_token_indices
                                        ),
                                        pair_right_token_indices=(
                                            batch.contact_right_token_indices
                                        ),
                                        distance_bin_targets=(
                                            batch.contact_distance_bins
                                        ),
                                    )
                            else:
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
                                + protocol["loss"]["contact"].get(
                                    "distance_bin_weight", 0.0
                                )
                                * distance_loss_for_record
                                + protocol["loss"]["contact"].get(
                                    "ranking_weight", 0.0
                                )
                                * ranking_loss_for_record
                                + protocol["loss"]["contact"].get(
                                    "centering_weight", 0.0
                                )
                                * centering_loss_for_record
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
                            contact_bce=contact_loss,
                            contact_ranking=ranking_loss_for_record,
                            contact_centering=centering_loss_for_record,
                            retrieval_accuracy_left_to_right=zero,
                            retrieval_accuracy_right_to_left=zero,
                            contact_count=contact_count,
                            contact_positive_count=contact_positive_count,
                            contact_row_comparison_count=(
                                contact_objective.row_comparison_count
                                if paircontact_residual
                                else 0
                            ),
                            contact_column_comparison_count=(
                                contact_objective.column_comparison_count
                                if paircontact_residual
                                else 0
                            ),
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
                            contact_bce=zero,
                            contact_ranking=zero,
                            contact_centering=zero,
                            retrieval_accuracy_left_to_right=(
                                retrieval_accuracy_left_to_right
                            ),
                            retrieval_accuracy_right_to_left=(
                                retrieval_accuracy_right_to_left
                            ),
                            contact_count=0,
                            contact_positive_count=0,
                            contact_row_comparison_count=0,
                            contact_column_comparison_count=0,
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
                        contact_bce=contact_loss,
                        contact_ranking=contact_loss.new_zeros(()),
                        contact_centering=contact_loss.new_zeros(()),
                        retrieval_accuracy_left_to_right=(
                            retrieval_accuracy_left_to_right
                        ),
                        retrieval_accuracy_right_to_left=(
                            retrieval_accuracy_right_to_left
                        ),
                        contact_count=contact_count,
                        contact_positive_count=contact_positive_count,
                        contact_row_comparison_count=0,
                        contact_column_comparison_count=0,
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
                if not alternating_efficiency or active_phase == "retrieval":
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
                        float(distance_loss_for_record.detach()),
                        float(ranking_loss_for_record.detach()),
                        float(centering_loss_for_record.detach()),
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
            totals[9] /= world_size * optimization["accumulation_steps"]
            totals[10] /= world_size * optimization["accumulation_steps"]
            totals[11] /= world_size * optimization["accumulation_steps"]
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            approximate_useful_token_mfu = None
            if alternating_efficiency:
                approximate_useful_token_mfu = (
                    (2.0 if head_only_geometry else 6.0)
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
                "approximate_backbone_forward_2n_mfu": (
                    approximate_useful_token_mfu if head_only_geometry else None
                ),
                "approximate_useful_token_6n_mfu": (
                    None if head_only_geometry else approximate_useful_token_mfu
                ),
                "contact_loss": float(totals[3]),
                "distance_loss": float(totals[9]),
                "ranking_loss": float(totals[10]),
                "centering_loss": float(totals[11]),
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
                        "format_version": (
                            "abpairfm_stage_a1_promotion_progress_v1"
                            if promotion
                            else "abpairfm_stage_a1_pilot_progress_v1"
                        ),
                        "latest": record,
                        "loss": (
                            {
                                name: _loss_summary(
                                    update_records,
                                    name,
                                    allow_inactive=name
                                    in {"mlm_loss", "retrieval_loss"},
                                )
                                for name in (
                                    "total_loss",
                                    "mlm_loss",
                                    "retrieval_loss",
                                    "contact_loss",
                                    "distance_loss",
                                    *(("ranking_loss",) if contact_ranking else ()),
                                    *(("centering_loss",) if paircontact_residual else ()),
                                )
                            }
                            if head_only_geometry
                            else None
                        ),
                        "status": (
                            "RUNNING"
                            if update + 1 < optimization["updates"]
                            else "TRAINING_COMPLETE_CHECKPOINT_PENDING"
                        ),
                        "test_access": False,
                        "update": update + 1,
                    },
                )
        internal_holdout_after = (
            (
                _evaluate_paircontact_residual_holdout
                if paircontact_residual
                else _evaluate_geometry_ranking_holdout
            )(
                backbone=backbone,
                heads=heads,
                holdout_records=holdout_records,
                protocol=protocol,
                arguments=arguments,
                device=device,
                rank=rank,
                world_size=world_size,
                microbatch_pairs_per_rank=microbatch_pairs_per_rank,
            )
            if contact_ranking
            else None
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
            output_generation,
            model=system,
            optimizer=optimizer,
            provenance_hashes=provenance,
            training_state={
                "input_consumed_residue_tokens": input_receipt["training_state"][
                    "consumed_residue_tokens"
                ],
                "production_training": promotion,
                "stage_a1_updates": optimization["updates"],
                "test_access": False,
            },
        )
        restore_receipt = _DCP.load_dcp_generation(
            arguments.checkpoint_root,
            output_generation,
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
        elif promotion:
            status = "PASS_STAGE_A1_LENGTH_BAND_PROMOTION_CHECKPOINT_VERIFIED"
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
                    "generation": output_generation,
                    "provenance_hashes": provenance,
                    "restore_content_sha256": restore_receipt["content_sha256"],
                    "root": str(arguments.checkpoint_root.resolve()),
                },
                "claim_boundary": protocol["claim_boundary"],
                "format_version": (
                    "abpairfm_stage_a1_promotion_result_v1"
                    if promotion
                    else "abpairfm_stage_a1_multitask_pilot_result_v1"
                ),
                "training_protocol_format_version": protocol["format_version"],
                "input_checkpoint": {
                    "consumed_residue_tokens": input_receipt["training_state"][
                        "consumed_residue_tokens"
                    ],
                    "content_sha256": arguments.expected_input_content_sha256,
                    "generation": arguments.input_generation,
                },
                "initialization": initialization,
                "internal_holdout": (
                    {
                        "after_training": internal_holdout_after,
                        "before_training": internal_holdout_before,
                        "receipt": internal_holdout_receipt,
                        "test_access": False,
                    }
                    if contact_ranking
                    else None
                ),
                "loss": {
                    name: _loss_summary(
                        update_records,
                        name,
                        allow_inactive=(
                            head_only_geometry
                            and name in {"mlm_loss", "retrieval_loss"}
                        ),
                    )
                    for name in (
                        "total_loss",
                        "mlm_loss",
                        "retrieval_loss",
                        "contact_loss",
                        "distance_loss",
                        *(("ranking_loss",) if contact_ranking else ()),
                        *(("centering_loss",) if paircontact_residual else ()),
                    )
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
                    "unique_records_available": unique_selected_records,
                },
                "memory": {
                    "maximum_allocated_fraction": peak_fraction,
                    "required_maximum_fraction": protocol["promotion_gates"][
                        "maximum_peak_memory_fraction"
                    ],
                },
                "optimization": {
                    **optimization,
                    "progress_interval": arguments.progress_interval,
                    "rank_phase_telemetry": arguments.rank_phase_telemetry,
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
                    "approximate_backbone_forward_2n_mfu_mean": (
                        statistics.mean(
                            record["approximate_backbone_forward_2n_mfu"]
                            for record in update_records
                        )
                        if head_only_geometry
                        else None
                    ),
                    "approximate_useful_token_6n_mfu_mean": (
                        statistics.mean(
                            record["approximate_useful_token_6n_mfu"]
                            for record in update_records
                            if record["approximate_useful_token_6n_mfu"] is not None
                        )
                        if alternating_efficiency and not head_only_geometry
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
                        if alternating_efficiency and not head_only_geometry
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
