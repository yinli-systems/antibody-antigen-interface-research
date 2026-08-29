#!/usr/bin/env python3
"""Train one pre-registered FrameContact seed on development roles only."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.attention_diagnostics import (  # noqa: E402
    AttentionDiagnosticsAccumulator,
    CrossAttentionDiagnosticsAccumulator,
)
from interaction_first.asep_external import AsepDevelopmentDataset  # noqa: E402
from interaction_first.framecontact import FrameContactModel  # noqa: E402
from interaction_first.framecontact_metrics import (  # noqa: E402
    binary_classification_metrics,
    partner_retrieval_metrics,
    ranked_pair_metrics,
)
from interaction_first.framecontact_objective import (  # noqa: E402
    FrameContactLossWeights,
    FrameContactObjective,
    ValidationTemperatureScaler,
    top_fraction_contact_score,
)
from interaction_first.partner_panel import PARTNER_PANEL_FORMAT  # noqa: E402
from interaction_first.reciprocal_partner_panel import (  # noqa: E402
    RECIPROCAL_PARTNER_PANEL_FORMAT,
)
from interaction_first.sabdab2_contacts import (  # noqa: E402
    Sabdab2ContactBatchCollator,
    Sabdab2ContactDataset,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    Sabdab2StructureBatchCollator,
    Sabdab2StructureEmbeddingDataset,
    canonical_json_bytes,
    sha256_file,
)
from interaction_first.source_manifest import verify_source_manifest  # noqa: E402


RESULT_FORMAT = "framecontact_same_protocol_adapter_development_v2"
EXPERIMENTS = {
    "sequence_dense": {
        "geometry_feature_mode": "none",
        "geometry_bias_mode": "pair_mlp",
        "geometry": False,
        "cross": False,
        "fusion_mode": "late",
    },
    "spatial_topology": {
        "geometry_feature_mode": "none",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": False,
        "fusion_mode": "late",
    },
    "distance": {
        "geometry_feature_mode": "distance",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": False,
        "fusion_mode": "late",
    },
    "distance_direction": {
        "geometry_feature_mode": "distance_direction",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": False,
        "fusion_mode": "late",
    },
    "full_pose": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": False,
        "fusion_mode": "late",
    },
    "frozen_frame_ipa": {
        "geometry_feature_mode": "none",
        "geometry_bias_mode": "invariant_points",
        "geometry": True,
        "cross": False,
        "fusion_mode": "late",
    },
    "framecontact": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
    },
    "framecontact_interleaved": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "interleaved",
        "cross_attention_contact_gate_mode": "learned",
    },
    "framecontact_unconditioned_interleaved": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "interleaved",
        "cross_attention_contact_gate_mode": "content_only",
    },
    "framecontact_distance_aux": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "interleaved",
        "cross_attention_contact_gate_mode": "content_only",
        "adapter_mode": "framecontact",
        "distance_bin_count": 5,
    },
    "epiformer_style_same_protocol": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "interleaved",
        "cross_attention_contact_gate_mode": "content_only",
        "adapter_mode": "epiformer_style_same_protocol",
        "distance_bin_count": 5,
    },
    "abpairfm_epiformer_style_same_protocol": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "interleaved",
        "cross_attention_contact_gate_mode": "content_only",
        "adapter_mode": "epiformer_style_same_protocol",
        "distance_bin_count": 5,
    },
    "abpairfm_epiformer_residual_fusion_same_protocol": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "interleaved",
        "cross_attention_contact_gate_mode": "content_only",
        "adapter_mode": "epiformer_style_same_protocol",
        "distance_bin_count": 5,
    },
    "mipe_style_same_protocol": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
        "cross_attention_contact_gate_mode": "content_only",
        "adapter_mode": "mipe_style_same_protocol",
        "distance_bin_count": 0,
    },
}
for _experiment in EXPERIMENTS.values():
    _experiment.setdefault("cross_attention_contact_gate_mode", "learned")
    _experiment.setdefault("adapter_mode", "framecontact")
    _experiment.setdefault("distance_bin_count", 0)


def _sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-backend", choices=("sabdab2", "asep"), default="sabdab2"
    )
    parser.add_argument("--structure-cache-root", type=Path)
    parser.add_argument("--contact-cache-root", type=Path)
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--embedding-manifest-sha256")
    parser.add_argument("--joint-embedding-root", type=Path)
    parser.add_argument("--joint-embedding-manifest-sha256")
    parser.add_argument("--joint-embedding-source-split-sha256")
    parser.add_argument("--matched-local-root", type=Path)
    parser.add_argument("--matched-local-manifest-sha256")
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split-manifest-sha256")
    parser.add_argument("--structure-manifest-sha256")
    parser.add_argument("--contact-manifest-sha256")
    parser.add_argument("--asep-cache-root", type=Path)
    parser.add_argument("--asep-cache-manifest-sha256")
    parser.add_argument(
        "--asep-feature-mode",
        choices=("official_esm2_35m", "official_igfold_esm2"),
        default="official_esm2_35m",
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--protocol-receipt-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--training-partner-panel", type=Path, required=True)
    parser.add_argument("--training-partner-panel-sha256", required=True)
    parser.add_argument("--validation-partner-panel", type=Path, required=True)
    parser.add_argument("--validation-partner-panel-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument(
        "--epitope-pooling",
        choices=("topk_mean", "noisy_or", "logmeanexp", "learned_softmax"),
        default="topk_mean",
    )
    parser.add_argument("--epitope-pooling-k", type=int, default=2)
    parser.add_argument(
        "--paratope-pooling",
        choices=("logmeanexp", "topk_mean"),
        default="logmeanexp",
    )
    parser.add_argument("--paratope-pooling-k", type=int, default=2)
    parser.add_argument(
        "--contact-readout",
        choices=("pair", "topk_context_equal_logit"),
        default="pair",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-self-layers", type=int, default=2)
    parser.add_argument("--num-rbf", type=int, default=16)
    parser.add_argument("--neighbor-k", type=int, default=32)
    parser.add_argument("--neighbor-radius", type=float, default=16.0)
    parser.add_argument("--sequence-window", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contact-loss-weight", type=float, default=1.0)
    parser.add_argument("--paratope-loss-weight", type=float, default=0.25)
    parser.add_argument("--epitope-loss-weight", type=float, default=0.25)
    parser.add_argument("--specificity-loss-weight", type=float, default=0.25)
    parser.add_argument("--specificity-margin", type=float, default=0.05)
    parser.add_argument("--score-fraction", type=float, default=0.01)
    parser.add_argument("--contact-focal-gamma", type=float, default=0.0)
    parser.add_argument("--epitope-dice-weight", type=float, default=0.0)
    parser.add_argument("--epitope-count-weight", type=float, default=0.0)
    parser.add_argument("--distance-loss-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--matched-local-loss-weight", type=float, default=0.0)
    parser.add_argument("--matched-local-margin", type=float, default=0.05)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-maximum-residues", type=int, default=256)
    parser.add_argument("--contrastive-dim", type=int, default=64)
    parser.add_argument("--calibration-max-pairs", type=int, default=2_000_000)
    parser.add_argument(
        "--offload-saved-tensors",
        action="store_true",
        help="Move autograd-saved tensors to CPU until backward for memory safety.",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _autocast(device: torch.device, use_bf16: bool) -> Any:
    if use_bf16:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _saved_tensor_context(enabled: bool) -> Any:
    return torch.autograd.graph.save_on_cpu(pin_memory=False) if enabled else nullcontext()


def _cognate_training_view(output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: output[key]
        for key in (
            "contact_logits",
            "contact_interaction_logits",
            "pair_candidate_mask",
            "paratope_logits",
            "epitope_logits",
            "antibody_mask",
            "antigen_mask",
            "distance_bin_logits",
            "contrastive_views",
        )
    }


def _wrong_training_view(output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: output[key]
        for key in (
            "contact_logits",
            "contact_interaction_logits",
            "pair_candidate_mask",
        )
    }


def _authenticated_panel(
    path: Path,
    expected_sha256: str,
    *,
    role: str,
    expected_size: int,
    expected_format: str,
) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{role} partner panel SHA-256 differs")
    panel = json.loads(path.read_text(encoding="utf-8"))
    if (
        panel.get("format_version") != expected_format
        or panel.get("role") != role
        or panel.get("authority", {}).get("task_label_columns_read") is not False
    ):
        raise ValueError(f"{role} partner panel is not an eligible frozen authority")
    size_field = (
        "panel_size_per_direction"
        if expected_format == RECIPROCAL_PARTNER_PANEL_FORMAT
        else "panel_size"
    )
    if int(panel.get(size_field, -1)) != expected_size:
        raise ValueError(f"{role} partner panel size differs from the protocol")
    return panel


def _load_authorities(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sabdab_values = (
        arguments.structure_cache_root,
        arguments.contact_cache_root,
        arguments.embedding_root,
        arguments.embedding_manifest_sha256,
        arguments.structure_manifest_sha256,
        arguments.contact_manifest_sha256,
    )
    joint_values = (
        arguments.joint_embedding_root,
        arguments.joint_embedding_manifest_sha256,
        arguments.joint_embedding_source_split_sha256,
    )
    asep_values = (
        arguments.asep_cache_root,
        arguments.asep_cache_manifest_sha256,
    )
    if arguments.dataset_backend == "sabdab2":
        if any(value is None for value in sabdab_values):
            raise ValueError("SAbDab2 backend requires structure/contact/embedding authorities")
        if any(value is not None for value in asep_values):
            raise ValueError("SAbDab2 backend may not receive AsEP cache arguments")
        if any(value is None for value in joint_values) != all(
            value is None for value in joint_values
        ):
            raise ValueError("Joint embedding root and SHA-256 must be provided together")
        structure_manifest = (
            arguments.structure_cache_root / "STRUCTURE_CACHE_MANIFEST.json"
        )
        contact_manifest = arguments.contact_cache_root / "CONTACT_CACHE_MANIFEST.json"
        if sha256_file(structure_manifest) != arguments.structure_manifest_sha256:
            raise ValueError("Structure cache manifest SHA-256 differs")
        if sha256_file(contact_manifest) != arguments.contact_manifest_sha256:
            raise ValueError("Contact cache manifest SHA-256 differs")
        embedding_manifest = (
            arguments.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
        )
        if sha256_file(embedding_manifest) != arguments.embedding_manifest_sha256:
            raise ValueError("ESM embedding manifest SHA-256 differs")
        if arguments.experiment in {
            "abpairfm_epiformer_style_same_protocol",
            "abpairfm_epiformer_residual_fusion_same_protocol",
        }:
            if any(value is None for value in joint_values):
                raise ValueError("ABPairFM adapter requires joint embedding authority")
            joint_manifest_path = (
                arguments.joint_embedding_root
                / "SABDAB2_ABPAIRFM_JOINT_EMBEDDING_MANIFEST.json"
            )
            if (
                sha256_file(joint_manifest_path)
                != arguments.joint_embedding_manifest_sha256
            ):
                raise ValueError("ABPairFM joint embedding manifest SHA-256 differs")
            joint_manifest = json.loads(joint_manifest_path.read_text(encoding="utf-8"))
            if (
                joint_manifest.get("format_version")
                != "sabdab2_abpairfm_joint_embedding_manifest_v1"
                or joint_manifest.get("status") != "COMPLETE_TRAIN_VALIDATION_ONLY"
                or joint_manifest.get("test_access") is not False
                or joint_manifest.get("authority", {}).get("task_labels_read") is not False
                or joint_manifest.get("authority", {}).get("test_loaded") is not False
                or joint_manifest.get("authority", {}).get("split_sha256")
                != arguments.joint_embedding_source_split_sha256
                or joint_manifest.get("role_counts")
                != {"train": 2565, "validation": 549}
            ):
                raise ValueError("ABPairFM joint embedding authority differs")
        elif any(value is not None for value in joint_values):
            raise ValueError("Only the ABPairFM adapter may receive joint embeddings")
    else:
        if any(value is not None for value in sabdab_values + joint_values):
            raise ValueError("AsEP backend may not receive SAbDab2 cache arguments")
        if any(
            value is not None
            for value in (
                arguments.label_root,
                arguments.split_manifest,
                arguments.split_manifest_sha256,
            )
        ):
            raise ValueError("AsEP backend may not receive SAbDab2 label/split arguments")
        if any(value is None for value in asep_values):
            raise ValueError("AsEP backend requires its cache root and manifest SHA-256")
        asep_manifest = arguments.asep_cache_root / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"
        if sha256_file(asep_manifest) != arguments.asep_cache_manifest_sha256:
            raise ValueError("AsEP cache manifest SHA-256 differs")
    if sha256_file(arguments.protocol_receipt) != arguments.protocol_receipt_sha256:
        raise ValueError("Protocol receipt SHA-256 differs")
    receipt = json.loads(arguments.protocol_receipt.read_text(encoding="utf-8"))
    protocol_raw = arguments.protocol.read_bytes()
    if hashlib.sha256(protocol_raw).hexdigest() != receipt.get("protocol_file_sha256"):
        raise ValueError("Protocol does not match its frozen receipt")
    protocol = json.loads(protocol_raw)
    protocol_backend = protocol.get("data", {}).get("dataset_backend", "sabdab2")
    if protocol_backend != arguments.dataset_backend:
        raise ValueError("Dataset backend differs from the pre-registered protocol")
    expected_feature_mode = protocol.get("data", {}).get(
        "input_feature_mode", "external_embedding_cache"
    )
    if arguments.joint_embedding_root is None:
        observed_feature_mode = "external_embedding_cache"
    elif arguments.experiment == "abpairfm_epiformer_residual_fusion_same_protocol":
        observed_feature_mode = "abpairfm_joint_residual_concatenation_v1"
    else:
        observed_feature_mode = "abpairfm_joint_cross_chain_v1"
    if expected_feature_mode != observed_feature_mode:
        raise ValueError("Input feature mode differs from the pre-registered protocol")
    matched_weight = float(
        protocol.get("training_objective", {})
        .get("loss_weights", {})
        .get("matched_local", 0.0)
    )
    matched_values = (
        arguments.matched_local_root,
        arguments.matched_local_manifest_sha256,
    )
    if matched_weight > 0:
        if arguments.dataset_backend != "sabdab2" or any(
            value is None for value in matched_values
        ):
            raise ValueError("Matched-local training requires its SAbDab2 authority")
        matched_path = (
            arguments.matched_local_root / "MATCHED_LOCAL_DECOY_MANIFEST.json"
        )
        if sha256_file(matched_path) != arguments.matched_local_manifest_sha256:
            raise ValueError("Matched-local manifest SHA-256 differs")
        matched = json.loads(matched_path.read_text(encoding="utf-8"))
        matched_authority = matched.get("authority", {})
        if (
            matched_authority.get("split_sha256") != arguments.split_manifest_sha256
            or matched_authority.get("task_labels_read_for_roles") != ["train"]
            or matched_authority.get("validation_labels_read") is not False
            or matched_authority.get("test_labels_read") is not False
            or matched_authority.get("model_predictions_read") is not False
            or int(matched.get("counts", {}).get("train_instances", -1)) != 512
        ):
            raise ValueError("Matched-local authority differs from the train-only contract")
    elif any(value is not None for value in matched_values):
        raise ValueError("Control training may not receive matched-local artifacts")
    if protocol.get("tasks", {}).get("winner", {}).get("experiment") != arguments.experiment:
        raise ValueError("Experiment differs from the pre-registered protocol")
    if arguments.seed not in protocol.get("model_selection", {}).get("seeds", []):
        raise ValueError("Seed was not pre-registered")
    if int(protocol["model_selection"]["maximum_epochs"]) != arguments.epochs:
        raise ValueError("Epoch count differs from the pre-registered maximum")
    if (
        int(protocol["model_selection"]["early_stopping_patience"])
        != arguments.early_stopping_patience
    ):
        raise ValueError("Early-stopping patience differs from the protocol")
    _validate_execution_configuration(protocol, arguments)
    panel_protocol = protocol["data"].get(
        "reciprocal_partner_panel", protocol["data"].get("negative_partner_panel")
    )
    if not isinstance(panel_protocol, dict):
        raise ValueError("Protocol lacks a partner-panel definition")
    panel_size = int(panel_protocol["size_per_query"])
    panel_format = str(panel_protocol.get("format_version", PARTNER_PANEL_FORMAT))
    if panel_format not in {PARTNER_PANEL_FORMAT, RECIPROCAL_PARTNER_PANEL_FORMAT}:
        raise ValueError("Protocol requests an unsupported partner-panel format")
    training_panel = _authenticated_panel(
        arguments.training_partner_panel,
        arguments.training_partner_panel_sha256,
        role="train",
        expected_size=panel_size,
        expected_format=panel_format,
    )
    validation_panel = _authenticated_panel(
        arguments.validation_partner_panel,
        arguments.validation_partner_panel_sha256,
        role="validation",
        expected_size=panel_size,
        expected_format=panel_format,
    )
    return protocol, training_panel, validation_panel


def _validate_execution_configuration(
    protocol: Mapping[str, Any], arguments: argparse.Namespace
) -> None:
    """Fail closed when an executable hyperparameter drifts from the protocol."""
    implementation = protocol.get("implementation")
    if implementation is None:
        return
    if not isinstance(implementation, Mapping):
        raise ValueError("Protocol implementation block is invalid")
    fields = {
        "d_model": "d_model",
        "num_heads": "num_heads",
        "num_self_layers": "num_self_layers",
        "num_rbf": "num_rbf",
        "neighbor_k": "neighbor_k",
        "neighbor_radius": "neighbor_radius_angstrom",
        "sequence_window": "sequence_window",
        "dropout": "dropout",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "batch_size": "batch_size",
        "evaluation_batch_size": "evaluation_batch_size",
        "precision": "precision",
        "calibration_max_pairs": "calibration_max_pairs",
        "offload_saved_tensors": "offload_saved_tensors",
        "contact_focal_gamma": "contact_focal_gamma",
        "epitope_dice_weight": "epitope_dice_weight",
        "epitope_count_weight": "epitope_count_weight",
    }
    for argument_name, protocol_name in fields.items():
        observed = getattr(arguments, argument_name)
        expected = implementation.get(protocol_name)
        if isinstance(expected, float):
            equal = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            equal = observed == expected
        if not equal:
            raise ValueError(
                f"Execution parameter {argument_name} differs from the protocol"
            )
    objective = protocol.get("training_objective", {})
    loss_weights = objective.get("loss_weights", {})
    for argument_name, protocol_name in (
        ("contact_loss_weight", "contact"),
        ("paratope_loss_weight", "paratope"),
        ("epitope_loss_weight", "epitope"),
        ("specificity_loss_weight", "specificity"),
        ("matched_local_loss_weight", "matched_local"),
    ):
        if not math.isclose(
            float(getattr(arguments, argument_name)),
            float(loss_weights.get(protocol_name, math.nan)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Loss weight {argument_name} differs from the protocol")
    for argument_name, protocol_name in (
        ("specificity_margin", "specificity_margin"),
        ("score_fraction", "score_fraction"),
        ("matched_local_margin", "matched_local_margin"),
    ):
        if not math.isclose(
            float(getattr(arguments, argument_name)),
            float(objective.get(protocol_name, math.nan)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Objective parameter {argument_name} differs from the protocol"
            )


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        collate_fn=Sabdab2ContactBatchCollator(),
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


class _JointEmbeddingOverrideDataset(torch.utils.data.Dataset):
    """Use authenticated per-complex ABPairFM states alone or with frozen ESM states."""

    def __init__(
        self,
        base_dataset: Sabdab2StructureEmbeddingDataset,
        joint_root: Path,
        manifest_sha256: str,
        *,
        split_role: str,
        concatenate_base: bool,
    ) -> None:
        self.base_dataset = base_dataset
        self.cache_root = base_dataset.cache_root
        self.records = base_dataset.records
        manifest_path = joint_root / "SABDAB2_ABPAIRFM_JOINT_EMBEDDING_MANIFEST.json"
        if sha256_file(manifest_path) != manifest_sha256:
            raise ValueError("ABPairFM joint embedding manifest SHA-256 differs")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_id = {
            str(record["instance_id"]): record for record in manifest.get("records", [])
        }
        expected_ids = {str(record["instance_id"]) for record in self.records}
        selected_ids = {
            identifier
            for identifier, record in by_id.items()
            if record.get("split_role") == split_role
        }
        if not expected_ids.issubset(selected_ids):
            raise ValueError("ABPairFM joint embedding IDs are missing from the selected split")
        self.joint_root = joint_root
        self.joint_by_id = {identifier: by_id[identifier] for identifier in expected_ids}
        self.checkpoint_sha256 = str(manifest["checkpoint"]["content_sha256"])
        self.concatenate_base = bool(concatenate_base)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.base_dataset[index])
        identifier = str(item["instance_id"])
        record = self.joint_by_id[identifier]
        path = self.joint_root / str(record["artifact_path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != record["artifact_sha256"]
        ):
            raise ValueError("ABPairFM joint embedding artifact differs")
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        antibody = artifact.get("antibody_features")
        antigen = artifact.get("antigen_features")
        source_record = self.records[index]
        if (
            artifact.get("format_version") != "sabdab2_abpairfm_joint_embedding_v1"
            or artifact.get("instance_id") != identifier
            or artifact.get("checkpoint_content_sha256") != self.checkpoint_sha256
            or artifact.get("antibody_sequence_sha256")
            != source_record["antibody_sequence_sha256"]
            or artifact.get("antigen_sequence_sha256")
            != source_record["antigen_sequence_sha256"]
            or not torch.is_tensor(antibody)
            or not torch.is_tensor(antigen)
            or tuple(antibody.shape)
            != (int(source_record["antibody_length"]), 2560)
            or tuple(antigen.shape)
            != (int(source_record["antigen_length"]), 2560)
            or not bool(torch.isfinite(antibody).all())
            or not bool(torch.isfinite(antigen).all())
        ):
            raise ValueError("ABPairFM joint embedding artifact metadata differs")
        if self.concatenate_base:
            base_antibody = item.get("antibody_features")
            base_antigen = item.get("antigen_features")
            if (
                not torch.is_tensor(base_antibody)
                or not torch.is_tensor(base_antigen)
                or tuple(base_antibody.shape)
                != (int(source_record["antibody_length"]), 1280)
                or tuple(base_antigen.shape)
                != (int(source_record["antigen_length"]), 1280)
                or not bool(torch.isfinite(base_antibody).all())
                or not bool(torch.isfinite(base_antigen).all())
            ):
                raise ValueError("Frozen base ESM feature authority differs")
            item["antibody_features"] = torch.cat(
                (base_antibody.float(), antibody.float()), dim=-1
            )
            item["antigen_features"] = torch.cat(
                (base_antigen.float(), antigen.float()), dim=-1
            )
        else:
            item["antibody_features"] = antibody.float()
            item["antigen_features"] = antigen.float()
        return item


class _MatchedLocalTrainingDataset(torch.utils.data.Dataset):
    """Attach authenticated train-only local motif pairs to each cognate complex."""

    def __init__(
        self,
        dataset: Sabdab2ContactDataset,
        artifact_root: Path,
        manifest_sha256: str,
    ) -> None:
        self.dataset = dataset
        self.records = dataset.records
        self.artifact_root = artifact_root
        manifest_path = artifact_root / "MATCHED_LOCAL_DECOY_MANIFEST.json"
        if sha256_file(manifest_path) != manifest_sha256:
            raise ValueError("Matched-local manifest SHA-256 differs")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_id = {
            str(record["instance_id"]): record
            for record in manifest.get("records", [])
        }
        expected_ids = {str(record["instance_id"]) for record in self.records}
        if set(by_id) != expected_ids:
            raise ValueError("Matched-local records differ from the training dataset")
        self.decoy_by_id = by_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        item = self.dataset[index]
        identifier = str(item["instance_id"])
        record = self.decoy_by_id[identifier]
        path = self.artifact_root / str(record["artifact_path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != int(record["artifact_size_bytes"])
            or sha256_file(path) != record["artifact_sha256"]
        ):
            raise ValueError("Matched-local artifact differs")
        with np.load(path, allow_pickle=False) as payload:
            decoy_count = int(payload["decoy_count"].item())
            antigen_length = int(payload["antigen_length"].item())
            if (
                str(payload["instance_id"].item()) != identifier
                or decoy_count != int(record["decoy_count"])
                or decoy_count > 32
                or antigen_length != int(item["antigen_features"].shape[0])
            ):
                raise ValueError("Matched-local artifact metadata differs")
            decoys = {
                "instance_id": identifier,
                "antigen_length": antigen_length,
                "decoy_count": decoy_count,
                "positive_offsets": torch.from_numpy(
                    payload["positive_motif_offsets"].astype(np.int64, copy=True)
                ),
                "positive_flat": torch.from_numpy(
                    payload["positive_motif_flat_indices"].astype(np.int64, copy=True)
                ),
                "negative_offsets": torch.from_numpy(
                    payload["negative_motif_offsets"].astype(np.int64, copy=True)
                ),
                "negative_flat": torch.from_numpy(
                    payload["negative_motif_flat_indices"].astype(np.int64, copy=True)
                ),
            }
        if (
            tuple(decoys["positive_offsets"].shape) != (decoy_count + 1,)
            or tuple(decoys["negative_offsets"].shape) != (decoy_count + 1,)
            or int(decoys["positive_offsets"][-1]) != decoys["positive_flat"].numel()
            or int(decoys["negative_offsets"][-1]) != decoys["negative_flat"].numel()
        ):
            raise ValueError("Matched-local motif offsets differ")
        return item, decoys


class _MatchedLocalTrainingCollator:
    def __init__(self) -> None:
        self.base_collator = Sabdab2ContactBatchCollator()

    def __call__(
        self, examples: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        items, decoys = zip(*examples, strict=True)
        return self.base_collator(items), list(decoys)


def _matched_local_ranking_loss(
    interaction_logits: torch.Tensor,
    pair_candidate_mask: torch.Tensor,
    decoys: list[dict[str, Any]],
    *,
    margin: float,
) -> torch.Tensor:
    """Equal-decoy/equal-complex ranking on interaction-only motif scores."""
    if (
        interaction_logits.ndim != 3
        or tuple(interaction_logits.shape) != tuple(pair_candidate_mask.shape)
        or pair_candidate_mask.dtype != torch.bool
        or len(decoys) != int(interaction_logits.shape[0])
    ):
        raise ValueError("Matched-local logits, mask, and batch must align")
    complex_losses = []
    for batch_index, decoy in enumerate(decoys):
        antigen_length = int(decoy["antigen_length"])
        relation_losses = []
        for relation_index in range(int(decoy["decoy_count"])):
            local_scores = []
            for kind in ("positive", "negative"):
                offsets = decoy[f"{kind}_offsets"]
                flat = decoy[f"{kind}_flat"][
                    int(offsets[relation_index]) : int(offsets[relation_index + 1])
                ].to(interaction_logits.device, non_blocking=True)
                antibody_index = torch.div(flat, antigen_length, rounding_mode="floor")
                antigen_index = flat.remainder(antigen_length)
                if flat.numel() == 0 or not bool(
                    pair_candidate_mask[
                        batch_index, antibody_index, antigen_index
                    ].all()
                ):
                    raise ValueError("Matched-local motif is empty or outside the candidate grid")
                local_scores.append(
                    torch.sigmoid(
                        interaction_logits[
                            batch_index, antibody_index, antigen_index
                        ].float()
                    ).mean()
                )
            positive_score, negative_score = local_scores
            relation_losses.append(
                torch.nn.functional.softplus(margin + negative_score - positive_score)
            )
        if relation_losses:
            complex_losses.append(torch.stack(relation_losses).mean())
        else:
            complex_losses.append(interaction_logits[batch_index].float().sum() * 0.0)
    return torch.stack(complex_losses).mean()


class _FrozenPanelTrainingDataset(torch.utils.data.Dataset):
    """Choose one legal frozen wrong partner per focal complex and epoch."""

    def __init__(self, dataset: Any, panel: dict[str, Any], seed: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.index_by_id = {
            record["instance_id"]: index for index, record in enumerate(dataset.records)
        }
        self.entries = {entry["instance_id"]: entry for entry in panel["entries"]}
        self.reciprocal = (
            panel.get("format_version") == RECIPROCAL_PARTNER_PANEL_FORMAT
        )
        if set(self.entries) != set(self.index_by_id):
            raise ValueError("Training panel and training dataset identifiers differ")
        candidate_keys = (
            ("wrong_antigen_instance_ids", "wrong_antibody_instance_ids")
            if self.reciprocal
            else ("wrong_partner_instance_ids",)
        )
        for entry in self.entries.values():
            for key in candidate_keys:
                if any(identifier not in self.index_by_id for identifier in entry[key]):
                    raise ValueError("Training panel references a non-training instance")

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _choose(self, focal_id: str, candidates: list[str], direction: str) -> dict[str, Any]:
        digest = hashlib.sha256(
            f"{self.seed}\0{self.epoch}\0{direction}\0{focal_id}".encode("utf-8")
        ).digest()
        candidate_id = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
        return self.dataset[self.index_by_id[candidate_id]]

    def __getitem__(self, index: int) -> tuple[dict[str, Any], ...]:
        focal = self.dataset[index]
        entry = self.entries[focal["instance_id"]]
        if not self.reciprocal:
            candidates = entry["wrong_partner_instance_ids"]
            digest = hashlib.sha256(
                f"{self.seed}\0{self.epoch}\0{focal['instance_id']}".encode("utf-8")
            ).digest()
            candidate_id = candidates[
                int.from_bytes(digest[:8], "big") % len(candidates)
            ]
            candidate = self.dataset[self.index_by_id[candidate_id]]
            return focal, _wrong_antigen_pair_item(focal, candidate)
        wrong_antigen = self._choose(
            focal["instance_id"],
            entry["wrong_antigen_instance_ids"],
            "wrong_antigen",
        )
        wrong_antibody = self._choose(
            focal["instance_id"],
            entry["wrong_antibody_instance_ids"],
            "wrong_antibody",
        )
        return (
            focal,
            _wrong_antigen_pair_item(focal, wrong_antigen),
            _wrong_antibody_pair_item(focal, wrong_antibody),
        )


class _FrozenPanelTrainingCollator:
    def __init__(self) -> None:
        self.cognate = Sabdab2ContactBatchCollator()
        self.wrong = Sabdab2StructureBatchCollator()

    def __call__(
        self, examples: list[tuple[dict[str, Any], ...]]
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        if len(examples[0]) == 2:
            cognate, wrong_antigen = zip(*examples)
            return self.cognate(cognate), {
                "fixed_antibody_wrong_antigen": self.wrong(wrong_antigen)
            }
        if len(examples[0]) != 3:
            raise ValueError("Frozen partner examples have an unsupported arity")
        cognate, wrong_antigen, wrong_antibody = zip(*examples)
        return self.cognate(cognate), {
            "fixed_antibody_wrong_antigen": self.wrong(wrong_antigen),
            "fixed_antigen_wrong_antibody": self.wrong(wrong_antibody),
        }


def _wrong_antigen_pair_item(
    focal: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    item = dict(focal)
    for key, value in candidate.items():
        if key.startswith("antigen_"):
            item[key] = value
    item["instance_id"] = f"{focal['instance_id']}__wrong__{candidate['instance_id']}"
    item.pop("pair_labels", None)
    item.pop("pair_candidate_mask", None)
    return item


def _wrong_antibody_pair_item(
    focal: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    item = dict(focal)
    for key, value in candidate.items():
        if key.startswith("antibody_"):
            item[key] = value
    item["instance_id"] = f"{focal['instance_id']}__wrong_ab__{candidate['instance_id']}"
    item.pop("pair_labels", None)
    item.pop("pair_candidate_mask", None)
    return item


def _best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(
        precision[:-1] + recall[:-1], 1e-12, None
    )
    return float(thresholds[int(np.argmax(f1))])


def _fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    maximum_pairs: int,
) -> tuple[float, int]:
    if len(logits) > maximum_pairs:
        indices = np.random.default_rng(seed).choice(
            len(logits), size=maximum_pairs, replace=False
        )
        logits_for_fit = logits[indices]
        labels_for_fit = labels[indices]
    else:
        logits_for_fit, labels_for_fit = logits, labels
    scaler = ValidationTemperatureScaler()
    logits_tensor = torch.from_numpy(logits_for_fit.astype(np.float32, copy=False))
    labels_tensor = torch.from_numpy(labels_for_fit.astype(np.float32, copy=False))
    mask = torch.ones_like(labels_tensor, dtype=torch.bool)
    with torch.enable_grad():
        temperature = scaler.fit(logits_tensor, labels_tensor, mask)
    return temperature, len(logits_for_fit)


@torch.no_grad()
def _evaluate_partner_panel(
    model: FrameContactModel,
    dataset: Any,
    panel: dict[str, Any],
    device: torch.device,
    *,
    correct_scores: Mapping[str, float],
    use_bf16: bool,
    evaluation_batch_size: int,
    score_fraction: float,
    specificity_score_source: str,
    specificity_temperature: float,
    batch_intervention: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any]:
    index_by_id = {
        record["instance_id"]: index for index, record in enumerate(dataset.records)
    }
    panel_entries = panel["entries"]
    if set(index_by_id) != {entry["instance_id"] for entry in panel_entries}:
        raise ValueError("Validation panel and validation dataset identifiers differ")
    structure_collator = Sabdab2StructureBatchCollator()
    ordered_entries = sorted(panel_entries, key=lambda value: value["instance_id"])
    item_by_id = {
        identifier: dataset[index]
        for identifier, index in sorted(index_by_id.items())
    }
    if panel.get("format_version") == RECIPROCAL_PARTNER_PANEL_FORMAT:
        direction_specs = {
            "fixed_antibody_wrong_antigen": (
                "wrong_antigen_instance_ids",
                _wrong_antigen_pair_item,
                "antibody_cluster_id",
            ),
            "fixed_antigen_wrong_antibody": (
                "wrong_antibody_instance_ids",
                _wrong_antibody_pair_item,
                "antigen_cluster_id",
            ),
        }
    else:
        direction_specs = {
            "fixed_antibody_wrong_antigen": (
                "wrong_partner_instance_ids",
                _wrong_antigen_pair_item,
                "antigen_cluster_id",
            )
        }
    retrieval_by_direction = {}
    for direction, (entry_key, item_builder, cluster_key) in direction_specs.items():
        wrong_scores: list[list[float]] = []
        cognate_scores: list[float] = []
        cluster_ids: list[str] = []
        for entry in ordered_entries:
            focal = item_by_id[entry["instance_id"]]
            wrong_items = [
                item_builder(focal, item_by_id[identifier])
                for identifier in entry[entry_key]
            ]
            local_scores: list[float] = []
            for start in range(0, len(wrong_items), evaluation_batch_size):
                raw_wrong = structure_collator(
                    wrong_items[start : start + evaluation_batch_size]
                )
                wrong_batch = _move_batch(raw_wrong, device)
                if batch_intervention is not None:
                    wrong_batch = batch_intervention(wrong_batch)
                with _autocast(device, use_bf16):
                    wrong_output = model(wrong_batch)
                score_logits = (
                    wrong_output["contact_interaction_logits"]
                    if specificity_score_source == "interaction_only"
                    else wrong_output["contact_logits"]
                )
                scores = top_fraction_contact_score(
                    score_logits.float(),
                    wrong_output["pair_candidate_mask"],
                    fraction=score_fraction,
                    temperature=specificity_temperature,
                )
                local_scores.extend(scores.cpu().tolist())
            cognate_scores.append(float(correct_scores[entry["instance_id"]]))
            wrong_scores.append(local_scores)
            cluster_ids.append(entry[cluster_key])
        retrieval_by_direction[direction] = partner_retrieval_metrics(
            cognate_scores, wrong_scores, cluster_ids
        )
    return retrieval_by_direction


@torch.no_grad()
def _evaluate(
    model: FrameContactModel,
    loader: DataLoader,
    dataset: Any,
    panel: dict[str, Any],
    device: torch.device,
    *,
    use_bf16: bool,
    evaluation_batch_size: int,
    score_fraction: float,
    seed: int,
    calibration_max_pairs: int,
    specificity_score_source: str,
    evaluate_partner_retrieval: bool = True,
    collect_attention: bool = False,
    frozen_calibration: Mapping[str, float] | None = None,
    batch_intervention: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    model.eval()
    diagnostics = AttentionDiagnosticsAccumulator() if collect_attention else None
    cross_diagnostics = (
        CrossAttentionDiagnosticsAccumulator() if collect_attention else None
    )
    contact_logits_all: list[np.ndarray] = []
    contact_labels_all: list[np.ndarray] = []
    residue: dict[str, dict[str, list[np.ndarray]]] = {
        "paratope": {"labels": [], "logits": [], "per_complex_auprc": []},
        "epitope": {"labels": [], "logits": [], "per_complex_auprc": []},
    }
    contact_per_complex_auprc: list[float] = []
    contact_ranked_by_instance: dict[str, dict[str, Any]] = {}
    top_specificity_logits_by_instance: dict[str, np.ndarray] = {}
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        if batch_intervention is not None:
            batch = batch_intervention(batch)
        with _autocast(device, use_bf16):
            output = model(batch, return_attention=collect_attention)
        if diagnostics is not None:
            details = output["self_attention_details"]
            if details is None:
                raise RuntimeError("Requested attention diagnostics were not returned")
            diagnostics.update(
                "antibody", details["antibody"]["geometry"], output["antibody_mask"]
            )
            diagnostics.update(
                "antigen", details["antigen"]["geometry"], output["antigen_mask"]
            )
            cross_details = output["cross_attention_details"]
            if cross_diagnostics is None or cross_details is None:
                raise RuntimeError("Requested cross-attention diagnostics were not returned")
            cross_diagnostics.update(
                cross_details,
                output["antibody_mask"],
                output["antigen_mask"],
            )
        for batch_index, instance_id in enumerate(raw_batch["instance_ids"]):
            pair_mask = output["pair_candidate_mask"][batch_index]
            logits = output["contact_logits"][batch_index][pair_mask].float().cpu().numpy()
            labels = batch["pair_labels"][batch_index][pair_mask].cpu().numpy().astype(
                np.int8, copy=False
            )
            contact_logits_all.append(logits)
            contact_labels_all.append(labels)
            contact_per_complex_auprc.append(
                float(average_precision_score(labels, _sigmoid_numpy(logits)))
            )
            ranked = ranked_pair_metrics(
                labels,
                _sigmoid_numpy(logits),
                antibody_length=int(output["antibody_mask"][batch_index].sum()),
                antigen_length=int(output["antigen_mask"][batch_index].sum()),
            )
            if str(instance_id) in contact_ranked_by_instance:
                raise ValueError(f"Duplicate evaluation instance: {instance_id}")
            contact_ranked_by_instance[str(instance_id)] = ranked
            top_count = max(1, math.ceil(score_fraction * len(logits)))
            if specificity_score_source == "interaction_only":
                specificity_logits = output["contact_interaction_logits"][
                    batch_index
                ][pair_mask].float().cpu().numpy()
            elif specificity_score_source == "total_contact":
                specificity_logits = logits
            else:
                raise ValueError("Unknown specificity score source")
            top_specificity_logits_by_instance[str(instance_id)] = np.partition(
                specificity_logits, len(specificity_logits) - top_count
            )[-top_count:]
            for name, labels_key, logits_key, mask_key in (
                ("paratope", "antibody_labels", "paratope_logits", "antibody_mask"),
                ("epitope", "antigen_labels", "epitope_logits", "antigen_mask"),
            ):
                mask = output[mask_key][batch_index]
                local_labels = batch[labels_key][batch_index][mask].cpu().numpy().astype(
                    np.int8, copy=False
                )
                local_logits = (
                    output[logits_key][batch_index][mask].float().cpu().numpy()
                )
                residue[name]["labels"].append(local_labels)
                residue[name]["logits"].append(local_logits)
                residue[name]["per_complex_auprc"].append(
                    float(
                        average_precision_score(
                            local_labels, _sigmoid_numpy(local_logits)
                        )
                    )
                )
    contact_logits = np.concatenate(contact_logits_all)
    contact_labels = np.concatenate(contact_labels_all)
    if frozen_calibration is None:
        temperature, calibration_fit_count = _fit_temperature(
            contact_logits,
            contact_labels,
            seed=seed,
            maximum_pairs=calibration_max_pairs,
        )
        calibration_source = "fit_on_current_validation"
    else:
        temperature = float(frozen_calibration["temperature"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Frozen temperature must be finite and positive")
        calibration_fit_count = 0
        calibration_source = "frozen_validation"
    contact_probabilities = _sigmoid_numpy(contact_logits / temperature)
    contact_threshold = (
        _best_f1_threshold(contact_labels, contact_probabilities)
        if frozen_calibration is None
        else float(frozen_calibration["contact_threshold"])
    )
    specificity_temperature = (
        temperature if specificity_score_source == "total_contact" else 1.0
    )
    correct_scores = {
        identifier: float(
            _sigmoid_numpy(values / specificity_temperature).mean()
        )
        for identifier, values in top_specificity_logits_by_instance.items()
    }

    retrieval_by_direction = (
        _evaluate_partner_panel(
            model,
            dataset,
            panel,
            device,
            correct_scores=correct_scores,
            use_bf16=use_bf16,
            evaluation_batch_size=evaluation_batch_size,
            score_fraction=score_fraction,
            specificity_score_source=specificity_score_source,
            specificity_temperature=specificity_temperature,
            batch_intervention=batch_intervention,
        )
        if evaluate_partner_retrieval
        else {}
    )
    selection_metric = (
        min(
            value["cluster_weighted_cognate_compatibility_margin"]
            for value in retrieval_by_direction.values()
        )
        if retrieval_by_direction
        else float(np.mean(residue["epitope"]["per_complex_auprc"]))
    )
    result: dict[str, Any] = {
        "selection_metric": selection_metric,
        "selection_metric_rule": (
            "minimum_direction_cluster_weighted_margin"
            if retrieval_by_direction
            else "epitope_mean_per_complex_auprc"
        ),
        "specificity_score_source": specificity_score_source,
        "specificity_temperature": specificity_temperature,
        "validation_temperature": temperature,
        "calibration_source": calibration_source,
        "calibration_fit_pair_count": calibration_fit_count,
        "contact": {
            **binary_classification_metrics(
                contact_labels, contact_probabilities, threshold=contact_threshold
            ),
            "threshold_selected_on_validation": contact_threshold,
            "mean_per_complex_auprc": float(np.mean(contact_per_complex_auprc)),
            "complex_count": len(contact_per_complex_auprc),
            "ranked_pair_definition": (
                "fixed K plus Lmin=min(antibody_length,antigen_length); "
                "macro mean gives each complex equal weight"
            ),
            "ranked_pair_macro": {
                cutoff: {
                    "mean_precision": float(
                        np.mean(
                            [
                                values["cutoffs"][cutoff]["precision"]
                                for values in contact_ranked_by_instance.values()
                            ]
                        )
                    ),
                    "mean_recall": float(
                        np.mean(
                            [
                                values["cutoffs"][cutoff]["recall"]
                                for values in contact_ranked_by_instance.values()
                            ]
                        )
                    ),
                    "mean_effective_k": float(
                        np.mean(
                            [
                                values["cutoffs"][cutoff]["effective_k"]
                                for values in contact_ranked_by_instance.values()
                            ]
                        )
                    ),
                }
                for cutoff in next(
                    iter(contact_ranked_by_instance.values())
                )["cutoffs"]
            },
            "ranked_pair_by_instance": contact_ranked_by_instance,
        },
        "partner_retrieval": (
            next(iter(retrieval_by_direction.values()))
            if len(retrieval_by_direction) == 1
            else None
        ),
        "partner_retrieval_by_direction": retrieval_by_direction,
    }
    for name, values in residue.items():
        labels = np.concatenate(values["labels"])
        logits = np.concatenate(values["logits"])
        if frozen_calibration is None:
            residue_temperature, residue_calibration_fit_count = _fit_temperature(
                logits,
                labels,
                seed=seed + (101 if name == "paratope" else 211),
                maximum_pairs=calibration_max_pairs,
            )
            residue_calibration_source = "fit_on_current_validation"
        else:
            residue_temperature = float(
                frozen_calibration[f"{name}_temperature"]
            )
            if not math.isfinite(residue_temperature) or residue_temperature <= 0:
                raise ValueError(f"Frozen {name} temperature must be finite and positive")
            residue_calibration_fit_count = 0
            residue_calibration_source = "frozen_validation"
        probabilities = _sigmoid_numpy(logits / residue_temperature)
        threshold = (
            _best_f1_threshold(labels, probabilities)
            if frozen_calibration is None
            else float(frozen_calibration[f"{name}_threshold"])
        )
        result[name] = {
            **binary_classification_metrics(labels, probabilities, threshold=threshold),
            "threshold_selected_on_validation": threshold,
            "validation_temperature": residue_temperature,
            "calibration_source": residue_calibration_source,
            "calibration_fit_residue_count": residue_calibration_fit_count,
            "mean_per_complex_auprc": float(np.mean(values["per_complex_auprc"])),
        }
    if diagnostics is not None:
        result["attention_diagnostics"] = diagnostics.result()
        assert cross_diagnostics is not None
        result["cross_attention_diagnostics"] = cross_diagnostics.result()
    return result


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _configure_experiment(model: FrameContactModel, experiment: str) -> None:
    specification = EXPERIMENTS[experiment]
    if model.fusion_mode != specification["fusion_mode"]:
        raise ValueError("model fusion mode differs from the experiment specification")
    if model.adapter_mode != specification["adapter_mode"]:
        raise ValueError("model adapter mode differs from the experiment specification")
    if model.fusion_mode == "late":
        assert model.self_encoder is not None and model.cross_attention is not None
        self_encoders = (model.self_encoder,)
        cross_attentions = (model.cross_attention,)
    else:
        assert model.interleaved_self_encoders is not None
        assert model.interleaved_cross_attentions is not None
        self_encoders = tuple(model.interleaved_self_encoders)
        cross_attentions = tuple(model.interleaved_cross_attentions)
    if not specification["geometry"]:
        for encoder in self_encoders:
            for stream in (
                encoder.antibody_encoder,
                encoder.antigen_encoder,
            ):
                with torch.no_grad():
                    stream.geometry_residual_gate.zero_()
                stream.geometry_residual_gate.requires_grad_(False)
    if not specification["cross"]:
        for attention in cross_attentions:
            for cross in (
                attention.antibody_from_antigen,
                attention.antigen_from_antibody,
            ):
                with torch.no_grad():
                    cross.residual_gate.zero_()
                cross.residual_gate.requires_grad_(False)


def _model_from_configuration(
    first: Mapping[str, Any],
    configuration: Mapping[str, Any],
    experiment: str,
) -> FrameContactModel:
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown FrameContact experiment")
    model = FrameContactModel(
        antibody_input_dim=int(first["antibody_features"].shape[-1]),
        antigen_input_dim=int(first["antigen_features"].shape[-1]),
        d_model=int(configuration["d_model"]),
        num_heads=int(configuration["num_heads"]),
        num_self_layers=int(configuration["num_self_layers"]),
        geometry_feature_mode=EXPERIMENTS[experiment]["geometry_feature_mode"],
        geometry_bias_mode=EXPERIMENTS[experiment]["geometry_bias_mode"],
        num_rbf=int(configuration["num_rbf"]),
        neighbor_k=int(configuration["neighbor_k"]),
        neighbor_radius=float(configuration["neighbor_radius"]),
        sequence_window=int(configuration["sequence_window"]),
        dropout=float(configuration["dropout"]),
        fusion_mode=EXPERIMENTS[experiment]["fusion_mode"],
        cross_attention_contact_gate_mode=EXPERIMENTS[experiment][
            "cross_attention_contact_gate_mode"
        ],
        adapter_mode=EXPERIMENTS[experiment]["adapter_mode"],
        epitope_pooling=str(configuration["epitope_pooling"]),
        epitope_pooling_k=int(configuration["epitope_pooling_k"]),
        paratope_pooling=str(configuration.get("paratope_pooling", "logmeanexp")),
        paratope_pooling_k=int(configuration.get("paratope_pooling_k", 2)),
        contact_readout=str(configuration.get("contact_readout", "pair")),
        distance_bin_count=int(EXPERIMENTS[experiment]["distance_bin_count"]),
        contrastive_dim=int(configuration["contrastive_dim"]),
    )
    _configure_experiment(model, experiment)
    return model


def main() -> None:
    arguments = _arguments()
    if (arguments.source_manifest is None) != (
        arguments.source_manifest_sha256 is None
    ):
        raise ValueError("source manifest path and SHA-256 must be provided together")
    if arguments.dataset_backend == "sabdab2" and arguments.source_manifest is None:
        raise ValueError("SAbDab2 training requires an authenticated source manifest")
    if arguments.source_manifest is not None:
        verify_source_manifest(
            REPOSITORY_ROOT,
            arguments.source_manifest,
            arguments.source_manifest_sha256,
        )
    if (arguments.split_manifest is None) != (
        arguments.split_manifest_sha256 is None
    ):
        raise ValueError("split manifest path and SHA-256 must be provided together")
    protocol, training_panel, panel = _load_authorities(arguments)
    readout = protocol.get("epitope_readout", {})
    if readout.get("pooling") != arguments.epitope_pooling:
        raise ValueError("Epitope pooling differs from the pre-registered protocol")
    expected_k = readout.get("k")
    if arguments.epitope_pooling == "topk_mean":
        if int(expected_k) != arguments.epitope_pooling_k:
            raise ValueError("Epitope pooling k differs from the pre-registered protocol")
    elif expected_k is not None:
        raise ValueError("Non-top-k pooling protocol must set k to null")
    if arguments.epitope_pooling_k < 1:
        raise ValueError("Epitope pooling k must be positive")
    paratope_readout = protocol.get("paratope_readout", {})
    if paratope_readout.get("pooling") != arguments.paratope_pooling:
        raise ValueError("Paratope pooling differs from the pre-registered protocol")
    expected_paratope_k = paratope_readout.get("k")
    if arguments.paratope_pooling == "topk_mean":
        if int(expected_paratope_k) != arguments.paratope_pooling_k:
            raise ValueError("Paratope pooling k differs from the protocol")
    elif expected_paratope_k is not None:
        raise ValueError("Non-top-k paratope pooling protocol must set k to null")
    if arguments.paratope_pooling_k < 1:
        raise ValueError("Paratope pooling k must be positive")
    contact_readout = protocol.get("contact_readout", {})
    if contact_readout.get("mode") != arguments.contact_readout:
        raise ValueError("Contact readout differs from the pre-registered protocol")
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError("output-root already exists")
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    if arguments.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if arguments.calibration_max_pairs <= 0:
        raise ValueError("calibration-max-pairs must be positive")
    device = _device(arguments.device)
    use_bf16 = arguments.precision == "bf16" and device.type == "cuda"
    if arguments.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 requires CUDA")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise ValueError("CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG")
    _set_seed(arguments.seed)
    if arguments.dataset_backend == "sabdab2":
        train_base = Sabdab2StructureEmbeddingDataset(
            arguments.structure_cache_root,
            arguments.embedding_root,
            arguments.label_root,
            split_role="train",
            split_path=arguments.split_manifest,
            split_sha256=arguments.split_manifest_sha256,
        )
        validation_base = Sabdab2StructureEmbeddingDataset(
            arguments.structure_cache_root,
            arguments.embedding_root,
            arguments.label_root,
            split_role="validation",
            split_path=arguments.split_manifest,
            split_sha256=arguments.split_manifest_sha256,
        )
        if arguments.joint_embedding_root is not None:
            concatenate_base = (
                arguments.experiment
                == "abpairfm_epiformer_residual_fusion_same_protocol"
            )
            train_base = _JointEmbeddingOverrideDataset(
                train_base,
                arguments.joint_embedding_root,
                arguments.joint_embedding_manifest_sha256,
                split_role="train",
                concatenate_base=concatenate_base,
            )
            validation_base = _JointEmbeddingOverrideDataset(
                validation_base,
                arguments.joint_embedding_root,
                arguments.joint_embedding_manifest_sha256,
                split_role="validation",
                concatenate_base=concatenate_base,
            )
        train_dataset = Sabdab2ContactDataset(
            train_base, arguments.contact_cache_root
        )
        validation_dataset = Sabdab2ContactDataset(
            validation_base, arguments.contact_cache_root
        )
        if arguments.matched_local_loss_weight > 0:
            train_dataset = _MatchedLocalTrainingDataset(
                train_dataset,
                arguments.matched_local_root,
                arguments.matched_local_manifest_sha256,
            )
    elif arguments.matched_local_loss_weight > 0:
        raise ValueError("Matched-local training is available only for SAbDab2")
    else:
        train_dataset = AsepDevelopmentDataset(
            arguments.asep_cache_root,
            manifest_sha256=arguments.asep_cache_manifest_sha256,
            split_role="train",
            feature_mode=arguments.asep_feature_mode,
        )
        validation_dataset = AsepDevelopmentDataset(
            arguments.asep_cache_root,
            manifest_sha256=arguments.asep_cache_manifest_sha256,
            split_role="val",
            feature_mode=arguments.asep_feature_mode,
        )
    if arguments.matched_local_loss_weight > 0 and arguments.specificity_loss_weight > 0:
        raise ValueError("Matched-local and partner-swap screens must remain separate")
    train_generator = torch.Generator().manual_seed(arguments.seed)
    paired_train_dataset = None
    matched_local_train_dataset = None
    if arguments.matched_local_loss_weight > 0:
        matched_local_train_dataset = train_dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=arguments.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=arguments.workers,
            collate_fn=_MatchedLocalTrainingCollator(),
            generator=train_generator,
            pin_memory=device.type == "cuda",
            persistent_workers=arguments.workers > 0,
        )
    elif arguments.specificity_loss_weight == 0:
        train_loader = _loader(
            train_dataset,
            batch_size=arguments.batch_size,
            workers=arguments.workers,
            shuffle=True,
            seed=arguments.seed,
            pin_memory=device.type == "cuda",
        )
    else:
        paired_train_dataset = _FrozenPanelTrainingDataset(
            train_dataset, training_panel, arguments.seed
        )
        train_loader = DataLoader(
            paired_train_dataset,
            batch_size=arguments.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=arguments.workers,
            collate_fn=_FrozenPanelTrainingCollator(),
            generator=train_generator,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
    validation_loader = _loader(
        validation_dataset,
        batch_size=arguments.evaluation_batch_size,
        workers=arguments.workers,
        shuffle=False,
        seed=arguments.seed,
        pin_memory=device.type == "cuda",
    )
    first = (
        train_dataset.dataset[0]
        if matched_local_train_dataset is not None
        else train_dataset[0]
    )
    model = _model_from_configuration(
        first,
        vars(arguments),
        arguments.experiment,
    )
    initial_state_sha256 = _state_sha256(model)
    model.to(device)
    weights = FrameContactLossWeights(
        contact=arguments.contact_loss_weight,
        paratope=arguments.paratope_loss_weight,
        epitope=arguments.epitope_loss_weight,
        specificity=arguments.specificity_loss_weight,
        distance=arguments.distance_loss_weight,
        contrastive=arguments.contrastive_loss_weight,
    )
    specificity_score_source = str(
        protocol.get("training_objective", {}).get(
            "specificity_score_source", "total_contact"
        )
    )
    objective = FrameContactObjective(
        weights=weights,
        specificity_margin=arguments.specificity_margin,
        score_fraction=arguments.score_fraction,
        specificity_score_source=specificity_score_source,
        contact_focal_gamma=arguments.contact_focal_gamma,
        epitope_dice_weight=arguments.epitope_dice_weight,
        epitope_count_weight=arguments.epitope_count_weight,
        contrastive_temperature=arguments.contrastive_temperature,
        contrastive_maximum_residues=arguments.contrastive_maximum_residues,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay
    )
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.tmp-",
            dir=arguments.output_root.parent,
        )
    )
    history = []
    best_metric = -math.inf
    best_tie_break_metric = -math.inf
    best_epoch = 0
    stale_epochs = 0
    best_path = temporary_root / "best_validation.pt"
    started = time.monotonic()
    try:
        for epoch in range(1, arguments.epochs + 1):
            model.train()
            if paired_train_dataset is not None:
                paired_train_dataset.set_epoch(epoch)
            component_sums: dict[str, float] = {}
            steps = 0
            for raw_item in train_loader:
                if matched_local_train_dataset is not None:
                    raw_batch, raw_matched_local = raw_item
                    raw_wrong_by_direction = {}
                elif paired_train_dataset is None:
                    raw_batch = raw_item
                    raw_matched_local = []
                    raw_wrong_by_direction = {}
                else:
                    raw_batch, raw_wrong_by_direction = raw_item
                    raw_matched_local = []
                batch = _move_batch(raw_batch, device)
                wrong_batches = {
                    direction: _move_batch(raw_wrong, device)
                    for direction, raw_wrong in raw_wrong_by_direction.items()
                }
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, use_bf16), _saved_tensor_context(
                    arguments.offload_saved_tensors
                ):
                    output = _cognate_training_view(model(batch))
                    wrong_outputs = {
                        direction: _wrong_training_view(model(wrong_batch))
                        for direction, wrong_batch in wrong_batches.items()
                    }
                    losses = objective(
                        output,
                        batch,
                        wrong_partner_outputs=(wrong_outputs if wrong_outputs else None),
                    )
                    if matched_local_train_dataset is not None:
                        matched_local_loss = _matched_local_ranking_loss(
                            output["contact_interaction_logits"],
                            output["pair_candidate_mask"],
                            raw_matched_local,
                            margin=arguments.matched_local_margin,
                        )
                        losses["matched_local_loss"] = matched_local_loss
                        losses["loss"] = (
                            losses["loss"]
                            + arguments.matched_local_loss_weight * matched_local_loss
                        )
                    else:
                        losses["matched_local_loss"] = losses["loss"].new_zeros(())
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                for key, value in losses.items():
                    if torch.is_tensor(value):
                        component_sums[key] = component_sums.get(key, 0.0) + float(
                            value.detach().cpu()
                        )
                steps += 1
            validation = _evaluate(
                model,
                validation_loader,
                validation_dataset,
                panel,
                device,
                use_bf16=use_bf16,
                evaluation_batch_size=arguments.evaluation_batch_size,
                score_fraction=arguments.score_fraction,
                seed=arguments.seed,
                calibration_max_pairs=arguments.calibration_max_pairs,
                specificity_score_source=specificity_score_source,
                evaluate_partner_retrieval=bool(
                    protocol["model_selection"].get(
                        "evaluate_partner_retrieval", True
                    )
                ),
            )
            record = {
                "epoch": epoch,
                "train": {key: value / steps for key, value in component_sums.items()},
                "validation": validation,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(record)
            primary_metric = protocol["model_selection"].get(
                "primary_metric", "minimum_direction_cluster_weighted_margin"
            )
            contact_auprc = float(validation["contact"]["mean_per_complex_auprc"])
            partner_margin = float(validation["selection_metric"])
            if primary_metric == "epitope_mean_per_complex_auprc":
                metric = float(validation["epitope"]["mean_per_complex_auprc"])
                tie_break_metric = contact_auprc
            elif primary_metric == "contact_mean_per_complex_auprc":
                metric = contact_auprc
                tie_break_metric = partner_margin
            elif primary_metric == "minimum_direction_cluster_weighted_margin":
                metric = partner_margin
                tie_break_metric = contact_auprc
            else:
                raise ValueError("Unsupported primary checkpoint-selection metric")
            improved = metric > best_metric + 1e-4 or (
                abs(metric - best_metric) <= 1e-4
                and tie_break_metric > best_tie_break_metric
            )
            if improved:
                best_metric = metric
                best_tie_break_metric = tie_break_metric
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "format_version": RESULT_FORMAT,
                        "experiment": arguments.experiment,
                        "seed": arguments.seed,
                        "selected_epoch": epoch,
                        "selection_rule": protocol["model_selection"]["checkpoint_rule"],
                        "specificity_score_source": specificity_score_source,
                        "validation_temperature": validation["validation_temperature"],
                        "model_state_dict": model.state_dict(),
                    },
                    best_path,
                )
            else:
                stale_epochs += 1
            print(json.dumps(record, sort_keys=True), flush=True)
            if stale_epochs >= arguments.early_stopping_patience:
                break

        checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_validation = _evaluate(
            model,
            validation_loader,
            validation_dataset,
            panel,
            device,
            use_bf16=use_bf16,
            evaluation_batch_size=arguments.evaluation_batch_size,
            score_fraction=arguments.score_fraction,
            seed=arguments.seed,
            calibration_max_pairs=arguments.calibration_max_pairs,
            specificity_score_source=specificity_score_source,
            collect_attention=True,
            evaluate_partner_retrieval=bool(
                protocol["model_selection"].get(
                    "evaluate_partner_retrieval", True
                )
            ),
        )
        configuration = {
            key: value
            for key, value in vars(arguments).items()
            if key
            not in {
                "structure_cache_root",
                "contact_cache_root",
                "embedding_root",
                "joint_embedding_root",
                "matched_local_root",
                "label_root",
                "asep_cache_root",
                "output_root",
            }
        }
        configuration = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in configuration.items()
        }
        result = {
            "format_version": RESULT_FORMAT,
            "experiment": arguments.experiment,
            "claim_boundary": (
                "Development train/validation result only. Test and external roles were not "
                "loaded; no generalization or benchmark-improvement claim is authorized."
            ),
            "authority": {
                "dataset_backend": arguments.dataset_backend,
                "structure_manifest_sha256": arguments.structure_manifest_sha256,
                "contact_manifest_sha256": arguments.contact_manifest_sha256,
                "embedding_manifest_sha256": arguments.embedding_manifest_sha256,
                "joint_embedding_manifest_sha256": arguments.joint_embedding_manifest_sha256,
                "joint_embedding_source_split_sha256": arguments.joint_embedding_source_split_sha256,
                "matched_local_manifest_sha256": arguments.matched_local_manifest_sha256,
                "asep_cache_manifest_sha256": arguments.asep_cache_manifest_sha256,
                "protocol_file_sha256": sha256_file(arguments.protocol),
                "protocol_receipt_sha256": arguments.protocol_receipt_sha256,
                "source_manifest_sha256": arguments.source_manifest_sha256,
                "validation_partner_panel_sha256": arguments.validation_partner_panel_sha256,
                "training_partner_panel_sha256": arguments.training_partner_panel_sha256,
                "split_manifest_sha256": arguments.split_manifest_sha256,
                "test_loaded": False,
                "external_test_loaded": False,
            },
            "configuration": configuration,
            "runtime": {
                "device": str(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "deterministic_algorithms": True,
            },
            "data": {
                "dataset_backend": arguments.dataset_backend,
                "feature_mode": (
                    arguments.asep_feature_mode
                    if arguments.dataset_backend == "asep"
                    else (
                        "external_embedding_cache"
                        if arguments.joint_embedding_root is None
                        else (
                            "abpairfm_joint_residual_concatenation_v1"
                            if arguments.experiment
                            == "abpairfm_epiformer_residual_fusion_same_protocol"
                            else "abpairfm_joint_cross_chain_v1"
                        )
                    )
                ),
                "train_complexes": len(train_dataset),
                "validation_complexes": len(validation_dataset),
            },
            "model": {
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "initial_state_sha256": initial_state_sha256,
                "selected_state_sha256": _state_sha256(model),
            },
            "selection": {
                "best_epoch": best_epoch,
                "best_validation_primary_metric": best_metric,
                "primary_metric": protocol["model_selection"].get("primary_metric"),
                "best_validation_tie_break_metric": best_tie_break_metric,
                "epochs_completed": len(history),
            },
            "history": history,
            "selected_checkpoint_validation": final_validation,
        }
        result_path = temporary_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
        result_path.write_bytes(canonical_json_bytes(result))
        selected_checkpoint = temporary_root / "FRAMECONTACT_SELECTED.pt"
        torch.save(checkpoint, selected_checkpoint)
        receipt = {
            "format_version": RESULT_FORMAT,
            "result_sha256": sha256_file(result_path),
            "selected_checkpoint_sha256": sha256_file(selected_checkpoint),
            "seed": arguments.seed,
            "selected_epoch": best_epoch,
            "matched_local_manifest_sha256": arguments.matched_local_manifest_sha256,
            "test_loaded": False,
            "external_test_loaded": False,
        }
        (temporary_root / "RUN_RECEIPT.json").write_bytes(canonical_json_bytes(receipt))
        os.replace(temporary_root, arguments.output_root)
        print(json.dumps({**receipt, "output_root": str(arguments.output_root)}, sort_keys=True))
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


if __name__ == "__main__":
    main()
