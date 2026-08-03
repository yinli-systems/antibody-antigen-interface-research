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
from torch.utils.checkpoint import checkpoint as activation_checkpoint


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
    positive_required_average_precision,
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
from interaction_first.sabdab2_pair_geometry import (  # noqa: E402
    Sabdab2PairGeometryBatchCollator,
    Sabdab2PairGeometryDataset,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    Sabdab2StructureBatchCollator,
    Sabdab2StructureEmbeddingDataset,
    canonical_json_bytes,
    sha256_file,
)
from interaction_first.source_manifest import verify_source_manifest  # noqa: E402


RESULT_FORMAT = "framecontact_development_training_v1"
EXTERNAL_BASELINE_AUTHORITY: dict[str, Any] | None = None
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
    },
    "framecontact_unary_only": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "binary_contact",
        "residue_output_mode": "unary_only",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
    },
    "framecontact_contact_marginal": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "binary_contact",
        "residue_output_mode": "contact_marginal_only",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
    },
    "framecontact_contact_hybrid": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "binary_contact",
        "residue_output_mode": "gated_hybrid",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
    },
    "framecontact_geometry_ablated_contact_hybrid": {
        # Keep the full-pose module topology and parameter shapes so this is a
        # same-code control.  _configure_experiment freezes the zero geometry
        # residual gates, making outputs coordinate-independent while retaining
        # the partner-aware pair and cross-attention paths.
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": False,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "binary_contact",
        "residue_output_mode": "gated_hybrid",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
    },
    "framecontact_distogram": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "distogram",
        "residue_output_mode": "gated_hybrid",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
    },
    "framecontact_geometry_ablated_distogram": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": False,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "distogram",
        "residue_output_mode": "gated_hybrid",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
    },
    "framecontact_distogram_triangle": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "distogram",
        "residue_output_mode": "gated_hybrid",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
    },
    # Gauge arms. The contact map splits exactly into two single-chain
    # propensity profiles plus a jointly dependent remainder, and only the
    # remainder can distinguish partners. The centered remainder is aggregated
    # with length-normalized log-mean-exp rather than probabilistic noisy-OR:
    # zero residual is absence of interaction evidence, not a 50% contact.
    # The contact prediction itself is untouched.
    "framecontact_gauge_marginal": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "binary_contact",
        "residue_output_mode": "gauge_marginal_only",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
        "pair_gauge_mode": "doubly_centered",
    },
    "framecontact_gauge_hybrid": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "binary_contact",
        "residue_output_mode": "gauge_hybrid",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
        "pair_gauge_mode": "doubly_centered",
    },
    # Identical architecture to framecontact_gauge_hybrid. The separate name
    # exists so the residue-level partner-contrastive weight, supplied on the
    # command line like the triangle weight, lands in its own run directory.
    "framecontact_gauge_hybrid_contrastive": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "binary_contact",
        "residue_output_mode": "gauge_hybrid",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
        "pair_gauge_mode": "doubly_centered",
    },
    "framecontact_gauge_hybrid_distogram": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "distogram",
        "residue_output_mode": "gauge_hybrid",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
        "pair_gauge_mode": "doubly_centered",
    },
    "framecontact_gauge_hybrid_distogram_triangle": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "distogram",
        "residue_output_mode": "gauge_hybrid",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
        "pair_gauge_mode": "doubly_centered",
    },
    # Gauge diagnostics with an unchanged raw-marginal residue head. This is
    # the read-only arm: it reports the partner-specific variance share of an
    # existing configuration without altering a single logit.
    "framecontact_contact_hybrid_gauge_audit": {
        "geometry_feature_mode": "full_pose",
        "geometry_bias_mode": "pair_mlp",
        "geometry": True,
        "cross": True,
        "fusion_mode": "late",
        "pair_output_mode": "binary_contact",
        "residue_output_mode": "gated_hybrid",
        "residue_unary_source": "intra_chain",
        "contact_marginalization": "noisy_or",
        "pair_gauge_mode": "doubly_centered",
    },
}


def _sigmoid_numpy(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _weight_distribution_diagnostics(
    weights: np.ndarray, mask: np.ndarray
) -> dict[str, float | bool | None]:
    """Describe attribution concentration without selecting a threshold."""

    values = np.asarray(weights, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        raise ValueError("weight diagnostics require at least one valid element")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("weight diagnostics require finite nonnegative values")
    total = float(values.sum())
    result: dict[str, float | bool | None] = {
        "valid_count": float(values.size),
        "mean_weight": float(values.mean()),
        "maximum_weight": float(values.max()),
        "positive_weight_fraction": float(np.mean(values > 0.0)),
        "total_weight": total,
        "zero_total_weight": total <= 1e-12,
    }
    if total <= 1e-12:
        return result | {
            "entropy": None,
            "normalized_entropy": None,
            "effective_support": 0.0,
            "top1_mass_fraction": None,
        }
    probabilities = values / total
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    normalized_entropy = (
        0.0 if values.size == 1 else entropy / math.log(float(values.size))
    )
    return result | {
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "effective_support": float(math.exp(entropy)),
        "top1_mass_fraction": float(probabilities.max()),
    }


def _aggregate_weight_diagnostics(
    values: list[Mapping[str, float | bool | None]],
) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot aggregate empty weight diagnostics")
    numeric_keys = sorted(
        key
        for key in values[0]
        if key != "zero_total_weight"
        and all(
            entry.get(key) is None
            or (
                isinstance(entry.get(key), (int, float))
                and not isinstance(entry.get(key), bool)
            )
            for entry in values
        )
    )
    return {
        "complex_count": len(values),
        "zero_total_weight_complex_count": sum(
            bool(entry["zero_total_weight"]) for entry in values
        ),
        "mean_by_metric_over_defined_complexes": {
            key: (
                None
                if not [entry[key] for entry in values if entry[key] is not None]
                else float(
                    np.mean(
                        [entry[key] for entry in values if entry[key] is not None]
                    )
                )
            )
            for key in numeric_keys
        },
    }


def _residue_logit_distribution_diagnostics(
    logits: np.ndarray, *, saturation_absolute_logit: float = 29.0
) -> dict[str, float | int]:
    """Fail-closed diagnostics for residue-score collapse and numeric clipping.

    ``masked_noisy_or_logits`` clamps its accumulated log-survival at -30, so
    the failed v1 gauge marginal produced residue logits pinned at +30.  The
    29-logit threshold is deliberately close to that implementation boundary;
    it is a numerical saturation alarm, not a classification operating point.
    Exact unique counts are computed after conversion to float32, matching the
    values written by the evaluation path.
    """

    values = np.asarray(logits, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("residue logit diagnostics require at least one value")
    if not np.isfinite(values).all():
        raise ValueError("residue logit diagnostics require finite values")
    if not math.isfinite(saturation_absolute_logit) or saturation_absolute_logit <= 0:
        raise ValueError("saturation threshold must be finite and positive")
    return {
        "valid_count": int(values.size),
        "unique_valid_logit_count": int(np.unique(values).size),
        "standard_deviation": float(values.astype(np.float64).std()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "saturation_absolute_logit": float(saturation_absolute_logit),
        "saturation_fraction": float(
            np.mean(np.abs(values) >= saturation_absolute_logit)
        ),
    }


def _aggregate_residue_logit_diagnostics(
    values: Mapping[str, Mapping[str, float | int]],
) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot aggregate empty residue logit diagnostics")
    entries = list(values.values())
    return {
        "definition": (
            "Descriptive pre-calibration valid-residue logits. Saturation means "
            "absolute raw logit >=29, a numerical alarm for the +/-30 noisy-OR "
            "clamp and not a selected decision threshold."
        ),
        "complex_count": len(entries),
        "collapsed_complex_count": sum(
            int(entry["unique_valid_logit_count"]) <= 1 for entry in entries
        ),
        "saturated_complex_count": sum(
            float(entry["saturation_fraction"]) > 0.0 for entry in entries
        ),
        "minimum_unique_valid_logit_count": min(
            int(entry["unique_valid_logit_count"]) for entry in entries
        ),
        "mean_standard_deviation": float(
            np.mean([float(entry["standard_deviation"]) for entry in entries])
        ),
        "maximum_saturation_fraction": max(
            float(entry["saturation_fraction"]) for entry in entries
        ),
        "by_instance": dict(values),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-backend", choices=("sabdab2", "asep"), default="sabdab2"
    )
    parser.add_argument("--structure-cache-root", type=Path)
    parser.add_argument("--contact-cache-root", type=Path)
    parser.add_argument("--pair-geometry-cache-root", type=Path)
    parser.add_argument("--pair-geometry-manifest-sha256")
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--embedding-manifest-sha256")
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
    parser.add_argument(
        "--attention-kernel",
        choices=("reference", "sdpa"),
        default="reference",
    )
    parser.add_argument(
        "--contact-bottleneck-projection-dim",
        type=int,
        default=0,
        help=(
            "Enable the validation-gated contact-marginal contrastive head with "
            "this projection width; zero keeps the candidate disabled."
        ),
    )
    parser.add_argument(
        "--contact-bottleneck-gradient-mode",
        choices=("stop_gradient", "joint"),
        default="stop_gradient",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contact-loss-weight", type=float, default=1.0)
    parser.add_argument("--paratope-loss-weight", type=float, default=0.25)
    parser.add_argument("--epitope-loss-weight", type=float, default=0.25)
    parser.add_argument("--specificity-loss-weight", type=float, default=0.25)
    parser.add_argument("--distogram-loss-weight", type=float, default=0.0)
    parser.add_argument("--triangle-loss-weight", type=float, default=0.0)
    parser.add_argument("--edm-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--residue-partner-contrastive-loss-weight", type=float, default=0.0
    )
    parser.add_argument("--residue-contrastive-margin", type=float, default=0.5)
    parser.add_argument(
        "--residue-contrastive-mode",
        choices=("pairwise_softplus", "listwise_contrastive"),
        default="pairwise_softplus",
    )
    parser.add_argument("--residue-contrastive-temperature", type=float, default=1.0)
    parser.add_argument("--specificity-margin", type=float, default=0.05)
    parser.add_argument("--score-fraction", type=float, default=0.01)
    parser.add_argument(
        "--specificity-loss-mode",
        choices=("pairwise_softplus", "listwise_contrastive"),
        default="pairwise_softplus",
    )
    parser.add_argument("--specificity-negative-count", type=int, default=1)
    parser.add_argument("--specificity-temperature", type=float, default=1.0)
    parser.add_argument(
        "--specificity-direction-aggregation",
        choices=("mean", "smooth_max"),
        default="mean",
    )
    parser.add_argument(
        "--specificity-direction-temperature", type=float, default=0.1
    )
    parser.add_argument(
        "--checkpoint-wrong-partners",
        action="store_true",
        help=(
            "Recompute wrong-partner forwards during backward to bound activation "
            "memory for multi-negative specificity training."
        ),
    )
    parser.add_argument("--calibration-max-pairs", type=int, default=2_000_000)
    parser.add_argument("--triangle-neighbor-radius", type=float, default=12.0)
    parser.add_argument("--triangle-neighbor-k", type=int, default=8)
    parser.add_argument("--triangle-slack", type=float, default=0.5)
    parser.add_argument("--antigen-patch-top-seeds", type=int, default=1)
    parser.add_argument("--antigen-patch-radius", type=float, default=12.0)
    parser.add_argument("--edm-numerical-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--offload-saved-tensors",
        action="store_true",
        help="Move autograd-saved tensors to CPU until backward for memory safety.",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--authority-preflight-only",
        action="store_true",
        help=(
            "Authenticate the frozen source/protocol/panels, construct both "
            "development datasets, load one train and validation example, and "
            "instantiate the configured model without creating an output root "
            "or starting training. This preserves the requested CUDA/bf16 "
            "execution contract while remaining runnable on a login node."
        ),
    )
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


# The training views below are memory reductions, not semantic filters.  Any
# key an enabled loss reads must survive them, so the key sets are declared once
# here and asserted against the objective by
# tests/test_training_view_loss_contract.py.  Dropping a key that a loss needs
# fails only when that loss is switched on, which is exactly the failure mode
# that kept the triangle arm from ever starting.
_COGNATE_VIEW_REQUIRED = (
    "contact_logits",
    "contact_interaction_logits",
    "pair_candidate_mask",
    "paratope_logits",
    "epitope_logits",
    "antibody_mask",
    "antigen_mask",
    # select_local_antigen_patch routes the triangle constraint through the
    # antigen unary prediction, so the triangle loss needs these two.
    "paratope_unary_logits",
    "epitope_unary_logits",
)
_COGNATE_VIEW_OPTIONAL = (
    "distogram_logits",
    "distance_bin_boundaries",
    "antibody_contact_propensity_logits",
    "antigen_contact_propensity_logits",
    "contact_bottleneck_compatibility",
    "contact_gauge_interaction_logits",
)
_WRONG_VIEW_REQUIRED = (
    "contact_logits",
    "contact_interaction_logits",
    "pair_candidate_mask",
)
# Kept optional so score-only stub models stay usable, but every one of these
# must survive for the loss that reads it.  The residue logits carry the decoy
# side of the residue-level partner-contrastive loss.
_WRONG_VIEW_OPTIONAL = (
    "paratope_logits",
    "epitope_logits",
    "contact_bottleneck_compatibility",
    "contact_gauge_interaction_logits",
)
# Which model attribute decides that an optional key will be emitted, so the
# activation-checkpointed path can pack the same keys as the plain path without
# running an extra forward.
_WRONG_VIEW_OPTIONAL_SOURCE = {
    "paratope_logits": lambda model: getattr(model, "paratope_head", None) is not None,
    "epitope_logits": lambda model: getattr(model, "epitope_head", None) is not None,
    "contact_bottleneck_compatibility": (
        lambda model: getattr(model, "contact_bottleneck_head", None) is not None
    ),
    "contact_gauge_interaction_logits": (
        lambda model: getattr(model, "pair_gauge_mode", "raw") != "raw"
    ),
}


def _cognate_training_view(output: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: output[key] for key in _COGNATE_VIEW_REQUIRED}
    for key in _COGNATE_VIEW_OPTIONAL:
        if output.get(key) is not None:
            result[key] = output[key]
    return result


def _wrong_training_view(output: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: output[key] for key in _WRONG_VIEW_REQUIRED}
    for key in _WRONG_VIEW_OPTIONAL:
        if output.get(key) is not None:
            result[key] = output[key]
    return result


def _checkpointed_wrong_training_view(
    model: torch.nn.Module, batch: Mapping[str, Any]
) -> dict[str, Any]:
    """Checkpoint a wrong-partner forward while preserving the exact score path."""

    tensor_keys = tuple(key for key, value in batch.items() if torch.is_tensor(value))
    tensor_values = tuple(batch[key] for key in tensor_keys)
    static_values = {
        key: value for key, value in batch.items() if key not in tensor_keys
    }

    packed_keys = _WRONG_VIEW_REQUIRED + tuple(
        key
        for key in _WRONG_VIEW_OPTIONAL
        if _WRONG_VIEW_OPTIONAL_SOURCE[key](model)
    )

    def forward(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        local_batch = dict(static_values)
        local_batch.update(zip(tensor_keys, values))
        view = _wrong_training_view(model(local_batch))
        return tuple(view[key] for key in packed_keys)

    checkpointed = activation_checkpoint(
        forward,
        *tensor_values,
        use_reentrant=False,
        preserve_rng_state=True,
    )
    return dict(zip(packed_keys, checkpointed))


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
    asep_values = (
        arguments.asep_cache_root,
        arguments.asep_cache_manifest_sha256,
    )
    if arguments.dataset_backend == "sabdab2":
        if any(value is None for value in sabdab_values):
            raise ValueError("SAbDab2 backend requires structure/contact/embedding authorities")
        if any(value is not None for value in asep_values):
            raise ValueError("SAbDab2 backend may not receive AsEP cache arguments")
        structure_manifest = (
            arguments.structure_cache_root / "STRUCTURE_CACHE_MANIFEST.json"
        )
        contact_manifest = arguments.contact_cache_root / "CONTACT_CACHE_MANIFEST.json"
        if sha256_file(structure_manifest) != arguments.structure_manifest_sha256:
            raise ValueError("Structure cache manifest SHA-256 differs")
        if sha256_file(contact_manifest) != arguments.contact_manifest_sha256:
            raise ValueError("Contact cache manifest SHA-256 differs")
        distance_required = EXPERIMENTS[arguments.experiment].get(
            "pair_output_mode", "binary_contact"
        ) == "distogram"
        geometry_values = (
            arguments.pair_geometry_cache_root,
            arguments.pair_geometry_manifest_sha256,
        )
        if distance_required and any(value is None for value in geometry_values):
            raise ValueError("Distogram training requires a pair geometry authority")
        if not distance_required and any(value is not None for value in geometry_values):
            raise ValueError("Binary-contact training may not receive pair geometry inputs")
        if distance_required:
            pair_geometry_manifest = (
                arguments.pair_geometry_cache_root
                / "PAIR_GEOMETRY_CACHE_MANIFEST.json"
            )
            if (
                sha256_file(pair_geometry_manifest)
                != arguments.pair_geometry_manifest_sha256
            ):
                raise ValueError("Pair geometry manifest SHA-256 differs")
        embedding_manifest = (
            arguments.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
        )
        if sha256_file(embedding_manifest) != arguments.embedding_manifest_sha256:
            raise ValueError("ESM embedding manifest SHA-256 differs")
    else:
        if any(value is not None for value in sabdab_values):
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
        if (
            arguments.pair_geometry_cache_root is not None
            or arguments.pair_geometry_manifest_sha256 is not None
        ):
            raise ValueError("AsEP backend has no authenticated pair geometry authority")
        if EXPERIMENTS[arguments.experiment].get(
            "pair_output_mode", "binary_contact"
        ) == "distogram":
            raise ValueError("Distogram experiments are not authorized on AsEP v1 cache")
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
    if arguments.seed not in protocol.get("model_selection", {}).get("seeds", []):
        raise ValueError("Seed was not pre-registered")
    if arguments.experiment not in protocol.get("development_matrix", []):
        raise ValueError("Experiment was not pre-registered in the development matrix")
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
        "attention_kernel": "attention_kernel",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "batch_size": "batch_size",
        "evaluation_batch_size": "evaluation_batch_size",
        "precision": "precision",
        "calibration_max_pairs": "calibration_max_pairs",
        "offload_saved_tensors": "offload_saved_tensors",
        "checkpoint_wrong_partners": "checkpoint_wrong_partners",
        "triangle_neighbor_radius": "triangle_neighbor_radius_angstrom",
        "triangle_neighbor_k": "triangle_neighbor_k",
        "triangle_slack": "triangle_slack_angstrom",
        "antigen_patch_top_seeds": "antigen_patch_top_seeds",
        "antigen_patch_radius": "antigen_patch_radius_angstrom",
        "edm_numerical_tolerance": "edm_numerical_tolerance",
        "contact_bottleneck_projection_dim": (
            "contact_bottleneck_projection_dim"
        ),
        "contact_bottleneck_gradient_mode": "contact_bottleneck_gradient_mode",
    }
    for argument_name, protocol_name in fields.items():
        observed = getattr(
            arguments,
            argument_name,
            1e-6 if argument_name == "edm_numerical_tolerance" else None,
        )
        expected = implementation.get(protocol_name)
        if expected is None:
            continue
        if isinstance(expected, float):
            equal = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            equal = observed == expected
        if not equal:
            raise ValueError(
                f"Execution parameter {argument_name} differs from the protocol"
            )
    objective = protocol.get("training_objective", {})
    loss_weights_by_condition = objective.get("loss_weights_by_condition")
    if loss_weights_by_condition is None:
        loss_weights = objective.get("loss_weights", {})
    else:
        if not isinstance(loss_weights_by_condition, Mapping):
            raise ValueError("Condition-specific loss weights are invalid")
        loss_weights = loss_weights_by_condition.get(arguments.experiment)
        if not isinstance(loss_weights, Mapping):
            raise ValueError("Experiment lacks condition-specific loss weights")
    for argument_name, protocol_name in (
        ("contact_loss_weight", "contact"),
        ("paratope_loss_weight", "paratope"),
        ("epitope_loss_weight", "epitope"),
        ("specificity_loss_weight", "specificity"),
        ("distogram_loss_weight", "distogram"),
        ("triangle_loss_weight", "triangle"),
        ("edm_loss_weight", "edm"),
        (
            "residue_partner_contrastive_loss_weight",
            "residue_partner_contrastive",
        ),
    ):
        observed_weight = getattr(
            arguments,
            argument_name,
            0.0
            if argument_name
            in {"edm_loss_weight", "residue_partner_contrastive_loss_weight"}
            else math.nan,
        )
        if protocol_name not in loss_weights:
            if math.isclose(
                float(observed_weight),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                continue
            raise ValueError(f"Loss weight {argument_name} is absent from the protocol")
        if not math.isclose(
            float(observed_weight),
            float(loss_weights.get(protocol_name, math.nan)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Loss weight {argument_name} differs from the protocol")
    for argument_name, protocol_name in (
        ("specificity_margin", "specificity_margin"),
        ("score_fraction", "score_fraction"),
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
    optional_objective_fields = {
        "specificity_loss_mode": "specificity_loss_mode",
        "specificity_negative_count": "specificity_negative_count",
        "specificity_temperature": "specificity_temperature",
        "specificity_direction_aggregation": (
            "specificity_direction_aggregation"
        ),
        "specificity_direction_temperature": (
            "specificity_direction_temperature"
        ),
        "residue_contrastive_margin": "residue_contrastive_margin",
        "residue_contrastive_mode": "residue_contrastive_mode",
        "residue_contrastive_temperature": "residue_contrastive_temperature",
    }
    for argument_name, protocol_name in optional_objective_fields.items():
        if protocol_name not in objective:
            continue
        observed = getattr(arguments, argument_name)
        expected = objective[protocol_name]
        if isinstance(expected, float):
            equal = math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
            )
        else:
            equal = observed == expected
        if not equal:
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


class _FrozenPanelTrainingDataset(torch.utils.data.Dataset):
    """Choose legal frozen wrong partners per focal complex and epoch."""

    def __init__(
        self,
        dataset: Any,
        panel: dict[str, Any],
        seed: int,
        negative_count: int = 1,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        if (
            isinstance(negative_count, bool)
            or not isinstance(negative_count, int)
            or negative_count <= 0
        ):
            raise ValueError("negative_count must be positive")
        self.negative_count = negative_count
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
                if len(entry[key]) < self.negative_count:
                    raise ValueError(
                        "Training panel has fewer candidates than negative_count"
                    )
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

    def _choose_many(
        self, focal_id: str, candidates: list[str], direction: str
    ) -> tuple[dict[str, Any], ...]:
        if self.negative_count == 1:
            return (self._choose(focal_id, candidates, direction),)
        ranked = sorted(
            candidates,
            key=lambda candidate_id: hashlib.sha256(
                (
                    f"{self.seed}\0{self.epoch}\0{direction}\0{focal_id}\0"
                    f"{candidate_id}"
                ).encode("utf-8")
            ).digest(),
        )
        return tuple(
            self.dataset[self.index_by_id[candidate_id]]
            for candidate_id in ranked[: self.negative_count]
        )

    def __getitem__(self, index: int) -> tuple[dict[str, Any], ...]:
        focal = self.dataset[index]
        entry = self.entries[focal["instance_id"]]
        if not self.reciprocal:
            candidates = entry["wrong_partner_instance_ids"]
            if self.negative_count > 1:
                chosen = self._choose_many(
                    focal["instance_id"], candidates, "wrong_antigen"
                )
                return focal, tuple(
                    _wrong_antigen_pair_item(focal, candidate)
                    for candidate in chosen
                )
            digest = hashlib.sha256(
                f"{self.seed}\0{self.epoch}\0{focal['instance_id']}".encode("utf-8")
            ).digest()
            candidate_id = candidates[
                int.from_bytes(digest[:8], "big") % len(candidates)
            ]
            candidate = self.dataset[self.index_by_id[candidate_id]]
            return focal, _wrong_antigen_pair_item(focal, candidate)
        wrong_antigens = self._choose_many(
            focal["instance_id"],
            entry["wrong_antigen_instance_ids"],
            "wrong_antigen",
        )
        wrong_antibodies = self._choose_many(
            focal["instance_id"],
            entry["wrong_antibody_instance_ids"],
            "wrong_antibody",
        )
        if self.negative_count == 1:
            return (
                focal,
                _wrong_antigen_pair_item(focal, wrong_antigens[0]),
                _wrong_antibody_pair_item(focal, wrong_antibodies[0]),
            )
        return (
            focal,
            tuple(
                _wrong_antigen_pair_item(focal, candidate)
                for candidate in wrong_antigens
            ),
            tuple(
                _wrong_antibody_pair_item(focal, candidate)
                for candidate in wrong_antibodies
            ),
        )


class _FrozenPanelTrainingCollator:
    def __init__(self, *, pair_geometry: bool = False) -> None:
        self.cognate = (
            Sabdab2PairGeometryBatchCollator()
            if pair_geometry
            else Sabdab2ContactBatchCollator()
        )
        self.wrong = Sabdab2StructureBatchCollator()

    def __call__(
        self, examples: list[tuple[dict[str, Any], ...]]
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        if len(examples[0]) == 2:
            cognate, wrong_antigen = zip(*examples)
            if isinstance(wrong_antigen[0], tuple):
                return self.cognate(cognate), {
                    "fixed_antibody_wrong_antigen": tuple(
                        self.wrong(tuple(items[negative_index] for items in wrong_antigen))
                        for negative_index in range(len(wrong_antigen[0]))
                    )
                }
            return self.cognate(cognate), {
                "fixed_antibody_wrong_antigen": self.wrong(wrong_antigen)
            }
        if len(examples[0]) != 3:
            raise ValueError("Frozen partner examples have an unsupported arity")
        cognate, wrong_antigen, wrong_antibody = zip(*examples)
        if isinstance(wrong_antigen[0], tuple):
            return self.cognate(cognate), {
                "fixed_antibody_wrong_antigen": tuple(
                    self.wrong(tuple(items[negative_index] for items in wrong_antigen))
                    for negative_index in range(len(wrong_antigen[0]))
                ),
                "fixed_antigen_wrong_antibody": tuple(
                    self.wrong(tuple(items[negative_index] for items in wrong_antibody))
                    for negative_index in range(len(wrong_antibody[0]))
                ),
            }
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
    collect_attention: bool = False,
    frozen_calibration: Mapping[str, float] | None = None,
    batch_intervention: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    compute_calibration_regression: bool = False,
) -> dict[str, Any]:
    model.eval()
    diagnostics = AttentionDiagnosticsAccumulator() if collect_attention else None
    cross_diagnostics = (
        CrossAttentionDiagnosticsAccumulator() if collect_attention else None
    )
    contact_logits_all: list[np.ndarray] = []
    contact_labels_all: list[np.ndarray] = []
    residue: dict[str, dict[str, list[Any]]] = {
        "paratope": {
            "labels": [],
            "logits": [],
            "logit_distribution_by_instance": {},
            "per_complex_auprc": [],
            "per_complex_auprc_by_instance": {},
            "positive_required_metric_ineligible_instance_ids": [],
        },
        "epitope": {
            "labels": [],
            "logits": [],
            "logit_distribution_by_instance": {},
            "per_complex_auprc": [],
            "per_complex_auprc_by_instance": {},
            "positive_required_metric_ineligible_instance_ids": [],
        },
    }
    contact_per_complex_auprc: list[float] = []
    contact_per_complex_auprc_by_instance: dict[str, float] = {}
    contact_positive_required_metric_ineligible_instance_ids: list[str] = []
    contact_ranked_by_instance: dict[str, dict[str, Any]] = {}
    specificity_evidence_by_instance: dict[str, np.ndarray] = {}
    contact_bottleneck_diagnostics_by_instance: dict[str, dict[str, Any]] = {}
    gauge_attribution_by_instance: dict[str, dict[str, float]] = {}
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
            contact_probabilities_local = _sigmoid_numpy(logits)
            antibody_length = int(output["antibody_mask"][batch_index].sum())
            antigen_length = int(output["antigen_mask"][batch_index].sum())
            contact_local_ap = positive_required_average_precision(
                labels, contact_probabilities_local
            )
            if not contact_local_ap["eligible"]:
                contact_positive_required_metric_ineligible_instance_ids.append(
                    str(instance_id)
                )
                ranked = {
                    "eligible": False,
                    "ineligible_reason": "zero_positive_contacts",
                    "length_definition": (
                        "Lmin=min(antibody_length,antigen_length)"
                    ),
                    "antibody_length": antibody_length,
                    "antigen_length": antigen_length,
                    "positive_count": 0,
                    "candidate_count": int(labels.size),
                    "cutoffs": None,
                }
            else:
                contact_local_value = float(contact_local_ap["value"])
                contact_per_complex_auprc.append(contact_local_value)
                contact_per_complex_auprc_by_instance[str(instance_id)] = (
                    contact_local_value
                )
                ranked = {
                    "eligible": True,
                    **ranked_pair_metrics(
                        labels,
                        contact_probabilities_local,
                        antibody_length=antibody_length,
                        antigen_length=antigen_length,
                    ),
                }
            if str(instance_id) in contact_ranked_by_instance:
                raise ValueError(f"Duplicate evaluation instance: {instance_id}")
            contact_ranked_by_instance[str(instance_id)] = ranked
            top_count = max(1, math.ceil(score_fraction * len(logits)))
            gauge_attribution = output.get("contact_gauge_attribution")
            if gauge_attribution is not None:
                gauge_residual = output["contact_gauge_centering_residual"]
                gauge_attribution_by_instance[str(instance_id)] = {
                    "partner_specific_share": float(
                        gauge_attribution["partner_specific_share"][batch_index]
                    ),
                    "antibody_propensity_share": float(
                        gauge_attribution["antibody_propensity_share"][batch_index]
                    ),
                    "antigen_propensity_share": float(
                        gauge_attribution["antigen_propensity_share"][batch_index]
                    ),
                    "partner_specific_variance": float(
                        gauge_attribution["partner_specific_variance"][batch_index]
                    ),
                    "total_variance": float(
                        gauge_attribution["total_variance"][batch_index]
                    ),
                    "variance_partition_gap": float(
                        gauge_attribution["partition_gap"][batch_index]
                    ),
                    "maximum_absolute_centering_residual": float(
                        gauge_residual["maximum_absolute_mean"][batch_index]
                    ),
                }
            if specificity_score_source == "contact_bottleneck_cosine":
                compatibility = output.get("contact_bottleneck_compatibility")
                if compatibility is None:
                    raise ValueError(
                        "contact_bottleneck_cosine evaluation requires "
                        "contact_bottleneck_compatibility"
                    )
                local_compatibility = float(
                    compatibility[batch_index].float().cpu().item()
                )
                if not math.isfinite(local_compatibility):
                    raise ValueError("contact bottleneck compatibility is non-finite")
                specificity_evidence_by_instance[str(instance_id)] = np.asarray(
                    [local_compatibility], dtype=np.float64
                )
                antibody_weights = output.get(
                    "contact_bottleneck_antibody_weights"
                )
                antigen_weights = output.get("contact_bottleneck_antigen_weights")
                pair_lift = output.get("contact_bottleneck_pair_contact_lift")
                if antibody_weights is None or antigen_weights is None or pair_lift is None:
                    raise ValueError(
                        "contact bottleneck evaluation requires attribution weights and lift"
                    )
                contact_bottleneck_diagnostics_by_instance[str(instance_id)] = {
                    "antibody": _weight_distribution_diagnostics(
                        antibody_weights[batch_index].float().cpu().numpy(),
                        output["antibody_mask"][batch_index].cpu().numpy(),
                    ),
                    "antigen": _weight_distribution_diagnostics(
                        antigen_weights[batch_index].float().cpu().numpy(),
                        output["antigen_mask"][batch_index].cpu().numpy(),
                    ),
                    "pair_contact_lift": _weight_distribution_diagnostics(
                        pair_lift[batch_index].float().cpu().numpy(),
                        pair_mask.cpu().numpy(),
                    ),
                }
            elif specificity_score_source in {
                "interaction_only",
                "gauge_interaction",
            }:
                specificity_key = (
                    "contact_interaction_logits"
                    if specificity_score_source == "interaction_only"
                    else "contact_gauge_interaction_logits"
                )
                specificity_source_tensor = output.get(specificity_key)
                if specificity_source_tensor is None:
                    raise ValueError(
                        f"{specificity_score_source} specificity requires "
                        f"{specificity_key} on the model output"
                    )
                specificity_logits = (
                    specificity_source_tensor[batch_index][pair_mask]
                    .float()
                    .cpu()
                    .numpy()
                )
                specificity_evidence_by_instance[str(instance_id)] = np.partition(
                    specificity_logits, len(specificity_logits) - top_count
                )[-top_count:]
            elif specificity_score_source == "total_contact":
                specificity_logits = logits
                specificity_evidence_by_instance[str(instance_id)] = np.partition(
                    specificity_logits, len(specificity_logits) - top_count
                )[-top_count:]
            else:
                raise ValueError("Unknown specificity score source")
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
                residue[name]["logit_distribution_by_instance"][
                    str(instance_id)
                ] = _residue_logit_distribution_diagnostics(local_logits)
                local_ap = positive_required_average_precision(
                    local_labels, _sigmoid_numpy(local_logits)
                )
                if not local_ap["eligible"]:
                    residue[name][
                        "positive_required_metric_ineligible_instance_ids"
                    ].append(str(instance_id))
                else:
                    local_value = float(local_ap["value"])
                    residue[name]["per_complex_auprc"].append(local_value)
                    residue[name]["per_complex_auprc_by_instance"][
                        str(instance_id)
                    ] = local_value
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
    if specificity_score_source == "contact_bottleneck_cosine":
        correct_scores = {
            identifier: float(values[0])
            for identifier, values in specificity_evidence_by_instance.items()
        }
        specificity_score_transform = "raw_cosine"
    else:
        correct_scores = {
            identifier: float(
                _sigmoid_numpy(values / specificity_temperature).mean()
            )
            for identifier, values in specificity_evidence_by_instance.items()
        }
        specificity_score_transform = "mean_sigmoid_of_top_fraction_logits"

    index_by_id = {
        record["instance_id"]: index for index, record in enumerate(dataset.records)
    }
    panel_entries = panel["entries"]
    if set(index_by_id) != {entry["instance_id"] for entry in panel_entries}:
        raise ValueError("Validation panel and validation dataset identifiers differ")
    structure_collator = Sabdab2StructureBatchCollator()
    ordered_entries = sorted(panel_entries, key=lambda value: value["instance_id"])
    # Every validation complex appears repeatedly across the reciprocal panel.
    # Cache each authenticated item once per evaluation so a 31x2 panel does
    # not re-hash and deserialize the same structure/embedding files tens of
    # thousands of times. The cache is local to this call and contains no test
    # role or predictions.
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
                if specificity_score_source == "contact_bottleneck_cosine":
                    scores = wrong_output.get("contact_bottleneck_compatibility")
                    if scores is None:
                        raise ValueError(
                            "contact_bottleneck_cosine evaluation requires "
                            "contact_bottleneck_compatibility"
                        )
                    scores = scores.float()
                else:
                    score_logits = {
                        "interaction_only": wrong_output["contact_interaction_logits"],
                        "gauge_interaction": wrong_output.get(
                            "contact_gauge_interaction_logits"
                        ),
                    }.get(specificity_score_source, wrong_output["contact_logits"])
                    if score_logits is None:
                        raise ValueError(
                            "gauge_interaction specificity requires a gauged model"
                        )
                    scores = top_fraction_contact_score(
                        score_logits.float(),
                        wrong_output["pair_candidate_mask"],
                        fraction=score_fraction,
                        temperature=specificity_temperature,
                    )
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError("partner retrieval score is non-finite")
                local_scores.extend(scores.cpu().tolist())
            cognate_scores.append(correct_scores[entry["instance_id"]])
            wrong_scores.append(local_scores)
            cluster_ids.append(entry[cluster_key])
        retrieval_by_direction[direction] = partner_retrieval_metrics(
            cognate_scores, wrong_scores, cluster_ids
        )
    selection_metric = min(
        value["cluster_weighted_cognate_compatibility_margin"]
        for value in retrieval_by_direction.values()
    )
    eligible_ranked_by_instance = {
        identifier: values
        for identifier, values in contact_ranked_by_instance.items()
        if values["eligible"]
    }
    if not eligible_ranked_by_instance:
        raise ValueError("No complex has a positive contact for ranked-pair metrics")
    ranked_cutoffs = next(iter(eligible_ranked_by_instance.values()))["cutoffs"]
    assert ranked_cutoffs is not None
    result: dict[str, Any] = {
        "selection_metric": selection_metric,
        "selection_metric_rule": "minimum_direction_cluster_weighted_margin",
        "specificity_score_source": specificity_score_source,
        "specificity_temperature": specificity_temperature,
        "specificity_score_transform": specificity_score_transform,
        "contact_gauge_attribution": (
            None
            if not gauge_attribution_by_instance
            else {
                "definition": (
                    "Two-way additive-effects decomposition of the validation "
                    "contact logit map. partner_specific_share is the fraction "
                    "of contact-map variance that no pair of single-chain "
                    "propensity profiles could reproduce; it is exactly zero "
                    "for an additively separable map."
                ),
                "mean_partner_specific_share": float(
                    np.mean(
                        [
                            values["partner_specific_share"]
                            for values in gauge_attribution_by_instance.values()
                        ]
                    )
                ),
                "mean_antibody_propensity_share": float(
                    np.mean(
                        [
                            values["antibody_propensity_share"]
                            for values in gauge_attribution_by_instance.values()
                        ]
                    )
                ),
                "mean_antigen_propensity_share": float(
                    np.mean(
                        [
                            values["antigen_propensity_share"]
                            for values in gauge_attribution_by_instance.values()
                        ]
                    )
                ),
                "maximum_absolute_centering_residual": float(
                    max(
                        values["maximum_absolute_centering_residual"]
                        for values in gauge_attribution_by_instance.values()
                    )
                ),
                "maximum_variance_partition_gap": float(
                    max(
                        values["variance_partition_gap"]
                        for values in gauge_attribution_by_instance.values()
                    )
                ),
                "by_instance": gauge_attribution_by_instance,
            }
        ),
        "contact_bottleneck_diagnostics": (
            None
            if not contact_bottleneck_diagnostics_by_instance
            else {
                "definition": (
                    "Descriptive concentration of pre-cross counterfactual positive "
                    "contact lift and its residue noisy-OR weights; no threshold or "
                    "checkpoint is selected from these diagnostics."
                ),
                "by_instance": contact_bottleneck_diagnostics_by_instance,
                "summary": {
                    scope: _aggregate_weight_diagnostics(
                        [
                            entry[scope]
                            for entry in contact_bottleneck_diagnostics_by_instance.values()
                        ]
                    )
                    for scope in (
                        "antibody",
                        "antigen",
                        "pair_contact_lift",
                    )
                },
            }
        ),
        "validation_temperature": temperature,
        "calibration_source": calibration_source,
        "calibration_fit_pair_count": calibration_fit_count,
        "contact": {
            **binary_classification_metrics(
                contact_labels,
                contact_probabilities,
                threshold=contact_threshold,
                include_calibration_regression=compute_calibration_regression,
            ),
            "fixed_raw_logit_threshold_zero": binary_classification_metrics(
                contact_labels,
                _sigmoid_numpy(contact_logits),
                threshold=0.5,
            ),
            "threshold_selected_on_validation": contact_threshold,
            "mean_per_complex_auprc": float(np.mean(contact_per_complex_auprc)),
            "per_complex_auprc_by_instance": (
                contact_per_complex_auprc_by_instance
            ),
            "complex_count": len(contact_ranked_by_instance),
            "positive_required_metric_eligible_complex_count": len(
                contact_per_complex_auprc
            ),
            "positive_required_metric_ineligible_complex_count": len(
                contact_positive_required_metric_ineligible_instance_ids
            ),
            "positive_required_metric_ineligible_instance_ids": sorted(
                contact_positive_required_metric_ineligible_instance_ids
            ),
            "zero_positive_policy": (
                "All candidates remain in pooled discrimination/calibration metrics. "
                "Complexes with zero positive contacts are explicitly ineligible for "
                "positive-required per-complex AP and ranked recall summaries."
            ),
            "ranked_pair_definition": (
                "fixed K plus Lmin=min(antibody_length,antigen_length); "
                "macro mean gives each positive-eligible complex equal weight"
            ),
            "ranked_pair_macro": {
                cutoff: {
                    "mean_precision": float(
                        np.mean(
                            [
                                values["cutoffs"][cutoff]["precision"]
                                for values in eligible_ranked_by_instance.values()
                            ]
                        )
                    ),
                    "mean_recall": float(
                        np.mean(
                            [
                                values["cutoffs"][cutoff]["recall"]
                                for values in eligible_ranked_by_instance.values()
                            ]
                        )
                    ),
                    "mean_effective_k": float(
                        np.mean(
                            [
                                values["cutoffs"][cutoff]["effective_k"]
                                for values in eligible_ranked_by_instance.values()
                            ]
                        )
                    ),
                }
                for cutoff in ranked_cutoffs
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
            **binary_classification_metrics(
                labels,
                probabilities,
                threshold=threshold,
                include_calibration_regression=compute_calibration_regression,
            ),
            "fixed_raw_logit_threshold_zero": binary_classification_metrics(
                labels,
                _sigmoid_numpy(logits),
                threshold=0.5,
            ),
            "threshold_selected_on_validation": threshold,
            "validation_temperature": residue_temperature,
            "calibration_source": residue_calibration_source,
            "calibration_fit_residue_count": residue_calibration_fit_count,
            "raw_logit_distribution": _aggregate_residue_logit_diagnostics(
                values["logit_distribution_by_instance"]
            ),
            "mean_per_complex_auprc": float(np.mean(values["per_complex_auprc"])),
            "per_complex_auprc_by_instance": values[
                "per_complex_auprc_by_instance"
            ],
            "complex_count": len(values["labels"]),
            "positive_required_metric_eligible_complex_count": len(
                values["per_complex_auprc"]
            ),
            "positive_required_metric_ineligible_complex_count": len(
                values["positive_required_metric_ineligible_instance_ids"]
            ),
            "positive_required_metric_ineligible_instance_ids": sorted(
                values["positive_required_metric_ineligible_instance_ids"]
            ),
            "zero_positive_policy": (
                "All residues remain in pooled discrimination/calibration metrics. "
                "Complexes with zero positive residues are explicitly ineligible for "
                "positive-required per-complex AP."
            ),
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
    bottleneck_projection_dim = configuration.get(
        "contact_bottleneck_projection_dim", 0
    )
    bottleneck_projection_dim = (
        None
        if bottleneck_projection_dim is None or int(bottleneck_projection_dim) == 0
        else int(bottleneck_projection_dim)
    )
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
        attention_kernel=str(configuration.get("attention_kernel", "reference")),
        fusion_mode=EXPERIMENTS[experiment]["fusion_mode"],
        pair_output_mode=EXPERIMENTS[experiment].get(
            "pair_output_mode", "binary_contact"
        ),
        residue_output_mode=EXPERIMENTS[experiment].get(
            "residue_output_mode", "gated_hybrid"
        ),
        residue_unary_source=EXPERIMENTS[experiment].get(
            "residue_unary_source", "post_cross"
        ),
        contact_marginalization=EXPERIMENTS[experiment].get(
            "contact_marginalization", "logmeanexp"
        ),
        pair_gauge_mode=EXPERIMENTS[experiment].get("pair_gauge_mode", "raw"),
        pair_gauge_iterations=int(
            EXPERIMENTS[experiment].get("pair_gauge_iterations", 1)
        ),
        contact_bottleneck_projection_dim=bottleneck_projection_dim,
        contact_bottleneck_gradient_mode=str(
            configuration.get(
                "contact_bottleneck_gradient_mode", "stop_gradient"
            )
        ),
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
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError("output-root already exists")
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    if arguments.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if arguments.calibration_max_pairs <= 0:
        raise ValueError("calibration-max-pairs must be positive")
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
        train_dataset = Sabdab2ContactDataset(
            train_base, arguments.contact_cache_root
        )
        validation_dataset = Sabdab2ContactDataset(
            validation_base, arguments.contact_cache_root
        )
        if arguments.pair_geometry_cache_root is not None:
            train_dataset = Sabdab2PairGeometryDataset(
                train_dataset,
                arguments.pair_geometry_cache_root,
                manifest_sha256=arguments.pair_geometry_manifest_sha256,
            )
            validation_dataset = Sabdab2PairGeometryDataset(
                validation_dataset,
                arguments.pair_geometry_cache_root,
                manifest_sha256=arguments.pair_geometry_manifest_sha256,
            )
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
    if arguments.authority_preflight_only:
        paired_preflight = _FrozenPanelTrainingDataset(
            train_dataset,
            training_panel,
            arguments.seed,
            negative_count=arguments.specificity_negative_count,
        )
        train_item = train_dataset[0]
        validation_item = validation_dataset[0]
        _FrozenPanelTrainingCollator(
            pair_geometry=arguments.pair_geometry_cache_root is not None
        )([paired_preflight[0]])
        validation_collator = (
            Sabdab2PairGeometryBatchCollator()
            if arguments.pair_geometry_cache_root is not None
            else Sabdab2ContactBatchCollator()
        )
        validation_collator([validation_item])
        model = _model_from_configuration(
            train_item,
            vars(arguments),
            arguments.experiment,
        )
        print(
            json.dumps(
                {
                    "format_version": "framecontact_authority_preflight_v1",
                    "status": "valid",
                    "dataset_backend": arguments.dataset_backend,
                    "experiment": arguments.experiment,
                    "seed": arguments.seed,
                    "train_complexes": len(train_dataset),
                    "validation_complexes": len(validation_dataset),
                    "training_panel_format": training_panel["format_version"],
                    "validation_panel_format": panel["format_version"],
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "requested_device": arguments.device,
                    "requested_precision": arguments.precision,
                    "output_root_created": False,
                    "test_role_loaded": False,
                    "claim_boundary": (
                        "Authority and construction preflight only; no optimizer "
                        "step, prediction metric, test access or hardware claim."
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    device = _device(arguments.device)
    use_bf16 = arguments.precision == "bf16" and device.type == "cuda"
    if arguments.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 requires CUDA")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise ValueError("CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG")
    paired_train_dataset = _FrozenPanelTrainingDataset(
        train_dataset,
        training_panel,
        arguments.seed,
        negative_count=arguments.specificity_negative_count,
    )
    train_generator = torch.Generator().manual_seed(arguments.seed)
    train_loader = DataLoader(
        paired_train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=arguments.workers,
        collate_fn=_FrozenPanelTrainingCollator(
            pair_geometry=arguments.pair_geometry_cache_root is not None
        ),
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
    first = train_dataset[0]
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
        distogram=arguments.distogram_loss_weight,
        triangle=arguments.triangle_loss_weight,
        edm=arguments.edm_loss_weight,
        residue_partner_contrastive=(
            arguments.residue_partner_contrastive_loss_weight
        ),
    )
    specificity_score_source = str(
        protocol.get("training_objective", {}).get(
            "specificity_score_source", "total_contact"
        )
    )
    bottleneck_enabled = model.contact_bottleneck_head is not None
    bottleneck_requested = specificity_score_source == "contact_bottleneck_cosine"
    if bottleneck_enabled != bottleneck_requested:
        raise ValueError(
            "contact bottleneck enablement must exactly match the frozen "
            "specificity_score_source"
        )
    objective = FrameContactObjective(
        weights=weights,
        specificity_margin=arguments.specificity_margin,
        score_fraction=arguments.score_fraction,
        specificity_score_source=specificity_score_source,
        residue_contrastive_margin=arguments.residue_contrastive_margin,
        residue_contrastive_mode=arguments.residue_contrastive_mode,
        residue_contrastive_temperature=arguments.residue_contrastive_temperature,
        specificity_loss_mode=arguments.specificity_loss_mode,
        specificity_temperature=arguments.specificity_temperature,
        specificity_direction_aggregation=(
            arguments.specificity_direction_aggregation
        ),
        specificity_direction_temperature=(
            arguments.specificity_direction_temperature
        ),
        triangle_neighbor_radius=arguments.triangle_neighbor_radius,
        triangle_neighbor_k=arguments.triangle_neighbor_k,
        triangle_slack=arguments.triangle_slack,
        antigen_patch_top_seeds=arguments.antigen_patch_top_seeds,
        antigen_patch_radius=arguments.antigen_patch_radius,
        edm_numerical_tolerance=arguments.edm_numerical_tolerance,
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
    best_contact_auprc = -math.inf
    best_epoch = 0
    stale_epochs = 0
    best_path = temporary_root / "best_validation.pt"
    started = time.monotonic()
    try:
        for epoch in range(1, arguments.epochs + 1):
            model.train()
            paired_train_dataset.set_epoch(epoch)
            component_sums: dict[str, float] = {}
            steps = 0
            for raw_batch, raw_wrong_by_direction in train_loader:
                batch = _move_batch(raw_batch, device)
                wrong_batches = {
                    direction: (
                        tuple(_move_batch(item, device) for item in raw_wrong)
                        if isinstance(raw_wrong, (tuple, list))
                        else _move_batch(raw_wrong, device)
                    )
                    for direction, raw_wrong in raw_wrong_by_direction.items()
                }
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, use_bf16), _saved_tensor_context(
                    arguments.offload_saved_tensors
                ):
                    output = _cognate_training_view(model(batch))
                    wrong_outputs = {
                        direction: (
                            tuple(
                                (
                                    _checkpointed_wrong_training_view(model, item)
                                    if arguments.checkpoint_wrong_partners
                                    else _wrong_training_view(model(item))
                                )
                                for item in wrong_batch
                            )
                            if isinstance(wrong_batch, tuple)
                            else (
                                _checkpointed_wrong_training_view(model, wrong_batch)
                                if arguments.checkpoint_wrong_partners
                                else _wrong_training_view(model(wrong_batch))
                            )
                        )
                        for direction, wrong_batch in wrong_batches.items()
                    }
                    losses = objective(
                        output, batch, wrong_partner_outputs=wrong_outputs
                    )
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
            )
            record = {
                "epoch": epoch,
                "train": {key: value / steps for key, value in component_sums.items()},
                "validation": validation,
                # The applied gate is tanh of the raw parameter and is the
                # amount of contact evidence the residue head actually uses.
                # A gate that stays at zero means the contact map contributed
                # nothing to residue prediction, whatever the losses did.
                "residue_contact_gate": {
                    "paratope_raw": float(model.paratope_contact_gate.detach()),
                    "epitope_raw": float(model.epitope_contact_gate.detach()),
                    "paratope_applied": float(
                        torch.tanh(model.paratope_contact_gate.detach())
                    ),
                    "epitope_applied": float(
                        torch.tanh(model.epitope_contact_gate.detach())
                    ),
                    "residue_output_mode": model.residue_output_mode,
                    "pair_gauge_mode": model.pair_gauge_mode,
                    "gauge_marginalization": (
                        "centered_logmeanexp"
                        if model.pair_gauge_mode == "doubly_centered"
                        else None
                    ),
                },
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(record)
            metric = float(validation["selection_metric"])
            contact_auprc = float(validation["contact"]["mean_per_complex_auprc"])
            improved = metric > best_metric + 1e-4 or (
                abs(metric - best_metric) <= 1e-4 and contact_auprc > best_contact_auprc
            )
            if improved:
                best_metric = metric
                best_contact_auprc = contact_auprc
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
        )
        configuration = {
            key: value
            for key, value in vars(arguments).items()
            if key
            not in {
                "structure_cache_root",
                "contact_cache_root",
                "pair_geometry_cache_root",
                "embedding_root",
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
                "pair_geometry_manifest_sha256": (
                    arguments.pair_geometry_manifest_sha256
                ),
                "embedding_manifest_sha256": arguments.embedding_manifest_sha256,
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
                    else "external_embedding_cache"
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
                "best_validation_compatibility_margin": best_metric,
                "tie_break_contact_mean_complex_auprc": best_contact_auprc,
                "epochs_completed": len(history),
            },
            "history": history,
            "selected_checkpoint_validation": final_validation,
        }
        if EXTERNAL_BASELINE_AUTHORITY is not None:
            result["external_baseline_authority"] = dict(
                EXTERNAL_BASELINE_AUTHORITY
            )
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
