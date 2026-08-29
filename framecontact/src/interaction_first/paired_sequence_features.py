"""Provenance-bound paired-sequence features for FrameContact.

This module is deliberately a clean adapter boundary.  It does not vendor or
import PPLM source code.  An external, pinned PPLM environment may materialize
safe ``.npz`` features, which are then authenticated before they enter the
FrameContact data path.

The boundary matters for two independent reasons:

* paired embeddings/attention are scientifically different from frozen
  single-sequence ESM embeddings because they already depend on the partner;
* the PPLM paper describes the code as MIT licensed while the current official
  repository README states PolyForm Noncommercial.  Until that discrepancy is
  resolved by the authors, this project treats PPLM as an external baseline and
  does not redistribute its code or weights.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - optional dependency boundary
    torch = None
    nn = None

from .sabdab2_structure import authenticate_file, sha256_file


PPLM_OFFICIAL_REPOSITORY = "https://github.com/junliu621/PPLM"
PPLM_OFFICIAL_REVISION = "1267db4592703095b889cc22da6cf1796ce038b5"
PPLM_PAPER_URL = "https://doi.org/10.1038/s41467-026-70457-5"
PPLM_CACHE_FORMAT = "framecontact_pplm_paired_features_v1"
PPLM_LICENSE_BOUNDARY = (
    "paper_says_mit_current_repository_says_polyform_noncommercial_unresolved"
)
PPLM_ATTENTION_REDUCTION = {
    "method": "symmetric_interchain_attention_layer_group_mean",
    "source_layer_count": 33,
    "source_head_count": 20,
    "layer_groups": [[0, 8], [8, 16], [16, 24], [24, 33]],
}

_ModuleBase = nn.Module if nn is not None else object
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for paired-sequence conditioning")
    return torch


def sequence_sha256(sequence: str) -> str:
    normalized = "".join(str(sequence).split()).upper()
    if not normalized or not normalized.isascii():
        raise ValueError("sequence must be nonempty ASCII")
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def paired_feature_cache_key(
    antibody_sequence_sha256: str, antigen_sequence_sha256: str
) -> str:
    """Return an ordered cache key bound to both interaction endpoints."""

    antibody_digest = _sha256(
        antibody_sequence_sha256, "antibody_sequence_sha256"
    )
    antigen_digest = _sha256(antigen_sequence_sha256, "antigen_sequence_sha256")
    payload = f"antibody:{antibody_digest}\nantigen:{antigen_digest}\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _sha256(value: Any, name: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _shape(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty shape list")
    shape = tuple(int(item) for item in value)
    if any(item <= 0 for item in shape):
        raise ValueError(f"{name} dimensions must be positive")
    return shape


def load_pplm_pair_feature_artifact(
    artifact_path: str | Path,
    receipt_path: str | Path,
    *,
    expected_antibody_sequence_sha256: str | None = None,
    expected_antigen_sequence_sha256: str | None = None,
) -> dict[str, Any]:
    """Load one safe PPLM feature artifact and fail closed on provenance drift.

    The artifact must contain exactly three numeric arrays:

    ``antibody_embedding``: ``[L_ab, C]``
    ``antigen_embedding``: ``[L_ag, C]``
    ``inter_attention``: ``[H, L_ab, L_ag]``

    Pickled object arrays are never accepted.  This is both a security property
    and a reproducibility property: the cache contract is independent of the
    external PPLM Python environment.
    """

    artifact_path = Path(artifact_path)
    receipt_path = Path(receipt_path)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("PPLM receipt must be a regular non-symlink file")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("PPLM receipt is not valid JSON") from error
    if not isinstance(receipt, dict):
        raise ValueError("PPLM receipt must be a JSON object")
    if receipt.get("format_version") != PPLM_CACHE_FORMAT:
        raise ValueError("unknown PPLM cache format")
    if receipt.get("source_repository") != PPLM_OFFICIAL_REPOSITORY:
        raise ValueError("PPLM receipt repository differs from the official source")
    if receipt.get("source_revision") != PPLM_OFFICIAL_REVISION:
        raise ValueError("PPLM receipt revision differs from the pinned revision")
    if receipt.get("license_boundary") != PPLM_LICENSE_BOUNDARY:
        raise ValueError("PPLM license boundary was not explicitly preserved")
    if receipt.get("attention_reduction") != PPLM_ATTENTION_REDUCTION:
        raise ValueError("PPLM attention reduction differs from the frozen contract")
    if receipt.get("chain_order") != {
        "sequence_A": "antibody",
        "sequence_B": "antigen",
    }:
        raise ValueError("PPLM chain order must be antibody then antigen")
    artifact_sha256 = _sha256(receipt.get("artifact_sha256"), "artifact_sha256")
    authenticate_file(artifact_path, artifact_sha256, "PPLM feature artifact")
    _sha256(receipt.get("source_checkpoint_sha256"), "source_checkpoint_sha256")
    antibody_digest = _sha256(
        receipt.get("antibody_sequence_sha256"), "antibody_sequence_sha256"
    )
    antigen_digest = _sha256(
        receipt.get("antigen_sequence_sha256"), "antigen_sequence_sha256"
    )
    if (
        expected_antibody_sequence_sha256 is not None
        and antibody_digest
        != _sha256(
            expected_antibody_sequence_sha256,
            "expected_antibody_sequence_sha256",
        )
    ):
        raise ValueError("PPLM antibody sequence digest differs")
    if (
        expected_antigen_sequence_sha256 is not None
        and antigen_digest
        != _sha256(
            expected_antigen_sequence_sha256,
            "expected_antigen_sequence_sha256",
        )
    ):
        raise ValueError("PPLM antigen sequence digest differs")

    declared_arrays = receipt.get("arrays")
    if not isinstance(declared_arrays, dict) or set(declared_arrays) != {
        "antibody_embedding",
        "antigen_embedding",
        "inter_attention",
    }:
        raise ValueError("PPLM receipt must declare exactly the contracted arrays")
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            if set(archive.files) != set(declared_arrays):
                raise ValueError("PPLM artifact arrays differ from its receipt")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("PPLM"):
            raise
        raise ValueError("PPLM artifact is not a safe numeric NPZ archive") from error

    for name, array in arrays.items():
        declaration = declared_arrays[name]
        if not isinstance(declaration, dict):
            raise ValueError(f"PPLM {name} declaration must be an object")
        if tuple(array.shape) != _shape(declaration.get("shape"), f"{name}.shape"):
            raise ValueError(f"PPLM {name} shape differs from its receipt")
        if str(array.dtype) != declaration.get("dtype"):
            raise ValueError(f"PPLM {name} dtype differs from its receipt")
        if array.dtype.kind not in {"f"} or not np.isfinite(array).all():
            raise ValueError(f"PPLM {name} must be finite floating-point data")
    antibody_embedding = arrays["antibody_embedding"]
    antigen_embedding = arrays["antigen_embedding"]
    inter_attention = arrays["inter_attention"]
    if antibody_embedding.ndim != 2 or antigen_embedding.ndim != 2:
        raise ValueError("PPLM residue embeddings must be rank two")
    if antibody_embedding.shape[1] != antigen_embedding.shape[1]:
        raise ValueError("PPLM residue embedding dimensions differ by partner")
    if inter_attention.ndim != 3 or inter_attention.shape[1:] != (
        antibody_embedding.shape[0],
        antigen_embedding.shape[0],
    ):
        raise ValueError("PPLM inter_attention does not align with residue lengths")
    if inter_attention.shape[0] != len(PPLM_ATTENTION_REDUCTION["layer_groups"]):
        raise ValueError("PPLM inter_attention depth groups differ from the contract")
    if np.any(inter_attention < 0):
        raise ValueError("PPLM inter_attention must be nonnegative")
    return {
        **arrays,
        "provenance": {
            "artifact_sha256": artifact_sha256,
            "receipt_sha256": sha256_file(receipt_path),
            "source_checkpoint_sha256": receipt["source_checkpoint_sha256"],
            "source_revision": PPLM_OFFICIAL_REVISION,
            "license_boundary": PPLM_LICENSE_BOUNDARY,
            "antibody_sequence_sha256": antibody_digest,
            "antigen_sequence_sha256": antigen_digest,
        },
    }


def reduce_pplm_inter_attention(attention_ab: Any, attention_ba: Any) -> Any:
    """Compress official 33x20 bidirectional attention into four pair maps.

    Inputs have shape ``[layer, head, L_ab, L_ag]`` and
    ``[layer, head, L_ag, L_ab]``.  The reduction first averages directions,
    then averages every frozen layer group and all 20 source heads.  Four maps
    retain coarse depth information while avoiding the prohibitive storage of
    all 660 source maps for every SAbDab2 pair.
    """

    torch_module = _require_torch()
    expected_layers = int(PPLM_ATTENTION_REDUCTION["source_layer_count"])
    expected_heads = int(PPLM_ATTENTION_REDUCTION["source_head_count"])
    if attention_ab.ndim != 4 or attention_ba.ndim != 4:
        raise ValueError("PPLM directional attention tensors must be rank four")
    if tuple(attention_ab.shape[:2]) != (expected_layers, expected_heads):
        raise ValueError("PPLM A-to-B attention layer/head dimensions differ")
    if tuple(attention_ba.shape[:2]) != (expected_layers, expected_heads):
        raise ValueError("PPLM B-to-A attention layer/head dimensions differ")
    if tuple(attention_ba.shape[2:]) != (
        int(attention_ab.shape[3]),
        int(attention_ab.shape[2]),
    ):
        raise ValueError("PPLM directional attention residue dimensions differ")
    if not torch_module.is_floating_point(attention_ab) or not torch_module.is_floating_point(
        attention_ba
    ):
        raise ValueError("PPLM attention tensors must be floating point")
    if not bool(torch_module.isfinite(attention_ab).all()) or not bool(
        torch_module.isfinite(attention_ba).all()
    ):
        raise ValueError("PPLM attention tensors must be finite")
    symmetric = 0.5 * (
        attention_ab.float() + attention_ba.transpose(-1, -2).float()
    )
    if bool((symmetric < 0).any()):
        raise ValueError("PPLM attention probabilities must be nonnegative")
    reduced = []
    for start, stop in PPLM_ATTENTION_REDUCTION["layer_groups"]:
        reduced.append(symmetric[int(start) : int(stop)].mean(dim=(0, 1)))
    result = torch_module.stack(reduced, dim=0)
    expected_shape = (
        len(PPLM_ATTENTION_REDUCTION["layer_groups"]),
        int(attention_ab.shape[2]),
        int(attention_ab.shape[3]),
    )
    if tuple(result.shape) != expected_shape:
        raise RuntimeError("PPLM attention reduction produced an invalid shape")
    return result


class PPLMPairFeatureStore:
    """Fail-closed exact-pair cache; cognate features cannot survive a swap."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("PPLM pair feature root must be a regular directory")

    def paths(
        self, antibody_sequence_sha256: str, antigen_sequence_sha256: str
    ) -> tuple[Path, Path]:
        key = paired_feature_cache_key(
            antibody_sequence_sha256, antigen_sequence_sha256
        )
        parent = self.root / key[:2]
        return parent / f"{key}.npz", parent / f"{key}.receipt.json"

    def load(
        self, antibody_sequence_sha256: str, antigen_sequence_sha256: str
    ) -> dict[str, Any]:
        artifact, receipt = self.paths(
            antibody_sequence_sha256, antigen_sequence_sha256
        )
        if not artifact.is_file() or not receipt.is_file():
            raise FileNotFoundError(
                "No exact PPLM feature artifact exists for this ordered pair"
            )
        return load_pplm_pair_feature_artifact(
            artifact,
            receipt,
            expected_antibody_sequence_sha256=antibody_sequence_sha256,
            expected_antigen_sequence_sha256=antigen_sequence_sha256,
        )


def validate_paired_sequence_tensors(
    antibody_base: Any,
    antigen_base: Any,
    antibody_paired: Any,
    antigen_paired: Any,
    inter_attention: Any,
    antibody_mask: Any,
    antigen_mask: Any,
    *,
    paired_embedding_dim: int,
    attention_head_count: int,
) -> None:
    torch_module = _require_torch()
    if antibody_base.ndim != 3 or antigen_base.ndim != 3:
        raise ValueError("base residue features must be rank three")
    batch_size, antibody_count = antibody_base.shape[:2]
    if int(antigen_base.shape[0]) != batch_size:
        raise ValueError("antibody and antigen batches differ")
    antigen_count = int(antigen_base.shape[1])
    if tuple(antibody_paired.shape) != (
        batch_size,
        antibody_count,
        paired_embedding_dim,
    ) or tuple(antigen_paired.shape) != (
        batch_size,
        antigen_count,
        paired_embedding_dim,
    ):
        raise ValueError("paired residue embeddings do not match the declared shape")
    if tuple(inter_attention.shape) != (
        batch_size,
        attention_head_count,
        antibody_count,
        antigen_count,
    ):
        raise ValueError("paired inter_attention does not match the declared shape")
    if (
        tuple(antibody_mask.shape) != (batch_size, antibody_count)
        or tuple(antigen_mask.shape) != (batch_size, antigen_count)
        or antibody_mask.dtype != torch_module.bool
        or antigen_mask.dtype != torch_module.bool
    ):
        raise ValueError("paired feature masks must be boolean and align with residues")
    for name, value in (
        ("antibody_paired", antibody_paired),
        ("antigen_paired", antigen_paired),
        ("inter_attention", inter_attention),
    ):
        if not torch_module.is_floating_point(value) or not bool(
            torch_module.isfinite(value).all()
        ):
            raise ValueError(f"{name} must contain finite floating-point values")
    if bool((inter_attention < 0).any()):
        raise ValueError("inter_attention must be nonnegative")


class PPLMPairedConditioner(_ModuleBase):
    """Zero-gated fusion of PPLM paired embeddings and inter-chain attention.

    Antibody and antigen residue projections are intentionally parameter
    unshared, matching the two-stream FrameContact encoder.  At initialization
    both residue gates and the pair-prior gate are exactly zero, so enabling the
    module cannot silently change the established single-sequence baseline.
    """

    def __init__(
        self,
        *,
        antibody_base_dim: int,
        antigen_base_dim: int,
        paired_embedding_dim: int,
        attention_head_count: int,
    ) -> None:
        _require_torch()
        super().__init__()
        dimensions = {
            "antibody_base_dim": antibody_base_dim,
            "antigen_base_dim": antigen_base_dim,
            "paired_embedding_dim": paired_embedding_dim,
            "attention_head_count": attention_head_count,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.paired_embedding_dim = paired_embedding_dim
        self.attention_head_count = attention_head_count
        self.antibody_norm = nn.LayerNorm(paired_embedding_dim)
        self.antigen_norm = nn.LayerNorm(paired_embedding_dim)
        self.antibody_projection = nn.Linear(
            paired_embedding_dim, antibody_base_dim, bias=False
        )
        self.antigen_projection = nn.Linear(
            paired_embedding_dim, antigen_base_dim, bias=False
        )
        self.antibody_residue_gate = nn.Parameter(torch.zeros(()))
        self.antigen_residue_gate = nn.Parameter(torch.zeros(()))
        self.attention_head_norm = nn.LayerNorm(attention_head_count)
        self.attention_projection = nn.Linear(attention_head_count, 1, bias=False)
        nn.init.normal_(
            self.attention_projection.weight,
            mean=0.0,
            std=1.0 / math.sqrt(attention_head_count),
        )
        self.pair_prior_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        antibody_base: Any,
        antigen_base: Any,
        antibody_paired: Any,
        antigen_paired: Any,
        inter_attention: Any,
        antibody_mask: Any,
        antigen_mask: Any,
    ) -> dict[str, Any]:
        validate_paired_sequence_tensors(
            antibody_base,
            antigen_base,
            antibody_paired,
            antigen_paired,
            inter_attention,
            antibody_mask,
            antigen_mask,
            paired_embedding_dim=self.paired_embedding_dim,
            attention_head_count=self.attention_head_count,
        )
        antibody_gate = torch.tanh(self.antibody_residue_gate)
        antigen_gate = torch.tanh(self.antigen_residue_gate)
        antibody = antibody_base + antibody_gate * self.antibody_projection(
            self.antibody_norm(antibody_paired)
        )
        antigen = antigen_base + antigen_gate * self.antigen_projection(
            self.antigen_norm(antigen_paired)
        )
        antibody = antibody.masked_fill(~antibody_mask[..., None], 0.0)
        antigen = antigen.masked_fill(~antigen_mask[..., None], 0.0)

        # Official PPLM returns nonnegative attention probabilities.  Log-space
        # prevents a few large heads from dominating before the learned head
        # projection; LayerNorm operates across heads at each residue pair.
        pair_features = torch.log(inter_attention.float().clamp_min(1e-12))
        pair_features = pair_features.permute(0, 2, 3, 1)
        raw_prior = self.attention_projection(
            self.attention_head_norm(pair_features)
        ).squeeze(-1)
        pair_mask = antibody_mask[:, :, None] & antigen_mask[:, None, :]
        raw_prior = raw_prior.masked_fill(~pair_mask, 0.0)
        pair_gate = torch.tanh(self.pair_prior_gate)
        prior = (pair_gate * raw_prior).masked_fill(~pair_mask, 0.0)
        return {
            "antibody_features": antibody,
            "antigen_features": antigen,
            "pair_prior_logits": prior,
            "raw_pair_prior_logits": raw_prior,
            "pair_mask": pair_mask,
            "antibody_residue_gate": antibody_gate,
            "antigen_residue_gate": antigen_gate,
            "pair_prior_gate": pair_gate,
            "source_revision": PPLM_OFFICIAL_REVISION,
            "license_boundary": PPLM_LICENSE_BOUNDARY,
        }


__all__ = [
    "PPLM_ATTENTION_REDUCTION",
    "PPLM_CACHE_FORMAT",
    "PPLM_LICENSE_BOUNDARY",
    "PPLM_OFFICIAL_REPOSITORY",
    "PPLM_OFFICIAL_REVISION",
    "PPLM_PAPER_URL",
    "PPLMPairedConditioner",
    "PPLMPairFeatureStore",
    "load_pplm_pair_feature_artifact",
    "paired_feature_cache_key",
    "reduce_pplm_inter_attention",
    "sequence_sha256",
    "validate_paired_sequence_tensors",
]
