"""Fail-closed adapter for an extracted official Boltz-1 Pairformer trunk.

The adapter deliberately separates two claims:

* the Pairformer weights came from a SHA-256-authenticated official checkpoint;
* the ESM/local-pose projections into those pretrained latent spaces are new
  task adapters and are not part of Boltz pretraining.

Consequently this is an adapted pretrained baseline, not a reproduction of the
native Boltz structure-prediction pipeline.
"""

from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .intra_chain_geometry import (
    build_backbone_frames,
    build_intra_chain_relative_geometry,
)

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - optional dependency boundary
    torch = None
    nn = None


_ModuleBase = nn.Module if nn is not None else object

BOLTZ_REVISION = "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc"
BOLTZ1_CHECKPOINT_SHA256 = (
    "fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5"
)
PAIRFORMER_ARTIFACT_FORMAT = "boltz1_pairformer_extracted_v1"
LOCAL_POSE_PAIR_FEATURE_TAIL_DIM = 14


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for the Boltz Pairformer adapter")


def _source_revision(source_root: Path) -> str:
    if not (source_root / ".git").is_dir():
        raise FileNotFoundError("Boltz source root is not a Git checkout")
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_official_pairformer(source_root: Path) -> Any:
    source_root = source_root.resolve()
    if _source_revision(source_root) != BOLTZ_REVISION:
        raise RuntimeError("Boltz source revision differs from the pinned revision")
    python_root = source_root / "src"
    if not python_root.is_dir():
        raise FileNotFoundError("Pinned Boltz source lacks its src directory")
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    module = importlib.import_module("boltz.model.modules.trunk")
    return module.PairformerModule


class OfficialBoltzPairformerAdapter(_ModuleBase):
    """Project task features through frozen, authenticated Boltz Pairformer weights."""

    def __init__(
        self,
        *,
        artifact_path: str | Path,
        receipt_path: str | Path,
        boltz_source_root: str | Path,
        single_input_dim: int,
        pair_input_dim: int,
        output_dim: int,
        freeze_pairformer: bool = True,
    ) -> None:
        _require_torch()
        super().__init__()
        artifact_path = Path(artifact_path)
        receipt = read_pairformer_receipt(receipt_path)
        if receipt.get("artifact_sha256") != _sha256(artifact_path):
            raise ValueError("Pairformer artifact SHA-256 differs from its receipt")
        if receipt.get("source_checkpoint_sha256") != BOLTZ1_CHECKPOINT_SHA256:
            raise ValueError("Pairformer receipt checkpoint SHA-256 differs")
        if receipt.get("boltz_source_revision") != BOLTZ_REVISION:
            raise ValueError("Pairformer receipt source revision differs")
        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if artifact.get("format_version") != PAIRFORMER_ARTIFACT_FORMAT:
            raise ValueError("Unknown Pairformer artifact format")
        if artifact.get("boltz_source_revision") != BOLTZ_REVISION:
            raise ValueError("Pairformer artifact source revision differs")
        if artifact.get("source_checkpoint_sha256") != BOLTZ1_CHECKPOINT_SHA256:
            raise ValueError("Pairformer artifact checkpoint SHA-256 differs")
        architecture = artifact.get("architecture")
        state_dict = artifact.get("state_dict")
        if not isinstance(architecture, dict) or not isinstance(state_dict, dict):
            raise ValueError("Pairformer artifact is incomplete")
        required_architecture = {
            "token_s",
            "token_z",
            "num_blocks",
            "num_heads",
            "dropout",
            "pairwise_head_width",
            "pairwise_num_heads",
        }
        if set(architecture) != required_architecture:
            raise ValueError("Pairformer architecture fields differ from the contract")
        pairformer_class = _import_official_pairformer(Path(boltz_source_root))
        self.pairformer = pairformer_class(**architecture)
        self.pairformer.load_state_dict(state_dict, strict=True)
        self.freeze_pairformer = bool(freeze_pairformer)
        if self.freeze_pairformer:
            self.pairformer.requires_grad_(False)
        token_s = int(architecture["token_s"])
        token_z = int(architecture["token_z"])
        self.single_input_projection = nn.Linear(single_input_dim, token_s)
        self.pair_input_projection = nn.Linear(pair_input_dim, token_z)
        self.output_projection = nn.Linear(token_s, output_dim)
        self.pair_output_norm = nn.LayerNorm(token_z)
        self.artifact_metadata = {
            key: artifact[key]
            for key in (
                "source_checkpoint_sha256",
                "boltz_source_revision",
                "state_prefix",
                "state_tensor_count",
            )
        }

    def train(self, mode: bool = True) -> "OfficialBoltzPairformerAdapter":
        super().train(mode)
        if self.freeze_pairformer:
            self.pairformer.eval()
        return self

    def forward(
        self,
        single_features: Any,
        pair_features: Any,
        residue_mask: Any,
        pair_mask: Any,
    ) -> dict[str, Any]:
        if single_features.ndim != 3 or pair_features.ndim != 4:
            raise ValueError("single and pair features must be rank 3 and 4")
        batch_size, residue_count, _ = single_features.shape
        if tuple(pair_features.shape[:3]) != (
            batch_size,
            residue_count,
            residue_count,
        ):
            raise ValueError("pair features must align with the residue axes")
        if residue_mask.dtype != torch.bool or tuple(residue_mask.shape) != (
            batch_size,
            residue_count,
        ):
            raise ValueError("residue_mask must be boolean and align with residues")
        if pair_mask.dtype != torch.bool or tuple(pair_mask.shape) != (
            batch_size,
            residue_count,
            residue_count,
        ):
            raise ValueError("pair_mask must be boolean and align with residue pairs")
        permitted = residue_mask[:, :, None] & residue_mask[:, None, :]
        if bool((pair_mask & ~permitted).any()):
            raise ValueError("pair_mask contains a padding edge")
        single = self.single_input_projection(single_features)
        pair = self.pair_input_projection(pair_features)
        single, pair = self.pairformer(
            single,
            pair,
            residue_mask.to(single.dtype),
            pair_mask.to(pair.dtype),
            use_kernels=False,
        )
        single = self.output_projection(single)
        single = single.masked_fill(~residue_mask[..., None], 0.0)
        pair = self.pair_output_norm(pair)
        pair = pair.masked_fill(~pair_mask[..., None], 0.0)
        return {
            "single_features": single,
            "pair_features": pair,
            "residue_mask": residue_mask,
            "pair_mask": pair_mask,
            "pretrained_pairformer_frozen": self.freeze_pairformer,
            "provenance": dict(self.artifact_metadata),
        }


class OfficialBoltzPairformerMonomerEncoder(_ModuleBase):
    """Adapt real ``N/CA/C`` local-pose features to the official Pairformer.

    This wrapper is the runnable bridge from the project's structure-batch
    contract to :class:`OfficialBoltzPairformerAdapter`.  It computes only
    intra-chain SE(3)-invariant pair features; it never subtracts coordinates
    belonging to two independently framed monomers.
    """

    def __init__(
        self,
        *,
        artifact_path: str | Path,
        receipt_path: str | Path,
        boltz_source_root: str | Path,
        single_input_dim: int,
        output_dim: int,
        geometry_feature_mode: str = "full_pose",
        attention_graph_mode: str = "dense",
        num_rbf: int = 16,
        distance_max: float = 20.0,
        neighbor_k: int | None = 32,
        neighbor_radius: float | None = None,
        sequence_window: int = 32,
        freeze_pairformer: bool = True,
    ) -> None:
        _require_torch()
        super().__init__()
        if isinstance(num_rbf, bool) or not isinstance(num_rbf, int) or num_rbf <= 0:
            raise ValueError("num_rbf must be a positive integer")
        self.single_input_dim = int(single_input_dim)
        self.geometry_feature_mode = geometry_feature_mode
        self.attention_graph_mode = attention_graph_mode
        self.num_rbf = num_rbf
        self.distance_max = distance_max
        self.neighbor_k = neighbor_k
        self.neighbor_radius = neighbor_radius
        self.sequence_window = sequence_window
        self.adapter = OfficialBoltzPairformerAdapter(
            artifact_path=artifact_path,
            receipt_path=receipt_path,
            boltz_source_root=boltz_source_root,
            single_input_dim=single_input_dim,
            pair_input_dim=num_rbf + LOCAL_POSE_PAIR_FEATURE_TAIL_DIM,
            output_dim=output_dim,
            freeze_pairformer=freeze_pairformer,
        )

    def forward(
        self,
        features: Any,
        residue_mask: Any,
        backbone: Any,
        chain_ids: Any | None = None,
    ) -> dict[str, Any]:
        if not torch.is_tensor(features) or features.ndim != 3:
            raise ValueError("features must have shape [batch, residue, channels]")
        if int(features.shape[-1]) != self.single_input_dim:
            raise ValueError(
                f"features must have {self.single_input_dim} input channels"
            )
        frames = build_backbone_frames(backbone, residue_mask)
        geometry = build_intra_chain_relative_geometry(
            frames,
            residue_mask,
            chain_ids=chain_ids,
            geometry_feature_mode=self.geometry_feature_mode,
            attention_graph_mode=self.attention_graph_mode,
            num_rbf=self.num_rbf,
            distance_max=self.distance_max,
            neighbor_k=self.neighbor_k,
            neighbor_radius=self.neighbor_radius,
            sequence_window=self.sequence_window,
        )
        output = self.adapter(
            features,
            geometry.features,
            residue_mask,
            geometry.neighbor_mask,
        )
        output.update(
            {
                "frame_valid_mask": frames.valid_mask,
                "geometry_valid_mask": geometry.geometry_valid_mask,
                "distance": geometry.distance,
                "geometry_feature_mode": self.geometry_feature_mode,
                "attention_graph_mode": self.attention_graph_mode,
            }
        )
        return output


class DualOfficialBoltzPairformerMonomerEncoder(_ModuleBase):
    """Two parameter-disjoint antibody/antigen Pairformer adapters.

    Both frozen trunks start from the same authenticated official weights, but
    they are distinct module instances and their trainable input/output
    adapters are never shared.  The explicit duplication is intentional: it
    preserves the two-encoder architecture contract at the cost of memory.
    """

    def __init__(self, **monomer_kwargs: Any) -> None:
        _require_torch()
        super().__init__()
        self.antibody_encoder = OfficialBoltzPairformerMonomerEncoder(
            **monomer_kwargs
        )
        self.antigen_encoder = OfficialBoltzPairformerMonomerEncoder(
            **monomer_kwargs
        )

    def forward(
        self,
        *,
        antibody_features: Any,
        antibody_residue_mask: Any,
        antibody_backbone: Any,
        antigen_features: Any,
        antigen_residue_mask: Any,
        antigen_backbone: Any,
        antibody_chain_ids: Any | None = None,
        antigen_chain_ids: Any | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        antibody = self.antibody_encoder(
            antibody_features,
            antibody_residue_mask,
            antibody_backbone,
            antibody_chain_ids,
        )
        antigen = self.antigen_encoder(
            antigen_features,
            antigen_residue_mask,
            antigen_backbone,
            antigen_chain_ids,
        )
        return antibody, antigen


def read_pairformer_receipt(path: str | Path) -> dict[str, Any]:
    """Read the small JSON receipt paired with an extracted weight artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != "boltz1_pairformer_extraction_receipt_v1":
        raise ValueError("Unknown Pairformer extraction receipt")
    return payload


__all__ = [
    "BOLTZ1_CHECKPOINT_SHA256",
    "BOLTZ_REVISION",
    "DualOfficialBoltzPairformerMonomerEncoder",
    "LOCAL_POSE_PAIR_FEATURE_TAIL_DIM",
    "OfficialBoltzPairformerAdapter",
    "OfficialBoltzPairformerMonomerEncoder",
    "PAIRFORMER_ARTIFACT_FORMAT",
    "read_pairformer_receipt",
]
