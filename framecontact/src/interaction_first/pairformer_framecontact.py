"""Contact-first task head over authenticated, frozen Boltz-1 Pairformer trunks.

This is an *adapted* baseline.  It does not reproduce native Boltz inputs or
structure heads, and it intentionally lives outside the frozen confirmatory
FrameContact registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .boltz_pairformer_adapter import OfficialBoltzPairformerMonomerEncoder
from .framecontact import (
    BidirectionalContactGatedCrossAttention,
    FrameContactModel,
    PairContactHead,
    masked_logmeanexp,
)


class AdaptedBoltzPairformerContactModel(nn.Module):
    """Two unshared frozen Pairformers plus the matched contact-first decoder."""

    def __init__(
        self,
        *,
        artifact_path: str | Path,
        receipt_path: str | Path,
        boltz_source_root: str | Path,
        antibody_input_dim: int,
        antigen_input_dim: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_rbf: int = 16,
        neighbor_k: int | None = 32,
        neighbor_radius: float | None = 16.0,
        sequence_window: int = 32,
        pair_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        shared = {
            "artifact_path": artifact_path,
            "receipt_path": receipt_path,
            "boltz_source_root": boltz_source_root,
            "output_dim": d_model,
            "geometry_feature_mode": "full_pose",
            "attention_graph_mode": "dense",
            "num_rbf": num_rbf,
            "neighbor_k": neighbor_k,
            "neighbor_radius": neighbor_radius,
            "sequence_window": sequence_window,
            "freeze_pairformer": True,
        }
        self.antibody_encoder = OfficialBoltzPairformerMonomerEncoder(
            single_input_dim=antibody_input_dim,
            **shared,
        )
        self.antigen_encoder = OfficialBoltzPairformerMonomerEncoder(
            single_input_dim=antigen_input_dim,
            **shared,
        )
        self.d_model = d_model
        self.fusion_mode = "late"
        self.contact_head = PairContactHead(d_model, pair_dim)
        self.cross_attention = BidirectionalContactGatedCrossAttention(
            d_model, num_heads, dropout=dropout
        )
        self.antibody_norm = nn.LayerNorm(d_model)
        self.antigen_norm = nn.LayerNorm(d_model)
        self.paratope_head = nn.Linear(d_model, 1)
        self.epitope_head = nn.Linear(d_model, 1)
        self.paratope_contact_gate = nn.Parameter(torch.zeros(()))
        self.epitope_contact_gate = nn.Parameter(torch.zeros(()))
        nn.init.constant_(self.paratope_head.bias, -2.5)
        nn.init.constant_(self.epitope_head.bias, -2.5)

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_attention: bool = False,
    ) -> dict[str, Any]:
        antibody_mask = batch["antibody_mask"]
        antigen_mask = batch["antigen_mask"]
        FrameContactModel._validate_frame_partition(batch, "antibody")
        FrameContactModel._validate_frame_partition(batch, "antigen")
        antibody_output = self.antibody_encoder(
            batch["antibody_features"],
            antibody_mask,
            batch["antibody_backbone"],
            batch["antibody_chain_ids"],
        )
        antigen_output = self.antigen_encoder(
            batch["antigen_features"],
            antigen_mask,
            batch["antigen_backbone"],
            batch["antigen_chain_ids"],
        )
        antibody = antibody_output["single_features"]
        antigen = antigen_output["single_features"]
        pair_mask = antibody_mask[:, :, None] & antigen_mask[:, None, :]
        if "pair_candidate_mask" in batch:
            candidate = batch["pair_candidate_mask"]
            if (
                candidate.dtype != torch.bool
                or tuple(candidate.shape) != tuple(pair_mask.shape)
                or bool((candidate & ~pair_mask).any())
            ):
                raise ValueError(
                    "pair_candidate_mask must be a subset of real residue pairs"
                )
            pair_mask = pair_mask & candidate
        if not bool(pair_mask.flatten(start_dim=1).any(dim=-1).all()):
            raise ValueError("every example must contain a candidate residue pair")
        initial_components = self.contact_head.components(
            antibody, antigen, pair_mask
        )
        initial_contact_logits = initial_components["logits"]
        antibody_fused, antigen_fused, cross_details = self.cross_attention(
            antibody,
            antigen,
            antibody_mask,
            antigen_mask,
            initial_contact_logits,
            pair_mask,
            return_attention=return_attention,
        )
        final_components = self.contact_head.components(
            antibody_fused, antigen_fused, pair_mask
        )
        contact_logits = final_components["logits"]
        antibody_fused = self.antibody_norm(antibody_fused).masked_fill(
            ~antibody_mask[..., None], 0.0
        )
        antigen_fused = self.antigen_norm(antigen_fused).masked_fill(
            ~antigen_mask[..., None], 0.0
        )
        paratope_contact = masked_logmeanexp(contact_logits, pair_mask, dim=2)
        epitope_contact = masked_logmeanexp(contact_logits, pair_mask, dim=1)
        paratope_logits = self.paratope_head(antibody_fused).squeeze(-1)
        epitope_logits = self.epitope_head(antigen_fused).squeeze(-1)
        paratope_logits = paratope_logits + torch.tanh(
            self.paratope_contact_gate
        ) * paratope_contact
        epitope_logits = epitope_logits + torch.tanh(
            self.epitope_contact_gate
        ) * epitope_contact
        paratope_logits = paratope_logits.masked_fill(~antibody_mask, 0.0)
        epitope_logits = epitope_logits.masked_fill(~antigen_mask, 0.0)
        return {
            "initial_contact_logits": initial_contact_logits,
            "initial_contact_interaction_logits": initial_components[
                "interaction_logits"
            ],
            "intermediate_contact_logits": (initial_contact_logits,),
            "contact_logits": contact_logits,
            "contact_interaction_logits": final_components["interaction_logits"],
            "antibody_contact_propensity_logits": final_components[
                "antibody_unary_logits"
            ].squeeze(-1),
            "antigen_contact_propensity_logits": final_components[
                "antigen_unary_logits"
            ].squeeze(1),
            "pair_candidate_mask": pair_mask,
            "paratope_contact_logits": paratope_contact,
            "epitope_contact_logits": epitope_contact,
            "paratope_logits": paratope_logits,
            "epitope_logits": epitope_logits,
            "antibody_mask": antibody_mask,
            "antigen_mask": antigen_mask,
            "self_attention_details": None,
            "cross_attention_details": cross_details,
            "pairformer_pair_features": {
                "antibody": antibody_output["pair_features"],
                "antigen": antigen_output["pair_features"],
            },
            "pairformer_provenance": {
                "antibody": antibody_output["provenance"],
                "antigen": antigen_output["provenance"],
            },
            "fusion_mode": self.fusion_mode,
            "baseline_family": "adapted_boltz1_pairformer_contact",
        }


__all__ = ["AdaptedBoltzPairformerContactModel"]
