"""Contact-guided CDR sequence proposal without structural diffusion claims.

This module implements a small MaskGIT-style iterative sampler.  It keeps the
antibody framework fixed, proposes canonical amino acids only at a caller-
supplied CDR mask, and reranks proposals with a frozen residue-pair contact
scorer against a caller-supplied antigen hotspot.  It is an integration PoC,
not an RFdiffusion-equivalent backbone generator or evidence of binding.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class ContactGuidedCDRProposal:
    antibody_token_ids: tuple[int, ...]
    proposal_log_probability: float
    hotspot_contact_score: float
    off_target_contact_score: float
    combined_score: float
    seed: int


def _validate_vector(tensor: Tensor, *, dtype: torch.dtype, label: str) -> None:
    if tensor.ndim != 1 or tensor.dtype != dtype:
        raise ValueError(f"{label} must be a one-dimensional {dtype} tensor")


def contact_guidance_objective(
    pair_logits: Tensor,
    cdr_mask: Tensor,
    hotspot_mask: Tensor,
    *,
    off_target_weight: float = 0.25,
) -> tuple[Tensor, Tensor, Tensor]:
    """Score target-hotspot contacts while penalizing off-target CDR contacts."""

    if pair_logits.ndim != 3:
        raise ValueError("pair_logits must have shape [K,L_ab,L_ag]")
    _validate_vector(cdr_mask, dtype=torch.bool, label="cdr_mask")
    _validate_vector(hotspot_mask, dtype=torch.bool, label="hotspot_mask")
    if pair_logits.shape[1:] != (len(cdr_mask), len(hotspot_mask)):
        raise ValueError("Contact map axes do not match the CDR and hotspot masks")
    if not bool(cdr_mask.any()) or not bool(hotspot_mask.any()):
        raise ValueError("Contact guidance requires non-empty CDR and hotspot masks")
    if not math.isfinite(off_target_weight) or off_target_weight < 0.0:
        raise ValueError("off_target_weight must be finite and non-negative")
    if not bool(torch.isfinite(pair_logits).all()):
        raise ValueError("Contact logits must be finite")

    probabilities = torch.sigmoid(pair_logits.float())[:, cdr_mask, :]
    hotspot = probabilities[:, :, hotspot_mask].mean(dim=(1, 2))
    if bool((~hotspot_mask).any()):
        off_target = probabilities[:, :, ~hotspot_mask].mean(dim=(1, 2))
    else:
        off_target = torch.zeros_like(hotspot)
    return hotspot - off_target_weight * off_target, hotspot, off_target


def _sample_one(
    framework_tokens: Tensor,
    antigen_tokens: Tensor,
    cdr_mask: Tensor,
    hotspot_mask: Tensor,
    denoiser: Callable[[Tensor, Tensor, Tensor], Tensor],
    contact_scorer: Callable[[Tensor, Tensor], Tensor],
    canonical_token_ids: Tensor,
    *,
    mask_token_id: int,
    steps: int,
    proposals_per_step: int,
    temperature: float,
    contact_guidance_weight: float,
    off_target_weight: float,
    seed: int,
) -> ContactGuidedCDRProposal:
    generator = torch.Generator(device=framework_tokens.device)
    generator.manual_seed(seed % (2**63 - 1))
    current = framework_tokens.clone()
    current[cdr_mask] = mask_token_id
    cdr_positions = torch.nonzero(cdr_mask, as_tuple=False).squeeze(1)
    confidence = torch.full(
        (len(cdr_positions),), -torch.inf, dtype=torch.float32, device=current.device
    )
    position_log_probability = torch.full_like(confidence, -torch.inf)
    last_values: tuple[float, float, float, float] | None = None

    for step in range(1, steps + 1):
        active_local = current[cdr_positions].eq(mask_token_id)
        active_positions = cdr_positions[active_local]
        if not bool(active_local.any()):
            raise RuntimeError("CDR remasking schedule terminated before the final step")
        batch = current.unsqueeze(0).repeat(proposals_per_step, 1)
        antigens = antigen_tokens.unsqueeze(0).repeat(proposals_per_step, 1)
        timestep = torch.full(
            (proposals_per_step,),
            (steps - step + 1) / steps,
            dtype=torch.float32,
            device=framework_tokens.device,
        )
        with torch.no_grad():
            logits = denoiser(batch, antigens, timestep)
        if logits.ndim != 3 or logits.shape[:2] != batch.shape:
            raise ValueError("Denoiser must return [K,L_ab,vocabulary] logits")
        if int(canonical_token_ids.max()) >= logits.shape[-1]:
            raise ValueError("A canonical token id is outside the denoiser vocabulary")
        canonical_logits = logits[:, active_positions][:, :, canonical_token_ids]
        if not bool(torch.isfinite(canonical_logits).all()):
            raise ValueError("Denoiser returned non-finite CDR logits")
        probabilities = torch.softmax(canonical_logits.float() / temperature, dim=-1)
        sampled_indices = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]),
            1,
            generator=generator,
        ).reshape(proposals_per_step, len(active_positions))
        sampled_tokens = canonical_token_ids[sampled_indices]
        candidates = batch.clone()
        candidates[:, active_positions] = sampled_tokens
        selected_probabilities = probabilities.gather(
            -1, sampled_indices.unsqueeze(-1)
        ).squeeze(-1)
        candidate_log_probability = position_log_probability.unsqueeze(0).repeat(
            proposals_per_step, 1
        )
        candidate_log_probability[:, active_local] = selected_probabilities.clamp_min(
            1e-12
        ).log()
        proposal_scores = candidate_log_probability.mean(dim=1)

        with torch.no_grad():
            pair_logits = contact_scorer(candidates, antigens)
        guidance, hotspot, off_target = contact_guidance_objective(
            pair_logits,
            cdr_mask,
            hotspot_mask,
            off_target_weight=off_target_weight,
        )
        combined = proposal_scores + contact_guidance_weight * guidance
        winner = int(torch.argmax(combined))
        current = candidates[winner].clone()
        confidence[active_local] = selected_probabilities[winner]
        position_log_probability = candidate_log_probability[winner].clone()
        last_values = (
            float(proposal_scores[winner]),
            float(hotspot[winner]),
            float(off_target[winner]),
            float(combined[winner]),
        )

        if step < steps:
            keep_count = math.floor(len(cdr_positions) * step / steps)
            keep_count = min(keep_count, len(cdr_positions) - 1)
            remask = torch.ones(len(cdr_positions), dtype=torch.bool, device=current.device)
            if keep_count:
                keep_local = torch.topk(confidence, keep_count, sorted=False).indices
                remask[keep_local] = False
            current[cdr_positions[remask]] = mask_token_id
            confidence[remask] = -torch.inf
            position_log_probability[remask] = -torch.inf

    if last_values is None or bool(current[cdr_mask].eq(mask_token_id).any()):
        raise RuntimeError("CDR generation did not produce a complete proposal")
    if not torch.equal(current[~cdr_mask], framework_tokens[~cdr_mask]):
        raise RuntimeError("CDR generation modified an immutable framework token")
    proposal_log_probability, hotspot_score, off_target_score, combined_score = last_values
    return ContactGuidedCDRProposal(
        antibody_token_ids=tuple(int(value) for value in current.tolist()),
        proposal_log_probability=proposal_log_probability,
        hotspot_contact_score=hotspot_score,
        off_target_contact_score=off_target_score,
        combined_score=combined_score,
        seed=seed,
    )


def generate_contact_guided_cdr_proposals(
    framework_tokens: Tensor,
    antigen_tokens: Tensor,
    cdr_mask: Tensor,
    hotspot_mask: Tensor,
    denoiser: Callable[[Tensor, Tensor, Tensor], Tensor],
    contact_scorer: Callable[[Tensor, Tensor], Tensor],
    canonical_token_ids: Sequence[int],
    *,
    mask_token_id: int,
    steps: int = 4,
    proposals_per_step: int = 8,
    output_count: int = 4,
    temperature: float = 1.0,
    contact_guidance_weight: float = 1.0,
    off_target_weight: float = 0.25,
    seed: int = 17,
) -> list[ContactGuidedCDRProposal]:
    """Generate deterministic-seed CDR proposals and rank them by frozen scores."""

    _validate_vector(framework_tokens, dtype=torch.long, label="framework_tokens")
    _validate_vector(antigen_tokens, dtype=torch.long, label="antigen_tokens")
    _validate_vector(cdr_mask, dtype=torch.bool, label="cdr_mask")
    _validate_vector(hotspot_mask, dtype=torch.bool, label="hotspot_mask")
    if len(framework_tokens) != len(cdr_mask) or len(antigen_tokens) != len(hotspot_mask):
        raise ValueError("Token and mask lengths differ")
    if framework_tokens.device != antigen_tokens.device:
        raise ValueError("Antibody and antigen tokens must share a device")
    if cdr_mask.device != framework_tokens.device or hotspot_mask.device != antigen_tokens.device:
        raise ValueError("Tokens and masks must share a device")
    if not bool(cdr_mask.any()) or not bool(hotspot_mask.any()):
        raise ValueError("Generation requires non-empty CDR and hotspot masks")
    if any(value < 1 for value in (steps, proposals_per_step, output_count)):
        raise ValueError("Generation counts must be positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(contact_guidance_weight) or contact_guidance_weight < 0.0:
        raise ValueError("contact_guidance_weight must be finite and non-negative")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    canonical = tuple(int(token) for token in canonical_token_ids)
    if not canonical or len(canonical) != len(set(canonical)) or min(canonical) < 0:
        raise ValueError("canonical_token_ids must be unique non-negative ids")
    if mask_token_id in canonical:
        raise ValueError("mask token cannot be a canonical amino-acid token")
    if bool(framework_tokens[~cdr_mask].eq(mask_token_id).any()):
        raise ValueError("immutable framework positions cannot contain mask tokens")
    if bool(antigen_tokens.eq(mask_token_id).any()):
        raise ValueError("antigen conditioning tokens cannot contain mask tokens")
    canonical_tensor = torch.tensor(canonical, dtype=torch.long, device=framework_tokens.device)

    proposals = [
        _sample_one(
            framework_tokens,
            antigen_tokens,
            cdr_mask,
            hotspot_mask,
            denoiser,
            contact_scorer,
            canonical_tensor,
            mask_token_id=mask_token_id,
            steps=steps,
            proposals_per_step=proposals_per_step,
            temperature=temperature,
            contact_guidance_weight=contact_guidance_weight,
            off_target_weight=off_target_weight,
            seed=seed + index,
        )
        for index in range(output_count)
    ]
    return sorted(proposals, key=lambda proposal: proposal.combined_score, reverse=True)
