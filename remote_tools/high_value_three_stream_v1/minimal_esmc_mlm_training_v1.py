#!/usr/bin/env python3
"""Trainable ESM-C 600M MLM wrapper for the frozen minimal runtime."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from minimal_esmc_runtime_v1 import MinimalESMC600M, PAD_TOKEN_ID, regression_head


class MinimalESMC600MForMLM(nn.Module):
    def __init__(self, body: MinimalESMC600M, sequence_head: nn.Module):
        super().__init__()
        self.body = body
        self.sequence_head = sequence_head

    def forward(self, sequence_tokens: torch.Tensor) -> torch.Tensor:
        sequence_id = sequence_tokens != PAD_TOKEN_ID
        hidden = self.body.embed(sequence_tokens)
        for block in self.body.transformer.blocks:
            if self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, sequence_id, use_reentrant=False)
            else:
                hidden = block(hidden, sequence_id)
        hidden = self.body.transformer.norm(hidden)
        return self.sequence_head(hidden)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_base_body_key(key: str) -> str | None:
    if key.startswith("lm_head.") or key.endswith("._extra_state"):
        return None
    if not key.startswith("esmc."):
        raise RuntimeError(f"unexpected base checkpoint key: {key}")
    key = key.removeprefix("esmc.")
    replacements = (
        (".layernorm_qkv.layer_norm_weight", ".layernorm_qkv.0.weight"),
        (".layernorm_qkv.layer_norm_bias", ".layernorm_qkv.0.bias"),
        (".layernorm_qkv.weight", ".layernorm_qkv.1.weight"),
        (".ffn.layer_norm_weight", ".ffn.0.weight"),
        (".ffn.layer_norm_bias", ".ffn.0.bias"),
        (".ffn.fc1_weight", ".ffn.1.weight"),
        (".ffn.fc2_weight", ".ffn.3.weight"),
    )
    for source, target in replacements:
        if source in key:
            return key.replace(source, target)
    return key


def load_base_model(checkpoint_path: Path, safetensors_site: Path) -> MinimalESMC600MForMLM:
    import sys

    sys.path.append(str(safetensors_site))
    from safetensors import safe_open

    body_state: dict[str, torch.Tensor] = {}
    head_state: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("lm_head."):
                head_key = key.removeprefix("lm_head.")
                if head_key in head_state:
                    raise RuntimeError("duplicate MLM head key")
                head_state[head_key] = handle.get_tensor(key)
                continue
            normalized = normalize_base_body_key(key)
            if normalized is None:
                continue
            if normalized in body_state:
                raise RuntimeError("duplicate normalized body key")
            body_state[normalized] = handle.get_tensor(key)
    with torch.device("meta"):
        body = MinimalESMC600M()
        head = regression_head(1152, 64)
    body_incompatible = body.load_state_dict(body_state, strict=True, assign=True)
    head_incompatible = head.load_state_dict(head_state, strict=True, assign=True)
    if (
        body_incompatible.missing_keys
        or body_incompatible.unexpected_keys
        or head_incompatible.missing_keys
        or head_incompatible.unexpected_keys
    ):
        raise RuntimeError("strict ESM-C MLM state load differs")
    body.materialize_nonpersistent_buffers(torch.device("cpu"))
    model = MinimalESMC600MForMLM(body, head)
    if sum(parameter.numel() for parameter in body.parameters()) != 573_632_640:
        raise RuntimeError("ESM-C body parameter count differs")
    return model


def state_probe_sha256(model: MinimalESMC600MForMLM) -> str:
    names = (
        "body.embed.weight",
        "body.transformer.blocks.0.attn.layernorm_qkv.1.weight",
        "body.transformer.blocks.35.ffn.3.weight",
        "sequence_head.0.weight",
        "sequence_head.3.weight",
    )
    parameters = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in names:
        tensor = parameters[name].detach().float().cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
