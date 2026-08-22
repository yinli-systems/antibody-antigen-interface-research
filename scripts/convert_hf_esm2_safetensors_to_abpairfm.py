#!/usr/bin/env python3
"""Convert an authenticated Hugging Face ESM2 checkpoint without changing values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch
from torch import Tensor


IGNORED_EXACT_KEYS = {
    "esm.contact_head.regression.bias",
    "esm.contact_head.regression.weight",
    "esm.embeddings.position_embeddings.weight",
    "esm.embeddings.position_ids",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_hf_esm2_key(name: str) -> str | None:
    """Map one HF ESM2 tensor name to the clean-room ESM2-compatible name."""
    if name in IGNORED_EXACT_KEYS:
        return None
    exact = {
        "esm.embeddings.word_embeddings.weight": "embed_tokens.weight",
        "esm.encoder.emb_layer_norm_after.bias": "emb_layer_norm_after.bias",
        "esm.encoder.emb_layer_norm_after.weight": "emb_layer_norm_after.weight",
        "lm_head.bias": "lm_head.bias",
        "lm_head.dense.bias": "lm_head.dense.bias",
        "lm_head.dense.weight": "lm_head.dense.weight",
        "lm_head.layer_norm.bias": "lm_head.layer_norm.bias",
        "lm_head.layer_norm.weight": "lm_head.layer_norm.weight",
    }
    if name in exact:
        return exact[name]
    prefix = "esm.encoder.layer."
    if not name.startswith(prefix):
        raise ValueError(f"Unsupported Hugging Face ESM2 key: {name}")
    layer, separator, suffix = name[len(prefix) :].partition(".")
    if not separator or not layer.isdigit():
        raise ValueError(f"Malformed Hugging Face ESM2 layer key: {name}")
    suffixes = {
        "attention.self.query.weight": "self_attn.q_proj.weight",
        "attention.self.query.bias": "self_attn.q_proj.bias",
        "attention.self.key.weight": "self_attn.k_proj.weight",
        "attention.self.key.bias": "self_attn.k_proj.bias",
        "attention.self.value.weight": "self_attn.v_proj.weight",
        "attention.self.value.bias": "self_attn.v_proj.bias",
        "attention.self.rotary_embeddings.inv_freq": "self_attn.rot_emb.inv_freq",
        "attention.output.dense.weight": "self_attn.out_proj.weight",
        "attention.output.dense.bias": "self_attn.out_proj.bias",
        "attention.LayerNorm.weight": "self_attn_layer_norm.weight",
        "attention.LayerNorm.bias": "self_attn_layer_norm.bias",
        "intermediate.dense.weight": "fc1.weight",
        "intermediate.dense.bias": "fc1.bias",
        "output.dense.weight": "fc2.weight",
        "output.dense.bias": "fc2.bias",
        "LayerNorm.weight": "final_layer_norm.weight",
        "LayerNorm.bias": "final_layer_norm.bias",
    }
    if suffix not in suffixes:
        raise ValueError(f"Unsupported Hugging Face ESM2 layer key: {name}")
    return f"layers.{int(layer)}.{suffixes[suffix]}"


def convert_state(state: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], list[str]]:
    """Map every admitted tensor and recreate the tied checkpoint-only duplicate."""
    converted: dict[str, Tensor] = {}
    ignored: list[str] = []
    for source_name, tensor in state.items():
        destination_name = map_hf_esm2_key(source_name)
        if destination_name is None:
            ignored.append(source_name)
            continue
        if destination_name in converted:
            raise ValueError(f"Duplicate converted ESM2 key: {destination_name}")
        converted[destination_name] = tensor
    embedding = converted.get("embed_tokens.weight")
    if embedding is None:
        raise ValueError("Converted checkpoint is missing token embeddings")
    converted["lm_head.weight"] = embedding
    return converted, sorted(ignored)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-size", type=int, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-layer-count", type=int, required=True)
    parser.add_argument("--expected-embed-dimension", type=int, required=True)
    return parser.parse_args()


def _publish_torch_create_only(payload: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, output)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> None:
    arguments = _arguments()
    if arguments.input.is_symlink() or not arguments.input.is_file():
        raise RuntimeError("Input checkpoint must be a regular file")
    if arguments.input.stat().st_size != arguments.expected_input_size:
        raise RuntimeError("Input checkpoint byte size differs")
    if sha256_file(arguments.input) != arguments.expected_input_sha256:
        raise RuntimeError("Input checkpoint SHA-256 differs")
    if arguments.output.exists() or arguments.receipt.exists():
        raise FileExistsError("Conversion outputs are create-only")

    from safetensors import safe_open

    source: dict[str, Tensor] = {}
    with safe_open(arguments.input, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            source[name] = handle.get_tensor(name)
    converted, ignored = convert_state(source)
    layer_indices = {
        int(name.split(".")[1])
        for name in converted
        if name.startswith("layers.")
    }
    if layer_indices != set(range(arguments.expected_layer_count)):
        raise RuntimeError("Converted ESM2 layer set differs")
    embedding = converted["embed_tokens.weight"]
    if tuple(embedding.shape) != (33, arguments.expected_embed_dimension):
        raise RuntimeError("Converted ESM2 embedding shape differs")

    _publish_torch_create_only({"model": converted}, arguments.output)
    receipt = {
        "claim_boundary": "Authenticated value-preserving format conversion only; single-chain numerical parity and downstream ability require separate gates.",
        "format_version": "abpairfm_hf_esm2_conversion_v1",
        "ignored_source_keys": ignored,
        "input": {
            "path": str(arguments.input),
            "sha256": arguments.expected_input_sha256,
            "size_bytes": arguments.expected_input_size,
        },
        "output": {
            "path": str(arguments.output),
            "sha256": sha256_file(arguments.output),
            "size_bytes": arguments.output.stat().st_size,
            "state_key_count": len(converted),
            "tensor_element_count": sum(tensor.numel() for tensor in converted.values()),
        },
        "status": "PASS_FORMAT_CONVERSION",
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    with arguments.receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
