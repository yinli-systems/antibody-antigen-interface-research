#!/usr/bin/env python3
"""Audit the pinned public MINT checkpoint without importing third-party code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch
from torch import Tensor


LAYER_COUNT = 33
EMBED_DIMENSION = 1280
FFN_DIMENSION = 5120
HEAD_COUNT = 20
VOCABULARY_SIZE = 33
EXPECTED_SOURCE_REVISION = "06694b7606e2d00b76ec58daf5c7aecdaf7cd283"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "emb_layer_norm_after.bias": (EMBED_DIMENSION,),
        "emb_layer_norm_after.weight": (EMBED_DIMENSION,),
        "embed_tokens.weight": (VOCABULARY_SIZE, EMBED_DIMENSION),
        "lm_head.bias": (VOCABULARY_SIZE,),
        "lm_head.dense.bias": (EMBED_DIMENSION,),
        "lm_head.dense.weight": (EMBED_DIMENSION, EMBED_DIMENSION),
        "lm_head.layer_norm.bias": (EMBED_DIMENSION,),
        "lm_head.layer_norm.weight": (EMBED_DIMENSION,),
        "lm_head.weight": (VOCABULARY_SIZE, EMBED_DIMENSION),
    }
    for layer in range(LAYER_COUNT):
        prefix = f"layers.{layer}."
        for attention in ("self_attn", "multimer_attn"):
            projections = ("k_proj", "q_proj", "v_proj")
            if attention == "self_attn":
                projections += ("out_proj",)
            for projection in projections:
                shapes[f"{prefix}{attention}.{projection}.bias"] = (EMBED_DIMENSION,)
                shapes[f"{prefix}{attention}.{projection}.weight"] = (
                    EMBED_DIMENSION,
                    EMBED_DIMENSION,
                )
        shapes[f"{prefix}self_attn.rot_emb.inv_freq"] = (
            EMBED_DIMENSION // HEAD_COUNT // 2,
        )
        shapes[f"{prefix}self_attn_layer_norm.bias"] = (EMBED_DIMENSION,)
        shapes[f"{prefix}self_attn_layer_norm.weight"] = (EMBED_DIMENSION,)
        shapes[f"{prefix}fc1.bias"] = (FFN_DIMENSION,)
        shapes[f"{prefix}fc1.weight"] = (FFN_DIMENSION, EMBED_DIMENSION)
        shapes[f"{prefix}fc2.bias"] = (EMBED_DIMENSION,)
        shapes[f"{prefix}fc2.weight"] = (EMBED_DIMENSION, FFN_DIMENSION)
        shapes[f"{prefix}final_layer_norm.bias"] = (EMBED_DIMENSION,)
        shapes[f"{prefix}final_layer_norm.weight"] = (EMBED_DIMENSION,)
    return shapes


def normalize_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    normalized: dict[str, Tensor] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not name.startswith("model."):
            raise ValueError("Every MINT checkpoint key must have the exact model. prefix")
        normalized_name = name.removeprefix("model.")
        if normalized_name in normalized:
            raise ValueError(f"Duplicate normalized MINT key: {normalized_name}")
        if not isinstance(tensor, Tensor):
            raise ValueError(f"MINT state value is not a tensor: {name}")
        normalized[normalized_name] = tensor
    return normalized


def audit_state(state: Mapping[str, Tensor]) -> dict[str, Any]:
    normalized = normalize_state(state)
    expected = expected_shapes()
    missing = sorted(set(expected) - set(normalized))
    unexpected = sorted(set(normalized) - set(expected))
    shape_mismatches = {
        name: {
            "checkpoint": list(normalized[name].shape),
            "expected": list(expected[name]),
        }
        for name in sorted(set(expected) & set(normalized))
        if tuple(normalized[name].shape) != expected[name]
    }
    tied_weight_verified = (
        "embed_tokens.weight" in normalized
        and "lm_head.weight" in normalized
        and torch.equal(
            normalized["embed_tokens.weight"], normalized["lm_head.weight"]
        )
    )
    multimer_projection_pairs_different = 0
    for layer in range(LAYER_COUNT):
        for projection in ("k_proj", "q_proj", "v_proj"):
            self_name = f"layers.{layer}.self_attn.{projection}.weight"
            multimer_name = f"layers.{layer}.multimer_attn.{projection}.weight"
            if self_name in normalized and multimer_name in normalized and not torch.equal(
                normalized[self_name], normalized[multimer_name]
            ):
                multimer_projection_pairs_different += 1
    compatible = (
        not missing
        and not unexpected
        and not shape_mismatches
        and tied_weight_verified
        and multimer_projection_pairs_different == LAYER_COUNT * 3
    )
    state_parameter_elements = sum(tensor.numel() for tensor in normalized.values())
    tied_elements = normalized.get("lm_head.weight")
    unique_parameter_elements = state_parameter_elements - (
        tied_elements.numel() if tied_weight_verified and tied_elements is not None else 0
    )
    return {
        "architecture_compatible": compatible,
        "expected_state_keys": len(expected),
        "missing_keys": missing,
        "multimer_attention": {
            "projection_key_count": sum(
                ".multimer_attn." in name for name in normalized
            ),
            "self_vs_multimer_qkv_weight_pairs_different": (
                multimer_projection_pairs_different
            ),
            "self_vs_multimer_qkv_weight_pairs_expected": LAYER_COUNT * 3,
        },
        "normalized_state_keys": len(normalized),
        "shape_mismatches": shape_mismatches,
        "state_parameter_elements_including_tied_duplicate": state_parameter_elements,
        "tied_lm_head_duplicate_verified": tied_weight_verified,
        "unexpected_keys": unexpected,
        "unique_parameter_elements": unique_parameter_elements,
    }


def publish_create_only(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite MINT audit: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    checkpoint = arguments.checkpoint.resolve()
    receipt_path = arguments.materialization_receipt.resolve()
    if checkpoint.is_symlink() or checkpoint.stat().st_size != arguments.expected_size:
        raise ValueError("MINT checkpoint file identity differs")
    observed_sha256 = sha256_file(checkpoint)
    if observed_sha256 != arguments.expected_sha256:
        raise ValueError("MINT checkpoint SHA-256 differs")
    materialization = json.loads(receipt_path.read_text(encoding="utf-8"))
    materialized = materialization.get("files", {}).get("mint.ckpt", {})
    if (
        materialization.get("format_version") != "abpairfm_public_materialization_v1"
        or materialization.get("status") != "MATERIALIZED_NOT_NORMALIZED"
        or materialization.get("test_access") is not False
        or materialized.get("bytes") != arguments.expected_size
        or materialized.get("sha256") != observed_sha256
    ):
        raise ValueError("MINT materialization receipt differs")
    if arguments.expected_source_revision != EXPECTED_SOURCE_REVISION:
        raise ValueError("MINT source revision differs from the frozen audit")

    with torch.serialization.safe_globals([argparse.Namespace]):
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    if not isinstance(payload, Mapping) or set(payload) != {
        "hyper_parameters",
        "lr_schedulers",
        "pytorch-lightning_version",
        "state_dict",
    }:
        raise ValueError("MINT checkpoint top-level schema differs")
    state = payload["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("MINT checkpoint lacks a state mapping")
    audit = audit_state(state)
    hyper_parameters = payload["hyper_parameters"]
    if not isinstance(hyper_parameters, Mapping):
        raise ValueError("MINT hyperparameters are missing")
    args = vars(hyper_parameters["args"])
    config = vars(hyper_parameters["cfg"])
    hyperparameters_match = (
        args.get("model") == "650M"
        and args.get("no_multimer") is False
        and args.get("copy_weights") is True
        and config.get("encoder_layers") == LAYER_COUNT
        and config.get("encoder_embed_dim") == EMBED_DIMENSION
        and config.get("encoder_attention_heads") == HEAD_COUNT
        and config.get("token_dropout") is True
        and config.get("max_tokens") == 1024
    )
    compatible = audit["architecture_compatible"] and hyperparameters_match
    receipt = {
        "architecture": {
            "backbone_label": "ESM2_650M",
            "embed_dimension": EMBED_DIMENSION,
            "ffn_dimension": FFN_DIMENSION,
            "head_count": HEAD_COUNT,
            "layer_count": LAYER_COUNT,
            "unique_parameter_elements": audit["unique_parameter_elements"],
            "vocabulary_size": VOCABULARY_SIZE,
        },
        "authorization": {
            "frozen_development_contact_baseline": False,
            "frozen_development_pair_mlm_baseline": compatible,
            "frozen_development_retrieval_baseline": compatible,
            "production_training": False,
            "test_access": False,
        },
        "checkpoint": {
            "bytes": checkpoint.stat().st_size,
            "name": checkpoint.name,
            "sha256": observed_sha256,
        },
        "claim_boundary": "Read-only safe-unpickling, schema, shape, tied-weight, and multimer-branch audit of the pinned public MINT checkpoint. Compatibility authorizes only prospective frozen-development pair-MLM and retrieval baselines; contact evaluation requires the same frozen train-only adapter, and production training, test access, biological improvement, and SOTA claims remain closed.",
        "compatibility": {
            **audit,
            "hyperparameters_match": hyperparameters_match,
            "pytorch_lightning_version": payload["pytorch-lightning_version"],
        },
        "format_version": "abpairfm_mint_650m_checkpoint_compatibility_v1",
        "provenance": {
            "materialization_receipt_sha256": sha256_file(receipt_path),
            "official_source_revision": arguments.expected_source_revision,
            "official_source_url": "https://github.com/VarunUllanat/mint",
        },
        "status": (
            "PAIR_MLM_RETRIEVAL_COMPATIBLE_CONTACT_ADAPTER_PENDING"
            if compatible
            else "CHECKPOINT_INCOMPATIBLE_FAIL_CLOSED"
        ),
        "test_access": False,
    }
    publish_create_only(arguments.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
