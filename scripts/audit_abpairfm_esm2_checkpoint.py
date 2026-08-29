#!/usr/bin/env python3
"""Create a fail-closed ESM2-to-ABPairFM checkpoint compatibility receipt."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from antibody_pretraining.pair_aware_esm2 import (
    PairAwareESM2,
    PairAwareESM2Config,
    authenticated_torch_checkpoint,
    canonicalize_tied_esm2_state,
    parameter_count,
    upgrade_esm2_state_dict,
)


def _model_cfg_value(payload: Mapping[str, Any], name: str) -> Any:
    cfg = payload.get("cfg")
    if cfg is None:
        raise RuntimeError("Official checkpoint is missing cfg")
    model_cfg = cfg["model"]
    if hasattr(model_cfg, name):
        return getattr(model_cfg, name)
    return model_cfg[name]


def audit_state_compatibility(
    model: PairAwareESM2,
    raw_state: Mapping[str, Tensor],
    *,
    checkpoint_dimensions: Mapping[str, Any],
) -> dict[str, Any]:
    expected_dimensions = {
        "encoder_attention_heads": model.config.head_count,
        "encoder_embed_dim": model.config.embed_dimension,
        "encoder_layers": model.config.layer_count,
        "token_dropout": model.config.token_dropout,
    }
    if dict(checkpoint_dimensions) != expected_dimensions:
        raise RuntimeError(
            f"Checkpoint architecture mismatch: {dict(checkpoint_dimensions)} != {expected_dimensions}"
        )
    upgraded = upgrade_esm2_state_dict(raw_state)
    if "lm_head.weight" not in upgraded:
        raise RuntimeError("Checkpoint is missing its tied LM-head duplicate")
    if not torch.equal(upgraded["embed_tokens.weight"], upgraded["lm_head.weight"]):
        raise RuntimeError("Checkpoint tied token and LM-head values differ")
    canonical = canonicalize_tied_esm2_state(
        upgraded,
        require_lm_head_duplicate=True,
    )
    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    found_keys = set(canonical)
    allowed_initialized = {
        f"layers.{index}.self_attn.cross_scale"
        for index in range(model.config.layer_count)
    }
    missing = expected_keys - found_keys
    unexpected = found_keys - expected_keys
    if missing != allowed_initialized:
        raise RuntimeError(f"Unexpected missing keys: {sorted(missing)}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {sorted(unexpected)}")
    shape_mismatches = {
        name: {
            "checkpoint": list(canonical[name].shape),
            "model": list(expected_state[name].shape),
        }
        for name in found_keys
        if tuple(canonical[name].shape) != tuple(expected_state[name].shape)
    }
    if shape_mismatches:
        raise RuntimeError(f"Checkpoint tensor shape mismatches: {shape_mismatches}")
    expected_rotary = 1.0 / (
        10_000.0
        ** (
            torch.arange(
                0,
                model.config.embed_dimension // model.config.head_count,
                2,
                dtype=torch.float32,
            )
            / (model.config.embed_dimension // model.config.head_count)
        )
    )
    for index in range(model.config.layer_count):
        key = f"layers.{index}.self_attn.rot_emb.inv_freq"
        if not torch.equal(canonical[key], expected_rotary.to(canonical[key].dtype)):
            raise RuntimeError(f"Rotary frequency buffer mismatch in layer {index}")
    initialized_parameter_count = sum(
        expected_state[name].numel() for name in allowed_initialized
    )
    return {
        "allowed_initialized_key_count": len(allowed_initialized),
        "allowed_initialized_keys": sorted(allowed_initialized),
        "architecture": expected_dimensions,
        "checkpoint_dtype_counts": dict(
            sorted(Counter(str(tensor.dtype) for tensor in upgraded.values()).items())
        ),
        "checkpoint_state_key_count": len(upgraded),
        "checkpoint_tensor_element_count_with_tied_state_duplicate": sum(
            tensor.numel() for tensor in upgraded.values()
        ),
        "compatible_model_parameter_count": parameter_count(model),
        "initialized_pair_aware_parameter_count": initialized_parameter_count,
        "shape_mismatch_count": 0,
        "unexpected_key_count": 0,
    }


def audit_checkpoint(
    protocol: Mapping[str, Any], checkpoint: Path
) -> dict[str, Any]:
    model_contract = protocol["model"]
    payload = authenticated_torch_checkpoint(
        checkpoint,
        expected_size=model_contract["checkpoint_size_bytes"],
        expected_sha256=model_contract["checkpoint_sha256"],
    )
    with torch.device("meta"):
        model = PairAwareESM2(PairAwareESM2Config())
    dimensions = {
        name: _model_cfg_value(payload, name)
        for name in (
            "encoder_attention_heads",
            "encoder_embed_dim",
            "encoder_layers",
            "token_dropout",
        )
    }
    compatibility = audit_state_compatibility(
        model, payload["model"], checkpoint_dimensions=dimensions
    )
    return {
        "authorization": {
            "checkpoint_initialization": True,
            "development_evaluation": False,
            "gpu_smoke": False,
            "production_training": False,
            "test_access": False,
        },
        "checkpoint": {
            "http_etag": "34533353c10c4fe3aceb2c1ad2867827-677",
            "http_last_modified": "Thu, 18 Aug 2022 13:27:28 GMT",
            "http_version_id": "qgHt_My5BTcMMWHSuORVt6kPsq_lYEyo",
            "sha256": model_contract["checkpoint_sha256"],
            "size_bytes": model_contract["checkpoint_size_bytes"],
            "url": model_contract["checkpoint_url"],
        },
        "claim_boundary": (
            "Authenticated official checkpoint and exact state-shape compatibility only. "
            "No GPU capacity, numerical forward parity, recovery, throughput, training, "
            "biological quality, development result, or SOTA claim is established."
        ),
        "compatibility": compatibility,
        "format_version": "abpairfm_esm2_checkpoint_audit_v1",
        "status": "CHECKPOINT_COMPATIBLE",
        "test_access": {
            "asep_test_accessed": False,
            "pinder_test_rows_used": False,
            "pinder_test_structure_bytes_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    data = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="ascii"))
    if protocol.get("format_version") != "abpairfm_3b_pretraining_v1":
        raise RuntimeError("Unexpected protocol format")
    _write_create_only(args.output, audit_checkpoint(protocol, args.checkpoint))


if __name__ == "__main__":
    main()
