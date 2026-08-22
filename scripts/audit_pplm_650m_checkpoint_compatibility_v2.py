#!/usr/bin/env python3
"""Audit whether the frozen public PPLM 650M state matches the clean-room model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import torch
from torch import Tensor

from antibody_pretraining.pair_aware_esm2 import PairAwareESM2, PairAwareESM2Config


PREFIX = re.compile(r"^(encoder\.sentence_encoder\.|encoder\.)")
AUXILIARY_CONTACT_HEAD_KEYS = {
    "contact_head.regression.bias",
    "contact_head.regression.weight",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_pplm_state(
    state: Mapping[str, Tensor], config: PairAwareESM2Config
) -> tuple[dict[str, Tensor], dict[str, object]]:
    normalized: dict[str, Tensor] = {}
    renamed_inter_attention: list[str] = []
    for raw_name, tensor in state.items():
        name = PREFIX.sub("", raw_name)
        if name.endswith(".self_attn.inter_attn_weight"):
            name = name.removesuffix("inter_attn_weight") + "cross_scale"
            renamed_inter_attention.append(name)
        if name in normalized:
            raise ValueError(f"Duplicate normalized checkpoint key: {name}")
        normalized[name] = tensor
    duplicate = normalized.pop("lm_head.weight", None)
    embedding = normalized.get("embed_tokens.weight")
    if duplicate is not None and (
        embedding is None
        or duplicate.shape != embedding.shape
        or not torch.equal(duplicate, embedding)
    ):
        raise ValueError("PPLM tied LM-head and token embedding differ")

    with torch.device("meta"):
        expected_state = PairAwareESM2(config).state_dict()
    auxiliary_contact_head = {
        name: normalized.pop(name)
        for name in sorted(AUXILIARY_CONTACT_HEAD_KEYS & set(normalized))
    }
    expected = set(expected_state)
    found = set(normalized)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    shape_mismatches = {
        name: {
            "checkpoint": list(normalized[name].shape),
            "expected": list(expected_state[name].shape),
        }
        for name in sorted(expected & found)
        if normalized[name].shape != expected_state[name].shape
    }
    expected_scales = {
        f"layers.{index}.self_attn.cross_scale"
        for index in range(config.layer_count)
    }
    observed_scales = set(renamed_inter_attention)
    encoder_mlm_compatible = (
        not missing
        and not unexpected
        and not shape_mismatches
        and observed_scales == expected_scales
    )
    scale_values = torch.cat(
        [normalized[name].detach().float().flatten() for name in sorted(observed_scales)]
    )
    audit = {
        "auxiliary_contact_head": {
            "exact_expected_key_set_present": set(auxiliary_contact_head)
            == AUXILIARY_CONTACT_HEAD_KEYS,
            "keys": {
                name: list(tensor.shape)
                for name, tensor in auxiliary_contact_head.items()
            },
            "not_implemented_in_clean_room_model": True,
        },
        "encoder_mlm_compatible": encoder_mlm_compatible,
        "end_to_end_contact_compatible": False,
        "expected_state_keys": len(expected),
        "missing_keys": missing,
        "normalized_state_keys": len(normalized),
        "pplm_inter_attention": {
            "maximum": float(scale_values.max()),
            "minimum": float(scale_values.min()),
            "renamed_layer_count": len(observed_scales),
            "tensor_elements": int(scale_values.numel()),
        },
        "shape_mismatches": shape_mismatches,
        "tied_lm_head_duplicate_verified": duplicate is not None,
        "unexpected_keys": unexpected,
    }
    return normalized, audit


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
        raise FileExistsError(f"Refusing to overwrite PPLM audit: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    checkpoint = arguments.checkpoint.resolve()
    if checkpoint.is_symlink() or checkpoint.stat().st_size != arguments.expected_size:
        raise ValueError("PPLM checkpoint file identity differs")
    observed_sha256 = sha256_file(checkpoint)
    if observed_sha256 != arguments.expected_sha256:
        raise ValueError("PPLM checkpoint SHA-256 differs")
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ValueError("PPLM checkpoint lacks a model state mapping")
    config = PairAwareESM2Config(
        layer_count=33,
        embed_dimension=1280,
        head_count=20,
        ffn_dimension=5120,
    )
    normalized, audit = normalize_pplm_state(payload["model"], config)
    receipt = {
        "architecture": {
            "embed_dimension": config.embed_dimension,
            "ffn_dimension": config.ffn_dimension,
            "head_count": config.head_count,
            "layer_count": config.layer_count,
            "parameter_elements_in_normalized_state": sum(
                tensor.numel() for tensor in normalized.values()
            ),
        },
        "authorization": {
            "frozen_development_contact_baseline": False,
            "frozen_development_mlm_baseline": audit["encoder_mlm_compatible"],
            "production_training": False,
            "test_access": False,
        },
        "checkpoint": {
            "bytes": checkpoint.stat().st_size,
            "name": checkpoint.name,
            "sha256": observed_sha256,
            "source_revision": arguments.source_revision,
        },
        "claim_boundary": "Read-only schema, shape, and exact tied-weight compatibility audit of the frozen public PPLM 650M checkpoint. It authorizes only a prospective frozen-development baseline when compatible; no training, test access, biological, or SOTA claim is established.",
        "compatibility": audit,
        "format_version": "abpairfm_pplm_650m_checkpoint_compatibility_v2",
        "status": (
            "ENCODER_MLM_COMPATIBLE_CONTACT_HEAD_PENDING"
            if audit["encoder_mlm_compatible"]
            else "ENCODER_MLM_INCOMPATIBLE_FAIL_CLOSED"
        ),
        "test_access": False,
    }
    publish_create_only(arguments.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
