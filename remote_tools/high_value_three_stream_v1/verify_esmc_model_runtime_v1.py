#!/usr/bin/env python3
"""Strict-load and inference canary for frozen base or AbCDR ESM-C weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from minimal_esmc_runtime_v1 import (
    MinimalESMC600M,
    extract_paired_residue_embeddings,
    tokenize_sequence,
)


HEAVY = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVA"
LIGHT = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIY"
ANTIGEN = "TNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNLAPFFTFKCYG"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def normalize_body_key(key: str, role: str) -> str | None:
    """Map both frozen checkpoint formats onto the esm==3.1.4 ESM-C body."""
    if role == "published_cdr":
        key = key.removeprefix("esmC_model.")
        if key.startswith("sequence_head."):
            return None
        return key
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
            key = key.replace(source, target)
            break
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--role", choices=("base", "published_cdr"), required=True)
    parser.add_argument("--safetensors-site", type=Path, required=True)
    parser.add_argument("--runtime-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if sha256_file(args.checkpoint) != args.expected_sha256:
        raise RuntimeError("checkpoint SHA differs")
    sys.path.append(str(args.safetensors_site))
    from safetensors import safe_open

    state = {}
    with safe_open(args.checkpoint, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            normalized = normalize_body_key(key, args.role)
            if normalized is None:
                continue
            if normalized in state:
                raise RuntimeError("duplicate normalized state key")
            state[normalized] = handle.get_tensor(key)
    with torch.device("meta"):
        model = MinimalESMC600M()
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict state-dict contract differs")
    model.materialize_nonpersistent_buffers(torch.device("cpu"))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 573_632_640:
        raise RuntimeError(f"parameter count differs: {parameter_count}")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    paired_tokens = tokenize_sequence(f"{HEAVY}-{LIGHT}").unsqueeze(0).to(device)
    antigen_tokens = tokenize_sequence(ANTIGEN).unsqueeze(0).to(device)
    with torch.inference_mode():
        paired = model(paired_tokens).embeddings
        antigen = model(antigen_tokens).embeddings
    residues = extract_paired_residue_embeddings(paired, len(HEAVY), len(LIGHT))
    antigen_residues = antigen[0, 1:-1]
    if (
        paired_tokens[0, 1 + len(HEAVY)].item() != 30
        or antigen_residues.shape != (len(ANTIGEN), 1152)
        or not torch.isfinite(residues).all()
        or not torch.isfinite(antigen_residues).all()
    ):
        raise RuntimeError("token/residue inference contract differs")
    result = {
        "format_version": "esmc_600m_runtime_canary_v1",
        "status": "PASS_STRICT_ESMC_600M_RUNTIME_CANARY",
        "role": args.role,
        "checkpoint_sha256": args.expected_sha256,
        "runtime_source_sha256": args.runtime_source_sha256,
        "state_tensor_count": len(state),
        "strict_body_only_load": True,
        "nonpersistent_rotary_buffers_materialized": True,
        "parameter_count": parameter_count,
        "embedding_dimension": 1152,
        "separator_token_id": 30,
        "paired_input_tokens": paired_tokens.shape[1],
        "exported_antibody_residues": residues.shape[0],
        "exported_antigen_residues": antigen_residues.shape[0],
        "antibody_embedding_sha256_fp32": tensor_fingerprint(residues),
        "antigen_embedding_sha256_fp32": tensor_fingerprint(antigen_residues),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "audit": {
            "strict_state_dict_load": True,
            "finite": True,
            "bos_eos_removed": True,
            "separator_removed_after_forward": True,
            "vh_then_vl_order_preserved": True,
            "test_access": False,
            "training_started": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(result))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({**result, "result_sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
