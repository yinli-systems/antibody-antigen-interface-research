#!/usr/bin/env python3
"""Verify single-chain numerical parity of an authenticated ESM2 conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from antibody_pretraining.pair_aware_esm2 import (  # noqa: E402
    PairAwareESM2,
    PairAwareESM2Config,
    load_compatible_esm2_state,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--expected-hf-size", type=int, required=True)
    parser.add_argument("--expected-hf-sha256", required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--expected-converted-size", type=int, required=True)
    parser.add_argument("--expected-converted-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--maximum-logit-absolute-error", type=float, default=1e-2)
    parser.add_argument(
        "--maximum-mean-logit-absolute-error", type=float, default=1e-3
    )
    parser.add_argument("--minimum-cosine-similarity", type=float, default=0.999999)
    return parser.parse_args()


def passes_parity(
    maximum: float,
    mean: float,
    cosine: float,
    *,
    maximum_threshold: float,
    mean_threshold: float,
    cosine_threshold: float,
) -> bool:
    return (
        maximum <= maximum_threshold
        and mean <= mean_threshold
        and cosine >= cosine_threshold
    )


def main() -> None:
    arguments = _arguments()
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("Numerical parity must run under Slurm gpu_4090")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError(f"Forbidden GPU model: {torch.cuda.get_device_name(0)}")
    hf_checkpoint = arguments.hf_root / "model.safetensors"
    for path, size, digest in (
        (hf_checkpoint, arguments.expected_hf_size, arguments.expected_hf_sha256),
        (
            arguments.converted,
            arguments.expected_converted_size,
            arguments.expected_converted_sha256,
        ),
    ):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("Parity input must be a regular file")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise RuntimeError("Parity input authentication failed")
    if arguments.receipt.exists():
        raise FileExistsError("Parity receipt is create-only")

    from transformers import EsmForMaskedLM

    device = torch.device("cuda", 0)
    official = EsmForMaskedLM.from_pretrained(
        arguments.hf_root, local_files_only=True
    ).eval().to(device)
    converted_payload = torch.load(
        arguments.converted, map_location="cpu", weights_only=False, mmap=True
    )
    converted = PairAwareESM2(
        PairAwareESM2Config(
            layer_count=33,
            embed_dimension=1280,
            head_count=20,
            ffn_dimension=5120,
            attention_implementation="dense",
            activation_checkpointing=False,
        )
    ).eval()
    load_receipt = load_compatible_esm2_state(converted, converted_payload["model"])
    converted.to(device)

    tokens = torch.tensor(
        [
            [0, 4, 5, 6, 32, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 1, 1],
            [0, 23, 22, 21, 20, 19, 18, 32, 16, 15, 14, 13, 12, 11, 10, 2, 1, 1],
        ],
        dtype=torch.long,
        device=device,
    )
    attention_mask = tokens.ne(1)
    chain_ids = torch.where(
        attention_mask,
        torch.zeros_like(tokens),
        torch.full_like(tokens, -1),
    )
    with torch.inference_mode():
        official_logits = official(
            input_ids=tokens, attention_mask=attention_mask
        ).logits
        converted_logits = converted(tokens, chain_ids).logits
    selected = attention_mask.unsqueeze(-1).expand_as(official_logits)
    difference = (official_logits - converted_logits).abs()[selected]
    maximum = float(difference.max().item())
    mean = float(difference.mean().item())
    official_selected = official_logits[selected].float()
    converted_selected = converted_logits[selected].float()
    cosine = float(
        torch.nn.functional.cosine_similarity(
            official_selected.unsqueeze(0), converted_selected.unsqueeze(0)
        ).item()
    )
    status = (
        "PASS_SINGLE_CHAIN_NUMERICAL_PARITY"
        if passes_parity(
            maximum,
            mean,
            cosine,
            maximum_threshold=arguments.maximum_logit_absolute_error,
            mean_threshold=arguments.maximum_mean_logit_absolute_error,
            cosine_threshold=arguments.minimum_cosine_similarity,
        )
        else "FAIL_SINGLE_CHAIN_NUMERICAL_PARITY"
    )
    receipt = {
        "claim_boundary": "Authenticated single-chain logit parity only; pair-aware behavior and downstream ability require separate gates.",
        "converted_checkpoint": {
            "sha256": arguments.expected_converted_sha256,
            "size_bytes": arguments.expected_converted_size,
        },
        "format_version": "abpairfm_hf_esm2_conversion_parity_v2",
        "hf_checkpoint": {
            "sha256": arguments.expected_hf_sha256,
            "size_bytes": arguments.expected_hf_size,
        },
        "load_receipt": load_receipt,
        "measurement": {
            "cosine_similarity": cosine,
            "maximum_logit_absolute_error": maximum,
            "mean_logit_absolute_error": mean,
            "nonpadding_logit_count": int(selected.sum().item()),
            "thresholds": {
                "maximum_logit_absolute_error": arguments.maximum_logit_absolute_error,
                "maximum_mean_logit_absolute_error": arguments.maximum_mean_logit_absolute_error,
                "minimum_cosine_similarity": arguments.minimum_cosine_similarity,
            },
        },
        "status": status,
        "test_access": False,
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    with arguments.receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    if status != "PASS_SINGLE_CHAIN_NUMERICAL_PARITY":
        raise RuntimeError("Converted ESM2 checkpoint failed numerical parity")


if __name__ == "__main__":
    main()
