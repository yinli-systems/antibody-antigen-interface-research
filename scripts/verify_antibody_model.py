#!/usr/bin/env python3
"""Verify tokenizer, masking, model numerics, and exact preset parameter counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import torch

import antibody_pretraining.masking as masking_module
import antibody_pretraining.model as model_module
import antibody_pretraining.tokenizer as tokenizer_module
from antibody_pretraining.masking import apply_uniform_mlm
from antibody_pretraining.model import (
    AntibodyMaskedLanguageModel,
    AntibodyModelConfig,
    config_from_preset,
    load_model_family,
    parameter_receipt,
)
from antibody_pretraining.tokenizer import encode_unpaired, pad_encoded


def _canonical_json_bytes(payload: object) -> bytes:
    """Encode finite sorted ASCII JSON with one terminal line feed."""
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    """Hash a regular non-symlink file."""
    if not path.is_file() or path.is_symlink():
        raise ValueError("Verification input must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_create_only(path: Path, payload: bytes) -> None:
    """Create one immutable verification receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink(missing_ok=True)
        finally:
            raise


def _parser() -> argparse.ArgumentParser:
    """Build the strict verification command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--family-file-sha256", required=True)
    parser.add_argument("--masking-source-sha256", required=True)
    parser.add_argument("--model-source-sha256", required=True)
    parser.add_argument("--tokenizer-source-sha256", required=True)
    parser.add_argument("--verifier-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _verify_source(path: Path, expected: str, label: str) -> None:
    """Require a source module to match its external digest."""
    observed = _file_sha256(path)
    if observed != expected:
        raise ValueError(f"{label} source digest differs")


def _small_model_evidence() -> dict[str, object]:
    """Run deterministic forward, selected-target loss, and backward checks."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = AntibodyModelConfig(
        d_model=64,
        ffn_hidden_size=128,
        maximum_sequence_tokens=96,
        n_heads=4,
        n_layers=2,
    )
    sequences = (
        "ACDEFGHIKLMNPQRSTVWY" * 4,
        "YWVTSRQPNMLKIHGFEDCA" * 4,
    )
    encoded = [encode_unpaired(sequence, "heavy") for sequence in sequences]
    token_lists, type_lists, masks = pad_encoded(encoded, pad_to_length=96)
    original_tokens = np.asarray(token_lists, dtype=np.int64)
    sample_keys = np.asarray([17, 73], dtype=np.uint64)
    masked = apply_uniform_mlm(
        original_tokens,
        sample_keys,
        update=11,
        seed=42,
    )
    replay = apply_uniform_mlm(
        original_tokens,
        sample_keys,
        update=11,
        seed=42,
    )
    if not np.array_equal(masked.input_ids, replay.input_ids):
        raise AssertionError("MLM corruption did not replay")
    if masked.selected_target_count <= 0:
        raise AssertionError("MLM verification selected no targets")

    input_ids = torch.from_numpy(masked.input_ids)
    type_ids = torch.tensor(type_lists, dtype=torch.long)
    attention_mask = torch.tensor(masks, dtype=torch.bool)
    labels = torch.from_numpy(masked.labels)
    torch.manual_seed(1234)
    model = AntibodyMaskedLanguageModel(config)
    output = model(input_ids, type_ids, attention_mask, labels)
    if output.loss_sum is None or not torch.isfinite(output.loss_sum):
        raise AssertionError("Small-model selected-target loss is not finite")
    output.loss_sum.backward()
    gradient = model.token_embedding.weight.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise AssertionError("Tied token embedding has no finite gradient")

    torch.manual_seed(1234)
    replay_model = AntibodyMaskedLanguageModel(config)
    replay_output = replay_model(input_ids, type_ids, attention_mask, labels)
    if not torch.equal(output.logits.detach(), replay_output.logits.detach()):
        raise AssertionError("CPU model forward did not replay bitwise")
    return {
        "gradient_l2_norm": float(gradient.float().norm().item()),
        "logits_sha256": hashlib.sha256(
            output.logits.detach().float().contiguous().numpy().tobytes()
        ).hexdigest(),
        "loss_sum": float(output.loss_sum.detach().item()),
        "selected_target_count": masked.selected_target_count,
    }


def main() -> int:
    """Run all checks and emit a source-bound receipt."""
    args = _parser().parse_args()
    _verify_source(Path(__file__), args.verifier_source_sha256, "Verifier")
    _verify_source(Path(masking_module.__file__), args.masking_source_sha256, "Masking")
    _verify_source(Path(model_module.__file__), args.model_source_sha256, "Model")
    _verify_source(
        Path(tokenizer_module.__file__),
        args.tokenizer_source_sha256,
        "Tokenizer",
    )
    if _file_sha256(args.family) != args.family_file_sha256:
        raise ValueError("Model-family file digest differs")
    family = load_model_family(args.family)
    parameter_receipts = {}
    with torch.device("meta"):
        for preset in sorted(family["presets"]):
            model = AntibodyMaskedLanguageModel(config_from_preset(family, preset))
            parameter_receipts[preset] = parameter_receipt(
                model,
                preset=preset,
                family_file_sha256=args.family_file_sha256,
            )
    receipt = {
        "checks": {
            "cpu_forward_replay_bitwise": True,
            "forward_loss_backward": True,
            "mask_replay_bitwise": True,
            "meta_parameter_count": True,
            "tied_embedding_gradient": True,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "family_file_sha256": args.family_file_sha256,
        "format_version": "antibody_model_verification_receipt_v1",
        "masking_source_sha256": args.masking_source_sha256,
        "model_source_sha256": args.model_source_sha256,
        "parameter_receipts": parameter_receipts,
        "small_model_evidence": _small_model_evidence(),
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "tokenizer_source_sha256": args.tokenizer_source_sha256,
        "verifier_source_sha256": args.verifier_source_sha256,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    _write_create_only(args.output, _canonical_json_bytes(receipt))
    print(_file_sha256(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
