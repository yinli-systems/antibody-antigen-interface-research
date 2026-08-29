#!/usr/bin/env python3
"""Verify a frozen 650M DCP for paired-sequence continuation without training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed.checkpoint as dcp


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abf-source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--model-family", type=Path, required=True)
    parser.add_argument("--paired-module", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Canary requires exactly one allocated CUDA device")
    device_name = torch.cuda.get_device_name(0)
    if "4090" not in device_name:
        raise RuntimeError(f"Canary requires RTX 4090, observed {device_name}")

    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes.decode("ascii"))
    if (
        protocol.get("format_version") != "oas_paired_650m_dcp_canary_v1"
        or protocol.get("training_authorized") is not False
        or protocol.get("test_access")
        != {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
            "temporal_holdout_accessed": False,
        }
    ):
        raise RuntimeError("Canary protocol authority differs")
    if _sha256(args.model_family) != protocol["model_family_sha256"]:
        raise RuntimeError("Model-family hash differs")
    if _sha256(args.checkpoint_receipt) != protocol["checkpoint_receipt_sha256"]:
        raise RuntimeError("Checkpoint receipt file hash differs")
    if _sha256(args.paired_module) != protocol["paired_module_sha256"]:
        raise RuntimeError("Paired module hash differs")

    receipt = json.loads(args.checkpoint_receipt.read_text(encoding="ascii"))
    expected_model_sha256 = protocol["checkpoint_model_sha256"]
    if (
        receipt.get("checkpoint_model_sha256") != expected_model_sha256
        or receipt.get("preset") != "abf_650m"
        or receipt.get("model_family_sha256") != protocol["model_family_sha256"]
    ):
        raise RuntimeError("Checkpoint receipt provenance differs")

    sys.path.insert(0, str(args.abf_source_root / "src"))
    sys.path.insert(0, str(args.abf_source_root / "scripts"))
    sys.path.insert(0, str(args.paired_module.parent))
    from antibody_pretraining.model import (  # pylint: disable=import-outside-toplevel
        AntibodyMaskedLanguageModel,
        config_from_preset,
        load_model_family,
    )
    from paired_pretraining import encode_pair  # pylint: disable=import-outside-toplevel
    from run_antibody_pretraining import _tree_digest  # pylint: disable=import-outside-toplevel

    family = load_model_family(args.model_family)
    original_config = config_from_preset(family, "abf_650m")
    paired_config = replace(original_config, maximum_sequence_tokens=320)
    original_model = AntibodyMaskedLanguageModel(original_config)
    paired_model = AntibodyMaskedLanguageModel(paired_config)
    original_keys = tuple(original_model.state_dict())
    paired_keys = tuple(paired_model.state_dict())
    if original_keys != paired_keys:
        raise RuntimeError("RoPE extension changed persistent model state keys")
    if any("rotary.cosine" in key or "rotary.sine" in key for key in paired_keys):
        raise RuntimeError("RoPE tables unexpectedly became persistent parameters")
    del original_model

    model_state = paired_model.state_dict()
    state = {"model": model_state}
    dcp.load(state, checkpoint_id=str(args.checkpoint_root))
    paired_model.load_state_dict(state["model"], strict=True)
    observed_model_sha256 = _tree_digest(paired_model.state_dict())
    if observed_model_sha256 != expected_model_sha256:
        raise RuntimeError("Loaded model payload hash differs from production receipt")

    encoded = encode_pair(
        "EVQLVESGGGLVQPGGSLRLSCAASGYDFTHYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPYYYGTSHWYFDVWGQGTLVTVSS",
        "DIQLTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIK",
        ("GYDFTHY", "INTYTGEPT", "AKYPYYYGTSHWYFDV"),
        ("SASQDISNY", "FTS", "QQYSTVPWT"),
        maximum_tokens=320,
    )
    device = torch.device("cuda", 0)
    paired_model.to(device).eval()
    input_ids = torch.as_tensor(encoded.input_ids[None, :], dtype=torch.long, device=device)
    type_ids = torch.as_tensor(encoded.type_ids[None, :], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = paired_model(input_ids, type_ids, attention_mask=attention_mask)
    if output.logits.shape != (1, len(encoded.input_ids), len(family["tokenizer"]["amino_acids"]) + 5):
        raise RuntimeError("Paired forward output shape differs")
    if not bool(torch.isfinite(output.logits).all()):
        raise RuntimeError("Paired forward produced nonfinite logits")

    result = {
        "checkpoint_model_sha256": observed_model_sha256,
        "checkpoint_receipt_sha256": _sha256(args.checkpoint_receipt),
        "device_name": device_name,
        "format_version": "oas_paired_650m_dcp_canary_result_v1",
        "input_checkpoint_root": str(args.checkpoint_root),
        "model_family_sha256": _sha256(args.model_family),
        "original_maximum_sequence_tokens": original_config.maximum_sequence_tokens,
        "paired_forward_finite": True,
        "paired_forward_shape": list(output.logits.shape),
        "paired_input_tokens": len(encoded.input_ids),
        "paired_maximum_sequence_tokens": paired_config.maximum_sequence_tokens,
        "paired_module_sha256": _sha256(args.paired_module),
        "parameter_state_keys_unchanged": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "status": "PASS_MODEL_ONLY_DCP_AND_PAIRED_FORWARD_VERIFIED",
        "test_access": protocol["test_access"],
        "training_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_bytes(_canonical_bytes(result))
    os.replace(temporary, args.output)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
