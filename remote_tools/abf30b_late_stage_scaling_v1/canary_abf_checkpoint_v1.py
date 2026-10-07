#!/usr/bin/env python3
"""Restore one frozen ABF-650M DCP and verify residue-hidden-state extraction."""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from antibody_pretraining.model import (
    AntibodyMaskedLanguageModel,
    config_from_preset,
    load_model_family,
)
from antibody_pretraining.tokenizer import encode_unpaired, pad_encoded


MODEL_FAMILY_SHA256 = "1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(value: object) -> str:
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            header = json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            digest.update(b"tensor\0" + header + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda current: str(current)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                visit(child)
        elif isinstance(item, type):
            identity = f"{item.__module__}.{item.__qualname__}".encode("ascii")
            digest.update(b"type\0" + identity + b"\0")
        elif isinstance(item, enum.Enum):
            identity = (
                f"{item.__class__.__module__}.{item.__class__.__qualname__}"
            ).encode("ascii")
            digest.update(b"enum\0" + identity + b"\0")
            visit(item.name)
            visit(item.value)
        elif isinstance(item, (torch.dtype, torch.device)):
            digest.update(b"torch_identity\0" + str(item).encode("ascii") + b"\0")
        else:
            digest.update(
                json.dumps(item, sort_keys=True, separators=(",", ":")).encode("ascii")
                + b"\0"
            )

    visit(value)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def restore_model(model: AntibodyMaskedLanguageModel, checkpoint: Path) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
    state = get_model_state_dict(model, options=options)
    dcp.load({"model": state}, checkpoint_id=checkpoint.resolve(), no_dist=True)
    incompatible = set_model_state_dict(model, state, options=options)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"DCP restore returned incompatible keys: {incompatible}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--chain-map", type=Path, required=True)
    parser.add_argument("--chain-map-sha256", required=True)
    parser.add_argument("--model-family", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--checkpoint-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    for path, expected in (
        (args.protocol, args.protocol_sha256),
        (args.chain_map, args.chain_map_sha256),
        (args.model_family, MODEL_FAMILY_SHA256),
    ):
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"authority SHA differs: {path}")
    if args.checkpoint.is_symlink() or not (args.checkpoint / ".metadata").is_file():
        raise ValueError("DCP checkpoint is missing or symlinked")
    protocol = json.loads(args.protocol.read_text())
    chain_map = json.loads(args.chain_map.read_text())
    checkpoint = protocol.get("checkpoints", {}).get(args.checkpoint_label, {})
    if (
        protocol.get("status") != "FROZEN_BEFORE_LATE_STAGE_SCALING_EXECUTION"
        or chain_map.get("status") != "PASS_LABEL_FREE_ANARCII_CHAIN_TYPING"
        or chain_map.get("protocol_sha256")
        != protocol.get("base_transfer_protocol_sha256")
        or checkpoint.get("checkpoint_model_sha256")
        != args.checkpoint_model_sha256
        or Path(checkpoint.get("checkpoint_path", "")).resolve()
        != args.checkpoint.resolve()
        or chain_map.get("task_labels_read") is not False
    ):
        raise ValueError("scaling protocol, checkpoint, or chain-map authority differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("deterministic CUBLAS workspace is required")

    family = load_model_family(args.model_family)
    model = AntibodyMaskedLanguageModel(config_from_preset(family, "abf_650m"))
    restore_model(model, args.checkpoint)
    observed_model_sha256 = tree_digest(model.state_dict())
    if observed_model_sha256 != args.checkpoint_model_sha256:
        raise RuntimeError(
            "restored model digest differs: "
            f"{observed_model_sha256} != {args.checkpoint_model_sha256}"
        )

    examples: dict[str, dict[str, Any]] = {}
    for row in chain_map["unique_chains"]:
        examples.setdefault(row["abf_chain_type"], row)
    if "heavy" not in examples or not ({"kappa", "lambda"} & set(examples)):
        raise ValueError("chain map lacks required canary chain types")
    selected_types = [kind for kind in ("heavy", "kappa", "lambda") if kind in examples]
    encoded = [
        encode_unpaired(examples[kind]["sequence"], kind) for kind in selected_types
    ]
    input_ids, type_ids, attention_mask = pad_encoded(encoded)
    device = torch.device("cuda")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], value: torch.Tensor) -> None:
        captured.append(value.detach())

    handle = model.final_norm.register_forward_hook(hook)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    type_tensor = torch.tensor(type_ids, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.bool, device=device)
    torch.manual_seed(20_260_830)
    torch.cuda.manual_seed_all(20_260_830)
    torch.use_deterministic_algorithms(True, warn_only=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        first = model(input_tensor, type_tensor, attention_mask=mask_tensor)
        first_hidden = captured.pop().float().cpu()
        second = model(input_tensor, type_tensor, attention_mask=mask_tensor)
        second_hidden = captured.pop().float().cpu()
    handle.remove()
    if captured:
        raise AssertionError("final-norm hook capture count differs")
    if (
        not torch.equal(first_hidden, second_hidden)
        or not torch.equal(first.logits.float().cpu(), second.logits.float().cpu())
        or not bool(torch.isfinite(first_hidden).all())
        or tuple(first_hidden.shape) != (
            len(selected_types),
            len(input_ids[0]),
            1280,
        )
    ):
        raise RuntimeError("embedding canary is non-finite, non-repeatable, or shape-invalid")
    residue_counts = [len(examples[kind]["sequence"]) for kind in selected_types]
    for index, length in enumerate(residue_counts):
        if not bool(torch.isfinite(first_hidden[index, 1 : 1 + length]).all()):
            raise RuntimeError("residue hidden state is non-finite")

    result = {
        "format_version": "abf_checkpoint_scaling_dcp_embedding_canary_v1",
        "status": "PASS_DCP_EXACT_RESTORE_AND_RESIDUE_EMBEDDING_CANARY",
        "protocol_sha256": args.protocol_sha256,
        "chain_map_sha256": args.chain_map_sha256,
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_model_sha256": observed_model_sha256,
        "embedding_definition": "final RMS-normalized residue hidden state excluding CLS and SEP",
        "embedding_dimension": 1280,
        "chain_types_tested": selected_types,
        "residue_counts": residue_counts,
        "repeat_bit_equal": True,
        "nonfinite_count": 0,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "autocast_dtype": "bfloat16",
        },
        "task_labels_read": False,
        "test_loaded": False,
    }
    result_sha = publish(args.output, result)
    receipt = {
        "format_version": "abf_checkpoint_scaling_dcp_embedding_canary_receipt_v1",
        "status": result["status"],
        "result_sha256": result_sha,
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_model_sha256": observed_model_sha256,
        "task_labels_read": False,
        "test_loaded": False,
    }
    receipt_sha = publish(args.receipt_output, receipt)
    print(f"result_sha256={result_sha}")
    print(f"receipt_sha256={receipt_sha}")


if __name__ == "__main__":
    main()
