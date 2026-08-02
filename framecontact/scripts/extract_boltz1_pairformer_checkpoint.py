#!/usr/bin/env python3
"""Extract the official Boltz-1 Pairformer weights into a hash-bound artifact."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.nodes import AnyNode


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.boltz_pairformer_adapter import (  # noqa: E402
    BOLTZ1_CHECKPOINT_SHA256,
    BOLTZ_REVISION,
    PAIRFORMER_ARTIFACT_FORMAT,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _revision(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _pairformer_state(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates = (
        "pairformer_module.",
        "model.pairformer_module.",
    )
    for prefix in candidates:
        selected = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if selected:
            return prefix, selected
    raise ValueError("Checkpoint contains no recognized trunk Pairformer prefix")


def _architecture(checkpoint: dict[str, Any]) -> dict[str, Any]:
    hyperparameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("Checkpoint lacks hyper_parameters")
    pairformer = hyperparameters.get("pairformer_args")
    if not isinstance(pairformer, Mapping):
        raise ValueError("Checkpoint lacks pairformer_args")
    architecture = {
        "token_s": int(hyperparameters["token_s"]),
        "token_z": int(hyperparameters["token_z"]),
        "num_blocks": int(pairformer["num_blocks"]),
        "num_heads": int(pairformer.get("num_heads", 16)),
        "dropout": float(pairformer.get("dropout", 0.0)),
        "pairwise_head_width": int(pairformer.get("pairwise_head_width", 32)),
        "pairwise_num_heads": int(pairformer.get("pairwise_num_heads", 4)),
    }
    if architecture["num_blocks"] <= 0:
        raise ValueError("Pairformer block count is invalid")
    return architecture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--boltz-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    for target in (arguments.output, arguments.receipt):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite: {target}")
    checkpoint_sha256 = _sha256(arguments.checkpoint)
    if checkpoint_sha256 != BOLTZ1_CHECKPOINT_SHA256:
        raise ValueError("Boltz-1 checkpoint SHA-256 differs from the official lock")
    if _revision(arguments.boltz_source_root) != BOLTZ_REVISION:
        raise ValueError("Boltz source revision differs from the official lock")
    # The official Lightning checkpoint stores OmegaConf metadata alongside
    # tensors.  Keep ``weights_only=True`` and allowlist only the minimal data
    # container types required by that metadata; never fall back to unrestricted
    # pickle execution.
    safe_globals = [
        DictConfig,
        ListConfig,
        ContainerMetadata,
        Metadata,
        AnyNode,
        Any,
        dict,
        defaultdict,
    ]
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(
            arguments.checkpoint,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint lacks a state_dict")
    prefix, selected = _pairformer_state(state)
    if not selected or not all(torch.is_tensor(value) for value in selected.values()):
        raise ValueError("Extracted Pairformer state is empty or non-tensor")
    artifact = {
        "format_version": PAIRFORMER_ARTIFACT_FORMAT,
        "source_checkpoint_sha256": checkpoint_sha256,
        "boltz_source_revision": BOLTZ_REVISION,
        "state_prefix": prefix,
        "state_tensor_count": len(selected),
        "architecture": _architecture(checkpoint),
        "state_dict": selected,
        "claim_boundary": (
            "Official Pairformer weights only; downstream ESM/local-pose projections "
            "are new adapters and native Boltz preprocessing is not reproduced."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, arguments.output)
    receipt = {
        "format_version": "boltz1_pairformer_extraction_receipt_v1",
        "artifact_path": str(arguments.output),
        "artifact_sha256": _sha256(arguments.output),
        "source_checkpoint_path": str(arguments.checkpoint),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_bytes": arguments.checkpoint.stat().st_size,
        "boltz_source_root": str(arguments.boltz_source_root),
        "boltz_source_revision": BOLTZ_REVISION,
        "state_prefix": prefix,
        "state_tensor_count": len(selected),
        "architecture": artifact["architecture"],
        "native_boltz_pipeline_reproduced": False,
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_bytes(_canonical_bytes(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
