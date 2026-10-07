"""Pinned ESM-2 sequence embeddings used by external evaluation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM_MODEL_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
ESM_MODEL_WEIGHT_SHA256 = "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"
ABF30B_MODEL_NAME = "abf_650m_oas30b_final"
ABF30B_MODEL_REVISION = "c36e0b1d812f97317109f1f8ab085a349beb2a782a6f4092b3074bfc1eaab3ea"
ABF30B_MODEL_WEIGHT_SHA256 = ABF30B_MODEL_REVISION
EMBED_DIM = 1280
MAX_TOKENS = 1024
STRIDE = 512
EMBEDDING_FORMAT = "sppider_seq_embedding_v1"


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.strip().upper().encode("ascii")).hexdigest()


def embedding_path(root: Path, sequence_key: str) -> Path:
    return root / sequence_key[:2] / f"{sequence_key}.pt"


def validate_embedding_artifact(
    artifact: Mapping[str, Any], *, expected_key: str | None = None
) -> None:
    required = {
        "format_version",
        "sequence_sha256",
        "sequence_length",
        "model_name",
        "max_tokens",
        "stride",
        "starts",
        "ends",
        "chunks",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"embedding artifact is missing keys: {sorted(missing)}")
    if artifact["format_version"] != EMBEDDING_FORMAT:
        raise ValueError("unexpected embedding format")
    model_name = artifact["model_name"]
    revision = artifact.get("model_revision")
    weight_sha256 = artifact.get("model_weight_sha256")
    if model_name == ESM_MODEL_NAME:
        if revision != ESM_MODEL_REVISION and weight_sha256 != ESM_MODEL_WEIGHT_SHA256:
            raise ValueError("ESM artifact lacks the pinned revision or exact weight digest")
        if int(artifact["max_tokens"]) != MAX_TOKENS or int(artifact["stride"]) != STRIDE:
            raise ValueError("ESM embedding chunk policy differs")
    elif model_name == ABF30B_MODEL_NAME:
        if (
            revision != ABF30B_MODEL_REVISION
            or weight_sha256 != ABF30B_MODEL_WEIGHT_SHA256
        ):
            raise ValueError("ABF-30B artifact lacks the exact checkpoint digest")
        if int(artifact["max_tokens"]) != 194 or int(artifact["stride"]) != 192:
            raise ValueError("ABF-30B per-chain embedding policy differs")
    else:
        raise ValueError("unexpected embedding model name")
    if expected_key is not None and artifact["sequence_sha256"] != expected_key:
        raise ValueError("embedding sequence digest differs")
    length = int(artifact["sequence_length"])
    starts = [int(value) for value in artifact["starts"]]
    ends = [int(value) for value in artifact["ends"]]
    chunks = list(artifact["chunks"])
    if length <= 0 or not chunks or len(chunks) != len(starts) or len(chunks) != len(ends):
        raise ValueError("embedding chunks are empty or inconsistent")
    covered = np.zeros(length, dtype=np.bool_)
    for start, end, chunk in zip(starts, ends, chunks):
        if start < 0 or start >= end or end > length:
            raise ValueError("embedding interval is invalid")
        if not torch.is_tensor(chunk) or tuple(chunk.shape) != (end - start, EMBED_DIM):
            raise ValueError("embedding chunk shape differs")
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("embedding contains non-finite values")
        covered[start:end] = True
    if not covered.all():
        raise ValueError("embedding chunks do not cover the sequence")


@torch.inference_mode()
def embed_sequence(
    sequence: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
) -> dict[str, Any]:
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("cannot embed an empty sequence")
    window = MAX_TOKENS - 2
    starts: list[int] = []
    ends: list[int] = []
    chunks: list[torch.Tensor] = []
    for start in range(0, len(sequence), STRIDE):
        end = min(start + window, len(sequence))
        encoded = tokenizer(sequence[start:end], return_tensors="pt", add_special_tokens=True)
        output = model(encoded["input_ids"].to(device))
        embedding = (
            output.last_hidden_state[:, 1:-1, :]
            .squeeze(0)
            .to(dtype=torch.float32, device="cpu")
            .contiguous()
        )
        if tuple(embedding.shape) != (end - start, EMBED_DIM):
            raise ValueError("ESM output does not preserve the residue axis")
        starts.append(start)
        ends.append(end)
        chunks.append(embedding)
        if end == len(sequence):
            break
    artifact = {
        "format_version": EMBEDDING_FORMAT,
        "sequence_sha256": sequence_sha256(sequence),
        "sequence_length": len(sequence),
        "model_name": ESM_MODEL_NAME,
        "model_revision": ESM_MODEL_REVISION,
        "max_tokens": MAX_TOKENS,
        "stride": STRIDE,
        "starts": starts,
        "ends": ends,
        "chunks": chunks,
    }
    validate_embedding_artifact(artifact, expected_key=artifact["sequence_sha256"])
    return artifact


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_embedding(path: Path, *, expected_key: str, expected_length: int) -> torch.Tensor:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing embedding: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    validate_embedding_artifact(artifact, expected_key=expected_key)
    if int(artifact["sequence_length"]) != expected_length:
        raise ValueError("embedding length differs")
    output = torch.zeros(expected_length, EMBED_DIM, dtype=torch.float32)
    counts = torch.zeros(expected_length, 1, dtype=torch.float32)
    for chunk, start, end in zip(
        artifact["chunks"], artifact["starts"], artifact["ends"]
    ):
        output[int(start) : int(end)] += chunk.float()
        counts[int(start) : int(end)] += 1.0
    if bool((counts == 0).any()):
        raise ValueError("embedding intervals leave uncovered residues")
    return output / counts
