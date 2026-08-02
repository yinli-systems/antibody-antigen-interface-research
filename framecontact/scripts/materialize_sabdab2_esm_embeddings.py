#!/usr/bin/env python3
"""Fill and authenticate pinned ESM-2 embeddings for the full SAbDab2 cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from interaction_first.esm_embeddings import (  # noqa: E402
    EMBED_DIM,
    ESM_MODEL_NAME,
    ESM_MODEL_REVISION,
    ESM_MODEL_WEIGHT_SHA256,
    atomic_torch_save,
    embed_sequence,
    embedding_path,
    sequence_sha256,
    validate_embedding_artifact,
)
from interaction_first.sabdab2_structure import authenticate_file, sha256_file  # noqa: E402


def _sequences(sequence_path: Path, prepared_path: Path) -> dict[str, str]:
    import pandas as pd

    source = pd.read_parquet(
        sequence_path, columns=["instance_id", "ab_seq", "ag_seq"]
    )
    prepared = pd.read_parquet(
        prepared_path,
        columns=[
            "instance_id", "ab_sequence_sha256", "ag_sequence_sha256",
            "ab_length", "ag_length",
        ],
    )
    if source["instance_id"].duplicated().any() or prepared["instance_id"].duplicated().any():
        raise ValueError("SAbDab2 sequence authorities contain duplicate instances")
    source_by_id = source.set_index("instance_id")
    prepared_by_id = prepared.set_index("instance_id")
    if set(source_by_id.index) != set(prepared_by_id.index):
        raise ValueError("SAbDab2 sequence authority cohorts differ")
    sequences: dict[str, str] = {}
    for identifier in sorted(source_by_id.index):
        for partner in ("ab", "ag"):
            sequence = str(source_by_id.loc[identifier, f"{partner}_seq"]).strip().upper()
            key = sequence_sha256(sequence)
            expected_key = str(
                prepared_by_id.loc[identifier, f"{partner}_sequence_sha256"]
            )
            expected_length = int(prepared_by_id.loc[identifier, f"{partner}_length"])
            if not sequence or key != expected_key or len(sequence) != expected_length:
                raise ValueError(f"{identifier}: {partner} sequence authority differs")
            previous = sequences.setdefault(key, sequence)
            if previous != sequence:
                raise AssertionError("SHA-256 sequence collision")
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--sequences-sha256", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-sha256", required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-sequences", type=int)
    arguments = parser.parse_args()
    authority = {
        "sequences_sha256": authenticate_file(
            arguments.sequences, arguments.sequences_sha256, "sequence authority"
        ),
        "prepared_sha256": authenticate_file(
            arguments.prepared, arguments.prepared_sha256, "prepared authority"
        ),
        "sequence_columns_read": ["instance_id", "ab_seq", "ag_seq"],
        "prepared_columns_read": [
            "instance_id", "ab_sequence_sha256", "ag_sequence_sha256",
            "ab_length", "ag_length",
        ],
        "task_labels_read": False,
    }
    model_weight = arguments.model_root / "model.safetensors"
    if sha256_file(model_weight) != ESM_MODEL_WEIGHT_SHA256:
        raise ValueError("Local ESM model weight SHA-256 differs from the frozen authority")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if arguments.device == "cuda" and cublas not in {":4096:8", ":16:8"}:
        raise SystemExit("CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG")
    sequences = _sequences(arguments.sequences, arguments.prepared)
    selected = sorted(sequences)
    if arguments.max_sequences is not None:
        if arguments.max_sequences <= 0:
            raise SystemExit("--max-sequences must be positive")
        selected = selected[: arguments.max_sequences]
    manifest_path = arguments.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError("Refusing to overwrite SAbDab2 ESM manifest")

    missing = [key for key in selected if not embedding_path(arguments.embedding_root, key).is_file()]
    tokenizer = model = None
    device = torch.device(arguments.device)
    if missing:
        from transformers import EsmModel, EsmTokenizer

        tokenizer = EsmTokenizer.from_pretrained(
            arguments.model_root, local_files_only=True
        )
        model = EsmModel.from_pretrained(arguments.model_root, local_files_only=True)
        if int(model.config.hidden_size) != EMBED_DIM:
            raise ValueError("ESM hidden dimension differs")
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

    random.seed(20_260_730)
    np.random.seed(20_260_730)
    torch.manual_seed(20_260_730)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20_260_730)
    records = []
    started = time.monotonic()
    for index, key in enumerate(selected, start=1):
        path = embedding_path(arguments.embedding_root, key)
        if path.is_file() and not path.is_symlink():
            if not arguments.resume:
                raise FileExistsError(f"Embedding exists without --resume: {path}")
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            validate_embedding_artifact(artifact, expected_key=key)
            if int(artifact["sequence_length"]) != len(sequences[key]):
                raise ValueError("Existing embedding length differs")
            state = "verified_existing"
        else:
            if tokenizer is None or model is None:
                raise AssertionError("ESM runtime was not initialized")
            artifact = embed_sequence(sequences[key], tokenizer, model, device)
            atomic_torch_save(artifact, path)
            state = "written"
        records.append(
            {
                "sequence_sha256": key,
                "sequence_length": len(sequences[key]),
                "artifact_path": path.relative_to(arguments.embedding_root).as_posix(),
                "artifact_sha256": sha256_file(path),
                "state": state,
            }
        )
        if index % 20 == 0 or index == len(selected):
            print(f"verified/embedded {index}/{len(selected)}", flush=True)
    manifest = {
        "schema_version": "sabdab2_esm_embedding_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": authority,
        "model_name": ESM_MODEL_NAME,
        "model_revision": ESM_MODEL_REVISION,
        "model_weight_sha256": ESM_MODEL_WEIGHT_SHA256,
        "embedding_dimension": EMBED_DIM,
        "selected_sequence_count": len(selected),
        "complete_cohort_sequence_count": len(sequences),
        "complete": len(selected) == len(sequences),
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cublas_workspace_config": cublas,
            "elapsed_seconds": time.monotonic() - started,
        },
        "records": records,
        "claim_boundary": "Frozen sequence embeddings only; no task label or performance claim.",
    }
    arguments.embedding_root.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
