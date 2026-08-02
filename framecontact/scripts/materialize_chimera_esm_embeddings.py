#!/usr/bin/env python3
"""Materialize pinned ESM-2 embeddings for the strict CHIMERA adapter cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from interaction_first.chimera_cache import load_cache_manifest  # noqa: E402
from interaction_first.esm_embeddings import (  # noqa: E402
    EMBED_DIM,
    ESM_MODEL_NAME,
    ESM_MODEL_REVISION,
    atomic_torch_save,
    embed_sequence,
    embedding_path,
    sequence_sha256,
    validate_embedding_artifact,
)
from interaction_first.sabdab2_structure import sha256_file  # noqa: E402


def _sequences(cache_root: Path, manifest: dict) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for record in manifest["records"]:
        path = cache_root / record["artifact_path"]
        if sha256_file(path) != record["artifact_sha256"]:
            raise ValueError("adapter artifact SHA-256 differs")
        with np.load(path, allow_pickle=False) as archive:
            for field in ("antibody_sequence", "antigen_sequence"):
                sequence = str(archive[field].item())
                key = sequence_sha256(sequence)
                previous = sequences.setdefault(key, sequence)
                if previous != sequence:
                    raise AssertionError("sequence SHA-256 collision")
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-sequences", type=int)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if args.device == "cuda" and cublas_workspace_config not in {":4096:8", ":16:8"}:
        raise SystemExit(
            "CUDA determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    device = torch.device(args.device)
    cache_manifest = load_cache_manifest(
        args.cache_root, expected_sha256=args.expected_cache_manifest_sha256
    )
    sequences = _sequences(args.cache_root, cache_manifest)
    selected = sorted(sequences)
    if args.max_sequences is not None:
        if args.max_sequences <= 0:
            raise SystemExit("--max-sequences must be positive")
        selected = selected[: args.max_sequences]

    random.seed(20260730)
    np.random.seed(20260730)
    torch.manual_seed(20260730)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260730)
    from transformers import EsmModel, EsmTokenizer

    tokenizer = EsmTokenizer.from_pretrained(
        ESM_MODEL_NAME,
        revision=ESM_MODEL_REVISION,
        cache_dir=str(args.hf_cache) if args.hf_cache else None,
        local_files_only=True,
    )
    model = EsmModel.from_pretrained(
        ESM_MODEL_NAME,
        revision=ESM_MODEL_REVISION,
        cache_dir=str(args.hf_cache) if args.hf_cache else None,
        local_files_only=True,
    )
    commit_hash = getattr(model.config, "_commit_hash", None)
    if commit_hash not in {None, ESM_MODEL_REVISION}:
        raise ValueError(f"resolved ESM revision differs: {commit_hash}")
    if int(model.config.hidden_size) != EMBED_DIM:
        raise ValueError("ESM hidden dimension differs")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    records = []
    started = time.monotonic()
    for index, key in enumerate(selected, start=1):
        path = embedding_path(args.embedding_root, key)
        state = "written"
        if args.resume and path.is_file() and not path.is_symlink():
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            validate_embedding_artifact(artifact, expected_key=key)
            if int(artifact["sequence_length"]) != len(sequences[key]):
                raise ValueError("existing embedding length differs")
            state = "verified_existing"
        else:
            artifact = embed_sequence(sequences[key], tokenizer, model, device)
            atomic_torch_save(artifact, path)
        records.append(
            {
                "sequence_sha256": key,
                "sequence_length": len(sequences[key]),
                "artifact_path": path.relative_to(args.embedding_root).as_posix(),
                "artifact_sha256": sha256_file(path),
                "state": state,
            }
        )
        if index % 20 == 0 or index == len(selected):
            print(f"embedded {index}/{len(selected)} sequences", flush=True)

    manifest = {
        "schema_version": "chimera_esm_embedding_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_manifest_sha256": args.expected_cache_manifest_sha256,
        "model_name": ESM_MODEL_NAME,
        "model_revision": ESM_MODEL_REVISION,
        "resolved_commit_hash": commit_hash,
        "embedding_dimension": EMBED_DIM,
        "selected_sequence_count": len(selected),
        "complete_cache_sequence_count": len(sequences),
        "complete": len(selected) == len(sequences),
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cublas_workspace_config": cublas_workspace_config,
            "elapsed_seconds": time.monotonic() - started,
        },
        "records": records,
    }
    args.embedding_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.embedding_root / "CHIMERA_ESM_EMBEDDING_MANIFEST.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
