#!/usr/bin/env python3
"""Materialize pinned ESM-2 embeddings for the label-free SABLE cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from types import ModuleType
from typing import Any

import numpy as np
import torch


EXPECTED_INPUT_PLAN_SHA256 = (
    "c82c12f08c87b42d21aee31cc555373ffbc4211415d9aff302444baefc971ad6"
)
EXPECTED_INPUT_RECEIPT_SHA256 = (
    "a6ba3357c282a8f375ab6e0890bb78d90b5cdbb41a75ff05b088796ab237b1d2"
)
EXPECTED_ESM_MODULE_SHA256 = (
    "2adc1d711f7d5a9d0fcfb34d4813468393d18ea45b0751fffa9d569857570dd8"
)
EXPECTED_ENTRIES = 97
EXPECTED_PDB_IDS = 88
EXPECTED_TRANSFORMERS_VERSION = "4.57.6"
EXPECTED_TOKENIZERS_VERSION = "0.22.2"
EXPECTED_SAFETENSORS_VERSION = "0.8.0"


class EmbeddingError(ValueError):
    """Raised when an embedding authority or artifact differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_esm_module(path: Path) -> ModuleType:
    if sha256_file(path) != EXPECTED_ESM_MODULE_SHA256:
        raise EmbeddingError("frozen ESM embedding module SHA256 differs")
    spec = importlib.util.spec_from_file_location("frozen_esm_embeddings", path)
    if spec is None or spec.loader is None:
        raise EmbeddingError("cannot import frozen ESM embedding module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "EMBED_DIM",
        "ESM_MODEL_NAME",
        "ESM_MODEL_REVISION",
        "ESM_MODEL_WEIGHT_SHA256",
        "atomic_torch_save",
        "embed_sequence",
        "embedding_path",
        "sequence_sha256",
        "validate_embedding_artifact",
    )
    for name in required:
        if not hasattr(module, name):
            raise EmbeddingError(f"frozen ESM module lacks {name}")
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--input-plan-receipt", type=Path, required=True)
    parser.add_argument("--esm-module", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dependency-site-packages", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256_file(args.input_plan) != EXPECTED_INPUT_PLAN_SHA256:
        raise EmbeddingError("AF3 label-free input plan SHA256 differs")
    if sha256_file(args.input_plan_receipt) != EXPECTED_INPUT_RECEIPT_SHA256:
        raise EmbeddingError("AF3 label-free input receipt SHA256 differs")
    plan = json.loads(args.input_plan.read_text(encoding="utf-8"))
    receipt = json.loads(args.input_plan_receipt.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "PASS_FROZEN_LABEL_FREE_AF3_INPUT_PLAN"
        or receipt.get("status") != plan.get("status")
        or receipt.get("plan_sha256") != EXPECTED_INPUT_PLAN_SHA256
        or plan.get("counts", {}).get("eligible_entries") != EXPECTED_ENTRIES
        or plan.get("counts", {}).get("eligible_unique_pdb_ids") != EXPECTED_PDB_IDS
        or plan.get("task_labels_read") is not False
        or plan.get("sable_bound_structures_downloaded") is not False
        or plan.get("selection_or_tuning") is not False
    ):
        raise EmbeddingError("label-free sequence authority differs")

    entities = plan.get("entities", [])
    entity_by_id = {row["entity_id"]: row for row in entities}
    bindings = plan.get("entry_bindings", [])
    if len(entity_by_id) != len(entities) or len(bindings) != EXPECTED_ENTRIES:
        raise EmbeddingError("entity or entry binding cardinality differs")

    esm = load_esm_module(args.esm_module)
    model_weight = args.model_root / "model.safetensors"
    if sha256_file(model_weight) != esm.ESM_MODEL_WEIGHT_SHA256:
        raise EmbeddingError("local ESM model weight SHA256 differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if args.device == "cuda" and cublas not in {":4096:8", ":16:8"}:
        raise SystemExit("CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG")

    sequences: dict[str, str] = {}
    entry_records: list[dict[str, Any]] = []
    for binding in sorted(bindings, key=lambda row: row["entry_id"]):
        if (
            binding.get("task_labels_read") is not False
            or binding.get("bound_structure_read") is not False
        ):
            raise EmbeddingError("entry binding is not label-free")
        antibody = entity_by_id.get(binding["antibody_entity_id"])
        antigen_entities = [
            entity_by_id.get(entity_id)
            for entity_id in binding["antigen_entity_ids_in_declared_order"]
        ]
        if antibody is None or not antigen_entities or any(
            entity is None for entity in antigen_entities
        ):
            raise EmbeddingError("entry binding references an unknown entity")
        antibody_sequence = "".join(str(value) for value in antibody["sequences"])
        antigen_sequence = "".join(
            str(sequence)
            for entity in antigen_entities
            for sequence in entity["sequences"]
        )
        if not antibody_sequence or not antigen_sequence:
            raise EmbeddingError("entry contains an empty model-axis sequence")
        antibody_key = esm.sequence_sha256(antibody_sequence)
        antigen_key = esm.sequence_sha256(antigen_sequence)
        for key, sequence in (
            (antibody_key, antibody_sequence),
            (antigen_key, antigen_sequence),
        ):
            previous = sequences.setdefault(key, sequence)
            if previous != sequence:
                raise AssertionError("sequence SHA256 collision")
        entry_records.append(
            {
                "entry_id": binding["entry_id"],
                "pdb_id": binding["pdb_id"],
                "antibody_sequence_sha256": antibody_key,
                "antibody_sequence_length": len(antibody_sequence),
                "antigen_sequence_sha256": antigen_key,
                "antigen_sequence_length": len(antigen_sequence),
            }
        )

    manifest_path = args.embedding_root / "SABLE_ESM_EMBEDDING_MANIFEST.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(manifest_path)
    selected = sorted(sequences)
    missing = [
        key
        for key in selected
        if not esm.embedding_path(args.embedding_root, key).is_file()
    ]
    tokenizer = model = None
    device = torch.device(args.device)
    if missing:
        if not args.dependency_site_packages.is_dir():
            raise EmbeddingError("read-only dependency site-packages is missing")
        sys.path.append(str(args.dependency_site_packages))
        import safetensors
        import tokenizers
        import transformers
        from transformers import EsmModel, EsmTokenizer

        observed_versions = {
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "safetensors": safetensors.__version__,
        }
        if observed_versions != {
            "transformers": EXPECTED_TRANSFORMERS_VERSION,
            "tokenizers": EXPECTED_TOKENIZERS_VERSION,
            "safetensors": EXPECTED_SAFETENSORS_VERSION,
        }:
            raise EmbeddingError("read-only dependency versions differ")

        tokenizer = EsmTokenizer.from_pretrained(
            args.model_root, local_files_only=True
        )
        model = EsmModel.from_pretrained(args.model_root, local_files_only=True)
        if int(model.config.hidden_size) != int(esm.EMBED_DIM):
            raise EmbeddingError("ESM hidden dimension differs")
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

    random.seed(20_260_829)
    np.random.seed(20_260_829)
    torch.manual_seed(20_260_829)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20_260_829)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, key in enumerate(selected, start=1):
        path = esm.embedding_path(args.embedding_root, key)
        if path.is_file() and not path.is_symlink():
            if not args.resume:
                raise FileExistsError(f"embedding exists without --resume: {path}")
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            esm.validate_embedding_artifact(artifact, expected_key=key)
            if int(artifact["sequence_length"]) != len(sequences[key]):
                raise EmbeddingError("existing embedding length differs")
            state = "verified_existing"
        else:
            if tokenizer is None or model is None:
                raise AssertionError("ESM runtime was not initialized")
            artifact = esm.embed_sequence(sequences[key], tokenizer, model, device)
            esm.atomic_torch_save(artifact, path)
            state = "written"
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
            print(f"verified/embedded {index}/{len(selected)}", flush=True)

    manifest = {
        "format_version": "sable_v0_2_0_esm2_embedding_manifest_v1",
        "status": "PASS_LABEL_FREE_ESM2_EMBEDDINGS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": {
            "input_plan_sha256": EXPECTED_INPUT_PLAN_SHA256,
            "input_plan_receipt_sha256": EXPECTED_INPUT_RECEIPT_SHA256,
            "esm_module_path": str(args.esm_module),
            "esm_module_sha256": EXPECTED_ESM_MODULE_SHA256,
            "dependency_site_packages": str(args.dependency_site_packages),
            "dependency_versions": {
                "transformers": EXPECTED_TRANSFORMERS_VERSION,
                "tokenizers": EXPECTED_TOKENIZERS_VERSION,
                "safetensors": EXPECTED_SAFETENSORS_VERSION,
            },
            "task_labels_read": False,
            "bound_structures_read": False,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
            "selection_or_tuning": False,
        },
        "model_name": esm.ESM_MODEL_NAME,
        "model_revision": esm.ESM_MODEL_REVISION,
        "model_weight_sha256": esm.ESM_MODEL_WEIGHT_SHA256,
        "embedding_dimension": int(esm.EMBED_DIM),
        "entry_count": len(entry_records),
        "unique_pdb_id_count": len({row["pdb_id"] for row in entry_records}),
        "unique_sequence_count": len(records),
        "entry_records": entry_records,
        "records": records,
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cublas_workspace_config": cublas,
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    if len(entry_records) != EXPECTED_ENTRIES:
        raise EmbeddingError("not every eligible entry was embedded")
    if manifest["unique_pdb_id_count"] != EXPECTED_PDB_IDS:
        raise EmbeddingError("embedded PDB cardinality differs")
    args.embedding_root.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
