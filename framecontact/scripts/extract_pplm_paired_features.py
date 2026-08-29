#!/usr/bin/env python3
"""Extract safe, provenance-bound paired features from an official PPLM checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.paired_sequence_features import (  # noqa: E402
    PPLM_ATTENTION_REDUCTION,
    PPLM_CACHE_FORMAT,
    PPLM_LICENSE_BOUNDARY,
    PPLM_OFFICIAL_REPOSITORY,
    PPLM_OFFICIAL_REVISION,
    reduce_pplm_inter_attention,
    sequence_sha256,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)


_ALLOWED_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWYBXZOU")


def _read_fasta(path: Path, name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} FASTA must be a regular non-symlink file")
    sequence = "".join(
        line.strip()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip() and not line.startswith(">")
    ).upper()
    if not sequence or any(residue not in _ALLOWED_RESIDUES for residue in sequence):
        raise ValueError(f"{name} FASTA contains an empty or unsupported sequence")
    return sequence


def _git_revision(source_root: Path) -> str:
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("PPLM source root must be a regular directory")
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PPLM_OFFICIAL_REVISION:
        raise ValueError("PPLM source revision differs from the pinned revision")
    tracked_status = subprocess.check_output(
        [
            "git",
            "-C",
            str(source_root),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if tracked_status:
        raise ValueError("PPLM source has modified tracked files")
    return revision


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite PPLM artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial.npz")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary PPLM artifact already exists: {temporary}")
    try:
        np.savez_compressed(temporary, **arrays)
        with np.load(temporary, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise RuntimeError("written PPLM artifact arrays differ")
            for name, expected in arrays.items():
                observed = np.asarray(archive[name])
                if observed.shape != expected.shape or observed.dtype != expected.dtype:
                    raise RuntimeError(f"written PPLM array differs: {name}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite PPLM receipt: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary PPLM receipt already exists: {temporary}")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pplm-source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--antibody-fasta", type=Path, required=True)
    parser.add_argument("--antigen-fasta", type=Path, required=True)
    parser.add_argument("--fixture-receipt", type=Path, required=True)
    parser.add_argument("--fixture-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    started = time.perf_counter()
    revision = _git_revision(arguments.pplm_source_root)
    if arguments.checkpoint.is_symlink() or not arguments.checkpoint.is_file():
        raise ValueError("PPLM checkpoint must be a regular non-symlink file")
    observed_checkpoint_sha256 = sha256_file(arguments.checkpoint)
    if observed_checkpoint_sha256 != arguments.checkpoint_sha256:
        raise ValueError("PPLM checkpoint SHA-256 differs")
    antibody_sequence = _read_fasta(arguments.antibody_fasta, "antibody")
    antigen_sequence = _read_fasta(arguments.antigen_fasta, "antigen")
    if arguments.fixture_receipt.is_symlink() or not arguments.fixture_receipt.is_file():
        raise ValueError("PPLM fixture receipt must be a regular non-symlink file")
    if sha256_file(arguments.fixture_receipt) != arguments.fixture_receipt_sha256:
        raise ValueError("PPLM fixture receipt SHA-256 differs")
    try:
        fixture = json.loads(arguments.fixture_receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("PPLM fixture receipt is invalid JSON") from error
    if (
        fixture.get("format_version") != "framecontact_pplm_train_fixture_v1"
        or fixture.get("role") != "train"
        or fixture.get("antibody_length") != len(antibody_sequence)
        or fixture.get("antigen_length") != len(antigen_sequence)
        or fixture.get("antibody_sequence_sha256")
        != sequence_sha256(antibody_sequence)
        or fixture.get("antigen_sequence_sha256")
        != sequence_sha256(antigen_sequence)
    ):
        raise ValueError("PPLM fixture receipt does not bind the train pair")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    sys.path.insert(0, str(arguments.pplm_source_root))
    from pplm import Alphabet, PPLM  # type: ignore[import-not-found]

    # weights_only=True rejects arbitrary checkpoint globals.  Do not weaken
    # this boundary to reproduce the official script's unrestricted pickle load.
    model_data = torch.load(
        arguments.checkpoint, map_location="cpu", weights_only=True
    )
    if not isinstance(model_data, dict) or set(model_data) < {"param", "model"}:
        raise ValueError("PPLM checkpoint does not contain param and model mappings")
    parameters = model_data["param"]
    state = model_data["model"]
    if not isinstance(parameters, dict) or not isinstance(state, dict):
        raise ValueError("PPLM checkpoint param/model entries must be mappings")
    architecture = {
        "encoder_layers": int(parameters.get("encoder_layers", -1)),
        "encoder_embed_dim": int(parameters.get("encoder_embed_dim", -1)),
        "encoder_attention_heads": int(
            parameters.get("encoder_attention_heads", -1)
        ),
    }
    if architecture != {
        "encoder_layers": 33,
        "encoder_embed_dim": 1280,
        "encoder_attention_heads": 20,
    }:
        raise ValueError(f"unexpected PPLM architecture: {architecture}")

    alphabet = Alphabet.from_architecture()
    model = PPLM(
        num_layers=architecture["encoder_layers"],
        embed_dim=architecture["encoder_embed_dim"],
        attention_heads=architecture["encoder_attention_heads"],
        token_dropout=False,
        alphabet=alphabet,
    )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "PPLM state_dict is not exact: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch_converter = alphabet.get_batch_converter()
    _, _, antibody_tokens = batch_converter(
        [("antibody", antibody_sequence)]
    )
    _, _, antigen_tokens = batch_converter([( "antigen", antigen_sequence)])
    tokens = torch.cat((antibody_tokens, antigen_tokens), dim=-1).to(device)
    antibody_token_count = len(antibody_sequence) + 2
    antigen_token_count = len(antigen_sequence) + 2
    total_token_count = antibody_token_count + antigen_token_count
    if tuple(tokens.shape) != (1, total_token_count):
        raise RuntimeError("PPLM tokenizer produced an unexpected shape")
    inter_chain_mask = torch.ones(
        total_token_count,
        total_token_count,
        device=device,
        dtype=torch.float32,
    )
    inter_chain_mask[:antibody_token_count, :antibody_token_count] = 0
    inter_chain_mask[antibody_token_count:, antibody_token_count:] = 0

    with torch.inference_mode():
        output = model(
            tokens,
            inter_chain_mask,
            repr_layers=[architecture["encoder_layers"]],
            need_head_weights=True,
            return_contacts=False,
        )
    representation = output["representations"][architecture["encoder_layers"]]
    attention = output["attentions"]
    expected_attention_shape = (
        1,
        architecture["encoder_layers"],
        architecture["encoder_attention_heads"],
        total_token_count,
        total_token_count,
    )
    if tuple(attention.shape) != expected_attention_shape:
        raise RuntimeError(
            "PPLM attention shape differs: "
            f"observed={tuple(attention.shape)} expected={expected_attention_shape}"
        )
    antibody_slice = slice(1, len(antibody_sequence) + 1)
    antigen_slice = slice(
        antibody_token_count + 1,
        antibody_token_count + 1 + len(antigen_sequence),
    )
    antibody_embedding = representation[0, antibody_slice].float()
    antigen_embedding = representation[0, antigen_slice].float()
    directional_ab = attention[0, :, :, antibody_slice, antigen_slice]
    directional_ba = attention[0, :, :, antigen_slice, antibody_slice]
    inter_attention = reduce_pplm_inter_attention(directional_ab, directional_ba)

    arrays = {
        "antibody_embedding": antibody_embedding.cpu().numpy().astype(
            np.float32, copy=False
        ),
        "antigen_embedding": antigen_embedding.cpu().numpy().astype(
            np.float32, copy=False
        ),
        "inter_attention": inter_attention.cpu().numpy().astype(
            np.float32, copy=False
        ),
    }
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise RuntimeError(f"PPLM output contains non-finite values: {name}")
    if np.any(arrays["inter_attention"] < 0):
        raise RuntimeError("PPLM reduced inter attention contains negative values")

    _atomic_npz(arguments.output, arrays)
    payload: dict[str, object] = {
        "format_version": PPLM_CACHE_FORMAT,
        "claim_boundary": (
            "Official-checkpoint feature extraction and GPU engineering evidence only. "
            "No trained FrameContact improvement, generalization, benchmark, novelty, "
            "commercial-use, or production-performance claim."
        ),
        "source_repository": PPLM_OFFICIAL_REPOSITORY,
        "source_revision": revision,
        "source_checkpoint_sha256": observed_checkpoint_sha256,
        "license_boundary": PPLM_LICENSE_BOUNDARY,
        "attention_reduction": PPLM_ATTENTION_REDUCTION,
        "chain_order": {"sequence_A": "antibody", "sequence_B": "antigen"},
        "antibody_sequence_sha256": sequence_sha256(antibody_sequence),
        "antigen_sequence_sha256": sequence_sha256(antigen_sequence),
        "fixture_receipt_sha256": arguments.fixture_receipt_sha256,
        "fixture_instance_id": fixture["instance_id"],
        "artifact_sha256": sha256_file(arguments.output),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "architecture": architecture,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    _atomic_json(arguments.receipt, payload)
    print(
        json.dumps(
            {
                "artifact_sha256": payload["artifact_sha256"],
                "receipt_sha256": sha256_file(arguments.receipt),
                "arrays": payload["arrays"],
                "runtime": payload["runtime"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
