#!/usr/bin/env python3
"""Materialize ABF-30B antibody embeddings and retain ESM-2 antigen embeddings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import enum
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from antibody_pretraining.model import (
    AntibodyMaskedLanguageModel,
    config_from_preset,
    load_model_family,
)
from antibody_pretraining.tokenizer import encode_unpaired, pad_encoded


FINAL_MODEL_SHA256 = "c36e0b1d812f97317109f1f8ab085a349beb2a782a6f4092b3074bfc1eaab3ea"
MODEL_FAMILY_SHA256 = "1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015"
SEQUENCE_VIEW_SHA256 = "27e0beb755c3d8e4b9910d4ad2ca979161668a1e1fe7b4d09bf6a60251e07468"
ESM2_MANIFEST_SHA256 = "9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545"
ABF_MODEL_NAME = "abf_650m_oas30b_final"
EMBEDDING_FORMAT = "sppider_seq_embedding_v1"
EXPECTED_RECORDS = 3114


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


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
            digest.update(
                b"type\0" + f"{item.__module__}.{item.__qualname__}".encode("ascii") + b"\0"
            )
        elif isinstance(item, enum.Enum):
            digest.update(
                b"enum\0"
                + f"{item.__class__.__module__}.{item.__class__.__qualname__}".encode("ascii")
                + b"\0"
            )
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


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def artifact_path(root: Path, digest: str) -> Path:
    return root / digest[:2] / f"{digest}.pt"


def validate_abf_artifact(artifact: dict[str, Any], expected_sha: str, length: int) -> None:
    if (
        artifact.get("format_version") != EMBEDDING_FORMAT
        or artifact.get("sequence_sha256") != expected_sha
        or int(artifact.get("sequence_length", -1)) != length
        or artifact.get("model_name") != ABF_MODEL_NAME
        or artifact.get("model_revision") != FINAL_MODEL_SHA256
        or artifact.get("model_weight_sha256") != FINAL_MODEL_SHA256
        or int(artifact.get("max_tokens", -1)) != 194
        or int(artifact.get("stride", -1)) != 192
    ):
        raise ValueError("ABF embedding artifact authority differs")
    starts = [int(value) for value in artifact.get("starts", [])]
    ends = [int(value) for value in artifact.get("ends", [])]
    chunks = artifact.get("chunks", [])
    if not chunks or len(chunks) != len(starts) or len(chunks) != len(ends):
        raise ValueError("ABF embedding chunks differ")
    covered = torch.zeros(length, dtype=torch.bool)
    for start, end, chunk in zip(starts, ends, chunks):
        if (
            start < 0
            or start >= end
            or end > length
            or not torch.is_tensor(chunk)
            or tuple(chunk.shape) != (end - start, 1280)
            or not bool(torch.isfinite(chunk).all())
        ):
            raise ValueError("ABF embedding chunk is invalid")
        covered[start:end] = True
    if not bool(covered.all()):
        raise ValueError("ABF embedding leaves uncovered residues")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--chain-map", type=Path, required=True)
    parser.add_argument("--chain-map-sha256", required=True)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--canary-sha256", required=True)
    parser.add_argument("--sequence-view", type=Path, required=True)
    parser.add_argument("--model-family", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--esm2-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--batch-antibodies", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_antibodies <= 0:
        raise ValueError("batch-antibodies must be positive")
    expected = (
        (args.protocol, args.protocol_sha256),
        (args.chain_map, args.chain_map_sha256),
        (args.canary, args.canary_sha256),
        (args.sequence_view, SEQUENCE_VIEW_SHA256),
        (args.model_family, MODEL_FAMILY_SHA256),
        (args.esm2_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json", ESM2_MANIFEST_SHA256),
    )
    for path, digest in expected:
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"authority SHA differs: {path}")
    protocol = json.loads(args.protocol.read_text())
    chain_map = json.loads(args.chain_map.read_text())
    canary = json.loads(args.canary.read_text())
    if (
        protocol.get("status")
        != "FROZEN_BEFORE_CHAIN_TYPING_DCP_RESTORE_AND_MODEL_SPECIFIC_MEASUREMENT"
        or chain_map.get("status") != "PASS_LABEL_FREE_ANARCII_CHAIN_TYPING"
        or canary.get("status") != "PASS_DCP_EXACT_RESTORE_AND_RESIDUE_EMBEDDING_CANARY"
        or canary.get("checkpoint_model_sha256") != FINAL_MODEL_SHA256
        or chain_map.get("task_labels_read") is not False
        or canary.get("task_labels_read") is not False
    ):
        raise ValueError("protocol, chain-map, or canary authority differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    sequence_rows: dict[str, dict[str, Any]] = {}
    antibodies: dict[str, str] = {}
    antigens: dict[str, str] = {}
    with args.sequence_view.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            identifier = str(row["instance_id"])
            if identifier in sequence_rows or row["role"] not in {"train", "validation"}:
                raise ValueError("development sequence row differs")
            ab_seq = str(row["ab_seq"]).strip().upper()
            ag_seq = str(row["ag_seq"]).strip().upper()
            if (
                sequence_sha256(ab_seq) != row["ab_sequence_sha256"]
                or sequence_sha256(ag_seq) != row["ag_sequence_sha256"]
            ):
                raise ValueError("development sequence SHA differs")
            sequence_rows[identifier] = row
            antibodies.setdefault(row["ab_sequence_sha256"], ab_seq)
            antigens.setdefault(row["ag_sequence_sha256"], ag_seq)
    if len(sequence_rows) != EXPECTED_RECORDS or set(antibodies) & set(antigens):
        raise ValueError("development sequence cardinality or partner overlap differs")

    unique_chain_by_sha = {row["sequence_sha256"]: row for row in chain_map["unique_chains"]}
    chain_spec_by_ab: dict[str, tuple[dict[str, Any], ...]] = {}
    for row in chain_map["records"]:
        identifier = row["instance_id"]
        source = sequence_rows.get(identifier)
        if source is None or row["antibody_sequence_sha256"] != source["ab_sequence_sha256"]:
            raise ValueError("chain map sequence binding differs")
        specs = tuple(row["chains"])
        previous = chain_spec_by_ab.setdefault(source["ab_sequence_sha256"], specs)
        if previous != specs:
            raise ValueError("repeated antibody chain map differs")
    if set(chain_spec_by_ab) != set(antibodies):
        raise ValueError("chain map does not cover unique development antibodies")
    for ab_sha, specs in chain_spec_by_ab.items():
        reconstructed = "".join(unique_chain_by_sha[spec["sequence_sha256"]]["sequence"] for spec in specs)
        if reconstructed != antibodies[ab_sha]:
            raise ValueError("chain map does not reconstruct antibody sequence")

    esm2_manifest = json.loads(
        (args.esm2_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json").read_text()
    )
    esm2_record_by_sha = {row["sequence_sha256"]: row for row in esm2_manifest["records"]}
    if not set(antigens) <= set(esm2_record_by_sha):
        raise ValueError("ESM-2 manifest does not cover development antigens")

    # The frozen FrameContact CLI discovers this exact filename; the manifest's
    # own format_version and per-record model fields distinguish the mixed cache.
    manifest_path = args.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
    receipt_path = args.embedding_root / "SABDAB2_ABF30B_MIXED_EMBEDDING_RECEIPT.json"
    if manifest_path.exists() or receipt_path.exists():
        raise FileExistsError("mixed embedding manifest or receipt already exists")
    args.embedding_root.mkdir(parents=True, exist_ok=True)

    family = load_model_family(args.model_family)
    model = AntibodyMaskedLanguageModel(config_from_preset(family, "abf_650m"))
    restore_model(model, args.checkpoint)
    observed_model_sha = tree_digest(model.state_dict())
    if observed_model_sha != FINAL_MODEL_SHA256:
        raise RuntimeError("restored model digest differs")
    device = torch.device("cuda")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], value: torch.Tensor) -> None:
        captured.append(value.detach())

    hook_handle = model.final_norm.register_forward_hook(hook)
    ordered = sorted(antibodies)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    torch.manual_seed(20_260_830)
    torch.cuda.manual_seed_all(20_260_830)
    torch.use_deterministic_algorithms(True, warn_only=True)
    for offset in range(0, len(ordered), args.batch_antibodies):
        batch_shas = ordered[offset : offset + args.batch_antibodies]
        encoded = []
        ownership: list[tuple[str, dict[str, Any], str]] = []
        for ab_sha in batch_shas:
            for spec in chain_spec_by_ab[ab_sha]:
                chain_row = unique_chain_by_sha[spec["sequence_sha256"]]
                chain_sequence = chain_row["sequence"]
                chain_type = spec["abf_chain_type"]
                encoded.append(encode_unpaired(chain_sequence, chain_type))
                ownership.append((ab_sha, spec, chain_sequence))
        input_ids, type_ids, masks = pad_encoded(encoded)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(
                torch.tensor(input_ids, dtype=torch.long, device=device),
                torch.tensor(type_ids, dtype=torch.long, device=device),
                attention_mask=torch.tensor(masks, dtype=torch.bool, device=device),
            )
        hidden = captured.pop().float().cpu()
        per_antibody: dict[str, list[tuple[dict[str, Any], torch.Tensor]]] = {
            sha: [] for sha in batch_shas
        }
        for index, (ab_sha, spec, chain_sequence) in enumerate(ownership):
            chain_hidden = hidden[index, 1 : 1 + len(chain_sequence)].contiguous()
            if tuple(chain_hidden.shape) != (len(chain_sequence), 1280) or not bool(
                torch.isfinite(chain_hidden).all()
            ):
                raise RuntimeError("ABF residue embedding is non-finite or shape-invalid")
            per_antibody[ab_sha].append((spec, chain_hidden))
        for ab_sha in batch_shas:
            path = artifact_path(args.embedding_root, ab_sha)
            specs_and_hidden = per_antibody[ab_sha]
            starts: list[int] = []
            ends: list[int] = []
            chunks: list[torch.Tensor] = []
            cursor = 0
            for spec, chain_hidden in specs_and_hidden:
                starts.append(cursor)
                cursor += int(chain_hidden.shape[0])
                ends.append(cursor)
                chunks.append(chain_hidden)
            artifact = {
                "format_version": EMBEDDING_FORMAT,
                "sequence_sha256": ab_sha,
                "sequence_length": len(antibodies[ab_sha]),
                "model_name": ABF_MODEL_NAME,
                "model_revision": FINAL_MODEL_SHA256,
                "model_weight_sha256": FINAL_MODEL_SHA256,
                "max_tokens": 194,
                "stride": 192,
                "starts": starts,
                "ends": ends,
                "chunks": chunks,
                "embedding_definition": "final RMS-normalized residue hidden state",
                "chain_types": [spec["abf_chain_type"] for spec, _ in specs_and_hidden],
                "chain_sequence_sha256s": [spec["sequence_sha256"] for spec, _ in specs_and_hidden],
            }
            validate_abf_artifact(artifact, ab_sha, len(antibodies[ab_sha]))
            if path.exists():
                if not args.resume or path.is_symlink():
                    raise FileExistsError(path)
                existing = torch.load(path, map_location="cpu", weights_only=True)
                validate_abf_artifact(existing, ab_sha, len(antibodies[ab_sha]))
                for expected_chunk, existing_chunk in zip(chunks, existing["chunks"]):
                    if not torch.equal(expected_chunk, existing_chunk):
                        raise RuntimeError("resumed ABF embedding differs from regenerated value")
                state = "verified_existing"
            else:
                atomic_torch_save(artifact, path)
                state = "written"
            records.append(
                {
                    "partner": "antibody",
                    "sequence_sha256": ab_sha,
                    "sequence_length": len(antibodies[ab_sha]),
                    "artifact_path": path.relative_to(args.embedding_root).as_posix(),
                    "artifact_sha256": sha256_file(path),
                    "model_name": ABF_MODEL_NAME,
                    "state": state,
                }
            )
        completed = min(offset + len(batch_shas), len(ordered))
        print(f"embedded antibodies {completed}/{len(ordered)}", flush=True)
    hook_handle.remove()
    if captured:
        raise AssertionError("final-norm hook capture count differs")

    for index, ag_sha in enumerate(sorted(antigens), start=1):
        source_record = esm2_record_by_sha[ag_sha]
        source = args.esm2_root / source_record["artifact_path"]
        destination = artifact_path(args.embedding_root, ag_sha)
        if source.is_symlink() or not source.is_file() or sha256_file(source) != source_record["artifact_sha256"]:
            raise ValueError("source ESM-2 antigen artifact differs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not args.resume or destination.is_symlink() or sha256_file(destination) != source_record["artifact_sha256"]:
                raise FileExistsError(destination)
            state = "verified_existing_hardlink"
        else:
            os.link(source, destination)
            state = "hardlinked_original_esm2"
        records.append(
            {
                "partner": "antigen",
                "sequence_sha256": ag_sha,
                "sequence_length": len(antigens[ag_sha]),
                "artifact_path": destination.relative_to(args.embedding_root).as_posix(),
                "artifact_sha256": sha256_file(destination),
                "model_name": "facebook/esm2_t33_650M_UR50D",
                "state": state,
            }
        )
        if index % 250 == 0 or index == len(antigens):
            print(f"linked antigens {index}/{len(antigens)}", flush=True)

    manifest = {
        "format_version": "sabdab2_abf30b_mixed_partner_embedding_manifest_v1",
        "status": "PASS_COMPLETE_DEVELOPMENT_MIXED_PARTNER_EMBEDDINGS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": args.protocol_sha256,
        "chain_map_sha256": args.chain_map_sha256,
        "canary_sha256": args.canary_sha256,
        "sequence_view_sha256": SEQUENCE_VIEW_SHA256,
        "control_esm2_manifest_sha256": ESM2_MANIFEST_SHA256,
        "antibody_model": {
            "model_name": ABF_MODEL_NAME,
            "checkpoint_model_sha256": FINAL_MODEL_SHA256,
            "embedding_dimension": 1280,
            "embedding_definition": "final RMS-normalized residue hidden state",
        },
        "antigen_model": {
            "model_name": "facebook/esm2_t33_650M_UR50D",
            "artifacts_reused_byte_for_byte": True,
        },
        "counts": {
            "development_instances": len(sequence_rows),
            "unique_antibody_sequences": len(antibodies),
            "unique_antigen_sequences": len(antigens),
            "records": len(records),
        },
        "records": sorted(records, key=lambda row: (row["partner"], row["sequence_sha256"])),
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "elapsed_seconds": time.monotonic() - started,
        },
        "task_labels_read": False,
        "old_sabdab2_test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    manifest_sha = publish(manifest_path, manifest)
    receipt = {
        "format_version": "sabdab2_abf30b_mixed_partner_embedding_receipt_v1",
        "status": manifest["status"],
        "manifest_sha256": manifest_sha,
        "checkpoint_model_sha256": FINAL_MODEL_SHA256,
        "counts": manifest["counts"],
        "task_labels_read": False,
        "test_loaded": False,
    }
    receipt_sha = publish(receipt_path, receipt)
    print(f"manifest_sha256={manifest_sha}")
    print(f"receipt_sha256={receipt_sha}")


if __name__ == "__main__":
    main()
