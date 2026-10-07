#!/usr/bin/env python3
"""Materialize ABF-30B antibody plus unchanged ESM-2 antigen SABLE embeddings."""

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

from antibody_pretraining.model import AntibodyMaskedLanguageModel, config_from_preset, load_model_family
from antibody_pretraining.tokenizer import encode_unpaired, pad_encoded


FINAL_MODEL_SHA256 = "c36e0b1d812f97317109f1f8ab085a349beb2a782a6f4092b3074bfc1eaab3ea"
MODEL_FAMILY_SHA256 = "1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015"
CONTROL_ESM_MANIFEST_SHA256 = "67670200608ac98573c9d0329fd06fb7f5973d0a47bd96aeea5861b94955abc9"
STRUCTURE_MANIFEST_SHA256 = "795223b2ebd02d6ce37cebf7dadafe5b1f50b9b212976ba15e36af3e62f23259"
EMBEDDING_FORMAT = "sppider_seq_embedding_v1"
ABF_MODEL_NAME = "abf_650m_oas30b_final"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def require_sha(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority SHA differs: {path}")


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
                visit(str(key)); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item: visit(child)
        elif isinstance(item, type):
            digest.update(b"type\0" + f"{item.__module__}.{item.__qualname__}".encode("ascii") + b"\0")
        elif isinstance(item, enum.Enum):
            digest.update(b"enum\0" + f"{item.__class__.__module__}.{item.__class__.__qualname__}".encode("ascii") + b"\0")
            visit(item.name); visit(item.value)
        elif isinstance(item, (torch.dtype, torch.device)):
            digest.update(b"torch_identity\0" + str(item).encode("ascii") + b"\0")
        else:
            digest.update(json.dumps(item, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\0")

    visit(value)
    return digest.hexdigest()


def restore_model(model: AntibodyMaskedLanguageModel, checkpoint: Path) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
    state = get_model_state_dict(model, options=options)
    dcp.load({"model": state}, checkpoint_id=checkpoint.resolve(), no_dist=True)
    incompatible = set_model_state_dict(model, state, options=options)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"DCP restore returned incompatible keys: {incompatible}")


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--chain-map", type=Path, required=True)
    parser.add_argument("--chain-map-sha256", required=True)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--canary-sha256", required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--control-embedding-root", type=Path, required=True)
    parser.add_argument("--model-family", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-antibodies", type=int, default=8)
    args = parser.parse_args()
    control_manifest_path = args.control_embedding_root / "SABLE_ESM_EMBEDDING_MANIFEST.json"
    for path, expected in (
        (args.protocol, args.protocol_sha256),
        (args.chain_map, args.chain_map_sha256),
        (args.canary, args.canary_sha256),
        (args.structure_manifest, STRUCTURE_MANIFEST_SHA256),
        (control_manifest_path, CONTROL_ESM_MANIFEST_SHA256),
        (args.model_family, MODEL_FAMILY_SHA256),
    ):
        require_sha(path, expected)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    protocol = json.loads(args.protocol.read_text())
    chain_map = json.loads(args.chain_map.read_text())
    canary = json.loads(args.canary.read_text())
    structures = json.loads(args.structure_manifest.read_text())
    control = json.loads(control_manifest_path.read_text())
    if (
        protocol.get("status")
        != "FROZEN_BEFORE_SABLE_LABEL_FREE_CHAIN_TYPING_AND_MIXED_EMBEDDINGS"
        or chain_map.get("status") != "PASS_LABEL_FREE_SABLE_ANARCII_CHAIN_TYPING"
        or canary.get("status") != "PASS_DCP_EXACT_RESTORE_AND_RESIDUE_EMBEDDING_CANARY"
        or canary.get("checkpoint_model_sha256") != FINAL_MODEL_SHA256
        or structures.get("status") != "PASS_LABEL_FREE_SEQUENCE_PREDICTED_STRUCTURE_CACHE"
        or control.get("status") != "PASS_LABEL_FREE_ESM2_EMBEDDINGS"
        or any(value.get("task_labels_read") is not False for value in (protocol, chain_map))
    ):
        raise ValueError("SABLE label-free embedding authority differs")

    antibodies = {row["antibody_sequence_sha256"]: row for row in chain_map["records"]}
    unique_chains = {row["sequence_sha256"]: row for row in chain_map["unique_chains"]}
    antigen_shas = {str(row["antigen_sequence_sha256"]) for row in structures["records"]}
    if set(antibodies) & antigen_shas:
        raise ValueError("SABLE antibody and antigen sequence sets overlap")
    control_by_sha = {str(row["sequence_sha256"]): row for row in control["records"]}
    if set(control_by_sha) != set(antibodies) | antigen_shas:
        raise ValueError("control embedding sequence cohort differs")

    family = load_model_family(args.model_family)
    model = AntibodyMaskedLanguageModel(config_from_preset(family, "abf_650m"))
    restore_model(model, args.checkpoint)
    if tree_digest(model.state_dict()) != FINAL_MODEL_SHA256:
        raise RuntimeError("restored model digest differs")
    device = torch.device("cuda")
    model.to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad = False
    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], value: torch.Tensor) -> None:
        captured.append(value.detach())

    handle = model.final_norm.register_forward_hook(hook)
    args.output_root.mkdir(parents=True)
    records_by_sha: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    ordered = sorted(antibodies)
    torch.manual_seed(20_260_830); torch.cuda.manual_seed_all(20_260_830)
    torch.use_deterministic_algorithms(True, warn_only=True)
    for offset in range(0, len(ordered), args.batch_antibodies):
        batch_shas = ordered[offset : offset + args.batch_antibodies]
        encoded = []
        ownership: list[tuple[str, dict[str, Any], str]] = []
        for antibody_sha in batch_shas:
            for spec in antibodies[antibody_sha]["chains"]:
                chain = unique_chains[spec["sequence_sha256"]]
                encoded.append(encode_unpaired(chain["sequence"], spec["abf_chain_type"]))
                ownership.append((antibody_sha, spec, chain["sequence"]))
        input_ids, type_ids, masks = pad_encoded(encoded)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(
                torch.tensor(input_ids, dtype=torch.long, device=device),
                torch.tensor(type_ids, dtype=torch.long, device=device),
                attention_mask=torch.tensor(masks, dtype=torch.bool, device=device),
            )
        hidden = captured.pop().float().cpu()
        grouped: dict[str, list[tuple[dict[str, Any], torch.Tensor]]] = {key: [] for key in batch_shas}
        for index, (antibody_sha, spec, sequence) in enumerate(ownership):
            tensor = hidden[index, 1 : 1 + len(sequence)].contiguous()
            if tuple(tensor.shape) != (len(sequence), 1280) or not bool(torch.isfinite(tensor).all()):
                raise RuntimeError("ABF SABLE residue embedding differs")
            grouped[antibody_sha].append((spec, tensor))
        for antibody_sha in batch_shas:
            starts, ends, chunks, cursor = [], [], [], 0
            for spec, tensor in grouped[antibody_sha]:
                starts.append(cursor); cursor += len(tensor); ends.append(cursor); chunks.append(tensor)
            if cursor != int(antibodies[antibody_sha]["antibody_sequence_length"]):
                raise RuntimeError("ABF SABLE antibody axis differs")
            artifact = {
                "format_version": EMBEDDING_FORMAT,
                "sequence_sha256": antibody_sha,
                "sequence_length": cursor,
                "model_name": ABF_MODEL_NAME,
                "model_revision": FINAL_MODEL_SHA256,
                "model_weight_sha256": FINAL_MODEL_SHA256,
                "max_tokens": 194,
                "stride": 192,
                "starts": starts,
                "ends": ends,
                "chunks": chunks,
                "embedding_definition": "final RMS-normalized residue hidden state",
                "chain_types": [spec["abf_chain_type"] for spec, _ in grouped[antibody_sha]],
                "chain_sequence_sha256s": [spec["sequence_sha256"] for spec, _ in grouped[antibody_sha]],
            }
            relative = Path(control_by_sha[antibody_sha]["artifact_path"])
            destination = args.output_root / relative
            atomic_torch_save(artifact, destination)
            records_by_sha[antibody_sha] = {
                "sequence_sha256": antibody_sha,
                "sequence_length": cursor,
                "artifact_path": relative.as_posix(),
                "artifact_sha256": sha256_file(destination),
                "state": "written_abf30b_antibody",
                "partner": "antibody",
            }
        print(f"embedded SABLE antibodies {min(offset + len(batch_shas), len(ordered))}/{len(ordered)}", flush=True)
    handle.remove()
    if captured: raise RuntimeError("final-norm capture count differs")

    for antigen_sha in sorted(antigen_shas):
        source_row = control_by_sha[antigen_sha]
        source = args.control_embedding_root / source_row["artifact_path"]
        require_sha(source, source_row["artifact_sha256"])
        relative = Path(source_row["artifact_path"])
        destination = args.output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
        if sha256_file(destination) != source_row["artifact_sha256"]:
            raise RuntimeError("hardlinked SABLE antigen embedding differs")
        records_by_sha[antigen_sha] = {
            **source_row,
            "state": "hardlinked_unchanged_esm2_antigen",
            "partner": "antigen",
        }
    if set(records_by_sha) != set(control_by_sha):
        raise RuntimeError("mixed SABLE embedding record cohort differs")

    manifest = {
        "format_version": "abf30b_sable_mixed_partner_embedding_manifest_v1",
        "status": "PASS_LABEL_FREE_SABLE_MIXED_PARTNER_EMBEDDINGS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "entry_count": 97,
        "unique_pdb_id_count": 88,
        "unique_sequence_count": len(records_by_sha),
        "embedding_dimension": 1280,
        "records": [records_by_sha[key] for key in sorted(records_by_sha)],
        "entry_records": control["entry_records"],
        "antibody_model": {
            "name": ABF_MODEL_NAME,
            "checkpoint_model_sha256": FINAL_MODEL_SHA256,
            "embedding_definition": "final RMS-normalized residue hidden state",
        },
        "antigen_model": {
            "name": control["model_name"],
            "revision": control["model_revision"],
            "weight_sha256": control["model_weight_sha256"],
            "control_manifest_sha256": CONTROL_ESM_MANIFEST_SHA256,
        },
        "authority": {
            "protocol_sha256": args.protocol_sha256,
            "chain_map_sha256": args.chain_map_sha256,
            "dcp_canary_sha256": args.canary_sha256,
            "predicted_structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
            "task_labels_read": False,
            "bound_label_artifacts_read": False,
            "performance_read": False,
            "selection_or_tuning": False,
            "old_sabdab2_test_loaded": False,
            "temporal_holdout_loaded": False,
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "elapsed_seconds": time.monotonic() - started,
        },
    }
    manifest_path = args.output_root / "SABLE_ESM_EMBEDDING_MANIFEST.json"
    manifest_sha = publish(manifest_path, manifest)
    receipt = {
        "format_version": "abf30b_sable_mixed_partner_embedding_receipt_v1",
        "status": manifest["status"],
        "manifest_sha256": manifest_sha,
        "checkpoint_model_sha256": FINAL_MODEL_SHA256,
        "record_count": len(records_by_sha),
        "antibody_count": len(antibodies),
        "antigen_count": len(antigen_shas),
        "task_labels_read": False,
        "bound_labels_read": False,
        "test_loaded": False,
    }
    receipt_sha = publish(args.output_root / "SABLE_ABF30B_MIXED_EMBEDDING_RECEIPT.json", receipt)
    print(json.dumps({"manifest_sha256": manifest_sha, "receipt_sha256": receipt_sha}))


if __name__ == "__main__":
    main()
