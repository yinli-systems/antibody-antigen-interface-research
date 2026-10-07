#!/usr/bin/env python3
"""Audit only authority-train embedding artifacts needed by strict retrieval."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID = set("ACDEFGHIKLMNPQRSTVWY")
FORMAT = "strict_partner_retrieval_source_bank_receipt_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seqkey(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def artifact_path(root: Path, key: str) -> Path:
    return root / key[:2] / f"{key}.pt"


def normalize(value: Any, *, allow_empty: bool = False) -> str:
    sequence = "".join(str(value or "").split()).upper()
    if not sequence and not allow_empty:
        raise ValueError("empty sequence")
    if any(char not in VALID for char in sequence):
        raise ValueError("noncanonical sequence")
    return sequence


def canonical_manifest(path: Path) -> dict[str, set[str]]:
    result = {"vh": set(), "vl": set(), "antigen": set()}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["INSTANCE", "vh_seq", "vl_seq", "ag_seq"]:
            raise ValueError(f"exact four-column manifest required, got {reader.fieldnames}")
        seen = set()
        for row in reader:
            instance = str(row["INSTANCE"])
            if instance in seen:
                raise ValueError(f"duplicate instance {instance}")
            seen.add(instance)
            vh = normalize(row["vh_seq"])
            vl = normalize(row["vl_seq"], allow_empty=True)
            antigen = normalize(row["ag_seq"])
            result["vh"].add(vh)
            if vl:
                result["vl"].add(vl)
            result["antigen"].add(antigen)
    return result


def merge_chunks(torch, artifact: dict[str, Any], length: int, dim: int) -> None:
    starts = artifact.get("starts")
    ends = artifact.get("ends")
    chunks = artifact.get("chunks")
    if not isinstance(starts, list) or not isinstance(ends, list) or not isinstance(chunks, list):
        raise ValueError("chunk metadata is missing")
    if not starts or not (len(starts) == len(ends) == len(chunks)):
        raise ValueError("chunk metadata length mismatch")
    count = torch.zeros(length, dtype=torch.int16)
    for start, end, chunk in zip(starts, ends, chunks, strict=True):
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > length:
            raise ValueError("invalid chunk boundary")
        tensor = chunk.detach().cpu()
        if tuple(tensor.shape) != (end - start, dim):
            raise ValueError("chunk tensor shape mismatch")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("nonfinite chunk tensor")
        count[start:end] += 1
    if bool((count == 0).any()):
        raise ValueError("incomplete residue coverage")


def audit_antiberty(torch, path: Path, key: str, sequence: str, revision: str) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("format_version") != "pretrained_representation_v1":
        raise ValueError("AntiBERTy artifact format")
    if artifact.get("sequence_sha256") != key:
        raise ValueError("AntiBERTy artifact key")
    if int(artifact.get("sequence_length", -1)) != len(sequence):
        raise ValueError("AntiBERTy sequence length")
    if artifact.get("model_name") != "antiberty" or artifact.get("model_revision") != revision:
        raise ValueError("AntiBERTy model identity")
    if int(artifact.get("representation_dim", -1)) != 512:
        raise ValueError("AntiBERTy dimension")
    merge_chunks(torch, artifact, len(sequence), 512)
    return {
        "bank": "antiberty",
        "sequence_sha256": key,
        "length": len(sequence),
        "dim": 512,
        "file_sha256": sha256(path),
        "relative_path": str(path.relative_to(path.parents[1])),
    }


def audit_esm2(
    torch,
    path: Path,
    key: str,
    sequence: str,
    model_name: str,
    weight_sha256: str,
) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("format_version") != "sppider_seq_embedding_v1":
        raise ValueError("ESM-2 artifact format")
    if artifact.get("sequence_sha256") != key:
        raise ValueError("ESM-2 artifact key")
    if int(artifact.get("sequence_length", -1)) != len(sequence):
        raise ValueError("ESM-2 sequence length")
    if artifact.get("model_name") != model_name:
        raise ValueError("ESM-2 model identity")
    if artifact.get("model_weight_sha256") != weight_sha256:
        raise ValueError("ESM-2 weight identity")
    chunks = artifact.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("ESM-2 chunks absent")
    dim = int(chunks[0].shape[-1])
    if dim != 1280:
        raise ValueError("ESM-2 dimension")
    merge_chunks(torch, artifact, len(sequence), dim)
    return {
        "bank": "esm2",
        "sequence_sha256": key,
        "length": len(sequence),
        "dim": dim,
        "file_sha256": sha256(path),
        "relative_path": str(path.relative_to(path.parents[1])),
        "model_weight_sha256": weight_sha256,
    }


def entries_sha256(entries: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--antiberty-root", type=Path, required=True)
    parser.add_argument("--antiberty-extraction-receipt", type=Path, required=True)
    parser.add_argument("--antiberty-extraction-receipt-sha256", required=True)
    parser.add_argument("--antiberty-revision", default="antiberty-pypi")
    parser.add_argument("--antiberty-model-weights-sha256", required=True)
    parser.add_argument("--esm2-root", type=Path, required=True)
    parser.add_argument("--esm2-model-name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--esm2-model-weight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.canonical_manifest) != args.canonical_manifest_sha256:
        raise ValueError("canonical manifest hash mismatch")
    if sha256(args.antiberty_extraction_receipt) != args.antiberty_extraction_receipt_sha256:
        raise ValueError("AntiBERTy extraction receipt hash mismatch")
    receipt = json.loads(args.antiberty_extraction_receipt.read_text(encoding="utf-8"))
    if receipt.get("format_version") != "pretrained_representation_receipt_v1":
        raise ValueError("AntiBERTy extraction receipt format")
    if receipt.get("backend") != "antiberty" or receipt.get("revision") != args.antiberty_revision:
        raise ValueError("AntiBERTy extraction receipt identity")
    if receipt.get("complete") is not True or receipt.get("failed") != []:
        raise ValueError("AntiBERTy extraction receipt is incomplete")
    if "No label, contact, prediction, or score was read" not in str(receipt.get("claim_boundary")):
        raise ValueError("AntiBERTy extraction receipt lacks label-free boundary")

    sequences = canonical_manifest(args.canonical_manifest)
    import torch

    entries: list[dict[str, Any]] = []
    for role in ("vh", "vl"):
        for sequence in sorted(sequences[role]):
            key = seqkey(sequence)
            path = artifact_path(args.antiberty_root, key)
            if not path.is_file():
                raise FileNotFoundError(path)
            entry = audit_antiberty(torch, path, key, sequence, args.antiberty_revision)
            entry["authority_role"] = role
            entries.append(entry)
    for sequence in sorted(sequences["antigen"]):
        key = seqkey(sequence)
        path = artifact_path(args.esm2_root, key)
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = audit_esm2(
            torch,
            path,
            key,
            sequence,
            args.esm2_model_name,
            args.esm2_model_weight_sha256,
        )
        entry["authority_role"] = "antigen"
        entries.append(entry)
    entries.sort(key=lambda item: (item["bank"], item["sequence_sha256"], item["authority_role"]))

    payload = {
        "format_version": FORMAT,
        "status": "PASS_AUTHORITY_TRAIN_SCOPED_BANK_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "authority_train_only",
        "canonical_manifest": str(args.canonical_manifest.resolve()),
        "canonical_manifest_sha256": sha256(args.canonical_manifest),
        "antiberty": {
            "root": str(args.antiberty_root.resolve()),
            "required_vh_count": len(sequences["vh"]),
            "required_vl_count": len(sequences["vl"]),
            "model_name": "antiberty",
            "model_revision": args.antiberty_revision,
            "representation_dim": 512,
            "model_weights_sha256_contract": args.antiberty_model_weights_sha256,
            "model_weight_bytes_verified_by_artifacts": False,
            "extraction_receipt": str(args.antiberty_extraction_receipt.resolve()),
            "extraction_receipt_sha256": sha256(args.antiberty_extraction_receipt),
            "extraction_receipt_note": "The historical unsupervised extraction receipt parsed sealed split sequence membership, but this audit opens only authority-train keys and no sealed artifact.",
        },
        "esm2": {
            "root": str(args.esm2_root.resolve()),
            "required_antigen_count": len(sequences["antigen"]),
            "model_name": args.esm2_model_name,
            "model_weight_sha256": args.esm2_model_weight_sha256,
            "representation_dim": 1280,
            "model_weight_bytes_verified_by_each_artifact": True,
        },
        "entry_count": len(entries),
        "entries_sha256": entries_sha256(entries),
        "entries": entries,
        "whole_bank_enumerated": False,
        "only_authority_train_keys_opened": True,
        "model_inference_performed": False,
        "contact_labels_read": False,
        "affinity_labels_read": False,
        "test_access": False,
        "ood_access": False,
        "training_authorized": False,
        "outer_evaluation_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "entries": len(entries),
        "entries_sha256": payload["entries_sha256"],
        "counts": {
            "vh": len(sequences["vh"]),
            "vl": len(sequences["vl"]),
            "antigen": len(sequences["antigen"]),
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
