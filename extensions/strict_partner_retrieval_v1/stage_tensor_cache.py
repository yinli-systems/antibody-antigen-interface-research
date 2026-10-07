#!/usr/bin/env python3
"""Stage only audited, allowlisted residue embeddings into one fold-local root."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from cdr import annotate_antibody

OUT_FORMAT = "strict_partner_retrieval_embedding_v1"
BANK_FORMAT = "strict_partner_retrieval_source_bank_receipt_v1"


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def save_new(torch, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def merge_chunks(
    torch,
    artifact: dict[str, Any],
    expected_length: int,
    expected_dim: int,
) -> "torch.Tensor":
    starts = artifact.get("starts")
    ends = artifact.get("ends")
    chunks = artifact.get("chunks")
    if not isinstance(starts, list) or not isinstance(ends, list) or not isinstance(chunks, list):
        raise ValueError("chunk metadata missing")
    if not starts or not (len(starts) == len(ends) == len(chunks)):
        raise ValueError("chunk metadata length mismatch")
    output = torch.zeros((expected_length, expected_dim), dtype=torch.float32)
    count = torch.zeros((expected_length, 1), dtype=torch.float32)
    for start, end, chunk in zip(starts, ends, chunks, strict=True):
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > expected_length:
            raise ValueError("invalid chunk boundary")
        tensor = chunk.detach().cpu().float()
        if tuple(tensor.shape) != (end - start, expected_dim):
            raise ValueError("chunk tensor shape mismatch")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("nonfinite source tensor")
        output[start:end] += tensor
        count[start:end] += 1
    if bool((count == 0).any()):
        raise ValueError("incomplete residue coverage")
    merged = (output / count).contiguous()
    if not bool(torch.isfinite(merged).all()):
        raise ValueError("nonfinite merged tensor")
    return merged


def bank_index(receipt: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    entries = receipt.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source-bank receipt has no entries")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("source-bank entry is not an object")
        bank = entry.get("bank")
        key = entry.get("sequence_sha256")
        file_sha = entry.get("file_sha256")
        if bank not in ("antiberty", "esm2") or not isinstance(key, str) or len(key) != 64:
            raise ValueError("source-bank entry identity")
        if not isinstance(file_sha, str) or len(file_sha) != 64:
            raise ValueError("source-bank file digest")
        identity = (bank, key)
        if identity in result and result[identity]["file_sha256"] != file_sha:
            raise ValueError("conflicting source-bank duplicate")
        result[identity] = entry
    return result


def require_audited_source(
    index: dict[tuple[str, str], dict[str, Any]],
    bank: str,
    key: str,
    path: Path,
) -> dict[str, Any]:
    entry = index.get((bank, key))
    if entry is None:
        raise PermissionError(f"source artifact is outside scoped bank audit: {bank}:{key}")
    actual = sha256(path)
    if actual != entry["file_sha256"]:
        raise ValueError(f"source artifact digest changed: {path}")
    return entry


def load_antiberty_chain(
    torch,
    path: Path,
    key: str,
    sequence: str,
    expected_revision: str,
) -> tuple["torch.Tensor", dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("format_version") != "pretrained_representation_v1":
        raise ValueError("AntiBERTy source format")
    if artifact.get("sequence_sha256") != key:
        raise ValueError("AntiBERTy sequence key")
    if int(artifact.get("sequence_length", -1)) != len(sequence):
        raise ValueError("AntiBERTy sequence length")
    if artifact.get("model_name") != "antiberty":
        raise ValueError("AntiBERTy model name")
    if artifact.get("model_revision") != expected_revision:
        raise ValueError("AntiBERTy model revision")
    if int(artifact.get("representation_dim", -1)) != 512:
        raise ValueError("AntiBERTy dimension")
    tensor = merge_chunks(torch, artifact, len(sequence), 512)
    return tensor, {
        "format_version": artifact["format_version"],
        "model_name": artifact["model_name"],
        "model_revision": artifact["model_revision"],
        "sequence_sha256": key,
        "dim": 512,
    }


def load_antibody(
    torch,
    root: Path,
    index: dict[tuple[str, str], dict[str, Any]],
    vh: str,
    vl: str,
    expected_revision: str,
    expected_weight_contract: str,
) -> tuple["torch.Tensor", dict[str, Any]]:
    tensors = []
    chains = []
    for role, sequence in (("vh", vh), ("vl", vl)):
        if not sequence:
            continue
        key = seqkey(sequence)
        path = artifact_path(root, key)
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = require_audited_source(index, "antiberty", key, path)
        if entry.get("length") != len(sequence) or entry.get("dim") != 512:
            raise ValueError("AntiBERTy bank-entry metadata")
        tensor, metadata = load_antiberty_chain(
            torch, path, key, sequence, expected_revision
        )
        tensors.append(tensor)
        chains.append(
            {
                "role": role,
                "source_path": str(path),
                "source_file_sha256": entry["file_sha256"],
                **metadata,
            }
        )
    if not tensors:
        raise ValueError("antibody has no nonempty chain")
    return torch.cat(tensors, dim=0).contiguous(), {
        "source_format": "pretrained_representation_v1_chainwise_concat",
        "model_name": "AntiBERTy_md_smooth",
        "artifact_model_name": "antiberty",
        "model_revision": expected_revision,
        "model_weights_sha256_contract": expected_weight_contract,
        "model_weight_bytes_verified_by_artifacts": False,
        "dim": 512,
        "chains": chains,
    }


def load_antigen(
    torch,
    path: Path,
    index: dict[tuple[str, str], dict[str, Any]],
    key: str,
    sequence: str,
    expected_model_name: str,
    expected_weight_sha256: str,
) -> tuple["torch.Tensor", dict[str, Any]]:
    entry = require_audited_source(index, "esm2", key, path)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("format_version") != "sppider_seq_embedding_v1":
        raise ValueError("ESM-2 source format")
    if artifact.get("sequence_sha256") != key:
        raise ValueError("ESM-2 sequence key")
    if int(artifact.get("sequence_length", -1)) != len(sequence):
        raise ValueError("ESM-2 sequence length")
    if artifact.get("model_name") != expected_model_name:
        raise ValueError("ESM-2 model name")
    if artifact.get("model_weight_sha256") != expected_weight_sha256:
        raise ValueError("ESM-2 model weight")
    chunks = artifact.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("ESM-2 chunks absent")
    dim = int(chunks[0].shape[-1])
    if dim != 1280 or entry.get("dim") != 1280 or entry.get("length") != len(sequence):
        raise ValueError("ESM-2 bank-entry metadata")
    tensor = merge_chunks(torch, artifact, len(sequence), dim)
    return tensor, {
        "source_format": artifact["format_version"],
        "model_name": artifact["model_name"],
        "model_weight_sha256": artifact["model_weight_sha256"],
        "source_path": str(path),
        "source_file_sha256": entry["file_sha256"],
        "dim": dim,
    }


def validate_bank_receipt(
    receipt: dict[str, Any],
    cache: dict[str, Any],
    antibody_root: Path,
    antigen_root: Path,
    expected_antiberty_revision: str,
    expected_antiberty_weight: str,
    expected_antigen_name: str,
    expected_antigen_weight: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    if receipt.get("format_version") != BANK_FORMAT:
        raise ValueError("source-bank receipt format")
    if receipt.get("status") != "PASS_AUTHORITY_TRAIN_SCOPED_BANK_AUDIT":
        raise ValueError("source-bank audit did not pass")
    if receipt.get("scope") != "authority_train_only":
        raise ValueError("source-bank scope")
    if receipt.get("only_authority_train_keys_opened") is not True:
        raise ValueError("source-bank audit opened a broader authority")
    for key in (
        "model_inference_performed",
        "contact_labels_read",
        "affinity_labels_read",
        "test_access",
        "ood_access",
        "training_authorized",
        "outer_evaluation_authorized",
    ):
        if receipt.get(key) is not False:
            raise ValueError(f"source-bank receipt unexpectedly enables {key}")
    if receipt.get("canonical_manifest_sha256") != cache.get("canonical_manifest_sha256"):
        raise ValueError("source-bank/canonical-manifest mismatch")
    anti = receipt.get("antiberty", {})
    esm = receipt.get("esm2", {})
    if Path(anti.get("root", "")).resolve() != antibody_root.resolve():
        raise ValueError("AntiBERTy root differs from bank audit")
    if Path(esm.get("root", "")).resolve() != antigen_root.resolve():
        raise ValueError("ESM-2 root differs from bank audit")
    if anti.get("model_revision") != expected_antiberty_revision:
        raise ValueError("AntiBERTy revision differs from bank audit")
    if anti.get("model_weights_sha256_contract") != expected_antiberty_weight:
        raise ValueError("AntiBERTy weight contract differs from bank audit")
    if esm.get("model_name") != expected_antigen_name:
        raise ValueError("ESM-2 model differs from bank audit")
    if esm.get("model_weight_sha256") != expected_antigen_weight:
        raise ValueError("ESM-2 weight differs from bank audit")
    return bank_index(receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--antibody-root", type=Path, required=True)
    parser.add_argument("--antigen-root", type=Path, required=True)
    parser.add_argument("--expected-antiberty-revision", default="antiberty-pypi")
    parser.add_argument("--expected-antiberty-weights-sha256", required=True)
    parser.add_argument(
        "--expected-antigen-model-name", default="facebook/esm2_t33_650M_UR50D"
    )
    parser.add_argument("--expected-antigen-model-weight-sha256", required=True)
    parser.add_argument("--source-bank-receipt", type=Path, required=True)
    parser.add_argument("--source-bank-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if sha256(args.source_bank_receipt) != args.source_bank_receipt_sha256:
        raise ValueError("source-bank receipt hash")
    cache = load_json(args.sequence_cache)
    role = cache.get("role")
    if role not in ("training_only", "evaluation_only"):
        raise ValueError("sequence-cache role")
    if cache.get("test_access") is not False or cache.get("ood_access") is not False:
        raise PermissionError("sequence cache has sealed access")
    receipt = load_json(args.source_bank_receipt)
    index = validate_bank_receipt(
        receipt,
        cache,
        args.antibody_root,
        args.antigen_root,
        args.expected_antiberty_revision,
        args.expected_antiberty_weights_sha256,
        args.expected_antigen_model_name,
        args.expected_antigen_model_weight_sha256,
    )

    import torch

    args.output_root.mkdir(parents=True)
    entries = []
    for entity, data in sorted(cache["antibodies"].items()):
        vh = data["vh"]
        vl = data.get("vl", "")
        tensor, metadata = load_antibody(
            torch,
            args.antibody_root,
            index,
            vh,
            vl,
            args.expected_antiberty_revision,
            args.expected_antiberty_weights_sha256,
        )
        cdr = annotate_antibody(vh, vl)
        if len(cdr["cdr_masks"]) != len(tensor):
            raise ValueError("CDR mask/tensor length mismatch")
        destination = args.output_root / "antibody" / entity[:2] / f"{entity}.pt"
        payload = {
            "format_version": OUT_FORMAT,
            "entity_id": entity,
            "side": "antibody",
            "role": role,
            "embedding": tensor,
            "mask": torch.ones(len(tensor), dtype=torch.bool),
            "vh_length": len(vh),
            "vl_length": len(vl),
            "cdr_masks": torch.tensor(cdr["cdr_masks"], dtype=torch.bool),
            "source_metadata": metadata,
            "cdr_receipt": cdr,
        }
        save_new(torch, destination, payload)
        entries.append(
            {
                "entity_id": entity,
                "side": "antibody",
                "relative_path": str(destination.relative_to(args.output_root)),
                "sha256": sha256(destination),
                "length": len(tensor),
                "dim": int(tensor.shape[1]),
            }
        )
    for entity, sequence in sorted(cache["antigens"].items()):
        key = seqkey(sequence)
        source = artifact_path(args.antigen_root, key)
        if not source.is_file():
            raise FileNotFoundError(source)
        tensor, metadata = load_antigen(
            torch,
            source,
            index,
            key,
            sequence,
            args.expected_antigen_model_name,
            args.expected_antigen_model_weight_sha256,
        )
        destination = args.output_root / "antigen" / entity[:2] / f"{entity}.pt"
        payload = {
            "format_version": OUT_FORMAT,
            "entity_id": entity,
            "side": "antigen",
            "role": role,
            "embedding": tensor,
            "mask": torch.ones(len(tensor), dtype=torch.bool),
            "source_metadata": metadata,
        }
        save_new(torch, destination, payload)
        entries.append(
            {
                "entity_id": entity,
                "side": "antigen",
                "relative_path": str(destination.relative_to(args.output_root)),
                "sha256": sha256(destination),
                "length": len(tensor),
                "dim": int(tensor.shape[1]),
            }
        )
    receipt_payload = {
        "format_version": "strict_partner_retrieval_tensor_cache_receipt_v1",
        "status": "COMPLETE_FOLD_LOCAL_TENSOR_CACHE",
        "role": role,
        "fold": cache["fold"],
        "sequence_cache_sha256": sha256(args.sequence_cache),
        "source_bank_receipt_sha256": sha256(args.source_bank_receipt),
        "source_bank_entries_sha256": receipt["entries_sha256"],
        "entry_count": len(entries),
        "entries": entries,
        "antibody_count": len(cache["antibodies"]),
        "antigen_count": len(cache["antigens"]),
        "training_and_model_selection_authorized": cache.get(
            "training_and_model_selection_authorized"
        )
        is True,
        "outer_evaluation_authorized": cache.get("outer_evaluation_authorized")
        is True,
        "whole_pool_bank_visible_to_model_process": False,
        "test_access": False,
        "ood_access": False,
    }
    with (args.output_root / "CACHE_RECEIPT.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt_payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": receipt_payload["status"],
                "fold": receipt_payload["fold"],
                "role": role,
                "entries": len(entries),
                "training_authorized": receipt_payload[
                    "training_and_model_selection_authorized"
                ],
                "outer_evaluation_authorized": receipt_payload[
                    "outer_evaluation_authorized"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
