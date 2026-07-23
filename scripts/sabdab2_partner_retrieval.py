#!/usr/bin/env python3
"""Train and evaluate a leakage-controlled SAbDab2 partner retriever."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FORMAT_VERSION = "sabdab2_partner_retrieval_v1"
SPLIT_FORMAT_VERSION = "sabdab2_interaction_first_v1"
HEX_DIGEST_LENGTH = 64


@dataclass(frozen=True)
class PairRecord:
    """One observed antibody-antigen pair represented by sequence digests."""

    instance_id: str
    antibody_key: str
    antigen_key: str
    antibody_length: int
    antigen_length: int


@dataclass(frozen=True)
class SequenceRecord:
    """One unique sequence and its validated residue length."""

    key: str
    length: int


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping with stable serialization."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_config_integrity(config: Mapping[str, Any]) -> None:
    """Reject a checkpoint whose canonical configuration hash is inconsistent."""
    stored = config.get("config_sha256")
    if not isinstance(stored, str) or not _is_sha256(stored):
        raise ValueError("Checkpoint configuration has no valid canonical hash")
    unhashed = {key: value for key, value in config.items() if key != "config_sha256"}
    observed = canonical_sha256(unhashed)
    if observed != stored:
        raise ValueError(
            f"Checkpoint configuration hash mismatch: expected={stored}, observed={observed}"
        )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict JSON through an adjacent temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _is_sha256(value: str) -> bool:
    return len(value) == HEX_DIGEST_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _pair_set(records: Iterable[PairRecord]) -> set[tuple[str, str]]:
    return {(record.antibody_key, record.antigen_key) for record in records}


def deduplicate_exact_pairs(
    records: Sequence[PairRecord],
) -> tuple[list[PairRecord], int]:
    """Retain one deterministic representative per exact sequence pair."""
    representatives: dict[tuple[str, str], PairRecord] = {}
    for record in sorted(records, key=lambda value: value.instance_id):
        key = (record.antibody_key, record.antigen_key)
        representatives.setdefault(key, record)
    deduplicated = sorted(
        representatives.values(),
        key=lambda value: (value.antibody_key, value.antigen_key, value.instance_id),
    )
    return deduplicated, len(records) - len(deduplicated)


def _records_from_frame(frame: Any) -> list[PairRecord]:
    records = []
    for row in frame.itertuples(index=False):
        record = PairRecord(
            instance_id=str(row.instance_id),
            antibody_key=str(row.ab_sequence_sha256),
            antigen_key=str(row.ag_sequence_sha256),
            antibody_length=int(row.ab_length),
            antigen_length=int(row.ag_length),
        )
        if not record.instance_id:
            raise ValueError("Prepared instances contain an empty instance ID")
        if not _is_sha256(record.antibody_key) or not _is_sha256(record.antigen_key):
            raise ValueError(f"Invalid sequence digest for {record.instance_id}")
        if record.antibody_length <= 0 or record.antigen_length <= 0:
            raise ValueError(f"Invalid sequence length for {record.instance_id}")
        records.append(record)
    return records


def load_dataset(
    prepare_dir: Path,
    split_manifest: Path,
) -> tuple[dict[str, list[PairRecord]], dict[str, Any]]:
    """Load prepared metadata and a fixed development-only v1 split."""
    import pandas as pd

    prepare_dir = prepare_dir.expanduser().resolve()
    instances_path = prepare_dir / "instances.parquet"
    split_manifest = split_manifest.expanduser().resolve()
    if not instances_path.is_file():
        raise FileNotFoundError(f"Prepared instances are missing: {instances_path}")
    if not split_manifest.is_file():
        raise FileNotFoundError(f"Split manifest is missing: {split_manifest}")

    columns = [
        "instance_id",
        "ab_sequence_sha256",
        "ag_sequence_sha256",
        "ab_length",
        "ag_length",
    ]
    frame = pd.read_parquet(instances_path, columns=columns)
    if frame.empty or frame["instance_id"].isna().any():
        raise ValueError("Prepared instances cannot be empty or contain null IDs")
    frame["instance_id"] = frame["instance_id"].astype(str)
    if frame["instance_id"].duplicated().any():
        raise ValueError("Prepared instances contain duplicate instance IDs")

    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    if payload.get("format_version") != SPLIT_FORMAT_VERSION:
        raise ValueError("Split manifest is not a compatible interaction-first v1 split")
    if payload.get("intended_use") != "development_only":
        raise ValueError("Partner training requires a fixed development-only split")
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("Split manifest does not contain split lists")

    identifiers = set(frame["instance_id"])
    assignment: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        split_ids = [str(value) for value in raw_splits.get(split_name, [])]
        if not split_ids or len(split_ids) != len(set(split_ids)):
            raise ValueError(f"Split {split_name!r} must contain unique instance IDs")
        for instance_id in split_ids:
            if instance_id in assignment:
                raise ValueError(f"Instance {instance_id} occurs in multiple splits")
            assignment[instance_id] = split_name
    if set(assignment) != identifiers:
        missing = sorted(identifiers - set(assignment))[:5]
        extra = sorted(set(assignment) - identifiers)[:5]
        raise ValueError(
            f"Prepared/split membership mismatch: missing={missing}, extra={extra}"
        )

    frame["retrieval_split"] = frame["instance_id"].map(assignment)
    split_records = {
        split_name: _records_from_frame(
            frame.loc[frame["retrieval_split"] == split_name, columns]
            .sort_values("instance_id", kind="stable")
        )
        for split_name in ("train", "validation", "test")
    }
    metadata = {
        "instances_path": str(instances_path),
        "instances_sha256": sha256_file(instances_path),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "split_intended_use": payload["intended_use"],
        "raw_instance_counts": {
            name: len(values) for name, values in split_records.items()
        },
    }
    return split_records, metadata


def audit_double_remote_isolation(
    train_records: Sequence[PairRecord],
    holdout_records: Sequence[PairRecord],
    holdout_name: str,
) -> dict[str, int]:
    """Reject exact partner-sequence overlap across a double-remote boundary."""
    train_antibodies = {record.antibody_key for record in train_records}
    train_antigens = {record.antigen_key for record in train_records}
    holdout_antibodies = {record.antibody_key for record in holdout_records}
    holdout_antigens = {record.antigen_key for record in holdout_records}
    audit = {
        "antibody_sequence_overlap": len(train_antibodies & holdout_antibodies),
        "antigen_sequence_overlap": len(train_antigens & holdout_antigens),
        "exact_pair_overlap": len(_pair_set(train_records) & _pair_set(holdout_records)),
    }
    if any(audit.values()):
        raise ValueError(
            f"Train/{holdout_name} violates exact double-remote isolation: {audit}"
        )
    return audit


def embedding_corpus_fingerprint(
    embedding_root: Path,
    records: Sequence[PairRecord],
    scope: str,
) -> dict[str, Any]:
    """Hash every frozen embedding artifact required by a data scope."""
    root = embedding_root.expanduser().resolve()
    sequence_keys = sorted(
        {
            key
            for record in records
            for key in (record.antibody_key, record.antigen_key)
        }
    )
    if not sequence_keys:
        raise ValueError("Cannot fingerprint an empty embedding corpus")
    manifest_digest = hashlib.sha256()
    total_bytes = 0
    for sequence_key in sequence_keys:
        artifact_path = root / sequence_key[:2] / f"{sequence_key}.pt"
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Frozen embedding is missing: {artifact_path}")
        artifact_size = artifact_path.stat().st_size
        artifact_digest = sha256_file(artifact_path)
        manifest_digest.update(
            (
                f"{sequence_key}\t{artifact_size}\t{artifact_digest}\n"
            ).encode("ascii")
        )
        total_bytes += artifact_size
    return {
        "scope": scope,
        "sequence_count": len(sequence_keys),
        "total_bytes": total_bytes,
        "manifest_algorithm": "sha256(sequence_key,size,file_sha256)",
        "content_manifest_sha256": manifest_digest.hexdigest(),
    }


def length_bucket_batches(
    records: Sequence[PairRecord],
    batch_size: int,
    bucket_width: int,
    seed: int,
    epoch: int,
) -> list[list[PairRecord]]:
    """Create deterministic two-dimensional length buckets without dropping rows."""
    if batch_size < 2:
        raise ValueError("batch_size must be at least two for contrastive learning")
    if bucket_width <= 0:
        raise ValueError("bucket_width must be positive")
    if len(records) < 2:
        raise ValueError("At least two pair records are required")

    buckets: dict[tuple[int, int], list[PairRecord]] = defaultdict(list)
    for record in records:
        key = (
            (record.antibody_length - 1) // bucket_width,
            (record.antigen_length - 1) // bucket_width,
        )
        buckets[key].append(record)

    complete: list[list[PairRecord]] = []
    remainders: list[PairRecord] = []
    for bucket_key in sorted(buckets):
        values = sorted(buckets[bucket_key], key=lambda value: value.instance_id)
        derived_seed = int.from_bytes(
            hashlib.sha256(
                f"{seed}:{epoch}:{bucket_key[0]}:{bucket_key[1]}".encode("ascii")
            ).digest()[:8],
            "big",
        )
        random.Random(derived_seed).shuffle(values)
        full_count = len(values) // batch_size
        for index in range(full_count):
            start = index * batch_size
            complete.append(values[start : start + batch_size])
        remainders.extend(values[full_count * batch_size :])

    remainders.sort(
        key=lambda value: (
            value.antigen_length,
            value.antibody_length,
            value.instance_id,
        )
    )
    for start in range(0, len(remainders), batch_size):
        complete.append(remainders[start : start + batch_size])
    if complete and len(complete[-1]) == 1:
        singleton = complete.pop()[0]
        target = min(
            range(len(complete)),
            key=lambda index: (
                abs(
                    max(value.antigen_length for value in complete[index])
                    - singleton.antigen_length
                )
                + abs(
                    max(value.antibody_length for value in complete[index])
                    - singleton.antibody_length
                ),
                len(complete[index]),
                index,
            ),
        )
        complete[target].append(singleton)
    if not complete or any(len(batch) < 2 for batch in complete):
        raise ValueError("Unable to construct non-singleton contrastive batches")

    batch_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{epoch}:batch-order".encode("ascii")).digest()[:8],
        "big",
    )
    random.Random(batch_seed).shuffle(complete)
    observed = [record for batch in complete for record in batch]
    if len(observed) != len(records) or len({id(value) for value in observed}) != len(records):
        raise AssertionError("Length bucketing lost or duplicated a pair record")
    return complete


def sequence_bucket_batches(
    records: Sequence[SequenceRecord],
    batch_size: int,
    bucket_width: int,
) -> list[list[SequenceRecord]]:
    """Group unique sequences by length for vectorized library encoding."""
    if batch_size <= 0 or bucket_width <= 0:
        raise ValueError("Sequence batch settings must be positive")
    buckets: dict[int, list[SequenceRecord]] = defaultdict(list)
    for record in records:
        buckets[(record.length - 1) // bucket_width].append(record)
    batches = []
    for bucket_key in sorted(buckets):
        values = sorted(buckets[bucket_key], key=lambda value: value.key)
        batches.extend(
            values[start : start + batch_size]
            for start in range(0, len(values), batch_size)
        )
    return batches


def unique_sequences(
    records: Sequence[PairRecord], side: str
) -> list[SequenceRecord]:
    """Collapse pair rows into one length-validated sequence library."""
    if side not in {"antibody", "antigen"}:
        raise ValueError("side must be 'antibody' or 'antigen'")
    lengths: dict[str, int] = {}
    for record in records:
        key = record.antibody_key if side == "antibody" else record.antigen_key
        length = record.antibody_length if side == "antibody" else record.antigen_length
        if key in lengths and lengths[key] != length:
            raise ValueError(f"Inconsistent length for sequence {key}")
        lengths[key] = length
    return [SequenceRecord(key, lengths[key]) for key in sorted(lengths)]


def build_positive_mask(
    antibody_keys: Sequence[str],
    antigen_keys: Sequence[str],
    known_positive_pairs: set[tuple[str, str]],
    device: Any,
) -> Any:
    """Build a multi-positive antibody-by-antigen target matrix."""
    import torch

    mask = torch.tensor(
        [
            [
                (antibody_key, antigen_key) in known_positive_pairs
                for antigen_key in antigen_keys
            ]
            for antibody_key in antibody_keys
        ],
        dtype=torch.bool,
        device=device,
    )
    if mask.numel() == 0 or not bool(mask.any(dim=1).all()) or not bool(mask.any(dim=0).all()):
        raise ValueError("Every retrieval anchor must have a known positive candidate")
    return mask


def _pad_sequence_features(
    records: Sequence[SequenceRecord],
    embedding_store: Any,
    input_dim: int,
    device: Any,
) -> tuple[Any, Any]:
    import torch

    from interaction_first.model import merge_chunk_embeddings

    if not records:
        raise ValueError("Cannot pad an empty sequence batch")
    features = torch.zeros(
        len(records),
        max(record.length for record in records),
        input_dim,
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros(features.shape[:2], dtype=torch.bool, device=device)
    for index, record in enumerate(records):
        chunks, starts, full_length = embedding_store.load(record.key, device)
        if full_length != record.length:
            raise ValueError(
                f"Embedding length mismatch for {record.key}: "
                f"metadata={record.length}, artifact={full_length}"
            )
        merged = merge_chunk_embeddings(chunks, starts, full_length, device)
        if tuple(merged.shape) != (record.length, input_dim):
            raise ValueError(f"Embedding dimension mismatch for {record.key}")
        features[index, : record.length] = merged
        mask[index, : record.length] = True
    return features, mask


def prepare_pair_batch(
    records: Sequence[PairRecord],
    known_positive_pairs: set[tuple[str, str]],
    embedding_store: Any,
    input_dim: int,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Materialize unique partner axes for one vectorized contrastive batch."""
    antibodies = unique_sequences(records, "antibody")
    antigens = unique_sequences(records, "antigen")
    antibody_features, antibody_mask = _pad_sequence_features(
        antibodies, embedding_store, input_dim, device
    )
    antigen_features, antigen_mask = _pad_sequence_features(
        antigens, embedding_store, input_dim, device
    )
    positive_mask = build_positive_mask(
        [record.key for record in antibodies],
        [record.key for record in antigens],
        known_positive_pairs,
        device,
    )
    return (
        antigen_features,
        antibody_features,
        antigen_mask,
        antibody_mask,
        positive_mask,
    )


def _stable_positive_ranks(similarities: Any, positive_mask: Any) -> list[int]:
    import torch

    ranks = []
    for row_index in range(int(similarities.shape[0])):
        try:
            order = torch.argsort(
                similarities[row_index], descending=True, stable=True
            )
        except TypeError:
            order = torch.argsort(similarities[row_index], descending=True)
        positions = torch.nonzero(
            positive_mask[row_index].index_select(0, order), as_tuple=False
        ).flatten()
        if positions.numel() == 0:
            raise ValueError("Retrieval anchor has no positive candidate")
        ranks.append(int(positions.min().item()) + 1)
    return ranks


def directional_retrieval_metrics(
    similarities: Any,
    positive_mask: Any,
    recall_ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Compute deterministic full-library retrieval metrics for one direction."""
    import torch

    if similarities.ndim != 2 or positive_mask.shape != similarities.shape:
        raise ValueError("Similarity and positive matrices must be two-dimensional and aligned")
    if positive_mask.dtype != torch.bool or not torch.isfinite(similarities).all():
        raise ValueError("Retrieval matrices have invalid types or values")
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("Every retrieval anchor must have at least one positive")
    ks = sorted({int(value) for value in recall_ks})
    if not ks or ks[0] <= 0:
        raise ValueError("recall_ks must contain positive integers")
    ranks = _stable_positive_ranks(similarities, positive_mask)
    metrics: dict[str, Any] = {
        "anchor_count": len(ranks),
        "candidate_count": int(similarities.shape[1]),
        "known_positive_count": int(positive_mask.sum().item()),
        "mean_positives_per_anchor": float(positive_mask.sum(dim=1).float().mean().item()),
        "mrr": float(sum(1.0 / rank for rank in ranks) / len(ranks)),
        "mean_rank": float(sum(ranks) / len(ranks)),
        "median_rank": float(statistics.median(ranks)),
    }
    for value in ks:
        metrics[f"recall_at_{value}"] = float(
            sum(rank <= value for rank in ranks) / len(ranks)
        )
    return metrics


def full_library_metrics(
    antibody_embeddings: Any,
    antigen_embeddings: Any,
    positive_mask: Any,
    temperature: Any,
    recall_ks: Sequence[int],
) -> dict[str, Any]:
    """Evaluate both retrieval directions over every holdout candidate."""
    import torch

    from interaction_first.retrieval import (
        cosine_similarity_matrix,
        symmetric_multi_positive_info_nce,
    )

    antibody_embeddings = antibody_embeddings.to(dtype=torch.float32, device="cpu")
    antigen_embeddings = antigen_embeddings.to(dtype=torch.float32, device="cpu")
    positive_mask = positive_mask.to(device="cpu")
    similarities = cosine_similarity_matrix(antibody_embeddings, antigen_embeddings)
    loss = symmetric_multi_positive_info_nce(
        similarities,
        positive_mask,
        temperature=temperature.detach().to(dtype=torch.float32, device="cpu"),
    )
    antibody_to_antigen = directional_retrieval_metrics(
        similarities, positive_mask, recall_ks
    )
    antigen_to_antibody = directional_retrieval_metrics(
        similarities.transpose(0, 1), positive_mask.transpose(0, 1), recall_ks
    )
    bidirectional_mrr = 0.5 * (
        antibody_to_antigen["mrr"] + antigen_to_antibody["mrr"]
    )
    return {
        "loss": float(loss.item()),
        "bidirectional_mrr": float(bidirectional_mrr),
        "antibody_to_antigen": antibody_to_antigen,
        "antigen_to_antibody": antigen_to_antibody,
    }


def cosine_warmup_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    """Return a linear-warmup and cosine-decay learning-rate multiplier."""
    if total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("Invalid warmup schedule lengths")
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in (0, 1]")
    if step < warmup_steps:
        return max(minimum_ratio, (step + 1) / max(1, warmup_steps))
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _optimizer(model: Any, learning_rate: float, weight_decay: float) -> Any:
    import torch

    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "temperature" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay or not no_decay:
        raise ValueError("Optimizer parameter partition is unexpectedly empty")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
    )


def _scheduler(
    optimizer: Any,
    total_steps: int,
    warmup_fraction: float,
    minimum_ratio: float,
) -> Any:
    import torch

    warmup_steps = min(total_steps - 1, max(1, round(total_steps * warmup_fraction)))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup_multiplier(
            step, total_steps, warmup_steps, minimum_ratio
        ),
    )


def _model_from_config(config: Mapping[str, Any], device: Any) -> Any:
    from interaction_first.retrieval import PartnerRetrievalModel

    architecture = config["model"]
    return PartnerRetrievalModel(
        antigen_input_dim=int(architecture["input_dim"]),
        antibody_input_dim=int(architecture["input_dim"]),
        embedding_dim=int(architecture["embedding_dim"]),
        attention_hidden_dim=int(architecture["attention_hidden_dim"]),
        projection_hidden_dim=int(architecture["projection_hidden_dim"]),
        dropout=float(architecture["dropout"]),
        initial_temperature=float(architecture["initial_temperature"]),
        minimum_temperature=float(architecture["minimum_temperature"]),
        maximum_temperature=float(architecture["maximum_temperature"]),
    ).to(device)


def _encode_library(
    model: Any,
    records: Sequence[SequenceRecord],
    side: str,
    embedding_store: Any,
    input_dim: int,
    batch_size: int,
    bucket_width: int,
    device: Any,
) -> tuple[list[str], Any]:
    import torch

    model.eval()
    encoded: dict[str, Any] = {}
    with torch.no_grad():
        for batch in sequence_bucket_batches(records, batch_size, bucket_width):
            features, mask = _pad_sequence_features(
                batch, embedding_store, input_dim, device
            )
            if side == "antibody":
                embeddings, _ = model.encode_antibody(features, mask)
            elif side == "antigen":
                embeddings, _ = model.encode_antigen(features, mask)
            else:
                raise ValueError("side must be 'antibody' or 'antigen'")
            for record, embedding in zip(batch, embeddings):
                encoded[record.key] = embedding.detach().to(device="cpu")
    keys = sorted(encoded)
    if keys != sorted(record.key for record in records):
        raise AssertionError("Library encoding did not preserve every unique sequence")
    return keys, torch.stack([encoded[key] for key in keys], dim=0)


def evaluate_records(
    model: Any,
    records: Sequence[PairRecord],
    embedding_store: Any,
    input_dim: int,
    batch_size: int,
    bucket_width: int,
    device: Any,
    recall_ks: Sequence[int],
) -> dict[str, Any]:
    """Encode and score a fixed split against its complete partner libraries."""
    antibodies = unique_sequences(records, "antibody")
    antigens = unique_sequences(records, "antigen")
    antibody_keys, antibody_embeddings = _encode_library(
        model,
        antibodies,
        "antibody",
        embedding_store,
        input_dim,
        batch_size,
        bucket_width,
        device,
    )
    antigen_keys, antigen_embeddings = _encode_library(
        model,
        antigens,
        "antigen",
        embedding_store,
        input_dim,
        batch_size,
        bucket_width,
        device,
    )
    positive_mask = build_positive_mask(
        antibody_keys,
        antigen_keys,
        _pair_set(records),
        device="cpu",
    )
    metrics = full_library_metrics(
        antibody_embeddings,
        antigen_embeddings,
        positive_mask,
        model.temperature_head(),
        recall_ks,
    )
    metrics.update(
        {
            "exact_pair_count": len(_pair_set(records)),
            "unique_antibody_count": len(antibody_keys),
            "unique_antigen_count": len(antigen_keys),
        }
    )
    return metrics


def _train_epoch(
    model: Any,
    records: Sequence[PairRecord],
    known_positive_pairs: set[tuple[str, str]],
    embedding_store: Any,
    optimizer: Any,
    scheduler: Any,
    device: Any,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, Any], int]:
    import torch

    from interaction_first.retrieval import symmetric_multi_positive_info_nce

    model.train()
    batches = length_bucket_batches(
        records, args.batch_size, args.bucket_width, args.seed, epoch
    )
    loss_total = 0.0
    anchor_total = 0
    processed = 0
    skipped_without_negatives = 0
    maximum_gradient_norm = 0.0
    for batch_index, batch in enumerate(batches, 1):
        (
            antigen_features,
            antibody_features,
            antigen_mask,
            antibody_mask,
            positive_mask,
        ) = prepare_pair_batch(
            batch,
            known_positive_pairs,
            embedding_store,
            args.input_dim,
            device,
        )
        if bool(positive_mask.all()):
            skipped_without_negatives += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            antigen_features,
            antibody_features,
            antigen_mask,
            antibody_mask,
        )
        loss = symmetric_multi_positive_info_nce(
            outputs["similarity_matrix"],
            positive_mask,
            temperature=outputs["temperature"],
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite contrastive loss in epoch {epoch}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"Non-finite gradient norm in epoch {epoch}")
        optimizer.step()
        scheduler.step()
        global_step += 1
        anchors = int(positive_mask.shape[0] + positive_mask.shape[1])
        loss_total += float(loss.detach().item()) * anchors
        anchor_total += anchors
        processed += 1
        maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm.item()))
        if batch_index % args.log_every == 0 or batch_index == len(batches):
            running = loss_total / max(1, anchor_total)
            print(
                f"epoch={epoch} batch={batch_index}/{len(batches)} "
                f"contrastive_loss={running:.8f}",
                flush=True,
            )
    if processed == 0 or anchor_total == 0:
        raise ValueError("Every training batch lacked valid negative pairs")
    return (
        {
            "loss": loss_total / anchor_total,
            "batch_count": len(batches),
            "optimized_batch_count": processed,
            "skipped_without_negatives": skipped_without_negatives,
            "anchor_count": anchor_total,
            "maximum_preclip_gradient_norm": maximum_gradient_norm,
        },
        global_step,
    )


def _validate_numeric_arguments(args: argparse.Namespace) -> None:
    if args.batch_size < 2 or args.validation_batch_size <= 0:
        raise ValueError("Training batch size must be at least two")
    if args.bucket_width <= 0 or args.input_dim <= 0:
        raise ValueError("Dimensions and bucket width must be positive")
    if args.max_epochs <= 0 or args.patience <= 0 or args.gradient_clip <= 0.0:
        raise ValueError("Training limits must be positive")
    if not 0.0 < args.minimum_learning_rate <= args.learning_rate:
        raise ValueError("Learning-rate bounds are invalid")
    if not 0.0 < args.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in (0, 1)")
    if args.weight_decay < 0.0 or args.min_delta < 0.0:
        raise ValueError("Regularization and min_delta cannot be negative")
    if args.min_train_pairs < 2 or args.min_validation_pairs < 2:
        raise ValueError("Minimum split sizes must be at least two")
    finite_values = (
        args.learning_rate,
        args.minimum_learning_rate,
        args.weight_decay,
        args.warmup_fraction,
        args.gradient_clip,
        args.min_delta,
        args.embedding_cache_gib,
    )
    if not all(math.isfinite(float(value)) for value in finite_values):
        raise ValueError("Numeric training arguments must be finite")
    if args.embedding_cache_gib < 0.0:
        raise ValueError("embedding_cache_gib cannot be negative")


def _source_hashes() -> dict[str, str]:
    """Return immutable hashes for every source file used by this entrypoint."""
    script = Path(__file__).resolve()
    package = script.parent / "interaction_first"
    return {
        "entrypoint": sha256_file(script),
        "retrieval": sha256_file(package / "retrieval.py"),
        "runtime": sha256_file(package / "runtime.py"),
        "model_merge_helpers": sha256_file(package / "model.py"),
        "interaction_scoring": sha256_file(package / "scoring.py"),
    }


def _config(
    args: argparse.Namespace,
    dataset_metadata: Mapping[str, Any],
    embedding_corpus: Mapping[str, Any],
    train_records: Sequence[PairRecord],
    validation_records: Sequence[PairRecord],
    duplicate_counts: Mapping[str, int],
    isolation_audit: Mapping[str, int],
    train_batch_count: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model": {
            "architecture": "PartnerRetrievalModel",
            "input_dim": args.input_dim,
            "embedding_dim": args.embedding_dim,
            "attention_hidden_dim": args.attention_hidden_dim,
            "projection_hidden_dim": args.projection_hidden_dim,
            "dropout": args.dropout,
            "initial_temperature": args.initial_temperature,
            "minimum_temperature": args.minimum_temperature,
            "maximum_temperature": args.maximum_temperature,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_fraction": args.warmup_fraction,
            "schedule": "linear_warmup_cosine_decay_per_optimizer_step",
            "gradient_clip": args.gradient_clip,
            "batch_size": args.batch_size,
            "bucket_width": args.bucket_width,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "seed": args.seed,
            "deterministic": not args.allow_nondeterministic,
            "train_batch_count": train_batch_count,
        },
        "validation": {
            "protocol": "fixed_full_library_bidirectional",
            "checkpoint_selection": "bidirectional_validation_mrr",
            "recall_ks": sorted(set(args.recall_ks)),
            "validation_batch_size": args.validation_batch_size,
            "pair_count": len(validation_records),
            "exact_pair_sha256": canonical_sha256(
                {
                    "pairs": sorted(
                        [list(value) for value in _pair_set(validation_records)]
                    )
                }
            ),
        },
        "training_data": {
            "pair_count_after_exact_deduplication": len(train_records),
            "exact_pair_sha256": canonical_sha256(
                {"pairs": sorted([list(value) for value in _pair_set(train_records)])}
            ),
            "duplicate_counts": dict(duplicate_counts),
            "train_validation_isolation": dict(isolation_audit),
            "positive_mask_policy": "all_observed_pairs_within_training_split",
            "unknown_pair_policy": "unobserved_pairs_are_contrastive_negatives",
        },
        "dataset": dict(dataset_metadata),
        "embedding_corpus": dict(embedding_corpus),
        "source_sha256": _source_hashes(),
    }
    config["config_sha256"] = canonical_sha256(config)
    return config


def _provenance(
    config: Mapping[str, Any], device: Any, embedding_root: Path
) -> dict[str, Any]:
    import torch

    script = Path(__file__).resolve()
    package = script.parent / "interaction_first"
    return {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config["config_sha256"],
        "source_files": {
            "entrypoint": {"path": str(script), "sha256": sha256_file(script)},
            "retrieval": {
                "path": str(package / "retrieval.py"),
                "sha256": sha256_file(package / "retrieval.py"),
            },
            "runtime": {
                "path": str(package / "runtime.py"),
                "sha256": sha256_file(package / "runtime.py"),
            },
            "model_merge_helpers": {
                "path": str(package / "model.py"),
                "sha256": sha256_file(package / "model.py"),
            },
            "interaction_scoring": {
                "path": str(package / "scoring.py"),
                "sha256": sha256_file(package / "scoring.py"),
            },
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "embedding_location": str(embedding_root.expanduser().resolve()),
        "scientific_scope": {
            "frozen_test_used": False,
            "model_selection_data": "fixed development validation only",
            "checkpoint_selection_metric": "bidirectional validation MRR",
        },
    }


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    epoch: int,
    global_step: int,
    best_validation_mrr: float,
    best_epoch: int,
    epochs_without_improvement: int,
) -> dict[str, Any]:
    import torch

    return {
        "format_version": FORMAT_VERSION,
        "config": dict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_mrr": best_validation_mrr,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "history": list(history),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def command_train(args: argparse.Namespace) -> None:
    """Train with symmetric multi-positive contrastive supervision."""
    import torch

    from interaction_first.runtime import (
        EmbeddingStore,
        atomic_torch_save,
        safe_torch_load,
        set_reproducible_seed,
    )

    _validate_numeric_arguments(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    set_reproducible_seed(args.seed, deterministic=not args.allow_nondeterministic)

    split_records, dataset_metadata = load_dataset(
        args.prepare_dir, args.split_manifest
    )
    train_records, train_duplicates = deduplicate_exact_pairs(split_records["train"])
    validation_records, validation_duplicates = deduplicate_exact_pairs(
        split_records["validation"]
    )
    if len(train_records) < args.min_train_pairs:
        raise ValueError(
            f"Training split has only {len(train_records)} exact pairs; "
            f"minimum={args.min_train_pairs}"
        )
    if len(validation_records) < args.min_validation_pairs:
        raise ValueError(
            f"Validation split has only {len(validation_records)} exact pairs; "
            f"minimum={args.min_validation_pairs}"
        )
    isolation = audit_double_remote_isolation(
        train_records, validation_records, "validation"
    )
    train_batches = length_bucket_batches(
        train_records, args.batch_size, args.bucket_width, args.seed, epoch=1
    )
    duplicate_counts = {
        "train_removed": train_duplicates,
        "validation_removed": validation_duplicates,
    }
    embedding_root = args.embedding_dir.expanduser().resolve()
    embedding_corpus = embedding_corpus_fingerprint(
        embedding_root,
        [*train_records, *validation_records],
        "deduplicated_train_and_validation_sequences",
    )
    config = _config(
        args,
        dataset_metadata,
        embedding_corpus,
        train_records,
        validation_records,
        duplicate_counts,
        isolation,
        len(train_batches),
    )

    model = _model_from_config(config, device)
    optimizer = _optimizer(model, args.learning_rate, args.weight_decay)
    total_steps = args.max_epochs * len(train_batches)
    scheduler = _scheduler(
        optimizer,
        total_steps,
        args.warmup_fraction,
        args.minimum_learning_rate / args.learning_rate,
    )
    embedding_store = EmbeddingStore(
        embedding_root, args.embedding_cache_gib
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.json"
    provenance_path = output_dir / "provenance.json"
    existing = [path for path in (last_path, best_path, history_path) if path.exists()]
    if not args.resume and existing:
        raise FileExistsError(
            "Refusing to overwrite an existing non-resumed run: "
            f"{[str(path) for path in existing]}"
        )
    if args.resume and not last_path.is_file() and existing:
        raise ValueError("Run artifacts exist but last.pt is missing; resume is unsafe")

    provenance = _provenance(config, device, embedding_root)
    if provenance_path.is_file():
        old_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if old_provenance.get("config_sha256") != config["config_sha256"]:
            raise ValueError("Existing provenance belongs to a different configuration")
    else:
        atomic_write_json(provenance_path, provenance)

    start_epoch = 1
    global_step = 0
    best_validation_mrr = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if args.resume and last_path.is_file():
        checkpoint = safe_torch_load(last_path, map_location=device)
        if checkpoint.get("format_version") != FORMAT_VERSION:
            raise ValueError("Checkpoint format is incompatible")
        previous_config = checkpoint.get("config", {})
        validate_config_integrity(previous_config)
        if previous_config.get("config_sha256") != config["config_sha256"]:
            raise ValueError("Checkpoint configuration or input digests changed")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_validation_mrr = float(checkpoint["best_validation_mrr"])
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history = list(checkpoint.get("history", []))
        torch.set_rng_state(checkpoint["torch_rng_state"].to(device="cpu"))
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(
                [state.to(device="cpu") for state in checkpoint["cuda_rng_state_all"]]
            )

    if epochs_without_improvement >= args.patience:
        print(
            "The resumed checkpoint already satisfies the early-stopping rule; "
            "no additional epochs will run.",
            flush=True,
        )
        start_epoch = args.max_epochs + 1

    known_positive_pairs = _pair_set(train_records)
    for epoch in range(start_epoch, args.max_epochs + 1):
        started = time.time()
        train_metrics, global_step = _train_epoch(
            model,
            train_records,
            known_positive_pairs,
            embedding_store,
            optimizer,
            scheduler,
            device,
            args,
            epoch,
            global_step,
        )
        validation_metrics = evaluate_records(
            model,
            validation_records,
            embedding_store,
            args.input_dim,
            args.validation_batch_size,
            args.bucket_width,
            device,
            args.recall_ks,
        )
        validation_mrr = float(validation_metrics["bidirectional_mrr"])
        if not math.isfinite(validation_mrr):
            raise FloatingPointError("Validation MRR is non-finite")
        improved = validation_mrr > best_validation_mrr + args.min_delta
        if improved:
            best_validation_mrr = validation_mrr
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_metric": "bidirectional_validation_mrr",
            "best_validation_mrr": best_validation_mrr,
            "best_epoch": best_epoch,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "temperature": float(model.temperature_head().detach().cpu().item()),
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        checkpoint_payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            config,
            history,
            epoch,
            global_step,
            best_validation_mrr,
            best_epoch,
            epochs_without_improvement,
        )
        if improved:
            atomic_torch_save(checkpoint_payload, best_path)
        atomic_torch_save(checkpoint_payload, last_path)
        atomic_write_json(
            history_path,
            {
                "format_version": FORMAT_VERSION,
                "config": config,
                "history": history,
                "embedding_cache": embedding_store.summary(),
            },
        )
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError("Training did not produce a best checkpoint")
    summary = {
        "format_version": FORMAT_VERSION,
        "stage": "train_complete",
        "config_sha256": config["config_sha256"],
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "best_validation_mrr": best_validation_mrr,
        "best_epoch": best_epoch,
        "completed_epoch": history[-1]["epoch"] if history else start_epoch - 1,
        "frozen_test_used": False,
    }
    atomic_write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


def command_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a checkpoint on validation or an explicitly unlocked test split."""
    import torch

    from interaction_first.runtime import EmbeddingStore, safe_torch_load, set_reproducible_seed

    if args.split == "test" and not args.unlock_frozen_test:
        raise PermissionError(
            "Frozen test evaluation is locked; pass --unlock-frozen-test explicitly"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    set_reproducible_seed(args.seed, deterministic=not args.allow_nondeterministic)
    split_records, dataset_metadata = load_dataset(
        args.prepare_dir, args.split_manifest
    )
    train_records, _ = deduplicate_exact_pairs(split_records["train"])
    validation_records, _ = deduplicate_exact_pairs(split_records["validation"])
    evaluation_records, removed = deduplicate_exact_pairs(split_records[args.split])
    audit = audit_double_remote_isolation(train_records, evaluation_records, args.split)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Checkpoint format is incompatible")
    config = checkpoint["config"]
    validate_config_integrity(config)
    if config.get("source_sha256") != _source_hashes():
        raise ValueError("Evaluation source code differs from the checkpoint provenance")
    if config["dataset"]["instances_sha256"] != dataset_metadata["instances_sha256"]:
        raise ValueError("Prepared instances changed after checkpoint training")
    if config["dataset"]["split_manifest_sha256"] != dataset_metadata["split_manifest_sha256"]:
        raise ValueError("Split manifest changed after checkpoint training")
    embedding_root = args.embedding_dir.expanduser().resolve()
    observed_embedding_corpus = embedding_corpus_fingerprint(
        embedding_root,
        [*train_records, *validation_records],
        "deduplicated_train_and_validation_sequences",
    )
    if config.get("embedding_corpus") != observed_embedding_corpus:
        raise ValueError("Frozen embedding corpus changed after checkpoint training")
    model = _model_from_config(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    embedding_store = EmbeddingStore(
        embedding_root, args.embedding_cache_gib
    )
    metrics = evaluate_records(
        model,
        evaluation_records,
        embedding_store,
        int(config["model"]["input_dim"]),
        args.batch_size,
        args.bucket_width,
        device,
        args.recall_ks,
    )
    output = {
        "format_version": FORMAT_VERSION,
        "stage": "full_library_evaluation",
        "split": args.split,
        "frozen_test_unlocked": bool(args.split == "test"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": config["config_sha256"],
        "dataset": dataset_metadata,
        "embedding_corpus": observed_embedding_corpus,
        "exact_duplicate_pairs_removed": removed,
        "train_holdout_isolation": audit,
        "metrics": metrics,
        "embedding_cache": embedding_store.summary(),
    }
    atomic_write_json(args.output.expanduser().resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


def command_smoke(args: argparse.Namespace) -> None:
    """Run a deterministic synthetic gradient and full-library metric check."""
    import torch

    from interaction_first.retrieval import (
        PartnerRetrievalModel,
        symmetric_multi_positive_info_nce,
    )

    torch.manual_seed(args.seed)
    model = PartnerRetrievalModel(
        antigen_input_dim=16,
        antibody_input_dim=16,
        embedding_dim=8,
        attention_hidden_dim=8,
        projection_hidden_dim=16,
        dropout=0.0,
    )
    antigen = torch.randn(4, 7, 16)
    antibody = torch.randn(4, 5, 16)
    antigen_mask = torch.tensor(
        [[1] * 7, [1] * 6 + [0], [1] * 5 + [0] * 2, [1] * 7],
        dtype=torch.bool,
    )
    antibody_mask = torch.tensor(
        [[1] * 5, [1] * 4 + [0], [1] * 5, [1] * 3 + [0] * 2],
        dtype=torch.bool,
    )
    outputs = model(antigen, antibody, antigen_mask, antibody_mask)
    positives = torch.eye(4, dtype=torch.bool)
    positives[0, 1] = True
    loss = symmetric_multi_positive_info_nce(
        outputs["similarity_matrix"], positives, outputs["temperature"]
    )
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    metrics = full_library_metrics(
        outputs["antibody_embeddings"].detach(),
        outputs["antigen_embeddings"].detach(),
        positives,
        outputs["temperature"].detach(),
        (1, 2, 4),
    )
    summary = {
        "format_version": FORMAT_VERSION,
        "stage": "synthetic_smoke",
        "loss": float(loss.item()),
        "finite_gradients": finite_gradients,
        "metrics": metrics,
    }
    if not finite_gradients:
        raise FloatingPointError("Synthetic smoke test produced invalid gradients")
    if args.output is not None:
        atomic_write_json(args.output.expanduser().resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument("--embedding-cache-gib", type=float, default=24.0)
    parser.add_argument("--bucket-width", type=int, default=128)
    parser.add_argument("--recall-ks", type=int, nargs="+", default=[1, 5, 10])


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone partner-retrieval command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    _add_data_arguments(train)
    _add_runtime_arguments(train)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--input-dim", type=int, default=1280)
    train.add_argument("--embedding-dim", type=int, default=128)
    train.add_argument("--attention-hidden-dim", type=int, default=128)
    train.add_argument("--projection-hidden-dim", type=int, default=256)
    train.add_argument("--dropout", type=float, default=0.15)
    train.add_argument("--initial-temperature", type=float, default=0.07)
    train.add_argument("--minimum-temperature", type=float, default=0.01)
    train.add_argument("--maximum-temperature", type=float, default=0.50)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--validation-batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--minimum-learning-rate", type=float, default=3e-6)
    train.add_argument("--weight-decay", type=float, default=1e-2)
    train.add_argument("--warmup-fraction", type=float, default=0.10)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--max-epochs", type=int, default=40)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--min-delta", type=float, default=1e-4)
    train.add_argument("--min-train-pairs", type=int, default=32)
    train.add_argument("--min-validation-pairs", type=int, default=32)
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser("evaluate")
    _add_data_arguments(evaluate)
    _add_runtime_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--unlock-frozen-test", action="store_true")
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.set_defaults(handler=command_evaluate)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--output", type=Path)
    smoke.set_defaults(handler=command_smoke)
    return parser


def main() -> None:
    """Dispatch one partner-retrieval pipeline stage."""
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
