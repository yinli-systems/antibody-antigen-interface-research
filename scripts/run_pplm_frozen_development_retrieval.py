#!/usr/bin/env python3
"""Run the official PPLM-PPI ensemble on one frozen development panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from run_mint_frozen_development_retrieval import (
    average_tie_rank,
    batch_work,
    endpoint_windows,
    load_inputs,
    percentile,
    publish_create_only,
    register_pair,
    sha256_file,
    summarize_anchor_metrics,
)


PPLM_REVISION = "1267db4592703095b889cc22da6cf1796ce038b5"
PPLM_CHECKPOINT_SHA256 = "c8ce3b75dfd7b5890dfa7d96de7f698b522a5cbf59d7ee2732cb80ad3e5df495"
PPLM_HEAD_SHA256 = "2695b886ef3ba2c1acacb74cd4f8c8173323addfc548b6d482143fb2b7070d8f"
HEAD_AUDIT_SHA256 = "d4d6700057038ac0c71f18125329f7f1cd299925003bf7cafc2e33c6110ec586"
SOURCE_FILE_SHA256 = {
    "pplm/__init__.py": "87744ffcf3b2caaf6123abf6e69e2d39ffa1ad1a59caa7beff035f84bf49d780",
    "pplm/constants.py": "8bf9231414c6e2e61efc53be627225597eeb9fc72310b756becb3c68139a59a1",
    "pplm/data.py": "79df68cc6d677fe76f2739326c9d8b753990786b44f66e565597c6a82e23312b",
    "pplm/modules.py": "92e10cb04771178a46bb6708099da11a2facbfcff3b85f82781d772c063aaa3c",
    "pplm/multihead_attention.py": "a3717bad059e9b5e45ba7fb5e0537253fff0f41ca8e31aa8a6bd8f78cc927b81",
    "pplm/pplm.py": "521cb8085069f7df1b154101de5c73bf5f4707d8314b7fefbb4319b80c021434",
    "pplm/rotary_embedding.py": "a856bd6cdc10eab6086a72cd1447a38956c51516a36a90ad815a8e9a52212230",
    "pplm_ppi/__init__.py": "523a71b86095b9f71031939b7e9cb0ef44ea2440856223398c8402465ad07d60",
    "pplm_ppi/model.py": "0e0812bbe23008f2d34cba247a1ce296ebe3f6252dc125f094a7db6b42c1733d",
    "run_pplm-ppi.py": "d4a7c6f5fa3e0b20b287d0646487b820f2d7bfad974bbeef691e6a78584526e7",
}
IMPLEMENTATION_FILE_SHA256 = {
    "src/antibody_pretraining/pair_aware_attention.py": "9d830b909936aa85c0f54f5a3e1aabde4457ac0c4675865a1a2d166260412d58",
    "src/antibody_pretraining/pair_aware_esm2.py": "0dcf3a156c0d4fc3d8099afdcb856171164bfc525ff4319b9bbc24dbb743cd24",
}
EQUIVALENCE_TOLERANCE = 5e-6
PADDING_EQUIVALENCE_TOLERANCE = 2e-5
PROBABILITY_EQUIVALENCE_TOLERANCE = 1e-6
ACCEPTED_SMOKE_PROTOCOL_SHA256 = "ad32eb302afb7721513733953cb0f5e37f4117c776f6c9ab89dca9b4ca13137f"


def verify_source(source_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in SOURCE_FILE_SHA256.items():
        path = source_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"PPLM source file is absent or linked: {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"PPLM source file SHA-256 differs: {relative}")
        observed[relative] = digest
    return observed


def verify_implementation(implementation_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in IMPLEMENTATION_FILE_SHA256.items():
        path = implementation_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"PPLM implementation file is absent or linked: {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"PPLM implementation file SHA-256 differs: {relative}")
        observed[relative] = digest
    return observed


def verify_smoke_gate(path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("PPLM equivalence smoke summary is absent or linked")
    if sha256_file(path) != expected_sha256:
        raise ValueError("PPLM equivalence smoke summary SHA-256 differs")
    summary = json.loads(path.read_text(encoding="utf-8"))
    equivalence = summary.get("equivalence", {})
    if (
        summary.get("status") != "SMOKE_NONCLAIM_COMPLETE"
        or summary.get("protocol_sha256") != ACCEPTED_SMOKE_PROTOCOL_SHA256
        or summary.get("authorization", {}).get("development_pair_retrieval") is not False
        or summary.get("test_access") is not False
        or equivalence.get("status") != "EXACT_FEATURE_SEMANTICS_ACCEPTED"
        or float(
            equivalence.get(
                "streaming_vs_official_full_attention_max_abs_delta", math.inf
            )
        )
        > EQUIVALENCE_TOLERANCE
        or float(
            equivalence.get(
                "clean_room_vs_official_probability_max_abs_delta", math.inf
            )
        )
        > PROBABILITY_EQUIVALENCE_TOLERANCE
        or float(
            equivalence.get("batch_padding_vs_singleton_max_abs_delta", math.inf)
        )
        > PADDING_EQUIVALENCE_TOLERANCE
    ):
        raise ValueError("PPLM equivalence smoke summary does not satisfy the gate")
    return {
        "required_smoke_summary_sha256": expected_sha256,
        "smoke_protocol_sha256": ACCEPTED_SMOKE_PROTOCOL_SHA256,
        "status": "ACCEPTED_BY_VERIFIED_SMOKE_GATE",
    }


def retrieval_pairs(
    records: list[dict[str, Any]],
    authority_records: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str, str], tuple[str, str]],
    dict[str, dict[str, Any]],
]:
    index = {record["instance_id"]: record for record in authority_records}
    pairs: dict[tuple[str, str, str], tuple[str, str]] = {}
    for record in records:
        antibody_hash = record["source_antibody_sequence_sha256"]
        antigen_hash = record["source_antigen_sequence_sha256"]
        register_pair(
            pairs,
            ("ordered", antibody_hash, antigen_hash),
            record["antibody_endpoint_sequence"],
            record["antigen_sequence"],
        )
        for identifier in record["wrong_antibody_instance_ids"]:
            candidate = index[identifier]
            register_pair(
                pairs,
                (
                    "ordered",
                    candidate["source_antibody_sequence_sha256"],
                    antigen_hash,
                ),
                candidate["antibody_endpoint_sequence"],
                record["antigen_sequence"],
            )
        for identifier in record["wrong_antigen_instance_ids"]:
            candidate = index[identifier]
            register_pair(
                pairs,
                (
                    "ordered",
                    antibody_hash,
                    candidate["source_antigen_sequence_sha256"],
                ),
                record["antibody_endpoint_sequence"],
                candidate["antigen_sequence"],
            )
    return pairs, index


def make_token_batch(
    batch: Sequence[tuple[tuple[str, str, str], str, str]],
    alphabet: Any,
    device: torch.device,
) -> tuple[Tensor, int, list[tuple[int, int]]]:
    encoded = [
        (
            [alphabet.cls_idx, *alphabet.encode(left), alphabet.eos_idx],
            [alphabet.cls_idx, *alphabet.encode(right), alphabet.eos_idx],
        )
        for _, left, right in batch
    ]
    maximum_left = max(len(left) for left, _ in encoded)
    maximum_right = max(len(right) for _, right in encoded)
    tokens = torch.full(
        (len(batch), maximum_left + maximum_right),
        alphabet.padding_idx,
        dtype=torch.long,
        device=device,
    )
    residue_lengths: list[tuple[int, int]] = []
    for row, (left, right) in enumerate(encoded):
        tokens[row, : len(left)] = torch.tensor(left, dtype=torch.long, device=device)
        tokens[row, maximum_left : maximum_left + len(right)] = torch.tensor(
            right, dtype=torch.long, device=device
        )
        residue_lengths.append((len(left) - 2, len(right) - 2))
    return tokens, maximum_left, residue_lengths


def inter_chain_mask(
    token_count: int,
    left_block_tokens: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if not 0 < left_block_tokens < token_count:
        raise ValueError("PPLM chain boundary is outside the token tensor")
    mask = torch.ones((token_count, token_count), device=device, dtype=dtype)
    mask[:left_block_tokens, :left_block_tokens] = 0
    mask[left_block_tokens:, left_block_tokens:] = 0
    return mask


def reduce_layer_attention(
    attention: Tensor,
    left_block_tokens: int,
    residue_lengths: Sequence[tuple[int, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Reduce one official PPLM layer from H,B,T,T to six B,H features."""
    if attention.ndim != 4:
        raise ValueError("Official PPLM attention must have shape H,B,T,T")
    head_count, batch_size, token_count, source_count = attention.shape
    if token_count != source_count or batch_size != len(residue_lengths):
        raise ValueError("Official PPLM attention shape differs from batch metadata")
    rows = attention.transpose(0, 1).float()
    mean_inter: list[Tensor] = []
    mean_aa: list[Tensor] = []
    mean_bb: list[Tensor] = []
    max_inter: list[Tensor] = []
    max_aa: list[Tensor] = []
    max_bb: list[Tensor] = []
    for row, (left_length, right_length) in zip(rows, residue_lengths):
        if left_length <= 0 or right_length <= 0:
            raise ValueError("PPLM retrieval cannot score an empty endpoint")
        left = slice(1, 1 + left_length)
        right = slice(left_block_tokens + 1, left_block_tokens + 1 + right_length)
        aa = row[:, left, left]
        ab = row[:, left, right]
        ba = row[:, right, left]
        bb = row[:, right, right]
        inter = (ab + ba.transpose(-1, -2)) * 0.5
        mean_inter.append(inter.mean(dim=(-1, -2)))
        mean_aa.append(aa.mean(dim=(-1, -2)))
        mean_bb.append(bb.mean(dim=(-1, -2)))
        max_inter.append(inter.amax(dim=(-1, -2)))
        max_aa.append(aa.amax(dim=(-1, -2)))
        max_bb.append(bb.amax(dim=(-1, -2)))
    outputs = tuple(
        torch.stack(values, dim=0)
        for values in (mean_inter, mean_aa, mean_bb, max_inter, max_aa, max_bb)
    )
    if any(value.shape != (batch_size, head_count) for value in outputs):
        raise RuntimeError("Reduced PPLM attention feature shape differs")
    return outputs  # type: ignore[return-value]


def reduce_embeddings(
    representation: Tensor,
    left_block_tokens: int,
    residue_lengths: Sequence[tuple[int, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mean_a: list[Tensor] = []
    mean_b: list[Tensor] = []
    max_a: list[Tensor] = []
    max_b: list[Tensor] = []
    for row, (left_length, right_length) in zip(representation.float(), residue_lengths):
        left = row[1 : 1 + left_length]
        right = row[
            left_block_tokens + 1 : left_block_tokens + 1 + right_length
        ]
        mean_a.append(left.mean(dim=0))
        mean_b.append(right.mean(dim=0))
        max_a.append(left.amax(dim=0))
        max_b.append(right.amax(dim=0))
    return tuple(
        torch.stack(values, dim=0)
        for values in (mean_a, mean_b, max_a, max_b)
    )  # type: ignore[return-value]


def streaming_features(
    model: nn.Module,
    tokens: Tensor,
    left_block_tokens: int,
    residue_lengths: Sequence[tuple[int, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run the batch-safe clean-room layers while retaining pooled features."""
    if getattr(model.config, "token_dropout", None) is not False:
        raise ValueError("PPLM-PPI requires token_dropout=False")
    valid_mask = tokens.ne(model.config.padding_index)
    chain_ids = torch.full_like(tokens, -1)
    chain_ids[:, :left_block_tokens] = 0
    chain_ids[:, left_block_tokens:] = 1
    chain_ids[~valid_mask] = -1
    hidden = model._embed(tokens, valid_mask)
    channel_count = model.config.layer_count * model.config.head_count
    attention_features = [
        torch.empty(
            (tokens.shape[0], channel_count),
            dtype=torch.float32,
            device=tokens.device,
        )
        for _ in range(6)
    ]
    for layer_index, layer in enumerate(model.layers):
        hidden, attention = layer(
            hidden,
            chain_ids,
            valid_mask,
            return_attention=True,
        )
        reduced = reduce_layer_attention(
            attention.transpose(0, 1), left_block_tokens, residue_lengths
        )
        start = layer_index * model.config.head_count
        stop = start + model.config.head_count
        for destination, values in zip(attention_features, reduced):
            destination[:, start:stop] = values
        del attention, reduced
    representation = model.emb_layer_norm_after(hidden)
    mean_a, mean_b, max_a, max_b = reduce_embeddings(
        representation, left_block_tokens, residue_lengths
    )
    mean_inter, mean_aa, mean_bb, max_inter, max_aa, max_bb = attention_features
    return (
        mean_inter,
        mean_aa,
        mean_bb,
        mean_a,
        mean_b,
        max_inter,
        max_aa,
        max_bb,
        max_a,
        max_b,
    )


def full_reference_features(
    model: nn.Module,
    tokens: Tensor,
    left_block_tokens: int,
    residue_lengths: Sequence[tuple[int, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    mask = inter_chain_mask(
        tokens.shape[1],
        left_block_tokens,
        device=tokens.device,
        dtype=model.embed_tokens.weight.dtype,
    )
    output = model(
        tokens,
        mask,
        repr_layers=[model.num_layers],
        need_head_weights=True,
        return_contacts=False,
    )
    attention = output["attentions"]
    representation = output["representations"][model.num_layers]
    batch_size, layer_count, head_count, _, _ = attention.shape
    flattened = [[] for _ in range(6)]
    for layer_index in range(layer_count):
        reduced = reduce_layer_attention(
            attention[:, layer_index].transpose(0, 1),
            left_block_tokens,
            residue_lengths,
        )
        for values, layer_values in zip(flattened, reduced):
            values.append(layer_values)
    attention_features = [torch.cat(values, dim=1) for values in flattened]
    if any(value.shape != (batch_size, layer_count * head_count) for value in attention_features):
        raise RuntimeError("Full-reference PPLM feature shape differs")
    mean_a, mean_b, max_a, max_b = reduce_embeddings(
        representation, left_block_tokens, residue_lengths
    )
    mean_inter, mean_aa, mean_bb, max_inter, max_aa, max_bb = attention_features
    return (
        mean_inter,
        mean_aa,
        mean_bb,
        mean_a,
        mean_b,
        max_inter,
        max_aa,
        max_bb,
        max_a,
        max_b,
    )


def maximum_feature_delta(
    left: Sequence[Tensor], right: Sequence[Tensor]
) -> float:
    if len(left) != len(right):
        raise ValueError("PPLM feature tuple lengths differ")
    return max(float((a.float() - b.float()).abs().max()) for a, b in zip(left, right))


def feature_deltas(left: Sequence[Tensor], right: Sequence[Tensor]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("PPLM feature tuple lengths differ")
    return [
        float((a.float() - b.float()).abs().max()) for a, b in zip(left, right)
    ]


def equivalence_audit(
    model: nn.Module,
    official_reference: nn.Module,
    mean_heads: nn.ModuleList,
    max_heads: nn.ModuleList,
    alphabet: Any,
    device: torch.device,
) -> dict[str, Any]:
    pairs = [
        (("audit", "a", "b"), "ACDEFGHIK", "LMNPQRSTVWY"),
        (("audit", "c", "d"), "ACDEFGHIKLMNPQ", "RSTVWY"),
    ]
    tokens, boundary, lengths = make_token_batch(pairs, alphabet, device)
    with torch.inference_mode():
        streaming = streaming_features(model, tokens, boundary, lengths)
    singleton_deltas: list[float] = []
    official_deltas: list[float] = []
    official_feature_deltas: list[list[float]] = []
    official_probability_deltas: list[float] = []
    for index, pair in enumerate(pairs):
        one_tokens, one_boundary, one_lengths = make_token_batch([pair], alphabet, device)
        with torch.inference_mode():
            one = streaming_features(model, one_tokens, one_boundary, one_lengths)
            reference = full_reference_features(
                official_reference, one_tokens, one_boundary, one_lengths
            )
        singleton_deltas.append(
            maximum_feature_delta(
                tuple(feature[index : index + 1] for feature in streaming), one
            )
        )
        official_deltas.append(maximum_feature_delta(one, reference))
        official_feature_deltas.append(feature_deltas(one, reference))
        with torch.inference_mode():
            clean_probability = ensemble_probability(
                tuple(feature.float() for feature in one), mean_heads, max_heads
            )
            official_probability = ensemble_probability(
                tuple(feature.float() for feature in reference), mean_heads, max_heads
            )
        official_probability_deltas.append(
            float((clean_probability - official_probability).abs().max())
        )
    streaming_delta = max(official_deltas)
    padding_delta = max(singleton_deltas)
    feature_names = (
        "mean_inter_attention",
        "mean_intra_a_attention",
        "mean_intra_b_attention",
        "mean_embedding_a",
        "mean_embedding_b",
        "max_inter_attention",
        "max_intra_a_attention",
        "max_intra_b_attention",
        "max_embedding_a",
        "max_embedding_b",
    )
    diagnostic = {
        "batch_padding_vs_singleton_max_abs_delta": padding_delta,
        "clean_room_vs_official_feature_max_abs_delta": {
            name: max(row[index] for row in official_feature_deltas)
            for index, name in enumerate(feature_names)
        },
        "clean_room_vs_official_probability_max_abs_delta": max(
            official_probability_deltas
        ),
        "streaming_vs_official_full_attention_max_abs_delta": streaming_delta,
    }
    if streaming_delta > EQUIVALENCE_TOLERANCE:
        raise RuntimeError(
            "Streaming features differ from official full attention: "
            + json.dumps(diagnostic, sort_keys=True)
        )
    if padding_delta > PADDING_EQUIVALENCE_TOLERANCE:
        raise RuntimeError(
            "Padded PPLM batch differs from singleton evaluation: "
            + json.dumps(diagnostic, sort_keys=True)
        )
    if (
        diagnostic["clean_room_vs_official_probability_max_abs_delta"]
        > PROBABILITY_EQUIVALENCE_TOLERANCE
    ):
        raise RuntimeError(
            "PPLM-PPI probability differs from the official full-attention path: "
            + json.dumps(diagnostic, sort_keys=True)
        )
    return {
        **diagnostic,
        "batch_padding_vs_singleton_tolerance": PADDING_EQUIVALENCE_TOLERANCE,
        "clean_room_vs_official_probability_tolerance": PROBABILITY_EQUIVALENCE_TOLERANCE,
        "streaming_vs_official_full_attention_tolerance": EQUIVALENCE_TOLERANCE,
        "status": "EXACT_FEATURE_SEMANTICS_ACCEPTED",
    }


def ensemble_probability(
    features: Sequence[Tensor], mean_heads: nn.ModuleList, max_heads: nn.ModuleList
) -> Tensor:
    if len(features) != 10 or len(mean_heads) != 5 or len(max_heads) != 5:
        raise ValueError("PPLM-PPI ensemble shape differs")
    mean_inter, mean_aa, mean_bb, mean_a, mean_b = features[:5]
    max_inter, max_aa, max_bb, max_a, max_b = features[5:]
    predictions: list[Tensor] = []
    for head in mean_heads:
        forward = head(mean_inter, mean_aa, mean_bb, mean_a, mean_b)
        reverse = head(mean_inter, mean_bb, mean_aa, mean_b, mean_a)
        predictions.append((forward + reverse) * 0.5)
    for head in max_heads:
        forward = head(max_inter, max_aa, max_bb, max_a, max_b)
        reverse = head(max_inter, max_bb, max_aa, max_b, max_a)
        predictions.append((forward + reverse) * 0.5)
    return torch.stack(predictions, dim=0).mean(dim=0).squeeze(-1)


def score_pairs(
    *,
    pairs: Mapping[tuple[str, str, str], tuple[str, str]],
    model: nn.Module,
    mean_heads: nn.ModuleList,
    max_heads: nn.ModuleList,
    alphabet: Any,
    device: torch.device,
    maximum_batch_size: int,
    maximum_attention_cells: int,
) -> tuple[dict[tuple[str, str, str], float], dict[str, Any]]:
    work: list[tuple[tuple[str, str, str], str, str]] = []
    expected_windows: dict[tuple[str, str, str], int] = {}
    for key, (left, right) in pairs.items():
        left_windows = endpoint_windows(left)
        right_windows = endpoint_windows(right)
        expected_windows[key] = len(left_windows) * len(right_windows)
        for left_window in left_windows:
            for right_window in right_windows:
                work.append((key, left_window, right_window))
    batches = batch_work(work, maximum_batch_size, maximum_attention_cells)
    sums = {key: 0.0 for key in pairs}
    counts = {key: 0 for key in pairs}
    scored_residues = 0
    scored_windows = 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(batches, start=1):
        tokens, boundary, lengths = make_token_batch(batch, alphabet, device)
        scored_residues += sum(left + right for left, right in lengths)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            features = streaming_features(model, tokens, boundary, lengths)
        with torch.inference_mode():
            probabilities = ensemble_probability(
                tuple(feature.float() for feature in features), mean_heads, max_heads
            )
        for item, probability in zip(batch, probabilities.tolist()):
            key = item[0]
            sums[key] += float(probability)
            counts[key] += 1
        scored_windows += len(batch)
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(batches):
            print(
                json.dumps(
                    {
                        "batch": batch_index,
                        "batches": len(batches),
                        "pair_windows_scored": scored_windows,
                        "pair_windows_total": len(work),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    for key in pairs:
        if counts[key] != expected_windows[key]:
            raise RuntimeError("PPLM pair window count differs")
    probabilities = {key: sums[key] / counts[key] for key in pairs}
    if any(not 0.0 <= value <= 1.0 for value in probabilities.values()):
        raise RuntimeError("PPLM-PPI probability is outside [0, 1]")
    return probabilities, {
        "batch_count": len(batches),
        "elapsed_seconds": elapsed,
        "nonpadding_residues": scored_residues,
        "nonpadding_residues_per_second": scored_residues / elapsed,
        "pair_windows": scored_windows,
        "pair_windows_per_second": scored_windows / elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "unique_ordered_pair_count": len(pairs),
    }


def stable_logit(probability: float) -> float:
    clipped = min(max(probability, 1e-7), 1.0 - 1e-7)
    return math.log(clipped) - math.log1p(-clipped)


def evaluate(
    records: list[dict[str, Any]],
    index: dict[str, dict[str, Any]],
    scores: Mapping[tuple[str, str, str], float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    for record in records:
        antibody_hash = record["source_antibody_sequence_sha256"]
        antigen_hash = record["source_antigen_sequence_sha256"]
        cognate_key = ("ordered", antibody_hash, antigen_hash)
        cognate_score = float(scores[cognate_key])
        anchor: dict[str, Any] = {
            "canonical_pair_sha256": record["canonical_pair_sha256"],
            "instance_id": record["instance_id"],
        }
        for direction, candidate_field, endpoint_field in (
            (
                "fixed_antigen_wrong_antibody",
                "wrong_antibody_instance_ids",
                "source_antibody_sequence_sha256",
            ),
            (
                "fixed_antibody_wrong_antigen",
                "wrong_antigen_instance_ids",
                "source_antigen_sequence_sha256",
            ),
        ):
            candidate_ids = [record["instance_id"], *record[candidate_field]]
            direction_rows: list[dict[str, Any]] = []
            for candidate_id in candidate_ids:
                candidate = index[candidate_id]
                key = (
                    (
                        "ordered",
                        candidate["source_antibody_sequence_sha256"],
                        antigen_hash,
                    )
                    if direction == "fixed_antigen_wrong_antibody"
                    else (
                        "ordered",
                        antibody_hash,
                        candidate["source_antigen_sequence_sha256"],
                    )
                )
                probability = float(scores[key])
                row = {
                    "anchor_instance_id": record["instance_id"],
                    "candidate_endpoint_sha256": candidate[endpoint_field],
                    "candidate_instance_id": candidate_id,
                    "canonical_pair_sha256": record["canonical_pair_sha256"],
                    "dataset": record["dataset"],
                    "direction": direction,
                    "is_cognate": candidate_id == record["instance_id"],
                    "official_probability": probability,
                    "probability_logit": stable_logit(probability),
                }
                predictions.append(row)
                direction_rows.append(row)
            instance_scores = [float(row["official_probability"]) for row in direction_rows]
            instance_rank = average_tie_rank(cognate_score, instance_scores)
            collapsed: dict[str, float] = {}
            for row in direction_rows:
                digest = str(row["candidate_endpoint_sha256"])
                previous = collapsed.setdefault(
                    digest, float(row["official_probability"])
                )
                if previous != float(row["official_probability"]):
                    raise RuntimeError("Exact endpoint duplicates received different scores")
            collapsed_rank = average_tie_rank(cognate_score, collapsed.values())
            anchor[f"{direction}.instance_candidate_count"] = len(instance_scores)
            anchor[f"{direction}.instance_mrr"] = 1.0 / instance_rank
            anchor[f"{direction}.instance_recall_at_1"] = float(instance_rank == 1.0)
            anchor[f"{direction}.unique_endpoint_candidate_count"] = len(collapsed)
            anchor[f"{direction}.unique_endpoint_mrr"] = 1.0 / collapsed_rank
            anchor[f"{direction}.unique_endpoint_recall_at_1"] = float(
                collapsed_rank == 1.0
            )
        for policy in ("instance", "unique_endpoint"):
            anchor[f"bidirectional.{policy}_mrr"] = math.fsum(
                anchor[f"{direction}.{policy}_mrr"]
                for direction in (
                    "fixed_antigen_wrong_antibody",
                    "fixed_antibody_wrong_antigen",
                )
            ) / 2
            anchor[f"bidirectional.{policy}_recall_at_1"] = math.fsum(
                anchor[f"{direction}.{policy}_recall_at_1"]
                for direction in (
                    "fixed_antigen_wrong_antibody",
                    "fixed_antibody_wrong_antigen",
                )
            ) / 2
        anchors.append(anchor)
    metrics: dict[str, Any] = {}
    for policy in ("unique_endpoint", "instance"):
        for direction in (
            "fixed_antigen_wrong_antibody",
            "fixed_antibody_wrong_antigen",
            "bidirectional",
        ):
            for metric in ("mrr", "recall_at_1"):
                field = f"{direction}.{policy}_{metric}"
                metrics[field] = summarize_anchor_metrics(anchors, field)
    for direction in (
        "fixed_antigen_wrong_antibody",
        "fixed_antibody_wrong_antigen",
    ):
        counts = [
            int(row[f"{direction}.unique_endpoint_candidate_count"])
            for row in anchors
        ]
        metrics[f"{direction}.unique_endpoint_candidate_count"] = {
            "maximum": max(counts),
            "median": statistics.median(counts),
            "minimum": min(counts),
        }
    metrics["pair_order_probability_absolute_delta"] = {
        "maximum": 0.0,
        "mean": 0.0,
        "reason": "The official PPLM-PPI head averages A-B and B-A feature-order predictions for every ensemble member.",
        "unique_canonical_exact_pair_count": len(
            {record["canonical_pair_sha256"] for record in records}
        ),
    }
    return predictions, metrics


def normalize_clean_room_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    normalized: dict[str, Tensor] = {}
    for raw_name, tensor in state.items():
        name = str(raw_name)
        for prefix in ("encoder.sentence_encoder.", "encoder."):
            if name.startswith(prefix):
                name = name.removeprefix(prefix)
                break
        if name.endswith(".self_attn.inter_attn_weight"):
            name = name.removesuffix("inter_attn_weight") + "cross_scale"
        if name in normalized:
            raise ValueError(f"Duplicate normalized PPLM state key: {name}")
        normalized[name] = tensor
    duplicate = normalized.pop("lm_head.weight", None)
    embedding = normalized.get("embed_tokens.weight")
    if (
        duplicate is None
        or embedding is None
        or duplicate.shape != embedding.shape
        or not torch.equal(duplicate, embedding)
    ):
        raise ValueError("PPLM tied token and LM-head weights differ")
    contact = {
        name: normalized.pop(name, None)
        for name in (
            "contact_head.regression.bias",
            "contact_head.regression.weight",
        )
    }
    if contact["contact_head.regression.bias"] is None or contact[
        "contact_head.regression.weight"
    ] is None:
        raise ValueError("PPLM auxiliary contact-head state is absent")
    return normalized


def load_models(
    *,
    source_root: Path,
    implementation_root: Path,
    checkpoint_path: Path,
    head_path: Path,
    device: torch.device,
    include_official_reference: bool,
) -> tuple[nn.Module, nn.Module | None, nn.ModuleList, nn.ModuleList, Any]:
    sys.path.insert(0, str(source_root))
    from pplm import Alphabet, PPLM  # type: ignore[import-not-found]
    from pplm_ppi import PPLM_PPI  # type: ignore[import-not-found]

    sys.path.insert(0, str(implementation_root / "src"))
    from antibody_pretraining.pair_aware_esm2 import (  # type: ignore[import-not-found]
        PairAwareESM2,
        PairAwareESM2Config,
    )

    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("PPLM checkpoint is not a mapping")
    parameters = checkpoint.get("param")
    state = checkpoint.get("model")
    if not isinstance(parameters, Mapping) or not isinstance(state, Mapping):
        raise ValueError("PPLM checkpoint lacks param/model mappings")
    expected = {
        "encoder_layers": 33,
        "encoder_embed_dim": 1280,
        "encoder_attention_heads": 20,
    }
    if any(int(parameters.get(key, -1)) != value for key, value in expected.items()):
        raise ValueError("PPLM checkpoint architecture differs")
    config = PairAwareESM2Config(
        layer_count=33,
        embed_dimension=1280,
        head_count=20,
        ffn_dimension=5120,
        token_dropout=False,
        attention_implementation="dense",
        cross_logit_bias=1.0,
        maximum_tokens=1024,
    )
    model = PairAwareESM2(config)
    model.load_state_dict(normalize_clean_room_state(state), strict=True)
    official_reference = None
    if include_official_reference:
        official_reference = PPLM(
            num_layers=33,
            embed_dim=1280,
            attention_heads=20,
            token_dropout=False,
            alphabet=Alphabet.from_architecture(),
        )
        incompatible = official_reference.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(
                "Official PPLM model state differs: "
                f"missing={incompatible.missing_keys} "
                f"unexpected={incompatible.unexpected_keys}"
            )
    head_states = torch.load(
        head_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(head_states, Mapping) or set(head_states) != {"mean", "max"}:
        raise ValueError("PPLM-PPI head checkpoint schema differs")
    heads: dict[str, nn.ModuleList] = {}
    for branch in ("mean", "max"):
        states = head_states[branch]
        if not isinstance(states, Sequence) or len(states) != 5:
            raise ValueError(f"PPLM-PPI {branch} ensemble count differs")
        modules = nn.ModuleList()
        for member_state in states:
            head = PPLM_PPI()
            head.load_state_dict(member_state, strict=True)
            modules.append(head.eval())
        heads[branch] = modules
    model = model.to(device).eval()
    if official_reference is not None:
        official_reference = official_reference.to(device).eval()
    mean_heads = heads["mean"].to(device)
    max_heads = heads["max"].to(device)
    del checkpoint, parameters, state, head_states, heads
    return (
        model,
        official_reference,
        mean_heads,
        max_heads,
        Alphabet.from_architecture(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("asep", "sabdab2"), required=True)
    parser.add_argument("--pair-panel", type=Path, required=True)
    parser.add_argument("--pair-panel-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--implementation-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--head-audit", type=Path, required=True)
    parser.add_argument("--equivalence-smoke-summary", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--maximum-batch-size", type=int, default=4)
    parser.add_argument("--maximum-attention-cells", type=int, default=1_048_576)
    parser.add_argument("--smoke-anchor-count", type=int, default=0)
    arguments = parser.parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("PPLM retrieval must run under Slurm gpu_4090")
    if sha256_file(arguments.protocol) != arguments.protocol_sha256:
        raise ValueError("PPLM retrieval protocol SHA-256 differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_BEFORE_GPU_EVALUATION"
        or protocol.get("authorization", {}).get("test_access") is not False
        or protocol.get("authorization", {}).get("development_contact") is not False
    ):
        raise ValueError("PPLM retrieval protocol authorization differs")
    if sha256_file(arguments.checkpoint) != PPLM_CHECKPOINT_SHA256:
        raise ValueError("PPLM checkpoint SHA-256 differs")
    if sha256_file(arguments.head) != PPLM_HEAD_SHA256:
        raise ValueError("PPLM-PPI head SHA-256 differs")
    if sha256_file(arguments.head_audit) != HEAD_AUDIT_SHA256:
        raise ValueError("Pair-retrieval head audit SHA-256 differs")
    head_audit = json.loads(arguments.head_audit.read_text(encoding="utf-8"))
    if (
        head_audit.get("status") != "PAIR_RETRIEVAL_HEADS_COMPATIBLE"
        or head_audit.get("authorization", {}).get("frozen_development_pair_retrieval")
        is not True
        or head_audit.get("test_access") is not False
    ):
        raise ValueError("Pair-retrieval head audit authorization differs")
    source_files = verify_source(arguments.source_root)
    implementation_files = verify_implementation(arguments.implementation_root)
    authority_records = load_inputs(
        arguments.pair_panel, arguments.pair_panel_receipt, arguments.dataset
    )
    if arguments.smoke_anchor_count < 0:
        raise ValueError("smoke-anchor-count cannot be negative")
    is_smoke = arguments.smoke_anchor_count > 0
    verified_smoke_gate = None
    if not is_smoke:
        if arguments.equivalence_smoke_summary is None:
            raise ValueError("Full PPLM retrieval requires an equivalence smoke summary")
        required_smoke_sha256 = protocol.get("equivalence_gate", {}).get(
            "required_smoke_summary_sha256"
        )
        if not isinstance(required_smoke_sha256, str):
            raise ValueError("Full PPLM protocol lacks a smoke-summary SHA-256")
        verified_smoke_gate = verify_smoke_gate(
            arguments.equivalence_smoke_summary, required_smoke_sha256
        )
    records = (
        authority_records[: arguments.smoke_anchor_count]
        if is_smoke
        else authority_records
    )
    if not records:
        raise ValueError("PPLM retrieval anchor selection is empty")
    pairs, index = retrieval_pairs(records, authority_records)
    device = torch.device("cuda")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 4090"
    ):
        raise RuntimeError("Exactly one RTX 4090 is required")
    torch.manual_seed(20260822)
    torch.cuda.manual_seed_all(20260822)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, official_reference, mean_heads, max_heads, alphabet = load_models(
        source_root=arguments.source_root,
        implementation_root=arguments.implementation_root,
        checkpoint_path=arguments.checkpoint,
        head_path=arguments.head,
        device=device,
        include_official_reference=is_smoke,
    )
    if is_smoke:
        if official_reference is None:
            raise RuntimeError("PPLM smoke requires the official reference model")
        equivalence = equivalence_audit(
            model,
            official_reference,
            mean_heads,
            max_heads,
            alphabet,
            device,
        )
    else:
        if verified_smoke_gate is None:
            raise RuntimeError("Verified PPLM smoke gate is absent")
        equivalence = verified_smoke_gate
    del official_reference
    torch.cuda.empty_cache()
    probabilities, runtime = score_pairs(
        pairs=pairs,
        model=model,
        mean_heads=mean_heads,
        max_heads=max_heads,
        alphabet=alphabet,
        device=device,
        maximum_batch_size=arguments.maximum_batch_size,
        maximum_attention_cells=arguments.maximum_attention_cells,
    )
    predictions, metrics = evaluate(records, index, probabilities)
    prediction_payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in predictions
    )
    publish_create_only(arguments.predictions, prediction_payload)
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    summary = {
        "authorization": {
            "development_pair_retrieval": not is_smoke,
            "development_contact": False,
            "production_training": False,
            "test_access": False,
        },
        "claim_boundary": (
            "Non-claim implementation and runtime smoke; no quality conclusion is authorized."
            if is_smoke
            else "Official PPLM 650M checkpoint plus ten-member PPLM-PPI ensemble on one frozen development dataset. Decoys are ranking controls rather than biological noninteraction labels; test, contact, calibration, biological-improvement, commercial-use, and SOTA claims remain closed."
        ),
        "dataset": arguments.dataset,
        "equivalence": equivalence,
        "format_version": "abpairfm_pplm_development_retrieval_result_v1",
        "input": {
            "anchor_instance_count": len(records),
            "unique_canonical_exact_pair_count": len(
                {record["canonical_pair_sha256"] for record in records}
            ),
        },
        "license_boundary": {
            "commercial_use_authorized": False,
            "source_repository_license": "PolyForm Noncommercial"
        },
        "metrics": metrics,
        "model": {
            "checkpoint_sha256": PPLM_CHECKPOINT_SHA256,
            "head_ensemble_member_count": len(mean_heads) + len(max_heads),
            "head_sha256": PPLM_HEAD_SHA256,
            "implementation_file_sha256": implementation_files,
            "source_file_sha256": source_files,
            "source_revision": PPLM_REVISION,
        },
        "predictions": {
            "bytes": arguments.predictions.stat().st_size,
            "row_count": len(predictions),
            "sha256": sha256_file(arguments.predictions),
        },
        "protocol_sha256": arguments.protocol_sha256,
        "runtime": {
            **runtime,
            "gpu_hours": runtime["elapsed_seconds"] / 3600,
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_allocated_memory_fraction": runtime["peak_cuda_bytes"] / total_memory,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "status": (
            "SMOKE_NONCLAIM_COMPLETE"
            if is_smoke
            else "FROZEN_DEVELOPMENT_RETRIEVAL_COMPLETE"
        ),
        "test_access": False,
    }
    publish_create_only(
        arguments.summary,
        (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
