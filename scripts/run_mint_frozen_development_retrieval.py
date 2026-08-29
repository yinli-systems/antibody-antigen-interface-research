#!/usr/bin/env python3
"""Run the frozen MINT development retrieval baseline on one dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn


MINT_REVISION = "06694b7606e2d00b76ec58daf5c7aecdaf7cd283"
MINT_CHECKPOINT_SHA256 = "84a4016365997cd9f0bccb07d746fa8f076ffd8e45aa0cbcf4e50a037161a342"
MINT_HEAD_SHA256 = "702849af78e245c7596d8c032391da52ff6657e9173af380e29a8ea3858fdab9"
PAIR_PANEL_SHA256 = "18a7efcc4f1cb5499be32938a4819208f5bb6e2095acdda3098e71496458fce2"
PAIR_PANEL_RECEIPT_SHA256 = "078dc59a5c1caba9fa3ecae4361a5bcd0fa3922f586039ff3a44d11e998f1346"
HEAD_AUDIT_SHA256 = "d4d6700057038ac0c71f18125329f7f1cd299925003bf7cafc2e33c6110ec586"
MAX_ENDPOINT_RESIDUES = 510
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260822


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(source_root: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != MINT_REVISION:
        raise ValueError("MINT source revision differs")
    tracked = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if tracked:
        raise ValueError("MINT source has modified tracked files")
    return revision


def endpoint_windows(sequence: str) -> list[str]:
    if len(sequence) <= MAX_ENDPOINT_RESIDUES:
        return [sequence]
    starts = sorted(
        {0, (len(sequence) - MAX_ENDPOINT_RESIDUES) // 2, len(sequence) - MAX_ENDPOINT_RESIDUES}
    )
    return [sequence[start : start + MAX_ENDPOINT_RESIDUES] for start in starts]


def average_tie_rank(cognate_score: float, scores: Iterable[float]) -> float:
    values = list(scores)
    greater = sum(value > cognate_score for value in values)
    tied_decoys = sum(value == cognate_score for value in values) - 1
    if tied_decoys < 0:
        raise ValueError("Cognate score is absent from candidate scores")
    return 1.0 + greater + 0.5 * tied_decoys


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute an empty percentile")
    return ordered[round((len(ordered) - 1) * fraction)]


def bootstrap_mean_ci(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("Cannot bootstrap empty values")
    generator = random.Random(BOOTSTRAP_SEED)
    count = len(values)
    draws = [
        math.fsum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def summarize_anchor_metrics(
    rows: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    values = [float(row[field]) for row in rows]
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["canonical_pair_sha256"]), []).append(float(row[field]))
    group_values = [math.fsum(group) / len(group) for group in grouped.values()]
    return {
        "canonical_exact_pair_macro": math.fsum(group_values) / len(group_values),
        "canonical_exact_pair_paired_bootstrap_ci95": bootstrap_mean_ci(group_values),
        "instance_macro_secondary": math.fsum(values) / len(values),
        "unique_canonical_exact_pair_count": len(group_values),
    }


def publish_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite MINT retrieval artifact: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


class BernettMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(2560, 2560),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2560, 1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.project(value)


def load_inputs(
    panel_path: Path,
    receipt_path: Path,
    dataset: str,
) -> list[dict[str, Any]]:
    if sha256_file(panel_path) != PAIR_PANEL_SHA256:
        raise ValueError("Development pair panel SHA-256 differs")
    if sha256_file(receipt_path) != PAIR_PANEL_RECEIPT_SHA256:
        raise ValueError("Development pair panel receipt SHA-256 differs")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("format_version") != "abpairfm_development_pair_panel_v1"
        or receipt.get("status") != "FROZEN_DEVELOPMENT_PAIR_PANEL"
        or receipt.get("authorization", {}).get("development_retrieval") is not True
        or receipt.get("authorization", {}).get("development_contact_labels") is not False
        or receipt.get("test_access") is not False
    ):
        raise ValueError("Development pair panel receipt authorization differs")
    records = [
        json.loads(line)
        for line in panel_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [record for record in records if record.get("dataset") == dataset]
    if not selected or any(record.get("role") != "development_validation" for record in selected):
        raise ValueError("Selected development dataset is empty or has a forbidden role")
    expected = 170 if dataset == "asep" else 549
    if len(selected) != expected:
        raise ValueError("Selected development dataset count differs")
    identifiers = {record["instance_id"] for record in selected}
    if len(identifiers) != len(selected):
        raise ValueError("Development instances are duplicated")
    for record in selected:
        for key in ("wrong_antibody_instance_ids", "wrong_antigen_instance_ids"):
            candidates = record[key]
            if len(candidates) != 31 or not set(candidates).issubset(identifiers):
                raise ValueError("Frozen candidate panel differs")
    return selected


def register_pair(
    pairs: dict[tuple[str, str, str], tuple[str, str]],
    key: tuple[str, str, str],
    left: str,
    right: str,
) -> None:
    previous = pairs.setdefault(key, (left, right))
    if previous != (left, right):
        raise ValueError("Endpoint digest maps to inconsistent sequences")


def retrieval_pairs(
    records: list[dict[str, Any]],
    authority_records: list[dict[str, Any]] | None = None,
) -> tuple[
    dict[tuple[str, str, str], tuple[str, str]],
    dict[str, dict[str, Any]],
    dict[str, tuple[str, str, str]],
]:
    index = {
        record["instance_id"]: record
        for record in (authority_records if authority_records is not None else records)
    }
    pairs: dict[tuple[str, str, str], tuple[str, str]] = {}
    swapped: dict[str, tuple[str, str, str]] = {}
    for record in records:
        antibody_hash = record["source_antibody_sequence_sha256"]
        antigen_hash = record["source_antigen_sequence_sha256"]
        key = ("ordered", antibody_hash, antigen_hash)
        register_pair(
            pairs, key, record["antibody_endpoint_sequence"], record["antigen_sequence"]
        )
        swap_key = ("swapped", record["canonical_pair_sha256"], "")
        register_pair(
            pairs, swap_key, record["antigen_sequence"], record["antibody_endpoint_sequence"]
        )
        swapped[record["canonical_pair_sha256"]] = swap_key
        for identifier in record["wrong_antibody_instance_ids"]:
            candidate = index[identifier]
            key = ("ordered", candidate["source_antibody_sequence_sha256"], antigen_hash)
            register_pair(
                pairs,
                key,
                candidate["antibody_endpoint_sequence"],
                record["antigen_sequence"],
            )
        for identifier in record["wrong_antigen_instance_ids"]:
            candidate = index[identifier]
            key = ("ordered", antibody_hash, candidate["source_antigen_sequence_sha256"])
            register_pair(
                pairs,
                key,
                record["antibody_endpoint_sequence"],
                candidate["antigen_sequence"],
            )
    return pairs, index, swapped


def batch_work(
    work: list[tuple[tuple[str, str, str], str, str]],
    maximum_batch_size: int,
    maximum_attention_cells: int,
) -> list[list[tuple[tuple[str, str, str], str, str]]]:
    ordered = sorted(work, key=lambda item: len(item[1]) + len(item[2]) + 4)
    batches: list[list[tuple[tuple[str, str, str], str, str]]] = []
    current: list[tuple[tuple[str, str, str], str, str]] = []
    current_max = 0
    for item in ordered:
        length = len(item[1]) + len(item[2]) + 4
        next_max = max(current_max, length)
        if current and (
            len(current) >= maximum_batch_size
            or next_max * next_max * (len(current) + 1) > maximum_attention_cells
        ):
            batches.append(current)
            current = []
            current_max = 0
        current.append(item)
        current_max = max(current_max, length)
    if current:
        batches.append(current)
    return batches


def score_pairs(
    *,
    pairs: dict[tuple[str, str, str], tuple[str, str]],
    model: nn.Module,
    head: nn.Module,
    alphabet: Any,
    device: torch.device,
    maximum_batch_size: int,
    maximum_attention_cells: int,
) -> tuple[dict[tuple[str, str, str], float], dict[str, Any]]:
    token_cache: dict[str, list[int]] = {}

    def encode(sequence: str) -> list[int]:
        if sequence not in token_cache:
            token_cache[sequence] = alphabet.encode("<cls>" + sequence + "<eos>")
        return token_cache[sequence]

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
    scored_pair_windows = 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(batches, start=1):
        encoded = [(encode(left), encode(right)) for _, left, right in batch]
        maximum = max(len(left) + len(right) for left, right in encoded)
        tokens = torch.full(
            (len(batch), maximum), alphabet.padding_idx, dtype=torch.long, device=device
        )
        chain_ids = torch.zeros((len(batch), maximum), dtype=torch.int32, device=device)
        for row, (left, right) in enumerate(encoded):
            left_end = len(left)
            right_end = left_end + len(right)
            tokens[row, :left_end] = torch.tensor(left, dtype=torch.long, device=device)
            tokens[row, left_end:right_end] = torch.tensor(right, dtype=torch.long, device=device)
            chain_ids[row, left_end:right_end] = 1
            scored_residues += len(left) + len(right) - 4
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            output = model(tokens, chain_ids, repr_layers=[33])
        representation = output["representations"][33].float()
        residue_mask = (
            tokens.ne(model.cls_idx)
            & tokens.ne(model.eos_idx)
            & tokens.ne(model.padding_idx)
        )
        left_mask = residue_mask & chain_ids.eq(0)
        right_mask = residue_mask & chain_ids.eq(1)
        left_mean = (representation * left_mask.unsqueeze(-1)).sum(1) / left_mask.sum(1, keepdim=True)
        right_mean = (representation * right_mask.unsqueeze(-1)).sum(1) / right_mask.sum(1, keepdim=True)
        logits = head(torch.cat((left_mean, right_mean), dim=-1)).squeeze(-1)
        for item, logit in zip(batch, logits.tolist()):
            key = item[0]
            sums[key] += float(logit)
            counts[key] += 1
        scored_pair_windows += len(batch)
        if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(batches):
            print(
                json.dumps(
                    {
                        "batch": batch_index,
                        "batches": len(batches),
                        "pair_windows_scored": scored_pair_windows,
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
            raise RuntimeError("MINT pair window count differs")
    scores = {key: sums[key] / counts[key] for key in pairs}
    return scores, {
        "batch_count": len(batches),
        "elapsed_seconds": elapsed,
        "nonpadding_residues": scored_residues,
        "nonpadding_residues_per_second": scored_residues / elapsed,
        "pair_windows": scored_pair_windows,
        "pair_windows_per_second": scored_pair_windows / elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "unique_ordered_and_swap_pair_count": len(pairs),
    }


def evaluate(
    records: list[dict[str, Any]],
    index: dict[str, dict[str, Any]],
    swapped: dict[str, tuple[str, str, str]],
    scores: Mapping[tuple[str, str, str], float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    for record in records:
        antibody_hash = record["source_antibody_sequence_sha256"]
        antigen_hash = record["source_antigen_sequence_sha256"]
        cognate_key = ("ordered", antibody_hash, antigen_hash)
        cognate_score = float(scores[cognate_key])
        anchor_result: dict[str, Any] = {
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
            direction_rows = []
            for candidate_id in candidate_ids:
                candidate = index[candidate_id]
                if direction == "fixed_antigen_wrong_antibody":
                    pair_key = (
                        "ordered",
                        candidate["source_antibody_sequence_sha256"],
                        antigen_hash,
                    )
                else:
                    pair_key = (
                        "ordered",
                        antibody_hash,
                        candidate["source_antigen_sequence_sha256"],
                    )
                score = float(scores[pair_key])
                row = {
                    "anchor_instance_id": record["instance_id"],
                    "candidate_endpoint_sha256": candidate[endpoint_field],
                    "candidate_instance_id": candidate_id,
                    "canonical_pair_sha256": record["canonical_pair_sha256"],
                    "dataset": record["dataset"],
                    "direction": direction,
                    "is_cognate": candidate_id == record["instance_id"],
                    "probability": sigmoid(score),
                    "raw_logit": score,
                }
                predictions.append(row)
                direction_rows.append(row)
            instance_scores = [float(row["raw_logit"]) for row in direction_rows]
            instance_rank = average_tie_rank(cognate_score, instance_scores)
            collapsed: dict[str, float] = {}
            for row in direction_rows:
                digest = str(row["candidate_endpoint_sha256"])
                previous = collapsed.setdefault(digest, float(row["raw_logit"]))
                if previous != float(row["raw_logit"]):
                    raise RuntimeError("Exact endpoint duplicates received different scores")
            collapsed_rank = average_tie_rank(cognate_score, collapsed.values())
            anchor_result[f"{direction}.instance_candidate_count"] = len(instance_scores)
            anchor_result[f"{direction}.instance_mrr"] = 1.0 / instance_rank
            anchor_result[f"{direction}.instance_recall_at_1"] = float(instance_rank == 1.0)
            anchor_result[f"{direction}.unique_endpoint_candidate_count"] = len(collapsed)
            anchor_result[f"{direction}.unique_endpoint_mrr"] = 1.0 / collapsed_rank
            anchor_result[f"{direction}.unique_endpoint_recall_at_1"] = float(
                collapsed_rank == 1.0
            )
        for policy in ("instance", "unique_endpoint"):
            anchor_result[f"bidirectional.{policy}_mrr"] = math.fsum(
                anchor_result[f"{direction}.{policy}_mrr"]
                for direction in (
                    "fixed_antigen_wrong_antibody",
                    "fixed_antibody_wrong_antigen",
                )
            ) / 2
            anchor_result[f"bidirectional.{policy}_recall_at_1"] = math.fsum(
                anchor_result[f"{direction}.{policy}_recall_at_1"]
                for direction in (
                    "fixed_antigen_wrong_antibody",
                    "fixed_antibody_wrong_antigen",
                )
            ) / 2
        anchors.append(anchor_result)
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
            int(row[f"{direction}.unique_endpoint_candidate_count"]) for row in anchors
        ]
        metrics[f"{direction}.unique_endpoint_candidate_count"] = {
            "maximum": max(counts),
            "median": statistics.median(counts),
            "minimum": min(counts),
        }
    order_deltas = []
    observed: set[str] = set()
    for record in records:
        canonical = record["canonical_pair_sha256"]
        if canonical in observed:
            continue
        observed.add(canonical)
        ordered = scores[
            (
                "ordered",
                record["source_antibody_sequence_sha256"],
                record["source_antigen_sequence_sha256"],
            )
        ]
        order_deltas.append(abs(float(ordered) - float(scores[swapped[canonical]])))
    metrics["pair_order_raw_logit_absolute_delta"] = {
        "maximum": max(order_deltas),
        "mean": math.fsum(order_deltas) / len(order_deltas),
        "median": statistics.median(order_deltas),
        "p95": percentile(order_deltas, 0.95),
        "unique_canonical_exact_pair_count": len(order_deltas),
    }
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("asep", "sabdab2"), required=True)
    parser.add_argument("--pair-panel", type=Path, required=True)
    parser.add_argument("--pair-panel-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--head-audit", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--maximum-batch-size", type=int, default=8)
    parser.add_argument("--maximum-attention-cells", type=int, default=4_194_304)
    parser.add_argument("--smoke-anchor-count", type=int, default=0)
    arguments = parser.parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("MINT retrieval must run under Slurm gpu_4090")
    if sha256_file(arguments.protocol) != arguments.protocol_sha256:
        raise ValueError("MINT retrieval protocol SHA-256 differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_BEFORE_GPU_EVALUATION"
        or protocol.get("authorization", {}).get("test_access") is not False
        or protocol.get("authorization", {}).get("development_contact") is not False
    ):
        raise ValueError("MINT retrieval protocol authorization differs")
    if sha256_file(arguments.checkpoint) != MINT_CHECKPOINT_SHA256:
        raise ValueError("MINT checkpoint SHA-256 differs")
    if sha256_file(arguments.head) != MINT_HEAD_SHA256:
        raise ValueError("MINT head SHA-256 differs")
    if sha256_file(arguments.head_audit) != HEAD_AUDIT_SHA256:
        raise ValueError("Retrieval-head audit SHA-256 differs")
    head_audit = json.loads(arguments.head_audit.read_text(encoding="utf-8"))
    if (
        head_audit.get("status") != "PAIR_RETRIEVAL_HEADS_COMPATIBLE"
        or head_audit.get("authorization", {}).get("frozen_development_pair_retrieval")
        is not True
        or head_audit.get("test_access") is not False
    ):
        raise ValueError("Retrieval-head audit authorization differs")
    revision = verify_source(arguments.source_root)
    authority_records = load_inputs(
        arguments.pair_panel, arguments.pair_panel_receipt, arguments.dataset
    )
    if arguments.smoke_anchor_count < 0:
        raise ValueError("smoke-anchor-count cannot be negative")
    is_smoke = arguments.smoke_anchor_count > 0
    records = (
        authority_records[: arguments.smoke_anchor_count]
        if is_smoke
        else authority_records
    )
    if not records:
        raise ValueError("MINT retrieval anchor selection is empty")
    pairs, index, swapped = retrieval_pairs(records, authority_records)
    sys.path.insert(0, str(arguments.source_root))
    from mint.data import Alphabet  # type: ignore[import-not-found]
    from mint.model.esm2 import ESM2  # type: ignore[import-not-found]

    device = torch.device("cuda")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("Exactly one RTX 4090 is required")
    torch.manual_seed(20260822)
    torch.cuda.manual_seed_all(20260822)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = ESM2(
        num_layers=33,
        embed_dim=1280,
        attention_heads=20,
        token_dropout=True,
        use_multimer=True,
    )
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(
            arguments.checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("MINT checkpoint state is missing")
    normalized = {str(key).removeprefix("model."): value for key, value in state.items()}
    model.load_state_dict(normalized, strict=True)
    head = BernettMLP()
    head_state = torch.load(arguments.head, map_location="cpu", mmap=True, weights_only=True)
    head.load_state_dict(head_state, strict=True)
    model = model.to(device).eval()
    head = head.to(device).eval()
    del checkpoint, state, normalized, head_state
    scores, runtime = score_pairs(
        pairs=pairs,
        model=model,
        head=head,
        alphabet=Alphabet.from_architecture("ESM-1b"),
        device=device,
        maximum_batch_size=arguments.maximum_batch_size,
        maximum_attention_cells=arguments.maximum_attention_cells,
    )
    predictions, metrics = evaluate(records, index, swapped, scores)
    prediction_payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
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
            "Non-claim runtime smoke on a deterministic prefix of the frozen development panel; no quality conclusion is authorized."
            if is_smoke
            else "Official MINT checkpoint plus Bernett PPI head on one frozen development dataset. Decoys are ranking controls rather than biological noninteraction labels; test, contact, calibration, biological-improvement, and SOTA claims remain closed."
        ),
        "dataset": arguments.dataset,
        "format_version": "abpairfm_mint_development_retrieval_result_v1",
        "input": {
            "anchor_instance_count": len(records),
            "pair_panel_sha256": PAIR_PANEL_SHA256,
            "pair_panel_receipt_sha256": PAIR_PANEL_RECEIPT_SHA256,
            "unique_canonical_exact_pair_count": len(
                {record["canonical_pair_sha256"] for record in records}
            ),
        },
        "metrics": metrics,
        "model": {
            "checkpoint_sha256": MINT_CHECKPOINT_SHA256,
            "head_sha256": MINT_HEAD_SHA256,
            "source_revision": revision,
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
        (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
