#!/usr/bin/env python3
"""Run the frozen distributed PINDER B-canary from an immutable Stage-A0 DCP."""

from __future__ import annotations

import argparse
from datetime import timedelta
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import statistics
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from antibody_pretraining.pair_aware_distributed import (  # noqa: E402
    fully_shard_pair_aware_esm2,
    load_authenticated_esm2_into_fsdp2,
    reshard_pair_aware_esm2,
)
from antibody_pretraining.pair_aware_esm2 import (  # noqa: E402
    ESM2_ALPHABET,
    PairAwareESM2,
    PairAwareESM2Config,
)
from antibody_pretraining.paircontact_pretraining import (  # noqa: E402
    BipartiteEdgeRefinementContactHead,
    SymmetricLowRankPairContactHead,
    build_partner_swap_permutation,
    paircontact_pretraining_loss,
    paircontact_ranking_metrics,
)
from interaction_first.frontier_fsdp2 import (  # noqa: E402
    load_dcp_generation,
    save_dcp_generation,
    verify_dcp_generation,
)


TOKEN_INDEX = {token: index for index, token in enumerate(ESM2_ALPHABET)}
OUTPUT_GENERATION = "b-canary-000000"


class CanarySystem(nn.Module):
    def __init__(self, backbone: PairAwareESM2, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(repository_root: Path) -> str:
    paths = (
        "remote_tools/abpairfm_distributed_contact_pretraining_v1/run_pinder_b_canary_v1.py",
        "src/antibody_pretraining/pair_aware_attention.py",
        "src/antibody_pretraining/pair_aware_distributed.py",
        "src/antibody_pretraining/pair_aware_esm2.py",
        "src/antibody_pretraining/paircontact_pretraining.py",
        "src/interaction_first/frontier_fsdp2.py",
    )
    digest = hashlib.sha256()
    for relative in paths:
        payload = (repository_root / relative).read_bytes()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _publish_create_only(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")
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
        raise FileExistsError(f"Refusing to overwrite output: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_progress(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-initial-checkpoint-size", type=int, required=True)
    parser.add_argument("--expected-initial-checkpoint-sha256", required=True)
    parser.add_argument("--input-checkpoint-root", type=Path, required=True)
    parser.add_argument("--input-generation", required=True)
    parser.add_argument("--expected-input-content-sha256", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, default=4)
    return parser.parse_args()


def _load_protocol(arguments: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(arguments.protocol) != arguments.expected_protocol_sha256:
        raise RuntimeError("B-canary protocol SHA-256 differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    stage = protocol.get("stage", {}).get("name", "B-canary")
    authorization = protocol.get("authorization", {})
    expected = {
        "B-canary": {
            "format": "abpairfm_pinder_b_canary_v1",
            "status": "FROZEN_B_CANARY_AUTHORIZED",
            "authorized": authorization.get("B_canary_training") is True,
            "next_authorized": authorization.get("B32K_training") is False,
            "partitions": {"gpu_4090"},
        },
        "B32K": {
            "format": "abpairfm_pinder_b32k_v1",
            "status": "FROZEN_B32K_AUTHORIZED",
            "authorized": authorization.get("B32K_training") is True,
            "next_authorized": authorization.get("B500K_training") is False,
            "partitions": {"gpu_4090", "gpu_5090"},
        },
    }.get(stage)
    optimization = protocol.get("optimization", {})
    data = protocol.get("data", {})
    weight_mean = data.get("system_weight_mean", data.get("canary_system_weight_mean"))
    if (
        expected is None
        or protocol.get("format_version") != expected["format"]
        or protocol.get("status") != expected["status"]
        or expected["authorized"] is not True
        or expected["next_authorized"] is not True
        or protocol.get("test_access") is not False
        or protocol.get("systems", {}).get("world_size") != arguments.expected_world_size
        or protocol.get("systems", {}).get("partition") not in expected["partitions"]
        or not isinstance(optimization.get("updates"), int)
        or optimization.get("updates", 0) < 1
        or not isinstance(optimization.get("accumulation_steps"), int)
        or optimization.get("accumulation_steps", 0) < 1
        or optimization.get("systems_per_rank") != 1
        or data.get("maximum_tokens", 1024) != 1024
        or not 1 <= int(protocol.get("model", {}).get("trainable_top_layers", 0)) <= 36
        or isinstance(weight_mean, bool)
        or not isinstance(weight_mean, (int, float))
        or not math.isfinite(float(weight_mean))
        or float(weight_mean) <= 0
        or int(data.get("holdout_systems", 16))
        != int(data.get("evaluation_groups", 4)) * arguments.expected_world_size
    ):
        raise RuntimeError(f"{stage} authorization or systems shape differs")
    checkpoint = protocol.get("checkpoint", {})
    if (
        checkpoint.get("generation") != arguments.input_generation
        or checkpoint.get("content_sha256") != arguments.expected_input_content_sha256
    ):
        raise RuntimeError(f"{stage} input checkpoint anchor differs")
    model = protocol.get("model", {})
    decoder = model.get("pair_decoder")
    if decoder not in {"symmetric_low_rank", "bipartite_edge_refinement"}:
        raise RuntimeError(f"{stage} pair decoder is not reviewed")
    if decoder == "bipartite_edge_refinement" and (
        not isinstance(model.get("edge_dimension"), int)
        or not 8 <= model["edge_dimension"] <= 128
        or not isinstance(model.get("refinement_steps"), int)
        or not 1 <= model["refinement_steps"] <= 4
    ):
        raise RuntimeError(f"{stage} bipartite decoder shape differs")
    loss = protocol.get("loss", {})
    pair_loss_mode = loss.get("pair_loss_mode", "adaptive_balanced")
    if pair_loss_mode not in {"adaptive_balanced", "balanced_focal"}:
        raise RuntimeError(f"{stage} pair loss is not reviewed")
    if pair_loss_mode == "balanced_focal" and (
        isinstance(loss.get("focal_gamma"), bool)
        or not isinstance(loss.get("focal_gamma"), (int, float))
        or not math.isfinite(float(loss["focal_gamma"]))
        or float(loss["focal_gamma"]) < 0.0
    ):
        raise RuntimeError(f"{stage} focal gamma differs")
    return protocol


def _build_pair_head(protocol: Mapping[str, Any]) -> nn.Module:
    model = protocol["model"]
    common = {
        "embedding_dimension": 2560,
        "pair_dimension": int(model["pair_dimension"]),
    }
    if model["pair_decoder"] == "symmetric_low_rank":
        return SymmetricLowRankPairContactHead(**common)
    if model["pair_decoder"] == "bipartite_edge_refinement":
        return BipartiteEdgeRefinementContactHead(
            **common,
            edge_dimension=int(model["edge_dimension"]),
            refinement_steps=int(model["refinement_steps"]),
        )
    raise RuntimeError("Pair decoder is not reviewed")


def _load_records(
    manifest: Path,
    receipt_path: Path,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    data = protocol["data"]
    if sha256_file(manifest) != data["manifest_sha256"]:
        raise RuntimeError("PINDER stage manifest SHA-256 differs")
    if sha256_file(receipt_path) != data["manifest_receipt_sha256"]:
        raise RuntimeError("PINDER stage manifest receipt SHA-256 differs")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    stage = protocol.get("stage", {}).get("name", "B-canary")
    receipt_contract = {
        "B-canary": (
            "PASS_B_CANARY_PAIR_LABELS_MODEL_INPUT_SEQUENCE_ONLY",
            "B_canary_training",
            512,
        ),
        "B32K": (
            "PASS_B32K_PAIR_LABELS_MODEL_INPUT_SEQUENCE_ONLY",
            "B32K_training",
            32_768,
        ),
    }.get(stage)
    if receipt_contract is None:
        raise RuntimeError("PINDER stage receipt contract is absent")
    receipt_status, authorization_key, expected_systems = receipt_contract
    if (
        receipt.get("status") != receipt_status
        or receipt.get("authorization", {}).get(authorization_key) is not True
        or receipt.get("test_access") is not False
        or receipt.get("checks", {}).get("bound_coordinates_published_as_model_input") is not False
        or receipt.get("counts", {}).get("systems") != expected_systems
    ):
        raise RuntimeError(f"{stage} manifest receipt is not eligible")
    records = []
    with gzip.open(manifest, "rt", encoding="ascii") as handle:
        for line in handle:
            record = json.loads(line)
            left = str(record["sequence1"])
            right = str(record["sequence2"])
            contacts = [tuple(pair) for pair in record["contact_pairs"]]
            if (
                record.get("source") != "pinder_train_bound_labels"
                or record.get("contact_supervision") is not True
                or not left
                or not right
                or len(left) + len(right) > 1020
                or not contacts
                or any(i < 0 or j < 0 or i >= len(left) or j >= len(right) for i, j in contacts)
                or not math.isfinite(float(record["system_weight"]))
                or float(record["system_weight"]) <= 0
            ):
                raise RuntimeError(f"{stage} record violates the frozen model-input contract")
            records.append(record)
    if len(records) != expected_systems or len({record["example_id"] for record in records}) != expected_systems:
        raise RuntimeError(f"{stage} manifest membership differs")
    return records


def _known_pair(record: Mapping[str, Any]) -> tuple[str, str]:
    return tuple(
        sorted((str(record["endpoint_exact1_sha256"]), str(record["endpoint_exact2_sha256"])))
    )


def _swap_allowed(
    group: Sequence[Mapping[str, Any]], known_pairs: set[tuple[str, str]]
) -> Tensor:
    size = len(group)
    allowed = torch.ones((size, size), dtype=torch.bool)
    for row, left_record in enumerate(group):
        left_exact = str(left_record["endpoint_exact1_sha256"])
        left_cluster = str(left_record["endpoint_cluster1"])
        for column, right_record in enumerate(group):
            right_exact = str(right_record["endpoint_exact2_sha256"])
            right_cluster = str(right_record["endpoint_cluster2"])
            endpoint_overlap = (
                left_exact
                in {
                    str(right_record["endpoint_exact1_sha256"]),
                    str(right_record["endpoint_exact2_sha256"]),
                }
                or right_exact
                in {
                    str(left_record["endpoint_exact1_sha256"]),
                    str(left_record["endpoint_exact2_sha256"]),
                }
            )
            cluster_overlap = (
                left_cluster
                in {
                    str(right_record["endpoint_cluster1"]),
                    str(right_record["endpoint_cluster2"]),
                }
                or right_cluster
                in {
                    str(left_record["endpoint_cluster1"]),
                    str(left_record["endpoint_cluster2"]),
                }
            )
            allowed[row, column] = (
                row != column
                and not endpoint_overlap
                and not cluster_overlap
                and tuple(sorted((left_exact, right_exact))) not in known_pairs
            )
    return allowed


def _pdb_disjoint_holdout(records: Sequence[dict[str, Any]], count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pdbs = sorted(
        {str(record["pdb_id"]) for record in records},
        key=lambda value: hashlib.sha256(f"1701\0holdout\0{value}".encode()).digest(),
    )
    selected_pdbs = set(pdbs[:count])
    representatives: dict[str, dict[str, Any]] = {}
    training = []
    for record in records:
        pdb = str(record["pdb_id"])
        if pdb in selected_pdbs:
            previous = representatives.get(pdb)
            if previous is None or str(record["example_id"]) < str(previous["example_id"]):
                representatives[pdb] = record
        else:
            training.append(record)
    holdout = [representatives[pdb] for pdb in sorted(representatives)]
    if len(holdout) != count or not training:
        raise RuntimeError("B-canary PDB-disjoint holdout construction differs")
    return training, holdout


def _legal_groups(
    records: Sequence[dict[str, Any]],
    *,
    group_count: int,
    group_size: int = 4,
    known_pairs: set[tuple[str, str]],
    seed: int,
    maximum_tokens: int = 1024,
) -> list[list[dict[str, Any]]]:
    if maximum_tokens < 5:
        raise ValueError("maximum_tokens must admit two nonempty chains")
    candidates = list(records)
    candidates.sort(
        key=lambda record: (
            len(str(record["sequence1"])) + len(str(record["sequence2"])),
            str(record["example_id"]),
        )
    )
    generator = random.Random(seed)
    blocks = [candidates[index : index + 32] for index in range(0, len(candidates), 32)]
    for block in blocks:
        generator.shuffle(block)
    generator.shuffle(blocks)
    candidates = [record for block in blocks for record in block]
    groups = []
    cursor = 0
    attempts = 0
    while len(groups) < group_count and attempts < len(candidates) * 20:
        group = [candidates[(cursor + offset) % len(candidates)] for offset in range(group_size)]
        cursor = (cursor + group_size) % len(candidates)
        attempts += 1
        maximum_left = max(len(str(record["sequence1"])) for record in group)
        maximum_right = max(len(str(record["sequence2"])) for record in group)
        if maximum_left + maximum_right + 4 > maximum_tokens:
            candidates[cursor % len(candidates)], candidates[(cursor + 1) % len(candidates)] = (
                candidates[(cursor + 1) % len(candidates)],
                candidates[cursor % len(candidates)],
            )
            continue
        try:
            build_partner_swap_permutation(
                _swap_allowed(group, known_pairs), seed=seed + len(groups)
            )
        except ValueError:
            candidates[cursor % len(candidates)], candidates[(cursor + 1) % len(candidates)] = (
                candidates[(cursor + 1) % len(candidates)],
                candidates[cursor % len(candidates)],
            )
            continue
        groups.append(group)
    if len(groups) != group_count:
        raise RuntimeError(
            f"B-canary could not construct legal {group_size}-way partner swaps"
        )
    return groups


def _tokenize(sequence: str) -> list[int]:
    unknown = TOKEN_INDEX["<unk>"]
    return [TOKEN_INDEX.get(residue, unknown) for residue in sequence]


def _model_input(
    left: str,
    right: str,
    *,
    seed: int,
    mask: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    first = [TOKEN_INDEX["<cls>"], *_tokenize(left), TOKEN_INDEX["<eos>"]]
    second = [TOKEN_INDEX["<cls>"], *_tokenize(right), TOKEN_INDEX["<eos>"]]
    tokens = torch.tensor([first + second], dtype=torch.long)
    chains = torch.tensor([[0] * len(first) + [1] * len(second)], dtype=torch.long)
    labels = torch.full_like(tokens, -100)
    if mask:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed % (2**63 - 1))
        for start, end in ((1, len(first) - 1), (len(first) + 1, len(first) + len(second) - 1)):
            count = max(1, math.ceil((end - start) * 0.15))
            selected = torch.randperm(end - start, generator=generator)[:count] + start
            labels[0, selected] = tokens[0, selected]
            replacement = torch.rand(count, generator=generator)
            tokens[0, selected[replacement < 0.8]] = TOKEN_INDEX["<mask>"]
            random_selected = selected[(replacement >= 0.8) & (replacement < 0.9)]
            if len(random_selected):
                tokens[0, random_selected] = torch.randint(
                    TOKEN_INDEX["L"], TOKEN_INDEX["C"] + 1, (len(random_selected),), generator=generator
                )
    return tokens, chains, labels


def _hidden_chains(hidden: Tensor, left_length: int, right_length: int) -> tuple[Tensor, Tensor]:
    first_length = left_length + 2
    left = hidden[:, 1 : 1 + left_length]
    right = hidden[:, first_length + 1 : first_length + 1 + right_length]
    if left.shape[1] != left_length or right.shape[1] != right_length:
        raise RuntimeError("B-canary hidden residue axes differ")
    return left, right


def _label_matrix(record: Mapping[str, Any], device: torch.device) -> Tensor:
    labels = torch.zeros(
        (1, len(str(record["sequence1"])), len(str(record["sequence2"]))),
        dtype=torch.bool,
        device=device,
    )
    pairs = torch.tensor(record["contact_pairs"], dtype=torch.long, device=device)
    labels[0, pairs[:, 0], pairs[:, 1]] = True
    return labels


def _average_precision(scores: Tensor, labels: Tensor) -> float:
    scores = scores.flatten().float().cpu()
    labels = labels.flatten().bool().cpu()
    if not bool(labels.any()):
        raise ValueError("Average precision requires positives")
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order].float()
    precision = ranked.cumsum(0) / torch.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / ranked.sum())


def _evaluate(
    backbone: PairAwareESM2,
    head: nn.Module,
    groups: Sequence[Sequence[dict[str, Any]]],
    known_pairs: set[tuple[str, str]],
    device: torch.device,
    rank: int,
) -> dict[str, Any]:
    backbone_was_training = backbone.training
    head_was_training = head.training
    backbone.eval()
    head.eval()
    totals = torch.zeros(12, dtype=torch.float64, device=device)
    local_system_metrics: list[dict[str, Any]] = []
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with torch.no_grad():
        for group_index, group in enumerate(groups):
            record = group[rank]
            permutation = build_partner_swap_permutation(
                _swap_allowed(group, known_pairs), seed=51_701 + group_index
            )
            swapped = group[int(permutation[rank])]
            left = str(record["sequence1"])
            right = str(record["sequence2"])
            swapped_right = str(swapped["sequence2"])
            tokens, chains, _ = _model_input(left, right, seed=0, mask=False)
            swap_tokens, swap_chains, _ = _model_input(left, swapped_right, seed=0, mask=False)
            tokens, chains = tokens.to(device), chains.to(device)
            swap_tokens, swap_chains = swap_tokens.to(device), swap_chains.to(device)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                output = backbone(tokens, chains, return_hidden_states=True, compute_logits=False)
                swap_output = backbone(
                    swap_tokens, swap_chains, return_hidden_states=True, compute_logits=False
                )
                if output.hidden_states is None or swap_output.hidden_states is None:
                    raise RuntimeError("B-canary evaluation hidden states are absent")
                left_hidden, right_hidden = _hidden_chains(output.hidden_states, len(left), len(right))
                swap_left, swap_right = _hidden_chains(
                    swap_output.hidden_states, len(left), len(swapped_right)
                )
                logits = head(left_hidden, right_hidden).float()
                swap_logits = head(swap_left, swap_right).float()
            labels = _label_matrix(record, device)
            paratope_labels = labels[0].any(dim=1)
            epitope_labels = labels[0].any(dim=0)
            pair_metrics = paircontact_ranking_metrics(logits[0], labels[0])
            paratope_ap = _average_precision(
                logits[0].amax(dim=1), paratope_labels
            )
            epitope_ap = _average_precision(logits[0].amax(dim=0), epitope_labels)
            matched_minus_swapped = float(logits.mean() - swap_logits.mean())
            local_system_metrics.append(
                {
                    "contact_ap": pair_metrics.average_precision,
                    "contact_auroc": pair_metrics.auroc,
                    "contact_mrr": pair_metrics.mean_reciprocal_rank,
                    "contact_precision_at_l_over_10": (
                        pair_metrics.precision_at_l_over_10
                    ),
                    "contact_recall_at_1": pair_metrics.recall_at_1,
                    "contact_recall_at_l_over_10": pair_metrics.recall_at_l_over_10,
                    "epitope_ap": epitope_ap,
                    "example_id": str(record["example_id"]),
                    "matched_minus_swapped_mean_logit": matched_minus_swapped,
                    "paratope_ap": paratope_ap,
                    "pdb_id": str(record["pdb_id"]),
                    "prevalence": pair_metrics.prevalence,
                }
            )
            totals += torch.tensor(
                [
                    pair_metrics.average_precision,
                    pair_metrics.auroc,
                    pair_metrics.mean_reciprocal_rank,
                    pair_metrics.recall_at_1,
                    pair_metrics.recall_at_l_over_10,
                    pair_metrics.precision_at_l_over_10,
                    paratope_ap,
                    epitope_ap,
                    matched_minus_swapped,
                    pair_metrics.prevalence,
                    float(len(left) + len(right)),
                    1.0,
                ],
                dtype=torch.float64,
                device=device,
            )
    torch.distributed.all_reduce(totals)
    if backbone_was_training:
        backbone.train()
    if head_was_training:
        head.train()
    gathered_system_metrics: list[list[dict[str, Any]] | None] = [
        None
    ] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered_system_metrics, local_system_metrics)
    per_system = sorted(
        (
            value
            for rank_values in gathered_system_metrics
            if rank_values is not None
            for value in rank_values
        ),
        key=lambda value: value["example_id"],
    )
    count = float(totals[11])
    return {
        "contact_mean_per_system_ap": float(totals[0] / count),
        "contact_mean_per_system_auroc": float(totals[1] / count),
        "contact_mean_per_system_mrr": float(totals[2] / count),
        "contact_mean_per_system_recall_at_1": float(totals[3] / count),
        "contact_mean_per_system_recall_at_l_over_10": float(totals[4] / count),
        "contact_mean_per_system_precision_at_l_over_10": float(totals[5] / count),
        "paratope_mean_per_system_ap": float(totals[6] / count),
        "epitope_mean_per_system_ap": float(totals[7] / count),
        "matched_minus_swapped_mean_logit": float(totals[8] / count),
        "contact_prevalence": float(totals[9] / count),
        "mean_residues": float(totals[10] / count),
        "per_system": per_system,
        "systems": count,
    }


def _load_model_only_dcp(model: PairAwareESM2, root: Path, generation: str) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
    state = get_model_state_dict(model, options=options)
    dcp.load({"model": state}, checkpoint_id=root.resolve() / generation, no_dist=False)
    incompatible = set_model_state_dict(model, state, options=options)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("B-canary DCP restore returned incompatible model keys")
    torch.distributed.barrier()


def _collective_verify_input_dcp(root: Path, generation: str) -> dict[str, Any]:
    """Hash a large immutable input generation once and broadcast its receipt."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return verify_dcp_generation(root, generation)
    payload: list[dict[str, Any]] = [{"error": None, "receipt": None}]
    if torch.distributed.get_rank() == 0:
        try:
            payload[0]["receipt"] = verify_dcp_generation(root, generation)
        except BaseException as error:
            payload[0]["error"] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0]["error"] is not None:
        raise RuntimeError(f"Input DCP verification failed: {payload[0]['error']}")
    receipt = payload[0]["receipt"]
    if not isinstance(receipt, dict):
        raise RuntimeError("Rank zero did not broadcast an input DCP receipt")
    return receipt


def _sync_head_gradients(head: nn.Module, world_size: int) -> None:
    for name, parameter in head.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"B-canary head parameter has no gradient: {name}")
        torch.distributed.all_reduce(parameter.grad)
        parameter.grad.div_(world_size)


def _clip_gradients(backbone: nn.Module, head: nn.Module, maximum: float) -> float:
    backbone_norm = torch.nn.utils.clip_grad_norm_(backbone.parameters(), float("inf"), foreach=False)
    head_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), float("inf"), foreach=False)
    total = math.sqrt(float(backbone_norm) ** 2 + float(head_norm) ** 2)
    if not math.isfinite(total):
        raise RuntimeError("B-canary gradient norm is nonfinite")
    scale = min(1.0, maximum / (total + 1e-6))
    if scale < 1:
        for parameter in [*backbone.parameters(), *head.parameters()]:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    return total


def _learning_rate_scale(
    update: int,
    total_updates: int,
    warmup_updates: int,
    minimum_ratio: float,
) -> float:
    if not 0 <= update < total_updates:
        raise ValueError("update is outside the learning-rate schedule")
    if not 0 <= warmup_updates < total_updates:
        raise ValueError("warmup_updates is outside the learning-rate schedule")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be within [0, 1]")
    if warmup_updates and update < warmup_updates:
        return (update + 1) / warmup_updates
    decay_steps = total_updates - warmup_updates
    progress = (update - warmup_updates) / max(1, decay_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def main() -> None:
    arguments = _arguments()
    project_root = arguments.project_root.resolve()
    protocol = _load_protocol(arguments)
    stage = protocol.get("stage", {}).get("name", "B-canary")
    output_generation = protocol.get("stage", {}).get(
        "output_generation", OUTPUT_GENERATION
    )
    result_format = protocol.get("stage", {}).get(
        "result_format", "abpairfm_pinder_b_canary_result_v1"
    )
    progress_format = protocol.get("stage", {}).get(
        "progress_format", "abpairfm_pinder_b_canary_progress_v1"
    )
    pass_status = protocol.get("stage", {}).get(
        "pass_status", "PASS_B_CANARY_SYSTEMS_AND_OBJECTIVE"
    )
    fail_status = protocol.get("stage", {}).get(
        "fail_status", "FAIL_B_CANARY_PROMOTION_GATE"
    )
    observed_source = source_sha256(project_root)
    if observed_source != arguments.expected_source_sha256:
        raise RuntimeError("B-canary reviewed source SHA-256 differs")
    records = _load_records(arguments.manifest, arguments.manifest_receipt, protocol)
    known_pairs = {_known_pair(record) for record in records}
    data = protocol["data"]
    holdout_systems = int(data.get("holdout_systems", 16))
    evaluation_groups = int(data.get("evaluation_groups", 4))
    training_records, holdout_records = _pdb_disjoint_holdout(records, holdout_systems)
    optimization = protocol["optimization"]
    training_groups = _legal_groups(
        training_records,
        group_count=optimization["updates"] * optimization["accumulation_steps"],
        group_size=arguments.expected_world_size,
        known_pairs=known_pairs,
        seed=1701,
        maximum_tokens=int(data.get("maximum_tokens", 1024)),
    )
    holdout_groups = _legal_groups(
        holdout_records,
        group_count=evaluation_groups,
        group_size=arguments.expected_world_size,
        known_pairs=known_pairs,
        seed=51_701,
        maximum_tokens=int(data.get("maximum_tokens", 1024)),
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="nccl", device_id=device, timeout=timedelta(minutes=15)
    )
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    try:
        if world_size != arguments.expected_world_size:
            raise RuntimeError("B-canary world size differs")
        properties = torch.cuda.get_device_properties(device)
        expected_device = {
            "gpu_4090": "NVIDIA GeForce RTX 4090",
            "gpu_5090": "NVIDIA GeForce RTX 5090",
        }[protocol["systems"]["partition"]]
        if properties.name != expected_device:
            raise RuntimeError(f"Forbidden GPU model: {properties.name}")
        input_receipt = _collective_verify_input_dcp(
            arguments.input_checkpoint_root, arguments.input_generation
        )
        if (
            input_receipt.get("content_sha256") != arguments.expected_input_content_sha256
            or input_receipt.get("training_state", {}).get("test_access") is not False
        ):
            raise RuntimeError("B-canary input DCP authority differs")

        torch.manual_seed(1701)
        torch.cuda.manual_seed(1701)
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(
            "cuda", (1, world_size), mesh_dim_names=("replicate", "shard")
        )
        with torch.device("meta"):
            backbone = PairAwareESM2(PairAwareESM2Config(activation_checkpointing=False))
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        for layer in backbone.layers[-protocol["model"]["trainable_top_layers"] :]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        for module in (backbone.emb_layer_norm_after, backbone.lm_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        fully_shard_pair_aware_esm2(
            backbone, mesh=mesh, precision="bfloat16", reshard_after_forward=True
        )
        initialization = load_authenticated_esm2_into_fsdp2(
            backbone,
            arguments.initial_checkpoint,
            expected_size=arguments.expected_initial_checkpoint_size,
            expected_sha256=arguments.expected_initial_checkpoint_sha256,
        )
        _load_model_only_dcp(
            backbone, arguments.input_checkpoint_root, arguments.input_generation
        )
        head = _build_pair_head(protocol).to(device)
        system = CanarySystem(backbone, head)
        trainable_backbone = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
        if not trainable_backbone:
            raise RuntimeError("B-canary selected no trainable backbone parameters")
        optimizer = torch.optim.AdamW(
            [
                {"params": trainable_backbone, "lr": optimization["backbone_learning_rate"]},
                {"params": list(head.parameters()), "lr": optimization["head_learning_rate"]},
            ],
            betas=tuple(optimization["betas"]),
            weight_decay=optimization["weight_decay"],
            fused=True,
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        backbone.train()
        head.train()
        torch.cuda.reset_peak_memory_stats(device)
        before = _evaluate(
            backbone, head, holdout_groups, known_pairs, device, rank
        )
        update_records: list[dict[str, float]] = []
        from torch.nn.attention import SDPBackend, sdpa_kernel

        for update in range(optimization["updates"]):
            learning_rate_scale = _learning_rate_scale(
                update,
                optimization["updates"],
                int(optimization.get("warmup_updates", 0)),
                float(optimization.get("minimum_learning_rate_ratio", 1.0)),
            )
            optimizer.param_groups[0]["lr"] = (
                optimization["backbone_learning_rate"] * learning_rate_scale
            )
            optimizer.param_groups[1]["lr"] = (
                optimization["head_learning_rate"] * learning_rate_scale
            )
            torch.distributed.barrier()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(8, dtype=torch.float64, device=device)
            for accumulation in range(optimization["accumulation_steps"]):
                microstep = update * optimization["accumulation_steps"] + accumulation
                group = training_groups[microstep]
                record = group[rank]
                permutation = build_partner_swap_permutation(
                    _swap_allowed(group, known_pairs), seed=1701 + microstep
                )
                swapped = group[int(permutation[rank])]
                left = str(record["sequence1"])
                right = str(record["sequence2"])
                swapped_right = str(swapped["sequence2"])
                tokens, chains, labels = _model_input(
                    left, right, seed=1701 + microstep * world_size + rank, mask=True
                )
                swap_tokens, swap_chains, _ = _model_input(
                    left,
                    swapped_right,
                    seed=971_701 + microstep * world_size + rank,
                    mask=True,
                )
                tokens, chains, labels = tokens.to(device), chains.to(device), labels.to(device)
                swap_tokens, swap_chains = swap_tokens.to(device), swap_chains.to(device)
                synchronize = accumulation == optimization["accumulation_steps"] - 1
                backbone.set_requires_gradient_sync(synchronize)
                backbone.set_is_last_backward(synchronize)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    output = backbone(
                        tokens, chains, labels=labels, return_hidden_states=True
                    )
                    swapped_output = backbone(
                        swap_tokens,
                        swap_chains,
                        return_hidden_states=True,
                        compute_logits=False,
                    )
                    if (
                        output.hidden_states is None
                        or output.balanced_chain_loss_sum is None
                        or swapped_output.hidden_states is None
                    ):
                        raise RuntimeError("B-canary backbone outputs are incomplete")
                    mlm_loss = output.balanced_chain_loss_sum / output.contributing_chain_count
                    left_hidden, right_hidden = _hidden_chains(
                        output.hidden_states, len(left), len(right)
                    )
                    swap_left, swap_right = _hidden_chains(
                        swapped_output.hidden_states, len(left), len(swapped_right)
                    )
                    pair_logits = head(left_hidden, right_hidden).float()
                    swapped_logits = head(swap_left, swap_right).float()
                pair_labels = _label_matrix(record, device)
                objective = paircontact_pretraining_loss(
                    pair_logits,
                    pair_labels,
                    torch.ones_like(pair_labels),
                    swapped_pair_logits=swapped_logits,
                    swapped_pair_valid=torch.ones_like(swapped_logits, dtype=torch.bool),
                    pair_weight=protocol["loss"]["pair_contact_weight"],
                    paratope_weight=protocol["loss"]["paratope_marginal_weight"],
                    epitope_weight=protocol["loss"]["epitope_marginal_weight"],
                    partner_swap_weight=protocol["loss"]["partner_swap_weight"],
                    pair_loss_mode=protocol["loss"].get(
                        "pair_loss_mode", "adaptive_balanced"
                    ),
                    focal_gamma=float(protocol["loss"].get("focal_gamma", 2.0)),
                    hard_negative_temperature=protocol["loss"]["hard_negative_temperature"],
                    marginal_temperature=protocol["loss"]["marginal_temperature"],
                )
                raw_total = (
                    objective.total
                    + protocol["loss"]["masked_language_modeling_weight"] * mlm_loss
                )
                multiplicity_scale = float(record["system_weight"]) / float(
                    data.get("system_weight_mean", data.get("canary_system_weight_mean"))
                )
                loss = raw_total * multiplicity_scale / optimization["accumulation_steps"]
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("B-canary loss is nonfinite")
                loss.backward()
                totals += torch.tensor(
                    [
                        float(raw_total.detach()),
                        float(objective.pair.detach()),
                        float(objective.paratope.detach()),
                        float(objective.epitope.detach()),
                        float(objective.partner_swap.detach()),
                        float(mlm_loss.detach()),
                        float(len(left) + len(right)),
                        float(pair_labels.sum()),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
            _sync_head_gradients(head, world_size)
            gradient_norm = _clip_gradients(
                backbone, head, optimization["gradient_clip_norm"]
            )
            optimizer.step()
            torch.distributed.all_reduce(totals)
            totals[:6] /= world_size * optimization["accumulation_steps"]
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            record = {
                "elapsed_seconds": elapsed,
                "epitope_loss": float(totals[3]),
                "gradient_norm": gradient_norm,
                "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                "head_learning_rate": optimizer.param_groups[1]["lr"],
                "masked_language_modeling_loss": float(totals[5]),
                "pair_contact_loss": float(totals[1]),
                "paratope_loss": float(totals[2]),
                "partner_swap_loss": float(totals[4]),
                "residues_per_second": float(totals[6] / elapsed),
                "total_loss": float(totals[0]),
                "update": float(update + 1),
            }
            update_records.append(record)
            if rank == 0:
                _publish_progress(
                    arguments.progress,
                    {
                        "format_version": progress_format,
                        "latest": record,
                        "status": "RUNNING",
                        "test_access": False,
                        "updates_complete": update + 1,
                        "updates_total": optimization["updates"],
                    },
                )

        after = _evaluate(backbone, head, holdout_groups, known_pairs, device, rank)
        rank_record = {
            "device_name": properties.name,
            "hostname": socket.gethostname(),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "rank": rank,
            "total_memory_bytes": properties.total_memory,
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, rank_record)
        ranks = [record for record in gathered if record is not None]
        reshard_pair_aware_esm2(backbone)
        provenance = {
            "input_checkpoint": arguments.expected_input_content_sha256,
            "manifest": protocol["data"]["manifest_sha256"],
            "manifest_receipt": protocol["data"]["manifest_receipt_sha256"],
            "protocol": arguments.expected_protocol_sha256,
            "source": observed_source,
        }
        checkpoint = save_dcp_generation(
            arguments.checkpoint_root,
            output_generation,
            model=system,
            optimizer=optimizer,
            provenance_hashes=provenance,
            training_state={
                "input_consumed_residue_tokens": input_receipt["training_state"][
                    "consumed_residue_tokens"
                ],
                "stage": stage,
                "test_access": False,
                "updates": optimization["updates"],
            },
            claim_boundary=protocol["claim_boundary"],
        )
        restored = load_dcp_generation(
            arguments.checkpoint_root,
            output_generation,
            model=system,
            optimizer=optimizer,
            expected_provenance_hashes=provenance,
            expected_claim_boundary=protocol["claim_boundary"],
        )
        if checkpoint["content_sha256"] != restored["content_sha256"]:
            raise RuntimeError("B-canary exact DCP restore content differs")
        peak_fraction = max(
            float(record["max_memory_allocated_bytes"]) / float(record["total_memory_bytes"])
            for record in ranks
        )
        finite = all(
            math.isfinite(value)
            for record in update_records
            for key, value in record.items()
            if key != "update"
        )
        promotion = protocol["promotion"]
        biological_checks = {
            "contact_ap_above_prevalence": (
                after["contact_mean_per_system_ap"] > after["contact_prevalence"]
            ),
            "contact_ap_non_decrease": (
                after["contact_mean_per_system_ap"] >= before["contact_mean_per_system_ap"]
            ),
            "matched_minus_swapped_positive": (
                after["matched_minus_swapped_mean_logit"] > 0.0
            ),
        }
        required_biological_checks = {
            key: value
            for key, value in biological_checks.items()
            if promotion.get(f"require_{key}", False)
        }
        status = (
            pass_status
            if finite
            and peak_fraction <= promotion["maximum_peak_memory_fraction"]
            and all(required_biological_checks.values())
            else fail_status
        )
        if rank == 0:
            result = {
                "authorization": protocol["authorization"],
                "checkpoint": {
                    "content_sha256": checkpoint["content_sha256"],
                    "generation": output_generation,
                    "restore_content_sha256": restored["content_sha256"],
                },
                "claim_boundary": protocol["claim_boundary"],
                "evaluation": {"after": after, "before": before},
                "format_version": result_format,
                "initialization": initialization,
                "input_checkpoint": {
                    "content_sha256": arguments.expected_input_content_sha256,
                    "generation": arguments.input_generation,
                },
                "optimization": {
                    "first_5_update_mean_seconds": statistics.mean(
                        record["elapsed_seconds"] for record in update_records[:5]
                    ),
                    "final_5_update_mean_seconds": statistics.mean(
                        record["elapsed_seconds"] for record in update_records[-5:]
                    ),
                    "final_5_update_mean_total_loss": statistics.mean(
                        record["total_loss"] for record in update_records[-5:]
                    ),
                    "peak_memory_fraction": peak_fraction,
                    "updates": len(update_records),
                },
                "objective": protocol["loss"],
                "promotion_checks": {
                    **biological_checks,
                    "finite_all_update_metrics": finite,
                    "peak_memory_within_limit": (
                        peak_fraction <= promotion["maximum_peak_memory_fraction"]
                    ),
                    "required_biological_checks": sorted(required_biological_checks),
                },
                "provenance": provenance,
                "ranks": ranks,
                "status": status,
                "test_access": False,
            }
            _publish_create_only(arguments.result, result)
            _publish_progress(
                arguments.progress,
                {
                    "format_version": progress_format,
                    "result_sha256": sha256_file(arguments.result),
                    "status": status,
                    "test_access": False,
                    "updates_complete": optimization["updates"],
                    "updates_total": optimization["updates"],
                },
            )
            print(json.dumps(result, allow_nan=False, sort_keys=True))
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
