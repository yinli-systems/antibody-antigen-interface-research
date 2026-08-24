#!/usr/bin/env python3
"""Score frozen SAbDab2 development contact and OOD slices with Stage-A1."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from antibody_pretraining.pair_aware_distributed import (  # noqa: E402
    fully_shard_pair_aware_esm2,
    load_authenticated_esm2_into_fsdp2,
)
from antibody_pretraining.pair_aware_esm2 import (  # noqa: E402
    ESM2_ALPHABET,
    PairAwareESM2,
    PairAwareESM2Config,
)
from antibody_pretraining.stage_a1_objective import StageA1Heads  # noqa: E402
from antibody_pretraining.stage_a1_geometry_objective import (  # noqa: E402
    StageA1GeometryHeads,
)
from interaction_first.frontier_fsdp2 import verify_dcp_generation  # noqa: E402


EXPECTED_PROTOCOL_FORMATS = {
    "abpairfm_stage_a1_development_contact_v1",
    "abpairfm_stage_a1_geometry_development_contact_v2",
}
EXPECTED_INITIAL_CHECKPOINT_SHA256 = (
    "7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500"
)
EXPECTED_INITIAL_CHECKPOINT_SIZE = 5_678_116_398
CONTACT_ARTIFACT_FORMAT = "sabdab2_framecontact_positive_flat_v1"
SOURCE_PATHS = (
    "scripts/run_abpairfm_stage_a1_frozen_development_contact.py",
    "src/antibody_pretraining/pair_aware_attention.py",
    "src/antibody_pretraining/pair_aware_distributed.py",
    "src/antibody_pretraining/pair_aware_esm2.py",
    "src/antibody_pretraining/stage_a1_objective.py",
    "src/antibody_pretraining/stage_a1_geometry_objective.py",
    "src/interaction_first/frontier_fsdp2.py",
)


class StageA1System(nn.Module):
    def __init__(self, backbone: PairAwareESM2, heads: StageA1Heads) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = heads


@dataclass(frozen=True)
class ContactRecord:
    instance_id: str
    antibody_sequence: str
    antigen_sequence: str
    antibody_length: int
    antigen_length: int
    artifact_path: Path
    artifact_sha256: str
    artifact_size: int
    candidate_pair_count: int
    positive_pair_count: int


@dataclass(frozen=True)
class WindowWork:
    instance_id: str
    antibody_start: int
    antibody_stop: int
    antigen_start: int
    antigen_stop: int
    antibody_sequence: str
    antigen_sequence: str

    @property
    def token_count(self) -> int:
        return len(self.antibody_sequence) + len(self.antigen_sequence) + 4


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--dcp-root", type=Path, required=True)
    parser.add_argument("--dcp-generation", required=True)
    parser.add_argument("--expected-dcp-content-sha256", required=True)
    parser.add_argument("--promotion-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-batch-size", type=int, default=7)
    parser.add_argument("--maximum-attention-cells", type=int, default=7_340_032)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Reviewed source is missing or symlinked: {relative}")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


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
        raise FileExistsError(f"Refusing to overwrite contact artifact: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_npz_create_only(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite contact predictions: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_protocol(arguments: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(arguments.protocol) != arguments.expected_protocol_sha256:
        raise RuntimeError("Stage-A1 contact protocol SHA-256 differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    authorization = protocol.get("authorization", {})
    compute = protocol.get("compute", {})
    if (
        protocol.get("format_version") not in EXPECTED_PROTOCOL_FORMATS
        or protocol.get("status") != "FROZEN_BEFORE_GPU_EVALUATION"
        or authorization.get("development_contact") is not True
        or authorization.get("development_pair_retrieval") is not False
        or authorization.get("production_training") is not False
        or authorization.get("test_access") is not False
        or protocol.get("test_access") is not False
        or compute.get("partition") != "gpu_4090"
        or compute.get("world_size") != 8
        or arguments.maximum_batch_size
        != compute.get("maximum_batch_windows_per_rank")
        or arguments.maximum_attention_cells
        != compute.get("maximum_attention_cells_per_rank")
        or protocol.get("calibration", {}).get("fitted_parameters") is not False
    ):
        raise RuntimeError("Stage-A1 contact protocol authorization differs")
    return protocol


def _verify_checkpoint_collectively(
    arguments: argparse.Namespace, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rank = torch.distributed.get_rank()
    payload: list[dict[str, Any]] = [{"error": None, "receipt": None}]
    if rank == 0:
        try:
            promotion = json.loads(arguments.promotion_result.read_text(encoding="utf-8"))
            selection = protocol["checkpoint_selection"]
            receipt = verify_dcp_generation(arguments.dcp_root, arguments.dcp_generation)
            if (
                promotion.get("status") != selection["required_result_status"]
                or promotion.get("test_access") is not False
                or promotion.get("checkpoint", {}).get("content_sha256")
                != arguments.expected_dcp_content_sha256
                or promotion.get("checkpoint", {}).get("generation")
                != arguments.dcp_generation
                or promotion.get("source_sha256")
                != selection["expected_training_source_sha256"]
                or receipt.get("content_sha256")
                != arguments.expected_dcp_content_sha256
                or receipt.get("generation") != arguments.dcp_generation
                or receipt.get("provenance_hashes", {}).get("protocol")
                != selection["expected_training_protocol_sha256"]
                or receipt.get("provenance_hashes", {}).get("source")
                != selection["expected_training_source_sha256"]
                or receipt.get("training_state", {}).get("production_training")
                is not selection["expected_production_training"]
                or receipt.get("training_state", {}).get("stage_a1_updates")
                != selection["expected_stage_a1_updates"]
                or receipt.get("training_state", {}).get("test_access") is not False
            ):
                raise RuntimeError("Stage-A1 promotion result or DCP receipt differs")
            payload[0]["receipt"] = receipt
        except BaseException as error:
            payload[0]["error"] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(payload, src=0)
    torch.distributed.barrier()
    if payload[0]["error"] is not None:
        raise RuntimeError(f"Stage-A1 checkpoint verification failed: {payload[0]['error']}")
    receipt = payload[0]["receipt"]
    if not isinstance(receipt, dict):
        raise RuntimeError("Rank zero did not broadcast the Stage-A1 receipt")
    return receipt


def _load_system_dcp(
    system: StageA1System, *, dcp_root: Path, generation: str
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
    state = get_model_state_dict(system, options=options)
    dcp.load(
        {"model": state}, checkpoint_id=dcp_root.resolve() / generation, no_dist=False
    )
    incompatible = set_model_state_dict(system, state, options=options)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Stage-A1 DCP restore returned incompatible model keys")
    torch.distributed.barrier()


def _anchored_json(anchor: Mapping[str, str]) -> dict[str, Any]:
    path = Path(anchor["path"])
    if path.is_symlink() or not path.is_file() or sha256_file(path) != anchor["sha256"]:
        raise RuntimeError(f"Anchored JSON differs: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _load_records(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, ContactRecord], dict[str, list[str]]]:
    anchors = protocol["source_anchors"]
    transfer = _anchored_json(anchors["development_transfer_manifest"])
    contact_manifest = _anchored_json(anchors["contact_manifest"])
    sequence_receipt = _anchored_json(anchors["development_sequence_panel_receipt"])
    if (
        transfer.get("format_version") != "abpairfm_development_transfer_manifest_v2"
        or transfer.get("status")
        != "PASS_DEVELOPMENT_ONLY_MANIFEST_PRIMARY_TEST_SEALED"
        or transfer.get("test_access") is not False
        or transfer.get("authority", {}).get("primary_sealed_role_authority")
        != "standard"
        or transfer.get("authority", {}).get("primary_sealed_role_ids_retained")
        is not False
        or transfer.get("authority", {}).get("task_labels_read") is not False
        or contact_manifest.get("format_version")
        != "sabdab2_framecontact_cohort_cache_v2"
    ):
        raise RuntimeError("Development transfer or contact authority differs")
    slices = {
        name: list(value["instance_ids"])
        for name, value in transfer["slices"].items()
        if name in protocol["split_slices"]
    }
    if set(slices) != set(protocol["split_slices"]):
        raise RuntimeError("Development contact slice names differ")
    for name, identifiers in slices.items():
        if (
            len(identifiers) != protocol["split_slices"][name]
            or identifiers != sorted(identifiers)
            or len(identifiers) != len(set(identifiers))
        ):
            raise RuntimeError(f"Development contact slice differs: {name}")
    selected = sorted(set().union(*map(set, slices.values())))

    contact_by_id = {row["instance_id"]: row for row in contact_manifest["records"]}
    if len(contact_by_id) != contact_manifest["counts"]["records"] or not set(
        selected
    ).issubset(contact_by_id):
        raise RuntimeError("Selected development contacts are absent from authority")

    panel_path = Path(anchors["development_sequence_panel"]["path"])
    if (
        panel_path.is_symlink()
        or not panel_path.is_file()
        or sha256_file(panel_path) != anchors["development_sequence_panel"]["sha256"]
        or sequence_receipt.get("format_version")
        != "abpairfm_development_contact_sequence_authority_v1"
        or sequence_receipt.get("status")
        != "FROZEN_DEVELOPMENT_CONTACT_SEQUENCE_AUTHORITY"
        or sequence_receipt.get("test_access") is not False
        or sequence_receipt.get("authority", {}).get(
            "primary_sealed_role_ids_retained"
        )
        is not False
        or sequence_receipt.get("authority", {}).get("task_labels_read") is not False
        or sequence_receipt.get("authority", {}).get("contact_labels_read") is not False
        or sequence_receipt.get("authority", {}).get("transfer_manifest_sha256")
        != anchors["development_transfer_manifest"]["sha256"]
        or sequence_receipt.get("panel", {}).get("sha256")
        != anchors["development_sequence_panel"]["sha256"]
    ):
        raise RuntimeError("Development sequence panel authority differs")
    sequence_rows = [
        json.loads(line)
        for line in panel_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sequence_by_id = {row.get("instance_id"): row for row in sequence_rows}
    if (
        len(sequence_by_id) != len(sequence_rows)
        or set(sequence_by_id) != set(selected)
        or sequence_receipt.get("panel", {}).get("row_count") != len(sequence_rows)
    ):
        raise RuntimeError("Development sequence panel membership differs")
    contact_root = Path(anchors["contact_manifest"]["path"]).parent
    records: dict[str, ContactRecord] = {}
    for identifier in selected:
        authority = sequence_by_id[identifier]
        if set(authority) != {
            "antibody_length",
            "antibody_sequence",
            "antibody_sequence_sha256",
            "antigen_length",
            "antigen_sequence",
            "antigen_sequence_sha256",
            "instance_id",
        }:
            raise RuntimeError(f"Development sequence row fields differ: {identifier}")
        antibody = str(authority["antibody_sequence"])
        antigen = str(authority["antigen_sequence"])
        contact = contact_by_id[identifier]
        antibody_length = int(authority["antibody_length"])
        antigen_length = int(authority["antigen_length"])
        artifact = contact_root / contact["artifact_path"]
        if (
            not antibody
            or not antigen
            or len(antibody) != antibody_length
            or len(antigen) != antigen_length
            or _sequence_sha256(antibody) != authority["antibody_sequence_sha256"]
            or _sequence_sha256(antigen) != authority["antigen_sequence_sha256"]
            or contact["antibody_length"] != antibody_length
            or contact["antigen_length"] != antigen_length
            or contact["candidate_pair_count"] != antibody_length * antigen_length
            or artifact.is_symlink()
            or not artifact.is_file()
        ):
            raise RuntimeError(f"Development contact record differs: {identifier}")
        records[identifier] = ContactRecord(
            instance_id=identifier,
            antibody_sequence=antibody,
            antigen_sequence=antigen,
            antibody_length=antibody_length,
            antigen_length=antigen_length,
            artifact_path=artifact,
            artifact_sha256=contact["artifact_sha256"],
            artifact_size=int(artifact.stat().st_size),
            candidate_pair_count=int(contact["candidate_pair_count"]),
            positive_pair_count=int(contact["positive_pair_count"]),
        )
        if records[identifier].artifact_size <= 0:
            raise RuntimeError("Contact artifact is empty")
    return records, slices


def window_spans(length: int, maximum: int) -> list[tuple[int, int]]:
    if length <= 0 or maximum <= 0:
        raise ValueError("Window lengths must be positive")
    if length <= maximum:
        return [(0, length)]
    starts = sorted({0, (length - maximum) // 2, length - maximum})
    spans = [(start, start + maximum) for start in starts]
    coverage = np.zeros(length, dtype=np.uint8)
    for start, stop in spans:
        coverage[start:stop] += 1
    if bool((coverage == 0).any()):
        raise RuntimeError("Endpoint windows do not cover every residue")
    return spans


def _allocate_records(
    records: Mapping[str, ContactRecord], world_size: int, maximum: int
) -> list[list[str]]:
    costs = []
    for identifier, record in records.items():
        cost = sum(
            (antibody_stop - antibody_start + antigen_stop - antigen_start + 4) ** 2
            for antibody_start, antibody_stop in window_spans(record.antibody_length, maximum)
            for antigen_start, antigen_stop in window_spans(record.antigen_length, maximum)
        )
        costs.append((cost, identifier))
    assignments = [[] for _ in range(world_size)]
    totals = [0] * world_size
    for cost, identifier in sorted(costs, key=lambda value: (-value[0], value[1])):
        rank = min(range(world_size), key=lambda value: (totals[value], value))
        assignments[rank].append(identifier)
        totals[rank] += cost
    if any(not assignment for assignment in assignments):
        raise RuntimeError("Every rank requires development contact work")
    return [sorted(assignment) for assignment in assignments]


def _window_work(
    records: Mapping[str, ContactRecord], identifiers: Iterable[str], maximum: int
) -> list[WindowWork]:
    result = []
    for identifier in identifiers:
        record = records[identifier]
        for antibody_start, antibody_stop in window_spans(record.antibody_length, maximum):
            for antigen_start, antigen_stop in window_spans(record.antigen_length, maximum):
                result.append(
                    WindowWork(
                        instance_id=identifier,
                        antibody_start=antibody_start,
                        antibody_stop=antibody_stop,
                        antigen_start=antigen_start,
                        antigen_stop=antigen_stop,
                        antibody_sequence=record.antibody_sequence[
                            antibody_start:antibody_stop
                        ],
                        antigen_sequence=record.antigen_sequence[antigen_start:antigen_stop],
                    )
                )
    return result


def batch_window_work(
    work: list[WindowWork], maximum_batch_size: int, maximum_attention_cells: int
) -> list[list[WindowWork]]:
    batches: list[list[WindowWork]] = []
    current: list[WindowWork] = []
    current_maximum = 0
    for item in work:
        proposed_maximum = max(current_maximum, item.token_count)
        proposed_size = len(current) + 1
        if current and (
            proposed_size > maximum_batch_size
            or proposed_size * proposed_maximum * proposed_maximum
            > maximum_attention_cells
        ):
            batches.append(current)
            current = []
            current_maximum = 0
        current.append(item)
        current_maximum = max(current_maximum, item.token_count)
        if len(current) * current_maximum * current_maximum > maximum_attention_cells:
            raise RuntimeError("One contact window exceeds the attention-cell budget")
    if current:
        batches.append(current)
    if not batches:
        raise RuntimeError("Contact work produced no batches")
    return batches


def _token_ids(sequence: str) -> list[int]:
    index = {token: position for position, token in enumerate(ESM2_ALPHABET)}
    unknown = index["<unk>"]
    return [
        index["<cls>"],
        *(index.get(residue, unknown) for residue in sequence),
        index["<eos>"],
    ]


def build_contact_batch(
    batch: list[WindowWork], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int, int]]]:
    encoded = [(_token_ids(item.antibody_sequence), _token_ids(item.antigen_sequence)) for item in batch]
    maximum = max(len(antibody) + len(antigen) for antibody, antigen in encoded)
    tokens = torch.full((len(batch), maximum), 1, dtype=torch.long)
    chain_ids = torch.full((len(batch), maximum), -1, dtype=torch.long)
    offsets = []
    for row, (antibody, antigen) in enumerate(encoded):
        antigen_token_start = len(antibody)
        stop = antigen_token_start + len(antigen)
        tokens[row, :antigen_token_start] = torch.tensor(antibody)
        tokens[row, antigen_token_start:stop] = torch.tensor(antigen)
        chain_ids[row, :antigen_token_start] = 0
        chain_ids[row, antigen_token_start:stop] = 1
        offsets.append((1, len(antibody) - 1, antigen_token_start + 1, stop - 1))
    return (
        tokens.to(device, non_blocking=True),
        chain_ids.to(device, non_blocking=True),
        offsets,
    )


def _load_positive_indices(record: ContactRecord) -> np.ndarray:
    if (
        record.artifact_path.stat().st_size != record.artifact_size
        or sha256_file(record.artifact_path) != record.artifact_sha256
    ):
        raise RuntimeError(f"Contact artifact authentication failed: {record.instance_id}")
    with np.load(record.artifact_path, allow_pickle=False) as artifact:
        required = {
            "format_version",
            "instance_id",
            "antibody_length",
            "antigen_length",
            "candidate_pair_count",
            "positive_flat_indices",
        }
        if set(artifact.files) != required:
            raise RuntimeError("Contact artifact fields differ")
        positives = np.asarray(artifact["positive_flat_indices"], dtype=np.int64)
        if (
            str(artifact["format_version"].item()) != CONTACT_ARTIFACT_FORMAT
            or str(artifact["instance_id"].item()) != record.instance_id
            or int(artifact["antibody_length"].item()) != record.antibody_length
            or int(artifact["antigen_length"].item()) != record.antigen_length
            or int(artifact["candidate_pair_count"].item())
            != record.candidate_pair_count
            or len(positives) != record.positive_pair_count
            or len(positives) == 0
            or bool((positives < 0).any())
            or bool((positives >= record.candidate_pair_count).any())
            or bool((np.diff(positives) <= 0).any())
        ):
            raise RuntimeError(f"Contact artifact payload differs: {record.instance_id}")
    return positives


def score_local_records(
    *,
    backbone: PairAwareESM2,
    heads: StageA1Heads,
    records: Mapping[str, ContactRecord],
    identifiers: list[str],
    output_root: Path,
    maximum_window: int,
    maximum_batch_size: int,
    maximum_attention_cells: int,
    device: torch.device,
) -> dict[str, Any]:
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    work = _window_work(records, identifiers, maximum_window)
    batches = batch_window_work(work, maximum_batch_size, maximum_attention_cells)
    maximum_batches = torch.tensor(len(batches), device=device, dtype=torch.long)
    torch.distributed.all_reduce(maximum_batches, op=torch.distributed.ReduceOp.MAX)
    accumulators = {
        identifier: (
            np.zeros(
                (records[identifier].antibody_length, records[identifier].antigen_length),
                dtype=np.float32,
            ),
            np.zeros(
                (records[identifier].antibody_length, records[identifier].antigen_length),
                dtype=np.uint8,
            ),
        )
        for identifier in identifiers
    }
    backbone.eval()
    heads.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.distributed.barrier()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    active_windows = 0
    for batch_index in range(int(maximum_batches.item())):
        active = batch_index < len(batches)
        batch = batches[batch_index] if active else batches[-1]
        tokens, chain_ids, offsets = build_contact_batch(batch, device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = backbone(
                tokens,
                chain_ids,
                allow_cross_chain_attention=True,
                return_hidden_states=True,
                compute_logits=False,
            )
            if output.hidden_states is None:
                raise RuntimeError("Stage-A1 contact hidden states are missing")
            for row, (item, offset) in enumerate(zip(batch, offsets)):
                antibody_start, antibody_stop, antigen_start, antigen_stop = offset
                if isinstance(heads, StageA1GeometryHeads):
                    left_count = antibody_stop - antibody_start
                    right_count = antigen_stop - antigen_start
                    pair_chunk = 4096
                    chunks = []
                    for pair_start in range(0, left_count * right_count, pair_chunk):
                        flat = torch.arange(
                            pair_start,
                            min(pair_start + pair_chunk, left_count * right_count),
                            device=device,
                        )
                        chunks.append(
                            heads.contact_logits(
                                output.hidden_states,
                                torch.full_like(flat, row),
                                antibody_start + torch.div(flat, right_count, rounding_mode="floor"),
                                antigen_start + flat.remainder(right_count),
                            )
                        )
                    logits = torch.cat(chunks).reshape(left_count, right_count)
                else:
                    left = heads.contact_left(
                        output.hidden_states[row, antibody_start:antibody_stop]
                    )
                    right = heads.contact_right(
                        output.hidden_states[row, antigen_start:antigen_stop]
                    )
                    logits = left @ right.transpose(0, 1) / math.sqrt(left.shape[-1])
                    logits = logits + heads.contact_bias
                if active:
                    sums, counts = accumulators[item.instance_id]
                    block = logits.float().cpu().numpy()
                    target = np.s_[
                        item.antibody_start:item.antibody_stop,
                        item.antigen_start:item.antigen_stop,
                    ]
                    sums[target] += block
                    counts[target] += 1
                    active_windows += 1
        if rank == 0 and (batch_index == 0 or (batch_index + 1) % 25 == 0):
            print(
                json.dumps(
                    {
                        "batch": batch_index + 1,
                        "batches": int(maximum_batches.item()),
                        "phase": "contact",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    prediction_root = output_root / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    prediction_bytes = 0
    for identifier in identifiers:
        sums, counts = accumulators[identifier]
        if bool((counts == 0).any()):
            raise RuntimeError(f"Contact windows did not cover {identifier}")
        logits = sums / counts.astype(np.float32)
        if not bool(np.isfinite(logits).all()):
            raise RuntimeError(f"Nonfinite contact logits for {identifier}")
        positives = _load_positive_indices(records[identifier])
        path = prediction_root / f"{hashlib.sha256(identifier.encode()).hexdigest()}.npz"
        _publish_npz_create_only(
            path,
            format_version=np.asarray("abpairfm_stage_a1_contact_prediction_v1"),
            instance_id=np.asarray(identifier),
            antibody_length=np.asarray(records[identifier].antibody_length, dtype=np.int64),
            antigen_length=np.asarray(records[identifier].antigen_length, dtype=np.int64),
            logits=logits,
            positive_flat_indices=positives,
        )
        prediction_bytes += path.stat().st_size
    counters = torch.tensor(
        [active_windows, prediction_bytes, torch.cuda.max_memory_allocated(device)],
        device=device,
        dtype=torch.long,
    )
    gathered = [torch.zeros_like(counters) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, counters)
    elapsed_tensor = torch.tensor(elapsed, device=device, dtype=torch.float64)
    torch.distributed.all_reduce(elapsed_tensor, op=torch.distributed.ReduceOp.MAX)
    return {
        "elapsed_seconds": float(elapsed_tensor.item()),
        "gpu_hours": float(elapsed_tensor.item()) * world_size / 3600.0,
        "prediction_bytes": sum(int(value[1].item()) for value in gathered),
        "scored_complexes": len(records),
        "scored_windows": sum(int(value[0].item()) for value in gathered),
        "peak_cuda_bytes_max_rank": max(int(value[2].item()) for value in gathered),
        "world_size": world_size,
    }


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive_count = int(labels.sum())
    if len(labels) == 0 or positive_count == 0:
        raise ValueError("Average precision requires a positive example")
    if positive_count == len(labels):
        return 1.0
    order = np.argsort(-scores, kind="stable")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    true_positive = np.cumsum(ordered_labels, dtype=np.int64)
    group_ends = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0), len(scores) - 1]
    tp = true_positive[group_ends].astype(np.float64)
    predicted = (group_ends + 1).astype(np.float64)
    recall = tp / positive_count
    precision = tp / predicted
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def fixed_raw_ece(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    order = np.argsort(probabilities, kind="stable")
    total = len(labels)
    value = 0.0
    for indices in np.array_split(order, bins):
        if len(indices):
            value += len(indices) / total * abs(
                float(probabilities[indices].mean() - labels[indices].mean())
            )
    return value


def _bootstrap_ci(values: list[float], replicates: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        raise ValueError("Cannot bootstrap empty values")
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    chunk = 512
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = generator.integers(0, len(array), size=(stop - start, len(array)))
        draws[start:stop] = array[indices].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _prediction_path(output_root: Path, identifier: str) -> Path:
    return output_root / "predictions" / f"{hashlib.sha256(identifier.encode()).hexdigest()}.npz"


def _load_prediction(
    output_root: Path, identifier: str
) -> tuple[np.ndarray, np.ndarray]:
    path = _prediction_path(output_root, identifier)
    with np.load(path, allow_pickle=False) as artifact:
        if (
            set(artifact.files)
            != {
                "format_version",
                "instance_id",
                "antibody_length",
                "antigen_length",
                "logits",
                "positive_flat_indices",
            }
            or str(artifact["format_version"].item())
            != "abpairfm_stage_a1_contact_prediction_v1"
            or str(artifact["instance_id"].item()) != identifier
        ):
            raise RuntimeError(f"Contact prediction artifact differs: {identifier}")
        logits = np.asarray(artifact["logits"], dtype=np.float32)
        positives = np.asarray(artifact["positive_flat_indices"], dtype=np.int64)
        if logits.shape != (
            int(artifact["antibody_length"].item()),
            int(artifact["antigen_length"].item()),
        ):
            raise RuntimeError("Contact prediction shape differs")
    return logits, positives


def _complex_metrics(logits: np.ndarray, positives: np.ndarray) -> dict[str, float]:
    flat = logits.reshape(-1)
    labels = np.zeros(len(flat), dtype=np.bool_)
    labels[positives] = True
    probabilities = 1.0 / (1.0 + np.exp(-flat.astype(np.float64)))
    order = np.argsort(-flat, kind="stable")
    first_true_rank = int(np.flatnonzero(labels[order])[0]) + 1
    pair_auprc = average_precision(labels, flat)
    brier = float(np.mean((probabilities - labels) ** 2))
    matrix_probabilities = probabilities.reshape(logits.shape)
    matrix_labels = labels.reshape(logits.shape)
    epitope_labels = matrix_labels.any(axis=0)
    paratope_labels = matrix_labels.any(axis=1)
    return {
        "pair_auprc": pair_auprc,
        "fixed_raw_brier": brier,
        "first_true_reciprocal_rank": 1.0 / first_true_rank,
        "hit_at_1": float(first_true_rank == 1),
        "epitope_auprc": average_precision(
            epitope_labels, matrix_probabilities.max(axis=0)
        ),
        "paratope_auprc": average_precision(
            paratope_labels, matrix_probabilities.max(axis=1)
        ),
    }


def analyze_slices(
    *,
    output_root: Path,
    slices: Mapping[str, list[str]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    replicates = int(protocol["metrics"]["uncertainty"]["bootstrap_replicates"])
    seed = int(protocol["metrics"]["uncertainty"]["seed"])
    for slice_index, (name, identifiers) in enumerate(sorted(slices.items())):
        complex_rows = []
        flat_logits = []
        flat_labels = []
        prediction_hashes = []
        candidate_count = 0
        positive_count = 0
        for identifier in identifiers:
            path = _prediction_path(output_root, identifier)
            logits, positives = _load_prediction(output_root, identifier)
            labels = np.zeros(logits.size, dtype=np.bool_)
            labels[positives] = True
            complex_rows.append({"instance_id": identifier, **_complex_metrics(logits, positives)})
            flat_logits.append(logits.reshape(-1))
            flat_labels.append(labels)
            prediction_hashes.append(f"{identifier}\t{sha256_file(path)}")
            candidate_count += logits.size
            positive_count += len(positives)
        logits = np.concatenate(flat_logits)
        labels = np.concatenate(flat_labels)
        probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        metrics: dict[str, Any] = {
            "global_pair_auprc": average_precision(labels, logits),
            "global_fixed_raw_brier": float(np.mean((probabilities - labels) ** 2)),
            "global_fixed_raw_ece_15": fixed_raw_ece(labels, probabilities, 15),
        }
        for field in (
            "pair_auprc",
            "fixed_raw_brier",
            "first_true_reciprocal_rank",
            "hit_at_1",
            "epitope_auprc",
            "paratope_auprc",
        ):
            values = [float(row[field]) for row in complex_rows]
            key = f"mean_per_complex_{field}"
            metrics[key] = float(np.mean(values))
            metrics[f"{key}_bootstrap_ci95"] = _bootstrap_ci(
                values, replicates, seed + slice_index
            )
        rows_path = output_root / f"{name}.complex_metrics.jsonl"
        rows_payload = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for row in complex_rows
        )
        publish_create_only(rows_path, rows_payload)
        summary = {
            "candidate_pair_count": candidate_count,
            "complex_count": len(identifiers),
            "complex_metrics": {
                "bytes": rows_path.stat().st_size,
                "sha256": sha256_file(rows_path),
            },
            "metrics": metrics,
            "positive_pair_count": positive_count,
            "prediction_set_sha256": hashlib.sha256(
                ("\n".join(prediction_hashes) + "\n").encode()
            ).hexdigest(),
        }
        result[name] = summary
        del logits, labels, probabilities, flat_logits, flat_labels
    return result


def main() -> None:
    arguments = _arguments()
    protocol = _load_protocol(arguments)
    project_root = arguments.project_root.resolve()
    source_sha256 = _source_sha256(project_root)
    if source_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("Reviewed Stage-A1 contact source digest differs")
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("Stage-A1 contact must run under Slurm gpu_4090")
    if (
        arguments.initial_checkpoint.stat().st_size != EXPECTED_INITIAL_CHECKPOINT_SIZE
        or sha256_file(arguments.initial_checkpoint) != EXPECTED_INITIAL_CHECKPOINT_SHA256
    ):
        raise RuntimeError("Initial ESM2 checkpoint authentication failed")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="nccl", device_id=device, timeout=timedelta(minutes=20)
    )
    rank = torch.distributed.get_rank()
    try:
        world_size = torch.distributed.get_world_size()
        if world_size != 8:
            raise RuntimeError("Stage-A1 contact requires exactly eight ranks")
        properties = torch.cuda.get_device_properties(device)
        if properties.name != "NVIDIA GeForce RTX 4090":
            raise RuntimeError(f"Forbidden GPU model: {properties.name}")
        torch.manual_seed(20260824 + rank)
        torch.cuda.manual_seed(20260824 + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        receipt = _verify_checkpoint_collectively(arguments, protocol)
        records, slices = _load_records(protocol)
        maximum_window = int(protocol["contact_score"]["endpoint_window_residues"])
        assignments = _allocate_records(records, world_size, maximum_window)

        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))
        with torch.device("meta"):
            backbone = PairAwareESM2(PairAwareESM2Config(activation_checkpointing=False))
        fully_shard_pair_aware_esm2(
            backbone, mesh=mesh, precision="bfloat16", reshard_after_forward=True
        )
        load_authenticated_esm2_into_fsdp2(
            backbone,
            arguments.initial_checkpoint,
            expected_size=EXPECTED_INITIAL_CHECKPOINT_SIZE,
            expected_sha256=EXPECTED_INITIAL_CHECKPOINT_SHA256,
        )
        if protocol["contact_score"].get("head") == "geometry_distance_mass_v1":
            geometry = protocol["contact_score"]["geometry_head"]
            heads = StageA1GeometryHeads(
                pair_dimension=geometry["pair_dimension"],
                num_distance_bins=geometry["num_distance_bins"],
                contact_distance_bin_count=geometry["contact_distance_bin_count"],
                temperature=0.07,
            ).to(device=device, dtype=torch.float32)
        else:
            heads = StageA1Heads(temperature=0.07).to(
                device=device, dtype=torch.float32
            )
        system = StageA1System(backbone, heads)
        _load_system_dcp(
            system, dcp_root=arguments.dcp_root, generation=arguments.dcp_generation
        )
        runtime = score_local_records(
            backbone=backbone,
            heads=heads,
            records=records,
            identifiers=assignments[rank],
            output_root=arguments.output_root,
            maximum_window=maximum_window,
            maximum_batch_size=arguments.maximum_batch_size,
            maximum_attention_cells=arguments.maximum_attention_cells,
            device=device,
        )
        torch.distributed.barrier()
        summaries = analyze_slices(
            output_root=arguments.output_root, slices=slices, protocol=protocol
        ) if rank == 0 else None
        torch.distributed.barrier()
        if rank == 0:
            result = {
                "authorization": protocol["authorization"],
                "checkpoint": {
                    "content_sha256": receipt["content_sha256"],
                    "generation": receipt["generation"],
                    "training_state": receipt["training_state"],
                },
                "claim_boundary": protocol["claim_boundary"],
                "format_version": "abpairfm_stage_a1_development_contact_result_v1",
                "protocol_sha256": arguments.expected_protocol_sha256,
                "runtime": {
                    **runtime,
                    "gpu_name": properties.name,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                },
                "slices": summaries,
                "source_sha256": source_sha256,
                "status": "FROZEN_DEVELOPMENT_CONTACT_COMPLETE",
                "test_access": False,
            }
            publish_create_only(
                arguments.output_root / "result.json",
                (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            )
            print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
