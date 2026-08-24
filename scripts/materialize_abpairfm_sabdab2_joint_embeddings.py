#!/usr/bin/env python3
"""Materialize train/validation-only joint ABPairFM features for SAbDab2."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from antibody_pretraining.pair_aware_distributed import (  # noqa: E402
    fully_shard_pair_aware_esm2,
    load_authenticated_esm2_into_fsdp2,
)
from antibody_pretraining.pair_aware_esm2 import (  # noqa: E402
    PairAwareESM2,
    PairAwareESM2Config,
)
from antibody_pretraining.stage_a1_geometry_objective import (  # noqa: E402
    StageA1GeometryHeads,
)
from run_abpairfm_stage_a1_frozen_development_contact import (  # noqa: E402
    EXPECTED_INITIAL_CHECKPOINT_SHA256,
    EXPECTED_INITIAL_CHECKPOINT_SIZE,
    StageA1System,
    WindowWork,
    _load_protocol,
    _load_system_dcp,
    _replicate_shard_mesh_shape,
    _verify_checkpoint_collectively,
    batch_window_work,
    build_contact_batch,
    sha256_file,
)


FORMAT_VERSION = "sabdab2_abpairfm_joint_embedding_manifest_v1"
ARTIFACT_FORMAT = "sabdab2_abpairfm_joint_embedding_v1"
ALLOWED_ROLES = frozenset({"train", "validation"})


@dataclass(frozen=True)
class SequenceRecord:
    instance_id: str
    role: str
    antibody_sequence: str
    antigen_sequence: str
    antibody_sequence_sha256: str
    antigen_sequence_sha256: str

    @property
    def antibody_length(self) -> int:
        return len(self.antibody_sequence)

    @property
    def antigen_length(self) -> int:
        return len(self.antigen_sequence)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-protocol", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-protocol-sha256", required=True)
    parser.add_argument("--embedding-protocol", type=Path, required=True)
    parser.add_argument("--expected-embedding-protocol-sha256", required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--dcp-root", type=Path, required=True)
    parser.add_argument("--dcp-generation", required=True)
    parser.add_argument("--expected-dcp-content-sha256", required=True)
    parser.add_argument("--promotion-result", type=Path, required=True)
    parser.add_argument("--sequence-view", type=Path, required=True)
    parser.add_argument("--expected-sequence-view-sha256", required=True)
    parser.add_argument("--sequence-view-receipt", type=Path, required=True)
    parser.add_argument("--expected-sequence-view-receipt-sha256", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--expected-split-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-window-residues", type=int, default=510)
    parser.add_argument("--maximum-batch-size", type=int, default=7)
    parser.add_argument("--maximum-attention-cells", type=int, default=7_340_032)
    return parser.parse_args()


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _artifact_name(instance_id: str) -> str:
    return hashlib.sha256(instance_id.encode("utf-8")).hexdigest() + ".pt"


def _window_spans(length: int, maximum: int) -> list[tuple[int, int]]:
    """Cover an arbitrary sequence with deterministic bounded windows."""
    if length <= 0 or maximum <= 0:
        raise ValueError("Window lengths must be positive")
    if length <= maximum:
        return [(0, length)]
    starts = list(range(0, length - maximum + 1, maximum))
    if starts[-1] != length - maximum:
        starts.append(length - maximum)
    spans = [(start, start + maximum) for start in starts]
    coverage = np.zeros(length, dtype=np.uint8)
    for start, stop in spans:
        coverage[start:stop] += 1
    if bool((coverage == 0).any()):
        raise RuntimeError("Joint embedding windows do not cover every residue")
    return spans


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _publish_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    try:
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _load_embedding_protocol(arguments: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(arguments.embedding_protocol) != arguments.expected_embedding_protocol_sha256:
        raise RuntimeError("Joint-embedding protocol SHA-256 differs")
    protocol = json.loads(arguments.embedding_protocol.read_text(encoding="utf-8"))
    compute = protocol.get("compute", {})
    data = protocol.get("data", {})
    if (
        protocol.get("format_version") != "abpairfm_sabdab2_joint_embedding_v1"
        or protocol.get("status") != "FROZEN_BEFORE_GPU_MATERIALIZATION"
        or protocol.get("test_access") is not False
        or protocol.get("task_labels_read") is not False
        or set(data.get("split_roles", [])) != ALLOWED_ROLES
        or data.get("split_sha256") != arguments.expected_split_sha256
        or data.get("sequence_view_sha256")
        != arguments.expected_sequence_view_sha256
        or data.get("sequence_view_receipt_sha256")
        != arguments.expected_sequence_view_receipt_sha256
        or compute.get("partition") != "gpu_4090"
        or int(compute.get("world_size", -1)) != int(os.environ.get("WORLD_SIZE", -2))
        or int(compute.get("maximum_window_residues", -1))
        != arguments.maximum_window_residues
        or int(compute.get("maximum_batch_windows_per_rank", -1))
        != arguments.maximum_batch_size
        or int(compute.get("maximum_attention_cells_per_rank", -1))
        != arguments.maximum_attention_cells
        or protocol.get("checkpoint", {}).get("content_sha256")
        != arguments.expected_dcp_content_sha256
        or protocol.get("checkpoint", {}).get("generation") != arguments.dcp_generation
    ):
        raise RuntimeError("Joint-embedding protocol differs")
    return protocol


def _load_records(arguments: argparse.Namespace) -> dict[str, SequenceRecord]:
    if sha256_file(arguments.sequence_view) != arguments.expected_sequence_view_sha256:
        raise RuntimeError("Train/validation sequence view SHA-256 differs")
    if (
        sha256_file(arguments.sequence_view_receipt)
        != arguments.expected_sequence_view_receipt_sha256
    ):
        raise RuntimeError("Train/validation sequence-view receipt SHA-256 differs")
    if sha256_file(arguments.split_manifest) != arguments.expected_split_sha256:
        raise RuntimeError("Split manifest SHA-256 differs")
    receipt = json.loads(arguments.sequence_view_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("format_version") != "sabdab2_train_validation_sequence_view_v1"
        or receipt.get("output_sha256") != arguments.expected_sequence_view_sha256
        or receipt.get("split_sha256") != arguments.expected_split_sha256
        or receipt.get("source_sequence_authority_sha256")
        != "9e87fb4241012e44837f700634ef4110c652822e6a1c0c162ea49487f830b107"
        or receipt.get("prepared_authority_sha256")
        != "a32aa68bb6780d1222b310bab5e4efbf365c9219e08eb8a897379ceded565335"
        or receipt.get("role_counts") != {"train": 2565, "validation": 549}
        or int(receipt.get("record_count", -1)) != 3114
        or receipt.get("task_labels_read") is not False
        or receipt.get("test_rows_emitted") is not False
        or receipt.get("test_loaded") is not False
    ):
        raise RuntimeError("Train/validation sequence-view receipt differs")
    role_by_id: dict[str, str] = {}
    with arguments.split_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"instance_id", "role"}.issubset(reader.fieldnames):
            raise RuntimeError("Split manifest lacks instance_id/role")
        for row in reader:
            identifier = str(row["instance_id"])
            role = str(row["role"])
            if identifier in role_by_id or role not in {"train", "validation", "test"}:
                raise RuntimeError("Split manifest identifiers or roles are invalid")
            role_by_id[identifier] = role
    records: dict[str, SequenceRecord] = {}
    for line_number, line in enumerate(
        arguments.sequence_view.read_text(encoding="utf-8").splitlines(), start=1
    ):
        row = json.loads(line)
        identifier = str(row["instance_id"])
        role = str(row["role"])
        if identifier in records or role not in ALLOWED_ROLES:
            raise RuntimeError(f"Sequence-view row {line_number} is invalid")
        if role_by_id.get(identifier) != role:
            raise RuntimeError(f"{identifier}: sequence-view split role differs")
        antibody = str(row["ab_seq"]).strip().upper()
        antigen = str(row["ag_seq"]).strip().upper()
        ab_sha = _sequence_sha256(antibody)
        ag_sha = _sequence_sha256(antigen)
        if (
            not antibody
            or not antigen
            or ab_sha != str(row["ab_sequence_sha256"])
            or ag_sha != str(row["ag_sequence_sha256"])
            or len(antibody) != int(row["ab_length"])
            or len(antigen) != int(row["ag_length"])
        ):
            raise RuntimeError(f"{identifier}: sequence authority differs")
        records[identifier] = SequenceRecord(
            instance_id=identifier,
            role=role,
            antibody_sequence=antibody,
            antigen_sequence=antigen,
            antibody_sequence_sha256=ab_sha,
            antigen_sequence_sha256=ag_sha,
        )
    expected_ids = {
        identifier for identifier, role in role_by_id.items() if role in ALLOWED_ROLES
    }
    if (
        set(records) != expected_ids
        or {record.role for record in records.values()} != ALLOWED_ROLES
    ):
        raise RuntimeError("Train/validation-only sequence selection is incomplete")
    return records


def _allocate_records(
    records: Mapping[str, SequenceRecord], world_size: int, maximum: int
) -> list[list[str]]:
    costs = []
    for identifier, record in records.items():
        cost = sum(
            (ab_stop - ab_start + ag_stop - ag_start + 4) ** 2
            for ab_start, ab_stop in _window_spans(record.antibody_length, maximum)
            for ag_start, ag_stop in _window_spans(record.antigen_length, maximum)
        )
        costs.append((cost, identifier))
    assignments = [[] for _ in range(world_size)]
    totals = [0] * world_size
    for cost, identifier in sorted(costs, key=lambda item: (-item[0], item[1])):
        rank = min(range(world_size), key=lambda value: (totals[value], value))
        assignments[rank].append(identifier)
        totals[rank] += cost
    if any(not assignment for assignment in assignments):
        raise RuntimeError("Every rank requires joint-embedding work")
    return [sorted(assignment) for assignment in assignments]


def _window_work(
    records: Mapping[str, SequenceRecord], identifiers: Iterable[str], maximum: int
) -> list[WindowWork]:
    result: list[WindowWork] = []
    for identifier in identifiers:
        record = records[identifier]
        for ab_start, ab_stop in _window_spans(record.antibody_length, maximum):
            for ag_start, ag_stop in _window_spans(record.antigen_length, maximum):
                result.append(
                    WindowWork(
                        instance_id=identifier,
                        antibody_start=ab_start,
                        antibody_stop=ab_stop,
                        antigen_start=ag_start,
                        antigen_stop=ag_stop,
                        antibody_sequence=record.antibody_sequence[ab_start:ab_stop],
                        antigen_sequence=record.antigen_sequence[ag_start:ag_stop],
                    )
                )
    return result


@torch.inference_mode()
def _materialize_local(
    *,
    backbone: PairAwareESM2,
    records: Mapping[str, SequenceRecord],
    identifiers: list[str],
    output_root: Path,
    checkpoint_sha256: str,
    maximum_window: int,
    maximum_batch_size: int,
    maximum_attention_cells: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    work = _window_work(records, identifiers, maximum_window)
    batches = batch_window_work(work, maximum_batch_size, maximum_attention_cells)
    accumulators: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for identifier in identifiers:
        record = records[identifier]
        accumulators[identifier] = (
            torch.zeros(record.antibody_length, 2560, dtype=torch.float32),
            torch.zeros(record.antibody_length, 1, dtype=torch.float32),
            torch.zeros(record.antigen_length, 2560, dtype=torch.float32),
            torch.zeros(record.antigen_length, 1, dtype=torch.float32),
        )
    peak_cuda = 0
    started = time.monotonic()
    backbone.eval()
    for batch_index, batch in enumerate(batches, start=1):
        tokens, chain_ids, offsets = build_contact_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = backbone(
                tokens,
                chain_ids=chain_ids,
                return_attention=False,
                return_hidden_states=True,
            )
        if output.hidden_states is None or output.hidden_states.shape[-1] != 2560:
            raise RuntimeError("ABPairFM hidden-state dimension differs")
        hidden = output.hidden_states.detach().float().cpu()
        for row, (item, offset) in enumerate(zip(batch, offsets)):
            ab_token_start, ab_token_stop, ag_token_start, ag_token_stop = offset
            ab_sum, ab_count, ag_sum, ag_count = accumulators[item.instance_id]
            ab_sum[item.antibody_start : item.antibody_stop] += hidden[
                row, ab_token_start:ab_token_stop
            ]
            ab_count[item.antibody_start : item.antibody_stop] += 1.0
            ag_sum[item.antigen_start : item.antigen_stop] += hidden[
                row, ag_token_start:ag_token_stop
            ]
            ag_count[item.antigen_start : item.antigen_stop] += 1.0
        peak_cuda = max(peak_cuda, int(torch.cuda.max_memory_allocated(device)))
        if batch_index % 50 == 0 or batch_index == len(batches):
            print(
                json.dumps(
                    {
                        "rank": torch.distributed.get_rank(),
                        "batches_complete": batch_index,
                        "batches_total": len(batches),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    result = []
    for identifier in identifiers:
        record = records[identifier]
        ab_sum, ab_count, ag_sum, ag_count = accumulators.pop(identifier)
        if bool((ab_count == 0).any()) or bool((ag_count == 0).any()):
            raise RuntimeError(f"{identifier}: joint windows leave uncovered residues")
        artifact = {
            "format_version": ARTIFACT_FORMAT,
            "instance_id": identifier,
            "split_role": record.role,
            "antibody_sequence_sha256": record.antibody_sequence_sha256,
            "antigen_sequence_sha256": record.antigen_sequence_sha256,
            "antibody_length": record.antibody_length,
            "antigen_length": record.antigen_length,
            "checkpoint_content_sha256": checkpoint_sha256,
            "maximum_window_residues": maximum_window,
            "aggregation": "arithmetic_mean_across_joint_ab_ag_windows",
            "antibody_features": (ab_sum / ab_count).to(torch.bfloat16).contiguous(),
            "antigen_features": (ag_sum / ag_count).to(torch.bfloat16).contiguous(),
        }
        relative = Path("artifacts") / _artifact_name(identifier)
        path = output_root / relative
        _atomic_torch_save(artifact, path)
        result.append(
            {
                "instance_id": identifier,
                "split_role": record.role,
                "artifact_path": relative.as_posix(),
                "artifact_sha256": sha256_file(path),
                "artifact_size": path.stat().st_size,
            }
        )
    return result, {
        "elapsed_seconds": time.monotonic() - started,
        "window_count": len(work),
        "batch_count": len(batches),
        "peak_cuda_bytes": peak_cuda,
    }


def main() -> None:
    arguments = _arguments()
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("Joint embedding materialization requires gpu_4090")
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError("Refusing to overwrite joint-embedding output root")
    checkpoint_protocol = _load_protocol(
        argparse.Namespace(
            **{
                **vars(arguments),
                "protocol": arguments.checkpoint_protocol,
                "expected_protocol_sha256": arguments.expected_checkpoint_protocol_sha256,
            }
        )
    )
    embedding_protocol = _load_embedding_protocol(arguments)
    if (
        arguments.initial_checkpoint.stat().st_size != EXPECTED_INITIAL_CHECKPOINT_SIZE
        or sha256_file(arguments.initial_checkpoint) != EXPECTED_INITIAL_CHECKPOINT_SHA256
    ):
        raise RuntimeError("Initial ESM2 checkpoint authentication failed")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="nccl", device_id=device, timeout=timedelta(minutes=30)
    )
    rank = torch.distributed.get_rank()
    try:
        world_size = torch.distributed.get_world_size()
        if world_size != embedding_protocol["compute"]["world_size"]:
            raise RuntimeError("Joint-embedding world size differs")
        if torch.cuda.get_device_properties(device).name != "NVIDIA GeForce RTX 4090":
            raise RuntimeError("Joint embedding materialization requires RTX 4090")
        torch.manual_seed(20260825 + rank)
        torch.cuda.manual_seed(20260825 + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        receipt = _verify_checkpoint_collectively(arguments, checkpoint_protocol)
        records = _load_records(arguments)
        assignments = _allocate_records(records, world_size, arguments.maximum_window_residues)

        temporary_root = arguments.output_root.with_name(
            f".{arguments.output_root.name}.tmp-{os.environ.get('SLURM_JOB_ID', 'manual')}"
        )
        if rank == 0:
            if temporary_root.exists() or temporary_root.is_symlink():
                raise FileExistsError("Joint-embedding temporary root already exists")
            temporary_root.mkdir(parents=True)
        torch.distributed.barrier()

        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(
            "cuda",
            _replicate_shard_mesh_shape(world_size),
            mesh_dim_names=("replicate", "shard"),
        )
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
        heads = StageA1GeometryHeads(
            pair_dimension=256,
            num_distance_bins=8,
            contact_distance_bin_count=2,
            temperature=0.07,
        ).to(device=device, dtype=torch.float32)
        system = StageA1System(backbone, heads)
        _load_system_dcp(
            system, dcp_root=arguments.dcp_root, generation=arguments.dcp_generation
        )
        local_records, local_runtime = _materialize_local(
            backbone=backbone,
            records=records,
            identifiers=assignments[rank],
            output_root=temporary_root,
            checkpoint_sha256=arguments.expected_dcp_content_sha256,
            maximum_window=arguments.maximum_window_residues,
            maximum_batch_size=arguments.maximum_batch_size,
            maximum_attention_cells=arguments.maximum_attention_cells,
            device=device,
        )
        gathered_records: list[list[dict[str, Any]] | None] = [None] * world_size
        gathered_runtime: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered_records, local_records)
        torch.distributed.all_gather_object(gathered_runtime, local_runtime)
        torch.distributed.barrier()
        if rank == 0:
            flattened = sorted(
                [record for part in gathered_records if part for record in part],
                key=lambda value: value["instance_id"],
            )
            if len(flattened) != len(records) or len({x["instance_id"] for x in flattened}) != len(records):
                raise RuntimeError("Joint-embedding gathered records differ")
            role_counts = {
                role: sum(record.role == role for record in records.values())
                for role in sorted(ALLOWED_ROLES)
            }
            manifest = {
                "format_version": FORMAT_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "COMPLETE_TRAIN_VALIDATION_ONLY",
                "checkpoint": {
                    "content_sha256": arguments.expected_dcp_content_sha256,
                    "generation": arguments.dcp_generation,
                    "training_state": receipt["training_state"],
                },
                "authority": {
                    "sequence_view_sha256": arguments.expected_sequence_view_sha256,
                    "sequence_view_receipt_sha256": arguments.expected_sequence_view_receipt_sha256,
                    "split_sha256": arguments.expected_split_sha256,
                    "checkpoint_protocol_sha256": arguments.expected_checkpoint_protocol_sha256,
                    "embedding_protocol_sha256": arguments.expected_embedding_protocol_sha256,
                    "columns_read": ["instance_id", "ab_seq", "ag_seq"],
                    "task_labels_read": False,
                    "test_loaded": False,
                },
                "feature_definition": {
                    "dimension": 2560,
                    "dtype": "bfloat16",
                    "representation": "joint_cross_chain_attention_hidden_state",
                    "maximum_window_residues": arguments.maximum_window_residues,
                    "aggregation": "arithmetic_mean_across_joint_ab_ag_windows",
                },
                "role_counts": role_counts,
                "record_count": len(flattened),
                "artifact_bytes": sum(record["artifact_size"] for record in flattened),
                "records": flattened,
                "runtime_by_rank": gathered_runtime,
                "test_access": False,
            }
            _publish_create_only(
                temporary_root / "SABDAB2_ABPAIRFM_JOINT_EMBEDDING_MANIFEST.json",
                _canonical_bytes(manifest),
            )
            os.rename(temporary_root, arguments.output_root)
            print(
                json.dumps(
                    {
                        "output_root": str(arguments.output_root),
                        "manifest_sha256": sha256_file(
                            arguments.output_root
                            / "SABDAB2_ABPAIRFM_JOINT_EMBEDDING_MANIFEST.json"
                        ),
                        "record_count": len(flattened),
                        "role_counts": role_counts,
                        "test_loaded": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        torch.distributed.barrier()
    except BaseException:
        if rank == 0:
            temporary = arguments.output_root.with_name(
                f".{arguments.output_root.name}.tmp-{os.environ.get('SLURM_JOB_ID', 'manual')}"
            )
            if temporary.exists() and not arguments.output_root.exists():
                failed = arguments.output_root.with_name(
                    f"{arguments.output_root.name}.failed-{os.environ.get('SLURM_JOB_ID', 'manual')}"
                )
                if not failed.exists():
                    shutil.move(temporary, failed)
        raise
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
