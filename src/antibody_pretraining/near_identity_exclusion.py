"""Compile authenticated fixed-IMGT and gap-aware hits into digest exclusions.

The two search tracks identify raw OAS authority ordinals.  Production OAS
shards, however, are deduplicated by full sequence SHA-256.  This module joins
the authenticated search evidence back to the authenticated target export,
then emits one globally sorted, unique digest stream suitable for filtering the
already materialized corpus.

MMseqs2 remains a heuristic search engine.  A successful compilation therefore
proves faithful union and mapping of the two completed search tracks; it does
not turn the gap-aware track into an exhaustive ground-truth oracle and does
not by itself authorize pretraining.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
from typing import Any, BinaryIO, Dict, Iterable, Mapping, Optional, Tuple
import uuid

from .imgt_search_output import FIXED_IMGT_MATCH_RECORD, unpack_fixed_imgt_match
from .mmseqs_gap_aware import (
    GLOBAL_TARGET_FORMAT,
    TEST_ACCESS,
    canonical_json_bytes,
    parse_target_identifier,
    publish_export_receipt,
    require_sha256,
    sha256_file,
    signed_json,
)
from .oas_imgt_export import (
    IMGT_TARGET_RECORD,
    UNUSUAL_TARGET_RECORD,
    unpack_imgt_target,
    unpack_unusual_target,
)
from .production_materialization import SPLIT_TO_ID
from .tokenizer import CHAIN_TYPE_TO_ID


FORMAT_VERSION = "oas_near_identity_exclusion_compilation_v1"
EXACT_GLOBAL_FORMAT = "oas_imgt_exact_search_global_v1"
EXACT_TASK_FORMAT = "oas_imgt_exact_search_task_v1"
GAP_GLOBAL_FORMAT = "oas_mmseqs_gap_search_global_v1"
GAP_TASK_FORMAT = "oas_mmseqs_gap_search_task_v1"
TARGET_TEST_ACCESS = {
    "downstream_metrics_used": False,
    "sabdab2_heldout_accessed": False,
}
EXCLUSION_DIGEST_RECORD = struct.Struct(">32s")
_RAW_DIGEST_MASK_RECORD = struct.Struct(">32sB")
TRACK_EXACT = 1
TRACK_GAP = 2
_TRACK_MASK = TRACK_EXACT | TRACK_GAP


class NearIdentityExclusionError(RuntimeError):
    """Report unauthenticated, incomplete, or contradictory exclusion evidence."""


def _regular_file(path: Path, label: str, *, frozen: bool = False) -> Path:
    candidate = Path(path).expanduser().absolute()
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise NearIdentityExclusionError("Cannot stat " + label) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise NearIdentityExclusionError(label + " is absent, symbolic, or non-regular")
    if frozen and metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise NearIdentityExclusionError(label + " remains writable")
    return candidate


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NearIdentityExclusionError(label + " is not an object")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise NearIdentityExclusionError(label + " is not a list")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NearIdentityExclusionError(label + " is not a non-negative integer")
    return value


def _expected_receipt(
    path: Path, *, label: str, expected_receipt_sha256: str
) -> Tuple[Mapping[str, Any], str]:
    expected = require_sha256(expected_receipt_sha256, label + " expected hash")
    try:
        receipt, file_sha256 = signed_json(path, label)
    except Exception as error:
        raise NearIdentityExclusionError(str(error)) from error
    if receipt.get("receipt_sha256") != expected:
        raise NearIdentityExclusionError(label + " does not match the frozen hash")
    return receipt, file_sha256


def _ordered_tasks(
    receipt: Mapping[str, Any], *, label: str, task_count: int
) -> list[Mapping[str, Any]]:
    tasks = _require_list(receipt.get("task_receipts"), label + " tasks")
    if len(tasks) != task_count:
        raise NearIdentityExclusionError(label + " task count differs")
    output = []
    for task_id, value in enumerate(tasks):
        task = _require_mapping(value, label + " task")
        if task.get("task_id") != task_id:
            raise NearIdentityExclusionError(label + " task ordering differs")
        output.append(task)
    return output


def _task_receipt(
    path: Path,
    *,
    label: str,
    global_task: Mapping[str, Any],
    expected_format: str,
    task_id: int,
) -> Mapping[str, Any]:
    try:
        receipt, file_sha256 = signed_json(path, label)
    except Exception as error:
        raise NearIdentityExclusionError(str(error)) from error
    if (
        receipt.get("format_version") != expected_format
        or receipt.get("task_id") != task_id
        or receipt.get("receipt_sha256") != global_task.get("receipt_sha256")
        or file_sha256 != global_task.get("receipt_file_sha256")
    ):
        raise NearIdentityExclusionError(label + " differs from the global receipt")
    return receipt


def _target_stream(
    path: Path,
    *,
    inventory: Mapping[str, Any],
    record_width: int,
    label: str,
) -> Tuple[Path, int]:
    candidate = _regular_file(path, label, frozen=True)
    records = _nonnegative_integer(inventory.get("records"), label + " records")
    expected_bytes = _nonnegative_integer(inventory.get("bytes"), label + " bytes")
    if (
        inventory.get("record_bytes") != record_width
        or expected_bytes != records * record_width
        or candidate.stat().st_size != expected_bytes
        or sha256_file(candidate) != inventory.get("file_sha256")
    ):
        raise NearIdentityExclusionError(label + " inventory differs")
    return candidate, records


def _iter_records(
    handle: BinaryIO, *, width: int, label: str
) -> Iterable[bytes]:
    while True:
        payload = handle.read(width)
        if not payload:
            return
        if len(payload) != width:
            raise NearIdentityExclusionError(label + " ended mid-record")
        yield payload


def _encode_metadata(*, split: str, chain_type: Optional[str], tracks: int) -> int:
    try:
        split_id = SPLIT_TO_ID[split]
        chain_id = 0 if chain_type is None else CHAIN_TYPE_TO_ID[chain_type]
    except KeyError as error:
        raise NearIdentityExclusionError("Matched target metadata is invalid") from error
    if tracks not in {TRACK_EXACT, TRACK_GAP} or chain_id not in {0, 1, 2, 3}:
        raise NearIdentityExclusionError("Matched target metadata is outside its domain")
    return tracks | ((split_id + 1) << 2) | (chain_id << 4)


def _merge_metadata(previous: int, incoming: int) -> int:
    previous_split = (previous >> 2) & 0b11
    incoming_split = (incoming >> 2) & 0b11
    previous_chain = (previous >> 4) & 0b11
    incoming_chain = (incoming >> 4) & 0b11
    if previous_split and incoming_split and previous_split != incoming_split:
        raise NearIdentityExclusionError("Search tracks disagree on target split")
    if previous_chain and incoming_chain and previous_chain != incoming_chain:
        raise NearIdentityExclusionError("Search tracks disagree on target chain")
    return (
        (previous | incoming) & _TRACK_MASK
        | (max(previous_split, incoming_split) << 2)
        | (max(previous_chain, incoming_chain) << 4)
    )


def _add_target_metadata(
    targets: Dict[int, int], *, ordinal: int, metadata: int
) -> bool:
    previous = targets.get(ordinal)
    if previous is None:
        targets[ordinal] = metadata
        return True
    merged = _merge_metadata(previous, metadata)
    changed = merged != previous
    targets[ordinal] = merged
    return changed


def _load_exact_targets(
    *,
    root: Path,
    task_id: int,
    global_task: Mapping[str, Any],
    targets: Dict[int, int],
) -> Counter[str]:
    task_root = root / "fixed-imgt" / f"mapper-{task_id:04d}"
    receipt = _task_receipt(
        task_root / "receipt.json",
        label=f"fixed-IMGT task receipt {task_id}",
        global_task=global_task,
        expected_format=EXACT_TASK_FORMAT,
        task_id=task_id,
    )
    output = _require_mapping(receipt.get("output"), "fixed-IMGT task output")
    matches = _regular_file(task_root / "matches.bin", "fixed-IMGT matches", frozen=True)
    expected_bytes = _nonnegative_integer(
        output.get("matches_bin_bytes"), "fixed-IMGT match bytes"
    )
    if (
        output.get("match_record_bytes") != FIXED_IMGT_MATCH_RECORD.size
        or expected_bytes != matches.stat().st_size
        or expected_bytes % FIXED_IMGT_MATCH_RECORD.size
        or sha256_file(matches) != output.get("matches_bin_sha256")
    ):
        raise NearIdentityExclusionError("Fixed-IMGT match inventory differs")
    observed = Counter()
    unique_targets = set()
    with matches.open("rb", buffering=16 * 1024 * 1024) as handle:
        for payload in _iter_records(
            handle, width=FIXED_IMGT_MATCH_RECORD.size, label="fixed-IMGT matches"
        ):
            match = unpack_fixed_imgt_match(payload)
            metadata = _encode_metadata(
                split=match.target_split,
                chain_type=None,
                tracks=TRACK_EXACT,
            )
            _add_target_metadata(
                targets,
                ordinal=match.target_authority_ordinal,
                metadata=metadata,
            )
            unique_targets.add(match.target_authority_ordinal)
            observed["match_records"] += 1
            observed["split:" + match.target_split] += 1
    summary = _require_mapping(global_task.get("match_summary"), "exact task summary")
    split_counts = _require_mapping(
        summary.get("split_match_counts"), "exact task split summary"
    )
    if (
        observed["match_records"] != summary.get("match_records")
        or len(unique_targets) != summary.get("matched_target_count")
        or observed["split:train"] != split_counts.get("train")
        or observed["split:validation"] != split_counts.get("validation")
    ):
        raise NearIdentityExclusionError("Fixed-IMGT task replay differs")
    observed["matched_targets"] = len(unique_targets)
    return observed


def _load_gap_targets(
    *,
    root: Path,
    task_id: int,
    global_task: Mapping[str, Any],
    targets: Dict[int, int],
) -> Counter[str]:
    task_root = root / "gap-aware" / f"mapper-{task_id:04d}"
    receipt = _task_receipt(
        task_root / "receipt.json",
        label=f"gap-aware task receipt {task_id}",
        global_task=global_task,
        expected_format=GAP_TASK_FORMAT,
        task_id=task_id,
    )
    output = _require_mapping(receipt.get("output"), "gap-aware task output")
    hits = _regular_file(task_root / "hits.tsv", "gap-aware hits", frozen=True)
    expected_bytes = _nonnegative_integer(output.get("hits_bytes"), "gap-aware hit bytes")
    if (
        hits.stat().st_size != expected_bytes
        or sha256_file(hits) != output.get("hits_sha256")
    ):
        raise NearIdentityExclusionError("Gap-aware hit inventory differs")
    observed = Counter()
    unique_targets = set()
    with hits.open("r", encoding="ascii", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise NearIdentityExclusionError(
                    f"Gap-aware hit row {line_number} has the wrong width"
                )
            metadata = parse_target_identifier(fields[1])
            if metadata.get("task_id") != task_id:
                raise NearIdentityExclusionError("Gap-aware hit crosses target shards")
            ordinal = int(metadata["authority_ordinal"])
            encoded = _encode_metadata(
                split=str(metadata["split"]),
                chain_type=str(metadata["chain_type"]),
                tracks=TRACK_GAP,
            )
            _add_target_metadata(targets, ordinal=ordinal, metadata=encoded)
            unique_targets.add(ordinal)
            observed["hit_records"] += 1
            observed["split:" + str(metadata["split"])] += 1
    summary = _require_mapping(global_task.get("hit_summary"), "gap task summary")
    split_counts = _require_mapping(
        summary.get("split_hit_counts"), "gap task split summary"
    )
    if (
        observed["hit_records"] != summary.get("alignment_rows")
        or len(unique_targets) != summary.get("matched_target_count")
        or observed["split:train"] != split_counts.get("train")
        or observed["split:validation"] != split_counts.get("validation")
    ):
        raise NearIdentityExclusionError("Gap-aware task replay differs")
    observed["matched_targets"] = len(unique_targets)
    return observed


def _append_raw_records(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(descriptor, "ab", buffering=16 * 1024 * 1024) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def _map_task_targets(
    *,
    target_root: Path,
    task_id: int,
    target_task: Mapping[str, Any],
    targets: Dict[int, int],
    raw_bucket_root: Path,
) -> Counter[str]:
    inventory = _require_mapping(target_task.get("output_inventory"), "target inventory")
    representable_inventory = _require_mapping(
        inventory.get("representable"), "representable target inventory"
    )
    unusual_inventory = _require_mapping(
        inventory.get("unusual"), "unusual target inventory"
    )
    task_root = target_root / "imgt-map" / f"mapper-{task_id:04d}"
    representable, representable_records = _target_stream(
        task_root / "representable.bin",
        inventory=representable_inventory,
        record_width=IMGT_TARGET_RECORD.size,
        label="representable target stream",
    )
    unusual, unusual_records = _target_stream(
        task_root / "unusual.bin",
        inventory=unusual_inventory,
        record_width=UNUSUAL_TARGET_RECORD.size,
        label="unusual target stream",
    )
    counts = _require_mapping(target_task.get("counts"), "target task counts")
    if (
        counts.get("representable_records") != representable_records
        or counts.get("unusual_records") != unusual_records
        or counts.get("accepted_records") != representable_records + unusual_records
    ):
        raise NearIdentityExclusionError("Target task count accounting differs")
    bucket_payloads = {}  # type: Dict[int, bytearray]
    observed = Counter()
    streams = (
        (representable, IMGT_TARGET_RECORD.size, unpack_imgt_target, True),
        (unusual, UNUSUAL_TARGET_RECORD.size, unpack_unusual_target, False),
    )
    for path, width, unpacker, fixed_representable in streams:
        with path.open("rb", buffering=16 * 1024 * 1024) as handle:
            for payload in _iter_records(handle, width=width, label="target stream"):
                record = unpacker(payload)
                metadata = targets.pop(record.authority_ordinal, None)
                if metadata is None:
                    continue
                tracks = metadata & _TRACK_MASK
                expected_split = ((metadata >> 2) & 0b11) - 1
                expected_chain = (metadata >> 4) & 0b11
                if (
                    tracks not in {TRACK_EXACT, TRACK_GAP, _TRACK_MASK}
                    or expected_split != SPLIT_TO_ID[record.split]
                    or (expected_chain and expected_chain != CHAIN_TYPE_TO_ID[record.chain_type])
                    or (tracks & TRACK_EXACT and not fixed_representable)
                ):
                    raise NearIdentityExclusionError(
                        "Matched target metadata differs from target export"
                    )
                digest = bytes.fromhex(record.sequence_sha256)
                bucket_payloads.setdefault(digest[0], bytearray()).extend(
                    _RAW_DIGEST_MASK_RECORD.pack(digest, tracks)
                )
                observed["matched_target_records"] += 1
                observed["track:" + str(tracks)] += 1
                observed["split:" + record.split] += 1
    if targets:
        raise NearIdentityExclusionError("Matched target ordinal is absent from its shard")
    for bucket, payload in sorted(bucket_payloads.items()):
        _append_raw_records(raw_bucket_root / f"bucket-{bucket:02x}.raw", bytes(payload))
    return observed


def _write_digest_union(
    *, raw_bucket_root: Path, output_path: Path
) -> Tuple[Counter[str], str]:
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    digest_hash = hashlib.sha256()
    counts = Counter()
    previous_global = None  # type: Optional[bytes]
    try:
        with os.fdopen(descriptor, "wb", buffering=16 * 1024 * 1024) as output:
            for bucket in range(256):
                path = raw_bucket_root / f"bucket-{bucket:02x}.raw"
                if not path.exists():
                    continue
                raw = _regular_file(path, "raw digest bucket").read_bytes()
                if len(raw) % _RAW_DIGEST_MASK_RECORD.size:
                    raise NearIdentityExclusionError("Raw digest bucket ended mid-record")
                rows = [
                    raw[index : index + _RAW_DIGEST_MASK_RECORD.size]
                    for index in range(0, len(raw), _RAW_DIGEST_MASK_RECORD.size)
                ]
                rows.sort()
                index = 0
                while index < len(rows):
                    digest, mask = _RAW_DIGEST_MASK_RECORD.unpack(rows[index])
                    end = index + 1
                    while end < len(rows) and rows[end][:32] == digest:
                        mask |= rows[end][32]
                        end += 1
                    if mask not in {TRACK_EXACT, TRACK_GAP, _TRACK_MASK}:
                        raise NearIdentityExclusionError("Digest track mask is invalid")
                    if digest[0] != bucket or (
                        previous_global is not None and digest <= previous_global
                    ):
                        raise NearIdentityExclusionError("Digest union ordering differs")
                    output.write(digest)
                    digest_hash.update(digest)
                    previous_global = digest
                    counts["unique_digests"] += 1
                    counts["track:" + str(mask)] += 1
                    index = end
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return counts, digest_hash.hexdigest()


def _resource_receipt() -> Mapping[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"", "NoDevFiles"}:
        raise NearIdentityExclusionError("Exclusion compiler must not expose a GPU")
    assigned = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS", "")
    assigned_count = len([item for item in assigned.split(",") if item])
    declared = os.environ.get("OAS_SCHEDULER_RESERVED_GPU_COUNT")
    if declared == "1" and assigned_count == 1:
        mode = "scheduler_mandated_idle_only"
    elif declared is None and assigned_count == 0:
        mode = "none"
    else:
        raise NearIdentityExclusionError("Compiler scheduler resource evidence differs")
    return {
        "cuda_visible_devices": False,
        "gpu_compute_permitted": False,
        "scheduler_reserved_gpu_count": assigned_count,
        "scheduler_reserved_gpu_mode": mode,
    }


def _remove_partial_tree(path: Path) -> None:
    """Remove a failed publication even if its final chmod already ran."""
    if not path.exists():
        return
    for child in (path, *path.rglob("*")):
        try:
            mode = child.lstat().st_mode
            if stat.S_ISDIR(mode):
                child.chmod(mode | stat.S_IWUSR | stat.S_IXUSR)
            elif stat.S_ISREG(mode):
                child.chmod(mode | stat.S_IWUSR)
        except FileNotFoundError:
            continue
    shutil.rmtree(path)


def compile_exclusions(
    *,
    exact_root: Path,
    gap_root: Path,
    target_root: Path,
    output_root: Path,
    expected_exact_receipt_sha256: str,
    expected_gap_receipt_sha256: str,
    expected_target_receipt_sha256: str,
    controller_source_sha256: str,
    task_count: int = 32,
) -> Mapping[str, Any]:
    """Authenticate both tracks and create one digest-level exclusion union."""
    if task_count != 32:
        raise NearIdentityExclusionError("Exclusion compiler requires all 32 tasks")
    controller_source_sha256 = require_sha256(
        controller_source_sha256, "Compiler controller source"
    )
    exact_root = Path(exact_root).expanduser().absolute()
    gap_root = Path(gap_root).expanduser().absolute()
    target_root = Path(target_root).expanduser().absolute()
    output_root = Path(output_root).expanduser().absolute()
    for root, label in (
        (exact_root, "exact root"),
        (gap_root, "gap root"),
        (target_root, "target root"),
    ):
        if root.is_symlink() or not root.is_dir():
            raise NearIdentityExclusionError(label + " is absent or symbolic")
    if output_root.exists() or output_root.is_symlink():
        raise NearIdentityExclusionError("Refusing to overwrite exclusion output")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = output_root.parent / (output_root.name + ".partial-" + uuid.uuid4().hex)
    partial.mkdir(mode=0o700)
    raw_bucket_root = partial / "raw-buckets"
    raw_bucket_root.mkdir(mode=0o700)
    try:
        exact, exact_file_sha256 = _expected_receipt(
            exact_root / "fixed_imgt_global_receipt.json",
            label="fixed-IMGT global receipt",
            expected_receipt_sha256=expected_exact_receipt_sha256,
        )
        gap, gap_file_sha256 = _expected_receipt(
            gap_root / "gap_aware_global_receipt.json",
            label="gap-aware global receipt",
            expected_receipt_sha256=expected_gap_receipt_sha256,
        )
        target, target_file_sha256 = _expected_receipt(
            target_root / "global_receipt.json",
            label="target global receipt",
            expected_receipt_sha256=expected_target_receipt_sha256,
        )
        exact_boundary = _require_mapping(exact.get("claim_boundary"), "exact boundary")
        gap_boundary = _require_mapping(gap.get("claim_boundary"), "gap boundary")
        target_boundary = _require_mapping(target.get("claim_boundary"), "target boundary")
        if (
            exact.get("format_version") != EXACT_GLOBAL_FORMAT
            or gap.get("format_version") != GAP_GLOBAL_FORMAT
            or target.get("format_version") != GLOBAL_TARGET_FORMAT
            or exact.get("task_count") != task_count
            or gap.get("task_count") != task_count
            or target.get("task_count") != task_count
            or exact.get("test_access") != TEST_ACCESS
            or gap.get("test_access") != TEST_ACCESS
            or target.get("test_access") != TARGET_TEST_ACCESS
            or exact_boundary.get("fixed_imgt_representable_target_search_complete")
            is not True
            or exact_boundary.get("gap_aware_search_complete") is not False
            or gap_boundary.get("gap_aware_search_complete") is not True
            or gap_boundary.get("hit_list_truncation_detected") is not False
            or gap_boundary.get("treat_mmseqs_as_exhaustive_ground_truth") is not False
            or target_boundary.get("global_task_set_complete") is not True
            or target_boundary.get("accepted_oas_domain_partition_complete") is not True
        ):
            raise NearIdentityExclusionError("Search or target global scope differs")
        exact_inputs = _require_mapping(exact.get("inputs"), "exact inputs")
        gap_inputs = _require_mapping(gap.get("input"), "gap inputs")
        if (
            exact_inputs.get("target_receipt_sha256") != target["receipt_sha256"]
            or exact_inputs.get("target_receipt_file_sha256") != target_file_sha256
            or gap_inputs.get("global_target_receipt_sha256") != target["receipt_sha256"]
            or gap_inputs.get("global_target_receipt_file_sha256") != target_file_sha256
        ):
            raise NearIdentityExclusionError("Search tracks bind different target exports")
        exact_tasks = _ordered_tasks(exact, label="exact global receipt", task_count=task_count)
        gap_tasks = _ordered_tasks(gap, label="gap global receipt", task_count=task_count)
        target_tasks = _ordered_tasks(target, label="target global receipt", task_count=task_count)
        raw_counts = Counter()
        task_summaries = []
        for task_id in range(task_count):
            targets = {}  # type: Dict[int, int]
            exact_counts = _load_exact_targets(
                root=exact_root,
                task_id=task_id,
                global_task=exact_tasks[task_id],
                targets=targets,
            )
            gap_counts = _load_gap_targets(
                root=gap_root,
                task_id=task_id,
                global_task=gap_tasks[task_id],
                targets=targets,
            )
            mapped = _map_task_targets(
                target_root=target_root,
                task_id=task_id,
                target_task=target_tasks[task_id],
                targets=targets,
                raw_bucket_root=raw_bucket_root,
            )
            raw_counts.update(mapped)
            task_summaries.append(
                {
                    "exact_match_records": exact_counts["match_records"],
                    "exact_matched_targets": exact_counts["matched_targets"],
                    "gap_hit_records": gap_counts["hit_records"],
                    "gap_matched_targets": gap_counts["matched_targets"],
                    "mapped_both_targets": mapped["track:3"],
                    "mapped_exact_only_targets": mapped["track:1"],
                    "mapped_gap_only_targets": mapped["track:2"],
                    "task_id": task_id,
                }
            )
        digest_path = partial / "excluded_sequence_sha256.bin"
        digest_counts, digest_sha256 = _write_digest_union(
            raw_bucket_root=raw_bucket_root, output_path=digest_path
        )
        shutil.rmtree(raw_bucket_root)
        receipt = {
            "claim_boundary": {
                "fixed_imgt_search_complete": True,
                "gap_aware_search_complete": True,
                "large_scale_pretraining_authorized": False,
                "matched_target_digest_compilation_complete": True,
                "near_identity_exclusion_list_complete": True,
                "original_training_corpus_filtered": False,
                "scientific_claim": False,
                "treat_mmseqs_as_exhaustive_ground_truth": False,
                "zero_overlap_replay_complete": False,
            },
            "counts": {
                "raw_matched_both_targets": raw_counts["track:3"],
                "raw_matched_exact_only_targets": raw_counts["track:1"],
                "raw_matched_gap_only_targets": raw_counts["track:2"],
                "raw_matched_target_records": raw_counts["matched_target_records"],
                "raw_matched_train_targets": raw_counts["split:train"],
                "raw_matched_validation_targets": raw_counts["split:validation"],
                "unique_both_digests": digest_counts["track:3"],
                "unique_exact_only_digests": digest_counts["track:1"],
                "unique_excluded_sequence_sha256": digest_counts["unique_digests"],
                "unique_gap_only_digests": digest_counts["track:2"],
            },
            "format_version": FORMAT_VERSION,
            "input": {
                "exact_global_receipt_file_sha256": exact_file_sha256,
                "exact_global_receipt_sha256": exact["receipt_sha256"],
                "gap_global_receipt_file_sha256": gap_file_sha256,
                "gap_global_receipt_sha256": gap["receipt_sha256"],
                "target_global_receipt_file_sha256": target_file_sha256,
                "target_global_receipt_sha256": target["receipt_sha256"],
            },
            "output": {
                "excluded_sequence_sha256_bytes": digest_path.stat().st_size,
                "excluded_sequence_sha256_file_sha256": digest_sha256,
                "excluded_sequence_sha256_record_bytes": EXCLUSION_DIGEST_RECORD.size,
                "excluded_sequence_sha256_records": digest_counts["unique_digests"],
                "global_order": "sequence_sha256_ascending",
            },
            "resource_receipt": _resource_receipt(),
            "source_sha256": {
                "controller": controller_source_sha256,
                "exclusion_core": sha256_file(Path(__file__)),
            },
            "task_count": task_count,
            "task_summaries": task_summaries,
            "test_access": TEST_ACCESS,
        }
        publish_export_receipt(partial / "receipt.json", receipt)
        for path in partial.iterdir():
            path.chmod(0o400)
        partial.chmod(0o500)
        os.replace(partial, output_root)
        return json.loads((output_root / "receipt.json").read_text(encoding="ascii"))
    except BaseException:
        _remove_partial_tree(partial)
        raise


__all__ = [
    "EXCLUSION_DIGEST_RECORD",
    "FORMAT_VERSION",
    "NearIdentityExclusionError",
    "TRACK_EXACT",
    "TRACK_GAP",
    "compile_exclusions",
]
