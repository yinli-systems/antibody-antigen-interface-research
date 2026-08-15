#!/usr/bin/env python3
"""Run and independently verify one exhaustive CUDA fixed-IMGT search shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

from antibody_pretraining.imgt_position import position_identity
from antibody_pretraining.imgt_query_export import (
    ASEP_QUERY_EXPORT_FORMAT,
    IMGT_QUERY_MANIFEST_FORMAT,
    IMGT_QUERY_RECORD,
    IMGTQueryRecord,
    unpack_imgt_query,
)
from antibody_pretraining.imgt_search_output import (
    FIXED_IMGT_MATCH_RECORD,
    FixedIMGTOutputMatch,
    unpack_fixed_imgt_match,
)
from antibody_pretraining.near_identity_export import publish_export_receipt
from antibody_pretraining.oas_imgt_export import (
    IMGT_TARGET_RECORD,
    unpack_imgt_target,
)


FORMAT_VERSION = "oas_imgt_exact_search_task_v1"
GLOBAL_TARGET_FORMAT = "oas_imgt_target_export_global_v1"
KERNEL_STATS_FORMAT = "oas_imgt_exact_cuda_stats_v1"
TEST_ACCESS = {
    "development_sequence_fields_accessed": True,
    "frozen_official_external_test_accessed": False,
    "model_predictions_or_scores_accessed": False,
    "task_label_values_used": False,
}


class OASIMGTSearchTaskError(RuntimeError):
    """Report unauthenticated input, kernel failure, or output disagreement."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise OASIMGTSearchTaskError("Payload is not finite canonical JSON") from error
    return (payload + "\n").encode("ascii")


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(
    path: Path, label: str, *, require_canonical: bool = True
) -> tuple[dict[str, Any], bytes]:
    path = _regular_file(path, label)
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise OASIMGTSearchTaskError(f"{label} is not strict ASCII JSON") from error
    if not isinstance(value, dict) or (
        require_canonical and raw != _canonical_json_bytes(value)
    ):
        raise OASIMGTSearchTaskError(f"{label} is not canonical JSON")
    return value, raw


def _regular_file(path: Path, label: str) -> Path:
    candidate = Path(path).expanduser().absolute()
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise OASIMGTSearchTaskError(f"Cannot stat {label}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise OASIMGTSearchTaskError(f"{label} is absent, symbolic, or non-regular")
    return candidate


def _sha256(path: Path) -> str:
    path = _regular_file(path, "hashed file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise OASIMGTSearchTaskError(f"{label} is not lowercase SHA-256")
    return value


def _require_self_hash(value: Mapping[str, Any], label: str) -> str:
    unsigned = dict(value)
    claimed = _require_sha256(unsigned.pop("receipt_sha256", None), f"{label} hash")
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise OASIMGTSearchTaskError(f"{label} self hash differs")
    return claimed


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OASIMGTSearchTaskError(f"{label} is not a non-negative integer")
    return value


def _authenticate_target(
    *, target_output_root: Path, task_id: int
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    target_output_root = Path(target_output_root).expanduser().absolute()
    global_receipt, _ = _strict_json(
        target_output_root / "global_receipt.json", "global target receipt"
    )
    _require_self_hash(global_receipt, "global target receipt")
    boundary = global_receipt.get("claim_boundary")
    tasks = global_receipt.get("task_receipts")
    if (
        global_receipt.get("format_version") != GLOBAL_TARGET_FORMAT
        or not isinstance(boundary, Mapping)
        or boundary.get("global_task_set_complete") is not True
        or boundary.get("accepted_oas_domain_partition_complete") is not True
        or boundary.get("near_identity_search_complete") is not False
        or not isinstance(tasks, list)
        or global_receipt.get("task_count") != len(tasks)
        or not 0 <= task_id < len(tasks)
    ):
        raise OASIMGTSearchTaskError("Global target receipt is incomplete or out of scope")
    task = tasks[task_id]
    if not isinstance(task, Mapping) or task.get("task_id") != task_id:
        raise OASIMGTSearchTaskError("Target task receipt ordering differs")
    inventory = task.get("output_inventory")
    if not isinstance(inventory, Mapping) or not isinstance(
        inventory.get("representable"), Mapping
    ):
        raise OASIMGTSearchTaskError("Target representable inventory is absent")
    representable = dict(inventory["representable"])
    records = _nonnegative_integer(
        representable.get("records"), "Target record count"
    )
    byte_count = _nonnegative_integer(representable.get("bytes"), "Target bytes")
    expected_hash = _require_sha256(
        representable.get("file_sha256"), "Target file hash"
    )
    target_path = _regular_file(
        target_output_root
        / "imgt-map"
        / f"mapper-{task_id:04d}"
        / "representable.bin",
        "target binary",
    )
    if (
        representable.get("record_bytes") != IMGT_TARGET_RECORD.size
        or byte_count != records * IMGT_TARGET_RECORD.size
        or target_path.stat().st_size != byte_count
        or _sha256(target_path) != expected_hash
    ):
        raise OASIMGTSearchTaskError("Target binary inventory or hash differs")
    return target_path, representable, global_receipt


def _read_query_records(path: Path) -> tuple[IMGTQueryRecord, ...]:
    payload = _regular_file(path, "query binary").read_bytes()
    if not payload or len(payload) % IMGT_QUERY_RECORD.size:
        raise OASIMGTSearchTaskError("Query binary size is not a record multiple")
    records = tuple(
        unpack_imgt_query(payload[offset : offset + IMGT_QUERY_RECORD.size])
        for offset in range(0, len(payload), IMGT_QUERY_RECORD.size)
    )
    ordinals = [record.query_ordinal for record in records]
    if len(ordinals) != len(set(ordinals)) or ordinals != sorted(ordinals):
        raise OASIMGTSearchTaskError("Query ordinals are duplicated or unordered")
    return records


def _authenticate_queries(
    query_root: Path,
) -> tuple[Path, tuple[IMGTQueryRecord, ...], dict[str, Any]]:
    query_root = Path(query_root).expanduser().absolute()
    receipt, _ = _strict_json(query_root / "EXPORT_RECEIPT.json", "query receipt")
    _require_self_hash(receipt, "query receipt")
    manifest, manifest_raw = _strict_json(
        query_root / "imgt_query_manifest.json", "query manifest"
    )
    _require_self_hash(manifest, "query manifest")
    outputs = receipt.get("outputs")
    if (
        receipt.get("format_version") != ASEP_QUERY_EXPORT_FORMAT
        or manifest.get("format_version") != IMGT_QUERY_MANIFEST_FORMAT
        or not isinstance(outputs, Mapping)
        or receipt.get("test_access") != TEST_ACCESS
        or manifest.get("test_access") != TEST_ACCESS
        or receipt.get("outside_canonical_query_count") != 3
        or receipt.get("gap_aware_query_count") != 3_106
    ):
        raise OASIMGTSearchTaskError("Query receipt scope or exhaustive handoff differs")
    query_path = _regular_file(query_root / "queries.imgt.bin", "query binary")
    for name, path in (
        ("queries.imgt.bin", query_path),
        ("imgt_query_manifest.json", query_root / "imgt_query_manifest.json"),
    ):
        inventory = outputs.get(name)
        if (
            not isinstance(inventory, Mapping)
            or inventory.get("bytes") != path.stat().st_size
            or inventory.get("sha256") != _sha256(path)
        ):
            raise OASIMGTSearchTaskError(f"Query output inventory differs for {name}")
    if hashlib.sha256(manifest_raw).hexdigest() != outputs[
        "imgt_query_manifest.json"
    ]["sha256"]:
        raise OASIMGTSearchTaskError("Query manifest raw hash differs")
    records = _read_query_records(query_path)
    if (
        len(records) != receipt.get("fixed_imgt_query_count")
        or len(records) != manifest.get("fixed_imgt_query_count")
        or len(records) != 3_103
    ):
        raise OASIMGTSearchTaskError("Fixed-IMGT query count differs")
    return query_path, records, receipt


def _iter_match_records(path: Path) -> Iterable[FixedIMGTOutputMatch]:
    path = _regular_file(path, "match binary")
    if path.stat().st_size % FIXED_IMGT_MATCH_RECORD.size:
        raise OASIMGTSearchTaskError("Match binary size is not a record multiple")
    with path.open("rb") as handle:
        while True:
            payload = handle.read(FIXED_IMGT_MATCH_RECORD.size)
            if not payload:
                return
            if len(payload) != FIXED_IMGT_MATCH_RECORD.size:
                raise OASIMGTSearchTaskError("Match binary ended mid-record")
            yield unpack_fixed_imgt_match(payload)


def _verify_output_matches(
    *,
    match_path: Path,
    query_records: tuple[IMGTQueryRecord, ...],
    target_path: Path,
) -> dict[str, Any]:
    queries = {record.query_ordinal: record for record in query_records}
    matches = tuple(_iter_match_records(match_path))
    wanted: dict[int, list[FixedIMGTOutputMatch]] = {}
    pairs: set[tuple[int, int]] = set()
    split_counts = {"train": 0, "validation": 0}
    for match in matches:
        key = (match.target_authority_ordinal, match.query_ordinal)
        if match.query_ordinal not in queries or key in pairs:
            raise OASIMGTSearchTaskError("Match query is absent or pair is duplicated")
        pairs.add(key)
        wanted.setdefault(match.target_authority_ordinal, []).append(match)
        split_counts[match.target_split] += 1

    observed_targets: set[int] = set()
    with target_path.open("rb") as handle:
        while True:
            payload = handle.read(IMGT_TARGET_RECORD.size)
            if not payload:
                break
            if len(payload) != IMGT_TARGET_RECORD.size:
                raise OASIMGTSearchTaskError("Target binary ended mid-record")
            ordinal = int.from_bytes(payload[32:40], "big")
            if ordinal not in wanted:
                continue
            if ordinal in observed_targets:
                raise OASIMGTSearchTaskError("Matched target ordinal is duplicated")
            observed_targets.add(ordinal)
            target = unpack_imgt_target(payload)
            for claimed in wanted[ordinal]:
                query = queries[claimed.query_ordinal]
                metrics = position_identity(query.alignment, target.alignment)
                reconstructed = FixedIMGTOutputMatch(
                    identical_positions=metrics.identical_positions,
                    overlapping_positions=metrics.overlapping_positions,
                    query_ordinal=query.query_ordinal,
                    query_positions=metrics.query_positions,
                    target_authority_ordinal=target.authority_ordinal,
                    target_positions=metrics.target_positions,
                    target_split=target.split,
                    union_positions=metrics.union_positions,
                )
                reconstructed.validate()
                if reconstructed != claimed:
                    raise OASIMGTSearchTaskError(
                        "Independent full-position match reconstruction differs"
                    )
    if observed_targets != set(wanted):
        raise OASIMGTSearchTaskError("At least one matched target ordinal is absent")
    return {
        "match_records": len(matches),
        "matched_query_count": len({match.query_ordinal for match in matches}),
        "matched_target_count": len(wanted),
        "split_match_counts": split_counts,
    }


def _validate_kernel_stats(
    *,
    path: Path,
    query_count: int,
    target_records: int,
    match_records: int,
    batch_records: int,
) -> dict[str, Any]:
    # The source-pinned C++ kernel emits finite, duplicate-free JSON, but its
    # 17-digit double rendering is not necessarily byte-identical to Python's
    # shortest round-trip float representation.  Parse that runtime telemetry
    # strictly and canonicalize it when embedding it in the signed receipt;
    # all persistent input receipts still require canonical bytes.
    stats, _ = _strict_json(
        path, "CUDA kernel stats", require_canonical=False
    )
    elapsed = stats.get("elapsed_seconds")
    expected_batches = (target_records + batch_records - 1) // batch_records
    if (
        stats.get("format_version") != KERNEL_STATS_FORMAT
        or stats.get("query_records") != query_count
        or stats.get("target_file_records") != target_records
        or stats.get("records_scanned") != target_records
        or stats.get("seed_lookups") != target_records * 25
        or stats.get("matches") != match_records
        or stats.get("batch_records") != batch_records
        or stats.get("batches") != expected_batches
        or not isinstance(stats.get("cuda_device"), str)
        or not stats["cuda_device"]
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed <= 0
    ):
        raise OASIMGTSearchTaskError("CUDA kernel statistics differ")
    for name in ("bucket_member_visits", "quorum_candidates"):
        _nonnegative_integer(stats.get(name), f"CUDA {name}")
    if stats["quorum_candidates"] < match_records:
        raise OASIMGTSearchTaskError("CUDA quorum count is below verified matches")
    return stats


def _freeze_tree(root: Path, *, include_root: bool = True) -> None:
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise OASIMGTSearchTaskError("Output tree contains a symbolic link")
        path.chmod(metadata.st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    if include_root:
        metadata = root.lstat()
        root.chmod(metadata.st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _remove_output_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in (root, *root.rglob("*")):
        try:
            metadata = path.lstat()
            path.chmod(metadata.st_mode | stat.S_IWUSR)
        except OSError:
            pass
    shutil.rmtree(root, ignore_errors=True)


def _resource_receipt() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assigned = os.environ.get("SLURM_JOB_GPUS") or os.environ.get(
        "SLURM_STEP_GPUS", ""
    )
    assigned_count = len([item for item in assigned.split(",") if item])
    if not visible or visible == "NoDevFiles" or assigned_count != 1:
        raise OASIMGTSearchTaskError("Exactly one visible scheduler GPU is required")
    return {
        "cuda_visible_devices": visible,
        "gpu_compute_permitted": True,
        "scheduler_reserved_gpu_count": assigned_count,
    }


def run_task(
    *,
    task_id: int,
    target_output_root: Path,
    query_root: Path,
    search_output_root: Path,
    kernel_binary: Path,
    kernel_source: Path,
    batch_records: int,
    expected_controller_source_sha256: str,
    expected_kernel_binary_sha256: str,
    expected_kernel_source_sha256: str,
    expected_match_schema_source_sha256: str,
) -> dict[str, Any]:
    """Authenticate inputs, run CUDA, reconstruct every match, and seal a receipt."""
    if not 0 <= task_id < 32 or not 1 <= batch_records <= 262_144:
        raise OASIMGTSearchTaskError("Task id or batch size is outside the frozen range")
    expected_hashes = {
        "controller": _require_sha256(
            expected_controller_source_sha256, "Expected controller source"
        ),
        "kernel_binary": _require_sha256(
            expected_kernel_binary_sha256, "Expected kernel binary"
        ),
        "kernel_source": _require_sha256(
            expected_kernel_source_sha256, "Expected kernel source"
        ),
        "match_schema": _require_sha256(
            expected_match_schema_source_sha256, "Expected match schema"
        ),
    }
    observed_hashes = {
        "controller": _sha256(Path(__file__)),
        "kernel_binary": _sha256(kernel_binary),
        "kernel_source": _sha256(kernel_source),
        "match_schema": _sha256(
            Path(sys.modules[FixedIMGTOutputMatch.__module__].__file__ or "")
        ),
    }
    if observed_hashes != expected_hashes:
        raise OASIMGTSearchTaskError("Executable search source identity differs")
    target_path, target_inventory, global_target_receipt = _authenticate_target(
        target_output_root=target_output_root, task_id=task_id
    )
    query_path, query_records, query_receipt = _authenticate_queries(query_root)
    resource = _resource_receipt()
    search_output_root = Path(search_output_root).expanduser().absolute()
    search_output_root.mkdir(parents=True, exist_ok=True)
    work_root = search_output_root / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    final_root = search_output_root / "fixed-imgt" / f"mapper-{task_id:04d}"
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists() or final_root.is_symlink():
        raise OASIMGTSearchTaskError("Refusing to overwrite a fixed-IMGT task")
    temporary = Path(
        tempfile.mkdtemp(prefix=f"fixed-imgt-{task_id:04d}.partial-", dir=work_root)
    )
    published = False
    try:
        match_path = temporary / "matches.bin"
        stats_path = temporary / "kernel_stats.json"
        completed = subprocess.run(
            [
                str(_regular_file(kernel_binary, "CUDA kernel binary")),
                "--queries",
                str(query_path),
                "--targets",
                str(target_path),
                "--matches",
                str(match_path),
                "--stats",
                str(stats_path),
                "--batch-records",
                str(batch_records),
                "--device",
                "0",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise OASIMGTSearchTaskError(
                "CUDA kernel failed: "
                + completed.stderr.decode("utf-8", errors="replace")[:2_000]
            )
        if completed.stdout or completed.stderr:
            raise OASIMGTSearchTaskError(
                "Successful CUDA kernel wrote unaudited stdout or stderr"
            )
        match_path = _regular_file(match_path, "match binary")
        match_count = match_path.stat().st_size // FIXED_IMGT_MATCH_RECORD.size
        independent = _verify_output_matches(
            match_path=match_path,
            query_records=query_records,
            target_path=target_path,
        )
        stats = _validate_kernel_stats(
            path=stats_path,
            query_count=len(query_records),
            target_records=target_inventory["records"],
            match_records=match_count,
            batch_records=batch_records,
        )
        if independent["match_records"] != match_count:
            raise OASIMGTSearchTaskError("Independent match count differs")
        receipt = {
            "claim_boundary": {
                "fixed_imgt_task_domain_complete": True,
                "full_200_position_verification_complete": True,
                "gap_aware_search_complete": False,
                "global_fixed_imgt_search_complete": False,
                "large_scale_pretraining_authorized": False,
                "near_identity_decontamination_complete": False,
                "scientific_claim": False,
            },
            "format_version": FORMAT_VERSION,
            "independent_output_verification": independent,
            "input": {
                "global_target_receipt_sha256": global_target_receipt["receipt_sha256"],
                "query_receipt_sha256": query_receipt["receipt_sha256"],
                "query_binary_sha256": _sha256(query_path),
                "target_binary_sha256": target_inventory["file_sha256"],
                "target_records": target_inventory["records"],
            },
            "kernel_stats": stats,
            "output": {
                "match_record_bytes": FIXED_IMGT_MATCH_RECORD.size,
                "matches_bin_bytes": match_path.stat().st_size,
                "matches_bin_sha256": _sha256(match_path),
            },
            "resource_receipt": resource,
            "source_sha256": observed_hashes,
            "task_id": task_id,
            "test_access": TEST_ACCESS,
        }
        publish_export_receipt(temporary / "receipt.json", receipt)
        # BeeGFS can reject cross-directory rename of a directory whose own
        # write bit was already cleared.  Seal every child first, atomically
        # publish the still-private task root, then remove the root write bits.
        # The global finalizer cannot run until every array task completes.
        _freeze_tree(temporary, include_root=False)
        os.rename(temporary, final_root)
        published = True
        metadata = final_root.lstat()
        final_root.chmod(
            metadata.st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        )
        return json.loads((final_root / "receipt.json").read_text(encoding="ascii"))
    except BaseException:
        _remove_output_tree(final_root if published else temporary)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--target-output-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--search-output-root", type=Path, required=True)
    parser.add_argument("--kernel-binary", type=Path, required=True)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--batch-records", type=int, default=32_768)
    parser.add_argument("--controller-source-sha256", required=True)
    parser.add_argument("--kernel-binary-sha256", required=True)
    parser.add_argument("--kernel-source-sha256", required=True)
    parser.add_argument("--match-schema-source-sha256", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    receipt = run_task(
        task_id=arguments.task_id,
        target_output_root=arguments.target_output_root,
        query_root=arguments.query_root,
        search_output_root=arguments.search_output_root,
        kernel_binary=arguments.kernel_binary,
        kernel_source=arguments.kernel_source,
        batch_records=arguments.batch_records,
        expected_controller_source_sha256=arguments.controller_source_sha256,
        expected_kernel_binary_sha256=arguments.kernel_binary_sha256,
        expected_kernel_source_sha256=arguments.kernel_source_sha256,
        expected_match_schema_source_sha256=arguments.match_schema_source_sha256,
    )
    sys.stdout.buffer.write(_canonical_json_bytes(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
