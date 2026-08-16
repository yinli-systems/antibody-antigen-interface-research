#!/usr/bin/env python3
"""Independently authenticate all sharded MMseqs2 gap-aware task evidence."""

from __future__ import annotations

import argparse
import io
import json
import os
import stat
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from antibody_pretraining.mmseqs_gap_aware import (
    ALIGNMENT_AUDIT_FORMAT,
    GLOBAL_TARGET_FORMAT,
    PREFILTER_AUDIT_FORMAT,
    TEST_ACCESS,
    MMseqsGapAwareError,
    audit_alignment_rows,
    authenticate_query_fasta,
    canonical_json_bytes,
    parse_target_identifier,
    publish_export_receipt,
    require_sha256,
    sha256_file,
    signed_json,
)

FORMAT_VERSION = "oas_mmseqs_gap_search_global_v1"
TASK_FORMAT_VERSION = "oas_mmseqs_gap_search_task_v1"


def _assert_frozen_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise MMseqsGapAwareError("Gap-aware task root is absent or symbolic")
    for path in (root, *root.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise MMseqsGapAwareError("Gap-aware task tree contains a symbolic link")
        if metadata.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise MMseqsGapAwareError("Gap-aware task tree remains writable")


def _same_json(left: object, right: object, label: str) -> None:
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        raise MMseqsGapAwareError(label + " differs")


def _same_signed_payload(
    signed_payload: Mapping[str, Any], unsigned_snapshot: object, label: str
) -> None:
    """Compare an authenticated receipt with its intentionally unsigned snapshot."""
    require_sha256(signed_payload.get("receipt_sha256"), label + " receipt")
    if not isinstance(unsigned_snapshot, Mapping) or "receipt_sha256" in unsigned_snapshot:
        raise MMseqsGapAwareError(label + " unsigned snapshot differs")
    payload = {
        key: value
        for key, value in signed_payload.items()
        if key != "receipt_sha256"
    }
    _same_json(payload, unsigned_snapshot, label)


def _read_hits(path: Path, *, task_id: int) -> tuple[set[tuple[str, str]], Counter[str]]:
    pairs = set()  # type: set[tuple[str, str]]
    splits = Counter()  # type: Counter[str]
    with path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise MMseqsGapAwareError(
                    f"Gap-aware hit row {line_number} has the wrong width"
                )
            query, target = fields[:2]
            metadata = parse_target_identifier(target)
            pair = (query, target)
            if metadata["task_id"] != task_id or pair in pairs:
                raise MMseqsGapAwareError("Gap-aware hit pair is duplicated or cross-shard")
            pairs.add(pair)
            splits[str(metadata["split"])] += 1
    return pairs, splits


def _audit_task(
    task_root: Path,
    *,
    task_id: int,
    query: Mapping[str, Any],
    expected_controller_sha256: str,
    expected_core_sha256: str,
    expected_mmseqs_sha256: str,
    expected_protocol_sha256: str,
) -> Mapping[str, Any]:
    _assert_frozen_tree(task_root)
    receipt, receipt_file_sha256 = signed_json(
        task_root / "receipt.json", f"gap-aware task receipt {task_id}"
    )
    boundary = receipt.get("claim_boundary")
    sources = receipt.get("source_sha256")
    inputs = receipt.get("input")
    outputs = receipt.get("output")
    mmseqs = receipt.get("mmseqs")
    resource = receipt.get("resource_receipt")
    if (
        receipt.get("format_version") != TASK_FORMAT_VERSION
        or receipt.get("task_id") != task_id
        or receipt.get("task_count") != 32
        or receipt.get("test_access") != TEST_ACCESS
        or not isinstance(boundary, Mapping)
        or boundary.get("gap_aware_task_search_complete") is not True
        or boundary.get("global_gap_aware_search_complete") is not False
        or boundary.get("hit_list_truncation_detected") is not False
        or boundary.get("large_scale_pretraining_authorized") is not False
        or boundary.get("near_identity_decontamination_complete") is not False
        or boundary.get("scientific_claim") is not False
        or not all(
            isinstance(value, Mapping)
            for value in (sources, inputs, outputs, mmseqs, resource)
        )
    ):
        raise MMseqsGapAwareError("Gap-aware task claim boundary differs")
    if (
        sources.get("controller") != expected_controller_sha256
        or sources.get("gap_aware_core") != expected_core_sha256
        or mmseqs.get("binary_sha256") != expected_mmseqs_sha256
        or inputs.get("protocol_file_sha256") != expected_protocol_sha256
        or inputs.get("query_fasta_sha256") != query["fasta_sha256"]
        or inputs.get("query_manifest_file_sha256")
        != query["manifest_file_sha256"]
        or inputs.get("query_receipt_file_sha256")
        != query["query_receipt_file_sha256"]
        or inputs.get("query_receipt_sha256") != query["query_receipt_sha256"]
        or resource.get("gpu_compute_permitted") is not True
        or resource.get("scheduler_reserved_gpu_count") != 1
        or not isinstance(resource.get("cuda_visible_devices"), str)
        or not resource.get("cuda_visible_devices")
    ):
        raise MMseqsGapAwareError("Gap-aware task provenance differs")

    prefilter_receipt, prefilter_file_sha256 = signed_json(
        task_root / "prefilter_audit.json", f"prefilter audit {task_id}"
    )
    alignment_receipt, alignment_file_sha256 = signed_json(
        task_root / "alignment_audit.json", f"alignment audit {task_id}"
    )
    if (
        prefilter_receipt.get("format_version") != PREFILTER_AUDIT_FORMAT
        or prefilter_receipt.get("query_count") != 3_106
        or prefilter_receipt.get("saturation_detected") is not False
        or prefilter_receipt.get("hit_count_maximum")
        >= prefilter_receipt.get("maximum_sequences")
        or alignment_receipt.get("format_version") != ALIGNMENT_AUDIT_FORMAT
    ):
        raise MMseqsGapAwareError("Gap-aware task audit receipt differs")
    _same_signed_payload(
        prefilter_receipt, outputs.get("prefilter_audit"), "Prefilter audit"
    )
    _same_signed_payload(
        alignment_receipt, outputs.get("alignment_audit"), "Alignment audit"
    )

    alignments = task_root / "alignments.tsv"
    hits = task_root / "hits.tsv"
    if (
        outputs.get("alignments_bytes") != alignments.stat().st_size
        or outputs.get("alignments_sha256") != sha256_file(alignments)
        or outputs.get("hits_bytes") != hits.stat().st_size
        or outputs.get("hits_sha256") != sha256_file(hits)
    ):
        raise MMseqsGapAwareError("Gap-aware task output inventory differs")
    replayed_hits = io.StringIO()
    with alignments.open("r", encoding="utf-8") as handle:
        replayed_alignment = audit_alignment_rows(
            handle,
            query_lengths=query["query_lengths"],
            expected_task_id=task_id,
            hits_output=replayed_hits,
        )
    _same_signed_payload(
        alignment_receipt, replayed_alignment, "Independent alignment replay"
    )
    if replayed_hits.getvalue().encode("ascii") != hits.read_bytes():
        raise MMseqsGapAwareError("Independent five-column hit replay differs")
    pairs, splits = _read_hits(hits, task_id=task_id)
    if (
        len(pairs) != alignment_receipt.get("alignment_rows")
        or splits["train"] != alignment_receipt["split_hit_counts"]["train"]
        or splits["validation"]
        != alignment_receipt["split_hit_counts"]["validation"]
    ):
        raise MMseqsGapAwareError("Gap-aware task hit accounting differs")
    return {
        "alignment_audit_file_sha256": alignment_file_sha256,
        "global_target_receipt_file_sha256": inputs[
            "global_target_receipt_file_sha256"
        ],
        "global_target_receipt_sha256": inputs["global_target_receipt_sha256"],
        "hit_pairs": pairs,
        "hit_summary": alignment_receipt,
        "maximum_sequences": prefilter_receipt["maximum_sequences"],
        "mmseqs_version": mmseqs.get("version"),
        "prefilter_audit_file_sha256": prefilter_file_sha256,
        "prefilter_hit_count_maximum": prefilter_receipt["hit_count_maximum"],
        "prefilter_hit_count_total": prefilter_receipt["hit_count_total"],
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "target_records": inputs["target_records"],
        "task_id": task_id,
    }


def _resource_receipt() -> Mapping[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"", "NoDevFiles"}:
        raise MMseqsGapAwareError("Gap-aware finalizer must not expose a GPU")
    assigned = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS", "")
    assigned_count = len([item for item in assigned.split(",") if item])
    declared = os.environ.get("OAS_SCHEDULER_RESERVED_GPU_COUNT")
    if declared == "1" and assigned_count == 1:
        mode = "scheduler_mandated_idle_only"
    elif declared is None and assigned_count == 0:
        mode = "none"
    else:
        raise MMseqsGapAwareError("Finalizer scheduler resource evidence differs")
    return {
        "cuda_visible_devices": False,
        "gpu_compute_permitted": False,
        "scheduler_reserved_gpu_count": assigned_count,
        "scheduler_reserved_gpu_mode": mode,
    }


def finalize(arguments: argparse.Namespace) -> Mapping[str, Any]:
    if arguments.task_count != 32 or not 1 <= arguments.worker_count <= 8:
        raise MMseqsGapAwareError("Gap-aware finalizer task or worker domain differs")
    expected_controller = require_sha256(
        arguments.expected_controller_sha256, "Expected controller"
    )
    expected_core = require_sha256(arguments.expected_core_sha256, "Expected core")
    expected_mmseqs = require_sha256(
        arguments.expected_mmseqs_sha256, "Expected MMseqs binary"
    )
    expected_protocol = require_sha256(
        arguments.expected_protocol_sha256, "Expected protocol"
    )
    query = authenticate_query_fasta(arguments.query_root)
    output_root = Path(arguments.output_root).expanduser().absolute()
    output_path = output_root / "gap_aware_global_receipt.json"
    if output_path.exists() or output_path.is_symlink():
        raise MMseqsGapAwareError("Refusing to overwrite gap-aware global receipt")
    if any((output_root / "work").glob("gap-aware-*.partial-*")):
        raise MMseqsGapAwareError("Partial gap-aware tasks remain")

    def audit(task_id: int) -> Mapping[str, Any]:
        return _audit_task(
            output_root / "gap-aware" / f"mapper-{task_id:04d}",
            task_id=task_id,
            query=query,
            expected_controller_sha256=expected_controller,
            expected_core_sha256=expected_core,
            expected_mmseqs_sha256=expected_mmseqs,
            expected_protocol_sha256=expected_protocol,
        )

    with ThreadPoolExecutor(max_workers=arguments.worker_count) as pool:
        tasks = list(pool.map(audit, range(arguments.task_count)))
    tasks.sort(key=lambda value: value["task_id"])
    if [value["task_id"] for value in tasks] != list(range(arguments.task_count)):
        raise MMseqsGapAwareError("Gap-aware task ordering differs")
    target_receipt, target_file_sha256 = signed_json(
        Path(arguments.target_output_root) / "global_receipt.json",
        "global target receipt",
    )
    target_counts = target_receipt.get("counts")
    target_boundary = target_receipt.get("claim_boundary")
    if (
        target_receipt.get("format_version") != GLOBAL_TARGET_FORMAT
        or not isinstance(target_counts, Mapping)
        or not isinstance(target_boundary, Mapping)
        or target_boundary.get("global_task_set_complete") is not True
        or sum(value["target_records"] for value in tasks)
        != target_counts.get("accepted_records")
        or any(
            value["global_target_receipt_sha256"]
            != target_receipt["receipt_sha256"]
            or value["global_target_receipt_file_sha256"] != target_file_sha256
            for value in tasks
        )
    ):
        raise MMseqsGapAwareError("Gap-aware target-domain accounting differs")
    all_pairs = set()  # type: set[tuple[str, str]]
    split_counts = Counter()  # type: Counter[str]
    matched_queries = set()
    matched_targets = set()
    for task in tasks:
        pairs = task["hit_pairs"]
        if all_pairs.intersection(pairs):
            raise MMseqsGapAwareError("Gap-aware hit pair repeats across target shards")
        all_pairs.update(pairs)
        for query_key, target_id in pairs:
            matched_queries.add(query_key)
            matched_targets.add(target_id)
            split_counts[str(parse_target_identifier(target_id)["split"])] += 1
    task_receipts = []
    for value in tasks:
        task_receipts.append(
            {
                key: item
                for key, item in value.items()
                if key != "hit_pairs"
            }
        )
    receipt = {
        "claim_boundary": {
            "gap_aware_search_complete": True,
            "hit_list_truncation_detected": False,
            "large_scale_pretraining_authorized": False,
            "near_identity_decontamination_complete": False,
            "scientific_claim": False,
            "treat_mmseqs_as_exhaustive_ground_truth": False,
        },
        "counts": {
            "hit_pairs": len(all_pairs),
            "matched_queries": len(matched_queries),
            "matched_targets": len(matched_targets),
            "query_accounting_per_task": 3_106,
            "target_records_scoped": sum(value["target_records"] for value in tasks),
            "train_hits": split_counts["train"],
            "validation_hits": split_counts["validation"],
        },
        "format_version": FORMAT_VERSION,
        "input": {
            "global_target_receipt_file_sha256": target_file_sha256,
            "global_target_receipt_sha256": target_receipt["receipt_sha256"],
            "protocol_file_sha256": expected_protocol,
            "query_fasta_sha256": query["fasta_sha256"],
            "query_manifest_file_sha256": query["manifest_file_sha256"],
            "query_receipt_file_sha256": query["query_receipt_file_sha256"],
            "query_receipt_sha256": query["query_receipt_sha256"],
        },
        "resource_receipt": _resource_receipt(),
        "source_sha256": {
            "controller": expected_controller,
            "finalizer": sha256_file(Path(__file__)),
            "gap_aware_core": expected_core,
            "mmseqs_binary": expected_mmseqs,
        },
        "task_count": arguments.task_count,
        "task_receipts": task_receipts,
        "test_access": TEST_ACCESS,
    }
    publish_export_receipt(output_path, receipt)
    return json.loads(output_path.read_text(encoding="ascii"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-output-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=32)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--expected-controller-sha256", required=True)
    parser.add_argument("--expected-core-sha256", required=True)
    parser.add_argument("--expected-mmseqs-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    return parser


def main() -> None:
    result = finalize(_parser().parse_args())
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
