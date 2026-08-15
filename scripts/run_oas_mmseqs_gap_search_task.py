#!/usr/bin/env python3
"""Run one authenticated, cap-audited MMseqs2 gap-aware target shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid

from antibody_pretraining.mmseqs_gap_aware import (
    ALIGNMENT_COLUMNS,
    MMseqsGapAwareError,
    TEST_ACCESS,
    audit_alignment_rows,
    audit_prefilter_database,
    authenticate_query_fasta,
    canonical_json_bytes,
    export_authenticated_target_task,
    publish_export_receipt,
    sha256_file,
)


FORMAT_VERSION = "oas_mmseqs_gap_search_task_v1"


def _absolute_executable(path: Path, label: str) -> Path:
    candidate = Path(path).expanduser().absolute()
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise MMseqsGapAwareError("Cannot stat " + label) from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or not os.access(candidate, os.X_OK)
    ):
        raise MMseqsGapAwareError(label + " is not a regular executable")
    return candidate


def _one_reserved_gpu() -> Mapping[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    values = [value for value in visible.split(",") if value]
    if len(values) != 1 or visible == "NoDevFiles":
        raise MMseqsGapAwareError("Task requires exactly one scheduler-visible GPU")
    assigned = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS", "")
    assigned_values = [value for value in assigned.split(",") if value]
    if os.environ.get("SLURM_JOB_ID") and len(assigned_values) != 1:
        raise MMseqsGapAwareError("Scheduler GPU allocation evidence differs")
    return {
        "cuda_visible_devices": visible,
        "gpu_compute_permitted": True,
        "scheduler_reserved_gpu_count": 1,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def _run(
    command: Sequence[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    environment: Mapping[str, str],
) -> Mapping[str, Any]:
    started = time.monotonic()
    with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
        "ab", buffering=0
    ) as stderr:
        result = subprocess.run(
            list(command),
            check=False,
            env=dict(environment),
            stdout=stdout,
            stderr=stderr,
        )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise MMseqsGapAwareError(
            "MMseqs command failed with exit code " + str(result.returncode)
        )
    return {
        "argv": list(command),
        "elapsed_seconds": elapsed,
        "returncode": result.returncode,
    }


def _write_signed_json(path: Path, value: Mapping[str, Any]) -> None:
    publish_export_receipt(path, value)


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise MMseqsGapAwareError("Task output contains a symbolic link")
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def run_task(arguments: argparse.Namespace) -> Mapping[str, Any]:
    if arguments.task_count != 32 or not 0 <= arguments.task_id < arguments.task_count:
        raise MMseqsGapAwareError("Task domain differs from the frozen 32-shard plan")
    if not 10_000 <= arguments.maximum_sequences <= 2_000_000:
        raise MMseqsGapAwareError("--maximum-sequences is outside the audited range")
    if not 1 <= arguments.threads <= 64:
        raise MMseqsGapAwareError("Thread count is outside the audited range")
    mmseqs = _absolute_executable(arguments.mmseqs_bin, "MMseqs binary")
    protocol = Path(arguments.protocol).expanduser().absolute()
    if sha256_file(protocol) != arguments.protocol_sha256:
        raise MMseqsGapAwareError("Near-identity protocol hash differs")
    resources = _one_reserved_gpu()
    query = authenticate_query_fasta(arguments.query_root)

    output_root = Path(arguments.output_root).expanduser().absolute()
    if output_root.is_symlink() or not output_root.is_dir():
        raise MMseqsGapAwareError("Search output root is absent or symbolic")
    final_root = output_root / "gap-aware" / f"mapper-{arguments.task_id:04d}"
    if final_root.exists() or final_root.is_symlink():
        raise MMseqsGapAwareError("Refusing to overwrite a gap-aware task")
    work_parent = output_root / "work"
    work_parent.mkdir(parents=True, exist_ok=True)
    partial = work_parent / (
        f"gap-aware-{arguments.task_id:04d}.partial-{uuid.uuid4().hex}"
    )
    partial.mkdir()
    durable = partial / "durable"
    scratch = partial / "scratch"
    durable.mkdir()
    scratch.mkdir()
    stdout_path = durable / "mmseqs.stdout"
    stderr_path = durable / "mmseqs.stderr"

    environment = dict(os.environ)
    environment["TMPDIR"] = str(scratch / "tmp")
    environment["XDG_CACHE_HOME"] = str(scratch / "cache")
    Path(environment["TMPDIR"]).mkdir()
    Path(environment["XDG_CACHE_HOME"]).mkdir()

    target_fasta = scratch / "targets.fasta"
    target_manifest = export_authenticated_target_task(
        target_output_root=arguments.target_output_root,
        task_id=arguments.task_id,
        output_fasta=target_fasta,
    )
    _write_signed_json(durable / "target_fasta_receipt.json", target_manifest)

    query_db = scratch / "query_db"
    target_db = scratch / "target_db"
    target_gpu = scratch / "target_gpu"
    prefilter_db = scratch / "prefilter_db"
    alignment_db = scratch / "alignment_db"
    alignments = durable / "alignments.tsv"
    commands = []
    common = ["--threads", str(arguments.threads), "-v", "3"]
    commands.append(
        _run(
            [str(mmseqs), "createdb", str(query["fasta_path"]), str(query_db), "--shuffle", "0"],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            environment=environment,
        )
    )
    commands.append(
        _run(
            [str(mmseqs), "createdb", str(target_fasta), str(target_db), "--shuffle", "0"],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            environment=environment,
        )
    )
    commands.append(
        _run(
            [
                str(mmseqs),
                "makepaddedseqdb",
                str(target_db),
                str(target_gpu),
                "--mask",
                "0",
                *common,
            ],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            environment=environment,
        )
    )
    commands.append(
        _run(
            [
                str(mmseqs),
                "ungappedprefilter",
                str(query_db),
                str(target_gpu),
                str(prefilter_db),
                "--gpu",
                "1",
                "--prefilter-mode",
                "1",
                "--max-seqs",
                str(arguments.maximum_sequences),
                "--comp-bias-corr",
                "0",
                "-e",
                "10000",
                *common,
            ],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            environment=environment,
        )
    )
    prefilter = audit_prefilter_database(
        prefix=prefilter_db,
        expected_query_count=len(query["query_lengths"]),
        maximum_sequences=arguments.maximum_sequences,
    )
    _write_signed_json(durable / "prefilter_audit.json", prefilter)
    commands.append(
        _run(
            [
                str(mmseqs),
                "align",
                str(query_db),
                str(target_db),
                str(prefilter_db),
                str(alignment_db),
                "--alignment-mode",
                "3",
                "--min-seq-id",
                "0.9",
                "-c",
                "0.9",
                "--cov-mode",
                "0",
                "--seq-id-mode",
                "0",
                "--max-accept",
                "2147483647",
                "--max-rejected",
                "2147483647",
                *common,
            ],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            environment=environment,
        )
    )
    commands.append(
        _run(
            [
                str(mmseqs),
                "convertalis",
                str(query_db),
                str(target_db),
                str(alignment_db),
                str(alignments),
                "--format-output",
                ",".join(ALIGNMENT_COLUMNS),
                *common,
            ],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            environment=environment,
        )
    )
    hits = durable / "hits.tsv"
    with alignments.open("r", encoding="utf-8") as source, hits.open(
        "x", encoding="ascii", newline=""
    ) as destination:
        alignment_audit = audit_alignment_rows(
            source,
            query_lengths=query["query_lengths"],
            expected_task_id=arguments.task_id,
            hits_output=destination,
        )
        destination.flush()
        os.fsync(destination.fileno())
    _write_signed_json(durable / "alignment_audit.json", alignment_audit)

    version = subprocess.run(
        [str(mmseqs), "version"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    ).stdout.strip()
    receipt = {
        "claim_boundary": {
            "gap_aware_task_search_complete": True,
            "global_gap_aware_search_complete": False,
            "hit_list_truncation_detected": False,
            "large_scale_pretraining_authorized": False,
            "near_identity_decontamination_complete": False,
            "scientific_claim": False,
        },
        "commands": commands,
        "format_version": FORMAT_VERSION,
        "input": {
            "global_target_receipt_file_sha256": target_manifest[
                "global_target_receipt_file_sha256"
            ],
            "global_target_receipt_sha256": target_manifest[
                "global_target_receipt_sha256"
            ],
            "protocol_file_sha256": arguments.protocol_sha256,
            "query_fasta_sha256": query["fasta_sha256"],
            "query_manifest_file_sha256": query["manifest_file_sha256"],
            "query_receipt_file_sha256": query["query_receipt_file_sha256"],
            "query_receipt_sha256": query["query_receipt_sha256"],
            "target_fasta_sha256": target_manifest["fasta_sha256"],
            "target_records": target_manifest["counts"]["records"],
        },
        "mmseqs": {
            "binary_path": str(mmseqs),
            "binary_sha256": sha256_file(mmseqs),
            "version": version,
        },
        "output": {
            "alignment_audit": alignment_audit,
            "alignments_bytes": alignments.stat().st_size,
            "alignments_sha256": sha256_file(alignments),
            "hits_bytes": hits.stat().st_size,
            "hits_sha256": sha256_file(hits),
            "prefilter_audit": prefilter,
        },
        "resource_receipt": resources,
        "source_sha256": {
            "controller": sha256_file(Path(__file__)),
            "gap_aware_core": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/antibody_pretraining/mmseqs_gap_aware.py"
            ),
        },
        "task_count": arguments.task_count,
        "task_id": arguments.task_id,
        "test_access": TEST_ACCESS,
    }
    _write_signed_json(durable / "receipt.json", receipt)
    shutil.rmtree(scratch)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(durable, final_root)
    partial.rmdir()
    _freeze_tree(final_root)
    value = json.loads((final_root / "receipt.json").read_text(encoding="ascii"))
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-count", type=int, default=32)
    parser.add_argument("--target-output-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mmseqs-bin", type=Path, required=True)
    parser.add_argument("--maximum-sequences", type=int, default=500_000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = run_task(arguments)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
