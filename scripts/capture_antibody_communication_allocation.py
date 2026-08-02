#!/usr/bin/env python3
"""Capture immutable hardware identity for the communication paired gate."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import socket
import subprocess


GPU_FIELDS = (
    "index",
    "uuid",
    "name",
    "pci.bus_id",
    "driver_version",
    "pstate",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
    "memory.total",
)
RUN_ORDER = (
    "safe_eager_r1",
    "fp32_compiled_r1",
    "bf16_selected_r1",
    "bf16_selected_r2",
    "fp32_compiled_r2",
    "safe_eager_r2",
)


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="ascii",
    )
    return completed.stdout


def _gpu_inventory() -> list[dict[str, str]]:
    query = "--query-gpu=" + ",".join(GPU_FIELDS)
    output = _run(["nvidia-smi", query, "--format=csv,noheader,nounits"])
    rows = list(csv.reader(io.StringIO(output)))
    inventory = [
        {field: value.strip() for field, value in zip(GPU_FIELDS, row)}
        for row in rows
    ]
    if len(inventory) != 8:
        raise RuntimeError("Communication gate requires exactly eight visible GPUs")
    if len({item["uuid"] for item in inventory}) != len(inventory):
        raise RuntimeError("GPU UUIDs are not unique")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-order", required=True)
    args = parser.parse_args()
    run_order = tuple(args.run_order.split(","))
    if run_order != RUN_ORDER:
        raise RuntimeError("Run order differs from the frozen counterbalanced order")
    if len(args.protocol_sha256) != 64:
        raise RuntimeError("Protocol SHA-256 is malformed")
    job_id = os.environ.get("SLURM_JOB_ID")
    node_list = os.environ.get("SLURM_JOB_NODELIST")
    if not job_id or not node_list:
        raise RuntimeError("Hardware capture must run inside a Slurm allocation")
    payload = {
        "claim_boundary": "Hardware identity only; no efficiency or scaling claim",
        "format_version": "antibody_communication_paired_allocation_v1",
        "gpu_count": 8,
        "gpu_inventory": _gpu_inventory(),
        "hostname": socket.gethostname(),
        "protocol_sha256": args.protocol_sha256,
        "run_order": list(run_order),
        "slurm": {
            "job_id": job_id,
            "job_nodelist": node_list,
            "node_count": int(os.environ.get("SLURM_JOB_NUM_NODES", "0")),
        },
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "topology": _run(["nvidia-smi", "topo", "-m"]),
    }
    if payload["slurm"]["node_count"] != 1:
        raise RuntimeError("Communication gate requires exactly one node")
    if payload["hostname"] not in node_list:
        raise RuntimeError("Allocated node and hostname disagree")
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)


if __name__ == "__main__":
    main()
