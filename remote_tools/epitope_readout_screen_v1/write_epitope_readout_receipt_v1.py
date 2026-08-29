#!/usr/bin/env python3
"""Write an immutable receipt for one epitope readout replay result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    result = read_json(args.result)
    protocol_receipt = read_json(args.protocol_receipt)
    protocol_sha256 = sha256_file(args.protocol)
    if (
        result.get("status") != "PASS_MEASUREMENT_ONLY_REPLAY"
        or result.get("audit", {}).get("protocol_sha256") != protocol_sha256
        or result.get("audit", {}).get("test_loaded") is not False
        or result.get("audit", {}).get("temporal_holdout_loaded") is not False
        or result.get("audit", {}).get("model_parameters_updated") is not False
        or protocol_receipt.get("protocol_sha256") != protocol_sha256
        or protocol_receipt.get("phase") != result.get("phase")
        or protocol_receipt.get("frozen_before_measurement") is not True
        or os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090"
    ):
        raise ValueError("result, protocol, or execution authority differs")
    payload = {
        "format_version": "sabdab2_epitope_readout_seed_receipt_v1",
        "seed": int(result["seed"]),
        "phase": result["phase"],
        "panel_role": result["panel_role"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "partition": "gpu_4090",
        "result_sha256": sha256_file(args.result),
        "protocol_sha256": protocol_sha256,
        "protocol_receipt_sha256": sha256_file(args.protocol_receipt),
        "measurement_only": True,
        "model_parameters_updated": False,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(payload))
    print(
        json.dumps(
            {
                "status": "PASS_RECEIPT_WRITTEN",
                "seed": payload["seed"],
                "receipt_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
