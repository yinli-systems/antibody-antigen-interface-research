#!/usr/bin/env python3
"""Freeze exact-condition smoke receipts into a hash-bound manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-protocol", required=True, type=Path)
    parser.add_argument("--campaign-protocol-sha256", required=True)
    parser.add_argument("--smoke-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(sha(args.campaign_protocol) == args.campaign_protocol_sha256,
            "Campaign protocol hash differs")
    campaign = json.loads(args.campaign_protocol.read_text())
    require(sha(Path(__file__).resolve()) == campaign["implementation"]["freeze_smokes.py"],
            "Smoke freezer differs from preregistered implementation")
    receipts = {}
    for condition in campaign["conditions"]:
        name = condition["name"]
        result = (args.smoke_root / name / "RESULT.json").resolve()
        validation = result.parent / "VALIDATION_EVIDENCE.json"
        checkpoint_candidates = sorted(result.parent.glob("checkpoint-epoch-*.pt"))
        require(result.is_file() and validation.is_file() and len(checkpoint_candidates) == 1,
                f"Incomplete smoke artifacts: {name}")
        payload = json.loads(result.read_text())
        require(payload["status"] == "PASS_CONDITION_SMOKE", f"Smoke failed: {name}")
        require(payload["condition"] == name, f"Smoke condition mismatch: {name}")
        require(payload["campaign_protocol_sha256"] == args.campaign_protocol_sha256,
                f"Smoke campaign mismatch: {name}")
        require(payload["seed"] == 17, f"Unexpected smoke seed: {name}")
        receipts[name] = {
            "result_path": str(result),
            "result_sha256": sha(result),
            "validation_path": str(validation.resolve()),
            "validation_sha256": sha(validation),
            "checkpoint_path": str(checkpoint_candidates[0].resolve()),
            "checkpoint_sha256": sha(checkpoint_candidates[0]),
            "slurm_job_id": payload.get("slurm_job_id"),
            "slurm_array_task_id": payload.get("slurm_array_task_id"),
        }
    output = {
        "status": "PASS_ALL_CONDITION_SMOKES_FROZEN",
        "format_version": campaign["format_version"],
        "campaign_protocol_sha256": args.campaign_protocol_sha256,
        "condition_count": len(receipts),
        "receipts": receipts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("COMPLETE", args.output, sha(args.output))


if __name__ == "__main__":
    main()
