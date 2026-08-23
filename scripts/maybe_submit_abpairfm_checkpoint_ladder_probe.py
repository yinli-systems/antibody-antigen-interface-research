#!/usr/bin/env python3
"""Submit the first immutable Stage-A0 checkpoint at each frozen token target."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPOSITORY_ROOT / "scripts"
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (SCRIPT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_abpairfm_checkpoint_transfer_probe import _source_sha256  # noqa: E402
from run_mint_frozen_development_retrieval import sha256_file  # noqa: E402


TARGETS = (250_000_000, 500_000_000, 1_000_000_000)
MAXIMUM_OVERSHOOT = 20_000_000


def selected_target(consumed: int) -> int | None:
    selected = [
        target
        for target in TARGETS
        if target <= consumed <= target + MAXIMUM_OVERSHOOT
    ]
    if len(selected) > 1:
        raise RuntimeError("Stage-A0 result ambiguously matches token targets")
    return selected[0] if selected else None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if sha256_file(arguments.protocol) != arguments.expected_protocol_sha256:
        raise RuntimeError("Checkpoint ladder protocol SHA-256 differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_BEFORE_GPU_EVALUATION"
        or tuple(protocol["checkpoint_ladder"]["targets"]) != TARGETS
        or protocol["checkpoint_ladder"]["maximum_target_overshoot_residue_tokens"]
        != MAXIMUM_OVERSHOOT
        or protocol.get("test_access") is not False
    ):
        raise RuntimeError("Checkpoint ladder protocol differs")
    result = json.loads(arguments.result.read_text(encoding="utf-8"))
    if (
        result.get("status") != "PASS_CHECKPOINT_VERIFIED"
        or result.get("test_access") is not False
    ):
        raise RuntimeError("Stage-A0 result does not authorize a ladder probe")
    consumed = result["training_state"]["consumed_residue_tokens"]
    target = selected_target(consumed)
    if target is None:
        print(json.dumps({"status": "NO_FROZEN_TOKEN_TARGET_REACHED"}))
        return
    arguments.submission_root.mkdir(parents=True, exist_ok=True)
    receipt_path = arguments.submission_root / f"target-{target}.json"
    output_root = (
        Path("/ssd/scxi253/model_deployment/results/pretraining")
        / "abpairfm_checkpoint_transfer_probe_v1"
        / f"stagea0-{target // 1_000_000}m-full"
    )
    if receipt_path.exists() or output_root.exists():
        print(json.dumps({"status": "TARGET_ALREADY_SUBMITTED", "target": target}))
        return
    transfer_source_sha256 = _source_sha256(arguments.project_root)
    checkpoint = result["checkpoint"]
    environment = os.environ.copy()
    environment.update(
        {
            "ABPAIRFM_PROBE_MODEL_VARIANT": "stage_a0_3b",
            "ABPAIRFM_PROBE_OUTPUT_ROOT": str(output_root),
            "ABPAIRFM_PROBE_PROTOCOL_SHA256": arguments.expected_protocol_sha256,
            "ABPAIRFM_PROBE_SOURCE_SHA256": transfer_source_sha256,
            "ABPAIRFM_PROBE_SMOKE_ANCHORS": "0",
            "ABPAIRFM_PROBE_CHECKPOINT_TARGET": str(target),
            "ABPAIRFM_PROBE_DCP_ROOT": checkpoint["root"],
            "ABPAIRFM_PROBE_DCP_GENERATION": checkpoint["generation"],
            "ABPAIRFM_PROBE_DCP_CONTENT_SHA256": checkpoint["content_sha256"],
        }
    )
    submission = subprocess.run(
        [
            "sbatch",
            "--parsable",
            str(arguments.project_root / "scripts/submit_abpairfm_checkpoint_transfer_probe.sbatch"),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    job_id = submission.stdout.strip()
    if not job_id.isdigit():
        raise RuntimeError("Slurm did not return one numeric transfer-probe job ID")
    receipt = {
        "checkpoint": {
            "content_sha256": checkpoint["content_sha256"],
            "generation": checkpoint["generation"],
            "root": checkpoint["root"],
        },
        "consumed_residue_tokens": consumed,
        "format_version": "abpairfm_checkpoint_ladder_submission_v1",
        "job_id": int(job_id),
        "output_root": str(output_root),
        "protocol_sha256": arguments.expected_protocol_sha256,
        "status": "SUBMITTED_FROZEN_DEVELOPMENT_TRANSFER_PROBE",
        "target_residue_tokens": target,
        "test_access": False,
        "transfer_source_sha256": transfer_source_sha256,
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
