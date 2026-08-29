#!/usr/bin/env python3
"""Measurement-only recovery for zero-positive SAbDab2 test complexes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ORIGINAL_EVALUATOR = Path(
    "/ssd/scxi253/model_deployment/tools/"
    "evaluate_task_routed_exposed_test_component_v1.py"
)
ORIGINAL_EVALUATOR_SHA256 = (
    "5380816388ee7ab8354590cabef549260c18a3b01b6dcf6c41519b03589c1467"
)
ROBUST_EVALUATION_SOURCE = Path(
    "/ssd/scxi253/model_deployment/data/framecontact_20260730/"
    "sabdab_eval_source_v1/scripts/train_framecontact.py"
)
ROBUST_EVALUATION_SOURCE_SHA256 = (
    "0f58e90596c540079196980cd2e604a470d759642d74e0d25732b8f4caa92722"
)
ROBUST_SOURCE_ROOT = Path(
    "/ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab_eval_source_v1"
)
FAILED_STDERR_SHA256 = {
    "baseline_current": "8a9c8c52e671c334eed887ecc04ae92604560cdb5079ef4ac959cf489647a78e",
    "edge_topk2_supervised": "17d6f62871ab0fb09f2accb09e42b24dd51678883aad9fefb48d725fc46b2ac3",
}
FAILED_RECOVERY_STDERR_SHA256 = {
    "baseline_current": "c9a276b3c86a497a62bb54ca0e1fd1fb682e3c6befd2663d083a92aa81827019",
    "edge_topk2_supervised": "c9a276b3c86a497a62bb54ca0e1fd1fb682e3c6befd2663d083a92aa81827019",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(FAILED_STDERR_SHA256), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    known, _ = parser.parse_known_args()

    if sha256_file(ORIGINAL_EVALUATOR) != ORIGINAL_EVALUATOR_SHA256:
        raise RuntimeError("original frozen evaluator SHA-256 differs")
    if sha256_file(ROBUST_EVALUATION_SOURCE) != ROBUST_EVALUATION_SOURCE_SHA256:
        raise RuntimeError("robust frozen evaluation source SHA-256 differs")
    recovery_authority_path = (
        known.campaign_root / "authority" / "MEASUREMENT_RECOVERY_V2.json"
    )
    recovery_authority = json.loads(recovery_authority_path.read_text())
    if (
        recovery_authority.get("status")
        != "FROZEN_MEASUREMENT_ONLY_SIGNATURE_ADAPTER_RECOVERY"
        or recovery_authority.get("failed_array_job_id") != 1472578
        or recovery_authority.get("failed_recovery_array_job_id") != 1472589
        or recovery_authority.get("original_evaluator_sha256")
        != ORIGINAL_EVALUATOR_SHA256
        or recovery_authority.get("recovery_wrapper_sha256")
        != sha256_file(Path(__file__))
        or recovery_authority.get("robust_evaluation_source_sha256")
        != ROBUST_EVALUATION_SOURCE_SHA256
        or recovery_authority.get("failed_stderr_sha256") != FAILED_STDERR_SHA256
        or recovery_authority.get("failed_recovery_stderr_sha256")
        != FAILED_RECOVERY_STDERR_SHA256
        or recovery_authority.get("model_checkpoint_or_routing_changed") is not False
        or recovery_authority.get("split_candidates_thresholds_or_calibration_changed")
        is not False
        or recovery_authority.get("failed_attempt_wrote_result") is not False
    ):
        raise RuntimeError("measurement recovery authority differs")
    failed_stderr = (
        known.campaign_root
        / "logs"
        / ("component-1472578_0.err" if known.arm == "baseline_current" else "component-1472578_1.err")
    )
    if sha256_file(failed_stderr) != FAILED_STDERR_SHA256[known.arm]:
        raise RuntimeError("failed-attempt stderr SHA-256 differs")
    failed_recovery_stderr = (
        known.campaign_root
        / "logs"
        / (
            "recovery-1472589_0.err"
            if known.arm == "baseline_current"
            else "recovery-1472589_1.err"
        )
    )
    if (
        sha256_file(failed_recovery_stderr)
        != FAILED_RECOVERY_STDERR_SHA256[known.arm]
    ):
        raise RuntimeError("failed recovery stderr SHA-256 differs")
    if known.output.exists() or known.receipt.exists():
        raise FileExistsError("the failed attempt unexpectedly wrote an output")

    robust_paths = [
        str(ROBUST_SOURCE_ROOT / "scripts"),
        str(ROBUST_SOURCE_ROOT / "src"),
    ]
    sys.path[:0] = robust_paths
    robust_training = load_module(
        "framecontact_robust_test_evaluation_v1", ROBUST_EVALUATION_SOURCE
    )
    for path in robust_paths:
        sys.path.remove(path)
    for name in list(sys.modules):
        if name == "interaction_first" or name.startswith("interaction_first."):
            del sys.modules[name]

    source_root = known.source_root.resolve()
    sys.path[:0] = [str(source_root / "scripts"), str(source_root / "src")]
    import train_framecontact as model_training  # noqa: PLC0415

    def robust_evaluate_adapter(*args, **kwargs):
        evaluate_partner_retrieval = kwargs.pop("evaluate_partner_retrieval", None)
        if evaluate_partner_retrieval is not True:
            raise ValueError("frozen recovery requires partner retrieval evaluation")
        return robust_training._evaluate(*args, **kwargs)

    model_training._evaluate = robust_evaluate_adapter
    sys.modules["train_framecontact"] = model_training
    original = load_module("task_routed_exposed_test_original_v1", ORIGINAL_EVALUATOR)
    original.main()


if __name__ == "__main__":
    main()
