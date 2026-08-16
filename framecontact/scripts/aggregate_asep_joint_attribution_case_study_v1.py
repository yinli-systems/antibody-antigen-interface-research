#!/usr/bin/env python3
"""Aggregate same-antigen development case-study outputs without selecting models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

FORMAT = "framecontact_asep_joint_attribution_case_study_v1_aggregate"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric {label}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {label}")
    return value


def _summary(values: list[float]) -> dict[str, float]:
    if len(values) != 5:
        raise ValueError("Case-study aggregation requires five seeds")
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "min": min(values), "max": max(values)}


def _member(path: Path, *, condition: str, seed: int, cache_sha: str, challenge_sha: str) -> dict[str, Any]:
    result_path = path / "ASEP_SAME_ANTIGEN_CHALLENGE_RESULT.json"
    receipt_path = path / "RUN_RECEIPT.json"
    if any(item.is_symlink() or not item.is_file() for item in (result_path, receipt_path)):
        raise ValueError(f"Incomplete case-study result: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("result_sha256") != _sha256(result_path) or receipt.get("official_test_loaded") is not False or receipt.get("used_for_model_selection") is not False:
        raise ValueError(f"Case-study receipt failed authentication: {path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    authority = result.get("authority", {})
    if result.get("format_version") != "framecontact_asep_same_antigen_challenge_result_v1" or result.get("experiment") != condition or result.get("seed") != seed or authority.get("cache_manifest_sha256") != cache_sha or authority.get("challenge_sha256") != challenge_sha or authority.get("official_test_loaded") is not False:
        raise ValueError(f"Case-study authority differs: {path}")
    records = result.get("directional_records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Case-study payload is incomplete: {path}")
    margins = [_number(record.get("assignment_margin"), "assignment_margin") for record in records]
    successes = [bool(record.get("directional_success")) for record in records]
    shifts = [_number(record.get("mean_absolute_prediction_shift"), "prediction_shift") for record in records]
    return {"condition": condition, "seed": seed, "result_sha256": receipt["result_sha256"], "directional_case_count": len(records), "mean_assignment_margin": statistics.mean(margins), "directional_success_fraction": statistics.mean(successes), "mean_absolute_prediction_shift": statistics.mean(shifts), "authority": authority}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--conditions", nargs=3, required=True)
    parser.add_argument("--seeds", nargs=5, type=int, required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--challenge-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refusing to overwrite a case-study aggregate")
    conditions = tuple(args.conditions)
    seeds = tuple(args.seeds)
    by_condition = {condition: [_member(args.runs_root / condition / f"seed-{seed}", condition=condition, seed=seed, cache_sha=args.cache_manifest_sha256, challenge_sha=args.challenge_sha256) for seed in seeds] for condition in conditions}
    authorities = {json.dumps(member["authority"], sort_keys=True) for members in by_condition.values() for member in members}
    if len(authorities) != 1:
        raise ValueError("Case-study authorities differ across conditions")
    payload = {"format_version": FORMAT, "cache_manifest_sha256": args.cache_manifest_sha256, "challenge_sha256": args.challenge_sha256, "conditions": {condition: {"seed_count": 5, "directional_case_count_per_seed": sorted({member["directional_case_count"] for member in members}), "mean_assignment_margin": _summary([member["mean_assignment_margin"] for member in members]), "directional_success_fraction": _summary([member["directional_success_fraction"] for member in members]), "mean_absolute_prediction_shift": _summary([member["mean_absolute_prediction_shift"] for member in members]), "members": [{"seed": member["seed"], "result_sha256": member["result_sha256"]} for member in members]} for condition, members in by_condition.items()}, "authority": json.loads(next(iter(authorities))), "claim_boundary": "Same-antigen partner-retrieval diagnostic on the AsEP validation challenge panel; development-only, no checkpoint or architecture selection, and no frozen/official/external test role."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(payload))
    receipt = {"format_version": "framecontact_asep_joint_attribution_case_study_v1_aggregate_receipt", "aggregate_sha256": _sha256(args.output), "cache_manifest_sha256": args.cache_manifest_sha256, "challenge_sha256": args.challenge_sha256, "run_count": len(conditions) * len(seeds), "test_loaded": False, "frozen_test_loaded": False, "official_test_loaded": False, "external_test_loaded": False}
    receipt_path = args.output.with_name(args.output.name + ".receipt.json")
    receipt_path.write_bytes(_canonical(receipt))
    print(json.dumps({"output": str(args.output), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
