#!/usr/bin/env python3
"""Analyze adaptive-balanced versus class-balanced focal pair losses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping


EXPECTED = {
    "adaptive_balanced_control": {
        "mode": "adaptive_balanced",
        "protocol": "b4dd3ab8cabe9889204c5b32c0c8d24db2f8629c03d2f5880e34ff4f222e10f1",
        "status": "PASS_PAIR_LOSS_ADAPTIVE_CONTROL",
    },
    "balanced_focal": {
        "mode": "balanced_focal",
        "protocol": "ea5c0dbb216df312efd93fb585654bd3ceb10cfeda35ba9c0f963f3aa0f96249",
        "status": "PASS_PAIR_LOSS_FOCAL_CANARY",
    },
}
EXPECTED_SOURCE = "b47eff405951c07c08feaf6b4928e0daea040617356ae729b71619fd301aff4c"
EXPECTED_CHECKPOINT = "be22eb1889dd4f15b3d2c2ad449cb5084bde91435dd8d4d392f9295d6f10f839"
METRICS = (
    "contact_ap",
    "contact_auroc",
    "contact_mrr",
    "contact_recall_at_1",
    "contact_recall_at_l_over_10",
    "contact_precision_at_l_over_10",
    "paratope_ap",
    "epitope_ap",
    "matched_minus_swapped_mean_logit",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_create_only(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def load_arm(root: Path, arm: str) -> dict[str, Any]:
    path = root / arm / "RESULT.json"
    value = json.loads(path.read_text(encoding="ascii"))
    expected = EXPECTED[arm]
    if (
        value.get("status") != expected["status"]
        or value.get("test_access") is not False
        or value.get("objective", {}).get("pair_loss_mode") != expected["mode"]
        or value.get("provenance", {}).get("protocol") != expected["protocol"]
        or value.get("provenance", {}).get("source") != EXPECTED_SOURCE
        or value.get("input_checkpoint", {}).get("content_sha256")
        != EXPECTED_CHECKPOINT
        or value.get("checkpoint", {}).get("content_sha256")
        != value.get("checkpoint", {}).get("restore_content_sha256")
        or value.get("promotion_checks", {}).get("finite_all_update_metrics") is not True
    ):
        raise RuntimeError(f"{arm} result authority or execution checks differ")
    for phase in ("before", "after"):
        per_system = value.get("evaluation", {}).get(phase, {}).get("per_system")
        if not isinstance(per_system, list) or len(per_system) != 16:
            raise RuntimeError(f"{arm} {phase} per-system evidence differs")
        if len({record["example_id"] for record in per_system}) != 16:
            raise RuntimeError(f"{arm} {phase} membership is not unique")
        if any(
            not math.isfinite(float(record[metric]))
            for record in per_system
            for metric in METRICS
        ):
            raise RuntimeError(f"{arm} {phase} metrics are nonfinite")
    value["result_sha256"] = sha256_file(path)
    return value


def keyed(value: Mapping[str, Any], phase: str) -> dict[str, Mapping[str, Any]]:
    return {
        str(record["example_id"]): record
        for record in value["evaluation"][phase]["per_system"]
    }


def bootstrap_interval(values: list[float], *, seed: int = 2901) -> list[float]:
    generator = random.Random(seed)
    replicates = []
    for _ in range(10_000):
        sample = [values[generator.randrange(len(values))] for _ in values]
        replicates.append(sum(sample) / len(sample))
    replicates.sort()
    return [replicates[249], replicates[9749]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arms = {arm: load_arm(arguments.campaign_root, arm) for arm in EXPECTED}
    membership = set(keyed(arms["adaptive_balanced_control"], "before"))
    if any(
        set(keyed(value, phase)) != membership
        for value in arms.values()
        for phase in ("before", "after")
    ):
        raise RuntimeError("Matched pair-loss arms use different systems")

    evidence: dict[str, Any] = {}
    for arm, value in arms.items():
        before = keyed(value, "before")
        after = keyed(value, "after")
        evidence[arm] = {
            metric: sum(float(after[key][metric]) for key in membership) / len(membership)
            for metric in METRICS
        }
        evidence[arm]["learned_gain"] = {
            metric: sum(
                float(after[key][metric]) - float(before[key][metric])
                for key in membership
            )
            / len(membership)
            for metric in METRICS
        }

    control_before = keyed(arms["adaptive_balanced_control"], "before")
    control_after = keyed(arms["adaptive_balanced_control"], "after")
    focal_before = keyed(arms["balanced_focal"], "before")
    focal_after = keyed(arms["balanced_focal"], "after")
    delta_of_deltas = {}
    for metric in METRICS:
        paired = [
            (float(focal_after[key][metric]) - float(focal_before[key][metric]))
            - (float(control_after[key][metric]) - float(control_before[key][metric]))
            for key in sorted(membership)
        ]
        delta_of_deltas[metric] = {
            "mean": sum(paired) / len(paired),
            "paired_bootstrap_95_ci": bootstrap_interval(paired),
        }

    focal = evidence["balanced_focal"]
    control = evidence["adaptive_balanced_control"]
    checks = {
        "contact_ap_gain_over_control": delta_of_deltas["contact_ap"]["mean"] > 0.0,
        "contact_ap_above_prevalence": (
            focal["contact_ap"]
            > arms["balanced_focal"]["evaluation"]["after"]["contact_prevalence"]
        ),
        "contact_mrr_noninferior_to_control": focal["contact_mrr"] >= control["contact_mrr"],
        "recall_at_1_noninferior_to_control": (
            focal["contact_recall_at_1"] >= control["contact_recall_at_1"]
        ),
    }
    promote = all(checks.values())
    result = {
        "arm_result_sha256": {
            arm: value["result_sha256"] for arm, value in arms.items()
        },
        "checks": checks,
        "delta_of_deltas": delta_of_deltas,
        "format_version": "abpairfm_pair_loss_canary_analysis_v1",
        "holdout_systems": len(membership),
        "means_after_training": evidence,
        "next_action": (
            "PROMOTE_BALANCED_FOCAL_TO_PREREGISTERED_B32K_ABLATION"
            if promote
            else "RETAIN_ADAPTIVE_BALANCED_PAIR_LOSS"
        ),
        "status": (
            "PASS_BALANCED_FOCAL_PAIR_RANKING_GATE"
            if promote
            else "FAIL_BALANCED_FOCAL_PAIR_RANKING_GATE"
        ),
        "test_access": False,
    }
    publish_create_only(arguments.output, result)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
