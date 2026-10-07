#!/usr/bin/env python3
"""Analyze the frozen matched low-rank versus bipartite decoder canary."""

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
    "low_rank_control": {
        "protocol": "ee4ea5a195975adf2e02f3560372b71bbf0b0f8d265f759d31b9383b97c56c34",
        "status": "PASS_BIPARTITE_DECODER_LOW_RANK_CONTROL",
    },
    "bipartite_edge_refinement": {
        "protocol": "682f6b5667912acc41c227b1e6e8b2913a8d9e922a7d1729125fe95b2a32bef2",
        "status": "PASS_BIPARTITE_EDGE_REFINEMENT_CANARY",
    },
}
EXPECTED_SOURCE = "78736d49f17eb622bcb2f48cc7f1853563d111b42f5969f5d21c0ebbe2700b51"
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
            raise RuntimeError(f"{arm} {phase} system membership is not unique")
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


def bootstrap_interval(values: list[float], *, seed: int = 1701) -> list[float]:
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
    evidence: dict[str, Any] = {}
    delta_of_deltas: dict[str, Any] = {}
    membership: set[str] | None = None
    for arm, value in arms.items():
        before = keyed(value, "before")
        after = keyed(value, "after")
        if set(before) != set(after):
            raise RuntimeError(f"{arm} before/after membership differs")
        if membership is None:
            membership = set(before)
        elif membership != set(before):
            raise RuntimeError("Matched decoder arms use different systems")
        evidence[arm] = {
            metric: sum(float(after[key][metric]) for key in after) / len(after)
            for metric in METRICS
        }
        evidence[arm]["learned_gain"] = {
            metric: sum(
                float(after[key][metric]) - float(before[key][metric])
                for key in after
            )
            / len(after)
            for metric in METRICS
        }

    assert membership is not None
    control_before = keyed(arms["low_rank_control"], "before")
    control_after = keyed(arms["low_rank_control"], "after")
    edge_before = keyed(arms["bipartite_edge_refinement"], "before")
    edge_after = keyed(arms["bipartite_edge_refinement"], "after")
    for metric in METRICS:
        paired = [
            (float(edge_after[key][metric]) - float(edge_before[key][metric]))
            - (float(control_after[key][metric]) - float(control_before[key][metric]))
            for key in sorted(membership)
        ]
        delta_of_deltas[metric] = {
            "mean": sum(paired) / len(paired),
            "paired_bootstrap_95_ci": bootstrap_interval(paired),
        }

    edge = evidence["bipartite_edge_refinement"]
    control = evidence["low_rank_control"]
    checks = {
        "contact_ap_gain_over_control": (
            delta_of_deltas["contact_ap"]["mean"] > 0.0
        ),
        "contact_ap_above_prevalence": (
            edge["contact_ap"]
            > arms["bipartite_edge_refinement"]["evaluation"]["after"][
                "contact_prevalence"
            ]
        ),
        "contact_mrr_noninferior_to_control": edge["contact_mrr"] >= control["contact_mrr"],
        "recall_at_1_noninferior_to_control": (
            edge["contact_recall_at_1"] >= control["contact_recall_at_1"]
        ),
    }
    promote = all(checks.values())
    result = {
        "arm_result_sha256": {
            arm: value["result_sha256"] for arm, value in arms.items()
        },
        "checks": checks,
        "delta_of_deltas": delta_of_deltas,
        "format_version": "abpairfm_bipartite_decoder_canary_analysis_v1",
        "holdout_systems": len(membership),
        "means_after_training": evidence,
        "next_action": (
            "PROMOTE_EDGE_REFINEMENT_TO_PREREGISTERED_B32K_ABLATION"
            if promote
            else "RETAIN_LOW_RANK_DECODER_AND_ARCHIVE_EDGE_REFINEMENT"
        ),
        "status": (
            "PASS_BIPARTITE_EDGE_REFINEMENT_RANKING_GATE"
            if promote
            else "FAIL_BIPARTITE_EDGE_REFINEMENT_RANKING_GATE"
        ),
        "test_access": False,
    }
    publish_create_only(arguments.output, result)
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
