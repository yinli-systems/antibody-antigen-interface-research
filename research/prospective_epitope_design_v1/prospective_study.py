#!/usr/bin/env python3
"""Freeze and analyze a blinded prospective three-policy candidate study."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable


ARMS = ("generator_baseline", "partner_specific", "uncertainty_diversity")
CANDIDATE_COLUMNS = (
    "candidate_id",
    "payload_sha256",
    "eligible",
    "baseline_score",
    "partner_score",
    "acquisition_score",
    "diversity_cluster",
)
RESULT_COLUMNS = (
    "blinded_id",
    "primary_hit",
    "primary_value",
    "confirmatory_hit",
    "expression_pass",
)
CONTROL_COLUMNS = ("control_id", "control_role", "primary_value", "control_pass")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_nonempty(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def parse_bool(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be true/false or 1/0")


def parse_binary_or_blank(value: str, label: str) -> bool | None:
    if not value.strip():
        return None
    return parse_bool(value, label)


def parse_finite(value: str | int | float, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def read_exact_tsv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(
                f"{path} columns differ; expected {columns}, observed {reader.fieldnames}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty TSV: {path}")
    return rows


def validate_protocol_draft(protocol: dict[str, Any]) -> None:
    if protocol.get("format_version") != "prospective_epitope_selection_protocol_v1":
        raise ValueError("protocol format_version differs")
    if protocol.get("status") != "DRAFT_BLOCKED_ON_TARGET_ASSAY_BUDGET_AND_SAFETY_REVIEW":
        raise ValueError("input must be an unfrozen protocol draft")
    target = protocol.get("target", {})
    for key in (
        "target_id",
        "molecule_type",
        "epitope_definition",
        "bound_state_or_structure_source_disclosure",
    ):
        require_nonempty(target.get(key), f"target.{key}")
    if not SHA256_RE.fullmatch(str(target.get("antigen_artifact_sha256", ""))):
        raise ValueError("target.antigen_artifact_sha256 must be lowercase SHA-256")

    safety = protocol.get("safety", {})
    if safety.get("approved") is not True:
        raise ValueError("independent safety approval is required")
    require_nonempty(safety.get("reviewer"), "safety.reviewer")
    require_nonempty(safety.get("review_date"), "safety.review_date")
    if safety.get("target_is_pathogen_or_toxin") is not False:
        raise ValueError("this scaffold excludes pathogen and toxin targets")
    if safety.get("gain_of_function_in_scope") is not False:
        raise ValueError("gain-of-function work is out of scope")

    authority = protocol.get("computational_authority", {})
    for key in (
        "generator_id",
        "eligibility_rule",
        "baseline_score_definition",
        "partner_specific_model_id",
        "partner_score_definition",
        "acquisition_score_definition",
        "diversity_cluster_definition",
    ):
        require_nonempty(authority.get(key), f"computational_authority.{key}")
    for key in (
        "generator_revision_or_weight_sha256",
        "generation_protocol_sha256",
        "proposal_pool_manifest_sha256",
        "partner_specific_checkpoint_sha256",
        "acquisition_score_implementation_sha256",
        "diversity_cluster_implementation_sha256",
    ):
        if not SHA256_RE.fullmatch(str(authority.get(key, ""))):
            raise ValueError(f"computational_authority.{key} must be lowercase SHA-256")
    if authority.get("candidate_outcomes_used_to_fit_or_select_any_authority") is not False:
        raise ValueError("computational authorities must be frozen without candidate outcomes")

    assay = protocol.get("assay", {})
    for key in (
        "primary_name",
        "primary_hit_rule",
        "threshold_unit",
        "positive_control",
        "negative_control",
        "confirmatory_assay",
    ):
        require_nonempty(assay.get(key), f"assay.{key}")
    parse_finite(assay.get("threshold_value"), "assay.threshold_value")
    if assay.get("direction") not in {"lower_is_better", "higher_is_better"}:
        raise ValueError("assay.direction must be lower_is_better or higher_is_better")
    if assay.get("positive_control") == assay.get("negative_control"):
        raise ValueError("positive and negative controls must differ")
    if assay.get("lab_blinded_to_selection_arm") is not True:
        raise ValueError("laboratory arm blinding is required")

    design = protocol.get("design", {})
    budget = design.get("candidate_budget_per_arm")
    if not isinstance(budget, int) or isinstance(budget, bool) or not 3 <= budget <= 96:
        raise ValueError("design.candidate_budget_per_arm must be an integer in [3, 96]")
    for key in (
        "proposal_pool_frozen_before_selection",
        "overlap_across_policy_slates_allowed",
        "overlapping_candidate_assayed_once",
    ):
        if design.get(key) is not True:
            raise ValueError(f"design.{key} must be true")
    if design.get("candidate_outcomes_available_before_freeze") is not False:
        raise ValueError("candidate outcomes must be unavailable before freeze")
    if tuple(design.get("arms", {}).keys()) != ARMS:
        raise ValueError(f"design arms must be ordered exactly as {ARMS}")
    expected_scores = {
        "generator_baseline": "baseline_score",
        "partner_specific": "partner_score",
        "uncertainty_diversity": "acquisition_score",
    }
    for arm, score in expected_scores.items():
        if design["arms"][arm].get("score_column") != score:
            raise ValueError(f"{arm} score column differs")
    expected_selection = {
        "generator_baseline": "descending_score_then_candidate_id",
        "partner_specific": "descending_score_then_candidate_id",
        "uncertainty_diversity": "distinct_diversity_clusters_first_then_descending_score_then_candidate_id",
    }
    for arm, selection in expected_selection.items():
        if design["arms"][arm].get("selection") != selection:
            raise ValueError(f"{arm} selection rule differs")
    bootstrap = design.get("bootstrap", {})
    if bootstrap.get("method") != "candidate_level_bayesian_bootstrap":
        raise ValueError("bootstrap method differs")
    if not isinstance(bootstrap.get("replicates"), int) or bootstrap["replicates"] < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    if not isinstance(bootstrap.get("seed"), int):
        raise ValueError("bootstrap seed must be an integer")

    roles = protocol.get("role_separation", {})
    require_nonempty(roles.get("private_mapping_custodian"), "role_separation.private_mapping_custodian")
    require_nonempty(roles.get("lab_results_custodian"), "role_separation.lab_results_custodian")
    if roles.get("same_person_allowed") is not False:
        raise ValueError("mapping and results custodians must be role-separated")
    if roles["private_mapping_custodian"] == roles["lab_results_custodian"]:
        raise ValueError("mapping and results custodians must differ")


def freeze_protocol(args: argparse.Namespace) -> None:
    protocol = load_json(args.input)
    validate_protocol_draft(protocol)
    frozen = dict(protocol)
    frozen["status"] = "FROZEN_BEFORE_CANDIDATE_OUTCOME_ACCESS"
    frozen["frozen_at_utc"] = args.frozen_at_utc or datetime.now(timezone.utc).isoformat()
    frozen["tool_binding"] = {
        "prospective_study_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_write(args.output, canonical_bytes(frozen))
    receipt = {
        "format_version": "prospective_epitope_protocol_receipt_v1",
        "status": frozen["status"],
        "source_draft_sha256": sha256_file(args.input),
        "protocol_sha256": sha256_file(args.output),
        "candidate_outcomes_read": False,
        "target_safety_review_approved": True,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))


def validate_candidates(path: Path) -> list[dict[str, Any]]:
    raw = read_exact_tsv(path, CANDIDATE_COLUMNS)
    rows: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    payloads: set[str] = set()
    for index, row in enumerate(raw, start=2):
        candidate_id = row["candidate_id"].strip()
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError(f"duplicate or empty candidate_id at line {index}")
        payload_sha = row["payload_sha256"].strip()
        if not SHA256_RE.fullmatch(payload_sha) or payload_sha in payloads:
            raise ValueError(f"invalid or duplicate payload_sha256 at line {index}")
        cluster = row["diversity_cluster"].strip()
        if not cluster:
            raise ValueError(f"empty diversity_cluster at line {index}")
        rows.append(
            {
                "candidate_id": candidate_id,
                "payload_sha256": payload_sha,
                "eligible": parse_bool(row["eligible"], f"eligible line {index}"),
                "baseline_score": parse_finite(row["baseline_score"], f"baseline_score line {index}"),
                "partner_score": parse_finite(row["partner_score"], f"partner_score line {index}"),
                "acquisition_score": parse_finite(row["acquisition_score"], f"acquisition_score line {index}"),
                "diversity_cluster": cluster,
            }
        )
        candidate_ids.add(candidate_id)
        payloads.add(payload_sha)
    return rows


def descending(rows: Iterable[dict[str, Any]], score: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-row[score], row["candidate_id"]))


def select_diverse(rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    ranked = descending(rows, "acquisition_score")
    selected: list[dict[str, Any]] = []
    seen_clusters: set[str] = set()
    for row in ranked:
        if row["diversity_cluster"] not in seen_clusters:
            selected.append(row)
            seen_clusters.add(row["diversity_cluster"])
            if len(selected) == budget:
                return selected
    selected_ids = {row["candidate_id"] for row in selected}
    for row in ranked:
        if row["candidate_id"] not in selected_ids:
            selected.append(row)
            if len(selected) == budget:
                return selected
    raise ValueError("insufficient candidates for diversity selection")


def freeze_slate(args: argparse.Namespace) -> None:
    protocol = load_json(args.protocol)
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise ValueError("protocol SHA-256 differs")
    if protocol.get("status") != "FROZEN_BEFORE_CANDIDATE_OUTCOME_ACCESS":
        raise ValueError("protocol is not frozen")
    if "private" not in args.private_mapping.parts:
        raise ValueError("private mapping path must contain a private directory")
    rows = validate_candidates(args.candidates)
    eligible = [row for row in rows if row["eligible"]]
    budget = protocol["design"]["candidate_budget_per_arm"]
    if len(eligible) < budget:
        raise ValueError("insufficient eligible candidates")
    selections = {
        "generator_baseline": descending(eligible, "baseline_score")[:budget],
        "partner_specific": descending(eligible, "partner_score")[:budget],
        "uncertainty_diversity": select_diverse(eligible, budget),
    }
    key = args.blinding_key_file.read_bytes()
    if len(key) < 32:
        raise ValueError("blinding key must contain at least 32 bytes")
    memberships: dict[str, list[str]] = {}
    for arm in ARMS:
        if len(selections[arm]) != budget:
            raise ValueError(f"{arm} budget differs")
        for row in selections[arm]:
            memberships.setdefault(row["candidate_id"], []).append(arm)
    by_id = {row["candidate_id"]: row for row in eligible}
    records: list[dict[str, Any]] = []
    for candidate_id, arms in memberships.items():
        digest = hmac.new(
            key,
            f"{args.protocol_sha256}\0{candidate_id}".encode(),
            hashlib.sha256,
        ).hexdigest()
        records.append(
            {
                "blinded_id": f"C-{digest[:16]}",
                "candidate_id": candidate_id,
                "payload_sha256": by_id[candidate_id]["payload_sha256"],
                "arms": arms,
                "scores": {
                    "baseline_score": by_id[candidate_id]["baseline_score"],
                    "partner_score": by_id[candidate_id]["partner_score"],
                    "acquisition_score": by_id[candidate_id]["acquisition_score"],
                },
                "diversity_cluster": by_id[candidate_id]["diversity_cluster"],
            }
        )
    if len({record["blinded_id"] for record in records}) != len(records):
        raise ValueError("blinded ID collision")
    records.sort(
        key=lambda record: hmac.new(
            key, f"order\0{record['blinded_id']}".encode(), hashlib.sha256
        ).digest()
    )
    public_records = [
        {
            "lab_order": index,
            "blinded_id": record["blinded_id"],
            "payload_sha256": record["payload_sha256"],
        }
        for index, record in enumerate(records, start=1)
    ]
    public = {
        "format_version": "prospective_blinded_candidate_manifest_v1",
        "status": "FROZEN_BLINDED_CANDIDATE_SLATE",
        "protocol_sha256": args.protocol_sha256,
        "unique_candidate_count": len(records),
        "candidate_budget_per_arm": budget,
        "candidates": public_records,
        "arm_membership_disclosed": False,
        "model_scores_disclosed": False,
    }
    private = {
        "format_version": "prospective_private_candidate_mapping_v1",
        "status": public["status"],
        "protocol_sha256": args.protocol_sha256,
        "candidate_source_sha256": sha256_file(args.candidates),
        "records": records,
    }
    atomic_write(args.public_manifest, canonical_bytes(public))
    atomic_write(args.private_mapping, canonical_bytes(private))
    overlap_counts = {
        str(size): sum(len(record["arms"]) == size for record in records)
        for size in (1, 2, 3)
    }
    receipt = {
        "format_version": "prospective_blinded_candidate_manifest_receipt_v1",
        "status": public["status"],
        "protocol_sha256": args.protocol_sha256,
        "candidate_source_sha256": sha256_file(args.candidates),
        "public_manifest_sha256": sha256_file(args.public_manifest),
        "private_mapping_sha256": sha256_file(args.private_mapping),
        "private_mapping_path_recorded": False,
        "eligible_candidate_count": len(eligible),
        "unique_selected_candidate_count": len(records),
        "candidate_budget_per_arm": budget,
        "membership_count_distribution": overlap_counts,
        "candidate_outcomes_read": False,
        "arm_membership_disclosed_to_lab": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))


def wilson_interval(hits: int, total: int) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    z = 1.959963984540054
    p = hits / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def hit_from_value(value: float, assay: dict[str, Any]) -> bool:
    threshold = parse_finite(assay["threshold_value"], "assay.threshold_value")
    if assay["direction"] == "higher_is_better":
        return value >= threshold
    if assay["direction"] == "lower_is_better":
        return value <= threshold
    raise ValueError("assay direction differs")


def validate_controls(path: Path, assay: dict[str, Any]) -> dict[str, Any]:
    rows = read_exact_tsv(path, CONTROL_COLUMNS)
    if len(rows) != 2:
        raise ValueError("control result file must contain exactly two rows")
    expected = {
        "positive": assay["positive_control"],
        "negative": assay["negative_control"],
    }
    observed: dict[str, Any] = {}
    for index, row in enumerate(rows, start=2):
        role = row["control_role"].strip()
        if role not in expected or role in observed:
            raise ValueError(f"invalid or duplicate control role at line {index}")
        if row["control_id"].strip() != expected[role]:
            raise ValueError(f"{role} control ID differs from protocol")
        value = parse_finite(row["primary_value"], f"control primary_value line {index}")
        derived_pass = hit_from_value(value, assay) if role == "positive" else not hit_from_value(value, assay)
        declared_pass = parse_bool(row["control_pass"], f"control_pass line {index}")
        if declared_pass != derived_pass:
            raise ValueError(f"{role} control_pass differs from frozen hit rule")
        observed[role] = {
            "control_id": expected[role],
            "primary_value": value,
            "pass": derived_pass,
        }
    if set(observed) != set(expected):
        raise ValueError("positive and negative control rows are required")
    if not all(item["pass"] for item in observed.values()):
        raise ValueError("assay controls did not pass")
    return observed


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of empty values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def analyze(args: argparse.Namespace) -> None:
    protocol = load_json(args.protocol)
    if sha256_file(args.protocol) != args.protocol_sha256:
        raise ValueError("protocol SHA-256 differs")
    if protocol.get("status") != "FROZEN_BEFORE_CANDIDATE_OUTCOME_ACCESS":
        raise ValueError("protocol is not frozen")
    public = load_json(args.public_manifest)
    private = load_json(args.private_mapping)
    candidate_receipt = load_json(args.candidate_receipt)
    if public.get("protocol_sha256") != args.protocol_sha256 or private.get("protocol_sha256") != args.protocol_sha256:
        raise ValueError("candidate authority differs")
    if (
        candidate_receipt.get("status") != "FROZEN_BLINDED_CANDIDATE_SLATE"
        or candidate_receipt.get("protocol_sha256") != args.protocol_sha256
        or candidate_receipt.get("public_manifest_sha256") != sha256_file(args.public_manifest)
        or candidate_receipt.get("private_mapping_sha256") != sha256_file(args.private_mapping)
        or candidate_receipt.get("candidate_outcomes_read") is not False
        or candidate_receipt.get("arm_membership_disclosed_to_lab") is not False
    ):
        raise ValueError("candidate receipt or bound candidate artifacts differ")
    public_ids = [row["blinded_id"] for row in public.get("candidates", [])]
    records = private.get("records", [])
    private_ids = [row["blinded_id"] for row in records]
    if len(public_ids) != len(set(public_ids)) or set(public_ids) != set(private_ids):
        raise ValueError("public/private candidate IDs differ")
    control_results = validate_controls(args.control_results, protocol["assay"])
    raw_results = read_exact_tsv(args.assay_results, RESULT_COLUMNS)
    outcomes: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw_results, start=2):
        blinded_id = row["blinded_id"].strip()
        if blinded_id in outcomes:
            raise ValueError(f"duplicate blinded_id at line {index}")
        primary_value = parse_finite(row["primary_value"], f"primary_value line {index}")
        primary_hit = parse_bool(row["primary_hit"], f"primary_hit line {index}")
        if primary_hit != hit_from_value(primary_value, protocol["assay"]):
            raise ValueError(f"primary_hit differs from frozen hit rule at line {index}")
        outcomes[blinded_id] = {
            "primary_hit": primary_hit,
            "primary_value": primary_value,
            "confirmatory_hit": parse_binary_or_blank(row["confirmatory_hit"], f"confirmatory_hit line {index}"),
            "expression_pass": parse_binary_or_blank(row["expression_pass"], f"expression_pass line {index}"),
        }
    if set(outcomes) != set(public_ids):
        raise ValueError("assay result IDs must exactly match the blinded manifest")
    record_by_id = {row["blinded_id"]: row for row in records}
    arm_members = {
        arm: sorted(row["blinded_id"] for row in records if arm in row["arms"])
        for arm in ARMS
    }
    budget = protocol["design"]["candidate_budget_per_arm"]
    if any(len(arm_members[arm]) != budget for arm in ARMS):
        raise ValueError("private mapping arm budget differs from protocol")
    arm_metrics: dict[str, Any] = {}
    for arm in ARMS:
        members = arm_members[arm]
        hits = sum(outcomes[candidate]["primary_hit"] for candidate in members)
        confirmed = [outcomes[candidate]["confirmatory_hit"] for candidate in members]
        expressed = [outcomes[candidate]["expression_pass"] for candidate in members]
        arm_metrics[arm] = {
            "n": len(members),
            "primary_hits": hits,
            "primary_hit_rate": hits / len(members),
            "primary_hit_rate_wilson_95ci": wilson_interval(hits, len(members)),
            "primary_value_mean": sum(outcomes[candidate]["primary_value"] for candidate in members) / len(members),
            "confirmatory_observed_n": sum(value is not None for value in confirmed),
            "confirmatory_hits": sum(value is True for value in confirmed),
            "expression_observed_n": sum(value is not None for value in expressed),
            "expression_passes": sum(value is True for value in expressed),
        }

    rng = random.Random(protocol["design"]["bootstrap"]["seed"])
    replicates = protocol["design"]["bootstrap"]["replicates"]
    union_ids = sorted(record_by_id)
    pairwise_samples = {arm: [] for arm in ARMS if arm != "generator_baseline"}
    for _ in range(replicates):
        weights = {candidate: rng.expovariate(1.0) for candidate in union_ids}
        rates: dict[str, float] = {}
        for arm in ARMS:
            denominator = sum(weights[candidate] for candidate in arm_members[arm])
            numerator = sum(
                weights[candidate] * outcomes[candidate]["primary_hit"]
                for candidate in arm_members[arm]
            )
            rates[arm] = numerator / denominator
        for arm in pairwise_samples:
            pairwise_samples[arm].append(rates[arm] - rates["generator_baseline"])
    pairwise = {}
    baseline_rate = arm_metrics["generator_baseline"]["primary_hit_rate"]
    for arm, samples in pairwise_samples.items():
        pairwise[f"{arm}_minus_generator_baseline"] = {
            "risk_difference": arm_metrics[arm]["primary_hit_rate"] - baseline_rate,
            "candidate_level_bayesian_bootstrap_95ci": [
                percentile(samples, 0.025),
                percentile(samples, 0.975),
            ],
        }
    result = {
        "format_version": "prospective_blinded_assay_result_v1",
        "status": "PASS_BLINDED_ASSAY_ANALYSIS",
        "protocol_sha256": args.protocol_sha256,
        "public_manifest_sha256": sha256_file(args.public_manifest),
        "private_mapping_sha256": sha256_file(args.private_mapping),
        "candidate_receipt_sha256": sha256_file(args.candidate_receipt),
        "assay_results_sha256": sha256_file(args.assay_results),
        "control_results_sha256": sha256_file(args.control_results),
        "unique_assayed_candidates": len(union_ids),
        "candidate_budget_per_arm": budget,
        "arm_metrics": arm_metrics,
        "pairwise_primary_risk_differences": pairwise,
        "controls": control_results,
        "bootstrap": protocol["design"]["bootstrap"],
        "audit": {
            "all_blinded_ids_accounted_for": True,
            "arm_membership_loaded_only_during_analysis": True,
            "primary_hit_rule_refit": False,
            "candidate_slate_changed_after_assay": False,
            "unreported_assayed_candidates": 0,
            "positive_and_negative_controls_passed": True,
            "primary_hits_recomputed_from_frozen_rule": True,
        },
    }
    atomic_write(args.output, canonical_bytes(result))
    receipt = {
        "format_version": "prospective_blinded_assay_result_receipt_v1",
        "status": result["status"],
        "result_sha256": sha256_file(args.output),
        "protocol_sha256": args.protocol_sha256,
        "assay_results_sha256": sha256_file(args.assay_results),
        "control_results_sha256": sha256_file(args.control_results),
    }
    atomic_write(args.receipt, canonical_bytes(receipt))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-protocol")
    freeze.add_argument("--input", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--receipt", type=Path, required=True)
    freeze.add_argument("--frozen-at-utc")
    freeze.set_defaults(function=freeze_protocol)

    slate = subparsers.add_parser("freeze-slate")
    slate.add_argument("--protocol", type=Path, required=True)
    slate.add_argument("--protocol-sha256", required=True)
    slate.add_argument("--candidates", type=Path, required=True)
    slate.add_argument("--blinding-key-file", type=Path, required=True)
    slate.add_argument("--public-manifest", type=Path, required=True)
    slate.add_argument("--private-mapping", type=Path, required=True)
    slate.add_argument("--receipt", type=Path, required=True)
    slate.set_defaults(function=freeze_slate)

    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--protocol", type=Path, required=True)
    analysis.add_argument("--protocol-sha256", required=True)
    analysis.add_argument("--public-manifest", type=Path, required=True)
    analysis.add_argument("--private-mapping", type=Path, required=True)
    analysis.add_argument("--candidate-receipt", type=Path, required=True)
    analysis.add_argument("--assay-results", type=Path, required=True)
    analysis.add_argument("--control-results", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--receipt", type=Path, required=True)
    analysis.set_defaults(function=analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
