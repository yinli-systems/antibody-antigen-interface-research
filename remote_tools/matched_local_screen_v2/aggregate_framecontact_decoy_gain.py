#!/usr/bin/env python3
"""Freeze and aggregate a five-seed FrameContact decoy-gain study.

The script is deliberately read-only with respect to run directories.  It
verifies every result/checkpoint receipt before atomically creating a new,
non-overwritable aggregate directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable


SEEDS = (17, 42, 73, 101, 137)
BOOTSTRAP_SEED = 20_260_821
BOOTSTRAP_REPLICATES = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError("/".join(keys))
        current = current[key]
    return current


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {result}")
    return result


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    data = [finite_number(value, "aggregate value") for value in values]
    if not data:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "sample_sd": statistics.stdev(data) if len(data) > 1 else 0.0,
        "minimum": min(data),
        "maximum": max(data),
    }


def quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a quantile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap(deltas: list[float]) -> dict[str, float | int]:
    if len(deltas) != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} paired seed deltas, got {len(deltas)}")
    generator = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        samples.append(
            statistics.fmean(generator.choice(deltas) for _ in range(len(deltas)))
        )
    samples.sort()
    return {
        "estimate": statistics.fmean(deltas),
        "ci95_lower": quantile(samples, 0.025),
        "ci95_upper": quantile(samples, 0.975),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "paired_training_seed",
    }


def collect_metrics(result: dict[str, Any]) -> dict[str, float]:
    validation = nested(result, "selected_checkpoint_validation")
    metrics: dict[str, float] = {
        "selection_metric": finite_number(validation["selection_metric"], "selection_metric"),
        "selected_epoch": finite_number(nested(result, "selection", "best_epoch"), "best_epoch"),
        "contact_auprc": finite_number(nested(validation, "contact", "auprc"), "contact_auprc"),
        "contact_auroc": finite_number(nested(validation, "contact", "auroc"), "contact_auroc"),
        "contact_macro_auprc": finite_number(
            nested(validation, "contact", "mean_per_complex_auprc"), "contact_macro_auprc"
        ),
        "contact_precision": finite_number(nested(validation, "contact", "precision"), "contact_precision"),
        "contact_recall": finite_number(nested(validation, "contact", "recall"), "contact_recall"),
        "contact_f1": finite_number(nested(validation, "contact", "f1"), "contact_f1"),
        "contact_mcc": finite_number(nested(validation, "contact", "mcc"), "contact_mcc"),
        "contact_brier": finite_number(nested(validation, "contact", "brier"), "contact_brier"),
        "contact_ece_15": finite_number(nested(validation, "contact", "ece_15"), "contact_ece_15"),
        "contact_threshold": finite_number(
            nested(validation, "contact", "threshold_selected_on_validation"), "contact_threshold"
        ),
        "validation_temperature": finite_number(validation["validation_temperature"], "validation_temperature"),
    }
    for target in ("epitope", "paratope"):
        for name in ("auprc", "auroc", "f1", "mcc", "brier", "ece_15"):
            metrics[f"{target}_{name}"] = finite_number(
                nested(validation, target, name), f"{target}_{name}"
            )
    for cutoff in ("10", "20", "50", "lmin_over_10", "lmin_over_5", "lmin"):
        for name in ("mean_precision", "mean_recall"):
            metrics[f"pair_{name}_{cutoff}"] = finite_number(
                nested(validation, "contact", "ranked_pair_macro", cutoff, name),
                f"pair_{name}_{cutoff}",
            )
    directions = nested(validation, "partner_retrieval_by_direction")
    for direction in ("fixed_antibody_wrong_antigen", "fixed_antigen_wrong_antibody"):
        short = "ab_to_ag" if direction == "fixed_antibody_wrong_antigen" else "ag_to_ab"
        for source_name, output_name in (
            ("top1_retrieval_accuracy", "recall_at_1"),
            ("mean_reciprocal_rank", "mrr"),
            ("cluster_weighted_cognate_compatibility_margin", "cluster_margin"),
            ("complex_weighted_cognate_compatibility_margin", "complex_margin"),
        ):
            metrics[f"partner_{short}_{output_name}"] = finite_number(
                nested(directions, direction, source_name),
                f"partner_{short}_{output_name}",
            )
    return metrics


def load_member(family_root: Path, seed: int) -> dict[str, Any]:
    run_root = family_root / "framecontact" / f"seed-{seed}"
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
    for path in (result_path, checkpoint_path, receipt_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    result_sha256 = sha256_file(result_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    receipt = read_json(receipt_path)
    if receipt.get("seed") != seed:
        raise ValueError(f"Seed mismatch in {receipt_path}")
    if receipt.get("result_sha256") != result_sha256:
        raise ValueError(f"Result SHA256 mismatch in {receipt_path}")
    if receipt.get("selected_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"Checkpoint SHA256 mismatch in {receipt_path}")
    if receipt.get("test_loaded") is not False or receipt.get("external_test_loaded") is not False:
        raise ValueError(f"Test authority was opened in {receipt_path}")

    result = read_json(result_path)
    authority = nested(result, "authority")
    if authority.get("test_loaded") is not False or authority.get("external_test_loaded") is not False:
        raise ValueError(f"Test authority was opened in {result_path}")
    if nested(result, "configuration", "seed") != seed:
        raise ValueError(f"Configuration seed mismatch in {result_path}")
    if int(nested(result, "selection", "best_epoch")) != int(receipt["selected_epoch"]):
        raise ValueError(f"Selected epoch mismatch in {receipt_path}")

    return {
        "seed": seed,
        "run_root": str(run_root),
        "result_sha256": result_sha256,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "authority": authority,
        "configuration": result["configuration"],
        "data": result["data"],
        "metrics": collect_metrics(result),
    }


def consistent_authority(members: list[dict[str, Any]]) -> dict[str, Any]:
    first = members[0]["authority"]
    for member in members[1:]:
        if member["authority"] != first:
            raise ValueError("Authority differs across seeds")
    return first


def aggregate_metrics(members: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    names = sorted(members[0]["metrics"])
    if any(sorted(member["metrics"]) != names for member in members):
        raise ValueError("Metric schema differs across seeds")
    return {name: summarize(member["metrics"][name] for member in members) for name in names}


def paired_deltas(
    candidate: list[dict[str, Any]], control: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_seed_control = {member["seed"]: member for member in control}
    output = {}
    for name in sorted(candidate[0]["metrics"]):
        deltas = [
            member["metrics"][name] - by_seed_control[member["seed"]]["metrics"][name]
            for member in candidate
        ]
        output[name] = {
            "candidate_minus_control_by_seed": dict(zip(map(str, SEEDS), deltas)),
            "summary": summarize(deltas),
            "paired_bootstrap": paired_bootstrap(deltas),
        }
    return output


def format_metric(aggregate: dict[str, Any], name: str) -> str:
    value = aggregate[name]
    return f"{value['mean']:.6f} +/- {value['sample_sd']:.6f}"


def render_markdown(payload: dict[str, Any]) -> str:
    candidate = payload["candidate"]["aggregate_metrics"]
    control = payload["control"]["aggregate_metrics"]
    delta = payload["paired_deltas"]
    lines = [
        "# FrameContact local hard-decoy gain audit",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "All five result JSONs and selected checkpoints passed SHA256 receipt verification. "
        "The SAbDab2 test and every external test remained sealed.",
        "",
        "## Five-seed validation summary",
        "",
        "| Metric | Hard-decoy | Control | Paired delta (95% seed bootstrap CI) |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "contact_auprc",
        "contact_macro_auprc",
        "contact_auroc",
        "contact_f1",
        "contact_mcc",
        "contact_brier",
        "contact_ece_15",
        "pair_mean_recall_10",
        "pair_mean_recall_lmin",
        "partner_ab_to_ag_recall_at_1",
        "partner_ab_to_ag_mrr",
        "partner_ag_to_ab_recall_at_1",
        "partner_ag_to_ab_mrr",
        "epitope_auprc",
        "paratope_auprc",
    ):
        ci = delta[name]["paired_bootstrap"]
        lines.append(
            f"| `{name}` | {format_metric(candidate, name)} | {format_metric(control, name)} | "
            f"{ci['estimate']:.6f} [{ci['ci95_lower']:.6f}, {ci['ci95_upper']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Scope and missing outputs",
            "",
            "- Pair-contact AUPRC/AUROC, fixed-K ranked recall, Brier and ECE are native pair-head metrics.",
            "- Partner Recall@1/MRR are antibody-antigen retrieval metrics; they are not pair-contact Recall@1/MRR.",
            "- Pair-contact Recall@1/MRR and full per-pair frozen logits were not emitted by these training runs and remain explicit follow-up work.",
            "- This is development-validation evidence on the fixed double-cold antigen-fold split, not a frozen-test or SOTA claim.",
            "",
            "## Provenance",
            "",
            f"- Candidate: `{payload['candidate']['root']}`",
            f"- Control: `{payload['control']['root']}`",
            f"- Split receipt SHA256: `{payload['split_receipt']['sha256']}`",
            f"- Superseded slow job: `{payload['superseded_job_id']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--split-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--superseded-job-id", default="none")
    arguments = parser.parse_args()

    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {arguments.output_root}")
    candidate = [load_member(arguments.candidate_root, seed) for seed in SEEDS]
    control = [load_member(arguments.control_root, seed) for seed in SEEDS]
    candidate_authority = consistent_authority(candidate)
    control_authority = consistent_authority(control)
    split_receipt = read_json(arguments.split_receipt)
    antigen_fold_audit = nested(split_receipt, "audits", "antigen_fold")
    if antigen_fold_audit.get("valid") is not True:
        raise ValueError("Antigen-fold split receipt is not valid")
    for key in ("antibody_sequence_sha256", "antigen_sequence_sha256", "pdb_assembly_id"):
        if nested(antigen_fold_audit, "cross_role_overlap_counts", key) != 0:
            raise ValueError(f"Non-zero split overlap: {key}")
    expected_split_sha256 = candidate_authority["split_manifest_sha256"]
    if control_authority["split_manifest_sha256"] != expected_split_sha256:
        raise ValueError("Candidate and control use different split manifests")
    split_manifest = arguments.split_receipt.parent / "antigen_fold.csv"
    if sha256_file(split_manifest) != expected_split_sha256:
        raise ValueError("Split manifest SHA256 does not match run authority")

    payload = {
        "format_version": "framecontact_local_decoy_gain_aggregate_v1",
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "claim_boundary": (
            "SAbDab2 double-cold antigen-fold development validation only; test and external "
            "roles remained sealed; no benchmark-improvement claim is authorized."
        ),
        "seeds": list(SEEDS),
        "candidate": {
            "root": str(arguments.candidate_root),
            "authority": candidate_authority,
            "members": candidate,
            "aggregate_metrics": aggregate_metrics(candidate),
        },
        "control": {
            "root": str(arguments.control_root),
            "authority": control_authority,
            "members": control,
            "aggregate_metrics": aggregate_metrics(control),
        },
        "paired_deltas": paired_deltas(candidate, control),
        "split_receipt": {
            "path": str(arguments.split_receipt),
            "sha256": sha256_file(arguments.split_receipt),
            "antigen_fold_audit": antigen_fold_audit,
        },
        "superseded_job_id": arguments.superseded_job_id,
        "metric_boundaries": {
            "native_pair_metrics_present": [
                "AUPRC",
                "AUROC",
                "Brier",
                "ECE-15",
                "precision/recall at 10, 20, 50, Lmin/10, Lmin/5, Lmin",
            ],
            "partner_retrieval_metrics_present": ["Recall@1", "MRR"],
            "missing": [
                "pair-contact Recall@1",
                "pair-contact MRR",
                "full per-pair frozen logits for post-hoc re-evaluation",
            ],
        },
    }

    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.tmp-", dir=arguments.output_root.parent
        )
    )
    try:
        json_path = temporary / "FRAMECONTACT_LOCAL_DECOY_GAIN_AGGREGATE.json"
        report_path = temporary / "FRAMECONTACT_LOCAL_DECOY_GAIN_REPORT.md"
        atomic_write_json(json_path, payload)
        report_path.write_text(render_markdown(payload), encoding="utf-8")
        manifest = {
            "format_version": "framecontact_local_decoy_gain_aggregate_receipt_v1",
            "aggregate_sha256": sha256_file(json_path),
            "report_sha256": sha256_file(report_path),
            "seed_count": len(SEEDS),
            "verified_result_count": len(SEEDS) * 2,
            "verified_checkpoint_count": len(SEEDS) * 2,
        }
        atomic_write_json(temporary / "RUN_RECEIPT.json", manifest)
        temporary.replace(arguments.output_root)
    except Exception:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise
    print(arguments.output_root)


if __name__ == "__main__":
    main()
