#!/usr/bin/env python3
"""Score the preregistered CHIMERA external subset without tuning.

The scorer refuses to upgrade the result to a confirmatory claim when either
co-primary split misses the frozen 95% adapter-coverage gate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from interaction_first.chimera_cache import load_cache_manifest  # noqa: E402
from interaction_first.chimera_external import verify_frozen_authority  # noqa: E402
from interaction_first.sabdab2_structure import sha256_file  # noqa: E402


CANDIDATES = ("sequence_dense", "spatial_topology")
SEEDS = (17, 42, 73)
PRIMARY_SPLITS = ("temporal.test", "antigen_fold.test")
PARTNERS = {
    "epitope": ("antigen_offsets", "antigen_labels", "epitope_probabilities"),
    "paratope": (
        "antibody_offsets",
        "antibody_labels",
        "paratope_probabilities",
    ),
}


def _load_predictions(root: Path, manifest: dict) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    records = manifest.get("predictions")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("prediction manifest must contain exactly six runs")
    output: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    required = {
        "instance_ids",
        "antibody_offsets",
        "antigen_offsets",
        "antibody_labels",
        "antigen_labels",
        "paratope_probabilities",
        "epitope_probabilities",
    }
    reference: tuple[np.ndarray, ...] | None = None
    for record in records:
        key = (record.get("candidate"), int(record.get("seed", -1)))
        if key in output or key[0] not in CANDIDATES or key[1] not in SEEDS:
            raise ValueError("prediction run identity differs")
        path = root / str(record.get("prediction_path", ""))
        if path.is_symlink() or not path.is_file():
            raise ValueError("prediction artifact is missing or unsafe")
        if sha256_file(path) != record.get("prediction_sha256"):
            raise ValueError("prediction artifact SHA-256 differs")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise ValueError("prediction artifact schema differs")
            arrays = {name: archive[name].copy() for name in archive.files}
        instance_ids = arrays["instance_ids"]
        if instance_ids.ndim != 1 or len(set(instance_ids.tolist())) != len(instance_ids):
            raise ValueError("prediction instance IDs are invalid")
        for partner, (offset_key, label_key, probability_key) in PARTNERS.items():
            offsets = arrays[offset_key]
            labels = arrays[label_key]
            probabilities = arrays[probability_key]
            if (
                offsets.dtype != np.int64
                or offsets.shape != (len(instance_ids) + 1,)
                or offsets[0] != 0
                or np.any(np.diff(offsets) <= 0)
                or offsets[-1] != len(labels)
                or labels.shape != probabilities.shape
                or not np.isin(labels, (0.0, 1.0)).all()
                or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0.0) | (probabilities > 1.0))
            ):
                raise ValueError(f"invalid {partner} prediction axis")
        authority = (
            arrays["instance_ids"],
            arrays["antibody_offsets"],
            arrays["antigen_offsets"],
            arrays["antibody_labels"],
            arrays["antigen_labels"],
        )
        if reference is None:
            reference = tuple(value.copy() for value in authority)
        elif any(not np.array_equal(left, right) for left, right in zip(reference, authority)):
            raise ValueError("labels or ordering differ between prediction runs")
        output[key] = arrays
    if set(output) != {(candidate, seed) for candidate in CANDIDATES for seed in SEEDS}:
        raise ValueError("prediction run matrix is incomplete")
    return output


def _ensemble(
    predictions: dict[tuple[str, int], dict[str, np.ndarray]],
    candidate: str,
    probability_key: str,
) -> np.ndarray:
    return np.mean(
        np.stack([predictions[(candidate, seed)][probability_key] for seed in SEEDS]),
        axis=0,
        dtype=np.float64,
    )


def _safe_auprc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if not np.any(labels == 1.0):
        return 0.0
    return float(average_precision_score(labels, probabilities))


def _safe_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(labels)) != 2:
        return None
    return float(roc_auc_score(labels, probabilities))


def _threshold_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    binary = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels.astype(np.int64), binary, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "threshold": 0.5,
        "precision": float(precision_score(labels, binary, zero_division=0)),
        "recall": float(recall_score(labels, binary, zero_division=0)),
        "f1": float(f1_score(labels, binary, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, binary)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _absolute_metrics(
    labels_by_complex: list[np.ndarray], probabilities_by_complex: list[np.ndarray]
) -> dict[str, Any]:
    auprc = np.asarray(
        [_safe_auprc(labels, probabilities) for labels, probabilities in zip(labels_by_complex, probabilities_by_complex)],
        dtype=np.float64,
    )
    auroc_values = [
        _safe_auroc(labels, probabilities)
        for labels, probabilities in zip(labels_by_complex, probabilities_by_complex)
    ]
    finite_auroc = np.asarray([value for value in auroc_values if value is not None])
    pooled_labels = np.concatenate(labels_by_complex)
    pooled_probabilities = np.concatenate(probabilities_by_complex)
    pooled_auroc = _safe_auroc(pooled_labels, pooled_probabilities)
    prevalence = float(pooled_labels.mean())
    return {
        "complex_count": len(labels_by_complex),
        "residue_count": int(len(pooled_labels)),
        "positive_count": int(pooled_labels.sum()),
        "prevalence": prevalence,
        "mean_per_complex_auprc": float(auprc.mean()),
        "mean_per_complex_auroc": (
            float(finite_auroc.mean()) if len(finite_auroc) else None
        ),
        "per_complex_auroc_defined_count": int(len(finite_auroc)),
        "pooled_auprc": _safe_auprc(pooled_labels, pooled_probabilities),
        "pooled_auroc": pooled_auroc,
        "pooled_auprc_minus_prevalence": (
            _safe_auprc(pooled_labels, pooled_probabilities) - prevalence
        ),
        "fixed_threshold": _threshold_metrics(pooled_labels, pooled_probabilities),
    }


def _group_bootstrap(
    delta: np.ndarray,
    groups: list[str],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    group_names = sorted(members)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(group_names), size=len(group_names))
        indices = [index for group_index in sampled for index in members[group_names[group_index]]]
        bootstrap[replicate] = float(delta[indices].mean())
    tail_low = (int(np.count_nonzero(bootstrap <= 0.0)) + 1) / (replicates + 1)
    tail_high = (int(np.count_nonzero(bootstrap >= 0.0)) + 1) / (replicates + 1)
    p_value = min(1.0, 2.0 * min(tail_low, tail_high))
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return (
        {
            "group_count": len(group_names),
            "replicates": replicates,
            "seed": seed,
            "confidence": 0.95,
            "mean_difference": float(delta.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
            "two_sided_bootstrap_tail_p": float(p_value),
            "p_value_definition": (
                "2 * min((count(bootstrap <= 0)+1)/(R+1), "
                "(count(bootstrap >= 0)+1)/(R+1)), capped at 1"
            ),
        },
        bootstrap,
    )


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CHIMERA strict external subset result",
        "",
        result["claim_boundary"],
        "",
        "| Split | Coverage | Candidate | Epitope mean complex AUPRC | Epitope pooled AUPRC | Paratope mean complex AUPRC |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for split_name in PRIMARY_SPLITS:
        split = result["splits"][split_name]
        for candidate in CANDIDATES:
            metrics = split["absolute_metrics"][candidate]
            lines.append(
                f"| {split_name} | {split['usable']}/{split['eligible']} ({split['coverage']:.3%}) "
                f"| {candidate} | {metrics['epitope']['mean_per_complex_auprc']:.6f} "
                f"| {metrics['epitope']['pooled_auprc']:.6f} "
                f"| {metrics['paratope']['mean_per_complex_auprc']:.6f} |"
            )
        primary = split["primary_endpoint"]
        lines.extend(
            [
                "",
                f"- `{split_name}` primary delta (spatial - dense): "
                f"{primary['mean_difference']:+.6f}, 95% group-bootstrap CI "
                f"[{primary['ci_low']:+.6f}, {primary['ci_high']:+.6f}], "
                f"Holm p={primary['holm_adjusted_p']:.6g}.",
                f"- Confirmatory support: **{str(primary['confirmatory_support']).lower()}** "
                f"(coverage gate passed: {str(split['coverage_gate_passed']).lower()}).",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--adapter-audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-cache-manifest-sha256", required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--expected-prediction-manifest-sha256")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise SystemExit("output-root already exists")
    protocol = verify_frozen_authority(args.protocol, args.freeze_receipt)
    if tuple(protocol["evaluation"]["primary_splits"]) != PRIMARY_SPLITS:
        raise ValueError("primary split contract differs")
    bootstrap_contract = protocol["evaluation"]["bootstrap"]
    if int(bootstrap_contract["replicates"]) != 10_000 or int(bootstrap_contract["seed"]) != 20260730:
        raise ValueError("bootstrap contract differs")
    cache_manifest = load_cache_manifest(
        args.cache_root, expected_sha256=args.expected_cache_manifest_sha256
    )
    audit = json.loads(args.adapter_audit.read_text(encoding="utf-8"))
    supported_audit_schemas = {
        "chimera_external_adapter_audit_v1",
        "chimera_pdb_anchored_coverage_audit_v3",
    }
    if (
        audit.get("schema_version") not in supported_audit_schemas
        or audit.get("protocol_sha256") != sha256_file(args.protocol)
        or audit.get("unique_usable_complexes") != cache_manifest["record_count"]
        or cache_manifest.get("adapter_audit_sha256")
        != sha256_file(args.adapter_audit)
        or audit.get("model_scores_read") is not False
        or audit.get("metrics_computed") is not False
    ):
        raise ValueError("adapter audit authority differs")
    audit_confirmatory_interpretation_permitted = bool(
        audit.get(
            "confirmatory_interpretation_permitted",
            audit.get("schema_version") == "chimera_external_adapter_audit_v1",
        )
    )
    prediction_manifest_path = args.prediction_root / "PREDICTION_MANIFEST.json"
    prediction_manifest_sha = sha256_file(prediction_manifest_path)
    if (
        args.expected_prediction_manifest_sha256 is not None
        and prediction_manifest_sha != args.expected_prediction_manifest_sha256
    ):
        raise ValueError("prediction manifest SHA-256 differs")
    prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    if (
        prediction_manifest.get("schema_version") != "chimera_frozen_external_predictions_v1"
        or prediction_manifest.get("protocol_sha256") != sha256_file(args.protocol)
        or prediction_manifest.get("cache_manifest_sha256") != args.expected_cache_manifest_sha256
        or prediction_manifest.get("metrics_computed") is not False
    ):
        raise ValueError("prediction manifest authority differs")
    predictions = _load_predictions(args.prediction_root, prediction_manifest)
    reference = predictions[(CANDIDATES[0], SEEDS[0])]
    identifier_to_index = {
        str(identifier): index for index, identifier in enumerate(reference["instance_ids"])
    }
    if set(identifier_to_index) != {record["instance_id"] for record in cache_manifest["records"]}:
        raise ValueError("prediction and adapter-cache instance sets differ")

    antigen_groups: dict[str, str] = {}
    membership: dict[str, set[str]] = {}
    for record in cache_manifest["records"]:
        identifier = record["instance_id"]
        groups = record.get("split_memberships")
        if not isinstance(groups, list) or not groups or any(group not in PRIMARY_SPLITS for group in groups):
            raise ValueError("adapter-cache split memberships are absent or invalid")
        membership[identifier] = set(groups)
        artifact_path = args.cache_root / record["artifact_path"]
        if sha256_file(artifact_path) != record["artifact_sha256"]:
            raise ValueError("adapter artifact SHA-256 differs during grouping")
        with np.load(artifact_path, allow_pickle=False) as archive:
            antigen_sequence = str(archive["antigen_sequence"].item()).strip().upper()
        antigen_groups[identifier] = hashlib.sha256(antigen_sequence.encode("ascii")).hexdigest()

    split_outputs: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for split_name in PRIMARY_SPLITS:
        identifiers = [
            str(identifier)
            for identifier in reference["instance_ids"]
            if split_name in membership[str(identifier)]
        ]
        indices = [identifier_to_index[identifier] for identifier in identifiers]
        audit_split = audit["splits"][split_name]
        if len(indices) != int(audit_split["usable"]):
            raise ValueError("split usable count differs from adapter audit")
        absolute: dict[str, Any] = {}
        primary_candidate_scores: dict[str, np.ndarray] = {}
        for candidate in CANDIDATES:
            candidate_metrics: dict[str, Any] = {}
            for partner, (offset_key, label_key, probability_key) in PARTNERS.items():
                offsets = reference[offset_key]
                labels = reference[label_key]
                probabilities = _ensemble(predictions, candidate, probability_key)
                local_labels = [labels[offsets[index] : offsets[index + 1]] for index in indices]
                local_probabilities = [
                    probabilities[offsets[index] : offsets[index + 1]] for index in indices
                ]
                candidate_metrics[partner] = _absolute_metrics(local_labels, local_probabilities)
                if partner == "epitope":
                    primary_candidate_scores[candidate] = np.asarray(
                        [
                            _safe_auprc(local_label, local_probability)
                            for local_label, local_probability in zip(local_labels, local_probabilities)
                        ],
                        dtype=np.float64,
                    )
            absolute[candidate] = candidate_metrics
        delta = primary_candidate_scores["spatial_topology"] - primary_candidate_scores["sequence_dense"]
        interval, _ = _group_bootstrap(
            delta,
            [antigen_groups[identifier] for identifier in identifiers],
            replicates=int(bootstrap_contract["replicates"]),
            seed=int(bootstrap_contract["seed"]),
        )
        p_values[split_name] = interval["two_sided_bootstrap_tail_p"]
        coverage = float(audit_split["coverage"])
        split_outputs[split_name] = {
            "eligible": int(audit_split["eligible"]),
            "usable": len(indices),
            "excluded": int(audit_split["excluded"]),
            "coverage": coverage,
            "coverage_gate_threshold": 0.95,
            "coverage_gate_passed": bool(coverage >= 0.95),
            "absolute_metrics": absolute,
            "primary_endpoint": interval,
        }
    adjusted = _holm(p_values)
    for split_name, split in split_outputs.items():
        endpoint = split["primary_endpoint"]
        endpoint["holm_adjusted_p"] = adjusted[split_name]
        endpoint["statistical_success_before_coverage_gate"] = bool(
            endpoint["mean_difference"] > 0.0
            and endpoint["ci_low"] > 0.0
            and adjusted[split_name] < 0.05
        )
        endpoint["confirmatory_support"] = bool(
            audit_confirmatory_interpretation_permitted
            and split["coverage_gate_passed"]
            and endpoint["statistical_success_before_coverage_gate"]
        )

    result = {
        "schema_version": "chimera_strict_external_subset_result_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            (
                "Real zero-shot external residue-interface scoring on the pre-access "
                "strict mapped CHIMERA subset. Confirmatory support additionally "
                "requires every preregistered statistical and coverage gate."
                if audit_confirmatory_interpretation_permitted
                else (
                    "Real zero-shot external residue-interface scoring on a post-access "
                    "adapter expansion. This is secondary robustness evidence only, "
                    "regardless of coverage or statistical significance."
                )
            )
            + " This is not the canonical CHIMERA CDR-design benchmark."
        ),
        "adapter_confirmatory_interpretation_permitted": (
            audit_confirmatory_interpretation_permitted
        ),
        "confirmatory_claim_allowed": bool(
            audit_confirmatory_interpretation_permitted
            and all(split["coverage_gate_passed"] for split in split_outputs.values())
        ),
        "digests": {
            "protocol_sha256": sha256_file(args.protocol),
            "freeze_receipt_sha256": sha256_file(args.freeze_receipt),
            "adapter_audit_sha256": sha256_file(args.adapter_audit),
            "cache_manifest_sha256": args.expected_cache_manifest_sha256,
            "prediction_manifest_sha256": prediction_manifest_sha,
            "scorer_source_sha256": sha256_file(Path(__file__)),
        },
        "seed_ensemble": list(SEEDS),
        "comparison": "spatial_topology minus sequence_dense",
        "primary_endpoint": "mean per-complex epitope AUPRC difference",
        "multiplicity": "Holm across two co-primary split-level hypotheses",
        "group_key": "sha256 of normalized antigen sequence",
        "splits": split_outputs,
    }
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.tmp-", dir=args.output_root.parent)
    )
    try:
        json_path = temporary_root / "CHIMERA_STRICT_EXTERNAL_RESULT.json"
        md_path = temporary_root / "CHIMERA_STRICT_EXTERNAL_RESULT.md"
        json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        md_path.write_text(_markdown(result))
        receipt = {
            "result_json_sha256": sha256_file(json_path),
            "result_markdown_sha256": sha256_file(md_path),
        }
        (temporary_root / "RESULT_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        os.chmod(temporary_root, 0o755)
        os.replace(temporary_root, args.output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(f"result={args.output_root / 'CHIMERA_STRICT_EXTERNAL_RESULT.json'}")
    print(f"result_sha256={sha256_file(args.output_root / 'CHIMERA_STRICT_EXTERNAL_RESULT.json')}")


if __name__ == "__main__":
    main()
