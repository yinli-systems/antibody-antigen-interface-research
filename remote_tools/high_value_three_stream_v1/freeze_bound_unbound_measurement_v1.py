#!/usr/bin/env python3
"""Freeze the bound/unbound measurement after its label-free eligibility gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ARMS = ("baseline_current", "edge_topk2_supervised")
SEEDS = (17, 29, 43)


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


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-free-manifest", type=Path, required=True)
    parser.add_argument("--label-free-manifest-sha256", required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol-sha256", required=True)
    parser.add_argument("--full-analysis", type=Path, required=True)
    parser.add_argument("--full-analysis-sha256", required=True)
    parser.add_argument("--baseline-source-root", type=Path, required=True)
    parser.add_argument("--baseline-source-manifest", type=Path, required=True)
    parser.add_argument("--baseline-source-manifest-sha256", required=True)
    parser.add_argument("--expert-source-root", type=Path, required=True)
    parser.add_argument("--expert-source-manifest", type=Path, required=True)
    parser.add_argument("--expert-source-manifest-sha256", required=True)
    parser.add_argument("--esm2-model-root", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    for path, expected, label in (
        (args.label_free_manifest, args.label_free_manifest_sha256, "label-free manifest"),
        (args.parent_protocol, args.parent_protocol_sha256, "parent protocol"),
        (args.full_analysis, args.full_analysis_sha256, "full-development analysis"),
        (args.baseline_source_manifest, args.baseline_source_manifest_sha256, "baseline source manifest"),
        (args.expert_source_manifest, args.expert_source_manifest_sha256, "expert source manifest"),
    ):
        require_sha(path, expected, label)
    label_free = json.loads(args.label_free_manifest.read_text())
    analysis = json.loads(args.full_analysis.read_text())
    if (
        label_free.get("status") != "PASS_BOUND_UNBOUND_LABEL_FREE_FREEZE"
        or label_free.get("counts", {}).get("eligible_cases") != 52
        or label_free.get("audit", {}).get("contact_labels_computed") is not False
        or label_free.get("audit", {}).get("cross_chain_distances_computed") is not False
        or analysis.get("status") != "PASS_CURRENT_BEST_FULL_DEVELOPMENT_TASK_ROUTING"
        or analysis.get("old_test_loaded") is not False
        or analysis.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("pre-measurement gate differs")

    source = {
        "baseline_current": {
            "root": args.baseline_source_root,
            "manifest": args.baseline_source_manifest,
            "manifest_sha256": args.baseline_source_manifest_sha256,
        },
        "edge_topk2_supervised": {
            "root": args.expert_source_root,
            "manifest": args.expert_source_manifest,
            "manifest_sha256": args.expert_source_manifest_sha256,
        },
    }
    bindings: dict[str, dict[str, object]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            run = analysis["run_bindings"][arm][str(seed)]
            run_root = Path(run["run_root"])
            checkpoint = run_root / "FRAMECONTACT_SELECTED.pt"
            result = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
            receipt = run_root / "RUN_RECEIPT.json"
            for path, expected, label in (
                (checkpoint, run["checkpoint_sha256"], "checkpoint"),
                (result, run["result_sha256"], "development result"),
                (receipt, run["receipt_sha256"], "development receipt"),
            ):
                require_sha(path, expected, f"{arm}/seed-{seed} {label}")
            development = json.loads(result.read_text())
            if (
                development.get("configuration", {}).get("seed") != seed
                or development.get("authority", {}).get("test_loaded") is not False
            ):
                raise ValueError(f"development authority differs: {arm}/seed-{seed}")
            bindings[arm][str(seed)] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": run["checkpoint_sha256"],
                "result": str(result),
                "result_sha256": run["result_sha256"],
                "receipt": str(receipt),
                "receipt_sha256": run["receipt_sha256"],
                "selected_epoch": run["selected_epoch"],
                "source_root": str(source[arm]["root"]),
                "source_manifest": str(source[arm]["manifest"]),
                "source_manifest_sha256": source[arm]["manifest_sha256"],
                "framecontact_sha256": sha256_file(source[arm]["root"] / "src/interaction_first/framecontact.py"),
                "trainer_sha256": sha256_file(source[arm]["root"] / "scripts/train_framecontact.py"),
            }

    model_weight = args.esm2_model_root / "model.safetensors"
    expected_weight_sha = "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"
    require_sha(model_weight, expected_weight_sha, "ESM-2 model weight")
    protocol = {
        "format_version": "framecontact_bound_unbound_measurement_protocol_v1",
        "status": "FROZEN_AFTER_LABEL_FREE_GATE_BEFORE_CONTACTS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_label_free_authority": {
            "protocol": str(args.parent_protocol),
            "protocol_sha256": args.parent_protocol_sha256,
            "manifest": str(args.label_free_manifest),
            "manifest_sha256": args.label_free_manifest_sha256,
            "eligible_case_count": 52,
            "eligibility_may_not_be_relaxed_after_freeze": True,
        },
        "dataset": {
            "repository": label_free["source"]["repository"],
            "commit": label_free["source"]["commit"],
            "archive_sha256": label_free["source"]["archive_sha256"],
            "subsets": label_free["selection"]["subsets"],
            "license_file_present_at_pinned_commit": False,
            "redistribution_authorized": False,
        },
        "common_residue_axis": {
            "selection": "exact same amino-acid identity in the frozen bound-to-unbound alignment and finite N/CA/C coordinates in both conformations",
            "same_axis_across_all_four_conditions": True,
            "no_post_label_exclusion": True,
        },
        "contact_labels": {
            "source": "bound complex coordinates only",
            "definition": "minimum inter-residue heavy-atom distance <= 4.5 angstrom",
            "hydrogens_excluded": True,
            "candidate_grid": "complete Cartesian product of retained antibody and antigen axes exactly once",
            "same_labels_across_all_four_conditions": True,
            "zero-positive_case_policy": "pooled metrics retain all cases; mean-per-complex AP excludes zero-positive cases and reports the denominator",
        },
        "geometry_conditions": [
            "bound_bound",
            "unbound_ab_bound_ag",
            "bound_ab_unbound_ag",
            "unbound_unbound",
        ],
        "embedding": {
            "model_id": "facebook/esm2_t33_650M_UR50D",
            "revision": "08e4846e537177426273712802403f7ba8261b6c",
            "model_root": str(args.esm2_model_root),
            "model_weight_sha256": expected_weight_sha,
            "embedding_dimension": 1280,
            "computed_once_per_common_sequence_and_reused_across_conditions": True,
        },
        "full_development_authority": {
            "analysis": str(args.full_analysis),
            "analysis_sha256": args.full_analysis_sha256,
            "validation_was_checkpoint_selection_data": True,
            "no_retraining_for_this_robustness_measurement": True,
        },
        "model_bindings": bindings,
        "task_routing": {
            "contact": "baseline_current",
            "paratope": "baseline_current",
            "epitope": "edge_topk2_supervised",
            "seeds": list(SEEDS),
            "mixture_or_route_tuning": False,
        },
        "measurement": {
            "primary_metric": "mean per eligible complex non-interpolated average precision",
            "paired_reference": "bound_bound",
            "pooled_metrics": ["average_precision", "auroc", "brier", "ece_15", "precision", "recall", "f1", "mcc"],
            "bootstrap": {
                "unit": "eligible case",
                "replicates": 10000,
                "seed": 20260828,
                "seed_handling": "mean paired per-case delta over fixed seeds before resampling cases",
                "conditional_on_fixed_trained_seeds": True,
            },
            "calibration_and_thresholds": "reuse each development-selected checkpoint's frozen validation temperature and threshold without refitting",
        },
        "execution": {
            "partition": "gpu_4090",
            "python": "/ssd/scxi253/model_deployment/envs/contact/bin/python3.10",
            "materializer_sha256": sha256_file(args.materializer),
            "evaluator_sha256": sha256_file(args.evaluator),
            "analyzer_sha256": sha256_file(args.analyzer),
        },
        "claim_boundary": {
            "robustness_measurement_only": True,
            "not_model_selection": True,
            "not_a_blind_test": True,
            "old_exposed_sabdab2_test_forbidden": True,
            "temporal_holdout_forbidden": True,
            "cross_chain_coordinates_are_never_model_inputs": True,
            "bound_state_intra_chain_geometry_is_the_bound_reference_input": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(protocol))
    receipt = {
        "format_version": "framecontact_bound_unbound_measurement_protocol_receipt_v1",
        "status": protocol["status"],
        "protocol_sha256": sha256_file(args.output),
        "label_free_manifest_sha256": args.label_free_manifest_sha256,
        "frozen_before_contact_label_materialization": True,
    }
    args.receipt.write_bytes(canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
