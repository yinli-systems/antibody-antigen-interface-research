#!/usr/bin/env python3
"""Freeze the once-only temporal measurement after the label-free cold gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


ARMS = ("baseline_current", "edge_topk2_supervised")
SEEDS = (17, 29, 43)
EXPECTED_ESM2_WEIGHT_SHA256 = "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"


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


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-free-manifest", type=Path, required=True)
    parser.add_argument("--label-free-manifest-sha256", required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol-sha256", required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--execution-plan-sha256", required=True)
    parser.add_argument("--full-analysis", type=Path, required=True)
    parser.add_argument("--full-analysis-sha256", required=True)
    parser.add_argument("--baseline-source-root", type=Path, required=True)
    parser.add_argument("--baseline-source-manifest", type=Path, required=True)
    parser.add_argument("--baseline-source-manifest-sha256", required=True)
    parser.add_argument("--expert-source-root", type=Path, required=True)
    parser.add_argument("--expert-source-manifest", type=Path, required=True)
    parser.add_argument("--expert-source-manifest-sha256", required=True)
    parser.add_argument("--esm2-model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    for path, expected, label in (
        (args.label_free_manifest, args.label_free_manifest_sha256, "label-free manifest"),
        (args.parent_protocol, args.parent_protocol_sha256, "parent protocol"),
        (args.execution_plan, args.execution_plan_sha256, "execution plan"),
        (args.full_analysis, args.full_analysis_sha256, "full-development analysis"),
        (args.baseline_source_manifest, args.baseline_source_manifest_sha256, "baseline source manifest"),
        (args.expert_source_manifest, args.expert_source_manifest_sha256, "expert source manifest"),
    ):
        require_sha(path, expected, label)
    label_free = json.loads(args.label_free_manifest.read_text())
    parent = json.loads(args.parent_protocol.read_text())
    plan = json.loads(args.execution_plan.read_text())
    analysis = json.loads(args.full_analysis.read_text())
    counts = label_free.get("counts", {})
    if (
        label_free.get("status") != "PASS_LABEL_FREE_DOUBLE_COLD_FREEZE"
        or label_free.get("protocol_sha256") != args.parent_protocol_sha256
        or label_free.get("audit", {}).get("task_labels_read") is not False
        or label_free.get("audit", {}).get("cross_chain_distances_computed") is not False
        or label_free.get("audit", {}).get("contact_labels_computed") is not False
        or label_free.get("audit", {}).get("old_test_accessed") is not False
        or label_free.get("audit", {}).get("selection_performed") is not False
        or parent.get("status") != "FROZEN_BEFORE_STRUCTURE_OR_LABEL_ACCESS"
        or plan.get("status") != "FROZEN_BEFORE_LABEL_FREE_SCREEN_COMPLETION"
        or plan.get("parent_protocol_sha256") != args.parent_protocol_sha256
        or analysis.get("status") != "PASS_CURRENT_BEST_FULL_DEVELOPMENT_TASK_ROUTING"
        or analysis.get("old_test_loaded") is not False
        or analysis.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("pre-opening authority differs")
    minimum_instances = int(plan["gate"]["minimum_double_cold_instances"])
    minimum_pdbs = int(plan["gate"]["minimum_double_cold_unique_pdbs"])
    if (
        int(counts.get("double_cold_instances", -1)) < minimum_instances
        or int(counts.get("double_cold_unique_pdbs", -1)) < minimum_pdbs
    ):
        raise RuntimeError("FAIL_CLOSED_INSUFFICIENT_LABEL_FREE_DOUBLE_COLD_COHORT")
    selected = set(label_free["double_cold_instance_ids"])
    selected_rows = [
        row for row in label_free["instances"] if row["instance_id"] in selected
    ]
    if (
        len(selected_rows) != int(counts["double_cold_instances"])
        or len({row["pdb_id"] for row in selected_rows})
        != int(counts["double_cold_unique_pdbs"])
        or any(not row.get("double_cold") for row in selected_rows)
    ):
        raise ValueError("selected double-cold rows differ")

    sources = {
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
                (result, run["result_sha256"], "result"),
                (receipt, run["receipt_sha256"], "receipt"),
            ):
                require_sha(path, expected, f"{arm}/seed-{seed} {label}")
            development = json.loads(result.read_text())
            if (
                development.get("configuration", {}).get("seed") != seed
                or development.get("authority", {}).get("test_loaded") is not False
            ):
                raise ValueError(f"development authority differs: {arm}/seed-{seed}")
            source = sources[arm]
            bindings[arm][str(seed)] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": run["checkpoint_sha256"],
                "result": str(result),
                "result_sha256": run["result_sha256"],
                "receipt": str(receipt),
                "receipt_sha256": run["receipt_sha256"],
                "selected_epoch": run["selected_epoch"],
                "source_root": str(source["root"]),
                "source_manifest": str(source["manifest"]),
                "source_manifest_sha256": source["manifest_sha256"],
                "framecontact_sha256": sha256_file(source["root"] / "src/interaction_first/framecontact.py"),
                "trainer_sha256": sha256_file(source["root"] / "scripts/train_framecontact.py"),
            }
    require_sha(
        args.esm2_model_root / "model.safetensors",
        EXPECTED_ESM2_WEIGHT_SHA256,
        "ESM-2 weight",
    )
    protocol = {
        "format_version": "framecontact_temporal_double_cold_measurement_protocol_v1",
        "status": "FROZEN_AFTER_LABEL_FREE_GATE_BEFORE_SINGLE_LABEL_OPENING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_protocol_sha256": args.parent_protocol_sha256,
        "execution_plan_sha256": args.execution_plan_sha256,
        "label_free_manifest": str(args.label_free_manifest),
        "label_free_manifest_sha256": args.label_free_manifest_sha256,
        "cohort": {
            "instances": len(selected_rows),
            "unique_pdbs": len({row["pdb_id"] for row in selected_rows}),
            "instance_ids": sorted(selected),
            "strict_deposition_after": parent["identity_cohort"]["strict_deposition_after"],
            "antibody_and_antigen_sequence_double_cold": True,
            "gate_thresholds_relaxed": False,
        },
        "contact_definition": plan["contact_definition"],
        "embedding": {
            "model_id": "facebook/esm2_t33_650M_UR50D",
            "revision": "08e4846e537177426273712802403f7ba8261b6c",
            "model_root": str(args.esm2_model_root),
            "model_weight_sha256": EXPECTED_ESM2_WEIGHT_SHA256,
            "embedding_dimension": 1280,
        },
        "full_development_authority": {
            "analysis": str(args.full_analysis),
            "analysis_sha256": args.full_analysis_sha256,
            "validation_was_checkpoint_selection_data": True,
            "no_training_on_temporal_data": True,
        },
        "model_bindings": bindings,
        "task_routing": plan["measurement"]["task_route"],
        "measurement": plan["measurement"],
        "execution": plan["execution"],
        "opening": {
            "opening_count_before_materialization": 0,
            "opening_count_after_materialization": 1,
            "maximum": 1,
            "selection_performed_on_temporal": False,
            "result_feedback_into_development": False,
        },
        "claim_boundary": {
            "prospective_relative_to_training_snapshot": True,
            "single_opening": True,
            "independent_of_checkpoint_selection": True,
            "old_exposed_test_access": False,
            "not_an_official_sabdab2_split": True,
            "bound_complex_derived_intra_chain_backbone_geometry": True,
            "cross_chain_coordinates_are_never_model_inputs": True,
        },
    }
    atomic_write(args.output, canonical_bytes(protocol))
    receipt = {
        "format_version": "framecontact_temporal_double_cold_measurement_protocol_receipt_v1",
        "status": protocol["status"],
        "protocol_sha256": sha256_file(args.output),
        "label_free_manifest_sha256": args.label_free_manifest_sha256,
        "frozen_before_single_label_opening": True,
        "opening_count": 0,
        "selection_performed": False,
    }
    atomic_write(args.receipt, canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.receipt)}, sort_keys=True))


if __name__ == "__main__":
    main()
