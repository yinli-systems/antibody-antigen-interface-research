#!/usr/bin/env python3
"""Bind six fixed ABF-30B checkpoints before the exposed test label opening."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SEEDS = (17, 29, 43)
FINAL_ANALYSIS_SHA256 = "c83a025c0f895ceecf61d0a079095dd182f3a8610cc419b1482604176af3b68d"
CONTACT_MANIFEST_SHA256 = "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33"
TEST_PANEL_SHA256 = "ea70fef770448302fb4ed1ccd7da51a1cc56cba5ea153076e077971735874bd1"
SPLIT_SHA256 = "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def require(path: Path, expected: str | None = None) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required file missing or symlinked: {path}")
    observed = sha256_file(path)
    if expected is not None and observed != expected:
        raise ValueError(f"authority SHA differs: {path}")
    return observed


def calibration(result: dict[str, Any]) -> dict[str, float]:
    selected = result["selected_checkpoint_validation"]
    return {
        "temperature": float(selected["validation_temperature"]),
        "contact_threshold": float(selected["contact"]["threshold_selected_on_validation"]),
        "epitope_temperature": float(selected["epitope"]["validation_temperature"]),
        "epitope_threshold": float(selected["epitope"]["threshold_selected_on_validation"]),
        "paratope_temperature": float(selected["paratope"]["validation_temperature"]),
        "paratope_threshold": float(selected["paratope"]["threshold_selected_on_validation"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--prep-protocol", type=Path, required=True)
    parser.add_argument("--prep-protocol-sha256", required=True)
    parser.add_argument("--final-development-root", type=Path, required=True)
    parser.add_argument("--baseline-source-root", type=Path, required=True)
    parser.add_argument("--topk2-source-root", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--test-panel", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--sbatch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    require(args.prep_protocol, args.prep_protocol_sha256)
    prep = json.loads(args.prep_protocol.read_text(encoding="utf-8"))
    if prep.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_STRUCTURE_GENERATION_AND_LABEL_ARTIFACT_READ" or prep.get("claim_boundary", {}).get("cumulative_test_access_ordinal") != 3:
        raise ValueError("preparation protocol differs")
    final_analysis_path = args.final_development_root / "ABF30B_FRAMECONTACT_DEVELOPMENT_ANALYSIS_V1.json"
    require(final_analysis_path, FINAL_ANALYSIS_SHA256)
    final_analysis = json.loads(final_analysis_path.read_text(encoding="utf-8"))
    if final_analysis.get("status") != "PASS_ABF30B_ANTIBODY_REPRESENTATION_DEVELOPMENT" or final_analysis.get("distance_loss_weight") != 0.0 or final_analysis.get("old_sabdab2_test_loaded") is not False:
        raise ValueError("final development analysis differs")
    structure_manifest_path = args.structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    structure_receipt_path = args.structure_root / "STRUCTURE_CACHE_RECEIPT.json"
    structure_binding_path = args.structure_root / "TEST_STRUCTURE_CACHE_BINDING_V1.json"
    embedding_manifest_path = args.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
    embedding_receipt_path = args.embedding_root / "SABDAB2_ABF30B_MIXED_EMBEDDING_RECEIPT.json"
    for path in (structure_manifest_path, structure_receipt_path, structure_binding_path, embedding_manifest_path, embedding_receipt_path):
        require(path)
    structure = json.loads(structure_manifest_path.read_text(encoding="utf-8"))
    structure_binding = json.loads(structure_binding_path.read_text(encoding="utf-8"))
    embeddings = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    embedding_receipt = json.loads(embedding_receipt_path.read_text(encoding="utf-8"))
    if (
        structure.get("counts", {}).get("roles") != {"test": 549}
        or structure.get("authority", {}).get("experimental_bound_complex_coordinates_read") is not False
        or structure.get("authority", {}).get("cross_partner_coordinates_or_distances_used") is not False
        or structure_binding.get("prep_protocol_sha256") != args.prep_protocol_sha256
        or embeddings.get("status") != "PASS_COMPLETE_EXPOSED_TEST_MIXED_PARTNER_EMBEDDINGS"
        or embeddings.get("old_sabdab2_test_loaded") is not True
        or embedding_receipt.get("manifest_sha256") != sha256_file(embedding_manifest_path)
    ):
        raise ValueError("test model-input authority differs")
    require(args.contact_manifest, CONTACT_MANIFEST_SHA256)
    require(args.split, SPLIT_SHA256)
    require(args.test_panel, TEST_PANEL_SHA256)

    execution_root = args.final_development_root / "authority/development_execution_v1"
    execution_path = execution_root / "ABF30B_DEVELOPMENT_EXECUTION_V1.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    components: dict[str, dict[str, Any]] = {}
    source_roots = {"baseline": args.baseline_source_root, "edge_topk2_supervised": args.topk2_source_root}
    for arm, source_root in source_roots.items():
        components[arm] = {}
        arm_authority = execution["arms"][arm]
        for seed in SEEDS:
            run_root = args.final_development_root / "runs_development_v1" / arm / f"seed-{seed}"
            result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
            checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
            receipt_path = run_root / "RUN_RECEIPT.json"
            for path in (result_path, checkpoint_path, receipt_path):
                require(path)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            run_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            binding = final_analysis["run_bindings"][arm][str(seed)]
            if (
                binding["result_sha256"] != sha256_file(result_path)
                or binding["checkpoint_sha256"] != sha256_file(checkpoint_path)
                or binding["receipt_sha256"] != sha256_file(receipt_path)
                or run_receipt.get("test_loaded") is not False
                or result.get("configuration", {}).get("distance_loss_weight") != 0.0
            ):
                raise ValueError(f"development checkpoint binding differs: {arm}/{seed}")
            components[arm][str(seed)] = {
                "source_root": str(source_root.resolve()),
                "source_manifest_sha256": arm_authority["source_manifest_sha256"],
                "trainer_sha256": sha256_file(source_root / "scripts/train_framecontact.py"),
                "framecontact_sha256": sha256_file(source_root / "src/interaction_first/framecontact.py"),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "development_result_sha256": sha256_file(result_path),
                "development_receipt_sha256": sha256_file(receipt_path),
                "configuration": result["configuration"],
                "frozen_calibration": calibration(result),
                "selected_epoch": int(result["selection"]["best_epoch"]),
                "specificity_score_source": result["selected_checkpoint_validation"]["specificity_score_source"],
            }
    protocol = {
        "format_version": "abf30b_sabdab2_exposed_historical_test_protocol_v1",
        "status": "FROZEN_BEFORE_EXPOSED_TEST_LABEL_OPENING",
        "prep_protocol_sha256": args.prep_protocol_sha256,
        "final_development_analysis_sha256": FINAL_ANALYSIS_SHA256,
        "seeds": list(SEEDS), "arms": list(source_roots),
        "task_routing": prep["task_routing"], "components": components,
        "data": {
            "test_complex_count": 549,
            "structure_manifest_sha256": sha256_file(structure_manifest_path),
            "structure_receipt_sha256": sha256_file(structure_receipt_path),
            "structure_binding_sha256": sha256_file(structure_binding_path),
            "embedding_manifest_sha256": sha256_file(embedding_manifest_path),
            "embedding_receipt_sha256": sha256_file(embedding_receipt_path),
            "contact_manifest_sha256": CONTACT_MANIFEST_SHA256,
            "split_sha256": SPLIT_SHA256, "test_panel_sha256": TEST_PANEL_SHA256,
        },
        "execution": {
            "evaluator_sha256": require(args.evaluator), "analyzer_sha256": require(args.analyzer),
            "sbatch_sha256": require(args.sbatch), "evaluation_batch_size": 8,
            "workers": 6, "bootstrap_replicates": 10000, "bootstrap_seed": 20260831,
        },
        "claim_boundary": prep["claim_boundary"],
        "checkpoint_or_model_selected_on_test": False,
        "calibration_refit_on_test": False,
        "model_update_on_test": False,
        "test_label_artifacts_read_during_freeze": False,
        "temporal_holdout_loaded": False,
    }
    protocol_sha = publish(args.output, protocol)
    receipt_sha = publish(args.receipt_output, {
        "format_version": "abf30b_sabdab2_exposed_historical_test_protocol_receipt_v1",
        "status": protocol["status"], "protocol_sha256": protocol_sha,
        "checkpoint_count": 6, "frozen_before_test_label_artifact_read": True,
        "selection_performed_on_test": False, "fresh_blind_test": False,
        "cumulative_test_access_ordinal": 3, "temporal_holdout_loaded": False,
    })
    print(json.dumps({"status": protocol["status"], "protocol_sha256": protocol_sha, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
