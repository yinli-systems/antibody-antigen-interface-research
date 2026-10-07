#!/usr/bin/env python3
"""Freeze the no-privileged-distance bilateral Top-K readout audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any


ANALYSIS_SHA256 = "1382c6fc18624b53f50657c0c0d61408fa03f0acc5aa3a812ffa3f84bd23b78c"
SEEDS = (17, 29, 43)
K_VALUES = (1, 2, 4, 8)
TASK_FAMILIES = {
    "paratope": "baseline",
    "epitope": "edge_topk2_supervised",
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    analysis_path = (
        args.campaign_root
        / "PREDICTED_GEOMETRY_NO_PRIVILEGED_DISTANCE_ANALYSIS_V1.json"
    )
    if sha256_file(analysis_path) != ANALYSIS_SHA256:
        raise ValueError("frozen no-privileged-distance analysis differs")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis["status"] != "PASS_PREDICTED_GEOMETRY_NO_BOUND_DISTANCE_AUXILIARY_DEVELOPMENT":
        raise ValueError("source campaign did not pass")
    if analysis["old_sabdab2_test_loaded"] is not False:
        raise ValueError("source campaign accessed test")
    if analysis["temporal_holdout_loaded"] is not False:
        raise ValueError("source campaign accessed temporal holdout")
    if analysis["development_only"] is not True:
        raise ValueError("source campaign is not development-only")
    if analysis["distance_loss_weight"] != 0.0:
        raise ValueError("source campaign used privileged distance supervision")
    if analysis["distance_label_cache_supplied"] is not False:
        raise ValueError("source campaign supplied a distance-label cache")

    global_paths = {
        "structure_manifest": args.campaign_root
        / "data/structure_cache_v1/STRUCTURE_CACHE_MANIFEST.json",
        "contact_manifest": args.campaign_root
        / "data/contact_cache_predicted_geometry_authority_v1/CONTACT_CACHE_MANIFEST.json",
        "embedding_manifest": args.model_root
        / "data/sppider_seq_sabdab2_embeddings/SABDAB2_ESM_EMBEDDING_MANIFEST.json",
        "development_split": args.campaign_root
        / "data/structure_cache_v1/DEVELOPMENT_SPLIT_VIEW.csv",
    }
    authorities = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in global_paths.items()
    }

    task_records: dict[str, Any] = {}
    for task, family in TASK_FAMILIES.items():
        source_root = (
            args.model_root
            / "immutable_stage_a1_candidates"
            / (
                "abpairfm_energy_balanced_residual_distance_labels_source_v4"
                if family == "baseline"
                else "edge_topk2_distance_labels_source_v2"
            )
        )
        source_manifest = (
            args.campaign_root
            / "authority/no_privileged_distance_v1"
            / (
                "SOURCE_MANIFEST_BASELINE_NO_DISTANCE_V1.json"
                if family == "baseline"
                else "SOURCE_MANIFEST_TOPK2_NO_DISTANCE_V1.json"
            )
        )
        seed_records: dict[str, Any] = {}
        for seed in SEEDS:
            run_root = (
                args.campaign_root
                / f"runs_no_privileged_distance_v1/{family}/seed-{seed}"
            )
            result = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
            checkpoint = run_root / "FRAMECONTACT_SELECTED.pt"
            receipt = run_root / "RUN_RECEIPT.json"
            run = json.loads(result.read_text(encoding="utf-8"))
            if int(run["configuration"]["seed"]) != seed:
                raise ValueError(f"seed mismatch: {task}/{seed}")
            if float(run["configuration"]["distance_loss_weight"]) != 0.0:
                raise ValueError(f"distance auxiliary enabled: {task}/{seed}")
            if run["authority"]["test_loaded"] is not False:
                raise ValueError(f"test accessed: {task}/{seed}")
            if run["authority"]["external_test_loaded"] is not False:
                raise ValueError(f"external test accessed: {task}/{seed}")
            seed_records[str(seed)] = {
                "run_root": str(run_root),
                "run_result_path": str(result),
                "run_result_sha256": sha256_file(result),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "run_receipt_path": str(receipt),
                "run_receipt_sha256": sha256_file(receipt),
                "selected_epoch": int(run["selection"]["best_epoch"]),
                "selected_state_sha256": run["model"]["selected_state_sha256"],
            }
        task_records[task] = {
            "checkpoint_family": family,
            "source_root": str(source_root),
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "trainer_sha256": sha256_file(source_root / "scripts/train_framecontact.py"),
            "seeds": seed_records,
        }

    protocol = {
        "format_version": "bilateral_topk_readout_protocol_v1",
        "status": "FROZEN_BEFORE_FIXED_CHECKPOINT_READOUT_MEASUREMENT",
        "question": "K sensitivity for epitope and paratope readouts under the final no-privileged-distance development protocol",
        "k_values": list(K_VALUES),
        "tasks": task_records,
        "authorities": authorities,
        "source_analysis": {
            "path": str(analysis_path),
            "sha256": ANALYSIS_SHA256,
        },
        "readouts": {
            "epitope": "for each antigen residue, mean of the top min(K,n_valid) pair probabilities, clipped and mapped back to a logit",
            "paratope_baseline": "direct paratope node logit plus masked contact log-mean-exp",
            "paratope_topk": "the same direct paratope node logit plus the antibody-row Top-K probability-mean logit",
            "paratope_edge_only_control": "antibody-row Top-K probability-mean logit without the node head",
        },
        "execution": {
            "evaluator_sha256": sha256_file(args.evaluator),
            "launcher_sha256": sha256_file(args.launcher),
            "analyzer_sha256": sha256_file(args.analyzer),
            "partition": "gpu_4090",
            "python": str(args.model_root / "envs/contact/bin/python3.10"),
        },
        "systems_recovery": {
            "prior_campaign": str(
                args.model_root
                / "results/development/sabdab2_bilateral_topk_readout_v1"
            ),
            "prior_protocol_sha256": "25eebe6e341ab16abdc9c57a87cd17a94430e1d7cc0863742f1dae092d0eeefa",
            "prior_array_job": 1538190,
            "reason": "baseline source lacked an exported Top-K helper and scheduler requeue collided with immutable output paths",
            "scientific_protocol_changed": False,
            "model_or_checkpoint_changed": False,
            "data_or_labels_changed": False,
            "prior_outputs_used_for_selection": False,
        },
        "claim_boundary": {
            "development_validation_only": True,
            "fixed_checkpoint_readout_sensitivity": True,
            "not_a_from_scratch_training_ablation": True,
            "no_optimal_k_claim": True,
            "no_architecture_selection_authorized": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent)
    )
    try:
        (temporary / "authority").mkdir()
        (temporary / "runs").mkdir()
        (temporary / "logs").mkdir()
        protocol_path = temporary / "authority/BILATERAL_TOPK_READOUT_PROTOCOL_V1.json"
        protocol_path.write_bytes(canonical_bytes(protocol))
        protocol_sha = sha256_file(protocol_path)
        receipt = {
            "format_version": "bilateral_topk_readout_protocol_receipt_v1",
            "status": protocol["status"],
            "protocol_sha256": protocol_sha,
            "source_analysis_sha256": ANALYSIS_SHA256,
            "k_values": list(K_VALUES),
            "seeds": list(SEEDS),
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        }
        receipt_path = temporary / "authority/BILATERAL_TOPK_READOUT_PROTOCOL_V1_RECEIPT.json"
        receipt_path.write_bytes(canonical_bytes(receipt))
        temporary.rename(args.output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "protocol_sha256": protocol_sha,
                "receipt_sha256": sha256_file(
                    args.output_root
                    / "authority/BILATERAL_TOPK_READOUT_PROTOCOL_V1_RECEIPT.json"
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
