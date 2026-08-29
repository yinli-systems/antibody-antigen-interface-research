#!/usr/bin/env python3
"""Freeze whole-chain versus CDR-focused adapted ESM-C downstream study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def audit_manifest(path: Path, expected_sha: str, expected_model: str) -> dict[str, object]:
    if sha256_file(path) != expected_sha:
        raise RuntimeError("embedding manifest SHA differs")
    value = json.loads(path.read_text())
    authority = value.get("authority", {})
    if (
        value.get("schema_version") != "sabdab2_esm_embedding_manifest_v1"
        or value.get("complete") is not True
        or value.get("embedding_dimension") != 1152
        or value.get("selected_sequence_count") != 2512
        or value.get("complete_cohort_sequence_count") != 2512
        or value.get("model_name") != expected_model
        or authority.get("train_instances") != 2565
        or authority.get("validation_instances") != 549
        or authority.get("task_labels_read") is not False
        or authority.get("test_access") is not False
    ):
        raise RuntimeError("embedding manifest contract differs")
    return {
        "path": str(path),
        "sha256": expected_sha,
        "model_name": value["model_name"],
        "model_revision": value["model_revision"],
        "model_weight_sha256": value["model_weight_sha256"],
        "component_models": value["component_models"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-receipt", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--cdr-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    args = parser.parse_args()
    authority = args.output_root / "authority"
    protocol_path = authority / "PROTOCOL_ADAPTED_ESMC_DOWNSTREAM_V1.json"
    receipt_path = authority / "PROTOCOL_ADAPTED_ESMC_DOWNSTREAM_V1_RECEIPT.json"
    campaign_path = authority / "CAMPAIGN_ADAPTED_ESMC_DOWNSTREAM_V1.json"
    if any(path.exists() or path.is_symlink() for path in (protocol_path, receipt_path, campaign_path)):
        raise FileExistsError("frozen downstream authority already exists")

    embedding_receipt = json.loads(args.embedding_receipt.read_text())
    if (
        embedding_receipt.get("status") != "PASS_ADAPTED_ESMC_DEVELOPMENT_EMBEDDINGS_VERIFIED"
        or embedding_receipt.get("same_protocol_downstream_development_screen_authorized") is not True
        or embedding_receipt.get("test_access") is not False
    ):
        raise RuntimeError("embedding materialization receipt differs")
    base = audit_manifest(
        args.base_manifest,
        embedding_receipt["whole_chain_manifest_sha256"],
        "local/ESMC-600M-whole-chain-adapted-epoch4",
    )
    cdr = audit_manifest(
        args.cdr_manifest,
        embedding_receipt["cdr_focused_manifest_sha256"],
        "local/ESMC-600M-cdr-focused-adapted-epoch4",
    )
    if (
        base["component_models"]["antigen"] != cdr["component_models"]["antigen"]
        or base["component_models"]["antibody"]["weight_sha256"]
        != "50b27675df33f843e7a9a71eca4423c449603246980e3296dfbb21fc27ffa1fb"
        or cdr["component_models"]["antibody"]["weight_sha256"]
        != "5fd67e390f4602f88e9a59d03a11bff332da487ebe4425bdb562bcce8a3a9d9a"
    ):
        raise RuntimeError("adapted model component binding differs")
    source_manifest = args.source_root / "SOURCE_MANIFEST_EDGE_TOPK2_V1.json"
    trainer = args.source_root / "scripts/train_framecontact.py"
    framecontact = args.source_root / "src/interaction_first/framecontact.py"
    expected_sources = {
        str(source_manifest): "f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55",
        str(trainer): "ef989bbaa8878c285c46440126bdcbfc22e8261c70db9b5c57d48446ed490b2e",
        str(framecontact): "121b3e06e24db0eeb79db100a3870d19b299b44f77355b590c97ab45cc8f38fe",
    }
    if any(sha256_file(Path(path)) != digest for path, digest in expected_sources.items()):
        raise RuntimeError("frozen downstream source differs")

    protocol = {
        "schema_version": "sabdab2_adapted_esmc_downstream_development_v1",
        "protocol_status": "frozen_before_same_protocol_development_training",
        "arm": "whole_chain_adapted_vs_cdr_focused_adapted",
        "research_question": (
            "Does controlled CDR-focused adaptation improve partner-specific contact "
            "prediction over a matched whole-chain adaptation from the same base ESM-C, "
            "corpus, token budget, and fixed epoch-4 endpoint?"
        ),
        "representation_arms": {
            "whole_chain_adapted": base,
            "cdr_focused_adapted": cdr,
        },
        "data": {
            "dataset_backend": "sabdab2",
            "input_feature_mode": "external_embedding_cache",
            "structure_manifest_sha256": "edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0",
            "contact_manifest_sha256": "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33",
            "split_manifest_sha256": "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6",
            "train_instances": 2565,
            "validation_instances": 549,
            "candidate_grid_invariant": "complete_cartesian_product_exactly_once",
            "reciprocal_partner_panel": {
                "format_version": "framecontact_reciprocal_partner_panel_v2",
                "size_per_query": 31,
                "training_sha256": "fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6",
                "validation_sha256": "222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5",
            },
        },
        "tasks": {"winner": {"experiment": "epiformer_style_same_protocol"}},
        "implementation": {
            "d_model": 128,
            "num_heads": 8,
            "num_self_layers": 2,
            "num_rbf": 16,
            "neighbor_k": 32,
            "neighbor_radius_angstrom": 16.0,
            "sequence_window": 32,
            "dropout": 0.1,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "batch_size": 1,
            "evaluation_batch_size": 1,
            "precision": "bf16",
            "calibration_max_pairs": 500000,
            "offload_saved_tensors": True,
            "contact_focal_gamma": 1.0,
            "epitope_dice_weight": 1.0,
            "epitope_count_weight": 0.2,
        },
        "training_objective": {
            "loss_weights": {
                "contact": 1.0,
                "distance": 0.1,
                "paratope": 0.1,
                "epitope": 0.5,
                "specificity": 0.0,
                "contrastive": 0.0,
                "matched_local": 0.0,
            },
            "specificity_margin": 0.05,
            "score_fraction": 0.01,
            "matched_local_margin": 0.05,
        },
        "model_selection": {
            "seeds": [17, 29, 43],
            "maximum_epochs": 20,
            "early_stopping_patience": 5,
            "primary_metric": "contact_mean_per_complex_auprc",
            "checkpoint_rule": (
                "maximize validation contact mean-per-complex average precision; "
                "break ties by frozen partner-retrieval margin and then earlier epoch"
            ),
            "evaluate_partner_retrieval": True,
        },
        "frozen_analysis_gate": {
            "candidate": "cdr_focused_adapted",
            "reference": "whole_chain_adapted",
            "minimum_positive_contact_seeds": 2,
            "minimum_median_contact_delta": 0.0,
            "maximum_median_epitope_regression": 0.01,
            "maximum_median_paratope_regression": 0.01,
        },
        "authority": {
            "embedding_materialization_receipt_sha256": sha256_file(args.embedding_receipt),
            "source_files": expected_sources,
            "launcher_sha256": sha256_file(args.launcher),
            "analyzer_sha256": sha256_file(args.analyzer),
        },
        "claim_boundary": {
            "development_only": True,
            "validation_used_for_checkpoint_selection": True,
            "independent_test": False,
            "matched_fixed_epoch4_adaptation_comparison": True,
            "antigen_representation_identical_base_esmc600m": True,
            "antibody_representation_is_the_only_factor": True,
            "task_labels_read_during_embedding": False,
            "old_test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    atomic_write(protocol_path, protocol)
    receipt = {
        "format_version": "sabdab2_adapted_esmc_downstream_protocol_receipt_v1",
        "status": "FROZEN_BEFORE_SAME_PROTOCOL_DEVELOPMENT_TRAINING",
        "protocol_file_sha256": sha256_file(protocol_path),
        "embedding_materialization_receipt_sha256": sha256_file(args.embedding_receipt),
        "test_access": False,
    }
    atomic_write(receipt_path, receipt)
    campaign = {
        "format_version": "sabdab2_adapted_esmc_downstream_campaign_v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "protocol_sha256": receipt["protocol_file_sha256"],
        "protocol_receipt_sha256": sha256_file(receipt_path),
        "arms": ["whole_chain_adapted", "cdr_focused_adapted"],
        "seeds": [17, 29, 43],
        "training_started": False,
        "test_loaded": False,
    }
    atomic_write(campaign_path, campaign)
    print(json.dumps({"protocol_sha256": receipt["protocol_file_sha256"], "receipt_sha256": sha256_file(receipt_path), "campaign_sha256": sha256_file(campaign_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
