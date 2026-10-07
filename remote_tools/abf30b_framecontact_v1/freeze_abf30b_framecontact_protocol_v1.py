#!/usr/bin/env python3
"""Freeze the ABF-650M/OAS-30B FrameContact transfer and external-measurement plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FINAL_RECEIPT_SHA256 = "9dcc9c8fb35a08dc95af8772335671f2d40ef4c5993c313e0d69fbfabea3e5f2"
FINAL_MODEL_SHA256 = "c36e0b1d812f97317109f1f8ab085a349beb2a782a6f4092b3074bfc1eaab3ea"
MODEL_FAMILY_SHA256 = "1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015"
MODEL_SOURCE_SHA256 = "f33e90400f6e7a55a434adbc72534fc7c971b7e60b946763c410287f1cc412c3"
TOKENIZER_SOURCE_SHA256 = "270a90b6c092a2890d0d84209578d4ba1e9b8c871c5377b57b5c72e3af24e274"
SEQUENCE_VIEW_SHA256 = "27e0beb755c3d8e4b9910d4ad2ca979161668a1e1fe7b4d09bf6a60251e07468"
SEQUENCE_RECEIPT_SHA256 = "36a9249d5d1cec4065292e79d1efae8623987a4643675557f32252bcde7752ee"
STRUCTURE_MANIFEST_SHA256 = "bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614"
CONTACT_MANIFEST_SHA256 = "d772a8dce8181d161318135a1bf4947506210363bb8ed3527bfa41dc7131ad88"
SPLIT_SHA256 = "ae1d72ade95c343c186915c12f0bfe0d392877cfb5bacd89ff0e42d8422d2700"
ESM2_MANIFEST_SHA256 = "9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545"
NO_DISTANCE_CAMPAIGN_SHA256 = "9eddf93aca94c1018287620e3720f873ece3730d9e3fe85d340b372a9ef0b4d0"
NO_DISTANCE_ANALYSIS_SHA256 = "1382c6fc18624b53f50657c0c0d61408fa03f0acc5aa3a812ffa3f84bd23b78c"
SABLE_PRIOR_ANALYSIS_SHA256 = "d14016529b15f281eea049ab5ac6153f672dbf13b4acd6b515e658d7e46611fc"
SABLE_PRIOR_BINDING_SHA256 = "ab4c8e6876ad5767fea2b4af281e746ada1213f8f71570a3670ffd4bdfada1b0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def publish(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def require_sha(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority SHA differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    pretrain = root / "results/pretraining/antibody_pretraining_rtx4090_30b_5f1f6ac_v1"
    dev = root / "results/development/sabdab2_sequence_predicted_geometry_development_v1"
    sable = root / "results/development/sable_v0_2_0_framecontact_external_v1"
    paths = {
        "final_pretraining_receipt": (
            pretrain / "segment-000069/run/receipt.json",
            FINAL_RECEIPT_SHA256,
        ),
        "model_family": (
            root
            / "results/pretraining/antibody-pretraining-rtx4090-5f1f6ac-v1/protocols/antibody_model_family_v1.json",
            MODEL_FAMILY_SHA256,
        ),
        "sequence_view": (
            root
            / "data/framecontact_20260730/authorities/abpairfm_joint_embedding_v1/sabdab2_train_validation_sequences_v1.jsonl",
            SEQUENCE_VIEW_SHA256,
        ),
        "sequence_receipt": (
            root
            / "data/framecontact_20260730/authorities/abpairfm_joint_embedding_v1/SABDAB2_TRAIN_VALIDATION_SEQUENCE_VIEW_V1_RECEIPT.json",
            SEQUENCE_RECEIPT_SHA256,
        ),
        "predicted_structure_manifest": (
            dev / "data/structure_cache_v1/STRUCTURE_CACHE_MANIFEST.json",
            STRUCTURE_MANIFEST_SHA256,
        ),
        "contact_manifest": (
            dev
            / "data/contact_cache_predicted_geometry_authority_v1/CONTACT_CACHE_MANIFEST.json",
            CONTACT_MANIFEST_SHA256,
        ),
        "development_split": (
            dev / "data/structure_cache_v1/DEVELOPMENT_SPLIT_VIEW.csv",
            SPLIT_SHA256,
        ),
        "esm2_manifest": (
            root
            / "data/sppider_seq_sabdab2_embeddings/SABDAB2_ESM_EMBEDDING_MANIFEST.json",
            ESM2_MANIFEST_SHA256,
        ),
        "no_distance_campaign": (
            dev / "authority/no_privileged_distance_v1/CAMPAIGN_NO_PRIVILEGED_DISTANCE_V1.json",
            NO_DISTANCE_CAMPAIGN_SHA256,
        ),
        "no_distance_analysis": (
            dev / "PREDICTED_GEOMETRY_NO_PRIVILEGED_DISTANCE_ANALYSIS_V1.json",
            NO_DISTANCE_ANALYSIS_SHA256,
        ),
        "sable_prior_analysis": (
            sable
            / "analysis_external_once_v1/SABLE_V0_2_0_EXTERNAL_ANALYSIS_V1.json",
            SABLE_PRIOR_ANALYSIS_SHA256,
        ),
        "sable_prior_binding": (
            sable
            / "authority/SABLE_V0_2_0_NO_DISTANCE_CHECKPOINT_BINDING_V1.json",
            SABLE_PRIOR_BINDING_SHA256,
        ),
    }
    for path, digest in paths.values():
        require_sha(path, digest)

    receipt = json.loads(paths["final_pretraining_receipt"][0].read_text())
    family = json.loads(paths["model_family"][0].read_text())
    dev_analysis = json.loads(paths["no_distance_analysis"][0].read_text())
    sable_analysis = json.loads(paths["sable_prior_analysis"][0].read_text())
    if (
        receipt.get("campaign", {}).get("complete") is not True
        or receipt.get("campaign", {}).get("token_limit") != 30_000_000_000
        or receipt.get("preset") != "abf_650m"
        or receipt.get("checkpoint_model_sha256") != FINAL_MODEL_SHA256
        or receipt.get("final_model_sha256") != FINAL_MODEL_SHA256
        or receipt.get("implementation", {}).get("src/antibody_pretraining/model.py")
        != MODEL_SOURCE_SHA256
        or receipt.get("implementation", {}).get("src/antibody_pretraining/tokenizer.py")
        != TOKENIZER_SOURCE_SHA256
        or receipt.get("test_access", {}).get("sabdab2_heldout_accessed") is not False
        or receipt.get("test_access", {}).get("downstream_labels_used") is not False
    ):
        raise ValueError("final 30B pretraining receipt differs")
    config = family.get("presets", {}).get("abf_650m", {})
    if (
        config.get("d_model") != 1280
        or config.get("n_layers") != 32
        or config.get("n_heads") != 20
        or family.get("maximum_sequence_tokens") != 194
    ):
        raise ValueError("ABF-650M architecture differs")
    if (
        dev_analysis.get("status")
        != "PASS_PREDICTED_GEOMETRY_NO_BOUND_DISTANCE_AUXILIARY_DEVELOPMENT"
        or dev_analysis.get("distance_loss_weight") != 0.0
        or dev_analysis.get("old_sabdab2_test_loaded") is not False
        or dev_analysis.get("temporal_holdout_loaded") is not False
        or sable_analysis.get("status")
        != "PASS_SABLE_V0_2_0_SINGLE_OPENING_EXTERNAL_EVALUATION"
    ):
        raise ValueError("upstream development or SABLE authority differs")

    output.mkdir(parents=True)
    protocol = {
        "format_version": "abf30b_framecontact_transfer_protocol_v1",
        "status": "FROZEN_BEFORE_CHAIN_TYPING_DCP_RESTORE_AND_MODEL_SPECIFIC_MEASUREMENT",
        "pretrained_model": {
            "name": "ABF-650M OAS-30B final checkpoint",
            "preset": "abf_650m",
            "checkpoint_path": str(pretrain / "segment-000069/run/checkpoint"),
            "checkpoint_model_sha256": FINAL_MODEL_SHA256,
            "final_receipt_sha256": FINAL_RECEIPT_SHA256,
            "model_family_sha256": MODEL_FAMILY_SHA256,
            "model_source_sha256": MODEL_SOURCE_SHA256,
            "tokenizer_source_sha256": TOKENIZER_SOURCE_SHA256,
            "embedding_layer": "final RMS-normalized residue hidden state",
            "embedding_dimension": 1280,
        },
        "chain_typing": {
            "implementation": "ANARCII 2.0.8 frozen environment",
            "accepted_chain_types": ["H", "K", "L"],
            "mapping": {"H": "heavy", "K": "kappa", "L": "lambda"},
            "paired_axis_rule": "one H chain followed by one K or L chain",
            "single_chain_rule": "accept exactly one H, K, or L classification",
            "failure_policy": "fail closed on error, ambiguity, unsupported length, or inconsistent repeated sequence segmentation",
            "paired token types not used": True,
            "light_unknown token type not used": True,
        },
        "development_comparison": {
            "cohort": {"train": 2565, "validation": 549},
            "sequence_view_sha256": SEQUENCE_VIEW_SHA256,
            "predicted_structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
            "contact_manifest_sha256": CONTACT_MANIFEST_SHA256,
            "split_sha256": SPLIT_SHA256,
            "control_embedding_manifest_sha256": ESM2_MANIFEST_SHA256,
            "single_factor": "replace antibody ESM-2 residue embeddings with ABF-650M OAS-30B embeddings",
            "antigen_embeddings": "unchanged pinned ESM-2 t33 650M",
            "arms": ["baseline", "edge_topk2_supervised"],
            "seeds": [17, 29, 43],
            "task_routing": {
                "contact": "seed-matched baseline checkpoint",
                "paratope": "seed-matched baseline checkpoint",
                "epitope": "seed-matched edge_topk2_supervised checkpoint",
            },
            "optimization_and_selection": "identical to frozen no-privileged-distance v1 protocol",
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
        "external_measurement": {
            "dataset": "SABLE v0.2.0 frozen 97-entry external cohort",
            "execution": "exactly one model-specific measurement after six development checkpoints are complete and SHA-bound",
            "checkpoint_selection": "development validation only",
            "calibration_refit": False,
            "model_update": False,
            "selection_or_tuning_from_external_results": False,
            "feedback_to_development": False,
            "analyst_has_seen_prior_model_results_on_dataset": True,
            "fresh_blind_test": False,
            "claim_role": "analyst-exposed, model-unseen, measurement-only external evaluation",
            "prior_sable_analysis_sha256": SABLE_PRIOR_ANALYSIS_SHA256,
            "prior_sable_checkpoint_binding_sha256": SABLE_PRIOR_BINDING_SHA256,
        },
        "claim_boundary": {
            "development_result_is_not_independent_test": True,
            "sable_result_is_not_fresh_blind": True,
            "old_sabdab2_test_authorized": False,
            "temporal_holdout_authorized": False,
            "sota_claim_authorized": False,
            "performance_gain_claim_authorized_before_external_result": False,
        },
    }
    protocol_path = output / "ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1.json"
    protocol_sha = publish(protocol_path, protocol)
    receipt_payload = {
        "format_version": "abf30b_framecontact_transfer_protocol_receipt_v1",
        "status": protocol["status"],
        "protocol_sha256": protocol_sha,
        "authorities": {name: digest for name, (_, digest) in paths.items()},
        "frozen_before_chain_typing": True,
        "frozen_before_dcp_restore": True,
        "frozen_before_model_specific_sable_measurement": True,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_path = output / "ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1_RECEIPT.json"
    receipt_sha = publish(receipt_path, receipt_payload)
    print(f"protocol_sha256={protocol_sha}")
    print(f"receipt_sha256={receipt_sha}")


if __name__ == "__main__":
    main()
