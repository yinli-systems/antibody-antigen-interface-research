#!/usr/bin/env python3
"""Freeze the corrected ESM-C 600M authority without mutating protocol v1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


EXPECTED_V1_SHA256 = "efc1d75031e171228bc897ed9a2459409ff63202de5e1c6f1e98a35de6872049"
OFFICIAL_ESM_CODE_REVISION = "43ccece2ad485f27db46afdb67da2a9601e8f106"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-v1", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    observed_v1_sha = sha256_file(args.protocol_v1)
    if observed_v1_sha != EXPECTED_V1_SHA256:
        raise RuntimeError(
            f"protocol v1 SHA mismatch: {observed_v1_sha} != {EXPECTED_V1_SHA256}"
        )
    v1 = json.loads(args.protocol_v1.read_text())
    if v1.get("format_version") != "abpairfm_esmc_cdr_study_protocol_v1":
        raise RuntimeError("unexpected protocol v1 format")

    v2 = copy.deepcopy(v1)
    v2["format_version"] = "abpairfm_esmc_cdr_study_protocol_v2"
    v2["status"] = "FROZEN_DATA_GATE_BEFORE_ESMC_600M_WEIGHT_BINDING"
    v2["supersession"] = {
        "supersedes_protocol_sha256": observed_v1_sha,
        "reason": (
            "Protocol v1 incorrectly recorded a 960-dimensional representation for "
            "ESM-C 600M. The official architecture uses d_model=1152; d_model=960 "
            "belongs to ESM-C 300M. No model binding, embedding materialization, or "
            "training occurred under the incorrect field."
        ),
        "dimension_independent_hit_sharding_remains_valid": True,
        "v1_forbidden_for_model_binding_or_training": True,
    }
    v2["official_architecture_authority"] = {
        "repository": "https://github.com/Biohub/esm",
        "source_file": "esm/pretrained.py",
        "repository_revision": OFFICIAL_ESM_CODE_REVISION,
        "esmc_600m": {"d_model": 1152, "n_heads": 18, "n_layers": 36},
        "esmc_300m": {"d_model": 960, "n_heads": 15, "n_layers": 30},
        "literature_doi": "10.1038/s44488-026-00010-2",
    }
    v2["model_binding"] = {
        "status": "REQUIRED_BEFORE_TRAINING",
        "canonical_model_id": "biohub/ESMC-600M",
        "legacy_model_alias": "biohub/esmc-600m-2024-12",
        "revision": None,
        "weight_sha256": None,
        "embedding_dimension_expected": 1152,
        "architecture": {"d_model": 1152, "n_heads": 18, "n_layers": 36},
        "license_status": "REQUIRES_EXACT_MODEL_CARD_AND_WEIGHT_REVISION_BINDING",
        "license_must_be_accepted_and_recorded": True,
        "training_authorized": False,
    }
    v2["existing_960_dim_embedding_boundary"] = {
        "allowed_as_esmc_600m_input": False,
        "reason": "960 dimensions are incompatible with the official ESM-C 600M architecture",
        "possible_secondary_300m_control": (
            "allowed only after its own model ID, revision, weight SHA, tokenizer, "
            "special-token policy, and artifact manifest are verified"
        ),
    }
    v2["claim_boundary"].update(
        {
            "data_gate_may_run_before_model_weights_arrive": True,
            "hit_sharding_is_model_dimension_independent": True,
            "training_may_not_start_without_exact_model_revision_weight_sha_and_license": True,
            "existing_960_dim_cache_may_not_be_described_as_esmc_600m": True,
        }
    )

    protocol_path = args.output_root / "PROTOCOL_PAIRED_ESMC_CDR_V2.json"
    protocol_sha = write_canonical_json(protocol_path, v2)
    receipt = {
        "format_version": "abpairfm_esmc_cdr_study_protocol_v2_receipt",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "superseded_protocol_path": str(args.protocol_v1),
        "superseded_protocol_sha256": observed_v1_sha,
        "official_esm_code_revision": OFFICIAL_ESM_CODE_REVISION,
        "model_binding_complete": False,
        "training_authorized": False,
        "dimension_independent_hit_sharding_remains_valid": True,
    }
    receipt_path = args.output_root / "PROTOCOL_PAIRED_ESMC_CDR_V2_RECEIPT.json"
    receipt_sha = write_canonical_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "protocol_sha256": protocol_sha,
                "receipt_sha256": receipt_sha,
                "status": v2["status"],
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
