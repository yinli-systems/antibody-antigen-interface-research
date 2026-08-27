#!/usr/bin/env python3
"""Freeze the source-verified ESM-C 600M CDR study authority."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


EXPECTED_V2_SHA256 = "8b9e66287f3d702720d4199c4370ccb5a18cd06411d8b2c20b5e2a0f2404819a"
OFFICIAL_SOURCE_ARCHIVE_SHA256 = "316c0a2d8dd92acf7f8bc03fa90bbac5822fdd4355d9685699d10e531a7e2e83"
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
    parser.add_argument("--protocol-v2", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    observed_v2_sha = sha256_file(args.protocol_v2)
    if observed_v2_sha != EXPECTED_V2_SHA256:
        raise RuntimeError(
            f"protocol v2 SHA mismatch: {observed_v2_sha} != {EXPECTED_V2_SHA256}"
        )
    v2 = json.loads(args.protocol_v2.read_text())
    if v2.get("format_version") != "abpairfm_esmc_cdr_study_protocol_v2":
        raise RuntimeError("unexpected protocol v2 format")

    v3 = copy.deepcopy(v2)
    v3["format_version"] = "abpairfm_esmc_cdr_study_protocol_v3"
    v3["status"] = "FROZEN_SOURCE_VERIFIED_DATA_GATE_BEFORE_ESMC_600M_WEIGHT_BINDING"
    v3["supersession"] = {
        "supersedes_protocol_sha256": observed_v2_sha,
        "reason": (
            "Protocol v3 binds the authors' public source archive and corrects the "
            "paired input from an abstract dedicated separator to the literal '-' "
            "token used by the released ESM-C dataset loader. No model binding, "
            "embedding materialization, or training occurred under v2."
        ),
        "dimension_independent_hit_sharding_remains_valid": True,
        "v1_and_v2_forbidden_for_model_binding_or_training": True,
    }
    v3["official_training_source_authority"] = {
        "zenodo_record": "https://zenodo.org/records/19900188",
        "zenodo_record_id": 19900188,
        "archive_name": "AbCDR-ESM.zip",
        "archive_sha256": OFFICIAL_SOURCE_ARCHIVE_SHA256,
        "archive_md5_reported_by_zenodo": "73c722e0b3b7e811f8063ec0425de55c",
        "training_script": "AbCDR-ESM/AbCDR-ESM/fine-tune/esmc/esmc_paired_train.py",
        "dataset_script": "AbCDR-ESM/AbCDR-ESM/utility/protein_dataset.py",
        "collator_script": "AbCDR-ESM/AbCDR-ESM/utility/data_collator.py",
        "wrapper_script": "AbCDR-ESM/AbCDR-ESM/utility/protein_modeling.py",
        "literature_doi": "10.1038/s44488-026-00010-2",
    }
    v3["official_architecture_authority"]["repository_revision"] = (
        OFFICIAL_ESM_CODE_REVISION
    )
    v3["paired_input"] = {
        "format": "VH + literal '-' token + VL",
        "separator_literal": "-",
        "tokenizer_add_special_tokens": True,
        "maximum_tokenized_length": 164,
        "heavy_light_pairing_preserved": True,
        "cdr_annotation_authority": "OAS IMGT annotations in authenticated paired archive",
        "cdr_indices": (
            "zero-based, chain-local half-open [start,end) intervals, offset after "
            "tokenization by any leading special token and the literal separator"
        ),
    }
    v3["controlled_arms"] = {
        "base_esmc_600m": "frozen base model; no antibody adaptation",
        "whole_chain_control": "15% whole-chain masking, BERT 80/10/10 corruption",
        "cdr_focused": "50% masking within CDR1-3 of VH and VL, BERT 80/10/10 corruption",
        "hybrid": "excluded from this minimal study; authors report no ESM-C benefit",
    }
    v3["optimization_after_model_binding"].update(
        {
            "epochs_maximum": 4,
            "duration_boundary": (
                "resource-bounded, literature-informed four-epoch study; not an exact "
                "reproduction of every ten-epoch trajectory in the paper"
            ),
            "checkpoint_rule": (
                "pre-frozen epoch 4 for the adapted arms, matching the authors' "
                "validation-selected direct-CDR ESM-C checkpoint; no downstream metric selection"
            ),
            "random_token_sampling": (
                "match the source collator exactly, including sampling from the tokenizer vocabulary"
            ),
        }
    )
    v3["implementation_gate"] = {
        "official_source_archive_sha_verified": True,
        "model_dimension_verified": True,
        "separator_and_cdr_offset_semantics_verified": True,
        "exact_model_revision_weight_sha_tokenizer_and_license_bound": False,
        "training_authorized": False,
    }
    v3["claim_boundary"].update(
        {
            "not_an_exact_full_compute_reproduction_of_the_published_study": True,
            "no_downstream_selection_of_pretraining_checkpoint": True,
            "source_verified_adaptation_not_claimed_beneficial_until_fixed_transfer_passes": True,
        }
    )

    protocol_path = args.output_root / "PROTOCOL_PAIRED_ESMC_CDR_V3.json"
    protocol_sha = write_canonical_json(protocol_path, v3)
    receipt = {
        "format_version": "abpairfm_esmc_cdr_study_protocol_v3_receipt",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "superseded_protocol_path": str(args.protocol_v2),
        "superseded_protocol_sha256": observed_v2_sha,
        "official_source_archive_sha256": OFFICIAL_SOURCE_ARCHIVE_SHA256,
        "official_esm_code_revision": OFFICIAL_ESM_CODE_REVISION,
        "model_binding_complete": False,
        "training_authorized": False,
        "dimension_independent_hit_sharding_remains_valid": True,
    }
    receipt_path = args.output_root / "PROTOCOL_PAIRED_ESMC_CDR_V3_RECEIPT.json"
    receipt_sha = write_canonical_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "protocol_sha256": protocol_sha,
                "receipt_sha256": receipt_sha,
                "status": v3["status"],
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
