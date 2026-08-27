#!/usr/bin/env python3
"""Freeze exact public ESM-C authorities before any embedding materialization.

This addendum deliberately separates a published positive-control checkpoint
from locally trained adaptation arms.  It authorizes neither training nor
downstream evaluation; a byte-level download receipt is still required first.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


EXPECTED_V3_SHA256 = "a7d32158d3ab27ca93d4e730abdb7c05d9c7d4e930091a0505bde67ca6c448ea"
ESM_SOURCE_REVISION = "43ccece2ad485f27db46afdb67da2a9601e8f106"
BASE_REPOSITORY = "biohub/ESMC-600M"
BASE_REVISION = "a7e82012c83126b9eedb055fea9fa84b6c02f094"
BASE_WEIGHT_SHA256 = "e4232c30fd35fe2f57051ec88a703996ac94520580b4b836894207a3d45d9ff8"
ADAPTED_REPOSITORY = "NOC-Lab/AbCDR-ESMC"
ADAPTED_REVISION = "d6e08c7a771f6c7b399e9f94cd3b9b36d0ac12b8"
ADAPTED_WEIGHT_SHA256 = "0490932ca515c6fe1738d070ede8b3ade28745d990a5e1325f286f3624c1d4e6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_new(path: Path, payload: object) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(payload)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-v3", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    observed = sha256_file(args.protocol_v3)
    if observed != EXPECTED_V3_SHA256:
        raise RuntimeError(f"protocol v3 SHA mismatch: {observed}")
    protocol_v3 = json.loads(args.protocol_v3.read_text())
    if (
        protocol_v3.get("format_version") != "abpairfm_esmc_cdr_study_protocol_v3"
        or protocol_v3.get("implementation_gate", {}).get("training_authorized") is not False
    ):
        raise RuntimeError("unexpected protocol v3 authority")

    binding = {
        "format_version": "abpairfm_esmc_public_model_binding_v1",
        "status": "FROZEN_PUBLIC_MODEL_AUTHORITIES_BEFORE_DOWNLOAD",
        "parent_protocol_sha256": observed,
        "source_runtime": {
            "repository": "Biohub/esm",
            "revision": ESM_SOURCE_REVISION,
            "package_installation_allowed": False,
            "execution_rule": "use an immutable source checkout; do not pip-install on ParaCloud",
        },
        "base_model": {
            "repository": BASE_REPOSITORY,
            "revision": BASE_REVISION,
            "filename": "model.safetensors",
            "expected_sha256": BASE_WEIGHT_SHA256,
            "hidden_size": 1152,
            "layers": 36,
            "attention_heads": 18,
            "role": "pre-specified unadapted ESM-C 600M control",
        },
        "published_cdr_model": {
            "repository": ADAPTED_REPOSITORY,
            "revision": ADAPTED_REVISION,
            "filename": "model.safetensors",
            "expected_sha256": ADAPTED_WEIGHT_SHA256,
            "xet_hash": "38c151e183a361c012a3cfda5c817eb60ca41ba0ca4fad765de70bdaf2092641",
            "hidden_size": 1152,
            "tensor_dtype": "BF16",
            "pair_format": "VH + literal '-' + VL",
            "adaptation": "paired antibody MLM with 50% CDR masking",
            "role": "pre-specified published positive control; no local checkpoint selection",
            "license_reported_by_model_card": "MIT",
        },
        "tokenizer": {
            "authority": "get_esmc_model_tokenizers from the pinned Biohub/esm revision",
            "special_tokens_retained_during_model_forward": True,
            "residue_export": (
                "remove BOS/EOS and the literal heavy-light separator only after the forward pass; "
                "assert VH and VL residue counts and preserve VH-then-VL order"
            ),
        },
        "frozen_comparison": {
            "first_stage": ["base_model", "published_cdr_model"],
            "same_sequences_and_downstream_split": True,
            "same_FrameContact_architecture_loss_optimizer_and_seed": True,
            "only_embedding_authority_changes": True,
            "downstream_selection_from_either_arm": False,
            "local_whole_chain_or_cdr_retraining_deferred": True,
        },
        "gates": {
            "published_checkpoint_download_sha_verified": False,
            "base_checkpoint_download_sha_verified": False,
            "tokenizer_contract_test_passed": False,
            "embedding_manifest_complete": False,
            "downstream_training_authorized": False,
            "adaptation_training_authorized": False,
        },
        "claim_boundary": {
            "published_checkpoint_is_external_work": True,
            "positive_control_is_not_our_pretraining_result": True,
            "no_old_exposed_test_or_temporal_holdout_access": True,
            "commercial_use_requires_independent_license_review": True,
        },
    }
    output = args.output_root / "MODEL_BINDING_PAIRED_ESMC_CDR_V1.json"
    binding_sha = write_new(output, binding)
    receipt = {
        "format_version": "abpairfm_esmc_public_model_binding_receipt_v1",
        "status": binding["status"],
        "binding_path": str(output),
        "binding_sha256": binding_sha,
        "parent_protocol_sha256": observed,
        "download_complete": False,
        "downstream_training_authorized": False,
        "adaptation_training_authorized": False,
    }
    receipt_path = args.output_root / "MODEL_BINDING_PAIRED_ESMC_CDR_V1_RECEIPT.json"
    receipt_sha = write_new(receipt_path, receipt)
    print(json.dumps({**receipt, "receipt_sha256": receipt_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
