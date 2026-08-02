#!/usr/bin/env python3
"""Validate and seal the AsEP development protocol before model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "framecontact_asep_epitope_group_protocol_v1"
RECEIPT_SCHEMA = "framecontact_asep_protocol_freeze_receipt_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--cache-receipt", type=Path, required=True)
    parser.add_argument("--partner-metadata", type=Path, required=True)
    parser.add_argument("--train-panel", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.receipt.exists() or arguments.receipt.is_symlink():
        raise FileExistsError(f"Refusing to overwrite protocol receipt: {arguments.receipt}")
    raw = arguments.protocol.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("Unknown AsEP protocol schema")
    boundary = payload.get("evidence_boundary_at_freeze", {})
    for key in (
        "model_fitting_started",
        "official_test_graph_tensors_opened",
        "official_test_labels_read",
        "official_test_predictions_or_scores_read",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"AsEP protocol is not eligible: {key}")
    data = payload.get("data", {})
    panel = data.get("reciprocal_partner_panel", {})
    authorities = {
        "development_cache_manifest_sha256": _sha(arguments.cache_manifest),
        "development_cache_receipt_sha256": _sha(arguments.cache_receipt),
        "metadata_sha256": _sha(arguments.partner_metadata),
        "train_panel_sha256": _sha(arguments.train_panel),
        "validation_panel_sha256": _sha(arguments.validation_panel),
    }
    expected = {
        "development_cache_manifest_sha256": data.get(
            "development_cache_manifest_sha256"
        ),
        "development_cache_receipt_sha256": data.get(
            "development_cache_receipt_sha256"
        ),
        "metadata_sha256": panel.get("metadata_sha256"),
        "train_panel_sha256": panel.get("train_panel_sha256"),
        "validation_panel_sha256": panel.get("validation_panel_sha256"),
    }
    if authorities != expected:
        raise ValueError("AsEP protocol data authorities differ")
    if (
        data.get("dataset_backend") != "asep"
        or data.get("split_method") != "epitope_group"
        or panel.get("format_version") != "framecontact_reciprocal_partner_panel_v2"
        or int(panel.get("size_per_query", -1)) != 31
        or payload.get("training_objective", {}).get("specificity_score_source")
        != "interaction_only"
        or len(payload.get("model_selection", {}).get("seeds", [])) < 5
        or payload.get("model_selection", {}).get("test_role_forbidden") is not True
    ):
        raise ValueError("AsEP protocol invariants differ")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_path": str(arguments.protocol),
        "protocol_file_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_canonical_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "authorities": authorities,
        "five_seed_rule_validated": True,
        "reciprocal_panel_validated": True,
        "interaction_only_specificity_validated": True,
        "official_test_sealed_at_freeze": True,
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_bytes(_canonical(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
