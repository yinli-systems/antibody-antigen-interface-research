#!/usr/bin/env python3
"""Freeze one SAbDab2 architecture winner using development validation only."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)


ENSEMBLE_FORMAT = "framecontact_development_ensemble_v1"
SELECTION_FORMAT = "framecontact_sabdab2_development_selection_v1"
RESULT_NAME = "FRAMECONTACT_DEVELOPMENT_ENSEMBLE.json"
RECEIPT_NAME = "ENSEMBLE_RECEIPT.json"
FROZEN_SEEDS = (17, 42, 73, 101, 137)
ARCHITECTURE_MATRIX = (
    "sequence_dense",
    "spatial_topology",
    "distance",
    "distance_direction",
    "full_pose",
    "frozen_frame_ipa",
    "framecontact",
    "framecontact_interleaved",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-roots", type=Path, nargs=8, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--protocol-receipt-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _load_protocol(
    path: Path, receipt_path: Path, receipt_sha256: str
) -> dict[str, Any]:
    authenticate_file(receipt_path, receipt_sha256, "SAbDab2 protocol receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    authenticate_file(
        path, str(receipt.get("protocol_file_sha256", "")), "SAbDab2 protocol"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != "framecontact_primary_protocol_v5"
        or tuple(protocol.get("development_matrix", ())) != ARCHITECTURE_MATRIX
        or tuple(protocol.get("model_selection", {}).get("seeds", ()))
        != FROZEN_SEEDS
        or protocol.get("model_selection", {}).get("temperature_and_thresholds")
        != "validation only"
    ):
        raise ValueError("SAbDab2 protocol is not architecture-selection eligible")
    return protocol


def authenticated_ensemble(root: Path) -> dict[str, Any]:
    result_path = root / RESULT_NAME
    receipt_path = root / RECEIPT_NAME
    for path, name in ((result_path, "ensemble result"), (receipt_path, "ensemble receipt")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular {name}: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("format_version") != ENSEMBLE_FORMAT
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or receipt.get("result_sha256") != sha256_file(result_path)
    ):
        raise ValueError(f"SAbDab2 ensemble receipt is ineligible: {root}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    authority = result.get("authority", {})
    validation = result.get("validation", {})
    experiment = result.get("experiment")
    selection_metric = validation.get("selection_metric")
    contact_metric = validation.get("contact", {}).get("mean_per_complex_auprc")
    if (
        result.get("format_version") != ENSEMBLE_FORMAT
        or experiment not in ARCHITECTURE_MATRIX
        or result.get("seeds") != list(FROZEN_SEEDS)
        or result.get("ensemble_rule", {}).get("member_weighting") != "uniform 1/5"
        or len(authority.get("members", [])) != 5
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
        or validation.get("selection_metric_rule")
        != "minimum_direction_cluster_weighted_margin"
        or validation.get("specificity_score_source") != "interaction_only"
        or not isinstance(selection_metric, (int, float))
        or not math.isfinite(float(selection_metric))
        or not isinstance(contact_metric, (int, float))
        or not math.isfinite(float(contact_metric))
    ):
        raise ValueError(f"SAbDab2 ensemble result is ineligible: {root}")
    return {
        "root": str(root.resolve()),
        "experiment": str(experiment),
        "selection_metric": float(selection_metric),
        "contact_mean_per_complex_auprc": float(contact_metric),
        "result_sha256": receipt["result_sha256"],
        "receipt_sha256": sha256_file(receipt_path),
        "authority": authority,
    }


def freeze_selection(
    *,
    ensemble_roots: list[Path],
    protocol: Path,
    protocol_receipt: Path,
    protocol_receipt_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite SAbDab2 selection: {output_root}")
    protocol_payload = _load_protocol(
        protocol, protocol_receipt, protocol_receipt_sha256
    )
    architecture_order = tuple(protocol_payload["development_matrix"])
    records = [authenticated_ensemble(root) for root in ensemble_roots]
    if {record["experiment"] for record in records} != set(architecture_order):
        raise ValueError("Exactly one ensemble per frozen architecture is required")
    authority_keys = (
        "structure_manifest_sha256",
        "contact_manifest_sha256",
        "embedding_manifest_sha256",
        "split_manifest_sha256",
        "validation_partner_panel_sha256",
        "protocol_file_sha256",
        "protocol_receipt_sha256",
        "test_loaded",
        "external_test_loaded",
    )
    reference = records[0]["authority"]
    if any(
        any(record["authority"].get(key) != reference.get(key) for key in authority_keys)
        for record in records[1:]
    ):
        raise ValueError("SAbDab2 ensemble data/protocol authorities differ")
    if (
        reference.get("protocol_file_sha256") != sha256_file(protocol)
        or reference.get("protocol_receipt_sha256") != protocol_receipt_sha256
    ):
        raise ValueError("SAbDab2 ensemble and requested protocol authorities differ")
    order_index = {name: index for index, name in enumerate(architecture_order)}
    ranked = sorted(
        records,
        key=lambda record: (
            -record["selection_metric"],
            -record["contact_mean_per_complex_auprc"],
            order_index[record["experiment"]],
        ),
    )
    selected = ranked[0]
    payload = {
        "format_version": SELECTION_FORMAT,
        "claim_boundary": (
            "Architecture and five-seed ensemble selected from SAbDab2 development "
            "validation only. Frozen test features, labels, predictions, and scores "
            "were not read."
        ),
        "selection_rule": {
            "primary": "maximize minimum_direction_cluster_weighted_margin",
            "first_tie_break": "maximize mean_per_complex_contact_auprc",
            "exact_tie_break": "frozen development_matrix order",
            "test_role_forbidden": True,
        },
        "selected_experiment": selected["experiment"],
        "selected_ensemble": {
            key: selected[key]
            for key in (
                "root",
                "result_sha256",
                "receipt_sha256",
                "selection_metric",
                "contact_mean_per_complex_auprc",
            )
        },
        "all_candidates": [
            {
                key: record[key]
                for key in (
                    "experiment",
                    "root",
                    "result_sha256",
                    "receipt_sha256",
                    "selection_metric",
                    "contact_mean_per_complex_auprc",
                )
            }
            for record in ranked
        ],
        "authority": {
            "protocol_sha256": sha256_file(protocol),
            "protocol_receipt_sha256": protocol_receipt_sha256,
            "common_ensemble_authority": {
                key: reference.get(key) for key in authority_keys
            },
            "test_loaded": False,
            "external_test_loaded": False,
        },
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        result_path = temporary / "SABDAB2_DEVELOPMENT_SELECTION.json"
        result_path.write_bytes(canonical_json_bytes(payload))
        receipt = {
            "format_version": SELECTION_FORMAT,
            "result_sha256": sha256_file(result_path),
            "selected_experiment": selected["experiment"],
            "selected_ensemble_result_sha256": selected["result_sha256"],
            "test_loaded": False,
            "external_test_loaded": False,
        }
        (temporary / "SELECTION_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> None:
    arguments = _arguments()
    print(
        json.dumps(
            freeze_selection(
                ensemble_roots=list(arguments.ensemble_roots),
                protocol=arguments.protocol,
                protocol_receipt=arguments.protocol_receipt,
                protocol_receipt_sha256=arguments.protocol_receipt_sha256,
                output_root=arguments.output_root,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
