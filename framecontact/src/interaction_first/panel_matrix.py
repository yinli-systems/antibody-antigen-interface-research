"""Atomic partner-metadata and reciprocal-panel matrix for every frozen split."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .reciprocal_partner_panel import build_reciprocal_partner_panel
from .sabdab2_partner_metadata import (
    PARTNER_METADATA_BASE_FORMAT,
    bind_partner_metadata_to_split,
)
from .sabdab2_split_matrix import SPLIT_MATRIX_FORMAT
from .sabdab2_structure import authenticate_file, canonical_json_bytes, sha256_file


FORMAT = "framecontact_split_panel_matrix_v1"
SPLITS = ("original", "temporal", "antigen_fold")
ROLES = ("train", "validation", "test")


def build_split_panel_matrix(
    *,
    partner_metadata_base_path: Path,
    partner_metadata_base_sha256: str,
    partner_metadata_base_receipt_path: Path,
    partner_metadata_base_receipt_sha256: str,
    split_matrix_root: Path,
    split_matrix_receipt_sha256: str,
    output_root: Path,
    panel_size_per_direction: int = 31,
    protocol_seed: int = 20_260_730,
) -> dict[str, Any]:
    """Bind one label-free base table and freeze all nine reciprocal panels."""
    if (
        isinstance(panel_size_per_direction, bool)
        or not isinstance(panel_size_per_direction, int)
        or panel_size_per_direction <= 0
    ):
        raise ValueError("panel_size_per_direction must be a positive integer")
    split_receipt_path = split_matrix_root / "SPLIT_MATRIX_RECEIPT.json"
    authority = {
        "partner_metadata_base_sha256": authenticate_file(
            partner_metadata_base_path,
            partner_metadata_base_sha256,
            "partner metadata base",
        ),
        "partner_metadata_base_receipt_sha256": authenticate_file(
            partner_metadata_base_receipt_path,
            partner_metadata_base_receipt_sha256,
            "partner metadata base receipt",
        ),
        "split_matrix_receipt_sha256": authenticate_file(
            split_receipt_path,
            split_matrix_receipt_sha256,
            "split matrix receipt",
        ),
        "task_labels_read": False,
        "predictions_read": False,
    }
    base_receipt = json.loads(
        partner_metadata_base_receipt_path.read_text(encoding="utf-8")
    )
    split_receipt = json.loads(split_receipt_path.read_text(encoding="utf-8"))
    if (
        base_receipt.get("format_version") != PARTNER_METADATA_BASE_FORMAT
        or base_receipt.get("metadata_sha256") != partner_metadata_base_sha256
        or base_receipt.get("authority", {}).get("task_labels_read") is not False
    ):
        raise ValueError("Partner metadata base receipt is not eligible")
    if (
        split_receipt.get("format_version") != SPLIT_MATRIX_FORMAT
        or split_receipt.get("authority", {}).get("task_labels_read") is not False
        or split_receipt.get("authority", {}).get("predictions_read") is not False
        or set(split_receipt.get("manifests", {})) != set(SPLITS)
    ):
        raise ValueError("Split matrix receipt is not eligible")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite panel matrix: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    matrix: dict[str, Any] = {}
    try:
        for split_name in SPLITS:
            split_specification = split_receipt["manifests"][split_name]
            split_path = split_matrix_root / split_specification["path"]
            split_sha256 = str(split_specification["sha256"])
            if sha256_file(split_path) != split_sha256:
                raise ValueError(f"{split_name} split SHA-256 differs")
            metadata_root = temporary / split_name / "metadata"
            metadata_receipt = bind_partner_metadata_to_split(
                base_path=partner_metadata_base_path,
                base_sha256=partner_metadata_base_sha256,
                split_path=split_path,
                split_sha256=split_sha256,
                output_root=metadata_root,
            )
            metadata_path = metadata_root / "PARTNER_METADATA.csv"
            metadata_sha256 = str(metadata_receipt["metadata_sha256"])
            panels = {}
            panel_root = temporary / split_name / "panels"
            panel_root.mkdir(parents=True, exist_ok=True)
            for role in ROLES:
                panel_path = panel_root / f"{role}.json"
                payload = build_reciprocal_partner_panel(
                    metadata_path=metadata_path,
                    metadata_sha256=metadata_sha256,
                    output_path=panel_path,
                    role=role,
                    panel_size_per_direction=panel_size_per_direction,
                    protocol_seed=protocol_seed,
                )
                panels[role] = {
                    "path": str(panel_path.relative_to(temporary)),
                    "sha256": sha256_file(panel_path),
                    "entry_count": len(payload["entries"]),
                }
            matrix[split_name] = {
                "split_path": str(split_path.relative_to(split_matrix_root)),
                "split_sha256": split_sha256,
                "metadata_path": str(metadata_path.relative_to(temporary)),
                "metadata_sha256": metadata_sha256,
                "metadata_role_counts": metadata_receipt["role_counts"],
                "panels": panels,
            }
        receipt = {
            "format_version": FORMAT,
            "authority": authority,
            "panel_size_per_direction": panel_size_per_direction,
            "protocol_seed": protocol_seed,
            "matrix": matrix,
            "claim_boundary": (
                "Task-label-free split binding and wrong-partner selection only; no "
                "model, specificity, generalization, or performance claim."
            ),
        }
        (temporary / "SPLIT_PANEL_MATRIX_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = ["FORMAT", "build_split_panel_matrix"]
