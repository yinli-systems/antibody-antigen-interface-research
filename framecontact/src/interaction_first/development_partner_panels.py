"""Atomic train/validation reciprocal panels that never materialize a test panel."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .reciprocal_partner_panel import build_reciprocal_partner_panel
from .sabdab2_partner_metadata import PARTNER_METADATA_SPLIT_FORMAT
from .sabdab2_structure import authenticate_file, canonical_json_bytes, sha256_file


FORMAT = "framecontact_development_partner_panel_bundle_v1"
ROLES = ("train", "validation")


def build_development_partner_panel_bundle(
    *,
    metadata_path: Path,
    metadata_sha256: str,
    metadata_receipt_path: Path,
    metadata_receipt_sha256: str,
    output_root: Path,
    panel_size_per_direction: int = 31,
    protocol_seed: int = 20_260_730,
) -> dict[str, Any]:
    """Freeze train/validation panels while leaving the test role unmaterialized."""

    authority = {
        "metadata_sha256": authenticate_file(
            metadata_path, metadata_sha256, "split-bound partner metadata"
        ),
        "metadata_receipt_sha256": authenticate_file(
            metadata_receipt_path,
            metadata_receipt_sha256,
            "split-bound partner metadata receipt",
        ),
        "task_labels_read": False,
        "predictions_read": False,
    }
    try:
        metadata_receipt = json.loads(
            metadata_receipt_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read partner metadata receipt") from error
    if (
        metadata_receipt.get("format_version") != PARTNER_METADATA_SPLIT_FORMAT
        or metadata_receipt.get("metadata_sha256") != metadata_sha256
        or metadata_receipt.get("authority", {}).get("task_labels_read") is not False
    ):
        raise ValueError("Partner metadata receipt is not eligible")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite development panels: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        panels: dict[str, Any] = {}
        for role in ROLES:
            panel_path = temporary / f"{role}-reciprocal-{panel_size_per_direction}.json"
            payload = build_reciprocal_partner_panel(
                metadata_path=metadata_path,
                metadata_sha256=metadata_sha256,
                output_path=panel_path,
                role=role,
                panel_size_per_direction=panel_size_per_direction,
                protocol_seed=protocol_seed,
            )
            panels[role] = {
                "path": panel_path.name,
                "sha256": sha256_file(panel_path),
                "entry_count": len(payload["entries"]),
            }
        receipt = {
            "format_version": FORMAT,
            "authority": authority,
            "panel_size_per_direction": panel_size_per_direction,
            "protocol_seed": protocol_seed,
            "roles_materialized": list(ROLES),
            "test_panel_materialized": False,
            "panels": panels,
            "claim_boundary": (
                "Task-label-free train/validation wrong-partner selection only; "
                "no test panel, model prediction, or performance result was materialized."
            ),
        }
        (temporary / "DEVELOPMENT_PANEL_BUNDLE_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = ["FORMAT", "ROLES", "build_development_partner_panel_bundle"]
