#!/usr/bin/env python3
"""Mask low-confidence AF3 backbone frames in a development-only cache.

The transform is label-free.  It authenticates the existing predicted-geometry
cache and the exact AF3 model CIFs that produced it, then invalidates residue
frames whose mean N/CA/C pLDDT is below a pre-registered threshold.  Sequence
embeddings, split membership, contact labels, and experimental structures are
never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable


SOURCE_MANIFEST_SHA256 = (
    "bf6ea4a21ff303f5e0bbf0f8edca5d824cc5024ec6c41fea9213172c64a8f614"
)
SOURCE_FORMAT = "sabdab2_intra_chain_structure_cohort_cache_v2"
ARTIFACT_FORMAT = "sabdab2_intra_chain_structure_cohort_v2"
ATOM_ORDER = ("N", "CA", "C")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--canary-records", type=int, default=0)
    return parser.parse_args()


def _coordinate_key(values: Any) -> tuple[int, ...]:
    return tuple(int(round(float(value) * 1000.0)) for value in values.reshape(-1))


def _model_residue_index(model_path: Path) -> dict[tuple[int, ...], tuple[float, Any]]:
    import numpy as np
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(model_path.stem, str(model_path))
        model = next(structure.get_models())
    except Exception as error:
        raise ValueError(f"cannot parse authenticated AF3 model: {model_path}") from error

    result: dict[tuple[int, ...], tuple[float, Any]] = {}
    for chain in model:
        for residue in chain:
            if not all(name in residue for name in ATOM_ORDER):
                continue
            atoms = [residue[name] for name in ATOM_ORDER]
            coordinates = np.stack([atom.coord for atom in atoms]).astype(np.float32)
            plddt = float(sum(float(atom.bfactor) for atom in atoms) / len(atoms))
            if not math.isfinite(plddt) or not 0.0 <= plddt <= 100.0:
                raise ValueError(f"invalid N/CA/C pLDDT in {model_path}")
            key = _coordinate_key(coordinates)
            previous = result.get(key)
            if previous is not None:
                previous_plddt, previous_coordinates = previous
                if (
                    abs(previous_plddt - plddt) > 1e-6
                    or not np.allclose(previous_coordinates, coordinates, atol=5e-4, rtol=0.0)
                ):
                    raise ValueError(f"ambiguous coordinate key in {model_path}")
            else:
                result[key] = (plddt, coordinates)
    if not result:
        raise ValueError(f"AF3 model has no complete N/CA/C residue frames: {model_path}")
    return result


def _partner_plddt(
    backbone: Any,
    valid_mask: Any,
    model_path: Path,
    expected_model_sha256: str,
) -> list[float | None]:
    import numpy as np

    if sha256_file(model_path) != expected_model_sha256:
        raise ValueError(f"AF3 model SHA-256 differs: {model_path}")
    residue_index = _model_residue_index(model_path)
    coordinates = backbone.detach().cpu().numpy().astype(np.float32, copy=False)
    valid = valid_mask.detach().cpu().numpy().astype(bool, copy=False)
    if coordinates.shape != (len(valid), 3, 3):
        raise ValueError("backbone and frame-valid axes differ")
    result: list[float | None] = []
    for index, (row, is_valid) in enumerate(zip(coordinates, valid)):
        if not is_valid:
            result.append(None)
            continue
        if not np.isfinite(row).all():
            raise ValueError(f"valid frame {index} contains non-finite coordinates")
        matched = residue_index.get(_coordinate_key(row))
        if matched is None:
            raise ValueError(
                f"valid frame {index} does not match the authenticated AF3 model"
            )
        plddt, model_coordinates = matched
        if not np.allclose(row, model_coordinates, atol=5e-4, rtol=0.0):
            raise ValueError(f"frame {index} exceeds coordinate matching tolerance")
        result.append(plddt)
    return result


def _transform_record(
    source_root: Path,
    record: dict[str, Any],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch

    source_artifact_path = source_root / record["artifact_path"]
    if source_artifact_path.is_symlink() or not source_artifact_path.is_file():
        raise ValueError(f"source artifact is not a regular file: {source_artifact_path}")
    if sha256_file(source_artifact_path) != record["artifact_sha256"]:
        raise ValueError(f"source artifact SHA-256 differs: {source_artifact_path}")
    artifact = torch.load(source_artifact_path, map_location="cpu", weights_only=True)
    if (
        artifact.get("format_version") != ARTIFACT_FORMAT
        or artifact.get("instance_id") != record["instance_id"]
    ):
        raise ValueError(f"source artifact metadata differs: {record['instance_id']}")

    transformed = dict(artifact)
    partner_summary: dict[str, Any] = {}
    for partner in ("antibody", "antigen"):
        prediction = record[f"{partner}_prediction"]
        model_path = Path(prediction["model_path"])
        plddts = _partner_plddt(
            artifact[f"{partner}_backbone"],
            artifact[f"{partner}_frame_valid_mask"],
            model_path,
            prediction["model_sha256"],
        )
        old_valid = artifact[f"{partner}_frame_valid_mask"].to(dtype=torch.bool)
        confidence_valid = torch.tensor(
            [value is not None and value >= threshold for value in plddts],
            dtype=torch.bool,
        )
        new_valid = old_valid & confidence_valid
        backbone = artifact[f"{partner}_backbone"].clone()
        backbone[~new_valid] = 0.0
        transformed[f"{partner}_backbone"] = backbone
        transformed[f"{partner}_frame_valid_mask"] = new_valid
        transformed[f"{partner}_missing_frame_mask"] = ~new_valid
        finite_plddts = [float(value) for value in plddts if value is not None]
        partner_summary[partner] = {
            "source_valid_frame_count": int(old_valid.sum().item()),
            "confidence_valid_frame_count": int(new_valid.sum().item()),
            "confidence_masked_frame_count": int((old_valid & ~new_valid).sum().item()),
            "preexisting_invalid_frame_count": int((~old_valid).sum().item()),
            "matched_plddt_count": len(finite_plddts),
            "plddt_min": min(finite_plddts),
            "plddt_max": max(finite_plddts),
            "plddt_mean": sum(finite_plddts) / len(finite_plddts),
        }

    transformed["confidence_geometry_mask"] = {
        "source_structure_artifact_sha256": record["artifact_sha256"],
        "confidence_source": "AF3 model CIF N/CA/C B_iso_or_equiv pLDDT",
        "threshold_rule": "mean_N_CA_C_plddt_greater_than_or_equal",
        "threshold": threshold,
        "low_confidence_backbone_replacement": "all_zero_degenerate_frame",
        "low_confidence_frame_valid_mask": False,
        "task_labels_read": False,
    }
    return transformed, partner_summary, {
        "source_artifact_path": str(source_artifact_path),
        "source_artifact_sha256": record["artifact_sha256"],
    }


def _write_cache(
    source_root: Path,
    output_root: Path,
    source_manifest: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    import torch

    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    records: list[dict[str, Any]] = []
    totals = {
        partner: {
            "source_valid_frame_count": 0,
            "confidence_valid_frame_count": 0,
            "confidence_masked_frame_count": 0,
            "preexisting_invalid_frame_count": 0,
        }
        for partner in ("antibody", "antigen")
    }
    try:
        for source_record in source_manifest["records"]:
            artifact, summary, provenance = _transform_record(
                source_root, source_record, threshold
            )
            artifact_path = temporary_root / source_record["artifact_path"]
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(artifact, artifact_path)
            record = dict(source_record)
            record.update(
                {
                    "artifact_sha256": sha256_file(artifact_path),
                    "artifact_size_bytes": artifact_path.stat().st_size,
                    "source_artifact_sha256": provenance["source_artifact_sha256"],
                    "confidence_geometry": summary,
                    "antibody_valid_frame_count": summary["antibody"][
                        "confidence_valid_frame_count"
                    ],
                    "antigen_valid_frame_count": summary["antigen"][
                        "confidence_valid_frame_count"
                    ],
                }
            )
            records.append(record)
            for partner in totals:
                for key in totals[partner]:
                    totals[partner][key] += int(summary[partner][key])

        split_name = source_manifest["authority"]["development_split_view_path"]
        source_split = source_root / split_name
        output_split = temporary_root / split_name
        shutil.copyfile(source_split, output_split)
        if sha256_file(output_split) != source_manifest["authority"][
            "development_split_view_sha256"
        ]:
            raise ValueError("development split view SHA-256 differs")

        authority = dict(source_manifest["authority"])
        authority.update(
            {
                "geometry_source": "independent_sequence_predicted_af3_plddt70_masked",
                "source_structure_manifest_sha256": SOURCE_MANIFEST_SHA256,
                "confidence_masking": {
                    "confidence_source": "AF3 model CIF N/CA/C B_iso_or_equiv pLDDT",
                    "threshold_rule": "mean_N_CA_C_plddt_greater_than_or_equal",
                    "threshold": threshold,
                    "low_confidence_backbone_replacement": "all_zero_degenerate_frame",
                    "low_confidence_frame_valid_mask": False,
                    "sequence_embeddings_changed": False,
                    "task_labels_read": False,
                    "selection_or_tuning": False,
                },
            }
        )
        required_false = (
            "experimental_bound_complex_coordinates_read",
            "experimental_bound_structure_artifacts_read",
            "experimental_bound_structure_manifest_read",
            "cross_partner_coordinates_or_distances_used",
            "task_labels_read",
            "selection_or_tuning",
            "temporal_holdout_access",
            "test_structures_resolved",
        )
        if any(authority.get(key) is not False for key in required_false):
            raise ValueError("source cache authority crosses a forbidden evidence boundary")

        manifest = dict(source_manifest)
        manifest["authority"] = authority
        manifest["records"] = records
        manifest["confidence_geometry_totals"] = totals
        manifest_path = temporary_root / "STRUCTURE_CACHE_MANIFEST.json"
        manifest_path.write_bytes(canonical_bytes(manifest))
        receipt = {
            "format_version": "sabdab2_confidence_masked_structure_cache_receipt_v1",
            "status": "PASS_LABEL_FREE_CONFIDENCE_MASKED_STRUCTURE_CACHE",
            "manifest_sha256": sha256_file(manifest_path),
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "record_count": len(records),
            "role_counts": source_manifest["counts"]["roles"],
            "threshold": threshold,
            "confidence_geometry_totals": totals,
            "experimental_bound_coordinates_read": False,
            "cross_partner_geometry_read": False,
            "task_labels_read": False,
            "test_or_temporal_loaded": False,
        }
        receipt_path = temporary_root / "STRUCTURE_CACHE_RECEIPT.json"
        receipt_path.write_bytes(canonical_bytes(receipt))
        os.replace(temporary_root, output_root)
        return {
            **receipt,
            "receipt_sha256": sha256_file(output_root / receipt_path.name),
            "output_root": str(output_root),
        }
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.threshold) or not 0.0 < args.threshold < 100.0:
        raise ValueError("threshold must be finite and strictly between 0 and 100")
    source_manifest_path = args.source_cache_root / "STRUCTURE_CACHE_MANIFEST.json"
    if sha256_file(source_manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("source structure manifest SHA-256 differs")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("format_version") != SOURCE_FORMAT
        or source_manifest.get("counts", {}).get("roles")
        != {"train": 2565, "validation": 549}
        or len(source_manifest.get("records", [])) != 3114
    ):
        raise ValueError("source structure cache cohort differs")

    if args.canary_records:
        if args.output_root is not None or args.canary_records < 1:
            raise ValueError("canary mode does not accept an output root")
        summaries = []
        for record in source_manifest["records"][: args.canary_records]:
            _, summary, _ = _transform_record(args.source_cache_root, record, args.threshold)
            summaries.append({"instance_id": record["instance_id"], **summary})
        print(
            json.dumps(
                {
                    "status": "PASS_CONFIDENCE_EXTRACTION_CANARY",
                    "threshold": args.threshold,
                    "records": summaries,
                    "task_labels_read": False,
                },
                sort_keys=True,
            )
        )
        return
    if args.output_root is None:
        raise ValueError("full materialization requires --output-root")
    print(
        json.dumps(
            _write_cache(
                args.source_cache_root,
                args.output_root,
                source_manifest,
                args.threshold,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
