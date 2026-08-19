"""Authenticated residue-pair contact labels for FrameContact.

This adapter never derives pair labels from paratope or epitope marginals.  It
accepts only an externally digested ``pairs.parquet`` whose row groups each
contain one instance and whose candidate rows form an exact antibody-by-
antigen Cartesian grid.  Only positive flat indices are stored after that
completeness proof, so reconstructing the negative complement is lossless.

Rows belonging to instances outside the development structure cache are
identified from the ``instance_id`` column and skipped before the label column
is read.  This preserves the sealed-test boundary of the upstream release.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - optional dependency boundary
    torch = None
    Dataset = object

from .sabdab2_structure import (
    STRUCTURE_CACHE_FORMAT,
    STRUCTURE_COHORT_CACHE_FORMAT,
    Sabdab2StructureBatchCollator,
    StructureCacheError,
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)


CONTACT_CACHE_FORMAT = "sabdab2_framecontact_cache_v1"
CONTACT_RECEIPT_FORMAT = "sabdab2_framecontact_receipt_v1"
CONTACT_ARTIFACT_FORMAT = "sabdab2_framecontact_positive_flat_v1"
CONTACT_COHORT_CACHE_FORMAT = "sabdab2_framecontact_cohort_cache_v2"
CONTACT_COHORT_RECEIPT_FORMAT = "sabdab2_framecontact_cohort_receipt_v2"
REQUIRED_PAIR_COLUMNS = ("instance_id", "ab_res_idx", "ag_res_idx", "label")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for FrameContact contact batches")
    return torch


def _load_authenticated_structure_manifest(
    cache_root: Path, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    path = cache_root / "STRUCTURE_CACHE_MANIFEST.json"
    observed = authenticate_file(path, expected_sha256, "structure cache manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructureCacheError("Cannot parse structure cache manifest") from error
    if payload.get("format_version") not in {
        STRUCTURE_CACHE_FORMAT,
        STRUCTURE_COHORT_CACHE_FORMAT,
    }:
        raise StructureCacheError("Unknown structure cache format")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise StructureCacheError("Structure cache contains no records")
    return payload, observed


def _safe_numpy_integer(values: Any, name: str) -> Any:
    import numpy as np

    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"i", "u"}:
        raise StructureCacheError(f"{name} must be a one-dimensional integer column")
    return array.astype(np.int64, copy=False)


def _safe_numpy_binary(values: Any, name: str) -> Any:
    """Normalize a boolean or explicitly binary integer label column."""
    import numpy as np

    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"b", "i", "u"}:
        raise StructureCacheError(
            f"{name} must be a one-dimensional boolean or integer column"
        )
    normalized = array.astype(np.int64, copy=False)
    if not bool(((normalized == 0) | (normalized == 1)).all()):
        raise StructureCacheError(f"{name} must contain binary values only")
    return normalized


def materialize_contact_cache(
    *,
    pairs_path: Path,
    pairs_sha256: str,
    structure_cache_root: Path,
    structure_manifest_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """Authenticate and compact complete pair grids without reading test labels."""
    import numpy as np

    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise RuntimeError("pyarrow is required to materialize contact labels") from error

    authenticated_pairs = authenticate_file(pairs_path, pairs_sha256, "pairs parquet")
    structure_manifest, authenticated_structure = _load_authenticated_structure_manifest(
        structure_cache_root, structure_manifest_sha256
    )
    selected: dict[str, dict[str, Any]] = {}
    cohort_mode = structure_manifest.get("format_version") == STRUCTURE_COHORT_CACHE_FORMAT
    for record in structure_manifest["records"]:
        identifier = str(record.get("instance_id", ""))
        if not identifier or identifier in selected:
            raise StructureCacheError("Structure cache identifiers must be nonempty and unique")
        if not cohort_mode and record.get("split_role") not in {"train", "validation"}:
            raise StructureCacheError("Contact cache may contain development roles only")
        selected[identifier] = record

    pair_file = parquet.ParquetFile(pairs_path)
    missing_columns = set(REQUIRED_PAIR_COLUMNS) - set(pair_file.schema_arrow.names)
    if missing_columns:
        raise StructureCacheError(
            f"Pair authority is missing columns: {sorted(missing_columns)}"
        )
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite contact cache: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    records: list[dict[str, Any]] = []
    encountered: set[str] = set()
    skipped_row_groups = 0
    try:
        for row_group in range(pair_file.num_row_groups):
            id_table = pair_file.read_row_group(row_group, columns=["instance_id"])
            identifiers = id_table.column("instance_id").to_pylist()
            unique_identifiers = {str(value) for value in identifiers}
            if len(unique_identifiers) != 1:
                raise StructureCacheError(
                    f"Pair row group {row_group} must contain exactly one instance"
                )
            identifier = unique_identifiers.pop()
            if identifier not in selected:
                skipped_row_groups += 1
                continue
            if identifier in encountered:
                raise StructureCacheError(
                    f"Pair instance {identifier!r} spans more than one row group"
                )
            encountered.add(identifier)
            table = pair_file.read_row_group(
                row_group, columns=["ab_res_idx", "ag_res_idx", "label"]
            )
            antibody_index = _safe_numpy_integer(
                table.column("ab_res_idx").to_numpy(zero_copy_only=False),
                "ab_res_idx",
            )
            antigen_index = _safe_numpy_integer(
                table.column("ag_res_idx").to_numpy(zero_copy_only=False),
                "ag_res_idx",
            )
            labels = _safe_numpy_binary(
                table.column("label").to_numpy(zero_copy_only=False), "label"
            )
            record = selected[identifier]
            antibody_length = int(record["antibody_length"])
            antigen_length = int(record["antigen_length"])
            candidate_count = antibody_length * antigen_length
            if not (
                len(antibody_index) == len(antigen_index) == len(labels) == candidate_count
            ):
                raise StructureCacheError(
                    f"{identifier}: pair rows are not the complete residue Cartesian grid"
                )
            if (
                bool((antibody_index < 0).any())
                or bool((antibody_index >= antibody_length).any())
                or bool((antigen_index < 0).any())
                or bool((antigen_index >= antigen_length).any())
            ):
                raise StructureCacheError(f"{identifier}: pair indices are out of bounds")
            flat = antibody_index * antigen_length + antigen_index
            multiplicity = np.bincount(flat, minlength=candidate_count)
            if len(multiplicity) != candidate_count or not bool((multiplicity == 1).all()):
                raise StructureCacheError(
                    f"{identifier}: pair grid has duplicate or missing candidates"
                )
            positive_flat = np.sort(flat[labels == 1]).astype(np.int64, copy=False)
            identifier_sha = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
            relative_path = Path("artifacts") / identifier_sha[:2] / f"{identifier_sha}.npz"
            target = temporary_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target,
                format_version=np.asarray(CONTACT_ARTIFACT_FORMAT),
                instance_id=np.asarray(identifier),
                antibody_length=np.asarray(antibody_length, dtype=np.int64),
                antigen_length=np.asarray(antigen_length, dtype=np.int64),
                candidate_pair_count=np.asarray(candidate_count, dtype=np.int64),
                positive_flat_indices=positive_flat,
            )
            output_record = {
                    "instance_id": identifier,
                    "antibody_length": antibody_length,
                    "antigen_length": antigen_length,
                    "candidate_pair_count": candidate_count,
                    "positive_pair_count": int(len(positive_flat)),
                    "artifact_path": relative_path.as_posix(),
                    "artifact_sha256": sha256_file(target),
                    "source_row_group": row_group,
                }
            if not cohort_mode:
                output_record["split_role"] = record["split_role"]
            records.append(output_record)
        missing_instances = sorted(set(selected) - encountered)
        if missing_instances:
            preview = ", ".join(missing_instances[:5])
            raise StructureCacheError(
                f"Pair authority lacks {len(missing_instances)} structure instances: {preview}"
            )
        records.sort(key=lambda value: value["instance_id"])
        role_counts = None
        if not cohort_mode:
            role_counts = {
                role: sum(record["split_role"] == role for record in records)
                for role in ("train", "validation")
            }
        manifest = {
            "format_version": (
                CONTACT_COHORT_CACHE_FORMAT if cohort_mode else CONTACT_CACHE_FORMAT
            ),
            "authority": {
                "pairs_path": str(pairs_path),
                "pairs_sha256": authenticated_pairs,
                "structure_manifest_sha256": authenticated_structure,
                "source_pair_columns": list(REQUIRED_PAIR_COLUMNS),
                "source_row_group_invariant": "exactly_one_instance",
                "candidate_grid_invariant": "complete_cartesian_product_exactly_once",
                "unselected_pair_label_columns_read": False,
                "split_membership_read": not cohort_mode,
                "complete_cohort_pair_labels_read": cohort_mode,
                "frozen_test_labels_read": False if not cohort_mode else None,
            },
            "counts": {
                "records": len(records),
                "by_role": role_counts,
                "skipped_unselected_row_groups": skipped_row_groups,
                "candidate_pairs": sum(record["candidate_pair_count"] for record in records),
                "positive_pairs": sum(record["positive_pair_count"] for record in records),
            },
            "records": records,
        }
        manifest_path = temporary_root / "CONTACT_CACHE_MANIFEST.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        receipt = {
            "format_version": (
                CONTACT_COHORT_RECEIPT_FORMAT if cohort_mode else CONTACT_RECEIPT_FORMAT
            ),
            "manifest_sha256": sha256_file(manifest_path),
            "record_count": len(records),
            "role_counts": role_counts,
            "frozen_test_labels_read": False if not cohort_mode else None,
            "split_membership_read": not cohort_mode,
            "claim_boundary": (
                "Authenticated split-independent pair supervision cache; no model "
                "prediction, model selection, generalization, or performance claim."
                if cohort_mode
                else "Authenticated development pair labels only; no held-out, external, "
                "generalization, or performance claim."
            ),
        }
        (temporary_root / "CONTACT_CACHE_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary_root, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


def _load_contact_manifest(cache_root: Path) -> dict[str, Any]:
    path = cache_root / "CONTACT_CACHE_MANIFEST.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructureCacheError("Cannot read contact cache manifest") from error
    if payload.get("format_version") not in {
        CONTACT_CACHE_FORMAT,
        CONTACT_COHORT_CACHE_FORMAT,
    }:
        raise StructureCacheError("Unknown contact cache format")
    return payload


class Sabdab2ContactDataset(Dataset):
    """Add exact residue-pair targets to a structure/ESM dataset."""

    def __init__(self, base_dataset: Any, contact_cache_root: Path) -> None:
        _require_torch()
        if not hasattr(base_dataset, "records") or not hasattr(base_dataset, "cache_root"):
            raise TypeError("base_dataset must be a Sabdab2StructureEmbeddingDataset")
        self.base_dataset = base_dataset
        self.contact_cache_root = contact_cache_root
        manifest = _load_contact_manifest(contact_cache_root)
        structure_manifest_path = Path(base_dataset.cache_root) / "STRUCTURE_CACHE_MANIFEST.json"
        if sha256_file(structure_manifest_path) != manifest.get("authority", {}).get(
            "structure_manifest_sha256"
        ):
            raise StructureCacheError("Contact and structure cache authorities differ")
        contact_records = {record["instance_id"]: record for record in manifest["records"]}
        base_ids = [record["instance_id"] for record in base_dataset.records]
        missing = sorted(set(base_ids) - set(contact_records))
        if missing:
            raise StructureCacheError(f"Contact cache lacks selected instance {missing[0]!r}")
        self.records = tuple(contact_records[identifier] for identifier in base_ids)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np

        torch_module = _require_torch()
        item = dict(self.base_dataset[index])
        record = self.records[index]
        if item["instance_id"] != record["instance_id"]:
            raise StructureCacheError("Structure/contact dataset order differs")
        path = self.contact_cache_root / record["artifact_path"]
        if sha256_file(path) != record["artifact_sha256"]:
            raise StructureCacheError("Contact artifact digest mismatch")
        with np.load(path, allow_pickle=False) as artifact:
            required = {
                "format_version",
                "instance_id",
                "antibody_length",
                "antigen_length",
                "candidate_pair_count",
                "positive_flat_indices",
            }
            if set(artifact.files) != required:
                raise StructureCacheError("Contact artifact schema mismatch")
            if (
                str(artifact["format_version"].item()) != CONTACT_ARTIFACT_FORMAT
                or str(artifact["instance_id"].item()) != item["instance_id"]
            ):
                raise StructureCacheError("Contact artifact metadata mismatch")
            antibody_length = int(artifact["antibody_length"].item())
            antigen_length = int(artifact["antigen_length"].item())
            positive_flat = artifact["positive_flat_indices"].astype(np.int64, copy=True)
        if antibody_length != int(item["antibody_features"].shape[0]) or antigen_length != int(
            item["antigen_features"].shape[0]
        ):
            raise StructureCacheError("Contact targets differ from structure residue axes")
        pair_labels = torch_module.zeros(
            antibody_length * antigen_length, dtype=torch_module.float32
        )
        if len(positive_flat):
            if positive_flat[0] < 0 or positive_flat[-1] >= pair_labels.numel():
                raise StructureCacheError("Positive contact indices are out of bounds")
            pair_labels[torch_module.from_numpy(positive_flat)] = 1.0
        item["pair_labels"] = pair_labels.view(antibody_length, antigen_length)
        item["pair_candidate_mask"] = torch_module.ones(
            antibody_length, antigen_length, dtype=torch_module.bool
        )
        derived_antibody = item["pair_labels"].bool().any(dim=1).to(torch_module.float32)
        derived_antigen = item["pair_labels"].bool().any(dim=0).to(torch_module.float32)
        if "antibody_labels" in item and not torch_module.equal(
            item["antibody_labels"], derived_antibody
        ):
            raise StructureCacheError("Stored paratope labels differ from contact marginals")
        if "antigen_labels" in item and not torch_module.equal(
            item["antigen_labels"], derived_antigen
        ):
            raise StructureCacheError("Stored epitope labels differ from contact marginals")
        item["antibody_labels"] = derived_antibody
        item["antigen_labels"] = derived_antigen
        return item


class Sabdab2ContactBatchCollator:
    """Pad pair targets and prove they project to the stored residue labels."""

    def __init__(self, *, verify_residue_marginals: bool = True) -> None:
        self.verify_residue_marginals = bool(verify_residue_marginals)
        self.structure_collator = Sabdab2StructureBatchCollator()

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        torch_module = _require_torch()
        batch = self.structure_collator(examples)
        batch_size, max_antibody = batch["antibody_mask"].shape
        max_antigen = int(batch["antigen_mask"].shape[1])
        pair_labels = torch_module.zeros(
            batch_size, max_antibody, max_antigen, dtype=torch_module.float32
        )
        pair_candidate_mask = torch_module.zeros(
            batch_size, max_antibody, max_antigen, dtype=torch_module.bool
        )
        for batch_index, item in enumerate(examples):
            antibody_length, antigen_length = item["pair_labels"].shape
            expected_shape = (
                int(item["antibody_features"].shape[0]),
                int(item["antigen_features"].shape[0]),
            )
            if (antibody_length, antigen_length) != expected_shape or tuple(
                item["pair_candidate_mask"].shape
            ) != expected_shape:
                raise StructureCacheError("Pair targets do not align with residue axes")
            if item["pair_candidate_mask"].dtype != torch_module.bool or not bool(
                item["pair_candidate_mask"].all()
            ):
                raise StructureCacheError("FrameContact v1 requires a complete candidate grid")
            labels = item["pair_labels"]
            if not bool(((labels == 0) | (labels == 1)).all()):
                raise StructureCacheError("Pair targets must be binary")
            if self.verify_residue_marginals:
                expected_paratope = labels.bool().any(dim=1).to(torch_module.float32)
                expected_epitope = labels.bool().any(dim=0).to(torch_module.float32)
                if not torch_module.equal(expected_paratope, item["antibody_labels"]) or not torch_module.equal(
                    expected_epitope, item["antigen_labels"]
                ):
                    raise StructureCacheError(
                        "Pair labels do not reproduce paratope/epitope marginals"
                    )
            pair_labels[batch_index, :antibody_length, :antigen_length] = labels
            pair_candidate_mask[
                batch_index, :antibody_length, :antigen_length
            ] = item["pair_candidate_mask"]
        batch["pair_labels"] = pair_labels
        batch["pair_candidate_mask"] = pair_candidate_mask
        motif_keys = (
            "local_motif_positive_mask",
            "local_motif_native_noncontact_mask",
            "local_motif_matched_decoy_mask",
        )
        present = [key in item for key in motif_keys for item in examples]
        if any(present) and not all(present):
            raise StructureCacheError(
                "Local motif masks must be present for every item in a batch"
            )
        if all(present):
            for key in motif_keys:
                mask = torch_module.zeros(
                    batch_size, max_antibody, max_antigen, dtype=torch_module.bool
                )
                for batch_index, item in enumerate(examples):
                    expected_shape = tuple(item["pair_labels"].shape)
                    if tuple(item[key].shape) != expected_shape or item[key].dtype != torch_module.bool:
                        raise StructureCacheError(
                            f"{key} must be a boolean pair-aligned mask"
                        )
                    mask[batch_index, : expected_shape[0], : expected_shape[1]] = item[key]
                batch[key] = mask
        return batch


__all__ = [
    "CONTACT_ARTIFACT_FORMAT",
    "CONTACT_CACHE_FORMAT",
    "CONTACT_COHORT_CACHE_FORMAT",
    "Sabdab2ContactBatchCollator",
    "Sabdab2ContactDataset",
    "materialize_contact_cache",
]
