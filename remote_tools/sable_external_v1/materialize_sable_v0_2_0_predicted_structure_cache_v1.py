#!/usr/bin/env python3
"""Materialize a label-free SABLE cache from independent AF3 predictions.

Antibody and antigen entities were predicted separately from sequence.  This
script authenticates every AF3 batch, reuses the frozen SAbDab2 AF3 parser, and
concatenates antigen-chain predictions only in the declared sequence order.
It never opens a SABLE bound structure or task label.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any

import torch


EXPECTED_INPUT_PLAN_SHA256 = (
    "c82c12f08c87b42d21aee31cc555373ffbc4211415d9aff302444baefc971ad6"
)
EXPECTED_INPUT_RECEIPT_SHA256 = (
    "a6ba3357c282a8f375ab6e0890bb78d90b5cdbb41a75ff05b088796ab237b1d2"
)
EXPECTED_PARSER_SHA256 = (
    "4326a5af92f1ff17102db0f10570d80f9d258bfc9fd1d88425554bd183b0b3fe"
)
EXPECTED_ENTRIES = 97
EXPECTED_PDB_IDS = 88
EXPECTED_ENTITIES = 179
EXPECTED_BATCHES = 18


class MaterializationError(ValueError):
    """Raised when a frozen authority or prediction artifact differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> str:
    data = canonical_bytes(payload)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def load_parser(path: Path) -> ModuleType:
    if sha256_file(path) != EXPECTED_PARSER_SHA256:
        raise MaterializationError("frozen AF3 parser SHA256 differs")
    spec = importlib.util.spec_from_file_location("frozen_af3_parser", path)
    if spec is None or spec.loader is None:
        raise MaterializationError("cannot import frozen AF3 parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("load_prediction", "concatenate_structure", "prediction_record"):
        if not hasattr(module, name):
            raise MaterializationError(f"frozen AF3 parser lacks {name}")
    return module


def verify_batch_checksums(prediction_root: Path) -> list[dict[str, Any]]:
    batch_records: list[dict[str, Any]] = []
    for batch_index in range(EXPECTED_BATCHES):
        batch_root = prediction_root / "batches" / f"batch-{batch_index:03d}"
        checksum_path = batch_root / "RESULT_SHA256SUMS.txt"
        if not checksum_path.is_file():
            raise MaterializationError(f"missing batch checksum: {checksum_path}")
        verified = 0
        seen: set[str] = set()
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, separator, relative_name = line.partition("  ")
            if separator != "  " or len(digest) != 64 or not relative_name:
                raise MaterializationError(f"invalid checksum row: {checksum_path}")
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise MaterializationError("unsafe relative path in batch checksum")
            if relative_name in seen:
                raise MaterializationError("duplicate path in batch checksum")
            seen.add(relative_name)
            target = batch_root / relative
            if not target.is_file() or sha256_file(target) != digest:
                raise MaterializationError(f"batch artifact SHA differs: {target}")
            verified += 1
        if verified == 0:
            raise MaterializationError(f"empty batch checksum: {checksum_path}")
        input_count = len(list((batch_root / "inputs").glob("*.json")))
        model_count = len(list((batch_root / "results").glob("*/*_model.cif")))
        if input_count == 0 or input_count != model_count:
            raise MaterializationError(
                f"batch {batch_index} input/model count differs: "
                f"{input_count}/{model_count}"
            )
        batch_records.append(
            {
                "batch_index": batch_index,
                "checksum_path": str(checksum_path),
                "checksum_sha256": sha256_file(checksum_path),
                "verified_file_count": verified,
                "input_count": input_count,
                "model_count": model_count,
            }
        )
    return batch_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--input-plan-receipt", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--prediction-parser", type=Path, required=True)
    parser.add_argument("--af3-runner", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    if sha256_file(args.input_plan) != EXPECTED_INPUT_PLAN_SHA256:
        raise MaterializationError("AF3 label-free input plan SHA256 differs")
    if sha256_file(args.input_plan_receipt) != EXPECTED_INPUT_RECEIPT_SHA256:
        raise MaterializationError("AF3 label-free input receipt SHA256 differs")

    plan = json.loads(args.input_plan.read_text(encoding="utf-8"))
    receipt = json.loads(args.input_plan_receipt.read_text(encoding="utf-8"))
    counts = plan.get("counts", {})
    if (
        plan.get("status") != "PASS_FROZEN_LABEL_FREE_AF3_INPUT_PLAN"
        or receipt.get("status") != plan.get("status")
        or receipt.get("plan_sha256") != EXPECTED_INPUT_PLAN_SHA256
        or counts.get("eligible_entries") != EXPECTED_ENTRIES
        or counts.get("eligible_unique_pdb_ids") != EXPECTED_PDB_IDS
        or counts.get("prediction_entities") != EXPECTED_ENTITIES
        or counts.get("batch_count") != EXPECTED_BATCHES
        or plan.get("task_labels_read") is not False
        or plan.get("sable_bound_structures_downloaded") is not False
        or plan.get("experimental_bound_geometry_used_as_prediction_input") is not False
        or plan.get("cross_partner_geometry_used") is not False
        or plan.get("selection_or_tuning") is not False
    ):
        raise MaterializationError("AF3 label-free input authority differs")

    entities = plan.get("entities", [])
    bindings = plan.get("entry_bindings", [])
    entity_by_id = {row["entity_id"]: row for row in entities}
    if len(entity_by_id) != EXPECTED_ENTITIES or len(bindings) != EXPECTED_ENTRIES:
        raise MaterializationError("entity or entry binding cardinality differs")
    if len({row["entry_id"] for row in bindings}) != EXPECTED_ENTRIES:
        raise MaterializationError("entry IDs are not unique")
    if len({row["pdb_id"] for row in bindings}) != EXPECTED_PDB_IDS:
        raise MaterializationError("entry binding PDB cardinality differs")

    batch_records = verify_batch_checksums(args.prediction_root)
    parser_module = load_parser(args.prediction_parser)
    prediction_cache: dict[str, Any] = {}
    for entity_id, entity in sorted(entity_by_id.items()):
        batch_index = int(entity["batch_index"])
        directory = (
            args.prediction_root
            / "batches"
            / f"batch-{batch_index:03d}"
            / "results"
            / entity_id
        )
        source = parser_module.load_prediction(directory)
        expected_sequences = tuple(str(value) for value in entity["sequences"])
        if source.structure.sequences != expected_sequences:
            raise MaterializationError(f"predicted sequence differs: {entity_id}")
        prediction_cache[entity_id] = source

    used_entities: set[str] = set()
    records: list[dict[str, Any]] = []
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.tmp-", dir=args.output_root.parent
        )
    )
    try:
        for binding in sorted(bindings, key=lambda row: row["entry_id"]):
            if (
                binding.get("task_labels_read") is not False
                or binding.get("bound_structure_read") is not False
            ):
                raise MaterializationError("entry binding is not label-free")
            entry_id = str(binding["entry_id"])
            antibody_id = str(binding["antibody_entity_id"])
            antigen_ids = tuple(binding["antigen_entity_ids_in_declared_order"])
            if antibody_id not in entity_by_id or not antigen_ids:
                raise MaterializationError(f"invalid entity binding: {entry_id}")
            if any(entity_id not in entity_by_id for entity_id in antigen_ids):
                raise MaterializationError(f"unknown antigen entity: {entry_id}")
            used_entities.add(antibody_id)
            used_entities.update(antigen_ids)

            antibody_entity = entity_by_id[antibody_id]
            antibody_source = prediction_cache[antibody_id]
            antibody_sequences = tuple(antibody_entity["sequences"])
            antibody_backbone, antibody_valid, antibody_chain_ids = (
                parser_module.concatenate_structure(
                    antibody_source, antibody_sequences
                )
            )

            antigen_backbones: list[torch.Tensor] = []
            antigen_valid_masks: list[torch.Tensor] = []
            antigen_chain_ids: list[torch.Tensor] = []
            antigen_sequences: list[str] = []
            antigen_source_records: list[dict[str, Any]] = []
            for chain_index, antigen_id in enumerate(antigen_ids):
                antigen_entity = entity_by_id[antigen_id]
                expected = tuple(antigen_entity["sequences"])
                if len(expected) != 1:
                    raise MaterializationError("antigen entity is not a single chain")
                source = prediction_cache[antigen_id]
                backbone, valid, _ = parser_module.concatenate_structure(
                    source, expected
                )
                antigen_sequences.append(expected[0])
                antigen_backbones.append(backbone)
                antigen_valid_masks.append(valid)
                antigen_chain_ids.append(
                    torch.full(
                        (len(expected[0]),), chain_index, dtype=torch.int64
                    )
                )
                antigen_source_records.append(parser_module.prediction_record(source))
            antigen_backbone = torch.cat(antigen_backbones, dim=0)
            antigen_valid = torch.cat(antigen_valid_masks, dim=0)
            antigen_chain_id = torch.cat(antigen_chain_ids, dim=0)

            source_bundle = {
                "antibody_model_sha256": antibody_source.model_sha256,
                "antigen_model_sha256_in_declared_order": [
                    prediction_cache[entity_id].model_sha256
                    for entity_id in antigen_ids
                ],
            }
            source_digest = hashlib.sha256(canonical_bytes(source_bundle)).hexdigest()
            antibody_sequence = "".join(antibody_sequences)
            antigen_sequence = "".join(antigen_sequences)
            artifact = {
                "format_version": "sable_v0_2_0_sequence_predicted_structure_v1",
                "entry_id": entry_id,
                "pdb_id": str(binding["pdb_id"]),
                "source_prediction_bundle_sha256": source_digest,
                "antibody_sequence_sha256": sequence_sha256(antibody_sequence),
                "antigen_sequence_sha256": sequence_sha256(antigen_sequence),
                "antibody_backbone": antibody_backbone,
                "antigen_backbone": antigen_backbone,
                "antibody_chain_ids": antibody_chain_ids,
                "antigen_chain_ids": antigen_chain_id,
                "antibody_frame_valid_mask": antibody_valid,
                "antigen_frame_valid_mask": antigen_valid,
                "antibody_missing_frame_mask": ~antibody_valid,
                "antigen_missing_frame_mask": ~antigen_valid,
                "geometry_source": "independent_sequence_predicted_af3",
            }
            identifier_sha = hashlib.sha256(entry_id.encode("utf-8")).hexdigest()
            relative = Path("artifacts") / identifier_sha[:2] / f"{identifier_sha}.pt"
            target = temporary_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(artifact, target)
            records.append(
                {
                    "entry_id": entry_id,
                    "pdb_id": str(binding["pdb_id"]),
                    "artifact_path": relative.as_posix(),
                    "artifact_sha256": sha256_file(target),
                    "artifact_size_bytes": target.stat().st_size,
                    "source_prediction_bundle_sha256": source_digest,
                    "antibody_sequence_sha256": artifact["antibody_sequence_sha256"],
                    "antigen_sequence_sha256": artifact["antigen_sequence_sha256"],
                    "antibody_length": len(antibody_sequence),
                    "antigen_length": len(antigen_sequence),
                    "antibody_chain_count": len(antibody_sequences),
                    "antigen_chain_count": len(antigen_sequences),
                    "antibody_valid_frame_count": int(antibody_valid.sum().item()),
                    "antigen_valid_frame_count": int(antigen_valid.sum().item()),
                    "antibody_prediction": parser_module.prediction_record(
                        antibody_source
                    ),
                    "antigen_predictions_in_declared_order": antigen_source_records,
                }
            )

        if used_entities != set(entity_by_id):
            raise MaterializationError("not every predicted entity is used")
        if len(records) != EXPECTED_ENTRIES:
            raise MaterializationError("not every eligible entry was materialized")
        manifest = {
            "format_version": "sable_v0_2_0_predicted_structure_cache_v1",
            "status": "PASS_LABEL_FREE_SEQUENCE_PREDICTED_STRUCTURE_CACHE",
            "authority": {
                "input_plan_path": str(args.input_plan),
                "input_plan_sha256": EXPECTED_INPUT_PLAN_SHA256,
                "input_plan_receipt_path": str(args.input_plan_receipt),
                "input_plan_receipt_sha256": EXPECTED_INPUT_RECEIPT_SHA256,
                "prediction_parser_path": str(args.prediction_parser),
                "prediction_parser_sha256": EXPECTED_PARSER_SHA256,
                "af3_runner_path": str(args.af3_runner),
                "af3_runner_sha256": sha256_file(args.af3_runner),
                "experimental_bound_complex_coordinates_read": False,
                "cross_partner_coordinates_or_distances_used": False,
                "task_labels_read": False,
                "old_sabdab2_test_loaded": False,
                "temporal_holdout_loaded": False,
                "selection_or_tuning": False,
            },
            "atom_order": ["N", "CA", "C"],
            "counts": {
                "records": len(records),
                "unique_pdb_ids": len({row["pdb_id"] for row in records}),
                "prediction_entities": len(prediction_cache),
                "batches": len(batch_records),
            },
            "batch_checksum_records": batch_records,
            "records": records,
        }
        manifest_path = temporary_root / "STRUCTURE_CACHE_MANIFEST.json"
        manifest_sha256 = write_json(manifest_path, manifest)
        all_artifacts_verified = all(
            sha256_file(temporary_root / row["artifact_path"])
            == row["artifact_sha256"]
            for row in records
        )
        if not all_artifacts_verified:
            raise MaterializationError("structure artifact SHA verification failed")
        receipt_payload = {
            "format_version": "sable_v0_2_0_predicted_structure_cache_receipt_v1",
            "status": manifest["status"],
            "manifest_path": "STRUCTURE_CACHE_MANIFEST.json",
            "manifest_sha256": manifest_sha256,
            "record_count": len(records),
            "unique_pdb_id_count": EXPECTED_PDB_IDS,
            "prediction_entity_count": EXPECTED_ENTITIES,
            "all_batch_sha256_verified": True,
            "all_artifact_sha256_verified": True,
            "experimental_bound_complex_coordinates_read": False,
            "cross_partner_coordinates_or_distances_used": False,
            "task_labels_read": False,
        }
        receipt_sha256 = write_json(
            temporary_root / "STRUCTURE_CACHE_RECEIPT.json", receipt_payload
        )
        temporary_root.rename(args.output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                **receipt_payload,
                "receipt_sha256": receipt_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
