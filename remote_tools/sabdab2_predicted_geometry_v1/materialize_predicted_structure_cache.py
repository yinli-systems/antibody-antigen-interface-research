#!/usr/bin/env python3
"""Materialize a development-only SAbDab2 cache from AF3 monomer predictions.

The script resolves only train/validation instance IDs before touching sequence
or structure fields. Antibody and antigen predictions are independent, so no
experimental antibody-antigen pose or cross-chain distance is consumed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import torch


ARTIFACT_FORMAT = "sabdab2_intra_chain_structure_cohort_v2"
CACHE_FORMAT = "sabdab2_intra_chain_structure_cohort_cache_v2"
RECEIPT_FORMAT = "sabdab2_sequence_predicted_structure_receipt_v1"
ATOM_ORDER = ("N", "CA", "C")
ALLOWED_ROLES = frozenset({"train", "validation"})
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


class CacheError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-manifest-csv", type=Path, required=True)
    parser.add_argument("--source-split-csv", type=Path, required=True)
    parser.add_argument("--paired-antibody-root", type=Path, required=True)
    parser.add_argument("--sdh-plan", type=Path, required=True)
    parser.add_argument("--sdh-result-root", type=Path, required=True)
    parser.add_argument("--antigen-root", type=Path, required=True)
    parser.add_argument("--af3-runner-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CacheError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def development_split(
    path: Path,
    requested_ids: set[str] | None,
) -> tuple[list[str], list[dict[str, str]], dict[str, str]]:
    fieldnames, rows = read_csv(path)
    if not {"instance_id", "role"}.issubset(fieldnames):
        raise CacheError("split CSV lacks instance_id/role")
    role_by_id: dict[str, str] = {}
    selected: list[dict[str, str]] = []
    for row in rows:
        instance_id = row["instance_id"].strip()
        role = row["role"].strip()
        if instance_id in role_by_id or role not in {"train", "validation", "test"}:
            raise CacheError("split contains duplicate IDs or invalid roles")
        role_by_id[instance_id] = role
        if role in ALLOWED_ROLES and (
            requested_ids is None or instance_id in requested_ids
        ):
            selected.append(row)
    if requested_ids is not None:
        missing = requested_ids.difference(row["instance_id"] for row in selected)
        if missing:
            raise CacheError(f"requested IDs are absent/non-development: {sorted(missing)}")
    if not selected:
        raise CacheError("development split selection is empty")
    return fieldnames, selected, role_by_id


def write_split_view(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


@dataclass(frozen=True)
class ParsedStructure:
    sequences: tuple[str, ...]
    backbones: tuple[torch.Tensor, ...]
    valid_masks: tuple[torch.Tensor, ...]


def parse_af3_cif(path: Path) -> ParsedStructure:
    residues: dict[str, dict[int, dict[str, Any]]] = {}
    chain_order: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM "):
                continue
            fields = line.split()
            if len(fields) < 18:
                raise CacheError(f"malformed AF3 atom row in {path}")
            atom_name = fields[3]
            residue_name = fields[5]
            chain_name = fields[6]
            try:
                residue_index = int(fields[8])
                coordinate = tuple(float(value) for value in fields[10:13])
            except ValueError as error:
                raise CacheError(f"invalid AF3 atom row in {path}") from error
            if chain_name not in residues:
                residues[chain_name] = {}
                chain_order.append(chain_name)
            residue = residues[chain_name].setdefault(
                residue_index, {"name": residue_name, "atoms": {}}
            )
            if residue["name"] != residue_name:
                raise CacheError(f"inconsistent residue identity in {path}")
            if atom_name in ATOM_ORDER:
                if atom_name in residue["atoms"]:
                    raise CacheError(f"duplicate {atom_name} atom in {path}")
                residue["atoms"][atom_name] = coordinate
    if not chain_order:
        raise CacheError(f"AF3 CIF contains no ATOM records: {path}")

    sequences: list[str] = []
    backbones: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    for chain_name in chain_order:
        chain = residues[chain_name]
        indices = sorted(chain)
        if indices != list(range(1, len(indices) + 1)):
            raise CacheError(f"AF3 label_seq_id is not contiguous in {path}")
        sequence: list[str] = []
        backbone = torch.zeros((len(indices), 3, 3), dtype=torch.float32)
        atom_complete = torch.zeros(len(indices), dtype=torch.bool)
        for offset, residue_index in enumerate(indices):
            residue = chain[residue_index]
            try:
                sequence.append(AA3_TO_AA1[residue["name"]])
            except KeyError as error:
                raise CacheError(
                    f"unsupported predicted residue {residue['name']} in {path}"
                ) from error
            atoms = residue["atoms"]
            if all(atom in atoms for atom in ATOM_ORDER):
                backbone[offset] = torch.tensor(
                    [atoms[atom] for atom in ATOM_ORDER], dtype=torch.float32
                )
                atom_complete[offset] = True
        if not bool(torch.isfinite(backbone).all()):
            raise CacheError(f"non-finite predicted coordinates in {path}")
        frame_valid = frame_valid_mask(backbone) & atom_complete
        sequences.append("".join(sequence))
        backbones.append(backbone)
        valid_masks.append(frame_valid)
    return ParsedStructure(tuple(sequences), tuple(backbones), tuple(valid_masks))


def frame_valid_mask(backbone: torch.Tensor) -> torch.Tensor:
    n_atom, ca_atom, c_atom = backbone.unbind(dim=1)
    x_raw = c_atom - ca_atom
    x_norm = torch.linalg.vector_norm(x_raw, dim=-1)
    x_axis = x_raw / x_norm.clamp_min(1e-8).unsqueeze(-1)
    n_raw = n_atom - ca_atom
    y_raw = n_raw - (n_raw * x_axis).sum(dim=-1, keepdim=True) * x_axis
    y_norm = torch.linalg.vector_norm(y_raw, dim=-1)
    return (
        torch.isfinite(backbone).all(dim=-1).all(dim=-1)
        & (x_norm > 1e-5)
        & (y_norm > 1e-5)
    )


@dataclass(frozen=True)
class PredictionSource:
    directory: Path
    model_path: Path
    model_sha256: str
    data_path: Path
    data_sha256: str
    ranking_path: Path
    ranking_sha256: str
    selected_seed: int
    selected_sample: int
    selected_ranking_score: float
    chain_count: int
    unpaired_msa_modes: tuple[str, ...]
    paired_msa_modes: tuple[str, ...]
    templates_empty: bool
    structure: ParsedStructure


def only_path(paths: Iterable[Path], description: str) -> Path:
    values = sorted(paths)
    if len(values) != 1:
        raise CacheError(f"expected one {description}, observed {len(values)}")
    return values[0]


def msa_mode(msa: str) -> str:
    if not msa:
        return "empty"
    sequence_count = sum(line.startswith(">") for line in msa.splitlines())
    if sequence_count == 1:
        return "query_only"
    if sequence_count > 1:
        return "homologs"
    raise CacheError("AF3 MSA text has no FASTA header")


def load_prediction(directory: Path) -> PredictionSource:
    if not directory.is_dir():
        raise CacheError(f"prediction directory is missing: {directory}")
    model_path = only_path(directory.glob("*_model.cif"), "top-ranked model CIF")
    data_path = only_path(directory.glob("*_data.json"), "AF3 input data JSON")
    ranking_path = directory / "ranking_scores.csv"
    if not ranking_path.is_file():
        raise CacheError(f"ranking_scores.csv is missing: {directory}")

    with ranking_path.open("r", encoding="utf-8", newline="") as handle:
        ranking_rows = list(csv.DictReader(handle))
    if not ranking_rows or not {"seed", "sample", "ranking_score"}.issubset(
        ranking_rows[0]
    ):
        raise CacheError(f"invalid ranking_scores.csv: {directory}")
    try:
        selected = max(ranking_rows, key=lambda row: float(row["ranking_score"]))
        seed = int(selected["seed"])
        sample = int(selected["sample"])
        score = float(selected["ranking_score"])
    except (TypeError, ValueError) as error:
        raise CacheError(f"non-numeric AF3 ranking score: {directory}") from error
    if not math.isfinite(score):
        raise CacheError(f"non-finite AF3 ranking score: {directory}")
    selected_dir = directory / f"seed-{seed}_sample-{sample}"
    selected_model = only_path(selected_dir.glob("*model.cif"), "selected sample CIF")
    model_sha256 = sha256_file(model_path)
    structure = parse_af3_cif(model_path)
    selected_structure = parse_af3_cif(selected_model)
    if (
        structure.sequences != selected_structure.sequences
        or len(structure.backbones) != len(selected_structure.backbones)
        or any(
            not torch.equal(observed, selected)
            for observed, selected in zip(
                structure.backbones, selected_structure.backbones
            )
        )
    ):
        raise CacheError(f"top-level model is not the max-ranking sample: {directory}")
    data_payload = data_path.read_bytes()
    data_sha256 = hashlib.sha256(data_payload).hexdigest()
    try:
        data_json = json.loads(data_payload)
    except json.JSONDecodeError as error:
        raise CacheError(f"invalid AF3 data JSON: {data_path}") from error
    proteins = [item.get("protein") for item in data_json.get("sequences", [])]
    if any(not isinstance(protein, dict) for protein in proteins):
        raise CacheError(f"AF3 data JSON contains non-protein inputs: {data_path}")
    chain_count = len(structure.sequences)
    if len(proteins) != chain_count:
        raise CacheError(f"AF3 input/model chain counts differ: {directory}")
    input_sequences = tuple(str(protein.get("sequence", "")) for protein in proteins)
    if input_sequences != structure.sequences:
        raise CacheError(f"AF3 input/model sequences differ: {directory}")
    unpaired_modes = tuple(
        msa_mode(str(protein.get("unpairedMsa", ""))) for protein in proteins
    )
    paired_modes = tuple(
        msa_mode(str(protein.get("pairedMsa", ""))) for protein in proteins
    )
    templates_empty = all(protein.get("templates") == [] for protein in proteins)
    if not templates_empty:
        raise CacheError(f"AF3 templates are not explicitly disabled: {directory}")
    return PredictionSource(
        directory=directory,
        model_path=model_path,
        model_sha256=model_sha256,
        data_path=data_path,
        data_sha256=data_sha256,
        ranking_path=ranking_path,
        ranking_sha256=sha256_file(ranking_path),
        selected_seed=seed,
        selected_sample=sample,
        selected_ranking_score=score,
        chain_count=chain_count,
        unpaired_msa_modes=unpaired_modes,
        paired_msa_modes=paired_modes,
        templates_empty=templates_empty,
        structure=structure,
    )


def prediction_record(source: PredictionSource) -> dict[str, Any]:
    return {
        "directory": str(source.directory),
        "model_path": str(source.model_path),
        "model_sha256": source.model_sha256,
        "data_path": str(source.data_path),
        "data_sha256": source.data_sha256,
        "ranking_path": str(source.ranking_path),
        "ranking_sha256": source.ranking_sha256,
        "selected_seed": source.selected_seed,
        "selected_sample": source.selected_sample,
        "selected_ranking_score": source.selected_ranking_score,
        "chain_count": source.chain_count,
        "unpaired_msa_modes": list(source.unpaired_msa_modes),
        "paired_msa_modes": list(source.paired_msa_modes),
        "templates_empty": source.templates_empty,
    }


def concatenate_structure(
    source: PredictionSource,
    expected_sequences: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if source.structure.sequences != expected_sequences:
        raise CacheError(
            f"predicted sequence differs at {source.directory}: "
            f"expected lengths {[len(x) for x in expected_sequences]}, "
            f"observed lengths {[len(x) for x in source.structure.sequences]}"
        )
    backbone = torch.cat(source.structure.backbones, dim=0)
    valid = torch.cat(source.structure.valid_masks, dim=0)
    chain_ids = torch.cat(
        [torch.full((len(sequence),), index, dtype=torch.int64)
         for index, sequence in enumerate(expected_sequences)],
        dim=0,
    )
    return backbone, valid, chain_ids


def find_paired_prediction(
    root: Path,
    instance_id: str,
    antibody_type: str,
    expected_sequences: tuple[str, str],
    directory_names: tuple[str, ...],
    cache: dict[Path, PredictionSource],
) -> PredictionSource:
    prefix = f"{instance_id}_{antibody_type}".lower()
    candidate_names = [name for name in directory_names if name == prefix]
    candidate_names.extend(
        name for name in directory_names if name.startswith(prefix + "_")
    )
    matches: list[PredictionSource] = []
    for name in dict.fromkeys(candidate_names):
        directory = root / name
        source = cached_prediction(cache, directory)
        if source.structure.sequences == expected_sequences:
            matches.append(source)
    if len(matches) != 1:
        raise CacheError(
            f"paired AF3 source is not unique for {instance_id}: {len(matches)} matches"
        )
    return matches[0]


def cached_prediction(
    cache: dict[Any, PredictionSource], key: Any, directory: Path | None = None
) -> PredictionSource:
    if key not in cache:
        cache[key] = load_prediction(directory if directory is not None else key)
    return cache[key]


def main() -> None:
    args = parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"output root already exists: {args.output_root}")
    requested_ids = set(args.instance_ids) if args.instance_ids else None
    split_fields, split_rows, role_by_id = development_split(
        args.source_split_csv, requested_ids
    )
    selected_ids = {row["instance_id"] for row in split_rows}
    split_by_id = {row["instance_id"]: row for row in split_rows}

    manifest_fields, manifest_rows = read_csv(args.instance_manifest_csv)
    required_fields = {
        "instance_id", "type", "vh_seq", "vl_seq", "ag_seq", "ag_seq_hash"
    }
    if not required_fields.issubset(manifest_fields):
        raise CacheError(f"instance manifest lacks {sorted(required_fields)}")
    rows_by_id = {
        row["instance_id"]: row
        for row in manifest_rows
        if row["instance_id"] in selected_ids
    }
    if set(rows_by_id) != selected_ids:
        raise CacheError("development split and instance manifest differ")

    sdh_plan = json.loads(args.sdh_plan.read_text(encoding="utf-8"))
    if (
        sdh_plan.get("allowed_roles") != ["train", "validation"]
        or sdh_plan.get("test_access") is not False
        or sdh_plan.get("temporal_holdout_access") is not False
        or sdh_plan.get("task_labels_loaded") is not False
    ):
        raise CacheError("SD-H plan is not development-only")
    sdh_by_sha = {record["sequence_sha256"]: record for record in sdh_plan["records"]}

    paired_names = tuple(path.name for path in args.paired_antibody_root.iterdir() if path.is_dir())
    source_cache: dict[Path, PredictionSource] = {}
    antigen_cache: dict[str, PredictionSource] = {}
    sdh_cache: dict[str, PredictionSource] = {}
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.tmp-", dir=args.output_root.parent)
    )
    records: list[dict[str, Any]] = []
    regime_counts: dict[str, int] = {}
    try:
        split_view_path = temporary_root / "DEVELOPMENT_SPLIT_VIEW.csv"
        split_view_sha256 = write_split_view(split_view_path, split_fields, split_rows)
        for instance_id in sorted(selected_ids):
            row = rows_by_id[instance_id]
            vh = row["vh_seq"].strip().upper()
            vl = row["vl_seq"].strip().upper()
            ag = row["ag_seq"].strip().upper()
            if not vh or not ag:
                raise CacheError(f"empty required sequence for {instance_id}")
            antibody_sequence = vh + vl
            split_row = split_by_id[instance_id]
            if (
                sequence_sha256(antibody_sequence)
                != split_row["antibody_sequence_sha256"]
                or sequence_sha256(ag) != split_row["antigen_sequence_sha256"]
            ):
                raise CacheError(f"sequence authority mismatch for {instance_id}")

            if row["type"] == "SD-H":
                sequence_digest = sequence_sha256(vh)
                plan_record = sdh_by_sha.get(sequence_digest)
                if plan_record is None or instance_id not in plan_record["instance_ids"]:
                    raise CacheError(f"SD-H plan does not cover {instance_id}")
                name = plan_record["name"]
                directory = (
                    args.sdh_result_root
                    / "batches"
                    / f"batch-{int(plan_record['batch_index']):03d}"
                    / "results"
                    / name
                )
                antibody_source = cached_prediction(
                    sdh_cache, sequence_digest, directory
                )
                expected_antibody = (vh,)
            else:
                if not vl:
                    raise CacheError(f"paired antibody lacks VL for {instance_id}")
                expected_antibody = (vh, vl)
                antibody_source = find_paired_prediction(
                    args.paired_antibody_root,
                    instance_id,
                    row["type"],
                    expected_antibody,
                    paired_names,
                    source_cache,
                )
            antibody_backbone, antibody_valid, antibody_chain_ids = concatenate_structure(
                antibody_source, expected_antibody
            )

            ag_key = row["ag_seq_hash"].strip().lower()
            antigen_directory = args.antigen_root / ag_key
            antigen_source = cached_prediction(
                antigen_cache, ag_key, antigen_directory
            )
            antigen_backbone, antigen_valid, antigen_chain_ids = concatenate_structure(
                antigen_source, (ag,)
            )

            source_bundle = {
                "antibody_model_sha256": antibody_source.model_sha256,
                "antigen_model_sha256": antigen_source.model_sha256,
            }
            source_digest = hashlib.sha256(canonical_json_bytes(source_bundle)).hexdigest()
            artifact = {
                "format_version": ARTIFACT_FORMAT,
                "instance_id": instance_id,
                "source_cif_sha256": source_digest,
                "antibody_sequence_sha256": split_row["antibody_sequence_sha256"],
                "antigen_sequence_sha256": split_row["antigen_sequence_sha256"],
                "antibody_backbone": antibody_backbone,
                "antigen_backbone": antigen_backbone,
                "antibody_chain_ids": antibody_chain_ids,
                "antigen_chain_ids": antigen_chain_ids,
                "antibody_frame_valid_mask": antibody_valid,
                "antigen_frame_valid_mask": antigen_valid,
                "antibody_missing_frame_mask": ~antibody_valid,
                "antigen_missing_frame_mask": ~antigen_valid,
                "antibody_axis_projections": [],
                "geometry_source": "independent_sequence_predicted_af3",
            }
            identifier_sha = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()
            relative_path = Path("artifacts") / identifier_sha[:2] / f"{identifier_sha}.pt"
            target = temporary_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(artifact, target)
            ab_source_record = prediction_record(antibody_source)
            ag_source_record = prediction_record(antigen_source)
            for partner, source in (("antibody", antibody_source), ("antigen", antigen_source)):
                unique_modes = sorted(set(source.unpaired_msa_modes))
                mode = unique_modes[0] if len(unique_modes) == 1 else "mixed"
                regime = f"{partner}:unpaired_msa_{mode}"
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            records.append(
                {
                    "instance_id": instance_id,
                    "role": role_by_id[instance_id],
                    "artifact_path": relative_path.as_posix(),
                    "artifact_sha256": sha256_file(target),
                    "artifact_size_bytes": target.stat().st_size,
                    "source_cif_sha256": source_digest,
                    "antibody_sequence_sha256": split_row["antibody_sequence_sha256"],
                    "antigen_sequence_sha256": split_row["antigen_sequence_sha256"],
                    "antibody_length": int(antibody_backbone.shape[0]),
                    "antigen_length": int(antigen_backbone.shape[0]),
                    "antibody_valid_frame_count": int(antibody_valid.sum().item()),
                    "antigen_valid_frame_count": int(antigen_valid.sum().item()),
                    "antibody_axis_projection_count": 0,
                    "antibody_axis_projections": [],
                    "antibody_prediction": ab_source_record,
                    "antigen_prediction": ag_source_record,
                }
            )

        if len(records) != len(selected_ids):
            raise CacheError("not every development instance was materialized")
        role_counts = {
            role: sum(record["role"] == role for record in records)
            for role in sorted(ALLOWED_ROLES)
        }
        manifest = {
            "format_version": CACHE_FORMAT,
            "authority": {
                "geometry_source": "independent_sequence_predicted_af3",
                "experimental_bound_complex_coordinates_read": False,
                "cross_partner_coordinates_or_distances_used": False,
                "task_labels_read": False,
                "test_structures_resolved": False,
                "temporal_holdout_access": False,
                "selection_or_tuning": False,
                "instance_manifest_path": str(args.instance_manifest_csv),
                "instance_manifest_sha256": sha256_file(args.instance_manifest_csv),
                "source_split_path": str(args.source_split_csv),
                "source_split_sha256": sha256_file(args.source_split_csv),
                "development_split_view_path": "DEVELOPMENT_SPLIT_VIEW.csv",
                "development_split_view_sha256": split_view_sha256,
                "experimental_bound_structure_manifest_read": False,
                "experimental_bound_structure_artifacts_read": False,
                "paired_antibody_root": str(args.paired_antibody_root),
                "sdh_plan_path": str(args.sdh_plan),
                "sdh_plan_sha256": sha256_file(args.sdh_plan),
                "sdh_result_root": str(args.sdh_result_root),
                "antigen_root": str(args.antigen_root),
                "af3_runner_script_path": str(args.af3_runner_script),
                "af3_runner_script_sha256": sha256_file(args.af3_runner_script),
                "templates": "disabled in every authenticated AF3 input",
                "prediction_regime_note": (
                    "Historical paired-antibody and antigen AF3 inputs may include "
                    "homolog MSAs; newly completed SD-H inputs contain only the query."
                ),
            },
            "atom_order": list(ATOM_ORDER),
            "counts": {
                "records": len(records),
                "exclusions": 0,
                "roles": role_counts,
                "antibody_axis_projections": 0,
                "prediction_regimes": regime_counts,
            },
            "records": records,
            "exclusions": [],
        }
        manifest_path = temporary_root / "STRUCTURE_CACHE_MANIFEST.json"
        manifest_sha256 = write_json(manifest_path, manifest)
        receipt = {
            "format_version": RECEIPT_FORMAT,
            "status": "PASS_DEVELOPMENT_ONLY_PREDICTED_STRUCTURE_CACHE",
            "manifest_path": "STRUCTURE_CACHE_MANIFEST.json",
            "manifest_sha256": manifest_sha256,
            "development_split_view_sha256": split_view_sha256,
            "record_count": len(records),
            "role_counts": role_counts,
            "all_artifact_sha256_verified": all(
                sha256_file(temporary_root / record["artifact_path"])
                == record["artifact_sha256"]
                for record in records
            ),
            "experimental_bound_complex_coordinates_read": False,
            "task_labels_read": False,
            "test_structures_resolved": False,
            "temporal_holdout_access": False,
        }
        write_json(temporary_root / "STRUCTURE_CACHE_RECEIPT.json", receipt)
        temporary_root.rename(args.output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": "PASS_DEVELOPMENT_ONLY_PREDICTED_STRUCTURE_CACHE",
                "records": len(records),
                "manifest_sha256": manifest_sha256,
                "split_view_sha256": split_view_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
