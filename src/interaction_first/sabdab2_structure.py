"""Authenticated SAbDab2 backbone cache and structure-aware residue batches.

The materializer intersects a pre-existing development-only split with the
native training role of a separately authenticated SAbDab2 ML release.  It
reads only the selected mmCIF members, maps ``N/CA/C`` atoms to the canonical
residue axes used by the existing embeddings and labels, and records every
exclusion and input digest.  No native-release test structure is materialized.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None
    nn = None
    Dataset = object

from .intra_chain_geometry import DualIntraChainGeometryEncoder, build_backbone_frames


STRUCTURE_ARTIFACT_FORMAT = "sabdab2_intra_chain_structure_v1"
STRUCTURE_CACHE_FORMAT = "sabdab2_intra_chain_structure_cache_v1"
STRUCTURE_RECEIPT_FORMAT = "sabdab2_intra_chain_structure_receipt_v1"
NATIVE_TRAIN_ROLE = "train"
ATOM_ORDER = ("N", "CA", "C")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_INSTANCE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.-]{0,255}")


class StructureCacheError(ValueError):
    """Raised when structure provenance or residue alignment fails closed."""


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for SAbDab2 structure batches")
    return torch


def sha256_file(path: Path) -> str:
    """Hash one regular file without loading it completely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _verified_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise StructureCacheError(f"{name} must be a lowercase SHA-256 digest")
    return value


def authenticate_file(path: Path, expected_sha256: str, name: str) -> str:
    """Require a regular, non-symlink file with an externally supplied digest."""
    expected_sha256 = _verified_digest(expected_sha256, f"{name}.sha256")
    if path.is_symlink() or not path.is_file():
        raise StructureCacheError(f"{name} must be a regular non-symlink file")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise StructureCacheError(
            f"{name} SHA-256 differs: expected {expected_sha256}, observed {observed}"
        )
    return observed


def _string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() == "nan" else text


def _instance_id(value: Any) -> str:
    text = _string(value)
    if not _INSTANCE_PATTERN.fullmatch(text):
        raise StructureCacheError(f"Invalid instance identifier: {value!r}")
    return text


def _read_csv_index(path_or_handle: Any, key: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(path_or_handle)
    if reader.fieldnames is None or key not in reader.fieldnames:
        raise StructureCacheError(f"CSV is missing required key column {key!r}")
    for raw in reader:
        identifier = _instance_id(raw.get(key))
        if identifier in rows:
            raise StructureCacheError(f"Duplicate CSV identifier: {identifier}")
        rows[identifier] = {str(name): _string(value) for name, value in raw.items()}
    return rows


def _load_csv_index(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return _read_csv_index(handle, key)


def _load_prepared_index(path: Path) -> dict[str, dict[str, Any]]:
    try:
        import pandas as pd
    except ModuleNotFoundError as error:
        raise RuntimeError("pandas with a Parquet engine is required") from error
    frame = pd.read_parquet(path)
    required = {
        "instance_id",
        "ab_sequence_sha256",
        "ag_sequence_sha256",
        "ab_length",
        "ag_length",
        "paratope_positive_count",
        "epitope_positive_count",
    }
    if not required.issubset(frame.columns):
        raise StructureCacheError(
            f"Prepared instances are missing columns: {sorted(required - set(frame.columns))}"
        )
    if frame["instance_id"].duplicated().any():
        raise StructureCacheError("Prepared instances contain duplicate identifiers")
    return {
        _instance_id(row["instance_id"]): row
        for row in frame.to_dict(orient="records")
    }


def _load_split(path: Path) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructureCacheError("Cannot parse the locked split") from error
    if (
        payload.get("intended_use") != "development_only"
        or payload.get("valid") is not True
        or payload.get("frozen_test_labels_or_predictions_read") is not False
        or payload.get("frozen_test_sequences_aligned") is not False
    ):
        raise StructureCacheError("Split is not a valid development-only authority")
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict) or set(raw_splits) != {"train", "validation"}:
        raise StructureCacheError("Split must contain exactly train and validation")
    result: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for role in ("train", "validation"):
        values = raw_splits[role]
        if not isinstance(values, list) or not values:
            raise StructureCacheError(f"Split role {role!r} must be nonempty")
        identifiers = tuple(_instance_id(value) for value in values)
        if len(set(identifiers)) != len(identifiers) or seen.intersection(identifiers):
            raise StructureCacheError("Split roles must be unique and disjoint")
        seen.update(identifiers)
        result[role] = identifiers
    return result, payload


def _validate_split_verification(
    path: Path,
    *,
    split_sha256: str,
    expected_instance_count: int,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructureCacheError("Cannot parse the split verification authority") from error
    audited = (
        payload.get("provenance", {})
        .get("audited_splits", {})
        .get("nw_chain_strict", {})
    )
    roles = payload.get("results", {}).get("nw_chain_strict", {}).get("roles", {})
    if (
        payload.get("intended_use") != "development_only"
        or payload.get("frozen_test_labels_or_predictions_read") is not False
        or payload.get("frozen_test_sequences_aligned") is not False
        or payload.get("development_instance_count") != expected_instance_count
        or audited.get("sha256") != split_sha256
        or set(roles) != {"antibody", "antigen"}
    ):
        raise StructureCacheError("Split verification does not authenticate this split")
    for partner, audit in roles.items():
        for metric in (
            "calm_scaled_similarity_violation_count",
            "identity_and_reciprocal_coverage_violation_count",
            "exact_sequence_cross_split_overlap_count",
        ):
            if audit.get(metric) != 0:
                raise StructureCacheError(
                    f"Split verification reports {partner} {metric} != 0"
                )
    return payload


def _archive_metadata_rows(archive_path: Path) -> dict[str, dict[str, str]]:
    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            member = archive.getmember("splits_final/abag_split.csv")
            binary = archive.extractfile(member)
        except (KeyError, tarfile.TarError) as error:
            raise StructureCacheError("Archive lacks splits_final/abag_split.csv") from error
        if binary is None:
            raise StructureCacheError("Cannot read archive metadata")
        with io.TextIOWrapper(binary, encoding="utf-8", newline="") as handle:
            return _read_csv_index(handle, "INSTANCE")


def _chain_names(value: Any) -> tuple[str, ...]:
    text = _string(value)
    if not text:
        return ()
    names = tuple(part.strip() for part in text.split("/") if part.strip() not in {"", "+"})
    if len(set(names)) != len(names):
        raise StructureCacheError(f"Duplicate chain names in {text!r}")
    return names


@dataclass(frozen=True)
class ResidueSpec:
    amino_acid: str
    number: int
    insertion_code: str


def _insertion_code(value: Any) -> str:
    text = _string(value)
    return " " if not text else text


def _parse_numbering_list(raw: str) -> tuple[ResidueSpec, ...] | None:
    text = _string(raw)
    if text == "__FULL_CHAIN__":
        return None
    if not text:
        return ()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise StructureCacheError("Residue numbering list is not valid JSON") from error
    if value is None:
        return ()
    if not isinstance(value, list):
        raise StructureCacheError("Residue numbering list must be an array")
    result: list[ResidueSpec] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or len(item[0]) != 1
            or not isinstance(item[1], list)
            or len(item[1]) != 2
            or isinstance(item[1][0], bool)
            or not isinstance(item[1][0], int)
        ):
            raise StructureCacheError(f"Malformed residue numbering entry {index}")
        result.append(
            ResidueSpec(
                amino_acid=item[0].upper(),
                number=int(item[1][0]),
                insertion_code=_insertion_code(item[1][1]),
            )
        )
    keys = [(item.number, item.insertion_code) for item in result]
    if len(set(keys)) != len(keys):
        raise StructureCacheError("Residue numbering list contains duplicate keys")
    return tuple(result)


def _split_antigen_specs(raw: str, chain_count: int) -> tuple[str, ...]:
    parts = tuple(_string(part) for part in _string(raw).split("/"))
    if chain_count == 0:
        return ()
    if len(parts) != chain_count:
        raise StructureCacheError(
            f"Antigen residue specs have {len(parts)} parts for {chain_count} chains"
        )
    return parts


def _bio_parser() -> Any:
    try:
        from Bio.PDB import MMCIFParser
    except ModuleNotFoundError as error:
        raise RuntimeError("Biopython is required for mmCIF structure parsing") from error
    return MMCIFParser(QUIET=True)


def _is_polymer_residue(residue: Any) -> bool:
    try:
        from Bio.PDB.Polypeptide import is_aa
    except ModuleNotFoundError as error:
        raise RuntimeError("Biopython is required for residue parsing") from error
    return bool(is_aa(residue, standard=False))


def _residue_one_letter(residue: Any) -> str:
    try:
        from Bio.SeqUtils import seq1
    except ModuleNotFoundError as error:
        raise RuntimeError("Biopython is required for residue parsing") from error
    return seq1(residue.get_resname(), custom_map={"MSE": "M"}, undef_code="X").upper()


def _chain_lookup(model: Any, chain_name: str) -> Any:
    try:
        return model[chain_name]
    except KeyError as error:
        raise StructureCacheError(f"mmCIF lacks chain {chain_name!r}") from error


def _observed_residues(chain: Any) -> tuple[Any, ...]:
    residues = tuple(residue for residue in chain if _is_polymer_residue(residue))
    if not residues:
        raise StructureCacheError(f"Chain {chain.id!r} has no polymer residues")
    return residues


def _residue_lookup(chain: Any) -> dict[tuple[int, str], Any]:
    result: dict[tuple[int, str], Any] = {}
    for residue in _observed_residues(chain):
        key = (int(residue.id[1]), _insertion_code(residue.id[2]))
        if key in result:
            raise StructureCacheError(f"Chain {chain.id!r} has duplicate residue key {key}")
        result[key] = residue
    return result


def _atom_coordinates(residue: Any | None) -> Any:
    import numpy as np

    coordinates = np.full((3, 3), np.nan, dtype=np.float32)
    if residue is None:
        return coordinates
    for index, atom_name in enumerate(ATOM_ORDER):
        if atom_name not in residue:
            continue
        atom = residue[atom_name]
        value = np.asarray(atom.get_coord(), dtype=np.float32)
        if value.shape == (3,) and np.isfinite(value).all():
            coordinates[index] = value
    return coordinates


def _full_chain_specs(chain: Any) -> tuple[ResidueSpec, ...]:
    return tuple(
        ResidueSpec(
            amino_acid=_residue_one_letter(residue),
            number=int(residue.id[1]),
            insertion_code=_insertion_code(residue.id[2]),
        )
        for residue in _observed_residues(chain)
    )


def _chain_backbone(
    chain: Any,
    specs: tuple[ResidueSpec, ...] | None,
) -> tuple[str, Any]:
    import numpy as np

    if specs is None:
        specs = _full_chain_specs(chain)
    lookup = _residue_lookup(chain)
    coordinates = np.full((len(specs), 3, 3), np.nan, dtype=np.float32)
    sequence: list[str] = []
    for index, spec in enumerate(specs):
        sequence.append(spec.amino_acid)
        residue = lookup.get((spec.number, spec.insertion_code))
        if residue is not None:
            observed = _residue_one_letter(residue)
            if observed != "X" and spec.amino_acid != "X" and observed != spec.amino_acid:
                raise StructureCacheError(
                    f"Residue identity mismatch at {chain.id}:{spec.number}{spec.insertion_code}"
                )
        coordinates[index] = _atom_coordinates(residue)
    return "".join(sequence), coordinates


def _frame_valid_mask(backbone: Any) -> Any:
    import numpy as np

    finite = np.isfinite(backbone).all(axis=(-1, -2))
    safe = np.nan_to_num(backbone, nan=0.0)
    n_atom, ca_atom, c_atom = safe[:, 0], safe[:, 1], safe[:, 2]
    x_raw = c_atom - ca_atom
    x_norm = np.linalg.norm(x_raw, axis=-1)
    x_axis = x_raw / np.maximum(x_norm[:, None], 1e-8)
    n_raw = n_atom - ca_atom
    y_raw = n_raw - (n_raw * x_axis).sum(axis=-1, keepdims=True) * x_axis
    y_norm = np.linalg.norm(y_raw, axis=-1)
    return finite & (x_norm > 1e-5) & (y_norm > 1e-5)


def _canonical_sequence(row: Mapping[str, Any], field: str) -> str:
    sequence = _string(row.get(field)).upper()
    if any(character not in "ACDEFGHIKLMNPQRSTVWYX" for character in sequence):
        raise StructureCacheError(f"Canonical sequence {field!r} contains invalid residues")
    return sequence


def _tensor_artifact(
    instance_id: str,
    role: str,
    metadata: Mapping[str, Any],
    canonical: Mapping[str, Any],
    prepared: Mapping[str, Any],
    cif_payload: bytes,
) -> dict[str, Any]:
    import numpy as np

    torch_module = _require_torch()
    try:
        structure = _bio_parser().get_structure(
            instance_id,
            io.StringIO(cif_payload.decode("utf-8", errors="strict")),
        )
        model = next(structure.get_models())
    except Exception as error:
        if isinstance(error, StructureCacheError):
            raise
        raise StructureCacheError("Cannot parse mmCIF structure") from error

    antibody_parts: list[Any] = []
    antibody_chain_ids: list[Any] = []
    antibody_sequence_parts: list[str] = []
    antibody_definitions = (
        ("Hchain", "VH_numbering_list", "vh_seq", 0),
        ("Lchain", "VL_numbering_list", "vl_seq", 1),
    )
    for chain_field, numbering_field, sequence_field, chain_index in antibody_definitions:
        expected = _canonical_sequence(canonical, sequence_field)
        chain_name = _string(metadata.get(chain_field))
        if not expected:
            if chain_name not in {"", "+"}:
                raise StructureCacheError(f"{sequence_field} is empty but its chain is present")
            continue
        if not chain_name or chain_name == "+":
            raise StructureCacheError(f"{sequence_field} is present but its chain is absent")
        chain = _chain_lookup(model, chain_name)
        observed_sequence, backbone = _chain_backbone(
            chain,
            _parse_numbering_list(_string(metadata.get(numbering_field))),
        )
        if observed_sequence != expected:
            raise StructureCacheError(
                f"New-release {sequence_field} differs from canonical sequence"
            )
        antibody_sequence_parts.append(observed_sequence)
        antibody_parts.append(backbone)
        antibody_chain_ids.append(
            np.full(len(observed_sequence), chain_index, dtype=np.int64)
        )
    if not antibody_parts:
        raise StructureCacheError("No antibody variable-domain residues were mapped")

    antigen_chains = _chain_names(metadata.get("agchains"))
    if not antigen_chains:
        raise StructureCacheError("No antigen chains are declared")
    raw_antigen_specs = _split_antigen_specs(
        _string(metadata.get("ag_index_list")),
        len(antigen_chains),
    )
    antigen_parts: list[Any] = []
    antigen_chain_ids: list[Any] = []
    antigen_sequence_parts: list[str] = []
    for chain_index, (chain_name, raw_spec) in enumerate(
        zip(antigen_chains, raw_antigen_specs)
    ):
        chain = _chain_lookup(model, chain_name)
        observed_sequence, backbone = _chain_backbone(
            chain,
            _parse_numbering_list(raw_spec),
        )
        antigen_sequence_parts.append(observed_sequence)
        antigen_parts.append(backbone)
        antigen_chain_ids.append(
            np.full(len(observed_sequence), chain_index, dtype=np.int64)
        )

    antibody_sequence = "".join(antibody_sequence_parts)
    antigen_sequence = "".join(antigen_sequence_parts)
    if antibody_sequence != _canonical_sequence(canonical, "vh_seq") + _canonical_sequence(
        canonical, "vl_seq"
    ):
        raise StructureCacheError("Concatenated antibody sequence differs from canonical")
    if antigen_sequence != _canonical_sequence(canonical, "ag_seq"):
        raise StructureCacheError("New-release antigen sequence differs from canonical sequence")

    antibody_backbone = np.concatenate(antibody_parts, axis=0)
    antigen_backbone = np.concatenate(antigen_parts, axis=0)
    antibody_ids = np.concatenate(antibody_chain_ids, axis=0)
    antigen_ids = np.concatenate(antigen_chain_ids, axis=0)
    expected_ab_length = int(prepared["ab_length"])
    expected_ag_length = int(prepared["ag_length"])
    if antibody_backbone.shape[0] != expected_ab_length:
        raise StructureCacheError("Antibody structure length differs from embedding/label axis")
    if antigen_backbone.shape[0] != expected_ag_length:
        raise StructureCacheError("Antigen structure length differs from embedding/label axis")
    antibody_valid = _frame_valid_mask(antibody_backbone)
    antigen_valid = _frame_valid_mask(antigen_backbone)
    if not antibody_valid.any() or not antigen_valid.any():
        raise StructureCacheError("One residue stream has no valid backbone frame")

    return {
        "format_version": STRUCTURE_ARTIFACT_FORMAT,
        "instance_id": instance_id,
        "split_role": role,
        "source_cif_sha256": sha256_bytes(cif_payload),
        "antibody_sequence_sha256": _verified_digest(
            str(prepared["ab_sequence_sha256"]),
            "ab_sequence_sha256",
        ),
        "antigen_sequence_sha256": _verified_digest(
            str(prepared["ag_sequence_sha256"]),
            "ag_sequence_sha256",
        ),
        "antibody_backbone": torch_module.from_numpy(antibody_backbone),
        "antigen_backbone": torch_module.from_numpy(antigen_backbone),
        "antibody_chain_ids": torch_module.from_numpy(antibody_ids),
        "antigen_chain_ids": torch_module.from_numpy(antigen_ids),
        "antibody_frame_valid_mask": torch_module.from_numpy(antibody_valid),
        "antigen_frame_valid_mask": torch_module.from_numpy(antigen_valid),
        "antibody_missing_frame_mask": torch_module.from_numpy(~antibody_valid),
        "antigen_missing_frame_mask": torch_module.from_numpy(~antigen_valid),
        "paratope_positive_count": int(prepared["paratope_positive_count"]),
        "epitope_positive_count": int(prepared["epitope_positive_count"]),
    }


def materialize_structure_cache(
    *,
    archive_path: Path,
    archive_sha256: str,
    split_path: Path,
    split_sha256: str,
    split_verification_path: Path,
    split_verification_sha256: str,
    canonical_manifest_path: Path,
    canonical_manifest_sha256: str,
    prepared_instances_path: Path,
    prepared_instances_sha256: str,
    output_root: Path,
    max_instances: int | None = None,
) -> dict[str, Any]:
    """Build a dual-authority, development-only structure cache atomically."""
    torch_module = _require_torch()
    authenticated = {
        "archive_sha256": authenticate_file(archive_path, archive_sha256, "archive"),
        "split_sha256": authenticate_file(split_path, split_sha256, "locked split"),
        "split_verification_sha256": authenticate_file(
            split_verification_path,
            split_verification_sha256,
            "locked split verification",
        ),
        "canonical_manifest_sha256": authenticate_file(
            canonical_manifest_path,
            canonical_manifest_sha256,
            "canonical manifest",
        ),
        "prepared_instances_sha256": authenticate_file(
            prepared_instances_path,
            prepared_instances_sha256,
            "prepared instances",
        ),
    }
    if max_instances is not None and (
        isinstance(max_instances, bool)
        or not isinstance(max_instances, int)
        or max_instances <= 0
    ):
        raise StructureCacheError("max_instances must be a positive integer or None")
    if output_root.exists() or output_root.is_symlink():
        raise StructureCacheError("Output root already exists")

    split_roles, split_payload = _load_split(split_path)
    split_verification = _validate_split_verification(
        split_verification_path,
        split_sha256=authenticated["split_sha256"],
        expected_instance_count=sum(len(values) for values in split_roles.values()),
    )
    native_rows = _archive_metadata_rows(archive_path)
    canonical_rows = _load_csv_index(canonical_manifest_path, "INSTANCE")
    prepared_rows = _load_prepared_index(prepared_instances_path)
    role_by_instance: dict[str, str] = {}
    exclusions: list[dict[str, str]] = []
    for role in ("train", "validation"):
        for instance_id in split_roles[role]:
            native = native_rows.get(instance_id)
            if native is None:
                exclusions.append(
                    {"instance_id": instance_id, "reason": "absent_from_new_release"}
                )
                continue
            if native.get("ab_ag_split") != NATIVE_TRAIN_ROLE:
                exclusions.append(
                    {
                        "instance_id": instance_id,
                        "reason": "new_release_native_role_not_train",
                    }
                )
                continue
            if instance_id not in canonical_rows:
                exclusions.append(
                    {"instance_id": instance_id, "reason": "absent_from_canonical_manifest"}
                )
                continue
            if instance_id not in prepared_rows:
                exclusions.append(
                    {"instance_id": instance_id, "reason": "absent_from_prepared_instances"}
                )
                continue
            role_by_instance[instance_id] = role
    if max_instances is not None:
        selected = sorted(role_by_instance)[:max_instances]
        role_by_instance = {identifier: role_by_instance[identifier] for identifier in selected}
    if not role_by_instance:
        raise StructureCacheError("No dual-authority instances remain")

    output_parent = output_root.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_parent)
    )
    records: list[dict[str, Any]] = []
    encountered: set[str] = set()
    try:
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.startswith("splits_final/"):
                    continue
                filename = member.name.rsplit("/", 1)[-1]
                if not filename.endswith(".cif"):
                    continue
                instance_id = filename[:-4]
                if instance_id not in role_by_instance:
                    continue
                if instance_id in encountered:
                    raise StructureCacheError(f"Duplicate archive member for {instance_id}")
                encountered.add(instance_id)
                binary = archive.extractfile(member)
                if binary is None:
                    exclusions.append(
                        {"instance_id": instance_id, "reason": "unreadable_cif_member"}
                    )
                    continue
                cif_payload = binary.read()
                try:
                    artifact = _tensor_artifact(
                        instance_id,
                        role_by_instance[instance_id],
                        native_rows[instance_id],
                        canonical_rows[instance_id],
                        prepared_rows[instance_id],
                        cif_payload,
                    )
                except StructureCacheError as error:
                    exclusions.append(
                        {"instance_id": instance_id, "reason": str(error)}
                    )
                    continue
                identifier_sha = hashlib.sha256(instance_id.encode("ascii")).hexdigest()
                relative_path = Path("artifacts") / identifier_sha[:2] / f"{identifier_sha}.pt"
                target = temporary_root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                torch_module.save(artifact, target)
                records.append(
                    {
                        "instance_id": instance_id,
                        "split_role": role_by_instance[instance_id],
                        "artifact_path": relative_path.as_posix(),
                        "artifact_sha256": sha256_file(target),
                        "artifact_size_bytes": target.stat().st_size,
                        "source_cif_sha256": artifact["source_cif_sha256"],
                        "antibody_sequence_sha256": artifact[
                            "antibody_sequence_sha256"
                        ],
                        "antigen_sequence_sha256": artifact[
                            "antigen_sequence_sha256"
                        ],
                        "antibody_length": int(artifact["antibody_backbone"].shape[0]),
                        "antigen_length": int(artifact["antigen_backbone"].shape[0]),
                        "antibody_valid_frame_count": int(
                            artifact["antibody_frame_valid_mask"].sum().item()
                        ),
                        "antigen_valid_frame_count": int(
                            artifact["antigen_frame_valid_mask"].sum().item()
                        ),
                        "paratope_positive_count": artifact[
                            "paratope_positive_count"
                        ],
                        "epitope_positive_count": artifact[
                            "epitope_positive_count"
                        ],
                    }
                )
        missing_members = sorted(set(role_by_instance) - encountered)
        exclusions.extend(
            {"instance_id": identifier, "reason": "missing_cif_member"}
            for identifier in missing_members
        )
        records.sort(key=lambda value: value["instance_id"])
        exclusions.sort(key=lambda value: (value["instance_id"], value["reason"]))
        counts = {
            role: sum(record["split_role"] == role for record in records)
            for role in ("train", "validation")
        }
        if max_instances is None and (counts["train"] == 0 or counts["validation"] == 0):
            raise StructureCacheError("Materialized cache must retain both split roles")
        manifest = {
            "format_version": STRUCTURE_CACHE_FORMAT,
            "authority": {
                **authenticated,
                "split_scientific_payload_sha256": split_payload.get(
                    "scientific_payload_sha256"
                ),
                "split_verification_format": split_verification.get("format_version"),
                "native_release_required_role": NATIVE_TRAIN_ROLE,
                "selection": "locked_development_membership_intersection_native_release_train",
                "sealed_test_cif_members_read": False,
            },
            "atom_order": list(ATOM_ORDER),
            "counts": {
                "records": len(records),
                "exclusions": len(exclusions),
                "by_role": counts,
            },
            "records": records,
            "exclusions": exclusions,
        }
        manifest_path = temporary_root / "STRUCTURE_CACHE_MANIFEST.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        manifest_sha256 = sha256_file(manifest_path)
        receipt = {
            "format_version": STRUCTURE_RECEIPT_FORMAT,
            "manifest_sha256": manifest_sha256,
            "record_count": len(records),
            "role_counts": counts,
            "claim_boundary": (
                "Development-only structural ablation cache; no confirmatory, "
                "held-out, external, clinical, or performance-improvement claim."
            ),
        }
        (temporary_root / "STRUCTURE_CACHE_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary_root, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


def _load_manifest(cache_root: Path) -> dict[str, Any]:
    path = cache_root / "STRUCTURE_CACHE_MANIFEST.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructureCacheError("Cannot read structure cache manifest") from error
    if payload.get("format_version") != STRUCTURE_CACHE_FORMAT:
        raise StructureCacheError("Unknown structure cache format")
    return payload


def _load_esm_embedding(path: Path, expected_sha256: str, expected_length: int) -> Any:
    torch_module = _require_torch()
    if not path.is_file() or path.is_symlink():
        raise StructureCacheError(f"Missing ESM embedding: {path}")
    payload = torch_module.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != "sppider_seq_embedding_v1"
        or payload.get("sequence_sha256") != expected_sha256
        or int(payload.get("sequence_length", -1)) != expected_length
    ):
        raise StructureCacheError("ESM embedding metadata differs from structure axis")
    chunks = payload.get("chunks")
    starts = payload.get("starts")
    ends = payload.get("ends")
    if not isinstance(chunks, list) or not chunks or not (
        isinstance(starts, list)
        and isinstance(ends, list)
        and len(chunks) == len(starts) == len(ends)
    ):
        raise StructureCacheError("ESM embedding chunks are malformed")
    embedding_dim = int(chunks[0].shape[-1])
    output = torch_module.zeros(expected_length, embedding_dim, dtype=torch_module.float32)
    counts = torch_module.zeros(expected_length, 1, dtype=torch_module.float32)
    for chunk, start, end in zip(chunks, starts, ends):
        start, end = int(start), int(end)
        if (
            not torch_module.is_tensor(chunk)
            or chunk.ndim != 2
            or tuple(chunk.shape) != (end - start, embedding_dim)
            or start < 0
            or end > expected_length
            or start >= end
        ):
            raise StructureCacheError("ESM embedding chunk bounds are invalid")
        output[start:end] += chunk.to(dtype=torch_module.float32)
        counts[start:end] += 1.0
    if bool((counts == 0).any()):
        raise StructureCacheError("ESM chunks do not cover the residue axis")
    return output / counts


class Sabdab2StructureEmbeddingDataset(Dataset):
    """Lazily join authenticated structures, frozen ESM features, and labels."""

    def __init__(
        self,
        cache_root: Path,
        embedding_root: Path,
        label_root: Path,
        *,
        split_role: str,
        max_antigen_length: int | None = None,
    ) -> None:
        _require_torch()
        if split_role not in {"train", "validation"}:
            raise ValueError("split_role must be train or validation")
        if max_antigen_length is not None and (
            isinstance(max_antigen_length, bool)
            or not isinstance(max_antigen_length, int)
            or max_antigen_length <= 0
        ):
            raise ValueError("max_antigen_length must be positive or None")
        self.cache_root = cache_root
        self.embedding_root = embedding_root
        self.label_root = label_root
        manifest = _load_manifest(cache_root)
        records = [
            record
            for record in manifest["records"]
            if record["split_role"] == split_role
            and (
                max_antigen_length is None
                or int(record["antigen_length"]) <= max_antigen_length
            )
        ]
        if not records:
            raise StructureCacheError("Dataset selection is empty")
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np

        torch_module = _require_torch()
        record = self.records[index]
        artifact_path = self.cache_root / record["artifact_path"]
        if sha256_file(artifact_path) != record["artifact_sha256"]:
            raise StructureCacheError("Structure artifact digest mismatch")
        artifact = torch_module.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        )
        if (
            artifact.get("format_version") != STRUCTURE_ARTIFACT_FORMAT
            or artifact.get("instance_id") != record["instance_id"]
            or artifact.get("split_role") != record["split_role"]
        ):
            raise StructureCacheError("Structure artifact metadata mismatch")
        ab_length = int(record["antibody_length"])
        ag_length = int(record["antigen_length"])
        ab_sha = record["antibody_sequence_sha256"]
        ag_sha = record["antigen_sequence_sha256"]
        antibody_features = _load_esm_embedding(
            self.embedding_root / ab_sha[:2] / f"{ab_sha}.pt",
            ab_sha,
            ab_length,
        )
        antigen_features = _load_esm_embedding(
            self.embedding_root / ag_sha[:2] / f"{ag_sha}.pt",
            ag_sha,
            ag_length,
        )
        label_path = self.label_root / f"{record['instance_id']}.npz"
        if not label_path.is_file() or label_path.is_symlink():
            raise StructureCacheError(f"Missing residue labels: {label_path}")
        with np.load(label_path, allow_pickle=False) as labels:
            if set(labels.files) != {"ab_labels", "ag_labels"}:
                raise StructureCacheError("Residue label schema mismatch")
            antibody_labels = torch_module.from_numpy(
                labels["ab_labels"].astype(np.float32, copy=True)
            )
            antigen_labels = torch_module.from_numpy(
                labels["ag_labels"].astype(np.float32, copy=True)
            )
        if tuple(antibody_labels.shape) != (ab_length,) or tuple(
            antigen_labels.shape
        ) != (ag_length,):
            raise StructureCacheError("Residue labels do not align with structure axes")
        if not bool(((antibody_labels == 0) | (antibody_labels == 1)).all()) or not bool(
            ((antigen_labels == 0) | (antigen_labels == 1)).all()
        ):
            raise StructureCacheError("Residue labels must be binary")
        return {
            "instance_id": record["instance_id"],
            "antibody_features": antibody_features,
            "antigen_features": antigen_features,
            "antibody_backbone": artifact["antibody_backbone"],
            "antigen_backbone": artifact["antigen_backbone"],
            "antibody_chain_ids": artifact["antibody_chain_ids"],
            "antigen_chain_ids": artifact["antigen_chain_ids"],
            "antibody_frame_valid_mask": artifact["antibody_frame_valid_mask"],
            "antigen_frame_valid_mask": artifact["antigen_frame_valid_mask"],
            "antibody_missing_frame_mask": artifact["antibody_missing_frame_mask"],
            "antigen_missing_frame_mask": artifact["antigen_missing_frame_mask"],
            "antibody_labels": antibody_labels,
            "antigen_labels": antigen_labels,
        }


class Sabdab2StructureBatchCollator:
    """Pad two variable-length residue streams without fabricating frames."""

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        torch_module = _require_torch()
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        batch_size = len(examples)
        max_ab = max(int(item["antibody_features"].shape[0]) for item in examples)
        max_ag = max(int(item["antigen_features"].shape[0]) for item in examples)
        ab_dim = int(examples[0]["antibody_features"].shape[1])
        ag_dim = int(examples[0]["antigen_features"].shape[1])

        def features(length: int, channels: int) -> Any:
            return torch_module.zeros(batch_size, length, channels, dtype=torch_module.float32)

        def backbones(length: int) -> Any:
            return torch_module.full(
                (batch_size, length, 3, 3),
                float("nan"),
                dtype=torch_module.float32,
            )

        antibody_features = features(max_ab, ab_dim)
        antigen_features = features(max_ag, ag_dim)
        antibody_backbone = backbones(max_ab)
        antigen_backbone = backbones(max_ag)
        antibody_mask = torch_module.zeros(batch_size, max_ab, dtype=torch_module.bool)
        antigen_mask = torch_module.zeros(batch_size, max_ag, dtype=torch_module.bool)
        antibody_chain_ids = torch_module.full((batch_size, max_ab), -1, dtype=torch_module.long)
        antigen_chain_ids = torch_module.full((batch_size, max_ag), -1, dtype=torch_module.long)
        antibody_frame_valid = torch_module.zeros(batch_size, max_ab, dtype=torch_module.bool)
        antigen_frame_valid = torch_module.zeros(batch_size, max_ag, dtype=torch_module.bool)
        antibody_missing = torch_module.zeros(batch_size, max_ab, dtype=torch_module.bool)
        antigen_missing = torch_module.zeros(batch_size, max_ag, dtype=torch_module.bool)
        antibody_labels = torch_module.zeros(batch_size, max_ab, dtype=torch_module.float32)
        antigen_labels = torch_module.zeros(batch_size, max_ag, dtype=torch_module.float32)
        instance_ids: list[str] = []
        for batch_index, item in enumerate(examples):
            ab_length = int(item["antibody_features"].shape[0])
            ag_length = int(item["antigen_features"].shape[0])
            if int(item["antibody_features"].shape[1]) != ab_dim or int(
                item["antigen_features"].shape[1]
            ) != ag_dim:
                raise StructureCacheError("Embedding dimensions differ within a batch")
            instance_ids.append(str(item["instance_id"]))
            antibody_features[batch_index, :ab_length] = item["antibody_features"]
            antigen_features[batch_index, :ag_length] = item["antigen_features"]
            antibody_backbone[batch_index, :ab_length] = item["antibody_backbone"]
            antigen_backbone[batch_index, :ag_length] = item["antigen_backbone"]
            antibody_mask[batch_index, :ab_length] = True
            antigen_mask[batch_index, :ag_length] = True
            antibody_chain_ids[batch_index, :ab_length] = item["antibody_chain_ids"]
            antigen_chain_ids[batch_index, :ag_length] = item["antigen_chain_ids"]
            antibody_frame_valid[batch_index, :ab_length] = item[
                "antibody_frame_valid_mask"
            ]
            antigen_frame_valid[batch_index, :ag_length] = item[
                "antigen_frame_valid_mask"
            ]
            antibody_missing[batch_index, :ab_length] = item[
                "antibody_missing_frame_mask"
            ]
            antigen_missing[batch_index, :ag_length] = item[
                "antigen_missing_frame_mask"
            ]
            antibody_labels[batch_index, :ab_length] = item["antibody_labels"]
            antigen_labels[batch_index, :ag_length] = item["antigen_labels"]
        if not torch_module.equal(antibody_frame_valid | antibody_missing, antibody_mask):
            raise StructureCacheError("Antibody frame masks do not partition valid residues")
        if not torch_module.equal(antigen_frame_valid | antigen_missing, antigen_mask):
            raise StructureCacheError("Antigen frame masks do not partition valid residues")
        return {
            "instance_ids": tuple(instance_ids),
            "antibody_features": antibody_features,
            "antigen_features": antigen_features,
            "antibody_backbone": antibody_backbone,
            "antigen_backbone": antigen_backbone,
            "antibody_mask": antibody_mask,
            "antigen_mask": antigen_mask,
            "antibody_chain_ids": antibody_chain_ids,
            "antigen_chain_ids": antigen_chain_ids,
            "antibody_frame_valid_mask": antibody_frame_valid,
            "antigen_frame_valid_mask": antigen_frame_valid,
            "antibody_missing_frame_mask": antibody_missing,
            "antigen_missing_frame_mask": antigen_missing,
            "antibody_labels": antibody_labels,
            "antigen_labels": antigen_labels,
        }


class IntraChainResidueClassifier(nn.Module if nn is not None else object):
    """Self-attention-only paratope/epitope classifier for geometry ablations."""

    def __init__(
        self,
        *,
        antibody_input_dim: int = 1280,
        antigen_input_dim: int = 1280,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        geometry_feature_mode: str = "full_pose",
        num_rbf: int = 16,
        neighbor_k: int | None = 32,
        neighbor_radius: float | None = 16.0,
        attention_topology: str = "spatial",
        dropout: float = 0.1,
    ) -> None:
        _require_torch()
        super().__init__()
        if d_model <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        self.antibody_projection = nn.Sequential(
            nn.LayerNorm(antibody_input_dim),
            nn.Linear(antibody_input_dim, d_model),
        )
        self.antigen_projection = nn.Sequential(
            nn.LayerNorm(antigen_input_dim),
            nn.Linear(antigen_input_dim, d_model),
        )
        self.encoder = DualIntraChainGeometryEncoder(
            d_model,
            num_heads,
            num_layers,
            geometry_feature_mode=geometry_feature_mode,
            num_rbf=num_rbf,
            neighbor_k=neighbor_k,
            neighbor_radius=neighbor_radius,
            attention_topology=attention_topology,
            dropout=dropout,
        )
        self.antibody_norm = nn.LayerNorm(d_model)
        self.antigen_norm = nn.LayerNorm(d_model)
        self.paratope_head = nn.Linear(d_model, 1)
        self.epitope_head = nn.Linear(d_model, 1)
        nn.init.constant_(self.paratope_head.bias, -2.5)
        nn.init.constant_(self.epitope_head.bias, -2.5)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        antibody_mask = batch["antibody_mask"]
        antigen_mask = batch["antigen_mask"]
        for partner, residue_mask in (
            ("antibody", antibody_mask),
            ("antigen", antigen_mask),
        ):
            valid = batch[f"{partner}_frame_valid_mask"]
            missing = batch[f"{partner}_missing_frame_mask"]
            if not torch.equal(valid | missing, residue_mask) or bool(
                (valid & missing).any()
            ):
                raise StructureCacheError(
                    f"{partner} valid/missing masks do not partition real residues"
                )
            derived = build_backbone_frames(
                batch[f"{partner}_backbone"], residue_mask
            ).valid_mask
            if not torch.equal(derived, valid):
                raise StructureCacheError(
                    f"{partner} frame mask differs from the N/CA/C geometry"
                )
        antibody = self.antibody_projection(batch["antibody_features"])
        antigen = self.antigen_projection(batch["antigen_features"])
        antibody, antigen, _ = self.encoder(
            antibody,
            antigen,
            antibody_mask,
            antigen_mask,
            batch["antibody_backbone"],
            batch["antigen_backbone"],
            antibody_chain_ids=batch["antibody_chain_ids"],
            antigen_chain_ids=batch["antigen_chain_ids"],
        )
        antibody = self.antibody_norm(antibody).masked_fill(
            ~antibody_mask[..., None], 0.0
        )
        antigen = self.antigen_norm(antigen).masked_fill(
            ~antigen_mask[..., None], 0.0
        )
        return {
            "paratope_logits": self.paratope_head(antibody).squeeze(-1).masked_fill(
                ~antibody_mask, 0.0
            ),
            "epitope_logits": self.epitope_head(antigen).squeeze(-1).masked_fill(
                ~antigen_mask, 0.0
            ),
            "antibody_mask": antibody_mask,
            "antigen_mask": antigen_mask,
        }


__all__ = [
    "ATOM_ORDER",
    "IntraChainResidueClassifier",
    "Sabdab2StructureBatchCollator",
    "Sabdab2StructureEmbeddingDataset",
    "StructureCacheError",
    "authenticate_file",
    "materialize_structure_cache",
    "sha256_file",
]
