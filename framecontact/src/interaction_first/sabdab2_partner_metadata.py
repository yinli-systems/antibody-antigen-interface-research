"""Label-free partner metadata with monomer antigen solvent exposure.

The base table is derived only from authenticated sequences, structural
metadata, and antigen coordinates.  Antibody atoms are removed before SASA is
computed, so ``surface_residue_count`` is not a holo-interface label.  A
second, cheap step binds the invariant base table to one frozen split.
"""

from __future__ import annotations

import copy
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any, Mapping

from .sabdab2_antigen_export import _metadata_index, _prepared_index
from .sabdab2_structure import (
    StructureCacheError,
    _bio_parser,
    _chain_lookup,
    _chain_names,
    _full_chain_specs,
    _parse_numbering_list,
    _residue_lookup,
    _split_antigen_specs,
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)


PARTNER_METADATA_BASE_FORMAT = "sabdab2_partner_metadata_base_v1"
PARTNER_METADATA_SPLIT_FORMAT = "sabdab2_partner_metadata_split_v1"
RSA_THRESHOLD = 0.20
SASA_PROBE_RADIUS_ANGSTROM = 1.40
SASA_POINTS_PER_ATOM = 100
TIEN_2013_THEORETICAL_MAX_ASA = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "E": 223.0,
    "Q": 225.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
}
BASE_COLUMNS = (
    "instance_id",
    "antigen_cluster_id",
    "antigen_sequence_sha256",
    "antigen_length",
    "surface_residue_count",
    "surface_observed_residue_count",
    "antigen_class",
    "antibody_cluster_id",
    "antibody_sequence_sha256",
    "antibody_length",
    "cdr_residue_count",
    "antibody_class",
    "documented_cognate_group_id",
)


def _clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.casefold() == "nan" else text


def _antibody_class(value: Any) -> str:
    normalized = _clean(value).upper()
    if normalized in {"FAB", "FV"}:
        return "conventional_fv"
    if normalized == "SD-H":
        return "single_domain_heavy"
    raise StructureCacheError(f"Unsupported antibody type for partner matching: {value!r}")


def _antigen_surface_profile(
    identifier: str,
    metadata: Mapping[str, Any],
    cif_payload: bytes,
    *,
    rsa_threshold: float = RSA_THRESHOLD,
    probe_radius: float = SASA_PROBE_RADIUS_ANGSTROM,
    n_points: int = SASA_POINTS_PER_ATOM,
) -> tuple[str, int, int]:
    """Return canonical antigen sequence, exposed count, and observed count."""
    try:
        from Bio.PDB.Chain import Chain
        from Bio.PDB.Model import Model
        from Bio.PDB.SASA import ShrakeRupley
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("Biopython is required for antigen SASA") from error
    if not 0 < rsa_threshold < 1:
        raise ValueError("rsa_threshold must lie in (0, 1)")
    if probe_radius <= 0 or n_points <= 0:
        raise ValueError("SASA probe radius and point count must be positive")

    structure = _bio_parser().get_structure(
        identifier, io.StringIO(cif_payload.decode("utf-8", errors="strict"))
    )
    source_model = next(structure.get_models())
    antigen_model = Model(0)
    chain_names = _chain_names(metadata.get("agchains"))
    raw_specs = _split_antigen_specs(
        _clean(metadata.get("ag_index_list")), len(chain_names)
    )
    if not chain_names:
        raise StructureCacheError("No antigen chains are declared")

    sequence: list[str] = []
    ordered_residues: list[tuple[str, Any | None]] = []
    for chain_name, raw_spec in zip(chain_names, raw_specs):
        source_chain = _chain_lookup(source_model, chain_name)
        specs = _parse_numbering_list(raw_spec)
        if specs is None:
            specs = _full_chain_specs(source_chain)
        source_lookup = _residue_lookup(source_chain)
        antigen_chain = Chain(chain_name)
        antigen_model.add(antigen_chain)
        for spec in specs:
            sequence.append(spec.amino_acid)
            source_residue = source_lookup.get((spec.number, spec.insertion_code))
            if source_residue is None:
                ordered_residues.append((spec.amino_acid, None))
                continue
            copied_residue = copy.deepcopy(source_residue)
            antigen_chain.add(copied_residue)
            ordered_residues.append((spec.amino_acid, copied_residue))

    ShrakeRupley(probe_radius=probe_radius, n_points=n_points).compute(
        antigen_model, level="R"
    )
    observed = 0
    exposed = 0
    for amino_acid, residue in ordered_residues:
        if residue is None:
            continue
        maximum = TIEN_2013_THEORETICAL_MAX_ASA.get(amino_acid)
        if maximum is None:
            raise StructureCacheError(
                f"No Tien-2013 MaxASA for antigen residue {amino_acid!r}"
            )
        observed += 1
        exposed += float(residue.sasa) / maximum >= rsa_threshold
    if observed == 0 or exposed == 0:
        raise StructureCacheError("Antigen SASA produced no observed exposed residues")
    return "".join(sequence), int(exposed), observed


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _partner_metadata_row(
    identifier: str,
    source: Mapping[str, Any],
    expected: Mapping[str, Any],
    cif_payload: bytes,
) -> dict[str, Any]:
    sequence, surface_count, observed_count = _antigen_surface_profile(
        identifier, source, cif_payload
    )
    observed_sha = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    if observed_sha != str(expected["ag_sequence_sha256"]):
        raise StructureCacheError(f"{identifier}: antigen sequence digest differs")
    if len(sequence) != int(expected["ag_length"]):
        raise StructureCacheError(f"{identifier}: antigen length differs")
    cdr_count = len(_clean(source.get("concatCDRseq")))
    antibody_length = int(expected["ab_length"])
    if cdr_count <= 0 or cdr_count > antibody_length:
        raise StructureCacheError(f"{identifier}: invalid CDR residue count")
    antigen_cluster = _clean(source.get("agclusters"))
    antibody_cluster = _clean(source.get("ab_cluster"))
    cognate_group = _clean(source.get("PDB_ID"))
    antigen_class = _clean(source.get("agtypes")).casefold()
    if not all((antigen_cluster, antibody_cluster, cognate_group, antigen_class)):
        raise StructureCacheError(f"{identifier}: incomplete partner metadata")
    return {
        "instance_id": identifier,
        "antigen_cluster_id": f"ag:{antigen_cluster}",
        "antigen_sequence_sha256": observed_sha,
        "antigen_length": len(sequence),
        "surface_residue_count": surface_count,
        "surface_observed_residue_count": observed_count,
        "antigen_class": antigen_class,
        "antibody_cluster_id": f"ab:{antibody_cluster}",
        "antibody_sequence_sha256": str(expected["ab_sequence_sha256"]),
        "antibody_length": antibody_length,
        "cdr_residue_count": cdr_count,
        "antibody_class": _antibody_class(source.get("type")),
        "documented_cognate_group_id": cognate_group,
    }


def materialize_partner_metadata_base(
    *,
    archive_path: Path,
    archive_sha256: str,
    metadata_path: Path,
    metadata_sha256: str,
    prepared_path: Path,
    prepared_sha256: str,
    output_root: Path,
    workers: int = 1,
) -> dict[str, Any]:
    """Compute an atomic, role-independent partner matching table."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    authority = {
        "archive_sha256": authenticate_file(archive_path, archive_sha256, "structure archive"),
        "metadata_sha256": authenticate_file(metadata_path, metadata_sha256, "metadata"),
        "prepared_sha256": authenticate_file(prepared_path, prepared_sha256, "prepared cohort"),
        "task_labels_read": False,
    }
    metadata = _metadata_index(metadata_path)
    prepared = _prepared_index(prepared_path)
    if set(prepared) - set(metadata):
        raise StructureCacheError("Prepared cohort contains instances absent from metadata")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite partner metadata: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    rows: list[dict[str, Any]] = []
    encountered: set[str] = set()
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    pending: dict[Any, str] = {}

    def collect_completed() -> None:
        if not pending:
            return
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            identifier = pending.pop(future)
            try:
                rows.append(future.result())
            except BaseException as error:
                raise StructureCacheError(
                    f"Partner metadata worker failed for {identifier}"
                ) from error

    try:
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".cif"):
                    continue
                identifier = Path(member.name).stem
                if identifier not in prepared:
                    continue
                if identifier in encountered:
                    raise StructureCacheError(f"Duplicate structure member: {identifier}")
                encountered.add(identifier)
                binary = archive.extractfile(member)
                if binary is None:
                    raise StructureCacheError(f"Cannot read structure member: {identifier}")
                source = metadata[identifier]
                expected = prepared[identifier]
                payload = binary.read()
                if executor is None:
                    rows.append(
                        _partner_metadata_row(identifier, source, expected, payload)
                    )
                else:
                    future = executor.submit(
                        _partner_metadata_row,
                        identifier,
                        source,
                        expected,
                        payload,
                    )
                    pending[future] = identifier
                    if len(pending) >= workers * 2:
                        collect_completed()
        while pending:
            collect_completed()
        missing = sorted(set(prepared) - encountered)
        if missing:
            raise StructureCacheError(
                f"Structure archive lacks {len(missing)} prepared instances: {missing[:5]}"
            )
        rows.sort(key=lambda row: str(row["instance_id"]))
        table_path = temporary / "PARTNER_METADATA_BASE.csv"
        _write_csv(table_path, rows, BASE_COLUMNS)
        receipt = {
            "format_version": PARTNER_METADATA_BASE_FORMAT,
            "authority": authority,
            "instance_count": len(rows),
            "metadata_sha256": sha256_file(table_path),
            "runtime_workers": workers,
            "surface_definition": {
                "entity": "antigen monomer; antibody atoms removed",
                "algorithm": "Shrake-Rupley",
                "probe_radius_angstrom": SASA_PROBE_RADIUS_ANGSTROM,
                "points_per_atom": SASA_POINTS_PER_ATOM,
                "rsa_threshold": RSA_THRESHOLD,
                "normalization": "Tien et al. 2013 theoretical ALLOWED MaxASA",
                "doi": "10.1371/journal.pone.0080635",
            },
            "claim_boundary": (
                "Label-free partner matching metadata only; monomer surface exposure is "
                "not an interface label or performance result."
            ),
        }
        (temporary / "PARTNER_METADATA_BASE_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def bind_partner_metadata_to_split(
    *,
    base_path: Path,
    base_sha256: str,
    split_path: Path,
    split_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """Bind a base metadata authority to exactly one frozen split."""
    authenticated = {
        "base_sha256": authenticate_file(base_path, base_sha256, "partner metadata base"),
        "split_sha256": authenticate_file(split_path, split_sha256, "frozen split"),
        "task_labels_read": False,
    }
    with base_path.open("r", encoding="utf-8", newline="") as handle:
        base_reader = csv.DictReader(handle)
        if base_reader.fieldnames is None or not set(BASE_COLUMNS).issubset(
            base_reader.fieldnames
        ):
            raise StructureCacheError("Partner metadata base schema is incomplete")
        base_rows = {row["instance_id"]: row for row in base_reader}
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        split_reader = csv.DictReader(handle)
        if split_reader.fieldnames is None or not {"instance_id", "role"}.issubset(
            split_reader.fieldnames
        ):
            raise StructureCacheError("Frozen split must contain instance_id and role")
        roles: dict[str, str] = {}
        for row in split_reader:
            identifier, role = str(row["instance_id"]).strip(), str(row["role"]).strip()
            if identifier in roles or role not in {"train", "validation", "test"}:
                raise StructureCacheError("Frozen split identifiers/roles are invalid")
            roles[identifier] = role
    if set(base_rows) != set(roles):
        raise StructureCacheError("Partner metadata and frozen split cohorts differ")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite bound partner metadata: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        rows = [
            {"instance_id": identifier, "role": roles[identifier], **base_rows[identifier]}
            for identifier in sorted(base_rows)
        ]
        columns = ("instance_id", "role", *BASE_COLUMNS[1:])
        table_path = temporary / "PARTNER_METADATA.csv"
        _write_csv(table_path, rows, columns)
        role_counts = {
            role: sum(row["role"] == role for row in rows)
            for role in ("train", "validation", "test")
        }
        if any(value == 0 for value in role_counts.values()):
            raise StructureCacheError("Every frozen split role must be nonempty")
        receipt = {
            "format_version": PARTNER_METADATA_SPLIT_FORMAT,
            "authority": authenticated,
            "instance_count": len(rows),
            "role_counts": role_counts,
            "metadata_sha256": sha256_file(table_path),
            "claim_boundary": "Split-bound label-free partner metadata only.",
        }
        (temporary / "PARTNER_METADATA_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def project_split_bound_partner_metadata_to_base(
    *,
    bound_path: Path,
    bound_sha256: str,
    bound_receipt_path: Path,
    bound_receipt_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """Remove only a frozen role column from authenticated label-free metadata.

    This recovery path is useful when the invariant base table is no longer
    available but a receipt-bound split table is.  It accepts exactly the
    label-free split schema produced by :func:`bind_partner_metadata_to_split`;
    task columns or arbitrary extra columns are rejected before projection.
    """

    authenticated = {
        "bound_metadata_sha256": authenticate_file(
            bound_path, bound_sha256, "split-bound partner metadata"
        ),
        "bound_receipt_sha256": authenticate_file(
            bound_receipt_path,
            bound_receipt_sha256,
            "split-bound partner metadata receipt",
        ),
        "task_labels_read": False,
        "predictions_read": False,
    }
    try:
        bound_receipt = json.loads(bound_receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructureCacheError("Cannot read split-bound metadata receipt") from error
    if (
        bound_receipt.get("format_version") != PARTNER_METADATA_SPLIT_FORMAT
        or bound_receipt.get("metadata_sha256") != bound_sha256
        or bound_receipt.get("authority", {}).get("task_labels_read") is not False
    ):
        raise StructureCacheError("Split-bound metadata receipt is not eligible")

    expected_columns = {"role", *BASE_COLUMNS}
    try:
        with bound_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(reader.fieldnames) != expected_columns:
                raise StructureCacheError(
                    "Split-bound metadata must contain exactly the label-free schema"
                )
            rows: list[dict[str, str]] = []
            identifiers: set[str] = set()
            role_counts = {role: 0 for role in ("train", "validation", "test")}
            for source in reader:
                identifier = str(source.get("instance_id", "")).strip()
                role = str(source.get("role", "")).strip()
                if not identifier or identifier in identifiers or role not in role_counts:
                    raise StructureCacheError(
                        "Split-bound metadata contains invalid identifiers or roles"
                    )
                identifiers.add(identifier)
                role_counts[role] += 1
                rows.append(
                    {name: str(source.get(name, "")).strip() for name in BASE_COLUMNS}
                )
    except OSError as error:
        raise StructureCacheError("Cannot read split-bound partner metadata") from error
    if not rows or any(value == 0 for value in role_counts.values()):
        raise StructureCacheError("Every source role must be nonempty")
    rows.sort(key=lambda row: row["instance_id"])

    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite partner metadata base: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        table_path = temporary / "PARTNER_METADATA_BASE.csv"
        _write_csv(table_path, rows, BASE_COLUMNS)
        receipt = {
            "format_version": PARTNER_METADATA_BASE_FORMAT,
            "authority": authenticated,
            "instance_count": len(rows),
            "metadata_sha256": sha256_file(table_path),
            "source_role_counts": role_counts,
            "projection": {
                "removed_columns": ["role"],
                "preserved_columns": list(BASE_COLUMNS),
                "values_recomputed": False,
            },
            "claim_boundary": (
                "Role-column projection of authenticated label-free partner metadata "
                "only; no task label, prediction, or performance result was read."
            ),
        }
        (temporary / "PARTNER_METADATA_BASE_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, output_root)
        return {**receipt, "output_root": str(output_root)}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "PARTNER_METADATA_BASE_FORMAT",
    "PARTNER_METADATA_SPLIT_FORMAT",
    "bind_partner_metadata_to_split",
    "materialize_partner_metadata_base",
    "project_split_bound_partner_metadata_to_base",
]
