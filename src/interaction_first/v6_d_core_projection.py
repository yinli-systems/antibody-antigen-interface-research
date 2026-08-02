"""Build an exact solver-neutral projection for a diagnosed V6-D family core.

The exporter keeps every source Boolean variable and selects only source rows
whose recorded constraint family belongs to the authenticated core. It proves
row-by-row subset containment before writing a create-only bundle. The result
is diagnostic evidence only and grants no split, training, selection, outer
evaluation, or test authority.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence

from .v6_cp_sat_formulation import DirectCPModel
from .v6_joint_ir import build_v6_joint_ir_from_direct_cp_sat
from .v6_linear_projection_bundle import export_linear_projection_bundle


RECEIPT_FORMAT = "v6_d_exact_family_core_projection_receipt_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class V6DCoreProjectionError(ValueError):
    """Raised when an exact family-core projection cannot be established."""


def canonical_json_bytes(payload: object) -> bytes:
    """Encode finite canonical ASCII JSON with one trailing newline."""
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise V6DCoreProjectionError(
            "core projection metadata must be finite ASCII JSON"
        ) from error


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_proto(model: Any) -> str:
    try:
        with tempfile.TemporaryDirectory(prefix="v6-d-core-proto-") as directory:
            path = Path(directory) / "model.binpb"
            if not model.export_to_file(str(path)):
                raise V6DCoreProjectionError("CP-SAT model binary export failed")
            payload = path.read_bytes()
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise V6DCoreProjectionError(
            "CP-SAT model cannot be serialized deterministically"
        ) from error
    if not payload:
        raise V6DCoreProjectionError("CP-SAT model serialization is empty")
    return hashlib.sha256(payload).hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(
            os,
            "O_CLOEXEC",
            0,
        )
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_indices(path: Path, indices: Sequence[int]) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(
            os,
            "O_CLOEXEC",
            0,
        )
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        for reduced_index, source_index in enumerate(indices):
            payload = canonical_json_bytes(
                {
                    "reduced_row_index": reduced_index,
                    "source_row_index": source_index,
                }
            )
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _dimensions(model: Any) -> dict[str, int]:
    proto = model.Proto()
    if any(
        row.enforcement_literal or not row.has_linear() for row in proto.constraints
    ):
        raise V6DCoreProjectionError(
            "core projection requires unconditional linear source rows"
        )
    return {
        "columns": len(proto.variables),
        "rows": len(proto.constraints),
        "nonzeros": sum(len(row.linear.vars) for row in proto.constraints),
    }


def _linear_row_payload(row: Any) -> dict[str, object]:
    if row.enforcement_literal or not row.has_linear():
        raise V6DCoreProjectionError(
            "core projection requires unconditional linear source rows"
        )
    return {
        "name": str(row.name),
        "variables": [int(value) for value in row.linear.vars],
        "coefficients": [int(value) for value in row.linear.coeffs],
        "domain": [int(value) for value in row.linear.domain],
    }


def _normalized_families(
    source_families: Sequence[str],
    active_families: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(active_families, (str, bytes)):
        raise V6DCoreProjectionError("active families must be a sequence")
    normalized = tuple(sorted(active_families))
    if not normalized or len(normalized) != len(set(normalized)):
        raise V6DCoreProjectionError(
            "active families must be non-empty, unique, and canonical"
        )
    if any(
        not isinstance(family, str)
        or not family
        or not family.isascii()
        or family.strip() != family
        for family in normalized
    ):
        raise V6DCoreProjectionError("active family identifier is malformed")
    unknown = set(normalized) - set(source_families)
    if unknown:
        raise V6DCoreProjectionError(f"active families are unknown: {sorted(unknown)}")
    return normalized


@dataclass(frozen=True)
class ExactFamilyCoreProjection:
    """One exact row subset with source-row containment provenance."""

    direct_model: DirectCPModel
    active_families: tuple[str, ...]
    source_row_indices: tuple[int, ...]
    full_proto_sha256: str
    reduced_proto_sha256: str
    full_dimensions: Mapping[str, int]
    reduced_dimensions: Mapping[str, int]
    row_subset_containment_passed: bool


def select_exact_family_core(
    direct_model: DirectCPModel,
    active_families: Sequence[str],
) -> ExactFamilyCoreProjection:
    """Clone the source model and retain exactly the requested family rows."""
    if not isinstance(direct_model, DirectCPModel):
        raise V6DCoreProjectionError("direct_model must be a DirectCPModel")
    if (
        not isinstance(direct_model.formulation_sha256, str)
        or _SHA256.fullmatch(direct_model.formulation_sha256) is None
    ):
        raise V6DCoreProjectionError("source formulation digest is malformed")
    source_proto = direct_model.model.Proto()
    source_families = tuple(direct_model.constraint_families)
    if len(source_families) != len(source_proto.constraints):
        raise V6DCoreProjectionError("source family sequence does not cover every row")
    observed_counts = {
        family: source_families.count(family) for family in set(source_families)
    }
    if observed_counts != dict(direct_model.constraint_family_counts):
        raise V6DCoreProjectionError("source family counts differ")
    normalized = _normalized_families(source_families, active_families)
    selected = tuple(
        index
        for index, family in enumerate(source_families)
        if family in set(normalized)
    )
    if not selected:
        raise V6DCoreProjectionError("selected family core contains no rows")

    reduced_model = direct_model.model.clone()
    reduced_proto = reduced_model.Proto()
    selected_rows = [
        copy.deepcopy(source_proto.constraints[index]) for index in selected
    ]
    reduced_proto.constraints.clear()
    reduced_proto.constraints.extend(selected_rows)
    reduced_families = tuple(source_families[index] for index in selected)
    reduced_counts = tuple(
        sorted(
            (family, reduced_families.count(family)) for family in set(reduced_families)
        )
    )
    reduced_identity = {
        "active_families": list(normalized),
        "full_formulation_sha256": direct_model.formulation_sha256,
        "source_row_indices_sha256": _sha256_json(list(selected)),
    }
    reduced_direct = DirectCPModel(
        model=reduced_model,
        variables=direct_model.variables,
        authority_payload_sha256=direct_model.authority_payload_sha256,
        constraint_family_counts=reduced_counts,
        constraint_families=reduced_families,
        formulation_sha256=_sha256_json(reduced_identity),
    )
    containment = len(reduced_proto.constraints) == len(selected) and all(
        _linear_row_payload(reduced_proto.constraints[reduced_index])
        == _linear_row_payload(source_proto.constraints[source_index])
        for reduced_index, source_index in enumerate(selected)
    )
    if not containment:
        raise V6DCoreProjectionError("row-subset containment audit failed")
    if len(reduced_proto.variables) != len(source_proto.variables):
        raise V6DCoreProjectionError("core projection changed variable domains")
    if any(
        (
            str(reduced_proto.variables[index].name),
            tuple(int(value) for value in reduced_proto.variables[index].domain),
        )
        != (
            str(source_proto.variables[index].name),
            tuple(int(value) for value in source_proto.variables[index].domain),
        )
        for index in range(len(source_proto.variables))
    ):
        raise V6DCoreProjectionError("core projection changed variable bytes")
    return ExactFamilyCoreProjection(
        direct_model=reduced_direct,
        active_families=normalized,
        source_row_indices=selected,
        full_proto_sha256=_sha256_proto(direct_model.model),
        reduced_proto_sha256=_sha256_proto(reduced_model),
        full_dimensions=_dimensions(direct_model.model),
        reduced_dimensions=_dimensions(reduced_model),
        row_subset_containment_passed=True,
    )


def export_exact_family_core_bundle(
    direct_model: DirectCPModel,
    active_families: Sequence[str],
    *,
    solver_policy_sha256: str,
    direct_formulation_source_sha256: str,
    exact_oracle_source_sha256: str,
    diagnostic_receipt_sha256: str,
    output_directory: Path,
) -> dict[str, object]:
    """Export one create-only exact core bundle and containment receipt."""
    for value, label in (
        (solver_policy_sha256, "solver policy"),
        (direct_formulation_source_sha256, "formulation source"),
        (exact_oracle_source_sha256, "exact oracle source"),
        (diagnostic_receipt_sha256, "diagnostic receipt"),
    ):
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise V6DCoreProjectionError(f"{label} digest is malformed")
    output = Path(os.path.abspath(os.path.expanduser(str(output_directory))))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only V6-D core output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise V6DCoreProjectionError("V6-D core output parent is invalid")
    os.mkdir(output, 0o700)
    try:
        projection = select_exact_family_core(direct_model, active_families)
        ir = build_v6_joint_ir_from_direct_cp_sat(
            projection.direct_model,
            solver_policy_sha256=solver_policy_sha256,
            direct_formulation_source_sha256=(direct_formulation_source_sha256),
            exact_oracle_source_sha256=exact_oracle_source_sha256,
        )
        bundle = export_linear_projection_bundle(ir, output / "bundle")
        index_path = output / "source_row_indices.jsonl"
        _write_indices(index_path, projection.source_row_indices)
        family_counts = dict(projection.direct_model.constraint_family_counts)
        receipt: dict[str, object] = {
            "format_version": RECEIPT_FORMAT,
            "active_families": list(projection.active_families),
            "active_family_counts": family_counts,
            "active_family_counts_sha256": _sha256_json(family_counts),
            "diagnostic_receipt_sha256": diagnostic_receipt_sha256,
            "full_formulation_sha256": direct_model.formulation_sha256,
            "full_proto_sha256": projection.full_proto_sha256,
            "reduced_formulation_sha256": (projection.direct_model.formulation_sha256),
            "reduced_proto_sha256": projection.reduced_proto_sha256,
            "full_dimensions": dict(projection.full_dimensions),
            "reduced_dimensions": dict(projection.reduced_dimensions),
            "source_row_count": len(projection.source_row_indices),
            "source_row_indices_file": index_path.name,
            "source_row_indices_file_sha256": _sha256_file(index_path),
            "source_row_indices_sha256": _sha256_json(
                list(projection.source_row_indices)
            ),
            "row_subset_containment_passed": (projection.row_subset_containment_passed),
            "all_source_variables_and_domains_preserved": True,
            "bundle_directory": "bundle",
            "bundle_sha256": bundle["bundle_sha256"],
            "bundle_ir_sha256": bundle["ir_sha256"],
            "full_protocol_infeasibility_claim_authorized": False,
            "split_gate0_authorized": False,
            "training_authorized": False,
            "model_selection_authorized": False,
            "outer_evaluation_authorized": False,
            "test_access_authorized": False,
        }
        receipt["receipt_sha256"] = _sha256_json(receipt)
        _write_exclusive(
            output / "receipt.json",
            canonical_json_bytes(receipt),
        )
        for path in output.iterdir():
            if path.is_file() and not path.is_symlink():
                os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.chmod(
            output,
            stat.S_IRUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH,
        )
        return receipt
    except BaseException:
        for path in output.iterdir():
            if path.is_file() and not path.is_symlink():
                os.chmod(path, 0o600)
        os.chmod(output, 0o700)
        raise


__all__ = [
    "ExactFamilyCoreProjection",
    "RECEIPT_FORMAT",
    "V6DCoreProjectionError",
    "canonical_json_bytes",
    "export_exact_family_core_bundle",
    "select_exact_family_core",
]
