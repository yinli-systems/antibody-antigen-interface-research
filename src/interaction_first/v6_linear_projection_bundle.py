"""Content-addressed sparse bundle for solver-isolated V6 linear projections."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .v6_joint_ir import CanonicalJointIR


BUNDLE_FORMAT = "v6_solver_neutral_linear_projection_bundle_v1"
MAX_EXACT_FLOAT64_INTEGER = 1 << 53
ARRAY_FILES = {
    "row_starts": ("row_starts.npy", "int64"),
    "column_indices": ("column_indices.npy", "int32"),
    "coefficients": ("coefficients.npy", "int64"),
    "row_lower": ("row_lower.npy", "int64"),
    "row_lower_present": ("row_lower_present.npy", "uint8"),
    "row_upper": ("row_upper.npy", "int64"),
    "row_upper_present": ("row_upper_present.npy", "uint8"),
}


class LinearProjectionBundleError(RuntimeError):
    """Raised when a solver-neutral bundle cannot be authenticated."""


def canonical_json_bytes(payload: object) -> bytes:
    """Encode one canonical ASCII JSON object."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
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


def _write_jsonl(path: Path, rows: object) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        for row in rows:
            payload = canonical_json_bytes(row)
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_array_exclusive(path: Path, array: object, numpy: object) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            numpy.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def export_linear_projection_bundle(
    ir: CanonicalJointIR,
    output_directory: Path,
) -> dict[str, object]:
    """Export exact integer rows without importing either production solver."""
    try:
        import numpy as np
    except ImportError as error:
        raise LinearProjectionBundleError("NumPy is required") from error

    output = Path(os.path.abspath(os.path.expanduser(str(output_directory))))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only bundle already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise LinearProjectionBundleError("bundle parent is absent or symbolic")
    os.mkdir(output, 0o700)
    variable_index = {key: index for index, key in enumerate(ir.variable_keys)}
    row_count = len(ir.linear_constraints)
    nonzero_count = sum(len(row.coefficients) for row in ir.linear_constraints)
    row_starts = np.empty(row_count + 1, dtype=np.int64)
    column_indices = np.empty(nonzero_count, dtype=np.int32)
    coefficients = np.empty(nonzero_count, dtype=np.int64)
    row_lower = np.zeros(row_count, dtype=np.int64)
    row_lower_present = np.zeros(row_count, dtype=np.uint8)
    row_upper = np.zeros(row_count, dtype=np.int64)
    row_upper_present = np.zeros(row_count, dtype=np.uint8)
    offset = 0
    maximum_abs_bound = 0
    maximum_abs_coefficient = 0
    row_starts[0] = 0
    for row_index, row in enumerate(ir.linear_constraints):
        for key, coefficient in row.coefficients:
            column_indices[offset] = variable_index[key]
            coefficients[offset] = coefficient
            maximum_abs_coefficient = max(
                maximum_abs_coefficient,
                abs(int(coefficient)),
            )
            offset += 1
        row_starts[row_index + 1] = offset
        if row.lower_bound is not None:
            row_lower[row_index] = row.lower_bound
            row_lower_present[row_index] = 1
            maximum_abs_bound = max(maximum_abs_bound, abs(int(row.lower_bound)))
        if row.upper_bound is not None:
            row_upper[row_index] = row.upper_bound
            row_upper_present[row_index] = 1
            maximum_abs_bound = max(maximum_abs_bound, abs(int(row.upper_bound)))
    if offset != nonzero_count:
        raise LinearProjectionBundleError("nonzero count changed during export")
    if max(maximum_abs_coefficient, maximum_abs_bound) > MAX_EXACT_FLOAT64_INTEGER:
        raise LinearProjectionBundleError(
            "linear projection integer exceeds exact float64 representation"
        )

    arrays = {
        "row_starts": row_starts,
        "column_indices": column_indices,
        "coefficients": coefficients,
        "row_lower": row_lower,
        "row_lower_present": row_lower_present,
        "row_upper": row_upper,
        "row_upper_present": row_upper_present,
    }
    for name, array in arrays.items():
        _save_array_exclusive(output / ARRAY_FILES[name][0], array, np)
    variable_path = output / "variable_dictionary.jsonl"
    row_path = output / "row_dictionary.jsonl"
    _write_jsonl(
        variable_path,
        (
            {"index": index, "key": list(key)}
            for index, key in enumerate(ir.variable_keys)
        ),
    )
    _write_jsonl(
        row_path,
        (
            {"index": index, "name": row.name}
            for index, row in enumerate(ir.linear_constraints)
        ),
    )
    artifact_paths = [
        *(output / filename for filename, _ in ARRAY_FILES.values()),
        variable_path,
        row_path,
    ]
    artifacts = {
        path.name: {
            "sha256": _sha256_file(path),
            "byte_size": path.stat().st_size,
        }
        for path in artifact_paths
    }
    metadata: dict[str, object] = {
        "format_version": BUNDLE_FORMAT,
        "ir_sha256": ir.ir_sha256,
        "source_linear_projection_sha256": ir.source_linear_projection_sha256,
        "construction_scope": ir.construction_scope,
        "independent_raw_authority_reconstruction": (
            ir.independent_raw_authority_reconstruction
        ),
        "source_bindings": dict(ir.source_bindings),
        "objective_stages": list(ir.objective_stages),
        "dimensions": {
            "columns": len(ir.variable_keys),
            "rows": row_count,
            "nonzeros": nonzero_count,
        },
        "numeric_envelope": {
            "maximum_abs_bound": maximum_abs_bound,
            "maximum_abs_coefficient": maximum_abs_coefficient,
            "maximum_exact_float64_integer": MAX_EXACT_FLOAT64_INTEGER,
            "all_integers_exactly_representable_as_float64": True,
        },
        "arrays": {
            name: {
                "file": ARRAY_FILES[name][0],
                "dtype": ARRAY_FILES[name][1],
                "shape": list(array.shape),
            }
            for name, array in arrays.items()
        },
        "variable_dictionary_file": variable_path.name,
        "row_dictionary_file": row_path.name,
        "artifacts": artifacts,
        "cp_sat_or_highs_native_object_serialized": False,
        "complete_score_free_feasible_region_certified": False,
        "training_authorized": False,
    }
    metadata["bundle_sha256"] = hashlib.sha256(
        canonical_json_bytes(metadata)
    ).hexdigest()
    _write_exclusive(output / "bundle.json", canonical_json_bytes(metadata))
    for path in output.iterdir():
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
    return metadata


@dataclass(frozen=True)
class LoadedLinearProjectionBundle:
    """Authenticated memory-mapped sparse arrays and their metadata."""

    root: Path
    metadata: Mapping[str, object]
    arrays: Mapping[str, object]


def load_linear_projection_bundle(
    bundle_directory: Path,
    *,
    expected_bundle_sha256: str,
) -> LoadedLinearProjectionBundle:
    """Authenticate every byte and load arrays read-only through NumPy mmap."""
    try:
        import numpy as np
    except ImportError as error:
        raise LinearProjectionBundleError("NumPy is required") from error
    supplied = Path(os.path.abspath(os.path.expanduser(str(bundle_directory))))
    if supplied.is_symlink():
        raise LinearProjectionBundleError("bundle root is absent or symbolic")
    root = supplied.resolve(strict=True)
    if not root.is_dir():
        raise LinearProjectionBundleError("bundle root is absent or symbolic")
    metadata_path = root / "bundle.json"
    raw = metadata_path.read_bytes()
    metadata = json.loads(raw.decode("ascii"))
    if not isinstance(metadata, dict) or metadata.get("format_version") != BUNDLE_FORMAT:
        raise LinearProjectionBundleError("bundle metadata format differs")
    claimed = metadata.pop("bundle_sha256", None)
    observed = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    metadata["bundle_sha256"] = claimed
    if (
        raw != canonical_json_bytes(metadata)
        or claimed != observed
        or claimed != expected_bundle_sha256
    ):
        raise LinearProjectionBundleError("bundle metadata digest differs")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise LinearProjectionBundleError("bundle artifacts are absent")
    for filename, receipt in artifacts.items():
        if not isinstance(receipt, Mapping):
            raise LinearProjectionBundleError("bundle artifact receipt is malformed")
        path = root / str(filename)
        if path.is_symlink() or not path.is_file():
            raise LinearProjectionBundleError("bundle artifact is absent or symbolic")
        if path.stat().st_size != receipt.get("byte_size"):
            raise LinearProjectionBundleError("bundle artifact size differs")
        if _sha256_file(path) != receipt.get("sha256"):
            raise LinearProjectionBundleError("bundle artifact digest differs")
    array_receipts = metadata.get("arrays")
    if not isinstance(array_receipts, Mapping) or set(array_receipts) != set(ARRAY_FILES):
        raise LinearProjectionBundleError("bundle array registry differs")
    arrays: dict[str, object] = {}
    for name, expected in array_receipts.items():
        if not isinstance(expected, Mapping):
            raise LinearProjectionBundleError("bundle array receipt is malformed")
        array = np.load(root / str(expected["file"]), mmap_mode="r", allow_pickle=False)
        if str(array.dtype) != expected["dtype"] or list(array.shape) != expected["shape"]:
            raise LinearProjectionBundleError("bundle array dtype or shape differs")
        arrays[name] = array
    dimensions = metadata.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise LinearProjectionBundleError("bundle dimensions are absent")
    rows = int(dimensions["rows"])
    columns = int(dimensions["columns"])
    nonzeros = int(dimensions["nonzeros"])
    numeric = metadata.get("numeric_envelope")
    if (
        not isinstance(numeric, Mapping)
        or numeric.get("maximum_exact_float64_integer")
        != MAX_EXACT_FLOAT64_INTEGER
        or numeric.get("all_integers_exactly_representable_as_float64") is not True
        or not isinstance(numeric.get("maximum_abs_bound"), int)
        or not isinstance(numeric.get("maximum_abs_coefficient"), int)
        or max(
            int(numeric["maximum_abs_bound"]),
            int(numeric["maximum_abs_coefficient"]),
        )
        > MAX_EXACT_FLOAT64_INTEGER
    ):
        raise LinearProjectionBundleError("bundle numeric envelope differs")
    if (
        arrays["row_starts"].shape != (rows + 1,)
        or arrays["column_indices"].shape != (nonzeros,)
        or arrays["coefficients"].shape != (nonzeros,)
        or arrays["row_lower"].shape != (rows,)
        or arrays["row_upper"].shape != (rows,)
        or arrays["row_starts"][0] != 0
        or arrays["row_starts"][-1] != nonzeros
        or np.any(arrays["row_starts"][1:] < arrays["row_starts"][:-1])
        or np.any(arrays["column_indices"] < 0)
        or np.any(arrays["column_indices"] >= columns)
        or np.any(arrays["row_lower_present"] > 1)
        or np.any(arrays["row_upper_present"] > 1)
    ):
        raise LinearProjectionBundleError("bundle sparse invariants differ")
    minimum_int64 = np.iinfo(np.int64).min
    if np.any(arrays["coefficients"] == minimum_int64):
        raise LinearProjectionBundleError("bundle coefficient magnitude overflows int64")
    observed_maximum_abs_coefficient = (
        int(np.max(np.abs(arrays["coefficients"]))) if nonzeros else 0
    )
    observed_bounds = np.concatenate(
        (
            arrays["row_lower"][arrays["row_lower_present"].astype(bool)],
            arrays["row_upper"][arrays["row_upper_present"].astype(bool)],
        )
    )
    if np.any(observed_bounds == minimum_int64):
        raise LinearProjectionBundleError("bundle bound magnitude overflows int64")
    observed_maximum_abs_bound = (
        int(np.max(np.abs(observed_bounds))) if observed_bounds.size else 0
    )
    if (
        observed_maximum_abs_coefficient
        != numeric["maximum_abs_coefficient"]
        or observed_maximum_abs_bound != numeric["maximum_abs_bound"]
    ):
        raise LinearProjectionBundleError("bundle numeric envelope audit differs")
    return LoadedLinearProjectionBundle(root=root, metadata=metadata, arrays=arrays)
