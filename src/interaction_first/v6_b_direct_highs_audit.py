"""Registered direct-HiGHS audit for the authenticated V6-B linear bundle."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import sys
import time
from pathlib import Path
from typing import Mapping

from .v6_highs_bundle_backend import LOCKED_OPTIONS
from .v6_highs_bundle_backend import solve_highs_projection_bundle


REGISTRATION_FORMAT = "v6_b_direct_highs_audit_registration_v1"
RECEIPT_FORMAT = "v6_b_direct_highs_audit_receipt_v1"


class DirectHighsAuditError(RuntimeError):
    """Raised when the registered direct-HiGHS audit cannot fail closed."""


def canonical_json_bytes(payload: object) -> bytes:
    """Encode one canonical ASCII JSON value."""
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


def sha256_payload(payload: object) -> str:
    """Hash one canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_verified(path: Path, expected_sha256: str, label: str) -> bytes:
    if not path.is_absolute() or not _is_sha256(expected_sha256):
        raise DirectHighsAuditError(f"{label} reference is malformed")
    if path.is_symlink():
        raise DirectHighsAuditError(f"{label} cannot be symbolic")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise DirectHighsAuditError(f"{label} cannot be opened") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise DirectHighsAuditError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            chunks.append(chunk)
        if digest.hexdigest() != expected_sha256:
            raise DirectHighsAuditError(f"{label} SHA-256 differs")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _strict_json(payload: bytes, label: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DirectHighsAuditError(f"{label} contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DirectHighsAuditError(f"{label} is not canonical ASCII JSON") from error
    if not isinstance(value, dict):
        raise DirectHighsAuditError(f"{label} is not a JSON object")
    if canonical_json_bytes(value) != payload:
        raise DirectHighsAuditError(f"{label} is not canonically encoded")
    return value


def _verify_receipt_self_hash(
    payload: Mapping[str, object],
    expected_self_sha256: str,
    label: str,
) -> None:
    if not _is_sha256(expected_self_sha256):
        raise DirectHighsAuditError(f"{label} self hash is malformed")
    field = "receipt_sha256"
    if payload.get(field) != expected_self_sha256:
        raise DirectHighsAuditError(f"{label} registered self hash differs")
    unsigned = dict(payload)
    unsigned.pop(field)
    if sha256_payload(unsigned) != expected_self_sha256:
        raise DirectHighsAuditError(f"{label} self hash is invalid")


def _validate_parent_reference(
    reference: object,
    label: str,
) -> dict[str, object]:
    if not isinstance(reference, Mapping) or set(reference) != {
        "path",
        "file_sha256",
        "receipt_sha256",
        "format_version",
        "required_fields",
    }:
        raise DirectHighsAuditError(f"{label} reference fields differ")
    if (
        not Path(str(reference["path"])).is_absolute()
        or not _is_sha256(reference["file_sha256"])
        or not _is_sha256(reference["receipt_sha256"])
        or not isinstance(reference["format_version"], str)
        or not isinstance(reference["required_fields"], Mapping)
    ):
        raise DirectHighsAuditError(f"{label} reference is malformed")
    return dict(reference)


def validate_direct_highs_registration(
    registration: Mapping[str, object],
) -> dict[str, object]:
    """Validate the immutable Direct-HiGHS execution envelope."""
    required = {
        "format_version",
        "authorization",
        "bundle_export",
        "primary_cp_result",
        "independent_cp_result",
        "runtime",
        "expected",
        "source",
        "limits",
        "slurm",
    }
    if set(registration) != required:
        raise DirectHighsAuditError("direct-HiGHS registration fields differ")
    if registration.get("format_version") != REGISTRATION_FORMAT:
        raise DirectHighsAuditError("direct-HiGHS registration format differs")
    authorization = registration.get("authorization")
    if (
        not isinstance(authorization, Mapping)
        or set(authorization)
        != {
            "full_protocol_infeasibility_claim",
            "split_gate0",
            "training",
            "model_selection",
            "outer_evaluation",
            "test_access",
        }
        or any(value is not False for value in authorization.values())
    ):
        raise DirectHighsAuditError("direct-HiGHS registration authorizes science")
    bundle_export = registration.get("bundle_export")
    if not isinstance(bundle_export, Mapping) or set(bundle_export) != {
        "receipt_path",
        "receipt_file_sha256",
        "receipt_sha256",
        "bundle_directory",
        "bundle_sha256",
        "bundle_json_file_sha256",
    }:
        raise DirectHighsAuditError("bundle export reference fields differ")
    if (
        not Path(str(bundle_export["receipt_path"])).is_absolute()
        or not Path(str(bundle_export["bundle_directory"])).is_absolute()
        or any(
            not _is_sha256(bundle_export[name])
            for name in (
                "receipt_file_sha256",
                "receipt_sha256",
                "bundle_sha256",
                "bundle_json_file_sha256",
            )
        )
    ):
        raise DirectHighsAuditError("bundle export reference is malformed")
    _validate_parent_reference(registration.get("primary_cp_result"), "primary CP")
    _validate_parent_reference(
        registration.get("independent_cp_result"),
        "independent CP",
    )
    primary_reference = registration["primary_cp_result"]
    independent_reference = registration["independent_cp_result"]
    assert isinstance(primary_reference, Mapping)
    assert isinstance(independent_reference, Mapping)
    if primary_reference["required_fields"] != {
        "decomposition_terminal_status": "PROVEN_INFEASIBLE",
        "master_status": "INFEASIBLE",
        "primary_controller_proven_infeasible": True,
        "full_protocol_infeasibility_claim_authorized": False,
        "split_gate0_authorized": False,
        "training_authorized": False,
    }:
        raise DirectHighsAuditError("primary CP required fields differ")
    if independent_reference["required_fields"] != {
        "projection_status": "INFEASIBLE",
        "linear_agreement_with_primary": True,
        "full_protocol_infeasibility_claim_authorized": False,
        "split_gate0_authorized": False,
        "training_authorized": False,
    }:
        raise DirectHighsAuditError("independent CP required fields differ")
    runtime = registration.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "receipt_path",
        "receipt_file_sha256",
        "receipt_sha256",
        "python_executable",
        "python_version",
        "highspy_version",
        "numpy_version",
        "system",
        "machine",
        "libc_family",
        "libc_version",
        "native_process_isolation",
    }:
        raise DirectHighsAuditError("direct-HiGHS runtime fields differ")
    if (
        not Path(str(runtime["receipt_path"])).is_absolute()
        or not Path(str(runtime["python_executable"])).is_absolute()
        or not _is_sha256(runtime["receipt_file_sha256"])
        or not _is_sha256(runtime["receipt_sha256"])
        or runtime["native_process_isolation"] is not True
        or any(
            not isinstance(runtime[name], str)
            for name in (
                "python_version",
                "highspy_version",
                "numpy_version",
                "system",
                "machine",
                "libc_family",
                "libc_version",
            )
        )
    ):
        raise DirectHighsAuditError("direct-HiGHS runtime is malformed")
    expected = registration.get("expected")
    if not isinstance(expected, Mapping) or set(expected) != {
        "ir_sha256",
        "source_linear_projection_sha256",
        "dimensions",
        "cp_consensus_status",
    }:
        raise DirectHighsAuditError("direct-HiGHS expected fields differ")
    dimensions = expected.get("dimensions")
    if (
        not _is_sha256(expected.get("ir_sha256"))
        or not _is_sha256(expected.get("source_linear_projection_sha256"))
        or expected.get("cp_consensus_status") != "INFEASIBLE"
        or not isinstance(dimensions, Mapping)
        or set(dimensions) != {"columns", "rows", "nonzeros"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions.values()
        )
    ):
        raise DirectHighsAuditError("direct-HiGHS expected identity is malformed")
    source = registration.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "source_root",
        "implementation_commit",
        "files",
    }:
        raise DirectHighsAuditError("direct-HiGHS source fields differ")
    source_root = Path(str(source.get("source_root", "")))
    files = source.get("files")
    if (
        not source_root.is_absolute()
        or not _is_git_commit(source.get("implementation_commit"))
        or not isinstance(files, Mapping)
        or set(files) != {"backend", "bundle", "controller", "launcher"}
    ):
        raise DirectHighsAuditError("direct-HiGHS source identity is malformed")
    for label, reference in files.items():
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str)
            or Path(reference["path"]).is_absolute()
            or ".." in Path(reference["path"]).parts
            or not _is_sha256(reference["sha256"])
        ):
            raise DirectHighsAuditError(
                f"direct-HiGHS {label} source reference is malformed"
            )
    limits = registration.get("limits")
    if (
        not isinstance(limits, Mapping)
        or set(limits) != {"maximum_time_seconds"}
        or isinstance(limits["maximum_time_seconds"], bool)
        or not isinstance(limits["maximum_time_seconds"], (int, float))
        or float(limits["maximum_time_seconds"]) <= 0.0
    ):
        raise DirectHighsAuditError("direct-HiGHS limits differ")
    slurm = registration.get("slurm")
    if not isinstance(slurm, Mapping) or set(slurm) != {
        "partition",
        "nodes",
        "tasks",
        "gpus_requested",
        "allocated_gpu_purpose",
        "accelerator_visible_to_solver",
        "cpus_per_task",
        "solver_threads",
        "explicit_memory_request",
        "wall_time",
    }:
        raise DirectHighsAuditError("direct-HiGHS Slurm fields differ")
    if (
        slurm["nodes"] != 1
        or slurm["tasks"] != 1
        or slurm["gpus_requested"] != 1
        or slurm["allocated_gpu_purpose"] != "site_scheduler_requirement_only"
        or slurm["accelerator_visible_to_solver"] is not False
        or slurm["solver_threads"] != 1
        or slurm["explicit_memory_request"] is not False
        or not isinstance(slurm["partition"], str)
        or not isinstance(slurm["cpus_per_task"], int)
        or slurm["cpus_per_task"] <= 0
        or not isinstance(slurm["wall_time"], str)
    ):
        raise DirectHighsAuditError("direct-HiGHS Slurm envelope is malformed")
    return dict(registration)


def _authenticate_parent(
    reference: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    payload = _strict_json(
        _read_verified(
            Path(str(reference["path"])),
            str(reference["file_sha256"]),
            label,
        ),
        label,
    )
    if payload.get("format_version") != reference["format_version"]:
        raise DirectHighsAuditError(f"{label} format differs")
    _verify_receipt_self_hash(
        payload,
        str(reference["receipt_sha256"]),
        label,
    )
    required_fields = reference["required_fields"]
    assert isinstance(required_fields, Mapping)
    for name, expected in required_fields.items():
        if payload.get(name) != expected:
            raise DirectHighsAuditError(f"{label} required field {name} differs")
    return {
        "path": str(reference["path"]),
        "file_sha256": str(reference["file_sha256"]),
        "receipt_sha256": str(reference["receipt_sha256"]),
    }


def _authenticate_bundle_export(
    reference: Mapping[str, object],
    expected: Mapping[str, object],
) -> dict[str, object]:
    payload = _strict_json(
        _read_verified(
            Path(str(reference["receipt_path"])),
            str(reference["receipt_file_sha256"]),
            "bundle export receipt",
        ),
        "bundle export receipt",
    )
    _verify_receipt_self_hash(
        payload,
        str(reference["receipt_sha256"]),
        "bundle export receipt",
    )
    bundle = payload.get("bundle")
    if (
        payload.get("format_version")
        != "v6_b_linear_projection_bundle_export_receipt_v1"
        or payload.get("terminal_status")
        != "FULL_LINEAR_PROJECTION_BUNDLE_EXPORTED"
        or payload.get("highs_solver_invoked") is not False
        or payload.get("highspy_imported") is not False
        or payload.get("split_gate0_authorized") is not False
        or payload.get("training_authorized") is not False
        or not isinstance(bundle, Mapping)
        or bundle.get("directory") != reference["bundle_directory"]
        or bundle.get("bundle_sha256") != reference["bundle_sha256"]
        or bundle.get("bundle_json_file_sha256")
        != reference["bundle_json_file_sha256"]
        or payload.get("ir_sha256") != expected["ir_sha256"]
    ):
        raise DirectHighsAuditError("bundle export receipt identity differs")
    dimensions = bundle.get("artifacts")
    if not isinstance(dimensions, Mapping):
        raise DirectHighsAuditError("bundle export artifact inventory is absent")
    bundle_json = Path(str(reference["bundle_directory"])) / "bundle.json"
    bundle_metadata = _strict_json(
        _read_verified(
            bundle_json,
            str(reference["bundle_json_file_sha256"]),
            "bundle metadata",
        ),
        "bundle metadata",
    )
    if (
        bundle_metadata.get("format_version")
        != "v6_solver_neutral_linear_projection_bundle_v1"
        or bundle_metadata.get("bundle_sha256") != reference["bundle_sha256"]
        or bundle_metadata.get("ir_sha256") != expected["ir_sha256"]
        or bundle_metadata.get("source_linear_projection_sha256")
        != expected["source_linear_projection_sha256"]
        or bundle_metadata.get("dimensions") != expected["dimensions"]
        or bundle_metadata.get("training_authorized") is not False
        or bundle_metadata.get("complete_score_free_feasible_region_certified")
        is not False
    ):
        raise DirectHighsAuditError("bundle metadata identity differs")
    return {
        "receipt_path": str(reference["receipt_path"]),
        "receipt_file_sha256": str(reference["receipt_file_sha256"]),
        "receipt_sha256": str(reference["receipt_sha256"]),
        "bundle_directory": str(reference["bundle_directory"]),
        "bundle_sha256": str(reference["bundle_sha256"]),
    }


def _authenticate_runtime(reference: Mapping[str, object]) -> dict[str, object]:
    payload = _strict_json(
        _read_verified(
            Path(str(reference["receipt_path"])),
            str(reference["receipt_file_sha256"]),
            "HiGHS runtime receipt",
        ),
        "HiGHS runtime receipt",
    )
    _verify_receipt_self_hash(
        payload,
        str(reference["receipt_sha256"]),
        "HiGHS runtime receipt",
    )
    solver_smoke = payload.get("solver_smoke")
    python_receipt = payload.get("python")
    if (
        payload.get("format_version") != "v6_a_solver_runtime_lock_v1"
        or payload.get("backend") != "highs"
        or payload.get("training_authorized") is not False
        or not isinstance(solver_smoke, Mapping)
        or solver_smoke.get("native_library_process_isolation") is not True
        or not isinstance(solver_smoke.get("highs"), Mapping)
        or solver_smoke["highs"].get("highs_version")
        != reference["highspy_version"]
        or solver_smoke["highs"].get("locked_options") != LOCKED_OPTIONS
        or not isinstance(python_receipt, Mapping)
        or python_receipt.get("executable") != reference["python_executable"]
        or python_receipt.get("version") != reference["python_version"]
    ):
        raise DirectHighsAuditError("HiGHS runtime receipt identity differs")
    if any(name == "ortools" or name.startswith("ortools.") for name in sys.modules):
        raise DirectHighsAuditError("OR-Tools is loaded in the HiGHS audit process")
    try:
        import highspy
        import numpy
    except ImportError as error:
        raise DirectHighsAuditError(
            "registered HiGHS runtime packages are absent"
        ) from error
    observed_libc = platform.libc_ver()
    if (
        str(Path(sys.executable).resolve())
        != str(Path(str(reference["python_executable"])).resolve())
        or platform.python_version() != reference["python_version"]
        or highspy.Highs().version() != reference["highspy_version"]
        or numpy.__version__ != reference["numpy_version"]
        or platform.system() != reference["system"]
        or platform.machine() != reference["machine"]
        or observed_libc[0] != reference["libc_family"]
        or observed_libc[1] != reference["libc_version"]
    ):
        raise DirectHighsAuditError("observed HiGHS runtime differs")
    return {
        "receipt_path": str(reference["receipt_path"]),
        "receipt_file_sha256": str(reference["receipt_file_sha256"]),
        "receipt_sha256": str(reference["receipt_sha256"]),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "highspy_version": highspy.Highs().version(),
        "numpy_version": numpy.__version__,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _authenticate_sources(
    registration: Mapping[str, object],
) -> dict[str, object]:
    source = registration["source"]
    assert isinstance(source, Mapping)
    source_root = Path(str(source["source_root"]))
    if not source_root.is_dir() or source_root.is_symlink():
        raise DirectHighsAuditError("registered source root is absent or symbolic")
    files = source["files"]
    assert isinstance(files, Mapping)
    receipts: dict[str, object] = {}
    for label, raw_reference in files.items():
        assert isinstance(raw_reference, Mapping)
        path = source_root / str(raw_reference["path"])
        payload = _read_verified(path, str(raw_reference["sha256"]), f"{label} source")
        receipts[str(label)] = {
            "path": str(path),
            "sha256": str(raw_reference["sha256"]),
            "byte_size": len(payload),
        }
    return receipts


def _seal_tree(output: Path) -> None:
    for root, directories, files in os.walk(output, topdown=False):
        root_path = Path(root)
        for name in files:
            path = root_path / name
            if not path.is_symlink():
                os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        for name in directories:
            path = root_path / name
            if not path.is_symlink():
                os.chmod(
                    path,
                    stat.S_IRUSR
                    | stat.S_IXUSR
                    | stat.S_IRGRP
                    | stat.S_IXGRP
                    | stat.S_IROTH
                    | stat.S_IXOTH,
                )
    os.chmod(
        output,
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH,
    )


def _publish(output: Path, receipt: dict[str, object]) -> None:
    report = (
        "# V6-B direct HiGHS linear audit\n\n"
        f"Terminal status: `{receipt['terminal_status']}`\n\n"
        "This audit is linear-projection evidence only. It does not authorize "
        "Gate 0, training, model selection, outer evaluation, or test access.\n"
    ).encode("ascii")
    receipt["report_sha256"] = hashlib.sha256(report).hexdigest()
    receipt["receipt_sha256"] = sha256_payload(receipt)
    _write_exclusive(output / "receipt.json", canonical_json_bytes(receipt))
    _write_exclusive(output / "report.md", report)
    _seal_tree(output)


def execute_direct_highs_audit(
    *,
    registration: Mapping[str, object],
    registration_sha256: str,
    controller_sha256: str,
    output_directory: Path,
) -> dict[str, object]:
    """Authenticate every parent and run one isolated direct-HiGHS audit."""
    checked = validate_direct_highs_registration(registration)
    if sha256_payload(checked) != registration_sha256:
        raise DirectHighsAuditError("direct-HiGHS registration digest differs")
    output = Path(os.path.abspath(os.path.expanduser(str(output_directory))))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only direct-HiGHS output exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise DirectHighsAuditError("direct-HiGHS output parent is absent or symbolic")
    os.mkdir(output, 0o700)
    started_ns = time.monotonic_ns()
    try:
        sources = _authenticate_sources(checked)
        controller_receipt = sources["controller"]
        assert isinstance(controller_receipt, Mapping)
        if controller_receipt["sha256"] != controller_sha256:
            raise DirectHighsAuditError("direct-HiGHS controller hash differs")
        expected = checked["expected"]
        assert isinstance(expected, Mapping)
        bundle_reference = checked["bundle_export"]
        runtime_reference = checked["runtime"]
        primary_reference = checked["primary_cp_result"]
        independent_reference = checked["independent_cp_result"]
        assert isinstance(bundle_reference, Mapping)
        assert isinstance(runtime_reference, Mapping)
        assert isinstance(primary_reference, Mapping)
        assert isinstance(independent_reference, Mapping)
        bundle = _authenticate_bundle_export(bundle_reference, expected)
        primary = _authenticate_parent(primary_reference, "primary CP result")
        independent = _authenticate_parent(
            independent_reference,
            "independent CP result",
        )
        runtime = _authenticate_runtime(runtime_reference)
        solver_output = output / "solver"
        solver_receipt = solve_highs_projection_bundle(
            bundle_directory=Path(str(bundle_reference["bundle_directory"])),
            expected_bundle_sha256=str(bundle_reference["bundle_sha256"]),
            output_directory=solver_output,
            maximum_time_seconds=float(checked["limits"]["maximum_time_seconds"]),
        )
        if (
            solver_receipt.get("ir_sha256") != expected["ir_sha256"]
            or solver_receipt.get("original_dimensions") != expected["dimensions"]
            or solver_receipt.get("bundle_sha256")
            != bundle_reference["bundle_sha256"]
        ):
            raise DirectHighsAuditError("direct-HiGHS solver identity differs")
        solver_file_sha256 = _sha256_file(solver_output / "receipt.json")
        _verify_receipt_self_hash(
            solver_receipt,
            str(solver_receipt["receipt_sha256"]),
            "direct-HiGHS solver receipt",
        )
        agreement = (
            solver_receipt.get("linear_projection_certified") is True
            and solver_receipt.get("linear_projection_infeasible") is True
            and solver_receipt.get("linear_projection_optimal") is False
        )
        if agreement:
            terminal_status = "DIRECT_HIGHS_LINEAR_INFEASIBLE_AGREEMENT"
        elif solver_receipt.get("linear_projection_certified") is True:
            terminal_status = "DIRECT_HIGHS_LINEAR_DISAGREEMENT"
        else:
            terminal_status = "DIRECT_HIGHS_LINEAR_UNCERTIFIED"
        receipt: dict[str, object] = {
            "format_version": RECEIPT_FORMAT,
            "terminal_status": terminal_status,
            "registration_sha256": registration_sha256,
            "controller_sha256": controller_sha256,
            "elapsed_ns": time.monotonic_ns() - started_ns,
            "source_receipts": sources,
            "runtime_receipt": runtime,
            "bundle_export_receipt": bundle,
            "primary_cp_receipt": primary,
            "independent_cp_receipt": independent,
            "cp_consensus_status": expected["cp_consensus_status"],
            "solver_receipt": {
                "path": str((solver_output / "receipt.json").resolve()),
                "file_sha256": solver_file_sha256,
                "receipt_sha256": solver_receipt["receipt_sha256"],
                "model_status": solver_receipt["model_status"],
                "presolved_dimensions": solver_receipt["presolved_dimensions"],
                "mip_gap_is_finite_zero": solver_receipt[
                    "mip_gap_is_finite_zero"
                ],
                "linear_projection_infeasible": solver_receipt[
                    "linear_projection_infeasible"
                ],
                "linear_projection_optimal": solver_receipt[
                    "linear_projection_optimal"
                ],
                "linear_projection_certified": solver_receipt[
                    "linear_projection_certified"
                ],
            },
            "cross_solver_linear_agreement": agreement,
            "direct_highs_audit_completed": True,
            "full_protocol_infeasibility_claim_authorized": False,
            "complete_score_free_feasible_region_certified": False,
            "split_gate0_authorized": False,
            "training_authorized": False,
            "model_selection_authorized": False,
            "outer_evaluation_authorized": False,
            "test_access_authorized": False,
        }
        _publish(output, receipt)
        return receipt
    except BaseException as error:
        failure = {
            "format_version": RECEIPT_FORMAT,
            "terminal_status": "DIRECT_HIGHS_AUDIT_EXCEPTION",
            "registration_sha256": registration_sha256,
            "controller_sha256": controller_sha256,
            "elapsed_ns": time.monotonic_ns() - started_ns,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_sha256": sha256_payload([type(error).__name__, str(error)]),
            "direct_highs_audit_completed": False,
            "full_protocol_infeasibility_claim_authorized": False,
            "complete_score_free_feasible_region_certified": False,
            "split_gate0_authorized": False,
            "training_authorized": False,
            "model_selection_authorized": False,
            "outer_evaluation_authorized": False,
            "test_access_authorized": False,
        }
        try:
            _publish(output, failure)
        except BaseException:
            try:
                _seal_tree(output)
            except BaseException:
                pass
        raise
