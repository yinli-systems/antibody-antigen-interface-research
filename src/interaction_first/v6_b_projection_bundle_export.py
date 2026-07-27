"""Registered CP-only export of the full V6-B solver-neutral linear bundle."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Mapping

from . import score_free_independent_auditor
from . import v6_b_independent_cp_audit as independent_cp
from . import v6_b_production as production
from . import v6_cp_sat_formulation as direct_cp
from . import v6_joint_ir
from . import v6_linear_projection_bundle as bundle_io


REGISTRATION_FORMAT = "v6_b_linear_projection_bundle_export_registration_v1"
RECEIPT_FORMAT = "v6_b_linear_projection_bundle_export_receipt_v1"


class ProjectionBundleExportError(RuntimeError):
    """Raised when a registered projection bundle export fails closed."""


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


def sha256_payload(payload: object) -> str:
    """Hash one canonical JSON payload."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


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
        "# V6-B solver-neutral linear projection bundle export\n\n"
        f"Terminal status: `{receipt['terminal_status']}`\n\n"
        "This CP-only export does not authorize Gate 0 or training.\n"
    ).encode("ascii")
    receipt["report_sha256"] = hashlib.sha256(report).hexdigest()
    receipt["receipt_sha256"] = sha256_payload(receipt)
    _write_exclusive(output / "receipt.json", canonical_json_bytes(receipt))
    _write_exclusive(output / "report.md", report)
    _seal_tree(output)


def validate_export_registration(
    registration: Mapping[str, object],
) -> dict[str, object]:
    """Validate the immutable execution envelope for a CP-only export."""
    required = {
        "format_version",
        "authorization",
        "production_registration",
        "formal_cp_result",
        "expected",
        "runtime",
        "source",
        "slurm",
    }
    if set(registration) != required:
        raise ProjectionBundleExportError("export registration fields differ")
    if registration.get("format_version") != REGISTRATION_FORMAT:
        raise ProjectionBundleExportError("export registration format differs")
    authorization = registration.get("authorization")
    if (
        not isinstance(authorization, Mapping)
        or set(authorization)
        != {"full_protocol_infeasibility_claim", "split_gate0", "training"}
        or any(value is not False for value in authorization.values())
    ):
        raise ProjectionBundleExportError("export registration authorizes science")
    production_reference = registration.get("production_registration")
    formal_reference = registration.get("formal_cp_result")
    if (
        not isinstance(production_reference, Mapping)
        or set(production_reference) != {"path", "sha256"}
        or not Path(str(production_reference.get("path", ""))).is_absolute()
        or not _is_sha256(production_reference.get("sha256"))
        or not isinstance(formal_reference, Mapping)
        or set(formal_reference) != {"path", "sha256", "receipt_sha256"}
        or not Path(str(formal_reference.get("path", ""))).is_absolute()
        or not _is_sha256(formal_reference.get("sha256"))
        or not _is_sha256(formal_reference.get("receipt_sha256"))
    ):
        raise ProjectionBundleExportError("export parent reference differs")
    expected = registration.get("expected")
    if not isinstance(expected, Mapping) or set(expected) != {
        "normalized_authority_sha256",
        "direct_formulation_sha256",
        "ir_sha256",
        "dimensions",
    }:
        raise ProjectionBundleExportError("export expected identity differs")
    dimensions = expected["dimensions"]
    if not isinstance(dimensions, Mapping) or set(dimensions) != {
        "variables",
        "linear_constraints",
        "exact_gate_declarations",
    }:
        raise ProjectionBundleExportError("export expected dimensions differ")
    if (
        any(
            not _is_sha256(expected[name])
            for name in (
                "normalized_authority_sha256",
                "direct_formulation_sha256",
                "ir_sha256",
            )
        )
        or any(not isinstance(value, int) or value < 0 for value in dimensions.values())
    ):
        raise ProjectionBundleExportError("export expected identity is malformed")
    runtime = registration.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "cp_sat_runtime_receipt_path",
        "cp_sat_runtime_receipt_sha256",
        "python_executable",
        "python_version",
        "ortools_version",
        "highspy_imported",
    }:
        raise ProjectionBundleExportError("export runtime envelope differs")
    if (
        not Path(str(runtime["cp_sat_runtime_receipt_path"])).is_absolute()
        or not _is_sha256(runtime["cp_sat_runtime_receipt_sha256"])
        or not Path(str(runtime["python_executable"])).is_absolute()
        or runtime["highspy_imported"] is not False
    ):
        raise ProjectionBundleExportError("export runtime identity is malformed")
    source = registration.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "implementation_commit",
        "source_root",
        "files",
    }:
        raise ProjectionBundleExportError("export source envelope differs")
    files = source["files"]
    if (
        not _is_git_commit(source["implementation_commit"])
        or not Path(str(source["source_root"])).is_absolute()
        or not isinstance(files, Mapping)
        or set(files) != {"controller", "implementation", "bundle", "launcher"}
    ):
        raise ProjectionBundleExportError("export source identity is malformed")
    for reference in files.values():
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"path", "sha256"}
            or Path(str(reference["path"])).is_absolute()
            or not _is_sha256(reference["sha256"])
        ):
            raise ProjectionBundleExportError("export source file identity differs")
    slurm = registration.get("slurm")
    if (
        not isinstance(slurm, Mapping)
        or slurm.get("solver_workers") != 0
        or slurm.get("gpus_requested") != 1
        or slurm.get("accelerator_visible_to_process") is not False
        or slurm.get("explicit_memory_request") is not False
    ):
        raise ProjectionBundleExportError("export Slurm envelope differs")
    return dict(registration)


def validate_formal_cp_result(
    payload: Mapping[str, object],
    *,
    expected_registration_sha256: str,
) -> dict[str, object]:
    """Require the immutable independent direct-CP infeasibility agreement."""
    checked = dict(payload)
    claimed = checked.pop("receipt_sha256", None)
    if claimed != sha256_payload(checked):
        raise ProjectionBundleExportError("formal CP result self hash differs")
    projection = checked.get("projection_receipt")
    if (
        checked.get("format_version")
        != "v6_b_independent_full_linear_cp_audit_receipt_v1"
        or checked.get("registration_sha256") != expected_registration_sha256
        or checked.get("terminal_status")
        != "INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT"
        or checked.get("primary_independent_authority_byte_agreement") is not True
        or checked.get("primary_independent_linear_agreement") is not True
        or not isinstance(projection, Mapping)
        or projection.get("status") != "INFEASIBLE"
        or projection.get("accepted_projection_terminal_status") is not True
        or checked.get("direct_highs_audit_completed") is not False
        or checked.get("full_protocol_infeasibility_claim_authorized") is not False
        or checked.get("split_gate0_authorized") is not False
        or checked.get("training_authorized") is not False
    ):
        raise ProjectionBundleExportError("formal CP result state differs")
    checked["receipt_sha256"] = claimed
    return checked


def _authenticate_export_sources(
    registration: Mapping[str, object],
) -> dict[str, object]:
    source = registration["source"]
    assert isinstance(source, Mapping)
    files = source["files"]
    assert isinstance(files, Mapping)
    modules = {
        "implementation": sys.modules[__name__],
        "bundle": bundle_io,
    }
    receipts: dict[str, object] = {}
    source_root = Path(str(source["source_root"]))
    for label, reference in files.items():
        if not isinstance(reference, Mapping):
            raise ProjectionBundleExportError(f"registered {label} source is absent")
        path = (source_root / str(reference["path"])).resolve(strict=True)
        if label in modules:
            imported_path = Path(str(modules[label].__file__)).resolve(strict=True)
            if _sha256_file(imported_path) != reference.get("sha256"):
                raise ProjectionBundleExportError(
                    f"imported registered {label} source differs"
                )
        observed = _sha256_file(path)
        if observed != reference.get("sha256"):
            raise ProjectionBundleExportError(f"registered {label} source differs")
        receipts[label] = {
            "path": str(path),
            "sha256": observed,
            "byte_size": path.stat().st_size,
        }
    return receipts


def execute_projection_bundle_export(
    *,
    export_registration: Mapping[str, object],
    export_registration_sha256: str,
    production_registration: Mapping[str, object],
    production_registration_sha256: str,
    formal_cp_result: Mapping[str, object],
    formal_cp_result_file_sha256: str,
    controller_sha256: str,
    output_directory: Path,
) -> dict[str, object]:
    """Rebuild the full IR and export it without invoking either solver."""
    registered = validate_export_registration(export_registration)
    if sha256_payload(registered) != export_registration_sha256:
        raise ProjectionBundleExportError("export registration digest differs")
    production_reference = registered["production_registration"]
    formal_reference = registered["formal_cp_result"]
    assert isinstance(production_reference, Mapping)
    assert isinstance(formal_reference, Mapping)
    if (
        production_reference.get("sha256") != production_registration_sha256
        or formal_reference.get("sha256") != formal_cp_result_file_sha256
    ):
        raise ProjectionBundleExportError("registered parent file digest differs")
    checked_production = production.validate_registration(production_registration)
    if production.sha256_payload(checked_production) != production_registration_sha256:
        raise ProjectionBundleExportError("production registration digest differs")
    checked_formal = validate_formal_cp_result(
        formal_cp_result,
        expected_registration_sha256=production_registration_sha256,
    )
    if checked_formal["receipt_sha256"] != formal_reference["receipt_sha256"]:
        raise ProjectionBundleExportError("registered formal CP self hash differs")
    runtime_reference = registered["runtime"]
    production_runtime = checked_production["runtime"]
    assert isinstance(runtime_reference, Mapping)
    assert isinstance(production_runtime, Mapping)
    for name in (
        "cp_sat_runtime_receipt_path",
        "cp_sat_runtime_receipt_sha256",
        "python_executable",
        "python_version",
        "ortools_version",
    ):
        if runtime_reference[name] != production_runtime[name]:
            raise ProjectionBundleExportError(f"registered runtime {name} differs")
    expected = registered["expected"]
    assert isinstance(expected, Mapping)
    for field in (
        "normalized_authority_sha256",
        "direct_formulation_sha256",
        "ir_sha256",
        "dimensions",
    ):
        formal_field = "normalized_authority_sha256" if field == "normalized_authority_sha256" else field
        if checked_formal.get(formal_field) != expected[field]:
            raise ProjectionBundleExportError(f"formal CP {field} differs")
    output = Path(os.path.abspath(os.path.expanduser(str(output_directory))))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only bundle output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ProjectionBundleExportError("bundle output parent is absent or symbolic")
    os.mkdir(output, 0o700)
    started_ns = time.monotonic_ns()
    try:
        if any(name == "highspy" or name.startswith("highspy.") for name in sys.modules):
            raise ProjectionBundleExportError("highspy is loaded in the CP-only process")
        runtime_receipt = production._runtime_receipt(checked_production)
        export_sources = _authenticate_export_sources(registered)
        production_sources = independent_cp._authenticate_registered_sources(
            checked_production
        )
        authority_reference = checked_production["authority"]
        assert isinstance(authority_reference, Mapping)
        published = production.read_verified(
            Path(str(authority_reference["path"])),
            str(authority_reference["sha256"]),
            "published authority",
        )
        scientific, publication = production.extract_scientific_authority(published)
        normalized_primary, primary_normalization = (
            production.normalize_scientific_pair_order(scientific)
        )
        authority_bytes, manifest_bytes, edge_bytes, raw_receipts = (
            production._registered_input_bytes(checked_production)
        )
        primary_payload = production.strict_json(
            normalized_primary,
            "normalized primary authority",
        )
        independently_rebuilt = (
            score_free_independent_auditor.reconstruct_score_free_authority(
                authority_bytes=authority_bytes,
                manifest_bytes=manifest_bytes,
                edge_ledger_bytes=edge_bytes,
                expected_primary_implementation_sha256=primary_payload[
                    "implementation_source_sha256"
                ],
                expected_counts=production.EXPECTED_COUNTS,
            )
        )
        normalized_independent, independent_normalization = (
            production.normalize_scientific_pair_order(
                production.canonical_json_bytes(independently_rebuilt)
            )
        )
        if normalized_independent != normalized_primary:
            raise ProjectionBundleExportError("independent authority bytes differ")
        authority_sha256 = hashlib.sha256(normalized_primary).hexdigest()
        if authority_sha256 != expected["normalized_authority_sha256"]:
            raise ProjectionBundleExportError("normalized authority digest differs")
        problem = direct_cp.direct_problem_from_authority_payload(
            normalized_primary,
            edge_bytes,
            expected_authority_payload_sha256=authority_sha256,
            expected_counts=production.EXPECTED_COUNTS,
        )
        direct_model = direct_cp.build_independent_cp_sat_formulation(problem)
        source_files = checked_production["source"]["files"]
        parent = checked_production["parent_evidence"]
        ir = v6_joint_ir.build_v6_joint_ir_from_direct_cp_sat(
            direct_model,
            solver_policy_sha256=parent["v6_b_policy"]["sha256"],
            direct_formulation_source_sha256=source_files[
                "v6_cp_sat_formulation"
            ]["sha256"],
            exact_oracle_source_sha256=source_files[
                "v6_exact_separation_oracle"
            ]["sha256"],
        )
        dimensions = {
            "variables": len(ir.variable_keys),
            "linear_constraints": len(ir.linear_constraints),
            "exact_gate_declarations": len(ir.separation_gates),
        }
        if (
            direct_model.formulation_sha256 != expected["direct_formulation_sha256"]
            or ir.ir_sha256 != expected["ir_sha256"]
            or dimensions != expected["dimensions"]
        ):
            raise ProjectionBundleExportError("rebuilt linear identity differs")
        bundle = bundle_io.export_linear_projection_bundle(ir, output / "bundle")
        if any(name == "highspy" or name.startswith("highspy.") for name in sys.modules):
            raise ProjectionBundleExportError("bundle export imported highspy")
        receipt: dict[str, object] = {
            "format_version": RECEIPT_FORMAT,
            "terminal_status": "FULL_LINEAR_PROJECTION_BUNDLE_EXPORTED",
            "export_registration_sha256": export_registration_sha256,
            "production_registration_sha256": production_registration_sha256,
            "formal_cp_result_file_sha256": formal_cp_result_file_sha256,
            "formal_cp_result_receipt_sha256": formal_reference["receipt_sha256"],
            "controller_sha256": controller_sha256,
            "elapsed_ns": time.monotonic_ns() - started_ns,
            "runtime_receipt": runtime_receipt,
            "export_source_receipts": export_sources,
            "production_source_receipts": production_sources,
            "authority_publication": publication,
            "primary_normalization": primary_normalization,
            "independent_normalization": independent_normalization,
            "primary_independent_authority_byte_agreement": True,
            "normalized_authority_sha256": authority_sha256,
            "raw_input_receipts": raw_receipts,
            "direct_formulation_sha256": direct_model.formulation_sha256,
            "ir_sha256": ir.ir_sha256,
            "dimensions": dimensions,
            "bundle": {
                "directory": str((output / "bundle").resolve()),
                "bundle_sha256": bundle["bundle_sha256"],
                "bundle_json_file_sha256": _sha256_file(output / "bundle" / "bundle.json"),
                "artifacts": bundle["artifacts"],
                "numeric_envelope": bundle["numeric_envelope"],
            },
            "cp_sat_model_built": True,
            "cp_sat_solver_invoked": False,
            "highspy_imported": False,
            "highs_solver_invoked": False,
            "historical_milp_matrix_consumed": False,
            "v6_b_partition_or_cut_consumed": False,
            "complete_score_free_feasible_region_certified": False,
            "full_protocol_infeasibility_claim_authorized": False,
            "split_gate0_authorized": False,
            "training_authorized": False,
        }
        _publish(output, receipt)
        return receipt
    except BaseException as error:
        failure = {
            "format_version": RECEIPT_FORMAT,
            "terminal_status": "LINEAR_PROJECTION_BUNDLE_EXPORT_EXCEPTION",
            "export_registration_sha256": export_registration_sha256,
            "production_registration_sha256": production_registration_sha256,
            "formal_cp_result_file_sha256": formal_cp_result_file_sha256,
            "controller_sha256": controller_sha256,
            "elapsed_ns": time.monotonic_ns() - started_ns,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_sha256": sha256_payload([type(error).__name__, str(error)]),
            "complete_score_free_feasible_region_certified": False,
            "full_protocol_infeasibility_claim_authorized": False,
            "split_gate0_authorized": False,
            "training_authorized": False,
        }
        try:
            _publish(output, failure)
        except BaseException:
            try:
                _seal_tree(output)
            except BaseException:
                pass
        raise
