"""Execute a registered V6-D two-CP family-core audit."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import sys
import time
from typing import Any, Mapping

from . import score_free_independent_auditor
from . import v6_b_production as production
from . import v6_constraint_family_diagnostic as diagnostic
from . import v6_cp_sat_formulation as direct_cp
from . import v6_d_core_projection as core_projection
from . import v6_d_two_cp_core_registration as registration_io
from . import v6_joint_ir


RECEIPT_FORMAT = "v6_d_two_cp_family_core_audit_receipt_v1"
TERMINAL_IRREDUCIBLE = "V6_D_IRREDUCIBLE_LINEAR_FAMILY_CORE_EXPORTED"
TERMINAL_SUFFICIENT = "V6_D_SUFFICIENT_LINEAR_FAMILY_CORE_EXPORTED"
TERMINAL_NONPROOF = "V6_D_LINEAR_FAMILY_CORE_NONPROOF"


class V6DTwoCPCoreAuditError(RuntimeError):
    """Raised when registered V6-D evidence is invalid or incomplete."""


def canonical_json_bytes(payload: object) -> bytes:
    """Encode one canonical ASCII JSON payload."""
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_receipt_self_hash(
    payload: Mapping[str, Any],
    expected_sha256: str,
    label: str,
) -> None:
    unsigned = dict(payload)
    claimed = unsigned.pop("receipt_sha256", None)
    if claimed != expected_sha256 or sha256_payload(unsigned) != expected_sha256:
        raise V6DTwoCPCoreAuditError(f"{label} self hash differs")


def _read_registered_receipt(
    reference: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    payload = production.read_verified(
        Path(str(reference["path"])),
        str(reference["file_sha256"]),
        label,
    )
    parsed = production.strict_json(payload, label)
    _verify_receipt_self_hash(
        parsed,
        str(reference["receipt_sha256"]),
        label,
    )
    required = reference["required_fields"]
    assert isinstance(required, Mapping)
    for name, expected in required.items():
        observed = parsed.get(name)
        if (
            observed is not expected
            if isinstance(expected, bool)
            else observed != expected
        ):
            raise V6DTwoCPCoreAuditError(f"{label} required field {name} differs")
    return parsed


def _authenticate_parents(
    registration: Mapping[str, Any],
    *,
    v6_b_registration_sha256: str,
) -> dict[str, Any]:
    references = registration["parent_results"]
    assert isinstance(references, Mapping)
    primary = _read_registered_receipt(
        references["primary_cp"],
        "V6-D primary CP parent",
    )
    independent = _read_registered_receipt(
        references["independent_cp"],
        "V6-D independent CP parent",
    )
    if primary.get("registration_sha256") != v6_b_registration_sha256:
        raise V6DTwoCPCoreAuditError("primary CP registration differs")
    if (
        independent.get("registration_sha256") != v6_b_registration_sha256
        or independent.get("primary_result_file_sha256")
        != references["primary_cp"]["file_sha256"]
        or independent.get("ir_sha256") != registration["expected"]["parent_ir_sha256"]
    ):
        raise V6DTwoCPCoreAuditError("independent CP lineage differs")
    primary_projection = primary.get("decomposition_result")
    independent_projection = independent.get("projection_receipt")
    if (
        not isinstance(primary_projection, Mapping)
        or primary_projection.get("terminal_status") != "PROVEN_INFEASIBLE"
        or primary_projection.get("candidate_present") is not False
        or not isinstance(independent_projection, Mapping)
        or independent_projection.get("status") != "INFEASIBLE"
        or independent_projection.get("accepted_projection_terminal_status") is not True
    ):
        raise V6DTwoCPCoreAuditError("two-CP infeasibility evidence differs")
    return {
        "primary_cp": {
            "path": str(references["primary_cp"]["path"]),
            "file_sha256": str(references["primary_cp"]["file_sha256"]),
            "receipt_sha256": str(references["primary_cp"]["receipt_sha256"]),
            "terminal_status": primary["terminal_status"],
        },
        "independent_cp": {
            "path": str(references["independent_cp"]["path"]),
            "file_sha256": str(references["independent_cp"]["file_sha256"]),
            "receipt_sha256": str(references["independent_cp"]["receipt_sha256"]),
            "terminal_status": independent["terminal_status"],
        },
        "primary_independent_linear_agreement": True,
        "direct_highs_full_model_proof_consumed": False,
    }


def _authenticate_policy(registration: Mapping[str, Any]) -> dict[str, Any]:
    reference = registration["policy"]
    assert isinstance(reference, Mapping)
    payload = production.read_verified(
        Path(str(reference["path"])),
        str(reference["sha256"]),
        "V6-D policy",
    )
    policy = production.strict_json(payload, "V6-D policy")
    if (
        policy.get("format_version") != "v6_d_two_cp_consensus_core_policy_v1"
        or policy.get("authority_counts")
        != registration["expected"]["authority_counts"]
        or policy.get("parent_requirements")
        != {
            "independent_cp_sat": ("INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT"),
            "primary_cp_sat": "PROVEN_INFEASIBLE",
            "same_linear_projection_required": True,
        }
        or any(value is not False for value in policy.get("authorization", {}).values())
        or policy.get("solver", {}).get("random_seed") != direct_cp.LOCKED_RANDOM_SEED
        or policy.get("highs_core_audit", {}).get("full_model_timeout_is_not_a_proof")
        is not True
    ):
        raise V6DTwoCPCoreAuditError("V6-D policy identity differs")
    return {
        "path": str(reference["path"]),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
    }


def _authenticate_v6_b_registration(
    registration: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    reference = registration["v6_b_registration"]
    assert isinstance(reference, Mapping)
    payload = production.read_verified(
        Path(str(reference["path"])),
        str(reference["sha256"]),
        "V6-B production registration",
    )
    checked = production.validate_registration(
        production.strict_json(payload, "V6-B production registration")
    )
    observed = hashlib.sha256(payload).hexdigest()
    if (
        production._sha256_payload(checked) != observed
        or checked["authority"]["counts"]
        != registration["expected"]["authority_counts"]
    ):
        raise V6DTwoCPCoreAuditError("V6-B registration identity differs")
    return checked, observed


def _authenticate_runtime(registration: Mapping[str, Any]) -> dict[str, Any]:
    runtime = registration["runtime"]
    assert isinstance(runtime, Mapping)
    try:
        import ortools
    except ImportError as error:
        raise V6DTwoCPCoreAuditError("OR-Tools is unavailable") from error
    libc_family, libc_version = platform.libc_ver()
    observed = {
        "libc_family": libc_family,
        "libc_version": libc_version,
        "machine": platform.machine(),
        "ortools_module_sha256": _sha256_file(
            Path(str(ortools.__file__)).resolve(strict=True)
        ),
        "ortools_version": ortools.__version__,
        "python_executable": str(Path(sys.executable).absolute()),
        "python_version": platform.python_version(),
        "system": platform.system(),
    }
    if observed != {name: runtime[name] for name in observed}:
        raise V6DTwoCPCoreAuditError("V6-D runtime identity differs")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise V6DTwoCPCoreAuditError("a GPU is visible to the V6-D solver")
    locked = production.read_verified(
        Path(str(runtime["cp_sat_runtime_receipt_path"])),
        str(runtime["cp_sat_runtime_receipt_sha256"]),
        "V6-D CP-SAT runtime receipt",
    )
    return {
        **observed,
        "accelerator_visible_to_solver": False,
        "cp_sat_runtime_receipt_sha256": hashlib.sha256(locked).hexdigest(),
    }


def _authenticate_sources(
    registration: Mapping[str, Any],
    *,
    controller_path: Path,
    controller_sha256: str,
) -> dict[str, Any]:
    source = registration["source"]
    assert isinstance(source, Mapping)
    root = Path(str(source["source_root"]))
    if not root.is_dir() or root.is_symlink():
        raise V6DTwoCPCoreAuditError("V6-D source root is invalid")
    files = source["files"]
    assert isinstance(files, Mapping)
    imported = {
        "audit": Path(__file__).resolve(strict=True),
        "core_projection": Path(str(core_projection.__file__)).resolve(strict=True),
        "cp_formulation": Path(str(direct_cp.__file__)).resolve(strict=True),
        "diagnostic": Path(str(diagnostic.__file__)).resolve(strict=True),
        "independent_auditor": Path(
            str(score_free_independent_auditor.__file__)
        ).resolve(strict=True),
        "joint_ir": Path(str(v6_joint_ir.__file__)).resolve(strict=True),
        "production": Path(str(production.__file__)).resolve(strict=True),
        "registration": Path(str(registration_io.__file__)).resolve(strict=True),
    }
    receipts = {}
    for label, reference in files.items():
        registered_path = root / str(reference["path"])
        payload = production.read_verified(
            registered_path,
            str(reference["sha256"]),
            f"V6-D source {label}",
        )
        if label in imported and registered_path.resolve() != imported[label]:
            raise V6DTwoCPCoreAuditError(f"imported V6-D source differs for {label}")
        receipts[label] = {
            "path": str(registered_path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        }
    registered_controller = root / str(files["controller"]["path"])
    if (
        registered_controller.resolve() != controller_path.resolve()
        or receipts["controller"]["sha256"] != controller_sha256
    ):
        raise V6DTwoCPCoreAuditError("V6-D controller identity differs")
    return receipts


def _projection_dimensions(model: direct_cp.DirectCPModel) -> dict[str, int]:
    proto = model.model.Proto()
    return {
        "rows": len(proto.constraints),
        "columns": len(proto.variables),
        "nonzeros": sum(len(row.linear.vars) for row in proto.constraints),
    }


def _reconstruct_direct_model(
    registration: Mapping[str, Any],
    v6_b_registration: Mapping[str, Any],
) -> tuple[direct_cp.DirectCPModel, dict[str, Any], str, str]:
    authority_reference = v6_b_registration["authority"]
    published = production.read_verified(
        Path(str(authority_reference["path"])),
        str(authority_reference["sha256"]),
        "published score-free authority",
    )
    scientific, publication = production.extract_scientific_authority(published)
    normalized_primary, primary_normalization = (
        production.normalize_scientific_pair_order(scientific)
    )
    if (
        publication["scientific_payload_sha256"]
        != authority_reference["scientific_payload_sha256"]
    ):
        raise V6DTwoCPCoreAuditError("published authority identity differs")
    authority_bytes, manifest_bytes, edge_bytes, raw_receipts = (
        production._registered_input_bytes(v6_b_registration)
    )
    primary_payload = production.strict_json(
        normalized_primary,
        "normalized score-free authority",
    )
    independent = score_free_independent_auditor.reconstruct_score_free_authority(
        authority_bytes=authority_bytes,
        manifest_bytes=manifest_bytes,
        edge_ledger_bytes=edge_bytes,
        expected_primary_implementation_sha256=primary_payload[
            "implementation_source_sha256"
        ],
        expected_counts=registration_io.EXPECTED_COUNTS,
    )
    normalized_independent, independent_normalization = (
        production.normalize_scientific_pair_order(
            production.canonical_json_bytes(independent)
        )
    )
    if normalized_independent != normalized_primary:
        raise V6DTwoCPCoreAuditError("independent authority bytes differ")
    authority_sha256 = hashlib.sha256(normalized_primary).hexdigest()
    problem = direct_cp.direct_problem_from_authority_payload(
        normalized_primary,
        edge_bytes,
        expected_authority_payload_sha256=authority_sha256,
        expected_counts=registration_io.EXPECTED_COUNTS,
    )
    direct_model = direct_cp.build_independent_cp_sat_formulation(problem)
    policy_reference = v6_b_registration["parent_evidence"]["v6_b_policy"]
    policy_bytes = production.read_verified(
        Path(str(policy_reference["path"])),
        str(policy_reference["sha256"]),
        "V6-B solver policy",
    )
    policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    exact_source_sha256 = v6_b_registration["source"]["files"][
        "v6_exact_separation_oracle"
    ]["sha256"]
    source_files = registration["source"]["files"]
    ir = v6_joint_ir.build_v6_joint_ir_from_direct_cp_sat(
        direct_model,
        solver_policy_sha256=policy_sha256,
        direct_formulation_source_sha256=source_files["cp_formulation"]["sha256"],
        exact_oracle_source_sha256=exact_source_sha256,
    )
    dimensions = _projection_dimensions(direct_model)
    expected = registration["expected"]
    if (
        dimensions != expected["dimensions"]
        or ir.source_linear_projection_sha256
        != expected["source_linear_projection_sha256"]
    ):
        raise V6DTwoCPCoreAuditError("rebuilt projection identity differs")
    identity = {
        "authority_sha256": authority_sha256,
        "publication": publication,
        "primary_normalization": primary_normalization,
        "independent_normalization": independent_normalization,
        "raw_input_receipts": raw_receipts,
        "primary_independent_authority_agreement": True,
        "direct_formulation_sha256": direct_model.formulation_sha256,
        "source_linear_projection_sha256": ir.source_linear_projection_sha256,
        "ir_sha256": ir.ir_sha256,
        "dimensions": dimensions,
        "constraint_family_counts": dict(direct_model.constraint_family_counts),
    }
    del ir
    gc.collect()
    return direct_model, identity, policy_sha256, exact_source_sha256


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


def _seal(output: Path) -> None:
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


def _publish(output: Path, receipt: dict[str, Any]) -> None:
    report = (
        "# V6-D two-CP family-core audit\n\n"
        f"Terminal status: `{receipt['terminal_status']}`\n\n"
        "This result is diagnostic only. It does not authorize Gate 0, "
        "training, selection, outer evaluation, or test access.\n"
    ).encode("ascii")
    receipt["report_sha256"] = hashlib.sha256(report).hexdigest()
    receipt["receipt_sha256"] = sha256_payload(receipt)
    _write_exclusive(output / "receipt.json", canonical_json_bytes(receipt))
    _write_exclusive(output / "report.md", report)
    _seal(output)


def execute_two_cp_core_audit(
    *,
    registration: Mapping[str, Any],
    registration_sha256: str,
    controller_path: Path,
    controller_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Authenticate, rebuild, diagnose, export, and publish one V6-D attempt."""
    checked = registration_io.validate_two_cp_core_registration(registration)
    if sha256_payload(checked) != registration_sha256:
        raise V6DTwoCPCoreAuditError("V6-D registration digest differs")
    output = Path(os.path.abspath(os.path.expanduser(str(output_directory))))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only V6-D output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise V6DTwoCPCoreAuditError("V6-D output parent is invalid")
    os.mkdir(output, 0o700)
    started_ns = time.monotonic_ns()
    try:
        policy_receipt = _authenticate_policy(checked)
        v6_b, v6_b_sha256 = _authenticate_v6_b_registration(checked)
        parents = _authenticate_parents(
            checked,
            v6_b_registration_sha256=v6_b_sha256,
        )
        runtime = _authenticate_runtime(checked)
        sources = _authenticate_sources(
            checked,
            controller_path=controller_path,
            controller_sha256=controller_sha256,
        )
        direct_model, projection_identity, policy_sha256, exact_source_sha256 = (
            _reconstruct_direct_model(checked, v6_b)
        )
        limits = checked["limits"]
        diagnostic_receipt = diagnostic.diagnose_constraint_families(
            direct_model,
            maximum_time_seconds_per_solve=float(
                limits["maximum_time_seconds_per_solve"]
            ),
            minimize_core=bool(limits["minimize_core"]),
        )
        sufficient = (
            diagnostic_receipt["linear_projection_infeasibility_diagnosed"] is True
        )
        irreducible = sufficient and (
            diagnostic_receipt["irreducibility_proven"] is True
        )
        core_receipt = None
        if sufficient:
            core_receipt = core_projection.export_exact_family_core_bundle(
                direct_model,
                diagnostic_receipt["sufficient_core"],
                solver_policy_sha256=policy_sha256,
                direct_formulation_source_sha256=checked["source"]["files"][
                    "cp_formulation"
                ]["sha256"],
                exact_oracle_source_sha256=exact_source_sha256,
                diagnostic_receipt_sha256=diagnostic_receipt["receipt_sha256"],
                output_directory=output / "core_projection",
            )
        terminal = (
            TERMINAL_IRREDUCIBLE
            if irreducible
            else TERMINAL_SUFFICIENT
            if sufficient
            else TERMINAL_NONPROOF
        )
        receipt: dict[str, Any] = {
            "format_version": RECEIPT_FORMAT,
            "terminal_status": terminal,
            "registration_sha256": registration_sha256,
            "controller_sha256": controller_sha256,
            "elapsed_ns": time.monotonic_ns() - started_ns,
            "policy_receipt": policy_receipt,
            "v6_b_registration_sha256": v6_b_sha256,
            "parent_receipts": parents,
            "runtime_receipt": runtime,
            "source_receipts": sources,
            "projection_identity": projection_identity,
            "diagnostic_receipt": diagnostic_receipt,
            "core_projection_receipt": core_receipt,
            "two_cp_consensus_consumed": True,
            "direct_highs_full_model_proof_consumed": False,
            "common_linear_projection_infeasibility_diagnosed": sufficient,
            "irreducible_family_core_proven": irreducible,
            "highs_core_audit_completed": False,
            "minimum_cardinality_core_claim_authorized": False,
            "full_protocol_infeasibility_claim_authorized": False,
            "threshold_change_authorized": False,
            "split_gate0_authorized": False,
            "training_authorized": False,
            "model_selection_authorized": False,
            "outer_evaluation_authorized": False,
            "test_access_authorized": False,
        }
        _publish(output, receipt)
        return receipt
    except BaseException as error:
        failure: dict[str, Any] = {
            "format_version": RECEIPT_FORMAT,
            "terminal_status": "V6_D_TWO_CP_CORE_AUDIT_EXCEPTION",
            "registration_sha256": registration_sha256,
            "controller_sha256": controller_sha256,
            "elapsed_ns": time.monotonic_ns() - started_ns,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_sha256": sha256_payload([type(error).__name__, str(error)]),
            "common_linear_projection_infeasibility_diagnosed": False,
            "irreducible_family_core_proven": False,
            "highs_core_audit_completed": False,
            "full_protocol_infeasibility_claim_authorized": False,
            "threshold_change_authorized": False,
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
                _seal(output)
            except BaseException:
                pass
        raise


__all__ = [
    "RECEIPT_FORMAT",
    "TERMINAL_IRREDUCIBLE",
    "TERMINAL_NONPROOF",
    "TERMINAL_SUFFICIENT",
    "V6DTwoCPCoreAuditError",
    "canonical_json_bytes",
    "execute_two_cp_core_audit",
    "sha256_payload",
]
