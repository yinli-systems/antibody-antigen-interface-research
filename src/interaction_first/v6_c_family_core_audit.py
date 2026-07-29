"""Execute a registered V6-C constraint-family infeasibility audit."""

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
from . import v6_c_family_core_registration as registration_io
from . import v6_constraint_family_diagnostic as diagnostic
from . import v6_cp_sat_formulation as direct_cp
from . import v6_joint_ir


RECEIPT_FORMAT = "v6_c_family_core_audit_receipt_v1"
TERMINAL_IRREDUCIBLE = "V6_C_IRREDUCIBLE_LINEAR_FAMILY_CORE_DIAGNOSED"
TERMINAL_SUFFICIENT = "V6_C_SUFFICIENT_LINEAR_FAMILY_CORE_DIAGNOSED"
TERMINAL_NONPROOF = "V6_C_LINEAR_FAMILY_CORE_NONPROOF"


class V6CFamilyCoreAuditError(RuntimeError):
    """Raised when registered V6-C evidence is invalid or incomplete."""


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


def _strict_ascii_json(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V6CFamilyCoreAuditError(
                    f"{label} contains duplicate keys"
                )
            result[key] = value
        return result

    try:
        parsed = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise V6CFamilyCoreAuditError(
            f"{label} is not ASCII JSON"
        ) from error
    if not isinstance(parsed, dict):
        raise V6CFamilyCoreAuditError(f"{label} is not a JSON object")
    return parsed


def _verify_receipt_self_hash(
    payload: Mapping[str, Any],
    expected_sha256: str,
    label: str,
) -> None:
    claimed = payload.get("receipt_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    if (
        claimed != expected_sha256
        or sha256_payload(unsigned) != expected_sha256
    ):
        raise V6CFamilyCoreAuditError(
            f"{label} self hash differs"
        )


def _read_registered_receipt(
    reference: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    path = Path(str(reference["path"]))
    payload = production.read_verified(
        path,
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
            raise V6CFamilyCoreAuditError(
                f"{label} required field {name} differs"
            )
    return parsed


def _authenticate_parent_results(
    registration: Mapping[str, Any],
    *,
    v6_b_registration_sha256: str,
) -> dict[str, Any]:
    references = registration["parent_results"]
    assert isinstance(references, Mapping)
    payloads = {
        label: _read_registered_receipt(reference, f"V6-C parent {label}")
        for label, reference in references.items()
    }
    primary = payloads["primary_cp"]
    independent = payloads["independent_cp"]
    direct = payloads["direct_highs"]
    if primary.get("registration_sha256") != v6_b_registration_sha256:
        raise V6CFamilyCoreAuditError(
            "primary CP parent registration differs"
        )
    if (
        independent.get("registration_sha256")
        != v6_b_registration_sha256
        or independent.get("primary_result_file_sha256")
        != references["primary_cp"]["file_sha256"]
    ):
        raise V6CFamilyCoreAuditError(
            "independent CP parent lineage differs"
        )
    primary_anchor = direct.get("primary_cp_receipt")
    independent_anchor = direct.get("independent_cp_receipt")
    if (
        not isinstance(primary_anchor, Mapping)
        or primary_anchor.get("file_sha256")
        != references["primary_cp"]["file_sha256"]
        or primary_anchor.get("receipt_sha256")
        != references["primary_cp"]["receipt_sha256"]
        or not isinstance(independent_anchor, Mapping)
        or independent_anchor.get("file_sha256")
        != references["independent_cp"]["file_sha256"]
        or independent_anchor.get("receipt_sha256")
        != references["independent_cp"]["receipt_sha256"]
        or direct.get("cp_consensus_status") != "INFEASIBLE"
    ):
        raise V6CFamilyCoreAuditError(
            "direct-HiGHS CP lineage differs"
        )
    expected = registration["expected"]
    assert isinstance(expected, Mapping)
    bundle_anchor = direct.get("bundle_export_receipt")
    solver_anchor = direct.get("solver_receipt")
    if (
        not isinstance(bundle_anchor, Mapping)
        or bundle_anchor.get("bundle_sha256") != expected["bundle_sha256"]
        or not isinstance(solver_anchor, Mapping)
        or solver_anchor.get("linear_projection_infeasible") is not True
        or solver_anchor.get("linear_projection_certified") is not True
        or solver_anchor.get("linear_projection_optimal") is not False
    ):
        raise V6CFamilyCoreAuditError(
            "direct-HiGHS linear proof differs"
        )
    solver_receipt = production.strict_json(
        production.read_verified(
            Path(str(solver_anchor["path"])),
            str(solver_anchor["file_sha256"]),
            "direct-HiGHS nested solver receipt",
        ),
        "direct-HiGHS nested solver receipt",
    )
    _verify_receipt_self_hash(
        solver_receipt,
        str(solver_anchor["receipt_sha256"]),
        "direct-HiGHS nested solver receipt",
    )
    if (
        solver_receipt.get("bundle_sha256") != expected["bundle_sha256"]
        or solver_receipt.get("original_dimensions")
        != expected["dimensions"]
        or solver_receipt.get("linear_projection_infeasible") is not True
        or solver_receipt.get("linear_projection_certified") is not True
        or solver_receipt.get(
            "complete_score_free_feasible_region_certified"
        )
        is not False
    ):
        raise V6CFamilyCoreAuditError(
            "direct-HiGHS nested solver identity differs"
        )
    return {
        label: {
            "path": str(references[label]["path"]),
            "file_sha256": str(references[label]["file_sha256"]),
            "receipt_sha256": str(references[label]["receipt_sha256"]),
            "terminal_status": payloads[label]["terminal_status"],
        }
        for label in sorted(payloads)
    }


def _authenticate_policy(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    reference = registration["policy"]
    assert isinstance(reference, Mapping)
    payload = production.read_verified(
        Path(str(reference["path"])),
        str(reference["sha256"]),
        "V6-C policy",
    )
    policy = _strict_ascii_json(payload, "V6-C policy")
    authorization = policy.get("authorization")
    solver = policy.get("solver")
    if (
        policy.get("format_version")
        != "v6_c_linear_infeasibility_diagnostic_policy_v1"
        or not isinstance(authorization, Mapping)
        or any(value is not False for value in authorization.values())
        or not isinstance(solver, Mapping)
        or policy.get("authority_counts")
        != registration["expected"]["authority_counts"]
        or policy.get("parent_requirements")
        != {
            "direct_highs": (
                "DIRECT_HIGHS_LINEAR_INFEASIBLE_AGREEMENT"
            ),
            "independent_cp_sat": (
                "INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT"
            ),
            "primary_cp_sat": "PROVEN_INFEASIBLE",
            "same_linear_projection_required": True,
        }
        or solver.get("random_seed") != direct_cp.LOCKED_RANDOM_SEED
    ):
        raise V6CFamilyCoreAuditError("V6-C policy identity differs")
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
    parsed = production.strict_json(payload, "V6-B production registration")
    checked = production.validate_registration(parsed)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if (
        observed_sha256 != reference["sha256"]
        or production._sha256_payload(checked) != reference["sha256"]
        or checked["authority"]["counts"]
        != registration["expected"]["authority_counts"]
    ):
        raise V6CFamilyCoreAuditError(
            "V6-B production registration identity differs"
        )
    return checked, observed_sha256


def _authenticate_runtime(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = registration["runtime"]
    assert isinstance(runtime, Mapping)
    try:
        import ortools
    except ImportError as error:
        raise V6CFamilyCoreAuditError("OR-Tools is unavailable") from error
    executable = Path(sys.executable).resolve()
    ortools_path = Path(str(ortools.__file__)).resolve(strict=True)
    libc_family, libc_version = platform.libc_ver()
    observed = {
        "libc_family": libc_family,
        "libc_version": libc_version,
        "machine": platform.machine(),
        "ortools_module_sha256": _sha256_file(ortools_path),
        "ortools_version": ortools.__version__,
        "python_executable": str(executable),
        "python_version": platform.python_version(),
        "system": platform.system(),
    }
    expected = {
        name: runtime[name]
        for name in observed
    }
    if observed != expected:
        raise V6CFamilyCoreAuditError(
            "V6-C runtime identity differs"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise V6CFamilyCoreAuditError(
            "a GPU is visible to the V6-C solver"
        )
    locked = production.read_verified(
        Path(str(runtime["cp_sat_runtime_receipt_path"])),
        str(runtime["cp_sat_runtime_receipt_sha256"]),
        "V6-C CP-SAT runtime receipt",
    )
    return {
        **observed,
        "accelerator_visible_to_solver": False,
        "cuda_visible_devices": "",
        "cp_sat_runtime_receipt_path": str(
            runtime["cp_sat_runtime_receipt_path"]
        ),
        "cp_sat_runtime_receipt_sha256": hashlib.sha256(
            locked
        ).hexdigest(),
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
        raise V6CFamilyCoreAuditError(
            "V6-C source root is absent or symbolic"
        )
    files = source["files"]
    assert isinstance(files, Mapping)
    imported = {
        "audit": Path(__file__).resolve(strict=True),
        "cp_formulation": Path(str(direct_cp.__file__)).resolve(strict=True),
        "diagnostic": Path(str(diagnostic.__file__)).resolve(strict=True),
        "independent_auditor": Path(
            str(score_free_independent_auditor.__file__)
        ).resolve(strict=True),
        "joint_ir": Path(str(v6_joint_ir.__file__)).resolve(strict=True),
        "production": Path(str(production.__file__)).resolve(strict=True),
        "registration": Path(
            str(registration_io.__file__)
        ).resolve(strict=True),
    }
    receipts: dict[str, Any] = {}
    for label, reference in files.items():
        assert isinstance(reference, Mapping)
        registered_path = root / str(reference["path"])
        payload = production.read_verified(
            registered_path,
            str(reference["sha256"]),
            f"V6-C source {label}",
        )
        if label in imported and registered_path.resolve() != imported[label]:
            raise V6CFamilyCoreAuditError(
                f"imported V6-C source differs for {label}"
            )
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
        raise V6CFamilyCoreAuditError(
            "V6-C controller identity differs"
        )
    return receipts


def _projection_dimensions(direct_model: direct_cp.DirectCPModel) -> dict[str, int]:
    proto = direct_model.model.Proto()
    if any(not row.has_linear() for row in proto.constraints):
        raise V6CFamilyCoreAuditError(
            "V6-C source projection contains a nonlinear CP row"
        )
    return {
        "rows": len(proto.constraints),
        "columns": len(proto.variables),
        "nonzeros": sum(
            len(row.linear.vars) for row in proto.constraints
        ),
    }


def _reconstruct_direct_model(
    registration: Mapping[str, Any],
    v6_b_registration: Mapping[str, Any],
) -> tuple[direct_cp.DirectCPModel, dict[str, Any]]:
    authority_reference = v6_b_registration["authority"]
    assert isinstance(authority_reference, Mapping)
    published = production.read_verified(
        Path(str(authority_reference["path"])),
        str(authority_reference["sha256"]),
        "published score-free authority",
    )
    scientific, publication = production.extract_scientific_authority(
        published
    )
    normalized_primary, primary_normalization = (
        production.normalize_scientific_pair_order(scientific)
    )
    if (
        publication["scientific_payload_sha256"]
        != authority_reference["scientific_payload_sha256"]
    ):
        raise V6CFamilyCoreAuditError(
            "published scientific authority differs"
        )
    authority_bytes, manifest_bytes, edge_bytes, raw_receipts = (
        production._registered_input_bytes(v6_b_registration)
    )
    primary_payload = production.strict_json(
        normalized_primary,
        "normalized score-free authority",
    )
    implementation_hashes = primary_payload.get(
        "implementation_source_sha256"
    )
    if not isinstance(implementation_hashes, Mapping):
        raise V6CFamilyCoreAuditError(
            "authority implementation anchors are absent"
        )
    independent = (
        score_free_independent_auditor.reconstruct_score_free_authority(
            authority_bytes=authority_bytes,
            manifest_bytes=manifest_bytes,
            edge_ledger_bytes=edge_bytes,
            expected_primary_implementation_sha256=implementation_hashes,
            expected_counts=registration_io.EXPECTED_COUNTS,
        )
    )
    normalized_independent, independent_normalization = (
        production.normalize_scientific_pair_order(
            production.canonical_json_bytes(independent)
        )
    )
    if normalized_independent != normalized_primary:
        raise V6CFamilyCoreAuditError(
            "primary and independent score-free authority differ"
        )
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
    source_files = registration["source"]["files"]
    exact_source = v6_b_registration["source"]["files"][
        "v6_exact_separation_oracle"
    ]["sha256"]
    ir = v6_joint_ir.build_v6_joint_ir_from_direct_cp_sat(
        direct_model,
        solver_policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        direct_formulation_source_sha256=source_files[
            "cp_formulation"
        ]["sha256"],
        exact_oracle_source_sha256=exact_source,
    )
    expected = registration["expected"]
    dimensions = _projection_dimensions(direct_model)
    if (
        dimensions != expected["dimensions"]
        or ir.source_linear_projection_sha256
        != expected["source_linear_projection_sha256"]
    ):
        raise V6CFamilyCoreAuditError(
            "rebuilt linear projection identity differs"
        )
    identity = {
        "authority_sha256": authority_sha256,
        "publication": publication,
        "primary_normalization": primary_normalization,
        "independent_normalization": independent_normalization,
        "raw_input_receipts": raw_receipts,
        "primary_independent_authority_agreement": True,
        "direct_formulation_sha256": direct_model.formulation_sha256,
        "source_linear_projection_sha256": (
            ir.source_linear_projection_sha256
        ),
        "ir_sha256": ir.ir_sha256,
        "dimensions": dimensions,
        "constraint_family_counts": dict(
            direct_model.constraint_family_counts
        ),
    }
    del ir
    gc.collect()
    return direct_model, identity


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(
        os,
        "O_CLOEXEC",
        0,
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


def _seal_output(output: Path) -> None:
    for path in output.iterdir():
        if path.is_file() and not path.is_symlink():
            os.chmod(
                path,
                stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH,
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


def _publish(output: Path, receipt: dict[str, Any]) -> None:
    diagnostic_receipt = receipt.get("diagnostic_receipt")
    core = (
        diagnostic_receipt.get("sufficient_core", [])
        if isinstance(diagnostic_receipt, Mapping)
        else []
    )
    report = (
        "# V6-C linear constraint-family diagnosis\n\n"
        f"Terminal status: `{receipt['terminal_status']}`\n\n"
        f"Reported family count: {len(core)}\n\n"
        "This result diagnoses only the common linear projection. It does not "
        "authorize a split, threshold change, training, model selection, "
        "outer evaluation, or test access.\n"
    ).encode("ascii")
    receipt["report_sha256"] = hashlib.sha256(report).hexdigest()
    receipt["receipt_sha256"] = sha256_payload(receipt)
    _write_exclusive(output / "receipt.json", canonical_json_bytes(receipt))
    _write_exclusive(output / "report.md", report)
    _seal_output(output)


def execute_family_core_audit(
    *,
    registration: Mapping[str, Any],
    registration_sha256: str,
    controller_path: Path,
    controller_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Authenticate, rebuild, diagnose, and publish one V6-C attempt."""
    checked = registration_io.validate_family_core_registration(
        registration
    )
    if sha256_payload(checked) != registration_sha256:
        raise V6CFamilyCoreAuditError(
            "V6-C registration digest differs"
        )
    output = Path(
        os.path.abspath(os.path.expanduser(str(output_directory)))
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"create-only V6-C output already exists: {output}"
        )
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise V6CFamilyCoreAuditError(
            "V6-C output parent is absent or symbolic"
        )
    os.mkdir(output, 0o700)
    started_ns = time.monotonic_ns()
    try:
        policy_receipt = _authenticate_policy(checked)
        v6_b_registration, v6_b_sha256 = (
            _authenticate_v6_b_registration(checked)
        )
        parent_receipts = _authenticate_parent_results(
            checked,
            v6_b_registration_sha256=v6_b_sha256,
        )
        runtime_receipt = _authenticate_runtime(checked)
        source_receipts = _authenticate_sources(
            checked,
            controller_path=controller_path,
            controller_sha256=controller_sha256,
        )
        direct_model, projection_identity = _reconstruct_direct_model(
            checked,
            v6_b_registration,
        )
        limits = checked["limits"]
        assert isinstance(limits, Mapping)
        diagnostic_receipt = diagnostic.diagnose_constraint_families(
            direct_model,
            maximum_time_seconds_per_solve=float(
                limits["maximum_time_seconds_per_solve"]
            ),
            minimize_core=bool(limits["minimize_core"]),
        )
        sufficient = (
            diagnostic_receipt[
                "linear_projection_infeasibility_diagnosed"
            ]
            is True
        )
        irreducible = (
            sufficient
            and diagnostic_receipt["irreducibility_proven"] is True
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
            "parent_receipts": parent_receipts,
            "runtime_receipt": runtime_receipt,
            "source_receipts": source_receipts,
            "projection_identity": projection_identity,
            "diagnostic_receipt": diagnostic_receipt,
            "common_linear_projection_infeasibility_diagnosed": (
                sufficient
            ),
            "irreducible_family_core_proven": irreducible,
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
            "terminal_status": "V6_C_FAMILY_CORE_AUDIT_EXCEPTION",
            "registration_sha256": registration_sha256,
            "controller_sha256": controller_sha256,
            "elapsed_ns": time.monotonic_ns() - started_ns,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_sha256": sha256_payload(
                [type(error).__name__, str(error)]
            ),
            "common_linear_projection_infeasibility_diagnosed": False,
            "irreducible_family_core_proven": False,
            "minimum_cardinality_core_claim_authorized": False,
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
                _seal_output(output)
            except BaseException:
                pass
        raise


__all__ = [
    "RECEIPT_FORMAT",
    "TERMINAL_IRREDUCIBLE",
    "TERMINAL_NONPROOF",
    "TERMINAL_SUFFICIENT",
    "V6CFamilyCoreAuditError",
    "canonical_json_bytes",
    "execute_family_core_audit",
    "sha256_payload",
]
