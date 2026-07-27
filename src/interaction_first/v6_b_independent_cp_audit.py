"""Independent full-linear CP-SAT audit for a terminal V6-B production result."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path
from typing import Mapping

from . import score_free_independent_auditor
from . import v6_b_production as production
from . import v6_cp_sat_formulation as direct_cp
from . import v6_joint_ir


RECEIPT_FORMAT = "v6_b_independent_full_linear_cp_audit_receipt_v1"


class IndependentCPAuditError(RuntimeError):
    """Raised when independent CP audit evidence is invalid or incomplete."""


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


def validate_primary_result(
    payload: Mapping[str, object],
    *,
    expected_registration_sha256: str,
) -> dict[str, object]:
    """Require the exact no-candidate primary terminal state under audit."""
    if payload.get("registration_sha256") != expected_registration_sha256:
        raise IndependentCPAuditError("primary result registration differs")
    if payload.get("terminal_status") != "PROVEN_INFEASIBLE":
        raise IndependentCPAuditError("primary result is not terminal infeasible")
    result = payload.get("decomposition_result")
    if not isinstance(result, Mapping):
        raise IndependentCPAuditError("primary decomposition result is absent")
    required = {
        "terminal_status": "PROVEN_INFEASIBLE",
        "complete": True,
        "proven_infeasible": True,
        "candidate_present": False,
        "combined_candidate_count": 0,
        "all_66_exact_gates_passed": False,
    }
    if any(result.get(name) != value for name, value in required.items()):
        raise IndependentCPAuditError("primary decomposition state differs")
    if result.get("oracle_history") != []:
        raise IndependentCPAuditError("primary result unexpectedly evaluated exact gates")
    if payload.get("direct_highs_fixed_assignment_audit_completed") is not False:
        raise IndependentCPAuditError("primary result claims an inapplicable HiGHS audit")
    if payload.get("split_gate0_authorized") is not False:
        raise IndependentCPAuditError("primary result authorizes Gate 0")
    if payload.get("training_authorized") is not False:
        raise IndependentCPAuditError("primary result authorizes training")
    return dict(payload)


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


def _authenticate_registered_sources(
    registration: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate registered bytes and equal imported scientific module bytes."""
    source = registration["source"]
    assert isinstance(source, Mapping)
    source_root = Path(str(source["source_root"]))
    files = source["files"]
    assert isinstance(files, Mapping)
    receipts: dict[str, object] = {}
    for label, reference in files.items():
        assert isinstance(reference, Mapping)
        registered_path = source_root / str(reference["path"])
        registered_bytes = production.read_verified(
            registered_path,
            str(reference["sha256"]),
            f"registered source {label}",
        )
        imported_sha256 = None
        module = production.EXPECTED_SOURCE_FILES.get(label)
        if module is not None:
            module_path = Path(str(module.__file__)).resolve(strict=True)
            imported_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
            if imported_sha256 != reference["sha256"]:
                raise IndependentCPAuditError(
                    f"imported source bytes differ for {label}"
                )
        receipts[label] = {
            "registered_path": str(registered_path),
            "registered_sha256": hashlib.sha256(registered_bytes).hexdigest(),
            "imported_sha256": imported_sha256,
            "byte_size": len(registered_bytes),
        }
    return receipts


def _publish(output: Path, receipt: dict[str, object]) -> None:
    receipt["receipt_sha256"] = sha256_payload(receipt)
    report = (
        "# V6-B independent full-linear CP audit\n\n"
        f"Terminal status: `{receipt['terminal_status']}`\n\n"
        "This audit is linear-only, does not authorize Gate 0, and does not "
        "authorize training.\n"
    ).encode("ascii")
    receipt["report_sha256"] = hashlib.sha256(report).hexdigest()
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = sha256_payload(receipt)
    _write_exclusive(output / "receipt.json", canonical_json_bytes(receipt))
    _write_exclusive(output / "report.md", report)
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


def execute_independent_cp_audit(
    *,
    registration: Mapping[str, object],
    registration_sha256: str,
    primary_result: Mapping[str, object],
    primary_result_file_sha256: str,
    controller_sha256: str,
    output_directory: Path,
) -> dict[str, object]:
    """Rebuild raw authority and solve the independent full linear formulation."""
    checked = production.validate_registration(registration)
    if sha256_payload(checked) != registration_sha256:
        raise IndependentCPAuditError("registration digest differs")
    validate_primary_result(
        primary_result,
        expected_registration_sha256=registration_sha256,
    )
    output = Path(os.path.abspath(os.path.expanduser(str(output_directory))))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise IndependentCPAuditError("output parent is absent or symbolic")
    os.mkdir(output, 0o700)
    started_ns = time.monotonic_ns()
    try:
        runtime_receipt = production._runtime_receipt(checked)
        source_receipts = _authenticate_registered_sources(checked)
        parent_receipts = production._parent_evidence_receipts(checked)
        authority_reference = checked["authority"]
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
            production._registered_input_bytes(checked)
        )
        primary_payload = production.strict_json(
            normalized_primary,
            "normalized primary authority",
        )
        independent_payload = (
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
                production.canonical_json_bytes(independent_payload)
            )
        )
        if normalized_independent != normalized_primary:
            raise IndependentCPAuditError("independent authority bytes differ")
        authority_sha256 = hashlib.sha256(normalized_primary).hexdigest()
        problem = direct_cp.direct_problem_from_authority_payload(
            normalized_primary,
            edge_bytes,
            expected_authority_payload_sha256=authority_sha256,
            expected_counts=production.EXPECTED_COUNTS,
        )
        direct_model = direct_cp.build_independent_cp_sat_formulation(problem)
        source_files = checked["source"]["files"]
        parent = checked["parent_evidence"]
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
        limits = checked["limits"]
        projection = direct_cp.solve_direct_linear_projection(
            direct_model,
            maximum_time_seconds=float(limits["maximum_time_seconds_per_solve"]),
        )
        independent_agreement = (
            projection["status"] == "INFEASIBLE"
            and projection["accepted_projection_terminal_status"] is True
        )
        terminal = (
            "INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT"
            if independent_agreement
            else "INDEPENDENT_DIRECT_CP_DISAGREEMENT_OR_NONPROOF"
        )
        receipt: dict[str, object] = {
            "format_version": RECEIPT_FORMAT,
            "terminal_status": terminal,
            "registration_sha256": registration_sha256,
            "primary_result_file_sha256": primary_result_file_sha256,
            "controller_sha256": controller_sha256,
            "elapsed_ns": time.monotonic_ns() - started_ns,
            "runtime_receipt": runtime_receipt,
            "source_receipts": source_receipts,
            "parent_evidence_receipts": parent_receipts,
            "authority_publication": publication,
            "primary_normalization": primary_normalization,
            "independent_normalization": independent_normalization,
            "primary_independent_authority_byte_agreement": True,
            "normalized_authority_sha256": authority_sha256,
            "raw_input_receipts": raw_receipts,
            "direct_formulation_sha256": direct_model.formulation_sha256,
            "ir_sha256": ir.ir_sha256,
            "dimensions": {
                "variables": len(ir.variable_keys),
                "linear_constraints": len(ir.linear_constraints),
                "exact_gate_declarations": len(ir.separation_gates),
            },
            "projection_receipt": projection,
            "primary_independent_linear_agreement": independent_agreement,
            "historical_milp_matrix_consumed": False,
            "v6_b_partition_or_cut_consumed": False,
            "highs_artifact_consumed": False,
            "exact_score_free_separation_required_after_linear_infeasibility": False,
            "direct_highs_audit_completed": False,
            "full_protocol_infeasibility_claim_authorized": False,
            "split_gate0_authorized": False,
            "training_authorized": False,
        }
        _publish(output, receipt)
        return receipt
    except BaseException as error:
        failure = {
            "format_version": RECEIPT_FORMAT,
            "terminal_status": "INDEPENDENT_DIRECT_CP_AUDIT_EXCEPTION",
            "registration_sha256": registration_sha256,
            "primary_result_file_sha256": primary_result_file_sha256,
            "controller_sha256": controller_sha256,
            "elapsed_ns": time.monotonic_ns() - started_ns,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_sha256": sha256_payload([type(error).__name__, str(error)]),
            "direct_highs_audit_completed": False,
            "full_protocol_infeasibility_claim_authorized": False,
            "split_gate0_authorized": False,
            "training_authorized": False,
        }
        try:
            _publish(output, failure)
        except BaseException:
            pass
        raise
