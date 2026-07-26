#!/usr/bin/env python3
"""Run the single registered V6-A 10 percent pre-graph feasibility check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import stat
from pathlib import Path
from typing import Mapping, Sequence


FORMAT_VERSION = "v6_a_canonical_dry_run_preflight_receipt_v1"
REGISTRATION_FORMAT = "v6_a_canonical_dry_run_registration_v1"
AUTHORITY_FORMAT = "score_free_authority_v2"


class DryRunPreflightError(ValueError):
    """Raised when registration, input, or publication semantics differ."""


def canonical_json_bytes(payload: object) -> bytes:
    """Encode canonical ASCII JSON with one terminal LF."""
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


def _sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DryRunPreflightError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_object(payload_bytes: bytes, label: str) -> dict[str, object]:
    if not payload_bytes or payload_bytes.startswith(b"\xef\xbb\xbf"):
        raise DryRunPreflightError(f"{label} is empty or contains a BOM")
    try:
        text = payload_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise DryRunPreflightError(f"{label} must be ASCII JSON") from error

    def reject_duplicates(items: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise DryRunPreflightError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        result = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DryRunPreflightError(f"{label} contains non-finite {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise DryRunPreflightError(f"{label} is not strict JSON") from error
    if not isinstance(result, dict):
        raise DryRunPreflightError(f"{label} must be a JSON object")
    return result


def _read_verified(path: Path, expected_sha256: str, label: str) -> bytes:
    _digest(expected_sha256, label)
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    current = Path(absolute.parts[0])
    for part in absolute.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise DryRunPreflightError(f"{label} path contains a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DryRunPreflightError(f"{label} is not a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fingerprint = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if fingerprint(before) != fingerprint(after) or len(payload) != before.st_size:
        raise DryRunPreflightError(f"{label} changed during authentication")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DryRunPreflightError(f"{label} SHA-256 differs")
    return payload


def _ceiling_ratio(count: int, numerator: int, denominator: int) -> int:
    if count < 0 or numerator < 1 or denominator < numerator:
        raise DryRunPreflightError("registered selection fraction is invalid")
    return -(-(count * numerator) // denominator)


def _entity_ids(authority: Mapping[str, object], axis: str) -> tuple[str, ...]:
    key = "antibodies" if axis == "antibody" else "antigens"
    rows = authority.get(key)
    if not isinstance(rows, list):
        raise DryRunPreflightError(f"authority {key} are malformed")
    identifiers = tuple(str(row.get("entity_id")) for row in rows if isinstance(row, dict))
    if len(identifiers) != len(rows) or identifiers != tuple(sorted(set(identifiers))):
        raise DryRunPreflightError(f"authority {axis} IDs are not canonical")
    for identifier in identifiers:
        _digest(identifier, f"{axis} entity ID")
    return identifiers


def evaluate_preflight(
    authority: Mapping[str, object],
    registration: Mapping[str, object],
) -> dict[str, object]:
    """Select vertices before graph access and prove static necessary bounds."""
    if authority.get("format_version") != AUTHORITY_FORMAT:
        raise DryRunPreflightError("authority format differs")
    if registration.get("format_version") != REGISTRATION_FORMAT:
        raise DryRunPreflightError("registration format differs")
    if registration.get("single_registered_run") is not True:
        raise DryRunPreflightError("dry run must be registered as single use")
    selection = registration.get("selection")
    necessary = registration.get("necessary_condition")
    inventory = registration.get("authority_inventory")
    if not all(isinstance(value, dict) for value in (selection, necessary, inventory)):
        raise DryRunPreflightError("registration sections are malformed")
    if selection != {
        "axis_scope": "independently_per_antibody_and_antigen_axis",
        "fraction_denominator": 10,
        "fraction_numerator": 1,
        "graph_or_pair_incidence_may_be_inspected_before_selection": False,
        "order": "lowercase_sha256_entity_id_lexicographic",
        "rounding": "ceiling",
    }:
        raise DryRunPreflightError("registered selection rule differs")
    antibody_ids = _entity_ids(authority, "antibody")
    antigen_ids = _entity_ids(authority, "antigen")
    counts = authority.get("counts")
    if counts != inventory:
        raise DryRunPreflightError("authority inventory differs from registration")
    if len(antibody_ids) != inventory["antibody_entities"]:
        raise DryRunPreflightError("antibody count differs")
    if len(antigen_ids) != inventory["antigen_entities"]:
        raise DryRunPreflightError("antigen count differs")
    numerator = selection["fraction_numerator"]
    denominator = selection["fraction_denominator"]
    selected_antibodies = antibody_ids[
        : _ceiling_ratio(len(antibody_ids), numerator, denominator)
    ]
    selected_antigens = antigen_ids[
        : _ceiling_ratio(len(antigen_ids), numerator, denominator)
    ]
    fold_count = necessary.get("outer_fold_count")
    endpoint_minimum = necessary.get("minimum_validation_endpoints_per_axis_per_fold")
    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, int)
        or fold_count < 1
        or isinstance(endpoint_minimum, bool)
        or not isinstance(endpoint_minimum, int)
        or endpoint_minimum < 1
        or necessary.get("outer_holdout_entities_are_disjoint") is not True
    ):
        raise DryRunPreflightError("registered necessary condition differs")
    required_per_axis = fold_count * endpoint_minimum
    failures = []
    for axis, selected in (
        ("antibody", selected_antibodies),
        ("antigen", selected_antigens),
    ):
        if len(selected) < required_per_axis:
            failures.append(
                {
                    "axis": axis,
                    "inequality": f"{len(selected)} < {required_per_axis}",
                    "selected_entity_count": len(selected),
                    "required_distinct_outer_validation_entities": required_per_axis,
                    "reason": "three_disjoint_outer_validation_endpoint_minima_cannot_fit",
                }
            )
    payload: dict[str, object] = {
        "registered_run_id": registration.get("registered_run_id"),
        "selection_completed_before_graph_or_pair_incidence_access": True,
        "selected_antibody_count": len(selected_antibodies),
        "selected_antigen_count": len(selected_antigens),
        "selected_antibody_membership_sha256": _sha256(selected_antibodies),
        "selected_antigen_membership_sha256": _sha256(selected_antigens),
        "selected_antibody_ids": list(selected_antibodies),
        "selected_antigen_ids": list(selected_antigens),
        "static_necessary_condition": {
            "outer_fold_count": fold_count,
            "minimum_validation_endpoints_per_axis_per_fold": endpoint_minimum,
            "required_distinct_outer_validation_entities_per_axis": required_per_axis,
        },
        "static_failures": failures,
        "static_preflight_passed": not failures,
        "pre_graph_infeasibility_proven": bool(failures),
        "graph_or_pair_incidence_accessed": False,
        "cp_sat_invoked": False,
        "highs_invoked": False,
        "complete_exact_candidate_present": False,
        "dry_run_accepted": False,
        "v6_a_must_stop": bool(failures),
        "new_numbered_decomposition_protocol_required": bool(failures),
        "training_authorized": False,
    }
    payload["selection_and_bound_sha256"] = _sha256(payload)
    return payload


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _markdown(receipt: Mapping[str, object]) -> bytes:
    failures = receipt["preflight"]["static_failures"]
    lines = [
        "# V6-A canonical 10 percent dry-run preflight",
        "",
        f"- Run: `{receipt['registered_run_id']}`",
        f"- Status: `{receipt['terminal_status']}`",
        f"- Selected antibodies: {receipt['preflight']['selected_antibody_count']}",
        f"- Selected antigens: {receipt['preflight']['selected_antigen_count']}",
        "- Pair graph opened: no",
        "- Training authorized: no",
        "",
        "## Static proof",
        "",
    ]
    for failure in failures:
        lines.append(
            f"- {failure['axis']}: `{failure['inequality']}`; {failure['reason']}."
        )
    lines.append("")
    if receipt["preflight"]["pre_graph_infeasibility_proven"]:
        lines.append(
            "V6-A stops before solver construction. A separately numbered "
            "decomposition protocol is required; no threshold was changed."
        )
    else:
        lines.append(
            "The static preflight passed, but this receipt does not accept the "
            "solver dry run or authorize training."
        )
    lines.append("")
    return "\n".join(lines).encode("ascii")


def run(args: argparse.Namespace) -> dict[str, object]:
    """Authenticate inputs, evaluate once, and publish immutable evidence."""
    authority_bytes = _read_verified(
        args.authority,
        args.authority_sha256,
        "authority",
    )
    registration_bytes = _read_verified(
        args.registration,
        args.registration_sha256,
        "registration",
    )
    policy_bytes = _read_verified(args.policy, args.policy_sha256, "policy")
    authority = _strict_object(authority_bytes, "authority")
    registration = _strict_object(registration_bytes, "registration")
    _strict_object(policy_bytes, "policy")
    if registration.get("authority_file_sha256") != args.authority_sha256:
        raise DryRunPreflightError("registration authority binding differs")
    if registration.get("parent_policy_sha256") != args.policy_sha256:
        raise DryRunPreflightError("registration policy binding differs")
    preflight = evaluate_preflight(authority, registration)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "registered_run_id": registration["registered_run_id"],
        "input_sha256": {
            "authority": args.authority_sha256,
            "registration": args.registration_sha256,
            "policy": args.policy_sha256,
        },
        "preflight": preflight,
        "peak_rss_kib_linux": peak_rss,
        "terminal_status": (
            "STATIC_NECESSARY_CONDITION_INFEASIBLE"
            if preflight["pre_graph_infeasibility_proven"]
            else "READY_FOR_NUMBERED_SOLVER_DRY_RUN"
        ),
        "implementation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "complete_score_free_feasible_region_certified": False,
        "training_authorized": False,
    }
    payload["receipt_sha256"] = _sha256(payload)
    output = Path(os.path.abspath(os.path.expanduser(str(args.output_directory))))
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise FileExistsError("dry-run output must be new under an existing parent")
    os.mkdir(output, 0o700)
    _write_exclusive(output / "preflight_receipt.json", canonical_json_bytes(payload))
    _write_exclusive(output / "preflight_report.md", _markdown(payload))
    directory_descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    os.chmod(output, 0o500)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    receipt = run(parse_args(argv))
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
