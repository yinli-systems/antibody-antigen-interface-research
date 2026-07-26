#!/usr/bin/env python3
"""Create physical V6-A train-only inputs from authenticated historical evidence.

The controller uses only the train-authority membership and exact direct graphs
from an externally anchored historical receipt. Historical fold assignments are
ignored. The mixed canonical manifest is scanned by its first identifier field;
sequence columns are parsed only after that identifier is known to be authorized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ASSIGNMENT_FORMAT = "train_only_nested_direct80_four_view_neighborhood_folds_v2"
AUDIT_FORMAT = "train_only_nested_direct80_four_view_independent_audit_v2"
AUTHORITY_FORMAT = "gate0_physical_train_only_allowlist_v1"
EDGE_FORMAT = "score_free_direct_entity_edges_v1"
RECEIPT_FORMAT = "v6_a_score_free_input_derivation_receipt_v1"
OUTPUT_MANIFEST_COLUMNS = ("INSTANCE", "vh_seq", "vl_seq", "ag_seq")
SOURCE_MANIFEST_COLUMNS = (
    "INSTANCE",
    "PDB_ID",
    "SABDAB_ID",
    "type",
    "split",
    "ab_ag_cluster",
    "Hchain",
    "Lchain",
    "agchains",
    "vh_seq",
    "vl_seq",
    "ag_seq",
    "vh_len",
    "vl_len",
    "ag_len",
    "vh_seq_hash",
    "vl_seq_hash",
    "ag_seq_hash",
    "vh_source_start",
    "vh_source_end",
    "vl_source_start",
    "vl_source_end",
    "vh_interval_raw",
    "vl_interval_raw",
    "vh_interval_convention",
    "vl_interval_convention",
    "status",
    "flags",
)
VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")
EXPECTED_COUNTS = {
    "source_rows": 3229,
    "exact_pairs": 1830,
    "antibody_entities": 1383,
    "antigen_entities": 1172,
}
EXPECTED_AUTHORITY_MEMBERSHIP_SHA256 = (
    "442095581ae9bf3c7440c1a818612298b5f81e5a62ef46f431e0867b9c8b466b"
)
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class InputDerivationError(ValueError):
    """Raised when historical evidence or a projected input fails closed."""


def _canonical_json_bytes(payload: object) -> bytes:
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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InputDerivationError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or any(character in value for character in (",", '"'))
    ):
        raise InputDerivationError(f"{label} must be exact delimiter-free ASCII")
    return value


def _membership_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        digest.update(_identifier(value, "membership identifier").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_regular_file_nofollow(path: Path, expected_sha256: str, label: str) -> bytes:
    expected = _digest(expected_sha256, f"expected {label}")
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise InputDerivationError(f"{label} must be a single-link regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        fingerprint = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if fingerprint(before) != fingerprint(after):
            raise InputDerivationError(f"{label} changed during authenticated read")
        payload = b"".join(chunks)
        if len(payload) != before.st_size or _sha256_bytes(payload) != expected:
            raise InputDerivationError(f"{label} differs from its external anchor")
        return payload
    finally:
        os.close(descriptor)


def _strict_object(payload_bytes: bytes, label: str) -> dict[str, Any]:
    if not payload_bytes or payload_bytes.startswith(b"\xef\xbb\xbf"):
        raise InputDerivationError(f"{label} is empty or contains a BOM")
    try:
        text = payload_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise InputDerivationError(f"{label} must be ASCII JSON") from error

    def reject_duplicates(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise InputDerivationError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                InputDerivationError(f"{label} contains non-finite {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise InputDerivationError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise InputDerivationError(f"{label} must be a JSON object")
    return value


def _collect_source_instance_ids(payload: object) -> tuple[str, ...]:
    values = []
    pending = [payload]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key == "source_instance_ids":
                    if not isinstance(value, list):
                        raise InputDerivationError("source_instance_ids is malformed")
                    values.extend(
                        _identifier(item, "source instance identifier")
                        for item in value
                    )
                else:
                    pending.append(value)
        elif isinstance(current, list):
            pending.extend(current)
    unique = tuple(sorted(set(values)))
    if len(unique) != EXPECTED_COUNTS["source_rows"]:
        raise InputDerivationError(
            "historical receipt does not cover the locked train membership"
        )
    if _membership_sha256(unique) != EXPECTED_AUTHORITY_MEMBERSHIP_SHA256:
        raise InputDerivationError("historical train membership differs from policy")
    return unique


def _validate_historical_evidence(
    assignment: Mapping[str, Any],
    audit: Mapping[str, Any],
    assignment_sha256: str,
) -> tuple[tuple[str, ...], dict[str, tuple[tuple[str, str], ...]]]:
    if assignment.get("format_version") != ASSIGNMENT_FORMAT:
        raise InputDerivationError("historical assignment format differs")
    if audit.get("format_version") != AUDIT_FORMAT:
        raise InputDerivationError("historical audit format differs")
    if audit.get("valid_audit_execution") is not True:
        raise InputDerivationError("historical independent audit was not valid")
    audit_inputs = audit.get("inputs")
    if not isinstance(audit_inputs, dict) or not isinstance(audit_inputs.get("assignment"), dict):
        raise InputDerivationError("historical audit assignment binding is malformed")
    if audit_inputs["assignment"].get("sha256") != assignment_sha256:
        raise InputDerivationError("historical audit binds a different assignment")
    if audit.get("assignment_scientific_payload_sha256") != assignment.get(
        "scientific_payload_sha256"
    ):
        raise InputDerivationError("historical audit scientific binding differs")
    gates = audit.get("independent_gates")
    required_true_gates = (
        "authority_train_only_firewall_passed",
        "exhaustive_alignment_accounting_passed",
        "independent_inventory_exactly_matches_assignment",
    )
    if not isinstance(gates, dict) or any(gates.get(key) is not True for key in required_true_gates):
        raise InputDerivationError("historical raw-authority gates did not pass")
    firewall = assignment.get("data_firewall")
    if not isinstance(firewall, dict) or any(
        firewall.get(key) != expected
        for key, expected in {
            "authorized_source_split": "train",
            "authorized_rows_materialized": EXPECTED_COUNTS["source_rows"],
            "sealed_rows_materialized": 0,
            "validation_sequence_fields_parsed": 0,
            "test_sequence_fields_parsed": 0,
        }.items()
    ):
        raise InputDerivationError("historical assignment firewall differs")
    authority = assignment.get("authority")
    if not isinstance(authority, dict) or authority.get(
        "authority_train_membership_sha256"
    ) != EXPECTED_AUTHORITY_MEMBERSHIP_SHA256:
        raise InputDerivationError("historical assignment authority differs")
    inventory = assignment.get("canonical_inventory")
    if not isinstance(inventory, dict) or {
        "source_rows": inventory.get("authority_train_source_row_count"),
        "exact_pairs": inventory.get("unique_exact_pair_count"),
        "antibody_entities": inventory.get("unique_antibody_entity_count"),
        "antigen_entities": inventory.get("unique_antigen_entity_count"),
    } != EXPECTED_COUNTS:
        raise InputDerivationError("historical canonical inventory differs")
    source_ids = _collect_source_instance_ids(assignment.get("construction"))
    graph_payload = assignment.get("exact_direct_graphs")
    if not isinstance(graph_payload, dict) or set(graph_payload) != set(VIEWS):
        raise InputDerivationError("historical direct graphs are incomplete")
    graphs = {}
    for view in VIEWS:
        row = graph_payload[view]
        raw_edges = row.get("direct_entity_edges") if isinstance(row, dict) else None
        if not isinstance(raw_edges, list):
            raise InputDerivationError(f"historical {view} edges are malformed")
        edges = []
        for edge in raw_edges:
            if not isinstance(edge, dict) or set(edge) != {
                "left_entity_id",
                "right_entity_id",
            }:
                raise InputDerivationError(f"historical {view} edge schema differs")
            left = _identifier(edge["left_entity_id"], f"{view} edge left")
            right = _identifier(edge["right_entity_id"], f"{view} edge right")
            if left >= right:
                raise InputDerivationError(f"historical {view} edge is noncanonical")
            edges.append((left, right))
        canonical = tuple(edges)
        if canonical != tuple(sorted(set(canonical))):
            raise InputDerivationError(f"historical {view} edges are unsorted or repeated")
        if row.get("lifted_direct_entity_edge_count") != len(canonical):
            raise InputDerivationError(f"historical {view} edge count differs")
        if row.get("direct_entity_edge_membership_sha256") != _membership_sha256(
            f"{left}\0{right}" for left, right in canonical
        ):
            raise InputDerivationError(f"historical {view} edge digest differs")
        graphs[view] = canonical
    return source_ids, graphs


def _project_authorized_manifest(
    payload_bytes: bytes,
    authorized_ids: Sequence[str],
) -> tuple[bytes, dict[str, int]]:
    if b"\0" in payload_bytes or b"\r" in payload_bytes or not payload_bytes.endswith(b"\n"):
        raise InputDerivationError("canonical manifest byte framing differs")
    lines = payload_bytes.split(b"\n")[:-1]
    if not lines or any(not line for line in lines):
        raise InputDerivationError("canonical manifest contains an empty physical line")
    try:
        header = next(csv.reader([lines[0].decode("ascii")], strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise InputDerivationError("canonical manifest header is malformed") from error
    if tuple(header) != SOURCE_MANIFEST_COLUMNS:
        raise InputDerivationError("canonical manifest header differs")
    column_index = {name: header.index(name) for name in OUTPUT_MANIFEST_COLUMNS}
    authorized = set(authorized_ids)
    observed_ids = set()
    projected_rows = []
    total_rows = 0
    for line_number, raw_line in enumerate(lines[1:], 2):
        total_rows += 1
        raw_identifier = raw_line.split(b",", 1)[0]
        try:
            instance_id = _identifier(raw_identifier.decode("ascii"), "manifest identifier")
        except UnicodeDecodeError as error:
            raise InputDerivationError(
                f"manifest identifier on line {line_number} is not ASCII"
            ) from error
        if instance_id in observed_ids:
            raise InputDerivationError("canonical manifest contains a duplicate identifier")
        observed_ids.add(instance_id)
        if instance_id not in authorized:
            continue
        try:
            row = next(csv.reader([raw_line.decode("ascii")], strict=True))
        except (UnicodeDecodeError, csv.Error) as error:
            raise InputDerivationError(
                f"authorized manifest line {line_number} is malformed"
            ) from error
        if len(row) != len(header) or row[0] != instance_id:
            raise InputDerivationError("authorized manifest row shape differs")
        projected = tuple(row[column_index[name]] for name in OUTPUT_MANIFEST_COLUMNS)
        for sequence_index, sequence in enumerate(projected[1:], 1):
            if not sequence and sequence_index == 2:
                continue
            if not sequence or set(sequence) - VALID_AMINO_ACIDS:
                raise InputDerivationError("authorized sequence is noncanonical")
        projected_rows.append(projected)
    if set(authorized_ids) != {row[0] for row in projected_rows}:
        raise InputDerivationError("not every authorized ID has one manifest row")
    projected_rows.sort(key=lambda row: row[0])
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(OUTPUT_MANIFEST_COLUMNS)
    writer.writerows(projected_rows)
    return buffer.getvalue().encode("ascii"), {
        "physical_source_rows_scanned": total_rows,
        "authorized_sequence_rows_parsed": len(projected_rows),
        "non_authorized_sequence_rows_parsed": 0,
    }


def _create_file(path: Path, payload: bytes) -> None:
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


def derive_inputs(
    *,
    assignment_bytes: bytes,
    audit_bytes: bytes,
    manifest_bytes: bytes,
    implementation_source_sha256: str,
) -> dict[str, bytes | object]:
    """Derive all physical output bytes from already authenticated inputs."""
    implementation_digest = _digest(
        implementation_source_sha256,
        "implementation source",
    )
    assignment = _strict_object(assignment_bytes, "historical assignment")
    audit = _strict_object(audit_bytes, "historical independent audit")
    source_ids, graphs = _validate_historical_evidence(
        assignment,
        audit,
        _sha256_bytes(assignment_bytes),
    )
    projected_manifest, firewall = _project_authorized_manifest(
        manifest_bytes,
        source_ids,
    )
    authority_bytes = _canonical_json_bytes(
        {
            "format_version": AUTHORITY_FORMAT,
            "authorized_split": "train",
            "instance_ids": list(source_ids),
        }
    )
    edge_ledgers = {
        view: _canonical_json_bytes(
            {
                "format_version": EDGE_FORMAT,
                "view": view,
                "edges": [list(edge) for edge in graphs[view]],
            }
        )
        for view in VIEWS
    }
    receipt: dict[str, object] = {
        "format_version": RECEIPT_FORMAT,
        "inputs": {
            "historical_assignment_sha256": _sha256_bytes(assignment_bytes),
            "historical_independent_audit_sha256": _sha256_bytes(audit_bytes),
            "full_canonical_manifest_sha256": _sha256_bytes(manifest_bytes),
        },
        "implementation_source_sha256": implementation_digest,
        "historical_fold_assignment_reused": False,
        "historical_fold_feasibility_claim_reused": False,
        "historical_source_scope": "train_authority_membership_and_exact_direct_graphs_only",
        "authority_membership_sha256": _membership_sha256(source_ids),
        "locked_counts": EXPECTED_COUNTS,
        "manifest_firewall": firewall,
        "graph_edge_counts": {view: len(graphs[view]) for view in VIEWS},
        "outputs": {
            "train_authority_allowlist.json": {
                "sha256": _sha256_bytes(authority_bytes),
                "byte_size": len(authority_bytes),
            },
            "train_canonical_manifest.csv": {
                "sha256": _sha256_bytes(projected_manifest),
                "byte_size": len(projected_manifest),
            },
            **{
                f"edge_ledger_{view}.json": {
                    "sha256": _sha256_bytes(edge_ledgers[view]),
                    "byte_size": len(edge_ledgers[view]),
                }
                for view in VIEWS
            },
        },
        "contact_or_interface_labels_read": False,
        "predictions_losses_embeddings_ranks_or_model_scores_read": False,
        "training_authorized": False,
        "model_selection_authorized": False,
        "outer_evaluation_authorized": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    return {
        "train_authority_allowlist.json": authority_bytes,
        "train_canonical_manifest.csv": projected_manifest,
        **{f"edge_ledger_{view}.json": edge_ledgers[view] for view in VIEWS},
        "derivation_receipt.json": _canonical_json_bytes(receipt),
        "receipt": receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-assignment", type=Path, required=True)
    parser.add_argument("--historical-assignment-sha256", required=True)
    parser.add_argument("--historical-independent-audit", type=Path, required=True)
    parser.add_argument("--historical-independent-audit-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--implementation-source-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    output = Path(os.path.abspath(os.path.expanduser(str(args.output_directory))))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"create-only output already exists: {output}")
    output.parent.mkdir(parents=False, exist_ok=True)
    assignment_bytes = _read_regular_file_nofollow(
        args.historical_assignment,
        args.historical_assignment_sha256,
        "historical assignment",
    )
    audit_bytes = _read_regular_file_nofollow(
        args.historical_independent_audit,
        args.historical_independent_audit_sha256,
        "historical independent audit",
    )
    manifest_bytes = _read_regular_file_nofollow(
        args.canonical_manifest,
        args.canonical_manifest_sha256,
        "canonical manifest",
    )
    _read_regular_file_nofollow(
        Path(__file__),
        args.implementation_source_sha256,
        "implementation source",
    )
    artifacts = derive_inputs(
        assignment_bytes=assignment_bytes,
        audit_bytes=audit_bytes,
        manifest_bytes=manifest_bytes,
        implementation_source_sha256=args.implementation_source_sha256,
    )
    os.mkdir(output, 0o500)
    os.chmod(output, 0o700)
    try:
        for name in sorted(key for key in artifacts if key != "receipt"):
            payload = artifacts[name]
            assert isinstance(payload, bytes)
            _create_file(output / name, payload)
    finally:
        os.chmod(output, 0o500)
    print(json.dumps(artifacts["receipt"], sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
