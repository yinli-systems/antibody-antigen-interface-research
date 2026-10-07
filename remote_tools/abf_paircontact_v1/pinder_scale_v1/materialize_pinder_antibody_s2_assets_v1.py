#!/usr/bin/env python3
"""Materialize immutable sequence-only assets for the antibody S2-80 gate."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
FORMAT_VERSION = "pinder_antibody_s2_asset_result_v1"


class S2AssetError(RuntimeError):
    """Raised when a frozen source or canonical sequence invariant fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise S2AssetError(
            f"{label} SHA mismatch: expected {expected}, observed {observed}"
        )
    return observed


def normalize_sequence(value: str, allow_empty: bool = False) -> str:
    sequence = "".join(value.split()).upper()
    if not sequence and allow_empty:
        return ""
    if not sequence:
        raise S2AssetError("required sequence is empty")
    return sequence


def sequence_views(row: Mapping[str, str]) -> tuple[str, str, str]:
    topology = row["topology"]
    if topology == "fab":
        vh = normalize_sequence(row["heavy_sequence"])
        vl = normalize_sequence(row["light_sequence"])
    elif topology == "heavy_only_or_vhh":
        vh = normalize_sequence(row["heavy_sequence"])
        vl = ""
    elif topology == "scfv":
        vh = normalize_sequence(row["fused_sequence"])
        vl = ""
    else:
        raise S2AssetError(f"unsupported topology: {topology}")
    antigen = normalize_sequence(row["antigen_sequence"])
    return vh, vl, antigen


def noncanonical_symbols(*sequences: str) -> tuple[str, ...]:
    return tuple(sorted(set().union(*(set(sequence) for sequence in sequences)) - VALID_AMINO_ACIDS))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 4_075 or len({row["system_id"] for row in rows}) != len(rows):
        raise S2AssetError("adaptation manifest count or uniqueness mismatch")
    if any(row["source_split"] != "train" for row in rows):
        raise S2AssetError("adaptation manifest contains non-train row")
    return rows


def fasta_bytes(sequences: Mapping[str, str]) -> bytes:
    lines: list[str] = []
    for digest, sequence in sorted(sequences.items()):
        if sequence_sha256(sequence) != digest:
            raise S2AssetError("FASTA identifier does not authenticate sequence")
        lines.extend((f">{digest}", sequence))
    return ("\n".join(lines) + "\n").encode("ascii")


def csv_bytes(fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def atomic_create_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_create_bytes(
        path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(arguments.protocol_json.read_text(encoding="utf-8"))
    if protocol.get("format_version") != "pinder_antibody_s2_asset_protocol_v1":
        raise S2AssetError("protocol format mismatch")
    observed_sha = {
        "adaptation_manifest": require_sha(
            arguments.adaptation_manifest,
            protocol["source_sha256"]["adaptation_manifest"],
            "adaptation_manifest",
        ),
        "adaptation_result": require_sha(
            arguments.adaptation_result,
            protocol["source_sha256"]["adaptation_result"],
            "adaptation_result",
        ),
        "protocol_json": sha256_file(arguments.protocol_json),
        "asset_source": sha256_file(Path(__file__)),
    }
    upstream = json.loads(arguments.adaptation_result.read_text(encoding="utf-8"))
    if upstream.get("status") != "PASS_PINDER_ANTIBODY_ADAPTATION_QUALITY_AND_MULTIPLICITY_GATE":
        raise S2AssetError("upstream adaptation quality result is not PASS")
    rows = read_manifest(arguments.adaptation_manifest)

    canonical_rows: list[dict[str, str]] = []
    exclusions: list[dict[str, Any]] = []
    vh_sequences: dict[str, str] = {}
    vl_sequences: dict[str, str] = {}
    antibody_sequences: dict[str, str] = {}
    antigen_sequences: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: item["system_id"]):
        vh, vl, antigen = sequence_views(row)
        unexpected = noncanonical_symbols(vh, vl, antigen)
        if unexpected:
            exclusions.append(
                {
                    "system_id": row["system_id"],
                    "reason": "noncanonical_residue_not_supported_by_exact_audit",
                    "symbols": list(unexpected),
                }
            )
            continue
        antibody = vh + vl
        vh_digest = sequence_sha256(vh)
        vl_digest = sequence_sha256(vl) if vl else ""
        antibody_digest = sequence_sha256(antibody)
        antigen_digest = sequence_sha256(antigen)
        expected_primary_digest = (
            row["fused_sequence_sha256"]
            if row["topology"] == "scfv"
            else row["heavy_sequence_sha256"]
        )
        if vh_digest != expected_primary_digest:
            raise S2AssetError("primary antibody-chain digest differs from authority")
        if vl_digest != row["light_sequence_sha256"]:
            raise S2AssetError("light-chain digest differs from authority")
        if antigen_digest != row["antigen_sequence_sha256"]:
            raise S2AssetError("antigen digest differs from authority")
        vh_sequences[vh_digest] = vh
        if vl:
            vl_sequences[vl_digest] = vl
        antibody_sequences[antibody_digest] = antibody
        antigen_sequences[antigen_digest] = antigen
        canonical_rows.append(
            {
                "INSTANCE": row["system_id"],
                "pdb_id": row["pdb_id"],
                "topology": row["topology"],
                "vh_seq": vh,
                "vl_seq": vl,
                "ag_seq": antigen,
                "system_loss_weight": row["system_loss_weight"],
            }
        )

    expected = protocol["expected_counts"]
    observed_counts = {
        "input_systems": len(rows),
        "retained_canonical_systems": len(canonical_rows),
        "excluded_noncanonical_systems": len(exclusions),
    }
    if observed_counts != expected:
        raise S2AssetError(
            f"frozen asset counts differ: expected {expected}, observed {observed_counts}"
        )

    output_dir = arguments.output_dir.expanduser().resolve()
    artifacts: dict[str, dict[str, Any]] = {}
    payloads = {
        "canonical_manifest": (
            output_dir / "PINDER_ANTIBODY_S2_CANONICAL_V1.csv",
            csv_bytes(
                [
                    "INSTANCE",
                    "pdb_id",
                    "topology",
                    "vh_seq",
                    "vl_seq",
                    "ag_seq",
                    "system_loss_weight",
                ],
                canonical_rows,
            ),
            len(canonical_rows),
        ),
        "vh_fasta": (
            output_dir / "PINDER_ANTIBODY_S2_VH_V1.fasta",
            fasta_bytes(vh_sequences),
            len(vh_sequences),
        ),
        "vl_fasta": (
            output_dir / "PINDER_ANTIBODY_S2_VL_V1.fasta",
            fasta_bytes(vl_sequences),
            len(vl_sequences),
        ),
        "concatenated_antibody_fasta": (
            output_dir / "PINDER_ANTIBODY_S2_CONCATENATED_ANTIBODY_V1.fasta",
            fasta_bytes(antibody_sequences),
            len(antibody_sequences),
        ),
        "antigen_fasta": (
            output_dir / "PINDER_ANTIBODY_S2_ANTIGEN_V1.fasta",
            fasta_bytes(antigen_sequences),
            len(antigen_sequences),
        ),
    }
    for label, (path, payload, row_count) in payloads.items():
        atomic_create_bytes(path, payload)
        artifacts[label] = {
            "filename": path.name,
            "sha256": sha256_file(path),
            "row_count": row_count,
        }
    authority_path = output_dir / "PINDER_ANTIBODY_S2_AUTHORITY_V1.json"
    authority = {
        "format_version": "pinder_antibody_s2_train_only_authority_v1",
        "splits": {
            "train": sorted(row["INSTANCE"] for row in canonical_rows),
            "validation": [],
            "test": [],
        },
        "claim_boundary": "All identifiers are PINDER-train systems; folds are not yet assigned.",
    }
    atomic_create_json(authority_path, authority)
    artifacts["authority"] = {
        "filename": authority_path.name,
        "sha256": sha256_file(authority_path),
        "row_count": len(canonical_rows),
    }
    exclusion_path = output_dir / "PINDER_ANTIBODY_S2_EXCLUSIONS_V1.json"
    atomic_create_json(
        exclusion_path,
        {
            "format_version": "pinder_antibody_s2_exclusion_ledger_v1",
            "excluded_system_count": len(exclusions),
            "excluded_systems": exclusions,
        },
    )
    artifacts["exclusion_ledger"] = {
        "filename": exclusion_path.name,
        "sha256": sha256_file(exclusion_path),
        "row_count": len(exclusions),
    }

    result: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "PASS_PINDER_ANTIBODY_S2_SEQUENCE_ASSET_GATE",
        "source_sha256": observed_sha,
        "firewall": protocol["authorization"],
        "similarity_policy": protocol["similarity_policy"],
        "counts": {
            **observed_counts,
            "unique_vh_sequences": len(vh_sequences),
            "unique_nonempty_vl_sequences": len(vl_sequences),
            "unique_concatenated_antibody_sequences": len(antibody_sequences),
            "unique_antigen_sequences": len(antigen_sequences),
            "unique_pdbs": len({row["pdb_id"] for row in canonical_rows}),
        },
        "artifacts": artifacts,
        "checks": {
            "all_retained_views_canonical_20_amino_acids": True,
            "all_rows_pinder_train": True,
            "every_fasta_identifier_authenticates_its_sequence": True,
            "five_noncanonical_systems_explicitly_excluded": True,
            "test_rows_read": False,
            "validation_rows_read": False,
        },
        "training_authorized": False,
        "model_selection_authorized": False,
        "required_next_gate": (
            "Compute frozen all-versus-all similarities, solve a PDB-indivisible "
            "two-axis assignment, then pass the independent exact four-view audit."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    result["scientific_payload_sha256"] = canonical_sha256(result)
    result_path = output_dir / "PINDER_ANTIBODY_S2_ASSET_RESULT_V1.json"
    atomic_create_json(result_path, result)
    receipt = {
        "format_version": "pinder_antibody_s2_asset_receipt_v1",
        "status": result["status"],
        "result": {"filename": result_path.name, "sha256": sha256_file(result_path)},
        "artifacts": artifacts,
        "scientific_payload_sha256": result["scientific_payload_sha256"],
        "test_access": False,
        "training_authorized": False,
        "model_selection_authorized": False,
    }
    atomic_create_json(output_dir / "PINDER_ANTIBODY_S2_ASSET_RECEIPT_V1.json", receipt)
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--adaptation-manifest", type=Path, required=True)
    parser.add_argument("--adaptation-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
