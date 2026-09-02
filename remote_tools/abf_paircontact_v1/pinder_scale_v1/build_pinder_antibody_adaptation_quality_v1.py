#!/usr/bin/env python3
"""Freeze a high-quality, multiplicity-weighted complete-antibody inventory."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


FORMAT_VERSION = "pinder_antibody_adaptation_quality_result_v1"
OUTPUT_NAME = "PINDER_ANTIBODY_ADAPTATION_SYSTEMS_V1.tsv.gz"


class QualityPolicyError(RuntimeError):
    """Raised when an immutable input or policy invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise QualityPolicyError(
            f"{label} SHA mismatch: expected {expected}, observed {observed}"
        )
    return observed


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != "pinder_antibody_adaptation_quality_policy_v1":
        raise QualityPolicyError("quality policy format_version mismatch")
    if payload.get("authorization", {}).get("training_authorized") is not False:
        raise QualityPolicyError("quality policy must not authorize training")
    return payload


def official_like_xray_quality(metadata: pd.DataFrame) -> pd.Series:
    numeric = lambda column: pd.to_numeric(metadata[column], errors="coerce")
    return (
        metadata["method"].eq("X-RAY DIFFRACTION")
        & numeric("resolution").le(3.5)
        & numeric("length1").ge(40)
        & numeric("length2").ge(40)
        & numeric("num_atom_types").ge(3)
        & numeric("interface_atom_gaps_4A").eq(0)
        & numeric("max_var_1").le(0.98)
        & numeric("max_var_2").le(0.98)
        & numeric("number_of_components_1").eq(1)
        & numeric("number_of_components_2").eq(1)
    )


def semicolon_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in value.split(";") if token)


def antibody_antigen_edges(row: Mapping[str, str]) -> tuple[str, ...]:
    edges = (
        semicolon_tokens(row["support_heavy_antigen_edge_ids"])
        + semicolon_tokens(row["support_light_antigen_edge_ids"])
        + semicolon_tokens(row["support_fused_antigen_edge_ids"])
    )
    if not edges:
        raise QualityPolicyError(f"system has no antibody-antigen support: {row['system_id']}")
    return tuple(sorted(set(edges)))


def read_full_systems(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "system_id",
        "pdb_id",
        "topology",
        "antibody_axis_sequence_sha256",
        "antigen_sequence_sha256",
        "support_heavy_antigen_edge_ids",
        "support_light_antigen_edge_ids",
        "support_fused_antigen_edge_ids",
        "support_interface_cluster_ids",
        "source_split",
        "model_input_modality",
    }
    if not rows or not required.issubset(fieldnames):
        raise QualityPolicyError("full-system manifest schema mismatch")
    if len(rows) != 24_240 or len({row["system_id"] for row in rows}) != len(rows):
        raise QualityPolicyError("full-system manifest count or uniqueness mismatch")
    if any(row["source_split"] != "train" for row in rows):
        raise QualityPolicyError("full-system manifest contains non-train rows")
    if any(row["model_input_modality"] != "sequence_only" for row in rows):
        raise QualityPolicyError("unexpected model_input_modality")
    return fieldnames, rows


def weighted_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact_key = lambda row: (
        row["antibody_axis_sequence_sha256"],
        row["antigen_sequence_sha256"],
    )
    exact_counts = Counter(exact_key(row) for row in rows)
    pdb_counts = Counter(row["pdb_id"] for row in rows)
    cluster_counts = Counter(row["support_interface_cluster_ids"] for row in rows)

    raw_weights = [
        (
            exact_counts[exact_key(row)]
            * pdb_counts[row["pdb_id"]]
            * cluster_counts[row["support_interface_cluster_ids"]]
        )
        ** (-1.0 / 3.0)
        for row in rows
    ]
    initial_mean = statistics.fmean(raw_weights)
    initially_normalized = [weight / initial_mean for weight in raw_weights]
    clipped = [min(10.0, max(0.1, weight)) for weight in initially_normalized]
    final_mean = statistics.fmean(clipped)
    final_weights = [weight / final_mean for weight in clipped]

    enriched: list[dict[str, Any]] = []
    for row, raw_weight, final_weight in zip(rows, raw_weights, final_weights):
        output = dict(row)
        output.update(
            {
                "exact_sequence_pair_multiplicity": exact_counts[exact_key(row)],
                "pdb_multiplicity": pdb_counts[row["pdb_id"]],
                "interface_cluster_signature_multiplicity": cluster_counts[
                    row["support_interface_cluster_ids"]
                ],
                "raw_inverse_geometric_multiplicity_weight": f"{raw_weight:.12g}",
                "system_loss_weight": f"{final_weight:.12g}",
            }
        )
        enriched.append(output)

    sum_weights = math.fsum(final_weights)
    effective_sample_size = sum_weights * sum_weights / math.fsum(
        weight * weight for weight in final_weights
    )
    summary = {
        "exact_pair_multiplicity_max": max(exact_counts.values(), default=0),
        "interface_cluster_signature_multiplicity_max": max(
            cluster_counts.values(), default=0
        ),
        "pdb_multiplicity_max": max(pdb_counts.values(), default=0),
        "final_weight_effective_sample_size": effective_sample_size,
        "final_weight_max": max(final_weights, default=0.0),
        "final_weight_mean": statistics.fmean(final_weights) if final_weights else 0.0,
        "final_weight_min": min(final_weights, default=0.0),
    }
    return enriched, summary


def deterministic_gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def tsv_bytes(fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
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
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_create_bytes(path, encoded)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol(arguments.protocol_json)
    observed_sha = {
        "full_system_manifest": require_sha(
            arguments.full_system_manifest,
            protocol["source_sha256"]["full_system_manifest"],
            "full_system_manifest",
        ),
        "full_system_result": require_sha(
            arguments.full_system_result,
            protocol["source_sha256"]["full_system_result"],
            "full_system_result",
        ),
        "metadata_parquet": require_sha(
            arguments.metadata_parquet,
            protocol["source_sha256"]["metadata_parquet"],
            "metadata_parquet",
        ),
        "protocol_json": sha256_file(arguments.protocol_json),
        "quality_source": sha256_file(Path(__file__)),
    }
    full_result = json.loads(arguments.full_system_result.read_text(encoding="utf-8"))
    if full_result.get("status") != "PASS_PINDER_TRAIN_FULL_ANTIBODY_SYSTEM_RECONSTRUCTION":
        raise QualityPolicyError("full-system result is not PASS")
    if full_result.get("training_authorized") is not False:
        raise QualityPolicyError("unexpected upstream training authorization")

    input_fields, rows = read_full_systems(arguments.full_system_manifest)
    metadata = pd.read_parquet(arguments.metadata_parquet)
    if metadata["id"].duplicated().any():
        raise QualityPolicyError("PINDER metadata id is not unique")
    quality_ids = set(metadata.loc[official_like_xray_quality(metadata), "id"])
    all_support_ids = {edge for row in rows for edge in antibody_antigen_edges(row)}
    metadata_ids = set(metadata["id"])
    if not all_support_ids.issubset(metadata_ids):
        raise QualityPolicyError("antibody-antigen support edge missing from metadata")

    retained = [
        row
        for row in rows
        if all(edge in quality_ids for edge in antibody_antigen_edges(row))
    ]
    retained.sort(key=lambda row: row["system_id"])
    expected = protocol["expected_counts"]
    topology_counts = Counter(row["topology"] for row in retained)
    exact_pairs = {
        (row["antibody_axis_sequence_sha256"], row["antigen_sequence_sha256"])
        for row in retained
    }
    observed_counts = {
        "total_systems": len(retained),
        "fab": topology_counts["fab"],
        "heavy_only_or_vhh": topology_counts["heavy_only_or_vhh"],
        "scfv": topology_counts["scfv"],
        "unique_exact_sequence_pairs": len(exact_pairs),
        "unique_interface_cluster_signatures": len(
            {row["support_interface_cluster_ids"] for row in retained}
        ),
        "unique_pdbs": len({row["pdb_id"] for row in retained}),
    }
    if observed_counts != expected:
        raise QualityPolicyError(
            f"frozen quality counts differ: expected {expected}, observed {observed_counts}"
        )

    enriched, weight_summary = weighted_rows(retained)
    output_fields = input_fields + [
        "exact_sequence_pair_multiplicity",
        "pdb_multiplicity",
        "interface_cluster_signature_multiplicity",
        "raw_inverse_geometric_multiplicity_weight",
        "system_loss_weight",
    ]
    output_dir = arguments.output_dir.expanduser().resolve()
    output_path = output_dir / OUTPUT_NAME
    atomic_create_bytes(
        output_path,
        deterministic_gzip(tsv_bytes(output_fields, enriched)),
    )

    result: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "PASS_PINDER_ANTIBODY_ADAPTATION_QUALITY_AND_MULTIPLICITY_GATE",
        "pinder_release": protocol["pinder_release"],
        "source_sha256": observed_sha,
        "firewall": protocol["authorization"],
        "quality_policy": protocol["quality_filter"],
        "multiplicity_policy": protocol["multiplicity_weight"],
        "retained_summary": observed_counts,
        "weight_summary": weight_summary,
        "artifact": {
            "filename": output_path.name,
            "row_count": len(enriched),
            "sha256": sha256_file(output_path),
        },
        "checks": {
            "all_antibody_antigen_support_edges_official_like_xray_quality": True,
            "all_rows_train_only": True,
            "all_rows_sequence_only_at_this_gate": True,
            "final_weight_mean_is_one": abs(weight_summary["final_weight_mean"] - 1.0)
            < 1e-12,
            "system_ids_unique": len({row["system_id"] for row in enriched})
            == len(enriched),
            "test_rows_read": False,
            "validation_rows_read": False,
        },
        "training_authorized": False,
        "model_selection_authorized": False,
        "required_next_gate": (
            "Construct a deterministic antibody-and-antigen two-axis split, keep exact "
            "sequence pairs and PDB groups indivisible, and independently audit VH, VL, "
            "concatenated-antibody, and antigen cross-partition similarity before training."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    result["scientific_payload_sha256"] = canonical_sha256(result)
    result_path = output_dir / "PINDER_ANTIBODY_ADAPTATION_QUALITY_RESULT_V1.json"
    atomic_create_json(result_path, result)
    receipt = {
        "format_version": "pinder_antibody_adaptation_quality_receipt_v1",
        "status": result["status"],
        "result": {"filename": result_path.name, "sha256": sha256_file(result_path)},
        "artifact": result["artifact"],
        "scientific_payload_sha256": result["scientific_payload_sha256"],
        "test_access": False,
        "training_authorized": False,
        "model_selection_authorized": False,
    }
    atomic_create_json(
        output_dir / "PINDER_ANTIBODY_ADAPTATION_QUALITY_RECEIPT_V1.json", receipt
    )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--full-system-manifest", type=Path, required=True)
    parser.add_argument("--full-system-result", type=Path, required=True)
    parser.add_argument("--metadata-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
