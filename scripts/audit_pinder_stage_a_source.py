#!/usr/bin/env python3
"""Audit PINDER train/validation metadata without projecting test records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument("--duckdb-sha256", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _authenticate(path: Path, expected_sha256: str) -> Path:
    selected = path.resolve()
    if _sha256(selected) != expected_sha256:
        raise RuntimeError(f"SHA-256 differs from frozen anchor: {selected}")
    return selected


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _run_json(duckdb: Path, sql: str) -> list[dict[str, object]]:
    result = subprocess.run(
        [str(duckdb), "-json", "-c", sql],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RuntimeError("DuckDB audit query did not return a JSON row array")
    for row in value:
        for key, field in tuple(row.items()):
            if isinstance(field, str) and field.isdigit():
                row[key] = int(field)
    return value


def _publish_create_only(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite PINDER audit: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def audit(arguments: argparse.Namespace) -> dict[str, object]:
    duckdb = _authenticate(arguments.duckdb, arguments.duckdb_sha256)
    index = _authenticate(arguments.index, arguments.index_sha256)
    metadata = _authenticate(arguments.metadata, arguments.metadata_sha256)
    index_sql = _sql_path(index)
    metadata_sql = _sql_path(metadata)
    version_rows = _run_json(duckdb, "SELECT version() AS version")
    if len(version_rows) != 1 or version_rows[0].get("version") != "v1.5.4":
        raise RuntimeError("DuckDB version differs from the audited tool anchor")

    inventory_sql = f"""
    WITH eligible AS (
      SELECT * FROM read_parquet('{index_sql}') WHERE split IN ('train','val')
    ), metadata AS (
      SELECT * FROM read_parquet('{metadata_sql}')
    ), joined AS (
      SELECT e.*, m.resolution, m.length1, m.length2,
             m.length_resolved_1, m.length_resolved_2, m.num_atom_types,
             m.intermolecular_contacts, m.interface_atom_gaps_4A,
             m.missing_interface_residues_4A, m.id IS NOT NULL AS metadata_present
      FROM eligible e LEFT JOIN metadata m USING (id)
    )
    SELECT split, count(*) AS rows, count(DISTINCT id) AS unique_ids,
      count(DISTINCT cluster_id) AS unique_clusters,
      count(DISTINCT CASE WHEN uniprot_R IS NOT NULL AND uniprot_L IS NOT NULL
        THEN least(uniprot_R,uniprot_L)||'|'||greatest(uniprot_R,uniprot_L) END)
        AS unique_uniprot_pairs,
      count_if(NOT metadata_present) AS missing_metadata_rows,
      count_if(holo_R AND holo_L) AS both_holo_rows,
      count_if(contains_antibody) AS antibody_rows,
      count_if(contains_antigen) AS antigen_rows,
      count_if(resolution IS NULL) AS missing_resolution_rows,
      median(resolution) AS resolution_p50,
      quantile_cont(resolution,0.95) AS resolution_p95,
      median(length1+length2) AS combined_length_p50,
      quantile_cont(length1+length2,0.95) AS combined_length_p95,
      quantile_cont(length1+length2,0.99) AS combined_length_p99,
      median(intermolecular_contacts) AS contacts_p50,
      quantile_cont(intermolecular_contacts,0.05) AS contacts_p05,
      count_if(coalesce(intermolecular_contacts,0)<5) AS contacts_below_5_or_missing,
      count_if(coalesce(num_atom_types,0)<4) AS atom_types_below_4_or_missing,
      count_if(coalesce(length_resolved_1,0)<20 OR coalesce(length_resolved_2,0)<20)
        AS short_resolved_chain_rows,
      count_if(coalesce(missing_interface_residues_4A,0)>0)
        AS missing_interface_residue_rows
    FROM joined GROUP BY split ORDER BY split
    """
    inventory_rows = _run_json(duckdb, inventory_sql)
    inventory = {str(row["split"]): row for row in inventory_rows}
    if set(inventory) != {"train", "val"}:
        raise RuntimeError("PINDER eligible split inventory differs from train/val")

    overlap_sql = f"""
    WITH eligible AS (
      SELECT *, CASE WHEN uniprot_R IS NOT NULL AND uniprot_L IS NOT NULL
        THEN least(uniprot_R,uniprot_L)||'|'||greatest(uniprot_R,uniprot_L) END AS pair_key
      FROM read_parquet('{index_sql}') WHERE split IN ('train','val')
    ), train AS (SELECT * FROM eligible WHERE split='train'),
       val AS (SELECT * FROM eligible WHERE split='val')
    SELECT
      (SELECT count(*) FROM (SELECT id FROM train INTERSECT SELECT id FROM val)) AS id_overlap,
      (SELECT count(*) FROM (SELECT cluster_id FROM train INTERSECT SELECT cluster_id FROM val)) AS cluster_overlap,
      (SELECT count(*) FROM (SELECT pair_key FROM train WHERE pair_key IS NOT NULL
        INTERSECT SELECT pair_key FROM val WHERE pair_key IS NOT NULL))
        AS exact_uniprot_pair_overlap,
      (SELECT count(*)-count(DISTINCT id) FROM train) AS train_duplicate_ids,
      (SELECT count(*)-count(DISTINCT id) FROM val) AS val_duplicate_ids
    """
    overlap_rows = _run_json(duckdb, overlap_sql)
    if len(overlap_rows) != 1:
        raise RuntimeError("PINDER overlap query returned an invalid row count")

    quality_sql = f"""
    WITH train AS (
      SELECT i.*, m.resolution, m.length1, m.length2,
        m.length_resolved_1, m.length_resolved_2, m.num_atom_types,
        m.intermolecular_contacts, m.interface_atom_gaps_4A,
        m.missing_interface_residues_4A
      FROM read_parquet('{index_sql}') i
      JOIN read_parquet('{metadata_sql}') m USING(id)
      WHERE i.split='train'
    )
    SELECT count(*) AS all_train,
      count_if(resolution<=4.0) AS resolution_le_4,
      count_if(resolution<=4.0 AND intermolecular_contacts>=5) AS plus_contacts,
      count_if(resolution<=4.0 AND intermolecular_contacts>=5 AND num_atom_types>=4)
        AS plus_atom_types,
      count_if(resolution<=4.0 AND intermolecular_contacts>=5 AND num_atom_types>=4
        AND length_resolved_1>=20 AND length_resolved_2>=20) AS plus_resolved_length,
      count_if(resolution<=4.0 AND intermolecular_contacts>=5 AND num_atom_types>=4
        AND length_resolved_1>=20 AND length_resolved_2>=20
        AND missing_interface_residues_4A=0 AND interface_atom_gaps_4A=0)
        AS strict_interface_complete,
      count_if(resolution<=4.0 AND intermolecular_contacts>=5 AND num_atom_types>=4
        AND length_resolved_1>=20 AND length_resolved_2>=20
        AND missing_interface_residues_4A=0 AND interface_atom_gaps_4A=0
        AND length1+length2<=1020) AS strict_model_fit,
      count(DISTINCT CASE WHEN resolution<=4.0 AND intermolecular_contacts>=5
        AND num_atom_types>=4 AND length_resolved_1>=20 AND length_resolved_2>=20
        AND missing_interface_residues_4A=0 AND interface_atom_gaps_4A=0
        AND length1+length2<=1020 THEN cluster_id END) AS strict_model_fit_clusters
    FROM train
    """
    quality_rows = _run_json(duckdb, quality_sql)
    if len(quality_rows) != 1:
        raise RuntimeError("PINDER quality query returned an invalid row count")

    train = inventory["train"]
    val = inventory["val"]
    overlaps = overlap_rows[0]
    counts_match = train["rows"] == 1560682 and val["rows"] == 1958
    index_integrity = (
        counts_match
        and train["rows"] == train["unique_ids"]
        and val["rows"] == val["unique_ids"]
        and train["missing_metadata_rows"] == 0
        and val["missing_metadata_rows"] == 0
        and overlaps["id_overlap"] == 0
        and overlaps["cluster_overlap"] == 0
    )
    return {
        "authorization": {
            "normalization": False,
            "stage_a_training": False,
            "structure_download": index_integrity,
        },
        "claim_boundary": (
            "Authenticated PINDER 2024-02 train/validation index and metadata evidence. "
            "Only split labels needed to select train/val were scanned; test records were "
            "not projected or analyzed and test structures were not downloaded."
        ),
        "files": {
            "index": {
                "sha256": arguments.index_sha256,
                "size_bytes": index.stat().st_size,
            },
            "metadata": {
                "sha256": arguments.metadata_sha256,
                "size_bytes": metadata.stat().st_size,
            },
        },
        "format_version": "pinder_2024_02_stage_a_source_audit_v1",
        "index_integrity_passed": index_integrity,
        "inventory": inventory,
        "license_and_training_authority": {
            "pinder_challenge_explicitly_allows_train_and_validation_rows": True,
            "pinder_code_license": "Apache-2.0",
            "underlying_wwpdb_structure_data_license": "CC0-1.0",
        },
        "overlap": overlaps,
        "quality_filter": {
            "criteria": {
                "combined_sequence_length_maximum": 1020,
                "interface_atom_gaps_4A": 0,
                "intermolecular_contacts_minimum": 5,
                "missing_interface_residues_4A": 0,
                "num_atom_types_minimum": 4,
                "per_chain_resolved_length_minimum": 20,
                "resolution_maximum_angstrom": 4.0,
            },
            "counts": quality_rows[0],
            "development_decontamination_pending": True,
        },
        "status": (
            "MATERIALIZED_TRAINVAL_INDEX_AUDITED_STRUCTURES_PENDING"
            if index_integrity
            else "FAILED_CLOSED"
        ),
        "structure_archive": {
            "expected_md5": "28e6e2724c1597514c7f6b6774ff32b4",
            "expected_size_bytes": 168884420877,
            "gcs_generation": "1718880407172355",
            "test_set_archive_requested": False,
            "url": "https://storage.googleapis.com/pinder/2024-02/pdbs.zip",
        },
        "test_access": {
            "asep_test_accessed": False,
            "pinder_test_rows_projected_or_used": False,
            "pinder_test_structure_bytes_accessed": False,
            "sabdab2_heldout_accessed": False,
        },
        "tool": {
            "duckdb_sha256": arguments.duckdb_sha256,
            "duckdb_version": "v1.5.4",
        },
    }


def main() -> None:
    arguments = _arguments()
    receipt = audit(arguments)
    _publish_create_only(arguments.output, receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False))
    if receipt["status"] == "FAILED_CLOSED" and not arguments.allow_blocked:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
