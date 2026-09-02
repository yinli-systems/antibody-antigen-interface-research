#!/usr/bin/env python3
"""Freeze a metadata-only PINDER-Train antibody-antigen candidate inventory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


EXPECTED_INDEX_SHA256 = "ced1936c4f36ea79161579df8d4b23bb4862b9c88f30af9defeeb85af99adac4"
EXPECTED_SABDAB_SHA256 = "afcf824aa992ccb1a0a1abfa913b5313929d8860e04e53402c442b6401be7380"
EXPECTED_INDEX_ROWS = 2_319_564
EXPECTED_STRICT_TRAIN_ROWS = 35_366


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapped_monomers(frame: pd.DataFrame, columns: list[str]) -> set[str]:
    monomers: set[str] = set()
    for column in columns:
        for value in frame[column].dropna().astype(str):
            for token in value.split(";"):
                token = token.strip()
                if token.endswith(("-R", "-L")):
                    monomers.add(token[:-2])
    return monomers


def write_deterministic_tsv_gz(frame: pd.DataFrame, path: Path) -> None:
    payload = frame.to_csv(index=False, sep="\t", lineterminator="\n").encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            handle.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sabdab-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=False)

    index_sha256 = sha256_file(arguments.index)
    sabdab_sha256 = sha256_file(arguments.sabdab_metadata)
    assert index_sha256 == EXPECTED_INDEX_SHA256
    assert sabdab_sha256 == EXPECTED_SABDAB_SHA256

    sabdab_columns = [
        "pinder_Hchain_monomers",
        "pinder_Lchain_monomers",
        "pinder_antigen_chain_monomers",
    ]
    sabdab = pq.read_table(arguments.sabdab_metadata, columns=sabdab_columns).to_pandas()
    antibody_monomers = mapped_monomers(
        sabdab, ["pinder_Hchain_monomers", "pinder_Lchain_monomers"]
    )
    antigen_monomers = mapped_monomers(sabdab, ["pinder_antigen_chain_monomers"])

    index_columns = [
        "split",
        "id",
        "pdb_id",
        "cluster_id",
        "cluster_id_R",
        "cluster_id_L",
        "uniprot_R",
        "uniprot_L",
    ]
    index = pq.read_table(arguments.index, columns=index_columns).to_pandas()
    assert len(index) == EXPECTED_INDEX_ROWS
    assert index["id"].is_unique

    endpoints = index["id"].astype(str).str.split("--", n=1, expand=True)
    index["monomer_R"] = endpoints[0]
    index["monomer_L"] = endpoints[1]
    index["R_is_antibody"] = index["monomer_R"].isin(antibody_monomers)
    index["R_is_antigen"] = index["monomer_R"].isin(antigen_monomers)
    index["L_is_antibody"] = index["monomer_L"].isin(antibody_monomers)
    index["L_is_antigen"] = index["monomer_L"].isin(antigen_monomers)

    strict_role = (
        index["R_is_antibody"]
        & ~index["R_is_antigen"]
        & index["L_is_antigen"]
        & ~index["L_is_antibody"]
    ) | (
        index["L_is_antibody"]
        & ~index["L_is_antigen"]
        & index["R_is_antigen"]
        & ~index["R_is_antibody"]
    )
    any_role = (
        index["R_is_antibody"] & index["L_is_antigen"]
    ) | (
        index["L_is_antibody"] & index["R_is_antigen"]
    )
    train = index["split"].astype(str).eq("train")
    selected = index.loc[strict_role & train].copy()
    assert len(selected) == EXPECTED_STRICT_TRAIN_ROWS

    antibody_on_R = selected["R_is_antibody"]
    manifest = pd.DataFrame(
        {
            "id": selected["id"].astype(str),
            "pdb_id": selected["pdb_id"].astype(str),
            "antibody_body": antibody_on_R.map({True: "R", False: "L"}),
            "antigen_body": antibody_on_R.map({True: "L", False: "R"}),
            "antibody_monomer": selected["monomer_R"].where(
                antibody_on_R, selected["monomer_L"]
            ),
            "antigen_monomer": selected["monomer_L"].where(
                antibody_on_R, selected["monomer_R"]
            ),
            "interface_cluster_id": selected["cluster_id"].astype(str),
            "antibody_cluster_id": selected["cluster_id_R"].where(
                antibody_on_R, selected["cluster_id_L"]
            ).astype(str),
            "antigen_cluster_id": selected["cluster_id_L"].where(
                antibody_on_R, selected["cluster_id_R"]
            ).astype(str),
            "antibody_uniprot": selected["uniprot_R"].where(
                antibody_on_R, selected["uniprot_L"]
            ).astype(str),
            "antigen_uniprot": selected["uniprot_L"].where(
                antibody_on_R, selected["uniprot_R"]
            ).astype(str),
            "source_split": "train",
            "label_structure_kind": "experimental_holo_dimer",
            "model_input_modality": "sequence_only",
        }
    ).sort_values("id", kind="stable", ignore_index=True)
    assert manifest["id"].is_unique
    assert manifest["source_split"].eq("train").all()
    assert set(manifest["antibody_body"]) == {"R", "L"}

    manifest_path = arguments.output_dir / "PINDER_ANTIBODY_TRAIN_CANDIDATES_V1.tsv.gz"
    write_deterministic_tsv_gz(manifest, manifest_path)
    manifest_sha256 = sha256_file(manifest_path)

    split_counts = {
        str(key): int(value)
        for key, value in index.groupby("split", observed=True).size().items()
    }
    strict_split_counts = {
        str(key): int(value)
        for key, value in index.loc[strict_role]
        .groupby("split", observed=True)
        .size()
        .items()
    }
    train_any_role_count = int((any_role & train).sum())
    result = {
        "authorizations": {
            "download_structures": False,
            "materialize_contact_labels": False,
            "model_selection": False,
            "test_access": False,
            "train_model": False,
        },
        "candidate_manifest": {
            "filename": manifest_path.name,
            "row_count": int(len(manifest)),
            "sha256": manifest_sha256,
        },
        "checks": {
            "all_candidates_train_split": bool(manifest["source_split"].eq("train").all()),
            "ambiguous_endpoint_roles_excluded": True,
            "candidate_ids_unique": bool(manifest["id"].is_unique),
            "index_sha256_verified": index_sha256 == EXPECTED_INDEX_SHA256,
            "sabdab_sha256_verified": sabdab_sha256 == EXPECTED_SABDAB_SHA256,
            "test_rows_selected": 0,
            "validation_rows_selected": 0,
        },
        "claim_boundary": (
            "Metadata-only PINDER-Train antibody-antigen inventory; no structures, "
            "contact labels, validation/test examples, training, or performance claims."
        ),
        "counts": {
            "ambiguous_train_pairs_excluded": train_any_role_count - int(len(manifest)),
            "antibody_monomers": len(antibody_monomers),
            "antibody_monomers_on_R": int((manifest["antibody_body"] == "R").sum()),
            "antibody_monomers_on_L": int((manifest["antibody_body"] == "L").sum()),
            "antibody_role_antigen_role_overlap": len(antibody_monomers & antigen_monomers),
            "antigen_monomers": len(antigen_monomers),
            "candidate_antibody_clusters": int(manifest["antibody_cluster_id"].nunique()),
            "candidate_antigen_clusters": int(manifest["antigen_cluster_id"].nunique()),
            "candidate_interface_clusters": int(manifest["interface_cluster_id"].nunique()),
            "candidate_pdb_ids": int(manifest["pdb_id"].nunique()),
            "candidate_rows": int(len(manifest)),
            "index_rows": int(len(index)),
            "index_split_counts": split_counts,
            "strict_role_split_counts": strict_split_counts,
        },
        "format_version": "pinder_antibody_inventory_result_v1",
        "source_sha256": {
            "index": index_sha256,
            "sabdab_metadata": sabdab_sha256,
        },
        "status": "PASS_METADATA_ONLY_PINDER_ANTIBODY_TRAIN_INVENTORY",
    }
    result_path = arguments.output_dir / "PINDER_ANTIBODY_INVENTORY_RESULT_V1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    receipt = {
        "candidate_manifest_sha256": manifest_sha256,
        "format_version": "pinder_antibody_inventory_receipt_v1",
        "result_sha256": sha256_file(result_path),
        "status": result["status"],
        "test_access": False,
    }
    (arguments.output_dir / "PINDER_ANTIBODY_INVENTORY_RECEIPT_V1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
