"""Tests for label-free three-way SAbDab2 split construction."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from interaction_first.sabdab2_split_matrix import (
    SPLIT_MATRIX_FORMAT,
    SplitMatrixError,
    _antigen_fold_roles,
    _base_rows,
    audit_split_manifest_shortcuts,
    build_split_matrix,
    symmetric_foldseek_score,
    structure_ids_sha256,
)
from interaction_first.structural_novelty import FOLDSEEK_RECEIPT_FORMAT


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_base_rows_rejects_prepared_instance_missing_from_metadata() -> None:
    prepared = {
        "missing": {
            "instance_id": "missing",
            "ab_sequence_sha256": "a" * 64,
            "ag_sequence_sha256": "b" * 64,
            "paratope_split": "train",
            "epitope_split": "train",
        }
    }
    with pytest.raises(SplitMatrixError, match="Every prepared instance"):
        _base_rows({}, prepared)


def test_reciprocal_min_requires_global_similarity_in_both_directions() -> None:
    assert symmetric_foldseek_score(0.9, 0.3, "reciprocal_min") == 0.3
    assert symmetric_foldseek_score(0.9, 0.3, "reciprocal_max") == 0.9
    assert symmetric_foldseek_score(0.9, 0.3, "geometric_mean") == pytest.approx(
        (0.27) ** 0.5
    )


def test_double_cold_policy_keeps_both_metadata_clusters_in_one_role() -> None:
    rows = []
    # Enough disconnected components are required for a meaningful 70/15/15
    # three-role assignment; the production splitter intentionally rejects an
    # empty role instead of silently relaxing the policy.
    for group in range(10):
        for offset in range(4):
            index = group * 4 + offset
            rows.append(
                {
                    "instance_id": f"case_{index}",
                    "pdb_assembly_id": f"pdb_{index}",
                    "antigen_sequence_sha256": hashlib.sha256(
                        f"ag-seq-{index}".encode()
                    ).hexdigest(),
                    "antibody_sequence_sha256": hashlib.sha256(
                        f"ab-seq-{index}".encode()
                    ).hexdigest(),
                    "antigen_cluster_id": f"ag-cluster-{group}",
                    "antibody_cluster_id": f"ab-cluster-{group}",
                }
            )
    roles, audit = _antigen_fold_roles(
        rows,
        edges=[],
        reported_alignment_rows=len(rows) ** 2,
        tm_threshold=0.5,
        seed=20260730,
        fractions=(0.7, 0.15, 0.15),
        metadata_cluster_policy="double_cold",
    )
    assert audit["valid"] is True
    assert audit["metadata_cluster_isolation"]["valid"] is True
    for column in ("antigen_cluster_id", "antibody_cluster_id"):
        by_cluster = {}
        for row in rows:
            by_cluster.setdefault(row[column], set()).add(roles[row["instance_id"]])
        assert all(len(values) == 1 for values in by_cluster.values())


def test_double_cold_unions_any_shared_multichain_antigen_cluster() -> None:
    rows = []
    for group in range(10):
        for offset, antigen_clusters in enumerate(
            (f"ag-{group}-a/shared-{group}", f"shared-{group}/ag-{group}-b")
        ):
            index = group * 2 + offset
            rows.append(
                {
                    "instance_id": f"case_{index}",
                    "pdb_assembly_id": f"pdb_{index}",
                    "antigen_sequence_sha256": f"{index + 1:064x}",
                    "antibody_sequence_sha256": f"{index + 101:064x}",
                    "antigen_cluster_id": antigen_clusters,
                    "antibody_cluster_id": f"ab-{index}",
                }
            )
    roles, audit = _antigen_fold_roles(
        rows,
        edges=[],
        reported_alignment_rows=len(rows) ** 2,
        tm_threshold=0.5,
        seed=17,
        fractions=(0.7, 0.15, 0.15),
        metadata_cluster_policy="double_cold",
    )
    for group in range(10):
        assert roles[f"case_{group * 2}"] == roles[f"case_{group * 2 + 1}"]
    antigen_audit = audit["metadata_cluster_isolation"]["fields"][
        "antigen_cluster_id"
    ]
    assert antigen_audit["value_semantics"] == "slash_separated_chain_cluster_tokens"
    assert antigen_audit["cross_role_value_count"] == 0


def test_split_shortcut_audit_reports_cluster_leakage_without_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "instance_id",
                "role",
                "pdb_assembly_id",
                "antigen_sequence_sha256",
                "antibody_sequence_sha256",
                "antigen_cluster_id",
                "antibody_cluster_id",
            ],
        )
        writer.writeheader()
        for index, role in enumerate(("train", "validation", "test")):
            writer.writerow(
                {
                    "instance_id": f"case_{index}",
                    "role": role,
                    "pdb_assembly_id": f"pdb_{index}",
                    "antigen_sequence_sha256": hashlib.sha256(
                        f"ag-{index}".encode()
                    ).hexdigest(),
                    "antibody_sequence_sha256": hashlib.sha256(
                        f"ab-{index}".encode()
                    ).hexdigest(),
                    "antigen_cluster_id": "shared-antigen" if index < 2 else "other",
                    "antibody_cluster_id": f"ab-cluster-{index}",
                }
            )
    audit = audit_split_manifest_shortcuts(path, _sha(path))
    assert audit["exact_isolation"]["valid"] is True
    assert audit["metadata_cluster_isolation"]["valid"] is False
    assert audit["metadata_cluster_isolation"]["fields"][
        "antigen_cluster_id"
    ]["cross_role_value_count"] == 1
    assert audit["double_cold_valid"] is False
    assert audit["task_labels_read"] is False


def test_build_split_matrix_preserves_hard_components(tmp_path: Path) -> None:
    ids = [f"case_{index}" for index in range(9)]
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["INSTANCE", "PDB_ID", "PDBdepo", "agclusters", "ab_cluster"],
        )
        writer.writeheader()
        dates = [
            "2020-01-01",
            "2021-01-01",
            "2022-01-01",
            "2023-02-01",
            "2023-03-01",
            "2024-02-01",
            "2024-07-01",
            "2025-01-01",
            "2025-02-01",
        ]
        for index, identifier in enumerate(ids):
            writer.writerow(
                {
                    "INSTANCE": identifier,
                    "PDB_ID": f"pdb_{index}",
                    "PDBdepo": dates[index],
                    "agclusters": f"ag_{index}",
                    "ab_cluster": f"ab_{index}",
                }
            )
        writer.writerow(
            {
                "INSTANCE": "outside_prepared_cohort",
                "PDB_ID": "outside_pdb",
                "PDBdepo": "2020-01-01",
                "agclusters": "outside_ag",
                "ab_cluster": "outside_ab",
            }
        )
    prepared = tmp_path / "prepared.parquet"
    frame = pd.DataFrame(
        {
            "instance_id": ids,
            "ab_sequence_sha256": [hashlib.sha256(f"ab{i}".encode()).hexdigest() for i in range(9)],
            "ag_sequence_sha256": [hashlib.sha256(f"ag{i}".encode()).hexdigest() for i in range(9)],
            "paratope_split": ["train"] * 3 + ["validation"] * 3 + ["test"] * 3,
            "epitope_split": ["train"] * 3 + ["validation"] * 3 + ["test"] * 3,
        }
    )
    frame.to_parquet(prepared, index=False)
    alignments = tmp_path / "foldseek.tsv"
    alignment_lines = ["query\ttarget\tqtmscore\tttmscore\n"]
    for query in ids:
        for target in ids:
            score = 1.0 if query == target else 0.1
            if {query, target} == {"case_0", "case_1"}:
                score = 0.6
            alignment_lines.append(f"{query}\t{target}\t{score}\t{score}\n")
    alignments.write_text("".join(alignment_lines), encoding="utf-8")
    receipt_path = tmp_path / "foldseek.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": FOLDSEEK_RECEIPT_FORMAT,
                "query_target_same_database": True,
                "alignment_type": 1,
                "test_labels_read": False,
                "structure_count": 9,
                "complete_ordered_cartesian_grid": True,
                "expected_ordered_pair_count": 81,
                "raw_row_count": 81,
                "structure_ids_sha256": structure_ids_sha256(ids),
                "foldseek_version": "test",
                "format_output": ["query", "target", "qtmscore", "ttmscore"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "splits"
    result = build_split_matrix(
        metadata_path=metadata,
        metadata_sha256=_sha(metadata),
        prepared_path=prepared,
        prepared_sha256=_sha(prepared),
        foldseek_alignments_path=alignments,
        foldseek_alignments_sha256=_sha(alignments),
        foldseek_receipt_path=receipt_path,
        foldseek_receipt_sha256=_sha(receipt_path),
        output_root=output,
    )
    assert result["format_version"] == SPLIT_MATRIX_FORMAT
    antigen_fold = pd.read_csv(output / "antigen_fold.csv").set_index("instance_id")
    assert antigen_fold.at["case_0", "role"] == antigen_fold.at["case_1", "role"]
    temporal = pd.read_csv(output / "temporal.csv")
    assert temporal.role.value_counts().to_dict() == {
        "train": 3,
        "validation": 3,
        "test": 3,
    }
    stored = json.loads((output / "SPLIT_MATRIX_RECEIPT.json").read_text())
    assert stored["authority"]["task_labels_read"] is False
    assert stored["cohort"]["source_metadata_instance_count"] == 10
    assert stored["cohort"]["source_metadata_instances_outside_prepared_cohort"] == 1
    assert stored["cohort"]["prepared_is_complete_metadata_subset"] is True
    assert stored["audits"]["antigen_fold"]["hard_edge_cross_role_count"] == 0
    antigen_fold_audit = stored["audits"]["antigen_fold"]
    assert antigen_fold_audit["component_count"] == antigen_fold_audit[
        "policy_component_count"
    ]
    assert antigen_fold_audit["structural_component_count"] >= antigen_fold_audit[
        "policy_component_count"
    ]
    assert antigen_fold_audit["component_definition"].startswith(
        "foldseek_hard_edges_plus_exact"
    )
