"""Tests for the antigen-fold split receipt gate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(values) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in sorted(values)).encode()
    ).hexdigest()


def _authority(tmp_path: Path, *, cross_role_score: float):
    structures = ("ag_train", "ag_validation", "ag_test")
    manifest = tmp_path / "split.csv"
    rows = []
    for index, (role, structure) in enumerate(
        zip(("train", "validation", "test"), structures)
    ):
        rows.append(
            {
                "instance_id": f"instance_{role}",
                "role": role,
                "antigen_structure_id": structure,
                "antigen_sequence_sha256": hashlib.sha256(
                    f"antigen_{role}".encode()
                ).hexdigest(),
                "antibody_sequence_sha256": hashlib.sha256(
                    f"antibody_{role}".encode()
                ).hexdigest(),
                "pdb_assembly_id": f"pdb_{index}",
            }
        )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    alignments = tmp_path / "alignments.tsv"
    with alignments.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["query", "target", "qtmscore", "ttmscore"],
        )
        writer.writeheader()
        for query in structures:
            for target in structures:
                if query == target:
                    score = 1.0
                elif {query, target} == {structures[0], structures[2]}:
                    score = cross_role_score
                else:
                    score = 0.1
                writer.writerow(
                    {
                        "query": query,
                        "target": target,
                        "qtmscore": score,
                        "ttmscore": score,
                    }
                )
    receipt = tmp_path / "foldseek.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "framecontact_foldseek_all_vs_all_receipt_v1",
                "query_target_same_database": True,
                "alignment_type": 1,
                "test_labels_read": False,
                "structure_count": len(structures),
                "structure_ids_sha256": _digest(structures),
                "raw_row_count": len(structures) ** 2,
                "expected_ordered_pair_count": len(structures) ** 2,
                "complete_ordered_cartesian_grid": True,
                "format_output": ["query", "target", "qtmscore", "ttmscore"],
                "foldseek_version": "synthetic-test",
            }
        ),
        encoding="utf-8",
    )
    return manifest, alignments, receipt


def _audit(paths):
    from interaction_first.structural_novelty import audit_antigen_structure_split

    manifest, alignments, receipt = paths
    return audit_antigen_structure_split(
        manifest_path=manifest,
        manifest_sha256=_sha256(manifest),
        alignments_path=alignments,
        alignments_sha256=_sha256(alignments),
        foldseek_receipt_path=receipt,
        foldseek_receipt_sha256=_sha256(receipt),
    )


def test_structurally_isolated_split_passes_with_audited_authorities(tmp_path: Path):
    result = _audit(_authority(tmp_path, cross_role_score=0.31))
    assert result["valid"] is True
    assert result["violations"]["cross_role_hard_edge_count"] == 0
    assert result["maximum_reported_cross_role_symmetric_tm_score"] == 0.31
    assert result["authority"]["test_labels_read"] is False


def test_cross_role_tm_edge_fails_the_split(tmp_path: Path):
    result = _audit(_authority(tmp_path, cross_role_score=0.61))
    assert result["valid"] is False
    assert result["violations"]["cross_role_hard_edge_count"] == 1
    assert result["violations"]["cross_role_component_count"] == 1
    assert result["violation_examples"]["cross_role_hard_edges"] == [
        ["ag_test", "ag_train"]
    ]


def test_reciprocal_min_does_not_treat_one_sided_containment_as_hard_edge(
    tmp_path: Path,
) -> None:
    manifest, alignments, receipt = _authority(tmp_path, cross_role_score=0.1)
    rows = list(csv.DictReader(alignments.open(newline=""), delimiter="\t"))
    with alignments.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        for row in rows:
            if {row["query"], row["target"]} == {"ag_train", "ag_test"}:
                row["qtmscore"], row["ttmscore"] = "0.9", "0.3"
            writer.writerow(row)
    from interaction_first.structural_novelty import audit_antigen_structure_split

    result = audit_antigen_structure_split(
        manifest_path=manifest,
        manifest_sha256=_sha256(manifest),
        alignments_path=alignments,
        alignments_sha256=_sha256(alignments),
        foldseek_receipt_path=receipt,
        foldseek_receipt_sha256=_sha256(receipt),
        foldseek_score_rule="reciprocal_min",
    )
    assert result["valid"] is True
    assert result["thresholds"]["foldseek_score_rule"] == "reciprocal_min"


def test_audit_detects_hard_edge_present_only_in_reverse_query_direction(
    tmp_path: Path,
) -> None:
    manifest, alignments, receipt = _authority(tmp_path, cross_role_score=0.1)
    rows = list(csv.DictReader(alignments.open(newline=""), delimiter="\t"))
    with alignments.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        for row in rows:
            if row["query"] == "ag_train" and row["target"] == "ag_test":
                row["qtmscore"], row["ttmscore"] = "0.1", "0.1"
            if row["query"] == "ag_test" and row["target"] == "ag_train":
                row["qtmscore"], row["ttmscore"] = "0.8", "0.8"
            writer.writerow(row)
    from interaction_first.structural_novelty import audit_antigen_structure_split

    result = audit_antigen_structure_split(
        manifest_path=manifest,
        manifest_sha256=_sha256(manifest),
        alignments_path=alignments,
        alignments_sha256=_sha256(alignments),
        foldseek_receipt_path=receipt,
        foldseek_receipt_sha256=_sha256(receipt),
        foldseek_score_rule="reciprocal_min",
    )
    assert result["valid"] is False
    assert result["violations"]["cross_role_hard_edge_count"] == 1
