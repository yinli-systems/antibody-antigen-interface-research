"""Tests for authenticated US-align Foldseek-boundary spot checks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parse_usalign_tm_scores_requires_two_normalizations() -> None:
    from interaction_first.usalign_boundary import parse_usalign_tm_scores

    stdout = """
TM-score= 0.48100 (normalized by length of Structure_1: L=100, d0=3.0)
TM-score= 0.51200 (normalized by length of Structure_2: L=90, d0=2.0)
"""
    assert parse_usalign_tm_scores(stdout) == (0.481, 0.512)
    with pytest.raises(ValueError, match="both"):
        parse_usalign_tm_scores(stdout.splitlines()[1])


def test_boundary_spot_check_binds_tool_and_detects_no_cross_role_edge(
    tmp_path: Path,
) -> None:
    from interaction_first.structural_novelty import STRUCTURAL_AUDIT_FORMAT
    from interaction_first.usalign_boundary import run_usalign_boundary_spot_checks

    binary = tmp_path / "USalign"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "print('TM-score= 0.49000 (normalized by length of Structure_1: L=10, d0=1)')\n"
        "print('TM-score= 0.48000 (normalized by length of Structure_2: L=12, d0=1)')\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    binary_sha = _sha256(binary)
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "format_version": "framecontact_usalign_tool_lock_v1",
                "remote_binary_sha256": binary_sha,
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "format_version": STRUCTURAL_AUDIT_FORMAT,
                "valid": True,
                "thresholds": {
                    "hard_edge_symmetric_tm_score": 0.5,
                    "foldseek_score_rule": "reciprocal_min",
                },
                "counts": {"boundary_pair_count": 1},
                "boundary_spot_check_pairs": [
                    {
                        "query": "ag_a",
                        "target": "ag_b",
                        "symmetric_tm_score": 0.49,
                        "pair_stratum": "cross_role",
                    }
                ],
                "boundary_spot_check_pairs_truncated": False,
            }
        ),
        encoding="utf-8",
    )
    split = tmp_path / "split.csv"
    export = tmp_path / "export.csv"
    with split.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["antigen_structure_id", "role"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {"antigen_structure_id": "ag_a", "role": "train"},
                {"antigen_structure_id": "ag_b", "role": "test"},
            ]
        )
    with export.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["antigen_structure_id", "foldseek_structure_id"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"antigen_structure_id": "ag_a", "foldseek_structure_id": "hash_a"},
                {"antigen_structure_id": "ag_b", "foldseek_structure_id": "hash_b"},
            ]
        )
    structures = tmp_path / "structures"
    structures.mkdir()
    (structures / "hash_a.pdb").write_text("ATOM\n", encoding="utf-8")
    (structures / "hash_b.pdb").write_text("ATOM\n", encoding="utf-8")
    result = run_usalign_boundary_spot_checks(
        structural_audit_path=audit,
        structural_audit_sha256=_sha256(audit),
        split_manifest_path=split,
        split_manifest_sha256=_sha256(split),
        export_manifest_path=export,
        export_manifest_sha256=_sha256(export),
        structure_root=structures,
        usalign_binary_path=binary,
        usalign_binary_sha256=binary_sha,
        usalign_lock_path=lock,
        usalign_lock_sha256=_sha256(lock),
        output_root=tmp_path / "result",
        workers=1,
    )
    assert result["valid"] is True
    assert result["spot_check_count"] == 1
    assert result["cross_role_usalign_hard_edge_count"] == 0
    assert result["sampled_cross_role_pair_count"] == 1
    assert result["requested_pair_strata"] == {"cross_role": 1}
    assert result["foldseek_usalign_classification_disagreement_count"] == 0
