from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize_antigen_fold_cohort.py"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_summary_reports_partner_axes_lengths_and_overlap(tmp_path: Path) -> None:
    split = tmp_path / "split.csv"
    metadata = tmp_path / "metadata.csv"
    output = tmp_path / "summary.json"
    common = [
        {
            "instance_id": "a",
            "role": "train",
            "antigen_sequence_sha256": "ag-a",
            "antibody_sequence_sha256": "ab-a",
            "antigen_cluster_id": "agc-a",
            "antibody_cluster_id": "abc-a",
            "pdb_assembly_id": "pdb-a",
        },
        {
            "instance_id": "b",
            "role": "validation",
            "antigen_sequence_sha256": "ag-b",
            "antibody_sequence_sha256": "ab-b",
            "antigen_cluster_id": "agc-b",
            "antibody_cluster_id": "abc-b",
            "pdb_assembly_id": "pdb-b",
        },
        {
            "instance_id": "c",
            "role": "test",
            "antigen_sequence_sha256": "ag-c",
            "antibody_sequence_sha256": "ab-c",
            "antigen_cluster_id": "agc-c",
            "antibody_cluster_id": "abc-c",
            "pdb_assembly_id": "pdb-c",
        },
    ]
    _write_csv(split, list(common[0]), common)
    meta_rows = []
    for index, row in enumerate(common, start=1):
        meta_rows.append(
            {
                "instance_id": row["instance_id"],
                "antigen_sequence_sha256": row["antigen_sequence_sha256"],
                "antibody_sequence_sha256": row["antibody_sequence_sha256"],
                "antigen_length": 100 * index,
                "antibody_length": 50 * index,
                "antigen_class": "protein",
                "antibody_class": "fv",
                "documented_cognate_group_id": f"g-{index}",
            }
        )
    _write_csv(metadata, list(meta_rows[0]), meta_rows)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--split",
            str(split),
            "--metadata",
            str(metadata),
            "--output",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["roles"]["validation"]["samples"] == 1
    assert summary["roles"]["validation"]["antigen_length"]["median"] == 200
    assert summary["roles"]["test"]["unique_antibody_sequences"] == 1
    assert not any(summary["cross_role_overlap_value_counts"].values())
    assert summary["authority"]["task_labels_read"] is False
