from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builder_freezes_all_roles_without_task_labels(tmp_path: Path) -> None:
    metadata = tmp_path / "PARTNER_METADATA.csv"
    fieldnames = [
        "instance_id",
        "role",
        "antigen_sequence_sha256",
        "antibody_sequence_sha256",
        "antibody_cluster_id",
    ]
    rows = []
    for role in ("train", "validation", "test"):
        rows.extend(
            [
                {
                    "instance_id": f"{role}-a",
                    "role": role,
                    "antigen_sequence_sha256": f"{role}-ag",
                    "antibody_sequence_sha256": f"{role}-ab-a",
                    "antibody_cluster_id": f"{role}-cluster-a",
                },
                {
                    "instance_id": f"{role}-b",
                    "role": role,
                    "antigen_sequence_sha256": f"{role}-ag",
                    "antibody_sequence_sha256": f"{role}-ab-b",
                    "antibody_cluster_id": f"{role}-cluster-b",
                },
            ]
        )
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata_sha = _sha256(metadata)
    receipt = tmp_path / "PARTNER_METADATA_RECEIPT.json"
    receipt.write_text(
        json.dumps(
            {
                "format_version": "sabdab2_partner_metadata_split_v1",
                "instance_count": len(rows),
                "metadata_sha256": metadata_sha,
                "authority": {"task_labels_read": False, "split_sha256": "split"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "challenge"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_sabdab2_same_antigen_challenge.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--metadata",
            str(metadata),
            "--metadata-sha256",
            metadata_sha,
            "--metadata-receipt",
            str(receipt),
            "--metadata-receipt-sha256",
            _sha256(receipt),
            "--output-root",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["receipt_sha256"] == _sha256(
        output / "SAME_ANTIGEN_CHALLENGE_RECEIPT.json"
    )
    for role in ("train", "validation", "test"):
        panel = json.loads((output / f"{role}.json").read_text())
        assert panel["authority"]["task_labels_read"] is False
        assert panel["focal_instance_count"] == 2
        assert panel["ordered_pair_count"] == 2
