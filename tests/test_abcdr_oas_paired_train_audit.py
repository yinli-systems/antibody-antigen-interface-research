"""Train-only audit tests for the public AbCDR OAS archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_abcdr_oas_paired_train.py"


def _module():
    specification = importlib.util.spec_from_file_location("abcdr_audit", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _row(heavy: str, light: str) -> list[str]:
    return [
        heavy,
        light,
        heavy[1:3],
        heavy[3:5],
        heavy[5:7],
        light[1:3],
        light[3:5],
        light[5:7],
        "[1, 3]",
        "[3, 5]",
        "[5, 7]",
        "[1, 3]",
        "[3, 5]",
        "[5, 7]",
    ]


def test_audit_reads_train_and_keeps_holdouts_sealed(tmp_path: Path) -> None:
    module = _module()
    archive = tmp_path / "OAS_paired_set.zip"
    csv_text = io.StringIO(newline="")
    writer = csv.writer(csv_text)
    writer.writerow(module.EXPECTED_COLUMNS)
    writer.writerow(_row("ACDEFGHIK", "LMNPQRSTV"))
    writer.writerow(_row("ACDEFGHIK", "LMNPQRSTV"))
    writer.writerow(_row("CDEFGHIKL", "MNPQRSTVW"))
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(module.TRAIN_MEMBER, csv_text.getvalue())
        handle.writestr("OAS_paired_set/validation.csv", "SEALED_VALIDATION")
        handle.writestr("OAS_paired_set/test.csv", "SEALED_TEST")
    metadata = tmp_path / "provider.json"
    metadata.write_text(
        json.dumps(
            {
                "id": 18762309,
                "doi": "10.5281/zenodo.18762309",
                "metadata": {"license": {"id": "cc-by-4.0"}},
                "files": [
                    {
                        "key": archive.name,
                        "size": archive.stat().st_size,
                        "checksum": "md5:fixture",
                    }
                ],
            }
        )
    )
    receipt = module.audit(
        argparse.Namespace(
            archive=archive,
            provider_metadata=metadata,
            expected_size=archive.stat().st_size,
            expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            target_native_pairs=5,
        )
    )
    assert receipt["training_inventory"]["row_count"] == 3
    assert receipt["training_inventory"]["unique_exact_pair_count"] == 2
    assert receipt["training_inventory"]["duplicate_exact_pair_count"] == 1
    assert receipt["training_inventory"]["malformed_cdr_row_count"] == 0
    assert receipt["test_access"]["abcdr_test_member_opened"] is False
    assert receipt["status"] == "MATERIALIZED_NOT_SUFFICIENT_FOR_STAGE_B"
