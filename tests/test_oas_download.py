from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest

from antibody_pretraining.oas_authority import (
    build_oas_authority_manifest,
    canonical_json_bytes,
)
from antibody_pretraining.oas_download import (
    OASDownloadError,
    inspect_gzip_data_unit,
    load_authority_manifest,
    select_task_data_units,
)


def _authority_html() -> bytes:
    return b"""
    <p>Your search yielded <b>6</b> unique sequences from <b>1</b> studies.</p>
    <script>var CSV = [
      "wget https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Study_2020/csv/a.csv.gz",
      "wget https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Study_2020/csv/b.csv.gz",
      "wget https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Study_2020/csv/c.csv.gz"
    ];</script>
    <table id="results_table"><tr><th>Details</th><th>DS Name</th><th>#Unique Sequences</th><th>Chain</th></tr>
    <tr><td><a href="../dataunit?unit=Study_2020%2Fcsv%2Fa.csv.gz">Details</a></td><td>Study_2020</td><td>1</td><td>Heavy</td></tr>
    <tr><td><a href="../dataunit?unit=Study_2020%2Fcsv%2Fb.csv.gz">Details</a></td><td>Study_2020</td><td>2</td><td>Heavy</td></tr>
    <tr><td><a href="../dataunit?unit=Study_2020%2Fcsv%2Fc.csv.gz">Details</a></td><td>Study_2020</td><td>3</td><td>Light</td></tr>
    </table>
    """


def _write_data_unit(path: Path, row_count: int) -> None:
    metadata = json.dumps({"Unique sequences": row_count}, sort_keys=True)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([metadata])
    writer.writerow(
        ["sequence_alignment_aa", "productive", "ANARCI_status", "locus"]
    )
    for index in range(row_count):
        writer.writerow(["ACDEFGHIK", "T", "ok", f"H{index}"])
    with gzip.open(path, "wb") as handle:
        handle.write(output.getvalue().encode("utf-8"))


def test_loaded_authority_requires_file_payload_and_semantic_agreement(
    tmp_path: Path,
) -> None:
    manifest = build_oas_authority_manifest(
        _authority_html(),
        source_class="unpaired",
    )
    payload = canonical_json_bytes(manifest)
    path = tmp_path / "manifest.json"
    path.write_bytes(payload)

    loaded = load_authority_manifest(
        path,
        expected_file_sha256=hashlib.sha256(payload).hexdigest(),
        expected_payload_sha256=manifest["manifest_sha256"],
    )
    assert loaded == manifest

    with pytest.raises(OASDownloadError, match="file digest differs"):
        load_authority_manifest(
            path,
            expected_file_sha256="0" * 64,
            expected_payload_sha256=manifest["manifest_sha256"],
        )


def test_modulo_shards_are_disjoint_complete_and_deterministic() -> None:
    units = build_oas_authority_manifest(
        _authority_html(), source_class="unpaired"
    )["data_units"]
    shards = [
        select_task_data_units(units, task_id=task_id, task_count=2)
        for task_id in range(2)
    ]

    assert [entry["relative_unit"] for entry in shards[0]] == [
        "Study_2020/csv/a.csv.gz",
        "Study_2020/csv/c.csv.gz",
    ]
    assert [entry["relative_unit"] for entry in shards[1]] == [
        "Study_2020/csv/b.csv.gz"
    ]
    assert {entry["relative_unit"] for shard in shards for entry in shard} == {
        entry["relative_unit"] for entry in units
    }
    assert select_task_data_units(
        units,
        task_id=0,
        task_count=2,
        maximum_units=1,
    ) == shards[0][:1]


def test_gzip_inspection_validates_complete_stream_schema_and_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unit.csv.gz"
    _write_data_unit(path, 3)

    inspection = inspect_gzip_data_unit(
        path,
        expected_unique_sequences=3,
        source_class="unpaired",
    )
    assert inspection.data_row_count == 3
    assert inspection.compressed_bytes == path.stat().st_size
    assert inspection.compressed_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert inspection.metadata == {"Unique sequences": 3}
    assert "sequence_alignment_aa" in inspection.header

    with pytest.raises(OASDownloadError, match="unique count differs"):
        inspect_gzip_data_unit(
            path,
            expected_unique_sequences=2,
            source_class="unpaired",
        )


def test_gzip_inspection_rejects_corruption_and_missing_schema(tmp_path: Path) -> None:
    valid = tmp_path / "valid.csv.gz"
    _write_data_unit(valid, 1)
    corrupt = tmp_path / "corrupt.csv.gz"
    corrupt.write_bytes(valid.read_bytes()[:-4])
    with pytest.raises(OASDownloadError, match="gzip validation"):
        inspect_gzip_data_unit(
            corrupt,
            expected_unique_sequences=1,
            source_class="unpaired",
        )

    missing = tmp_path / "missing.csv.gz"
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([json.dumps({"Unique sequences": 1})])
    writer.writerow(["sequence_alignment_aa", "productive"])
    writer.writerow(["ACDE", "T"])
    with gzip.open(missing, "wb") as handle:
        handle.write(output.getvalue().encode("utf-8"))
    with pytest.raises(OASDownloadError, match="missing required columns"):
        inspect_gzip_data_unit(
            missing,
            expected_unique_sequences=1,
            source_class="unpaired",
        )


def test_task_selection_rejects_invalid_ranges() -> None:
    with pytest.raises(OASDownloadError, match="task_count"):
        select_task_data_units([], task_id=0, task_count=0)
    with pytest.raises(OASDownloadError, match="task_id"):
        select_task_data_units([], task_id=2, task_count=2)
    with pytest.raises(OASDownloadError, match="maximum_units"):
        select_task_data_units([], task_id=0, task_count=1, maximum_units=0)
