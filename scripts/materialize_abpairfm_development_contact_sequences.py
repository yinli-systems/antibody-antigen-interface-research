#!/usr/bin/env python3
"""Materialize a label-free development-only sequence authority from Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


EXPECTED_TRANSFER_FORMAT = "abpairfm_development_transfer_manifest_v2"
EXPECTED_TRANSFER_STATUS = "PASS_DEVELOPMENT_ONLY_MANIFEST_PRIMARY_TEST_SEALED"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-manifest", type=Path, required=True)
    parser.add_argument("--expected-transfer-manifest-sha256", required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--expected-sequences-sha256", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--expected-prepared-sha256", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    return parser.parse_args()


def _publish_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite sequence authority: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _selected_ids(transfer: dict[str, Any]) -> list[str]:
    if (
        transfer.get("format_version") != EXPECTED_TRANSFER_FORMAT
        or transfer.get("status") != EXPECTED_TRANSFER_STATUS
        or transfer.get("test_access") is not False
        or transfer.get("authority", {}).get("primary_sealed_role_authority")
        != "standard"
        or transfer.get("authority", {}).get("primary_sealed_role_ids_retained")
        is not False
        or transfer.get("authority", {}).get("task_labels_read") is not False
    ):
        raise RuntimeError("Development transfer authority differs")
    identifiers: set[str] = set()
    for value in transfer.get("slices", {}).values():
        rows = value.get("instance_ids", [])
        if (
            len(rows) != value.get("count")
            or rows != sorted(rows)
            or len(rows) != len(set(rows))
        ):
            raise RuntimeError("Development transfer slice differs")
        identifiers.update(rows)
    if not identifiers:
        raise RuntimeError("Development transfer authority is empty")
    return sorted(identifiers)


def main() -> None:
    arguments = _arguments()
    anchors = (
        (arguments.transfer_manifest, arguments.expected_transfer_manifest_sha256),
        (arguments.sequences, arguments.expected_sequences_sha256),
        (arguments.prepared, arguments.expected_prepared_sha256),
    )
    for path, expected in anchors:
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Input authority differs: {path}")
    transfer = json.loads(arguments.transfer_manifest.read_text(encoding="utf-8"))
    selected = _selected_ids(transfer)

    import pandas as pd
    import pyarrow

    sequences = pd.read_parquet(
        arguments.sequences,
        columns=["instance_id", "ab_seq", "ag_seq"],
        filters=[("instance_id", "in", selected)],
    )
    prepared = pd.read_parquet(
        arguments.prepared,
        columns=[
            "instance_id",
            "ab_sequence_sha256",
            "ag_sequence_sha256",
            "ab_length",
            "ag_length",
        ],
        filters=[("instance_id", "in", selected)],
    )
    if (
        sequences["instance_id"].duplicated().any()
        or prepared["instance_id"].duplicated().any()
        or set(sequences["instance_id"]) != set(selected)
        or set(prepared["instance_id"]) != set(selected)
    ):
        raise RuntimeError("Filtered sequence authority differs")
    sequences = sequences.set_index("instance_id")
    prepared = prepared.set_index("instance_id")
    rows = []
    for identifier in selected:
        antibody = str(sequences.loc[identifier, "ab_seq"]).strip().upper()
        antigen = str(sequences.loc[identifier, "ag_seq"]).strip().upper()
        authority = prepared.loc[identifier]
        row = {
            "antibody_length": len(antibody),
            "antibody_sequence": antibody,
            "antibody_sequence_sha256": _sequence_sha256(antibody),
            "antigen_length": len(antigen),
            "antigen_sequence": antigen,
            "antigen_sequence_sha256": _sequence_sha256(antigen),
            "instance_id": identifier,
        }
        if (
            not antibody
            or not antigen
            or row["antibody_length"] != int(authority["ab_length"])
            or row["antigen_length"] != int(authority["ag_length"])
            or row["antibody_sequence_sha256"] != authority["ab_sequence_sha256"]
            or row["antigen_sequence_sha256"] != authority["ag_sequence_sha256"]
        ):
            raise RuntimeError(f"Sequence authority differs: {identifier}")
        rows.append(row)
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    _publish_create_only(arguments.output_jsonl, payload)
    receipt = {
        "authority": {
            "columns_read": {
                "prepared": [
                    "instance_id",
                    "ab_sequence_sha256",
                    "ag_sequence_sha256",
                    "ab_length",
                    "ag_length",
                ],
                "sequences": ["instance_id", "ab_seq", "ag_seq"],
            },
            "contact_labels_read": False,
            "prepared_sha256": arguments.expected_prepared_sha256,
            "primary_sealed_role_ids_retained": False,
            "sequences_sha256": arguments.expected_sequences_sha256,
            "task_labels_read": False,
            "transfer_manifest_sha256": arguments.expected_transfer_manifest_sha256,
        },
        "format_version": "abpairfm_development_contact_sequence_authority_v1",
        "panel": {
            "bytes": arguments.output_jsonl.stat().st_size,
            "row_count": len(rows),
            "sha256": sha256_file(arguments.output_jsonl),
        },
        "runtime": {
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "python": sys.executable,
            "python_version": sys.version.split()[0],
        },
        "status": "FROZEN_DEVELOPMENT_CONTACT_SEQUENCE_AUTHORITY",
        "test_access": False,
    }
    _publish_create_only(
        arguments.output_receipt,
        (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
