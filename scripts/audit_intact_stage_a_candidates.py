#!/usr/bin/env python3
"""Build a strict, sequence-unresolved IntAct Stage A candidate inventory."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Iterable
import zipfile


EXPECTED_HEADER = (
    "#ID(s) interactor A",
    "ID(s) interactor B",
    "Alt. ID(s) interactor A",
    "Alt. ID(s) interactor B",
    "Alias(es) interactor A",
    "Alias(es) interactor B",
    "Interaction detection method(s)",
    "Publication 1st author(s)",
    "Publication Identifier(s)",
    "Taxid interactor A",
    "Taxid interactor B",
    "Interaction type(s)",
    "Source database(s)",
    "Interaction identifier(s)",
    "Confidence value(s)",
    "Expansion method(s)",
    "Biological role(s) interactor A",
    "Biological role(s) interactor B",
    "Experimental role(s) interactor A",
    "Experimental role(s) interactor B",
    "Type(s) interactor A",
    "Type(s) interactor B",
    "Xref(s) interactor A",
    "Xref(s) interactor B",
    "Interaction Xref(s)",
    "Annotation(s) interactor A",
    "Annotation(s) interactor B",
    "Interaction annotation(s)",
    "Host organism(s)",
    "Interaction parameter(s)",
    "Creation date",
    "Update date",
    "Checksum(s) interactor A",
    "Checksum(s) interactor B",
    "Interaction Checksum(s)",
    "Negative",
)
_ACCESSION = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9][A-Z0-9]{3}[0-9])(?:-[0-9]+)?$"
)
_ALLOWED_INTERACTION_TYPES = {"MI:0407", "MI:0915"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, temporary: Path) -> None:
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _publish_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                (
                    json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        _publish(path, temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _mi_ids(field: str) -> set[str]:
    return set(re.findall(r"MI:[0-9]{4}", field))


def _uniprot_accession(*fields: str) -> str | None:
    for field in fields:
        for token in field.split("|"):
            prefix, separator, raw = token.partition(":")
            if separator and prefix.lower() == "uniprotkb":
                candidate = raw.split("(", 1)[0]
                if _ACCESSION.fullmatch(candidate):
                    return candidate.split("-", 1)[0]
    return None


def _miscore(field: str) -> float | None:
    for token in field.split("|"):
        if token.startswith("intact-miscore:"):
            try:
                value = float(token.split(":", 1)[1])
            except ValueError:
                return None
            return value if math.isfinite(value) else None
    return None


def _taxid(field: str) -> str:
    match = re.search(r"taxid:([^|(]+)", field)
    return match.group(1) if match else "-"


def _classify(row: list[str]) -> tuple[str, dict[str, Any] | None]:
    if len(row) != len(EXPECTED_HEADER):
        return "column_count", None
    negative = row[35].lower()
    if negative not in {"true", "false"}:
        return "invalid_negative", None
    if 'MI:0326' not in _mi_ids(row[20]) or 'MI:0326' not in _mi_ids(row[21]):
        return "non_protein_endpoint", None
    accession_a = _uniprot_accession(row[0], row[2])
    accession_b = _uniprot_accession(row[1], row[3])
    if accession_a is None or accession_b is None:
        return "unresolved_uniprot", None
    if accession_a == accession_b:
        return "self_pair", None
    if row[15] != "-":
        return "expanded_complex", None
    if "pubmed:" not in row[8]:
        return "missing_pubmed", None
    interaction_types = _mi_ids(row[11]) & _ALLOWED_INTERACTION_TYPES
    if negative == "false":
        score = _miscore(row[14])
        if not interaction_types:
            return "positive_interaction_type", None
        if score is None or score < 0.6:
            return "positive_miscore", None
        label = 1
    else:
        score = _miscore(row[14])
        label = 0
    endpoint_a, endpoint_b = sorted((accession_a, accession_b))
    return "accepted", {
        "accession_a": endpoint_a,
        "accession_b": endpoint_b,
        "interaction_ids": row[13],
        "interaction_types": "|".join(sorted(interaction_types)) or "-",
        "label": label,
        "miscore": score,
        "publication_ids": row[8],
        "row_sha256": hashlib.sha256("\t".join(row).encode("utf-8")).hexdigest(),
        "taxid_a": _taxid(row[9]),
        "taxid_b": _taxid(row[10]),
    }


def _rows(archive: Path) -> Iterable[tuple[str, list[str]]]:
    with zipfile.ZipFile(archive) as source:
        names = set(source.namelist())
        expected = {"intact-micluster.txt", "intact-micluster_negative.txt"}
        if names != expected:
            raise RuntimeError(f"Unexpected IntAct archive members: {sorted(names)}")
        for name in sorted(names):
            with source.open(name) as binary:
                text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
                reader = csv.reader(text, delimiter="\t", quoting=csv.QUOTE_NONE)
                header = tuple(next(reader))
                if header != EXPECTED_HEADER:
                    raise RuntimeError(f"Unexpected MITAB header in {name}")
                for row in reader:
                    yield name, row


def _insert(connection: sqlite3.Connection, record: dict[str, Any]) -> None:
    connection.execute(
        """
        INSERT INTO candidates (
            accession_a, accession_b, label, miscore, interaction_types,
            publication_ids, interaction_ids, taxid_a, taxid_b, row_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(accession_a, accession_b, label) DO UPDATE SET
            miscore = CASE
                WHEN excluded.miscore IS NULL THEN candidates.miscore
                WHEN candidates.miscore IS NULL THEN excluded.miscore
                ELSE MAX(candidates.miscore, excluded.miscore)
            END,
            interaction_types = MIN(candidates.interaction_types, excluded.interaction_types),
            publication_ids = MIN(candidates.publication_ids, excluded.publication_ids),
            interaction_ids = MIN(candidates.interaction_ids, excluded.interaction_ids),
            row_sha256 = MIN(candidates.row_sha256, excluded.row_sha256)
        """,
        (
            record["accession_a"],
            record["accession_b"],
            record["label"],
            record["miscore"],
            record["interaction_types"],
            record["publication_ids"],
            record["interaction_ids"],
            record["taxid_a"],
            record["taxid_b"],
            record["row_sha256"],
        ),
    )


def _export(connection: sqlite3.Connection, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    counts = Counter()
    accessions: set[str] = set()
    try:
        with temporary_path.open("wb") as binary:
            with gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.writer(text, delimiter="\t", lineterminator="\n")
                    writer.writerow(
                        (
                            "accession_a",
                            "accession_b",
                            "label",
                            "miscore",
                            "interaction_types",
                            "publication_ids",
                            "interaction_ids",
                            "taxid_a",
                            "taxid_b",
                            "row_sha256",
                        )
                    )
                    query = connection.execute(
                        "SELECT accession_a, accession_b, label, miscore, "
                        "interaction_types, publication_ids, interaction_ids, "
                        "taxid_a, taxid_b, row_sha256 FROM candidates "
                        "ORDER BY accession_a, accession_b, label"
                    )
                    for row in query:
                        writer.writerow("" if value is None else value for value in row)
                        counts[int(row[2])] += 1
                        accessions.update((str(row[0]), str(row[1])))
            binary.flush()
            os.fsync(binary.fileno())
        _publish(output, temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "negative_pair_count": counts[0],
        "positive_pair_count": counts[1],
        "unique_accession_count": len(accessions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    archive = arguments.archive.resolve()
    source_sha256 = _sha256_file(Path(__file__).resolve())
    if source_sha256 != arguments.expected_source_sha256:
        raise RuntimeError("IntAct audit source SHA-256 differs from the launch anchor")
    protocol = arguments.protocol.resolve()
    protocol_sha256 = _sha256_file(protocol)
    if protocol_sha256 != arguments.expected_protocol_sha256:
        raise RuntimeError("IntAct protocol SHA-256 differs from the launch anchor")
    if _sha256_file(archive) != arguments.expected_archive_sha256:
        raise RuntimeError("IntAct archive SHA-256 differs from the materialization anchor")
    rejection = Counter()
    source_rows = Counter()
    descriptor, database_name = tempfile.mkstemp(prefix="intact-candidates-", suffix=".sqlite3")
    os.close(descriptor)
    database = Path(database_name)
    try:
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            """
            CREATE TABLE candidates (
                accession_a TEXT NOT NULL,
                accession_b TEXT NOT NULL,
                label INTEGER NOT NULL,
                miscore REAL,
                interaction_types TEXT NOT NULL,
                publication_ids TEXT NOT NULL,
                interaction_ids TEXT NOT NULL,
                taxid_a TEXT NOT NULL,
                taxid_b TEXT NOT NULL,
                row_sha256 TEXT NOT NULL,
                PRIMARY KEY (accession_a, accession_b, label)
            ) WITHOUT ROWID
            """
        )
        for index, (source_name, row) in enumerate(_rows(archive), start=1):
            source_rows[source_name] += 1
            reason, record = _classify(row)
            rejection[reason] += 1
            if record is not None:
                _insert(connection, record)
            if index % 100000 == 0:
                connection.commit()
                print(json.dumps({"rows": index, "accepted_rows": rejection["accepted"]}), flush=True)
        connection.commit()
        inventory = _export(connection, arguments.output.resolve())
        connection.close()
    finally:
        database.unlink(missing_ok=True)
    output = arguments.output.resolve()
    receipt = {
        "authorization": {
            "development_decontamination_complete": False,
            "sequence_resolution_complete": False,
            "stage_a_training": False,
        },
        "claim_boundary": "Sequence-unresolved IntAct candidate inventory only. No candidate is authorized for training until sequence resolution, length filtering, conflict removal, and development decontamination pass.",
        "format_version": "abpairfm_intact_candidate_audit_v1",
        "input": {
            "archive_sha256": arguments.expected_archive_sha256,
            "protocol_sha256": protocol_sha256,
            "source_sha256": source_sha256,
            "source_rows": dict(sorted(source_rows.items())),
        },
        "inventory": inventory,
        "output": {
            "bytes": output.stat().st_size,
            "sha256": _sha256_file(output),
        },
        "rejection_counts": dict(sorted(rejection.items())),
        "status": "CANDIDATES_NOT_AUTHORIZED",
        "test_access": False,
    }
    _publish_json(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
