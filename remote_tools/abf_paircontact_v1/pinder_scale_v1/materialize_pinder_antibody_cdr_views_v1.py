#!/usr/bin/env python3
"""Materialize fixed-IMGT CDR sequences for PINDER-train antibody systems."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


CDR_RANGES = ((27, 38), (56, 65), (105, 117))
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
ANARCII_VERSION = "2.0.8"
ANARCII_WHEEL_SHA256 = "6ac81caef04a9cad1af1ea320b35bdced620fe2bfdd0b8b17b98131495b584ca"
ANARCII_MODEL_SHA256 = "03e9d07d13a3ec98a9d9fe57a0b2ff0853218aa2d7bdd50a25b6d83ff2bea2e0"
ANARCII_PARAMETERS_SHA256 = "4d8e5fc9d8580f5e26a190a21853dd3f3770fbf83ce45a46a53d92b083063c39"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def atomic_create(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_sequence(value: str, label: str) -> str:
    sequence = "".join(value.split()).upper()
    if not sequence or set(sequence) - VALID_AMINO_ACIDS:
        raise ValueError(f"invalid canonical sequence: {label}")
    return sequence


def authenticate_anarcii_runtime(wheel: Path) -> dict[str, Any]:
    if sha256_file(wheel) != ANARCII_WHEEL_SHA256:
        raise ValueError("ANARCII wheel SHA mismatch")
    distribution = metadata.distribution("anarcii")
    if distribution.version != ANARCII_VERSION:
        raise ValueError("ANARCII package version differs")
    root = Path(distribution.locate_file("")).resolve()
    assets = {
        "model": (
            root / "anarcii/models/antibody/antibody_4_2_128_512.pt",
            ANARCII_MODEL_SHA256,
        ),
        "parameters": (
            root / "anarcii/models/antibody/antibody_4_2_128_512.json",
            ANARCII_PARAMETERS_SHA256,
        ),
    }
    observed = {}
    for label, (path, expected) in assets.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"ANARCII {label} asset differs")
        observed[label] = {"path": str(path), "sha256": expected}
    return {
        "package": "anarcii",
        "package_version": ANARCII_VERSION,
        "python": ".".join(map(str, sys.version_info[:3])),
        "device": "cpu",
        "wheel_sha256": ANARCII_WHEEL_SHA256,
        "assets": observed,
    }


def extract_ordered_cdr_sequence(numbering: object) -> str:
    """Extract ungapped IMGT CDR1/2/3 residues from one ANARCII result."""
    if not isinstance(numbering, list):
        raise ValueError("ANARCII numbering is not a list")
    residues: list[tuple[int, str, str]] = []
    seen: set[tuple[int, str]] = set()
    for item in numbering:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("malformed numbering entry")
        coordinate, residue = item
        if not isinstance(coordinate, (tuple, list)) or len(coordinate) != 2:
            raise ValueError("malformed numbering coordinate")
        position, insertion = coordinate
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or not isinstance(insertion, str)
            or not isinstance(residue, str)
            or len(residue) != 1
        ):
            raise ValueError("malformed numbered residue")
        key = (position, insertion)
        if key in seen:
            raise ValueError("duplicate IMGT coordinate")
        seen.add(key)
        if residue != "-" and any(start <= position <= end for start, end in CDR_RANGES):
            if residue not in VALID_AMINO_ACIDS:
                raise ValueError("noncanonical CDR residue")
            residues.append((position, insertion, residue))
    sequence = "".join(residue for _, _, residue in residues)
    if not sequence:
        raise ValueError("empty concatenated CDR sequence")
    for start, end in CDR_RANGES:
        if not any(start <= position <= end for position, _, _ in residues):
            raise ValueError(f"missing IMGT CDR range {start}-{end}")
    return sequence


def require_domain(result: Mapping[str, Any], expected: set[str], label: str) -> str:
    if result.get("error") is not None or result.get("chain_type") not in expected:
        raise ValueError(f"ANARCII domain classification differs: {label}")
    return extract_ordered_cdr_sequence(result.get("numbering"))


def numbered_domains(model: Any, rows: list[dict[str, str]]) -> dict[str, tuple[str, str]]:
    ordinary: dict[str, str] = {}
    scfv: dict[str, str] = {}
    row_queries: dict[str, tuple[str, str | None]] = {}
    for row in rows:
        instance = row["INSTANCE"]
        vh = normalize_sequence(row["vh_seq"], f"{instance}:VH")
        vl = "".join(row["vl_seq"].split()).upper()
        if vl:
            vl = normalize_sequence(vl, f"{instance}:VL")
        if row["topology"] == "scfv":
            key = f"scfv-{len(scfv):06d}"
            scfv[key] = vh
            row_queries[instance] = (key, None)
        else:
            vh_key = f"vh-{sequence_sha256(vh)}"
            ordinary.setdefault(vh_key, vh)
            vl_key = None
            if vl:
                vl_key = f"vl-{sequence_sha256(vl)}"
                ordinary.setdefault(vl_key, vl)
            row_queries[instance] = (vh_key, vl_key)

    ordinary_results = model.number(ordinary, scfv=False)
    scfv_results = model.number(scfv, scfv=True) if scfv else {}
    domains: dict[str, tuple[str, str]] = {}
    for row in rows:
        instance = row["INSTANCE"]
        first, second = row_queries[instance]
        if row["topology"] == "scfv":
            candidates = [
                value
                for key, value in scfv_results.items()
                if key.startswith(f"{first}-")
            ]
            if any(value.get("error") is not None for value in candidates):
                raise ValueError(f"scFv domain recovery differs: {instance}")
            heavy = [value for value in candidates if value.get("chain_type") == "H"]
            light = [value for value in candidates if value.get("chain_type") in {"K", "L"}]
            if (
                len(candidates) not in {1, 2}
                or len(heavy) > 1
                or len(light) > 1
                or len(heavy) + len(light) != len(candidates)
            ):
                raise ValueError(f"scFv domain recovery differs: {instance}")
            domains[instance] = (
                (
                    require_domain(heavy[0], {"H"}, f"{instance}:H")
                    if heavy
                    else ""
                ),
                (
                    require_domain(light[0], {"K", "L"}, f"{instance}:L")
                    if light
                    else ""
                ),
            )
        else:
            heavy = require_domain(ordinary_results[first], {"H"}, f"{instance}:H")
            light = (
                require_domain(ordinary_results[second], {"K", "L"}, f"{instance}:L")
                if second is not None
                else ""
            )
            domains[instance] = (heavy, light)
    return domains


def csv_bytes(rows: Iterable[Mapping[str, str]]) -> bytes:
    fields = (
        "INSTANCE",
        "pdb_id",
        "topology",
        "heavy_cdr_sequence",
        "light_cdr_sequence",
        "combined_cdr_sequence",
        "combined_cdr_sequence_sha256",
    )
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def fasta_bytes(sequences: Iterable[str]) -> bytes:
    unique = {sequence_sha256(sequence): sequence for sequence in sequences}
    return "".join(f">{digest}\n{unique[digest]}\n" for digest in sorted(unique)).encode("ascii")


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    if protocol.get("format_version") != "pinder_antibody_cdr_antigen_s40_protocol_v1":
        raise ValueError("protocol format mismatch")
    canonical_sha = protocol["source"]["canonical_manifest_sha256"]
    if sha256_file(arguments.canonical_manifest) != canonical_sha:
        raise ValueError("canonical manifest SHA mismatch")
    with arguments.canonical_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) != len({row["INSTANCE"] for row in rows}):
        raise ValueError("canonical rows are empty or duplicated")

    from anarcii import Anarcii
    runtime = authenticate_anarcii_runtime(arguments.anarcii_wheel)
    model = Anarcii(cpu=True, ncpu=arguments.workers, batch_size=32, verbose=False)
    domains = numbered_domains(model, rows)
    output_rows = []
    for row in rows:
        heavy, light = domains[row["INSTANCE"]]
        combined = heavy + light
        output_rows.append(
            {
                "INSTANCE": row["INSTANCE"],
                "pdb_id": row["pdb_id"].lower(),
                "topology": row["topology"],
                "heavy_cdr_sequence": heavy,
                "light_cdr_sequence": light,
                "combined_cdr_sequence": combined,
                "combined_cdr_sequence_sha256": sequence_sha256(combined),
            }
        )
    output_rows.sort(key=lambda row: row["INSTANCE"])
    atomic_create(arguments.output_csv, csv_bytes(output_rows))
    atomic_create(
        arguments.output_fasta,
        fasta_bytes(row["combined_cdr_sequence"] for row in output_rows),
    )
    result: dict[str, Any] = {
        "format_version": "pinder_antibody_cdr_view_result_v1",
        "status": "PASS_PINDER_TRAIN_FIXED_IMGT_CDR_VIEWS",
        "counts": {
            "systems": len(output_rows),
            "unique_combined_cdr_sequences": len(
                {row["combined_cdr_sequence_sha256"] for row in output_rows}
            ),
            "topologies": dict(Counter(row["topology"] for row in output_rows)),
        },
        "artifacts": {
            "csv_sha256": sha256_file(arguments.output_csv),
            "fasta_sha256": sha256_file(arguments.output_fasta),
        },
        "cdr_definition": protocol["cdr_definition"],
        "runtime": runtime,
        "source": {
            "canonical_manifest_sha256": canonical_sha,
            "protocol_sha256": sha256_file(arguments.protocol),
        },
        "data_firewall": protocol["authorization"],
        "training_authorized": False,
        "model_selection_authorized": False,
    }
    result["scientific_payload_sha256"] = canonical_sha256(result)
    atomic_create(
        arguments.output_result,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--anarcii-wheel", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--output-result", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    payload = run(parse_arguments())
    print(json.dumps(payload, indent=2, sort_keys=True))
