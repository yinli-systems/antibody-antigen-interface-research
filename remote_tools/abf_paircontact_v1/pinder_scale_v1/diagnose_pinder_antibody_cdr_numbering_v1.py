#!/usr/bin/env python3
"""Read-only diagnosis of ANARCII domain/CDR failures in the frozen PINDER cohort."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from materialize_pinder_antibody_cdr_views_v1 import (
    CDR_RANGES,
    authenticate_anarcii_runtime,
    canonical_sha256,
    extract_ordered_cdr_sequence,
    normalize_sequence,
    sequence_sha256,
    sha256_file,
)


def summarize_numbering(
    result: Mapping[str, Any], expected_chain_types: set[str]
) -> dict[str, Any]:
    numbering = result.get("numbering")
    positions: list[int] = []
    if isinstance(numbering, list):
        for item in numbering:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                continue
            coordinate, residue = item
            if (
                isinstance(coordinate, (tuple, list))
                and len(coordinate) == 2
                and isinstance(coordinate[0], int)
                and isinstance(residue, str)
                and residue != "-"
            ):
                positions.append(coordinate[0])
    missing_ranges = [
        f"{start}-{end}"
        for start, end in CDR_RANGES
        if not any(start <= position <= end for position in positions)
    ]
    failures: list[str] = []
    if result.get("error") is not None:
        failures.append(f"anarcii_error:{result.get('error')}")
    if result.get("chain_type") not in expected_chain_types:
        failures.append(f"chain_type:{result.get('chain_type')}")
    try:
        cdr = extract_ordered_cdr_sequence(numbering)
    except ValueError as error:
        failures.append(str(error))
        cdr = ""
    return {
        "status": "PASS" if not failures else "FAIL",
        "chain_type": result.get("chain_type"),
        "anarcii_error": result.get("error"),
        "numbered_residue_count": len(positions),
        "numbered_position_min": min(positions) if positions else None,
        "numbered_position_max": max(positions) if positions else None,
        "missing_cdr_ranges": missing_ranges,
        "cdr_length": len(cdr),
        "failures": failures,
    }


def resolve_scfv_source_key(result_key: str, source_keys: set[str]) -> str | None:
    """Map ANARCII's original or domain-suffixed key back to its input key."""
    if result_key in source_keys:
        return result_key
    return next(
        (key for key in source_keys if result_key.startswith(f"{key}-")), None
    )


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    if protocol.get("format_version") != "pinder_antibody_cdr_antigen_s40_protocol_v1":
        raise ValueError("protocol format mismatch")
    if sha256_file(arguments.canonical_manifest) != protocol["source"][
        "canonical_manifest_sha256"
    ]:
        raise ValueError("canonical manifest SHA mismatch")
    with arguments.canonical_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) != len({row["INSTANCE"] for row in rows}):
        raise ValueError("canonical rows are empty or duplicated")

    from anarcii import Anarcii

    runtime = authenticate_anarcii_runtime(arguments.anarcii_wheel)
    model = Anarcii(cpu=True, ncpu=arguments.workers, batch_size=32, verbose=False)

    ordinary: dict[str, str] = {}
    ordinary_context: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    scfv: dict[str, str] = {}
    scfv_instances: dict[str, str] = {}
    for row in rows:
        instance = row["INSTANCE"]
        vh = normalize_sequence(row["vh_seq"], f"{instance}:VH")
        vl = "".join(row["vl_seq"].split()).upper()
        if vl:
            vl = normalize_sequence(vl, f"{instance}:VL")
        if row["topology"] == "scfv":
            key = f"scfv-{len(scfv):06d}"
            scfv[key] = vh
            scfv_instances[key] = instance
            continue
        for role, sequence, expected in (
            ("heavy", vh, "H"),
            ("light", vl, "K,L"),
        ):
            if not sequence:
                continue
            key = f"{role}-{sequence_sha256(sequence)}"
            ordinary.setdefault(key, sequence)
            ordinary_context[key].append(
                {"instance": instance, "role": role, "expected_chain_types": expected}
            )

    ordinary_results = model.number(ordinary, scfv=False)
    scfv_results = model.number(scfv, scfv=True) if scfv else {}
    failures: list[dict[str, Any]] = []
    passing_domains = 0

    for key in sorted(ordinary):
        contexts = ordinary_context[key]
        expected = set(contexts[0]["expected_chain_types"].split(","))
        if any(set(item["expected_chain_types"].split(",")) != expected for item in contexts):
            raise ValueError("one sequence was assigned conflicting domain roles")
        summary = summarize_numbering(ordinary_results[key], expected)
        if summary["status"] == "PASS":
            passing_domains += 1
            continue
        failures.append(
            {
                "query_key": key,
                "sequence_sha256": sequence_sha256(ordinary[key]),
                "sequence_length": len(ordinary[key]),
                "contexts": sorted(contexts, key=lambda item: (item["instance"], item["role"])),
                "diagnostic": summary,
            }
        )

    by_scfv: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result_key, value in scfv_results.items():
        source_key = resolve_scfv_source_key(result_key, set(scfv))
        if source_key is None:
            raise ValueError("ANARCII returned an unknown scFv result key")
        by_scfv[source_key].append(value)
    for key in sorted(scfv):
        domains = by_scfv.get(key, [])
        instance = scfv_instances[key]
        if not domains:
            failures.append(
                {
                    "query_key": key,
                    "sequence_sha256": sequence_sha256(scfv[key]),
                    "sequence_length": len(scfv[key]),
                    "contexts": [{"instance": instance, "role": "scfv"}],
                    "diagnostic": {"status": "FAIL", "failures": ["no_domains"]},
                }
            )
            continue
        for domain_index, result in enumerate(domains):
            chain_type = result.get("chain_type")
            expected = {"H"} if chain_type == "H" else {"K", "L"}
            summary = summarize_numbering(result, expected)
            if summary["status"] == "PASS":
                passing_domains += 1
                continue
            failures.append(
                {
                    "query_key": f"{key}-domain-{domain_index}",
                    "sequence_sha256": sequence_sha256(scfv[key]),
                    "sequence_length": len(scfv[key]),
                    "contexts": [{"instance": instance, "role": "scfv"}],
                    "diagnostic": summary,
                }
            )

    payload: dict[str, Any] = {
        "format_version": "pinder_antibody_cdr_numbering_diagnostic_v1",
        "status": "PASS_DIAGNOSTIC_COMPLETE_WITHOUT_TRAINING_AUTHORIZATION",
        "source": {
            "protocol_sha256": sha256_file(arguments.protocol),
            "canonical_manifest_sha256": sha256_file(arguments.canonical_manifest),
            "anarcii_runtime": runtime,
        },
        "counts": {
            "canonical_systems": len(rows),
            "ordinary_unique_queries": len(ordinary),
            "scfv_queries": len(scfv),
            "passing_domains": passing_domains,
            "failing_domains": len(failures),
        },
        "failures": failures,
        "authorizations": {"split": False, "training": False, "evaluation": False},
    }
    payload["evidence_sha256"] = canonical_sha256(payload)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_arguments(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--anarcii-wheel", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_arguments())
    print(json.dumps(result["counts"], sort_keys=True))
