#!/usr/bin/env python3
"""Build an immutable cross-source-deduplicated Stage-A1 pilot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable


EXPECTED_FORMAT = "abpairfm_stage_a1_multitask_pilot_v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_pair_hash(left: str, right: str) -> str:
    endpoint_hashes = sorted(
        hashlib.sha256(sequence.encode("ascii")).hexdigest()
        for sequence in (left, right)
    )
    return hashlib.sha256((endpoint_hashes[0] + endpoint_hashes[1]).encode("ascii")).hexdigest()


def selection_hash(seed: int, source: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}\0{source}\0{identifier}".encode("utf-8")).hexdigest()


def zstd_lines(path: Path) -> Iterable[str]:
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    try:
        yield from process.stdout
    finally:
        process.stdout.close()
        error = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstd failed with {return_code}: {error[-1000:]}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def _load_source(protocol: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    anchor = protocol["sources"][name]
    manifest = Path(anchor["manifest_path"])
    receipt_path = Path(anchor["receipt_path"])
    if (
        manifest.is_symlink()
        or receipt_path.is_symlink()
        or not manifest.is_file()
        or not receipt_path.is_file()
        or sha256_file(manifest) != anchor["manifest_sha256"]
        or sha256_file(receipt_path) != anchor["receipt_sha256"]
    ):
        raise RuntimeError(f"Stage-A1 source authentication failed: {name}")
    return manifest, json.loads(receipt_path.read_text(encoding="utf-8"))


def _publish_zstd_create_only(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            process = subprocess.Popen(
                ["zstd", "-q", "-8", "-c"],
                stdin=subprocess.PIPE,
                stdout=raw,
                stderr=subprocess.PIPE,
            )
            assert process.stdin is not None
            for row in rows:
                process.stdin.write(
                    (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                        "utf-8"
                    )
                )
            process.stdin.close()
            error = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
            if return_code:
                raise RuntimeError(
                    f"zstd compression failed with {return_code}: {error[-1000:].decode('utf-8', errors='replace')}"
                )
            raw.flush()
            os.fsync(raw.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    arguments = _arguments()
    if sha256_file(arguments.protocol) != arguments.expected_protocol_sha256:
        raise RuntimeError("Stage-A1 pilot protocol SHA-256 differs")
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("format_version") != EXPECTED_FORMAT
        or protocol.get("status") != "FROZEN_BEFORE_PILOT_MANIFEST_AND_GPU_TRAINING"
        or protocol.get("authorization", {}).get("pilot_training") is not True
        or protocol.get("authorization", {}).get("production_training") is not False
        or protocol.get("test_access") is not False
    ):
        raise RuntimeError("Stage-A1 pilot protocol authorization differs")
    if arguments.output.exists() or arguments.receipt.exists():
        raise FileExistsError("Stage-A1 pilot manifest outputs are create-only")

    ppiref_path, ppiref_receipt = _load_source(protocol, "ppiref50k")
    string_path, string_receipt = _load_source(protocol, "string_segment")
    if (
        ppiref_receipt.get("status")
        != "PINDER_PRIORITY_DEDUPLICATED_STRING_MERGE_PENDING"
        or ppiref_receipt.get("test_access") is not False
        or string_receipt.get("status") != "PRODUCTION_STAGE_A0_SEGMENT_READY"
        or string_receipt.get("test_access") is not False
    ):
        raise RuntimeError("Stage-A1 source receipt status differs")

    ppiref_by_pair: dict[str, dict[str, Any]] = {}
    ppiref_input_count = 0
    ppiref_short_full_endpoint_excluded = 0
    ppiref_short_patch_endpoint_excluded = 0
    layout = protocol["pilot_manifest"]
    minimum_full_endpoint = layout["minimum_full_endpoint_residues"]
    minimum_patch_endpoint = layout["minimum_patch_endpoint_residues"]
    for line in zstd_lines(ppiref_path):
        value = json.loads(line)
        ppiref_input_count += 1
        left = str(value["chain1_full_sequence"])
        right = str(value["chain2_full_sequence"])
        patch_left = str(value["chain1_patch_sequence"])
        patch_right = str(value["chain2_patch_sequence"])
        contacts = value["contact_pairs"]
        if min(len(left), len(right)) < minimum_full_endpoint:
            ppiref_short_full_endpoint_excluded += 1
            continue
        if min(len(patch_left), len(patch_right)) < minimum_patch_endpoint:
            ppiref_short_patch_endpoint_excluded += 1
            continue
        if (
            not contacts
            or any(
                i < 0 or j < 0 or i >= len(patch_left) or j >= len(patch_right)
                for i, j in contacts
            )
        ):
            raise RuntimeError("PPIRef contact record violates pilot invariants")
        pair_hash = canonical_pair_hash(left, right)
        identifier = str(value["example_id"])
        record = {
            "contact_pairs": contacts,
            "contact_supervision": True,
            "evidence_role": "structural_interface",
            "example_id": identifier,
            "full_pair_hash_sha256": pair_hash,
            "known_positive_group": str(value["pdb_id"]),
            "retrieval_supervision": True,
            "sequence1": patch_left,
            "sequence2": patch_right,
            "source": "ppiref50k",
        }
        previous = ppiref_by_pair.get(pair_hash)
        if previous is None or identifier < previous["example_id"]:
            ppiref_by_pair[pair_hash] = record

    ppiref_pair_hashes = set(ppiref_by_pair)
    string_candidates: list[dict[str, Any]] = []
    string_cross_source_removed = 0
    for line in zstd_lines(string_path):
        value = json.loads(line)
        left = str(value["sequence1"])
        right = str(value["sequence2"])
        pair_hash = canonical_pair_hash(left, right)
        if pair_hash in ppiref_pair_hashes:
            string_cross_source_removed += 1
            continue
        evidence = str(value["evidence_role"])
        string_candidates.append(
            {
                "contact_pairs": [],
                "contact_supervision": False,
                "evidence_role": evidence,
                "example_id": str(value["pair_hash_sha256"]),
                "full_pair_hash_sha256": pair_hash,
                "known_positive_group": str(value["pair_hash_sha256"]),
                "retrieval_supervision": evidence == "direct",
                "sequence1": left,
                "sequence2": right,
                "source": "string_v12_physical",
            }
        )

    microbatch_count = (
        protocol["optimization"]["updates"]
        * protocol["optimization"]["accumulation_steps"]
        * protocol["systems"]["world_size"]
    )
    ppiref_required = microbatch_count * layout["ppiref_records_per_microbatch"]
    string_required = microbatch_count * layout["string_records_per_microbatch"]
    seed = layout["selection_seed"]
    ppiref_selected = sorted(
        ppiref_by_pair.values(),
        key=lambda row: selection_hash(seed, "ppiref50k", row["example_id"]),
    )[:ppiref_required]
    string_selected = sorted(
        string_candidates,
        key=lambda row: selection_hash(seed, "string", row["example_id"]),
    )[:string_required]
    if len(ppiref_selected) != ppiref_required or len(string_selected) != string_required:
        raise RuntimeError("Stage-A1 pilot source does not have enough unique records")

    rows: list[dict[str, Any]] = []
    for microbatch in range(microbatch_count):
        p_start = microbatch * layout["ppiref_records_per_microbatch"]
        s_start = microbatch * layout["string_records_per_microbatch"]
        batch = [
            *ppiref_selected[
                p_start : p_start + layout["ppiref_records_per_microbatch"]
            ],
            *string_selected[
                s_start : s_start + layout["string_records_per_microbatch"]
            ],
        ]
        for row in batch:
            published = dict(row)
            published["pilot_record_index"] = len(rows)
            rows.append(published)
    if len(rows) != layout["total_records"]:
        raise RuntimeError("Stage-A1 pilot manifest count differs")
    selected_ppiref_pairs = {
        row["full_pair_hash_sha256"] for row in rows if row["source"] == "ppiref50k"
    }
    selected_string_pairs = {
        row["full_pair_hash_sha256"]
        for row in rows
        if row["source"] == "string_v12_physical"
    }
    if selected_ppiref_pairs & selected_string_pairs:
        raise RuntimeError("Stage-A1 pilot retained a cross-source exact pair")

    _publish_zstd_create_only(arguments.output, rows)
    receipt = {
        "authorization": {
            "pilot_training": True,
            "production_training": False,
            "test_access": False,
        },
        "claim_boundary": protocol["claim_boundary"],
        "counts": {
            "contact_positive_pairs_selected": sum(
                len(row["contact_pairs"]) for row in rows
            ),
            "ppiref_input_interfaces": ppiref_input_count,
            "ppiref_short_full_endpoint_excluded": ppiref_short_full_endpoint_excluded,
            "ppiref_short_patch_endpoint_excluded": ppiref_short_patch_endpoint_excluded,
            "ppiref_selected_unique_exact_pairs": len(selected_ppiref_pairs),
            "records": len(rows),
            "retrieval_supervised_records": sum(
                bool(row["retrieval_supervision"]) for row in rows
            ),
            "string_cross_source_exact_pairs_removed": string_cross_source_removed,
            "string_selected_unique_exact_pairs": len(selected_string_pairs),
        },
        "cross_source_exact_pair_overlap_after_filter": 0,
        "format_version": "abpairfm_stage_a1_pilot_manifest_receipt_v1",
        "inputs": {
            "ppiref_manifest_sha256": protocol["sources"]["ppiref50k"][
                "manifest_sha256"
            ],
            "ppiref_receipt_sha256": protocol["sources"]["ppiref50k"][
                "receipt_sha256"
            ],
            "protocol_sha256": arguments.expected_protocol_sha256,
            "string_manifest_sha256": protocol["sources"]["string_segment"][
                "manifest_sha256"
            ],
            "string_receipt_sha256": protocol["sources"]["string_segment"][
                "receipt_sha256"
            ],
        },
        "output": {
            "path": str(arguments.output),
            "sha256": sha256_file(arguments.output),
            "size_bytes": arguments.output.stat().st_size,
        },
        "status": "PASS_STAGE_A1_PILOT_SOURCE_ADMISSION",
        "test_access": False,
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    with arguments.receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
