#!/usr/bin/env python3
"""Audit exact, reversed-chain, and absent PPIRef50K archive members."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

from build_ppiref50k_interfaces_v2 import archive_inventory, read_split, sha256_file


def classify(
    split: list[tuple[str, str, str]], members: dict[tuple[str, str, str], str]
) -> tuple[dict[str, int], list[dict[str, str]], list[dict[str, str]], list[str]]:
    exact = 0
    folded: dict[tuple[str, str, str], list[tuple[tuple[str, str, str], str]]] = {}
    for key, member in members.items():
        folded.setdefault((key[0], key[1].casefold(), key[2].casefold()), []).append((key, member))
    aliases: list[dict[str, str]] = []
    reverse_only: list[dict[str, str]] = []
    absent: list[str] = []
    for pdb_id, chain1, chain2 in split:
        key = (pdb_id, chain1, chain2)
        identifier = f"{pdb_id}_{chain1}_{chain2}"
        if key in members:
            exact += 1
            continue
        matches = folded.get((pdb_id, chain1.casefold(), chain2.casefold()), [])
        if len(matches) == 1:
            archive_key, member = matches[0]
            aliases.append(
                {
                    "archive_interface_id": "_".join(archive_key),
                    "archive_member": member,
                    "requested_interface_id": identifier,
                }
            )
            continue
        reverse = (pdb_id, chain2, chain1)
        if reverse in members:
            reverse_only.append(
                {
                    "archive_member": members[reverse],
                    "requested_interface_id": identifier,
                    "reversed_archive_interface_id": f"{pdb_id}_{chain2}_{chain1}",
                }
            )
        else:
            absent.append(identifier)
    return {
        "absent_interfaces": len(absent),
        "archive_pdb_members": len(members),
        "casefold_same_order_interfaces": len(aliases),
        "exact_order_interfaces": exact,
        "reverse_order_only_interfaces": len(reverse_only),
        "split_interfaces": len(split),
    }, aliases, reverse_only, absent


def publish_create_only(path: Path, value: object) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
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
        raise FileExistsError(f"Refusing to overwrite PPIRef archive audit: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    archive_path = arguments.archive.resolve()
    split_path = arguments.split.resolve()
    if sha256_file(archive_path) != arguments.archive_sha256:
        raise ValueError("PPIRef archive SHA-256 differs")
    if sha256_file(split_path) != arguments.split_sha256:
        raise ValueError("PPIRef split SHA-256 differs")
    split = read_split(split_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive_inventory(archive)
    counts, aliases, reverse_only, absent = classify(split, members)
    if sum(counts[key] for key in ("exact_order_interfaces", "casefold_same_order_interfaces", "reverse_order_only_interfaces", "absent_interfaces")) != counts["split_interfaces"]:
        raise AssertionError("PPIRef archive coverage does not partition the split")
    receipt = {
        "authorization": {"unique_same_order_casefold_recovery": counts["absent_interfaces"] == 0 and counts["reverse_order_only_interfaces"] == 0, "stage_a_training": False},
        "claim_boundary": "PPIRef50K split-to-archive filename coverage only; no structure content, normalization, decontamination, or training authorization.",
        "counts": counts,
        "format_version": "abpairfm_ppiref50k_archive_coverage_v2",
        "inputs": {"archive_sha256": arguments.archive_sha256, "split_sha256": arguments.split_sha256},
        "reverse_order_only": reverse_only,
        "same_order_casefold_aliases": aliases,
        "truly_absent": absent,
        "status": "AUDITED_NOT_NORMALIZED_NOT_AUTHORIZED",
        "test_access": False,
    }
    publish_create_only(arguments.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
