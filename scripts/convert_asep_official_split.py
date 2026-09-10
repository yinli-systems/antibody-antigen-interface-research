#!/usr/bin/env python3
"""Convert AsEP's own split into a manifest this pipeline can read.

    python3 scripts/convert_asep_official_split.py \
        --split-dict asep/split/split_dict_corrected.pt \
        --identifier-list asep/split/asepv1-AbDb-IDs-corrected.txt \
        --instances interaction_first_asep/instances.parquet \
        --split-name epitope_group --output asep_epitope_group_split.json

Nothing in this project is currently comparable to published work. AsEP is the
benchmark those numbers are quoted on -- WALLE reports MCC 0.077 on the
``epitope_group`` split -- so reproducing that split exactly is what makes a
comparison possible at all.

The official split stores **integer indices**, not identifiers: entries are
``tensor(0)``, ``tensor(1)`` and so on, indexing the canonical AbDb identifier
list in file order. A conversion that guessed at identifiers, or that silently
tolerated an index it could not resolve, would produce a split that looks like
AsEP's and is not, which is worse than having no comparison. Every index is
therefore resolved through the identifier list and checked against the instances
actually present, and any gap is refused rather than dropped.

``val`` is renamed to ``validation`` because that is what the training script
requires; the membership is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FORMAT_VERSION = "sabdab2_interaction_first_v1"
SPLIT_RENAMES = {"val": "validation"}
REQUIRED_SPLITS = ("train", "validation", "test")


class AsepSplitError(RuntimeError):
    """Raised when the official split cannot be reproduced exactly."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_identifiers(path: Path) -> list[str]:
    """Read the canonical identifier list, in file order."""
    identifiers = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    identifiers = [value for value in identifiers if value]
    if len(set(identifiers)) != len(identifiers):
        raise AsepSplitError("identifier list contains duplicates")
    return identifiers


def resolve(index_values, identifiers: list[str], where: str) -> list[str]:
    """Map the official integer indices onto identifiers, refusing any gap."""
    resolved = []
    for value in index_values:
        try:
            position = int(value)
        except (TypeError, ValueError) as error:
            raise AsepSplitError(f"{where}: {value!r} is not an index") from error
        if not 0 <= position < len(identifiers):
            raise AsepSplitError(
                f"{where}: index {position} is outside the identifier list "
                f"of {len(identifiers)} entries"
            )
        resolved.append(identifiers[position])
    return resolved


def build_manifest(
    split_dict, identifiers: list[str], split_name: str, known: set[str] | None
) -> dict:
    if split_name not in split_dict:
        raise AsepSplitError(
            f"{split_name!r} is not in the split dictionary; it has {sorted(split_dict)}"
        )
    groups = split_dict[split_name]
    splits: dict[str, list[str]] = {}
    for name, values in groups.items():
        splits[SPLIT_RENAMES.get(name, name)] = resolve(values, identifiers, f"{split_name}/{name}")

    missing = [name for name in REQUIRED_SPLITS if not splits.get(name)]
    if missing:
        raise AsepSplitError(f"split {split_name!r} has no {missing}")

    seen: dict[str, str] = {}
    for name in REQUIRED_SPLITS:
        for identifier in splits[name]:
            if identifier in seen:
                raise AsepSplitError(
                    f"{identifier} is in both {seen[identifier]!r} and {name!r}"
                )
            seen[identifier] = name

    if known is not None:
        absent = sorted(set(seen) - known)
        if absent:
            raise AsepSplitError(
                f"{len(absent)} split members have no prepared instance, "
                f"first few: {absent[:5]}"
            )

    return {
        "format_version": FORMAT_VERSION,
        "stage": "convert_asep_official_split",
        "intended_use": "external_benchmark_comparison",
        "splits": {name: splits[name] for name in REQUIRED_SPLITS},
        # AsEP ships no per-instance group labels, so the cluster bootstrap over
        # components has nothing to resample and is left out rather than faked
        # from the split assignment, which would resample three clusters.
        "component_id_by_instance": {},
        "limitations": [
            "This is AsEP's own split, reproduced from its published index lists. "
            "It is the benchmark's split, not a split built by this project.",
            "The test split here is the benchmark's test set. Reading it is a "
            "benchmark measurement, not development, and belongs at the end of a "
            "development cycle rather than inside one.",
            "AsEP publishes no per-instance group labels, so no cluster bootstrap "
            "is available on this split.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split-dict", type=Path, required=True)
    parser.add_argument("--identifier-list", type=Path, required=True)
    parser.add_argument("--instances", type=Path, default=None,
                        help="instances.parquet; every split member must appear in it")
    parser.add_argument("--split-name", default="epitope_group")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    import torch

    identifiers = load_identifiers(arguments.identifier_list)
    split_dict = torch.load(arguments.split_dict, map_location="cpu", weights_only=False)

    known = None
    if arguments.instances is not None:
        import pandas

        known = set(pandas.read_parquet(arguments.instances)["instance_id"].astype(str))

    try:
        manifest = build_manifest(split_dict, identifiers, arguments.split_name, known)
    except AsepSplitError as error:
        print(f"{error}", file=sys.stderr)
        return 2

    manifest["source"] = {
        "split_dict": str(arguments.split_dict),
        "split_dict_sha256": digest(arguments.split_dict),
        "identifier_list": str(arguments.identifier_list),
        "identifier_list_sha256": digest(arguments.identifier_list),
        "identifier_count": len(identifiers),
        "split_name": arguments.split_name,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")

    counts = {name: len(manifest["splits"][name]) for name in REQUIRED_SPLITS}
    print(f"{arguments.split_name}: " + "  ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
