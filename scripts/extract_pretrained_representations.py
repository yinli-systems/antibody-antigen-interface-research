#!/usr/bin/env python3
"""Extract frozen per-residue representations from external pretrained encoders.

Reads the canonical sequence manifest, encodes each distinct chain sequence with
one backend, and writes validated artifacts plus a receipt.

    python3 scripts/extract_pretrained_representations.py \
        --manifest canonical_manifest.csv --backend antiberty \
        --chains vh vl --output-root /path/antiberty --receipt receipt.json

The manifest carries train and test rows. Splits other than ``train`` are
refused unless named explicitly, and the receipt records exactly which splits
were parsed, mirroring the ``data_firewall`` accounting the split artifacts use.
Frozen unsupervised features are not labels, but parsing a sealed sequence is
still an event this project counts, so it is counted here.

The manifest's own train/test division is **not** the double-cold split. Rows
the manifest calls ``train`` include instances the split manifest assigns to
validation, and instances the split manifest assigns to train are found under
the manifest's ``test``. Selecting by manifest split therefore cannot express
"the development instances and nothing else": it either misses some of them or
parses the frozen test split to reach them. ``--instance-split-manifest`` selects
by instance instead, which can.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.pretrained_representations import (  # noqa: E402
    RepresentationError,
    RepresentationUnit,
    extract,
    write_receipt,
)
from interaction_first.representation_backends import BACKENDS, build_backend  # noqa: E402

# ``ab`` is the concatenated VH+VL chain, which is the axis the stored backbone
# file uses and the axis the models already concatenate on.
CHAIN_COLUMNS = {
    "vh": ("vh_seq",),
    "vl": ("vl_seq",),
    "ag": ("ag_seq",),
    "ab": ("vh_seq", "vl_seq"),
}
# The backbone store is keyed by instance and holds one entry per side, so only
# these two chains can carry a structure.
BACKBONE_FIELD = {"ab": "antibody_backbone", "ag": "antigen_backbone"}


# The split whose sequences this project does not read. Naming it in an instance
# selection is refused rather than counted: every other split here is a
# development split, and a frozen one reached by accident is not recoverable.
SEALED_SPLIT = "test"


def load_split_instances(
    split_manifest: Path, split_names: list[str]
) -> tuple[set[str], dict[str, int]]:
    """Return the instance ids the split manifest assigns to the named splits."""
    document = json.loads(split_manifest.read_text(encoding="utf-8"))
    splits = document.get("splits")
    if not isinstance(splits, Mapping):
        raise RepresentationError("split manifest has no 'splits' mapping")
    sealed = [name for name in split_names if name == SEALED_SPLIT]
    if sealed:
        raise RepresentationError(
            f"refusing to select instances from the {SEALED_SPLIT!r} split; "
            "this project does not read it"
        )
    unknown = [name for name in split_names if name not in splits]
    if unknown:
        raise RepresentationError(
            f"split manifest has no split(s) {unknown}; it has {sorted(splits)}"
        )
    selected: set[str] = set()
    counts: dict[str, int] = {}
    for name in split_names:
        members = {str(instance) for instance in splits[name]}
        counts[name] = len(members)
        selected |= members
    return selected, counts


def load_units(
    manifest: Path,
    chains: list[str],
    splits: set[str],
    backbones: Mapping[str, Any] | None,
    instances: set[str] | None = None,
) -> tuple[list[RepresentationUnit], dict[str, int], dict[str, int]]:
    """Read the manifest and build one unit per requested chain of each row.

    ``instances`` restricts the rows to an explicit set of instance ids. It is
    applied in addition to ``splits``, so a caller that wants the development
    instances wherever the manifest happens to file them passes every manifest
    split and the instance ids it actually wants.
    """
    if backbones is not None:
        unsupported = [name for name in chains if name not in BACKBONE_FIELD]
        if unsupported:
            raise RepresentationError(
                f"no backbone is stored for chain(s) {unsupported}; "
                f"structure encoders accept {sorted(BACKBONE_FIELD)}"
            )

    units: list[RepresentationUnit] = []
    parsed: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = {column for name in chains for column in CHAIN_COLUMNS[name]}
        for column in ("INSTANCE", "split", *sorted(columns)):
            if column not in (reader.fieldnames or []):
                raise RepresentationError(f"manifest has no {column!r} column")
        for row in reader:
            split = (row["split"] or "").strip()
            if split not in splits:
                continue
            if instances is not None and row["INSTANCE"] not in instances:
                dropped["instance_not_selected"] += 1
                continue
            parsed[split] += 1
            for name in chains:
                sequence = "".join(
                    (row[column] or "").strip() for column in CHAIN_COLUMNS[name]
                )
                if not sequence:
                    dropped["empty_sequence"] += 1
                    continue
                backbone = None
                if backbones is not None:
                    entry = backbones.get(row["INSTANCE"])
                    backbone = (
                        entry.get(BACKBONE_FIELD[name]) if isinstance(entry, Mapping) else None
                    )
                    if backbone is None:
                        dropped["missing_backbone"] += 1
                        continue
                    if len(backbone) != len(sequence):
                        # The backend would reject this; drop it here so the
                        # receipt separates a data gap from an encoder failure.
                        dropped["backbone_length_mismatch"] += 1
                        continue
                units.append(RepresentationUnit(sequence=sequence, backbone=backbone))
    return units, dict(parsed), dict(dropped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", default=None, help="fail closed on a digest mismatch")
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--chains", nargs="+", required=True, choices=sorted(CHAIN_COLUMNS))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="manifest splits to parse; anything beyond train must be named explicitly",
    )
    parser.add_argument(
        "--instance-split-manifest",
        type=Path,
        default=None,
        help="select rows by instance id from this split manifest instead of by manifest split alone",
    )
    parser.add_argument(
        "--instance-splits",
        nargs="+",
        default=None,
        help=f"splits of --instance-split-manifest to select; {SEALED_SPLIT!r} is refused",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbones", type=Path, default=None,
                        help="torch file of per-instance backbones, for structure encoders")
    parser.add_argument("--proteinmpnn-checkout", type=Path, default=None,
                        help="directory holding protein_mpnn_utils.py and "
                             "vanilla_model_weights/; there is no packaged distribution "
                             "whose provenance this project assumes")
    parser.add_argument("--limit", type=int, default=None, help="encode at most N units")
    arguments = parser.parse_args(argv)

    digest = hashlib.sha256(arguments.manifest.read_bytes()).hexdigest()
    if arguments.manifest_sha256 and digest != arguments.manifest_sha256:
        print(
            f"manifest digest {digest} does not match the expected "
            f"{arguments.manifest_sha256}",
            file=sys.stderr,
        )
        return 2

    backbones = None
    if arguments.backbones is not None:
        import torch

        backbones = torch.load(arguments.backbones, map_location="cpu", weights_only=False)

    backend = build_backend(arguments.backend, device=arguments.device)
    if backend.requires_backbone and backbones is None:
        print(f"{arguments.backend} requires --backbones", file=sys.stderr)
        return 2
    if arguments.backend == "proteinmpnn":
        if arguments.proteinmpnn_checkout is None:
            print("proteinmpnn requires --proteinmpnn-checkout", file=sys.stderr)
            return 2
        backend.checkout = arguments.proteinmpnn_checkout

    instances = None
    instance_counts: dict[str, int] = {}
    instance_digest = None
    if (arguments.instance_split_manifest is None) != (arguments.instance_splits is None):
        print(
            "--instance-split-manifest and --instance-splits are used together",
            file=sys.stderr,
        )
        return 2
    if arguments.instance_split_manifest is not None:
        instance_digest = hashlib.sha256(
            arguments.instance_split_manifest.read_bytes()
        ).hexdigest()
        instances, instance_counts = load_split_instances(
            arguments.instance_split_manifest, arguments.instance_splits
        )

    units, parsed, dropped = load_units(
        arguments.manifest, arguments.chains, set(arguments.splits), backbones, instances
    )
    if arguments.limit is not None:
        units = units[: arguments.limit]
    if not units:
        print("no units matched the requested chains and splits", file=sys.stderr)
        return 2

    receipt = extract(units, backend, arguments.output_root)
    receipt.update(
        {
            "manifest": str(arguments.manifest),
            "manifest_sha256": digest,
            "chains": list(arguments.chains),
            "device": arguments.device,
            "splits_parsed": parsed,
            "dropped_units": dropped,
            "sealed_splits_parsed": sorted(set(arguments.splits) - {"train"}),
            "instance_selection": (
                None
                if instances is None
                else {
                    "split_manifest": str(arguments.instance_split_manifest),
                    "split_manifest_sha256": instance_digest,
                    "splits": list(arguments.instance_splits),
                    "instances_per_split": instance_counts,
                    "instances_selected": len(instances),
                }
            ),
            "claim_boundary": (
                "Frozen unsupervised per-residue features. No label, contact, prediction, "
                "or score was read. Extraction authorizes no training, selection, or "
                "evaluation."
            ),
        }
    )
    write_receipt(arguments.receipt, receipt)

    print(
        f"{receipt['backend']}: wrote {receipt['written']}, "
        f"skipped {receipt['skipped_existing']}, failed {len(receipt['failed'])} "
        f"of {receipt['requested_unique_sequences']} unique sequences"
    )
    for failure in receipt["failed"][:5]:
        print(f"  {failure['sequence_sha256'][:12]}  {failure['error']}", file=sys.stderr)
    return 0 if receipt["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
