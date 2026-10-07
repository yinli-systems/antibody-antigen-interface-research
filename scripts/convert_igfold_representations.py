#!/usr/bin/env python3
"""Convert AsEP's stored IgFold embeddings into this project's artifact format.

    python3 scripts/convert_igfold_representations.py --graphs asep/asepv1_interim_graphs \
        --instances interaction_first_asep/instances.parquet --output-root igfold \
        --receipt igfold_receipt.json

The attribution evidence so far rests on one structural encoder. A second one
from a different model family, showing the same pattern, is what separates
"ESM-IF1 happens to carry useful information" from "residue-level correspondence
is how structural representations transfer at all".

IgFold is that second encoder and it is already computed: AsEP's interim graphs
carry a 512-dimensional per-residue embedding for every antibody, for all 1,721
instances. Nothing is downloaded and nothing is re-run.

Two details make the conversion non-trivial.

The embedding is indexed over **SEQRES**, 224 residues for a typical antibody,
while the model consumes the CDR residues the cohort actually models, about 60.
The interim graph's own ``mapping.ab.seqres2cdr`` selects them, and the selected
count is checked against ``ab_length`` for every instance rather than assumed.

The artifact key must be the digest the model's store is keyed by,
``ab_sequence_sha256``, which is ``sha256(ab_seq)`` for this cohort. A digest
computed any other way produces artifacts nothing will ever look up, and the
composite store would silently zero-fill instead of failing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.pretrained_representations import (  # noqa: E402
    RepresentationError,
    RepresentationUnit,
    build_artifact,
    write_artifact,
    write_receipt,
)


def select_cdr(embedding, mask, identifier: str, expected: int):
    """Restrict a SEQRES-indexed embedding to the residues the cohort models."""
    import numpy
    import torch

    mask = numpy.asarray(mask)
    if int(embedding.shape[0]) != int(mask.shape[0]):
        raise RepresentationError(
            f"{identifier}: embedding covers {int(embedding.shape[0])} residues and the "
            f"mask covers {int(mask.shape[0])}"
        )
    indices = torch.as_tensor(numpy.flatnonzero(mask), dtype=torch.long)
    selected = torch.as_tensor(embedding).index_select(0, indices).to(torch.float32)
    if int(selected.shape[0]) != expected:
        raise RepresentationError(
            f"{identifier}: mask selects {int(selected.shape[0])} residues but the "
            f"instance table records {expected}"
        )
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="restrict to the instances this split manifest names")
    arguments = parser.parse_args(argv)

    import pandas
    import torch

    frame = pandas.read_parquet(arguments.instances).set_index("instance_id")
    wanted = set(frame.index.astype(str))
    if arguments.split_manifest is not None:
        payload = json.loads(arguments.split_manifest.read_text(encoding="utf-8"))
        named = {str(i) for members in payload["splits"].values() for i in members}
        wanted &= named

    written = 0
    seen_keys: dict[str, str] = {}
    failures: list[dict[str, str]] = []
    dimensions: set[int] = set()
    for identifier in sorted(wanted):
        path = arguments.graphs / f"{identifier}.pt"
        if not path.exists():
            failures.append({"instance_id": identifier, "error": "no interim graph"})
            continue
        try:
            graph = torch.load(path, map_location="cpu", weights_only=False)
            embedding = graph["embedding"]["ab"]["igfold"]
            mask = graph["mapping"]["ab"]["seqres2cdr"]
            row = frame.loc[identifier]
            selected = select_cdr(embedding, mask, identifier, int(row["ab_length"]))
            key = str(row["ab_sequence_sha256"])
            dimensions.add(int(selected.shape[1]))
            if key in seen_keys:
                # Distinct instances share antibody sequences; one artifact serves
                # them all, and writing it twice would be wasted work, not an error.
                continue
            artifact = build_artifact(
                RepresentationUnit(sequence=str(row["ab_seq"])),
                selected,
                model_name="igfold",
                model_revision="asep-interim-graphs",
            )
            if artifact["sequence_sha256"] != key:
                raise RepresentationError(
                    f"{identifier}: artifact key {artifact['sequence_sha256'][:12]} does "
                    f"not match ab_sequence_sha256 {key[:12]}"
                )
            write_artifact(arguments.output_root, artifact)
            seen_keys[key] = identifier
            written += 1
        except Exception as error:  # noqa: BLE001 - recorded, not raised
            failures.append({"instance_id": identifier, "error": f"{type(error).__name__}: {error}"})

    if len(dimensions) > 1:
        print(f"embeddings disagree on width: {sorted(dimensions)}", file=sys.stderr)
        return 2

    receipt = {
        "format_version": "pretrained_representation_receipt_v1",
        "backend": "igfold",
        "revision": "asep-interim-graphs",
        "chains": ["ab"],
        "representation_dim": dimensions.pop() if dimensions else None,
        "instances_considered": len(wanted),
        "unique_sequences_written": written,
        "failed": failures,
        "complete": not failures,
        "claim_boundary": (
            "Frozen per-residue features copied from AsEP's own interim graphs and "
            "restricted to the modelled CDR residues. No label, contact, prediction "
            "or score was read."
        ),
    }
    write_receipt(arguments.receipt, receipt)
    print(f"igfold: {written} unique sequences from {len(wanted)} instances, "
          f"{len(failures)} failed")
    for failure in failures[:5]:
        print(f"  {failure['instance_id']}: {failure['error']}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
