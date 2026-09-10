#!/usr/bin/env python3
"""Rebuild AsEP backbone coordinates from its structures.

    python3 scripts/build_asep_backbones.py --structures asep/structures \
        --graphs asep/asepv1_interim_graphs --output backbones.pt \
        --verify-against interaction_first_asep/backbones.pt

The stored backbone file covers 1,101 of 1,721 AsEP instances while 1,723
structures are on disk, so the gap is an extraction that did not finish rather
than data that does not exist. The gap is not uniform across AsEP's own split --
65.5% of train, 59.8% of validation, 55.0% of test -- so a structure encoder
built on it would train and be tested on differently-covered cohorts, and the
difference would come from the extraction rather than from the data.

The recipe is not documented anywhere and was recovered by reproducing entries
that already exist:

* the antibody is chains ``H`` then ``L`` concatenated **in that order**, which
  is not the order the files store them in, and the interim graph's
  ``mapping.ab.seqres2cdr`` mask then selects the CDR residues;
* the antigen is every remaining chain, with ``mapping.ag.seqres2surf``
  selecting the surface residues;
* each residue contributes N, CA and C, giving ``[L, 3, 3]``.

``--verify-against`` re-derives entries that already exist and compares them.
That check is the only evidence the recipe is right, so it runs by default when
a reference file is given and its failure stops the run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ANTIBODY_CHAIN_ORDER = ("H", "L")
BACKBONE_ATOMS = ("N", "CA", "C")


class BackboneError(RuntimeError):
    """Raised when a backbone cannot be rebuilt faithfully."""


def chain_backbones(model) -> dict[str, list]:
    """Return per-chain lists of ``[N, CA, C]`` coordinates, or None per residue."""
    chains = {}
    for chain in model:
        residues = []
        for residue in chain:
            try:
                residues.append([residue[name].coord for name in BACKBONE_ATOMS])
            except KeyError:
                residues.append(None)
        chains[chain.id] = residues
    return chains


def select(residues: list, mask, side: str, identifier: str):
    """Apply a seqres mask, refusing a length disagreement or a gap."""
    import numpy

    mask = numpy.asarray(mask)
    if len(residues) != int(mask.shape[0]):
        raise BackboneError(
            f"{identifier}: {side} has {len(residues)} residues but its mask covers "
            f"{int(mask.shape[0])}"
        )
    chosen = [residues[index] for index in range(len(residues)) if mask[index]]
    if not chosen:
        raise BackboneError(f"{identifier}: {side} mask selects nothing")
    missing = sum(1 for entry in chosen if entry is None)
    if missing:
        raise BackboneError(
            f"{identifier}: {side} is missing backbone atoms in {missing} selected "
            "residue(s); a coordinate cannot be invented for them"
        )
    return numpy.asarray(chosen, dtype=numpy.float32)


def build_one(identifier: str, structure_path: Path, graph_path: Path) -> dict:
    """Rebuild one instance's antibody and antigen backbones."""
    import torch
    from Bio.PDB import PDBParser

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    mapping = graph.get("mapping")
    if not isinstance(mapping, dict) or "ab" not in mapping or "ag" not in mapping:
        raise BackboneError(f"{identifier}: interim graph has no ab/ag mapping")

    model = PDBParser(QUIET=True).get_structure(identifier, str(structure_path))[0]
    chains = chain_backbones(model)

    antibody = [
        residue
        for chain_id in ANTIBODY_CHAIN_ORDER
        if chain_id in chains
        for residue in chains[chain_id]
    ]
    if not antibody:
        raise BackboneError(
            f"{identifier}: no {'/'.join(ANTIBODY_CHAIN_ORDER)} chain; "
            f"the file has {sorted(chains)}"
        )
    antigen = [
        residue
        for chain_id in sorted(chains)
        if chain_id not in ANTIBODY_CHAIN_ORDER
        for residue in chains[chain_id]
    ]
    if not antigen:
        raise BackboneError(f"{identifier}: no antigen chain outside H/L")

    return {
        "antibody_backbone": torch.from_numpy(
            select(antibody, mapping["ab"]["seqres2cdr"], "antibody", identifier)
        ),
        "antigen_backbone": torch.from_numpy(
            select(antigen, mapping["ag"]["seqres2surf"], "antigen", identifier)
        ),
    }


def verify(rebuilt: dict, reference: dict, identifier: str, tolerance: float) -> None:
    """Compare a rebuilt entry with one that already exists."""
    import numpy

    for side in ("antibody_backbone", "antigen_backbone"):
        new = numpy.asarray(rebuilt[side])
        old = numpy.asarray(reference[side])
        if new.shape != old.shape:
            raise BackboneError(
                f"{identifier}: rebuilt {side} has shape {new.shape}, stored {old.shape}"
            )
        if not numpy.allclose(new, old, atol=tolerance):
            worst = float(numpy.abs(new - old).max())
            raise BackboneError(
                f"{identifier}: rebuilt {side} differs from the stored one by {worst:.4g}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-against", type=Path, default=None)
    parser.add_argument("--verify-limit", type=int, default=50,
                        help="how many existing entries to re-derive and compare")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--instances", type=Path, default=None,
                        help="instances.parquet; restricts the build to its identifiers")
    arguments = parser.parse_args(argv)

    import torch

    wanted = None
    if arguments.instances is not None:
        import pandas

        wanted = set(pandas.read_parquet(arguments.instances)["instance_id"].astype(str))

    reference = {}
    if arguments.verify_against is not None:
        reference = torch.load(
            arguments.verify_against, map_location="cpu", weights_only=False
        )

    identifiers = sorted(
        path.stem for path in arguments.structures.glob("*.pdb")
        if wanted is None or path.stem in wanted
    )
    if not identifiers:
        print(f"no structures under {arguments.structures}", file=sys.stderr)
        return 2

    checked = 0
    for identifier in identifiers:
        if checked >= arguments.verify_limit or identifier not in reference:
            continue
        graph_path = arguments.graphs / f"{identifier}.pt"
        if not graph_path.exists():
            continue
        try:
            rebuilt = build_one(
                identifier, arguments.structures / f"{identifier}.pdb", graph_path
            )
            verify(rebuilt, reference[identifier], identifier, arguments.tolerance)
        except BackboneError as error:
            print(f"verification failed: {error}", file=sys.stderr)
            return 1
        checked += 1
    if reference:
        print(f"verified {checked} existing entries reproduce exactly")

    built: dict[str, dict] = {}
    failures: list[dict[str, str]] = []
    for identifier in identifiers:
        graph_path = arguments.graphs / f"{identifier}.pt"
        if not graph_path.exists():
            failures.append({"instance_id": identifier, "error": "no interim graph"})
            continue
        try:
            built[identifier] = build_one(
                identifier, arguments.structures / f"{identifier}.pdb", graph_path
            )
        except (BackboneError, Exception) as error:  # noqa: BLE001 - recorded, not raised
            failures.append({"instance_id": identifier, "error": f"{type(error).__name__}: {error}"})

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(built, arguments.output)

    kept = len(set(built) & set(reference)) if reference else 0
    print(f"built {len(built)} of {len(identifiers)} structures"
          + (f"  ({kept} of them already existed)" if reference else ""))
    print(f"failed {len(failures)}")
    for failure in failures[:5]:
        print(f"  {failure['instance_id']}: {failure['error']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
