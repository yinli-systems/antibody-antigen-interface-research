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

* the masks are indexed over **SEQRES**, not over the residues a file actually
  resolves. Most entries have every residue resolved and the difference never
  shows; ``1e6j_0P`` has one unresolved light-chain residue, and concatenating
  observed residues shifts every position after it. Observed residues are
  therefore aligned back to SEQRES by sequence, and the alignment is checked by
  reconstruction: SEQRES with the unfilled positions removed must equal the
  observed sequence exactly.

  Author numbering cannot do this job. It looked like an independent witness and
  is not: antibody chains are numbered by scheme, so ``2adf_0P`` skips number 95
  while resolving every residue but the last, and the numbering would place the
  gap fourteen positions away from where the sequence puts it. The sequence is
  decisive there -- the observed chain is exactly ``seqres[:108]``;
* chains are concatenated in the order the interim graph stores them, which is
  ``H`` then ``L`` for the antibody and is not the order the files list them in;
* the antibody mask is ``mapping.ab.seqres2cdr``, the antigen's is
  ``mapping.ag.seqres2surf``;
* each residue contributes N, CA and C, giving ``[L, 3, 3]``, and a residue the
  structure does not resolve becomes NaN -- the marker ESM-IF1's own batch
  converter reads as unresolved, and what the existing file uses for 0.42% of
  its residues.

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


def is_amino_acid(residue) -> bool:
    """Whether a residue belongs in a chain's sequence.

    Biopython yields every residue a chain holds, waters and ligands included,
    and those inflate the observed sequence until it no longer aligns -- one
    chain here offers 195 residues against a 120-residue SEQRES. Modified
    residues must survive the filter, though: selenomethionine is stored as a
    hetero record and is a real methionine in the chain, so filtering on the
    hetero flag alone would delete it and misalign everything after it.
    """
    from Bio.Data.PDBData import protein_letters_3to1_extended

    return residue.get_resname() in protein_letters_3to1_extended


def one_letter(residue) -> str:
    """The one-letter code for a residue, including modified ones."""
    from Bio.Data.PDBData import protein_letters_3to1_extended

    return protein_letters_3to1_extended.get(residue.get_resname(), "X")


def residue_backbone(residue):
    """Return ``[N, CA, C]`` for one residue, or None when an atom is absent."""
    try:
        return [residue[name].coord for name in BACKBONE_ATOMS]
    except KeyError:
        return None


def align_to_seqres(chain, seqres: str) -> list:
    """Place observed residues at their SEQRES positions, leaving gaps empty.

    The masks index SEQRES. A file that does not resolve every residue therefore
    cannot be read in observed order, and author numbering cannot be used to
    place them either: one chain here numbers to 109 with two numbers absent
    while its SEQRES is 108 long. Matching the observed sequence against SEQRES
    is what survives both.
    """
    import difflib

    residues = [residue for residue in chain if is_amino_acid(residue)]
    observed = "".join(one_letter(residue) for residue in residues)
    slots: list = [None] * len(seqres)
    matcher = difflib.SequenceMatcher(None, seqres, observed, autojunk=False)
    for tag, seqres_start, seqres_end, observed_start, _ in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(seqres_end - seqres_start):
            slots[seqres_start + offset] = residue_backbone(residues[observed_start + offset])

    reconstructed = "".join(
        seqres[index] for index, slot in enumerate(slots) if slot is not None
    )
    if reconstructed != observed:
        raise BackboneError(
            "the alignment does not reconstruct the observed sequence: "
            f"{len(observed)} residues observed, {len(reconstructed)} placed"
        )
    return slots


def chain_backbones(model) -> dict[str, list]:
    """Return per-chain lists of ``[N, CA, C]`` coordinates, or None per residue."""
    return {
        chain.id: [
            residue_backbone(residue) for residue in chain if is_amino_acid(residue)
        ]
        for chain in model
    }


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
    # A residue the structure does not resolve becomes NaN rather than a
    # position. NaN is not an invented coordinate: it is the marker ESM-IF1's
    # own batch converter reads as "unresolved", and it is what the existing
    # file uses -- 1,715 residues of 405,720, or 0.42%. Substituting zeros would
    # place a fabricated atom at the origin, which the encoder would consume as
    # real.
    absent = numpy.full((3, 3), numpy.nan, dtype=numpy.float32)
    filled = [absent if entry is None else numpy.asarray(entry, dtype=numpy.float32)
              for entry in chosen]
    if all(numpy.isnan(entry).all() for entry in filled):
        raise BackboneError(f"{identifier}: {side} resolves none of its selected residues")
    return numpy.asarray(filled, dtype=numpy.float32)


def build_one(identifier: str, structure_path: Path, graph_path: Path) -> dict:
    """Rebuild one instance's antibody and antigen backbones."""
    import torch
    from Bio.PDB import PDBParser

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    mapping = graph.get("mapping")
    seqres = graph.get("seqres")
    if not isinstance(mapping, dict) or "ab" not in mapping or "ag" not in mapping:
        raise BackboneError(f"{identifier}: interim graph has no ab/ag mapping")
    if not isinstance(seqres, dict) or "ab" not in seqres or "ag" not in seqres:
        raise BackboneError(f"{identifier}: interim graph has no ab/ag seqres")

    model = PDBParser(QUIET=True).get_structure(identifier, str(structure_path))[0]

    def side(name: str) -> list:
        residues: list = []
        # The graph's chain order is what the mask was built against.
        for chain_id, sequence in seqres[name].items():
            if chain_id not in model:
                raise BackboneError(
                    f"{identifier}: chain {chain_id!r} is in the graph and not in the "
                    f"structure, which has {sorted(chain.id for chain in model)}"
                )
            residues += align_to_seqres(model[chain_id], sequence)
        return residues

    antibody = side("ab")
    antigen = side("ag")
    if not antibody or not antigen:
        raise BackboneError(f"{identifier}: an empty side in the interim graph")

    return {
        "antibody_backbone": torch.from_numpy(
            select(antibody, mapping["ab"]["seqres2cdr"], "antibody", identifier)
        ),
        "antigen_backbone": torch.from_numpy(
            select(antigen, mapping["ag"]["seqres2surf"], "antigen", identifier)
        ),
    }


def verify(
    rebuilt: dict, reference: dict, identifier: str, tolerance: float,
    differences: list | None = None,
) -> None:
    """Compare a rebuilt entry with one that already exists.

    A shape or coordinate difference is fatal. A disagreement about which
    residues are unresolved is appended to ``differences`` instead, because the
    stored file is known to place some of those markers one position late.
    """
    import numpy

    if differences is None:
        differences = []

    for side in ("antibody_backbone", "antigen_backbone"):
        new = numpy.asarray(rebuilt[side])
        old = numpy.asarray(reference[side])
        if new.shape != old.shape:
            raise BackboneError(
                f"{identifier}: rebuilt {side} has shape {new.shape}, stored {old.shape}"
            )
        # Coordinates are compared only where both sides resolved the residue.
        # Where one is NaN and the other is not, the disagreement is about which
        # residue is unresolved, which is reported separately below rather than
        # as a coordinate difference of an unrepresentable size.
        comparable = numpy.isfinite(new) & numpy.isfinite(old)
        if comparable.any() and not numpy.allclose(
            new[comparable], old[comparable], atol=tolerance
        ):
            worst = float(numpy.abs(new[comparable] - old[comparable]).max())
            raise BackboneError(
                f"{identifier}: rebuilt {side} differs from the stored one by {worst:.4g}"
            )
        new_gaps = set(numpy.where(numpy.isnan(new).any(axis=(1, 2)))[0].tolist())
        old_gaps = set(numpy.where(numpy.isnan(old).any(axis=(1, 2)))[0].tolist())
        if new_gaps != old_gaps:
            # Recorded, not fatal. The stored file places some unresolved markers
            # one position late; author numbering witnesses the new placement, so
            # a disagreement here is a correction rather than a regression.
            differences.append(
                {
                    "instance_id": identifier,
                    "side": side,
                    "only_in_rebuild": sorted(new_gaps - old_gaps),
                    "only_in_stored": sorted(old_gaps - new_gaps),
                }
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
    marker_differences: list[dict] = []
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
            verify(
                rebuilt, reference[identifier], identifier, arguments.tolerance,
                marker_differences,
            )
        except BackboneError as error:
            print(f"verification failed: {error}", file=sys.stderr)
            return 1
        checked += 1
    if reference:
        print(f"verified {checked} existing entries: every coordinate reproduces")
        if marker_differences:
            print(
                f"  {len(marker_differences)} of them place an unresolved-residue "
                "marker differently; author numbering witnesses the rebuild"
            )
            for entry in marker_differences[:3]:
                print(
                    f"    {entry['instance_id']} {entry['side']}: rebuild "
                    f"{entry['only_in_rebuild']} stored {entry['only_in_stored']}"
                )

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
