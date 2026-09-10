#!/usr/bin/env python3
"""Are consecutive rows consecutive residues? Ask the geometry, not the metadata.

The backbone indexes residues that have a CA atom, so a residue unresolved in
the structure is dropped and leaves no marker. Counting NaN rows cannot find
that. But a peptide bond fixes the CA-CA distance of genuinely adjacent residues
at about 3.8 A, so a dropped residue shows up as a jump.

This measures the distance between consecutive rows and counts how often it is
too large to be a peptide bond.
"""
import csv
import numpy as np
import torch

PEPTIDE = 3.80
TOLERANCE = 0.7          # 3.1 - 4.5 A counts as adjacent; cis/trans and strain fit inside
import argparse

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--manifest", required=True)
parser.add_argument("--backbones", required=True)
arguments = parser.parse_args()

backbones = torch.load(arguments.backbones, map_location="cpu", weights_only=False)
rows = list(csv.DictReader(open(arguments.manifest, newline="")))

def audit(side, key, junction_of):
    steps = breaks = 0
    junction_breaks = 0
    distances = []
    for row in rows:
        entry = backbones.get(row["INSTANCE"])
        if entry is None or entry.get(key) is None:
            continue
        coords = np.asarray(entry[key], dtype=np.float64)
        alpha = coords[:, 1, :]
        if alpha.shape[0] < 2:
            continue
        step = np.linalg.norm(alpha[1:] - alpha[:-1], axis=1)
        ok = np.isfinite(step)
        step = step[ok]
        junction = junction_of(row)
        broken = np.abs(step - PEPTIDE) > TOLERANCE
        steps += step.size
        breaks += int(broken.sum())
        if junction is not None and 0 < junction <= step.size:
            # the VH/VL boundary is a break by construction, not a gap
            if broken[junction - 1]:
                junction_breaks += 1
        distances.append(step)
    alld = np.concatenate(distances) if distances else np.array([0.0])
    print(f"{side}:")
    print(f"  consecutive steps            {steps}")
    print(f"  outside {PEPTIDE}+-{TOLERANCE} A          {breaks}  ({100*breaks/steps:.3f}%)")
    print(f"  of which the VH/VL junction  {junction_breaks}")
    print(f"  median step                  {np.median(alld):.3f} A")
    print(f"  p1 / p99                     {np.quantile(alld,0.01):.3f} / "
          f"{np.quantile(alld,0.99):.3f} A")
    return breaks, steps, junction_breaks

ab_break, ab_steps, ab_junc = audit(
    "antibody", "antibody_backbone",
    lambda r: len((r.get("vh_seq") or "").strip()) or None)
print()
ag_break, ag_steps, _ = audit("antigen", "antigen_backbone", lambda r: None)
print()
real_ab = ab_break - ab_junc
print("chain breaks that are not the VH/VL junction -- these are structural gaps,")
print("and a shift across one moves a row by more residues than it moves rows:")
print(f"  antibody {real_ab} of {ab_steps} steps ({100*real_ab/ab_steps:.3f}%)")
print(f"  antigen  {ag_break} of {ag_steps} steps ({100*ag_break/ag_steps:.3f}%)")
