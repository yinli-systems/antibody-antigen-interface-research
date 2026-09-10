# sabdab2 and AsEP overlap by PDB entry

Status: a measurement, not a result. It constrains what may be claimed from
either cohort about the other. Nothing here required reading any model
prediction.

Date: 2026-09-10

Machine-readable: [`overlap.json`](overlap.json).

## Why it had to be measured before any comparison

This project has no number comparable to published work. AsEP is the benchmark
WALLE and EpiFormer quote, so the obvious move is to evaluate on it. The obvious
move has a trap: both cohorts are assembled from the Protein Data Bank, their
identifier schemes differ (`pdb_00001j1p_H_L` against `1j1p_0P`), and sequence
digests cannot detect the overlap either, because the cohorts select different
residues — AsEP stores CDR residues for the antibody and surface residues for
the antigen, sabdab2 stores the full variable region. The same structure hashes
differently in each.

The PDB accession is what survives both schemes.

## The result

Share of the AsEP split that sabdab2 has already trained on:

| sabdab2 split → AsEP split | shared | of | share |
|---|---:|---:|---:|
| train → train | 361 | 1321 | 27.3% |
| **train → validation** | **57** | **168** | **33.9%** |
| **train → test** | **39** | **168** | **23.2%** |
| validation → test | 3 | 168 | 1.8% |
| test → test | 5 | 168 | 3.0% |

**Twenty-three percent of AsEP's test structures are in sabdab2's training
split.** A third of its validation structures are too.

## What this forbids, and what it does not

**Forbidden.** Reporting an AsEP number from a model trained on sabdab2. That
number would be measured on a test set of which nearly a quarter was seen in
training. This is the natural-sounding claim — "our model reaches X on AsEP" —
and it is the one the overlap invalidates.

**Forbidden.** Merging the cohorts for training without first removing the AsEP
test accessions from the merged training set. AsEP brings 1,129 PDB entries that
sabdab2 does not have, a 44% increase on sabdab2's 2,548, which is the most
attractive source of additional data available. Taking it without excluding the
benchmark's test entries spends the only external comparison this project has.

**Not forbidden, and unaffected.** Training on AsEP's own train split and
evaluating on AsEP's own validation or test split. AsEP's published split is
internally consistent, and its 3.0% test-to-test overlap with sabdab2 is
irrelevant to it. That is the configuration the benchmark runs use.

## Limits

1. **Accession granularity.** Two entries of the same structure count as shared
   regardless of which chains, which resolution, or which preparation. This
   overstates contamination where a shared accession contributed different
   complexes to the two cohorts, and understates nothing.
2. It says nothing about sequence-level or structural similarity between
   *different* accessions, which is the leakage the double-cold split exists to
   control and which this measurement does not address.
