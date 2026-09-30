# RePaRank same-grid reproduction protocol v1

Frozen 2026-09-30 before any RePaRank score was computed on this project's
validation contacts. Official source is `NaturalAntibody/reparank` commit
`6910d950c2e483289973be41387c8913b859766a`, license CC BY-NC 4.0.

## Scientific question

On exactly the existing SAbDab2 development-validation complexes, residue axes,
contact labels and complete candidate pair grids, how do the official RePaRank
sequence and sequence+structure checkpoints rank true antibody-antigen contacts?

This is a checkpoint transfer diagnostic. It is not a clean independent
benchmark because the public checkpoint's training accession list is not bound
in the official repository and overlap with this cohort is unknown.

## Official authority

- Sequence checkpoint, official Drive file id
  `1MhuWKCwQkzPxPDQDrR5l8hZvPMcpAmhI`, 157,835,707 bytes, SHA-256
  `b9bbb418aeeb2c96ffa00ccbce36ad169a183c81d57200b503bdb8c1435a4c14`.
- Sequence+structure checkpoint, official Drive file id
  `1ey8KROvrCEj0PwaROky5yeBMPWvqj5es`, 171,471,477 bytes, SHA-256
  `ce95a698e866246b647197f5b12e11d86c2392d462d2cb2a9ed63c57cf51ae0d`.
- The files must be parsed successfully as PyTorch checkpoints; model
  hyperparameters, state-key coverage and source commit are frozen before
  contact scores are opened.

The official `contact_detection.py` defines contacts at 4.5 Å, fits an F1-optimal
latent-distance threshold on its training loader, and then reports mean
precision/recall. That procedure is retained only as a secondary diagnostic.
It is not used to replace the primary threshold-free endpoint below.

## Inputs and firewall

Evaluation uses only the 217 complexes in the existing `validation` role of
`double_remote_split.json`. Train and permanently closed test contact labels are
not loaded. Existing contact NPZ files define antibody and antigen residue axes,
the complete candidate grid and 4.5 Å binary labels.

Sequences come from the content-bound canonical manifest and are split into
heavy, light and antigen exactly as stored. For single-domain antibodies the
light chain is omitted rather than fabricated as padding.

The structure checkpoint may use only `structure_units_v2`: independently
predicted chain-level N/CA/C/O coordinates selected without contact labels.
Heavy, light and antigen are separately positioned in their own predicted-chain
frames. Experimental bound antibody-antigen pose is forbidden. If checkpoint
hyperparameters require an additional feature tensor not present in this
frozen authority, the structure arm fails closed rather than silently becoming
a sequence arm.

## Score and endpoints

The official model returns a 512-dimensional complex embedding per residue.
The continuous pair score is exactly negative Euclidean distance between the
selected antibody and antigen residue embeddings. Higher is better. No learned
head, recalibration, temperature, pooling or task-specific fine-tuning is added.

Primary endpoint: macro mean-per-complex non-interpolated pair AP on the exact
candidate grid. Secondary endpoints: pooled AP/AUROC; per-complex AUROC, MRR,
Recall@1 and precision at the true-contact count; epitope/paratope marginals
derived only from the same pair score; and equal-component uncertainty.

A validation-derived threshold may be reported for descriptive precision,
recall, F1 and MCC, but it is fitted and scored through an explicit inner split
of validation authority or by cross-fitting. A threshold fitted to all 217
complexes may not be called held-out performance. Threshold metrics never
replace AP.

## Statistics and comparisons

The official checkpoint is deterministic and is not reported as a multi-seed
model. Repeated inference is an engineering reproducibility check, not seed
uncertainty. Complex effects are aggregated within the 68 existing authority
components and bootstrapped by component. Comparisons to FrameContact, marginal
outer product or structural-PLM arms require identical complex and candidate
sets and name every preprocessing/input difference.

## Execution gates

1. Official source commit, license, checkpoint URL, size and SHA-256 pass.
2. Checkpoint audit extracts hyperparameters and loads every model state key with
   no missing or unexpected model parameter.
3. Sequence engineering canary on three validation complexes repeats bitwise or
   within 1e-6 and preserves model state.
4. Structure canary additionally proves exact sequence/PDB residue alignment,
   N/CA/C/O completeness and predicted-chain authority.
5. Full validation scoring runs in new exclusive directories. Every result
   records checkpoint/source/input hashes and per-complex rows.
6. No result is promoted if checkpoint training-overlap remains unaudited; it is
   reported as an exposed pretrained diagnostic.

## Separate future training reproduction

A true three-seed same-split RePaRank retraining campaign requires continuous
pair-distance supervision. The current contact cache contains only binary 4.5 Å
labels. Before training, a separate authority must materialise continuous
heavy-atom or explicitly selected distance targets from development-train bound
structures, bind atom policy and residue mapping, and audit leakage. Replacing
the ranking loss with binary BCE would be a different model and is forbidden.

No blind, SOTA, first-place or production claim is authorised by this protocol.
