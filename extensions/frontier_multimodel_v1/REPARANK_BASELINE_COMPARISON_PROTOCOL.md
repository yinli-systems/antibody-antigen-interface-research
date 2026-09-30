# RePaRank same-grid baseline comparison protocol v1

Frozen 2026-09-30 after the official sequence checkpoint's 217-complex score
was opened, but before the official sequence+structure 217-complex score and
before any paired baseline difference was computed. Accordingly, sequence-arm
comparisons are explicitly post-hoc; structure-arm comparisons are predeclared.

## Question

On exactly the 217 development-validation complexes and the exact complete
antibody-by-antigen residue-pair candidate grid, how do the deterministic
official RePaRank checkpoint scores compare with existing interaction-first
models trained on this project's development split?

This comparison is not a clean architecture ranking. The official RePaRank
checkpoint's training accessions are unavailable in the source repository, so
training overlap is unknown. Model sizes, pretraining, input information and
training budgets differ. The purpose is a same-grid numerical diagnostic.

## Arms

Official checkpoints:

- `reparank_sequence`: deterministic official sequence checkpoint.
- `reparank_sequence_structure`: deterministic official checkpoint with
  independently predicted chain-local N/CA/C/O coordinates; no bound interchain
  pose.

Existing baseline arms, primary seeds 1–10:

- `esm2`: interaction-first model with ESM-2 base inputs.
- `esm_if1_only`: same downstream model with aligned ESM-IF1 features.
- `proteinmpnn_aligned`: same downstream model with aligned ProteinMPNN features.
- `antiberty_only`: same downstream model with antibody-side AntiBERTy features.

ESM-2 seeds 11–20 are a separately labelled stability sensitivity. They may not
be pooled into the primary comparison after seeing the result.

## Exact candidate and label parity

For every baseline seed and every complex, compare the canonical sorted tuples
`(instance_id, antigen_residue_index, antibody_residue_index, label)` against
the RePaRank raw NPZ. Pair count, positive count, residue indices and every label
must match. Any difference fails the comparison. Scores are never used in this
parity digest.

## Primary estimand

For a checkpoint and baseline arm, compute non-interpolated AP separately for
each complex. For each baseline training seed, compute the paired complex effect

`RePaRank AP - baseline AP`.

Within each seed, average complex effects inside each of the 68 existing split
authority components. The primary point estimate equally weights components and
baseline training seeds. Its descriptive 95% interval resamples baseline seeds
and whole authority components. Also report:

- complex-weighted mean difference;
- component-only interval conditional on the fixed baseline seeds;
- fraction of components with positive/negative mean effect;
- each baseline seed's macro AP and paired difference;
- unpaired raw means, explicitly secondary.

RePaRank is deterministic here; baseline seeds quantify downstream optimisation
variation, not RePaRank seed variation or biological population uncertainty.
Intervals are pointwise and exploratory, not multiplicity-adjusted.

## Secondary diagnostics

Report paired per-complex AUROC, MRR and precision at the true-contact count only
where both files contain enough information to reproduce the same definition.
No secondary endpoint can replace AP. Do not compare partner-retrieval MRR from
the interaction-first model with residue-pair MRR from RePaRank; they are
different tasks.

## Evidence boundary

- Same 217 development-validation complexes; no test or rejected OOD labels.
- Sequence comparison is post-hoc because its aggregate AP was already known.
- Structure comparison is predeclared before its full result is opened.
- Official checkpoint training overlap remains unaudited.
- Existing baselines were selected/trained previously on this validation
  authority; this is not independent confirmation.
- No SOTA, superiority, first-place, blind or production claim is authorised.
