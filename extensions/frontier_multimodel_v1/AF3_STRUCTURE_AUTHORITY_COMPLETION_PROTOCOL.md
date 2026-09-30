# AF3 validation-chain structure-authority completion protocol v1

Frozen 2026-09-30 before any new AF3 prediction was launched.

## Motivation and frozen cohort

The RePaRank sequence+structure full validation canary passed, but the first
217-case run failed closed when the existing `structure_units_v2` authority
lacked a heavy-chain sequence. A label-free coverage audit over the frozen 217
validation complexes found:

- 187 complexes already complete;
- 30 incomplete complexes;
- 40 unique missing chain sequences: 18 heavy, 3 light and 19 antigen;
- 7,985 total unique missing residues; lengths 54–573.

No contact labels, cross-chain distances or model scores were used to choose
these sequences. The completion set is exactly the missing sequence set induced
by the validation role, canonical heavy/light/antigen sequences and the existing
structure manifest.

## Predictor parity

Use the same AlphaFold 3 authority as the existing manifest:

- AF3 model version 3.0.0;
- the existing model parameters, databases, Singularity image and
  `run_5090_singularity.sh` runner on ParaCloud;
- one protein chain `A` per input;
- no paired/unpaired MSA and no templates;
- model seeds `[42, 100, 18, 37, 79]`;
- five diffusion samples per seed under the existing runner;
- one missing sequence per batch/task, so a failure is attributable and does
  not suppress unrelated chains.

A different predictor, recycled homolog structure, zero coordinates or bound
complex coordinates is forbidden. The longest sequence is not silently routed
to a different method.

## Selection and validation

For each sequence, select the AF3 top-level `<name>_model.cif`, which the runner
constructs from the highest ranking score over seed/sample outputs. Independently
verify that its score equals the maximum in `ranking_scores.csv`, with source
path as deterministic tie-break only if scores are exactly tied.

The selected CIF must parse to exactly one protein chain with the exact requested
amino-acid sequence. Every residue must contain finite N/CA/C/O coordinates.
The materializer writes a chain-local PDB, verifies sequence and backbone atoms
again, and records input JSON, ranking CSV, selected CIF and output PDB SHA-256.

## Execution gates

1. Freeze canonical/split/base-manifest hashes, missing-sequence list, AF3 runner
   hash, task map and 40 input JSON hashes before submission.
2. Task 0 is the shortest missing sequence and serves as an actual authority
   canary. It uses the same production input and output contract and is retained
   as task 0 if it passes; it is not rerun in the formal array.
3. The canary must pass scheduler exit, ranking-score selection, exact sequence
   parsing and complete N/CA/C/O validation.
4. Submit tasks 1–39 with bounded concurrency. Failures remain; retries require
   a separate attempt/receipt.
5. Materialisation requires all 40 tasks and produces an extension manifest plus
   a combined manifest. No incomplete subset is silently accepted.
6. Re-run the 217-case RePaRank structure canary/full plan against the combined
   manifest; no closed test or rejected OOD data is opened.

## Evidence boundary

This predicts chain-local structures to complete an input authority. It is not a
co-folding result, binding prediction, contact metric, blind evaluation or proof
that AF3 structures are experimentally correct. No SOTA, first-place or
production claim is authorised.
