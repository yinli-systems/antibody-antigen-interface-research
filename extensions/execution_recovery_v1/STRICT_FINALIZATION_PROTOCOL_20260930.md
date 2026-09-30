# Strict post-inference AF3 finalization protocol — 2026-09-30

## Scope

The remaining 39 AF3 chains were already submitted once by the exclusive
controller as Slurm array `1641712` under source `79f6a23`. They are not
cancelled, resubmitted or rerun. This protocol adds a success-dependent,
no-inference strict audit and authority assembly using the semantic auditor
introduced at `93aa319`.

## Inputs

- Existing execution plan SHA-256:
  `397193cd9248d4157b836de3db6c1e3f729cc9665817b9af49117c5f59b880ea`.
- Existing prediction array: `1641712`, tasks 1–39, concurrency four.
- Task 0 prediction source: preserved `1641607_0` output; its original failure
  receipt remains unchanged.
- Strict auditor: `af3.py` with complete seed/sample, input, confidence,
  full-mmCIF, all-atom, chain/sequence and coordinate checks.

## Pre-output gate

Before an output directory is created, all 40 source tasks must have a valid
`COMPLETE.json`, no `FAILURE.json`, the exact source plan/scientific-plan hashes,
25 samples, unchanged prediction-tree hashes and the same selected maximum-score
sample under strict re-audit. The old execution code, all source inputs and large
runtime resources are independently reverified. A missing or changed task aborts
without a partial authority.

## Strict audit overlay

For every task the finalizer records:

- original terminal receipt path and SHA-256;
- strict auditor commit and source SHA-256;
- every prediction-file SHA-256;
- exact selected seed, sample and ranking score;
- raw and normalized mmCIF difference keys;
- top/sample all-atom and backbone fingerprints;
- full confidence-payload hashes and equality result;
- explicit `model_inference_performed=false` and label-access flags.

## Assembly

Only after all 40 strict audits pass, the finalizer writes backbone-only PDBs,
verifies the declared coordinate-rounding bound, proves the extension set is
exactly the 40 frozen missing chains, and closes predicted-chain coverage from
187/217 to 217/217 validation complexes. The base structure authority and live
SaProt/ProSST feature banks remain unchanged.

## Evidence boundary

This is structure-input authority completion, not structure-accuracy evaluation,
binding prediction, a blind result or model selection. No contact, test or OOD
labels are read. Scheduler success alone is not completion; only the terminal
strict authority `COMPLETE.json` is completion evidence.
