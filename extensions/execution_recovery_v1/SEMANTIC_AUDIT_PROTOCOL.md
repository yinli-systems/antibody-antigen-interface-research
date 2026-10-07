# Strict semantic AF3 post-audit — 2026-09-30

## Recovered state, not a new prediction campaign

At the initial read, the controller had already admitted task 0 without inference
under execution plan `397193cd9248d4157b836de3db6c1e3f729cc9665817b9af49117c5f59b880ea`
and submitted array `1641712` for tasks 1–39. No submission is repeated here.
The original failed prediction is retained under `af3_attempt2/runs/task_000`.
Its job `1641607_0` completed the 25 predictions and failed its byte-equality audit.

An uncommitted strict-audit patch was present in the local recovery worktree.
It was copied into a separate worktree, recorded in PATCH_ADOPTION.json, reviewed,
and further tested. Neither that working directory nor a running source snapshot
is modified. The new auditor is a separately versioned post-audit; it does not
rewrite old execution-plan hashes or old COMPLETE/FAILURE receipts.

## Primary-source contract

Read 2026-09-30:

- https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md
  describes root outputs as the highest-ranked prediction selected by the CSV.
- https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/model/post_processing.py
  computes a wall-clock timestamp for each mmCIF serialization.
- https://github.com/google-deepmind/alphafold3/blob/main/run_alphafold.py
  supplies the output-writing and ranking implementation context.

This supports allowing a serialization timestamp difference, not ignoring
coordinate, model-version, atom, sequence, confidence or ranking differences.

## Admissible equivalence

All parsed mmCIF categories and values must match exactly. The only normalised
substring is a valid calendar timestamp in `_ma_model_list.model_group_name`.
The AlphaFold identifier, model version and remaining prefix/suffix remain exact.
Duplicate CIF items are rejected. No numerical tolerance, rigid alignment,
coordinate rounding, side-chain dropping or score reselection is performed.

The independent all-atom digest binds residue IDs, residue names, atom names,
alternate locations, coordinates, occupancy, B factors and chemical elements.
All 25 samples require the exact seed/sample grid, one model and one chain,
requested sequence, complete finite N/CA/C/O and consistent atom topology.
Every confidence JSON is parsed, finite and shape/range checked. Its atom pLDDTs
must equal CIF B factors; token IDs and chain IDs must agree. Top-level full and
summary confidence payloads must exactly match the maximum-score sample.
The original CSV's full-precision ranking scores select the maximum; rounded
summary scores are checked only at their documented stored precision.

## Reproducibility and preservation

`semantic_audit.py` validates a pinned execution plan by importing the original
executor from that plan's source root, then invokes the new strict auditor on
existing predictions. It binds both source versions separately, hashes the full
task tree before/after, checks any original COMPLETE output manifest and selected
seed/sample, and optionally rehashes large runtime files. Recovered copies must
match the failed-audit source's prediction files byte for byte. New reports must
be written outside the immutable task run directory, with exclusive creation.

The actual timestamp regression fixture preserves the top and seed-42/sample-2
CIFs from job 1641607_0, the full selected confidence, summary, CSV and original
file hashes. Unit mutations operate only in temporary directories.

No model execution, job submission, label scoring, test/OOD read, layer selection
or live-job/source modification is performed by this post-audit. Predictions can
be file-consistent yet scientifically inaccurate; passing is not a fold-accuracy
or binding-quality claim. Full structure assembly and downstream scoring remain
blocked until all 40 outputs receive the required complete audits.
