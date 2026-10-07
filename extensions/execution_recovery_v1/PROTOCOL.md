# Frozen-input execution recovery v1 — 30 September 2026

This is additive to `multimodel@28dffada`, not an edit to frozen training files.
The parent 40-chain completion plan remains byte-identical. Existing training
array 1640503, dependent analyzer 1640514, raw outputs and historical attempts
are untouched. No new data, loss, seed, layer or validation selection is made.

## Defects verified before engineering

1. AF3 task 1640914_0 failed 2:0 before inference. Its actual log says the legacy
   launcher expected `batches/batch-000/inputs`; the builder wrote
   `inputs/batch-000`. Source inspection confirms this contract mismatch.
2. The runner's default shared sandbox no longer exists. The original SIF it
   names does exist. Its embedded `version.py` reports **3.0.1**, not the 3.0.0
   parameter-directory label. Do not retroactively claim every historical
   structure used this exact software revision.
3. Completed training evaluation receipts use `mean_reciprocal_rank`, while
   the frozen complete-campaign analyzer indexes `mrr`. Modifying that frozen
   source would fail the still-running tasks' end-of-run source checks.
4. AF3 summary JSON rounds ranking scores to two decimals, while ranking CSV
   retains full precision. Use CSV for top selection; compare summary with the
   correctly rounded value, never rank rounded scores.

## Recovery input and resource contract

Forty exact parent-plan sequences, one chain each, no MSA/templates, five
predeclared seeds [42,100,18,37,79] and five samples per seed. Exact input JSON
bytes, sequence SHA/length, original metadata/source hashes, new executor hashes,
SIF and parameter hashes are recorded before launch. Copy inputs into new
independent task directories; never hardlink or edit original inputs.

Use the runner's original SIF directly (`AF3_USE_NODE_SANDBOX=0`), with FUSE
loaded. This avoids touching or populating shared node-local sandbox caches.
The model resource is the existing 3.0.0 af3.bin; actual container code is 3.0.1.
Large runtime files receive full hashes at freeze and final deep verification;
tasks check size/mtime as well as full small source/input hashes. This is not
mislabelled as full per-task runtime byte verification.

## Admission, submission and output gates

Use a fresh execution root. One exclusive intent per task is written BEFORE
sbatch. An ambiguous acknowledgement stays locked; no automatic resubmission.
An active same-plan allocation blocks another submission. A task directory
cannot be reused and automatic Slurm restarts are rejected. Old failures remain.

Task 0 is production work, retained on pass. Tasks 1–39 may be submitted only
after task 0's COMPLETE output audit and scheduler termination. Four single-GPU
jobs maximum, six CPUs/GPU, three-hour per-task ceiling (120 GPU-hours allocation
ceiling, not predicted cost). No user priority or unrelated job is changed.

A successful process is NOT enough: every task must have the exact unique
five-seed/five-sample grid, 25 complete one-chain models, exact sequence, finite
N/CA/C/O, ranking CSV/rounded-summary parity, and byte-identical top-level model
to a maximum-full-precision-score sample. No timestamp-suffixed accidental
second output directory is admitted. Input/source hashes are checked again.
All files receive output hashes; mismatch produces FAILURE, not COMPLETE.

## Evidence boundaries

AF3 ranking_score is used only to select structures; not binding specificity or
model accuracy. Chain-only predictions are not a co-folder comparison. No contact,
closed-test or rejected-OOD labels are read. These engineering tests do not count
as new biological evidence; the full 40-chain authority is incomplete until all
40 complete receipts exist and are validated.

## Amendment: top-level AF3 serialization semantics (2026-09-30)

Official AF3 output semantics use `ranking_scores.csv` to select the root
highest-ranked prediction. Root and sample mmCIF files are not assumed to be
byte-identical because AF3 can stamp their separate serialization times. The
corrected audit compares full mmCIF semantics and all atom coordinates exactly,
normalising only the timestamp in `_ma_model_list.model_group_name`. See
`AF3_SEMANTIC_AUDIT_FIX_20260930.md`. This amendment does not change sequences,
seeds, sample count, ranking, coordinates, confidence checks or scientific data.
