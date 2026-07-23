# Partner-specific retrieval protocol v1.6 — execution hardening

Status: locked before the single final refit and before any selected checkpoint is
scored on either corrected authority-validation boundary.

Date: 2026-07-23 CST

Parent protocol: `PARTNER_RETRIEVAL_PROTOCOL_V1_5_EXECUTION_LOCK.md`, SHA-256
`43638fdd17b9ce3e4cac8b83e42c16261ecbf79c2e628d0b494e5dcbe02f9fc9`.

## Purpose

Protocol v1.5 fixed the scientific model, optimization, evaluation, control, and
statistical contracts. This amendment closes execution and provenance bypasses found by
independent pre-submission review. It changes no architecture, training example,
hyperparameter, endpoint, threshold, or promotion gate.

## Immutable data inputs

- Prepared instance table SHA-256:
  `a9131d52323bae40572b7de81c5e41353ccc39a73309b7d7043485350facb009`.
- Canonical sequence manifest SHA-256:
  `548bcfab75923ec6621c1a3fef1aa6673c034f64a9bdc752e7ee15b41b4a4e63`.
- Selected AntiBERTy training-universe embedding content-manifest SHA-256:
  `9eb10c64fd05897162548a81871841724d9471ba5ed1c800b3d10a2c9229a45e`.
- Shared ESM-2 antibody-control training-universe embedding content-manifest SHA-256:
  `03a9e0e4b4acf8b3b76cca651d65c4f278f5ac5a8985d3b12fe001da7809bf36`.
- Antigen ESM-2 training-universe embedding content-manifest SHA-256:
  `d4157971ba2e977d1125e896fe13147371292495fa04a47f62b32ebe8f4ac97e`.

The physical instance table and canonical manifest must be hashed before row
materialization. The exact training-universe embedding artifact inventory must be
hashed before optimization and match the reviewed fold-screen audit. Self-recording a
new input hash after loading is not sufficient.

## Exclusive final refit

The canonical refit output directory has one process-lifetime, nonblocking exclusive
lock. A concurrent second refit must fail before dataset or embedding access. A process
failure releases the operating-system lock so the authenticated state can be resumed;
the persistent lock file is not itself a completed-run marker.

Resume state must have an epoch within the locked range, contiguous history ending at
that epoch, coherent nonnegative global steps, finite model state, compatible
optimizer/scheduler state, identical submission and configuration provenance, and a
compatible deterministic runtime. Job identifiers may change; scientific runtime
semantics may not.

## One checkpoint for both mandatory boundaries

The sole acceptable checkpoint path is `<refit_output_dir>/final.pt`. Before either
holdout is materialized, evaluation creates or authenticates one immutable central
checkpoint-suite claim keyed by the submission-manifest SHA-256. The claim binds:

- submission-manifest SHA-256;
- exact final-checkpoint path and SHA-256;
- central registry path;
- both declared boundary names and split SHA-256 values; and
- the literal requirement that both boundary evaluations use this checkpoint.

The second boundary evaluation must reject any different checkpoint bytes, even if
their configuration or model tensors are semantically similar. Re-serializing a
checkpoint cannot create a new evaluation attempt. Each boundary retains its separate
O_EXCL one-shot ledger, and each completed ledger binds the shared suite claim.

Evaluation receives the reviewed checkpoint SHA-256 explicitly, reads the checkpoint
bytes once, hashes those exact bytes, and loads the model from the same in-memory byte
buffer. The final-checkpoint object has an exact versioned top-level schema. Rehashing a
mutable path before and after model loading is not accepted as evidence that the loaded
model and registered digest are identical.

After both boundary ledgers are complete, a separate authenticated suite-finalization
step verifies both declared evaluation JSON and score artifacts and creates one central
completion receipt. Confirmation statistics and control comparison require this
receipt; a single completed boundary is not a complete P0 result.

## Direct evaluation evidence

Every final-refit evaluation JSON directly exposes a strict
`locked_submission_evidence` object containing the physical submission manifest,
protocol v1.5, independent boundary verification v2, fixed evaluation contract, and
selected boundary. The score-artifact contract repeats their canonical hashes. The
statistics and control-comparison programs physically rehash these references and the
checkpoint-suite claim before accepting a result.

All two-boundary evaluation JSON and score-artifact paths are globally pairwise
distinct and cannot overlap the refit checkpoint directory or ledger registry. A path
cannot change semantic role between boundaries.

## Claim boundary

These changes harden reproducibility and prevent duplicate or selectively substituted
execution. They do not make the reused authority-validation subsets independent, do not
remove protein-language-model pretraining exposure, and do not authorize access to the
frozen SAbDab2 test.
