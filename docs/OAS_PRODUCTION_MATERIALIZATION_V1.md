# OAS production materialization v1

**Status:** protocol frozen; execution is not yet authorized as a training corpus

## Purpose

The downloaded OAS corpus contains 6,373 plan-selected data units and
265,834,820 source rows. The independent download audit rehashed
120,714,440,652 compressed bytes. Those facts prove download integrity, but
they do not prove that the final 250M-plan corpus has passed row filtering,
global deduplication, or validation isolation.

This protocol closes that gap before any run is called production
pretraining. It is independent of SAbDab2 labels, predictions, losses, and
held-out metrics.

## Deterministic map stage

Thirty-two CPU-only map tasks consume disjoint modulo partitions of the
canonical `selected_data_units` list. Every task rechecks its plan and receipt
bindings and recomputes each compressed file SHA-256 before parsing it. It then
applies the frozen rational sampling rule and row-quality policy.

Accepted rows are routed by the first byte of the canonical sequence SHA-256
into 256 binary buckets. The fixed-width records preserve the full digest,
canonical sequence, authority order, split, chain type, stratum, study, and
bounded ANARCI warning flags. Failed tasks retain their partial directory and
cannot publish a completion receipt.

## Exact reduce stage

One reducer owns each digest bucket. Records are sorted by full sequence
digest, split priority, and authority order. Validation has unconditional
priority over training: if a digest occurs in validation, all matching
training records are excluded. Within a split, the minimum authority ordinal
is retained.

An equal digest with different sequence bytes is treated as a hash collision
and stops the run. A collision in the uint64 training sample key also stops the
run. Thresholds and split membership are never changed in response to a
failure.

## Publication and authorization

Reducers publish immutable fixed-width NumPy shards for train and validation.
The final manifest is create-only and binds every map and reduce receipt, all
counts, hashes, source versions, and the study vocabulary. A separate auditor
that imports neither mapper nor reducer must replay the complete bucket
inventory and validation-precedence rule.

Only that independent audit plus a one-GPU loader/checkpoint replay can mark
the corpus eligible for a separately registered pretraining run. This stage
does not authorize training, downstream evaluation, or a scientific claim.
