# V6-C linear-infeasibility diagnosis protocol

Status: frozen diagnostic protocol. This protocol does not authorize a split,
model training, model selection, outer-fold inference, or test access.

Date: 2026-07-30 CST

## Trigger and scope

V6-C may run only after the registered V6-B primary CP-SAT result, independent
CP-SAT result, and direct-HiGHS result all certify infeasibility of the same
content-addressed linear projection. A timeout, unknown status, nonzero gap,
identity mismatch, or solver disagreement stops V6-C before diagnosis.

Even when all three solvers agree, the conclusion is limited to the common
linear projection. It is not a proof that the complete Decimal/Fraction
score-free feasible region is infeasible. No result from V6-C may be called a
Gate 0 result.

## Frozen scientific inputs

The diagnostic rebuilds the same authority-train-only problem from 3,229
source rows, 1,830 observed antibody-antigen pairs, 1,383 antibodies, and 1,172
antigens. It may read only the registered allowlist, four-column canonical
manifest, four direct-edge ledgers, observed pair incidence, and independently
reconstructed score-free authority.

Contact labels, contact density, paratope or epitope size, affinity, assay
outcome, sequence or structure embeddings, predictions, losses, ranks, model
scores, model results, outer labels, and test bytes remain forbidden.

All V6-A and V6-B thresholds, graph semantics, retention definitions, and
quarantine accounting remain byte-identical. V6-C cannot weaken a threshold or
silently replace the split.

## Constraint-family provenance

The direct CP-SAT builder records one ASCII family identifier for every source
linear row, in exact row order. The family sequence, family counts, source
model binary, formulation payload, authority payload, and implementation bytes
are content addressed.

The diagnostic clones the source satisfaction model and adds one Boolean
activation literal per distinct family. Every source row is enforced only by
its recorded family literal. With all family literals assumed true, the
diagnostic model must have the same variables, source rows, and feasible set as
the registered linear projection.

For a family-subset replay, every included family is assumed true and every
excluded family is explicitly fixed false. Merely omitting an assumption is not
an exact subset and is forbidden.

## Deterministic core extraction and replay

CP-SAT runs with one search worker, the registered random seed, randomized
search disabled, and zero absolute and relative gap limits. Only `INFEASIBLE`
is an infeasibility proof. `OPTIMAL` or `FEASIBLE` disproves the tested core.
`UNKNOWN`, `MODEL_INVALID`, timeout, exception, or resource exhaustion is
non-authorizing.

The diagnostic procedure is:

1. solve with all constraint families active and require `INFEASIBLE`;
2. obtain a sufficient assumption core from CP-SAT;
3. reject unknown, duplicate, negative, or absent core literals;
4. rebuild and solve the exact returned family subset;
5. apply deterministic deletion minimization in canonical family order;
6. replay the final core once more; and
7. remove each final family independently and require a decided feasible
   result before claiming the core is irreducible.

A sufficient core may be reported when its exact replay is infeasible. It may
be called irreducible only when every final single-family deletion returns
`OPTIMAL` or `FEASIBLE`. No minimum-cardinality claim is allowed.

## Receipt and failure preservation

The create-only receipt binds:

- all three parent result files and receipt digests;
- the authority, manifest, edge-ledger, and source-byte digests;
- source and diagnostic CP model binaries;
- formulation and constraint-family sequence digests;
- solver version and locked parameters;
- every core literal, replay status, deletion attempt, and minimality check;
- wall time, resource telemetry, error evidence, and output-file digests; and
- explicit false values for split, training, selection, outer-evaluation, and
  test authorization.

The Markdown report is derived from the same payload. Successful and failed
attempts are immutable and pushed. A failure must not be erased or renamed as
infeasibility.

## Decision boundary after diagnosis

A valid family core is an explanation of the locked linear projection, not
permission to tune on evaluation data. Any proposed repair requires a new
numbered protocol that states one scientifically justified design change
before rebuilding the authority-train split. The old thresholds and all failed
receipts remain visible. Structural leakage, power, nested five-seed training,
outer refit, and one-shot test evaluation remain downstream and blocked until
a new split is frozen and independently replayed.
