# V6-D two-CP consensus core protocol

Status: frozen non-authorizing diagnostic protocol. This protocol does not
authorize a split, model training, model selection, outer-fold inference, or
test access.

Date: 2026-07-30 CST

## Motivation and trigger

The registered V6-B primary CP-SAT formulation returned `PROVEN_INFEASIBLE`
and the independently reconstructed CP-SAT formulation returned
`INDEPENDENT_DIRECT_CP_INFEASIBLE_AGREEMENT` for the same content-addressed
linear projection. The registered full direct-HiGHS attempt reached its time
limit without a proof or incumbent. Therefore V6-C is not triggered: its
three-solver parent condition remains false.

V6-D is a new numbered protocol. It does not reinterpret the HiGHS timeout as
infeasibility and does not bypass V6-C. It uses the two independent CP-SAT
proofs only to diagnose an exact constraint-family subset, then asks isolated
direct HiGHS to solve that smaller subset. Every historical result remains
immutable.

## Frozen inputs and forbidden data

V6-D rebuilds the same authority-train-only problem from 3,229 source rows,
1,830 observed antibody-antigen pairs, 1,383 antibodies, and 1,172 antigens.
The only scientific inputs are the registered allowlist, four-column canonical
manifest, four direct-edge ledgers, observed pair incidence, and independently
reconstructed score-free authority.

Contact labels, contact density, paratope or epitope size, affinity, assay
outcome, embeddings, predictions, losses, ranks, model scores, model results,
outer labels, and test bytes are forbidden. All V6-A and V6-B thresholds,
retention definitions, graph semantics, and quarantine accounting remain
unchanged.

## Stage A: exact CP-SAT family diagnosis

The controller authenticates both CP-SAT parent receipts, their common V6-B
registration, raw authority bytes, runtime lock, and current source bytes. It
then independently rebuilds the complete linear projection and requires the
same dimensions and projection digest recorded by both parents.

One Boolean activation literal is added per recorded constraint family. With
all families active, CP-SAT must again return `INFEASIBLE`. A returned
assumption core is replayed exactly. Excluded families are fixed false rather
than merely omitted. Deterministic deletion minimization proceeds in canonical
family order, followed by a final replay and every single-family deletion.

Only `INFEASIBLE` proves a tested subset infeasible. `OPTIMAL` or `FEASIBLE`
proves that the tested deletion is feasible. `UNKNOWN`, timeout,
`MODEL_INVALID`, exception, or resource exhaustion is non-authorizing. An
irreducible-family claim requires every final single deletion to be decided
feasible. No minimum-cardinality claim is allowed.

## Stage B: exact core projection and direct HiGHS

The final CP family set selects rows from the original direct CP-SAT proto by
their recorded family identifiers. Variables, Boolean domains, integer
coefficients, row bounds, and source row indices are preserved. The exporter
records:

- the full model and family-sequence digests;
- selected family names and family-count digest;
- ordered source row indices and their digest;
- the reduced CP proto digest and exact dimensions;
- the solver-neutral sparse bundle digest; and
- a row-by-row subset-containment audit.

An isolated Python runtime then loads only the solver-neutral bundle. Direct
HiGHS runs with one thread, parallelism disabled, fixed seed, zero MIP gaps,
locked feasibility and integrality tolerances, and a finite registered time
limit. Original and presolved MPS, options, log, solution, `HighsInfo`, and MIP
telemetry are create-only artifacts.

If HiGHS returns certified infeasibility for the exact CP-infeasible row
subset, subset containment proves only that the common full linear projection
is infeasible. A timeout or any other unproved status remains a nonproof. A
feasible or optimal HiGHS result is a solver disagreement that stops the
protocol and is preserved.

## Receipt and decision boundary

Stage A and Stage B publish separate immutable receipts. A final V6-D consensus
receipt may report full-linear infeasibility only when:

1. both registered CP-SAT parent proofs authenticate;
2. the complete projection identity replays exactly;
3. the exact CP family subset replays `INFEASIBLE`;
4. row-subset containment is byte-audited; and
5. isolated direct HiGHS certifies the same subset infeasible.

Even that result is limited to the common linear projection. It does not prove
the complete Decimal/Fraction score-free feasible region infeasible, does not
authorize Gate 0, and cannot start training. Any repair requires a later
numbered protocol with one explicit scientific change. The frozen outer test
remains untouched.

