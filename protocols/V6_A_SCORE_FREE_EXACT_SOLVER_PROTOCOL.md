# V6-A score-free exact-solver amendment

Status: frozen engineering amendment. This amendment authorizes implementation and
score-free solver diagnostics only. It does not authorize model training, model
selection, outer-fold inference, authority-validation access, or frozen-test access.

Date: 2026-07-27 CST

## Scope and precedence

This amendment extends
`TRAIN_ONLY_FOUR_VIEW_DUAL_AXIS_NEIGHBORHOOD_EXCLUSION_PROTOCOL_V0.md` and the v5
full-authority component-gate contract. It does not reinterpret or erase any prior
attempt, timeout, audit failure, or inspection. Where the older draft is silent about
the exact solver implementation for nonlinear score-free gates, this amendment takes
precedence. All older stop rules and claim boundaries remain active.

The current v5 component gate and score-free statistics v1 module are engineering
precursors. They have not produced a real assignment or training authorization.

## Frozen statistical semantics

The standardized mean difference remains

```text
abs(mean(X) - mean(Y)) / sqrt((sample_var(X) + sample_var(Y)) / 2)
```

with sample variance denominators and the fail-closed zero-variance behavior already
implemented by the score-free statistics oracle. The locked distribution thresholds
are:

| Scope | Maximum SMD | Maximum TV | Maximum KS |
|---|---:|---:|---:|
| Outer | 0.35 | 0.20 | 0.20 |
| Pooled entity-disjoint outer validation | 0.25 | 0.15 | 0.20 |
| Inner | 0.40 | 0.25 | 0.25 |

These values are project-specific preregistered thresholds. They are not literature
defaults and may not be relaxed after a feasibility result is observed.

Contact density, paratope or epitope size, affinity, assay outcome, embeddings,
predictions, losses, ranks, model scores, and model results remain forbidden split
inputs. The only permitted density is observed antibody--antigen pair-incidence
density.

## Complete feasible-region authority

The complete feasible-region authority is the conjunction of:

1. a CP-SAT integer formulation independently reconstructed from authenticated
   authority-train inputs;
2. deterministic exact separation by the Decimal/Fraction score-free oracle for gates
   that cannot be represented without changing their frozen arithmetic; and
3. an independent raw-source auditor that does not import the builder, oracle, or
   generated solver matrix.

Each rejected candidate produces a deterministic local-membership conflict cut. A cut
may exclude only assignments that reproduce the exact fold, axis, role, and membership
state that the oracle proved invalid. Every cut binds the failed gate payload, support
bits, source hashes, iteration, and canonical SHA-256.

Only CP-SAT `OPTIMAL` or `INFEASIBLE` is terminal evidence. `FEASIBLE`, `UNKNOWN`, a
timeout, an incomplete separation sequence, an invalid independent audit, or any
cross-implementation disagreement is non-authorizing.

## Direct-HiGHS evidence boundary

The direct `highspy` backend certifies the shared linear projection and its stated
linear objectives. It never certifies the complete nonlinear feasible region by
itself. Every HiGHS receipt must carry
`complete_score_free_feasible_region_authorized=false`.

The locked HiGHS options are one thread, parallel mode off, a fixed random seed, zero
relative and absolute MIP gaps, and explicitly recorded primal and integrality
tolerances. The backend must preserve the original model, presolved model, options,
complete log, solution, `HighsInfo`, variable dictionary, matrix digest, and MIP
logging trajectory. It must not use a user-solution callback.

Solver wheels, interpreter, operating system, architecture, and dependency hashes are
frozen only after an isolated compatibility probe. A system Python installation is not
modified.

## Execution order and stop rules

The mandatory order is:

1. raw-source score-free authority derivation;
2. independent raw-source audit and adversarial conformance;
3. complete 3 outer plus 9 dependent inner formulation;
4. direct-HiGHS projection backend and independent CP-SAT formulation;
5. one canonical digest-selected 10 percent dry run of the frozen formulation;
6. retention-first staged objectives with exact optimum fixing;
7. provisional discovery, external digest freeze, numbered pinned replay, and
   independent replay audit;
8. post-freeze three-dimensional interface and lineage audit;
9. frozen attainable-precision and power simulation;
10. D0, R1, and R2 nested training only after every preceding authorization gate; and
11. multi-node model scaling only after prospective partner-dependence gates pass.

No incomplete feasible region may be optimized. No weighted surrogate may replace the
locked staged objectives. Every stage must be proven optimal and fixed by an exact
integer equality before the next stage. Canonical bit fixing tests zero before one in
the frozen digest order.

The 10 percent dry run is selected before graph inspection using the smallest canonical
entity digests. It is a single registered scaling experiment, not a formulation or seed
search. It records build and solver memory, sparse dimensions, presolve dimensions,
first incumbent, primal and dual bounds, gap, node rate, separation iterations, stage
costs, and projected sequential-fixing cost. An unscalable result stops this version and
requires a separately preregistered decomposition protocol.

After assignment freeze, a structural leakage hit from either preregistered structural
audit path stops the version. The split is not repaired after opening interface labels.
The power gate and all existing data-firewall rules remain mandatory before training.

## Publication and claim boundary

Every protocol, source, test, model, log, failure, candidate, cut, receipt, and decision
is append-only and content addressed. Human-readable reports are derived from the same
machine payload. Code comments are English and source files are ASCII unless an
upstream byte-exact fixture requires otherwise.

This amendment does not promise feasibility, a positive partner-dependence result,
useful scaling, state of the art, or world-leading performance. Such claims require
completed identical-benchmark evidence and the separately frozen evaluation contract.
