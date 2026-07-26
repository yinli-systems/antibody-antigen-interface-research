# V6-B full-authority logic-based decomposition protocol

Status: frozen engineering protocol. This protocol authorizes implementation and
score-free solver diagnostics only. It does not authorize model training, model
selection, outer-fold inference, authority-validation access, or frozen-test access.

Date: 2026-07-27 CST

## Reason for the new protocol

V6-A stopped when its single registered ten-percent vertex dry run selected 139
antibodies and 118 antigens. Three disjoint outer validation folds with the unchanged
minimum of 60 endpoints per axis require at least 180 distinct entities on each axis.
The receipt therefore proved only that the registered sampled universe cannot satisfy
the static integer projection. It did not inspect pair incidence, invoke a solver, or
prove the full 1830-pair authority universe infeasible.

V6-B does not rerun, enlarge, or reinterpret that sample. It replaces the monolithic
execution strategy with a separately numbered logic-based decomposition over the full
authenticated authority. Every statistical definition, threshold, forbidden input,
stop rule, and claim boundary from V6-A remains unchanged.

## Research basis and evidence boundary

Logic-based Benders decomposition permits a master problem to learn logically valid
consequences of failed subordinate problems. A cut is evidence only when its support
and inference are independently reproducible; a heuristic exclusion is not a proof.
This protocol follows the inference-dual framing introduced by Hooker and Ottosson
(Mathematical Programming 96, 33--60, 2003;
https://doi.org/10.1007/s10107-003-0375-9).

OR-Tools documents that CP-SAT uses integer models and distinguishes `OPTIMAL`,
`FEASIBLE`, `INFEASIBLE`, `MODEL_INVALID`, and `UNKNOWN`. In this protocol only
`OPTIMAL` and a fully audited `INFEASIBLE` proof are terminal. Assumption cores are
sufficient subsets, not automatically minimum cores, so every reported core must be
replayed before it can support a cut. See
https://developers.google.com/optimization/cp/cp_solver and
https://or-tools.github.io/docs/pdoc/ortools/sat/python/cp_model.

## Frozen full-authority inputs

The only scientific authority is the authenticated V6-A train-only authority bundle:
3229 source rows, 1830 observed pairs, 1383 antibodies, and 1172 antigens. V6-B reads
the same allowlist, four-column canonical manifest, four direct-edge ledgers, observed
pair incidence, and independently reconstructed `score_free_authority_v2` payload.

Contact density, paratope or epitope size, affinity, assay outcome, embeddings,
predictions, losses, ranks, model scores, and model results remain forbidden split
inputs. The only permitted density is observed antibody--antigen pair-incidence
density. No authority-validation or frozen-test bytes may be opened.

## Decomposition graph

The canonical V6-B problem contains one outer master and three dependent inner
subproblems:

1. The outer master assigns all authority entities and pairs across three outer folds,
   their training roles, and quarantine. It contains every exact integer row whose
   truth depends only on the outer assignment.
2. Outer candidate `o` deterministically defines three isolated outer-training
   universes `U_o`. No inner subproblem may read an entity or pair outside its exact
   `U_o`.
3. Inner subproblem `o` assigns `U_o` across its three dependent inner folds. It is
   reconstructed directly from the authenticated authority and the outer membership,
   not from a matrix, row list, variable index, or solution produced by another
   backend.
4. A combined candidate exists only when the outer master and all three inner
   subproblems return `OPTIMAL`, every exact primary gate passes, and the independent
   raw-byte auditor agrees field by field.

The complete 66-gate registry is partitioned without omission: 18 outer gates and 16
gates in each of three inner groups. The outer group contains 15 fold-level gates, one
cross-fold gate, and two pooled-axis gates. Each inner group contains 15 fold-level
gates and one cross-fold gate. Gate identifiers and threshold payloads must equal the
frozen V6-A registry byte for byte.

## Candidate rejection and cuts

The controller processes candidates and subproblems in canonical digest order.
Failing exact statistics generate the already frozen local membership no-good cut for
the precise fold, axis, role, and support state. An infeasible inner subproblem may
generate a master cut only through this sequence:

1. solve the inner model with one assumption literal for each canonical outer-training
   membership decision;
2. require CP-SAT `INFEASIBLE` and record the returned sufficient assumption subset;
3. rebuild the subproblem from raw authority using only that subset, with all other
   outer membership decisions free;
4. require a second `INFEASIBLE` result under the same locked solver settings;
5. independently rebuild and replay the support; and
6. add exactly one Boolean no-good clause that forbids simultaneous reproduction of
   every signed literal in the replayed support.

The cut must not exclude an outer state that differs on any signed support literal.
If assumption-core replay is unsupported, disagrees, returns `FEASIBLE`, `UNKNOWN`, or
`MODEL_INVALID`, or exceeds a resource limit, the protocol stops. It must not fall
back to an unproved full-outer-state cut. Duplicate cuts, undeclared gates, missing
inner universes, and primary/independent disagreement also stop.

Every master iteration, inner attempt, exact failure, and cut is append-only and binds
the authority, policy, protocol, solver runtime, implementation source, candidate,
signed support, replay, model proto, and output bytes by SHA-256.

## Telemetry-only row shards

V6-B has no sampled feasibility run. Before production solving, one deterministic
telemetry pass constructs the full variable dictionary and all full-authority row
metadata. It selects exactly one tenth of eligible constraint rows independently per
declared row family by the smallest lowercase SHA-256 of the canonical row payload,
rounding upward. It never samples vertices or pairs.

The telemetry pass records build peak RSS, row and nonzero counts, COO/CSC estimates,
model proto bytes, construction time, and projected memory with at least 25 percent
headroom. It does not solve the shard and cannot prove feasibility, infeasibility,
optimality, Gate 0, or training authorization. Failure to build or a projection without
the required headroom stops V6-B and requires another numbered execution protocol.

## Feasibility and objective execution

The first production phase is feasibility only. Acceptance requires outer master
`OPTIMAL`, all three inner subproblems `OPTIMAL`, all 66 exact primary gates passing,
the independent raw-source auditor agreeing, and the direct HiGHS fixed-assignment
linear projection audit returning optimal with zero gap. HiGHS remains non-authorizing
for the nonlinear feasible region.

Only after feasibility is proven may the unchanged retention-first lexicographic
objectives run. Each stage is solved through the same decomposition, independently
recomputed, and fixed by an exact integer equality before the next stage. Canonical
zero-first bit fixing may begin only after all earlier optima are frozen. `FEASIBLE`,
`UNKNOWN`, timeout, nonzero gap, an incomplete separation sequence, or disagreement is
non-authorizing at every stage.

## Freeze, post-freeze gates, and training boundary

A provisional digest, external create-only freeze receipt, and independently pinned
replay are still required. The independent auditor must reconstruct all 12 folds, 48
sequence boundaries, all 66 score-free gates, model allowlists, and quarantine
accounting from raw inputs.

Only the exactly replayed frozen assignment may proceed to the preregistered 3D
interface-leakage audit and frozen power simulation. D0/R1/R2 caches and training remain
blocked until those gates pass. Large-scale training remains blocked until the nested
small-model Gate 1 scientific criteria pass.

V6-B does not promise feasibility, an accepted split, useful model performance,
distributed scaling, state of the art, or world-leading results. Those claims require
their own completed identical-benchmark evidence.
