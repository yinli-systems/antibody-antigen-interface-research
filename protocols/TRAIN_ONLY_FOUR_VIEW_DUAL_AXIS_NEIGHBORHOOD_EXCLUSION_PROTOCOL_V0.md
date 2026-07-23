# Train-only four-view dual-axis fold-local neighborhood-exclusion protocol v0

Status: prospective preregistration draft. It is not frozen and authorizes neither
split use nor model execution. Authorization requires a frozen machine-readable
configuration, a complete exact homology graph, a proven canonical nested assignment,
an independent zero-edge audit, and the feasibility and precision gates below.

Date: 2026-07-23 CST

## Scientific question

Can an entire, prospectively fixed antibody--antigen partner-retrieval selection
procedure generalize within authority train under a fold-local pair-inductive boundary
in which neither member of a held-out pair, nor any direct
80%-identity/80%-reciprocal-coverage homolog of either member in any required sequence
view, is available to that fold's training procedure?

In this document, **development validation** means a held-out subset constructed only
from the authorized `train` membership. The source authority-validation pool is not an
untouched confirmation set: an earlier ESM-2 exploration inspected it, and the repaired
subsets have already been used as reused development confirmation. That history cannot
be undone. The current experiment may not read that pool, but this is a prospective
firewall for the new nested analysis, not a claim that the pool is sealed historically.
The frozen SAbDab2 test remains untouched. Only a separately frozen use of that test or
a prospective external experiment can provide independent confirmation.

The estimand is **foldwise direct-80/80 four-view homology-separated C3,
S2-like generalization**. It is not a global DataSAIL S2 partition and does not separate
every pair of entities connected by an arbitrarily long similarity path. The phrase
`S2-like` is permitted only with `foldwise` and `direct-80/80`; the unqualified phrases
`S2 split`, `global S2`, and `lineage-independent` are forbidden.

## Prospective boundary and historical ledger

This design is prospective only relative to model scores produced after its final
assignment is frozen. Prior authority-train screens, the failed provisional split, the
exact-audit failure, the strict-closure infeasibility result, and every prior inspection
of authority validation must be enumerated in a create-only history ledger with dates,
artifact hashes, score families viewed, and the decisions they informed. The candidate
family was motivated by prior work and is not presented as discovery on untouched data.

No outer or inner assignment may be redrawn, replaced, rebalanced, or relabeled after
the first score from that assignment is opened. This prohibition begins with the first
inner or outer model score for this assignment; it does not falsely declare that no
score has ever been seen. Any prospectively justified amendment before model execution
requires a new numbered draft, an amendment entry in the public history ledger, and a
new immutable hash. After model execution begins, a changed split is a new experiment,
not a repair or continuation.

## Why a new split is required

The first provisional assignment, historically labeled `S2`, failed its independent exact audit. The
[exact four-view audit receipt](train_only_s2_exact_nw/assignment_8dce37bec026/audit_receipt.json)
(file SHA-256
`f6bfd37f5ad062c1a3a48974b17f7bfdb24ac70a1c8ffd8ebcd479c9eb2670b3`)
found direct cross-fold threshold edges and therefore correctly withheld training and
model-selection authorization.

Adding every discovered VH, VL, concatenated-antibody, antigen, and exact-digest edge
transitively before repartitioning also failed for a principled reason. The
[strict-closure infeasibility receipt](train_only_s2_iterative_closure/iteration_001/infeasibility_receipt.json)
(file SHA-256
`1a4b74e1048f854d8d25b55f479e1fd6eb7c4dc5472a0c7b5ed8c811472b2323`;
scientific-payload SHA-256
`6a1eb3408f466dc7ebb76e1c89b17445d296e5f8a098fbf946adce02d9ffbea5`)
showed that the largest antibody component contains 92.697% of antibody entities and
93.115% of unique-pair degree. Such a component cannot be assigned intact while meeting
the locked three-fold balance bound.

This is not evidence that a direct train--validation homology boundary is impossible.
It is evidence that single-linkage transitive closure is too strong for this connected
antibody graph. A bridge chain `A--B--C` does not imply that `A` and `C` violate the
locked pairwise threshold. The next design therefore quarantines direct homology
neighborhoods separately for each held-out fold and audits the actual train--validation
boundary exhaustively.

## Primary-source basis

- [Petti and Eddy](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009492)
  formalize a bipartite independent pair: two retained sets with no threshold edge
  between them. They show why bridge entities can make single-linkage clustering fail
  even when a valid direct-edge-separated train/test pair exists, and allow entities to
  remain unassigned. They also show that the splitting algorithm can change benchmark
  difficulty, which motivates the representativeness diagnostics below.
- [GraphPart](https://doi.org/10.1093/nargab/lqad088)
  targets zero above-threshold cross-partition edges while retaining as many sequences
  as possible. It uses exact global Needleman--Wunsch alignment as a certification
  option and permits movement or removal when initial clusters are not separable. Its
  results also make clear that separation takes precedence over balance and that
  difficult graphs can lose entire partitions. GraphPart's reported identity and
  coverage conventions are not assumed to equal the exact metric in this protocol;
  GraphPart is methodological precedent, not the certifier of an edge here.
- [DataSAIL](https://www.nature.com/articles/s41467-025-58606-8) treats paired-input
  data as two-dimensional, defines similarity-aware S2 splitting on both entity axes,
  and discards an interaction when its two endpoints receive conflicting assignments.
  It also establishes that similarity-aware paired splitting is a constrained,
  NP-hard optimization problem rather than a routine random split.
- [Park and Marcotte](https://www.nature.com/articles/nmeth.2259) distinguish C1, C2,
  and C3 pair-input evaluation. C3 requires that neither endpoint of a held-out pair be
  represented in training. This protocol strengthens C3 by excluding direct homologs
  on both axes as well as exact endpoint reuse.
- [Disconnecting the Dots](https://openreview.net/forum?id=ifK9NFyrhn) motivates sparse
  removal of central connector points when stringent similarity graphs collapse into
  giant components. It is an OpenReview submission rather than a peer-reviewed method,
  so it is used here as design motivation and not as an evidentiary standard.
- [Owen's pigeonhole bootstrap](https://doi.org/10.1214/07-AOAS122) motivates separate
  resampling of row and column units for crossed, unbalanced data. It does not make
  evolutionary components independent or turn this observational collection into a
  probability sample; the interval language below is therefore deliberately
  conditional.

## Locked authority and data firewall

The split controller must accept only the following conceptual inputs, all restricted
to the authorized authority-train membership:

1. the pre-authenticated authority-train allowlist and its SHA-256, without mounting the
   full authority split;
2. canonical antibody and antigen sequences and canonical entity identifiers;
3. the observed exact antibody--antigen positive-pair incidence graph;
4. duplicate/source-row membership needed to keep all rows of one exact pair together;
5. the exact four-view homology edge ledgers defined below; and
6. score-free sequence and graph-topology covariates explicitly listed in this protocol.

For the currently pinned snapshot, the authority-train membership is 3,229 source rows,
1,830 unique exact antibody--antigen pairs, 1,383 antibody entities, and 1,172 antigen
entities. The authority membership SHA-256 is
`442095581ae9bf3c7440c1a818612298b5f81e5a62ef46f431e0867b9c8b466b`.
These counts are evidence, not tunable targets.

The current split process must not load any of the following:

- authority-validation or frozen-test sequence fields, coordinates, embeddings,
  predictions, or scores;
- contact, paratope, epitope, affinity, binder/non-binder, or experimental outcome
  labels;
- model checkpoints, learned embeddings used as a similarity measure, predictions,
  losses, ranks, retrieval metrics, or control results; or
- any inner or outer score produced from the assignment being constructed.

The observed positive-pair incidence graph `P` is itself a label: it identifies which
antibody--antigen combinations were observed as cognate structural pairs. The split is
therefore **label-aware but model-score-free**, not label-blind. `P` may be used only
for pair retention, validation utility, positive-degree and library-shape constraints,
and exhaustive quarantine accounting declared below. Contact, interface, affinity,
assay, and model outputs remain forbidden. An absent edge is a retrieval decoy, not an
experimentally verified non-binder. Exact-pair multiplicity is never an optimization
weight.

Any current-version process that parses an out-of-scope sequence field or a
model-derived quantity invalidates the complete split version. The event and every
accessed path must be written to the history ledger. A new version must start from the
authenticated authority-train inputs; the contaminated version cannot be repaired.

## Exact preprocessing and canonical units

The final machine-readable configuration must bind the input byte hashes, column names,
CSV dialect, UTF-8 decoder, parser and runtime versions, and the following normalization
contract. A changed byte, parser rule, or normalization rule is a new dataset version.

For each authority-train row, `VH`, `VL`, and antigen are parsed only after the leading
source identifier has been authenticated as authority train. A sequence is normalized
by deleting every Unicode code point for which the pinned runtime's `str.isspace()` is
true, converting to uppercase, requiring ASCII, and requiring every remaining symbol to
belong to `ACDEFGHIKLMNPQRSTVWY`. `VH` and antigen must be non-empty. `VL` may be empty;
no other chain may be empty. Ambiguous, non-canonical, stop, gap, and unknown symbols
(`B`, `J`, `O`, `U`, `X`, `Z`, `*`, `-`, and all others) cause a hard failure and are
never deleted, imputed, or mapped. Source identifiers are exact, non-empty strings with
no leading or trailing whitespace and no Unicode normalization.

The concatenated-antibody alignment view is exactly normalized `VH` followed by
normalized `VL`, heavy chain first, with an empty delimiter. When `VL` is missing this
view equals `VH`. The canonical antibody entity is nevertheless the ordered tuple
`(VH, VL)`, not the unframed concatenation. Define `frame(x)` as the ASCII decimal byte
length of `x`, followed by byte `:`, followed by the ASCII bytes of `x`. Canonical IDs
are

```text
a_id = SHA256(b"antibody-v1\0" + frame(VH) + frame(VL))
g_id = SHA256(b"antigen-v1\0" + frame(antigen))
p_id = SHA256(b"pair-v1\0" + ASCII_HEX(a_id) + b"\0" + ASCII_HEX(g_id))
```

The pinned snapshot counts above must be independently reproduced under this framed
definition before feasibility work. A count mismatch invalidates the draft snapshot;
the implementation may not coerce entities to recover the expected numbers.

The builder must prove that no two antibody tuples collapse accidentally under raw
heavy-then-light concatenation; if they do, they remain distinct entities but their
identical concatenated view creates an exact similarity edge. Every source row with the
same `(a_id, g_id)` belongs to one indivisible exact-pair group.

The source-row extractor, canonicalizer, and digest implementation used by the
independent verifier must not import the builder implementation. They must reproduce
the inventory from the authenticated raw authority-train bytes and pass a published
cross-implementation golden-vector suite covering whitespace, case, missing VL, invalid
symbols, length framing, chain order, duplicate rows, and malformed CSV. A shared
function called from both paths is not independent evidence.

Let:

- `A` be the set of canonical antibody entities;
- `G` be the set of canonical antigen entities; and
- `P` be the set of unique observed exact antibody--antigen pairs in `A x G`.

All source rows mapping to one element of `P` are an indivisible group. Optimization,
retention, and balance use unique exact pairs as the primary unit. Source-row counts are
reported secondarily and may not dominate an objective.

An antibody entity carries three audit views:

1. VH;
2. non-empty VL; and
3. the locked concatenated-antibody representation.

An antigen entity carries the antigen sequence view. A genuinely missing VL is absent
from the VL inventory; it is never represented by an empty token, copied VH, or other
imputation. VH and concatenated-antibody checks still apply to that antibody.

Distinct canonical identifiers with identical non-empty sequences are connected by an
edge. Canonical identifier equality is itself disallowed across a fold boundary even if
an upstream manifest error prevented a sequence edge from being emitted. Counts are
computed after exact-pair collapse; source-row counts are disclosure-only.

## Exact four-view edge definition

For each view, build a graph whose vertices are unique canonical sequences and whose
edges are unordered sequence pairs meeting both locked conditions under exact global
alignment:

- aligned residue identity is at least `0.80`; and
- coverage of each full raw sequence by paired residue--residue columns is at least
  `0.80`.

Equality at either threshold is an edge. Identity is the number of identical
residue--residue columns divided by the number of residue--residue columns; gap columns
are excluded from this denominator. Reciprocal coverage is the number of paired
residue--residue columns divided separately by each full raw sequence length.

The operational aligner contract is the same as the completed train-only exact audit:

- global Needleman--Wunsch-family dynamic programming;
- Biopython `PairwiseAligner` version `1.83`, with the complete Python, operating-system,
  architecture, and dependency lock hashes recorded;
- BLOSUM62 with canonical labeled-matrix payload SHA-256
  `2c62ca4bb7a4c3ec5fe5faee64276841028a7e6638cf846c801dd9c5e64527e2`;
- affine gap-open penalty `10.0` and gap-extension penalty `1.0`;
- terminal gaps penalized identically to internal gaps;
- the first optimal traceback returned by that pinned runtime, with a golden tie-case
  conformance suite and the ordered aligned-block coordinates recorded; and
- integer cross-products for threshold decisions, avoiding floating-point boundary
  comparisons.

All unordered pairs in every view must be considered. The only permitted pre-alignment
prune is the mathematically safe rule
`min(raw_length_1, raw_length_2) / max(raw_length_1, raw_length_2) < 0.80`, because such
a pair cannot attain 80% reciprocal coverage. Every prune must be counted and its length
pair retained in a digest-only ledger.

Let `H_A` be the union of the VH, non-empty-VL, and concatenated-antibody edge relations
lifted to canonical antibody entities. Let `H_G` be the antigen edge relation lifted to
canonical antigen entities. These are direct-edge graphs. Their connected-component
closures are used for diagnostics and bootstrap units, not as fold-equality constraints.

Approximate MMseqs2 or embedding neighbors may be used only as a runtime diagnostic.
They cannot certify a non-edge, seed the final optimization without exact completion,
or replace exhaustive exact construction.

The edge definition is protocol-specific. It must not be described as numerically
identical to GraphPart, MMseqs2, CD-HIT, BLAST, or any paper unless the other method's
denominators, coverage rules, substitution scores, terminal-gap handling, and traceback
are reproduced and cross-validated exactly.

## Recommended primary design: repeated closed-neighborhood exclusion

The primary design is three outer development folds. It is a deterministic,
pair-incidence-aware adaptation of a bipartite independent-pair construction, applied
jointly to the antibody and antigen axes.

For outer fold `f`, choose score-free held-out endpoint sets `V_A[f] subset A` and
`V_G[f] subset G`. Define the closed homology neighborhoods

```text
closed_A[f] = V_A[f] union {a in A: exists v in V_A[f] with (a, v) in H_A}
closed_G[f] = V_G[f] union {g in G: exists v in V_G[f] with (g, v) in H_G}
```

The allowed training endpoint sets are

```text
T_A[f] = A minus closed_A[f]
T_G[f] = G minus closed_G[f]
```

The retained unique-pair sets are then defined mechanically:

```text
P_val[f]   = P intersect (V_A[f] x V_G[f])
P_train[f] = P intersect (T_A[f] x T_G[f])
P_quarantine[f] = P minus (P_val[f] union P_train[f])
```

Define the only legal model-facing entity inventories as

```text
A_model[f] = {a: exists g with (a, g) in P_train[f]}
G_model[f] = {g: exists a with (a, g) in P_train[f]}
```

The training process must prove exact equality between its accessible antibody-ID
inventory and `A_model[f]`, and between its accessible antigen-ID inventory and
`G_model[f]`. Subset checks are insufficient. Necessarily `A_model[f] subset T_A[f]`
and `G_model[f] subset T_G[f]`; an endpoint in `T_A[f]` or `T_G[f]` with no retained
training pair is not model-accessible. The same equality applies independently to every
inner fold.

This allowlist covers every loader, sequence file, embedding cache, coordinate and
label table, hard-negative or retrieval index, tokenizer cache, feature normalizer,
batch statistic, sampler, and diagnostic invoked by a model process. No model process
may parse an antibody in `A minus T_A[f]`, an antigen in `G minus T_G[f]`, or a source
row whose exact pair is not in `P_train[f]`, even as an unlabeled example, normalization
row, decoy, cached embedding, or precomputed index member. Whole-pool embedding files
must not be mounted. Fold-local caches are built by an audited, create-only extractor
that filters the authenticated leading ID before any sequence, coordinate, or label
field is parsed and publishes exact ID-set and content hashes.

Every endpoint in `V_A[f]` and `V_G[f]` must occur in at least one member of
`P_val[f]`; dangling validation-library entities are forbidden. Every pair involving
one held-out endpoint and one non-held-out endpoint, a direct neighbor of a held-out
endpoint, or endpoints assigned to conflicting roles is quarantined for that fold.

The outer held-out antibody sets must be pairwise disjoint across folds, as must the
outer held-out antigen sets. Consequently, outer validation exact pairs are disjoint.
An entity held out in one outer fold may be used by a different fold's model only when
it is outside that other fold's closed validation neighborhood, as in ordinary
cross-validation with fold-specific training data.

This construction guarantees, before looking at any score, that for every fold:

1. `P_train[f]` and `P_val[f]` share no exact antibody entity;
2. they share no exact antigen entity;
3. no training antibody has a VH, VL, or concatenated-antibody edge to a validation
   antibody; and
4. no training antigen has an antigen-view edge to a validation antigen.

Thus each development validation pair is C3 with respect to its fold and the boundary
is direct-80/80 S2-like on both axes. It is not a reusable global S2 assignment. A long
graph path may connect a training entity to a validation entity, but a direct
locked-threshold edge may not.

### Direct-edge guarantee

Take any effective training antibody `a_t` and validation antibody `a_v` in fold `f`.
By construction, `a_t` belongs to `T_A[f]`, while `a_v` belongs to `V_A[f]`. Because
`T_A[f]` is the complement of the closed `H_A` neighborhood of `V_A[f]`, `a_t` is
neither equal nor adjacent in `H_A` to `a_v`. Since `H_A` is the union of the VH, VL,
and concatenated-antibody relations, none of those three threshold edges can cross the
boundary. The identical argument on `H_G` excludes every antigen threshold edge.

The guarantee is conditional only on complete and correct edge construction, which is
why the graph is built exhaustively and a separate verifier independently enumerates
every actual boundary pair. It does not depend on solver heuristics, a probabilistic
cluster assignment, or model behavior.

## Feasibility-first execution order

No result from this protocol is usable, and no model process may start, until every
step below has completed in order with create-only receipts:

1. authenticate authority-train bytes, exact preprocessing, the history ledger, and the
   model-data firewall;
2. complete the exhaustive four-view exact graph and safe-prune accounting;
3. publish topology-only graph summaries and the preregistered global graph-cut
   sensitivity, without fitting or scoring a model;
4. run one deterministic solver-scaling dry run on a canonical digest-selected 10%
   induced subproblem, using the final solver formulation, limits, and thread count;
5. solve a single joint outer-plus-inner feasibility model containing every hard gate;
6. only if feasibility is proven, solve the retention-first lexicographic objective and
   canonical tie break;
7. independently reconstruct and audit all outer and inner boundaries; and
8. run the locked attainable-precision and power simulation on the frozen fold
   structure.

The dry run is a computational scaling check, not a search over seeds or formulations.
Its 10% vertex inventory is fixed by the smallest canonical entity digests before the
graph is opened. The final feasibility model includes all three outer folds and all nine
inner folds; an outer assignment is not accepted merely because a different, unrecorded
outer assignment might admit inner folds. If feasibility is infeasible, unknown at the
resource limit, unauditable, or fails the precision gate, the protocol stops before
training. A smaller validation panel, relaxed homology threshold, changed objective,
new random seed, or graph-cut model comparison requires a new prospectively numbered
protocol.

## Immutable outer-fold gates

All gates are evaluated on unique exact pairs and effective endpoints after pairing.
Let `M = |P|`. A split is ineligible unless every outer fold satisfies all gates.

### Validation utility

- `max(120, ceil(0.07 M)) <= |P_val[f]| <= floor(0.09 M)`.
- At least 60 unique antibody endpoints and 60 unique antigen endpoints occur in
  `P_val[f]`.
- The held-out sets intersect at least 25 full-graph connected components of `H_A` and
  at least 25 full-graph connected components of `H_G`. Component IDs are computed on
  the complete authority-train graphs before folding. Components of the graph induced
  only on held-out endpoints are reported descriptively but never counted as independent
  sampling units.
- Every retained held-out query has at least one observed cognate in the opposite
  held-out library.

For the pinned `M = 1,830`, the validation-pair range is 129--164 per outer fold. These
values are fixed before feasibility is tested and cannot be relaxed in this version.

### Training retention

- `|P_train[f]| >= ceil(0.60 M)`.
- Effective training antibody and antigen endpoint counts are each at least 60% of
  their corresponding full authority-train counts.
- `|P_quarantine[f]| <= floor(0.33 M)`.
- No exact pair or any of its source rows occurs in both training and validation for the
  same fold.

### Cross-fold balance

- The outer-fold validation pair-count relative range,
  `(maximum - minimum) / mean`, is at most 0.15.
- The same relative range is at most 0.20 separately for validation antibody counts,
  validation antigen counts, training pair retention, quarantined pair counts,
  singleton-positive query fractions in each direction, validation bipartite density,
  and exact finite-chance bidirectional MRR.
- No fold may be dropped, merged, or assigned zero weight in the final aggregation.

### Label-aware, score-free panel shape and representativeness

Before optimization, construct deterministic quantile strata over the full
authority-train pool for the following covariates:

- VH, non-empty VL, concatenated-antibody, and antigen raw lengths;
- `log1p` direct degree in each view and in `H_A` or `H_G`;
- `log1p` observed exact-pair incidence degree on each entity axis; and
- missing-VL status as its own binary stratum.

For a covariate with sorted observations `x_(1),...,x_(n)`, tentative quintile cuts are
the type-1 empirical quantiles `x_(ceil(q n))` for `q` in `{0.2,0.4,0.6,0.8}`. Repeated
cuts are merged. A value equal to a cut stays wholly in the lower closed bin; identical
values are never divided by digest or identifier. VL length bins and summaries are
computed only among non-missing VL entities, while missing-VL is a separate binary
category. Fewer than two remaining bins is disclosed and the covariate contributes no
pseudo-stratum.

For a continuous covariate, the absolute standardized mean difference between groups
`X` and `Y` is
`abs(mean(X)-mean(Y))/sqrt((sample_var(X)+sample_var(Y))/2)`. If the denominator is
zero, SMD is zero only when the means are equal and infinity otherwise. KS is the exact
supremum of the two empirical CDFs evaluated at the union of unique observed values;
ties remain blocks. Total variation is `0.5 * sum_b abs(p_X[b]-p_Y[b])` over the fixed
full-pool bins, including explicit missing categories.

No biochemical outcome or model-derived feature is permitted. For every fold, all
SMD, TV, and KS gates are computed twice: held-out effective endpoints versus the full
authorized entity pool, and held-out effective endpoints versus the actual effective
endpoints of `P_train[f]`. The maximum absolute SMD must be at most 0.35, TV at most
0.20, and KS for any length or degree covariate at most 0.20 in both comparisons. Across
the pooled entity-disjoint outer validation sets, the full-pool thresholds are 0.25,
0.15, and 0.20, respectively. An effective endpoint means an entity incident to at
least one retained pair in the named set; unpaired assigned entities never enter a
denominator.

Because `P` is an observed positive-incidence label, every final fold manifest must also
freeze, before model execution, the complete validation antibody and antigen libraries,
positive IDs per query, singleton and multi-positive query counts, both degree
distributions, bipartite density, degree Gini and top-1%/5%/10% concentration, and exact
finite-library chance MRR in each direction. For a query with `d` positives in a library
of size `L`, chance MRR is computed exactly as
`sum_(r=1)^(L-d+1) [C(L-r,d-1)/C(L,d)] / r`; fold chance is the mean over queries.
Candidate libraries, positives, and chance values cannot be downsampled or redefined
after any score is visible.

These gates intentionally prevent the optimizer from constructing a nominally isolated
validation set consisting only of unusually short or extremely low-degree outliers.

## Immutable retention-first optimization objective

The three outer and nine inner folds are solved in one feasibility model under the hard
zero-edge, utility, retention, panel-shape, balance, and representativeness constraints
above. No random restart, convenient seed, or visually preferred split may be selected.
After joint feasibility is proven, the solver optimizes the following exact
lexicographic tuple, one stage at a time while fixing every earlier optimum:

For an entity axis `X`, role `R` in `{validation, training}`, fold `f`, and locked
stratum bin `b`, let `N_X` and `N_X[b]` be the full-pool entity counts and let
`n[f,X,R]` and `n[f,X,R,b]` be the corresponding retained counts. Define

```text
D_rep = max over (f, X, R, b)
        abs(n[f,X,R,b] / n[f,X,R] - N_X[b] / N_X)
```

`D_rep` is an exact rational objective over a finite assignment space. An
implementation may enumerate permitted retained-count denominators or use an exact
linear-fractional reformulation, but it may not approximate `D_rep` with a sampled
histogram or rounded decimal.

1. maximize `min_f |P_train[f]|` over the three outer folds;
2. maximize `sum_f |P_train[f]|`;
3. minimize `sum_f |P_quarantine[f]|` and then its maximum across outer folds;
4. after the outer values above are fixed, maximize the minimum normalized inner
   training retention across all nine inner folds, then maximize its sum;
5. minimize the worst absolute deviation of `|P_val[f]|` from `0.08 M`, followed by
   validation-pair, effective-endpoint, exact-chance-MRR, density, and quarantine
   cross-fold ranges in that order;
6. minimize `D_rep` subject to the already fixed meaningful SMD, TV, and KS tolerances;
7. maximize the minimum number of intersected full-graph homology components across
   both axes and all folds; and
8. choose the lexicographically smallest complete outer-plus-inner assignment
   bit-vector, ordered by canonical entity digest, outer fold, inner fold, axis, and
   role.

Representativeness deviations must be compared with integer cross-products of bin
counts, not rounded floating-point proportions. The final tie break must be obtained by
deterministic sequential variable fixing, not by relying on solver enumeration order.

Every mixed-integer stage must have a proven optimal solution with zero reported
relative gap. An incumbent from a timeout is not a valid split. Solver name, version,
parameters, thread count, time limit, status, primal objective, dual bound, gap, and
canonical tie-break trace must be published.

This ordering prevents a cosmetically representative but heavily quarantined split from
defeating a more informative feasible split. Covariate balance remains a hard gate and
an objective only after retention, not a license to optimize away difficult entities.

If feasibility or any objective stage is infeasible or not proven optimal, this
protocol stops. It does not fall back automatically to a graph-cut split, enlarge the
validation range, reduce the training-retention gate, weaken a representativeness gate,
or change the 80/80 boundary.

## Nested model-selection semantics

The outer folds estimate the performance of the entire inner-selection algorithm, not
the performance of one architecture selected after outer results are viewed. They are
not used for epoch selection, hyperparameter selection, calibration, hard-negative
mining, candidate promotion, or architecture repair. The separate
`TRAIN_ONLY_NESTED_PARTNER_RETRIEVAL_MODEL_PROTOCOL_V1_DRAFT.md` defines the only
compatible future model procedure and explicitly supersedes
`PARTNER_RERANKER_PROTOCOL_V0.md` for this experiment.

All three outer assignments and all nine inner assignments must be constructed,
audited, and published before the first model process starts. No inner split may be
created or replaced after any inner or outer model score is available.

For each outer fold `f`:

1. remove `P_val[f]`, every quarantined pair, and all outer held-out endpoint data from
   the model-selection workspace;
2. construct three inner folds only within `P_train[f]`, using the same closed-neighborhood
   equations, exact graphs induced on the outer-training universe, score-free objective,
   and canonical solver policy;
3. select among a separately hashed, preregistered finite candidate/hyperparameter grid
   using only the three inner validation results and a locked primary statistic;
4. resolve metric ties by lower trainable capacity and then lexical configuration ID;
5. use the fixed epoch count embedded in the selected configuration; epoch selection,
   early stopping, and checkpoint selection are forbidden;
6. refit the selected configuration once on all of `P_train[f]`; and
7. evaluate `P_val[f]` once, producing immutable out-of-fold predictions.

At steps 1--6, the exact model-facing antibody inventory must equal the antibody
endpoints of the relevant inner `P_train`, and the antigen inventory must equal its
antigen endpoints. Outer held-out endpoints and all homologous or quarantined endpoints
remain inaccessible even when they are not members of a particular inner validation
set. Each inner evaluation process is separate from its training process and may not
write a checkpoint, optimizer state, cache, normalizer, or index.

Inner gates are scaled to the outer-training universe `P_outer`:

- `max(80, ceil(0.07 |P_outer|)) <= |P_inner_val| <= floor(0.09 |P_outer|)`;
- at least 40 effective held-out endpoints on each axis;
- at least 20 intersected full-graph components on each axis;
- at least 55% retention of outer-training unique pairs and effective endpoints;
- no more than 38% quarantined unique pairs; and
- per-inner-fold standardized-mean-difference, total-variation, and KS limits of 0.40,
  0.25, and 0.25, respectively.

For each outer nest, inner quantile cuts are rebuilt once from the effective endpoint
universe of `P_outer` using the same type-1, tie-preserving rule and are then shared by
its three inner folds. Each inner held-out set is compared both with that outer-training
universe and with its actual inner `P_train` endpoints. Full-authority-train component
IDs remain the dependence and component-count IDs; an induced inner component is
descriptive only. No outer-held-out field may enter an inner cut, covariate, or graph.

Inner held-out endpoints are entity-disjoint across the three inner folds belonging to
one outer fold. Every inner boundary must pass the same independent four-view exact
audit before any inner model starts.

All three inner selections are completed and create-only frozen before any outer
prediction is opened. The outer results then characterize only the locked selection
algorithm. They cannot name a winning candidate, promote an architecture, modify a
configuration, or justify a rerun. Candidate selection frequencies are descriptive.

If a single final configuration is needed for a later full-authority-train refit, it is
the modal configuration selected by the three already-frozen outer nests; ties prefer
the lower-capacity candidate and then lexical configuration ID. This summary is created
before any outer score is materialized and is unaffected by outer performance. Neither
it nor the outer result makes authority validation independent. Any future frozen-test
use requires a separate protocol frozen before test access.

## Validation retrieval and statistical unit

For each outer fold, bidirectional retrieval uses exactly the effective held-out
antibody and antigen libraries. All observed multi-positive cognates in `P_val[f]` are
positives for ranking and are masked correctly in contrastive denominators. Unobserved
cross-pairs are retrieval decoys, not verified non-binders.

Because held-out entities may be homologous to other held-out entities, query-level
averages alone can overstate effective sample size. Reports must therefore include:

- ordinary query-weighted metrics;
- metrics giving equal total weight to each represented full-graph `H_A` or `H_G`
  component;
- all three outer-fold metrics without selecting the best fold;
- a crossed row/column pigeonhole bootstrap over full-graph antibody and antigen
  components; and
- the number of unique queries, exact pairs, and homology components contributing to
  every interval.

The ordinary finite-panel statistic is the equal-fold mean of the two direction-specific
query means, with folds weighted `1/3` and directions weighted `1/2`. A
component-balanced sensitivity gives equal total weight to each full-graph component
represented in a fold, then equal weights to folds and directions. A three-point t-test
over folds is not an adequate uncertainty analysis.

### Dependence units and redundancy sensitivities

Full-authority-train connected-component IDs of `H_A` and `H_G`, computed before any
fold restriction, are the conservative sequence-dependence units. Components of a
held-out-induced subgraph are descriptive only: removing bridge entities can split one
lineage proxy into apparently independent pieces and must not inflate sample size.

Two frozen sensitivities accompany the full-component analysis. First, a
`q=0.80` redundancy-reduced panel is constructed separately on each held-out axis by a
proven maximum independent set in the direct graph, with a canonical-digest
lexicographic tie break; retrieval includes only queries retaining at least one cognate
in the redundancy-reduced opposite library. Second, if authority-train germline/V-gene
or curated lineage metadata were authenticated before the split, results are reweighted
by those lineage clusters. Metadata cannot be inferred or chosen after scores are seen,
and this sensitivity cannot select the split or model.

Population-style confidence-interval language is allowed only if both axes have at
least 30 represented full-graph components, the largest component contributes less than
20% of effective queries in its direction, at least 99% of attempted bootstrap
replicates are valid, and the full-component and q-reduced conclusions agree in sign.
Otherwise, report the resampling output as a **conditional finite-panel stability
interval** or, if degenerate, only a descriptive range. It is not a population CI and
does not establish lineage-independent generalization.

### Locked crossed bootstrap

The primary paired comparison is the nested selection algorithm minus the locked dual
encoder on exactly matched out-of-fold predictions. It uses `20,000` valid replicates,
NumPy `PCG64DXSM`, seed `2026072301`, and no seed search. Let `C_A` and `C_G` be the full
`H_A` and `H_G` component IDs represented anywhere in the three outer validation
panels. In each attempt, independently draw `|C_A|` antibody component IDs with
replacement from `C_A` and `|C_G|` antigen component IDs with replacement from `C_G`.
The resulting multinomial counts are shared across every fold in which a component
appears; folds are fixed blocks, never resampled.

Within each fold, expand every entity once for each draw of its component. Repeated
entities receive distinct bootstrap-copy IDs but retain the original score and positive
edges. For Ab-to-Ag retrieval, each sampled antibody copy is a query and all sampled
antigen copies form the candidate library; the reverse direction is analogous. A
cognate edge is copied at every sampled row--column intersection. Queries with no
sampled cognate are undefined and omitted. Candidate copies are ordered by descending
score, canonical entity digest, component-draw ordinal, and within-component copy
ordinal; rank is the one-based position of the first cognate copy. A fold-direction is invalid
if it has no defined query. An attempt is invalid if any of the six fold-directions is
invalid or any required statistic is non-finite.

The same component draws, copy order, tie handling, query exclusions, and fold/direction
weights are used for every candidate, baseline, control, and paired difference.
Replicates aggregate query MRR within each fold-direction, directions by `1/2`, and
folds by `1/3`; they do not pool folds by query count. Attempt at most `25,000` draws to
obtain exactly `20,000` valid replicates. If that fails, no interval is reported. The
interval is the type-7 `2.5%` and `97.5%` empirical quantiles of paired replicate
differences. The point estimate is always the original, unresampled statistic, not the
bootstrap mean. Direction-specific intervals are secondary.

There is one primary preregistered train-only comparison: the selection algorithm versus the
locked baseline on the bidirectional statistic. Secondary directions, controls,
subgroups, contact endpoints, q-reduced panels, and lineage sensitivities are reported
with effect sizes and intervals but no unadjusted discovery claims. If simultaneous
claims are made across the two directions or across controls, two-sided bootstrap
p-values use plus-one counting and Holm adjustment within the explicitly named family.
For point difference `d_hat` and paired replicate differences `d_b`, the two-sided
null-centered plus-one value is
`(1+count(abs(d_b-d_hat)>=abs(d_hat)))/(B+1)`; Holm uses raw double-precision values and
lexical endpoint ID for exact p-value ties.
Training seeds are a fixed finite set, not bootstrap units; report each seed, their
arithmetic mean, sample standard deviation, and worst seed without treating three seeds
as an inferential population.

### Locked attainable-precision and power simulation

After assignments and component IDs are frozen but before a model process starts, run a
score-free planning simulation using only fold membership, component membership,
library sizes, and positive incidence. No empirical
model prediction, rank, loss, or pilot effect may enter it. The minimum meaningful
absolute improvement is fixed at `Delta*=0.020` in equal-fold bidirectional MRR.

For each simulated experiment and each of three fixed model seeds, generate paired
query-level improvements

```text
d[f,s,q] = Delta + F[f] + R[component_A(q)]
           + Cbar[positive component_G(q)] + U[s] + E[f,s,q]
```

for Ab-to-Ag, swapping row and column roles for Ag-to-Ab. `Cbar` is the arithmetic mean
of the positive-partner component effects for that query. All normal variates are
independent except that the same fold, row, column, and seed effects are reused across
directions where their IDs coincide. Scale the variance components to marginal SD
`sigma` and use the full grid
`sigma in {0.10,0.20,0.30}`, row ICC `in {0.05,0.15,0.30}`, column ICC
`in {0.05,0.15,0.30}`, fold ICC `=0.05`, and seed SD
`in {0.005,0.015,0.030}`. With `V=sigma^2`, set
`Var(F)=0.05V`, `Var(R)=rho_row V`, `Var(C)=rho_col V`,
`Var(U)=seed_sd^2`, and `Var(E)=V-Var(F)-Var(R)-Var(C)-Var(U)` before the
positive-component averaging in `Cbar`. A grid point with negative residual variance is
invalid and reported, not silently changed. Clip
`d` to `[-1,1]`. Simulations use `PCG64DXSM` seed `2026072302`; experiment and
bootstrap streams are derived from
`SHA256("power-v1" || scenario_index || Delta || experiment_index)` reduced modulo
`2^128`.

Within a planning-bootstrap replicate, draw full antibody and antigen components exactly
as in the locked bootstrap. For an Ab-to-Ag query `a`, its multiplier is the sampled
count of `component_A(a)` times the arithmetic mean of sampled counts over the distinct
antigen components containing its positives. For an Ag-to-Ab query, swap the axes.
Zero-weight queries are omitted; a zero-total-weight fold-direction invalidates the
replicate. Average `d[f,s,q]` over the three fixed seeds first, take the
multiplier-weighted mean within each fold-direction, then aggregate directions by `1/2`
and folds by `1/3`. In the full variance grid, run `500` experiments at each of
`Delta=0` and `Delta=0.020`, using exactly `1,000` valid bootstrap replicates within
`1,250` attempts per experiment. In the preregistered moderate scenario, run `2,000`
experiments for each `Delta in {0,0.01,0.02,0.03,0.05}`, using exactly `2,000` valid
replicates within `2,500` attempts. An experiment that misses its cap is invalid. Use
the same type-7 percentile limits as the planned analysis.

For every scenario report the fraction of intervals whose lower bound exceeds zero,
median and 90th-percentile interval width, empirical null false-positive rate,
invalid-experiment and invalid-replicate fractions, and bias. Report Wilson 95% Monte
Carlo intervals for power and false-positive rates, using the standard Wilson score
formula with `z=1.959963984540054` and the number of valid experiments as `n`. The preregistered planning gate is
evaluated at `sigma=0.20`, row ICC `0.15`, column ICC `0.15`, and seed SD `0.015`:
the lower Wilson limit for power at `Delta*=0.020` must be at least `0.80`, the upper Wilson limit for the null
false-positive rate must be at most `0.075`, and median interval half-width must be at
most `0.020`. These are assumption-conditional planning results, not evidence of
achieved biological power. Failure stops model execution under this version rather
than changing the panel or meaningful-effect threshold.

## Independent exact certification

The optimization edge graph and final certification must be separate code paths. The
independent verifier must use its independent extractor/canonicalizer and published
golden-vector conformance suite to reconstruct canonical sequences from authenticated
authority-train bytes. For every outer and inner fold it enumerates all unique
train--validation sequence pairs in each of the four views.

The verifier must record, by fold and view:

- total possible cross-boundary pairs;
- mathematically safe length prunes;
- exact dynamic-programming alignments;
- exact-digest equality crossings;
- threshold-edge crossings;
- maximum observed identity among pairs meeting both coverage requirements;
- input, configuration, source, inventory, edge-ledger, assignment, and output hashes;
  and
- proof that the current process parsed no authority-validation or frozen-test
  membership beyond the pre-authenticated train allowlist and no out-of-scope sequence
  field.

Authorization requires exactly zero direct crossing edges in VH, non-empty VL,
concatenated antibody, and antigen for every boundary. One crossing edge is a complete
failure, not a warning. Because the optimizer already consumed a complete exact graph,
an audit crossing indicates an implementation, canonicalization, or integrity defect.
The same protocol version may not add the edge and try a new split. Repair requires a
new version, a new immutable graph, and a new preregistration hash.

The verifier must also prove exact endpoint disjointness, exact-pair grouping, complete
quarantine accounting, all numerical gates, canonical fold-labeling, and exact equality
of the proposed antibody and antigen model allowlists to the endpoints of `P_train`.
Training is permitted only when a create-only receipt explicitly states
`authorized_for_nested_model_selection=true`.

## Bias and selection diagnostics

All diagnostics below are computed and published before model execution. Hard-gate
diagnostics are named above; the remaining diagnostics are mandatory disclosures and
cannot be used to search for another split.

Report separately for the full authorized pool, each training set, each held-out set,
each quarantine set, and the pooled entity-disjoint held-out sets:

- unique pair, source-row, antibody, antigen, and homology-component counts;
- sequence-length, missing-VL, direct-degree, component-size, and observed pair-degree
  distributions;
- nearest opposite-boundary exact identity and reciprocal coverage below the threshold;
- source-row multiplicity per exact pair;
- quarantine reason counts: held-out endpoint, antibody neighborhood, antigen
  neighborhood, both neighborhoods, or mixed-role pair;
- quarantine concentration by entity, including Gini coefficient and the fractions
  attributable to the top 1%, 5%, and 10% of entities;
- overlap and Jaccard similarity among the three fold-specific training sets;
- strata-specific retention rates and the worst retained fraction among strata
  containing at least 5% of the full pool; and
- where authenticated metadata are already available within authority train, species,
  experimental method, deposition year, resolution missingness, and antigen-chain-count
  distributions.

Contact density, paratope size, epitope size, affinity, and other outcome-derived
quantities may be summarized only after the assignment and all score-free diagnostics
are immutable. Such summaries are descriptive. An inconvenient label distribution does
not authorize re-splitting; it limits the claim or motivates a prospectively registered
new dataset version.

## Global graph-cut removal versus repeated neighborhood exclusion

### Global graph-cut removal

A global graph-cut design would choose permanent connector quarantine sets on `H_A`
and `H_G`, remove every pair incident to those entities, assign the remaining homology
components to three folds, and retain only pairs whose two endpoint components receive
the same fold. This follows the removal intuition in GraphPart, Petti--Eddy, and the
Disconnecting-the-Dots submission.

Its principal advantage is one reusable global partition: each retained entity has a
single fold and all models see structurally comparable training pools. If exact edge
construction and final auditing are complete, it can guarantee the same zero direct
cross-fold edges.

Its liabilities are material in this dataset:

- a connector is removed from every fold rather than only from the training set of a
  fold whose validation boundary it threatens;
- central VH/VL lineages and high-incidence entities are preferentially removed, which
  can change the biological population and erase scientifically important difficult
  cases;
- the paired antibody--antigen graph couples two separate vertex-separator problems, so
  a small entity cut need not imply high exact-pair retention;
- the already observed giant antibody component makes retention uncertain and exposes
  the result to the heuristic used to define "central"; and
- removal plus component assignment may satisfy homology separation while producing an
  unrepresentative or tiny validation fold.

### Repeated closed-neighborhood exclusion

The recommended design removes no entity globally. It chooses a representative held-out
anchor set for each fold, excludes only that set's direct homology neighborhoods from
that fold's training pool, and quarantines incompatible pairs. It directly enforces the
scientific boundary that will be claimed and does not convert a chain of individually
local edges into a permanent equality constraint.

Its costs are explicit: training populations differ by fold; a connector may be
quarantined repeatedly; held-out anchor selection can favor low-degree entities; and
models must be fit separately inside a true nested design. The joint score-free
representativeness objective, hard bias gates, component-balanced metrics, exact
quarantine ledger, and training-set-overlap diagnostics are included to control these
costs.

### Recommendation

Use repeated closed-neighborhood exclusion as the primary train-only protocol. It is
the better match to the intended foldwise direct-80/80 four-view C3/S2-like estimand,
preserves a usable training population without permanently deleting central lineages, and remains
valid even though strict transitive three-fold closure is infeasible.

Run global graph-cut removal only as a score-free topology sensitivity analysis. It may
report connector counts, pair retention, component balance, solver scaling, and the
same bias diagnostics, but no model may be trained on it in this version. The two schemes must
never be compared by model performance and then selected retrospectively. Training on
a graph-cut split requires a separate preregistration written before any graph-cut model
score is observed.

## Publication and immutability contract

Before any model process starts, publish create-only artifacts containing:

1. the historical-inspection and decision ledger;
2. authenticated input paths, byte sizes, SHA-256 values, and the exact preprocessing
   and cross-implementation conformance receipt;
3. the frozen aligner, BLOSUM payload, runtime, traceback, and threshold configuration;
4. complete digest-only four-view edge ledgers and safe-prune accounting;
5. topology sensitivity, scaling dry-run, joint feasibility, exact solver model,
   solver log, every lexicographic optimum, and canonical
   tie-break trace;
6. outer and inner entity, pair, source-row, model-allowlist, and quarantine memberships;
7. frozen retrieval libraries, positives per query, finite chance, panel-shape values,
   every hard gate, and every mandatory bias diagnostic;
8. the independent exact-audit source hash, execution receipt, and edge count by view;
9. a data-firewall receipt proving that only fold-legal authority-train data were
   materialized by each model-facing extractor;
10. crossed-bootstrap and attainable-precision simulation source/configuration hashes
    and complete planning results; and
11. a single authorization Boolean that is false unless every requirement passes.

Every artifact path must be collision-free and every immutable artifact must use
no-clobber publication. A rerun may authenticate and return an existing identical
artifact; it may not replace it. Human-readable reports must be derived from the same
hashed machine-readable payload.

No alternative successful split may be kept private. All preregistered failures,
timeouts, infeasibilities, and audit crossings are scientific outputs and remain in the
ledger.

## Stop rules

Stop without training or model-score access if any of the following occurs:

1. an input, source, configuration, or authority-membership hash differs from the
   preregistered value;
2. any authority-validation or frozen-test field is read or materialized by the current
   experiment, recognizing that authority validation was inspected historically;
3. an exact all-pairs edge graph is incomplete, a non-mathematical prune is used, or an
   aligner contract differs across build and audit;
4. any optimization stage is infeasible, times out without a proof of optimality, has a
   nonzero gap, or cannot complete canonical tie-breaking;
5. an outer or inner retention, utility, balance, or score-free representativeness gate
   fails;
6. any direct crossing edge or exact endpoint overlap is found by the independent
   audit;
7. any exact pair or source-row group is split across incompatible roles;
8. a model process can access any endpoint, cache, index, coordinate, label, or source
   row outside the exact model allowlist, or accesses outer validation for early
   stopping, negative mining,
   checkpoint choice, calibration, architecture choice, or debugging;
9. the candidate grid, metric, tie breaker, epoch rule, or architecture changes after an
   inner or outer score is opened; or
10. the planning power/precision gate or the train-only algorithm-comparison gate fails.

After a stop, do not weaken thresholds, choose a favorable fold subset, try additional
split seeds, switch automatically to graph-cut removal, or inspect authority-validation
or frozen-test data. A scientifically justified change requires a new numbered protocol, a clear amendment
rationale, and a new freeze before any further score is observed.

## Claim boundary

Passing this protocol supports only the statement that nested development results were
obtained on authority-train pairs under a foldwise direct, exact, four-view
80%-identity/80%-reciprocal-coverage C3/S2-like boundary with disclosed fold-specific
quarantine. Any interval is conditional on the observed finite panel unless the explicit
dependence-unit criteria above pass.

It does not establish:

- strict transitive family or lineage independence;
- structural-interface, temporal, laboratory, publication, or protein-language-model
  pretraining independence;
- experimentally measured non-binding, affinity, therapeutic specificity, or
  prospective design success; or
- independent confirmation on authority validation, the untouched frozen test, or an
  external prospective dataset.

Authority validation is reused development confirmation and cannot supply the missing
independence. Biological specificity requires measured negative panels or a prospective
experiment; unobserved cross-pairs in this structural collection remain retrieval
decoys. Independent performance confirmation requires the untouched frozen test under a
separately frozen protocol or a prospective external dataset.
