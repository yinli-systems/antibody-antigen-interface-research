# Train-only nested partner-retrieval model protocol v1 draft

Status: prospective preregistration draft. This document is not frozen and does not
authorize training, model selection, authority-validation access, or frozen-test
access. Model execution requires an authorized split receipt from
`TRAIN_ONLY_FOUR_VIEW_DUAL_AXIS_NEIGHBORHOOD_EXCLUSION_PROTOCOL_V0.md`, a frozen
machine-readable implementation contract, and all feasibility and precision gates.

Date: 2026-07-23 CST

## Supersession and historical truth

For this future fold-local experiment, this document explicitly supersedes
`PARTNER_RERANKER_PROTOCOL_V0.md`. The older draft remains an immutable historical
artifact, but its seed-42 screen, candidate-level outer promotion, and fixed-boundary
promotion semantics are incompatible with nested selection and must not be used here.
It is not edited or silently reinterpreted.

The source authority-validation pool was inspected by an earlier ESM-2 exploration,
and its repaired subsets have already served as reused development confirmation. They
are not an untouched confirmation set. This protocol excludes them from all new split,
training, selection, calibration, and model-repair processes, but that prospective
firewall does not erase the history. The frozen SAbDab2 test remains untouched. Only a
separately frozen test protocol or a prospective external experiment can provide
independent confirmation.

A create-only history ledger must enumerate prior train screens, inspected score
families, authority-validation inspections, failed split proposals, audit failures,
protocol versions, and decisions informed by each event. The architecture family below
was motivated by those prior results and literature; it is not presented as discovery
from untouched data.

## Design basis

- [DataSAIL](https://www.nature.com/articles/s41467-025-58606-8) motivates treating a
  paired problem on both entity axes and disclosing discarded interactions; the present
  fold-local boundary is S2-like, not a claim of a global DataSAIL S2 partition.
- [CALM](https://www.biorxiv.org/content/10.64898/2026.02.25.707916v1) motivates
  symmetric antibody--antigen retrieval and stringent identity-controlled evaluation.
- [EpiCLIP](https://openreview.net/pdf?id=JImubudlk9),
  [PPLM](https://www.nature.com/articles/s41467-026-70457-5), and
  [PLM-interact](https://www.nature.com/articles/s41467-025-64512-w) motivate
  residue-level pair-aware interaction rather than marginal propensity alone. They do
  not validate the specific architecture or hyperparameters below.
- [Negative-class composition](https://www.nature.com/articles/s42256-025-01089-5)
  motivates explicit decoy terminology and measured-negative validation rather than
  treating absent structural pairs as non-binders.
- [Owen's pigeonhole bootstrap](https://doi.org/10.1214/07-AOAS122) motivates crossed
  row/column resampling. Its assumptions do not make this finite evolutionary sample
  independent, which is why conditional interval language remains mandatory.

## Scientific question and estimand

Does the prospectively fixed **inner-selection algorithm** improve bidirectional
antibody--antigen retrieval over the locked asymmetric dual encoder on authority-train
development panels that satisfy foldwise direct-80/80 four-view homology-separated C3,
S2-like boundaries?

The outer estimand is the performance of the complete algorithm that constructs
fold-local training artifacts, trains every allowlisted inner candidate, applies the
fixed inner decision rule, refits the selected configuration, and predicts one outer
panel. It is not the performance of a named reranker chosen after outer scores are
seen. An outer result cannot promote a candidate, modify a hyperparameter, choose an
epoch, or select a training seed.

The target is finite-panel cognate retrieval. Unobserved cross-pairs are ranking decoys,
not measured non-binders. Contact prediction is a mechanistic auxiliary endpoint, not a
substitute for retrieval or evidence of experimental binding.

## Mandatory split and precision prerequisites

Before any model process starts, all three outer and all nine inner assignments must be
create-only, immutable, and independently authorized under the split protocol. The
authorization suite must bind:

- exact authority-train input and membership hashes;
- exact preprocessing, four-view graph, assignment, quarantine, and audit hashes;
- every outer and inner `P_train`, `P_val`, `T_A`, and `T_G` membership;
- exact model-facing entity and source-row allowlists;
- frozen validation libraries, positives per query, and finite-library chance values;
- the topology sensitivity and joint outer-plus-inner feasibility proof; and
- the passed attainable-precision/power receipt for `Delta*=0.020`.

The boundary is named **foldwise direct-80/80 four-view homology-separated C3,
S2-like**. It must not be shortened to `global S2`, `S2 split`, or
`lineage-independent`.

## Hard fold-local model-data firewall

For any outer or inner training fold `h`, define

```text
A_model[h] = {a: exists g with (a, g) in P_train[h]}
G_model[h] = {g: exists a with (a, g) in P_train[h]}
R_model[h] = {source row r: exact_pair(r) in P_train[h]}
```

The complete accessible antibody inventory of the training process must equal
`A_model[h]`; the antigen inventory must equal `G_model[h]`; and any accessible
coordinate or label row must belong to `R_model[h]`. These are equality requirements,
not subset checks. They also prove `A_model[h] subset T_A[h]` and
`G_model[h] subset T_G[h]`. An allowed endpoint with no retained training pair is not
model-accessible.

The equality covers raw sequences, token files, frozen and learned embeddings,
coordinates, contact labels, pair labels, normalizers, batch statistics, samplers,
retrieval and hard-negative indexes, caches, temporary files, logs, diagnostics, and
precomputed features. A whole-pool embedding cache may not be mounted or filtered by a
model process. An audited controller must create a fold-local cache first, using a
leading-ID filter before parsing any sequence, coordinate, or label field, and publish
its exact ID-set and content hashes. The model container receives only that cache and a
read-only allowlist.

No endpoint in `A minus T_A[h]` or `G minus T_G[h]`, and no pair outside
`P_train[h]`, may be used as an unlabeled example, hard decoy, normalization row,
language-model cache entry, index member, or debugging case. A cache hit for an
out-of-allowlist digest is a hard failure even if the bytes are never used in a loss.

Training and evaluation are separate processes. The outer-evaluation process receives
one immutable checkpoint and exactly the frozen `P_val` libraries; it cannot update a
parameter, optimizer, scheduler, normalizer, index, threshold, calibration constant, or
checkpoint. The training process cannot mount outer-validation artifacts.

## Exact sequence and embedding preprocessing

Sequence normalization, entity framing, heavy/light chain order, missing-VL handling,
and source-row grouping are exactly those in the split protocol. No model-specific
replacement or imputation is allowed. VH and non-empty VL are encoded separately by
AntiBERTy and concatenated in heavy-then-light residue order after encoding. A learned
two-valued chain-type vector marks heavy versus light residues; no artificial separator
is treated as a residue. Missing VL contributes no token and no pooled zero vector.

The frozen antibody encoder is `AntiBERTy_md_smooth`, weights SHA-256
`f1ae33eac8cc8784a7d4be5a600141d2fa7bc7d8d5b3f5324d64a6a63bd0f137`. The frozen
antigen encoder is `facebook/esm2_t33_650M_UR50D`, weights SHA-256
`a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0`. Tokenizer,
configuration, code, special-token, dtype, and dependency hashes must be added to the
machine-readable contract before freezing. A missing or mismatched hash stops the
experiment.

Encoder special tokens are removed before residue pooling and contact indexing.
Attention masks, not zero-value inference, define valid residues. Projection is applied
to each residue before any mean. The antigen is divided into score-independent windows
of at most `960` residues with stride `800`; starts are the sorted deduplicated set
`{800k: k>=0 and 800k+960<L} union {max(0,L-960)}`.
Overlapping residue embeddings are averaged with equal window weight. No validation
score, predicted surface patch, structure prediction, or partner identity may choose a
window or truncation point.

All fold-local frozen-embedding arrays are deterministic, create-only, finite, and
indexed by canonical entity ID. Their manifest must prove exact equality with
`A_model[h]` and `G_model[h]`, tensor shape and dtype, row-order digest, encoder hashes,
and zero duplicate, missing, or extra IDs. PLM pretraining exposure remains a claim
limitation and is not removed by fold-local caching.

## Fixed candidate and training allowlist

Only the following ordered configurations may enter the nested selector. Their IDs,
capacity order, and numerical settings are immutable after the protocol is frozen.
There is no fourth model, width/depth grid, top-K grid, encoder-finetuning variant,
structure-prediction feature, score calibration, or post-score ablation candidate.

### `D0_dual_encoder_locked`

This is the locked asymmetric dual encoder reproduced independently inside every fold:

- frozen AntiBERTy antibody and frozen ESM-2 antigen encoders;
- residue projection followed by masked length mean;
- 128-dimensional shared L2-normalized representation and dot-product score;
- batch size `1024` exact positive pairs, after exact-pair collapse;
- AdamW, learning rate `0.001`, weight decay `0.2`, betas `(0.9,0.98)`, and epsilon
  `1e-8`;
- cosine warm restarts with initial period `10` epochs and multiplier `2`;
- exactly `58` epochs, no early stopping, and no checkpoint selection; and
- symmetric multi-positive contrastive loss over all positives present in the batch.

This candidate is both the lowest-capacity fallback inside the selector and the locked
outer comparator. A fold-specific baseline is trained from the same seed and legal
training inventory as its paired selected procedure. `R1` and `R2` must reuse the exact
same fold-and-seed `D0` coarse-checkpoint bytes that serve as their paired comparator;
training a second coarse checkpoint is forbidden.

### `R1_joint_reranker`

`R1` first fits the fold-local `D0` coarse retriever, then freezes it. It projects frozen
residue embeddings to width `256` and applies exactly two pre-layer-normalized
bidirectional cross-attention blocks. Each block has eight attention heads, a
width-`512` GELU feed-forward layer, residual dropout `0.10`, and attention dropout
`0.10`. Antibody-to-antigen and antigen-to-antibody modules do not share weights.

Masked means of the updated antibody and antigen residues are `m_A` and `m_G`. The pair
head receives `[m_A, m_G, m_A*m_G, abs(m_A-m_G)]`, then applies
`Linear(1024,256)`, GELU, dropout `0.10`, and `Linear(256,1)` to produce `r`. For a
reranked pair the final score is `s_coarse + 0.5*tanh(r)`; every non-reranked pair keeps
`s_coarse` exactly.

The evaluation rerank set is the union of the coarse top `64` antigens for each
antibody and the coarse top `64` antibodies for each antigen, using descending coarse
score and canonical digest as the tie break; if a library has fewer than `64` entities,
all are used. Evaluation positives are never forced into
the set. During training, every observed positive is added to its query's candidate
set so that the loss is defined; this training-only addition is recorded and is not an
evaluation rule.

### `R2_joint_reranker_contact`

`R2` is tensor-for-tensor identical to `R1` and adds one native contact head. Project
the updated residues to dimension `64` with separate linear maps and define the contact
logit for residue pair `(i,j)` as
`dot(W_A h_i, W_G h_j)/sqrt(64) + b`. The retrieval score remains the declared `R1`
pair head; contact labels cannot enter evaluation ranking or select a candidate library.

### Reranker optimization

Both rerankers use exactly `40` epochs, batch size `32` unique exact pairs, AdamW with
learning rate `1e-4`, weight decay `0.01`, betas `(0.9,0.98)`, epsilon `1e-8`, global
gradient-norm clipping at `1.0`, linear warmup for the first `5%` of integer optimizer
steps, and cosine decay to zero thereafter. There is no early stopping or intermediate
checkpoint evaluation.

For each query with candidate set `C` and complete observed-positive set `Y`, the
directional retrieval loss is

```text
-log(sum_{y in C intersect Y} exp(s_y / 0.07)
     / sum_{c in C not masked_ambiguous} exp(s_c / 0.07))
```

and the two directions receive equal weight. All observed positives are positive, never
decoys. An unobserved antigen is masked for antibody query `a` if it has a direct
`H_G` edge to any observed cognate of `a`; the antibody-side rule is symmetric using
`H_A`. This ambiguity mask is computed only from the legal train graph, is frozen
before fitting, and cannot use validation behavior.

For `R2`, total loss is `retrieval_loss + 0.10 * contact_loss`. Within each structurally
labeled complex, contact loss gives total weight `0.5` to resolved contacts and `0.5` to
resolved noncontacts by separate mean binary cross-entropies. A complex lacking either
class contributes no contact loss. Multiple structural source rows for one exact pair
are averaged within that pair before pairs are averaged, preventing source multiplicity
from becoming an optimization weight.

All three candidates are executed at the fixed seeds `{17,42,73}` in every inner fold.
Seeds set Python, NumPy, PyTorch CPU/CUDA, data-loader generators, parameter
initialization, candidate ordering, and deterministic-kernel settings. A nondeterministic
kernel or unrecorded fallback is a hard failure. Seeds are not tried sequentially and
the best seed is never selected.

## Contact-label preprocessing

Contact labels are available only to `R2` training on source rows in `R_model[h]`. The
split builder never reads them. The exact mmCIF files, parser, chain map, atom filters,
and source hashes must be frozen before training.

A residue pair is a contact if at least one resolved non-hydrogen atom from the
antibody residue is within `4.5` Angstrom of at least one resolved non-hydrogen atom
from the antigen residue. A resolved pair farther than `4.5` Angstrom is a contact-head
negative; an unresolved or ambiguously mapped pair is masked, not negative. Alternate
locations use highest occupancy, with label `A` then lexical label as deterministic
ties. Waters, non-polymer ligands, hydrogens, and atoms with missing coordinates are
excluded.

Polymer residues map to normalized model residues through the frozen mmCIF sequence
mapping. A residue with no unique mapping is masked. If more than `5%` of otherwise
resolved polymer residues on either partner cannot map uniquely, the complete structure
is excluded from contact loss but retained for retrieval. Exclusions and reasons are
published by exact-pair ID. Chunk-overlap contact logits are averaged at the residue-pair
level, yielding one full antibody-residue by antigen-residue matrix.

These noncontact residue pairs are not antibody--antigen non-binders. Experimental
binder/non-binder classification requires measured negatives under a separate protocol.

## Hard decoys and negative terminology

The `D0` index used to mine reranker candidates is built only from the exact fold-local
model allowlists and one immutable fold-local coarse checkpoint. It is hashed before
reranker fitting and is never refreshed in response to validation. Decoys are selected
by the fixed top-64 rule and the fixed homology-ambiguity mask above.

Every absent cross-pair remains an **unobserved retrieval decoy**. Reports may not call
it a non-binder, specificity negative, experimental negative, or failed interaction.
Biological specificity claims require measured negative panels such as a prospectively
frozen experimental library, plus prospective confirmation where appropriate.

## Nested selection algorithm

All inner candidates and seed runs are trained without outer-validation access. For an
inner fold and seed, the score is the arithmetic mean of complete-library Ab-to-Ag MRR
and Ag-to-Ab MRR. For a candidate, average the three seeds within each inner fold, then
average the three inner folds equally. Query counts never weight folds.

For candidate `c`, let `delta[c,j]` be its seed-mean bidirectional MRR minus the matched
`D0` value in inner fold `j`. A reranker is eligible only when its three-fold mean
improvement is at least `0.010` and `delta[c,j] > 0` for all three inner folds. `R2` is
additionally eligible only if its per-complex macro contact AUPRC exceeds contact
prevalence in every inner fold. This contact condition is a mechanism screen, not
evidence of binding.

If neither reranker is eligible, select `D0`. Otherwise find the highest three-fold mean
retrieval score; among eligible candidates within absolute MRR `0.002` of that maximum,
select the lower-capacity order `D0 < R1 < R2`, then lexical ID. Raw unrounded
double-precision values determine comparisons; published displays do not. This complete
decision rule is run once per outer nest.

All three outer nests complete their inner runs and publish selected configuration IDs
before any outer score is materialized. The selected configuration is then refit on all
of that outer fold's `P_train` at each of the three seeds and evaluated once on the
outer panel. Only `D0` and the output of the selection algorithm are evaluated; rejected
candidates receive no outer score.

Outer results estimate the selection algorithm. They cannot declare `R1` or `R2` a
winner, promote a candidate, alter the eligibility margin, add a seed, change top-K, or
repair an architecture. Selection frequencies and reasons are descriptive.

## Retrieval endpoints and controls

The primary endpoint is the equal-fold, equal-direction mean of complete-library MRR,
averaged over the fixed three training seeds. Required secondary endpoints are
direction-specific MRR, normalized mean and median rank, and Recall/Hit at
`1, 5, 10, 20`, with exact finite-library chance for every query. All positives for a
query are honored. Complete native libraries are primary; no candidate-size curve or
downsampled panel may replace them. All ranks order candidates by descending raw score
and canonical entity digest; scores are never rounded before ranking.

Required paired falsification controls are:

1. exact finite-library random expectation;
2. degree-preserving partner relabeling;
3. antibody-only and antigen-only masks;
4. antibody composition-preserving residue shuffle;
5. antigen composition-preserving residue shuffle;
6. raw frozen-embedding similarity;
7. a random untrained reranker head with matched shapes; and
8. length/amino-acid-composition unary scoring.

Controls use the same frozen libraries, positives, seed pairing, score order, and
bootstrap draws. No control result may tune the selected model. Partner relabeling and
residue shuffles use `10,000` create-only permutations from `PCG64DXSM` seed
`2026072303`. Degree-preserving relabel replicate `k` starts from the original simple
positive bipartite graph and an independent stream derived from
`SHA256("partner-relabel-v1" || k)` modulo `2^128`. It proposes `40*|P_val|` double-edge
swaps by choosing two distinct edges uniformly without replacement and exchanging their
antigen endpoints; a proposal that creates an existing edge or duplicate edge is a
self-loop and leaves the graph unchanged. Discard the first `20*|P_val|` proposals as
burn-in and use the final graph only if both degree sequences are exact and at least
`5*|P_val|` swaps succeeded; otherwise the entire permutation suite fails. This is a
locked sensitivity generator, not a claim of exact uniform sampling over graphs.

For each composition shuffle, Fisher--Yates permute residue positions independently for
each entity using a stream derived from its axis, canonical digest, replicate, and seed;
missing VL remains missing and VH/VL are never mixed. The amino-acid count vector of
every chain must remain exact. A failed invariant stops the suite rather than triggering
an analyst-chosen redraw. Every stream derivation, rejection count, and final membership
digest is published.

For `R2`, report residue-pair AUROC and AUPRC, per-complex macro AUPRC, prevalence,
contact-map permutation, single-partner permutation, mapped-residue coverage, and
excluded-structure counts. Marginal paratope or epitope scores are secondary projections
of the native matrix and cannot replace pair-level evidence.

## Statistics, multiplicity, and seed stability

The exact full-component crossed row/column pigeonhole bootstrap, fold blocking,
paired-difference construction, `20,000` replicates, RNG, invalid-replicate rule,
direction aggregation, q-redundancy sensitivity, lineage sensitivity, and interval
language are inherited without modification from the split protocol. The same resample
is used for the selected procedure, matched `D0`, and every control.

There is one primary train-only comparison: selection algorithm minus `D0` on the
equal-fold bidirectional MRR. A robust positive train-only result requires point
improvement at least `0.020`, a two-sided 95% paired lower limit above zero, and positive
seed-mean improvement in every outer fold. This is an internal algorithm gate, not
candidate promotion and not independent confirmation.

The two direction-specific retrieval effects form one Holm family when directional
claims are made. The eight learned-model-versus-falsification-control effects numbered
`2` through `8` above, counting the two single-side masks separately, form a second Holm
family; exact chance is a mathematical reference rather than a permutation hypothesis.
Contact AUROC, contact AUPRC, macro contact AUPRC, subgroup,
q-reduced, lineage, and seed analyses are secondary unless a new protocol assigns them
a multiplicity family before execution. Bootstrap p-values use plus-one counting.

Report every seed, arithmetic mean, sample standard deviation, range, and worst seed.
Three seeds are a fixed sensitivity set, not an inferential sample from a seed
population. Do not use a t interval over seeds or folds. Population-style CI wording is
forbidden unless the split protocol's full-component adequacy criteria pass; otherwise
use `conditional finite-panel stability interval` or a descriptive range.

## Final configuration without outer-score selection

After all three inner decisions are frozen but before any outer score is opened, choose
the prospective full-authority-train configuration as the modal inner-selected ID.
Ties use lower capacity and then lexical ID. This final ID and all three refit seeds are
create-only and cannot respond to outer performance.

Outer predictions may then characterize the nested algorithm, but they cannot change
the final ID. A later full-authority-train refit uses exactly that ID and the same three
seeds. Any ensemble must be declared before fitting; this protocol defines its score as
the arithmetic mean of the three seed-specific pair scores before ranking.

Authority validation may be reported only as already reused development confirmation
under its locked historical protocols; it cannot independently confirm this new model.
Access to the untouched frozen test is not authorized here. A separate frozen-test
protocol must bind the final ID, seed ensemble, checkpoint hashes, complete libraries,
controls, multiplicity, and one-shot registry before any test field or score is read.

## Immutable artifacts and execution receipts

Before the first model process, publish create-only:

1. this protocol and machine-readable implementation hashes;
2. split authorization, planning-power, history-ledger, and model-allowlist receipts;
3. fold-local cache manifests proving exact model inventory equality;
4. candidate source, architecture, optimizer, tokenizer, encoder, runtime, and seed
   hashes;
5. frozen ambiguity masks, hard-negative indexes, candidate-order ledgers, and contact
   preprocessing receipts;
6. all nine inner-assignment IDs, all `81` inner candidate-by-seed run IDs, every outer
   refit/evaluation ID, and the deterministic selection-rule implementation hash; and
7. collision-free checkpoint, prediction, score-matrix, control, bootstrap, and
   statistics paths with no-clobber publication.

Each completed checkpoint binds its exact fold, candidate ID, seed, allowlists, cache
hashes, source hash, runtime, history, and optimizer history. A resumed run must prove
contiguous epochs, coherent optimizer/scheduler/global-step state, finite tensors, and
identical scientific provenance. Evaluation hashes checkpoint bytes once, loads those
same bytes, and writes one immutable attempt receipt before validation materialization.

## Stop rules

Stop the current protocol without reinterpretation if:

1. any prerequisite, input, source, runtime, encoder, cache, or allowlist hash differs;
2. a model process can access an ID or row outside its exact `A_model`, `G_model`, or
   `R_model` inventory;
3. authority-validation or frozen-test data are accessed by the current experiment;
4. any outer or inner boundary, graph, gate, audit, or planning-power receipt fails;
5. an inner or outer score is used to redraw a fold, change a candidate, alter a
   hyperparameter, choose a seed, modify a mask, refresh an index, calibrate a score, or
   select a checkpoint;
6. a nondeterministic kernel, NaN/Inf, missing positive, malformed contact map, or
   incomplete candidate library is observed;
7. fewer than all required seeds, folds, directions, controls, or contact diagnostics
   complete; or
8. the train-only algorithm gate fails.

A failure is retained as a result. It cannot be repaired by relaxing homology,
retention, power, eligibility, effect-size, or confidence limits after scores are
visible. A scientifically justified change is a new numbered protocol and new
experiment with an explicit history-ledger amendment.

## Claim boundary

Passing supports only a rigorous internal statement that the fixed nested selection
algorithm improved finite-panel cognate retrieval on authority-train under foldwise
direct-80/80 four-view homology-separated C3/S2-like boundaries, subject to the stated
conditional dependence analysis and PLM pretraining exposure.

It does not prove that a particular reranker is universally superior, that full
lineages are independent, that an unobserved pair does not bind, that contact logits are
physical affinities, or that the method has therapeutic, prospective, temporal,
external, clinical, or world-leading validity. Biological specificity requires
measured negative panels and prospective experimental confirmation. Independent
performance confirmation requires the untouched frozen test under a separately frozen
one-shot protocol or a prospective external dataset.
