# AsEP epitope-group test measurement

Status: the single test measurement registered in
`protocols/ASEP_TEST_MEASUREMENT_V1.md`. One configuration, frozen by the
registered rule before the split was read, measured once.

Date: 2026-09-10

Machine-readable: [`test_measurement.json`](test_measurement.json).

## Convention, and a correction to it

AsEP reports "the average performance metrics across the test set samples" —
**per-complex averages**, not a pooled confusion matrix. An earlier version of
this record used micro (pooled) figures and compared them to the published
table. That was wrong, and it inflated every headline:

| | micro (pooled) | **macro (per complex)** |
|---|---:|---:|
| MCC | 0.1893 | **0.1572** |
| AUROC | 0.7515 | **0.6970** |
| precision | 0.1754 | 0.1889 |
| recall | 0.4073 | 0.4288 |
| F1 | 0.2432 | 0.2445 |

The two are not equivalent and cannot be close on this data: AsEP's antigens
differ widely in length and in epitope prevalence, so pooling weights long
antigens more heavily than the per-complex average does. **The macro column is
the comparable one.**

Thresholds are selected on validation and applied unchanged to test, each
convention selecting its own (micro 0.847, macro 0.843). Both sit high, the same
shift seen on sabdab2.

## Result, ten seeds, per-complex convention

| | mean | 95% CI |
|---|---:|---|
| MCC | **0.1572** | [0.1538, 0.1608] |
| precision | 0.1889 | [0.1817, 0.1947] |
| recall | **0.4288** | [0.4015, 0.4642] |
| F1 | **0.2445** | [0.2420, 0.2473] |
| AUROC | 0.6970 | [0.6939, 0.7002] |

169 test complexes; AUROC is defined on all of them.

## Against the published table

Epitope-group split. Published figures for WALLE differ between sources — the
AsEP paper and later re-tabulations disagree — so both are shown and the
comparison is not rested on either.

| method | MCC | AUROC | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| WALLE (arXiv 2608.01092 Table 2b) | 0.077 | 0.544 | 0.143 | 0.266 | 0.145 |
| WALLE (later tabulation) | 0.152 | 0.596 | 0.207 | 0.299 | 0.204 |
| PEPNet | 0.139 | 0.612 | 0.230 | 0.143 | 0.177 |
| PEPNet+LE | 0.156 | 0.627 | **0.250** | 0.155 | 0.191 |
| GF3DRoPE | 0.159 | **0.706** | 0.212 | 0.216 | 0.212 |
| LF3DRoPE | **0.171** | 0.690 | 0.248 | 0.189 | 0.215 |
| **this work** | 0.157 | 0.697 | 0.189 | **0.429** | **0.245** |

**This is not above the published best.** On MCC it sits level with PEPNet+LE
and below LF3DRoPE; on AUROC it sits between LF3DRoPE and GF3DRoPE. Where it
leads is recall, by a wide margin, and F1. Where it trails is precision.

The honest characterisation is a **different operating point at comparable
overall quality**: this model finds far more of each epitope and is wrong more
often when it calls a residue positive.

## Ensembling the ten seeds

The ten checkpoints were already trained and their predictions already on disk,
so averaging them costs nothing. The aggregation and the threshold were both
selected on validation before the test figures were read.

| | single model | **ten-seed ensemble** |
|---|---:|---:|
| MCC | 0.1572 | **0.1743** |
| precision | 0.1889 | 0.1901 |
| recall | 0.4288 | **0.4836** |
| F1 | 0.2445 | **0.2579** |
| AUROC | 0.6970 | **0.7088** |

With the ensemble this work is at or above the published best on MCC (0.171),
AUROC (0.706), F1 (0.215) and recall (0.299), and still below on precision.

**This is not a like-for-like comparison.** The published figures are single
models reported with a seed standard deviation; ten averaged models are a
different object and the row is labelled accordingly. The single-model row is
the one to compare.

### The ensemble saturates at two, and the seed spread is the real finding

Measured on validation, averaging more seeds stops helping almost immediately:

| ensemble size | validation macro MCC |
|---:|---:|
| 1 | 0.2591 (range 0.1996 – 0.3185) |
| 2 | 0.3036 |
| 3 | 0.2967 |
| 5 | 0.2953 |
| 7 | 0.3024 |
| 10 | 0.3021 |

Two seeds capture the gain; three to ten are flat within noise. Training more
seeds would add nothing, so this lever is exhausted.

**The spread across single seeds is the part worth keeping.** Validation macro
MCC ranges 0.1996 to 0.3185 — the best seed is 1.6x the worst. The ensemble gain
is largely that variance being averaged away.

This bears directly on the comparison. LF3DRoPE is reported as 0.171 ± 0.010. If
that is a seed standard deviation, its variance is an order of magnitude below
what this cohort produces, which would itself need explaining; if it is over
something else — folds, say — then the error bars in the two rows are not the
same quantity and a gap of 0.003 between point estimates means very little
either way. **Neither has been checked**, and until it is, the point estimates
should be read as coarser than their decimals suggest.

### A selection that did not survive

Four ways of deriving a node score were tried: the model's own epitope head, and
the maximum, mean and top-three mean of its pair predictions over each antigen
residue's partners. Read off the test split, `top3` looked best at MCC 0.1764.
Selected on validation, as it must be, `head` wins — validation MCC 0.3024
against 0.2974 — and its test MCC is 0.1743.

The apparent 0.002 gain was selection on the split being estimated. The
ensembling gain of 0.017 survives the same discipline. Both are recorded because
the contrast is the useful part.

## Supervision parity: the advantage is not the richer training signal

LF3DRoPE trains on antigen residue labels alone. This model trains on residue
pairs and derives node marginals, so it uses strictly more of the same data, and
until this was tested the comparison could not separate a better method from a
richer signal. `protocols/ASEP_SUPERVISION_PARITY_V1.md` fixed what each outcome
would mean before the arm ran.

Switching off the edge loss and the partner-rank loss, leaving only the four node
terms, changes essentially nothing:

| test, per complex | pair-supervised | **node labels only** |
|---|---:|---:|
| MCC | 0.1572 [0.1538, 0.1608] | **0.1590** [0.1546, 0.1631] |
| precision | 0.1889 | 0.1953 |
| recall | 0.4288 | 0.3973 |
| F1 | 0.2445 | 0.2447 |
| AUROC | 0.6970 | 0.6976 |

MCC is marginally *higher* without the pair supervision and the intervals
overlap heavily. **The comparison to the published table therefore stands at
parity of supervision**, and the node-only row is the one to quote against
methods trained on node labels.

### An unexpected finding worth separating out

Pair supervision contributes nothing to the node metric. The edge loss carries
weight 1.0 and partner-rank 0.5, together the majority of the objective, and
removing both leaves every node figure where it was.

That is not the same as saying pair information is empty: on the sabdab2 cohort
the pair head beats the predicted-marginal outer product by 14.7% of baseline at
20/20 seeds, so the pairing does carry signal there. Two readings survive:

1. the node head learns everything it needs from node labels, and the pair
   machinery is redundant **for this endpoint**;
2. the pair information is real but does not reach the node head through this
   architecture.

The sabdab2 result makes the second more likely, and it points at a concrete
architectural gap rather than a null. Either way, this is direct evidence
against adding pair modelling in order to improve a node-level epitope
prediction.

## What this does not establish

1. **This compares against published numbers, not re-run models.** No baseline
   was re-executed here. Preprocessing differences, the definition of an epitope
   residue, and the corrected against original split file (169 against 170 test
   instances) could each absorb part of the difference.
2. **The problem setup may differ.** This model predicts residue **pairs** and
   derives node marginals from them, so it trains on which antibody residue
   contacts which antigen residue. A method trained on node labels alone uses
   strictly less of the same data. That is a difference in setup, not in
   architecture, and it has not been checked against each published method.
   Until it is, the defensible statement is that this number is higher in this
   setup, not that the method is better.
3. **Precision is lower and MCC is not higher.** For a use that needs every
   reported residue to be right, LF3DRoPE and PEPNet+LE are preferable. The
   advantage here is recall and F1.
4. Ten seeds of one configuration on one split. The `epitope_ratio` split was
   not measured.

## Provenance

| | |
|---|---|
| protocol | `protocols/ASEP_TEST_MEASUREMENT_V1.md`, with its amendment |
| arm | `esm2_esmif1`, frozen on validation before test was read |
| selection | `results/asep_benchmark_v1/arm_selection.json` |
| split | AsEP `epitope_group`, sha256 `678a20cd…` |
| test instances | 169 |
