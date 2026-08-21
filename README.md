# Partner-aware antibody-antigen interface research

This private repository is an evidence-gated research codebase for antibody-
antigen interface prediction on SAbDab2-derived data. It contains importable
models and metrics, data and split contracts, cluster launchers, immutable
protocols, and lightweight result receipts. It does **not** contain the raw
datasets, model weights, full embeddings, or complete prediction tensors.

The central research question is stricter than marginal binding-site prediction:
does a model identify the residues that interact for a **specific antibody-
antigen pair**, under dual-axis leakage control, rather than learning which
residues are generally interface-prone?

## Status at a glance

| Workstream | What is implemented | Current evidence boundary |
|---|---|---|
| SPPIDER-seq adaptation | SAbDab2 adapters, training/evaluation, cluster bootstrap, calibration, sequence-similarity audit, and matched-partner permutation | Locked reused-benchmark marginal result; not native residue-pair prediction and not a pristine blind test |
| MIPE reconstruction | Released-code-compatible five-fold evaluation, directional interaction export, pair-level metrics, OOF-only calibration, and a bound-geometry oracle audit | The completed `lr=1e-3` replay is not Appendix-exact; the fresh `lr=1e-4`, five-fold, 800-epoch replay is still running |
| Intra-chain geometry | Distance, receiver-frame direction, and full relative-pose attention biases with parameter-unshared antibody/antigen encoders | Two-A100 replay and three historical seeds are development evidence; the preregistered five-seed superiority gate did not pass and the outer test remains unread |
| Partner retrieval and PairScale | Dual encoders, bidirectional retrieval, native pair/contact models, hard-negative and one-sided controls, calibration, and fixed-budget runners | Development-only experiments did not pass the required partner-specificity gates; no candidate is promoted |
| Partner-geometry R2 / PAIRFormer | Explicit antibody-by-antigen pair state, bidirectional fusion, axial pair updates, contact/distance/orientation heads, losses, evaluator, execution plan, and one-shot freeze controls | Implementation complete; biological training, selection, and outer/test evaluation remain blocked at Gate 0 |
| V6 score-free split authority | Raw-byte authority reconstruction, independent auditor, exact statistics, joint IR, CP-SAT formulation, separation oracle, HiGHS projection backend, and V6-B/V6-C diagnostics | The linear projection was reported infeasible by the primary and independent CP-SAT paths, but the required direct-HiGHS terminal receipt is not committed; this does not authorize a full-protocol infeasibility claim or training |
| Distributed systems | Deterministic batching, distributed training helpers, immutable checkpoint/recovery, NCCL probes, and Slurm launchers | A real-OAS ABF-35M 4-to-2-to-4 DDP/DCP gate passed with exact terminal model state and bounded FP32 optimizer-moment drift; no 650M performance or 32/64-GPU scaling claim |
| Antibody foundation pretraining | Authenticated OAS materialization, exact checkpoint recovery, memory-mapped loading, and compile/eager confirmation tooling | The eight-RTX-5090 compile candidate improved measured pilot throughput but failed frozen loss and tail-latency gates; it is disqualified, with no production-scale or 16/32-GPU claim |

The authoritative gate manifest is
[`protocols/pairformer_gate_manifest_v1.json`](protocols/pairformer_gate_manifest_v1.json):
Gate 0 is blocked, and Gates 1--4 have not started. Model scale must not be used
to bypass missing split, provenance, leakage, or partner-dependence evidence.

## Headline scientific evidence

### SPPIDER-seq: strong paratope marginals, weak epitope specificity

The locked ensemble was evaluated on 455 reused benchmark complexes grouped
into 84 canonical test clusters. Thresholds and the final recipe were selected
on validation before this evaluation.

| Axis | AUROC | AUPRC | MCC | Cluster-bootstrap AUPRC 95% CI |
|---|---:|---:|---:|---:|
| Paratope | 0.948622 | 0.651089 | 0.602334 | 0.626906--0.678340 |
| Epitope | 0.671765 | 0.145445 | 0.126885 | 0.121265--0.166848 |

Matched-minus-permuted partner AUPRC was `+0.011850` for paratope
(`95% CI 0.000212--0.026145`) and `+0.002915` for epitope
(`95% CI -0.000489--0.006323`). The epitope interval crosses zero. The result
therefore supports useful marginal interface propensity prediction, especially
on the antibody side, but only limited partner identity dependence.

See the
[`final research report`](results/sppider_seq/FINAL_RESEARCH_REPORT.md) and
[`machine-readable summary`](results/sppider_seq/final_research_summary.json).

### MIPE: completed released-code replay, with the Appendix-exact replay pending

The completed five-fold released-code-compatible `lr=1e-3` ensemble scores
epitope AUROC/AUPRC/MCC `0.892775/0.513218/0.317949` and paratope
`0.875432/0.669642/0.492002` when validation-best checkpoints are fixed before
provided-test scoring. OOF-only threshold calibration raises the corresponding
MCC values to `0.484753` and `0.509856`.

Pair-level evaluation remains the limiting result. Native mean-bidirectional
attention has macro/pooled AUPRC `0.005535/0.002080`, Recall@1 `0`, and MRR
`0.008100`; the marginal residue outer-product baseline is much stronger at
`0.073666/0.041906`, Recall@1 `0.078125`, and MRR `0.207587`. A bound-coordinate
oracle reaches pair macro AUPRC `0.485569`, Recall@1 `0.609375`, and MRR
`0.785156`, but it directly uses the common bound-complex coordinate frame and
is not a deployable unbound or sequence-only predictor.

See the
[`audited report`](results/mipe/MIPE_OFFICIAL_REPLAY_20260821.md) and
[`machine-readable release`](results/mipe/MIPE_OFFICIAL_REPLAY_20260821.json).
The supplied split had pre-scoring schema access and is not pristine blind
evidence. The released `main.py` learning rate (`1e-3`) also differs from
Appendix section 2.3 (`1e-4`); a fresh five-fold, 800-epoch Appendix-correct
replay is running and has no terminal metrics yet.

#### Earlier records retained for provenance

The PAIRFormer protocol records a historical five-fold 800-epoch reconstruction
with native mean-bidirectional pair macro AUPRC `0.00342`
(`95% CI 0.00305--0.00381`), a predicted-marginal outer-product baseline of
`0.04045`, and validation-best epitope micro AUPRC `0.2804`. Those numbers are
preserved as historical protocol evidence in
[`PAIRFORMER_EXECUTION_PROTOCOL_V1.md`](protocols/PAIRFORMER_EXECUTION_PROTOCOL_V1.md)
and the gate manifest.

An earlier standalone MIPE result payload,
[`MIPE_INTERACTION_100E_SUMMARY.json`](results/mipe/MIPE_INTERACTION_100E_SUMMARY.json).
records a distinct five-fold 100-epoch held-out export: 11,431,418 residue
pairs, 18,316 positives, and pooled mean-bidirectional AUROC/AUPRC
`0.507190/0.001719`. It uses a different checkpoint budget and aggregation
scope from the new provided-test replay and must not be combined with it.
Complete 800-epoch checkpoints, raw predictions, and bootstrap artifacts are not
stored in Git, so the 800-epoch statement is not independently reproducible from
this repository alone.

The supplied external split had pre-scoring schema access and must be described
as a **provided external test-split evaluation with pre-scoring schema access**,
not as pristine blind evidence. Released code, paper, and appendix settings also
differ in learning rate, checkpoint selection, graph construction, and dataset
counts; the repository preserves those discrepancies instead of silently
reconciling them.

### Antibody foundation pretraining: throughput gain rejected by correctness gates

The authenticated OAS corpus contains 224,887,833 unique training records and
25,259,993,406 unique training residue tokens. Exact checkpoint recovery on the
real corpus passes, and a protocol-bound 1,024-array memory-mapped loader reduced
measured continuous-run data preparation from 19.19 seconds to 3.97 seconds in
the recovery gate.

An eight-RTX-5090, 600-update, two-repetition compile-only confirmation measured
202,795 useful residue tokens/s and 46.1% approximate 6N MFU, versus 150,991
tokens/s and 34.3% for eager execution. The candidate remains disqualified:
post-training masked cross-entropy was 2.03% worse than eager against a frozen
1% noninferiority margin, and compiled p95/p50 step-time ratios were 1.137 and
1.128 against a 1.10 ceiling. The failed gate, receipts, and telemetry are kept
under `results/pretraining/antibody_compile_only_confirmation_v1_1343024/`.
No production-scale pretraining or 16/32-GPU result is claimed from this pilot.

### Intra-chain geometry: a positive development signal that did not clear its gate

The recommended phase-one encoding is distance RBF plus receiver-frame direction
as an additive per-head self-attention bias. Antibody and antigen use
parameter-unshared encoders, and geometry is computed only within each chain.
No inter-chain coordinate frame is assumed.

On the exact A100 final checkpoints, per-complex joint macro AUPRC was `0.492936`
for distance+direction and `0.483061` for topology-only, a mean difference of
`+0.009876` (`component-bootstrap 95% CI 0.001836--0.016695`). The corresponding
Holm-adjusted one-sided p-value was `0.082492`, so the strict single-seed gate
failed. Three historical fixed-budget seeds all had a positive pooled joint
delta, but they are not the five preregistered confirmatory seeds. The frozen
outer test was not read.

See the
[`geometry report`](results/sabdab2_structure_self_attention/FINAL_GEOMETRY_RESEARCH_REPORT.md)
and [`tensor contract`](docs/intra_chain_geometry_attention.md).

### Partner retrieval, PairScale, and R2: negative development evidence before scale

The locked mean-pooled retrieval model did not beat its strongest raw-embedding
and single-side controls with the required paired confidence intervals. Later
PairScale contact, centering, multi-decoy, and unary-control experiments are
retained as development evidence, including apparent train-set fits, but their
required partner-specificity checks failed. They do not authorize model
selection, test access, or a performance claim.

The R2 code addresses the correct next task: construct an explicit
`[L_ab, L_ag, C_pair]` state, update it with partner-conditioned fusion, and
predict a native residue-pair contact matrix. Its training path is deliberately
blocked until the exact dual-axis split and all independent Gate 0 audits pass.
The current V6 evidence is compatible with an infeasible **linear projection**;
it is not yet a complete proof about every nonlinear score-free gate. No
threshold was relaxed and no test data were used to resolve the split.

## Code architecture

Reusable code lives in [`src/interaction_first/`](src/interaction_first/). The
small export surface in `interaction_first.__init__` covers the stable contact,
split, partner-sampling, frozen-embedding, structure-cache, and intra-chain
geometry interfaces; research-specific modules are imported explicitly.

| Module family | Responsibility |
|---|---|
| `contacts`, `geometry_labels`, `pair_geometry_schema`, `pairbench_schema` | Fail-closed sparse-contact, geometry-label, and benchmark contracts |
| `intra_chain_geometry`, `sabdab2_structure` | Local `N-CA-C` frames, invariant same-chain pair features, geometry-biased attention, and authenticated structure batches |
| `model`, `pair_geometry`, `pair_geometry_losses`, `pairformer_*` | Partner-conditioned residue fusion, explicit pair state, native contact/distance/orientation decoding, training-step provenance, and full-matrix evaluation |
| `retrieval`, `asymmetric_retrieval`, `partners`, `partner_controls` | Bidirectional partner retrieval, multi-positive objectives, false-negative-aware sampling, and matched-partner controls |
| `pairscale_*`, `development_*` | Train-only development cohorts, fixed-budget screens, one-sided controls, and cluster-aware statistics |
| `splits`, `nested_joint_milp`, `score_free_*`, `v6_*` | Dual-axis components, raw-source score-free authority, independent auditing, exact joint formulations, and solver evidence |
| `frontier_*` | Distributed batching, model construction, training, checkpoint/recovery, and parity helpers |
| `data_registry`, `data_admission`, `source_materializer`, `license_evidence` | Content-addressed source admission, provenance, and license enforcement |
| `avida_*` | Independently audited AVIDa annotation and normalization protocols |

Operator entry points are under [`scripts/`](scripts/README.md), scheduler
templates under `slurm/`, frozen governance under
[`protocols/`](protocols/README.md), and lightweight evidence under
[`results/`](results/README.md). The data registry is provenance metadata, not a
substitute for the licensed source data.

## Repository navigation

| Goal | Entry point |
|---|---|
| Understand directory roles and preservation rules | [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) |
| Reproduce the interaction-first development pipeline | [`docs/INTERACTION_FIRST_PIPELINE.md`](docs/INTERACTION_FIRST_PIPELINE.md) |
| Review active and historical protocol evidence | [`protocols/README.md`](protocols/README.md) |
| Locate result summaries and interpretation boundaries | [`results/README.md`](results/README.md) |
| Select an operator or Slurm entry point | [`scripts/README.md`](scripts/README.md) |
| Verify artifact and test-set provenance | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| Review third-party sources and licenses | [`THIRD_PARTY.md`](THIRD_PARTY.md) |

Some files are intentionally mirrored across `protocols/` and `results/`.
Receipts and validators may bind both their bytes and their paths. A newer
filename does not automatically supersede an older artifact.

## Installation and verification

Python 3.10 or newer is required. Solver development is pinned to
`highspy==1.15.1` and `ortools==9.15.6755`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'

# Add exact-solver dependencies only when running the V6 backends.
.venv/bin/python -m pip install -e '.[solver]'

# Portable syntax and shell checks.
.venv/bin/python -m compileall -q src scripts tests
find scripts slurm -type f \( -name '*.sh' -o -name '*.sbatch' \) \
  -exec bash -n {} +

# Full repository test suite.
.venv/bin/python -m pytest -q
```

The full local audit for this README read all 816 tracked files at base commit
`2e317823`, including 814 decodable text files (567,031 lines) and two binary
artifacts. It parsed 324 Python files with no syntax/AST errors, 309 JSON files
with no parse errors, 52 JSONL files with no parse errors, and 43 Markdown files
with no broken local links. All 75 tracked shell/Slurm scripts passed `bash -n`.

On macOS with Python 3.12, the current complete `pytest` run produced 1,330
passes, 39 failures, one expected skip for unavailable official SPPIDER weights,
and 37 passing subtests. Thirty-seven failures are path-bound historical
retrieval/closure integration tests whose original `outputs/` or colocated
script layout is absent from the curated tree; one is the intentional Biopython
`1.83` parity lock running against local `1.87`; one Direct-HiGHS test failed in
the full run but passed immediately when rerun alone, so it is recorded as an
order/environment-sensitive failure rather than a clean pass. These failures
must be resolved or replayed in their authenticated environment before the
portable suite can be called green.

## Claim and access boundary

- SPPIDER-seq predicts marginal paratope and epitope probabilities, not a native
  antibody-residue by antigen-residue interaction matrix.
- The adaptation uses concatenated VH/VL sequences and 4.5-A contact labels;
  these differ from the original SPPIDER-seq task semantics.
- The antibody side remains highly homologous across the historical split:
  97.58% of test instances have a high-coverage training neighbor at or above
  70% identity.
- The 455-case benchmark was used in earlier evaluations and is not a pristine
  one-shot confirmatory test.
- Unknown cross-pairs are not measured non-binders. They may supervise ranking
  controls, but they must not be assigned fabricated all-zero contact labels.
- No outer/test evaluation is authorized for R2 while Gate 0 is blocked.
- No clinical, prospective, external-generalization, docking-utility, 32/64-GPU
  scaling, or global state-of-the-art claim is supported by the tracked evidence.

## Preservation policy

Raw datasets, weights, embeddings, full prediction matrices, third-party source
trees, credentials, and cluster identities are intentionally excluded from Git.
Obtain upstream assets under their own licenses and verify pinned revisions and
hashes before use.

Evidence is append-only. Do not rewrite or delete a failed attempt, timeout,
superseded protocol, or mirrored receipt. Corrections require a new version or
numbered attempt with an explicit supersession link. Model selection and
threshold selection must remain development-only; a frozen test may be opened
only after the governing protocol and external freeze receipt authorize it.
