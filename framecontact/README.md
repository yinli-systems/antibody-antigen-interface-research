# FrameContact

Research-grade, fail-closed code for geometry-aware, partner-specific
antibody-antigen modeling under antigen structural novelty. The original
intra-chain self-attention ablation remains intact and is now the first stage of
an explicit residue-pair contact and contact-gated cross-attention model.

The package consumes frozen per-residue ESM embeddings plus real `N/CA/C`
coordinates, integer chain IDs, residue masks, and explicit valid/missing-frame
masks. Antibody and antigen use separate model parameters and may be expressed in
unrelated global coordinate frames.

## Implemented experiment matrix

| Name | Graph | Geometry features |
|---|---|---|
| `sequence_dense` | dense same-chain | none |
| `sequence_local` | same-chain sequence window | none |
| `spatial_topology` | same-chain kNN/radius | none |
| `distance` | same spatial graph | distance RBF |
| `distance_direction` | same spatial graph | distance + local direction |
| `full_pose` | same spatial graph | distance + direction + relative rotation |
| `frozen_frame_ipa` | same spatial graph | learned invariant local-frame points |
| `framecontact` | full-pose monomers + late contact-gated cross-attention | explicit pair contacts |
| `framecontact_interleaved` | full-pose monomers + per-layer contact-gated cross-attention | explicit pair contacts |
| `framecontact_geometry_ablated_contact_hybrid` | dense monomer encoders + partner-aware binary contact/cross-attention | full-pose path present but frozen at exact zero |
| `framecontact_geometry_ablated_distogram` | dense monomer encoders + partner-aware distogram/cross-attention | full-pose path present but frozen at exact zero |

Two development-only innovations, `local_pose_multibasis` and
`local_pose_rotary`, are implemented behind a separate hash-bound exploratory
protocol. They cannot be selected by the frozen eight-condition trainer.

The pair-MLP ablations keep the same geometry tensor width, modules, and
parameter count. The invariant-point baseline is a separately reported
architecture family because its bias parameterization differs.

## FrameContact architecture

`FrameContactModel` adds four auditable paths:

1. dense sequence self-attention for antibody and antigen with no shared
   parameters;
2. a zero-gated spatial/local-frame residual path using distance, local
   direction, or full relative pose;
3. an explicit low-rank antibody-by-antigen contact head supervised by genuine
   pair labels;
4. two independently parameterized cross-attention directions gated by the
   initial predicted contact probabilities, never by raw coordinates between
   unaligned monomers; the same contact head then scores the interaction-updated
   features to produce the final contact matrix.

Geometry, cross-attention, and contact-to-residue marginal residuals are
zero-initialized, so the initial residue path exactly preserves the dense
baseline. The contact head still receives direct gradients at initialization.

The contact cache adapter authenticates `pairs.parquet`, requires one instance
per row group, proves a complete Cartesian candidate grid, and never reconstructs
pair labels from residue marginals. Wrong partners receive a ranking loss; they
are not fabricated as all-negative contact matrices.

## Pre-registered primary endpoint

Protocol v3 froze the full confirmatory design before SAbDab2 antigen-fold
fitting. Runtime-only amendments v4 and v5 then reduced the batch size, enabled
autograd saved-tensor CPU offload, and defined observed-class-balanced BCE for a
genuine all-negative contact complex. Both amendments were made before any
SAbDab2 antigen-fold validation prediction was produced; they preserve the
hypotheses, split, losses, model-selection rule, eight-condition matrix, and five
seeds.

The primary endpoint is **fixed antigen, cognate versus wrong antibody**
retrieval; the reciprocal fixed-antibody endpoint is required and cannot replace
a failed primary endpoint. Both panels have 31 metadata-matched candidates.
Specificity is scored only from the jointly conditioned pair-interaction
component; antibody unary and antigen unary terms remain available for
calibrated cognate contact prediction but cannot establish partner specificity.
Checkpoint selection maximizes the weaker of the two validation directional
margins.

Residue-pair contact AUPRC, epitope/paratope AUPRC, calibration, both partner
shuffles, spatial enrichment, and causal geometry interventions remain mandatory.
Protocol v1 is preserved unchanged as provenance, but the first real pair run
must use v2 unless a result is explicitly labeled as a v1-only reproduction.

See [`docs/partner_specific_reply_and_research_20260730.md`](docs/partner_specific_reply_and_research_20260730.md),
the scientific authority in
[`protocols/framecontact_primary_v3.json`](protocols/framecontact_primary_v3.json),
and the current runtime amendment in
[`protocols/framecontact_primary_v5.json`](protocols/framecontact_primary_v5.json).

## What is measured

The training script saves fixed-final and development-best checkpoints, epoch
history, per-complex validation predictions, source/checkpoint hashes, and
head-resolved attention diagnostics. Diagnostics include entropy, conditional
C-alpha distance, spatial mass enrichment, sequence-local mass, geometry-bias
magnitude, and a same-graph counterfactual that removes only geometry bias.

The confirmatory matrix is seeds `17/42/73/101/137`, at most 200 epochs, and
validation-only early stopping with patience 20. The multi-seed aggregator
validates every prediction axis and digest and reports all seeds without
suppressing failures.

`scripts/aggregate_attention_diagnostics.py` separately aggregates final-epoch
entropy, spatial-neighbor enrichment, expected C-alpha distance, additive-bias
counterfactual sensitivity, and sequence-local mass. It preserves every
seed-by-head value and emphasizes within-seed head dispersion because head
indices are permutation-symmetric and are not semantically aligned across
independent seeds.

## Residue-head and distogram exploratory matrix

The teacher-requested residue prediction comparison is isolated from the
eight-condition primary architecture matrix. Five seeds compare unary-only
residue heads, direct noisy-OR contact marginals, and a zero-gated hybrid of the
two. A second five-seed matrix compares the binary hybrid with a single
categorical distogram whose registered near bins exactly define contact
probability, with and without a sparse pose-free reverse-triangle constraint.

The triangle constraint consumes only authenticated intra-chain distances. It
is routed to expanded IMGT CDR residues and a solvent-exposed antigen patch
seeded by the antigen encoder's intra-chain epitope-unary score; it never forms
a coordinate difference between independently framed monomers. Every epoch
stores both raw-logit-zero precision/recall and the separately named
validation-selected operating point. The final aggregator authenticates all
result/checkpoint receipts and reports only epochs shared by all five seeds.

This matrix is development-only. An interrupted preprocessing job read pair
labels for a subset of the current SAbDab2 test role before it was cancelled;
it did not read model predictions or test metrics, but the current test can no
longer support a procedural never-read claim. A new independent confirmatory
set is required before any generalization or benchmark-improvement statement.
See [`docs/DISTOGRAM_TRIANGLE_RESEARCH_NOTE.md`](docs/DISTOGRAM_TRIANGLE_RESEARCH_NOTE.md).

## Local verification

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

The authoritative CUDA-environment test count is recorded in the immutable
source handoff receipt. Test success verifies code behavior only; it is not a
biological benchmark result.

## Authenticated Boltz-1 Pairformer adapter

The pinned official Boltz source and official Boltz-1 checkpoint are now
authenticated, and the 48-block Pairformer trunk is extracted into a receipt-
bound artifact. `OfficialBoltzPairformerMonomerEncoder` converts the same real
`N/CA/C`, chain-ID, and missing-frame contract used by FrameContact into
SE(3)-invariant intra-chain pair features. The dual wrapper creates two distinct
antibody/antigen module instances; their parameter storage is not shared.

The strict local smoke loaded all 2,544 tensors (147,400,704 frozen Pairformer
parameters) and completed a finite synthetic forward. Reproduce it with:

```bash
python scripts/smoke_boltz1_pairformer_adapter.py \
  --artifact /path/to/boltz1_pairformer.pt \
  --receipt /path/to/BOLTZ1_PAIRFORMER_EXTRACTION_RECEIPT.json \
  --boltz-source-root /path/to/pinned/boltz \
  --output /new/path/BOLTZ1_PAIRFORMER_SMOKE.json \
  --device cpu
```

See [`docs/PAIRFORMER_BASELINE.md`](docs/PAIRFORMER_BASELINE.md) and
[`baselines/BASELINE_LOCK_V4.json`](baselines/BASELINE_LOCK_V4.json). This is an
adapted pretrained trunk, not native Boltz reproduction, a five-seed
identical-split result, an A100 performance result, or a biological benchmark.

## GPU execution and A100 handoff

The original residue-only matrix remains available below. The five-seed
FrameContact development runner and exact authority order are documented in
[`docs/AB03_FRAMECONTACT_RUNBOOK.md`](docs/AB03_FRAMECONTACT_RUNBOOK.md).

The current authenticated formal development job is running on a ParaCloud
**RTX 4090**, not on AB03 and not on an A100. Its source tree is read-only and
bound to a per-file manifest. No A100 training or biological-quality result may
be inferred from the ParaCloud run.

A separate AB03 A100 synthetic attention-kernel replication is complete. It
passed reference-versus-SDPA FP32/BF16 output and query/key/value/additive-bias
gradient checks and measured three shapes through 231 x 2363 residues. On that
runtime, the optional SDPA path reduced allocated training peak memory by
`6.1-8.6%`; median synthetic train-step speedup ranged from `1.007x` to
`1.068x`. This is shape-specific systems evidence only. See
[`protocols/framecontact_attention_kernel_a100_addendum_v1.json`](protocols/framecontact_attention_kernel_a100_addendum_v1.json)
and `receipts/attention-kernel-a100-v3-ab03/`.

The five AsEP `frozen_frame_ipa` seeds in the original array failed before
producing complete run receipts because the naive point-distance expression
materialized a `[B,L,L,H,P,3]` CUDA tensor. The registered runtime-only
amendment rewrites the same squared norm algebraically with a largest
condition-specific `[B,L,L,H]` intermediate; parameters, masks, objective,
data, and selection are unchanged. Naive values and gradients are regression-
tested. The matching not-yet-started SAbDab2 IPA tasks were removed from the
old array and may run only from the read-only recovery source. See
[`protocols/framecontact_invariant_point_memory_amendment_v1.json`](protocols/framecontact_invariant_point_memory_amendment_v1.json).

`scripts/run_a100_matrix.sh` is fail-closed and resumable at the
seed-by-experiment unit. It first runs the focused tests and BF16 CUDA smoke test,
requires the GPU name to match `A100` by default, and refuses to overwrite partial
or completed results.

Required environment variables:

```bash
export GEOM_REPOSITORY_ROOT=/path/to/intra-chain-geometry-attention
export GEOM_CACHE_ROOT=/path/to/authenticated/structure_cache
export GEOM_EMBEDDING_ROOT=/path/to/frozen/esm_embeddings
export GEOM_LABEL_ROOT=/path/to/development/labels
export GEOM_RUN_ROOT=/path/to/new/durable/run_root
export GEOM_PYTHON_BIN=/path/to/python
bash scripts/run_a100_matrix.sh
```

For dual-GPU workers sharing one run root, assign disjoint `GEOM_SEEDS`, unique
`GEOM_WORKER_TAG` values, and set `GEOM_AGGREGATE=0` on both workers. Run the
aggregator only after both workers finish. `GEOM_VALIDATE=0` may skip duplicate
tests/smoke on a secondary worker; the primary worker must retain validation.
The runner preserves an existing `PYTHONPATH`, so a read-only runtime dependency
directory can supply `pytest` without modifying the CUDA image.

The smoke receipt records a single synthetic BF16 forward/backward latency and
peak allocated memory for each experiment. These values catch gross regressions;
they are explicitly not throughput or biological benchmark measurements.

The script never reads a frozen/native-release test split. Inputs must already be
the authenticated locked development assets.

After the 18 training units finish, run the two independent aggregators once:

```bash
python scripts/aggregate_multiseed_ablation.py \
  --run-root "$GEOM_RUN_ROOT" \
  --output "$GEOM_RUN_ROOT/MULTISEED_ABLATION_SUMMARY.json" \
  --seeds 17 42 73 \
  --bootstrap-replicates 10000

python scripts/aggregate_attention_diagnostics.py \
  --run-root "$GEOM_RUN_ROOT" \
  --output "$GEOM_RUN_ROOT/ATTENTION_DIAGNOSTICS_SUMMARY.json" \
  --seeds 17 42 73
```

## Auditable source handoff

Build a deterministic source-only archive outside the repository:

```bash
python scripts/build_source_handoff.py \
  --output-dir ../handoff
```

The builder excludes interpreter/build caches, refuses symlinks and common
secret-bearing files, scans for private-key markers, normalizes tar/gzip
metadata, embeds `SOURCE_MANIFEST.json`, and writes a separate receipt with the
archive SHA-256. Datasets, checkpoints, and benchmark outputs are intentionally
not part of this source archive and need separate manifests.

## Documentation

- [`docs/intra_chain_geometry_attention.md`](docs/intra_chain_geometry_attention.md):
  tensor contract and adapter provenance.
- [`docs/research_landscape.md`](docs/research_landscape.md): method comparison,
  current primary literature, integration decisions, and falsifiable innovation
  proposals.
- [`docs/literature_catalog.json`](docs/literature_catalog.json): 85-method
  structured catalog with evidence grades and official links where available.
- [`docs/MSA_PAIRFORMER_BASELINE_AUDIT_20260801.md`](docs/MSA_PAIRFORMER_BASELINE_AUDIT_20260801.md):
  official Cell-2026 source, weight, MSA-comparability and safe-load audit.
- [`docs/CONTACT_ADAPTED_ESM_FACTORIAL_CANDIDATE.md`](docs/CONTACT_ADAPTED_ESM_FACTORIAL_CANDIDATE.md):
  post-freeze 2 x 2 design separating monomer contact adaptation from
  partner-aware pair-trunk gains.
- [`docs/HANDSHAKE_BASELINE_AUDIT_20260801.md`](docs/HANDSHAKE_BASELINE_AUDIT_20260801.md):
  pinned-source, license, split and deterministic-inference audit for the
  clean-room ProstT5 partner-aware sequence-only control.
- [`docs/FRONTIER_CONTACT_BASELINE_AUDIT_20260801.md`](docs/FRONTIER_CONTACT_BASELINE_AUDIT_20260801.md):
  pinned Handshake, ProstT5, DeepSSInter and ProTact authority ledger plus the
  contact-cutoff, representation-factorial and triangle-trunk comparison plan.
- [`docs/PARTNER_SPECIFIC_FRONTIER_UPDATE_20260801.md`](docs/PARTNER_SPECIFIC_FRONTIER_UPDATE_20260801.md):
  pinned SPPIDER-seq, PEPNet, CALM and AAMFM audit plus the default-off,
  late-fusion-only contact-bottleneck cosine candidate, its gradient ablations,
  regression evidence, and explicit not-yet-trained boundary.
- [`docs/FRAMECONTACT_GEOMETRY_ABLATED_CONTROL_20260801.md`](docs/FRAMECONTACT_GEOMETRY_ABLATED_CONTROL_20260801.md):
  same-code partner-aware control that removes coordinate information without
  disabling the pair or cross-attention paths.
- [`receipts/double-cold-source-v17-final/ENGINEERING_RECEIPT.json`](receipts/double-cold-source-v17-final/ENGINEERING_RECEIPT.json):
  current counterfactual contact-lift authority with dead/diffuse-bottleneck
  diagnostics: 528 authenticated files, 298/298 local tests and 293 passed plus
  five optional-Boltz skips remotely. The candidate remains default-off and
  untrained.
- [`receipts/double-cold-source-v16-final/ENGINEERING_RECEIPT.json`](receipts/double-cold-source-v16-final/ENGINEERING_RECEIPT.json):
  parent counterfactual contact-lift authority: 521 authenticated files,
  297/297 local tests and 292 passed plus five optional-Boltz skips remotely.
  The incomplete v15 SSH transfer is explicitly quarantined and has no manifest
  or authority.
- [`receipts/double-cold-source-v14-final/ENGINEERING_RECEIPT.json`](receipts/double-cold-source-v14-final/ENGINEERING_RECEIPT.json):
  superseded untrained engineering sketch whose direct interaction-logit
  noisy-OR was rejected during probability-semantics red-teaming.
- [`receipts/double-cold-source-v13-final/ENGINEERING_RECEIPT.json`](receipts/double-cold-source-v13-final/ENGINEERING_RECEIPT.json):
  historical parent authority with 498 authenticated source/evidence files,
  pinned partner-specific frontier ledger, and final 284-pass remote regression.
- [`receipts/live-array-audit-20260801T145157Z/README.md`](receipts/live-array-audit-20260801T145157Z/README.md):
  latest matched-epoch-18 five-seed development snapshot and full trajectory
  figure. Distogram raises fixed-logit precision in 5/5 seeds but lowers recall
  in 5/5; contact AUPRC is nearly unchanged and paratope AUPRC is lower, so the
  current evidence is a sparsity trade-off rather than promotion evidence.
- [`receipts/live-array-audit-20260801T134059Z/README.md`](receipts/live-array-audit-20260801T134059Z/README.md):
  preceding matched-epoch-17 five-seed development snapshot; the earlier
  positive retrieval-margin diagnostic reversed and paratope AUPRC was lower
  in 5/5 seeds.
- [`receipts/live-array-audit-20260801T113500Z/README.md`](receipts/live-array-audit-20260801T113500Z/README.md):
  earlier historical matched-epoch development snapshot.
- [`receipts/live-research-handoff-v4/HANDOFF_INDEX.json`](receipts/live-research-handoff-v4/HANDOFF_INDEX.json):
  verified remote handoff binding the cache/symlink-free v13 source authority,
  full regressions, latest five-seed audit, v6 frontier ledger, teacher updates,
  and an explicit development-only evidence boundary.
- [`docs/FRAMECONTACT_RETRIEVAL_ALIGNED_SPECIFICITY_20260801.md`](docs/FRAMECONTACT_RETRIEVAL_ALIGNED_SPECIFICITY_20260801.md):
  research rationale, objective definition, engineering evidence, and
  validation-only promotion gate for the K=4 listwise specificity candidate.
- [`docs/FRAMECONTACT_DOUBLE_COLD_SPLIT_AUDIT_20260801.md`](docs/FRAMECONTACT_DOUBLE_COLD_SPLIT_AUDIT_20260801.md):
  official SAbDab2 cluster semantics, tokenized multi-chain antigen audit,
  Foldseek double-cold confirmatory split, component caveat, and unopened-test
  execution boundary.
- [`receipts/listwise-specificity-v1-1346200/ENGINEERING_RECEIPT.json`](receipts/listwise-specificity-v1-1346200/ENGINEERING_RECEIPT.json):
  source-authenticated 269-test regression plus K=1/4/8 real train-role RTX
  4090 forward/backward stress evidence; it is not a model-quality benchmark.

## Claim boundary

The previous real four-mode run was a one-seed, three-epoch development result on
an RTX 5090. Its `none` condition retained the spatial graph, so it did not test a
true dense sequence-only baseline. This repository corrects that protocol. No
A100 training/model-quality or multi-seed final result may be claimed until the
corresponding AB03 receipts and checksums exist. The current A100 receipt covers
only synthetic kernel equivalence, latency, and allocated memory. No biological
benchmark, external-generalization, clinical, or state-of-the-art claim is
authorized by this code alone.

The authoritative SAbDab2 `pairs.parquet`, real N/CA/C caches, frozen ESM2-650M
embeddings, reciprocal partner panels, and reciprocal-min Foldseek antigen-fold
split have been authenticated. Foldseek reports zero cross-role hard edges under
the declared threshold, and US-align boundary spot checks found zero sampled
cross-role hard edges; this is an operational structural-novelty criterion, not
proof of no remote homology. The eight-condition, five-seed development matrix
has started. Its IPA recovery is explicitly recorded, but no frozen SAbDab2
test prediction or metric has been opened.

The retrieval-aligned K=4 listwise specificity path is now implemented and
train-role GPU stress-tested with wrong-partner activation checkpointing.  It
remains a post-freeze exploratory candidate: the running residue-head and
distogram matrices are unchanged, K=4 has not been trained or selected, and no
frozen/external test was opened.  Live unary/distogram summaries are descriptive
only because the conditions are unfinished. At matched epoch 18, distogram
increases fixed-logit precision in all five seeds but reduces fixed-logit recall
in all five; pooled contact AUPRC changes by only +0.00044 and paratope AUPRC by
-0.03022. This does not authorize a checkpoint or architecture promotion.

The label-free same-antigen/different-antibody test authority is also frozen. It
will require a swapped-antibody epitope map to move toward the alternate true
epitope, rather than merely change. Existing CHIMERA residue-classification
results remain secondary post-access evidence only. Canonical CHIMERA
epitope-conditioned CDR sequence-structure co-design remains a separate required
phase.

The SAbDab2 frozen scorer is now protected by a separate validation-only
architecture-selection receipt and a one-shot opening/consumption gate. The
opening binds all eight ensemble candidates, the unique five-seed winner, test
axis, counterfactual panels, bootstrap/intervention settings, and evaluator
source hashes. The scorer creates an exclusive token before loading the first
test item and verifies the opened dataset count. This is tested protocol
behavior only; no SAbDab2 frozen-test score exists yet.

The AsEP official test remains unopened. Its implemented gate requires all eight
five-seed development ensembles, freezes one validation-only winner, opens only
test PDB inputs for label-free SAbDab2 decontamination and reciprocal partner
panels, and then binds the five checkpoints, full/clean test axes, calibration
rule, bootstrap, evaluator, adapter, and evaluation-dependency hashes in one
opening receipt. The final evaluator creates an exclusive consumption token
before deserializing the first label; a crash is fail-closed and cannot be
silently rerun. It reports the full official test and the pre-flagged
SAbDab2-decontaminated subset separately. No AsEP test metric exists yet.

The current pinned external-baseline ledger is
[`baselines/BASELINE_LOCK_V4.json`](baselines/BASELINE_LOCK_V4.json). GVP, EGNN,
complete AlphaFold IPA, and the adapted Pairformer all remain biologically
unrun on the identical split. The implemented `frozen_frame_ipa` condition is
only an IPA-style point-bias baseline. For Pairformer, source, checkpoint,
extracted trunk, structure adapter, and a synthetic real-weight forward are now
authenticated; native Boltz preprocessing, five identical-split seeds, an A100
receipt, and frozen-test scoring do not exist yet.
