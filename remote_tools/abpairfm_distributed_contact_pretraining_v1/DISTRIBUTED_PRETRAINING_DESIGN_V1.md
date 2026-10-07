# ABPairFM distributed pair-contact pretraining v1

## Decision

Continue the authenticated 3B PairAware ESM-2 Stage-A0 checkpoint and add a
new, versioned contact curriculum.  Do not restart another antibody foundation
model from random initialization: the 7B MGnify scratch run is already using
six RTX 5090 GPUs, while the 3B checkpoint already contains paired-sequence
adaptation and is directly compatible with the contact heads.

The scientific target is partner-specific residue-pair ranking, not only
paratope or epitope propensity.  Experimental or predicted structures may be
used to create train labels in the general-PPI stages, but coordinates,
distances, and contact labels are never model inputs at inference.

## Curriculum

| Stage | Data | Purpose | Promotion |
|---|---|---|---|
| A0 | development-decontaminated STRING physical pairs | finish the existing 5B-residue paired MLM warm start | exact DCP restore, finite MLM, no repeated slices |
| B-canary | immutable 512-system PINDER-Train slice | validate objective, memory, collectives, and checkpointing | finite terms, no OOM, restart equivalence, no data-boundary violation |
| B32K | frozen 32,768-system PINDER-Train slice | first biological pair-contact gate | contact AP above prevalence, positive matched-minus-swapped signal, healthy marginals |
| B500K | frozen 500,000-system PINDER-Train slice | learn broad interface priors | B32K gate plus throughput/recovery pass |
| B-full | all 865,471 strict-quality, development-decontaminated PINDER-Train systems | final general-PPI contact pretraining | fixed token budget; no intermediate external-test access |
| C-Ab | reconstructed complete VH/VL-antigen systems | antibody-specific adaptation | only after exact four-view 80/80 split audit passes |

PINDER reports 1,560,682 bound train dimers.  The authenticated strict-quality
manifest contains 982,075 systems; removing every endpoint homologous to the
frozen development panel leaves 865,471 systems. Sampling is over physical
systems, not raw chain-pair rows. Exact-sequence-pair, PDB, and cluster
multiplicities are frozen before B32K. The weight of a system is proportional
to the inverse geometric mean of those three multiplicities and normalized to
mean one.

## Objective

The total loss is

`L = L_pair + 0.2 L_paratope + 0.2 L_epitope + 0.2 L_swap + 0.25 L_MLM`.

- `L_pair` is equal-weighted across physical systems and balances positive and
  negative residue pairs within each system.
- Negative contact edges are weighted by detached model confidence and inverse
  row/column degree.  This emphasizes difficult edges without allowing one
  long or highly sampled complex to dominate.
- Paratope and epitope targets are the row/column marginals of the contact
  matrix.  Their auxiliary logits are temperature-smoothed endpoint maxima.
- Same-batch partner swaps are all-negative only after exact endpoint,
  endpoint-cluster, and known-positive exclusions.  A deterministic bipartite
  matching fails closed when the batch has no legal derangement. Native pairs
  are never cropped: each eight-rank group is also accepted only when every
  possible left/right swap stays within the frozen 1,024-token model limit.
- MLM remains a low-weight stability regularizer.  It does not replace the
  pair objective.

The implementation is
`src/antibody_pretraining/paircontact_pretraining.py`.  It is backbone-neutral
and does not alter any previously frozen result runner.

The matched explicit-pair-state decoder decision is frozen separately in
`PAIR_LEVEL_RESEARCH_DECISION_V1.md`. It compares the current low-rank head
against a compact complete-bipartite edge-refinement head without changing the
active B32K run, data, objective, optimizer, or test boundary.

## Distributed system

### Selected safe path

- single node, 8 RTX 4090 GPUs for the B32K biological gate, using the
  immediately available full node;
- single node, 8 RTX 5090 GPUs for B500K and larger production stages after
  B32K passes;
- PyTorch FSDP2, transformer blocks sharded bottom-up and the root sharded last;
- BF16 forward/backward, FP32 optimizer state, fused AdamW;
- native variable-length Flash/SDPA attention, deterministic length buckets,
  and token-budget packing;
- selective activation checkpointing when it fits; full checkpointing is the
  fallback;
- one reduced metrics vector per optimizer update, no unconditional update
  barrier;
- DCP model/optimizer/data-cursor state, with asynchronous save promoted only
  after a restart-equivalence canary;
- eager execution is the scientific reference. `torch.compile` is only an
  efficiency candidate because the previous compile experiment failed the
  numerical non-inferiority gate.

The active 6-GPU 7B scratch job is not preempted. B-canary uses four RTX 4090
GPUs and B32K uses one complete eight-GPU RTX 4090 node. The 8-GPU 5090
production job starts only after the B32K biological gate passes and one
complete node is available, preventing an unmeasured multi-node communication
penalty.

### Efficiency selection

Run 20-update candidates after five warm-up updates:

1. eager + full activation checkpointing;
2. eager + selective activation checkpointing;
3. eager + selective checkpointing + `reshard_after_forward=false`, only if
   peak memory remains below 90%;
4. `torch.compile(mode="reduce-overhead")` only after candidates 1-3 pass.

Select the fastest candidate whose loss and gradients are finite, whose first
20-update mean loss differs by no more than 2% from eager, and whose exact DCP
restart reproduces the next batch identity and update number.  This is a
systems selection, not model tuning.

## Evaluation contract

B32K scores only a frozen, PDB-disjoint 512-system holdout drawn from
PINDER-Train and excluded from gradients. PINDER-Val is reserved for a later,
separately frozen development gate; PINDER test rows and structures remain
unopened. Report mean-per-system and pooled AP, AUROC, Recall@1, Recall@L/10,
MRR, matched-minus-partner-swapped score, paratope AP, and epitope AP. AP is
average precision, not trapezoidal AUPRC.

The complete-antibody C stage is unauthorized until the independent exact
four-view audit has zero train-validation crossings for VH, VL, concatenated
antibody, and antigen.  SABLE and historical SAbDab2 labels are not used for
selection.  MIPE and ParaSurf comparisons are reporting tracks with explicit
split/metric caveats, not promotion gates.

## External-review disposition

- The requested MIPE 1e-4/800-epoch five-fold training and frozen archival
  replay are complete. The replay remains analyst-exposed evidence and is not
  a promotion gate.
- The older binary split is now superseded by an exact full-relation
  infeasibility receipt and a vertex-separator protocol that keeps 80/80 while
  allowing bridge entities to be discarded.
- The official SAbDab2 split should be added as a comparability track, while the
  stricter four-view split remains the internal claim boundary.
- The pinned v0.2.0 split archive is now materialized and byte-audited. Its
  `ab_split`, `ab_split_sd`, `abag_split`, and `abag_split_sd` tables contain
  15,810/3,382 rows with no train-test `INSTANCE` overlap. They do retain
  train-test PDB overlap (228/39 for antibody-only and 112/27 for Ab-Ag), so
  these are official-comparability splits rather than strict PDB-disjoint or
  four-view double-cold evidence.
- Additional geometry seeds are not launched to chase p < 0.05.  A matched,
  current-model dual-branch versus single-branch ablation must be preregistered
  if that architectural claim is needed.
- Pair-level metrics and explicit pair state are primary.  Marginal-only gains
  cannot promote a scale stage.

## Publication path

- A systems/methods appendix may report the authenticated curriculum, failed
  gates, and B-canary engineering result, but it must not call the model SOTA.
- A main-track submission requires a passing B32K gate, one frozen B500K or
  B-full run, the official SAbDab2 `abag_split` comparability track, and a
  separately passing exact four-view antibody/antigen split. Pair AP, Recall@1,
  MRR, and matched-minus-swapped separation are primary; marginal residue
  scores are secondary.
- A high-impact biological-method claim additionally requires a post-cutoff or
  third-party sealed-label antibody/antigen double-cold cohort. Public SABLE,
  MIPE, and previously opened SAbDab2 labels remain analyst-exposed supporting
  evidence only.

## Primary references behind the design

- [PINDER](https://github.com/pinder-org/pinder) supplies the train-only
  bound-interface authority and its leakage-aware benchmark partitions.
- [PyTorch FSDP2](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
  and the [asynchronous Distributed Checkpoint recipe](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)
  define the sharding and recoverable-state path used here.
- [RaftPPI](https://openreview.net/pdf?id=Dp1RM3gPg8) motivates adaptive hard
  negatives for strict PPI benchmarks; this implementation additionally
  balances residue degrees and physical-system multiplicity.
- [PPLM](https://www.nature.com/articles/s41467-026-70457-5) supports paired
  protein-language adaptation at scale, while this campaign makes
  residue-pair contact ranking—not only paired sequence representation—the
  promotion target.
- [ParaSurf](https://academic.oup.com/bioinformatics/article/41/2/btaf062/8005857)
  is retained as an external marginal-interface comparison; it is not used to
  select B-stage checkpoints.

## Current execution state (2026-09-01)

- 7B scratch MGnify run: active on 6 RTX 5090 GPUs through job 1513514,
  target 50B tokens.
- 3B Stage-A0: verified `segment-000130` contains 2,070,335,724 / 5B residue
  tokens with content SHA-256 `be22eb18...f10f839`. Segment 131 first landed
  on a 23,028-MiB 4090 node and failed before update zero; recovery job
  1529027 preserves the model, optimizer, global batch 112, and all data while
  excluding the two confirmed low-memory nodes. Watchdog 1529028 resumes the
  same unique-pair ladder after the segment passes.
- PINDER antibody vertex-separator solve and exact audit: the first exact 80/80
  solve hit its frozen two-hour solver limit without publishing a split.
  Recovery-v2 job 1527693 extends only the solver window and is running; all
  four sequence-view constraints remain unchanged.
- PINDER train curricula: recovery job 1527028 completed 0:0. The frozen,
  deterministic nested manifests contain 512/32,768/500,000/865,471 systems;
  the receipt confirms that no test rows or structures were read.
- B-canary: job 1527640 completed 0:0 and passed all systems/objective gates.
  Contact mean-per-complex AP increased from 0.001465 to 0.002473 against
  prevalence 0.000914; matched-minus-swapped separation increased from 0.0150
  to 0.0395. Exact DCP restore matched content SHA-256
  `bb6cc3d7146e880e7589c682c59d7c7236c406bcd6970eff95fa6665a830c273`.
- B32K labels: 32,768 PINDER-Train systems, 12,372 PDBs, 1,301,323 primary
  4.5-A contacts, 2,416,353 auxiliary 6-A contacts, and 14,205,552 residues
  passed sequence/PDB-axis, finite-value, and train-only audits. Manifest and
  receipt SHA-256 are `9f76bb32...cca6637` and `b8b18478...8ec1034a`.
- B32K training: recovery-v5 job 1528772 uses 8 RTX 4090 GPUs, FSDP2/BF16,
  2,048 optimizer updates, global batch 32 systems, and the frozen Stage-A0
  `segment-000129`. The complete 32K preflight built 8,192 training groups and
  64 holdout groups and verified every native and partner-swapped input at
  no more than 1,024 tokens. Recovery-v3 reached the baseline evaluation but
  correctly failed before update zero because independently legal native pairs
  could create an over-length swapped pair. Recovery-v4 fixes only this group
  construction; no data, objective, optimizer, holdout, or promotion rule was
  changed. Recovery-v4 verified nine finite optimizer updates and was then
  deliberately cancelled because its original eight-hour Slurm limit was
  shorter than the measured end-to-end runtime; recovery-v5 changes only that
  execution limit to 20 hours and preserves the v4 protocol and source hashes.
- Official SAbDab2 v0.2.0 splits: the pinned 889,347,536-byte archive and all
  four CSV tables were extracted by an allowlisted, regular-file-only path and
  replay-audited. The tables have zero train-test instance overlap but retain
  PDB overlap, so they are restricted to the external comparability track.
