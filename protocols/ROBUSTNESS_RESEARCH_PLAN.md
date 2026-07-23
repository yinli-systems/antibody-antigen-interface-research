# Robust partner-specific antibody-antigen research plan

Last updated: 2026-07-23

## Goal and claim boundary

The immediate goal is partner-specific sequence retrieval: given one antibody or
antigen, rank its known cognate partner above a complete held-out library. This task is
strictly narrower than experimental binding classification, affinity prediction,
epitope/paratope labeling, or residue-pair contact prediction. Results from those tasks
must not be substituted for one another.

No result is described as world-class merely because it exceeds a random baseline.
The strongest supportable claim is determined by the hardest completed validation
layer below.

## Evidence from current methods

- [CALM](https://www.biorxiv.org/content/10.64898/2026.02.25.707916v1)
  establishes a direct sequence-to-specificity retrieval task and motivates frozen
  antibody/protein encoders, projection-before-pooling, symmetric contrastive learning,
  and 80%-identity leakage control.
- [EpiCLIP](https://openreview.net/pdf?id=JImubudlk9) shows that approximate interface
  candidates, six-CDR-aware antibody pooling, and antigen surface patches can focus a
  retriever on the interacting regions rather than global sequence identity.
- [DataSAIL](https://www.nature.com/articles/s41467-025-58606-8) motivates paired-input
  similarity splitting instead of random pair splitting when both molecular axes can
  leak.
- A controlled antibody-binding study in
  [Nature Machine Intelligence](https://www.nature.com/articles/s42256-025-01089-5)
  shows that biologically similar hard negatives can improve out-of-distribution
  generalization even when in-distribution scores become lower.
- A separate
  [Nature Machine Intelligence analysis](https://www.nature.com/articles/s42256-025-01176-7)
  demonstrates that frozen protein-language-model pretraining itself can contaminate
  downstream PPI evaluation. Downstream split isolation therefore cannot prove a fully
  pretraining-clean result.
- [CHIMERA-Bench](https://arxiv.org/abs/2603.13431) motivates distinct unseen-epitope,
  unseen-antigen-fold, and temporal evaluations rather than one generic random holdout.
- [AVIDa-SARS-CoV-2](https://proceedings.neurips.cc/paper_files/paper/2024/file/d2a1e47f7dc635fac77fbd6e2ec799e4-Paper-Datasets_and_Benchmarks_Track.pdf)
  supplies experimentally measured binders and non-binders, which are necessary to
  test real specificity rather than treating every unobserved SAbDab pair as a known
  negative.
- [Agate](https://openreview.net/pdf/d06b2dd898457df647788292fbcbce2e033e2a32.pdf)
  motivates joint antibody--antigen encoding, antigen-cluster-balanced sampling, and
  explicit binary interaction supervision at larger scale. Its task and negative
  construction differ from SAbDab retrieval, so its headline AUC is not a directly
  comparable target for the current experiment.
- [PPLM](https://www.nature.com/articles/s41467-026-70457-5) shows that a paired
  language model can expose inter-protein attention and support interaction, affinity,
  and contact-map heads. It motivates a pair-aware reranking stage after coarse
  retrieval rather than another marginal single-chain head.
- [PLM-interact](https://www.nature.com/articles/s41467-025-64512-w) supports a
  lightweight joint-input pair representation as a practical reranking alternative to
  immediately fine-tuning an entire paired protein language model.
- [SPPIDER-seq](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142516/) uses
  composition-preserving partner shuffles to test whether predictions depend on the
  supplied partner. This is a dependency diagnostic, not evidence that a shuffled or
  replaced pair is experimentally non-binding.
- [AsEP](https://proceedings.neurips.cc/paper_files/paper/2024/file/15add6732964d5b1f0954058bf3ccc88-Paper-Datasets_and_Benchmarks_Track.pdf)
  shows a large degradation under epitope-group splitting and provides residue-node
  and residue-edge supervision, motivating both unseen-epitope evaluation and a native
  contact head.
- [AbRank](https://arxiv.org/abs/2506.17857) motivates mutation-aware and scaffold-shift
  evaluation, as well as pairwise ranking with an explicit experimental noise margin
  instead of treating small affinity differences as exact regression targets.
- A 2026 [REAP specificity screen](https://pubmed.ncbi.nlm.nih.gov/41850278/)
  measures 174 clinical-stage or approved antibodies against 6,172 human extracellular
  proteins and reports off-target reactivity for 28% of the antibodies. Subject to
  sequence mapping and source-data availability, this is a substantially stronger
  specificity challenge than treating unobserved structural pairs as non-binders.
- The 2026 [Synthetic Epitope Atlas](https://www.biorxiv.org/content/10.64898/2026.04.17.719295v2.full)
  combines predicted VHH--synthetic-epitope complexes with high-throughput experimental
  affinity measurements, including designed complexes that fail to bind. It is a
  candidate prospective hard-negative benchmark, not an authorized tuning set.
- A 2025 [antibody--antigen affinity study](https://www.nature.com/articles/s43588-025-00823-8)
  reports strong sensitivity to train--test cut-offs and concludes that present
  experimental data remain insufficient for robust generalization. This reinforces the
  requirement to report split sensitivity and to separate measured from synthetic or
  predicted affinity labels.

## Validation ladder

### Layer 1: immutable data and split audit

- Pin canonical data, authority split, embedding corpus, model weights, source code,
  and every output with SHA-256.
- Remove exact duplicate pairs before training and evaluation.
- Audit both sequence axes with exhaustive global alignment, not only a heuristic
  search prefilter.
- Report concatenated antibody, VH, VL, and antigen leakage separately.
- Keep the frozen test inaccessible during architecture, hyperparameter, threshold,
  and seed selection.

### Layer 2a: completed first-stage nested development selection

- Compare only the three preregistered encoders/poolers on three train-only
  antigen-cluster folds.
- Use unweighted macro mean of fold bidirectional MRR; break ties by worst-fold MRR,
  then fewer trainable parameters.
- Use one seed for screening. Additional seeds are confirmation, not extra attempts at
  finding a favorable run.
- Stop by complete-library validation MRR, never by training loss.

This first-stage screen isolates antigen clusters but not both molecular axes
simultaneously. It is adequate for selecting the already locked P0 baseline, but it is
not the final generalization protocol for any new architecture.

### Layer 2b: required two-axis selection for every new architecture

- Build S2-style folds entirely inside authority train by clustering VH, VL,
  concatenated antibody, and antigen sequences.
- Assign antibody clusters and antigen clusters to folds as two distinct dimensions,
  then retain only interactions whose two endpoints receive the same fold. Discard
  cross-fold interactions explicitly, following the DataSAIL S2 semantics; never move
  a validation entity cluster into training to meet a size target.
- Report the discarded interaction count, entity and positive-pair counts per fold,
  connected-component sizes, clustering thresholds, and maximum exact-NW identity and
  reciprocal coverage between folds.
- Select loss, reranker depth, top-K, epoch, and all hard-negative rules only from these
  train-only folds. The corrected authority-validation boundaries cannot select a new
  architecture after their scores are visible.
- Compare vanilla InfoNCE, multi-positive InfoNCE, and multi-positive InfoNCE with
  ambiguous high-homology pairs masked from the denominator.

An exploratory authority-train-only feasibility calculation was run before this S2
protocol is frozen. Using the available MMseqs2 80%-identity/80%-reciprocal-coverage
relations for concatenated antibodies and antigens, the largest connected component of
the bipartite cluster-interaction graph contains 3,014 of 3,229 source-train rows
(93.34%). Therefore, assigning whole bipartite interaction components to folds is not a
viable three-fold strategy on this corpus. This calculation is diagnostic only: final
S2 folds still require exact Needleman--Wunsch auditing for concatenated antibody, VH,
VL, and antigen views, an immutable solver objective, and a complete discarded-edge
ledger. The immutable diagnostic is
`train_only_s2_feasibility/diagnostic.json` (SHA-256
`3d6b53506c81842e3e9e6aa8dfb434b706749e9d84590fb906e057fa27f8177d`), produced by
`analyze_train_only_s2_feasibility.py` (SHA-256
`65e7405f7c678af68d622e6a4a1470321426438845b1b4612a2f16d2debb02db`); its three
synthetic firewall and immutability tests pass.

A second authority-train-only feasibility calculation solved the separate-axis
assignment as an exact mixed-integer programme. With each axis constrained to
within 5% of one third of the 1,830 deduplicated exact sequence pairs, the proven
zero-gap optimum retains 1,607 pairs (87.81%) and discards 223 cross-fold pairs; it
also retains 2,935 of 3,229 source rows (90.90%). The retained exact-pair counts are
508, 538, and 561 across the three folds. Duplicate source rows are deliberately not
the optimization unit and remain imbalanced, because model selection deduplicates
exact sequence pairs before optimization. The approximate assignment is not exported
and does not authorize final folds. The immutable solver receipt is
`train_only_s2_feasibility/milp_diagnostic.json` (SHA-256
`1534377aa7dd4f4ab513492ea9694d904e2bdb12f0a6c5bbd6e38254e239330c`), produced by
`optimize_train_only_s2_feasibility.py` (SHA-256
`b4070a1ba623c20a191d07b92fe58d570b8ebffb2c93ba5c104628f92fa2c721`); both
synthetic optimizer tests pass.

The same optimum has now been exported as a create-only provisional assignment with
complete retained and discarded ledgers. It assigns 1,383 antibody sequences and 1,172
antigen sequences, retains 1,607 exact pairs / 2,935 source rows, and explicitly
discards 223 cross-fold exact pairs / 294 source rows. The artifact is
`train_only_s2_feasibility/s2_assignment_provisional_v1.json` (SHA-256
`8dce37bec026e76c1d8d7e917b946ced01fca378ad1bccac196d23aced587959`, scientific
payload SHA-256
`4b2fd9633685162af9fa48c33766bfbbea2722f5f04f76ee2bf86974d968d7ac`), produced by
`build_train_only_s2_folds.py` (SHA-256
`c1b5faf0d46f3111800da269c8a02d37cfdcb830e7ae75ee8c07058021994c1c`). Its hard
authorization flags remain false. It cannot be used for training or model selection
until a separate create-only exact-NW receipt reports zero cross-fold edges in VH, VL,
concatenated-antibody, and antigen views and binds both artifact hashes above.

### Layer 3: fixed development evaluation

- Evaluate the selected architecture once on both corrected, versioned boundaries:
  the 128-instance concatenated-antibody/antigen subset and the 43-instance four-view
  chain-aware stress subset.
- Treat both subsets as reused development confirmation only. They are clean relative
  to the AntiBERTy inner-fold selection, but the source authority-validation pool was
  inspected by an earlier ESM2 exploration and is therefore not globally untouched.
- Keep all 3,229 authority-train instances in training. The 88 and 173 ineligible
  authority-validation instances, respectively, remain excluded and are never moved
  into training or scored.
- Preserve every per-anchor rank and positive-key set.
- Report both retrieval directions, MRR, mean/median rank, normalized rank, and
  Recall/Hit@1, 5, 10, and 20.
- Report candidate-library-size curves as secondary analyses; the complete native
  library remains primary.

### Layer 4: falsification controls

- Exact finite-library random expectation for each direction and positive degree.
- At least 10,000 independent, degree-preserving label permutations.
- Raw frozen-embedding cosine similarity.
- Identical architecture with a random, untrained head.
- Length and amino-acid-composition-only shortcut model.
- Antibody-only and antigen-only/unary baselines.
- Partner permutation with a paired excess-over-chance effect and confidence interval.
- Composition-preserving sequence shuffles, length/composition-matched partner
  replacements, and lineage/family-matched difficult replacements. Every unmeasured
  replacement is called a non-cognate decoy, never a confirmed non-binder.
- Single-side mask/zero ablations and the probability that a cognate score exceeds its
  matched decoy score.
- Performance stratified by nearest training identity, sequence length, family size,
  antigen component, and presence/absence of VL.

### Layer 5: stability and uncertainty

- Repeat only the winning preregistered architecture at seeds 17, 42, and 73 after the
  seed-42 gate passes.
- Report every seed, mean, sample standard deviation, and worst seed.
- Use anchor bootstrap intervals plus cluster bootstrap intervals. For the 43-instance
  stress test, emphasize exact/permutation inference and interval width.
- Add a crossed antibody-by-antigen or connected-component bootstrap sensitivity
  analysis; repeated PDB rows and candidate samples are not independent biological
  observations.
- Treat secondary metrics as descriptive or control their false-discovery rate; do not
  mine them for a positive headline.

### Layer 6: experimental external validity

- Freeze the complete pipeline before touching AVIDa labels.
- Evaluate zero-shot or with a training partition whose antigens and VHH lineages are
  disjoint from the external test.
- Use experimental non-binders as negatives and report AUPRC, AUROC, MCC, balanced
  accuracy, Brier score, expected calibration error, and per-antigen grouped intervals.
- Make within-VHH mutation-specific pairwise accuracy/AUC the primary partner-specific
  endpoint when the same VHH changes binding state across antigen variants. Also report
  per-antigen macro AUPRC/AUROC and multi-positive MAP/nDCG/retrieval metrics.
- Add near-positive and lineage-matched hard negatives, but never relabel uncertain
  examples as certain non-binders.
- After the AVIDa protocol is complete, evaluate a separately frozen off-target panel
  from the REAP clinical-antibody screen if antibody variable-region sequences, antigen
  sequences, and quantitative source measurements can be mapped without manual
  score-guided filtering. Report target rank, macro average precision across antibodies,
  and the fraction of experimentally reported off-targets recovered at fixed library
  depths.
- Treat the Synthetic Epitope Atlas as a future prospective affinity challenge only if
  its row-level data and licenses become available. Designed failures must remain
  experimental hard negatives; predicted complex geometry is an input/provenance field,
  not a measured positive label.
- Do not use the ANDD aggregate as an independent headline benchmark without a complete
  source-level deduplication audit: it mixes multiple upstream collections and includes
  predicted affinity values alongside measurements.

### Layer 7: interface-specific modeling

Only after the global dual-encoder baseline is stable, open a new protocol for:

1. six-CDR or predicted-paratope antibody pooling;
2. solvent-accessible antigen surface patches with ambiguous patches ignored;
3. coarse dual-encoder retrieval followed by a cross-attention reranker;
4. an explicit residue-pair interaction matrix supervised by native contacts; and
5. partner-dropout, framework/context dropout, and counterfactual losses that penalize
   antigen-blind or antibody-blind predictions; and
6. an antigen-identity reconstruction or cycle-consistency diagnostic that must fail
   when the partner input is permuted.

This stage must show that partner permutation destroys the learned excess over chance.
Higher marginal epitope AUPRC alone is insufficient.

## Overfitting stop rules

- Do not add a model variant after inspecting a fixed-holdout score.
- Do not relax a similarity threshold because a strict split becomes small.
- Do not choose the best of many seeds and report it as representative.
- Do not tune on the external experimental test or frozen SAbDab2 test.
- Do not compare raw numbers across papers with different candidate libraries, splits,
  positive degrees, or definitions; reproduce a matched protocol first.
- Do not describe a statistically significant but practically tiny gain as robust.

## Advancement criteria

A model is promoted beyond an exploratory development result only when it:

1. passes every preregistered CALM-comparable gate;
2. remains above its matched null in both directions on the four-view stress test;
3. beats raw-embedding, random-head, shortcut, and unary controls;
4. shows a material partner-permutation collapse;
5. replicates across three seeds with bounded variance; and
6. later generalizes to experimentally verified binders/non-binders or a genuinely
   prospective/temporal benchmark.

Until all six conditions are met, the correct description is a rigorous development
result, not a world-leading binding predictor.

## Three promotion gates for the full programme

- **Gate A — current P0 confirmation:** the cognate-pair effect must survive both
  repaired boundaries and cannot be explained by raw-embedding, random-head,
  length/composition, antibody-only, or antigen-only controls.
- **Gate B — architecture improvement:** any pair reranker or contact-head model must
  improve paired MRR/R@k consistently across repeated train-only two-axis S2 folds,
  with a stronger partner perturbation effect than the dual-encoder baseline.
- **Gate C — external specificity:** one frozen checkpoint must exceed chance on an
  AVIDa within-VHH mutation-specific endpoint before a strong partner-specificity claim
  is made.
