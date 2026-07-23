# Partner-specific retrieval protocol v1

Locked: 2026-07-23, before any AntiBERTy retrieval result was observed.

## Scientific question

Can an independently encoded antibody and antigen be matched above a finite-library,
degree-preserving random baseline when both sequence axes are held remote from the
training data?

This is a development protocol. The frozen SAbDab2 test set remains sealed and is not
used for architecture selection, early stopping, or threshold selection.

## Existing exploratory result

The best ESM-2/ESM-2 exploratory configuration has bidirectional full-library MRR
0.079700 on 126 antibodies and 120 antigens. The exact finite-library random expectation
is 0.049345. A 10,000-replicate independent two-axis, fixed-degree label permutation
test gives p=0.001800. This is evidence of a matching signal, but the effect is only
1.615 times random and therefore does not pass the robustness gate below.

## Locked primary model

- Antibody residue representation: frozen AntiBERTy, VH and VL embedded separately.
- Antigen residue representation: frozen ESM-2 `esm2_t33_650M_UR50D`.
- Projection: a compact feed-forward head applied to each residue before pooling.
- Pooling: masked length-wise mean, matching the published CALM stage-1 ordering.
- Shared dimension: 128.
- Objective: symmetric multi-positive InfoNCE over every known exact positive in a
  batch.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.2.
- Temperature: learnable, initialized at 0.2 and numerically bounded.
- Batch size: use the largest deterministic batch up to 1,024 that fits without
  gradient accumulation changing the in-batch negative set.
- Selection: bidirectional full-library MRR only; never training loss.
- Early stopping: patience 15; best checkpoint is written atomically before the last
  checkpoint.

## Limited ablations

The initial screen contains only three prespecified representations, all at seed 42:

1. Frozen ESM-2 on both sides with the CALM projection-before-mean head.
2. AntiBERTy antibody plus ESM-2 antigen with strict length-wise mean pooling.
3. The same asymmetric backbones with VH/VL chain-balanced mean pooling.

No extra variant is added after inspecting a fixed holdout result. Additional ideas,
including CDR-only pooling, homology-uncertain negative masking, antigen patches, or
semi-hard negative mining, require a new versioned protocol and train-only evidence.

## Model-selection isolation

The three variants are screened on antigen-cluster folds constructed only from the
current training partition. The existing development validation labels are not used to
choose among them. The selected architecture is then trained on the entire training
partition and evaluated once on the fixed development validation library.

The fixed validation boundary has zero exact antibody, antigen, and pair overlap and
zero cross-boundary MMseqs2 hits at identity at least 80% with reciprocal coverage at
least 80%. A Needleman-Wunsch secondary audit is required before a leakage-safe claim.

## Primary metrics and nulls

- Primary: `0.5 * (MRR_antibody_to_antigen + MRR_antigen_to_antibody)`.
- Secondary: directional MRR and Hit/Recall at 1, 5, 10, and 20.
- Exact random expectation accounts for every anchor's number of positives.
- Statistical null: 10,000 independent permutations of antibody and antigen embedding
  labels against the fixed positive graph, preserving every graph degree.
- Uncertainty: 10,000 anchor bootstraps in both directions. Cluster bootstrap supersedes
  this interval once stable cluster identifiers are attached to all anchors.
- Required controls: raw frozen-embedding cosine, untrained projection head, sequence
  length/composition shortcut, partner permutation, and one-sided/unary ablations.

## Go/no-go gates

The model advances to three random seeds only if all conditions hold on the fixed
development validation:

1. Both directional MRR values exceed their permutation 99th percentiles.
2. Bidirectional MRR is at least twice the exact random expectation.
3. Both directional Recall@1 values are at least twice their exact random expectations.
4. Partner permutation removes at least 50% of the excess MRR above chance and the
   paired bootstrap interval for the drop has lower bound greater than zero.
5. Raw embedding, random-head, length/composition, and unary controls remain clearly
   below the trained paired model.

Three-seed confirmation uses seeds 17, 42, and 73. Every seed must exceed chance, and
the sample standard deviation must not exceed 25% of the mean excess above chance.

## Claim boundary

Passing this protocol supports a claim of partner-specific retrieval on this frozen
development benchmark. It does not establish binding, affinity, residue-pair contacts,
or state of the art across papers with different candidate libraries and splits.
External experimental non-binder data and a matched public CALM/EpiCLIP protocol are
required before a broader robustness or world-class claim.
