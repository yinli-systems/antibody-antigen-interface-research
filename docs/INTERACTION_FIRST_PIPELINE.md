# SAbDab2 interaction-first pipeline

This pipeline is an experimental successor to the frozen SPPIDER-seq baseline. It keeps the completed baseline immutable and adds three forms of supervision that the marginal classifier did not have:

1. connected-component leakage control across both antibody and antigen similarity;
2. explicit antibody-residue to antigen-residue contact prediction;
3. matched-versus-real-wrong-partner ranking.

All commands record strict machine-readable provenance. The split builder intentionally labels its output `development_only`: the current SAbDab2 instances have already been inspected, so a genuinely prospective final test must come from structures released after a future locked cutoff.

## Files

- `sabdab2_interaction_first.py`: staged CLI for preparation, splitting, training, evaluation, and smoke tests.
- `interaction_first/contacts.py`: validated sparse contact-map artifacts.
- `interaction_first/splits.py`: deterministic connected-component splitting.
- `interaction_first/partners.py`: false-negative-aware partner sampling.
- `interaction_first/model.py`: interleaved bidirectional cross-attention and a supervised pair decoder.
- `interaction_first/losses.py`: pair, marginal, count, and partner-ranking objectives.
- `interaction_first/metrics.py`: pair-map, marginal, retrieval, and cluster-bootstrap metrics.
- `interaction_first/runtime.py`: self-contained frozen-embedding and checkpoint runtime.
- `test_interaction_first.py`: dependency-aware unit and gradient tests.

## Required inputs

The pipeline reuses the existing immutable artifacts:

- frozen `instances.parquet`, `pairs.parquet`, and `MANIFEST.json`;
- prepared SPPIDER-seq instance manifest;
- frozen ESM-2 residue embeddings;
- MMseqs2 for the strict development split;
- optional antibody-numbering metadata with `instance_id,cdrh3,vh_gene,vl_gene`;
- optional grouping metadata such as canonical complex or epitope groups.

The contact preparation command preserves the labels already stored in `pairs.parquet`. It does not recompute contacts from PDB coordinates, which avoids residue-index projection drift.

## Recommended execution order

```bash
python sabdab2_interaction_first.py prepare-contacts \
  --benchmark-root "$BENCHMARK_ROOT" \
  --output-dir "$INTERACTION_DATA"

python sabdab2_interaction_first.py prepare-antibody-metadata \
  --benchmark-root "$BENCHMARK_ROOT" \
  --mapping-dir "$CANONICAL_MAPPING_DIR" \
  --output "$INTERACTION_DATA/antibody_cdrh3.csv"

python sabdab2_interaction_first.py build-split \
  --benchmark-root "$BENCHMARK_ROOT" \
  --mmseqs "$MMSEQS" \
  --work-dir "$INTERACTION_DATA/mmseqs" \
  --cdrh3-metadata "$INTERACTION_DATA/antibody_cdrh3.csv" \
  --group-manifest "$CANONICAL_MANIFEST" \
  --output "$INTERACTION_DATA/double_remote_split.json"

python sabdab2_interaction_first.py smoke \
  --output "$INTERACTION_RESULTS/smoke.json"

python sabdab2_interaction_first.py train \
  --prepare-dir "$SPPIDER_PREPARE_DIR" \
  --contact-dir "$INTERACTION_DATA/contacts" \
  --embedding-dir "$SPPIDER_EMBEDDING_DIR" \
  --split-manifest "$INTERACTION_DATA/double_remote_split.json" \
  --output-dir "$INTERACTION_RESULTS/seed42/checkpoints"

python sabdab2_interaction_first.py evaluate \
  --prepare-dir "$SPPIDER_PREPARE_DIR" \
  --contact-dir "$INTERACTION_DATA/contacts" \
  --embedding-dir "$SPPIDER_EMBEDDING_DIR" \
  --split-manifest "$INTERACTION_DATA/double_remote_split.json" \
  --checkpoint "$INTERACTION_RESULTS/seed42/checkpoints/best.pt" \
  --output-dir "$INTERACTION_RESULTS/seed42/validation" \
  --split validation
```

Run test evaluation only after model and threshold choices have been locked on validation data.

## Split policy

The default component graph joins instances when any configured relationship holds:

- full antibody identity at least 80% with at least 80% coverage in both directions;
- full antigen identity at least 70% with at least 80% coverage in both directions;
- CDRH3 identity at least 70% within the same CDRH3-length bucket; if VH/VL columns are supplied, the comparison is additionally gene-stratified;
- optional canonical complex, epitope, fold, or source grouping.

Connected components are indivisible. A component can occur in exactly one of train, validation, or test.

## Negative-pair policy

The sampler uses real partner sequences, never shuffled amino-acid strings. Known cognate pairs are excluded. Unknown mismatched pairs receive a relative ranking objective only; they are not assigned an all-zero contact map. If `partner_metadata` is provided, candidates are prioritized by antibody group and length while avoiding the anchor antigen group whenever possible.

## Primary evaluation outputs

- micro and per-complex macro pair AUPRC, AUROC, MCC, F1, contact IoU, and precision at the true contact count;
- micro and macro epitope/paratope metrics derived from the contact map;
- matched-partner Recall@1/5/10 and mean rank;
- cluster-bootstrap confidence intervals for matched-minus-wrong epitope AUPRC;
- aligned Parquet files for pair predictions, node predictions, and partner scores.

## Mandatory ablations

Run at least the following before making scientific claims:

- partner-ranking weight set to zero;
- edge-loss weight set to zero;
- focal-chain-only and null-partner controls;
- random versus hard real-partner negatives;
- mean versus top-k marginal pooling;
- three independent seeds;
- antibody-remote, antigen-remote, and double-remote subsets.

The frozen historical SPPIDER-seq test remains a benchmark comparison. It must not be used for model selection in this pipeline.
