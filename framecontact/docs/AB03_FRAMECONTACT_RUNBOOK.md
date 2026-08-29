# AB03 FrameContact runbook

## Current evidence state

| Item | State |
|---|---|
| FrameContact model and objectives | late and interleaved contact-gated variants implemented and locally tested |
| Protocol v3 + runtime amendment v5 | frozen; fixed-antigen/wrong-antibody is primary, specificity excludes unary propensity, and runtime changes are provenance-bound |
| Reciprocal partner panels | complete for train/validation/test; 31 candidates in both directions; test SHA-256 `ea70fef770448302fb4ed1ccd7da51a1cc56cba5ea153076e077971735874bd1` |
| Real `N/CA/C`, chain IDs, missing-frame adapter | 3,663-instance role-independent cache complete; manifest SHA-256 `edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0` |
| Exact residue-pair adapter | 190,509,766 candidates and 143,756 positives materialized; manifest SHA-256 `d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33` |
| Antigen structure export | 3,663 structures complete; manifest SHA-256 `4882930043a4fc26319119f56b55a15bd729d23fb22ab04c485ce3ac79ada3d5` |
| New split matrix | reciprocal-min antigen-fold split complete: train/validation/test = 2,565/549/549; manifest SHA-256 `fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6` |
| Foldseek + US-align audit | 725,687 declared hard edges, zero cross-role; all 103 cross-role boundary pairs plus 1,000 within-role controls spot-checked with US-align, zero cross-role hard edges |
| Full frozen ESM embedding cohort | 3,015 unique sequences complete; manifest SHA-256 `9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545` |
| Local validation | 189 tests, Python compile, and GPU validation smokes pass through 2026-07-31; rerun the full suite on the handed-off source |
| Five-seed matrix | formal development-only matrix submitted on ParaCloud RTX 4090 from read-only source v2; this is not AB03/A100 evidence |
| Five-seed validation-frozen ensemble | implemented and synthetic-tested; all eight ensembles and the unique validation-only winner must exist before opening test |
| Frozen SAbDab2 ensemble scorer | one-shot opening/consumption gate, zero-contact policy, calibration slope, cluster bootstrap, interventions, exact SE(3), and same-antigen challenge implemented; real test role not opened |
| Same-antigen challenge authority | label-free test panel frozen: 27 exact-antigen groups, 164 focal complexes, 1,710 ordered antibody swaps |
| AsEP benchmark | five-seed development matrix running; label-free test/decontamination/panel adapters and one-shot scorer are implemented, but official graph labels have not been opened and no external benchmark claim exists |
| Frozen external scorer/opening gate | implemented and synthetic-tested; no external truth opened |
| Canonical CHIMERA co-design | not run |

No biological result is authorized until the missing split/panel authorities,
real development checkpoints, validation-frozen ensemble, and hardware receipts
exist. ParaCloud 4090/5090 evidence must not be described as AB03 A100 evidence.

## 1. Authenticate the role-independent data authorities

The pair and structure authorities have already been materialized on the current
workspace. Before copying them to AB03, verify the immutable source and cache
digests rather than rematerializing from a guessed path:

```bash
sha256sum STRUCTURE_CACHE_MANIFEST.json CONTACT_CACHE_MANIFEST.json
sha256sum instances.parquet pairs.parquet abag_split.csv splits.tar.gz
```

The locked source digests are:

```text
splits.tar.gz       54e3b9cdae5f5ff57a4210866f87e86686203556eea6a033200ff55c0f81e2b7
abag_split.csv      01414df16af9d3343994f527e13e81a25bf8da08198858018493a96f173a8cfb
instances.parquet   a32aa68bb6780d1222b310bab5e4efbf365c9219e08eb8a897379ceded565335
pairs.parquet       6ee920d94466b3afd2c7a225ef792fae6ae50e2cfb3e7785ba225de586fe2432
```

The contact adapter expects `instance_id`, `ab_res_idx`, `ag_res_idx`, and
`label`, with exactly one instance per Parquet row group. The completed cache
contains the full Cartesian grid for all 3,663 instances. It was built before
the new roles existed, so changing the split never rewrites or selectively
opens labels.

```bash
python scripts/materialize_sabdab2_contact_cache.py \
  --pairs /exact/path/pairs.parquet \
  --pairs-sha256 PAIRS_SHA256 \
  --structure-cache-root /work/framecontact/structure-cache \
  --structure-manifest-sha256 STRUCTURE_MANIFEST_SHA256 \
  --output-root /work/framecontact/contact-cache
```

Any incomplete grid, duplicate, non-binary label, length mismatch, digest
mismatch, or missing instance stops materialization and removes the temporary
output. Transfer the cache with `rsync --partial --append-verify`, then verify
the manifest and every artifact from the receipt on the destination.

## 2. Build and audit the structural-novelty split

Use exactly the 3,663 authenticated antigen structures. The implementation can
run one full query job or residue-balanced query shards; every shard searches
the same complete target database. Sharding is an execution optimization only:
the merger verifies every rectangular query-by-target grid and the final
normalizer proves the complete 3,663 squared ordered Cartesian grid.

Current Foldseek documents `--alignment-type 1` as global TM-align mode and
supports query- and target-normalized `qtmscore` and `ttmscore` output:

```bash
foldseek easy-search antigen_structures/ antigen_structures/ alignments.tsv foldseek_tmp/ \
  --alignment-type 1 \
  --format-output query,target,qtmscore,ttmscore
```

Official reference: [Foldseek output and alignment modes](https://github.com/steineggerlab/foldseek#readme).

For a sharded run, first freeze a deterministic plan and then submit the array
and dependency-bound merger:

```bash
python scripts/prepare_foldseek_query_shards.py \
  --export-root /work/framecontact/antigen-structures \
  --manifest-sha256 4882930043a4fc26319119f56b55a15bd729d23fb22ab04c485ce3ac79ada3d5 \
  --output-root /work/framecontact/foldseek/query-plan \
  --shard-count 4

sbatch --array=0-3 scripts/slurm_foldseek_sabdab2_shards.sbatch
# Submit scripts/slurm_merge_foldseek_sabdab2_shards.sbatch only with
# --dependency=afterok:<array-job-id> and the frozen plan digest.
```

When copying from macOS, use `COPYFILE_DISABLE=1 rsync ...` and reject
`._*`/`.DS_Store` sidecars. As a second independent guard, inspect Foldseek's
database lookup files: the four query counts must sum to 3,663 and every target
lookup must contain exactly 3,663 entries.

Accept neither the raw TSV nor scheduler success alone. The final receipt must
bind the Foldseek version, export manifest, identifier-set digest, shard
receipts, command, exact row count, and `test_labels_read: false`.

Build all three label-free manifests in one atomic operation:

```text
python scripts/build_sabdab2_split_matrix.py \
  --metadata /work/framecontact/authority/abag_split.csv \
  --metadata-sha256 01414df16af9d3343994f527e13e81a25bf8da08198858018493a96f173a8cfb \
  --prepared /work/framecontact/authority/instances.parquet \
  --prepared-sha256 a32aa68bb6780d1222b310bab5e4efbf365c9219e08eb8a897379ceded565335 \
  --foldseek-alignments /work/framecontact/foldseek/alignments.normalized.tsv \
  --foldseek-alignments-sha256 ALIGNMENTS_SHA256 \
  --foldseek-receipt /work/framecontact/foldseek/FOLDSEEK_RECEIPT.json \
  --foldseek-receipt-sha256 FOLDSEEK_RECEIPT_SHA256 \
  --train-cutoff 2022-12-31 \
  --validation-cutoff 2024-06-30 \
  --tm-threshold 0.50 \
  --assignment-seed 20260730 \
  --output-root /work/framecontact/split-matrix-v1
```

The output is `original.csv`, `temporal.csv`, `antigen_fold.csv`, and one
`SPLIT_MATRIX_RECEIPT.json`. Each manifest contains:

```text
instance_id,role,antigen_structure_id,antigen_sequence_sha256,antibody_sequence_sha256,pdb_assembly_id,deposition_date,antigen_cluster_id,antibody_cluster_id,source_split
```

`temporal.csv` assigns an entire exact-sequence/PDB-connected component from its
latest deposition date, preventing an older duplicate from leaking forward.
`antigen_fold.csv` unions every pair with `min(qTM,tTM) >= 0.50` together with
exact antigen, exact focal-antibody, and PDB-assembly constraints before role
assignment. Any cross-role hard edge or empty role fails closed.

After split construction, run `audit_framecontact_structural_split.py`. It
streams the full authenticated Cartesian table, verifies the exact row count,
and emits at most 1,000 pairs closest to 0.50. Run those pairs through the
binary pinned in `baselines/USALIGN_TOOL_LOCK.json` with
`run_usalign_boundary_spot_checks.py`. Any independently detected cross-role
US-align edge at or above 0.50 invalidates the split; a passing sample is still
reported as a spot check, not as a proof of model performance.

## 3. Freeze reciprocal validation and test partner panels before training

Protocol v3 supersedes the one-direction v1 panel for the first real pair run.
The panel metadata may contain only identity and training-free matching fields:

```text
instance_id,role,antigen_cluster_id,antigen_sequence_sha256,antigen_length,surface_residue_count,antigen_class,antibody_cluster_id,antibody_sequence_sha256,antibody_length,cdr_residue_count,antibody_class,documented_cognate_group_id
```

First compute role-independent antigen monomer SASA with antibody atoms removed,
Shrake-Rupley (1.4-A probe, 100 points/atom), and Tien-2013 theoretical MaxASA
normalization at RSA >= 0.20. This table reads no task labels. The preferred
builder atomically binds the same base table to all three manifests and creates
all nine role panels; one failure removes the entire temporary matrix:

```bash
python scripts/build_framecontact_split_panel_matrix.py \
  --partner-metadata-base /work/framecontact/partner-metadata-base/PARTNER_METADATA_BASE.csv \
  --partner-metadata-base-sha256 PARTNER_METADATA_BASE_SHA256 \
  --partner-metadata-base-receipt /work/framecontact/partner-metadata-base/PARTNER_METADATA_BASE_RECEIPT.json \
  --partner-metadata-base-receipt-sha256 PARTNER_METADATA_BASE_RECEIPT_SHA256 \
  --split-matrix-root /work/framecontact/split-matrix-v1 \
  --split-matrix-receipt-sha256 SPLIT_MATRIX_RECEIPT_SHA256 \
  --panel-size-per-direction 31 \
  --protocol-seed 20260730 \
  --output-root /work/framecontact/split-panel-matrix-v1
```

The builder rejects task-label/prediction columns and excludes any known
antibody-cluster by antigen-cluster cognate edge across the full cohort, not
only within the requested role. Seal the test panel but do not pass it to the
development trainer. Never reuse a panel whose role authority came from another
manifest; every path and digest is listed in `SPLIT_PANEL_MATRIX_RECEIPT.json`.

## 4. Run five A100 development seeds

```bash
export FRAMECONTACT_REPOSITORY_ROOT=/work/framecontact/geometry-self-attention
export FRAMECONTACT_STRUCTURE_CACHE_ROOT=/work/framecontact/structure-cache
export FRAMECONTACT_CONTACT_CACHE_ROOT=/work/framecontact/contact-cache
export FRAMECONTACT_EMBEDDING_ROOT=/work/framecontact/esm-embeddings
export FRAMECONTACT_EMBEDDING_MANIFEST_SHA256=9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545
export FRAMECONTACT_SPLIT_MANIFEST=/work/framecontact/split-matrix-v1/antigen_fold.csv
export FRAMECONTACT_SPLIT_MANIFEST_SHA256=...
export FRAMECONTACT_STRUCTURE_MANIFEST_SHA256=...
export FRAMECONTACT_CONTACT_MANIFEST_SHA256=...
export FRAMECONTACT_PROTOCOL_RECEIPT_SHA256=...
export FRAMECONTACT_SOURCE_MANIFEST=/work/framecontact/execution-authority/SABDAB2_TRAINING_SOURCE_MANIFEST.json
export FRAMECONTACT_SOURCE_MANIFEST_SHA256=...
export FRAMECONTACT_TRAINING_PANEL=/work/framecontact/TRAINING_PARTNER_PANEL.json
export FRAMECONTACT_TRAINING_PANEL_SHA256=...
export FRAMECONTACT_VALIDATION_PANEL=/work/framecontact/VALIDATION_PARTNER_PANEL.json
export FRAMECONTACT_VALIDATION_PANEL_SHA256=...
export FRAMECONTACT_RUN_ROOT=/work/framecontact/runs/framecontact-primary-v5
export FRAMECONTACT_PYTHON_BIN=/opt/venv/bin/python
bash scripts/run_framecontact_a100.sh
```

The runner requires an A100 by name, runs the complete test suite and a BF16
CUDA backward smoke, uses seeds `17 42 73 101 137`, refuses partial outputs, and
never receives a test path. By default it runs eight conditions:
`sequence_dense`, `spatial_topology`, `distance`, `distance_direction`,
`full_pose`, `frozen_frame_ipa`, `framecontact` (late fusion), and
`framecontact_interleaved`. The invariant-point condition is an IPA-style
fixed-backbone point-bias baseline, not a reproduction of complete AlphaFold
IPA. The six pair-MLP late-fusion conditions share one parameter skeleton; the
point-bias and interleaved models are separately declared architecture families.
Each seed selects its checkpoint from the frozen minimum
of the two validation partner margins, with contact AUPRC and then earlier epoch
as registered tie-breaks. Split AB02/AB03 work by assigning disjoint
`FRAMECONTACT_EXPERIMENTS` values and separate run roots, then merge only
complete receipt-bearing directories.

After all 40 runs are present:

```bash
python scripts/aggregate_framecontact_multiseed.py \
  --run-root /work/framecontact/runs/framecontact-primary-v5 \
  --output /work/framecontact/runs/framecontact-primary-v5/FRAMECONTACT_DEVELOPMENT_5SEED_SUMMARY.json
```

The aggregator verifies every result/checkpoint digest, equal parameter count
and initialization within each declared architecture family, common
data/protocol authority, and the absence of test access before reporting paired
development deltas. It does not authorize a frozen-test claim.

## 5. Freeze one five-seed ensemble on validation only

Do not choose an architecture manually. First freeze one five-seed ensemble for
each of the eight registered conditions using only its validation outputs.
Exactly the five pre-registered checkpoints receive uniform probability weight.
Each member's validation temperature is applied before probability averaging;
one final ensemble temperature and all three decision thresholds are then fit
on validation only. Repeat the following command with `EXPERIMENT` set to each
registered condition and keep eight separate output roots:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/freeze_framecontact_development_ensemble.py \
  --run-roots \
    /work/framecontact/runs/framecontact-primary-v5/EXPERIMENT/seed-17 \
    /work/framecontact/runs/framecontact-primary-v5/EXPERIMENT/seed-42 \
    /work/framecontact/runs/framecontact-primary-v5/EXPERIMENT/seed-73 \
    /work/framecontact/runs/framecontact-primary-v5/EXPERIMENT/seed-101 \
    /work/framecontact/runs/framecontact-primary-v5/EXPERIMENT/seed-137 \
  --structure-cache-root /work/framecontact/structure-cache \
  --structure-manifest-sha256 STRUCTURE_MANIFEST_SHA256 \
  --contact-cache-root /work/framecontact/contact-cache \
  --contact-manifest-sha256 CONTACT_MANIFEST_SHA256 \
  --embedding-root /work/framecontact/esm-embeddings \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest /work/framecontact/split-matrix-v1/antigen_fold.csv \
  --split-manifest-sha256 ANTIGEN_FOLD_SPLIT_SHA256 \
  --validation-partner-panel /work/framecontact/VALIDATION_PARTNER_PANEL.json \
  --validation-partner-panel-sha256 VALIDATION_PANEL_SHA256 \
  --output-root /work/framecontact/runs/framecontact-primary-v5/ensembles/EXPERIMENT \
  --evaluation-batch-size 1 --device cuda --precision bf16
```

Every receipt must state `test_loaded: false` and bind all five
result/checkpoint digests, common data/protocol authority, split, validation
panel, calibration, thresholds, and the exact ensemble rule. Once all eight
exist, freeze the unique winner with the pre-registered primary metric and
tie-breaks:

```bash
python scripts/freeze_framecontact_development_selection.py \
  --ensemble-roots \
    /work/framecontact/runs/framecontact-primary-v5/ensembles/sequence_dense \
    /work/framecontact/runs/framecontact-primary-v5/ensembles/spatial_topology \
    /work/framecontact/runs/framecontact-primary-v5/ensembles/distance \
    /work/framecontact/runs/framecontact-primary-v5/ensembles/distance_direction \
    /work/framecontact/runs/framecontact-primary-v5/ensembles/full_pose \
    /work/framecontact/runs/framecontact-primary-v5/ensembles/frozen_frame_ipa \
    /work/framecontact/runs/framecontact-primary-v5/ensembles/framecontact \
    /work/framecontact/runs/framecontact-primary-v5/ensembles/framecontact_interleaved \
  --protocol /work/framecontact/protocol/framecontact_primary_protocol_v5.json \
  --protocol-receipt /work/framecontact/protocol/FRAMECONTACT_PROTOCOL_V5_RECEIPT.json \
  --protocol-receipt-sha256 PROTOCOL_RECEIPT_SHA256 \
  --output-root /work/framecontact/runs/framecontact-primary-v5/selection
```

The selection command authenticates exactly one candidate for every registered
condition and refuses to run if any ensemble has accessed test data. Record the
selected ensemble path from `SABDAB2_DEVELOPMENT_SELECTION.json`; do not edit or
copy the selection result to change that path.

## 6. Open the internal SAbDab2 frozen test exactly once

Only after Section 5 is immutable should the test role and test panel be
mounted. Create a new empty state directory and freeze one hash-bound opening
receipt before constructing the test dataset. `SELECTED_EXPERIMENT` and
`SELECTED_ENSEMBLE` below must be read from the immutable selection result:

```bash
mkdir /work/framecontact/test-state/sabdab2-antigen-fold-v1

python scripts/freeze_sabdab2_frozen_test_opening.py \
  --selection-root /work/framecontact/runs/framecontact-primary-v5/selection \
  --selection-receipt-sha256 SELECTION_RECEIPT_SHA256 \
  --run-roots \
    /work/framecontact/runs/framecontact-primary-v5/SELECTED_EXPERIMENT/seed-17 \
    /work/framecontact/runs/framecontact-primary-v5/SELECTED_EXPERIMENT/seed-42 \
    /work/framecontact/runs/framecontact-primary-v5/SELECTED_EXPERIMENT/seed-73 \
    /work/framecontact/runs/framecontact-primary-v5/SELECTED_EXPERIMENT/seed-101 \
    /work/framecontact/runs/framecontact-primary-v5/SELECTED_EXPERIMENT/seed-137 \
  --structure-manifest /work/framecontact/structure-cache/STRUCTURE_CACHE_MANIFEST.json \
  --structure-manifest-sha256 STRUCTURE_MANIFEST_SHA256 \
  --contact-manifest /work/framecontact/contact-cache/CONTACT_CACHE_MANIFEST.json \
  --contact-manifest-sha256 CONTACT_MANIFEST_SHA256 \
  --embedding-manifest /work/framecontact/esm-embeddings/SABDAB2_ESM_EMBEDDING_MANIFEST.json \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest /work/framecontact/split-matrix-v1/antigen_fold.csv \
  --split-manifest-sha256 ANTIGEN_FOLD_SPLIT_SHA256 \
  --test-partner-panel /work/framecontact/TEST_PARTNER_PANEL.json \
  --test-partner-panel-sha256 TEST_PANEL_SHA256 \
  --same-antigen-challenge /work/framecontact/same-antigen/test.json \
  --same-antigen-challenge-sha256 SAME_ANTIGEN_TEST_PANEL_SHA256 \
  --evaluator-source scripts/evaluate_framecontact_sabdab2_ensemble_test.py \
  --test-state-root /work/framecontact/test-state/sabdab2-antigen-fold-v1 \
  --output /work/framecontact/test-state/SABDAB2_FROZEN_TEST_OPENING.json
```

The opening command verifies the unique validation winner, five checkpoint
digests, complete test axis, test panel, same-antigen authority, all evaluation
source hashes, 31-partner panels, 10,000 bootstrap replicates, fixed seed, and
every required intervention. It does not load a test item or consume the
scoring session.

The scorer then reports every seed and the frozen ensemble, uses no test-time
fitting, and adds 10,000-replicate cluster bootstrap intervals to both
partner-retrieval directions:

```bash
python scripts/evaluate_framecontact_sabdab2_ensemble_test.py \
  --opening-receipt /work/framecontact/test-state/SABDAB2_FROZEN_TEST_OPENING.json \
  --opening-receipt-sha256 OPENING_RECEIPT_SHA256 \
  --selection-root /work/framecontact/runs/framecontact-primary-v5/selection \
  --selection-receipt-sha256 SELECTION_RECEIPT_SHA256 \
  --ensemble-root /work/framecontact/runs/framecontact-primary-v5/ensembles/SELECTED_EXPERIMENT \
  --run-roots \
    /work/framecontact/runs/framecontact-primary-v5/SELECTED_EXPERIMENT/seed-17 \
    /work/framecontact/runs/framecontact-primary-v5/SELECTED_EXPERIMENT/seed-42 \
    /work/framecontact/runs/framecontact-primary-v5/SELECTED_EXPERIMENT/seed-73 \
    /work/framecontact/runs/framecontact-primary-v5/SELECTED_EXPERIMENT/seed-101 \
    /work/framecontact/runs/framecontact-primary-v5/SELECTED_EXPERIMENT/seed-137 \
  --structure-cache-root /work/framecontact/structure-cache \
  --structure-manifest-sha256 STRUCTURE_MANIFEST_SHA256 \
  --contact-cache-root /work/framecontact/contact-cache \
  --contact-manifest-sha256 CONTACT_MANIFEST_SHA256 \
  --embedding-root /work/framecontact/esm-embeddings \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest /work/framecontact/split-matrix-v1/antigen_fold.csv \
  --split-manifest-sha256 ANTIGEN_FOLD_SPLIT_SHA256 \
  --test-partner-panel /work/framecontact/TEST_PARTNER_PANEL.json \
  --test-partner-panel-sha256 TEST_PANEL_SHA256 \
  --same-antigen-challenge /work/framecontact/same-antigen/test.json \
  --same-antigen-challenge-sha256 SAME_ANTIGEN_TEST_PANEL_SHA256 \
  --bootstrap-replicates 10000 --bootstrap-seed 20260730 \
  --evaluation-batch-size 1 \
  --output-root /work/framecontact/results/sabdab2-antigen-fold-frozen-v1 \
  --device cuda --precision bf16
```

Immediately before constructing the first authenticated test item, the scorer
creates `SABDAB2_TEST_SCORE_CONSUMED.json` with create-only filesystem
semantics. A crash is fail-closed: preserve the token and write a recovery
receipt; never delete the token or silently rerun. The scorer also checks that
the constructed dataset count equals the count frozen in the opening receipt.

The primary decision is not overall epitope AUPRC. It is the fixed-antigen,
wrong-antibody interaction-only compatibility margin with a cluster-bootstrap
95% lower bound above zero. Report contact AUPRC/P@K, epitope/paratope metrics,
calibration, reciprocal retrieval, and all negative controls even when the
primary endpoint fails.

## 7. Independent external opening gate

Do not mount or copy test labels into the training container. Before one-time
test scoring, freeze:

1. all five selected checkpoint digests;
2. ensemble rule;
3. validation temperatures and thresholds;
4. test partner-panel digest;
5. scorer source digest;
6. structural audit digest;
7. bootstrap seed and replicate count.

The external scorer must run in a separate output directory, refuse overwrite,
and state `test_open_count: 1`. The gate and scorer are implemented in
`scripts/freeze_framecontact_external_opening.py` and
`scripts/score_framecontact_external.py`. The scorer consumes label-free
prediction arrays and a separately authenticated truth archive so inference and
scoring custody can remain separate. Until real test features/truth authority,
five-seed ensemble predictions, and validation-only threshold authority exist,
the project remains confirmatory-blocked.

The expected array schemas and every digest check are executable in
`tests/test_framecontact_external_scorer.py`; that synthetic test proves scorer
behavior only, not external model performance.

Once the five-seed ensemble validation predictions and thresholds authority
exist, freeze the opening receipt before mounting truth:

```bash
python scripts/freeze_framecontact_external_opening.py \
  --development-summary FRAMECONTACT_DEVELOPMENT_5SEED_SUMMARY.json \
  --development-summary-sha256 ... \
  --structural-audit STRUCTURAL_SPLIT_AUDIT.json \
  --structural-audit-sha256 ... \
  --test-partner-panel TEST_PARTNER_PANEL.json \
  --test-partner-panel-sha256 ... \
  --thresholds FRAMECONTACT_VALIDATION_THRESHOLDS.json \
  --thresholds-sha256 ... \
  --checkpoints seed17.pt seed42.pt seed73.pt seed101.pt seed137.pt \
  --checkpoint-receipts seed17/RUN_RECEIPT.json seed42/RUN_RECEIPT.json seed73/RUN_RECEIPT.json seed101/RUN_RECEIPT.json seed137/RUN_RECEIPT.json \
  --test-truth-authority-sha256 DATA_CUSTODIAN_SUPPLIED_DIGEST \
  --scorer-source scripts/score_framecontact_external.py \
  --output FRAMECONTACT_EXTERNAL_OPENING_RECEIPT.json
```

After the inference custodian exports the label-free ensemble arrays, perform
the single opening:

```bash
python scripts/score_framecontact_external.py \
  --opening-receipt FRAMECONTACT_EXTERNAL_OPENING_RECEIPT.json \
  --opening-receipt-sha256 ... \
  --predictions FRAMECONTACT_EXTERNAL_PREDICTIONS.npz \
  --predictions-sha256 ... \
  --truth FRAMECONTACT_EXTERNAL_TRUTH.npz \
  --truth-sha256 DATA_CUSTODIAN_SUPPLIED_DIGEST \
  --thresholds FRAMECONTACT_VALIDATION_THRESHOLDS.json \
  --thresholds-sha256 ... \
  --output-root /work/framecontact/results/frozen-external-v1
```

## 8. AsEP official-test one-shot protocol

Do not use the generic external scorer as a substitute for the native AsEP
authority. Complete all 40 development runs and freeze all eight five-seed
ensembles, then run `freeze_asep_development_selection.py` with exactly those
eight roots. The unique winner is chosen by the registered minimum directional
validation margin, with mean per-complex contact AP and frozen architecture
order as tie-breaks. Its receipt must still state `test_loaded: false`.

Only after selection may `materialize_asep_label_free_test_metadata.py` open the
official test PDB members. It does not open graph tensors or labels. Use those
PDBs to run the complete AsEP-test versus SAbDab2-train/validation Foldseek
rectangle, finalize the normalized table, run
`audit_asep_sabdab2_decontamination.py`, and create both partner panels with
`build_asep_label_free_test_partner_panel.py`:

- `full_official_test` contains every official test instance;
- `sabdab2_decontaminated_primary` excludes only instances pre-flagged by exact
  PDB, exact PDB-SEQRES, or the frozen reciprocal-min Foldseek threshold.

The clean subset is a secondary reporting scope, not a replacement leaderboard
and never a prediction-driven exclusion. Wrong partners are label-free ranking
controls, not experimentally verified non-binders.

Freeze the opening before any official graph tensor is read:

```bash
mkdir -p /work/framecontact/asep/test-state
python scripts/freeze_asep_official_test_opening.py \
  --selection-root /work/framecontact/asep/selection \
  --selection-receipt-sha256 SELECTION_RECEIPT_SHA256 \
  --run-roots SEED17 SEED42 SEED73 SEED101 SEED137 \
  --inner-audit /work/framecontact/asep/INNER_AUDIT.json \
  --inner-audit-sha256 INNER_AUDIT_SHA256 \
  --split-root /work/framecontact/asep/splits \
  --split-receipt-sha256 SPLIT_RECEIPT_SHA256 \
  --split-method epitope_group \
  --label-free-test-root /work/framecontact/asep/label-free-test \
  --label-free-test-receipt-sha256 LABEL_FREE_RECEIPT_SHA256 \
  --decontamination-root /work/framecontact/asep/decontamination \
  --decontamination-receipt-sha256 DECONTAMINATION_RECEIPT_SHA256 \
  --full-test-panel /work/framecontact/asep/full-test-panel.json \
  --full-test-panel-sha256 FULL_PANEL_SHA256 \
  --clean-test-panel /work/framecontact/asep/clean-test-panel.json \
  --clean-test-panel-sha256 CLEAN_PANEL_SHA256 \
  --evaluator-source scripts/evaluate_asep_frozen_test.py \
  --test-state-root /work/framecontact/asep/test-state \
  --output /work/framecontact/asep/ASEP_OFFICIAL_TEST_OPENING.json
```

The receipt binds the unique selection, all five result/checkpoint hashes,
archive and split hashes, decontamination result, both panels, metrics,
validation-only calibration rule, bootstrap, evaluator, adapter, and imported
evaluation dependencies. Then materialize the graph/label cache exactly once:

```bash
python scripts/materialize_asep_frozen_test_cache.py \
  --authority-root /work/framecontact/asep/authority \
  --inner-audit /work/framecontact/asep/INNER_AUDIT.json \
  --inner-audit-sha256 INNER_AUDIT_SHA256 \
  --split-root /work/framecontact/asep/splits \
  --split-receipt-sha256 SPLIT_RECEIPT_SHA256 \
  --split-method epitope_group \
  --opening-receipt /work/framecontact/asep/ASEP_OFFICIAL_TEST_OPENING.json \
  --opening-receipt-sha256 OPENING_SHA256 \
  --output-root /work/framecontact/asep/frozen-test-cache
```

Finally run `evaluate_asep_frozen_test.py` once with that cache, the winner's
five run roots, and both panels. It reuses validation temperatures and thresholds,
reports all five seeds plus the ensemble on the full official test, reports the
ensemble separately on the decontaminated subset, adds cluster bootstrap,
calibration regression, reciprocal partner retrieval, independent-monomer SE(3)
invariance, geometry shuffle, local-frame corruption, degree-matched rewiring,
and 0.10/0.25/0.50-A coordinate perturbations. Before deserializing the first
cached label it creates `ASEP_TEST_SCORE_CONSUMED.json` with exclusive-create
semantics. If the process crashes, preserve the token, disclose the failure, and
freeze a recovery protocol; never delete it and rerun silently.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/evaluate_asep_frozen_test.py \
  --opening-receipt /work/framecontact/asep/ASEP_OFFICIAL_TEST_OPENING.json \
  --opening-receipt-sha256 OPENING_SHA256 \
  --selection-root /work/framecontact/asep/selection \
  --selection-receipt-sha256 SELECTION_RECEIPT_SHA256 \
  --ensemble-root /work/framecontact/asep/ensembles/SELECTED_EXPERIMENT \
  --run-roots SEED17 SEED42 SEED73 SEED101 SEED137 \
  --test-cache-root /work/framecontact/asep/frozen-test-cache \
  --test-cache-manifest-sha256 TEST_CACHE_MANIFEST_SHA256 \
  --test-cache-receipt-sha256 TEST_CACHE_RECEIPT_SHA256 \
  --full-test-panel /work/framecontact/asep/full-test-panel.json \
  --full-test-panel-sha256 FULL_PANEL_SHA256 \
  --clean-test-panel /work/framecontact/asep/clean-test-panel.json \
  --clean-test-panel-sha256 CLEAN_PANEL_SHA256 \
  --output-root /work/framecontact/asep/results/official-test-v1 \
  --evaluation-batch-size 8 --workers 4 --device cuda --precision bf16
```

## 9. External architecture baselines

`baselines/BASELINE_LOCK.json` preserves the first source audit;
`baselines/BASELINE_LOCK_V2.json` adds pinned official revisions for AsEP/WALLE,
EpiFormer, SPPIDER-seq, and SE3Bind without rewriting v1. Eight internal
conditions are runnable, including an explicitly labeled IPA-style point-bias
baseline. A pinned commit or implemented adapter proves no benchmark result.

`baselines/BASELINE_LOCK_V4.json` records the later GVP/EGNN source checkouts,
FlashIPA dependency blocker, and the authenticated real Boltz-1 Pairformer
extraction/adapter/synthetic smoke. None has an identical-split biological
result yet.

The ParaCloud 24-GiB run exposed a separate runtime boundary: all five original
AsEP `frozen_frame_ipa` tasks OOMed while materializing the naive
`[B,L,L,H,P,3]` point delta. The hash-bound
`framecontact_invariant_point_memory_amendment_v1.json` permits only an
algebraically equivalent norm expansion and resubmission of the five failed or
preemptively cancelled IPA units. Do not silently drop those seeds or mix their
recovery source with another condition. Preserve the original OOM logs and the
new recovery source manifest in the handoff.

Do not describe GVP/EGNN/complete AlphaFold IPA/Pairformer, WALLE, EpiFormer, SPPIDER-seq, or
SE3Bind as reproduced until an adapter and run/scorer receipts exist. Native
AsEP results must be reported separately from identical-split SAbDab2 results.
SPPIDER-seq's published training explicitly excluded antibody-epitope
complexes, so antibody-antigen use is transfer evidence. SE3Bind is a future
pose/docking reference, not a residue-contact baseline. A released checkpoint
and a from-scratch run must never be conflated.
