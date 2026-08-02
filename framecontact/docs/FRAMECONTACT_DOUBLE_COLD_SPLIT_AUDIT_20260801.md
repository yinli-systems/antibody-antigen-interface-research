# FrameContact SAbDab2 double-cold split audit

Date: 2026-08-01

## Outcome

A new **label-free, unopened-test confirmatory axis** is ready for the model
selected on the existing development arrays. It combines official SAbDab2
sequence-cluster authorities with reciprocal-min Foldseek structural hard
edges. It does not contain model predictions or task labels, and it does not
yet provide a generalization or benchmark-improvement result.

The final antigen-fold manifest is:

- path: `receipts/double-cold-split-v1-1346323/final-tokenized-v4-1346364/antigen_fold.csv`;
- SHA-256: `07b20645982971e9ff0004ac297d9a03ac0278269b4e8d84628fb0cbbb87804d`;
- roles: 2,584 train / 540 validation / 539 test;
- test labels read: false;
- predictions read: false.

The existing five-condition and distogram arrays were not modified. They remain
architecture-development experiments on antigen-fold v3. Once their
validation-only choice is frozen, the fixed winner must be retrained from
scratch on the new train/validation roles. The new test role remains unopened
until that second selection is sealed.

## Official SAbDab2 semantics

The authority is the official SAbDab2 ML Dataset v0.1.0, DOI
`10.5281/zenodo.20083995`. The authenticated local archive SHA-256 is
`54e3b9cdae5f5ff57a4210866f87e86686203556eea6a033200ff55c0f81e2b7`;
its `splits_final/README.md` SHA-256 is
`2d0a50eeb0ada3c4c452eb8a8e1a7a81ab63a9cf34f20df135db607ea1632690`.

The README defines the following authorities:

| Field / authority | Exact meaning used here |
|---|---|
| `ab_cluster` | final antibody cluster for the antibody-only split; connected through CDRH3, CDRH123 or CDRL123 clusters at 85% identity |
| `agclusters` | one antigen sequence-cluster ID per antigen chain, stored as a slash-separated string |
| antigen-aware antibody thresholds | CDRH3 95%; CDRH123 and CDRL123 100% |
| antigen thresholds | protein 40%; peptide, DNA and RNA 80% |
| `ab_ag_cluster` | official connected component joining the antigen-aware antibody and antigen base clusters |

FrameContact deliberately unions the more inclusive antibody-only
`ab_cluster`, every individual token in `agclusters`, and the structural
authorities below. These are sequence clusters. They are **not** germline,
clone, clonotype, or structural-fold labels.

The first implementation treated the entire slash-separated antigen field as a
single value. The official README audit caught that semantic error. The final
implementation tokenizes every chain cluster in both component construction and
post-hoc leakage auditing. On this 3,663-instance cohort, the corrected build
produced the same role CSV hash as the first build; therefore no realized
cross-role leak was present, but the v4 code now enforces the correct invariant
for this and future releases. The earlier receipts remain preserved as an
engineering failure/recovery trail.

## Exact isolation contract

Two instances are forced into the same connected component when any of these
conditions hold:

1. same PDB assembly;
2. exact antigen sequence SHA-256;
3. exact antibody sequence SHA-256;
4. any shared SAbDab2 antigen-chain cluster token;
5. same SAbDab2 antibody-only final cluster; or
6. a reciprocal-min Foldseek TM-score hard edge at or above 0.50.

The Foldseek authority contains 13,417,569 ordered alignment rows and yields
725,687 hard edges. The final split has zero hard edges crossing roles and zero
cross-role PDB, exact-sequence, antigen-cluster-token, or antibody-cluster
values.

| Audit | antigen-fold v3 development | new double-cold confirmatory |
|---|---:|---:|
| instances | 3,663 | 3,663 |
| exact antigen-sequence values crossing roles | 0 | 0 |
| exact antibody-sequence values crossing roles | 0 | 0 |
| PDB assemblies crossing roles | 0 | 0 |
| Foldseek hard edges crossing roles | 0 | 0 |
| SAbDab2 antigen-cluster tokens crossing roles | 8 values / 240 rows | 0 |
| SAbDab2 antibody clusters crossing roles | 11 values / 64 rows | 0 |
| double-cold valid | no | yes |

The split change is material: 1,482 of 3,663 instances change role relative to
development v3. The complete transition matrix is:

| old role -> new role | instances |
|---|---:|
| test -> test | 156 |
| test -> train | 177 |
| test -> validation | 216 |
| train -> test | 324 |
| train -> train | 1,971 |
| train -> validation | 270 |
| validation -> test | 59 |
| validation -> train | 436 |
| validation -> validation | 54 |

These migrations are why the current development checkpoints cannot simply be
scored on the new split. They must be retrained after architecture selection.

## Component-imbalance caveat

Foldseek alone gives 245 components with a largest component of 1,039. Adding
exact equivalence and both SAbDab2 cluster authorities gives 184 components and
one 2,584-instance component. That component is the entire train role, while
the remaining components fill validation and test.

This satisfies the isolation contract but changes the estimand: the holdouts
are the components disconnected from a very large antibody--antigen similarity
network. It is a hard extrapolation setting, not an IID estimate, and the
component imbalance may increase variance. Reports must therefore include:

- per-complex and pooled metrics;
- antigen- and antibody-cluster bootstrap intervals;
- component-size-stratified results;
- five seed dispersion;
- both partner-retrieval directions; and
- a comparison to the temporal split and official AsEP test, without using any
  frozen result for model selection.

## Label-free wrong-partner panels

The original role column was removed from an authenticated label-free metadata
table by a schema-exact projection. Any extra task column causes failure. The
projected base is then rebound to the new split.

- projected base SHA-256:
  `d7668d20a898a10569dc5fa0a4f0a888bd0f8636236a4c6cecc35d469307bd59`;
- new split-bound metadata SHA-256:
  `c1fd6b93bb063be0c17ae3bc8cb287136a19e71fae825f300482cf88fffee75d`;
- train reciprocal-31 panel SHA-256:
  `18fa306da6fe59905d58268b50a95b6fbd3479478bfb21acf88d0faed1f60c74`;
- validation reciprocal-31 panel SHA-256:
  `5b80013e62af2591a2e1247b26cdf9dee452262aca24cba18fe1204347a9ff75`.

There are 2,584 train queries and 540 validation queries in each reciprocal
direction. The bundle receipt explicitly records `test_panel_materialized:
false`. A wrong pair is a ranking example, never a fabricated all-zero contact
or distogram label.

## Research consequence

Three new results sharpen the project hypothesis:

1. The July-2026 germline-shortcut study reports that random splits inflated
   immune-receptor top-1 retrieval by 15--28 percentage points. This motivates
   CDR/cluster isolation and prevents interpreting frozen full-sequence ESM
   features as antigen specificity by default.
2. The 2026 shuffled-complex study found that AF3, Boltz-2 and Chai-1 confidence
   often failed to separate 106 cognate complexes from 11,342 non-cognate
   pairings. This makes cognate ranking a primary result, not a decorative
   shuffle control.
3. ICLR-2026 RaftPPI shows that confidence-weighted hard negatives and
   retrieval-scale evaluation improve practical PPI screening. FrameContact can
   borrow hard-negative mining and Recall@K, while retaining its explicit
   residue-contact head for localization.

The resulting high-value claim is narrower and harder than “geometry raises
epitope AUPRC”: a contact-first, independently invariant model should rank the
correct partner and localize its interface under both sequence-cluster and
structural novelty. Until the one-time test is run, this remains a registered
hypothesis rather than a result.

## Frozen execution order

1. Finish current five-seed development arrays unchanged.
2. Freeze the validation-only winning architecture, marginal head, calibration
   and checkpoint rule.
3. Compare K=1 versus K=4 listwise specificity on train/validation only, with
   contact and calibration non-degradation gates.
4. Retrain the fixed winner on the new double-cold train role and select only on
   its validation role.
5. Freeze all five checkpoints, ensemble rule, temperature, threshold, panel,
   scorer and source manifest.
6. Open the 539-instance test role once, then run official/decontaminated AsEP
   exactly once under its separately frozen gate.
7. Only after these scientific gates, move the selected contact representation
   into CHIMERA co-design and the sparse/fused systems track.

## Evidence boundary

This document proves split construction, leakage auditing, label-free partner
panel construction, regression behavior, and provenance. It does not prove
model quality, external generalization, state of the art, experimental binding,
or a GPU speedup. Those claims require the later frozen runs and their own
receipts.
