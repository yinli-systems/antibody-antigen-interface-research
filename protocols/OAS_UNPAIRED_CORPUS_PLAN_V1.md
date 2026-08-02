# OAS unpaired corpus plan v1

**Status:** independently audited plan frozen; full data admission has not started

**Frozen date:** 2026-07-28

## Scope

This plan selects a large, loss-independent phase-A corpus from the official
OAS unpaired authority manifest. It is independent of SAbDab2 held-out data,
predictions, losses, and downstream metrics. Expected prefilter counts are
planning quantities, not downloaded, admitted, deduplicated, or consumed
counts.

The authority contains 15,580 data units, 91 study labels, and 2,428,152,691
reported unique sequences. The training plan requests exactly 250,000,000
prefilter records:

- 180,000,000 human heavy-chain records;
- 45,000,000 human light-chain records;
- 23,500,000 nonhuman heavy-chain records;
- 1,500,000 nonhuman light-chain records.

The plan uses square-root study weighting and caps any one study at 12.5% of a
training stratum. The nonhuman-light stratum has an explicit two-thirds cap
because only ten authority studies contain such records and two are held out.
The realized maximum is 921,623 / 1,500,000. This rare-stratum limitation must
remain visible in any report.

The upstream `backup` directory is retained in authority totals but excluded
from training and validation because its directory label is not a canonical
study identifier. This is a provenance exclusion made before training.

## Validation isolation

Eight complete study labels are excluded from training before any loss is
observed:

- `Bernat_2019_1`
- `Chen_2020`
- `Cui_2019`
- `Fisher_2017`
- `Jaffe_2022`
- `Lindner_2015`
- `Prohaska_2018`
- `Setliff_2018`

The plan requests 1,910,348 validation records across those studies. Exact
sequence duplicates crossing validation and training are removed from
training during corpus materialization. A validation study never contributes
a training data unit.

## Frozen artifacts

The final plan requires 6,373 source data units containing 265,834,820 source
rows. Sixty-five terminal units use an exact rational SHA-256 sampling
threshold; all other selected units are consumed in full before row-quality
filtering and deduplication.

- cluster plan: `/ssd/scxi253/model_deployment/data/oas_pretraining/corpus_plan_250m_v1_v4/plan.json`
- plan file SHA-256: `45f7d002400dd81d9afa5f6b7daff6b6e67d7557aa2e379a031a5ac3b1f2b1b0`
- plan payload SHA-256: `fba741bc6666067f1601f2096d31507c1b38edb8af95118110eaf1cef7e059e0`
- independent audit file SHA-256: `e139ac62fec4b2a2d21c5015e0a70a1273f6eb429fce55acf17a2937007bbdbf`
- independent audit payload SHA-256: `5a26e4403ab4198c0de1f51340cfad81483f8cc7fcf7e988bc437b7501354e00`

The independent auditor imports neither the planner nor its allocation code.
It independently reclassifies authority entries, selects validation studies,
recomputes capped square-root allocations, reconstructs unit membership, and
compares the complete result.

## Failure history

The first real-manifest attempt used a 10% per-study cap. It failed closed
because human-light and nonhuman-heavy capacities were insufficient after the
whole-study holdout. The empty failure directory is preserved as
`corpus_plan_250m_v1_candidate`. Candidate v2 demonstrated feasibility at a
12.5% cap but did not bind the core planner module hash. Candidate v3 added the
maintenance-directory exclusion but still preceded the complete provenance
binding. Only numbered v4 is authoritative.

No pretraining run may describe these planned records as downloaded or
consumed. Production training begins only after raw admission, corpus
materialization, deterministic replay, and the one-GPU correctness gates pass.
