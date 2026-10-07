# Official RePaRank sequence checkpoint

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

Macro pair AP 0.0378078; pooled AP 0.00873858; pooled AUROC 0.786064. These are different estimators, not interchangeable scores.

| Field | Value |
|---|---|
| workstream | External baselines |
| cohort | 217 development complexes /68 components; 9,638,880 pairs /9,036 contacts |
| inputs | Released sequence checkpoint, no new head/fine-tuning |
| seeds | One released checkpoint |
| estimand | Macro, pooled and equal-component results separate |

## What this does not establish

External-baseline measurement is not a new algorithmic gain. Training overlap unknown; do not rank against FrameContact test 549.

## Pinned evidence

1. [extensions/frontier_multimodel_v1/evidence/reparank_same_protocol_v1/sequence_full/evaluation.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/extensions/frontier_multimodel_v1/evidence/reparank_same_protocol_v1/sequence_full/evaluation.json)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `e9d21731ff0c55952054dd199213d1b533574d31`; SHA-256 `3b41d0943eaea3113408274508a03fbc64282c06a26d56d58a8e3ac6837907ce`.
2. [docs/CLAIM_BOUNDARIES.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/docs/CLAIM_BOUNDARIES.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `36f24360004d53790cffe74199fdab909c77f2e6`; SHA-256 `285394e4f62f18c7bb8b0ee41a63aa22e3c3a586681e96801e9627ba256b194f`.

## Exact checked registry values

| ID | Value | Estimator |
|---|---|---|
| `reparank_macro_ap` | `0.03780779120862989` | one deterministic checkpoint; named macro/pooled/component statistic |
| `reparank_pooled_ap` | `0.008738584064940456` | one deterministic checkpoint; named macro/pooled/component statistic |
| `reparank_pairs` | `9638880` | one deterministic checkpoint; named macro/pooled/component statistic |
| `reparank_positives` | `9036` | one deterministic checkpoint; named macro/pooled/component statistic |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
