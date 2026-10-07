# Same-AA shuffling isolates correspondence damage

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

ESM-IF1 identity AP 0.027995 becomes 0.011100 under same-AA shuffle; component-weighted paired effect about −0.017151.

| Field | Value |
|---|---|
| workstream | Correspondence |
| cohort | 217 development complexes /68 components |
| inputs | Same-amino-acid permutation retains residue types and feature multiset but breaks correspondence |
| seeds | 3 fixed seeds |
| estimand | Complex-weighted levels versus component/seed-weighted effects; crossed intervals |

## What this does not establish

Synthetic sensitivity does not establish natural misregistration prevalence. Same-AA shuffling is not correspondence-preserving.

## Pinned evidence

1. [extensions/correspondence_v2/READOUT.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/14a385feaec24b3141331963ba71dda788a164e1/extensions/correspondence_v2/READOUT.md)
   - Branch `kevin/correspondence-upgrade-20260930`; commit `14a385feaec24b3141331963ba71dda788a164e1`.
   - Git blob `054ff8b771c6368b660b1b3eb408b571b51e0892`; SHA-256 `be6f680bac4d24e561c4466b279532d43b1bfd2d9e629565248caac18a4ddc88`.
2. [docs/CLAIM_BOUNDARIES.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/docs/CLAIM_BOUNDARIES.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `36f24360004d53790cffe74199fdab909c77f2e6`; SHA-256 `285394e4f62f18c7bb8b0ee41a63aa22e3c3a586681e96801e9627ba256b194f`.

## Exact checked registry values

| ID | Value | Estimator |
|---|---|---|
| `corr_esm_if1_identity` | `0.027994970888844756` | mean per-complex AP, mean of seeds 1/2/3 |
| `corr_esm_if1_same_aa` | `0.011100352687647625` | mean per-complex AP, mean of seeds 1/2/3 |
| `corr_esm_if1_delta` | `-0.017151325302456014` | equal-component/equal-seed paired effect; crossed seed/component interval |
| `corr_esm_if1_ci` | `[-0.022688197094407773, -0.012434039894905642]` | equal-component/equal-seed paired effect; crossed seed/component interval |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
