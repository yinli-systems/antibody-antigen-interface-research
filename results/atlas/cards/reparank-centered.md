# RePaRank structure: completed and correctly centered

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

Centered structure macro AP 0.0160136 versus sequence 0.0378078; pooled AP 0.00236625 /AUROC 0.672153. Complete 217/217: 12 reused+205 fresh predictions.

| Field | Value |
|---|---|
| workstream | External baselines |
| cohort | Same 217-complex pair grid as sequence checkpoint |
| inputs | Official-style per-chain centering; independently predicted chain-local geometry |
| seeds | One released checkpoint; reuse is not replication |
| estimand | Macro AP versus pooled metrics; component contrasts remain separate |

## What this does not establish

Uncentered results are a protocol-mismatch control. This does not establish that structure is useless: no correctly docked complex pose is supplied and checkpoint overlap is unknown.

## Pinned evidence

1. [extensions/frontier_multimodel_v1/evidence/reparank_centered_corrected_v1/evaluation.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc/extensions/frontier_multimodel_v1/evidence/reparank_centered_corrected_v1/evaluation.json)
   - Branch `kevin/rescue-qualification-20260930`; commit `a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc`.
   - Git blob `21475fd3021661f7b32c9c4e40fa693a42bfcab2`; SHA-256 `e6ce65b7219cbad93b456a1a011a7681fa84c0b9a034768b77365e9919887bc9`.
2. [results/INDEX.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/INDEX.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `383e5a2c8ef5f3fbebfe6251e65a71f6d0a2ef5d`; SHA-256 `d585cf1bcf7623ecdcae26a09bb5c1761d25a54884ea3f8b35159d8dbbdd1076`.

## Exact checked JSON fields

| Selector | Value |
|---|---|
| `["instance_count"]` | `217` |
| `["primary_macro_mean_per_complex_AP"]` | `0.016013622977683228` |
| `["pooled_pair_metrics", "average_precision"]` | `0.0023662535231304227` |
| `["pooled_pair_metrics", "auroc"]` | `0.6721526555729059` |
| `["reused_pair_prediction_count"]` | `12` |
| `["fresh_model_forward_count"]` | `205` |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
