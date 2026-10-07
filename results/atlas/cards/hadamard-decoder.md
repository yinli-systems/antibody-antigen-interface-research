# Hadamard decoder: no demonstrated win

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `metric-bound` — Committed numeric fields checked; no model rerun

Bilinear macro pair AP 0.0239708 versus Hadamard 0.0226796. No superiority or equivalence of decoder classes follows from one seed.

| Field | Value |
|---|---|
| workstream | Objectives & controls |
| cohort | 217 validation complexes /9,638,880 pairs |
| inputs | Matched bilinear versus Hadamard MLP |
| seeds | One seed 42; both checkpoint epoch 1 |
| estimand | Macro pair AP; other endpoints disagree in direction |

## What this does not establish

AP is not top-k precision: the legacy 97.6%-false-top-pairs deduction from AP 0.024 is invalid. Preserve the original post-prediction aggregation failure.

## Pinned evidence

1. [results/pair_decoder_ablation_v1/report_compact.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/347acbdfb10a15a2b4081a676b0364d3b83ee8dc/results/pair_decoder_ablation_v1/report_compact.json)
   - Branch `kevin/hadamard-pair-decoder-v1`; commit `347acbdfb10a15a2b4081a676b0364d3b83ee8dc`.
   - Git blob `206249f88275c888fc312f35150db395db415796`; SHA-256 `cfd1fc3432ed7ff2eef8e27d1827846cf0c16cac3fc5537487d893c4e9b32790`.
2. [results/pair_decoder_ablation_v1/README.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/pair_decoder_ablation_v1/README.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `85ec61d88ad6ad27c08db59a7dc9d385a27642a9`; SHA-256 `b4a78853bd601e0172875acc0d9deb84df6a7a65869570f1c9248ebda778b566`.

## Exact checked JSON fields

| Selector | Value |
|---|---|
| `["bilinear", "macro", "auprc"]` | `0.023970778057653425` |
| `["hadamard_mlp", "macro", "auprc"]` | `0.02267960882724628` |

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
