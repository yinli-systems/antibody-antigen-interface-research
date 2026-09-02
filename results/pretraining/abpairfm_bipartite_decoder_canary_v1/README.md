# Bipartite decoder canary v1

This directory records the frozen, matched canary comparing the existing
symmetric low-rank contact decoder with an explicit bipartite edge-refinement
decoder. Both arms used the same authenticated 512-system PINDER-Train subset,
the same 16-system PDB-disjoint holdout, initialization seed, optimizer,
objective, and 64-update budget. No test or temporal cohort was accessed.

## Result

| Metric after 64 updates | Low-rank control | Edge refinement |
|---|---:|---:|
| Contact AP | 0.002370 | 0.002521 |
| Contact AUROC | 0.641883 | 0.645109 |
| Contact MRR | 0.005332 | 0.006014 |
| Contact Recall@1 | 0 | 0 |
| Matched-minus-swapped mean logit | 0.060945 | 0.062673 |

The pre-registered decision uses paired difference-in-differences rather than
the raw endpoint. Edge refinement minus control was
`-0.0000424 [-0.0002954, 0.0001756]` for contact AP,
`-0.004319 [-0.022213, 0.013757]` for contact AUROC, and
`-0.001585 [-0.005851, 0.001567]` for contact MRR. Partner-swap separation
improved by `+0.008826 [0.000865, 0.016533]`, but this did not translate into
better pair ranking.

Decision: `FAIL_BIPARTITE_EDGE_REFINEMENT_RANKING_GATE`; retain the low-rank
decoder and archive this candidate without tuning it from the canary.

## Integrity

- Analysis SHA-256: `e370f0a35163830420c6f570f17c7efea45156a0dcda0baceaf9e1b41806d251`
- Low-rank result SHA-256: `407fd73f3d707adb57e64cd4a1076d23a09eef23da1a63f699c0c07fdc12b5a2`
- Edge-refinement result SHA-256: `3768cadb81c41e57fddbaf14c7f47b2513079a5679f98f116cf2f7ed52baba39`
- Frozen source SHA-256: `78736d49f17eb622bcb2f48cc7f1853563d111b42f5969f5d21c0ebbe2700b51`
- `test_access=false`

The machine-readable decision and per-system paired analysis are in
`ANALYSIS_V1.json`; each arm's execution record is in its corresponding
`RESULT.json` and `PROGRESS.json`.
