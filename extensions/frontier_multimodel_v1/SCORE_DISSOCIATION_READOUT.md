# Full unique-sequence score-dissociation readout

Status: **complete exploratory development analysis**. No replacement score is promoted and no binding-specificity claim is made.

## Executed scope

Successful attempt plan SHA-256: `d14805b3e0fe4f774753fe754df15c3084c147b0ee8ca60ff33a50aa6431f8b7`. The campaign contains nine GPU tasks: three fixed ESM-2 + ESM-IF1 checkpoints × aligned, chain-shift-8 and within-chain global-shuffle conditions.

Each task scores the complete deduplicated Cartesian product of **122 antibody sequences × 119 antigen sequences = 14,518 pairs**. Total pair evaluations: **130,662**.

- Canary `1639968_0`: COMPLETED, exit 0:0.
- Formal array `1639970_1`–`1639970_8`: 8/8 COMPLETED, all exit 0:0.
- Analysis `1639971`: COMPLETED, exit 0:0.
- Attempts `1639957_0` and `1639965_0` failed before inference and are preserved separately; neither produced a scientific score.

There are 150 known sequence pairs inside the 122×119 grid. Every other pair is an **unlabelled distractor**, not an experimentally verified non-binder. Exact duplicate sequences are one candidate, and multi-positive anchors are retained.

## The earlier MRR increase does not replicate as a robust full-grid effect

| Retrieval direction | Identity MRR | Shift-8 MRR | Global-shuffle MRR | Global minus identity, pointwise 95% CI |
|---|---:|---:|---:|---|
| Rank antigens for each antibody | 0.061197 | 0.062613 | 0.063283 | +0.002086 [-0.025596, +0.031408] |
| Rank antibodies for each antigen | 0.070451 | 0.068858 | 0.075521 | +0.005070 [-0.019347, +0.027706] |

Recall@1/5 and known-positive AP differences also have intervals spanning zero. Therefore the previous 16-candidate observation is best treated as panel-composition noise or instability, not evidence that corrupted correspondence improves partner retrieval. Contact AP still falls strongly in the existing matched-complex experiment; full-grid ranking is approximately unchanged and remains weak.

## What the current score measures

Across all 14,518 pairs, the checkpoint score has Spearman correlation **0.9743 on average** with the standard deviation of the double-centred interaction matrix; the nine seed/condition values range from `0.9589` to `0.9856`. The current score is therefore close to a monotone interaction-spread statistic over this heterogeneous global grid.

The score also has strong asymmetric within-anchor associations:

- when ranking antigens, mean per-anchor correlation with antigen length is about +0.35 to +0.47 and with the chain-local surface proxy about −0.45 to −0.50;
- when ranking antibodies, correlation with antibody length is about −0.35 to −0.48 and with the paratope-head mean about +0.44 to +0.58.

These are descriptive validation correlations, not proof of causal bias. Packing is available for 104/122 unique antibodies and 100/119 unique antigens; missing proxies remain missing and candidates are not dropped.

## Contact localisation and ranking-score changes are not coupled

Under global shuffle, mean matched-complex contact AP changes by −0.01860. The current-score change has mean per-seed Spearman 0.125 with contact-AP change and crossed seed/component interval [−0.142, 0.362]. Shift-8 is similarly uncertain. No inspected score provides multiplicity-corrected evidence of tracking contact loss.

An uncentred log-mean-exp score decreases under corruption in both directions, but it was inspected on validation and cannot be promoted as a replacement. Some other decompositions show isolated pointwise intervals that exclude zero; they arise from a fifteen-score exploratory family and are not corrected or independently confirmed.

## Consequence for the programme

The earlier “contact down, MRR up” finding is resolved as **not robust on the complete deduplicated grid**. The more durable issue is that the current score is largely interaction dispersion and has sizeable candidate-length, surface-proxy and node-prior associations. Downstream SaProt/ProSST work may proceed, but partner-ranking claims must keep the current score diagnostic; any replacement score requires a newly frozen evaluation on an admitted future cohort.

## Evidence boundary

`ANALYSIS.json` contains 300 paired retrieval contrasts, 30 contact-link analyses and source/artifact hashes. Per-task receipts and anchor metrics are committed. Full 14,518-row pair grids remain on ParaCloud at receipt-bound paths and are not copied into Git. No closed test, new training, experimentally confirmed non-binder panel or model promotion is part of this result.
