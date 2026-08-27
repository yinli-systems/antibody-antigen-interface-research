# Result index

This directory contains lightweight, reviewable evidence. It intentionally omits
raw datasets, model weights, full embeddings, and complete prediction tensors.

## Primary summaries

| Workstream | Recommended entry point | Status boundary |
|---|---|---|
| 2026-08-27 audited snapshot | [`LATEST_RESULTS_20260827.md`](LATEST_RESULTS_20260827.md) | Compact task-routing, pooling-readout, ESM-C, bound/unbound, and fail-closed temporal evidence; development and measurement boundaries are stated per result |
| Cluster activity | [`cluster/CLUSTER_ACTIVITY_SNAPSHOT_20260821.md`](cluster/CLUSTER_ACTIVITY_SNAPSHOT_20260821.md) | Point-in-time scheduler snapshot only; MIPE, the negative-sample ablation, and RTX 4090 pretraining were running at the recorded timestamp |
| Partner-negative ablation | [`negative_sample_ablation/README.md`](negative_sample_ablation/README.md) | Raw-Brier v1 failed closed; corrected paired five-seed AsEP/SAbDab2 v2 is running on 20 RTX 4090s, with no final comparison or test access yet |
| RTX 4090 pretraining | [`pretraining/RTX4090_PRODUCTION_CAMPAIGN_LAUNCH_20260821.md`](pretraining/RTX4090_PRODUCTION_CAMPAIGN_LAUNCH_20260821.md) | Exact recovery passed and the first 30B-token campaign segment is running; no terminal validation or downstream claim |
| ElasticABF recovery | [`pretraining/elasticabf_real_oas_4to2to4_v4_1457530/README.md`](pretraining/elasticabf_real_oas_4to2to4_v4_1457530/README.md) | Real-OAS ABF-35M 4-to-2-to-4 correctness gate passed; no 650M performance or scientific claim |
| SPPIDER-seq | [`sppider_seq/FINAL_RESEARCH_REPORT.md`](sppider_seq/FINAL_RESEARCH_REPORT.md) | Locked marginal paratope/epitope evaluation; not native pair prediction and not a pristine blind test |
| MIPE | [`mipe/MIPE_OFFICIAL_REPLAY_20260821.md`](mipe/MIPE_OFFICIAL_REPLAY_20260821.md) | Five-fold released-code replay, OOF calibration, and pair audit; external split had pre-scoring schema access and the Appendix-correct replay is still running |
| Intra-chain geometry | [`sabdab2_structure_self_attention/FINAL_GEOMETRY_RESEARCH_REPORT.md`](sabdab2_structure_self_attention/FINAL_GEOMETRY_RESEARCH_REPORT.md) | Development-only A100/RTX 5090 evidence; five-seed superiority gate did not pass |
| Partner retrieval | [`partner_retrieval/interaction_first/sequence_pair_seed42/validation/evaluation.json`](partner_retrieval/interaction_first/sequence_pair_seed42/validation/evaluation.json) | Development evidence; the locked dual encoder did not pass the formal specificity gate |
| PairScale | [`pairscale/`](pairscale/) | Development and control evidence only; consult the matching protocol before interpreting a metric |
| EpiFormer adapter | [`epiformer/`](epiformer/) | Adapter and verification evidence; comparison requires identical data and task semantics |

The root [`README.md`](../README.md) is the concise cross-workstream status report.
Protocol authority lives under [`protocols/`](../protocols/README.md).

## Interpretation rules

- Prefer a final report or verified receipt over a conveniently named summary.
- Verify the dataset role, contact definition, aggregation unit, and checkpoint
  selection policy before comparing metrics.
- Treat `research/`, `development`, `smoke`, and `screen` paths as non-confirmatory
  unless a governing protocol explicitly says otherwise.
- Do not describe the supplied external split as pristine blind test evidence.
- Do not claim partner specificity from marginal epitope/paratope scores alone.

## Mirrored files

Some files also appear under `protocols/`. These copies are retained because
existing manifests and validators may bind exact bytes and paths. They are not
unreviewed clutter and must not be removed by a generic cleanup.
