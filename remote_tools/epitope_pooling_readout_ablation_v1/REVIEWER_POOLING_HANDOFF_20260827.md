# Pooling and statistical follow-up (2026-08-27)

## Scope

This note separates two experiments:

1. A fixed-checkpoint diagnostic: one frozen seed-17 checkpoint and one unchanged
   549-complex development-validation pass; only the deterministic mapping from
   pair logits to epitope scores changes.
2. A strict controlled training screen: seven independently trained seed-17 arms
   with the same data, objective, optimizer, checkpoint rule, and training budget.

Neither experiment accesses the historical SAbDab2 test or the temporal holdout.
The 549-complex validation split was used for checkpoint selection, so these are
development diagnostics, not independent test results.

## 1. Why K=2?

K=2 was a compact candidate chosen before the current ablation, not a value proven
optimal by the official EpiFormer implementation. The earlier biological
explanation that EpiFormer selected K=2 because two or three paratope residues
usually support an epitope residue should not be used as evidence.

The fixed-checkpoint diagnostic now shows a broad plateau:

| Readout | Mean-per-complex AP | Delta vs K=2 | Paired complex-bootstrap 95% interval |
|---|---:|---:|---:|
| K=1 (max) | 0.235524 | -0.000347 | [-0.000732, 0.000017] |
| K=2 | 0.235871 | 0 | [0, 0] |
| K=4 | 0.235907 | +0.000036 | [-0.000294, 0.000353] |
| K=8 | 0.235891 | +0.000020 | [-0.000451, 0.000483] |

Thus K=2 is a stable low-complexity point on the K=1/2/4/8 plateau, but it is not
uniquely best. The strict seven-arm retraining screen is still running and is
required before reporting training-adaptive behavior.

## 2. Alternative pooling rules

On the same frozen pair logits:

| Readout | Mean-per-complex AP | Delta vs K=2 | Paired complex-bootstrap 95% interval |
|---|---:|---:|---:|
| max (=K=1) | 0.235524 | -0.000347 | [-0.000732, 0.000017] |
| top-K mean, K=2 | 0.235871 | 0 | [0, 0] |
| noisy-OR | 0.179440 | -0.056431 | [-0.065256, -0.048058] |
| log-mean-exp | 0.235520 | -0.000351 | [-0.001567, 0.000786] |

Noisy-OR is clearly worse because combining hundreds of candidate probabilities
causes diffuse low-confidence edges to accumulate. Max and log-mean-exp are close
to K=2 at this checkpoint. Learned-softmax pooling has an extra trainable
temperature and cannot be evaluated fairly by swapping only the readout of a K=2
checkpoint; its separately trained arm remains in the strict screen.

Frozen analysis:

`/ssd/scxi253/model_deployment/results/development/sabdab2_epitope_pooling_readout_v1/FIXED_CHECKPOINT_EPITOPE_READOUT_ANALYSIS_V1.json`

SHA-256: `8320b60686c4132a4ab0f232e339de02c17f58f2cefa613ca8a13c2f705dbaa6`

## 3. Why cluster weighting changes approximately 0.009 to 0.030

These are different estimands:

- complex-weighted, seed-averaged mean: +0.009559;
- median of the three seed-specific complex means: +0.008877;
- equal-antigen-cluster-weighted, seed-averaged mean: +0.029518.

The fresh156 set has 156 complexes in 49 antigen clusters. Twenty-three clusters
are singletons. Mean gains by cluster-size stratum are:

- size 1: +0.043697;
- size 2--3: +0.024897;
- size >=4: -0.004529.

Equal-cluster weighting gives each antigen cluster one vote, so the many small,
positive-gain clusters receive more weight and the larger weak/negative clusters
receive less weight than in the complex-weighted estimator. The reported
cluster-bootstrap interval [0.012630, 0.047429] is conditional on the three fixed
trained seeds; it does not include uncertainty from retraining new seeds.

Frozen explanation:

`/ssd/scxi253/model_deployment/results/development/sabdab2_edge_topk2_task_routed_holdout_v1/FRESH156_WEIGHTING_EXPLANATION_V1.json`

SHA-256: `593e6f05650c39b2de60b891eec1d9a43ec7658be23bcbd03120573d0ae51204`

## Baseline evidence for the collaborator

The recovered EpiFormer, MIPE, and SPPIDER-seq loss-curve package and the official
DiscoTope-3.0 source binding are complete. DiscoTope-3.0 is a fitted 100-model
XGBoost ensemble, so a neural per-epoch loss curve is not applicable and must not
be fabricated. Archived method evaluations use different splits and aggregation
rules and therefore must be reported per method rather than merged into a claimed
same-protocol ranking.

Full handoff:

`/ssd/scxi253/yangwuyu/baseline-evaluation/docs/2026-08-27-baseline-loss-evidence-handoff.md`
