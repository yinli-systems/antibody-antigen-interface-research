# Bilateral Top-K readout audit

This audit evaluates `K ∈ {1,2,4,8}` on the final sequence-predicted-geometry,
no-privileged-distance development checkpoints. It uses the same 549 validation
complexes and fixed seeds 17, 29, and 43. It does not access the old SAbDab2
test set or the temporal holdout.

## Main finding

Top-K should not be applied symmetrically merely for architectural appearance.
For paratope prediction, every node-plus-Top-K arm is worse than the native
node-plus-log-mean-exp readout on every seed. The closest arm, node plus TopK8,
still lowers median mean-per-complex AP from `0.696181` to `0.690259`; pooled AP,
AUROC, F1, and MCC also decline.

For epitope prediction, K1/K2/K4/K8 are nearly tied. K8 has the largest median
mean-per-complex AP (`0.232975`), but its median paired improvement over K2 is
only `+0.000292`. This is too small to support an optimal-K or replacement claim.

## Epitope fixed-checkpoint sensitivity

| Readout | Mean-per-complex AP | Pooled AP | AUROC | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| TopK1 | 0.232266 | 0.144446 | 0.657244 | 0.224778 | 0.128373 |
| TopK2 | 0.232682 | 0.144046 | 0.657212 | 0.225006 | 0.130552 |
| TopK4 | 0.232712 | 0.144597 | 0.657270 | 0.225092 | **0.131487** |
| TopK8 | **0.232975** | **0.144897** | **0.657491** | **0.225314** | 0.130291 |

## Paratope fixed-checkpoint sensitivity

| Readout | Mean-per-complex AP | Pooled AP | AUROC | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Native node + log-mean-exp | **0.696181** | **0.663613** | **0.953958** | **0.657242** | **0.622239** |
| Node + TopK1 | 0.684283 | 0.630044 | 0.948812 | 0.633178 | 0.596267 |
| Node + TopK2 | 0.686401 | 0.632144 | 0.949443 | 0.635711 | 0.599023 |
| Node + TopK4 | 0.688533 | 0.636178 | 0.950098 | 0.637431 | 0.600518 |
| Node + TopK8 | 0.690259 | 0.638390 | 0.950753 | 0.638691 | 0.600834 |

These are fixed-checkpoint readout sensitivities, not from-scratch K training
ablations. A pre-frozen eight-arm seed-17 retraining screen was submitted as
Slurm array `1538425`; its analyzer is `1538426`. Until that screen completes,
the paper should present the tables above as post-hoc readout sensitivity and
retain the native paratope route.

Raw analysis SHA-256:
`fb309de99d22f5d01b9854b0623bb727c26878b9df21a9f18e7ac6b0775cea73`.
