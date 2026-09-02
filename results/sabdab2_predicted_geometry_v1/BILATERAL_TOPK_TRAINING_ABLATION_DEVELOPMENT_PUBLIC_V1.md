# Bilateral Top-K from-scratch training ablation

Eight development-only runs compare `K in {1,2,4,8}` for epitope and
paratope pooling under the same sequence-predicted-geometry,
no-privileged-distance protocol. All arms use seed 17. The epitope arms retain
the native paratope readout; the paratope arms hold epitope pooling at TopK2.

## Epitope pooling

| K | Mean-per-complex AP | Pooled AP | AUROC | F1 | MCC |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.228217 | 0.138299 | 0.645819 | 0.217486 | 0.113597 |
| 2 | **0.237529** | **0.144046** | **0.649511** | **0.221773** | **0.125680** |
| 4 | 0.234279 | 0.139897 | 0.645966 | 0.220547 | 0.119503 |
| 8 | 0.237326 | 0.134207 | 0.622493 | 0.204677 | 0.104092 |

K2 and K8 are nearly tied on mean-per-complex AP, but K2 is stronger on every
other reported epitope metric. This supports retaining K2 in the current model;
it does not establish a universal optimal K.

## Paratope pooling

| Readout | Mean-per-complex AP | Pooled AP | AUROC | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Native reference | 0.679731 | 0.629400 | 0.951133 | 0.643753 | 0.607037 |
| TopK1 | **0.702509** | 0.629018 | 0.953033 | **0.656401** | **0.626935** |
| TopK2 | 0.701388 | **0.657073** | **0.954372** | 0.654526 | 0.618771 |
| TopK4 | 0.673524 | 0.585477 | 0.945263 | 0.622226 | 0.583647 |
| TopK8 | 0.680196 | 0.618129 | 0.950771 | 0.648468 | 0.611886 |

TopK1 and TopK2 improve mean-per-complex AP over the shared native reference
by `+0.022778` and `+0.021657`, respectively. TopK2 has the stronger pooled AP
and AUROC, whereas TopK1 has the stronger F1 and MCC. The result is promising
but remains a single-seed screen on checkpoint-selection validation data.

The fixed-checkpoint readout sensitivity remains separately preserved in
`BILATERAL_TOPK_READOUT_DEVELOPMENT_PUBLIC_V1.*`; its negative paratope result
is not replaced by this from-scratch training result. No old SAbDab2 test or
temporal holdout was loaded.

Internal frozen analysis SHA-256:
`6a1abbca777db679585451ad971039c82af7fac4a93b01146aa584a9e5eba3eb`.
