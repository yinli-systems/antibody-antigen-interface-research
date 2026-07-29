# SAbDab2 structure-aware self-attention ablation

This directory records the lightweight, development-only result of the real
SAbDab2 structure adapter. Raw structures, residue embeddings, labels, full
record manifests, logs, and model weights are intentionally excluded.

## Data boundary

- The authenticated Needleman--Wunsch development authority contains 3,393
  train and 52 validation instances.
- Intersecting it with the newer SAbDab2 release's native training role leaves
  2,986 train and 43 validation candidates. The 416 native-release test cases
  are excluded before reading their mmCIF members.
- Exact structure-to-canonical-axis validation accepts 2,985 train and all 43
  validation structures. One train complex is rejected because its new-release
  VH sequence differs from the canonical embedding/label axis.
- The preregistered `antigen_length <= 384` runtime cap leaves 2,676 train and
  36 validation complexes for this run.
- All 3,028 accepted artifacts have complete valid `N/CA/C` frames. The explicit
  missing-frame path is tested, but the accepted real cohort has zero positive
  missing-frame examples.

## Fixed-final-epoch result

All modes use frozen ESM2 residue inputs, separate antibody and antigen
self-attention parameters, the same seed and initialization, 271,670 parameters,
three epochs, and the same same-chain `k=32`, 16-A spatial graph. `none` is a
spatial-topology-only control, not a dense sequence-only Transformer.

| Edge features | Paratope AUPRC | Epitope AUPRC | Macro AUPRC | Macro AUROC |
|---|---:|---:|---:|---:|
| topology only | 0.627464 | 0.253331 | 0.440398 | 0.829702 |
| distance RBF | 0.630673 | 0.256434 | 0.443554 | 0.831124 |
| distance + local direction | 0.655229 | 0.255474 | 0.455351 | 0.829258 |
| full relative pose | 0.652331 | 0.257624 | 0.454978 | 0.830546 |

Metrics pool 5,149 antibody residues and 9,572 antigen residues from the 36
validation complexes. Epoch 3 is reported for every mode without validation-best
checkpoint selection.

The numerical pattern is a development signal only. This is one seed, a short
budget, a small validation intersection, and has no confidence interval or
held-out confirmation. It does not establish a benchmark improvement.

## Paired complex-level checkpoint audit

After the original pooled result was frozen, protocol
`SABDAB2_GEOMETRY_CONFIRMATORY_PROTOCOL_V1.md` preregistered a per-complex,
endpoint-component analysis. Immutable final checkpoints were replayed without
checkpoint selection. The replay reproduced all 20 historical pooled
AUPRC/AUROC values within the locked absolute tolerance `0.01`; the largest
device-related difference was `0.002471`.

Per-complex rather than pooled-residue macro-AUPRC is:

| Edge features | Paratope | Epitope | Joint macro |
|---|---:|---:|---:|
| topology only | 0.647477 | 0.302444 | 0.474961 |
| distance RBF | 0.648636 | 0.308101 | 0.478369 |
| distance + local direction | **0.672316** | **0.306036** | **0.489176** |
| full relative pose | 0.666767 | 0.302240 | 0.484504 |

The 36 validation complexes form 16 exact antibody-antigen
endpoint-connected components. With 10,000 paired component bootstrap
replicates and 10,000 component sign-flip permutations:

| Preregistered comparison | Mean AUPRC difference | Paired 95% CI | One-sided p |
|---|---:|---:|---:|
| joint distance+direction - topology | +0.014215 | [+0.001939, +0.022948] | 0.027397 |
| paratope distance+direction - topology | +0.024839 | [+0.018324, +0.031041] | 0.000100 |
| epitope distance+direction - topology | +0.003592 | [-0.015970, +0.017186] | 0.357464 |
| full pose - distance+direction, joint | -0.004672 | [-0.011390, +0.005731] | descriptive |

The joint effect is positive in 34 of 36 complexes. Nevertheless, the
preregistered three-comparison Holm-adjusted one-sided p-value is `0.082192`,
and the multi-seed requirement is not complete. The strict superiority gate
therefore **does not pass**. The defensible conclusion is narrower: local
direction has a reproducible development signal, driven primarily by the
paratope side; the current data do not establish a corresponding epitope gain,
and full relative pose does not beat the simpler direction representation.

This first per-complex analysis used the backed-up RTX 5090 final checkpoints
and a local CPU inference replay. An independent implementation reproduced 441
metrics, endpoint-component, bootstrap, permutation, and claim-boundary checks.

## AB03/A100 sealed reproduction

The complete AB03 archive was pulled back and verified independently. It binds
the exact two-A100 run, parent and additive runtime images, all final
checkpoints, 8,727 case-sensitive data-authority files, the original-run
manifest, an external seal, and three fixed-budget seed results. All 48
scientific replay gates pass. The archive SHA-256 is
`b6365093db7305b2d054c8a801e9aed002423156aae328075faf73466ab9c75b`.

The exact A100 checkpoints were then replayed for per-complex inference in FP32
on local CPU. The direction-minus-topology joint macro-AUPRC difference is
`+0.009876`, with endpoint-component bootstrap 95% CI
`[+0.001836, +0.016695]` and unadjusted one-sided sign-flip
`p=0.027497`. Its three-comparison Holm-adjusted p-value is `0.082492`;
the strict gate therefore remains failed. The independent implementation again
passed 441 checks.

The three archived A100 seeds give pooled direction-minus-topology joint AUPRC
deltas of `+0.017724`, `+0.008886`, and `+0.000886`. This is useful stability
evidence, but those seeds are not the five seeds frozen later in the
confirmatory protocol and cannot satisfy that requirement retroactively.

## Files

- `ablation.json`: exact configuration, cohort digests, runtime, and all epoch
  metrics. SHA-256:
  `4c5114e96dc4514c20695332a877afd9299c70a188d859aa5c713b7be84933cc`.
- `run_receipt.json`: result digest and final cohort counts.
- `structure_cache_receipt.json`: authenticated cache manifest digest and
  pre-length-cap record counts.
- `paired_statistics_summary.json`: preregistered complex-level effects,
  uncertainty, independent-verification counts, artifact hashes, and strict
  gate result. Raw per-residue prediction payloads are intentionally excluded
  from Git and retained in the local audit archive.
- `ab03_a100_replay_verification.json`: independent archive, manifest, runtime,
  gate, and three-seed recomputation receipt.
- `a100_checkpoint_paired_statistics_summary.json`: lightweight per-complex
  statistics and hashes from the exact A100 final checkpoints.
- `FINAL_GEOMETRY_RESEARCH_REPORT.md`: literature synthesis, implementation
  rationale, complete evidence interpretation, and the next partner-specific
  experiment.

## Claim boundary

No frozen test was read. This is not a real PairFormer checkpoint result, an
inter-chain contact/cross-attention result, a fused-kernel benchmark, or evidence
for biological, clinical, external-generalization, state-of-the-art, or
benchmark-improvement claims.
