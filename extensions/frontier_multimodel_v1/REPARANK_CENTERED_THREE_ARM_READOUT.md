# RePaRank sequence / uncentered-structure / centered-structure audit — 2026-09-30

## Evidence class

Exposed pretrained **development-validation diagnostic**. The public RePaRank checkpoint training accessions are not bound strongly enough to audit overlap with this 217-complex cohort. This is not a clean architecture comparison, blind test, OOD result, or SOTA claim.

## Protocol identity

All three arms use the same 217 validation complexes, the same complete antibody×antigen residue grid, and exactly **9,638,880 candidate pairs / 9,036 positives**. The canonical sorted `(instance, antigen residue index, antibody residue index, label)` digest is identical across all three arms: `ca55ae91723d14061c867ddf8046baac64183e33fbe6180a95102033d692cf82`.

- Sequence checkpoint SHA-256: `b9bbb418aeeb2c96ffa00ccbce36ad169a183c81d57200b503bdb8c1435a4c14`.
- Sequence+structure checkpoint SHA-256: `ce95a698e866246b647197f5b12e11d86c2392d462d2cb2a9ed63c57cf51ae0d`.
- Strict 217-case predicted-chain structure authority SHA-256: `bf47107325885a7f555767cf9f55f8d8074bdafaaae6956961ee2ea6408cc594`.
- Official RePaRank source commit: `6910d950c2e483289973be41387c8913b859766a`.
- Contact labels are the existing project 4.5 Å contact authority, matching the official RePaRank contact-detection definition. No closed test or OOD labels were read.
- Structure input is independently predicted **chain-local** N/CA/C/O geometry. No bound antibody–antigen pose or inter-chain coordinate frame is supplied.

## Why a centered correction was required

The first complete structure evaluation (`1642234`) passed raw chain-local PDB xyz directly to the official structure checkpoint. Inspection of the official RePaRank data path showed that validation/test applies `CenterMesh` independently to every chain: subtract the mean over residue and atom axes before inference. The uncentered run is retained as a real protocol-mismatch negative result rather than overwritten.

The corrected evaluator implements the same mathematical transform on float32 arrays. It is **formula/semantic equivalent, not bitwise equivalent to PyTorch's reduction kernel**: across synthetic arrays spanning 1–1000 residues, the maximum NumPy-vs-PyTorch centered-coordinate difference was `3.0517578125e-05`. Post-centering float32 means remain at ~1e-5 scale. Therefore an exact post-hoc `mean == 0` gate is invalid; this is consistent with standard float32 reduction behavior.

### Preserved failed attempts

- `1642386`: failed after an over-strict post-hoc zero-mean assertion rejected normal float32 residuals. Original stderr is preserved.
- `1642395`: failed before inference because the previous output directory already existed. Original stderr is preserved.
- `1642417`: hash-bound centered resume, **COMPLETED 0:0**. It reused 12 already-centered validated pair artifacts from the first centered attempt and performed 205 fresh forwards. It did not reuse uncentered predictions.

## Three-arm result

| Arm | Macro pair AP | Pooled pair AP | Pooled AUROC | Epitope AP | Paratope AP |
|---|---:|---:|---:|---:|---:|
| Official sequence | **0.0378078** | **0.00873858** | **0.786064** | **0.202039** | **0.557707** |
| Structure, raw/un-centered | 0.0107430 | 0.00289564 | 0.660171 | 0.182829 | 0.409777 |
| Structure, official-style centered | **0.0160136** | 0.00236625 | 0.672153 | **0.193265** | **0.479202** |

Centering recovers part of the structure checkpoint's expected behavior, especially paratope ranking, but does not close the gap to sequence-only on this predicted-chain input authority.

### 68-component paired inference

The 217 complexes belong to 68 frozen split-authority components. Effects are averaged within component first; the interval bootstraps the 68 components as whole units (20,000 deterministic resamples).

| Contrast | Endpoint | Equal-component Δ | 95% component-bootstrap CI |
|---|---|---:|---:|
| centered − uncentered | pair AP | +0.004142 | [−0.003506, +0.011360] |
| centered − sequence | pair AP | **−0.026838** | **[−0.043918, −0.011823]** |
| centered − uncentered | epitope AP | +0.015626 | [−0.005390, +0.037229] |
| centered − sequence | epitope AP | −0.031940 | [−0.072454, +0.006840] |
| centered − uncentered | paratope AP | **+0.065083** | **[+0.022944, +0.107171]** |
| centered − sequence | paratope AP | **−0.051005** | **[−0.098981, −0.002156]** |

Thus the contact-AP improvement from centering is directional but heterogeneous/undecided under the registered component grouping. The centered structure checkpoint remains clearly below sequence-only for pair AP, and also below sequence-only for paratope AP under this analysis.

## Why macro AP rose while raw pooled AP fell

This is not a contradiction. Macro AP ranks pairs **inside each complex** and weights each complex equally. Raw pooled AP mixes all 9.64M pair scores into one global ranking, so it is sensitive to cross-complex score calibration and candidate-grid size; AP is nonlinear and pooled AP is not a weighted average of per-complex APs.

A post-hoc calibration diagnostic supports this explanation:

| Arm | Raw pooled AP | Pooled AP after within-complex percentile rank | Pooled AP after within-complex z-score |
|---|---:|---:|---:|
| uncentered structure | 0.002896 | 0.003237 | 0.003249 |
| centered structure | 0.002366 | **0.004147** | **0.003982** |

The diagnostic normalization reverses the centered-vs-uncentered pooled ordering, consistent with centering improving some within-complex ranking while changing raw score calibration across complexes. These normalized numbers are **post-hoc diagnostics only** and do not replace the registered raw pooled endpoint.

The effect is highly heterogeneous: centered−uncentered pair AP is positive in 62.7% of complexes and 61.8% of components, with complex effects ranging from −0.1732 to +0.2970. The median complex effect is only +0.000111, so the +0.00527 equal-complex mean is influenced by large positive and negative outliers.

## Structure-confidence stratification

For 205 cases with direct chain receipts, an exploratory split by the minimum AF3 ranking score among supplied chains does **not** show a monotonic relationship between confidence and centered−uncentered contact-AP gain. Quartile mean gains are approximately `−0.00119`, `+0.00979`, `+0.00918`, `+0.00280`. AF3 ranking score is not experimental structure accuracy, so this cannot establish that structure quality causes the observed gap.

This distinction matters when comparing with antibody docking benchmarks: interface-recovery analyses restricted to correctly/reliably docked subsets condition on a successful complex pose. Our input authority contains independently predicted chains and no predicted inter-chain pose, so those subset results are not directly comparable.

## Scientific interpretation

The supported conclusion is **not** “structure is useless.” It is narrower:

1. Official preprocessing matters: centering substantially improves paratope AP and directionally improves macro contact AP relative to the uncentered protocol.
2. Under independently predicted chain-local geometry, the official structure checkpoint still underperforms the official sequence checkpoint on this same 217-case pair grid.
3. The structure arm is not being given a docked antibody–antigen complex, so it cannot exploit a correct inter-chain pose; only chain-local backbone geometry is available.
4. Checkpoint training overlap is unknown, and the cohort is development validation. No clean SOTA or generalisation claim is authorised.

The negative result strengthens the project's current thesis: **adding structural coordinates does not automatically solve partner-specific residue correspondence**, especially when structure is chain-local and predicted rather than a correctly docked complex.

## Canonical artifacts

- `evidence/reparank_centered_corrected_v1/evaluation.json`: final centered 217-case evaluation.
- `evidence/reparank_centered_corrected_v1/per_complex.csv`: centered per-complex table.
- `evidence/reparank_centered_corrected_v1/THREE_ARM_AUDIT.json`: exact grid hashes, paired CIs, heterogeneity, calibration diagnostic and confidence strata.
- `evidence/reparank_centered_corrected_v1/centered-1642386.err`: preserved float32 residual-gate failure.
- `evidence/reparank_centered_corrected_v1/centered-1642395.err`: preserved output-directory collision failure.
- `REPARANK_STRUCTURE_CENTERING_CORRECTION.md`: protocol-correction rationale.
