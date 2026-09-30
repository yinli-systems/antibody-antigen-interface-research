# Structural-PLM layer sweep final readout — 2026-09-30

## Final status

The formal structural-PLM sweep is complete: **36/36 training tasks completed with exit 0 and zero training failures**. The original analyzer job `1640514` failed after training because it expected the historical schema key `partner_retrieval.mrr`. The failure is preserved verbatim. The pre-registered schema-repair job `1641627` then completed `0:0` and produced the authoritative corrected analysis at:

`/ssd/scxi253/frontier_multimodel_v1/execution_recovery_v1/layer_analysis_v1/output/ANALYSIS.json`

The repair changed only schema adaptation. Its authority explicitly binds the original training attempt and plan and states that it uses the original AP/bootstrap/selection algorithms. No training was rerun.

## Frozen authority

- Formal training array: `1640503_0..35`
- Corrected analyzer: `1641627`
- Original failed analyzer: `1640514`
- Original plan SHA-256: `24d6c6cfb2cee44b9cc289ac8721b3ad183321db11a059c14b29c878dd7cea12`
- Original training source commit: `3eb4293464552199c66eff9b6add14d7ed75ae14`
- Schema-repair source commit: `b72e6bf1736e763135f2104303383b3e4ef0336a`
- Validation complexes: **217**
- Frozen independent split components: **68**
- Training seeds per arm: **1, 2, 3**
- Arms: **12** = five SaProt layers + one matched random-width control + five ProSST layers + one matched random-width control
- Primary endpoint: validation macro mean-per-complex pair AP from `best.pt`
- Test labels read: **false**
- OOD labels read: **false**
- Blind claim authorized: **false**
- SOTA claim authorized: **false**

## Development layer-selection values

The registered within-family development selector chose:

- **ProSST layer 12**, mean pair macro AP **0.0221756**.
- **SaProt layer 24**, mean pair macro AP **0.0244519**, because layer 33's raw high **0.0249890** lies within the predeclared 0.001 tie window.

These are development layer selections only. They are **not the representation-promotion decision**.

## Matched-random decision

The promotion question is whether a real structural-PLM layer reliably exceeds its contemporaneous width-matched random-feature control under the registered 68-component × three-seed paired analysis.

### ProSST

Width-matched random control mean pair macro AP: **0.0223573**.

| Layer | Real mean AP | Equal-component/equal-seed Δ vs random | 95% crossed seed×component CI |
|---|---:|---:|---:|
| 0 | 0.0194798 | −0.0016950 | [−0.0048718, +0.0012809] |
| 3 | 0.0210678 | −0.0019367 | [−0.0054927, +0.0017540] |
| 6 | 0.0203417 | −0.0021272 | [−0.0060799, +0.0012836] |
| 9 | 0.0205999 | −0.0005698 | [−0.0054493, +0.0045586] |
| **12** | **0.0221756** | **−0.0001479** | **[−0.0037666, +0.0029894]** |

No ProSST layer exceeds the matched random control under the registered interval. The selected layer 12 is effectively tied with, and pointwise slightly below, the control.

### SaProt

Width-matched random control mean pair macro AP: **0.0220751**.

| Layer | Real mean AP | Equal-component/equal-seed Δ vs random | 95% crossed seed×component CI |
|---|---:|---:|---:|
| 0 | 0.0205818 | −0.0017260 | [−0.0049084, +0.0016569] |
| 8 | 0.0210398 | −0.0005711 | [−0.0054398, +0.0042164] |
| 16 | 0.0230973 | +0.0024893 | [−0.0030949, +0.0084628] |
| **24** | **0.0244519** | **+0.0027893** | **[−0.0018430, +0.0084833]** |
| 33 | 0.0249890 | +0.0017850 | [−0.0021941, +0.0058604] |

SaProt layers 16/24/33 have positive point estimates, but **every crossed interval includes zero**. Layer 24 is positive in 57.4% of components; layer 33 in 58.8%. This is suggestive development signal, not evidence sufficient for promotion.

## Decision

**NO-PROMOTION_STRUCTURAL_PLM_FOR_STRICT_PARTNER_RETRIEVAL**

Under the prospectively stated rule, no SaProt or ProSST layer is promoted into the strict partner-retrieval model family. The experiment does not justify further layer/model chasing on this development cohort.

The result does **not** establish that structural language models are useless. It establishes the narrower claim that, in this exact downstream architecture and 217-complex development protocol, the tested real structural-PLM features do not reliably outperform contemporaneous width-matched random features on pair-contact AP.

Partner-ranking metrics remain secondary because this project has already observed score/contact dissociation. They are not used to override the primary matched-random contact gate.

## Failure provenance

`1640514` failed only in aggregation:

`KeyError: 'mrr'`

at the historical accessor `evaluation["partner_retrieval"]["mrr"]`. All 36 training receipts were already complete and remain unchanged. `1641627` was scheduled with dependency `afterok:1640503,afterany:1640514`, completed in 1m51s, and generated the corrected 36-task/12-arm analysis without submitting training.

## Scientific consequence

The structural-PLM branch is now closed as a development screen. The main research budget should move to the already-started **strict dual-axis partner-retrieval** program. SaProt/ProSST can remain as documented negative/suggestive evidence, but neither is an authorized feature arm for H0/D0/L1/R1/R2 unless a future genuinely independent protocol preregisters a new test.

## Evidence files

- `evidence/structural_plm_final_20260930/ANALYSIS.json` — authoritative corrected analysis.
- `evidence/structural_plm_final_20260930/ANALYSIS.csv` — arm-level summary.
- `evidence/structural_plm_final_20260930/AUTHORITY.json` — schema-repair execution authority.
- `evidence/structural_plm_final_20260930/1640514.err` — preserved original aggregation failure.
- `evidence/structural_plm_final_20260930/1641627.out` — corrected analyzer terminal summary.

