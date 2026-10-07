# Antibody–antigen results atlas

**A result is a claim plus its cohort, estimator and source — not just a score.**

[All 30 branches](BRANCHES.md) · [Every evidence location](FAMILIES.md) · [Cohorts & comparability](COHORTS.md) · [Audit findings](REVIEW_FINDINGS.md) · [Presentation guide](PRESENTATION_GUIDE.md)

Snapshot: **2026-10-07T11:36:54.800931+00:00**. Original main: `23480c507aeab4a5bb212631ad2b8eb09cb8867f`. **30 branch HEADs · 4,913 distinct path/blob versions · 103,552 tracked entries including repetition.** This is not every historical commit or an external GPU filesystem.

## Start here

The strongest artifact-backed story is controlled correspondence / representation evidence, a bounded three-output interface predictor, and explicit negative controls. It is not a single cross-protocol model leaderboard. AbCite is a separate multimodal evidence-retrieval application.

**Completion corrections:** final structural-PLM receipts record36/36 and no promotion; centered RePaRank structure covers217/217. Earlier12/36 / pending records are historical snapshots, not today's live status. See [PLMs](cards/structural-plm.md), [RePaRank](cards/reparank-centered.md), and [findings](REVIEW_FINDINGS.md).

**Newer does not mean better verified.** The October4 reviewer manuscript reports additional campaigns and an improved ensemble. These claims are shown as `report-only` where matching terminal artifacts were not identified in the inspected branch HEADs. This does not mean the runs did not happen or their files do not exist off-Git.

## Evidence levels

| Level | Meaning |
|---|---|
| `metric-bound` | Committed numeric fields checked; no model rerun |
| `readout-bound` | Committed research readout inspected; not independently recomputed |
| `report-only` | Newer narrative claim; matching terminal campaign artifacts not identified in the inspected HEAD trees |
| `protocol-only` | Protocol or implementation, not completed performance |
| `engineering` | Execution/provenance evidence, not model accuracy |
| `historical` | Earlier snapshot or superseded interpretation, retained unchanged |

## Result-by-result navigation

| Result | Level | Cohort | Recorded answer |
|---|---|---|---|
| [FrameContact: three outputs on the corrected historical test](cards/framecontact-historical.md) | metric-bound | Checkpoint-aligned historical 549 test, already exposed | Corrected test 549 routed medians: contact AP 0.02667, epitope AP 0.22031, paratope AP 0.69859. The old 455 cache was not held out for this checkpoint. |
| [Aligned features versus shuffled and width controls](cards/representation-transfer.md) | metric-bound | SAbDab2 development 217 | Aligned ESM-IF1 minus ESM-2: +0.005111 AP; shuffled minus ESM-2: −0.003093. The random-width control is undecided, not equivalent. |
| [Same-AA shuffling isolates correspondence damage](cards/same-aa-corruption.md) | metric-bound | 217 development complexes /68 components | ESM-IF1 identity AP 0.027995 becomes 0.011100 under same-AA shuffle; component-weighted paired effect about −0.017151. |
| [Registered holdout: one encoder confirms](cards/holdout-confirmation.md) | metric-bound | Registered 217-complex holdout, distinct from development 217 | ProteinMPNN aligned−shuffled: +0.003594 AP,9/10 favorable, interval above zero. ESM-IF1: +0.002260,6/10, interval includes zero. Both outcomes must remain visible. |
| [Learned rescue: synthetic gain with clean-input cost](cards/synthetic-rescue.md) | metric-bound | 217 development complexes /68 components; 3 synthetic conditions | Same-AA rescued−corrupted AP: +0.0157209,95% CI [0.0112560,0.0211527]. Clean inputs incur about −0.001430 component-weighted AP cost. |
| [Gating reduces clean damage but does not establish safety](cards/gated-rescue.md) | metric-bound | 217 development complexes /68 components; train-only threshold calibration | All registered synthetic corruptions trigger. Clean damage becomes −0.0004598 AP, crossed CI [−0.0011427,+0.0000198]; no clean-safety guarantee. |
| [Cheap selectors also detect synthetic corruption](cards/cheap-gate-control.md) | readout-bound | Same 217 development complexes; training-only calibration | Moved-fraction and related inexpensive statistics also detect same-AA/global permutations at or near 100%. The synthetic construction moves about 91% of same-AA rows and nearly all globally shuffled rows. |
| [Ten-seed pair head versus its own marginal baseline](cards/baseline-margin.md) | metric-bound | 217 validation complexes /68 clusters; documented split drift | Paired macro AP margin +0.00311345, CI [0.00196869,0.00429043], positive 10/10. Ratio of means≈16.1%; mean seed ratios≈16.7%; neither is the single-seed 32.7%. |
| [Hadamard decoder: no demonstrated win](cards/hadamard-decoder.md) | metric-bound | 217 validation complexes /9,638,880 pairs | Bilinear macro pair AP 0.0239708 versus Hadamard 0.0226796. No superiority or equivalence of decoder classes follows from one seed. |
| [Ranking-term removal improves two measured endpoints](cards/objective-factorial.md) | readout-bound | AsEP epitope_group validation | Node+edge pair AP 0.09055 versus full-objective 0.06917; paired gains about+0.0214 pair AP and+0.0369 node MCC, favorable 10/10. |
| [Gradient alignment changes during training](cards/gradient-dynamics.md) | readout-bound | AsEP/SAbDab2 training diagnostics | Recorded sign crossings occur at steps 77–110 on AsEP and 38–50 on SAbDab2. Converged positive cosine does not describe the early trajectory. |
| [Official RePaRank sequence checkpoint](cards/reparank-sequence.md) | metric-bound | 217 development complexes /68 components; 9,638,880 pairs /9,036 contacts | Macro pair AP 0.0378078; pooled AP 0.00873858; pooled AUROC 0.786064. These are different estimators, not interchangeable scores. |
| [RePaRank structure: completed and correctly centered](cards/reparank-centered.md) | metric-bound | Same 217-complex pair grid as sequence checkpoint | Centered structure macro AP 0.0160136 versus sequence 0.0378078; pooled AP 0.00236625 /AUROC 0.672153. Complete 217/217: 12 reused+205 fresh predictions. |
| [Structural PLMs: 36/36 complete, no promotion](cards/structural-plm.md) | metric-bound | 217 exposed development complexes /68 components | Complete SaProt/ProSST sweep: 36 tasks,12 arms; no real-layer crossed interval excludes zero against its matched random-width control. Decision NO-PROMOTION. |
| [Full partner grid does not confirm the small-panel anomaly](cards/full-partner-grid.md) | metric-bound | 122-antibody /119-antigen development library | 122×119=14,518 distinct pairings. The earlier small-panel MRR improvement after corruption is not robustly reproduced on the complete grid. |
| [Strict dual-axis retrieval: protocol, not a score](cards/strict-retrieval.md) | protocol-only | Prospectively separated authority-train development | Frozen retrieval contract and implementation support exist; no completed strict-model performance is promoted here. Passing builder and independent audit authorities are prerequisites. |
| [AsEP benchmark position without inflation](cards/asep-benchmark.md) | metric-bound | AsEP group/ratio test protocols with exposure histories | Ten-seed macro MCC≈0.1572 on epitope_group and 0.2584 on epitope_ratio. These constrain rather than establish benchmark leadership. |
| [Geometry gate remains failed after power analysis](cards/geometry-gate.md) | readout-bound | Historical geometry development comparison | The design analysis explains limited power and a Holm/Bonferroni field-label issue, but does not reverse the failed superiority gate. |
| [Powered replication is prospective](cards/geometry-replication.md) | protocol-only | Development-only geometry comparison | A separately numbered protocol specifies 24 prospective replicates and leaves the earlier failed gate unchanged. |
| [AsEP transfer: pair gain and epitope trade-off](cards/asep-pretraining.md) | readout-bound | Official 455 SAbDab2 transfer; separate 172-case AsEP-derived study | September 4 summary reports pooled pair AUROC 0.6220→0.6904 for1D versus3D+margin transfer, while epitope pooled AP falls 0.2210→0.1567. |
| [Routing: fresh 156 confirmation versus later development](cards/historical-routing.md) | readout-bound | Fresh 156 versus later 549 development validation | Fresh 156 routing gains are positive across 3 seeds. Later full-development routed medians 0.028576 /0.237464 /0.704746 are a distinct evidence stage. |
| [Pooling and ESM-C controls retain negative endpoints](cards/historical-pooling-esmc.md) | readout-bound | Development cohorts in August 27 snapshot | TopK1/2/4/8 and log-mean-exp are close; noisy-OR is worse. Published AbCDR-ESMC improves contact/epitope but lowers paratope. Local CDR adaptation trades CDR against full-chain quality. |
| [Prospective temporal admission failed closed](cards/temporal-gates.md) | readout-bound | Distinct prospective temporal/OOD attempts | August temporal cohort: 15 eligible complexes but 9 unique PDBs, below preregistered 10-PDB minimum. A separate later 14-candidate OOD admission also failed. |
| [AF3 semantic recovery is provenance, not accuracy](cards/af3-recovery.md) | engineering | Frozen 40-chain completion; dated operational snapshots | Task 0 originally failed after prediction due to a one-second valid CIF timestamp difference. Stricter semantics certified unchanged outputs while preserving the original failure. |
| [EPIC reviewer dispositions are not all-new experiments](cards/epic-closure.md) | readout-bound | Original submission versus post-submission/successor studies | Earlier package addresses 19 concerns through corrections, bounded follow-up evidence, explicit unavailable interventions and moderated claims. |
| [October 4 report: reconstructed ablation campaigns](cards/epic-new-ablations.md) | report-only | Reported official 455 test; 2,887 train /321 carved validation | Manuscript reports 36-model CampaignA and 39-model CampaignB; predicted-geometry epitope gains about+0.031 /+0.025 AP. Gate/cross-attention findings are mixed. |
| [October 4 report: consistency is not attributed to gating](cards/epic-new-consistency.md) | report-only | Reported reconstructed official 455 study | Head/contact-marginal rank agreement≈0.97–0.99 is reported; removing contact-derived readout changes little, while removing contact supervision damages paratope coupling. |
| [October 4 report: partner ranking near chance](cards/epic-new-partner.md) | report-only | Reported test 455 decoy diagnostic | One true antigen plus 9 decoys: reported less-length-confounded top 1 rates 0.103–0.110 versus chance 0.100; broad partner-conditioning claim withdrawn. |
| [October 4 report: ensemble gains and selection-transfer failure](cards/epic-v2-report.md) | report-only | Reported cluster-disjoint 324 validation and official 455 test | Reported AP 0.0214→0.0346 contact,0.2148→0.2502 epitope,0.6841→0.7215 paratope. Roughly half contact/epitope gain is seed ensembling. Contact-selected ensemble instead scores 0.0240 versus baseline ensemble 0.0273. |
| [October 4 report: predictor swap versus matched retraining](cards/epic-new-predictor.md) | report-only | Reported 321 validation swap; separate 455 retrained test | Inference-time validation OpenDDE swap and matched retraining are separate. Reported ensemble test epitope AP:AF3 0.2321 versus OpenDDE 0.2157; a long-entity training fallback remains AF3. |
| [October 4 report: specialist comparisons and input mismatch](cards/epic-new-specialists.md) | report-only | Reported official 455 and overlap-clean 386 | Reported EpiFormer epitope AP 0.372 exceeds reconstructed EPIC 0.214. Paraplume overlap excludes 69/455; on clean 386, improved EPIC 0.709 versus Paraplume 0.691 has CI crossing zero. |
| [October 4 report: overlap-stratified AsEP evaluation](cards/epic-new-asep.md) | report-only | Reported AsEP stratified external-cohort diagnostic | Reported 1,553 AsEP cases have median training similarity 0.97; below 0.5-identity stratum 412 has epitope AP 0.186 versus 0.316 in the near-duplicate stratum. |
| [October 4 report: label-definition correction](cards/epic-new-labels.md) | report-only | Reported 3,208 training/validation complexes | Reported heavy-atom≤4.5Å reproduces 3,047/3,208 complexes exactly; ≤4.502Å reproduces 3,169. Textual Cα definition conflicts with these labels. |
| [AbCite visual figure retrieval is a separate application](cards/abcite-visual.md) | readout-bound | PMC cropped figures; 295 total gold including 22 abstentions | Reported 273 answerable visual queries /15,949 figures:ColQwen2.5 R@1≈0.103,nDCG@5≈0.155. Retrieval and answer generation are separately evaluated. |
| [AbCite LoRA did not beat zero-shot](cards/abcite-lora.md) | readout-bound | 218 train /55 holdout figure IDs; frozen gallery | Reported 55-query holdout: zero-shot R@1≈0.0545 versus LoRA≈0.0182; nDCG@5≈0.0991 versus 0.050. |
| [AbCite ViDoRe reproduces released checkpoints](cards/abcite-vidore.md) | readout-bound | Official ViDoRe v1 full-page tasks | Official-harness mean nDCG@5 reported 0.8911 ColQwen2 and 0.8947 ColQwen2.5 over 10 tasks; checkpoint reproduction, not new-model invention. |
| [AbCite real audio/video and synthetic fixtures stay separate](cards/abcite-audio-video.md) | readout-bound | Real audio 29/video 26 galleries plus separate fixtures | Reported real-audio CLAP R@1≈0.414 on 29 items; video-frame R@1≈0.423 on 26. Synthetic hash/audio and keyframes have separate tables. |

## Beyond the curated cards

[337 evidence-location pages](FAMILIES.md) cover **1,530 candidate evidence-file versions**, including smaller studies, failure logs and protocols. The [4,913-version inventory](FILE_VERSIONS.jsonl.gz) contains every tracked path from all30 frozen HEADs, including code/assets. Neither number counts independent experiments.

The self-contained [HTML explorer](index.html) provides text, branch, file-kind, workstream and evidence-level filters. GitHub displays HTML source; open a saved copy in a browser for the interactive view. No external scripts, uploads, telemetry, model execution or local server. Markdown works directly on GitHub.

## Verify the presentation

```bash
python3 tools/evidence/build_atlas.py --check
python3 tools/evidence/verify.py --history
```

The clone must contain the pinned commits; no automatic fetch. [Verification receipt](VERIFICATION.json) · [Numeric bindings](METRIC_BINDINGS.json) · [Frozen refs](SOURCE_REFS.json) · [Manual definitions](CURATED.json).

**This verifies reporting integrity.** Source bytes, values, trees and reproducible pages are checked. Training, inference, unpublished cluster artifacts, label reconstruction and biological conclusions were not independently rerun. The old17-snapshot /747-link archive remains intact and is not relabelled as the new30-branch snapshot.
