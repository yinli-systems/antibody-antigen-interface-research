# Residue correspondence and partner-specific antibody–antigen interfaces

An evidence-gated research codebase for understanding **what residue-level protein representations know about exact antibody–antigen contacts — and what they do not**.

The project goes beyond marginal epitope/paratope propensity. Its central question is whether a model preserves the **partner-specific residue correspondence** needed to identify which antibody residue interacts with which antigen residue under strict split, provenance and leakage controls.

> **Research status — 30 Sep 2026.** The strongest completed evidence is consolidated below. We deliberately make **no state-of-the-art claim**: benchmark comparability, exposed historical tests, failed promotion gates and negative results are kept explicit. The current SaProt/ProSST layer sweep is still running and is not used to choose a headline result.

## What the project has established

| Finding | Audited result | Evidence boundary |
|---|---|---|
| **Residue correspondence governs frozen-feature transfer** | Aligned ESM-IF1 improves macro pair AUPRC over ESM-2 by **+0.005111 (+22.8%)** across 10 paired seeds; shuffled ESM-IF1 is **−0.003093 (−13.8%)** below ESM-2; aligned vs shuffled is **+0.008204 (+42.4%)**. A width-matched random control is undecided vs ESM-2. | Development attribution; [`results/representation_attribution_v1/report.json`](results/representation_attribution_v1/report.json) |
| **The effect survives stronger correspondence-preserving controls** | Same-amino-acid shuffling reduces contact AP from **0.027995 → 0.011100** for ESM-2+ESM-IF1; equal-component/equal-seed Δ **−0.017151**, pointwise crossed-bootstrap 95% CI **[−0.022688, −0.012434]**. | Exploratory development extension; [`extensions/correspondence_v2/READOUT.md`](extensions/correspondence_v2/READOUT.md) |
| **Correspondence is necessary for useful transfer, but not sufficient** | On a preregistered 217-complex holdout, ProteinMPNN aligned−shuffled is **+0.003594** with 9/10 seeds and CI **[0.001559, 0.005881]**; ESM-IF1 is **+0.002260** but undecided, CI **[−0.000007, 0.004901]**. | Single holdout read; [`results/holdout_confirmation_v1/confirmation.json`](results/holdout_confirmation_v1/confirmation.json) |
| **Auxiliary objectives are not interchangeable** | Contact supervision raises pair AUPRC by **+0.0833**; removing the complex-ranking term improves **pair AUPRC +0.0214** and **node MCC +0.0369** across 10 paired seeds. | Development factorial ablation; [`results/CLAIMS_LEDGER.md`](results/CLAIMS_LEDGER.md) |
| **Optimisation geometry changes sign during training** | Node/contact gradient cosine is negative early and crosses to durably positive at steps **77–110** on AsEP and **38–50** on SAbDab2; every registered seed shows the transition. | Training-time diagnostic, not a causal mechanism; [`results/gradient_alignment_dynamics_v1/step_trace_summary.json`](results/gradient_alignment_dynamics_v1/step_trace_summary.json) |
| **Exact contact correspondence remains much harder than interface localisation** | Official RePaRank sequence checkpoint on the exact 217-complex grid: **macro pair AP 0.037808**, pooled AP **0.008739**, AUROC **0.786064**, epitope AP **0.202039**, paratope AP **0.557707** over **9,638,880** residue pairs / 9,036 positives. | Exposed pretrained checkpoint; training-overlap not audited; [`evaluation.json`](extensions/frontier_multimodel_v1/evidence/reparank_same_protocol_v1/sequence_full/evaluation.json) |

### FrameContact: deployable predicted-geometry evidence

The no-privileged-distance three-seed development system reaches median mean-per-complex AP of approximately **0.0261 contact / 0.2327 epitope / 0.6962 paratope** using sequence plus independently predicted intra-chain geometry. On the correctly aligned **549-case historical test** the fixed three-seed routed medians are **0.02667 / 0.22031 / 0.69859**. The 549-case result is explicitly an **exposed historical test**, not fresh blind evidence and not proven antibody+antigen sequence-double-cold generalisation.

See [`paper/neurips_2026_framecontact/TEACHER_HANDOFF_20260903.md`](paper/neurips_2026_framecontact/TEACHER_HANDOFF_20260903.md) and [`results/aligned_test549_replay_v1/RESULT.json`](results/aligned_test549_replay_v1/RESULT.json).

### Partner ranking: a negative result that clarified the science

A complete **122 antibody × 119 antigen = 14,518 pair** diagnostic showed that the earlier small-panel “contact down, MRR up” observation **does not robustly reproduce**. The current complex score is instead strongly associated with interaction dispersion (mean Spearman ≈ **0.9743**) and has substantial length/surface/node-prior associations. This separates **global partner ranking** from **local residue correspondence** and prevents the project from overclaiming specificity.

See [`extensions/frontier_multimodel_v1/SCORE_DISSOCIATION_READOUT.md`](extensions/frontier_multimodel_v1/SCORE_DISSOCIATION_READOUT.md).

## External benchmark reality check

The AsEP benchmark is intentionally retained even though it is not a headline win. On `epitope_group`, the registered 10-seed single-model macro MCC is **0.1572**; on `epitope_ratio` it is **0.258**. Published configurations on those protocols can score higher. This repo therefore treats its contribution as **controlled attribution, mechanisms, failure boundaries and reproducibility**, not a disguised SOTA claim.

See [`results/asep_benchmark_v1/BENCHMARK.md`](results/asep_benchmark_v1/BENCHMARK.md).

## Active programme — not yet a result

A frozen **36-task SaProt/ProSST layer sweep** is running: 10 real structure-token layers × 3 seeds plus two exact-width/coverage random controls × 3 seeds. At the latest committed snapshot, six SaProt tasks are complete:

- SaProt layer 0, 3-seed mean validation pair AP ≈ **0.02058**.
- SaProt layer 8, 3-seed mean validation pair AP ≈ **0.02104**.

The difference is small and **no layer winner is selected** until all 36 tasks and matched controls finish. The RePaRank structure checkpoint also remains incomplete on the full 217 cases because the existing predicted-structure authority covers 187/217 complexes; a 40-chain AF3 completion authority is frozen separately.

Current execution details: [`results/CURRENT_STATUS_20260930.md`](results/CURRENT_STATUS_20260930.md).

## Evidence map

Start here instead of browsing branch names:

- **Canonical results:** [`results/INDEX.md`](results/INDEX.md)
- **Every paper-level number and its artifact:** [`results/CLAIMS_LEDGER.md`](results/CLAIMS_LEDGER.md)
- **Reproducibility notes:** [`results/REPRODUCIBILITY_NOTES.md`](results/REPRODUCIBILITY_NOTES.md)
- **Frozen representation evidence:** [`freeze/FREEZE.md`](freeze/FREEZE.md)
- **Cold reproduction:** [`reproduction/README.md`](reproduction/README.md)
- **Branch provenance / archive tags:** [`docs/BRANCH_PROVENANCE.md`](docs/BRANCH_PROVENANCE.md)
- **Machine-readable branch evidence index:** [`results/BRANCH_EVIDENCE_INDEX.json`](results/BRANCH_EVIDENCE_INDEX.json)
- **Complete pre-consolidation branch file inventory:** [`archive/BRANCH_FILE_INVENTORY_20260930.json`](archive/BRANCH_FILE_INVENTORY_20260930.json)

## Repository design

```text
results/        canonical measurements, ledgers, benchmark reports and receipts
paper/          paper-facing narratives, figures and corrected handoffs
protocols/      preregistered/frozen experimental contracts
freeze/         immutable claim/provenance freeze for the representation study
reproduction/   cold-reproduction entry points and receipts
extensions/     post-freeze programmes (correspondence, structural PLMs, RePaRank)
archive/        machine-readable preservation inventory; no experimental history deleted
```

The repository intentionally does **not** store raw datasets, full model weights, complete embedding banks or every dense prediction tensor. Large artifacts remain receipt-bound by path/hash; lightweight summaries and provenance stay in Git.

## Claims policy

1. **Absolute paired effects are primary.** Relative percentages are secondary because the pair-AP baselines are small.
2. **Development, historical exposed test, holdout and blind/OOD are different evidence classes.** They are never collapsed into one “test” label.
3. **Failed gates remain results.** Hadamard decoding, focal pair loss, temporal OOD admission failures, structure-authority failures and disqualified scaling candidates remain discoverable.
4. **No metric borrows another experiment's seed count, confidence bound or cohort.** See the claims ledger for the defect that motivated this rule.
5. **No branch was deleted during consolidation.** Every pre-consolidation branch head is pinned by an `archive/2026-09-30/...` tag.

## Why the main branch is curated this way

GitHub recommends using the README to explain what the project does, why it is useful and how to get started, while reproducibility guidance emphasises complete artifact inventories, executable workflows and clear links from claims to evidence. The main branch therefore keeps the scientific story compact and pushes complete provenance into machine-readable indexes rather than hiding it or flooding the front page.

---

**Current scientific thesis:** models can learn strong interface propensity and global interaction signals without learning the exact partner-specific residue correspondence required for reliable contact prediction. The most reproducible gains in this repository come from preserving that correspondence and from separating useful contact supervision from objectives that interfere with it.
