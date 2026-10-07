# All-branch integration and progress — 7 October 2026

[Results](../../RESULTS.md) · [Result cards](../atlas/README.md) · [Merge ledger](MERGE_LEDGER.json) · [Preserved variants](PRESERVED_VARIANTS.json)

**31 / 31 starting branch HEADs are ancestors of the integrated source commit `35dfab14143578c75b7aa7c838cff8a5d2215231`.** There are 18 explicit three-way merge commits; shared/ancestor branches are included through those commits rather than counted as extra experiments. The starting main already included PR21.

This is genuine Git-history integration, not an `ours`-strategy ancestry-only merge. Conflicts were resolved explicitly; both versions remain in `archive/branch_variants_20261007/` and original commits. Not every experiment-specific runtime is interchangeable with the canonical main runtime.

## Updated completion view

| Workstream | Latest committed status | Evidence |
|---|---|---|
| Structural PLMs | 36/36 training tasks; final matched-random decision NO-PROMOTION | [Final readout](../../extensions/frontier_multimodel_v1/STRUCTURAL_PLM_FINAL_READOUT.md) |
| Centered structure baseline | 217/217 evaluated; 12 reused + 205 fresh | [Terminal receipt](../../extensions/frontier_multimodel_v1/evidence/reparank_centered_corrected_v1/evaluation.json) |
| Synthetic rescue and gate | Completed development evidence; clean-input damage and calibration shift retained | [Readout](../../extensions/rescue_qualification_v2/GATED_RESCUE_READOUT.md) |
| Strict retrieval | Both infrastructure and independent model contract integrated; still no completed strict-model result claimed | [Source separation](../../extensions/strict_partner_retrieval_v1/README.md) |
| Newer EPIC campaigns | Manuscript claims remain report-only pending matching terminal artifacts | [Evidence gaps](../atlas/REVIEW_FINDINGS.md) |
| AbCite | Separate application, including the negative LoRA holdout result | [Application results](../../abcite/evals/latest.md) |

These are committed records, not live scheduler states. A merge cannot turn a pending protocol, a failed decision or report-only prose into a completed scientific result.

## All starting branches

| Current branch | Research / engineering state | Integration route | Pinned source |
|---|---|---|---|
| `abcite/multimodal-copilot` | Multimodal evidence-retrieval application: Released-checkpoint reproductions and negative LoRA result | direct three-way merge | [`fb871f35`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/fb871f35678c4462074c0868a7738f1dc9483b05) |
| `docs/results-atlas-20261007` | Results-atlas documentation: Already merged through PR21 | already in starting main | [`1841b985`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/1841b985c8df81f566457efa0088d3a82a4efb9d) |
| `feature/asep-pretraining-pipeline` | AsEP transfer/pretraining pipeline: Committed summary with task trade-offs | direct three-way merge | [`b730070e`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/b730070e4d3247e26b3f64f5c10f8f4e16bfccae) |
| `kevin/af3-semantic-audit-20260930` | Strict AF3 semantic audit: Engineering recovery; partial historical snapshot | direct three-way merge | [`859a7145`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/859a7145bfb10bab87ca72a7dd1fbebe5d6a2399) |
| `kevin/correspondence-upgrade-20260930` | Three-encoder correspondence expansion: Development perturbation evidence | included through descendant merge | [`14a385fe`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/14a385feaec24b3141331963ba71dda788a164e1) |
| `kevin/epic-camera-ready-20260930` | Reviewer package and frozen campaign: Earlier closure and pre-result protocol | direct three-way merge | [`a7bb1a0c`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/a7bb1a0ca76fd7edca603842425bfc1ec2e1523c) |
| `kevin/evaluate-sensitivity-ci-fix-v1` | Aggregation repair and baseline-margin replication: Ten-seed development result after repair | included through descendant merge | [`ed8c33fb`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/ed8c33fb3e25666a72cf1f368e03d83d5c7d89a4) |
| `kevin/execution-recovery-20260930` | AF3/analyzer execution repair: Engineering and recovery provenance | direct three-way merge | [`a5729c49`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/a5729c49e5d32fc05cd09a10f824a7cc6974fb6b) |
| `kevin/framecontact-multimodal-evidence-20260902` | Broad FrameContact/scaling evidence: Historical studies, controls and plans | direct three-way merge | [`27605737`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/27605737aea2b2a6c623c3f90711ec64dd626532) |
| `kevin/frontier-performance-v1-20260930` | Learned-rescue implementation: Earlier rescue development source | included through descendant merge | [`f1fe5aff`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/f1fe5affe119230635b862b0985f2ee92fd781fc) |
| `kevin/geometry-gate-power-reanalysis-v1` | Power and decision-rule reanalysis: Historical gate remains failed | direct three-way merge | [`96e2a244`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/96e2a244cacd3d3a6d6b413a45e1a0af36c5f69d) |
| `kevin/geometry-powered-replication-v1` | Prospective powered geometry study: Protocol, not a terminal outcome | direct three-way merge | [`f04dbeb9`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/f04dbeb92be4b8e072b17d382d2e3edf3dcdf969) |
| `kevin/graph-community-split-v1` | Community split and associated source changes: Implementation, not a validated split | included through descendant merge | [`822d9159`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/822d91590a42ce970c9d13c176cbe3ef671de42a) |
| `kevin/hadamard-pair-decoder-v1` | Decoder comparison: Single-seed non-win | included through descendant merge | [`347acbdf`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/347acbdfb10a15a2b4081a676b0364d3b83ee8dc) |
| `kevin/main-epic-evidence-index-20260930` | Reviewer evidence integration: Ancestor of main | already in starting main | [`3ac740d3`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/3ac740d354f9c5b73c39605cb09b0f600609fb82) |
| `kevin/main-evidence-consolidation-20260930` | Earlier evidence consolidation: Historical integration snapshot | already in starting main | [`d75ce2f2`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/d75ce2f28750a738dceb5f4f3a4f81d0b36973f5) |
| `kevin/multimodel-scale-program-20260930` | Multimodel extraction/sweep program: Earlier execution lineage | included through descendant merge | [`28dffada`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/28dffada7aeaf9e5969f87414e91529aef60f717) |
| `kevin/partner-pose-contract-v1` | No-bound-pose input contract: Engineering contract | included through descendant merge | [`c72fa83b`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/c72fa83be658c6a7f5a28900409649e37233f327) |
| `kevin/partner-retrieval-20260930` | Split feasibility canary: Readiness, no strict-model score | direct three-way merge | [`c9bcb012`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/c9bcb012795e743909376353b2cc450db83866cd) |
| `kevin/pretrained-representations-v1` | Representation extraction backends: Implementation and coverage | included through descendant merge | [`400022a2`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/400022a29f05f8ef43502f39a2e52c345199e644) |
| `kevin/protocol-binding-ci-v1` | Protocol/source binding and test portability: Engineering verification | direct three-way merge | [`191a8b24`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/191a8b241d73f0f6c6bfeef9728b530a376a0721) |
| `kevin/reparank-structure-resume-20260930` | Hash-bound structure resume: Earlier resume/partial qualification | direct three-way merge | [`2bf3f7b3`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/2bf3f7b3bc2cf9ff54aeb294e29291f76378f4e3) |
| `kevin/representation-ablation-v1` | Representation and objective evidence: Evidence Freeze v 2 source | direct three-way merge | [`55cfd137`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/55cfd137aee2b3b749f244fcb76dd11d30dda2c5) |
| `kevin/rescue-gating-20260930` | Train-only abstention gate: Earlier gate protocol | direct three-way merge | [`bd387938`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/bd3879384aa8ba30ef859c14b13af056f141654f) |
| `kevin/rescue-qualification-20260930` | Completed rescue and centered RePaRank: Final development results and countercontrols | included through descendant merge | [`a11a6ae7`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/a11a6ae7a9e2d56a3ff29bb8057e74f1c479b7cc) |
| `kevin/s2-split-readiness-v1` | S 2 readiness diagnosis: Implementation/diagnosis | direct three-way merge | [`053a0665`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/053a0665cb4ccb9df5ce325df330ebae2fc4f973) |
| `kevin/strict-dual-axis-retrieval-20260930` | Dual-axis split/scoped embedding infrastructure: Frozen protocol and readiness | direct three-way merge | [`e9c253ba`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/e9c253bace2edddf60cb4a10e1e839617b5cb927) |
| `kevin/strict-partner-retrieval-20260930` | Retrieval model/metric contract: Protocol and preflight | direct three-way merge | [`fdc2db39`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/fdc2db39bd9f8817ae414a17dcf666a565433d83) |
| `kevin/structural-plm-final-20260930` | Formal structural-PLM final analysis: 36/36 complete; NO-PROMOTION | direct three-way merge | [`96864049`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/96864049bdb935d14a4439c3f048231f59e78bee) |
| `kevin/teacher-paper-handoff-20260903` | Checkpoint-aligned paper handoff: Historical test 549 recovery | direct three-way merge | [`54afd798`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/54afd798700bc41ac1352065414d8687c9627871) |
| `main` | Curated evidence and paper hub: Mixed measured receipts and newer report-only prose | already in starting main | [`6e80c6d4`](https://github.com/yinli-systems/antibody-antigen-interface-research/tree/6e80c6d44be531edd6f40404521ba4f087a81fd4) |

## Neutral branch names

| Previous name | Current name | Same commit? |
|---|---|---|
| `kevin/epic-reviewer-response-20260930` | `kevin/epic-camera-ready-20260930` | Yes, `a7bb1a0c` |
| `kevin/main-epic-review-index-20260930` | `kevin/main-epic-evidence-index-20260930` | Yes, `3ac740d3` |

Historical freeze manifests retain the names that actually existed at their observation time. Current navigation uses the new names; evidence links use immutable commits. No historical result or commit was rewritten to erase review provenance.

## Source incompatibilities handled explicitly

The older AsEP pipeline replaced shared model/training/solver behavior and included a split-manifest mismatch bypass. Its separate adapters, scripts and reports are now present, while the newer strict canonical runtime is retained. Both shared-file versions are archived; reproduce that historical pipeline from its source commit, not by assuming main is its original runtime.

The two strict-retrieval branches define incompatible model interfaces. `models.py` retains the split/training-line implementation; the independent contract is imported byte-for-byte as `contract_models.py`. Its tests only change their import namespace. No architecture is synthesized by combining the two.

The stronger full-semantic audit is retained rather than removing duplicate-key, timestamp, atom/topology and confidence-integrity checks. Earlier implementations and failure evidence remain archived.

The optional multimodal pretraining stack also has an independent `antibody_pretraining.multimodal_v1` package. Its three modules are exact original source copies. The frozen core retains registered source hashes, while both frozen-core and optional-feature tests run separately. This prevents protocol-digest repairs from accidentally disabling the optional implementation. See [namespace provenance](NAMESPACE_INTEGRATION.json).

A full-suite integration regression also exposed an older pair-geometry variant that added cross-chain projection parameters and broke the registered pose/parameter-ownership contracts. The already-registered canonical source was restored; neither the tests nor the contracts were weakened. Both variants remain preserved. [Follow-up source decisions](FOLLOWUP_SOURCE_FIXES.json).

## Validation scope

See [VALIDATION.md](VALIDATION.md) and the machine-readable [verification receipt](VERIFICATION.json). Source integration does not replace scientific replication. Original 17 archive snapshots, the 30-branch atlas snapshot and its 87 checked fields remain separately verifiable.
