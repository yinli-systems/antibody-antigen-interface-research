# Structural PLM programme execution status — 2026-09-30

This is the live handoff for branch `kevin/multimodel-scale-program-20260930`.
It supersedes the older “not completed at recovery” status for SaProt/ProSST,
but it does not rewrite historical failure records or claim that the active
36-task layer sweep has finished.

## Interface / reasoning-mode boundary

The execution environment does not expose a control for inspecting or changing
the user's ChatGPT web-account “Max reasoning” selector. No account/model-tier
setting was changed programmatically. The work below was executed with the
reasoning capability available in the current session; this is an interface
limitation, not evidence about the user's subscription entitlement.

## Resumed state verified before new submission

- GitHub was first verified at `5eb4a45f27ed9f32eea07956701128fcb1222d58`.
- The 122 antibody × 119 antigen full-grid diagnostic was already complete and
  was not rerun. Its earlier small-panel MRR increase did not reproduce as a
  robust full-grid effect; see `SCORE_DISSOCIATION_READOUT.md`.
- The post-cutoff OOD metadata gate retained 14 candidates and admitted 0/14.
  It was not relaxed and no OOD coordinate/contact/performance label was read.
- Completed SaProt/ProSST shards were inspected before launch. No extraction,
  quantisation or feature shard was resubmitted.

## Completed structural-feature authority

The authoritative extraction root is
`/ssd/scxi253/frontier_multimodel_v1/structural_plm_full_v2`.

- Extraction plan SHA-256:
  `2ae83660e2b89d231fe48c9cb07308cada6502e1a755002e24f7572542b27cd1`.
- Coverage audit SHA-256:
  `03cc935e1defd1c47991b3b1182f03cf667bdc893fadaf2040850892b0141531`.
- **33,450 / 33,450** expected frozen representation artifacts are present.
- 3,345 structure-token records are covered: 2,174 antibody-chain units and
  1,171 antigen units.
- SaProt-650M layers `{0,8,16,24,33}` use genuine Foldseek AA+3Di tokens.
- ProSST-2048 layers `{0,3,6,9,12}` use the official 2048-state quantiser.
- Both antibody and antigen source specifications exist for every layer: 20
  immutable source specs total.
- Extraction audit reports `contact_labels_loaded=false`; this stage is feature
  authority, not downstream performance.

Recovery arrays `1640207_0–63` (ProSST quantisation) and `1640208_0–31`
(ProSST feature extraction) completed 0:0. Earlier `1640122` failures/cancellations
remain preserved. Exact scheduler history and output counts are committed under
`evidence/structural_plm_full_v2/`.

## Downstream training implementation and source commits

- `58740ea838a440462c52243107def84a5b97a09b`: frozen 10-layer × 3-seed
  training programme plus two width-matched controls.
- `4c4b8d15ae70a31b07534389cdc8bbda324b91dc`: explicit source `PYTHONPATH`
  for training/evaluation subprocesses.
- `3eb4293464552199c66eff9b6add14d7ed75ae14`: bounded smoke partner panel
  while formal 217-case evaluations retain nine distractors.

The exact source snapshot used by the active campaign is
`source_exact_3eb4293`. Targeted source/control/training tests: **63 passed**.

## Preserved smoke history

| Attempt | Job | Terminal result | Evidence |
|---|---:|---|---|
| attempt1 | `1640449` | FAILED 1:0 in 2 s | subprocess lacked explicit `source/src`; no training data or metric read completed |
| attempt2 | `1640468` | FAILED 1:0 after real 48-case training | training/checkpoint succeeded; 24-case smoke evaluation produced a ragged nine-distractor panel; no scientific score promoted |
| attempt3 | `1640488` | **COMPLETED 0:0** | real SaProt layer-16 feature input; one epoch; best/last validation receipts; stderr empty |

Successful smoke checks:

- observed per-side input width `2561 / 2561`, exactly the frozen SaProt width;
- parameter count `1,096,461`;
- model/source identity and revision checks passed while artifacts were loaded;
- best and last checkpoint evaluation each covered 24 smoke cases;
- `test_labels_read=false`, `ood_labels_read=false`, and
  `scientific_result=false`;
- smoke pair AP `0.0178728` is an engineering value and must not appear as a
  layer-sweep result.

## Active formal layer sweep

Attempt root:
`/ssd/scxi253/frontier_multimodel_v1/structural_training_v1/attempt3`.

- Frozen plan SHA-256:
  `24d6c6cfb2cee44b9cc289ac8721b3ad183321db11a059c14b29c878dd7cea12`.
- Formal array: **`1640503`**, tasks `0–35`, maximum six concurrent RTX 4090s.
- Dependent complete-campaign analysis: **`1640514`**, `afterok:1640503`.
- An earlier CPU-count-8 analysis submission was rejected before allocation
  because the partition permits six CPU cores per GPU. It was corrected to six
  cores; the formal training array was not resubmitted.

At the committed status snapshot (`2026-09-30T02:47:09Z`), tasks `0–5` were
RUNNING with empty stderr, tasks `6–35` were PENDING behind the six-task
concurrency gate, and no formal `FAILURE.json` existed. Tasks `0–2` are SaProt
layer 0 seeds 1–3; tasks `3–5` are SaProt layer 8 seeds 1–3. This is a running
state, not a completed result.

The analysis job requires all 36 `COMPLETE_DEVELOPMENT_TRAINING` receipts and
all 217 validation complexes per best/last checkpoint. It will fail rather than
silently dropping a task, seed or complex. The registered primary endpoint is
validation macro mean-per-complex pair AP. Partner ranking remains secondary
because score/contact dissociation has already been observed.

## Completed versus not completed

Completed in this recovery:

1. official-contract SaProt/ProSST structure tokenisation and full frozen
   feature extraction;
2. independent 33,450-artifact coverage audit;
3. exact generic-source and width/coverage-matched control wiring;
4. frozen downstream training protocol, exact 36-task plan and passing real-GPU
   smoke;
5. active formal training submission with fail-closed dependent aggregation;
6. integration of exact plans, source specs, failure logs and scheduler receipts
   into GitHub.

Not completed yet:

- terminal metrics from the active 36-task layer sweep;
- development layer selection or a validated improvement over the matched
  random controls;
- learned realignment rescue;
- an admitted untouched OOD measurement (the current 14/14 cohort failed gate);
- same-protocol RePaRank reproduction;
- large co-folder comparison;
- affinity / mutation endpoint;
- any blind, SOTA, “first place” or production claim.

The next scientific stage is conditioned on the layer-sweep evidence. No new
OOD, RePaRank, co-folder or affinity GPU campaign is launched from this status
snapshot, which avoids spending compute before the structural representation
choice and negative controls are known.
