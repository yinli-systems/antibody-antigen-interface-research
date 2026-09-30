# One held-out confirmation of the correspondence effect

Registered 2026-09-11, **before any sabdab2 test prediction exists**. This is an
amendment on top of `evidence-freeze-v1`, which is not modified.

## Why this is being run after a freeze

Every headline number in the frozen evidence comes from a validation split that
participated in this project's model development. That is the most specific
correctness objection available to a reader, and no amount of further
development-set work answers it. A single read of a split that never took part
does.

The freeze stands. If this confirmation completes, it becomes
`evidence-freeze-v2 = v1 + one preregistered, one-read sabdab2 test
confirmation, with no training and no model-selection changes`.

## The access audit that precedes it

`scripts/audit_test_access.py`, run before this protocol was written:

| check | result |
|---|---|
| the manifest has an independent test membership | 217 instances, disjoint from train (3229) and validation (217) |
| no test prediction or evaluation artifact exists | 0 found |
| every evaluation receipt names its split | 141 receipts, all `validation` |
| no logged command ran this cohort with `--split test` | **three scripts found** — see below |
| selection used validation only | 120 configs, none referencing a test manifest |
| what cannot be checked mechanically | stated, not asserted away |

**The fourth check did not pass and the finding is recorded rather than waved
through.** Three EpiFormer baseline scripts invoke `--split test`. They run on a
different machine, against a different split contract
(`sabdab2-baseline-outsourcing`), and they did run: `results/epiformer/` holds
four artifacts marked `split: test` over 455 instances.

That is a different partition. Its overlap with this project's sabdab2 test
membership is **32 of 217 instances, 14.7%**; it also overlaps this project's
validation by 58 and train by 365. So it is not a superset and not a copy — it is
an unrelated benchmark drawn from an overlapping pool.

**What this means.** No model in this paper has been evaluated on this test
split, and no configuration references it. For 32 of 217 complexes, a different
model's per-complex scores exist in this repository. Whether that could have
influenced a decision here cannot be excluded mechanically, so the design removes
it rather than arguing about it.

## The measurement

**Nothing is trained.** The existing frozen checkpoints are evaluated, once, on
the test split.

| | |
|---|---|
| contrast 1 | ESM-IF1 `aligned` − `shuffled` |
| contrast 2 | ProteinMPNN `aligned` − `shuffled` |
| metric | macro pair AUPRC |
| seeds | the same ten paired seeds, no others |
| checkpoints | the existing `best.pt`, unchanged |
| threshold | none — AUPRC is threshold-free |
| ensembling | none |
| representation, model, hyperparameters | unchanged |
| primary statistic | paired per-seed mean difference with a percentile bootstrap interval, the frozen convention |
| secondary | complex-level heterogeneity, the same decomposition used on validation |

**Two contrasts, and no others.** Not the random control, not AntiBERTy, not the
dose curve, not a companion metric sweep. Adding an arm after seeing a test
number is the freedom this registration exists to remove, and it would be
available precisely when the first result looks good.

### Two readouts from one read

Reported together, from the same single evaluation:

- **all 217 test complexes** — the registered split
- **the 185 complexes with no overlap of any kind** — the subset that no model,
  including EpiFormer, has been evaluated on

The second is a sensitivity check on the 14.7% exposure, not a second look. If
the two disagree in direction, that disagreement is the finding and is reported
as such.

## Outcomes, fixed now

**Both contrasts positive and decided** — the correspondence effect holds on a
split that took no part in development. The headline claim survives the strongest
objection available to it.

**Positive but smaller, or decided on one contrast only** — reported exactly that
way, with the development-set figures beside the test figures so the shrinkage is
visible rather than buried.

**Null or reversed** — the central claim of the paper contracts to a
development-set finding, and the abstract is rewritten before submission. This
outcome is the reason to run the confirmation now rather than to meet it in
review.

All three go into the paper. The decision rule is the one this project already
uses for these arms: `bootstrap_ci_excludes_zero`, reported alongside the
favourable seed count and the exact sign bound, with the word "decided" reserved
for the ledger and not the prose.

## After the read

The sabdab2 test split is **closed permanently**. No second look, no
re-evaluation under a different metric, no follow-up conditioned on the outcome.
Both AsEP test splits were already spent; this closes the third and last.
