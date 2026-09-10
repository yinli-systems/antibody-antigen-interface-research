# Checkpoint selection: discovery, replication, independent confirmation

Registered 2026-09-10, **before the group-validation replication reported and
before any ratio-split run using the new selection metric existed**.

## What was discovered, and where

An exploratory sweep of eighteen configurations on AsEP `epitope_group`
**validation** found that checkpoint selection dominates every other knob tried.
Selecting on the total validation objective instead of `partner_rank` moved
per-complex epitope MCC from a four-seed baseline mean of 0.2197 to 0.2566 at
`d_model` 128 and 0.2845 at 256, against a noise floor of sd 0.0055 and a
four-seed range of 0.0105. Learning rate and positive-class weight produced
nothing outside that floor, and `d_model` 256 under `partner_rank` selection
produced nothing either — the capacity only mattered once the selection changed.

Each of those is one seed. This protocol is what turns it into a result.

## The rule that is not negotiable

**The `epitope_group` test split is permanently frozen.** It has had the two
readings registered in `ASEP_TEST_MEASUREMENT_V1.md` and it will not be read
again — not for this configuration, not for any other.

The frozen group-test figures stand as they are: MCC 0.157, AUROC 0.697 for the
single model. Their value is that no post-hoc tuning touched them, and
re-evaluating a configuration discovered afterwards on that same split would
destroy exactly that.

Group-validation figures and published group-**test** figures are never placed
in one table. They are different splits and comparing them would be a category
error dressed as a result.

## Stage two: replication on group validation

Job 1577897, already running: `--selection-metric total` at `d_model` 128 and
256, ten seeds each, the same ten as every other arm, validation only.

Reported as a paired comparison against the `partner_rank` arm at the same
seeds, with the interval and the sign bound. If it does not replicate, the sweep
observation is recorded as a single-seed artefact and nothing further happens.

## Stage three: independent confirmation on the ratio split

The `epitope_ratio` test split has been read **once**, for the original frozen
configuration, giving MCC 0.2584. This registers the **second and final**
reading, for the selection-metric configuration.

**Everything is fixed here, before that split is touched:**

| | fixed to |
|---|---|
| selection metric | `total` |
| `d_model` | whichever of 128 / 256 has the higher mean epitope MCC in the **group-validation** replication |
| layers | 1 |
| loss weights | the defaults, unchanged |
| epochs | 12 |
| representations | ESM-2 + ESM-IF1 |
| seeds | 1-10 |
| threshold | selected on ratio **validation** to maximise per-complex MCC, applied unchanged to ratio test |
| primary figure | **single model**, mean over ten seeds, because the published rows are single models |
| secondary figure | ten-seed mean-score ensemble, reported separately and labelled |

`d_model` is chosen from group validation and from nothing else. The choice is
made before any ratio-split training starts.

**What will not happen on the ratio split**: no comparison of selection metrics
there, no choice of `d_model` after seeing its test numbers, no threshold
adjusted after a test look, no ensemble strategy picked on test performance.
Any of those would move the problem from one test split to the other rather
than solving it.

## What each outcome means

**Ratio test improves over 0.2584**: checkpoint-selection misalignment transfers
across splits and is a property of the model rather than of one validation set.

**Ratio test does not improve**: the misalignment is pronounced on group-split
validation and does not transfer. That is recorded as the finding, and the
group-validation replication is reported as split-specific.

**Ratio test degrades**: recorded as such. A configuration chosen on one split's
validation can be worse on another's test, and that would be worth knowing.

No further reading of either test split follows from any of these.
