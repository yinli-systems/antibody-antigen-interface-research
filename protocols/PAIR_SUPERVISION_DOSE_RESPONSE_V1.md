# Pair supervision, dose response

Registered 2026-09-10, **before the intermediate arms were submitted**. The two
endpoints already exist and their numbers are known; the two intermediate points
do not.

## The claim under test

`results/multitask_coupling_v1/` reports that removing pair supervision entirely
collapses the pair task and leaves the node tasks alone — and that epitope node
AUPRC is **3.1% higher without it, at 10 seeds out of 10**.

Unanimity across ten seeds is hard to read as noise, so the working explanation
is capacity: the pair terms carry 1.5 of the 2.5 total loss weight, and the
shared trunk spends parameters on them that the node heads would otherwise use.
If that is right, the effect is not a step between "on" and "off" but a
**gradient**, and node performance should rise monotonically as pair supervision
is dialled down.

That finding was a by-product with no protocol behind it. This is the protocol.

## The design

One multiplier λ scales both pair terms together, from the default
`edge 1.0, partner_rank 0.5`:

| λ | edge | partner_rank | status |
|---:|---:|---:|---|
| 1.00 | 1.00 | 0.50 | already run, 10 seeds |
| 0.50 | 0.50 | 0.25 | **to run**, seeds 1-10 |
| 0.25 | 0.25 | 0.125 | **to run**, seeds 1-10 |
| 0.00 | 0.00 | 0.00 | already run, 10 seeds |

Same seeds at every point, so every comparison is paired. Nothing else changes:
same split, same frozen representations, same twelve-epoch budget, same
`partner_rank` checkpoint selection — which stays on even at λ=0, where it
selects on a term the model is no longer trained on. That is deliberate: leaving
it fixed keeps the arms comparable, and changing it would confound the dose with
the selection rule.

**Validation only.** The AsEP test split has had its two registered readings and
gets no more. Every figure here is on validation.

## What each outcome means, fixed now

**Monotone rise in epitope node AUPRC as λ falls**: the capacity account is
supported, and the recorded claim becomes that pair supervision costs node
performance in proportion to its weight.

**Flat between λ=1 and λ=0.25, with the gain appearing only at λ=0**: the
capacity account is wrong. Something specific to removing the terms entirely —
most likely the loss landscape or the effective learning rate on the shared
trunk — explains it, and the finding stays an observation rather than a
mechanism.

**No consistent pattern**: the 10/10 unanimity at the endpoints was a property
of those two configurations and not a gradient. The multitask_coupling record is
then amended to say the direction is unexplained.

**Pair task collapses smoothly with λ**: expected, and it is the control. If the
pair metrics do *not* fall with λ, the multiplier is not doing what this protocol
assumes and nothing else here can be read.

## What will not be done

No test reading. No further λ values chosen after seeing these. If the pattern is
ambiguous at four points it is reported as ambiguous.
