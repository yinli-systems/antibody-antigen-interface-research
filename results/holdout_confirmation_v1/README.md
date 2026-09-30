# The held-out confirmation: one contrast holds, one does not

The single registered read of the sabdab2 test split, per
`protocols/HOLDOUT_CONFIRMATION_V1.md`. Nothing was trained. The existing frozen
checkpoints were evaluated once on 217 complexes that took no part in selecting
them. **The split is now closed.**

## Result

| contrast | readout | aligned | shuffled | difference | 95% CI | seeds up | sign bound |
|---|---|---:|---:|---:|---|---|---|
| **ESM-IF1** | all 217 | 0.027919 | 0.025659 | **+0.002260** | [**−0.000007**, +0.004901] | 6/10 | 0.3035 |
| ESM-IF1 | unseen 185 | 0.029271 | 0.026248 | +0.003022 | [+0.000310, +0.006084] | 7/10 | 0.3934 |
| **ProteinMPNN** | all 217 | 0.028490 | 0.024896 | **+0.003594** | [+0.001559, +0.005881] | **9/10** | **0.6058** |
| ProteinMPNN | unseen 185 | 0.030425 | 0.025516 | +0.004909 | [+0.002662, +0.007365] | 9/10 | 0.6058 |

The registered rule is `bootstrap_ci_excludes_zero`. Under it, **ProteinMPNN is
decided on the registered split and ESM-IF1 is not.** Under the project's other
rule, the exact sign bound above 0.5, the same split holds: ProteinMPNN at 0.6058
passes, ESM-IF1 at 0.3035 does not.

**The ESM-IF1 interval misses by 7 × 10⁻⁶.** That is not "essentially decided".
The rule was fixed before the read precisely so that a margin this thin could not
be argued across, and it is reported as undecided.

## Against the development figures

| | development | test, all 217 |
|---|---|---|
| ESM-IF1 | +0.008204, 10/10 | +0.002260, 6/10, **undecided** |
| ProteinMPNN | +0.003193, 10/10 | +0.003594, 9/10, **decided** |

**The encoder with the larger development effect shrank by a factor of 3.6 and
lost its decision; the one with the smaller effect did not shrink at all.** That
is the reverse of what the development set predicted, and it is the result.

Development and test are different complexes, so the two columns are not the
same estimand measured twice — but they are the same models, the same metric and
the same seeds, so the comparison is the one a reader will make and it is put
here rather than left to be made elsewhere.

## The 185-complex readout is a sensitivity check, not a rescue

32 of the 217 test complexes appear in an unrelated baseline's test evaluation
(`results/epiformer/`, a different split contract, a different model). The 185
with no overlap of any kind were reported from the same single read.

The two readouts **agree in direction**: both contrasts positive in both. They
differ in decidedness for ESM-IF1, which is decided on the subset and not on the
registered split.

**The registered split is 217 and its answer is undecided.** The subset is not
used to replace it. What the subset supports is narrower: the exposure does not
explain the direction, and removing it moves the estimate slightly up rather than
down. ESM-IF1's sign bound on the subset is still 0.3934, below 0.5.

## What may now be written

> On a split that took no part in development, destroying residue correspondence
> costs performance for both structural encoders. The effect is decided for
> ProteinMPNN on ten paired seeds and undecided for ESM-IF1, whose
> development-set advantage shrank roughly threefold.

## What may not

Not that the correspondence effect is confirmed on held-out data without
qualification. One of two contrasts confirmed.

Not that ESM-IF1's effect is absent. Positive on both readouts, 6 and 7 of 10
seeds, with an interval whose lower end sits at −7 × 10⁻⁶. Undecided is not
absence — and it is not presence either.

Not a second look. No re-evaluation under another metric, no threshold, no
ensemble, no additional arm. The two contrasts registered are the two reported.

## Scope

sabdab2 test, read once, now permanently closed. Both AsEP test splits were
already spent. This project has no unread test data remaining.
