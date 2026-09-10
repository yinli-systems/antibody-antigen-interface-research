# A second structural encoder: the ordering replicates, the magnitude does not

Registered in `protocols/SECOND_STRUCTURAL_ENCODER_V1.md` before any ProteinMPNN
feature was extracted. Three arms × ten seeds, sabdab2 validation, macro pair
AUPRC, paired within seed. ESM-2 alone is the existing twenty-seed arm, reused.

Matched to the ESM-IF1 arms in cohort, manifest, backbone store, chains, split,
seeds and endpoint. The encoder is the only deliberate difference.

## Arm means

| arm | macro pair AUPRC |
|---|---|
| ProteinMPNN shuffled | 0.019632 |
| ProteinMPNN random (width-matched) | 0.021387 |
| ESM-2 alone | 0.021764 |
| ProteinMPNN aligned | **0.022825** |

**Observed ordering: shuffled < random < esm2 < aligned.** The registered
prediction was `shuffled < random ≈ esm2 < aligned`, and `random ≈ esm2` holds:
−4.8%, interval containing zero, 2 of 10 seeds.

## Paired contrasts, both registered rules reported

| contrast | mean difference | bootstrap CI95 | seeds up | CI excludes 0 | sign bound > 0.5 |
|---|---|---|---|---|---|
| aligned − shuffled | +0.003193 | [+0.00220, +0.00421] | 10/10 | yes | **better** |
| aligned − random | +0.001438 | [+0.00030, +0.00260] | 8/10 | yes | **neither** |
| aligned − esm2 | +0.000365 | [−0.00134, +0.00200] | 7/10 | no | neither |
| random − esm2 | −0.001073 | [−0.00255, +0.00031] | 2/10 | no | neither |
| shuffled − esm2 | −0.002828 | [−0.00406, −0.00173] | 0/10 | yes | **worse** |
| shuffled − random | −0.001755 | [−0.00253, −0.00096] | 1/10 | yes | **worse** |

### The protocol did not name its decision rule, and it matters on one leg

`SECOND_STRUCTURAL_ENCODER_V1.md` uses the word "decided" and never says under
which rule. On `aligned − random` the two rules in use in this project disagree:
the bootstrap interval excludes zero, while 8 of 10 seeds gives an exact sign
lower bound of **0.4931**, just under the 0.5 the other rule requires.

That leg is therefore **rule-dependent** and no single verdict is claimed for it.
Choosing the favourable rule now is exactly the freedom the registration was
meant to remove. The sibling protocol this design copies
(`REPRESENTATION_ARMS_ANALYSIS_V1.md`) uses the interval rule, so inheritance
points that way — but it was not written down, so both are reported.

Every other leg agrees under both rules.

## Against ESM-IF1, leg by leg, same cohort

| leg | ESM-IF1 | ProteinMPNN |
|---|---|---|
| aligned − shuffled | +42.4%, 10/10 | +16.3%, 10/10 |
| aligned − random | +26.1%, 9/10 | +6.7%, 8/10 |
| aligned − esm2 | **+22.8%, 10/10** | **+1.6%, undecided** |
| random − esm2 | −2.6%, undecided | −4.8%, undecided |
| shuffled − esm2 | **−13.8%** | **−12.6%** |

**Every leg replicates in direction and the ordering replicates exactly.**

**The harm replicates in magnitude too**: −12.6% against −13.8%, and 0 of 10
seeds against 1 of 10. Displacing the rows of a second, architecturally unrelated
structural encoder costs almost exactly what displacing ESM-IF1's costs.

**The benefit does not.** Aligned ProteinMPNN beats its own width-matched control
by 6.7% where ESM-IF1 beats its by 26.1%, and against plain ESM-2 the ProteinMPNN
gain is +1.6% with an interval containing zero. Four to six times smaller on
every benefit leg.

## What this establishes

The phenomenon is not an ESM-IF1 artefact. A second structural encoder, sharing
the inverse-folding task and nothing else about how it solves it, reproduces the
whole ordering including the undecided middle.

**The asymmetry is the finding.** Correct correspondence is not sufficient to
make a frozen representation useful — ProteinMPNN aligned is not detectably
better than no structural features at all. Destroyed correspondence is sufficient
to make it harmful, by an amount that barely differs between encoders.

Three representations now show the harm: ESM-IF1, AntiBERTy, ProteinMPNN. One
shows a large benefit.

## What it does not establish

Not why. The two encoders share a task and differ in architecture, width and
input handling, and nothing here isolates which difference produces a benefit
four to six times smaller.

Not that the benefit is absent for ProteinMPNN. `aligned − esm2` is undecided at
ten seeds with 7 of 10 favourable and an interval reaching +0.0020 — a detection
failure, not a demonstrated absence. And under the interval rule, aligned does
beat its own width-matched control.

Not a family. Two structural encoders are two structural encoders.

## The reconstructed oxygen

ProteinMPNN featurises four backbone atoms and the store holds three, so the
carbonyl oxygen was reconstructed by ideal geometry. This was registered in
advance as a caveat on cross-encoder magnitude comparisons, and it is **not**
used to explain the smaller benefit away.

It also does not fit the data. The same features, with the same reconstructed
oxygen, produce a decided harm when their rows are displaced — 0 of 10 seeds
above ESM-2. Features degraded to the point of carrying nothing would be inert
under displacement, not harmful.

## Scope

Validation only. No test split was read for this. One cohort, one architecture,
one weighting.
