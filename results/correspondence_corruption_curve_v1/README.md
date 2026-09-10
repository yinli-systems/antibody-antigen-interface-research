# How much is residue correspondence worth?

Registered in `protocols/CORRESPONDENCE_CORRUPTION_CURVE_V1.md`, with the
endpoint and decision rule fixed in its 2026-09-11 amendment before any
evaluation was opened. Validation splits only; no test split was read.

> **`k` is an index-space perturbation.** It displaces **rows of the
> representation array after cohort-specific residue selection**, not residues
> along the peptide chain. The selected rows are the modelled CDR or interface
> residues, so consecutive rows can be separated by gaps, by segment boundaries,
> or by the heavy/light junction. `k = 8` therefore means *eight rows*, and
> whether that is eight contiguous biological residues is unverified. Read the
> saturation below as a computational observation about correspondence, not as a
> length scale in ångström or in residues.

Trained models are left alone. At evaluation the representation's rows are
displaced cyclically by `k` positions against the rows they are aligned to. The base ESM-2 columns are untouched, so `k` moves correspondence and
nothing else — the multiset of structural vectors is identical at every `k`.

Endpoint: macro pair AUPRC, paired to `k = 0` on the same seed and checkpoint.

## ESM-IF1, sabdab2, 10 seeds

| k | mean Δ AUPRC | 95% CI | damaged |
|---|---|---|---|
| 1 | −0.0097 | [−0.0104, −0.0090] | 10/10 |
| 2 | −0.0110 | [−0.0120, −0.0102] | 10/10 |
| 4 | −0.0165 | [−0.0175, −0.0154] | 10/10 |
| 8 | −0.0175 | [−0.0188, −0.0163] | 10/10 |
| 16 | −0.0169 | [−0.0179, −0.0158] | 10/10 |
| global | −0.0187 | [−0.0199, −0.0176] | 10/10 |

**Verdict: intermediate.** A one-residue error costs 52% of what destroying
correspondence entirely costs — above the registered slope boundary of 50% and
below the cliff boundary of 80%. The protocol says that band is reported as
itself rather than resolved by moving a boundary, so it is.

## AntiBERTy (AsEP cache, key `igfold`), AsEP `epitope_group`, 6 seeds

> **Relabelled 2026-09-11.** This ladder was reported as IgFold, a structural
> encoder. It is AntiBERTy, a sequence model, verified by tensor identity in
> `results/feature_provenance_v1/`. The numbers are unchanged.

| k | mean Δ AUPRC | 95% CI | damaged |
|---|---|---|---|
| 1 | −0.0130 | [−0.0153, −0.0109] | 6/6 |
| 2 | −0.0150 | [−0.0176, −0.0124] | 6/6 |
| 4 | −0.0255 | [−0.0303, −0.0211] | 6/6 |
| 8 | −0.0300 | [−0.0357, −0.0249] | 6/6 |
| 16 | −0.0298 | [−0.0350, −0.0249] | 6/6 |
| global | −0.0297 | [−0.0345, −0.0253] | 6/6 |

**Verdict: slope.** `k = 1` costs 44% and the ladder is non-increasing once the
`k = 8 → 16` step is judged on the step itself, which covers zero.

## What is actually going on

**Damage saturates by about eight rows of displacement.** On both encoders, `k = 8`, `k = 16` and the
global permutation sit within a hair of each other. Being eight residues wrong
about which residue is which is, to this model, the same as having no
correspondence at all. The interpretable endpoints of the ladder are `k = 1`
through `k = 8`; past that it is a plateau.

**The plateau is not flat on ESM-IF1.** The `k = 8 → 16` step is +0.0007, and
the paired design makes even that consistent across seeds, so the rule calls it
a real reversal rather than noise. It is 4% of the total damage. It is not
wrap-around: the shortest chain in the split is 51 residues, so every chain's
effective displacement is exactly `k`, which was checked rather than assumed.

**The sequence representation is damaged more than ESM-IF1, not less.** The protocol expected a flat
IgFold curve, because `results/igfold_generality_v1/` found IgFold's aligned
features no better than a random projection. The prediction was wrong in an
informative direction: correspondence-corrupted IgFold is worse than random
IgFold, even though aligned IgFold is not better than random IgFold.

`shuffled < random ≈ aligned` for the sequence representation, against
`shuffled < random ≪ aligned` for ESM-IF1. So displacement harms a frozen
per-residue representation whether or not that representation carries structure,
while the benefit of correct alignment appears only for the structural one.

That is a statement about **residue-indexed features**, not about structural
ones. The two ladders also differ in cohort, in which chains they cover, and in
whether the features were extracted here or copied from a cache, so the contrast
between them is not attributable to the encoder alone. An independent AntiBERTy
ladder on sabdab2, with features this project extracted and both chains covered,
is running to remove those confounds.

## Claim boundary

This measures what correspondence fidelity is worth, not how well the model
performs. Nothing here is a benchmark number and nothing here touched a test
split.
