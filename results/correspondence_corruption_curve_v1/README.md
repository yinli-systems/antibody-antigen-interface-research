# How much is residue correspondence worth?

Registered in `protocols/CORRESPONDENCE_CORRUPTION_CURVE_V1.md`, with the
endpoint and decision rule fixed in its 2026-09-11 amendment before any
evaluation was opened. Validation splits only; no test split was read.

> **What `k` displaces, measured rather than assumed.** `k` shifts rows of the
> representation array. Whether that is a shift of `k` residues along the chain
> depends on whether consecutive rows are consecutive residues, and the answer is
> **different for the two cohorts**. It was checked geometrically: a peptide bond
> fixes the Cα–Cα distance of adjacent residues near 3.8 Å, so a dropped residue
> shows up as a jump.
>
> | | non-junction chain breaks | Cα–Cα p99 |
> |---|---|---|
> | **sabdab2** (the ESM-IF1 ladder) | 0.43% antibody, 0.14% antigen | 3.885 Å |
> | **AsEP** (the AntiBERTy ladder) | 7.98% antibody, 7.03% antigen | 24.9 Å |
>
> The sabdab2 arrays are full VH+VL and full antigen chains and are essentially
> contiguous. Counting the two things that do break contiguity there — the VH/VL
> junction and the cyclic wrap — **an 8-row shift is an 8-residue shift within one
> chain for 90.2% of antibody positions and 97.0% of antigen positions**, and
> both exceptions are properties of the perturbation rather than of the data.
>
> The AsEP arrays are interface-selected: consecutive rows jump segments at 8% of
> steps, with a 99th-percentile step of 24.9 Å. For that ladder `k` is an
> index-space quantity and carries no residue reading.
>
> Reproduce with `scripts/audit_row_to_residue_mapping.py`.

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

**Damage saturates by about eight rows of displacement — eight residues on
sabdab2, for the ~90% of positions where the two coincide.** On both encoders, `k = 8`, `k = 16` and the
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


---

## AntiBERTy on sabdab2, extracted here, both chains, 10 seeds

Added 2026-09-11, registered before submission. This removes every confound
between the two earlier ladders. Same cohort as the ESM-IF1 ladder, features
this project extracted rather than copied, both antibody chains, the same
protocol, the same endpoint, ten seeds. `k = 0` was re-evaluated on the same
toolchain as the perturbed arms rather than reused.

| k | mean Δ AUPRC | 95% CI | damaged |
|---|---|---|---|
| 1 | −0.00273 | [−0.00302, −0.00248] | 10/10 |
| 2 | −0.00281 | [−0.00320, −0.00243] | 10/10 |
| 4 | −0.00415 | [−0.00474, −0.00356] | 10/10 |
| 8 | −0.00443 | [−0.00504, −0.00382] | 10/10 |
| 16 | −0.00444 | [−0.00513, −0.00376] | 10/10 |
| global | −0.00614 | [−0.00657, −0.00574] | 10/10 |

**Verdict: slope.**

### The shapes differ, and that is the result

Each ladder as a fraction of its own global-permutation damage, which is the
comparison registered in the extension because the arms have different baselines:

| k | ESM-IF1 (structure, sabdab2) | AntiBERTy (sequence, sabdab2) |
|---|---|---|
| 1 | 52% | 44% |
| 2 | 59% | 46% |
| 4 | 88% | 68% |
| 8 | **94%** | 72% |
| 16 | 90% | 72% |
| global | 100% | 100% |

**Displacement harms a sequence representation too.** Ten of ten seeds at every
ladder point. So the harm from destroyed correspondence is a property of frozen
residue-indexed features, not of structural ones — which is the outcome the
extension named as costing the current framing the most, and it is what happened.

**The structural ladder saturates and the sequence ladder does not.** ESM-IF1 is
at 94% of its total damage by eight rows and flat thereafter; AntiBERTy is at
72% and still climbing to the global permutation. Whatever ESM-IF1 carries that
correspondence unlocks is exhausted by a local displacement; what AntiBERTy
carries degrades gradually and is only fully destroyed by a global one.

That asymmetry is stated as an observation about these two encoders on this
cohort. It is not a mechanism, and two encoders are not a family.

### What this does to the earlier AntiBERTy ladder

`k = 1` costs 44% of the global damage in both AntiBERTy ladders — the AsEP one
read from the cache under the key `igfold`, and this one extracted here on
sabdab2 with both chains. Two cohorts, two extractions, two coverages, the same
figure. The earlier ladder's numbers survive the relabelling.
