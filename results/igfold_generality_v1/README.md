# A sequence representation shows the harm but not the benefit

> **Corrected 2026-09-11. This arm is not a structural encoder.** It is
> AntiBERTy — a sequence model with no coordinates — read from AsEP's cached
> interim graphs under the key `igfold`. Verified by running AntiBERTy and
> matching tensors: `results/feature_provenance_v1/`. This project never ran
> IgFold. The numbers below stand; every claim about "a second structural
> encoder" is withdrawn, and no such test was ever performed.
>
> The arm also differs from the ESM-IF1 arm in cohort (AsEP against sabdab2), in
> coverage (antibody side only against both), and in whether the features were
> extracted here or copied from a cache. Differences between the two arms are
> not attributable to the encoder.

Status: six paired seeds, AsEP `epitope_group` validation, antibody side only.
This was run believing it tested whether the ESM-IF1 attribution generalises
across structural encoders. It did not test that. What it does show is that a
frozen *sequence* representation shows the harm from misalignment without
showing a benefit from alignment.

Date: 2026-09-10

Machine-readable: [`report.json`](report.json).

## Why it was run

`results/representation_attribution_v1/` localises ESM-IF1's contribution to the
residue correspondence: aligned beats a width-matched random projection by 26.1%
at 9/10 seeds, while the same vectors permuted within the complex are worse than
no features at all.

One encoder cannot say whether that is how structural representations transfer
in general or a property of ESM-IF1. IgFold is a second structural encoder from
a different family — an antibody structure predictor rather than an inverse
folding model — and AsEP already ships its per-residue embeddings, so the same
three arms could be run without extracting anything.

## Result

| arm | node AUPRC | node MCC | pair AUPRC |
|---|---:|---:|---:|
| aligned | 0.345653 | 0.167767 | 0.054666 |
| shuffled | 0.316491 | 0.152446 | 0.042001 |
| random | 0.345472 | 0.163194 | 0.052633 |

| comparison | node AUPRC | 95% CI | seeds | verdict |
|---|---:|---|---:|---|
| **aligned vs random** | **+0.1%** | [−0.0073, +0.0082] | **2/6** | **no decision** |
| aligned vs shuffled | +9.2% | [+0.0022, +0.0621] | 3/6 | decided, barely |
| shuffled vs random | −8.4% | [−0.0568, −0.0039] | 1/6 | decided |

## What this does and does not say

**The generalised claim is withdrawn.** "Residue-level correspondence is how
pretrained structural representations transfer" is not supported. It holds for
ESM-IF1 and does not hold here.

**But the failure is not a counterexample to alignment.** IgFold's aligned
features do not beat a random projection at all — +0.1%, interval containing
zero, two seeds of six. This encoder contributes nothing over noise of the same
width in this setting, so **the alignment question does not arise for it**: one
cannot test whether the correspondence matters in features that carry nothing
either way. Writing this up as "alignment turned out not to matter" would
misdescribe it.

**One half does replicate.** Misaligned real structural features are worse than
a random projection, −8.4%, decided — the same direction ESM-IF1 shows. So:

> Misaligned structural features are harmful in both encoders. Aligned features
> are useful in one of the two.

**`aligned vs shuffled` is reported but not leaned on.** Its interval excludes
zero while only three seeds of six favour it, so the mean is carried by
magnitude rather than consistency. At six seeds that is weak.

## Why IgFold may contribute nothing here

Not established, and offered as the obvious candidate rather than a conclusion:
IgFold covers the **antibody side alone**, and AsEP's antibody side is CDR
residues only — about 60 against a full variable region's 220. ESM-IF1 covers
both sides. The comparison is therefore not encoder-against-encoder at equal
reach, and a fair generality test would need a second encoder covering the
antigen.

## Limits

1. Six seeds, against ten for the ESM-IF1 arms.
2. Antibody side only, which is confounded with the encoder identity.
3. Validation split, AsEP cohort. The ESM-IF1 attribution was measured on
   sabdab2, so this is also a different cohort — a second difference between the
   two experiments that this design does not separate.
