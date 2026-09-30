# Where the gain in a frozen structural representation actually sits

Status: development-only, ten paired seeds on the sabdab2 validation split, one
encoder. It localises the information; it does not establish that every
structural encoder behaves this way. A second encoder is being run for that.

Date: 2026-09-10

Machine-readable: [`report.json`](report.json).

## The question

Adding frozen ESM-IF1 features to the input is worth about 23% of pair macro
AUPRC. Three explanations survive that observation on its own:

1. the model gained **capacity** — 513 more input columns and the projection
   parameters that consume them;
2. the model gained **some structured signal** — any 512-dimensional vector
   field with the right statistics would do;
3. the model gained **residue-level structural information** — the vector at
   position *i* describes the local geometry of residue *i*.

Two controls separate them, and **they are matched to different things**. This
was checked against the run configs rather than taken from this file's earlier
wording, which claimed both were matched to the aligned arm. One is.

| arm | antibody / antigen width | parameters | added columns | what it removes |
|---|---|---|---|---|
| `esm_if1_only` (aligned) | 1793 / 1793 | 896,781 | real ESM-IF1 | — |
| `esm_if1_shuffled` | **1793 / 1793** | **896,781** | the real ESM-IF1 vectors, rows permuted per key | the residue correspondence; keeps the vectors, their distribution, the complex and its length |
| `random_control` | **2306** / 1793 | **963,471** | fixed random projection of the base embedding | all information; keeps width |

`esm_if1_shuffled` is matched to the aligned arm exactly — identical widths,
identical parameter count, the same tensors in a different row order.

`random_control` is **not**. It was built to mirror the three-encoder arm
`esm2_antiberty_esmif1`, whose widths and parameter count it matches to the
digit (963,471 against 963,471), and it is therefore **wider and larger than the
aligned arm it is compared to** — 513 more antibody columns and 66,690 more
parameters.

That mismatch runs in the conservative direction: the aligned arm beats a random
arm with **more** width and **more** parameters. So it does not weaken the
comparison, and the comparison may not be described as width-matched.

Audited by `scripts/audit_control_contract.py`.

## Result

Ten paired seeds, pair macro AUPRC:

| arm | mean | vs `esm2` | seeds | verdict |
|---|---:|---:|---:|---|
| `esm2` | 0.022460 | — | — | — |
| `random_control` | 0.021865 | −2.6% | 4/10 | **no decision** |
| **`esm_if1_shuffled`** | **0.019367** | **−13.8%** | **1/10** | **decided — worse** |
| `esm_if1_only` | 0.027571 | +22.8% | 10/10 | decided |

| comparison | difference | seeds |
|---|---:|---:|
| aligned vs shuffled | **+42.4%** | **10/10** |
| aligned vs random | +26.1% | 9/10 |

The ordering is

    shuffled  <  esm2  ≈  random  ≪  aligned

## What it means

**Capacity is not the explanation.** The comparison that establishes this is
`random_control` against `esm2`, where the parameter counts are 963,471 against
763,401 — two hundred thousand parameters of width carrying a linear function of
the model's own input leave the model where it was: interval containing zero,
four seeds of ten. Those two arms are the pair the width control was designed
for and its receipt records both counts.

The aligned-versus-random figure below (+26.1%) is a comparison **across**
widths, with the random arm the wider one, and it is reported as such.

**"Some structured signal" is not the explanation either, and the evidence is
stronger than a null.** Misaligned real features are **worse than no features at
all** — decided, with the interval entirely below zero and nine seeds of ten
falling. The model reaches for them and is misled. Vectors with exactly the
right distribution, from exactly the right complex, attached to the wrong
residues, actively cost 13.8%.

**What is left is the correspondence.** The same vectors, correctly assigned,
are worth +22.8% over the baseline and +42.4% over their own permutation.

The two controls are informative in different directions and that is the point:
an uninformative addition is harmless, a *mis*informative one is not.

## Limits

1. **One encoder, one cohort, one architecture.** This says where ESM-IF1's
   contribution sits on sabdab2. Whether structural representations transfer
   this way in general needs a second encoder, which is running.
2. **Within-complex permutation, not cross-complex.** Donating another
   complex's features is the more obvious control and is not well defined here:
   the features are per residue and complexes differ in length, so it would
   require truncation or padding. Permuting within the complex removes strictly
   less — it keeps even the complex-level statistics — which makes it the
   sharper test rather than the weaker one.
3. Validation split, development only. No test access is claimed.
4. The permutation is fixed per sequence key, not resampled per epoch. A fresh
   permutation each step would be noise injection, a different experiment.

## Provenance

| | |
|---|---|
| arms | `esm2`, `random_control`, `esm_if1_only`, `esm_if1_shuffled` |
| launcher | `slurm/representation_arms.sbatch` |
| shuffle | `--shuffle-representation-residues 20260911` |
| split | `double_remote_split.json`, sha256 `3da484ae…` |

---

## Where the variance is: seeds or complexes?

Added 2026-09-11. A margin averaged over seeds says the arm is reliably better
*given this validation set*. It does not say whether the advantage is spread
across complexes or carried by a few, and those imply different things for a new
antigen.

Aligned minus shuffled, ten seeds × 217 complexes, every cell present:

| resampling unit | sd | 95% interval |
|---|---:|---|
| seeds (10) | 0.002151 | [+0.006943, +0.009444] |
| **complexes (217)** | **0.015119** | [+0.006229, +0.010306] |

Overall mean paired difference +0.008204, the same figure the seed-level
comparison reports.

**The complex-level spread is seven times the seed-level spread.** Which seed was
trained matters far less than which complexes were drawn. The two intervals come
out comparable in width only because there are 217 complexes and 10 seeds.

**The advantage is broad, not carried by a tail.** 162 of 217 complexes favour
the aligned arm — 74.7%, exact sign lower bound 0.6933, which clears 0.5.

**But the mean overstates the typical complex.** Per-complex differences:

| p5 | p25 | p50 | p75 | p95 |
|---:|---:|---:|---:|---:|
| −0.0104 | −0.0003 | **+0.0043** | +0.0144 | +0.0321 |

The median is about half the mean, and the top ventile is four times it. A
quarter of complexes are no better or worse. So "+42.4% on average" is true of
the split and is not what to expect on an arbitrary new antigen; the median
complex gains about half that in absolute terms.

Both resampling units support the direction. Neither removes the fact that this
validation split informed model selection elsewhere in the project, and no
resampling can.
