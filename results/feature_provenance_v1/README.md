# The "IgFold" arm is AntiBERTy

Established 2026-09-11 by running the encoder and comparing tensors, not by
reading anyone's documentation.

## The claim that was wrong

This project reported an arm called `igfold` as **a second structural encoder**,
and used it to test whether the ESM-IF1 correspondence finding generalises
across structural encoders. It is not a structural encoder. It is AntiBERTy, a
sequence model, and the generality test across structural encoders was never
run.

## How it was established

`scripts/convert_igfold_representations.py` reads
`graph["embedding"]["ab"]["igfold"]` out of AsEP's own interim graphs. This
project never ran IgFold; the receipt has always recorded
`revision: asep-interim-graphs`. So "igfold" was a **cache key**, and what the
tensor holds was never verified.

`scripts/identify_cached_representation.py` verifies it. AntiBERTy is run here on
the full antibody domain (`ab_seq_full`, 224 residues), and every row of the
cached tensor is matched to its nearest row among the fresh ones. That needs no
knowledge of the CDR mask: if the cache is AntiBERTy of this domain, each cached
row must appear among AntiBERTy's rows for it.

The heavy/light boundary is not recorded and AntiBERTy encodes chains
separately, so the boundary is swept.

| instance | cached rows | best split | mean nearest-neighbour distance | cached row norm | ratio |
|---|---|---|---|---|---|
| 1a14_0P | 60 | 120 | 0.005626 | 22.310 | 2.5e-4 |
| 1adq_0P | 63 | 122 | 0.005473 | 22.360 | 2.4e-4 |

A ratio of 2.5e-4 is an identity. The splits, 120 and 122, are ordinary VH
lengths.

### The first attempt, and why it proved nothing

It fed AntiBERTy the manifest's antibody sequence — which is already
CDR-selected, 60 residues of 224 — and found distances of 27, comparable to the
row norm. That was not evidence against the identity. AntiBERTy is contextual, so
that compared AntiBERTy-of-a-fragment against AntiBERTy-of-a-domain. The test
was invalid and the correct one reverses its conclusion.

## What has to change

**Withdrawn:** "a second structural encoder does not replicate the alignment
finding." No second structural encoder was ever tested.

**What the arm actually shows:** a second *frozen per-residue representation* —
a sequence model, antibody side only, on a different cohort, with a different
metric emphasis — also loses performance when its rows are displaced. That is a
weaker statement about structure and a broader one about representations.

**The corruption curve's second ladder** is therefore a sequence-encoder ladder,
not a second structural one. Its numbers stand; its label was wrong.

**The paper's framing.** "Structured transfer" is not earned by two encoders when
one of them has no coordinates. The correspondence finding rests on **one**
structural encoder, ESM-IF1, and one sequence encoder as a boundary.

**Confounds that were being attributed to the encoder.** The two arms differ in
cohort (sabdab2 against AsEP), in coverage (both sides against antibody only),
and in the features' provenance (extracted here against copied from a cache).
Differences between them cannot be read as encoder differences.

## What is unaffected

The ESM-IF1 results, which were extracted by this project from backbone
coordinates and whose receipt records the model and revision. The four-arm
comparison, the corruption ladder for ESM-IF1, and the attribution chain are
untouched by this.

## Naming

The on-disk key stays `igfold`, because that is AsEP's key and renaming it would
break the stores and hide the history. Everything a reader sees is renamed. Where
the old name survives in code it is annotated with what the tensor is.
