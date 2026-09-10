# Does the correspondence effect hold for a second structural encoder?

Registered 2026-09-11, **before any ProteinMPNN feature was extracted** and
before any arm was trained.

## Why this is the one experiment left worth adding

`results/representation_attribution_v1/` localises ESM-IF1's contribution to
residue correspondence: aligned beats a width-matched random projection by 26.1%,
shuffled is decidedly worse than plain ESM-2, and aligned beats shuffled by 42.4%
on ten paired seeds.

That is **one encoder**. The arm that was reported as a second structural encoder
turned out to be AntiBERTy read from AsEP's cache
(`results/feature_provenance_v1/`), so the generality of the finding across
structural encoders has never been tested. The single most available objection to
this work is that the phenomenon is an ESM-IF1 artefact, and no amount of further
work on ESM-IF1 answers it.

## The encoder

ProteinMPNN, vanilla weights `v_48_020`, encoder half only.

| | ESM-IF1 | ProteinMPNN |
|---|---|---|
| architecture | GVP-transformer | message passing on a k-NN backbone graph |
| training objective | inverse folding, autoregressive over sequence | inverse folding, autoregressive over sequence |
| width | 512 | 128 |
| conditioned on sequence at encode time | no | no (decoder not run) |

They share a task and share nothing else about how they solve it. That is the
right amount of independence: an encoder trained on a different task would
confound the encoder with the task.

**Matched to the existing arms**: same cohort (sabdab2), same manifest, same
backbone store, both chains, the same split, the same ten seeds, the same
endpoint. The only deliberate difference is the encoder.

**Not matched, and recorded**: ProteinMPNN featurises four backbone atoms and
the store holds three, so the carbonyl oxygen is reconstructed by ideal geometry.
Every arm receives the same reconstruction, so it cannot create a difference
between arms, but it could depress all of them and would then understate this
encoder relative to ESM-IF1. Any cross-encoder magnitude comparison carries that
caveat.

## Arms

Ten seeds each, sabdab2 validation, pair macro AUPRC, paired within seed.

| arm | input |
|---|---|
| `esm2` | ESM-2 alone — already trained, reused |
| `proteinmpnn_aligned` | ESM-2 + ProteinMPNN in register |
| `proteinmpnn_shuffled` | ESM-2 + ProteinMPNN, rows permuted within the complex |
| `proteinmpnn_random` | ESM-2 + a width-matched random projection |

The random control is regenerated at **128** dimensions to match ProteinMPNN.
Reusing the 512-dimensional control built for ESM-IF1 would compare arms of
different widths and different parameter counts.

## The prediction, fixed now

If the phenomenon is a property of structure-conditioned representations rather
than of ESM-IF1:

    shuffled < random ≈ esm2 < aligned

with `aligned vs shuffled` decided, and `shuffled vs esm2` decided in the
negative direction — the ordering already established for ESM-IF1.

## What each outcome means

**The ordering replicates** — the effect is not an ESM-IF1 artefact, and the
claim becomes one about structure-conditioned per-residue representations rather
than about one model.

**Aligned does not beat random, but shuffled is still worse** — the same
asymmetry AntiBERTy showed: correct correspondence is not sufficient to make a
representation useful, while destroyed correspondence is enough to make it
harmful. The generality claim stays ESM-IF1-specific and the harm claim widens.

**Nothing is decided in either direction** — reported as a failure to replicate,
with the caveat that a 128-wide encoder given a reconstructed oxygen is a weaker
instrument than a 512-wide one given measured atoms, and that a null here does
not establish the encoders differ.

All three are reported. The endpoint, the seed count and the arms are fixed
above and no arm is added after the numbers are seen.

## What this does not become

A family. Two encoders that replicate are two encoders. The phrase "across model
families" requires more than this and will not be written from it.
