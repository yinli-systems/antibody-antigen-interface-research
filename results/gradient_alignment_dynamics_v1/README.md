# The conflict is real, and it lasts about a hundred steps

Registered in `protocols/GRADIENT_ALIGNMENT_DYNAMICS_V1.md` and its two
amendments, which fixed the endpoint, the crossing rule and all four possible
verdicts before any trace was written.

**Partial: AsEP only.** The sabdab2 traces are still running, and whether this
is a property of training or of this cohort is not yet decided.

## What was contradictory

The cosine between the node and contact-edge gradients on the shared trunk read
**+0.70** at converged sabdab2 checkpoints and **−0.059** in a smoke run. Two
explanations for that were proposed and both were wrong: it was blamed on the
partner-ranking term, which turned out to be 23-40× too small to matter, and
then on training stage measured in epochs, which the epoch-wise trajectories
refuted by reading +0.468 in epoch one.

The smoke was capped at 40 training instances against a split of 1383. It was
never an epoch. It was the **first 40 optimisation steps**.

## The trace

One epoch, full data, three seeds, per-step geometry.

| seed | steps 1-10 | steps 1-40 | last 40 | negative steps | crossing |
|---|---|---|---|---|---|
| 1 | −0.041 | −0.069 | +0.579 | 78 / 1383 | step 104 |
| 2 | −0.130 | −0.075 | +0.616 | 91 / 1383 | step 110 |
| 3 | +0.012 | −0.007 | +0.580 | 53 / 1383 | step 77 |

**Verdict: transient.** Every seed opens with a negative mean cosine and crosses
to a durably positive one — a step after which the smoothed cosine never returns
below zero — inside the first 8% of the first epoch. The registered rule allows
up to 25%.

The two figures that appeared to contradict each other are both correct and both
incomplete. A 40-step window sits inside the transient. A converged checkpoint
sits four hundred thousand steps past it.

## The toolchain replication

The same six runs on a different stack: torch 2.11+cu128 on sm_120 against
2.4.1+cu121 on sm_89, sharing almost no kernels.

| | 4090 crossing | 5090 crossing |
|---|---|---|
| seed 1 | 104 | 104 |
| seed 2 | 110 | 110 |

The opening means agree to four decimals and the first twenty steps agree to
three. The late-epoch means drift apart slightly (+0.6156 against +0.6106),
which is what accumulated float32 divergence over 1383 steps looks like.

A sign this close to zero is exactly what a numerical artefact could
manufacture. Two toolchains agreeing on where it flips is the evidence that it
does not.

## What may be said

That gradient measurements taken at convergence can conceal conflict that
occurred during learning. On this cohort, the node and contact-edge objectives
oppose each other for roughly a hundred steps and agree strongly thereafter, and
a measurement at either end alone would report the opposite of the other.

## What may not

Not that the transient causes anything. It is a description of optimisation
geometry, and nothing here shows that a downstream quantity depends on it.

Not that it generalises: the sabdab2 traces are unread. If sabdab2 shows no
transient, this is a property of the cohort and not of training.

Not that this settles the mechanism behind the pair-supervision effect. The
`results/gradient_conflict_v1/` figure keeps its numbers and its narrowed scope —
node against contact edges, at convergence, on sabdab2 — and the claim that
gradient conflict is ruled out stays unavailable, since it was a claim about a
trajectory made from its endpoint.
