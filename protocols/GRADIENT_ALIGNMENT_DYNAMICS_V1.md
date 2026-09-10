# Is gradient alignment stationary over training, and invariant across cohorts?

Registered 2026-09-11, **before any trajectory was read**. The two numbers it
sets out to reconcile are already on disk and are quoted here; nothing else is.

## The contradiction

The same quantity — the cosine between the node gradient and the contact-edge
gradient on the shared trunk — has been measured twice and disagrees in sign:

| | value | conditions |
|---|---|---|
| `results/gradient_conflict_v1/` | **+0.70**, negative on 0 of 240 complexes | trained sabdab2 checkpoint |
| `results/objective_component_localisation_v1/` | **−0.059**, **−0.051** | AsEP, epochs 1 and 2 from initialisation |

An earlier attempt blamed the partner-ranking term, which was present in one
decomposition and absent from the other. Isolating that term did not close the
gap: with the ranking gradient separated out and shown to be 23-40× smaller in
norm, node against edge is still negative early in training. That attribution
was wrong and is withdrawn.

Two candidate explanations remain, and they are separable because they vary
independently across the runs now in flight:

- **Stage.** The +0.70 is from a converged model; the −0.059 is from epoch 1.
- **Cohort.** The +0.70 is from sabdab2; the −0.059 is from AsEP.

## The measurement

Twelve epochs, five seeds, each cohort, with the `measure` balancer — which
applies the ordinary summed gradient and only reports geometry, so the
trajectories are the ordinary training trajectories. Per epoch it records the
mean over steps of `cos(g_node, g_edge)` and the task gradient norms.

Five seeds because one trajectory per cohort cannot show that anything is
reproducible, which is the only reason to look.

Let `c_t` be the epoch-wise cosine. For each cohort:

    early = mean of c_t over the first third of epochs
    late  = mean of c_t over the last third
    delta = late - early,  paired within seed

Reported per cohort: mean `early`, mean `late`, mean `delta`, each with its
percentile bootstrap 95% interval, and the exact sign bound on the fraction of
seeds with `delta > 0`.

A cohort shows a **sign transition** when all three hold:

    early < 0 on a decided majority of seeds
    late  > 0 on a decided majority of seeds
    delta > 0 with its bootstrap interval excluding zero

"Decided" is the sign bound exceeding 0.5, as everywhere else in this project.
A cohort that satisfies the third but not the second has a **rise without a
transition**, which is reported as itself.

### The direct comparison

The final epoch of the **sabdab2** trajectories is the closest thing to the
setting of `gradient_conflict_v1`: a converged sabdab2 model. Its value is
compared to +0.70 explicitly. That comparison is the reconciliation, and it is
reported whether or not it succeeds.

## Outcomes, fixed now

**Both cohorts transition** — the disagreement is stage. Recorded as: gradient
conflict is transient and stage-dependent, and geometry measured only at a
converged checkpoint misses conflict that occurred during learning.

**sabdab2 transitions, AsEP does not** — the disagreement is cohort. Recorded
as: gradient geometry depends on the data distribution. This is the outcome that
bears on the group-versus-ratio story, since it would make optimisation geometry
itself a property of the split.

**Both, with different magnitudes or endpoints** — recorded as: gradient
alignment is neither stationary over training nor invariant across cohorts. The
framing stays at that level and is not sharpened into a mechanism.

**Neither transitions** — then neither stage nor cohort reconciles the two
numbers, and the disagreement is in the measurement rather than in the model.
`gradient_conflict.py` would then be the thing to re-examine, starting with what
differs that is not stage and not cohort: it evaluates per complex under
`no_grad`-free single instances at one checkpoint, without negative sampling,
and any of those could matter. In that case **no claim about training dynamics
is written at all** until the instruments agree.

## What is not claimed either way

Not that conflict causes the pair-supervision effect on node metrics. The cosine
is a description of the optimisation geometry, not a demonstration that anything
downstream follows from it. The causal question belongs to
`OBJECTIVE_COMPONENT_LOCALISATION_V1.md` part two, which deletes objective terms
and measures what changes.

The paper's optimisation claim stays frozen at: relational supervision is
necessary for pair prediction, and its effect on node learning depends on
training dynamics and weighting. Neither "loss-scale is the mechanism" nor
"gradient conflict is ruled out" may be written, because the +0.70 and the
−0.059 are both measurements of this model and the first one is what would have
licensed the second claim.

## Scope

Validation and training splits of the development cohorts. No test split is read
for this. Gradients are taken on the shared trunk, defined as every parameter
that is not a node head and not the pair decoder — the same definition
`gradient_conflict.py` uses, so the two remain comparable.
