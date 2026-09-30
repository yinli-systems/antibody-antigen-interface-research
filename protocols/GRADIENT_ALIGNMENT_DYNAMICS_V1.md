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

---

## Amendment, 2026-09-11: the epoch is the wrong unit

Registered after the first five epochs of the AsEP trajectory appeared and
**before any early/late window was computed or any verdict taken**.

The epoch-wise cosine on full data is positive from the first epoch:

    AsEP     +0.468  +0.627  +0.632  +0.631  +0.640
    sabdab2  +0.528  +0.616

The −0.059 that motivated this protocol came from a smoke run with
`--max-train-instances 40`. AsEP's training split is **1383** instances. So that
figure is the mean over the first **40 optimisation steps**, and +0.468 is the
mean over **1383**. The two are not the same measurement at different stages;
one is a prefix of the other.

Both are consistent with a single account: the cosine is negative for the first
few dozen steps and positive thereafter. A 40-step window sees only the negative
part; a 1383-step window averages it into insignificance.

**Neither can demonstrate it**, which is the point. The criterion registered
above compares the first third of epochs to the last third, and a transient
confined to the first epoch is invisible to it at any number of seeds. The
criterion is not wrong; it answers a question about epochs, and the question
turned out to be about steps.

### What is added

`MeasuringBalancer` now keeps the per-step geometry, capped, and writes it to
`balancer_trace.json` beside the history. One entry per optimisation step in
order.

A dedicated run reads it: **one epoch**, full data, both cohorts, three seeds
each. One epoch suffices because the epoch mean is already positive, so whatever
happens, happens inside it.

### What each outcome means, fixed now

**The cosine starts negative and crosses within the first epoch** — the
disagreement between +0.70 and −0.059 is resolved as neither stage-across-epochs
nor cohort, but a transient of early optimisation. The reportable form is that
gradient agreement measured at convergence, or averaged over an epoch, conceals
conflict that occurred during learning. The crossing step is reported, per seed
and per cohort, with no claim that its value is universal.

**The cosine is positive from the first step** — then the smoke's negative mean
comes from something other than being early, and the candidates are the reduced
instance count itself and the truncated validation set. Both would be properties
of the smoke, not of training, and the honest report is that the −0.059 was an
artefact of a diagnostic configuration and never described the model. In that
case nothing about training dynamics is claimed and this protocol closes
without a finding.

**The cosine oscillates in sign without settling** — reported as that, with the
fraction of steps negative, and no crossing point claimed.

### What does not change

The epoch-level trajectories keep running and are still analysed by the
registered criterion, because whether the geometry differs between cohorts over
training is a separate question from whether there is an early transient. The
five-seeds-per-cohort design and the transition criterion stand as registered
for that question.

### Two refinements to the step criterion, before any trace existed

Both were found by testing the rule against constructed traces, and both make it
strictly more sensitive rather than more permissive.

**A short probe alongside the 40-step window.** `OPENING` was set to 40 because
that is what the smoke covered. But if the negative phase is briefer than 40
steps, the 40-step mean can already be positive and the run would be recorded as
never having started in conflict — missing exactly the transient the window was
chosen to find. A run now counts as opening negative if **either** the first 10
or the first 40 steps average below zero.

**A crossing must be early to count as a transient.** The crossing is the step
after which conflict never returns, so a run that goes negative again for a long
stretch and recovers near the end still has one — at the end. Classifying that
alongside a run which settles in the first few dozen steps would be wrong, so
`transient` now additionally requires every seed's crossing to fall within the
first quarter of the epoch. A universal but late crossing is reported as **late
crossing**, which is a different statement and not the one this protocol
predicted.
