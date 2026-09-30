# Correspondence corruption: a severity curve, not a switch

Registered 2026-09-10, **before the perturbation was implemented**.

## What is missing from the current evidence

`results/representation_attribution_v1/` compares two extremes: perfectly
aligned features, and features permuted globally within the complex. The obvious
question it cannot answer is whether the damage is a cliff or a slope — whether
being slightly wrong about which residue is which costs a little or everything.

A single ablation says misalignment is harmful. A curve says how harmful, as a
function of how misaligned, which is a quantitative statement about the same
phenomenon and a much harder one to dismiss as an artefact of one extreme
setting.

## The perturbation

At **evaluation only**, on models already trained with correctly aligned
features. Nothing is retrained. The structural representation's rows are shifted
cyclically by `k` positions relative to the residues they describe:

    s_i  ->  s_{(i + k) mod L}

with `k` drawn from a fixed ladder:

    k in {0, 1, 2, 4, 8, 16}  plus  global permutation

`k = 0` is the unperturbed model and reproduces the recorded figure exactly,
which is the control that the perturbation code is wired in correctly. The
global permutation is the existing shuffle and anchors the far end.

A cyclic shift is used rather than a random local permutation because it has one
interpretable parameter — every residue is displaced by exactly `k` — and
because it preserves the multiset of vectors exactly at every `k`, so the only
thing varying along the ladder is displacement.

## Predictions, fixed now

**If the damage is a slope**: performance falls monotonically with `k` and
approaches the global-permutation value from above. That would establish
correspondence fidelity as a graded quantity governing transfer, and the
headline becomes a dose-response law rather than an on/off ablation.

**If the damage is a cliff**: `k = 1` already costs most of what the global
permutation costs. That would say the representation is locally sharp — a
one-residue error is nearly as bad as no correspondence at all — which is a
different and equally reportable finding.

**If small `k` costs nothing**: correspondence is only needed at a coarse scale,
and the aligned advantage survives displacement of a few residues. That would
weaken the sharpness of the claim while keeping its direction.

All three are reported. No ladder point is added or removed after seeing the
curve.

## Scope

Run on both encoders that have trained aligned arms:

| cohort | encoder | checkpoints |
|---|---|---|
| sabdab2 | ESM-IF1 | `esm_if1_only`, 10 seeds |
| AsEP `epitope_group` | IgFold | `igfold_aligned`, 6 seeds |

Validation splits only. No test split is read for this at all.

The IgFold arm shows no aligned advantage over a random projection
(`results/igfold_generality_v1/`), so its curve is expected to be flat in the
useful region and is run as a control on that expectation rather than in hope of
a second slope. A flat IgFold curve beside a sloped ESM-IF1 curve is itself the
cleanest statement of "alignment matters where the encoder carries something".

---

## Amendment, 2026-09-11: readout and decision rule

Registered after the ESM-IF1 jobs finished writing and **before any of their
evaluations were opened**. The original text fixed the ladder and the three
predictions but never said which number reads them, which left the shape of the
curve decidable after the fact. That gap is closed here rather than excused.

### Endpoint

Macro pair `auprc`, the endpoint everywhere else in this project
(`REPRESENTATION_ARMS_ANALYSIS_V1.md`). `auroc`, `mcc`, `f1` and
`precision_at_true_contact_count` are reported beside it for every ladder point.
Macro, not micro: per-complex then averaged. No metric is promoted afterwards.

### Pairing

Every ladder point is compared to `k = 0` **on the same seed and the same
checkpoint** -- the perturbation is applied at evaluation to one trained model,
so the paired difference removes seed variance entirely:

    d_k(seed) = auprc(k, seed) - auprc(0, seed)

Reported per `k`: the mean of `d_k`, its percentile bootstrap 95% interval, and
the Clopper-Pearson one-sided lower bound on the fraction of seeds with
`d_k < 0`. A ladder point counts as **damaged** when that bound exceeds 0.5.

### Which of the three predictions is recorded

Let `D = |mean d_global|` be the damage at the far end. Decided on ESM-IF1:

**cliff** -- `|mean d_1| >= 0.8 D` and `k = 1` is damaged.

**slope** -- `k = 1` is damaged, `|mean d_1| < 0.5 D`, and the sequence
`mean d_1 ... mean d_16` is non-increasing. Non-increasing is judged on point
estimates; a violation whose bootstrap interval covers zero is noted as noise
rather than allowed to break the classification, and any violation is printed.

**coarse-only** -- `k = 1` is not damaged.

`0.5 D` and `0.8 D` leave a band between them that matches none of the three.
That band is a real outcome and is reported as **intermediate**, not resolved by
moving a boundary. Nothing here is conditioned on the IgFold curve, which is a
control on the expectation of flatness and is classified by the same rule but
reported separately.

### Scope correction

The submitting script derived cohort, `k` and seed from the array index, and the
IgFold branch received six task slots for the thirty-six it required, so its `k`
was pinned to the first ladder entry and only `k = 1` ran. It also covered six of
the ten ESM-IF1 seeds this protocol registered. The missing 30 IgFold points and
24 ESM-IF1 points are being run now from an explicit task list. The ESM-IF1 curve
is analysed at all ten registered seeds; if any point is still missing at
analysis time, the seed count actually used is printed beside every figure.

---

## Extension, 2026-09-11: is correspondence special to structural features?

Registered **before the AntiBERTy ladder was submitted**. Its aligned arm is
already trained (`antiberty_only`, ten seeds) and has never been perturbed.

Everything this protocol has measured so far used a **structural** encoder:
ESM-IF1 from backbone coordinates, IgFold from predicted antibody structure. The
finding is stated as being about structured transfer. But nothing in the
perturbation is structural — it displaces rows of a per-residue feature matrix,
and any per-residue representation has rows to displace.

So the claim as it stands has an untested boundary. AntiBERTy is a **sequence**
model over antibody repertoires. It produces one vector per residue, it is used
frozen in exactly the same way, and it carries no coordinates at all.

### What each outcome would mean

**AntiBERTy degrades comparably** — correspondence fidelity governs the use of
any frozen per-residue representation, not structural ones specifically. The
paper's framing narrows: the phenomenon is about **residue-indexed features**,
and "structured transfer" overstates it. This is the outcome that costs the
current framing the most, which is why it is worth running.

**AntiBERTy degrades much less** — the sensitivity is specific to
representations that encode spatial relations, and the structural framing is
earned rather than assumed.

**AntiBERTy degrades more** — unanticipated, reported as it stands, and the
framing question reopens rather than resolving.

No threshold separates "comparably" from "much less" in advance, because the
two curves are on different arms with different baselines and any cut would be
arbitrary. What is registered instead is the comparison that will be shown: each
encoder's damage **as a fraction of its own global-permutation damage**, at every
ladder point, so the shapes are compared rather than the magnitudes.

### Design

The same ladder, the same endpoint, the same pairing: `k ∈ {0, 1, 2, 4, 8, 16}`
plus global permutation, ten seeds, sabdab2 validation, evaluation only, macro
pair AUPRC paired to `k = 0` within seed.

`k = 0` is re-evaluated rather than reused from the existing run, because these
evaluations run on the cu128 toolchain and the existing baseline was produced on
cu121. Pairing a perturbed evaluation against a baseline from a different stack
would put a toolchain difference inside the quantity being measured. Both sides
of every pair come from the same stack, so the within-encoder curve is clean;
the cross-encoder comparison is of curve shapes, which is what is registered.
