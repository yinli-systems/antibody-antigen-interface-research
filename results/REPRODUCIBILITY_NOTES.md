# Defects found, and what they had in common

Not a contribution. Recorded because several were caught only by writing a test
before running an analysis that could be run once, and because they share a
shape worth stating.

## The silent inner join

`ratio_test_confirmation.py` forms a ten-seed ensemble by averaging each
residue's score across seeds. It merged the per-seed frames on
`(instance_id, res_idx)` with an inner join and then checked that each seed had
contributed a score column.

That check is satisfied when the seeds share **no rows at all**: every seed
still contributes a column. A run that indexed residues differently — a
different split manifest, a partially failed evaluation — would have been
averaged over whatever the intersection happened to be, and the ensemble figure
would have been computed from fewer complexes than it claimed, with nothing in
the output saying so.

The guard now compares the residue key sets directly, before any join, and names
which seed disagrees and by how many residues. Row counts would have been
sufficient — an inner join preserving every input's length implies the key sets
are equal — but only by an argument the next reader has to reconstruct, and the
failure it reports is a number rather than a diagnosis.

The regression test uses **partial** overlap. Fully disjoint seeds produce an
empty frame that almost any check would catch; partial intersection is the case
that corrupts silently, and it is the one worth testing.

Re-running the confirmation under the stricter guard reproduced every figure
exactly, so all ten seeds did share their residue index. The bug never fired.
That is the point: it would not have announced itself if it had.

## A copy of a source of truth, kept by hand

Three defects in this project were the same mistake:

| where | the copy | what it cost |
|---|---|---|
| `macro_metric_summary` | a hardcoded list of keys to exclude from averaging | a string field entered the numeric average and poisoned it; all ten replication seeds failed |
| the argument parser | a hardcoded statement of which flags each subcommand accepts | train-only flags were passed to `evaluate`; four trained seeds died after 57 minutes each |
| `--multitask-balancer` | a hardcoded tuple of valid choices | an implemented, working arm was rejected on a submitted job |

Each was a hand-maintained duplicate of something the code already knew. Each
went stale the moment the original changed. Each now derives from the original:
the metric summary filters by value **type**, the parser test builds its
namespace from the **real parser**, and the balancer choices come from the
**registry**.

## An index that truncated instead of failing

The corruption-curve submitter derived cohort, `k` and seed from a Slurm array
index. The IgFold branch was allocated six task slots for the thirty-six it
needed, so `K = KS[J/6]` was pinned to the first ladder entry: five of six
ladder points never ran, and every job exited zero.

Replaced by an explicit task list, written out and counted before submission,
with an index that names no line failing loudly.

## A measurement whose window did not match its phenomenon

The cosine between the node and contact-edge gradients read −0.059 in a smoke
run and +0.468 on full data. Both are correct. The smoke was capped at 40
training instances against a split of 1383, so it reported the mean over the
first 40 optimisation steps while the other reported the mean over 1383.

No code was wrong. The diagnostic configuration was mistaken for a small version
of the real one, and a prefix was compared against a whole. Two claims were
built on the difference and both were withdrawn.

## What generalises

The defects that mattered were not wrong formulas. They were **a duplicate
kept by hand, an index that truncated rather than failed, a join that dropped
rather than failed, and a window that did not match what it was measuring.**
Each is silent by construction, and each was caught by asserting an invariant
rather than by inspecting an output.
