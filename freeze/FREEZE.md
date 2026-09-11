# Evidence Freeze v1

Everything the paper will claim is pinned here. From this tag, **no new
scientific experiments**: paper writing, figure layout, typography, references
and anonymous packaging only.

## What is pinned

| file | contents |
|---|---|
| `MANIFEST.sha256` | every file under `results/`, `protocols/`, `scripts/`, `src/`, `reproduction/`, `paper/`, `tests/`, with its digest |
| `CLAIMS_LEDGER.json` | the ledger parsed from its own table, so the machine-readable form cannot drift from the prose |
| `PROVENANCE.json` | each feature store: model actually run, extraction layer, residue mapping, extracted here or copied, store digest |
| `REPRODUCTION_RECEIPT.json` | the four cold-reproduction receipts joined, with each step's final outcome |
| `FREEZE_INDEX.json` | the commit, the counts, and the digest of each of the above |

## The state being frozen

Eight analysis steps, **zero numeric mismatches**, recomputed from run
directories and configs in processes with no access to this project's prose. The
two heaviest recompute from prediction parquets and land on their frozen values
to 5e−18 and exactly. `reproduction/cold_v1/RESULT.md` records the three passes
it took and every failure along the way.

## What a freeze does not mean

It does not mean the numbers are right. It means they are **fixed and traceable**:
each is bound to a claim, a comparison, a cohort, a split, a threshold rule, a
seed set, a decision rule and a file, and a clean process rebuilds it.

Known limits are carried in the frozen documents rather than resolved by freezing
them: the ranking-term result is an intervention on the training procedure and
not an isolated gradient mechanism; the aligned-versus-random leg of the second
encoder is protocol-dependent because its protocol never named a decision rule;
the reconstructed oxygen is neither ruled out nor used as an excuse; two
inverse-folding encoders are not a family; and the published benchmark figures
come from a collaborator's audit and have not been re-read from the PDFs in this
project.

## Amendment procedure

If an implementation defect is found that would invalidate a frozen conclusion:

1. **Do not overwrite.** The frozen artifacts stay as they are.
2. Record the defect in `results/REPRODUCIBILITY_NOTES.md` with what it affects.
3. Mark the affected ledger rows **invalidated**, in place, with the reason.
4. Produce a new freeze, `evidence-freeze-v2`, with its own manifest, and state
   in it what changed and why.

A correction that cannot be seen in the history is not a correction. Nothing is
silently replaced, including by me.

## Not pinned, deliberately

Checkpoints, prediction parquets, extracted representations and backbones stay on
the cluster; they are large and regenerable, and their digests are recorded in
`PROVENANCE.json` and in each receipt. `.github/` remains untracked pending
credentials with `workflow` scope.
