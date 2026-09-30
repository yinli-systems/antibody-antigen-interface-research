# Rebuilding the core results

Every number in the paper comes from a JSON artifact produced by a script in
`scripts/`. This directory says which script, with which arguments, produces
which file, so a second person can rebuild the core tables without reading the
history or having an account on the cluster where the runs happened.

It does not ship the trained checkpoints or the prediction parquets, which are
large. It ships the aggregates and the commands that regenerate them from
predictions, and it states exactly what a rebuilder needs.

## What is here and what is not

| | |
|---|---|
| in the repository | every aggregate JSON, every protocol, every analysis script, the tests |
| **not** in the repository | checkpoints, prediction parquets, extracted representations, backbones |
| where those live | the compute cluster, under paths each aggregate records in its own receipt |

A rebuilder with the predictions can regenerate every aggregate. A rebuilder
without them can still check that each aggregate is internally consistent, that
the analysis scripts do what they claim on constructed inputs (`tests/`), and
that no number in the paper is absent from an artifact (`results/CLAIMS_LEDGER.md`).

## Environment

The analysis scripts need Python **3.10 or later** — the codebase uses
`zip(strict=)` and `X | None` — plus `numpy`, `scipy`, `pandas`, `pyarrow`.
`torch` is needed only for the scripts that read prediction tensors or run an
encoder. `scikit-learn` is used by tests as an independent reference for MCC and
rank correlation, never in the analyses themselves.

Running the suite on Python 3.9 produces a large number of collection errors and
failures that are the interpreter, not the code.

## The core rebuild

Each command writes the JSON named beside it. Paths marked `<runs>` are the
directories of per-seed predictions.

```bash
# 1. Margin over the predicted-marginal outer product (ledger row 1)
python3 scripts/aggregate_baseline_margin.py \
    --run-root <runs>/baseline_margin --json report.json

# 2-6. The four-arm attribution (ledger rows 2-6)
python3 scripts/aggregate_representation_ablation.py \
    --run-root <runs> --json report.json

# 7-8. Correspondence corruption ladders
python3 scripts/aggregate_corruption_curve.py \
    --aligned-root <runs>/esm_if1_only --corrupt-root <runs>/corrupt/esm_if1 \
    --label esm_if1 --json curve_esm_if1.json
python3 scripts/aggregate_corruption_curve.py \
    --aligned-root <runs>/antiberty_ladder/k0 --corrupt-root <runs>/antiberty_ladder \
    --label antiberty_sabdab2 --aligned-subdir '' --json curve_antiberty_sabdab2.json

# 9. Gradient dynamics: per-step geometry, then its classification and figure
python3 scripts/aggregate_step_trace.py \
    --cohort asep=<runs>/steptrace/asep_seed'*' \
    --cohort sabdab2=<runs>/steptrace/sabdab2_seed'*' --json step_trace_summary.json
python3 scripts/plot_gradient_trajectory.py \
    --trace asep=<runs>/steptrace/asep_seed'*'/balancer_trace.json \
    --trace sabdab2=<runs>/steptrace/sabdab2_seed'*'/balancer_trace.json \
    --output gradient_trajectory.png

# 12-13. Checkpoint-selection stages two and three
python3 scripts/aggregate_selection_metric_confirmation.py \
    --arm 128=<runs>/seltotal/d128 --arm 256=<runs>/seltotal/d256 --json stage_two.json
python3 scripts/ratio_test_confirmation.py \
    --run-root <runs>/ratio_seltotal/d256 --json ratio_test_confirmation.json

# Provenance of every frozen feature
python3 scripts/feature_provenance.py \
    --receipt esm_if1=<runs>/esm_if1_receipt.json ... --output feature_provenance.tsv

# Identity of a representation whose name is not evidence
python3 scripts/identify_cached_representation.py \
    <instances.parquet> <cached-representation-root> 2
```

## What fails loudly, and why that matters

The defects that cost this project the most were silent. Each is now an assertion
rather than a convention, and a rebuilder should see them fire on corrupted
input rather than get a plausible wrong answer:

| check | where | what it catches |
|---|---|---|
| seeds must share a residue key set before averaging | `ratio_test_confirmation.py` | an inner join silently averaging over an intersection |
| the three task losses must sum to the objective every step | training loop | a decomposition that does not reconstruct what is optimised |
| balancer choices come from the registry | argument parser | an implemented arm rejected because a literal list went stale |
| macro summaries filter by value type | `metrics.py` | a string field entering a numeric average |
| ladder jobs read an explicit task list | submission scripts | index arithmetic truncating a ladder to its first point |
| the figure imports its smoothing from the analysis | `plot_gradient_trajectory.py` | a caption and a table reporting different crossings |

## Reading order for a reviewer

1. `results/CLAIMS_LEDGER.md` — every paper number bound to its artifact.
2. `results/feature_provenance_v1/` — what each frozen representation actually is,
   including the one whose name was wrong and how that was established.
3. `results/EVIDENCE_INVENTORY.md` — what replicated, what was withdrawn, what may
   not yet be written.
4. `protocols/` — what was registered before each measurement, with amendments
   dated and the superseded text left in place rather than edited away.
5. `results/REPRODUCIBILITY_NOTES.md` — the defects found and what they had in
   common.
