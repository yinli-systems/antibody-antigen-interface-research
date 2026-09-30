# Layer-sweep receipt adapter: predeclared schema-only repair

Observed before sweep completion: the original aggregator reads
`partner_retrieval.mrr`, but the exact production driver writes
`partner_retrieval.mean_reciprocal_rank`. The latter describes partner retrieval,
not residue-pair rank. This will raise KeyError even for a correct completed run.
The 36-task scientific design, seed set, validation authority, selection metric,
AP definition, layer tie rule and bootstrap remain unchanged.

Do not patch source_exact_3eb4293: active training tasks bind its aggregator
bytes in PLAN.json and verify them at exit. The new adapter imports that exact
SHA-bound aggregator in a separate process and explicitly replaces only its
metric_snapshot reader. It adds finite/range, candidate-count, checkpoint-hash,
split-hash, task-set and within-family parameter-parity guards. It fails on
incomplete tasks rather than choosing a layer from partial results.

A new analysis output directory is required; existing artifacts are not amended.
Before reporting success, all 36 terminal receipts, both checkpoints' 217-case
evaluations and source hashes must pass. The original complete-campaign AP,
crossed bootstrap and layer selector then execute without algorithm changes.
Adapter/source hashes are separately recorded, never attributed to the old
commit alone. This is not new training or an independent confirmation.

Research: scikit-learn documents AP as a recall-increment-weighted precision
sum, not trapezoidal PR-AUC. Keep the existing average_precision_score call and
candidate grid. Slurm afterok on an array waits for all tasks to exit zero; a
failed dependency cannot be repaired by merely requeueing an antecedent.
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
https://slurm.schedmd.com/sbatch.html
https://slurm.schedmd.com/job_array.html
