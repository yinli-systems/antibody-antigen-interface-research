# Script index

The Python files in this directory are operator entry points. Reusable logic
belongs under `src/interaction_first/`; Slurm launch templates live under
`slurm/` or use the explicitly registered `submit_*.sbatch` launchers here.

## Script families

| Prefix or filename | Purpose |
|---|---|
| `sabdab2_*` | Dataset preparation, SPPIDER-seq, EpiFormer, retrieval, and interaction-first workflows |
| `sppider_seq_*` | SPPIDER-seq research, final figures, and shared utilities |
| `mipe_*`, `freeze_mipe_selections.py` | MIPE smoke, training, frozen checkpoint selection, provided-test evaluation, OOF calibration, interaction export, and bound-geometry diagnostic |
| `train_pairscale_*` | Development-only PairScale training and controls |
| `analyze_pairscale_*`, `audit_pairscale_*`, `compare_pairscale_*` | PairScale statistical analysis and gate evidence |
| `build_*`, `derive_*`, `materialize_*` | Authority-bound data, split, cache, and source construction |
| `audit_*`, `verify_*`, `reconcile_*` | Independent or replay verification |
| `freeze_*` | Create-only registration freezing |
| `run_v6_*` | V6 solver, export, and family-diagnostic controllers |
| `submit_v6_*` | Registered Slurm entry points for V6 evidence units |
| `frontier_*`, `distributed_*`, `nccl_*` | Distributed correctness and systems diagnostics |

## Safe execution order

1. Locate the governing document in [`protocols/`](../protocols/README.md).
2. Verify source, data, runtime, and registration hashes.
3. Run the relevant smoke or preflight command when the protocol permits it.
4. Use a create-only output directory for evidence-bearing runs.
5. Run the independent verifier before interpreting a result.
6. Keep outer/test access disabled until the governing receipts authorize it.

Scripts with similar names may belong to different historical protocol versions.
Do not select an entry point by filename alone; follow the registered source hash
or the protocol's explicit command.
