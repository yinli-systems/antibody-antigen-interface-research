# Remote workstream bundles

This directory preserves source and launch artifacts exactly as they were
deployed to remote research campaigns. New work belongs in a descriptive
workstream directory, for example:

- `sabdab2_predicted_geometry_v1/` for the development-only predicted-geometry
  sensitivity campaign;
- `matched_local_screen_v2/` for matched-local candidate and screen tooling;
- `mipe_replay_v1/` for the MIPE replay and finalization launchers;
- `sabdab2_frozen_ensemble_test_v1/` for the single authorized historical
  measurement and its recovery tooling;
- `oas_paired_continuation_v1/` and
  `abpairfm_3b_stage_a0_4gpu_production_v1/` for pretraining workstreams.

Files already committed at the directory root are retained for provenance.
Moving them can invalidate path-bound receipts or deployed commands. Python
bytecode, local workspaces, raw logs, model weights, caches, and large
predictions do not belong here.
