# Repository map

This document is the stable navigation layer for the repository. It does not
replace protocol-specific runbooks or result receipts.

## Directory roles

| Path | Role | Expected contents |
|---|---|---|
| `src/interaction_first/` | Importable implementation | Models, metrics, split formulations, data contracts, auditors, and solver backends |
| `framecontact/` | Isolated FrameContact snapshot | Independently installable historical source, protocols, receipts, reports, and tests whose relative paths are hash-bound |
| `scripts/` | Operator entry points | Build, train, evaluate, audit, freeze, verify, and submit commands |
| `slurm/` | Scheduler templates | Cluster launchers and orchestration helpers |
| `native/` | Native acceleration | Audited CUDA/C++ kernels used by bounded preprocessing workstreams |
| `remote_tools/` | Immutable deployed workstreams | Source snapshots, launchers, patches, and receipts grouped by remote campaign |
| `protocols/` | Frozen research governance | Preregistrations, amendments, execution locks, registrations, attempt receipts, and split evidence |
| `results/` | Lightweight evidence | Final summaries, verification receipts, statistical analyses, and selected figures |
| `data_registry/` | Data provenance | Source manifests, license receipts, normalization contracts, and materialization receipts |
| `schemas/` | Machine-readable contracts | Portable content and result schemas |
| `docs/` | Human-readable guidance | Reproduction, pipeline, geometry, and repository navigation documentation |
| `figures/` | Lightweight presentation artifacts | Final figures retained in Git |
| `paper/` | Manuscript workspaces | Venue-neutral paper source, paper-local figures, and compact metric/reference audits |
| `research/` | Active research notes | Bounded exploratory analyses that are not frozen result claims |
| `patches/` | Upstream compatibility records | Explicit patches required by pinned third-party code |

Raw datasets, model weights, full embeddings, prediction tensors, third-party
source trees, and credentials do not belong in Git. The root `.gitignore` records
the corresponding local-only paths and extensions.

## Recommended reading order

1. Read the root [`README.md`](../README.md) for the current scientific status and
   claim boundary.
2. Use [`protocols/README.md`](../protocols/README.md) to locate the governing
   protocol before running a command.
3. Use [`scripts/README.md`](../scripts/README.md) to find the matching operator
   entry point.
4. Use [`results/README.md`](../results/README.md) to locate concise evidence
   summaries.
5. Verify data and runtime provenance through
   [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and `data_registry/`.

## Preservation rules

- Never rewrite or remove a receipt that has already been used as evidence.
- Publish corrections as a new version or a new numbered attempt.
- Preserve failures, timeouts, and superseded protocols alongside successful
  evidence.
- Do not deduplicate mirrored protocol/result files without a manifest-aware
  migration: validators may depend on exact paths and hashes.
- Do not relocate an already-published `remote_tools/` bundle merely to improve
  aesthetics: deployed launchers and receipts may bind its path. New campaign
  files should be grouped under one descriptive workstream directory.
- Do not treat ignored local artifacts as published evidence.
- Do not infer test authorization from the presence of evaluation code. The
  governing protocol and its receipts control access.

## Naming conventions

- `*_PROTOCOL_V*.md` contains human-readable governance.
- `*_v*.json` contains a machine-readable contract or policy.
- `*_registration_*.json` freezes inputs, sources, runtimes, and expected outputs.
- `*_attempt_*` records one immutable execution attempt.
- `*_receipt.json` or `receipt.json` records machine-verifiable evidence.
- `*_summary.json` is a convenience view and is not automatically authoritative.

When two files appear equivalent, use the protocol or receipt lineage to decide
which path is authoritative; filename recency alone is insufficient.
