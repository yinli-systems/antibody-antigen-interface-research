# Repository map

[Results](../RESULTS.md) · [Integrated progress](../results/integration_20261007/BRANCH_PROGRESS.md) · [Cleanup and restore](../archive/2026-10-07/README.md)

| Location | Responsibility |
|---|---|
| `src/` and `scripts/` | Core modules and research entrypoints. Historical runs require their pinned revision. |
| `framecontact/` | FrameContact project, protocols and supporting evidence. |
| `extensions/` | Experimental programs, each with a local protocol and evidence boundary. |
| `abcite/` | Independent multimodal retrieval application; not an antibody-prediction benchmark. |
| `paper/` | Manuscripts, revisions and paper-level evidence. New prose is not a terminal experiment receipt. |
| `results/` | Canonical results, frozen atlas, integration and validation records. |
| `protocols/` | Frozen experimental contracts and source hashes; never rewrite them to manufacture a pass. |
| `archive/` | Original variants, branch snapshots, failure provenance and restoration maps. |
| `tools/evidence/` | Standard-library reporting, hash checks and exact-source checkout tools. |
| `docs/ci/` | Inactive CI template; no automatic model downloads introduced by cleanup. |

## What clean means here

One current navigation path, no transient caches or root-level one-line scratch logs, no duplicated workflow entrypoints, and only needed active branch references. It does not mean deleting negative results, flattening incompatible experimental variants, or making previously failing scientific gates look successful.

The `results/atlas/` 30-branch snapshot remains frozen. The integration report describes the subsequent 31 starting heads. The final branch snapshot records 32 heads before pruning. These are different observation times, not inconsistent experiment counts.

## Explicit implementation boundaries

`extensions/strict_partner_retrieval_v1/models.py` and `contract_models.py` preserve different model APIs. Optional multimodal source lives in `src/antibody_pretraining/multimodal_v1/`, separate from frozen core hashes. Source integration does not certify every runner against every protocol or authorize new model execution.

## Validation

Read the timestamped software validation records, including skips and expected failures. Missing external artifacts and historical source-drift alarms remain visible. Local CPU tests do not certify CUDA performance, biological claims or unseen cluster jobs.
