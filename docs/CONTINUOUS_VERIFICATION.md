# Continuous verification

This repository governs its own source with content hashes and keeps evidence
append-only. Both properties were unenforced: nothing recomputed a registered
digest, and the test suite was red for so many unrelated reasons that a real
regression could not be seen. This document describes the two mechanisms that
close that gap and how to operate them.

## 1. Protocol source bindings

Frozen protocols under `protocols/` register the SHA-256 of the source they
govern. Because evidence is append-only, an older protocol legitimately keeps an
older digest -- a superseded registration is not a defect. The defect is
narrower and more serious:

> a protocol-governed file for which **no** registration matches the bytes
> currently in the tree.

Such a file has left its own governance without a supersession record. Every
receipt that cites the protocol inherits the gap, silently.

```bash
python3 scripts/verify_protocol_source_bindings.py
```

The check resolves two registration shapes -- a key path that spells the file
name, and a `sha256` sitting beside a `path` -- against every source root,
including installable subprojects such as `framecontact/`, whose manifests
record paths relative to themselves. It is dependency-free by design: it must
keep working when the scientific stack cannot be installed.

### The baseline

`protocols/source_binding_drift_baseline_v1.json` records the drift that already
existed when the check was introduced, so continuous integration can fail on
*new* drift without rewriting history. Each entry carries the commit that first
left the file unregistered and a classification:

| Classification | Meaning |
|---|---|
| `historical_source_tree_snapshot` | The governing protocol is a point-in-time inventory. Later edits drift from it by construction. |
| `unreconciled_governance_drift` | A live protocol no longer describes the code it governs. Needs a numbered supersession. |

Recording drift does not reconcile it. A baselined entry classified as
`unreconciled_governance_drift` still means the affected protocol may not be
cited as current evidence.

To resolve an entry, add a numbered supersession that registers the current
bytes and links the protocol it supersedes, then drop the entry -- the check
reports baseline entries that no longer drift, so a stale baseline is visible.
`--write-baseline` exists to record drift you have decided to carry; it
preserves the `classification`, `first_unregistered_in` and `note` annotations
of entries whose bytes have not changed.

## 2. Test portability

`tests/portability_registry_v1.json` records, per test, the precondition it
needs to run. `tests/conftest.py` applies them at collection.

A recorded precondition never disables a test. It is skipped exactly while the
precondition is unmet, and runs -- and must pass -- on any machine that meets
it, so a maintainer with the cluster evidence mounted gets the full suite.

| Reason | Handling |
|---|---|
| `requires_external_artifacts` | Skipped while the named evidence is absent. |
| `absent_from_repository` | Skipped while absent. The named file was never committed on any branch: commit it or retire the test. |
| `requires_pinned_environment` | Skipped unless the exact pin or executable is present. |
| `requires_process_isolation` | Skipped when another test has already imported the module the audit must prove absent. |
| `known_source_drift` | Non-strict `xfail`; the live alarm for a baseline entry. |
| `known_repository_defect` | Non-strict `xfail`; fails against tracked but inconsistent data. |

`xfail` is non-strict on purpose: the moment a defect is fixed the test reports
`XPASS`, which is visible in the summary and tells you to remove the entry.

An entry that matches no collected test raises during a full-suite run, so the
registry cannot rot as tests are renamed or deleted.

### Adding an entry

Prefer fixing the cause. When a test genuinely depends on evidence that policy
keeps outside Git, add an entry naming the exact paths:

```json
{
  "nodeid_prefix": "tests/test_example.py::test_reads_cluster_receipt",
  "reason": "requires_external_artifacts",
  "detail": "binds the 4-GPU parity receipt kept beside the repository",
  "requires": [{"path": "../outputs/interaction_first/research/parity.json"}]
}
```

Supported requirement forms are `path`, `executable`, `module_version` and
`absent_module`.

## 3. Continuous integration

`.github/workflows/ci.yml` runs three independent jobs:

1. **static** -- `compileall` over every source tree and `bash -n` over every
   shell and Slurm entry point.
2. **protocol-bindings** -- the binding check above, with no dependencies
   installed.
3. **tests** -- the suite on Python 3.10 and 3.12 with the exact solver
   backends pinned by `pyproject.toml`.

The binding job is separate so that a dependency-resolution failure can never
mask a governance failure.
