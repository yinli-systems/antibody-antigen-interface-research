# Integration validation — 7 October 2026

[All branch progress](BRANCH_PROGRESS.md) · [Machine-readable software receipt](SOFTWARE_VALIDATION.json) · [Integration integrity](VERIFICATION.json)

## Completed checks

| Check | Actual result |
|---|---|
| Full canonical test collection, CPU | `2969 passed, 47 skipped, 36 xfailed, 14 warnings, 37 subtests passed in 106.67s (0:01:46)` |
| Execution-recovery software fixtures | `85 passed, 1 skipped, 13 subtests passed in 1.48s` |
| Previously failing runtime/contract tests | `61 passed in 3.65s` |
| Frozen and optional-stack compatibility tests | `54 passed in 6.66s` |
| Original frozen atlas tests | 14 passed |
| Deterministic atlas regeneration check | PASS, 415 generated products and 87 numeric bindings |
| Historical publication/preservation verifier | PASS: 59 original metric fields; 17 archived snapshots and 747 evidence paths intact |

The full suite's **47 skips and 36 expected failures remain explicit**. They are governed by the already-existing portability registry (missing external inputs/requirements and documented historical defects or frozen-source drift), not new exemptions created to make this merge green. Warnings are retained. Recovery/targeted/atlas suites overlap parts of the full check; do not sum them into independent coverage or scientific-replication counts.

## Failures encountered and fixed rather than hidden

The first collection attempt exposed missing optional solver dependencies and imports that worked only when a script directory happened to be on the Python path. Pinned solver versions were installed in an isolated Python 3.11 environment. Two imports were namespaced, and failed collection no longer triggers a misleading stale-registry internal error; underlying collection errors remain visible.

The first complete software run had 18 failures. A legacy pair-geometry change violated the registered pose and parameter-ownership contract. Restoring the already-registered canonical source fixed these regressions without changing model contracts or thresholds. A backend-registry test was updated to enumerate the fourth backend already supplied by the merged source; unknown-name rejection was not removed.

Frozen pretraining source hashes and optional multimodal enhancements could not share one file layout transparently. The optional three-module stack is preserved byte-for-byte in its own `multimodal_v1` package, while the frozen core and its original tests retain their registered versions. Optional-stack tests only change import namespace. Both stacks are tested, rather than deleting an enhancement or weakening the source-binding guard.

The strict-retrieval branches also define incompatible model classes. The independent matched-budget implementation is retained byte-for-byte as `contract_models.py`; its tests change only that namespace. Earlier alternate gate/geometry/execution files and conflict resolutions are preserved in the source-variant archive and merge ledger.

All failed/intermediate logs and JUnit reports are retained under [validation/](validation/), including the 18-failure run and the intermediate 3-failure/1-collection-error run before namespace separation. The latest full run is [full_tests_v4.log](validation/full_tests_v4.log), not either failed predecessor.

## Reproduction of software checks

```bash
# Python >=3.10; this verification used isolated Python3.11 and the recorded packages.
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q

python3 tools/evidence/test_atlas.py
python3 tools/evidence/build_atlas.py --check
python3 tools/evidence/verify.py --history
python3 tools/evidence/verify_integration.py
```

[Installed versions](validation/environment.txt) and source SHA-256 values are recorded. The integration check verifies every starting branch HEAD, original tree inventory, preserved source variants, exact optional-module bytes and local navigation. A fresh clone must include the pinned historical commits/tags for history checks; the verifier does not silently fetch or run experiments.

## Scope

No training, biological model inference, contact-label reconstruction, wet-lab work or GPU scheduling was performed. Synthetic tensor/model-contract fixtures test software behavior only. No current cluster state was queried. Merging a protocol does not mark its experiment complete; October4 manuscript-only claims retain their evidence-publication boundary.
