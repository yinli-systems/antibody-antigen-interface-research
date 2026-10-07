# Final repository cleanup — 7 October 2026

[Repository layout](../../docs/REPOSITORY_LAYOUT.md) · [Machine-readable validation](VALIDATION.json) · [Preservation and branch restoration](../../archive/2026-10-07/README.md)

Cleanup is based on integrated main `d8bca436981ee37bcabbde79d47ff8fdf0246386`; it does not overwrite the concurrently completed PR #22 integration.

## Changes

The two unused generated embedding caches were removed from the active tree, with exact restoration hashes. The one-line root diagnostic was moved into the archive. Two near-duplicate CI entrypoints became one inactive template. The frontend now has a package lock, and local TypeScript build state is ignored. Generated navigation is collapsed in code review without hiding handwritten evidence.

A fresh worktree exposed four declared archived variants and ten declared validation logs that prior broad ignore rules had omitted from Git. All fourteen files were restored byte-for-byte against their existing committed hashes; no output was fabricated. The cleanup verifier now checks both working-tree and Git-index bytes, so a locally present but uncommitted copy cannot pass.

The atlas compression differed only in gzip header byte 9 between Python versions. Canonicalizing that OS byte leaves the frozen decompressed inventory and original compressed artifact unchanged. Both Python 3.9 and 3.12 now reproduce it.

## Checks actually run

| Check | Result |
|---|---|
| Complete core CPU suite | 2,969 passed; 47 skipped; 36 existing expected failures; 37 subtests passed |
| AbCite fixture suite | 118 passed; 2 skipped |
| Execution-recovery fixtures | 85 passed; 1 skipped; 13 subtests passed |
| Atlas unit tests | 14 passed |
| Cleanup regression tests | 4 passed |
| Atlas regeneration | PASS on Python 3.9 and 3.12; 415 products, 87 metric bindings |
| Historical publication/preservation | PASS; original 17 snapshots and 747 evidence paths retained |
| Integrated history and preserved variants | PASS after exact restoration of omitted files |
| Frontend TypeScript/Vite build | PASS |

The AbCite full-gallery check is skipped only when that external metadata is absent; the minimum 20-query assertion is unchanged when it is available. The committed fixture captions are checked independently. Existing scientific/source-drift expected failures and missing-artifact prerequisites remain visible, not recategorized to manufacture a clean result.

Test suites overlap and subtests are not extra independent experiments. No model training, biological inference, new benchmark scoring, GPU scheduling, public-site deployment or Git-history rewriting occurred.

Actual branch pruning is recorded separately in `archive/2026-10-07/REMOTE_CLEANUP_RECEIPT.json` after the remote operations. The snapshot alone is not a pruning receipt.
