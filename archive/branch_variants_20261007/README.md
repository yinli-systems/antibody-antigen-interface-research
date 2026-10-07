# Exact source variants preserved during integration

Both sides of explicitly reviewed path conflicts are stored under their source-commit prefix, with original repository-relative paths. Source bytes are never edited in place. See [the resolution ledger](../../results/integration_20261007/MERGE_LEDGER.json) and [hash inventory](../../results/integration_20261007/PRESERVED_VARIANTS.json).

These files are provenance copies, not an alternative package to execute directly. The original parent commits preserve complete runnable source layouts. Main resolves incompatible defaults conservatively and retains separate namespaces where necessary. No historical branch is deleted merely because its commits have been merged.
