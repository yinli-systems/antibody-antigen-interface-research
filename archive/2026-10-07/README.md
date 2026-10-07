# 7 October 2026 preservation and cleanup

[Repository layout](../../docs/REPOSITORY_LAYOUT.md) · [Branch snapshot](BRANCH_SNAPSHOTS.json) · [Pre-integration backups](PREINTEGRATION_BACKUPS.json) · [Removed caches](REMOVED_CACHES.json)

## Preservation before pruning

All 32 heads observed before final cleanup are ancestors of integrated main `d8bca436981ee37bcabbde79d47ff8fdf0246386`. The tag `archive/2026-10-07/integrated-before-cleanup` preserves that exact integration. A separate 31-tag pre-integration snapshot also preserves every original observed HEAD and tree digest. Original manifests, failed runs and source variants are not rewritten.

The snapshot is not itself a deletion receipt. Actual remote changes are recorded separately in `REMOTE_CLEANUP_RECEIPT.json` after execution. No unmerged or changed branch may be deleted based on this file alone.

## Restore a historical branch

Use a new branch name and the full SHA from `BRANCH_SNAPSHOTS.json`:

```bash
git fetch origin --tags
git switch -c restored/my-experiment FULL_COMMIT_SHA_FROM_SNAPSHOT
```

No history rewrite or force push is required. Commit-pinned source links in the results atlas remain valid even when redundant active branch names are removed.

## Working-tree cleanup

Two unused, generated CLIP embedding caches were removed from the active tree, not from history. Their exact hashes and source commits are in `REMOVED_CACHES.json`; matching metric reports and ID metadata remain. The standalone `455_check.txt` diagnostic was relocated without changing its bytes. The application workflow variants were consolidated into one inactive template under `docs/ci/`.

The visual-query cache test now requires its actual full-gallery metadata; its existing minimum of 20 matching queries is unchanged. Committed fixture captions remain checked by their separate test. This is a declared optional-data prerequisite, not a weaker accuracy threshold.

No training, inference, dataset download, GPU scheduling, private-data publication or Git-history purge was performed by the cleanup.
