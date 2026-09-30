# Preservation and recovery

The cleanup changes navigation, presentation and audit tooling, not the survival
of the original experiments. **No original research branch was deleted,
rebased, force-pushed, or moved by this consolidation.**

## What is preserved

| Layer | Coverage | Evidence |
|---|---|---|
| GitHub archive tags | Original main plus all 16 original research branches | [Branch map](../docs/BRANCH_PROVENANCE.md) |
| Complete tracked snapshot trees | 55,630 entries across 17 trees; 4,526 distinct path/blob versions | [Full manifest](FULL_SNAPSHOT_MANIFEST_20260930.json) and [compressed per-file records](FULL_SNAPSHOT_TREES_20260930.jsonl.gz) |
| Branch-change catalogue | 1,782 differences and 747 indexed evidence-facing paths | [Original diff inventory](BRANCH_FILE_INVENTORY_20260930.json) and [clickable evidence directory](../results/ALL_BRANCHES.md) |
| Offline Git backup | All original snapshot tags and reachable history, restored into a separate bare repository and checked | [Restore verification](PRESERVATION_VERIFICATION_20260930.json) |
| Earlier front-page wording | Original main and first consolidation README bytes | [Original README](readmes/README_910b752e.md) and [first consolidation README](readmes/README_4fc0613f.md) |

Repeated files appear repeatedly in the 55,630-entry count; it is **not** a count
of unique experiments, complexes or scientific results. The 747-page/path index
is a navigation filter, not a claim that every branch-diff file was a result.
Unchanged files remain in the full tracked-tree manifest.

Archive tags are normal Git refs, not technically undeletable storage. The
independent bundle and the explicit tree identities provide an additional
recovery route. No raw evidence was deleted to improve presentation.

## Verify the GitHub and local snapshots

```bash
git fetch origin --tags
python3 tools/evidence/verify.py --history
python3 tools/evidence/verify.py --history --remote
```

The remote check is optional and explicit. A missing object in a shallow clone
is a missing-history condition, not evidence that the preserved source vanished.

## Offline restore

The bundle location, bytes, digest and restore-test result are recorded in the
[verification receipt](PRESERVATION_VERIFICATION_20260930.json). It is stored on
the authorised Mac, **not committed into Git**.

```bash
# Use the actual bundle path recorded in the receipt; choose a NEW destination.
git bundle verify /path/to/all-original-research-refs.bundle
git clone --mirror /path/to/all-original-research-refs.bundle restored.git
git --git-dir=restored.git fsck --full
```

To restore a specific experimental tree for inspection, use the [exact revision
helper](../docs/REPRODUCE_EXACT.md). Do not force-push or delete live branches as
part of a restore test.

## What this does not package

Untracked files, cluster datasets/weights/embeddings, external source-model
assets, jobs, GitHub discussions and account configuration are not part of a Git
bundle. The current task left those originals untouched. The archive receipt
records the LFS-pointer scan scope separately; it does not turn pointers into
model payloads.

Large cluster results remain at their original receipt-bound paths. A catalogue
link or hash is not equivalent to having copied every large tensor into GitHub.
No claim of complete off-machine backup of those non-Git assets is made here.
