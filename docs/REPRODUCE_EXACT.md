# Reproduce the exact experiment, not a mixture of branches

> **7 October integration:** all starting branch histories are now included in main. Exact experimental reproduction still requires the pinned source commits below. Incompatible runtime variants are explicitly preserved; the two strict-retrieval implementations use separate namespaces. See [integration decisions](../results/integration_20261007/BRANCH_PROGRESS.md).

**Main is an evidence hub, not a claim that every historical implementation has
been merged and retested together.** The first consolidation brought results
and extensions into main while leaving the old core training implementation in
place. In particular, the old core lacks the newer composite-embedding adapter.
Do not run a copied extension against an arbitrary main checkout.

## Three distinct operations

| Operation | What you need | What a pass establishes |
|---|---|---|
| Verify reporting | Python standard library; Git history for snapshot checks | Displayed metrics match their committed sources; archived tree identities match |
| Recompute metrics | Matching analysis source plus original prediction files | Reported aggregate can be rebuilt from existing predictions |
| Reproduce training | Matching source, dependencies, input data, model weights and compute budget | A new training/evaluation experiment; not authorised merely by this guide |

```bash
python3 tools/evidence/verify.py
# Requires archived refs/history, but no datasets or GPU:
python3 tools/evidence/verify.py --history
# Optional remote-ref check, explicitly uses the network:
python3 tools/evidence/verify.py --history --remote
```

A publication check is not an ACM artifact-evaluation award, an external
replication, a model-quality certificate or proof that every old test passes.

## Recover a compatible source snapshot

```bash
git fetch origin --tags
python3 tools/evidence/checkout_experiment.py --list
python3 tools/evidence/checkout_experiment.py multimodel ../antibody-multimodel
```

The helper creates a **new detached worktree**, refuses an existing destination,
checks the commit and entrypoints, never switches this working tree, and never
submits a GPU job. Use `--dry-run` to print the command without creating anything.
If the object is missing in a shallow clone, fetch complete history first.

| Route | Source revision | Read first in that worktree |
|---|---|---|
| `representation` | `55cfd137aee2b3b749f244fcb76dd11d30dda2c5` | `reproduction/README.md` |
| `correspondence` | `14a385feaec24b3141331963ba71dda788a164e1` | `extensions/correspondence_v2/PROTOCOL.md` |
| `multimodel` | `28dffada7aeaf9e5969f87414e91529aef60f717` | `extensions/frontier_multimodel_v1/STRUCTURAL_PLM_EXECUTION_STATUS.md` |
| `layer-sweep-runtime` | `3eb4293464552199c66eff9b6add14d7ed75ae14` | `extensions/frontier_multimodel_v1/STRUCTURAL_PLM_TRAINING_PROTOCOL.md` |
| `reparank-sequence-runtime` | `a9af80bc71e92e3c007e6ac4549c68e415900a66` | `extensions/frontier_multimodel_v1/REPARANK_SAME_PROTOCOL_PROTOCOL.md` |
| `framecontact-handoff` | `54afd798700bc41ac1352065414d8687c9627871` | `remote_tools/aligned_test549_handoff_v1/README.md` |
| `legacy-main` | `910b752ecf3e1493b80a7605151048bf2cd3e69b` | `README.md` |

Other historical branches are available through the [complete branch directory](../results/ALL_BRANCHES.md)
and their fixed commit URLs. Worktree preparation is not permission to reopen
closed tests, rerun healthy arrays, or alter frozen plans.

## Large artifacts and environments

Most trained weights, raw structures, dense prediction grids and embedding
stores are **not distributed in this Git repository**. Their recorded cluster
paths/hashes and access prerequisites remain in the original protocols. A
receipt is a pointer to a measurement, not a bundled copy of those large files.

The RePaRank checkpoint run used Torch `2.4.1+cu121`, while its official package
specified a newer version. Strict state loading and canaries passed, but that
compatibility deviation remains a limitation, not an exact-environment replay.
The official baseline's training-accession overlap also remains unknown.

`freeze/MANIFEST.sha256` belongs to its historical evidence freeze. Its original
paths are not a promise that the curated main tree has the same source bytes.
Use the freeze's own tag/revision and provenance for its historical validator.
This consolidation does not rewrite frozen protocols or declare new test access.

## Update the presentation

```bash
# Fails rather than silently adopting altered source artifacts:
python3 tools/evidence/render.py --check
python3 tools/evidence/render.py
python3 tools/evidence/verify.py
python3 tools/evidence/test_publication.py
```

New results require a reviewed registry entry with source digest, exact JSON
selector, cohort, evidence class, aggregation and interpretation limits. Do not
edit a rounded headline to conceal a changed raw artifact.

## Execution recovery

`python3 tools/evidence/checkout_experiment.py execution-recovery ../antibody-execution-recovery` prepares the new exact source. Read its `extensions/execution_recovery_v1/README.md` before execution: live jobs use separately pinned snapshots, and the helper itself submits no work.
