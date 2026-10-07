"""Add current repository-integration status without relabelling historical jobs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def enhance(readme: str, operations: list[str], root: Path):
    path = root / 'results/integration_20261007/STATUS.json'
    status = json.loads(path.read_text())
    for record in status['bindings']:
        data = (root / record['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Progress source digest changed: ' + record['path'])
        value = json.loads(data)
        for key in record['selector']:
            value = value[key]
        if value != record['value']:
            raise ValueError('Progress value changed: ' + record['path'])
    values = [record['value'] for record in status['bindings']]
    tasks, arms, cases, fresh, reused = values
    if cases != fresh + reused:
        raise ValueError('Reused/fresh case accounting differs')
    nav = '[Results](results/INDEX.md) · [Reproduce](docs/REPRODUCE_EXACT.md) · [All branches](results/ALL_BRANCHES.md) · [Run status](results/operations/README.md) · [Preservation](archive/README.md)'
    updated_nav = '[Results atlas](results/atlas/README.md) · [All branch progress](results/integration_20261007/BRANCH_PROGRESS.md) · [Cohorts](results/atlas/COHORTS.md) · [Reproduce](docs/REPRODUCE_EXACT.md) · [Historical jobs](results/operations/README.md) · [Repository layout](docs/REPOSITORY_LAYOUT.md) · [Preservation](archive/2026-10-07/README.md)'
    if readme.count(nav) != 1:
        raise ValueError('Expected one original navigation line')
    readme = readme.replace(nav, updated_nav)
    start = readme.index('## Ongoing — separate from completed findings')
    end = readme.index('## Nothing discarded to make the front page cleaner', start)
    current = f'''## Integrated progress — 7 October 2026

**All {status['branch_count']} starting branch HEADs are included in the integrated Git history**, through {status['merge_count']} explicit three-way merges and their ancestors. Conflicting experimental variants remain preserved; source-specific protocols and checkpoints are not silently made interchangeable. [Every branch and resolution](results/integration_20261007/BRANCH_PROGRESS.md)

| Workstream | Latest committed record | Result boundary |
|---|---|---|
| Structural PLMs | **{tasks}/{tasks} tasks complete**, {arms} arms | Final decision **NO-PROMOTION** versus matched random-width controls. [Final readout](extensions/frontier_multimodel_v1/STRUCTURAL_PLM_FINAL_READOUT.md) |
| Correctly centered RePaRank structure | **{cases}/{cases} cases**, {reused} reused + {fresh} fresh | Complete matched-grid diagnostic, not a new model or blind comparison. [Receipt](extensions/frontier_multimodel_v1/evidence/reparank_centered_corrected_v1/evaluation.json) |
| Synthetic rescue and gating | Completed development records | Clean-input cost and calibration shift retained. [Readout](extensions/rescue_qualification_v2/GATED_RESCUE_READOUT.md) |
| Strict retrieval | Infrastructure and separate model-contract implementations integrated | Still protocol/readiness evidence, not a completed strict-model benchmark. [Source separation](extensions/strict_partner_retrieval_v1/README.md) |
| Newer EPIC campaigns | October 4 manuscript reports additional experiments | **Report-only** where matching terminal artifacts were not found in the inspected branch trees; evidence may exist off-Git. [Evidence gaps](results/atlas/REVIEW_FINDINGS.md) |
| AbCite | Separate multimodal evidence-retrieval application | Preserve task-specific protocols and negative holdout-LoRA result. [Application results](abcite/evals/latest.md) |

This is **committed repository progress, not live scheduler status**. Old 12/36 and pending job records remain dated historical snapshots rather than current completion claims. [Historical operations](results/operations/README.md)

Review-related branch names were neutralized before redundant branch references were archived. Their commit histories are unchanged; frozen records retain the names observed at the time. [Rename mapping](results/integration_20261007/BRANCH_RENAMES.json) · [Branch snapshots and restoration](archive/2026-10-07/README.md)

'''
    readme = readme[:start] + current + readme[end:]
    readme = readme.replace('**Main is the curated evidence hub.** Historical results and experimental extensions have different source commits; copying their results here did not merge every training implementation.', '**Main now integrates the branch histories and their evidence.** Historical experiments still have different source commits. Divergent runtimes are preserved as source variants or separate namespaces; a combined main checkout is not proof of historical reproduction.')
    operations = list(operations)
    operations[0] = '# Historical operational snapshot — not a live monitor'
    operations[1:1] = ['', '**Superseded for completion status.** This page preserves its original observation time. See the [integrated branch progress](../integration_20261007/BRANCH_PROGRESS.md) for later committed terminal records. No live scheduler state was checked.', '']
    operations = [x.replace('## Still unfinished', '## Unfinished at this historical snapshot') for x in operations]
    readme = readme.replace('Original branches, archive tags, failed experiments and superseded interpretations remain available.', 'Original branch commits, archive tags, failed experiments and superseded interpretations remain available. Redundant active branch references may be pruned only after ancestry and backup verification.')
    return readme, operations
