#!/usr/bin/env python3
"""Render the evidence landing page from hashed measurement artifacts."""
import argparse
import json
from pathlib import Path
from core import ROOT, load_registry


def products(root=ROOT):
    root = Path(root)
    _, v = load_registry(root)
    manifest = json.loads((root/'archive/FULL_SNAPSHOT_MANIFEST_20260930.json').read_text())
    snapshot_path = sorted((root/'results/operations').glob('snapshot_*.json'))[-1]
    status = json.loads(snapshot_path.read_text())
    stamp = status['observed_at_utc'][:19] + ' UTC'
    fc = 'results/aligned_test549_replay_v1/RESULT.json'
    corr = 'extensions/correspondence_v2/READOUT.md'
    rr = 'extensions/frontier_multimodel_v1/evidence/reparank_same_protocol_v1/sequence_full/evaluation.json'
    ranking = 'extensions/frontier_multimodel_v1/SCORE_DISSOCIATION_READOUT.md'
    readme = f'''# Antibody–antigen interface research

**From interface localisation to residue-pair correspondence.**

FrameContact and controlled protein-representation experiments: contact maps, epitope/paratope prediction, partner retrieval, and the evidence needed to tell these capabilities apart.

[Results](results/INDEX.md) · [Reproduce](docs/REPRODUCE_EXACT.md) · [All branches](results/ALL_BRANCHES.md) · [Run status](results/operations/README.md) · [Preservation](archive/README.md)

## FrameContact: three endpoints, one documented evaluation

Sequence inputs plus independently predicted intra-chain geometry; **no experimental bound coordinates as model input**. On the checkpoint-aligned **549-case historical test**, fixed routed medians over seeds 17/29/43 are:

| Contact AP | Epitope AP | Paratope AP |
|---:|---:|---:|
| **{v['framecontact_contact_ap']:.5f}** | **{v['framecontact_epitope_ap']:.5f}** | **{v['framecontact_paratope_ap']:.5f}** |

Mean-per-complex AP, not pooled AP or accuracy. This is **exposed historical evidence**, not a fresh blind or proven sequence-double-cold test; some AF3 predictions use homolog MSAs. [Per-seed results]({fc}) · [455/549 split correction](paper/neurips_2026_framecontact/CHECKPOINT_TEST_ALIGNMENT_20260905.md)

## Controlled findings

| Question | Measured answer | Scope |
|---|---|---|
| Does aligned structural information transfer? | ESM-IF1 vs ESM-2: **+{v['rep_aligned_delta']:.6f} AP**, 95% CI [{v['rep_aligned_ci'][0]:.6f}, {v['rep_aligned_ci'][1]:.6f}]. Shuffled ESM-IF1: **{v['rep_shuffled_delta']:.6f} AP** vs ESM-2. | Ten paired training seeds; development data; intervals resample seeds. [Artifact](results/representation_attribution_v1/report.json) |
| Is the damage only an amino-acid-type mismatch? | Same-AA shuffling: **{v['corr_esm_if1_identity']:.6f} → {v['corr_esm_if1_same_aa']:.6f} AP**. Equal-component paired effect **{v['corr_esm_if1_delta']:.6f}**, CI [{v['corr_esm_if1_ci'][0]:.6f}, {v['corr_esm_if1_ci'][1]:.6f}]. | Three fixed seeds; 217 development complexes; crossed seed/component interval. [Readout]({corr}) |
| Does the correspondence contrast confirm on held-out data? | **One of two encoders confirms:** ProteinMPNN **+{v['holdout_ProteinMPNN_delta']:.6f} AP**; ESM-IF1 **+{v['holdout_ESM-IF1_delta']:.6f}**, interval includes zero. | One registered 217-case holdout read; now closed. Not a universal necessity claim. [Both results](results/holdout_confirmation_v1/README.md) |

![Frozen-feature transfer, four-arm development experiment](figures/evidence/representation_transfer.svg)

The random width control is **undecided**, not proven equivalent. The figure and headline metrics are derived from [hashed source artifacts](results/HEADLINE_REGISTRY.json); different seed sets and aggregation units are never pooled into a stronger-looking number.

**Objective ablation:** removing complex-ranking supervision improved both contact and node endpoints in the recorded factorial study. This is an intervention on the full training/selection procedure, not an isolated gradient mechanism. [Ablation and qualifications](results/objective_component_localisation_v1/PART_TWO.md)

## External baseline — measured here, not invented here

The **official RePaRank sequence checkpoint** scores **{v['reparank_macro_ap']:.6f} macro pair AP** and **{v['reparank_pooled_ap']:.6f} pooled pair AP** on **{v['reparank_pairs']:,} candidate residue pairs** ({v['reparank_positives']:,} contacts) from 217 validation complexes. Its epitope/paratope macro AP is **{v['reparank_epitope_ap']:.6f} / {v['reparank_paratope_ap']:.6f}**. [Raw measurement summary]({rr})

This is a pretrained-baseline diagnostic, **not an improvement produced by our model**. Checkpoint training overlap is unknown, and training budgets are not matched. Do not rank it against the 549-case table above.

## Evidence that limits the claims

The complete **122 × 119** partner grid ({v['ranking_unique_pair_count_per_task']:,} distinct pairs; {v['ranking_total_pair_evaluations']:,} repeated evaluations) **did not robustly reproduce** the earlier small-panel MRR increase under corruption. Unobserved pairings are unlabelled alternatives, not verified non-binders. [Full-grid counterevidence]({ranking})

On AsEP, ten-seed macro MCC is **{v['asep_group_mcc']:.4f}** on `epitope_group` and **{v['asep_ratio_mcc']:.4f}** on `epitope_ratio`; these are not benchmark-leading results. [Benchmark conventions](results/asep_benchmark_v1/BENCHMARK.md) · [Corrections and negative results](docs/CLAIM_BOUNDARIES.md)

## Reproduce without mixing source revisions

**Main is the curated evidence hub.** Historical results and experimental extensions have different source commits; copying their results here did not merge every training implementation.

```bash
# Standard-library reporting and preservation checks; no GPU or dataset required.
python3 tools/evidence/verify.py --history

# Prepare the exact experimental source in a NEW detached worktree.
python3 tools/evidence/checkout_experiment.py multimodel ../antibody-multimodel
```

[Exact-source routes and prerequisites](docs/REPRODUCE_EXACT.md) distinguish artifact verification, metric recomputation, and training reproduction. Large datasets, checkpoints, embedding stores and most raw prediction tensors remain outside Git.

## Ongoing — separate from completed findings

As of **{stamp}**, the frozen SaProt/ProSST sweep has **{len(status['completed'])}/36** formal tasks complete. No winning layer is selected before the matched controls finish. RePaRank's full structure arm remains incomplete; the first AF3 missing-chain job failed before inference. [Timestamped jobs and receipts](results/operations/README.md)

## Nothing discarded to make the front page cleaner

**{manifest['snapshot_count']} preserved pre-consolidation snapshots · {manifest['distinct_path_blob_versions']:,} distinct path/blob versions · {manifest['file_entries_including_repetition']:,} tracked-file entries including repetitions.**

Original branches, archive tags, failed experiments and superseded interpretations remain available. [747 evidence-path links](results/ALL_BRANCHES.md) · [Complete tracked-tree inventory](archive/FULL_SNAPSHOT_MANIFEST_20260930.json) · [Offline restore procedure](archive/README.md) · [Full result catalogue](results/INDEX.md)
'''
    # Quantitative vector chart. Four absolute means share a common zero origin.
    labels = [('ESM-2', 'rep_base'), ('ESM-2 + random width control', 'rep_random'), ('ESM-2 + aligned ESM-IF1', 'rep_aligned'), ('ESM-2 + shuffled ESM-IF1', 'rep_shuffled')]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="306" viewBox="0 0 960 306" role="img" aria-labelledby="title desc">', '<title id="title">Frozen-feature transfer on development validation</title>', '<desc id="desc">Mean per-complex contact AP averaged over the same ten training seeds. Values are not percentages or held-out performance. The horizontal axis starts at zero.</desc>', '<rect width="960" height="306" fill="white"/>', '<g font-family="sans-serif" fill="black">', '<text x="24" y="29" font-size="21" font-weight="bold">Frozen-feature transfer: what correct residue alignment adds</text>', '<text x="24" y="53" font-size="13">Development validation · mean-per-complex contact AP · same 10 seeds · paired effects in registry</text>']
    x0, scale = 310, 17000
    for i, (label,key) in enumerate(labels):
        y=84+i*42
        svg += [f'<text x="24" y="{y+16}" font-size="14">{label}</text>',f'<rect x="{x0}" y="{y}" width="{v[key]*scale:.3f}" height="23"/>',f'<text x="{x0+v[key]*scale+10:.3f}" y="{y+17}" font-size="14">{v[key]:.6f}</text>']
    svg.append(f'<path d="M{x0} 245 H{int(x0+.03*scale)}" fill="none" stroke="black"/>')
    for tick in [0,.01,.02,.03]:
        x=x0+tick*scale
        svg += [f'<path d="M{x} 245 v5" stroke="black"/>',f'<text x="{x}" y="268" font-size="13" text-anchor="middle">{tick:.2f}</text>']
    svg+=['<text x="24" y="296" font-size="12">Not a model leaderboard. No claim of random-control equivalence; see paired intervals and holdout qualifications.</text>','</g></svg>']
    ops=['# Operational snapshot — not a live monitor','',f'Observed: **{stamp}**. Generated from [{snapshot_path.name}]({snapshot_path.name}); no GPU jobs were submitted, cancelled or restarted by publication tooling.','','## Frozen structural-PLM sweep','',f"**{len(status['completed'])}/36** completed receipts; **{len(status['failures'])}** recorded formal failure files at this snapshot. The final analysis remains dependent on the full array.",'','| Task | Arm | Seed | Validation-best pair AP | Last-epoch pair AP |','|---:|---|---:|---:|---:|']
    for row in status['completed']:
        ops.append(f"| {row['task_id']} | {row['arm']} | {row['seed']} | {row['best_pair_ap']:.6f} | {row['last_pair_ap']:.6f} |")
    ops+=['','These intermediate values do not select a winner and are not independent test scores.','','## Actual scheduler records','','```text',status['accounting'].strip(),'```','','AF3 job `1640914_0` failed before prediction. Earlier references to `1640888` were not the actual ID found in `TASK0_JOB_ID.txt`. No structure result is promoted from that failed launch.','','## Still unfinished','','Full layer/control comparison; learned realignment; a newly admitted untouched OOD cohort; same-split RePaRank retraining; complete structure-RePaRank evaluation; co-folder comparison; affinity endpoint. The prior 14 OOD candidates were not admitted.','']
    return {'README.md':readme,'figures/evidence/representation_transfer.svg':'\n'.join(svg)+'\n','results/operations/README.md':'\n'.join(ops)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    for rel,text in products().items():
        path=ROOT/rel
        if args.check:
            if not path.exists() or path.read_text()!=text:
                raise SystemExit('generated publication file differs: '+rel)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text)
    print('PASS: generated reporting files match hashed inputs' if args.check else 'Rendered README, data chart and timestamped status')

if __name__=='__main__': main()
