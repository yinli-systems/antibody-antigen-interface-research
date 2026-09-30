# Main-branch evidence consolidation — 2026-09-30

## Goal

Make `main` the clean entry point for the strongest audited science without deleting or rewriting experimental history.

The consolidation follows three artifact principles: the README should explain the project and route readers quickly; claims should link to complete/reproducible artifacts; and preservation should use stable identifiers rather than relying on transient branch names.

## Preservation performed before consolidation

- Pre-consolidation main was pinned as `archive/2026-09-30/main-pre-consolidation` at `910b752ecf3e1493b80a7605151048bf2cd3e69b`.
- Every one of the 16 non-main branch heads was pinned under `archive/2026-09-30/branch/<branch>`.
- No branch was deleted, rebased or force-updated.
- [`archive/BRANCH_FILE_INVENTORY_20260930.json`](../archive/BRANCH_FILE_INVENTORY_20260930.json) records all 1,782 pre-consolidation branch-diff file entries with path/blob SHA/size.
- [`results/BRANCH_EVIDENCE_INDEX.json`](../results/BRANCH_EVIDENCE_INDEX.json) records the 747 evidence-facing result/paper/protocol/receipt entries.

## Canonical evidence imported into main

| Source branch | Canonical material promoted | Why |
|---|---|---|
| `kevin/representation-ablation-v1` @ `55cfd137…` | `results/`, `freeze/`, `reproduction/`, representation protocols, `paper/ABSTRACT_v1.md` | Most complete controlled correspondence/objective/benchmark evidence and claims ledger |
| `kevin/teacher-paper-handoff-20260903` @ `54afd798…` | corrected FrameContact paper handoff, aligned test549, Top-10 case study, predicted-geometry public readouts | Corrects the original test455/checkpoint split mismatch and preserves the final historical replay |
| `kevin/framecontact-multimodal-evidence-20260902` @ `27605737…` | terminal MIPE 800-epoch replay files | Later terminal replay not present in old main |
| `kevin/correspondence-upgrade-20260930` @ `14a385fe…` | `extensions/correspondence_v2/` | Strict same-AA/shift/global correspondence extension with complete GPU receipts |
| `kevin/multimodel-scale-program-20260930` @ `28dffada…` | `extensions/frontier_multimodel_v1/` | Score-dissociation, structural-PLM, RePaRank and OOD programme |
| ParaCloud terminal receipts | RePaRank sequence `evaluation.json`/per-complex summary; SaProt completed task receipts; AF3 completion frozen plan | Results existed on the receipt-bound compute authority after the branch snapshot |

Other branches remain discoverable through the branch evidence index and archive tags instead of being blindly merged into the working tree.

## README selection rule

A result is headline-worthy only when it is:

1. completed rather than merely scheduled/running;
2. supported by a committed artifact or newly imported terminal receipt;
3. scientifically central to residue correspondence / partner-specific contact prediction;
4. labelled by its true evidence class (development, holdout, exposed historical, exposed pretrained, active/incomplete);
5. not contradicted by a later correction.

This is why the active 36-task SaProt/ProSST sweep appears only as an active programme, why the invalid original 455 split is not a headline, and why benchmark losses/negative results remain in the canonical index.

## Files intended for readers

- Root scientific story: [`README.md`](../README.md)
- Canonical detailed results: [`results/INDEX.md`](../results/INDEX.md)
- Current compute snapshot: [`results/CURRENT_STATUS_20260930.md`](../results/CURRENT_STATUS_20260930.md)
- Claims ledger: [`results/CLAIMS_LEDGER.md`](../results/CLAIMS_LEDGER.md)
- Branch provenance: [`BRANCH_PROVENANCE.md`](BRANCH_PROVENANCE.md)

## Non-goals

This consolidation does not declare SOTA, does not convert exposed tests into blind evidence, does not promote intermediate layer-sweep values, and does not remove negative/failed experiments. It also does not duplicate multi-gigabyte raw data, model weights or dense prediction tensors into Git; those remain path/hash/receipt bound.
