# RePaRank sequence+structure same-grid readout — 2026-09-30

## Completed uncentered run (preserved negative result)

Job `1642234` completed `0:0` using the official sequence+structure checkpoint on the same 217 development-validation complexes and exact 9,638,880-pair / 9,036-positive candidate authority as the sequence checkpoint. The resume wrapper reused 26 previously computed NPZ files only after exact checkpoint, candidate/label, structure-chain and file-hash compatibility checks, and computed the remaining 191 cases. No test/OOD labels were read.

| Metric | sequence | sequence+structure, raw uncentered xyz | delta |
|---|---:|---:|---:|
| macro pair AP | 0.0378078 | 0.0107430 | -0.0270648 |
| pooled pair AP | 0.00873858 | 0.00289564 | -0.00584294 |
| pooled AUROC | 0.786064 | 0.660170 | -0.125894 |
| epitope macro AP | 0.202039 | 0.182829 | -0.019210 |
| paratope macro AP | 0.557707 | 0.409777 | -0.147930 |

An independent all-217 audit confirms exact residue-index/label parity between sequence and structure NPZ files, exact 26-file reuse hashes, and the same total pair/positive counts. For pair AP, structure−sequence is `−0.0270648` equal-complex and `−0.0309804` equal-component across 68 split-authority components; component bootstrap 95% CI is `[−0.0483945, −0.0155993]`. Only 31.8% of complexes and 27.9% of components have a positive pair-AP delta.

The negative effect is not confined to the 30 cases that required the new 40-chain AF3 completion: the 187 cases already covered by the prior authority have mean pair-AP delta `−0.0218131` (34.2% positive). The 30 completion-dependent cases are worse on average (`−0.0598000`, 16.7% positive), but include two extreme high-sequence-AP failures.

AF3 chain-ranking confidence does not explain the overall effect: Spearman rho between pair-AP delta and minimum chain ranking score is ~0.027, and with mean chain ranking score ~0.062. These are descriptive confidence proxies, not measured structure accuracy.

## Protocol defect discovered after the completed run

The uncentered result must **not** be promoted as the final fair official-preprocessing structure checkpoint result.

At official RePaRank commit `6910d950c2e483289973be41387c8913b859766a`:

- `AntibodyAntigenDatamodule` applies `CenterMesh()` for validation/test;
- `ComplexDataset.transform_example` applies that transform separately to each chain;
- `CenterMesh` returns `points - points.mean(dim=(0,1), keepdim=True)`;
- the tutorial likewise centers coordinates before structure inference;
- `PositionEmbedding` directly applies continuous sinusoidal encoding to xyz.

The completed `1642234` evaluator parsed exact N/CA/C/O coordinates but omitted `CenterMesh`. Since raw xyz enters a non-translation-invariant encoding, this omission is scientifically material. The result remains a valid **uncentered predicted-chain negative control**, not the final sequence+structure comparison.

A corrected protocol is frozen in `REPARANK_STRUCTURE_CENTERING_CORRECTION.md`. It changes only the official inference centering transform and preserves checkpoint, strict structure authority, candidate grid, labels, score definition, split and evidence boundary.

## Structure authority and input semantics

The strict predicted-chain authority is complete for 217/217 cases, contains 3,385 sequence-keyed chain units, and adds 40 strictly re-audited AF3 chains without rerunning AF3 inference. Contact/test/OOD labels were not used to build it.

These are **independently predicted chain-local structures**, not a bound antibody-antigen pose. Official RePaRank training explicitly random-rotates centered chain structures and describes this as imitating inference with separate protein structures. Therefore no bound pose should be injected to make this result look better.

## Contact protocol

The project contact authority uses the frozen 4.5 Å binary residue-contact definition. Official RePaRank `contact_detection.py` also defines true contacts at 4.5 Å, but then calibrates a latent-distance threshold by precision-recall/F1. This project instead keeps threshold-free AP/AUROC as primary same-grid diagnostics, avoiding a new threshold-selection degree of freedom.

## Claim boundary

- development validation, not blind/OOD;
- official checkpoint training accession overlap remains unknown;
- no clean SOTA claim;
- the uncentered negative result does **not** establish that structure is useless;
- interface recovery reported on correctly docked AF3 subsets is a different conditioning regime and cannot be substituted for this full-cohort independent-chain experiment.
