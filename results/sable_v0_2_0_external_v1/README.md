# SABLE v0.2.0 external evaluation

This directory contains compact public evidence for the completed single-opening
FrameContact measurement on SABLE v0.2.0. The frozen screen retained 97 entries
from 88 PDB structures after exact- and alignment-based decontamination against
the 3,114-complex SAbDab2 development cohort.

The label-free gate, ESM-2 representations, independently predicted structures,
six exact checkpoints, routing rule, calibration, metrics, and mapping gate were
frozen before the sole bound-label opening. Experimental bound structures were
used only to construct 4.5-Angstrom contact labels; they were not model inputs.

The pinned ESM-2 materialization is also label-free. It authenticates the 176
unique antibody/antigen model-axis sequences required by the 97-entry cohort;
no bound structure or task label was opened during embedding.

The independent sequence-predicted structure cache has now also passed. It
binds 179 independently predicted antibody or antigen entities to all 97
entries, with every batch and artifact SHA reverified. Antibody and antigen
were never predicted as a joint complex; no experimental bound coordinate,
cross-partner distance, or task label was read during this stage.

The completed analysis passed all six component bindings and the fixed router.
The routed median mean-per-complex AP values across seeds 17, 29, and 43 are
0.03982 for contact, 0.30893 for epitope, and 0.67720 for paratope. PDB-equal
weighted, seed-averaged estimates and 95% cluster-bootstrap intervals are stored
in `SABLE_V0_2_0_EXTERNAL_PUBLIC_V1.json`. No checkpoint selection, calibration
refit, model update, or feedback to the development protocol occurred.

The later ABF-650M OAS-30B checkpoint was also measured with the exact frozen
three-seed task router and the same single-opened 97-entry cohort. Its routed
median mean-per-complex AP values are 0.05205 for contact, 0.30901 for epitope,
and 0.68778 for paratope. The checkpoint was unseen on SABLE before this
measurement and no SABLE result was used for selection or tuning. However, the
analyst had already seen results from the earlier frozen model on this cohort,
so this is a leakage-aware, model-unseen, measurement-only external evaluation,
not a fresh blind test. Compact metrics and binding hashes are stored in
`ABF30B_SABLE_V0_2_0_MODEL_SPECIFIC_PUBLIC_V1.json`.
