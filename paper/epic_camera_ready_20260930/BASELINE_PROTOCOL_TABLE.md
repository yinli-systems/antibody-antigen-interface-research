# Baseline modality and protocol table

The original Table 1 mixes methods with different inputs and evaluation authorities. The camera-ready must not treat it as a controlled leaderboard.

| Method | Input modality stated in EPIC | Partner-specific? | Status in the submitted Table 1 | Camera-ready treatment |
|---|---|---:|---|---|
| EPIC | sequence + **predicted monomer backbone geometry** | yes | authors' experiment | report under exact EPIC protocol; do not call sequence-only |
| ImaPEp | antibody–antigen complex-derived structural patches | yes | literature/imported | literature context only; not same input/protocol |
| DiscoTope-3.0 | antigen sequence + structure/inverse-folding/surface features | no antibody partner in the EPIC comparison | literature/imported | literature context only |
| EpiFormer | antibody and antigen geometric graphs + sequence | yes | literature/imported | literature context unless rerun under bound protocol |
| MIPE | sequence + structural modalities | yes | literature/imported | literature context unless rerun under bound protocol |
| SPPIDER-seq | sequence/PLM, partner-aware site prediction | yes | literature/imported | literature context unless rerun under bound protocol |
| ParaDeep | antibody sequence; antigen-agnostic paratope | no | literature/imported | paratope context only, not a matched three-task baseline |
| Paraplume | antibody PLM; antigen-agnostic paratope | no | literature/imported | paratope context only, not a matched three-task baseline |

## Required caption

> Literature-context numbers are reproduced from the cited sources or benchmark reports and may differ in structure authority, split construction, candidate masks, per-complex versus pooled aggregation, and threshold selection. Boldface is not used to imply a controlled head-to-head comparison. Same-protocol measurements are reported separately.

## EPIC modality statement

> EPIC is a sequence-input, predicted-monomer-geometry method. It does not require the experimentally determined antibody–antigen complex at inference, but it is not structure-free.
