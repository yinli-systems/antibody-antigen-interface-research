# Ranking-term removal improves two measured endpoints

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `readout-bound` — Committed research readout inspected; not independently recomputed

Node+edge pair AP 0.09055 versus full-objective 0.06917; paired gains about+0.0214 pair AP and+0.0369 node MCC, favorable 10/10.

| Field | Value |
|---|---|
| workstream | Objectives & controls |
| cohort | AsEP epitope_group validation |
| inputs | Four node/contact/ranking objective combinations |
| seeds | 10 paired seeds per arm |
| estimand | Macro pair AP; node MCC fixed 0.5; paired seed effects |

## What this does not establish

Scheduler/checkpoint selection also change through the monitored loss. This is a full-procedure intervention, not isolated ranking-gradient causality or all-task Pareto improvement.

## Pinned evidence

1. [results/objective_component_localisation_v1/PART_TWO.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/objective_component_localisation_v1/PART_TWO.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `85ade4fe769fc68e59dd8c9b83660e682c7f7c14`; SHA-256 `fc5e0293364121df5fe38d52a7799e2149707d4e73635da196004c1db4a3da07`.
2. [results/objective_component_localisation_v1/RUN_CONTRACT.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/23480c507aeab4a5bb212631ad2b8eb09cb8867f/results/objective_component_localisation_v1/RUN_CONTRACT.md)
   - Branch `main`; commit `23480c507aeab4a5bb212631ad2b8eb09cb8867f`.
   - Git blob `a3572103a562766651d9b5bbaa0138607ef7eba8`; SHA-256 `69da2b9e21d2780cacec96c71d3fd0bc0f7427b8d9c4e543658d75e04826f3be`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
