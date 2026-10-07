# AbCite ViDoRe reproduces released checkpoints

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `readout-bound` — Committed research readout inspected; not independently recomputed

Official-harness mean nDCG@5 reported 0.8911 ColQwen2 and 0.8947 ColQwen2.5 over 10 tasks; checkpoint reproduction, not new-model invention.

| Field | Value |
|---|---|
| workstream | AbCite multimodal |
| cohort | Official ViDoRe v1 full-page tasks |
| inputs | Released checkpoints and official scorer |
| seeds | 10 tasks, not 10 training seeds |
| estimand | Mean task nDCG@5; custom scorer kept separate |

## What this does not establish

No current-leaderboard/SOTA claim. Different candidate spaces from the cropped-figure visual task.

## Pinned evidence

1. [abcite/evals/vidore_official_latest.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/fb871f35678c4462074c0868a7738f1dc9483b05/abcite/evals/vidore_official_latest.json)
   - Branch `abcite/multimodal-copilot`; commit `fb871f35678c4462074c0868a7738f1dc9483b05`.
   - Git blob `b0d92f8625dd3a3c77f61306c82c12a0e9b12f05`; SHA-256 `0b0ab120327367e3c7ab09d181d653c054f7028fa2d0334bcd515af252ac5d07`.
2. [abcite/evals/vidore_official_colqwen2_5_latest.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/fb871f35678c4462074c0868a7738f1dc9483b05/abcite/evals/vidore_official_colqwen2_5_latest.json)
   - Branch `abcite/multimodal-copilot`; commit `fb871f35678c4462074c0868a7738f1dc9483b05`.
   - Git blob `c689929894aa95231c7501cd6620ef354f9d40db`; SHA-256 `96b20a0af53a9e7dc1ed4591775aca60d6775f0495b033f1e07854f3c6c2cd89`.
3. [abcite/evals/latest.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/fb871f35678c4462074c0868a7738f1dc9483b05/abcite/evals/latest.md)
   - Branch `abcite/multimodal-copilot`; commit `fb871f35678c4462074c0868a7738f1dc9483b05`.
   - Git blob `9641b28c847eda8fd654a23d50753067c7f2faf1`; SHA-256 `83b900d112558c8915ad85eb0a6886d9768fbd643e4cecae74a79694473006f1`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
