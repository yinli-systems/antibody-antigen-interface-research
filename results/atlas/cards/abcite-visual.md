# AbCite visual figure retrieval is a separate application

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `readout-bound` — Committed research readout inspected; not independently recomputed

Reported 273 answerable visual queries /15,949 figures:ColQwen2.5 R@1≈0.103,nDCG@5≈0.155. Retrieval and answer generation are separately evaluated.

| Field | Value |
|---|---|
| workstream | AbCite multimodal |
| cohort | PMC cropped figures; 295 total gold including 22 abstentions |
| inputs | Released multimodal retrievers plus citation/abstention application |
| seeds | Fixed evaluation snapshot |
| estimand | R@K/nDCG separate from VQA |

## What this does not establish

Do not compare cropped-figure scores to official full-page ViDoRe as one leaderboard. Not an antibody contact model.

## Pinned evidence

1. [abcite/evals/latest.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/fb871f35678c4462074c0868a7738f1dc9483b05/abcite/evals/latest.md)
   - Branch `abcite/multimodal-copilot`; commit `fb871f35678c4462074c0868a7738f1dc9483b05`.
   - Git blob `9641b28c847eda8fd654a23d50753067c7f2faf1`; SHA-256 `83b900d112558c8915ad85eb0a6886d9768fbd643e4cecae74a79694473006f1`.
2. [abcite/evals/visual_qa_colqwen2_5_latest.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/fb871f35678c4462074c0868a7738f1dc9483b05/abcite/evals/visual_qa_colqwen2_5_latest.json)
   - Branch `abcite/multimodal-copilot`; commit `fb871f35678c4462074c0868a7738f1dc9483b05`.
   - Git blob `6af3106c61a0c0b05fac5694362ce73417938077`; SHA-256 `67fb80878bcb2fb42bc5eb87d5956a1271683b3e745f3f348f5468aa1b52ff4d`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
