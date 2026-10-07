# AbCite LoRA did not beat zero-shot

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `readout-bound` — Committed research readout inspected; not independently recomputed

Reported 55-query holdout: zero-shot R@1≈0.0545 versus LoRA≈0.0182; nDCG@5≈0.0991 versus 0.050.

| Field | Value |
|---|---|
| workstream | AbCite multimodal |
| cohort | 218 train /55 holdout figure IDs; frozen gallery |
| inputs | ColQwen2.5 query-side LoRA |
| seeds | Seed 7; 256 reported steps |
| estimand | Same-holdout retrieval metrics |

## What this does not establish

Small holdout does not imply general LoRA failure, but does not support promoting this adaptation.

## Pinned evidence

1. [abcite/evals/visual_qa_colqwen2_5_holdout_lora_latest.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/fb871f35678c4462074c0868a7738f1dc9483b05/abcite/evals/visual_qa_colqwen2_5_holdout_lora_latest.json)
   - Branch `abcite/multimodal-copilot`; commit `fb871f35678c4462074c0868a7738f1dc9483b05`.
   - Git blob `7a386075436dcea926296cf39bf554780cadfb21`; SHA-256 `b1050c0a97e4805ffaf64cf7cfb2d5015ce7affcd3026193f704205c2c375956`.
2. [abcite/evals/latest.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/fb871f35678c4462074c0868a7738f1dc9483b05/abcite/evals/latest.md)
   - Branch `abcite/multimodal-copilot`; commit `fb871f35678c4462074c0868a7738f1dc9483b05`.
   - Git blob `9641b28c847eda8fd654a23d50753067c7f2faf1`; SHA-256 `83b900d112558c8915ad85eb0a6886d9768fbd643e4cecae74a79694473006f1`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
