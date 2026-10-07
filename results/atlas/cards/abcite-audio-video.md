# AbCite real audio/video and synthetic fixtures stay separate

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `readout-bound` — Committed research readout inspected; not independently recomputed

Reported real-audio CLAP R@1≈0.414 on 29 items; video-frame R@1≈0.423 on 26. Synthetic hash/audio and keyframes have separate tables.

| Field | Value |
|---|---|
| workstream | AbCite multimodal |
| cohort | Real audio 29/video 26 galleries plus separate fixtures |
| inputs | Released CLAP/OpenCLIP and application retrieval |
| seeds | Fixed small-n snapshots |
| estimand | Within-modality retrieval |

## What this does not establish

Hash fixtures are not trained audio-model results; small galleries and heuristic VQA do not establish production multimodal reasoning.

## Pinned evidence

1. [abcite/evals/audio_qa_oa_latest.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/fb871f35678c4462074c0868a7738f1dc9483b05/abcite/evals/audio_qa_oa_latest.json)
   - Branch `abcite/multimodal-copilot`; commit `fb871f35678c4462074c0868a7738f1dc9483b05`.
   - Git blob `51df49bfee1cb93d616a0990fc23fa54c89bbb7f`; SHA-256 `6973d977b2fb46d947bc47c9d3b7f846614b2fa076909d100cf611983e9de40c`.
2. [abcite/evals/video_qa_oa_latest.json](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/fb871f35678c4462074c0868a7738f1dc9483b05/abcite/evals/video_qa_oa_latest.json)
   - Branch `abcite/multimodal-copilot`; commit `fb871f35678c4462074c0868a7738f1dc9483b05`.
   - Git blob `7c5d113966558aa76a495e4ca18a2965bcd80adf`; SHA-256 `7bc7d36adf8cbd1c37dd0951dbb95db1d78b682dfc0c21c572893bca28a4cfbc`.
3. [abcite/evals/latest.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/fb871f35678c4462074c0868a7738f1dc9483b05/abcite/evals/latest.md)
   - Branch `abcite/multimodal-copilot`; commit `fb871f35678c4462074c0868a7738f1dc9483b05`.
   - Git blob `9641b28c847eda8fd654a23d50753067c7f2faf1`; SHA-256 `83b900d112558c8915ad85eb0a6886d9768fbd643e4cecae74a79694473006f1`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
