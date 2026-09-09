# AbCite improvements (2026-09-09 web + local eval)

Sources: `evals/latest.md`, `README.md`. This note does not change FrameContact.

## 结论先行

**Impressive (keep, do not hide):** 100% abstain-on-unanswerable on the **fixture** with a CI gate; out-of-index PDBs now abstain on hard-dev too; cite-or-abstain product shape; HITL on high-risk claims; no LLM-as-judge in default CI.

**Honest after this slice:** fixture Hit@5 **99.0%** / nDCG@10 **0.899** is still in-corpus saturation (CI abstain 100%). Hard-dev (n=40) Hit@5 **96.7%**, nDCG@10 **0.809**, MRR **0.798**, multimodal citation-id **90.9%** after query rewrite + parent-child boost + caption RRF on figure queries — not a 99% Hit@5 requirement, still a closed toy index. Lexical faithfulness **73.7%**. Claim-span **80.4%** / claim-NLI **77.5%** (fixture) remain lexical, not RAGAS. CLIP **image-only** mean rank1−rank2 cosine gap **0.006**; caption BM25 gap **12.34** (TEXT-side contrast). Human labels **n=40, one annotator**. Cohen’s κ **human vs lexical = 0.0**, **human vs claim-NLI = 0.0** (machine almost always ≥0.5; not inter-human κ).

Honest one-liner: *safety + extractive citations on a closed toy index is strong engineering; caption fusion is contrast, not a bigger ViT; κ vs a second human is still N/A.*

## Sprint landed (2026-09-09)

| Item | Status |
|---|---|
| Fixture vs hard-dev tables + nDCG@10 / MRR / Hit@5 by `source` | done (do not re-split) |
| CLIP caption BM25 + image RRF; BiomedCLIP skip | done (`retrieval.py`, `evals/clip_margin.py`) |
| Claim labels ≥40 + human-vs-machine κ | done (`evals/human_labels.json` n=40; κ vs 2nd human **N/A**) |
| Hard taxonomy on all 40 | done (`evals/traces/hard_taxonomy.json`: ok 38, miss_multi_hop 1, miss_figure 1) |
| Query rewrite / parent-child boost | done (hard nDCG 0.652 → 0.809; Hit@5 not forced to 99%) |

## Gaps vs gold standard (unchanged science)

### 1. Hit@5 99% on fixture n=150 in-corpus

Still vanity. Hard-dev Hit@5 **96.7%** after rewrite is still a small closed index, not BEIR. Do not add gold that copies chunk text.

### 2. CLIP contrast

CI default encoder is local hashing: **image cosine gap 0.006** (collapse). Caption BM25 gap **12.34** is the discrimination we actually have. OpenCLIP path exists when weights are cached. BiomedCLIP: **skipped unless weights are present** (`ABCITE_CLIP_MODEL=biomedclip`). [NNN / hubness](https://arxiv.org/html/2410.24114).

### 3. Faithfulness

Lexical **73.7%**. Human n=40, one annotator. κ human-vs-machine **0.0**. Need a second human before inter-annotator κ. [Who Validates the Validators?](https://arxiv.org/abs/2404.12272).

### 4. Next bar (not this sprint)

PaperQA2-style live literature, GraphRAG, PubMedQA, residue overlap vs PDB beyond one cached 1N8Z demo; second human rater.

## What NOT to do

- Inflate gold size with more in-corpus / `synthetic_exam` until Hit@5 = 100%.
- Swap CLIP for a larger ViT and retweet 0.99 cosines.
- Turn on LLM-as-judge in CI without human calibration.
- Claim BEIR/RAGAS comparability or antibody SOTA from this fixture.
- Touch FrameContact training to juice copilot screenshots.
- Fake a second annotator by duplicating `human_labels.json`.
