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

The **14-SVG cite-id** number is a demo. Image cosine gap **0.006** on that toy set is collapse. The protocol that counts is `evals/run_multimodal.py` on OA paper figures (74 indexed this run; `download_figures.py --max 1200` to resume toward 1000). OpenCLIP ViT-B-32 on real jpg: image-only R@1/5/10 **0.016 / 0.125 / 0.188**, caption BM25 **0.047 / 0.109 / 0.156**, fusion **0.031 / 0.078 / 0.141**. i2i gap **0.162** (not 0.006). T→I gap still **0.0036** — CLIP still collapses for generic visual-class queries; not SciMMIR FT or BiomedCLIP PMC (weights missing).

### 3. Faithfulness

Lexical **73.7%**. Human n=40, one annotator. κ human-vs-machine **0.0**. Need a second human before inter-annotator κ. [Who Validates the Validators?](https://arxiv.org/abs/2404.12272).

### 4. Next bar (not this sprint)

PaperQA2-style live literature, GraphRAG, PubMedQA, residue overlap vs PDB beyond one cached 1N8Z demo; second human rater.

## Where this sits in *multimodal* retrieval (2026-09 web)

**Verdict:** demo / portfolio engineering, **not** multimodal SOTA and **not** in the same game as public figure corpora. Hard cite-id **90.9%** is 10/11 figure-id matches on a **14-SVG** gallery (`corpus/figures/`), driven by caption BM25 (gap **12.34**) after RRF; image-only CLIP gap **0.006** is collapse. Public work ranks among **10⁴–10⁶** real figures.

Closest public numbers (not apples-to-apples; AbCite is cite-or-abstain on a toy index):

| AbCite (Sep 2026) | Closest public multimodal number |
|---|---|
| cite-id 90.9% hard (n=11), 100% fixture (n=35); 14 SVGs | [SciMMIR](https://arxiv.org/abs/2401.13478) 530k paper figures/tables: CLIP-base **FT Hits@10 13.5%** T→I (ALL); most ZS VLMs Hits@10 **<0.23%** |
| CLIP image-only rank1−2 gap 0.006; self-hit 100% on 14 figs | [BiomedCLIP](https://arxiv.org/abs/2303.00915) **R@1 ~56% / R@5 ~77%** among **725,739** PMC pairs; [NNN / hubness](https://arxiv.org/abs/2410.24114) |
| nDCG@10 0.809 hard (text+caption fusion) | [ColPali](https://arxiv.org/abs/2407.01449) ViDoRe **nDCG@5 81.3** (page images, not 14 cartoons) |
| multimodal cite-id as RAG Hit | [MMDocRAG](https://arxiv.org/abs/2505.16470): hybrid ColQ+BGE image **R@20 85.2%** / text **47.7%** on 4,055 QA + 32k image quotes |
| OpenCLIP ViT-B-32 on SVG rasters | CLIP ViT-B Flickr30k T→I **R@1 ~58.8%** ([Radford 2021](https://arxiv.org/abs/2103.00020)); natural photos, ~1k gallery — still not diagrams |

**Real multimodal number next (not a bigger ViT):** ≥10³ unique **paper** figures with caption-ablated image-only Recall@K vs BiomedCLIP/SciMMIR; queries that cannot be solved by PDB id in the caption; report image-only gap (must not stay 0.006). PaperQA2 figures are **caption enrichment**, not CLIP ranking ([github.com/Future-House/paper-qa](https://github.com/Future-House/paper-qa)). MMMU / ScienceQA / DocVQA / FigureQA are **VQA**, not retrieval.

## What NOT to do

- Inflate gold size with more in-corpus / `synthetic_exam` until Hit@5 = 100%.
- Swap CLIP for a larger ViT and retweet 0.99 cosines.
- Turn on LLM-as-judge in CI without human calibration.
- Claim BEIR/RAGAS comparability or antibody SOTA from this fixture.
- Touch FrameContact training to juice copilot screenshots.
- Fake a second annotator by duplicating `human_labels.json`.
