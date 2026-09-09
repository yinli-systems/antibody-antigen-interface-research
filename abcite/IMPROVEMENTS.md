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

The **14-SVG cite-id** number is a demo. Image cosine gap **0.006** on that toy set is collapse. The protocol that counts is `evals/run_multimodal.py` on OA paper figures (**N=1196** PMC OA jpg this run). Caption-as-query OpenCLIP ViT-B-32: CLIP T→I R@1/5/10 **0.028 / 0.064 / 0.083**, caption BM25 **0.630 / 0.983 / 0.996**, fusion **0.114 / 0.305 / 0.528**. i2i gap **0.088**. Unique-gold visual (n=19) is a second table. Downloader: `--source pmc` (AWS pmc-oa-opendata / Europe PMC / NCBI fallback, retries, OA bins); SciMMIR skipped unless `--source scimmir`.

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

## N=1200：0.000 是协议，不是 ViT 坏了（2026-09-10）

**结论先行：** OpenCLIP ViT-B-32 / RTX 5090 / **N=1200 PMC OA jpg** 的 CLIP T→I R@1/5/10 = **0 / 0 / 0**，主要是 **class retrieval 被当成 pair retrieval**。64 条合成 visual-class query 实际只有 **6 个 unique 字符串**（50/64 是 `scientific figure panel panel from a paper figure`），gold 却是 **一张** 图；gallery 里该类有 **799** 张。完美分类器的 R@10 上限 ≈ 10/799 ≈ **1.25%**。caption BM25 R@10=**0.016** 已经摸到这个天花板。i2i gap **0.090** 说明像素编码器并未 collapse（14-SVG 的 0.006 才是）。T→I gap **0.002** 是 generic class query 的必然结果：同类图 CLIP 几乎打平。

Committed gold `evals/gold/multimodal_visual.jsonl`（12 条更具体的视觉问句）在 N=1200 **没用上**：id 是 `fix-*`，gallery 是 `pmc-*`，于是 `run_multimodal.py` fallback 到 `synthesize_visual_queries`。这是协议 bug，不是 5090 编码失败。

公开工作全是 **caption-as-query / 图文对检索**，不是 class：

| 协议 | Query | Gallery | 公开数字 |
|---|---|---|---|
| [SciMMIR](https://arxiv.org/abs/2401.13478) ([ACL](https://aclanthology.org/2024.findings-acl.746/)) | **caption** → 配对 figure | test **16,263** / 530k | CLIP-base **FT** T→I Hits@10 **13.34%**；ZS CLIP Hits@10 **0.67%**；多数 ZS VLM **<0.23%** |
| [BiomedCLIP](https://arxiv.org/abs/2303.00915) ([HF](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)) | **caption** ↔ figure | **725,739** PMC pairs | Fig 2A：R@1 **~56%** / R@5 **~77%**（不是 class） |
| [PMC-CLIP](https://arxiv.org/abs/2303.07240) | **caption** on ROCO | **2,000** pairs | T2I R@1/5/10 **28.0 / 58.3 / 69.7** |
| CLIP [Flickr30k](https://arxiv.org/abs/2103.00020) | unique scene caption | ~1k | ViT-B T→I R@1 **~58.8%** |
| [ColPali](https://arxiv.org/abs/2407.01449) / ViDoRe | visual **question** → page | pages | nDCG@5 **81.3**（ArxivQA figures 79.1） |

Flickr ~1k 上 R@1≈59% 是因为每条 query 是 **unique caption**。换成 “photo of a person” 同样会掉到接近 0。科学图更同质（western blot / plot 成百上千），generic class + unique gold → R@K=0 **是期望值**。SciMMIR 自己也写了：科学 caption 描述实验结论，不是风景；ZS CLIP 即使用 **配对 caption** 在 16k gallery 上也只有 Hits@10 **0.67%**。AbCite 还更粗一层（class）。

**0.000 拆开（诚实）：**

- CLIP T→I R@K=0：**协议为主**（class query + unique gold + 6 unique strings）。模型为辅：即便改成 caption-as-query，OpenCLIP ZS 在 SciMMIR 上也只有 0.67% Hits@10；BiomedCLIP **没跑**（HF 429）。
- caption BM25 0/0/0.016：**协议**。class 词撞 caption；1/64 进 top-10 正好是碰撞率。
- fusion 0.016：**协议**，RRF 吃到 BM25 那 1 hit。
- T→I gap 0.002：**协议期望**（同类打平），不是 encoder NaN。
- i2i gap 0.09：**模型还活着**（像素可分）。不要用 14-SVG 的 0.006 讲这个 run。
- **不是** SciMMIR FT 13.5%，**不是** BiomedCLIP R@1 56%。不要对标。

## Caption-as-query landed (2026-09-10)

**结论先行：** 同一 1196 张 PMC OA jpg / OpenCLIP ViT-B-32 / RTX 5090 上，**caption-as-query** 把 CLIP T→I R@10 从 class-query 的 **0.000** 抬到 **0.083**。caption BM25 R@10 **0.996** 是词面基线（重复 caption 让 R@1 停在 0.63）。这是 SciMMIR 风格的 pair retrieval，不是 class。BiomedCLIP **仍没跑**（HF 429；hf-mirror 308 到 huggingface.co）。

Committed `fix-*` gold 已 remap 到 `pmc-*`，unique-gold 视觉问句是**第二张表**（n=19），不要和 caption-as-query 混报。

### Next sprint（按「真能抬 R@K」排序）

1. **BiomedCLIP / PMC-CLIP 权重**：本地 cache，避开 HF 429。同一 caption-as-query split 上并排 OpenCLIP。预期量级是域内 pair retrieval，不是 class。
2. **ColPali**（页/图 late-interaction）：对「问句 → 哪一页/哪张图」更接近产品。ViDoRe 是 page retrieval，不是 1200 jpg class。
3. **不要**把 1N8Z / PDB id 写进 query（holdout 已禁；14-SVG cite-id 不要混进这条协议）。

## What NOT to do

- Inflate gold size with more in-corpus / `synthetic_exam` until Hit@5 = 100%.
- Swap CLIP for a larger ViT and retweet 0.99 cosines.
- Turn on LLM-as-judge in CI without human calibration.
- Claim BEIR/RAGAS comparability or antibody SOTA from this fixture.
- Touch FrameContact training to juice copilot screenshots.
- Fake a second annotator by duplicating `human_labels.json`.
- Put **1N8Z** / PDB ids into multimodal queries to juice T→I R@K.
- Report class-query R@K=0 as “CLIP cannot see figures” (i2i gap 0.09 says otherwise).
