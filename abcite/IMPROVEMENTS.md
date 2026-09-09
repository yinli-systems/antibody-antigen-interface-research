# AbCite improvements (2026-09-09 web + local eval)

Sources: `evals/latest.md`, `README.md`. This note does not change FrameContact.

## 结论先行

**Impressive (keep, do not hide):** 100% abstain-on-unanswerable on the **fixture** with a CI gate; out-of-index PDBs now abstain on hard-dev too; cite-or-abstain product shape; HITL on high-risk claims; no LLM-as-judge in default CI.

**Honest after this slice:** fixture Hit@5 **100.0%** / nDCG@10 **0.891** is still in-corpus saturation (CLIP fusion off made the toy RAG 99→100; not a bar we chased). Hard-dev (n=40) Hit@5 **96.7%**, nDCG@10 **0.801**, MRR **0.787**, multimodal citation-id **90.9%** after query rewrite + parent-child boost + caption BM25 on figure queries — not a 99% Hit@5 requirement, still a closed toy index. Lexical faithfulness **73.8%**. Claim-span **80.2%** / claim-NLI **77.2%** (fixture) remain lexical, not RAGAS. CLIP **image-only** mean rank1−rank2 cosine gap **0.006**; caption BM25 gap **12.34** (TEXT-side contrast). Human labels **n=40, one annotator**. Cohen’s κ **human vs lexical = 0.0**, **human vs claim-NLI = 0.0** (machine almost always ≥0.5; not inter-human κ).

Honest one-liner: *safety + extractive citations on a closed toy index is strong engineering; caption fusion is contrast, not a bigger ViT; κ vs a second human is still N/A.*

## Sprint landed (2026-09-09)

| Item | Status |
|---|---|
| Fixture vs hard-dev tables + nDCG@10 / MRR / Hit@5 by `source` | done (do not re-split) |
| CLIP caption BM25 + image RRF; BiomedCLIP skip | done (`retrieval.py`, `evals/clip_margin.py`) |
| Claim labels ≥40 + human-vs-machine κ | done (`evals/human_labels.json` n=40; κ vs 2nd human **N/A**) |
| Hard taxonomy on all 40 | done (`evals/traces/hard_taxonomy.json`: ok 37, miss_multi_hop 2, miss_figure 1) |
| Query rewrite / parent-child boost | done (hard nDCG 0.652 → 0.801 with CLIP fusion off; Hit@5 not forced to 99%) |

## Gaps vs gold standard (unchanged science)

### 1. Hit@5 99% on fixture n=150 in-corpus

Still vanity — now **100.0%** after CLIP fusion off, still a 150-item in-corpus set. Hard-dev Hit@5 **96.7%** after rewrite is still a small closed index, not BEIR. Do not add gold that copies chunk text.

### 2. CLIP contrast

The protocol that counts is `evals/run_multimodal.py` on OA paper figures (**N=15,949** PMC OA jpg this run). Held-out caption-as-query: BM25 **0.835 / 0.964 / 0.985**, BiomedCLIP T→I **0.601 / 0.736 / 0.764**, OpenCLIP T→I **0.047 / 0.096 / 0.124**, BiomedCLIP rerank-on-BM25-top50 **0.661 / 0.815 / 0.851** (**hurts BM25**). Unique-gold visual (n=33, small-n) is a second table. Downloader: `--source pmc --max 16000` (AWS pmc-oa-opendata / Europe PMC / NCBI fallback); SciMMIR skipped unless `--source scimmir`. CLIP fusion is **off** in the product default.

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
| nDCG@10 0.801 hard (text+caption BM25; CLIP fusion off) | [ColPali](https://arxiv.org/abs/2407.01449) ViDoRe **nDCG@5 81.3** (page images, not 14 cartoons) |
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

- CLIP T→I R@K=0：**协议为主**（class query + unique gold + 6 unique strings）。模型为辅：caption-as-query 上 OpenCLIP Hits@10 **8.3%**、BiomedCLIP **18.7%**；都不是 726k 上的 ~56% R@1。
- caption BM25 0/0/0.016：**协议**。class 词撞 caption；1/64 进 top-10 正好是碰撞率。
- fusion 0.016：**协议**，RRF 吃到 BM25 那 1 hit。
- T→I gap 0.002：**协议期望**（同类打平），不是 encoder NaN。
- i2i gap 0.09：**模型还活着**（像素可分）。不要用 14-SVG 的 0.006 讲这个 run。
- **不是** SciMMIR FT 13.5%，**不是** BiomedCLIP R@1 56%。不要对标。

## Caption-as-query landed (2026-09-10)

**结论先行：** 同一 1196 张 PMC OA jpg / RTX 5090 上，**caption-as-query** 把 OpenCLIP T→I R@10 从 class-query 的 **0.000** 抬到 **0.083**；BiomedCLIP T→I R@1/5/10 = **0.073 / 0.152 / 0.187**（约 2.2× OpenCLIP Hits@10）。caption BM25 R@10 **0.996** 是词面基线。这是 SciMMIR 风格的 pair retrieval，不是 class，也不是论文里 726k 的 R@1 ~56%。

Committed `fix-*` gold 已 remap 到 `pmc-*`，unique-gold 视觉问句是**第二张表**（n=19），不要和 caption-as-query 混报。

## 很丧：BiomedCLIP 18.7%@10 经不起看（2026-09-10）

**结论先行：** 同一 1196 张 PMC OA 上 BiomedCLIP T→I R@10 **18.7%** 不是「域内 CLIP 能用了」。按 gallery size 折完是弱零样本；fusion 把 caption BM25 从 R@10 **99.6%** 砸到 **58.2%**，视觉检索没有产品资格；1127 条 query 就是 gallery 自己的 caption，不是用户、也不是 held-out 论文。

### 1. 1.2k 上的 18.7% @10 调完 N 就是弱

随机 R@10 = 10/1196 ≈ **0.84%**。18.7% 只是 ~22× 随机。同量级 ~1k gallery 上，CLIP ViT-B Flickr30k T→I R@1 已经 ~59%（[Radford 2021](https://arxiv.org/abs/2103.00020)）。同一模型换 COCO-5K（gallery 大约 5×），T→I R@1 掉到 ~31%（[DCLIP 复现表](https://arxiv.org/html/2505.21549v3)）：**R@K 随 N 涨而掉**，不是脚注，是检索的定义。

rank 随 N 近似线性 → 小 recall 下 R@K 大致按 **1/N** 掉：

| 折到的 gallery | BiomedCLIP R@10 粗折 | 对照 |
|---|---|---|
| AbCite **N=1196**（实测） | **18.7%** | OpenCLIP 8.3%；随机 0.84% |
| SciMMIR test **16,263** | ~**1.4%** | OpenCLIP 8.3% 折过去 ~**0.61%**，贴他们 ZS CLIP Hits@10 **0.67–0.72%**（[arxiv 2401.13478](https://arxiv.org/abs/2401.13478) / [ACL](https://aclanthology.org/2024.findings-acl.746/) Table 4） |
| **50k** | ~**0.45%** | 产品级文献图库的下限附近；18.7% 会碎 |
| BiomedCLIP 论文 **725,739** | ~**0.03%** | 论文 R@1 **~56% / R@5 ~77%**（[arxiv 2303.00915](https://arxiv.org/abs/2303.00915) / [HTML Fig 2A](https://arxiv.org/html/2303.00915)） |

所以：AbCite 没有「超过 SciMMIR」。OpenCLIP 折到 16k 正好是人家的零样本；BiomedCLIP 2.2× OpenCLIP 仍然只是稍好的弱 ZS。SciMMIR CLIP-**FT** Hits@10 ~**13.5%** 是在 16k 上，不是 1.2k。论文 BiomedCLIP 在 **72 万** 候选里 R@1 过半；这边 1.2k 里 R@1 **7.3%**。同一个 checkpoint 名字，两个世界。

[PMC-CLIP](https://arxiv.org/abs/2303.07240) 在 ROCO **2,000** pairs 上 T2I R@1/5/10 = **28.0 / 58.3 / 69.7**。AbCite 更小的 gallery、R@1 只有 **7.3%**，连这个小数的 1/4 都不到。

### 2. Fusion 输给 BM25 = 视觉通路在伤害产品

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| caption BM25 | **0.630** | **0.983** | **0.996** |
| BiomedCLIP T→I | 0.073 | 0.152 | 0.187 |
| BiomedCLIP fusion (RRF) | 0.157 | 0.366 | **0.582** |

不是「还没追上」。词面已经能把对图放进 top-10（99.6%）；RRF 掺进噪声视觉 rank 之后 top-10 掉到 58%、top-1 从 63% 掉到 16%。产品结论只有一句：**关掉 CLIP**。视觉检索未产品就绪。

### 3. 1127 queries = gallery caption 自己（transductive，不是用户）

`caption_as_query_items`：query = 该图 caption（strip PDB/UniProt），gold = 该图 id。每条 query 都已经在 index 里。这是 pair retrieval 的常规协议（SciMMIR / BiomedCLIP 论文也这么评），所以**不是**把 CLIP 训练集当 test 的那种泄漏；但它**是**闭集 / transductive：没有 held-out 论文，没有用户问句。CLIP 的预训练目标就是「用这张图自己的 caption 把它找回来」——拿 pretext 当产品评测。

更丧：BiomedCLIP 在 **PMC-15M**（PMC OA 图文对，~1500 万）上预训练（[HF](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) / [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2400640)）。AbCite gallery 也是 PMC OA。**没有报告与 PMC-15M 的重叠**。可能见过这些对，R@1 仍然 7.3%。泄漏还这个数，要么预处理/编码没对齐论文，要么 blot/plot/拼图这块 CLIP 本来就看不见。

### 4. Reviewer / hiring manager 会说什么

- 「18.7% @10 on 1k 是 vanity。先报 16k / 50k，或者别写 multimodal retrieval。」
- 「Fusion 让 BM25 变差。视觉通路在伤害产品，不要 merge。」
- 「Caption-as-query 是 CLIP 的 pretext，不是 cite-or-abstain。真正像用户的 Protocol B 只有 n=19。」
- 「论文 56% R@1 @726k 你碰都没碰到。别把 BiomedCLIP 四个字写进 demo。」
- 工程能跑 GPU、协议能复现，够 portfolio；数字撑不起 multimodal 叙事。

**一句话：** 域内 CLIP 比 OpenCLIP 好一截，仍然是玩具 gallery 上的弱零样本；词面已经 99.6% @10 的时候再融图像是自残。

## 16k held-out + rerank（2026-09-10）：还是关掉 CLIP

**结论先行：** 扩到 **N=15,949**（目标 16,000；末尾 NCBI 429），按 **PMCID 80/20** 抽 query papers（498 / 2,491；n_queries=**2,805**）。四列：

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| caption BM25 | **0.835** | **0.964** | **0.985** |
| BiomedCLIP T→I full-gallery | 0.601 | 0.736 | 0.764 |
| OpenCLIP T→I | 0.047 | 0.096 | 0.124 |
| BiomedCLIP rerank BM25 top-50 | 0.661 | 0.815 | 0.851 |

Rerank **伤害** BM25（ΔR@1 **−0.174**）。Fusion ablation 同样输给 BM25。产品默认 **fusion off、rerank off**（`ABCITE_VISION_RERANK=1` 才开）。Protocol B 仍是 small-n **n=33**：BM25 0.576/0.818/0.849，BiomedCLIP T→I 0.091/0.212/0.242，rerank 照样伤 BM25。

1.2k 上 BiomedCLIP T→I R@1 **7.3%** 是另一套 transductive 全量 caption 协议 + 逐张 encode；不要和这次 16k / held-out / batched encode 混成一条曲线。60% R@1 @16k **不是** 论文 56% @726k。Caption-as-query 仍是 CLIP pretext：gold caption 在 BM25 索引里。

### Next sprint（按「真能抬 R@K」排序）

1. **ColPali**（页/图 late-interaction）：对「问句 → 哪一页/哪张图」更接近产品。ViDoRe 是 page retrieval，不是 16k jpg caption lookup。
2. **不要**把 rerank 默认打开：这次已经量过，它伤 BM25。
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
- Treat BiomedCLIP R@10 **18.7%** on N=1196 as a win, or compare it raw to SciMMIR FT Hits@10 / paper R@1 ~56% without gallery-size adjustment.
- Treat BiomedCLIP T→I R@1 **0.60** on N=15,949 caption-as-query as product-ready: rerank still **hurts** BM25; fusion stays off.
