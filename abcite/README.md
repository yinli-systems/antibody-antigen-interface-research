# AbCite — antibody evidence copilot

**Yin Li · 2026-09-10.** Cite-or-abstain **unified multimodal RAG** over papers, figures, audio, video, and PDB residue maps. Not a chatbot.

抗体证据副驾驶：问一篇论文声称的 epitope 是否与 PDB / 同一抗原上其他抗体重叠。答案必须引用 chunk（文本 / 图 / 音频 / 视频）；证据不足则弃权；高风险声明走 HITL。

## Problem

Antibody papers often claim epitope overlap without linking residue evidence to a PDB complex or to another antibody on the same antigen. AbCite retrieves **text + figure + audio + video + structure**, attaches claim-level citations (with thumbnails), and **abstains** when the index cannot support the claim.

FrameContact training code is **not** modified. Public development-validation overlap numbers are cited with their claim boundary (not official test / not SOTA): true epitope Jaccard **0.0566**, a0 predicted top-k Jaccard mean **0.484**.

## Architecture

```mermaid
flowchart LR
  UI[React desktop UI] --> API["FastAPI /query or /rag"]
  API --> G[Evidence graph]
  G --> R[Text BM25 + dense; CLIP fusion off]
  G --> V[Visual QA: CLIP / ColPali T→I]
  G --> A[Audio: Whisper + CLAP / wav-hash + transcript BM25]
  G --> VD[Video: keyframes + Whisper track]
  G --> T[MCP: RCSB / overlap]
  R --> C[Papers + PDB maps]
  V --> F[PMC OA figures]
  A --> W[OA / synthetic wav]
  VD --> K[Keyframes + wav]
  G -->|per-modality cite| ANS[Answer + citations]
  G -->|low evidence| X[Abstain]
```

One `/query` (`/rag` alias) accepts `task=evidence|visual_qa|audio_qa|video_qa|unified` plus optional `image_path` / `audio_path` / `video_path`. **Fusion is within-modality only.** Do not RRF vision into caption BM25 for the text-only protocol (that already hurt).

## 60-second demo

```bash
cd abcite
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# Real CLIP (optional): pip install -e ".[clip]" && python -m abcite_core.download_clip
python evals/build_corpus.py
pytest
python evals/run.py
python evals/download_figures.py --max 16000 --source pmc --workers 12   # OA jpg/png gallery; resume-able
python evals/build_av_fixtures.py                 # tiny committed wav + keyframe fixtures
python evals/run_visual_qa.py --fixture           # visual figure QA on 12 OA jpgs
python evals/run_visual_qa.py                     # visual QA + pixel retrieval (ColPali: --try-colpali)
# 5090, GPU idle: cite-or-abstain VLM (Qwen2-VL-2B-Instruct generate, not ColQwen2 MaxSim)
# ABCITE_VLM=1 python evals/run_visual_qa.py --try-colpali --colpali-only --vlm
# or: bash evals/run_vlm_vqa.sh
python evals/run_vidore.py --suite                # MaxSim ViDoRe v1 (not comparable to 89.2)
python evals/run_vidore_official.py --suite       # vidore-benchmark harness (comparable to 89.2)
python evals/run_audio_qa.py                      # transcript BM25 vs CLAP/hash (synthetic)
python evals/run_audio_qa.py --oa                 # after download_av.py (Commons + NASA)
python evals/run_video_qa.py                      # keyframes vs transcript (synthetic)
python evals/run_video_qa.py --oa
python evals/run_unified.py                       # four-modality snapshot
python evals/run_multimodal.py                    # appendix: held-out caption-as-query + Protocol B
python -m uvicorn apps.api.main:app --reload --port 8000
# 16k PMC visual_qa (npz on disk, not CI; same MaxSim as the n=273 receipt, not SOTA):
# ABCITE_PMC_GALLERY=1 ABCITE_COLPALI=1 python -m uvicorn apps.api.main:app --port 8000
# other terminal
cd apps/web && npm install && npm run dev
# open http://127.0.0.1:5173
```

### Docker Compose

```bash
cd abcite
# API only
docker compose -f infra/compose.yaml up --build api
curl -s http://127.0.0.1:8000/health

# API + web (Vite proxies /query → api:8000)
docker compose -f infra/compose.yaml up --build
# open http://127.0.0.1:5173
```

With keys (not baked into the image): `OPENAI_API_KEY=... docker compose -f infra/compose.yaml up api`

## Eval numbers (`python evals/run.py`, 2026-09-09)

CI gates **fixture** only (150 in-corpus items). **Hard-dev** is 40 paraphrases / multi-hop / out-of-index / residue questions.

| Metric | Fixture (n=150) | Hard-dev (n=40) |
|---|---|---|
| Gold | 105 answerable + 45 unanswerable (35 figure-grounded) | 30 answerable + 10 out-of-index abstain |
| Abstain-on-unanswerable | **100%** (CI fails if < 100%) | **100%** (unknown PDB / closed-index) |
| Hit@5 | **100.0%** (vanity / in-corpus; CLIP fusion off) | **96.7%** (rewrite; not a 99% target) |
| Recall@5 | 99.5% | 90.0% |
| nDCG@10 | 0.891 | **0.801** |
| MRR | 0.857 | 0.787 |
| Faithfulness (lexical vs cites) | 73.8% | 74.5% |
| Faithfulness (claim-span) | 80.2% | 73.6% |
| Claim-NLI heuristic | 77.2% | 74.6% |
| Judge (lexical heuristic, CI) | 73.8% | 74.5% |
| Multimodal citation-id match | 100% | 90.9% |

By `source` (answerable ranking only): fixture `public_pdb` n=105 Hit@5 100.0% / nDCG@10 0.891 / MRR 0.857. Hard `public_pdb` n=30 Hit@5 96.7% / nDCG@10 0.801 / MRR 0.787. `synthetic_exam` items are unanswerable (no ranking). Fixture Hit@5 went 99%→100% after CLIP fusion was turned **off** — still a closed toy index, not a bar we chased.

Hard auto-taxonomy (all 40): **ok 37**, **miss_multi_hop 2**, **miss_figure 1** — [`evals/traces/hard_taxonomy.json`](evals/traces/hard_taxonomy.json).

CLIP contrast (local hashing encoder in default CI, **14-SVG cite-id demo only**): **image-only** mean rank1−rank2 cosine gap **0.006** on that toy gallery; caption BM25 gap **12.34**; fused RRF gap is small by construction. Self-hit 100% on 14 figures; **not SOTA**. Do **not** call caption-only cite-id “CLIP multimodal”.

**Primary multimodal demo — visual figure QA + pixel retrieval** (2026-09-10). The user question must be answerable only by looking at the image (panel layout, stain, gel bands, colors, arrows, microscopy type). Query is **not** the figure caption. Gold figure is unique; no PDB id in the query. Every answerable item is `caption_bm25_should_fail` (≤1 content-token overlap with the caption). Visual retrieval prefers ColQwen2.5 late-interaction T→I when local `vidore/colqwen2.5-v0.2` + `vidore/colqwen2.5-base` weights are present (CUDA or `ABCITE_COLPALI=1`); fallback ColQwen2 (`vidore/colqwen2-v1.0`), then CLIP, then hash. Text RAG stays BM25. VQA is a separate cite-or-abstain step on the retrieved (or oracle) figure pixels. Product default stays extractive/heuristic. `ABCITE_VLM=1` runs `Qwen/Qwen2-VL-2B-Instruct` generate — not ColQwen2 MaxSim (`custom_text_proj` is retrieval-only). Product: `POST /query` with `task=visual_qa`. CLIP fusion/rerank **off**.

Gold: `evals/gold/visual_qa.jsonl` (**n=295**: 273 unique-gold answerable + 22 abstain; every answerable item is `caption_bm25_should_fail`). In-gallery answerable **n=273** on N=15,949 is the **eval** `FigureGallery` table. Default `POST /query` `task=visual_qa` MaxSim ranks the loaded fixture (CI: 14 SVGs). Opt-in 16k serving: `ABCITE_PMC_GALLERY=1` with `evals/embeddings-colqwen2.5-n15949.npz` (fallback `embeddings-colqwen2-n15949.npz`) and a ColQwen encoder — same GPU MaxSim scorer as the receipt, not a download in CI. npz-only is **retrieval-first**: missing jpg → cite-or-abstain **abstains**. When `evals/figures/cache` (or `ABCITE_PMC_IMAGE_DIR` / the 5090 path) has the top-hit jpg, product `task=visual_qa` runs the existing **heuristic** VQA on those pixels (`ABCITE_VLM=0`). `POST /search-image` MaxSim-ranks query-image tokens against that same gallery when ColQwen is loaded (CLIP/hash stay pooled); not ViDoRe, not the n=273 table. Encoder preference: ColQwen2.5 when weights are local; fallback ColQwen2 → CLIP → hash. Text RAG stays BM25. OpenCLIP and BiomedCLIP were re-scored on this **same n=273 / N=15,949** gold (npy cache hits; no 16k re-encode). Extra columns on the same gold / same 15,949 figures, each with its own npz (ColQwen2 `embeddings-colqwen2-n15949.npz` untouched): ColPali-v1.3 (`vidore/colpali-v1.3` LoRA on `vidore/colpaligemma-3b-pt-448-base` → `embeddings-colpali-v1.3-n15949.npz`) and ColQwen2.5 (`vidore/colqwen2.5-v0.2` LoRA on `vidore/colqwen2.5-base` → `embeddings-colqwen2.5-n15949.npz`). Both extra encodes were GPU MaxSim on the 5090 (`device=cuda`).

### Headline: vision beats caption BM25 (fair visual gold)

Same-gold columns on NVIDIA GeForce RTX 5090 (**n=273** / **N=15,949**). ColQwen2, ColPali-v1.3, and ColQwen2.5: GPU MaxSim (`maxsim GPU n_q=273 n_d=15949`), `Query: ` prefix. OpenCLIP / BiomedCLIP: pooled T→I, npy hits. All answerable items are `caption_bm25_should_fail`.

| Column | R@1 | R@5 | R@10 | nDCG@5 |
|---|---|---|---|---|
| caption BM25 | 0.000 | 0.000 | 0.000 | 0.000 |
| BiomedCLIP T→I | 0.040 | 0.084 | 0.114 | 0.062 |
| OpenCLIP ViT-B-32 T→I | 0.015 | 0.062 | 0.092 | 0.040 |
| ColPali-v1.3 late-interaction T→I | 0.044 | 0.132 | 0.172 | 0.090 |
| ColQwen2 late-interaction T→I | 0.073 | 0.179 | 0.212 | 0.130 |
| **ColQwen2.5 late-interaction T→I** | **0.103** | **0.209** | **0.256** | **0.155** |

ColQwen2.5 leads every cutoff on this cropped-PMC protocol. GPU MaxSim query pad **64** is not the bottleneck: a 128-token retry on the same npz (`ABCITE_MAXSIM_MAX_Q=128`, `max_d=768`) encoded questions at **obs_q=52** (q_trunc=0) and kept R@1/5/10/nDCG@5 at **0.1026 / 0.2088 / 0.2564 / 0.155**. Default cap stays 64. Cached docs go up to 779 tokens; 255/15949 are truncated at 768 (unchanged). ColPali-v1.3 is an extra column (GPU `device=cuda`, separate npz files). The ColPali paper's ViDoRe **81.3** is **page retrieval** (visual question → PDF page), not these cropped PMC figures — do not put 0.090, 0.130, or 0.155 next to 81.3. BM25 leak closed: `vqa-pmc-02280` dropped **haplotype** and `vqa-pmc-00657` dropped **panoramic**. Caption overlap still ≤1. vs **n=242** ColQwen2 **0.078 / 0.194 / 0.231 / 0.141**: R@1 **0.078 → 0.073**; R@5 **0.194 → 0.179**; R@10 **0.231 → 0.212**; nDCG@5 **0.141 → 0.130**. vs **n=166** **0.066 / 0.187 / 0.217 / 0.130**: R@1 still above 0.066; nDCG matches 0.130. Vision still beats caption BM25 at R@1. Product visual retrieval prefers ColQwen2.5 when those weights are local (`vidore/colqwen2.5-v0.2` LoRA on `vidore/colqwen2.5-base`); fallback ColQwen2 → CLIP → hash. `/query` `task=visual_qa` scores **MaxSim** (not mean-pool) on that encoder. Default index is the fixture; `ABCITE_PMC_GALLERY=1` MaxSim-searches `embeddings-colqwen2.5-n15949.npz` (or colqwen2) when present. Heuristic VQA then needs the jpg cache for the top hit; npz-only still abstains. Text RAG stays BM25. Do not score Qwen2-VL-2B as ColPali. Receipt: `evals/visual_qa_latest.json` / `evals/visual_qa_protocols_latest.json` / `evals/visual_qa_colpali-v1_3_latest.json` / `evals/visual_qa_colqwen2_5_latest.json`.

Prior **n=242** same-gallery four-column (not this table): BiomedCLIP 0.041 / 0.083 / 0.120 / 0.062; OpenCLIP 0.016 / 0.070 / 0.099 / 0.045; ColQwen2 0.078 / 0.194 / 0.231 / 0.141; BM25 0 / 0.008 / 0.008 / 0.004. Prior **n=166**: BiomedCLIP 0.042 / 0.078 / 0.102 / 0.060; OpenCLIP 0.018 / 0.072 / 0.102 / 0.048; ColQwen2 0.066 / 0.187 / 0.217 / 0.130; BM25 0.

N=1,200 n=51 reference (not the headline): ColQwen2 0.196 / 0.353 / 0.373 / 0.283; OpenCLIP 0.176 / 0.314 / 0.333 / 0.247; BiomedCLIP 0.098 / 0.176 / 0.274 / 0.135; BM25 0.020 / 0.039 / 0.059 / 0.029.

VQA cite-or-abstain is **not** retrieval R@K and **not** ViDoRe (official mean **89.11** vs published **89.2**). Heuristic on the new gold (n=295 queries / 273 answerable / 22 abstain, N=15,949, 5090). Product default stays extractive/heuristic (`ABCITE_VLM=0`). Qwen2-VL-2B was **not** re-run.

| Answerer | e2e | oracle-figure | abstain | cite@1 |
|---|---:|---:|---:|---:|
| heuristic pixel cues (n=295 / N=15949) | **9.5%** | **60.3%** | **73%** | 7.3% |
| Qwen2-VL-2B-Instruct generate (prior n=61 receipt; not re-run) | 1.6% | 42.6% | 0% | 7.8% |

The 2B VLM **does not beat** the heuristic on the prior gold. It over-answered (abstain **0%** on 10 unanswerable items). Do not set `ABCITE_VLM_MODEL=vidore/colqwen2-base`. Cached `embeddings-colqwen2-n15949.npz`; no 16k re-encode.

**Not SOTA on the PMC figure protocol.** ColQwen2.5 nDCG@5 **0.155**, ColQwen2 **0.130**, and ColPali-v1.3 **0.090** are cropped-figure T→I, not ViDoRe. ViDoRe is **page retrieval** (visual question → PDF page). Do not put 0.155, 0.130, or 0.090 next to paper ColPali **81.3** or official ColQwen2 **89.11** / published **89.2**.

### Official ViDoRe v1 (same ColQwen2 weights)

Same cached `vidore/colqwen2-v1.0` on the 5090 (`python evals/run_vidore.py --suite`). Official QA-format Hub subsets, our MaxSim + binary nDCG@5. Page retrieval, not PMC figure visual QA.

| Official task | queries | pages | R@1 | R@5 | R@10 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| ArxivQA | 500 | 500 | 0.820 | 0.920 | 0.946 | **0.875** |
| DocVQA | 500 | 500 | 0.484 | 0.622 | 0.700 | **0.557** |
| InfoVQA | 500 | 499 | 0.872 | 0.946 | 0.958 | **0.910** |
| TabFQuAD | 280 | 70 | 0.854 | 0.932 | 0.950 | **0.898** |
| TAT-DQA | 831 | 139 | 0.793 | 0.962 | 0.975 | **0.887** |
| ShiftProject | 100 | 83 | 0.780 | 0.830 | 0.830 | **0.810** |
| SyntheticDocQA-AI | 100 | 63 | 0.950 | 0.950 | 0.960 | **0.950** |
| SyntheticDocQA-Energy | 100 | 67 | 0.820 | 0.850 | 0.860 | **0.839** |
| SyntheticDocQA-Gov | 100 | 47 | 0.710 | 0.740 | 0.760 | **0.723** |
| SyntheticDocQA-Health | 100 | 53 | 0.830 | 0.870 | 0.880 | **0.848** |
| **10-task mean** | | | | | | **0.830** |

**Custom MaxSim 10-task mean nDCG@5 = 0.8296 (83.0)** — missing the training-time `Query: ` prefix and official query-dedup scorer. **Not the number to compare to 89.2.**

### Official vidore-benchmark harness (comparable to 89.2)

Same weights, `Query: ` prefix restored, `ViDoReEvaluatorQA` (`python evals/run_vidore_official.py --suite`). 10/10.

| Official task | published ColQwen2 | our nDCG@5 |
|---|---:|---:|
| ArxivQA | 0.880 | **0.878** |
| DocVQA | 0.615 | **0.602** |
| InfoVQA | 0.925 | **0.921** |
| TabFQuAD | 0.890 | **0.896** |
| TAT-DQA | 0.822 | **0.814** |
| ShiftProject | 0.899 | **0.907** |
| SyntheticDocQA-AI | 0.990 | **0.990** |
| SyntheticDocQA-Energy | 0.959 | **0.963** |
| SyntheticDocQA-Gov | 0.955 | **0.959** |
| SyntheticDocQA-Health | 0.988 | **0.983** |
| **10-task mean** | **0.892** | **0.891** |

**Official mean nDCG@5 = 0.8911 (89.11).** vs ColPali paper **81.3** (**+7.81**). vs published ColQwen2-v1.0 **89.2** (**−0.09**). Matches the published recipe to 0.1 nDCG. **Not ViDoRe SOTA** (`sota=false`; 2026 v1 pack ~92.7). DocVQA is still the drag (0.602 vs 0.615). Receipt: `evals/vidore_official_latest.json`. Page retrieval, not PMC cropped-figure visual QA.

### Official vidore-benchmark harness — ColQwen2.5 (same PMC checkpoint)

Same `vidore/colqwen2.5-v0.2` LoRA on `vidore/colqwen2.5-base` as the PMC extra column (`ABCITE_COLPALI_MODEL=vidore/colqwen2.5-v0.2`). `ViDoReEvaluatorQA` on the 5090, `Query: ` prefix, GPU. Separate receipts (does not overwrite ColQwen2 `vidore_official_latest.json`).

| Official task | our ColQwen2 | our ColQwen2.5 |
|---|---:|---:|
| ArxivQA | 0.878 | **0.896** |
| DocVQA | 0.602 | **0.637** |
| InfoVQA | 0.921 | **0.923** |
| TabFQuAD | 0.896 | **0.916** |
| TAT-DQA | 0.814 | 0.813 |
| ShiftProject | 0.907 | 0.867 |
| SyntheticDocQA-AI | 0.990 | **0.996** |
| SyntheticDocQA-Energy | 0.963 | 0.962 |
| SyntheticDocQA-Gov | 0.959 | **0.963** |
| SyntheticDocQA-Health | 0.983 | 0.975 |
| **10-task mean** | **0.891** | **0.895** |

**Official ColQwen2.5 mean nDCG@5 = 0.8947 (89.47).** Reproduced published ColQwen2.5-v0.2 **89.4** (colpali README; often rounded ~89.5) (**+0.07**). vs our ColQwen2 **89.11**. **Not ViDoRe SOTA** (`sota=false`; 2026 v1 pack ~**92.7** Argus/Nemotron). Matching 89.4 is a checkpoint reproduction, not a leaderboard claim. ShiftProject is the drag vs our ColQwen2 (0.867 vs 0.907). Do not put 89.47 next to PMC nDCG **0.155**. Receipt: `evals/vidore_official_colqwen2_5_latest.json`.

See `evals/visual_qa_protocols_latest.json` after `python evals/run_visual_qa.py`. Caption-as-query Protocols A/B stay **appendix** — BM25 already wins there and CLIP rerank hurts; that is the wrong product task.

### Audio + video (unified multimodal RAG)

Same cite-or-abstain product. Queries ask about **sound or scene**, not spoken words. RRF is **within-audio** or **within-video** only. CI fixtures are an **honest larger synthetic set** (distinct spectra / hue-coded keyframes; transcripts are generic lecture filler so BM25 should fail). Real OA clips: `python evals/download_av.py --max 64` then `ABCITE_CLAP=1 python evals/run_audio_qa.py --oa` / `ABCITE_CLIP=1 ABCITE_COLPALI=0 python evals/run_video_qa.py --oa` (or `evals/run_av_oa.sh` on the 5090). Commons + NASA public-domain `~small.mp4`; licenses in `evals/oa_av_LICENSE.md`. Keep `ABCITE_WHISPER=0` so transcripts stay lecture filler. Do not mix OA receipts with the synthetic wav-hash table. Whisper and LAION CLAP are opt-in; default CI uses a shared-space spectral / scene encoder.

Gold: `evals/gold/audio_qa.jsonl` (**n=24**: 20 answerable + 4 abstain) and `evals/gold/video_qa.jsonl` (**n=16**: 12 answerable + 4 abstain). All answerable items are `transcript_bm25_should_fail`.

**Audio** (`python evals/run_audio_qa.py`, wav-hash, N=24 synthetic):

| Column | R@1 | R@5 | R@10 | nDCG@5 |
|---|---|---|---|---|
| transcript BM25 | 0.000 | 0.100 | 0.500 | 0.041 |
| **wav-hash T→A** | **0.350** | **0.700** | **1.000** | **0.546** |
| within-audio RRF | 0.050 | 0.500 | 0.900 | 0.270 |

Embedding **beats transcript BM25** at R@1 (0.350 vs 0.000) on sound-description queries. **Not CLAP / not SOTA.**

**OA Wikimedia + NASA audio** (`python evals/download_av.py --max 64 && ABCITE_CLAP=1 python evals/run_audio_qa.py --oa`, N=29 real wav, 5090, 2026-09-10). Encoder is **LAION CLAP** (`laion/clap-htsat-unfused`) loaded from a complete local snapshot (`/workspace/hf-cache/clap-htsat-unfused`; Hub API still 429). `File:Cat.ogg` failed ffmpeg extract (not in the gallery). Small-n; not SOTA. Do not mix with the synthetic wav-hash table.

| Column | R@1 | R@5 | R@10 | nDCG@5 |
|---|---|---|---|---|
| transcript BM25 | 0.000 | 0.138 | 0.276 | 0.069 |
| **CLAP T→A** | **0.414** | **0.793** | **0.862** | **0.607** |
| within-audio RRF | 0.103 | 0.310 | 0.655 | 0.210 |

CLAP **beats** transcript BM25 at R@1 (0.414 vs 0.000) on this 29-clip Commons set (was N=9 R@1 0.333). Receipt: `evals/audio_qa_oa_latest.json`.

**Video** (`python evals/run_video_qa.py`, local-image scene cues, N=16 synthetic keyframes):

| Column | R@1 | R@5 | R@10 | nDCG@5 |
|---|---|---|---|---|
| transcript BM25 | 0.083 | 0.417 | 0.833 | 0.249 |
| **keyframe scene T→I** | **0.583** | **1.000** | **1.000** | **0.830** |
| audio track T→A | 0.083 | 0.417 | 0.833 | 0.246 |
| within-video RRF | 0.250 | 1.000 | 1.000 | 0.624 |

Frames **beat transcript BM25** at R@1 (0.583 vs 0.083) on color/layout queries. Receipts: `evals/audio_qa_latest.json`, `evals/video_qa_latest.json`. `POST /query` `task=audio_qa|video_qa|unified`.

**OA Wikimedia + NASA video** (`ABCITE_CLIP=1 ABCITE_COLPALI=0 python evals/run_video_qa.py --oa` after `download_av.py --max 64`, N=26 real keyframes, OpenCLIP ViT-B-32 on the 5090). Grew from 10 clips (Commons microscopy + NASA `~small.mp4`; Perseverance landing skipped as 111 MB). Small-n; not SOTA. Many clips have no audio stream; audio-track CLAP R@1 is **0.077**.

| Column | R@1 | R@5 | R@10 | nDCG@5 |
|---|---|---|---|---|
| transcript BM25 | 0.038 | 0.231 | 0.462 | 0.133 |
| **keyframe OpenCLIP T→I** | **0.423** | **0.692** | **0.885** | **0.559** |
| audio track CLAP T→A | 0.077 | 0.154 | 0.423 | 0.118 |
| within-video RRF | 0.192 | 0.423 | 0.769 | 0.318 |

Frames beat BM25 at R@1 (0.423 vs 0.038) on this 26-clip set (was N=10 R@1 0.300 vs 0.100). Receipt: `evals/video_qa_oa_latest.json`. Licenses: `evals/oa_av_LICENSE.md`.

OpenCLIP ViT-B-32 is the CLIP fallback when `[clip]` + weights are present. Visual retrieval prefers ColQwen2.5 (then ColQwen2) when local late-interaction weights exist on CUDA / `ABCITE_COLPALI=1`. `ABCITE_CLIP_MODEL=biomedclip` loads cached BiomedCLIP; `ABCITE_BIOMEDCLIP_DOWNLOAD=1` may fetch Hub weights (no download in default CI). Optional ColPali: `pip install -e ".[colpali]"`. Product retrieval does **not** fuse CLIP unless `ABCITE_VISION_FUSION=1`. Text RAG stays BM25.

Human labels: **40 items, one annotator**. Cohen’s κ vs a second human is **N/A**. **Human vs machine** (not inter-human): κ vs lexical **0.0**, κ vs claim-NLI **0.0** (automatic judges almost always score ≥0.5).

**Demo (real public PDB):** trastuzumab–HER2 **1N8Z** / Cho et al. Nature 2003 PMID 12610673 — [DOI](https://doi.org/10.1038/nature01392) — [RCSB 1N8Z](https://www.rcsb.org/structure/1N8Z). Receipt: [`evals/traces/demo_1n8z.json`](evals/traces/demo_1n8z.json).

Source tables: [`evals/latest.md`](evals/latest.md). Gaps: [`IMPROVEMENTS.md`](IMPROVEMENTS.md).

**CLIP:** default CI is fast (`ABCITE_CLIP=0`, hashing / tiny fallback). Real OpenCLIP ViT-B-32 is the default encoder after:

```bash
pip install -e ".[clip]"
python -m abcite_core.download_clip
```

SVG figures are rasterized to RGB before CLIP (PIL cannot open SVG). The GitHub `clip` job downloads/caches weights and **fails** if the encoder is not CLIP.

## Appendix — Protocol A held-out caption-as-query (wrong product task)

Caption-as-query is SciMMIR-style pair retrieval: BM25 already wins; CLIP rerank hurts. Kept as a diagnostic, not the demo. Real multimodal eval (OpenCLIP ViT-B-32 + BiomedCLIP, 2026-09-10, RTX 5090). PMC OA gallery: **N=15,949** readable jpg. Captions never enter the image encoder. Product default stays **hybrid text, CLIP fusion OFF, CLIP rerank OFF.**

Split **by PMCID 80/20** (seed=7): **2,491 papers** → 1,993 index / **498 query**. Queries = accession-stripped captions (**n_queries=2,805**). Gallery = full N=15,949.

| Column | R@1 | R@5 | R@10 | note |
|---|---|---|---|---|
| caption BM25 | **0.835** | **0.964** | **0.985** | lexical; duplicate captions keep R@1 off 1.0 |
| BiomedCLIP T→I full-gallery | 0.601 | 0.736 | 0.764 | i2i **0.284**; T→I gap **0.069** |
| OpenCLIP ViT-B-32 T→I | 0.047 | 0.096 | 0.124 | i2i **0.081**; T→I gap **0.007** |
| **rerank-on-BM25-top50 (BiomedCLIP)** | 0.661 | 0.815 | 0.851 | **hurts BM25** (ΔR@1 **−0.174**) |

See `evals/multimodal_caption_latest.json`.

## Appendix — Protocol B unique-gold visual questions (small-n)

n_queries=**33**. Caption-free visual gold remapped to `pmc-*`. Not mixed with Protocol A. Rerank still hurts BM25.

| Column | R@1 | R@5 | R@10 |
|---|---|---|---|
| caption BM25 | **0.576** | **0.818** | **0.849** |
| BiomedCLIP T→I | 0.091 | 0.212 | 0.242 |
| OpenCLIP T→I | 0.030 | 0.091 | 0.121 |
| rerank-on-BM25-top50 (BiomedCLIP) | 0.242 | 0.394 | 0.455 |

See `evals/multimodal_visual_latest.json`.

## Skill coverage (UK Applied AI Engineer)

See [`SKILL_COVERAGE.md`](SKILL_COVERAGE.md).

## Stub vs real in this commit

| Piece | Real now | Still stub / optional |
|---|---|---|
| Hit@5 + lexical faithfulness + judge | yes (`evals/run.py`) | Live LLM judge only with API keys; cache `evals/.cache` |
| Gold | **150 fixture** + **40 hard-dev** | Inter-human Cohen κ (one annotator; human-vs-machine κ = 0.0) |
| RCSB | allowlist + cache + MCP; live if `ABCITE_RCSB_LIVE=1` | tests never hit the network |
| Vision | SVG→RGB raster + OpenCLIP when `[clip]` is installed; 15,949 PMC OA figures on GPU; **visual QA** (`task=visual_qa`) is the primary figure demo; CLIP fusion **off** by default; VLM generate is opt-in (`ABCITE_VLM=1`, Qwen2-VL-2B-Instruct) | default CI hashing/tiny (`ABCITE_CLIP=0`); n=273 / N=15,949: ColQwen2.5 R@1 **0.103**, ColQwen2 **0.073**, ColPali-v1.3 **0.044**, BiomedCLIP **0.040**, OpenCLIP **0.015**, BM25 0; heuristic VQA e2e **9.5%** / oracle **60.3%** / abstain **73%** (ColQwen2 path); `ABCITE_PMC_GALLERY=1` MaxSim npz is retrieval-first unless jpg cache exists; prior 2B VLM n=61 e2e **1.6%** (not re-run; does not beat heuristic) |
| Audio / video | Whisper interface + CLAP/wav-hash + transcript BM25; video = keyframes + audio track; `/query` `task=audio_qa\|video_qa\|unified`; OA Commons+NASA path `evals/download_av.py` + `--oa` | OA CLAP encoder=`clap` N=29 R@1 **0.414** vs BM25 0; OA OpenCLIP video N=26 R@1 **0.423** vs BM25 0.038; CI uses synthetic wav/PNG fixtures; OA media is gitignored |
| OpenAI/Azure phrasing | called when keys set; citations from retrieval | needs a real key |
| Azure AI Search | `abcite_core/azure_search.py` hybrid+semantic body; mocked tests | live index needs `AZURE_SEARCH_*` |
| Terraform | `infra/terraform/` RG + Search + OpenAI account | not applied (no subscription in CI) |
| LangSmith | `emit_run` if tracing+key | no hosted project in this repo |
| Compose | `infra/compose.yaml` | not a cloud deploy |

## Tests

`pytest` (40+ + live CLIP skip-if-no-weights): cite-or-abstain, unanswerable, fixture≥150 + hard≥35, multimodal, **visual figure QA** on 12 OA jpgs, **VLM VQA parse / PDB-free query / heuristic default**, **audio QA** + **video QA** synthetic fixtures, caption-as-query appendix, paper holdout, vision rerank top-K, fusion off by default, SVG raster, tiny-CLIP NN, HITL, MCP/RCSB, fake OpenAI, Azure Search mock, judge prompt contract, LangSmith, eval gate.

## Security

Do not commit API keys. Copy `.env.example` → `.env`. Red-team cases: `evals/security.md`.
