# AbCite — antibody evidence copilot

**Yin Li · 2026-09-09.** Cite-or-abstain RAG over papers, figures, and PDB residue maps. Not a chatbot.

抗体证据副驾驶：问一篇论文声称的 epitope 是否与 PDB / 同一抗原上其他抗体重叠。答案必须引用 chunk；证据不足则弃权；高风险声明走 HITL。

## Problem

Antibody papers often claim epitope overlap without linking residue evidence to a PDB complex or to another antibody on the same antigen. AbCite retrieves **text + figure + structure** chunks, attaches claim-level citations (with thumbnails), and **abstains** when the index cannot support the claim.

FrameContact training code is **not** modified. Public development-validation overlap numbers are cited with their claim boundary (not official test / not SOTA): true epitope Jaccard **0.0566**, a0 predicted top-k Jaccard mean **0.484**.

## Architecture

```mermaid
flowchart LR
  UI[React desktop UI] --> API[FastAPI /query]
  API --> G[Evidence graph]
  G --> R[Hybrid BM25 + dense + image]
  G --> T[MCP: RCSB cache / overlap / literature]
  G --> L[LLM phrasing if keys else extractive]
  R --> C[Corpus: text / SVG figures / residue maps]
  G -->|low evidence| A[Abstain]
  G -->|clinical / annotation change| H[HITL interrupt]
  G -->|ok| ANS[Answer + citations]
```

## 60-second demo

```bash
cd abcite
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# Real CLIP (optional): pip install -e ".[clip]" && python -m abcite_core.download_clip
python evals/build_corpus.py
pytest
python evals/run.py
python evals/download_figures.py --max 1200   # OA jpg/png gallery; resume-able
python evals/run_multimodal.py                 # caption-as-query + unique-gold visual (two protocols)
python -m uvicorn apps.api.main:app --reload --port 8000
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
| Hit@5 | **99.0%** (vanity / in-corpus) | **96.7%** (rewrite; not a 99% target) |
| Recall@5 | 98.6% | 93.3% |
| nDCG@10 | 0.899 | **0.809** (was 0.652 before rewrite/caption fusion) |
| MRR | 0.866 | 0.798 |
| Faithfulness (lexical vs cites) | 73.7% | 73.7% |
| Faithfulness (claim-span) | 80.4% | 73.0% |
| Claim-NLI heuristic | 77.5% | 74.2% |
| Judge (lexical heuristic, CI) | 73.7% | 73.7% |
| Multimodal citation-id match | 100% | 90.9% |

By `source` (answerable ranking only): fixture `public_pdb` n=105 Hit@5 99.0% / nDCG@10 0.899 / MRR 0.866. Hard `public_pdb` n=30 Hit@5 96.7% / nDCG@10 0.809 / MRR 0.798. `synthetic_exam` items are unanswerable (no ranking).

Hard auto-taxonomy (all 40): **ok 38**, **miss_multi_hop 1**, **miss_figure 1** — [`evals/traces/hard_taxonomy.json`](evals/traces/hard_taxonomy.json).

CLIP contrast (local hashing encoder in default CI, **14-SVG cite-id demo only**): **image-only** mean rank1−rank2 cosine gap **0.006** on that toy gallery; caption BM25 gap **12.34**; fused RRF gap is small by construction. Self-hit 100% on 14 figures; **not SOTA**. Do **not** call caption-only cite-id “CLIP multimodal”.

**Real multimodal eval** (OpenCLIP ViT-B-32 + BiomedCLIP, 2026-09-10, RTX 5090). Same PMC OA gallery: **N=1196** readable jpg (1200 metadata rows, 4 missing files). Captions never enter the image encoder. PDB / UniProt / RefSeq tokens are stripped from queries. **Two protocols, not mixed.** BiomedCLIP weights were cached under `/workspace/hf-cache` (Hub 429 on the GPU IP; snapshot copied from a working Hub client). Not SciMMIR FT Hits@10 ~13.5% and not BiomedCLIP PMC R@1 ~56% on 726k pairs.

### Protocol A — caption-as-query pair retrieval (SciMMIR-style)

Query = accession-stripped figure caption; gold = that figure id. n_queries=**1127**. Caption BM25 is the lexical text baseline (captions as documents). CLIP image-only is T→I.

| Column | R@1 | R@5 | R@10 | gap |
|---|---|---|---|---|
| caption BM25 | 0.630 | 0.983 | 0.996 | lexical; duplicate captions keep R@1 off 1.0 |
| OpenCLIP ViT-B-32 T→I | 0.028 | 0.064 | 0.083 | i2i **0.088**; T→I **0.009** |
| **BiomedCLIP T→I** | **0.073** | **0.152** | **0.187** | i2i **0.257**; T→I **0.061** |
| OpenCLIP fusion (RRF) | 0.114 | 0.305 | 0.528 | — |
| BiomedCLIP fusion (RRF) | 0.157 | 0.366 | 0.582 | — |

Zero-shot OpenCLIP T→I Hits@10 **8.3%** vs BiomedCLIP **18.7%** on this ~1.2k gallery. Domain CLIP helps; it is still not the paper’s ~56% R@1 on 726k PMC pairs. See `evals/multimodal_caption_latest.json`.

### Protocol B — unique-gold visual questions

Caption-free visual queries (panel layout, stain, gel, microscopy; no PDB / **1N8Z**). Committed `fix-*` gold remapped to `pmc-*` via href, plus 7 extra unique visual items. n_queries=**19**. Small-n, reported separately.

| Column | R@1 | R@5 | R@10 |
|---|---|---|---|
| caption BM25 | 0.632 | 0.947 | 0.947 |
| **CLIP image-only (T→I)** | **0.105** | **0.316** | **0.368** |
| fusion (RRF) | 0.368 | 0.737 | 0.789 |

The previous class-query table (64 generic “scientific figure panel” strings, R@K≈0) was a protocol bug, not a 5090 encode failure. JPG binaries stay gitignored; committed vectors are `evals/embeddings-clip-vit-b-32-n1196.npy`. Fixture remains 12 OA jpgs for pytest caption-as-query. See `evals/multimodal_visual_latest.json`.

OpenCLIP ViT-B-32 is the default real path when `[clip]` + weights are present. `ABCITE_CLIP_MODEL=biomedclip` loads cached BiomedCLIP; `ABCITE_BIOMEDCLIP_DOWNLOAD=1` may fetch Hub weights (no download in default CI).

Human labels: **40 items, one annotator**. Cohen’s κ vs a second human is **N/A**. **Human vs machine** (not inter-human): κ vs lexical **0.0**, κ vs claim-NLI **0.0** (automatic judges almost always score ≥0.5).

**Demo (real public PDB):** trastuzumab–HER2 **1N8Z** / Cho et al. Nature 2003 PMID 12610673 — [DOI](https://doi.org/10.1038/nature01392) — [RCSB 1N8Z](https://www.rcsb.org/structure/1N8Z). Receipt: [`evals/traces/demo_1n8z.json`](evals/traces/demo_1n8z.json).

Source tables: [`evals/latest.md`](evals/latest.md). Gaps: [`IMPROVEMENTS.md`](IMPROVEMENTS.md).

**CLIP:** default CI is fast (`ABCITE_CLIP=0`, hashing / tiny fallback). Real OpenCLIP ViT-B-32 is the default encoder after:

```bash
pip install -e ".[clip]"
python -m abcite_core.download_clip
```

SVG figures are rasterized to RGB before CLIP (PIL cannot open SVG). The GitHub `clip` job downloads/caches weights and **fails** if the encoder is not CLIP.

## Skill coverage (UK Applied AI Engineer)

See [`SKILL_COVERAGE.md`](SKILL_COVERAGE.md).

## Stub vs real in this commit

| Piece | Real now | Still stub / optional |
|---|---|---|
| Hit@5 + lexical faithfulness + judge | yes (`evals/run.py`) | Live LLM judge only with API keys; cache `evals/.cache` |
| Gold | **150 fixture** + **40 hard-dev** | Inter-human Cohen κ (one annotator; human-vs-machine κ = 0.0) |
| RCSB | allowlist + cache + MCP; live if `ABCITE_RCSB_LIVE=1` | tests never hit the network |
| Vision | SVG→RGB raster + OpenCLIP ViT-B-32 when `[clip]` is installed; 1196 PMC OA figures encoded on GPU | default CI uses hashing/tiny (`ABCITE_CLIP=0`); BiomedCLIP loads from cache (`ABCITE_CLIP_MODEL=biomedclip`) |
| OpenAI/Azure phrasing | called when keys set; citations from retrieval | needs a real key |
| Azure AI Search | `abcite_core/azure_search.py` hybrid+semantic body; mocked tests | live index needs `AZURE_SEARCH_*` |
| Terraform | `infra/terraform/` RG + Search + OpenAI account | not applied (no subscription in CI) |
| LangSmith | `emit_run` if tracing+key | no hosted project in this repo |
| Compose | `infra/compose.yaml` | not a cloud deploy |

## Tests

`pytest` (40+ + live CLIP skip-if-no-weights): cite-or-abstain, unanswerable, fixture≥150 + hard≥35, multimodal, caption-as-query on 12 OA jpgs, SVG raster, tiny-CLIP NN, HITL, MCP/RCSB, fake OpenAI, Azure Search mock, judge prompt contract, LangSmith, eval gate (fixture vs hard, CLIP caption fusion, human-vs-machine κ, hard taxonomy n=40).

## Security

Do not commit API keys. Copy `.env.example` → `.env`. Red-team cases: `evals/security.md`.
