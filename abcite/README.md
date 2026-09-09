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
  G --> R[Hybrid BM25 + dense; CLIP fusion off]
  G --> V[Visual QA: text → figure pixels]
  G --> T[MCP: RCSB cache / overlap / literature]
  G --> L[LLM phrasing if keys else extractive]
  R --> C[Corpus: text / SVG figures / residue maps]
  V --> F[PMC OA figures]
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
python evals/download_figures.py --max 16000 --source pmc --workers 12   # OA jpg/png gallery; resume-able
python evals/run_visual_qa.py --fixture           # visual figure QA on 12 OA jpgs
python evals/run_visual_qa.py                     # primary multimodal: visual QA + pixel retrieval
python evals/run_multimodal.py                    # appendix: held-out caption-as-query + Protocol B
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

**Primary multimodal demo — visual figure QA + pixel retrieval** (2026-09-10, RTX 5090). The user question must be answerable only by looking at the image (panel layout, stain, gel bands, colors, arrows, microscopy type). Query is **not** the figure caption. Gold figure is unique; no PDB id in the query. Every answerable item is `caption_bm25_should_fail` (≤1 content-token overlap with the caption). VQA is cite-or-abstain on the retrieved figure (heuristic/extractive pixel cues, not a VLM). Product: `POST /query` with `task=visual_qa`, or `POST /search-image?task=visual_qa&query=...`. Default evidence RAG still keeps CLIP fusion/rerank **off**.

Gold: `evals/gold/visual_qa.jsonl` (**n=61**: 51 unique-gold answerable + 10 abstain). Gallery: **N=15,949** PMC OA jpg on the 5090. Encoder: BiomedCLIP T→I first. `colpali-engine` 0.3.18 imports; Hub **HTTP 429** and no cached `vidore/colpali` / ColQwen2 snapshot — ColPali column is empty, not a fake zero.

### Headline: vision beats caption BM25 (fair visual gold)

All 51 answerable queries are the BM25-should-fail subset. Caption BM25 is **0** at R@1/5/10. That is the protocol working, not a broken index.

| Column | R@1 | R@5 | R@10 | nDCG@5 |
|---|---|---|---|---|
| caption BM25 | 0.000 | 0.000 | 0.000 | 0.000 |
| **BiomedCLIP T→I** | **0.059** | 0.098 | 0.118 | 0.080 |
| OpenCLIP ViT-B-32 T→I | 0.039 | **0.137** | **0.176** | **0.096** |
| ColPali / ColQwen2 T→I | — | — | — | — |

Vision beats caption BM25 at R@1? **yes** (BiomedCLIP **0.059** > 0; OpenCLIP **0.039** > 0). Same at nDCG@5. Fusion/rerank stay **off**: BiomedCLIP rerank-on-BM25-top50 R@1 **0.000** (gold is outside the lexical top-50, as designed).

VQA cite-or-abstain (heuristic, not a VLM): e2e **16.4%**, oracle-figure **57.4%**, cite@1 **5.9%**, abstain **80%** (n_abstain=10). Receipt: `evals/visual_qa_latest.json`.

**Not SOTA.** ViDoRe ColPali nDCG@5 **81.3** is page retrieval with late interaction. Ours is figure-jpg CLIP on n=51 / N=15,949: nDCG@5 **0.096**. SciMMIR-FT Hits@10 **13.5%** and BiomedCLIP paper R@1 **~56%** are caption-as-query on different galleries — see appendix, do not mix.

See `evals/visual_qa_protocols_latest.json` after `python evals/run_visual_qa.py`. Caption-as-query Protocols A/B stay **appendix** — BM25 already wins there and CLIP rerank hurts; that is the wrong product task.

OpenCLIP ViT-B-32 is the default real path when `[clip]` + weights are present. `ABCITE_CLIP_MODEL=biomedclip` loads cached BiomedCLIP; `ABCITE_BIOMEDCLIP_DOWNLOAD=1` may fetch Hub weights (no download in default CI). Optional ColPali: `pip install -e ".[colpali]"` and `ABCITE_COLPALI=1`. Product retrieval does **not** fuse CLIP unless `ABCITE_VISION_FUSION=1`.

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
| Vision | SVG→RGB raster + OpenCLIP when `[clip]` is installed; 15,949 PMC OA figures on GPU; **visual QA** (`task=visual_qa`) is the primary multimodal demo; CLIP fusion **off** by default | default CI hashing/tiny (`ABCITE_CLIP=0`); BiomedCLIP from cache; ColPali engine installed, **weights blocked** (Hub 429) |
| OpenAI/Azure phrasing | called when keys set; citations from retrieval | needs a real key |
| Azure AI Search | `abcite_core/azure_search.py` hybrid+semantic body; mocked tests | live index needs `AZURE_SEARCH_*` |
| Terraform | `infra/terraform/` RG + Search + OpenAI account | not applied (no subscription in CI) |
| LangSmith | `emit_run` if tracing+key | no hosted project in this repo |
| Compose | `infra/compose.yaml` | not a cloud deploy |

## Tests

`pytest` (40+ + live CLIP skip-if-no-weights): cite-or-abstain, unanswerable, fixture≥150 + hard≥35, multimodal, **visual figure QA** on 12 OA jpgs, caption-as-query appendix, paper holdout, vision rerank top-K, fusion off by default, SVG raster, tiny-CLIP NN, HITL, MCP/RCSB, fake OpenAI, Azure Search mock, judge prompt contract, LangSmith, eval gate.

## Security

Do not commit API keys. Copy `.env.example` → `.env`. Red-team cases: `evals/security.md`.
