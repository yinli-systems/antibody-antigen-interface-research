# AbCite eval snapshot (fixture vs hard-dev)

Fixture is the in-corpus 150 used by CI (abstain-on-unanswerable must be 100%). Hard-dev is paraphrases / multi-hop / out-of-index / residue questions; Hit@5 is expected to drop. Do not tune retrieval to fake 99% on hard.

## Fixture

```
FIXTURE items                    150
unanswerable                     45
answerable (incl HITL)           105
multimodal (image id required)   35
abstain-on-unanswerable          100.0%
Hit@5                            100.0%
Recall@5                         99.5%
nDCG@10                          0.891
MRR                              0.857
faithfulness (lexical only)      73.8%
faithfulness (claim-span)        80.2%
claim-NLI heuristic              77.2%
judge score (lexical heuristic)  73.8%
multimodal citation-id match     100.0%
auto failure counts              {"ok": 149, "miss_multi_hop": 1}
taxonomy n                       150
abstain failures                 none
  public_pdb n=105               Hit@5 100.0%  nDCG@10 0.891  MRR 0.857
```

## Hard-dev

```
HARD-DEV items                   40
unanswerable                     10
answerable (incl HITL)           30
multimodal (image id required)   11
abstain-on-unanswerable          100.0%
Hit@5                            96.7%
Recall@5                         90.0%
nDCG@10                          0.801
MRR                              0.787
faithfulness (lexical only)      74.5%
faithfulness (claim-span)        73.6%
claim-NLI heuristic              74.6%
judge score (lexical heuristic)  74.5%
multimodal citation-id match     90.9%
auto failure counts              {"ok": 37, "miss_multi_hop": 2, "miss_figure": 1}
taxonomy n                       40
abstain failures                 none
  public_pdb n=30                Hit@5 96.7%  nDCG@10 0.801  MRR 0.787
```

## CLIP contrast (not SOTA)

```
CLIP encoder                   local-image
figure queries                 14
mean fused RRF gap             0.001326
mean image-only cosine gap     0.006296
mean caption BM25 gap          12.341795
min gap fused                  0.000529
self-hit rate                  100.0%
hub figs (rank1 for >1 query)  {}
BiomedCLIP                     not requested
CLIP SOTA claim                false
```

Judge alignment (**human vs machine**, not inter-human κ): Cohen's κ is human vs machine (lexical / claim-NLI), not inter-human κ. Second annotator is not a duplicate of the first; n_annotators=1. κ human-vs-lexical=0.0; κ human-vs-claim-NLI=0.0; κ human-vs-human=None.

Hard taxonomy counts (all 40 items): `{"ok": 37, "miss_multi_hop": 2, "miss_figure": 1}` — see `evals/traces/hard_taxonomy.json`.

Demo receipt: `evals/traces/demo_1n8z.json` (Cho 2003 / RCSB 1N8Z).
Lexical faithfulness is a smoke test only; claim-span / claim-NLI are still heuristics, not RAGAS.

## Visual figure QA — primary multimodal demo

Gallery **N=15949** (NVIDIA GeForce RTX 5090). n_answerable=**51** (same gold as the N=1,200 run). Query is a visual question (panel layout, stain, gel bands, colors, arrows, microscopy type), **not** the figure caption. Unique gold; no PDB in the query. VQA is cite-or-abstain on the retrieved figure (heuristic/extractive, not a VLM).

| Column | R@1 | R@5 | R@10 | nDCG@5 |
|---|---|---|---|---|
| caption BM25 | 0.000 | 0.000 | 0.000 | 0.000 |
| BiomedCLIP T→I | 0.059 | 0.098 | 0.118 | 0.080 |
| OpenCLIP ViT-B-32 T→I | 0.039 | 0.137 | 0.176 | 0.096 |
| ColPali / ColQwen2 T→I | 0.078 | 0.137 | 0.196 | 0.107 |

Vision beats caption BM25 at R@1? **yes**. ColQwen2 leads at R@1 (0.078 vs BiomedCLIP 0.059 vs OpenCLIP 0.039). Drop vs N=1,200: ColQwen2 R@1 0.196→0.078. VQA e2e=0.1803 oracle=0.5574 cite@1=0.0784. See `evals/visual_qa_latest.json`.

## Audio / video QA (unified multimodal RAG)

Audio gallery **N=24** (encoder=wav-hash; synthetic fixture). Query is not the transcript. RRF is within-audio only.

| Column | R@1 | R@5 | R@10 | nDCG@5 |
|---|---|---|---|---|
| transcript BM25 | 0.000 | 0.100 | 0.500 | 0.041 |
| CLAP / wav-hash T→A | 0.350 | 0.700 | 1.000 | 0.545 |
| within-audio RRF | 0.050 | 0.500 | 0.900 | 0.270 |

Video gallery **N=16** (encoder=local-image; synthetic keyframes). RRF is within-video (frames ⟂ audio), never into paper caption BM25.

| Column | R@1 | R@5 | R@10 | nDCG@5 |
|---|---|---|---|---|
| transcript BM25 | 0.083 | 0.417 | 0.833 | 0.249 |
| keyframe CLIP T→I | 0.583 | 1.000 | 1.000 | 0.830 |
| audio track T→A | 0.083 | 0.417 | 0.833 | 0.246 |
| within-video RRF | 0.250 | 1.000 | 1.000 | 0.624 |
See `evals/audio_qa_latest.json` and `evals/video_qa_latest.json`.

## Appendix — Protocol A held-out caption-as-query (PMC OA)

Wrong product task: BM25 already wins; CLIP rerank hurts. Kept as a diagnostic.

Gallery **N=15949** readable jpg (NVIDIA GeForce RTX 5090). Split by PMCID 80/20 (seed=7): **2491 papers** → 1993 index / **498 query**; query figures 3022; **n_queries=2805**. PDB / UniProt stripped. CLIP fusion off by default. Rerank = BiomedCLIP cosine on BM25 top-50.

| Column | R@1 | R@5 | R@10 |
|---|---|---|---|
| caption BM25 | 0.835 | 0.964 | 0.985 |
| BiomedCLIP T→I full-gallery | 0.601 | 0.736 | 0.764 |
| OpenCLIP ViT-B-32 T→I | 0.047 | 0.096 | 0.124 |
| rerank-on-BM25-top50 (BiomedCLIP) | 0.661 | 0.815 | 0.851 |

Rerank helps BM25? **no** (ΔR@1 -0.174). See `evals/multimodal_caption_latest.json`.

## Appendix — Protocol B unique visual questions (small-n)

n_queries=**33** (small-n, labeled). Caption-free visual gold remapped to `pmc-*`. Not mixed with Protocol A.

| Column | R@1 | R@5 | R@10 |
|---|---|---|---|
| caption BM25 | 0.576 | 0.818 | 0.849 |
| BiomedCLIP T→I | 0.091 | 0.212 | 0.242 |
| OpenCLIP T→I | 0.030 | 0.091 | 0.121 |
| rerank-on-BM25-top50 (BiomedCLIP) | 0.242 | 0.394 | 0.455 |

Rerank helps BM25? **no** (ΔR@1 -0.333). See `evals/multimodal_visual_latest.json`.
