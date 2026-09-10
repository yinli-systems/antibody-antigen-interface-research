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

Gallery **N=15949** (NVIDIA GeForce RTX 5090). Gold **n=182** (166 unique-gold answerable + 16 abstain). All 166 answerable IDs are in-gallery. Query is a visual question (panel layout, stain, gel bands, colors, arrows, microscopy type), **not** the figure caption. Unique gold; no PDB in the query; every answerable item is `caption_bm25_should_fail`. Retrieval is the four-column table below (same n=166 / N=15,949). VQA is scored separately. Do not mix with official ViDoRe **89.11** vs published **89.2**. Do not put 0.130 next to 89.2.

| Column | R@1 | R@5 | R@10 | nDCG@5 |
|---|---|---|---|---|
| caption BM25 | 0.000 | 0.000 | 0.000 | 0.000 |
| BiomedCLIP T→I | 0.042 | 0.078 | 0.102 | 0.060 |
| OpenCLIP ViT-B-32 T→I | 0.018 | 0.072 | 0.102 | 0.048 |
| ColQwen2 late-interaction T→I | 0.066 | 0.187 | 0.217 | 0.130 |

Vision beats caption BM25 at R@1? **yes**. ColQwen2 leads. GPU MaxSim (`maxsim GPU n_q=166 n_d=15949`) + `Query: ` prefix; npz hit `embeddings-colqwen2-n15949.npz`. CLIP/BiomedCLIP npy hits (`embeddings-clip-vit-b-32-n15949.npy`, `embeddings-biomedclip-n15949.npy`); no 16k re-encode. vs prior n=51 ColQwen2 **0.078 / 0.176 / 0.196 / 0.124**: R@1 0.078→0.066; R@5 0.176→0.187; R@10 0.196→0.217; nDCG@5 0.124→0.130. Same-gold CLIP vs that n=51 reference: BiomedCLIP R@1 0.059→0.042; OpenCLIP 0.039→0.018. Not ViDoRe; not SOTA.

VQA cite-or-abstain (separate from R@K). Heuristic on the new gold (n=182 / N=15949 / 5090). Qwen2-VL-2B was **not** re-run (prior n=61 receipt).

| Answerer | e2e | oracle | abstain | cite@1 |
|---|---:|---:|---:|---:|
| heuristic pixel cues (n=182) | **0.1099** | **0.5769** | **0.75** | 0.0663 |
| Qwen2-VL-2B-Instruct generate (prior n=61; not re-run) | 0.0164 | 0.4262 | 0.0 | 0.0784 |

`vqa.path` = extractive/heuristic pixel cues; not a VLM. Product default `ABCITE_VLM=0`. The prior 2B VLM does **not** beat the heuristic; it over-answers (abstain 0/10). See `evals/visual_qa_latest.json`.

## Official ViDoRe v1 — page retrieval (same ColQwen2)

Different protocol from PMC figure visual QA. Same `vidore/colqwen2-v1.0` on NVIDIA GeForce RTX 5090. 10/10 official QA-format tasks.

| Official task | queries | pages | R@1 | R@5 | R@10 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| ArxivQA | 500 | 500 | 0.820 | 0.920 | 0.946 | 0.875 |
| DocVQA | 500 | 500 | 0.484 | 0.622 | 0.700 | 0.557 |
| InfoVQA | 500 | 499 | 0.872 | 0.946 | 0.958 | 0.910 |
| TabFQuAD | 280 | 70 | 0.854 | 0.932 | 0.950 | 0.898 |
| TAT-DQA | 831 | 139 | 0.793 | 0.962 | 0.975 | 0.887 |
| ShiftProject | 100 | 83 | 0.780 | 0.830 | 0.830 | 0.810 |
| SyntheticDocQA-AI | 100 | 63 | 0.950 | 0.950 | 0.960 | 0.950 |
| SyntheticDocQA-Energy | 100 | 67 | 0.820 | 0.850 | 0.860 | 0.839 |
| SyntheticDocQA-Gov | 100 | 47 | 0.710 | 0.740 | 0.760 | 0.723 |
| SyntheticDocQA-Health | 100 | 53 | 0.830 | 0.870 | 0.880 | 0.848 |
| **10-task mean** | | | | | | **0.830** |

Custom MaxSim mean nDCG@5 **0.8296** is **not** comparable to 89.2 (no `Query: ` prefix / no official scorer).

## Official vidore-benchmark harness (comparable)

`python evals/run_vidore_official.py --suite`. Same ColQwen2-v1.0. 10/10. Mean nDCG@5 **0.8911** vs published **89.2** (−0.09). Not SOTA. See `evals/vidore_official_latest.json`.

| Official task | published | nDCG@5 |
|---|---:|---:|
| ArxivQA | 0.880 | 0.878 |
| DocVQA | 0.615 | 0.602 |
| InfoVQA | 0.925 | 0.921 |
| TabFQuAD | 0.890 | 0.896 |
| TAT-DQA | 0.822 | 0.814 |
| ShiftProject | 0.899 | 0.907 |
| SyntheticDocQA-AI | 0.990 | 0.990 |
| SyntheticDocQA-Energy | 0.959 | 0.963 |
| SyntheticDocQA-Gov | 0.955 | 0.959 |
| SyntheticDocQA-Health | 0.988 | 0.983 |
| **10-task mean** | **0.892** | **0.891** |

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

OA Wikimedia audio (N=9 real wav, encoder=`clap` on CUDA, 5090; Hub still 429 — weights from a complete local `clap-htsat-unfused` snapshot). Small-n; not SOTA. Do not mix with the synthetic wav-hash table.

| Column | R@1 | R@5 | R@10 | nDCG@5 |
|---|---|---|---|---|
| transcript BM25 | 0.000 | 0.444 | 1.000 | 0.217 |
| CLAP T→A | 0.333 | 0.778 | 1.000 | 0.577 |

CLAP beats transcript BM25 at R@1 (0.333 vs 0.000). Prior wav-hash on n=6 was tied at 0.167.

OA Wikimedia video N=10 (grew from 6; OpenCLIP frames). Frames R@1 **0.300** vs BM25 0.100. Audio-track CLAP R@1 **0.000** (several clips have no audio stream). See `evals/audio_qa_oa_latest.json`, `evals/video_qa_oa_latest.json`.

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
