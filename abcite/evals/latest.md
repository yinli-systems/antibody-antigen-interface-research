# AbCite eval snapshot (fixture vs hard-dev)

Fixture is the in-corpus 150 used by CI (abstain-on-unanswerable must be 100%). Hard-dev is paraphrases / multi-hop / out-of-index / residue questions; Hit@5 is expected to drop. Do not tune retrieval to fake 99% on hard.

## Fixture

```
FIXTURE items                    150
unanswerable                     45
answerable (incl HITL)           105
multimodal (image id required)   35
abstain-on-unanswerable          100.0%
Hit@5                            99.0%
Recall@5                         98.6%
nDCG@10                          0.899
MRR                              0.866
faithfulness (lexical only)      73.7%
faithfulness (claim-span)        80.4%
claim-NLI heuristic              77.5%
judge score (lexical heuristic)  73.7%
multimodal citation-id match     100.0%
auto failure counts              {"ok": 149, "miss_multi_hop": 1}
taxonomy n                       150
abstain failures                 none
  public_pdb n=105               Hit@5 99.0%  nDCG@10 0.899  MRR 0.866
```

## Hard-dev

```
HARD-DEV items                   40
unanswerable                     10
answerable (incl HITL)           30
multimodal (image id required)   11
abstain-on-unanswerable          100.0%
Hit@5                            96.7%
Recall@5                         93.3%
nDCG@10                          0.809
MRR                              0.798
faithfulness (lexical only)      73.7%
faithfulness (claim-span)        73.0%
claim-NLI heuristic              74.2%
judge score (lexical heuristic)  73.7%
multimodal citation-id match     90.9%
auto failure counts              {"ok": 38, "miss_multi_hop": 1, "miss_figure": 1}
taxonomy n                       40
abstain failures                 none
  public_pdb n=30                Hit@5 96.7%  nDCG@10 0.809  MRR 0.798
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

Hard taxonomy counts (all 40 items): `{"ok": 38, "miss_multi_hop": 1, "miss_figure": 1}` — see `evals/traces/hard_taxonomy.json`.

Demo receipt: `evals/traces/demo_1n8z.json` (Cho 2003 / RCSB 1N8Z).
Lexical faithfulness is a smoke test only; claim-span / claim-NLI are still heuristics, not RAGAS.
