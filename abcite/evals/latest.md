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
Recall@5                         99.0%
nDCG@10                          0.907
MRR                              0.880
faithfulness (lexical only)      73.7%
faithfulness (claim-span)        80.6%
claim-NLI heuristic              77.5%
judge score (lexical heuristic)  73.7%
multimodal citation-id match     100.0%
auto failure counts              {}
abstain failures                 none
  public_pdb n=105               Hit@5 99.0%  nDCG@10 0.907  MRR 0.880
```

## Hard-dev

```
HARD-DEV items                   40
unanswerable                     10
answerable (incl HITL)           30
multimodal (image id required)   11
abstain-on-unanswerable          100.0%
Hit@5                            86.7%
Recall@5                         80.0%
nDCG@10                          0.652
MRR                              0.597
faithfulness (lexical only)      76.9%
faithfulness (claim-span)        76.5%
claim-NLI heuristic              76.7%
judge score (lexical heuristic)  76.9%
multimodal citation-id match     54.5%
auto failure counts              {}
abstain failures                 none
  public_pdb n=30                Hit@5 86.7%  nDCG@10 0.652  MRR 0.597
```

## CLIP contrast (not SOTA)

```
CLIP encoder                   local-image
figure queries                 14
mean gap (rank1-rank2)         0.006296
min gap                        7e-06
self-hit rate                  100.0%
hub figs (rank1 for >1 query)  {}
CLIP SOTA claim                false
```

Judge alignment: Cohen's κ vs a second human is not available (one annotator). κ vs automatic judges is omitted when all human labels share one binary class.

Demo receipt: `evals/traces/demo_1n8z.json` (Cho 2003 / RCSB 1N8Z).
Lexical faithfulness is a smoke test only; claim-span / claim-NLI are still heuristics, not RAGAS.
