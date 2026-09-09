# AbCite eval snapshot (local extractive graph)

```
gold items                       150
unanswerable                     45
answerable (incl HITL)           105
multimodal (image id required)   35
abstain-on-unanswerable          100.0%
Hit@5 (any required chunk)       99.0%
Recall@5 (required chunks)       99.0%
faithfulness (lexical vs cites)  73.7%
judge score (heuristic)          73.7%
multimodal citation-id match     100.0%
abstain failures                 none
```

CI gate: abstain-on-unanswerable must be 100% on this fixture set.
Judge: HeuristicJudge in CI; LLMJudge when OPENAI_API_KEY/Azure is set (cache under evals/.cache). See evals/judge_alignment.json.
