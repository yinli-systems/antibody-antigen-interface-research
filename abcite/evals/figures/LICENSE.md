Figures are PMC Open Access (NIH/NLM OA subset via Europe PMC) and/or SciMMIR
(arXiv papers, 2023; HuggingFace `m-a-p/SciMMIR`). Redistribute only OA content.
This cache is for retrieval eval, not a republication of the papers.

To fetch ≥1000 jpg/png + metadata:

    cd abcite
    pip install datasets   # optional SciMMIR stream
    python evals/download_figures.py --max 1200

Resume-able: `evals/figures/cache/metadata.jsonl`. Gitignores the cache.
The committed `fixture/` folder is 8–16 OA thumbnails for unit tests only.
