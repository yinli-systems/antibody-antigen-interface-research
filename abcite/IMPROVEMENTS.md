# AbCite improvements (2026-09-09 web + local eval)

Sources: `evals/latest.md`, `README.md`. This note does not change FrameContact.

## 结论先行

**Impressive (keep, do not hide):** 100% abstain-on-unanswerable on the **fixture** with a CI gate; out-of-index PDBs now abstain on hard-dev too; cite-or-abstain product shape; HITL on high-risk claims; no LLM-as-judge in default CI.

**Honest after this sprint:** fixture Hit@5 **99.0%** / nDCG@10 **0.907** is still in-corpus saturation. Hard-dev (n=40) Hit@5 **86.7%**, nDCG@10 **0.652**, MRR **0.597**, multimodal citation-id **54.5%**. Lexical faithfulness **73.7%** (fixture) remains a heuristic. Claim-span **80.6%** / claim-NLI **77.5%** are still lexical, not RAGAS. CLIP mean rank1−rank2 gap **0.006** (local encoder in CI) is a collapse diagnostic, not SOTA. Human labels: **n=20, one annotator** — no Cohen’s κ vs a second human.

Honest one-liner: *safety + extractive citations on a closed toy index is strong engineering; hard-dev retrieval is no longer a fake 99%.*

## Sprint landed (2026-09-09)

| Item | Status |
|---|---|
| Fixture vs hard-dev tables + nDCG@10 / MRR / Hit@5 by `source` | done (`evals/run.py`, `evals/gold/fixture.jsonl` n=150, `hard.jsonl` n=40) |
| Claim-span faithfulness + 20 human labels | done (`abcite_core/metrics.py`, `evals/human_labels.json`); κ vs 2nd human **N/A** |
| CLIP gap / hubness | done (`evals/clip_margin.py`); `ABCITE_CLIP_MODEL=biomedclip` stub |
| 1N8Z paper↔PDB receipt | done (`evals/traces/demo_1n8z.json`, Cho 2003 + RCSB cache) |
| Error taxonomy auto-labels | done (`evals/failure_modes.md` + `label_failure`) |

## Gaps vs gold standard (unchanged science)

### 1. Hit@5 99% on fixture n=150 in-corpus

Still vanity on labels reverse-engineered from the same chunks. Hard-dev Hit@5 **86.7%** is the number to quote. Do not tune retrieval to recover 99% on hard.

BEIR / MTEB IR reports **nDCG@10**, often **~0.4–0.7** on scientific slices. See [BEIR](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf).

### 2. CLIP contrast

CI default encoder is local hashing: mean gap **0.006**. OpenCLIP path exists; BiomedCLIP is **not downloaded**. Do not report neighbor cosines as a win. [NNN / hubness](https://arxiv.org/html/2410.24114); [BiomedCLIP](https://ai.nejm.org/doi/full/10.1056/AIoa2400640).

### 3. Faithfulness

Lexical **73.7%** (fixture) is still the cheap CI smoke test. Claim-span / claim-NLI are the same family of overlap heuristics. Need a second annotator before κ. [Who Validates the Validators?](https://arxiv.org/abs/2404.12272).

### 4. Next bar (not this sprint)

PaperQA2-style live literature, GraphRAG, PubMedQA, residue overlap vs PDB beyond one cached 1N8Z demo.

## What NOT to do

- Inflate gold size with more in-corpus / `synthetic_exam` until Hit@5 = 100%.
- Swap CLIP for a larger ViT and retweet 0.99 cosines.
- Turn on LLM-as-judge in CI without human calibration.
- Claim BEIR/RAGAS comparability or antibody SOTA from this fixture.
- Touch FrameContact training to juice copilot screenshots.
