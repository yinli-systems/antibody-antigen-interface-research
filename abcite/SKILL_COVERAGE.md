# AbCite skill coverage — Yin Li, 2026-09-09

One page: UK Applied AI Engineer skills → files in this project.

| Skill | Evidence in this repo |
|---|---|
| Production RAG (hybrid BM25 + dense + image RRF, parent-child, cite-or-abstain) | `abcite_core/retrieval.py`, `ingest/chunking.py`, `agents/graph.py`, `prompts/cite_or_abstain_v1.txt` |
| Multimodal index (text / figure / structure) | `abcite_core/vision.py`, `abcite_core/structure.py`, `corpus/chunks.jsonl`, `corpus/figures/` |
| Figure retrieval (caption+BM25 **and** image similarity) | `abcite_core/retrieval.py` `search` / `search_by_image`; `tests/test_vision_retrieval.py` |
| Agents / LangGraph + HITL interrupt | `agents/graph.py` (`needs_hitl`, optional `build_langgraph`), `apps/web/src/App.tsx` HITL modal |
| MCP tools | `mcp/server.py` (literature, **cached RCSB**, overlap lookup) |
| Live RCSB with offline cache | `abcite_core/rcsb.py`, `corpus/rcsb_cache/`, `tests/test_rcsb.py` |
| FastAPI serving | `apps/api/main.py` (`/query` `/feedback` `/health` `/docs`) |
| Provider-used phrasing (keys → OpenAI/Azure; else extractive) | `providers/__init__.py` `get_llm`, `agents/graph.py` `compose_with_provider`, `tests/test_provider.py` |
| Desktop UI + citation thumbnails + residue chips | `apps/web/src/App.tsx`, `apps/web/src/styles.css` |
| Evals that are numbers, not placeholders | `abcite_core/metrics.py`, `evals/run.py`, `evals/latest.md`, `evals/gold/gold.jsonl` (90 items) |
| CI eval gate (abstain-on-unanswerable = 100%) | `.github/workflows/eval.yml`, `../../.github/workflows/abcite-eval.yml` |
| LLM-as-judge skepticism | `evals/judge_alignment.json`, `evals/failure_modes.md` |
| Azure OpenAI adapter + how-to | `providers/azure.py`, `infra/azure.md`, `docs/azure-vs-bedrock.md` |
| AWS Bedrock secondary | `providers/bedrock.py` |
| OpenAI + vision API path (called when key present) | `providers/openai.py`, `tests/test_provider.py` |
| Local extractive + local image encoder | `providers/local.py`, `abcite_core/vision.py` |
| LangSmith optional, no crash if unset | `abcite_core/tracing.py`, `tests/test_tracing.py` |
| Docker Compose (API + optional web) | `infra/Dockerfile`, `infra/compose.yaml` |
| Terraform (thin, not production) | `infra/terraform/main.tf` |
| Prompt versioning | `prompts/cite_or_abstain_v1.txt` |
| Security / injection / PHI | `evals/security.md` |
| Domain overlap citation (read-only FrameContact receipt numbers) | `abcite_core/overlap.py` |
| Settings / no secrets in git | `.env.example` |

FrameContact training/eval/slurm under `framecontact/` was not modified for this project.
