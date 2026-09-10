"""Official ViDoRe v1 harness (vidore-benchmark ViDoReEvaluatorQA).

Same ColQwen2-v1.0 weights as evals/run_vidore.py. Uses query dedup +
processor.score_multi_vector / get_scores, not AbCite MaxSim.
Restores the training-time ``Query: `` prefix dropped in colpali-engine>=0.3.13.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import importlib.util

_rv = importlib.util.spec_from_file_location("run_vidore", Path(__file__).with_name("run_vidore.py"))
run_vidore = importlib.util.module_from_spec(_rv)
assert _rv.loader is not None
_rv.loader.exec_module(run_vidore)
TASK_NAME = run_vidore.TASK_NAME
VIDORE_V1 = run_vidore.VIDORE_V1
_gpu_name = run_vidore._gpu_name
_hub_dataset_parquets = run_vidore._hub_dataset_parquets
_slug = run_vidore._slug

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = {
    "ArxivQA": 0.880,
    "DocVQA": 0.615,
    "InfoVQA": 0.925,
    "ShiftProject": 0.899,
    "SyntheticDocQA-AI": 0.990,
    "SyntheticDocQA-Energy": 0.959,
    "SyntheticDocQA-Gov": 0.955,
    "SyntheticDocQA-Health": 0.988,
    "TabFQuAD": 0.890,
    "TAT-DQA": 0.822,
}


def _load_hf_dataset(dataset: str, split: str):
    from datasets import Dataset, concatenate_datasets
    from PIL import Image
    import io

    paths = _hub_dataset_parquets(dataset, split)
    parts = [Dataset.from_parquet(str(p)) for p in paths]
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    def _row(ex, idx):
        img = ex.get("image")
        if isinstance(img, dict) and img.get("bytes"):
            img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        fn = ex.get("image_filename") or ex.get("doc_id") or f"page-{idx}"
        return {"image": img, "image_filename": str(fn), "query": ex.get("query")}

    return ds.map(_row, with_indices=True, remove_columns=ds.column_names)


def _restore_query_prefix(processor) -> str:
    from abcite_core.colpali_encoder import ensure_query_prefix

    old = getattr(processor, "query_prefix", None)
    now = ensure_query_prefix(processor)
    return f"processor.query_prefix {old!r} -> {now!r}"


def _load_retriever():
    from abcite_core.colpali_encoder import try_load_colpali
    from vidore_benchmark.retrievers import VisionRetriever

    col, skip = try_load_colpali()
    if col is None:
        raise SystemExit(skip or "ColQwen2 failed to load")
    prefix = _restore_query_prefix(col.processor)
    print(f"query_prefix_fix: {prefix}", flush=True)

    class _GpuVisionRetriever(VisionRetriever):
        def get_scores(self, query_embeddings, passage_embeddings, batch_size=128):
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            return self.processor.score(
                qs=query_embeddings,
                ps=passage_embeddings,
                batch_size=batch_size or 128,
                device=device,
            )

    return _GpuVisionRetriever(model=col.model, processor=col.processor), getattr(col, "name", "colqwen2")


def _eval_one(evaluator, dataset: str, split: str) -> dict:
    ds = _load_hf_dataset(dataset, split)
    print(f"official load {dataset} rows={len(ds)}", flush=True)
    metrics = evaluator.evaluate_dataset(
        ds=ds,
        batch_query=8,
        batch_passage=1,
        batch_score=32,
    )
    ndcg = metrics.get("ndcg_at_5")
    if ndcg is None:
        ndcg = metrics.get("ndcg@5")
    task = TASK_NAME.get(dataset) or dataset.split("/")[-1]
    report = {
        "protocol": "vidore_v1_official_harness",
        "harness": "vidore-benchmark ViDoReEvaluatorQA",
        "dataset": dataset,
        "split": split,
        "task": task,
        "n_rows": len(ds),
        "encoder": "colqwen2",
        "gpu": _gpu_name(),
        "metric": "nDCG@5 (vidore-benchmark QA, query-dedup)",
        "ndcg@5": None if ndcg is None else round(float(ndcg), 4),
        "raw_metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
        "published_colqwen2_v1_ndcg@5": PUBLISHED.get(task),
        "query_prefix": "Query: ",
    }
    dest = ROOT / "evals" / f"vidore_official_{_slug(dataset)}.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def _write_suite(reports: list[dict], missing: list[str]) -> dict:
    ndcgs = [r["ndcg@5"] for r in reports if r.get("ndcg@5") is not None]
    avg = round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else None
    suite = {
        "protocol": "vidore_v1_official_harness",
        "harness": "vidore-benchmark==5.0.0 ViDoReEvaluatorQA",
        "encoder": "colqwen2",
        "gpu": reports[0].get("gpu") if reports else None,
        "n_tasks": len(reports),
        "n_official_v1": 10,
        "mean_ndcg@5": avg,
        "vidore_colpali_paper_avg_ndcg@5": 0.813,
        "published_colqwen2_v1_avg_ndcg@5": 0.892,
        "gap_vs_colpali_paper": None if avg is None else round(avg - 0.813, 4),
        "gap_vs_published_colqwen2": None if avg is None else round(avg - 0.892, 4),
        "abcite_maxsim_mean_ndcg@5": 0.8296,
        "sota": bool(avg is not None and avg + 1e-9 >= 0.892),
        "query_prefix": "Query: ",
        "note": (
            "Official vidore-benchmark QA evaluator (query dedup + processor scores). "
            "Not AbCite MaxSim (0.8296) and not PMC figure visual QA. "
            "Do not claim SOTA unless mean >= published ColQwen2-v1.0 89.2."
        ),
        "subsets": {_slug(r["dataset"]): r for r in reports},
        "missing": missing,
    }
    dest = ROOT / "evals" / "vidore_official_latest.json"
    dest.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    return suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--split", default="test")
    ap.add_argument("--suite", action="store_true")
    args = ap.parse_args()
    try:
        from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorQA
    except Exception as exc:
        raise SystemExit(f"vidore-benchmark not importable: {type(exc).__name__}: {exc}") from exc

    datasets = list(VIDORE_V1) if args.suite else (args.dataset or ["vidore/docvqa_test_subsampled"])
    retriever, _name = _load_retriever()
    evaluator = ViDoReEvaluatorQA(retriever)
    reports: list[dict] = []
    missing: list[str] = []
    for name in datasets:
        existing = ROOT / "evals" / f"vidore_official_{_slug(name)}.json"
        if existing.is_file() and os.environ.get("ABCITE_VIDORE_RESUME", "1") not in {"0", "false"}:
            prev = json.loads(existing.read_text(encoding="utf-8"))
            if prev.get("ndcg@5") is not None and prev.get("dataset") == name:
                print(f"resume skip {name} nDCG@5={prev['ndcg@5']}", flush=True)
                reports.append(prev)
                continue
        try:
            reports.append(_eval_one(evaluator, name, args.split))
        except FileNotFoundError as exc:
            print(f"missing {name}: {exc}", flush=True)
            missing.append(name)
            continue
        except Exception as exc:
            print(f"failed {name}: {type(exc).__name__}: {exc}", flush=True)
            missing.append(name)
            continue
        _write_suite(reports, missing)
    suite = _write_suite(reports, missing)
    print(
        f"official n_tasks={suite['n_tasks']} mean_ndcg@5={suite['mean_ndcg@5']} "
        f"vs_paper={suite['gap_vs_colpali_paper']} vs_published={suite['gap_vs_published_colqwen2']} "
        f"sota={suite['sota']} missing={missing}",
        flush=True,
    )
    if missing:
        sys.exit(2)


if __name__ == "__main__":
    main()
