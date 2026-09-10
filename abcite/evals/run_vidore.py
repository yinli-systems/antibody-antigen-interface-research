"""Official ViDoRe v1 subset eval with the same ColQwen2 weights.

Uses a published ViDoRe QA-format dataset (query → page image) and the
ColPali late-interaction metric (nDCG@5). This is page retrieval, not
AbCite PMC figure visual QA. Do not mix the two protocols.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

import numpy as np
from PIL import Image

from abcite_core.colpali_encoder import try_load_colpali
from abcite_core.metrics import ndcg_at_k, recall_at_k
from abcite_core.visual_qa import maxsim_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "vidore/tabfquad_test_subsampled"
VIDORE_V1 = (
    "vidore/arxivqa_test_subsampled",
    "vidore/docvqa_test_subsampled",
    "vidore/infovqa_test_subsampled",
    "vidore/tabfquad_test_subsampled",
    "vidore/tatdqa_test",
    "vidore/shiftproject_test",
    "vidore/syntheticDocQA_artificial_intelligence_test",
    "vidore/syntheticDocQA_energy_test",
    "vidore/syntheticDocQA_government_reports_test",
    "vidore/syntheticDocQA_healthcare_industry_test",
)
TASK_NAME = {
    "vidore/arxivqa_test_subsampled": "ArxivQA",
    "vidore/docvqa_test_subsampled": "DocVQA",
    "vidore/infovqa_test_subsampled": "InfoVQA",
    "vidore/tabfquad_test_subsampled": "TabFQuAD",
    "vidore/tatdqa_test": "TAT-DQA",
    "vidore/shiftproject_test": "ShiftProject",
    "vidore/syntheticDocQA_artificial_intelligence_test": "SyntheticDocQA-AI",
    "vidore/syntheticDocQA_energy_test": "SyntheticDocQA-Energy",
    "vidore/syntheticDocQA_government_reports_test": "SyntheticDocQA-Gov",
    "vidore/syntheticDocQA_healthcare_industry_test": "SyntheticDocQA-Health",
}


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def _slug(dataset: str) -> str:
    return TASK_NAME.get(dataset, dataset.split("/")[-1]).lower().replace(" ", "-")


def _hub_dataset_parquets(dataset: str, split: str) -> list[Path]:
    name = "datasets--" + dataset.replace("/", "--")
    roots = []
    hf = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if hf:
        roots.append(Path(hf) / name if Path(hf).name == "hub" else Path(hf) / "hub" / name)
    roots.append(Path("/workspace/hf-cache/hub") / name)
    roots.append(Path.home() / ".cache/huggingface/hub" / name)
    for root in roots:
        snaps = root / "snapshots"
        if not snaps.is_dir():
            continue
        for snap in sorted(snaps.iterdir()):
            data = snap / "data"
            hits = sorted(data.glob(f"{split}*.parquet")) if data.is_dir() else []
            if not hits:
                hits = sorted(p for p in snap.glob(f"**/{split}*.parquet") if p.is_file())
            resolved = []
            for p in hits:
                target = p.resolve() if p.exists() else p
                if target.is_file() and target.stat().st_size > 1024:
                    resolved.append(target)
            if resolved:
                return resolved
    raise FileNotFoundError(f"No local parquet for {dataset} split={split}")


def _decode_image(raw) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict) and raw.get("bytes"):
        import io

        return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
    return Image.fromarray(np.asarray(raw)).convert("RGB")


def _load_rows(dataset: str, split: str) -> list[dict]:
    from datasets import Dataset, concatenate_datasets

    paths = _hub_dataset_parquets(dataset, split)
    parts = [Dataset.from_parquet(str(p)) for p in paths]
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        q = str(ex.get("query") or "").strip()
        img = ex.get("image")
        fn = str(ex.get("image_filename") or ex.get("doc_id") or f"page-{i}")
        page_id = Path(fn.split("?")[0].split("#")[0]).name or f"page-{i}"
        page_id = _SAFE.sub("_", page_id).strip("._") or f"page-{i}"
        if not q or img is None:
            continue
        rows.append({"query": q, "image": _decode_image(img), "page_id": page_id, "row": i})
    if not rows:
        raise SystemExit(f"{dataset} split={split} has no query/image rows")
    return rows


def _save_unique_pages(rows: list[dict], dest: Path) -> list[tuple[str, Path]]:
    dest.mkdir(parents=True, exist_ok=True)
    pages: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for row in rows:
        pid = row["page_id"]
        if pid in seen:
            continue
        seen.add(pid)
        suffix = Path(pid).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
            path = dest / f"{pid}.png"
        else:
            path = dest / pid
        img = row["image"]
        if not path.exists():
            if isinstance(img, Image.Image):
                img.convert("RGB").save(path)
            else:
                Image.fromarray(np.asarray(img)).convert("RGB").save(path)
        pages.append((pid, path))
    return pages


def _block(scores: np.ndarray, gold_idx: list[int]) -> dict:
    n = len(gold_idx)
    r1 = r5 = r10 = ndcg5 = 0.0
    for i, g in enumerate(gold_idx):
        ranked = np.argsort(-scores[i]).tolist()
        gold = [int(g)]
        ranked_ids = [int(x) for x in ranked]
        r1 += recall_at_k(ranked_ids, gold, 1)
        r5 += recall_at_k(ranked_ids, gold, 5)
        r10 += recall_at_k(ranked_ids, gold, 10)
        ndcg5 += ndcg_at_k(ranked_ids, gold, 5)
    n = max(1, n)
    return {
        "recall@1": round(r1 / n, 4),
        "recall@5": round(r5 / n, 4),
        "recall@10": round(r10 / n, 4),
        "ndcg@5": round(ndcg5 / n, 4),
        "n": n,
    }


def _eval_one(col, dataset: str, split: str) -> dict:
    rows = _load_rows(dataset, split)
    tmp = Path(tempfile.mkdtemp(prefix="vidore-pages-"))
    pages = _save_unique_pages(rows, tmp)
    page_ids = [p[0] for p in pages]
    by = {pid: i for i, pid in enumerate(page_ids)}
    print(
        f"loaded {dataset} queries={len(rows)} pages={len(pages)} "
        f"encoder={getattr(col, 'name', 'colqwen2')}",
        flush=True,
    )
    toks = col.encode_images_multi([p[1] for p in pages], batch_size=1)
    print(f"encoded {len(toks)} pages", flush=True)
    questions = [r["query"] for r in rows]
    gold_idx = [by[r["page_id"]] for r in rows]
    q_toks = col.encode_queries_multi(questions, batch_size=8)
    scores = maxsim_matrix(q_toks, [np.asarray(t, dtype=np.float32) for t in toks])
    metrics = _block(scores, gold_idx)
    report = {
        "protocol": "vidore_v1_page_retrieval",
        "dataset": dataset,
        "split": split,
        "task": TASK_NAME.get(dataset) or dataset.split("/")[-1],
        "n_queries": len(rows),
        "n_pages": len(pages),
        "encoder": getattr(col, "name", "colqwen2"),
        "gpu": _gpu_name(),
        "metric": "nDCG@5 (binary, official ColPali / ViDoRe page retrieval)",
        **metrics,
        "vidore_colpali_paper_avg_ndcg@5": 0.813,
        "published_colqwen2_v1_avg_ndcg@5": 0.8923,
        "note": (
            "Official ViDoRe v1 QA-format subset: visual question → document page. "
            "Not AbCite PMC figure visual QA (n=51 / N=15949). "
            "ColPali paper nDCG@5 81.3 is the 10-task ViDoRe v1 average."
        ),
    }
    slug = _slug(dataset)
    dest = ROOT / "evals" / f"vidore_{slug}.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(
        f"protocol=vidore_v1 dataset={dataset} pages={len(pages)} queries={len(rows)} "
        f"nDCG@5={metrics['ndcg@5']} R@1/5/10={metrics['recall@1']}/{metrics['recall@5']}/{metrics['recall@10']}",
        flush=True,
    )
    return report


def _write_suite(reports: list[dict]) -> dict:
    ndcgs = [r["ndcg@5"] for r in reports]
    avg = round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else None
    suite = {
        "protocol": "vidore_v1_page_retrieval",
        "encoder": "colqwen2",
        "gpu": reports[0].get("gpu") if reports else None,
        "metric": "nDCG@5 (binary, official ColPali / ViDoRe page retrieval)",
        "n_tasks": len(reports),
        "n_official_v1": 10,
        "mean_ndcg@5": avg,
        "vidore_colpali_paper_avg_ndcg@5": 0.813,
        "published_colqwen2_v1_avg_ndcg@5": 0.8923,
        "gap_vs_colpali_paper": None if avg is None else round(avg - 0.813, 4),
        "gap_vs_published_colqwen2": None if avg is None else round(avg - 0.8923, 4),
        "note": (
            "Official ViDoRe v1 QA-format tasks with cached vidore/colqwen2-v1.0. "
            "Page retrieval, not AbCite PMC figure visual QA. "
            "Paper 81.3 and published ColQwen2-v1.0 89.2 are 10-task averages."
        ),
        "subsets": {_slug(r["dataset"]): r for r in reports},
        "missing": [],
    }
    dest = ROOT / "evals" / "vidore_latest.json"
    dest.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    return suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--split", default="test")
    ap.add_argument("--suite", action="store_true", help="Run all 10 official ViDoRe v1 tasks")
    args = ap.parse_args()
    datasets = list(VIDORE_V1) if args.suite else (args.dataset or [DEFAULT_DATASET])
    col, skip = try_load_colpali()
    if col is None:
        raise SystemExit(skip or "ColQwen2 failed to load")
    reports: list[dict] = []
    missing: list[str] = []
    for name in datasets:
        existing = ROOT / "evals" / f"vidore_{_slug(name)}.json"
        if existing.is_file() and os.environ.get("ABCITE_VIDORE_RESUME", "1") not in {"0", "false"}:
            prev = json.loads(existing.read_text(encoding="utf-8"))
            if prev.get("ndcg@5") is not None and prev.get("dataset") == name:
                print(f"resume skip {name} nDCG@5={prev['ndcg@5']}", flush=True)
                reports.append(prev)
                continue
        try:
            reports.append(_eval_one(col, name, args.split))
        except FileNotFoundError as exc:
            print(f"missing {name}: {exc}", flush=True)
            missing.append(name)
            continue
        _write_suite(reports)
    suite = _write_suite(reports)
    suite["missing"] = missing
    dest = ROOT / "evals" / "vidore_latest.json"
    dest.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    print(
        f"suite n_tasks={suite['n_tasks']} mean_ndcg@5={suite['mean_ndcg@5']} "
        f"vs_paper={suite['gap_vs_colpali_paper']} vs_published={suite['gap_vs_published_colqwen2']} "
        f"missing={missing}",
        flush=True,
    )


if __name__ == "__main__":
    main()
