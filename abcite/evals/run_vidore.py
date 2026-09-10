"""Official ViDoRe v1 subset eval with the same ColQwen2 weights.

Uses a published ViDoRe QA-format dataset (query → page image) and the
ColPali late-interaction metric (nDCG@5). This is page retrieval, not
AbCite PMC figure visual QA. Do not mix the two protocols.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from abcite_core.colpali_encoder import try_load_colpali
from abcite_core.metrics import ndcg_at_k, recall_at_k
from abcite_core.visual_qa import maxsim_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "vidore/tabfquad_test_subsampled"
TASK_NAME = {
    "vidore/tabfquad_test_subsampled": "TabFQuAD",
    "vidore/docvqa_test_subsampled": "DocVQA",
    "vidore/arxivqa_test_subsampled": "ArxivQA",
}


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def _hub_dataset_parquet(dataset: str, split: str) -> Path:
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
            hits = sorted((snap / "data").glob(f"{split}*.parquet")) if (snap / "data").is_dir() else []
            hits += sorted(snap.glob(f"**/{split}*.parquet"))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"No local parquet for {dataset} split={split}")


def _decode_image(raw) -> Image.Image:
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict) and raw.get("bytes"):
        import io

        return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
    return Image.fromarray(np.asarray(raw)).convert("RGB")


def _load_rows(dataset: str, split: str) -> list[dict]:
    from datasets import Dataset

    path = _hub_dataset_parquet(dataset, split)
    ds = Dataset.from_parquet(str(path))
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        q = str(ex.get("query") or "").strip()
        img = ex.get("image")
        fn = str(ex.get("image_filename") or ex.get("doc_id") or f"page-{i}")
        page_id = Path(fn).name or f"page-{i}"
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
        path = dest / pid
        if not path.suffix:
            path = dest / f"{pid}.png"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    col, skip = try_load_colpali()
    if col is None:
        raise SystemExit(skip or "ColQwen2 failed to load")
    rows = _load_rows(args.dataset, args.split)
    tmp = Path(tempfile.mkdtemp(prefix="vidore-pages-"))
    pages = _save_unique_pages(rows, tmp)
    page_ids = [p[0] for p in pages]
    by = {pid: i for i, pid in enumerate(page_ids)}
    print(f"loaded {args.dataset} queries={len(rows)} pages={len(pages)} encoder={getattr(col, 'name', 'colqwen2')}", flush=True)
    encode_multi = col.encode_images_multi
    toks = encode_multi([p[1] for p in pages], batch_size=1)
    print(f"encoded {len(toks)} pages", flush=True)
    questions = [r["query"] for r in rows]
    gold_idx = [by[r["page_id"]] for r in rows]
    q_toks = col.encode_queries_multi(questions, batch_size=8)
    scores = maxsim_matrix(q_toks, [np.asarray(t, dtype=np.float32) for t in toks])
    metrics = _block(scores, gold_idx)
    report = {
        "protocol": "vidore_v1_page_retrieval",
        "dataset": args.dataset,
        "split": args.split,
        "task": TASK_NAME.get(args.dataset) or args.dataset.split("/")[-1],
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
            "ColPali paper nDCG@5 81.3 is the 10-task ViDoRe v1 average, not this subset alone."
        ),
    }
    dest = ROOT / "evals" / "vidore_latest.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(
        f"protocol=vidore_v1 dataset={args.dataset} pages={len(pages)} queries={len(rows)} "
        f"nDCG@5={metrics['ndcg@5']} R@1/5/10={metrics['recall@1']}/{metrics['recall@5']}/{metrics['recall@10']}"
    )


if __name__ == "__main__":
    main()
