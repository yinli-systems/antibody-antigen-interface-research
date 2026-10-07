"""Query-side ColQwen2.5 LoRA on a figure-id holdout. Frozen 16k npz docs.

Offline. Does not overwrite embeddings-colqwen2.5-n15949.npz.
Does not replace the n=273 zero-shot headline with train or holdout scores.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from abcite_core.colpali_encoder import encoder_cache_name, try_load_colpali
from abcite_core.figure_gallery import FigureGallery, load_figure_records
from abcite_core.visual_qa import (
    _recall_block,
    load_visual_qa_gold,
    sanitize_visual_query,
)
from abcite_core.visual_qa_holdout import (
    HOLD_SEED,
    TRAIN_FRAC,
    negative_index_pool,
    primary_gold_id,
    split_by_figure_id,
)
from evals.run_visual_qa import _align_recs_to_cached_late_ids, _attach_late_receipt, _gpu_name

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "evals" / "visual_qa_colqwen2_5_holdout_lora_latest.json"
ADAPTER_DIR = ROOT / "evals" / "artifacts" / "colqwen25-holdout-lora"
HEADLINE_R1 = 0.1026


def _enable_lora_train(model) -> int:
    n = 0
    cfg = getattr(model, "peft_config", None)
    if isinstance(cfg, dict):
        for key, pc in cfg.items():
            if hasattr(pc, "inference_mode"):
                pc.inference_mode = False
    for name, param in model.named_parameters():
        want = "lora_" in name.lower()
        param.requires_grad = want
        if want:
            n += 1
    return n


def _encode_query_grad(encoder, text: str):
    import torch
    import torch.nn.functional as F

    batch = encoder.processor.process_queries([text]).to(encoder.device)
    emb = encoder.model(**batch)
    if hasattr(emb, "float"):
        vec = emb.float()
    else:
        vec = torch.as_tensor(emb, device=encoder.device, dtype=torch.float32)
    if vec.ndim == 3:
        vec = vec[0]
    return F.normalize(vec, dim=-1)


def _maxsim_scores(query_toks, docs: list[np.ndarray]):
    import torch

    scores = []
    for doc in docs:
        dt = torch.from_numpy(np.asarray(doc, dtype=np.float32)).to(query_toks.device)
        sims = query_toks @ dt.T
        scores.append(sims.max(dim=1).values.sum())
    return torch.stack(scores)


def _score_vision(gallery: FigureGallery, items: list[dict], encoder) -> dict:
    questions = [sanitize_visual_query(it["question"]) for it in items]
    mat = gallery.clip_score_matrix(questions)
    rows = [mat[i] for i in range(len(items))]
    block = _recall_block(rows, items, gallery)
    report = {
        "protocol": "visual_qa_holdout",
        "n_gallery": gallery.n,
        "n_answerable": len(items),
        "clip_image_only": block,
        "encoder": getattr(encoder, "name", "colqwen2.5"),
        "gpu": _gpu_name(),
    }
    _attach_late_receipt(report, gallery, encoder)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--n-neg", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=HOLD_SEED)
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("ABCITE_VLM", "0")
    os.environ.setdefault("ABCITE_COLPALI", "1")
    os.environ.setdefault("ABCITE_COLPALI_MODEL", "vidore/colqwen2.5-v0.2")

    recs = load_figure_records(prefer_cache=True)
    recs = _align_recs_to_cached_late_ids(recs, "vidore/colqwen2.5-v0.2")
    items = load_visual_qa_gold({r.figure_id for r in recs}, recs)
    split = split_by_figure_id(items, train_frac=args.train_frac, seed=args.seed)
    train_items = split["train_items"]
    hold_items = split["holdout_items"]
    hold_figs = set(split["holdout_figure_ids"])
    print(
        f"holdout split seed={args.seed} train_frac={args.train_frac} "
        f"train_q={split['n_train_queries']} train_fig={split['n_train_figures']} "
        f"hold_q={split['n_holdout_queries']} hold_fig={split['n_holdout_figures']}",
        flush=True,
    )
    if not train_items or not hold_items:
        raise SystemExit("empty train or holdout after figure-id split")

    enc, skip = try_load_colpali()
    if enc is None:
        raise SystemExit(skip or "ColQwen2.5 failed to load offline")
    if encoder_cache_name(getattr(enc, "name", "")) != "colqwen2.5":
        raise SystemExit(f"refusing encoder={getattr(enc, 'name', None)}; need colqwen2.5")

    gal = FigureGallery(recs, enc)
    if gal.image_toks is None or gal.n != 15949:
        raise SystemExit(f"expected frozen npz n=15949, got n={gal.n} toks={gal.image_toks is not None}")
    cache = gal._cache_key().with_suffix(".npz")
    if not cache.exists():
        raise SystemExit("missing embeddings-colqwen2.5-n15949.npz")
    print(f"frozen docs npz={cache.name} n={gal.n} (will not rewrite)", flush=True)

    zs = _score_vision(gal, hold_items, enc)
    zs_vis = zs["clip_image_only"]
    print(f"zero-shot holdout {zs_vis}", flush=True)

    import torch
    import torch.nn.functional as F

    n_lora = _enable_lora_train(enc.model)
    print(f"trainable LoRA tensors={n_lora}", flush=True)
    if n_lora == 0:
        raise SystemExit(
            "no trainable LoRA parameters; existing adapter is frozen and peft train failed"
        )
    opt = torch.optim.AdamW(
        [p for p in enc.model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    rng = np.random.default_rng(args.seed + 1)
    id_to_idx = {fid: i for i, fid in enumerate(gal.ids)}
    losses: list[float] = []
    enc.model.train()
    for step in range(args.steps):
        it = train_items[int(rng.integers(0, len(train_items)))]
        pos_id = primary_gold_id(it)
        if pos_id is None or pos_id not in id_to_idx:
            continue
        pos_i = id_to_idx[pos_id]
        pool = negative_index_pool(
            gal.ids, holdout_figure_ids=hold_figs, positive_id=pos_id
        )
        if len(pool) < args.n_neg:
            raise SystemExit("negative pool smaller than n_neg")
        neg_i = [int(x) for x in rng.choice(pool, size=args.n_neg, replace=False)]
        docs = [gal.image_toks[pos_i], *[gal.image_toks[j] for j in neg_i]]
        q = sanitize_visual_query(it["question"])
        opt.zero_grad(set_to_none=True)
        qtok = _encode_query_grad(enc, q)
        scores = _maxsim_scores(qtok, docs)
        loss = F.cross_entropy(
            (scores / args.tau).unsqueeze(0),
            torch.zeros(1, dtype=torch.long, device=qtok.device),
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % 32 == 0 or step + 1 == args.steps:
            print(
                f"step {step + 1}/{args.steps} loss={losses[-1]:.4f} "
                f"mean_loss={sum(losses[-32:]) / len(losses[-32:]):.4f}",
                flush=True,
            )

    enc.model.eval()
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    save = getattr(enc.model, "save_pretrained", None)
    if callable(save):
        save(str(ADAPTER_DIR))
        print(f"saved adapter {ADAPTER_DIR}", flush=True)

    ft = _score_vision(gal, hold_items, enc)
    ft_vis = ft["clip_image_only"]
    print(f"lora holdout {ft_vis}", flush=True)
    beats = float(ft_vis.get("recall@1") or 0) > float(zs_vis.get("recall@1") or 0)

    receipt = {
        "protocol": "visual_qa_holdout_lora",
        "note": (
            "Figure-id 80/20 holdout of answerable caption_bm25_should_fail items. "
            "Query-side LoRA vs frozen embeddings-colqwen2.5-n15949.npz. "
            "Held-out gold figures are excluded from the negative pool. "
            "Not ViDoRe. Does not replace the n=273 zero-shot headline."
        ),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "n_gallery": gal.n,
        "n_answerable_full": sum(1 for it in items if it.get("gold_figure_ids")),
        "n_train_figures": split["n_train_figures"],
        "n_holdout_figures": split["n_holdout_figures"],
        "n_train_queries": split["n_train_queries"],
        "n_holdout_queries": split["n_holdout_queries"],
        "holdout_figure_ids": split["holdout_figure_ids"],
        "negatives": (
            f"{args.n_neg} in-gallery samples from N={gal.n}, "
            "excluding held-out gold figure ids and the positive"
        ),
        "doc_tokens": f"{cache.name} frozen; no 16k re-encode",
        "reencoded_16k": False,
        "npz_overwritten": False,
        "steps": args.steps,
        "n_neg": args.n_neg,
        "lr": args.lr,
        "tau": args.tau,
        "trainable_lora_tensors": n_lora,
        "mean_loss_last_32": round(sum(losses[-32:]) / max(1, len(losses[-32:])), 4),
        "zero_shot_holdout": zs_vis,
        "lora_holdout": ft_vis,
        "zero_shot_maxsim": zs.get("maxsim"),
        "lora_maxsim": ft.get("maxsim"),
        "lora_beats_zero_shot_r@1": beats,
        "headline_n273_unchanged": True,
        "headline_r@1": HEADLINE_R1,
        "gpu": _gpu_name(),
        "query_prefix": getattr(getattr(enc, "processor", None), "query_prefix", None)
        or "Query: ",
        "device": str(getattr(enc, "device", "")),
        "cuda": True,
        "adapter_dir": str(ADAPTER_DIR.relative_to(ROOT)) if ADAPTER_DIR.exists() else None,
    }
    DEST.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print(
        f"wrote {DEST.name} lora_beats_r@1={beats} "
        f"headline stays {HEADLINE_R1}",
        flush=True,
    )


if __name__ == "__main__":
    main()
