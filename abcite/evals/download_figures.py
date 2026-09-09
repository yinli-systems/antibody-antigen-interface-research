"""Download real OA paper figures (SciMMIR stream or Europe PMC).

Usage:
  python evals/download_figures.py --max 1200
  python evals/download_figures.py --max 20 --fixture-only

Cache: abcite/evals/figures/cache/ (gitignored).
Tiny fixture: abcite/evals/figures/fixture/ (committed, 8–16 PNGs).
License: PMC Open Access / arXiv source papers (see LICENSE.md). Resume-able via metadata.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "evals" / "figures"
CACHE = FIG_ROOT / "cache"
FIXTURE = FIG_ROOT / "fixture"
UA = "abcite-figure-downloader/1.0 (research eval; mailto:abcite@local)"
CTX = ssl.create_default_context()

# Europe PMC OA papers that typically expose several figures.
_SEED_PMC = [
    "PMC4221156",
    "PMC4311744",
    "PMC7104295",
    "PMC7641623",
    "PMC8006103",
    "PMC8123516",
    "PMC8484734",
    "PMC8714046",
    "PMC8923475",
    "PMC9022011",
    "PMC9150046",
    "PMC9362871",
    "PMC9665892",
    "PMC10025890",
]

_VISUAL_HINTS = (
    ("blot", "western blot"),
    ("gel", "gel electrophoresis"),
    ("microscop", "microscopy"),
    ("fluorescence", "fluorescence microscopy"),
    ("ribbon", "ribbon diagram"),
    ("cartoon", "protein cartoon"),
    ("schematic", "schematic panel"),
    ("workflow", "workflow diagram"),
    ("heatmap", "heatmap"),
    ("histogram", "histogram"),
    ("scatter", "scatter plot"),
    ("plot", "scientific plot"),
    ("graph", "scientific plot"),
    ("table", "results table"),
    ("crystal", "structure snapshot"),
    ("em ", "electron microscopy"),
    ("cryo", "cryo-em"),
)


def _visual_class(caption: str) -> str:
    low = caption.lower()
    for needle, label in _VISUAL_HINTS:
        if needle in low:
            return label
    return "scientific figure panel"


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
        return resp.read()


def _esearch_pmc(term: str, retmax: int = 400) -> list[str]:
    ids: list[str] = []
    retstart = 0
    while len(ids) < retmax:
        n = min(100, retmax - len(ids))
        q = urllib.parse.urlencode(
            {
                "db": "pmc",
                "term": term,
                "retmax": n,
                "retstart": retstart,
                "retmode": "json",
            }
        )
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{q}"
        try:
            payload = json.loads(_get(url, timeout=30).decode("utf-8"))
        except Exception as exc:
            print(f"esearch failed: {exc}", flush=True)
            break
        batch = payload.get("esearchresult", {}).get("idlist") or []
        if not batch:
            break
        ids.extend(batch)
        retstart += len(batch)
        time.sleep(0.34)
    return [f"PMC{i}" if not str(i).startswith("PMC") else str(i) for i in ids]


def _pmc_figures_from_xml(pmcid: str) -> list[tuple[str, str]]:
    numeric = pmcid.replace("PMC", "")
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pmc&id={numeric}&rettype=xml"
    )
    try:
        xml = _get(url, timeout=40).decode("utf-8", errors="ignore")
    except Exception as exc:
        print(f"  xml fail {pmcid}: {exc}", flush=True)
        return []
    out: list[tuple[str, str]] = []
    for fig in re.finditer(r"<fig[\s\S]*?</fig>", xml, flags=re.I):
        block = fig.group(0)
        cap_m = re.search(r"<caption[\s\S]*?</caption>", block, flags=re.I)
        cap = re.sub(r"<[^>]+>", " ", cap_m.group(0) if cap_m else "")
        cap = " ".join(cap.split())
        for h in re.findall(r'xlink:href="([^"]+)"', block):
            low = h.lower()
            if any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif")):
                out.append((h.split("/")[-1], cap[:2000]))
    if not out:
        for h in re.findall(r'xlink:href="([^"]+\.(?:jpg|jpeg|png|gif))"', xml, flags=re.I):
            name = h.split("/")[-1]
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                out.append((name, ""))
    return out[:12]


def _download_pmc_graphic(pmcid: str, href: str, dest: Path) -> bool:
    name = href.split("/")[-1]
    numeric = pmcid.replace("PMC", "")
    urls = [
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/instance/{numeric}/bin/{name}",
    ]
    for url in urls:
        try:
            raw = _get(url, timeout=30)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            continue
        if _save_image_bytes(dest, raw):
            return True
    return False


def _search_pmc_ids(need: int) -> list[str]:
    terms = [
        '("PLoS One"[journal]) AND 2019[dp] AND open access[filter]',
        '("PLoS One"[journal]) AND 2020[dp] AND open access[filter]',
        '("Sci Rep"[journal]) AND 2020[dp] AND open access[filter]',
        '("PLoS Pathog"[journal]) AND antibody AND open access[filter]',
    ]
    ids: list[str] = []
    per = max(40, need // len(terms))
    for term in terms:
        if len(ids) >= need:
            break
        ids.extend(_esearch_pmc(term, retmax=per))
        time.sleep(0.34)
    # unique preserve order
    seen: set[str] = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:need]


def _try_pmc(max_n: int, image_dir: Path, existing: set[str], start_i: int) -> list[dict]:
    rows: list[dict] = []
    pmc_ids = list(_SEED_PMC)
    extra = _search_pmc_ids(max(400, max_n))
    for p in extra:
        if p not in pmc_ids:
            pmc_ids.append(p)
    n = start_i
    print(f"NCBI PMC OA: {len(pmc_ids)} papers", flush=True)
    accumulated = _load_meta(CACHE / "metadata.jsonl")
    for pmcid in pmc_ids:
        if len(accumulated) >= max_n:
            break
        print(f"  paper {pmcid} saved={len(accumulated)}", flush=True)
        figs = _pmc_figures_from_xml(pmcid)
        time.sleep(0.34)
        got = 0
        for href, cap in figs:
            if len(accumulated) >= max_n:
                break
            n += 1
            fid = f"pmc-{n:05d}"
            if fid in existing:
                continue
            dest = image_dir / f"{fid}.jpg"
            if not _download_pmc_graphic(pmcid, href, dest):
                dest.unlink(missing_ok=True)
                continue
            visual = _visual_class(cap or href)
            rec = {
                "figure_id": fid,
                "path": str(dest.relative_to(FIG_ROOT)),
                "pmcid": pmcid,
                "caption": cap,
                "split": "oa",
                "visual_class": visual,
                "source": "pmc-oa",
                "href": href,
            }
            rows.append(rec)
            accumulated.append(rec)
            existing.add(fid)
            got += 1
            time.sleep(0.08)
        if got:
            _write_meta(CACHE / "metadata.jsonl", accumulated)
            print(f"  PMC +{got} total={len(accumulated)} (last {pmcid})", flush=True)
    return rows


def _load_meta(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_meta(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_image_bytes(dest: Path, raw: bytes) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(raw) < 800:
        return False
    # JPEG / PNG / GIF / WEBP magic
    if not (
        raw[:3] == b"\xff\xd8\xff"
        or raw[:8] == b"\x89PNG\r\n\x1a\n"
        or raw[:6] in {b"GIF87a", b"GIF89a"}
        or raw[:4] == b"RIFF"
    ):
        return False
    dest.write_bytes(raw)
    return True


def _try_scimmir(max_n: int, image_dir: Path, existing: set[str]) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed; skip HuggingFace SciMMIR", flush=True)
        return []
    rows: list[dict] = []
    print("Streaming m-a-p/SciMMIR (HF)…", flush=True)
    try:
        ds = load_dataset("m-a-p/SciMMIR", split="test", streaming=True)
    except Exception as exc:
        print(f"SciMMIR load failed: {exc}", flush=True)
        return []
    for i, item in enumerate(ds):
        if len(rows) + len(existing) >= max_n:
            break
        fid = f"scimmir-{i:05d}"
        if fid in existing:
            continue
        cap = str(item.get("text") or item.get("caption") or "")
        cls = str(item.get("class") or _visual_class(cap))
        img = item.get("image")
        if img is None:
            continue
        dest = image_dir / f"{fid}.png"
        try:
            if hasattr(img, "convert"):
                rgb = img.convert("RGB")
                rgb.save(dest, format="PNG")
            else:
                continue
        except Exception as exc:
            print(f"  skip {fid}: {exc}", flush=True)
            continue
        if not dest.is_file() or dest.stat().st_size < 800:
            dest.unlink(missing_ok=True)
            continue
        rows.append(
            {
                "figure_id": fid,
                "path": str(dest.relative_to(FIG_ROOT)),
                "pmcid": None,
                "caption": cap[:2000],
                "split": "test",
                "visual_class": cls,
                "source": "scimmir-arxiv-oa",
            }
        )
        existing.add(fid)
        if len(rows) % 50 == 0:
            print(f"  SciMMIR saved {len(rows)}", flush=True)
    return rows


def _copy_fixture(rows: list[dict], limit: int = 12) -> None:
    FIXTURE.mkdir(parents=True, exist_ok=True)
    chosen = rows[:limit]
    out_meta = []
    for i, row in enumerate(chosen, start=1):
        src = FIG_ROOT / row["path"]
        if not src.is_file():
            continue
        dest = FIXTURE / f"fix-{i:02d}{src.suffix.lower() or '.jpg'}"
        dest.write_bytes(src.read_bytes())
        rec = dict(row)
        rec["figure_id"] = f"fix-{i:02d}"
        rec["path"] = f"fixture/{dest.name}"
        rec["split"] = "fixture"
        out_meta.append(rec)
    _write_meta(FIXTURE / "metadata.jsonl", out_meta)
    gold_path = ROOT / "evals" / "gold" / "multimodal_visual.jsonl"
    if gold_path.exists() and gold_path.stat().st_size > 100:
        print(f"Wrote fixture n={len(out_meta)} → {FIXTURE} (kept existing gold)", flush=True)
        return
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    templates = {
        "western blot": "western blot panel of protein bands with molecular-weight ladder",
        "fluorescence microscopy": "fluorescence microscopy of stained cells in a multi-channel panel",
        "ribbon diagram": "ribbon diagram of an antibody-antigen protein complex",
        "schematic panel": "schematic workflow panel of an experimental assay pipeline",
        "scatter plot": "scientific scatter plot of binding measurements with error bars",
        "electron microscopy": "electron microscopy micrograph of a macromolecular particle",
        "heatmap": "heatmap panel of expression or binding across conditions",
        "gel electrophoresis": "gel electrophoresis panel of nucleic-acid or protein lanes",
        "histogram": "histogram of a quantitative assay readout",
        "residue schematic": "cartoon schematic of residue contacts at an interface",
        "results table": "results table image from a scientific paper figure",
        "structure snapshot": "structure snapshot cartoon of a folded protein domain",
        "scientific plot": "scientific plot panel from a paper figure",
        "microscopy": "microscopy panel of cells or tissue",
        "protein cartoon": "protein cartoon schematic panel",
        "cryo-em": "cryo-em micrograph panel of a macromolecular complex",
        "scientific figure panel": "scientific figure panel from a paper",
    }
    lines = []
    for rec in out_meta:
        vis = rec.get("visual_class") or "scientific figure panel"
        q = templates.get(vis, f"{vis} panel from a paper figure")
        lines.append(
            json.dumps(
                {
                    "id": f"mv-{rec['figure_id']}",
                    "question": q,
                    "gold_figure_ids": [rec["figure_id"]],
                    "visual_class": vis,
                }
            )
        )
    gold_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote fixture n={len(out_meta)} → {FIXTURE}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=1200)
    ap.add_argument("--fixture-only", action="store_true")
    ap.add_argument("--source", choices=["auto", "scimmir", "pmc"], default="auto")
    args = ap.parse_args()
    image_dir = CACHE / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE / "metadata.jsonl"
    rows = _load_meta(meta_path)
    existing = {r["figure_id"] for r in rows}
    target = min(args.max, 20) if args.fixture_only else args.max
    print(f"resume n={len(rows)} target={target}", flush=True)
    if len(rows) < target and args.source in {"auto", "scimmir"}:
        rows.extend(_try_scimmir(target, image_dir, existing))
        _write_meta(meta_path, rows)
    if len(rows) < target and args.source in {"auto", "pmc"}:
        start = max(
            len(rows),
            max(
                [0]
                + [
                    int(p.stem.split("-")[1])
                    for p in image_dir.glob("pmc-*.jpg")
                    if p.stem.split("-")[-1].isdigit()
                ]
            ),
        )
        rows.extend(_try_pmc(target, image_dir, existing, start_i=start))
        _write_meta(meta_path, rows)
    _write_meta(meta_path, rows)
    print(f"Indexed metadata n={len(rows)} at {meta_path}", flush=True)
    if rows:
        _copy_fixture(rows, limit=12)
    if len(rows) < 1000:
        print(
            "NOTE: fewer than 1000 figures in this run. Re-run:\n"
            "  pip install datasets\n"
            "  python evals/download_figures.py --max 1200 --source auto\n"
            "HF SciMMIR is large; PMC OA is the resume-able fallback.",
            flush=True,
        )


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
