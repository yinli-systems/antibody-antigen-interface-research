"""Download real OA paper figures (PMC Cloud / Europe PMC / NCBI fallback).

Fast path (2026): AWS Open Data bucket ``pmc-oa-opendata`` (us-east-1),
anonymous HTTPS — not per-article NCBI ``/pmc/articles/instance/.../bin``
and not E-utilities per figure.

Usage:
  python evals/download_figures.py --max 16000 --source pmc --workers 12
  python evals/download_figures.py --max 20 --fixture-only
  python evals/download_figures.py --max 16000 --source scimmir

SciMMIR/HuggingFace is opt-in (--source scimmir). Default pmc skips HF.

Cache: abcite/evals/figures/cache/ (gitignored).
Tiny fixture: abcite/evals/figures/fixture/ (committed, 8–16 PNGs).
License: PMC Open Access (see LICENSE.md). Resume-able via metadata.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
FIG_ROOT = ROOT / "evals" / "figures"
CACHE = FIG_ROOT / "cache"
FIXTURE = FIG_ROOT / "fixture"
UA = "abcite-figure-downloader/1.2 (research eval; mailto:abcite@local)"
CTX = ssl.create_default_context()
# NCBI PMC Cloud on AWS Open Data (no login). Legacy oa_bulk FTP packages
# were removed the week of 2026-08-24.
S3_HTTPS = "https://pmc-oa-opendata.s3.amazonaws.com"
_COMMERCIAL_LICENSES = frozenset({"CC0", "CC BY", "CC BY-SA", "CC BY-ND"})

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

# Multiple OA journal/year bins so NCBI/EuropePMC rate limits on one query
# cannot starve the gallery.
_OA_BINS = [
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2015-01-01 TO 2015-06-30]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2015-07-01 TO 2015-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2016-01-01 TO 2016-06-30]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2016-07-01 TO 2016-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2017-01-01 TO 2017-06-30]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2017-07-01 TO 2017-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2018-01-01 TO 2018-06-30]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2018-07-01 TO 2018-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2019-01-01 TO 2019-06-30]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2019-07-01 TO 2019-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2020-01-01 TO 2020-06-30]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2020-07-01 TO 2020-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2021-01-01 TO 2021-06-30]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2021-07-01 TO 2021-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2022-01-01 TO 2022-06-30]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2022-07-01 TO 2022-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS ONE" AND FIRST_PDATE:[2023-01-01 TO 2023-06-30]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"Scientific Reports" AND FIRST_PDATE:[2018-01-01 TO 2018-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"Scientific Reports" AND FIRST_PDATE:[2019-01-01 TO 2019-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"Scientific Reports" AND FIRST_PDATE:[2020-01-01 TO 2020-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"Scientific Reports" AND FIRST_PDATE:[2021-01-01 TO 2021-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"Scientific Reports" AND FIRST_PDATE:[2022-01-01 TO 2022-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"PLoS Pathogens" AND FIRST_PDATE:[2018-01-01 TO 2022-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"Frontiers in Immunology" AND FIRST_PDATE:[2019-01-01 TO 2021-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"Nature Communications" AND FIRST_PDATE:[2019-01-01 TO 2020-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND (antibody OR epitope OR immunoglobulin) AND FIRST_PDATE:[2019-01-01 TO 2021-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"BMC Biology" AND FIRST_PDATE:[2018-01-01 TO 2022-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"eLife" AND FIRST_PDATE:[2019-01-01 TO 2021-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"BMC Genomics" AND FIRST_PDATE:[2018-01-01 TO 2022-12-31]',
    'OPEN_ACCESS:y AND HAS_FT:y AND JOURNAL:"Frontiers in Microbiology" AND FIRST_PDATE:[2019-01-01 TO 2021-12-31]',
]

_NCBI_TERMS = [
    '("PLoS One"[journal]) AND 2018[dp] AND open access[filter]',
    '("PLoS One"[journal]) AND 2019[dp] AND open access[filter]',
    '("PLoS One"[journal]) AND 2020[dp] AND open access[filter]',
    '("PLoS One"[journal]) AND 2021[dp] AND open access[filter]',
    '("Sci Rep"[journal]) AND 2019[dp] AND open access[filter]',
    '("Sci Rep"[journal]) AND 2020[dp] AND open access[filter]',
    '("PLoS Pathog"[journal]) AND antibody AND open access[filter]',
    '("Front Immunol"[journal]) AND 2020[dp] AND open access[filter]',
    '("Nat Commun"[journal]) AND 2019[dp] AND open access[filter]',
    '("Elife"[journal]) AND 2020[dp] AND open access[filter]',
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


def _get(url: str, timeout: int = 25, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            code = int(getattr(exc, "code", 0) or 0)
            if code == 404:
                raise
            if code in {429, 500, 502, 503, 504}:
                delay = min(20.0, (1.5**attempt) + random.random())
                print(f"  HTTP {code} retry {attempt + 1}/{retries} sleep={delay:.1f}s {url[:80]}", flush=True)
                time.sleep(delay)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, TimeoutError, OSError) as exc:
            last = exc
            delay = min(20.0, (1.5**attempt) + random.random())
            print(f"  net fail retry {attempt + 1}/{retries} {exc} sleep={delay:.1f}s", flush=True)
            time.sleep(delay)
    assert last is not None
    raise last


def _europepmc_search(query: str, retmax: int = 200) -> list[str]:
    ids: list[str] = []
    cursor = "*"
    while len(ids) < retmax:
        page = min(1000, retmax - len(ids))
        q = urllib.parse.urlencode(
            {
                "query": query,
                "resultType": "lite",
                "pageSize": page,
                "format": "json",
                "cursorMark": cursor,
            }
        )
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{q}"
        try:
            payload = json.loads(_get(url, timeout=45).decode("utf-8"))
        except Exception as exc:
            print(f"europepmc search failed: {exc}", flush=True)
            break
        hits = (payload.get("resultList") or {}).get("result") or []
        if not hits:
            break
        for hit in hits:
            pmcid = hit.get("pmcid") or ""
            if pmcid:
                ids.append(str(pmcid) if str(pmcid).startswith("PMC") else f"PMC{pmcid}")
        nxt = payload.get("nextCursorMark")
        if not nxt or nxt == cursor:
            break
        cursor = nxt
        time.sleep(0.25)
    seen: set[str] = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _esearch_pmc(term: str, retmax: int = 200) -> list[str]:
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
            payload = json.loads(_get(url, timeout=40).decode("utf-8"))
        except Exception as exc:
            print(f"esearch failed: {exc}", flush=True)
            break
        batch = payload.get("esearchresult", {}).get("idlist") or []
        if not batch:
            break
        ids.extend(batch)
        retstart += len(batch)
        time.sleep(0.4)
    return [f"PMC{i}" if not str(i).startswith("PMC") else str(i) for i in ids]


def _parse_fig_xml(xml: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for fig in re.finditer(r"<fig[\s\S]*?</fig>", xml, flags=re.I):
        block = fig.group(0)
        cap_m = re.search(r"<caption[\s\S]*?</caption>", block, flags=re.I)
        cap = re.sub(r"<[^>]+>", " ", cap_m.group(0) if cap_m else "")
        cap = " ".join(cap.split())
        hrefs = re.findall(r'(?:xlink:href|href)="([^"]+)"', block)
        for h in hrefs:
            low = h.lower()
            if any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif")):
                out.append((h.split("/")[-1], cap[:2000]))
    if not out:
        for h in re.findall(r'(?:xlink:href|href)="([^"]+\.(?:jpg|jpeg|png|gif))"', xml, flags=re.I):
            name = h.split("/")[-1]
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                out.append((name, ""))
    # de-dupe
    seen: set[str] = set()
    uniq = []
    for name, cap in out:
        if name in seen:
            continue
        seen.add(name)
        uniq.append((name, cap))
    return uniq[:32]


def _s3_key_to_https(url_or_key: str) -> str:
    raw = url_or_key.strip()
    if raw.startswith("s3://pmc-oa-opendata/"):
        raw = raw[len("s3://pmc-oa-opendata/") :]
    if raw.startswith("https://"):
        return raw.split("?", 1)[0]
    return f"{S3_HTTPS}/{raw.split('?', 1)[0]}"


def _s3_latest_prefix(pmcid: str) -> str | None:
    pmcid = pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"
    url = f"{S3_HTTPS}/?list-type=2&prefix={pmcid}.&delimiter=/"
    try:
        xml = _get(url, timeout=20, retries=2).decode("utf-8", errors="ignore")
    except Exception as exc:
        print(f"  s3 list fail {pmcid}: {exc}", flush=True)
        return None
    prefixes = re.findall(r"<Prefix>(PMC\d+\.\d+)/</Prefix>", xml)
    if not prefixes:
        return None
    prefixes.sort(key=lambda p: int(p.rsplit(".", 1)[-1]))
    return prefixes[-1]


def _s3_article_meta(prefix: str) -> dict | None:
    url = f"{S3_HTTPS}/metadata/{prefix}.json"
    try:
        return json.loads(_get(url, timeout=20, retries=2).decode("utf-8"))
    except Exception as exc:
        print(f"  s3 meta fail {prefix}: {exc}", flush=True)
        return None


def _pmc_figures_from_xml(pmcid: str, *, prefix: str | None = None) -> list[tuple[str, str]]:
    urls: list[str] = []
    if prefix:
        urls.append(f"{S3_HTTPS}/{prefix}/{prefix}.xml")
    numeric = pmcid.replace("PMC", "")
    urls.extend(
        [
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={numeric}&retmode=xml",
            f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{numeric}&metadataPrefix=pmc",
        ]
    )
    for url in urls:
        try:
            xml = _get(url, timeout=25, retries=2).decode("utf-8", errors="ignore")
        except Exception as exc:
            print(f"  xml fail {pmcid}: {exc}", flush=True)
            continue
        figs = _parse_fig_xml(xml)
        if figs:
            return figs
    return []


def _download_pmc_graphic(
    pmcid: str,
    href: str,
    dest: Path,
    *,
    prefix: str | None = None,
    media_https: list[str] | None = None,
) -> bool:
    name = href.split("/")[-1].split("?")[0]
    numeric = pmcid.replace("PMC", "")
    urls: list[str] = []
    if media_https:
        for u in media_https:
            if u.rsplit("/", 1)[-1].split("?")[0] == name:
                urls.append(u)
    if prefix:
        urls.append(f"{S3_HTTPS}/{prefix}/{name}")
    urls.extend(
        [
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/instance/{numeric}/bin/{name}",
            f"https://europepmc.org/backend/ptpmcrender.fcgi?acc={pmcid}&blobtype=image&blobname={urllib.parse.quote(name)}",
            f"https://journals.plos.org/plosone/article/figure/image?size=medium&id=10.1371/journal.pone.{name.split('.')[1]}.{name.split('.')[2]}"
            if name.startswith("pone.") and len(name.split(".")) >= 3
            else "",
        ]
    )
    urls = [u for u in urls if u]
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        try:
            raw = _get(url, timeout=18, retries=2)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
            continue
        if _save_image_bytes(dest, raw):
            return True
    return False


def _search_pmc_ids(need: int) -> list[str]:
    cache_ids = CACHE / "pmc_ids.json"
    ids: list[str] = []
    if cache_ids.exists():
        try:
            ids = json.loads(cache_ids.read_text(encoding="utf-8"))
            print(f"  reused {len(ids)} cached PMC ids", flush=True)
        except Exception:
            ids = []
    per = max(120, need // max(1, len(_OA_BINS)))
    if len(ids) < need:
        for query in _OA_BINS:
            if len(ids) >= need:
                break
            print(f"  europepmc bin: {query[:70]}…", flush=True)
            ids.extend(_europepmc_search(query, retmax=per))
            time.sleep(0.2)
    if len(ids) < need // 2:
        per_n = max(40, need // max(1, len(_NCBI_TERMS)))
        for term in _NCBI_TERMS:
            if len(ids) >= need:
                break
            print(f"  ncbi bin: {term[:70]}…", flush=True)
            ids.extend(_esearch_pmc(term, retmax=per_n))
            time.sleep(0.4)
    seen: set[str] = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_ids.write_text(json.dumps(out), encoding="utf-8")
    return out[: max(need, len(out))]


def _next_pmc_index(image_dir: Path, existing: set[str]) -> int:
    nums = [0]
    for fid in existing:
        if fid.startswith("pmc-") and fid.split("-")[-1].isdigit():
            nums.append(int(fid.split("-")[-1]))
    for p in image_dir.glob("pmc-*.jpg"):
        tail = p.stem.split("-")[-1]
        if tail.isdigit():
            nums.append(int(tail))
    return max(nums)


def _harvest_paper(
    pmcid: str, *, commercial_only: bool
) -> tuple[str, str, str | None, str, list[str], list[tuple[str, str]]]:
    prefix = _s3_latest_prefix(pmcid)
    meta = _s3_article_meta(prefix) if prefix else None
    license_code = str((meta or {}).get("license_code") or "")
    if commercial_only and license_code and license_code.upper() not in _COMMERCIAL_LICENSES:
        return pmcid, "skip-license", prefix, license_code, [], []
    media_https: list[str] = []
    if meta:
        for u in meta.get("media_urls") or []:
            https = _s3_key_to_https(str(u))
            low = https.lower()
            if any(low.split("?")[0].endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif")):
                media_https.append(https)
    figs = _pmc_figures_from_xml(pmcid, prefix=prefix)
    if not figs and media_https:
        figs = [(u.rsplit("/", 1)[-1].split("?")[0], "") for u in media_https]
    status = "ok" if figs else "empty"
    return pmcid, status, prefix, license_code, media_https, figs


def _try_pmc(
    max_n: int,
    image_dir: Path,
    existing: set[str],
    start_i: int,
    *,
    workers: int = 8,
    commercial_only: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    pmc_ids = list(_SEED_PMC)
    extra = _search_pmc_ids(max(2500, (max_n * 2) // 5))
    for p in extra:
        if p not in pmc_ids:
            pmc_ids.append(p)
    n = start_i
    paper_workers = max(1, min(8, workers))
    print(
        f"PMC OA papers queued: {len(pmc_ids)} (S3 first, fig_workers={workers}, "
        f"paper_workers={paper_workers})",
        flush=True,
    )
    accumulated = _load_meta(CACHE / "metadata.jsonl")
    done_pmc = {str(r.get("pmcid") or "") for r in accumulated}
    skip_path = CACHE / "skip_pmc.json"
    skip_pmc: set[str] = set()
    if skip_path.exists():
        try:
            skip_pmc = set(json.loads(skip_path.read_text(encoding="utf-8")))
        except Exception:
            skip_pmc = set()
    lock = Lock()

    def _one_fig(
        item: tuple[int, str, str, str, str | None, list[str]],
    ) -> tuple[int, str, str, Path, bool]:
        idx, paper_id, href, cap, prefix, media = item
        dest = image_dir / f"pmc-{idx:05d}.jpg"
        ok = _download_pmc_graphic(paper_id, href, dest, prefix=prefix, media_https=media)
        if not ok:
            dest.unlink(missing_ok=True)
        return idx, href, cap, dest, ok

    todo = [p for p in pmc_ids if p not in done_pmc and p not in skip_pmc]
    fig_pool = ThreadPoolExecutor(max_workers=max(1, workers))
    fig_futs: dict = {}
    pending_meta: dict[int, tuple[str, str, str, str | None]] = {}

    def _flush_figs() -> None:
        nonlocal accumulated
        done = [f for f in list(fig_futs) if f.done()]
        if not done:
            return
        got_by_paper: dict[str, int] = {}
        for fut in done:
            pmcid_job = fig_futs.pop(fut)
            del pmcid_job
            try:
                idx, href, cap, dest, ok = fut.result()
            except Exception as exc:
                print(f"  fig fail {exc}", flush=True)
                continue
            meta_row = pending_meta.pop(idx, None)
            if not ok or meta_row is None:
                continue
            paper_id, license_code, prefix, _cap = meta_row
            fid = f"pmc-{idx:05d}"
            visual = _visual_class(cap or href)
            rec = {
                "figure_id": fid,
                "path": str(dest.relative_to(FIG_ROOT)),
                "pmcid": paper_id,
                "caption": cap,
                "split": "oa",
                "visual_class": visual,
                "source": "pmc-oa-s3" if prefix else "pmc-oa",
                "href": href,
                "license_code": license_code or None,
                "s3_prefix": prefix,
            }
            with lock:
                rows.append(rec)
                accumulated.append(rec)
                existing.add(fid)
            got_by_paper[paper_id] = got_by_paper.get(paper_id, 0) + 1
        if got_by_paper:
            with lock:
                _write_meta(CACHE / "metadata.jsonl", accumulated)
            total = len(accumulated)
            last = next(iter(got_by_paper))
            print(f"  PMC +{sum(got_by_paper.values())} total={total} (last {last})", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=paper_workers) as papers:
            paper_futs = {
                papers.submit(_harvest_paper, pmcid, commercial_only=commercial_only): pmcid
                for pmcid in todo
            }
            for fut in as_completed(paper_futs):
                if len(accumulated) >= max_n:
                    break
                try:
                    pmcid, status, prefix, license_code, media_https, figs = fut.result()
                except Exception as exc:
                    print(f"  paper fail {paper_futs[fut]}: {exc}", flush=True)
                    skip_pmc.add(paper_futs[fut])
                    continue
                if status != "ok" or not figs:
                    skip_pmc.add(pmcid)
                    continue
                pending: list[tuple[int, str, str, str, str | None, list[str]]] = []
                with lock:
                    if len(accumulated) >= max_n:
                        break
                    for href, cap in figs:
                        if len(accumulated) + len(pending) + len(fig_futs) >= max_n:
                            break
                        n += 1
                        fid = f"pmc-{n:05d}"
                        if fid in existing:
                            continue
                        pending.append((n, pmcid, href, cap, prefix, media_https))
                        pending_meta[n] = (pmcid, license_code, prefix, cap)
                for item in pending:
                    fig_futs[fig_pool.submit(_one_fig, item)] = pmcid
                done_pmc.add(pmcid)
                if len(fig_futs) >= workers * 4:
                    _flush_figs()
        while fig_futs:
            _flush_figs()
            if fig_futs:
                time.sleep(0.05)
    finally:
        fig_pool.shutdown(wait=True)
        _flush_figs()
        skip_path.write_text(json.dumps(sorted(skip_pmc)), encoding="utf-8")
        _write_meta(CACHE / "metadata.jsonl", accumulated)
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _save_image_bytes(dest: Path, raw: bytes) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(raw) < 800:
        return False
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
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    print("Streaming m-a-p/SciMMIR (HF, streaming=True; hf_transfer if installed)…", flush=True)
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
    existing_fix = list(FIXTURE.glob("fix-*"))
    if len(existing_fix) >= 8:
        print(f"Keeping existing fixture n={len(existing_fix)}", flush=True)
        return
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
    ap.add_argument("--max", type=int, default=16000)
    ap.add_argument("--fixture-only", action="store_true")
    ap.add_argument("--source", choices=["pmc", "scimmir", "auto"], default="pmc")
    ap.add_argument("--workers", type=int, default=8, help="parallel S3/HTTPS figure fetches")
    ap.add_argument(
        "--commercial-only",
        action="store_true",
        help="skip CC-BY-NC / TDM (keep CC0, CC BY, CC BY-SA, CC BY-ND)",
    )
    args = ap.parse_args()
    image_dir = CACHE / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE / "metadata.jsonl"
    rows = _load_meta(meta_path)
    existing = {r["figure_id"] for r in rows}
    target = min(args.max, 20) if args.fixture_only else args.max
    print(f"resume n={len(rows)} target={target} source={args.source}", flush=True)
    if len(rows) < target and args.source == "scimmir":
        rows.extend(_try_scimmir(target, image_dir, existing))
        _write_meta(meta_path, rows)
    if len(rows) < target and args.source == "auto":
        print("source=auto: PMC OA first; SciMMIR only if still short and datasets is installed", flush=True)
    if len(rows) < target and args.source in {"auto", "pmc"}:
        start = _next_pmc_index(image_dir, existing)
        rows.extend(
            _try_pmc(
                target,
                image_dir,
                existing,
                start_i=start,
                workers=max(1, args.workers),
                commercial_only=args.commercial_only,
            )
        )
        _write_meta(meta_path, rows)
    if len(rows) < target and args.source == "auto":
        rows.extend(_try_scimmir(target, image_dir, existing))
        _write_meta(meta_path, rows)
    _write_meta(meta_path, rows)
    print(f"Indexed metadata n={len(rows)} at {meta_path}", flush=True)
    if rows:
        _copy_fixture(rows, limit=12)
    if len(rows) < 1000:
        print(
            "NOTE: fewer than 1000 figures in this run. Re-run:\n"
            "  python evals/download_figures.py --max 16000 --source pmc --workers 12\n"
            "SciMMIR is opt-in: --source scimmir (needs `datasets`).",
            flush=True,
        )
    elif len(rows) < 16000:
        print(
            f"NOTE: indexed n={len(rows)} < 16000. Resume with:\n"
            "  python evals/download_figures.py --max 16000 --source pmc --workers 12",
            flush=True,
        )


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
