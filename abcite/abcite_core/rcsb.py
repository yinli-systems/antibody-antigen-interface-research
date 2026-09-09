"""RCSB metadata + mmCIF header fetch with on-disk cache.

Live HTTP only when ABCITE_RCSB_LIVE=1. Tests use bundled cache and never require network.
Allowlist keeps the demo scoped to complexes already in the corpus.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

ALLOWLIST = frozenset(
    {"1N8Z", "4JHW", "3NGB", "2NY7", "6W41", "1VFB", "1S78", "7C01", "1HZH", "4FQI", "3IXT"}
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "corpus" / "rcsb_cache"
ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


class RcsbError(RuntimeError):
    pass


def cache_dir() -> Path:
    override = os.environ.get("ABCITE_RCSB_CACHE")
    return Path(override) if override else DEFAULT_CACHE


def fetch_entry(pdb_id: str, *, live: bool | None = None, client: httpx.Client | None = None) -> dict:
    pid = pdb_id.upper()
    if pid not in ALLOWLIST:
        raise RcsbError(f"{pid} is not on the AbCite RCSB allowlist")
    dest = cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    json_path = dest / f"{pid}.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        header = dest / f"{pid}.cif.header"
        if header.exists():
            payload["cif_header_path"] = str(header)
        payload["cache_hit"] = True
        payload["live"] = False
        payload.setdefault("pdb_id", pid)
        payload.setdefault("rcsb", f"https://www.rcsb.org/structure/{pid}")
        return payload

    use_live = os.environ.get("ABCITE_RCSB_LIVE", "0") in {"1", "true", "TRUE"} if live is None else live
    if not use_live:
        raise RcsbError(f"no cache for {pid} and live fetch disabled")

    http = client or httpx.Client(timeout=30.0)
    r = http.get(ENTRY_URL.format(pdb_id=pid))
    r.raise_for_status()
    entry = r.json()
    cif_header = ""
    try:
        cif = http.get(CIF_URL.format(pdb_id=pid), headers={"Range": "bytes=0-2047"})
        if cif.status_code in {200, 206}:
            cif_header = cif.text[:2048]
            (dest / f"{pid}.cif.header").write_text(cif_header, encoding="utf-8")
    except httpx.HTTPError:
        cif_header = ""

    payload = {
        "pdb_id": pid,
        "rcsb_id": entry.get("rcsb_id", pid),
        "title": (entry.get("struct") or {}).get("title") or entry.get("struct_title"),
        "method": _method(entry),
        "released": (entry.get("rcsb_accession_info") or {}).get("initial_release_date"),
        "resolution": _resolution(entry),
        "rcsb": f"https://www.rcsb.org/structure/{pid}",
        "cif_header_path": str(dest / f"{pid}.cif.header") if cif_header else None,
        "source": "rcsb_rest_v1",
        "live": True,
        "cache_hit": False,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _method(entry: dict) -> str:
    expt = entry.get("exptl") or []
    if expt and isinstance(expt, list):
        return str(expt[0].get("method") or "unknown")
    return "unknown"


def _resolution(entry: dict) -> float | None:
    refine = entry.get("refine") or []
    if refine and isinstance(refine, list):
        val = refine[0].get("ls_d_res_high")
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None
    return None
