"""MCP tools: literature search, RCSB cache fetch, overlap lookup."""

from __future__ import annotations

import json
from pathlib import Path

from abcite_core.overlap import lookup_predicted_overlap
from abcite_core.rcsb import RcsbError, fetch_entry

try:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("abcite")
except Exception:  # pragma: no cover
    mcp = None


def literature_search(query: str, corpus_path: str = "corpus/chunks.jsonl") -> list[dict]:
    hits = []
    path = Path(corpus_path)
    if not path.exists():
        return [{"error": "corpus missing", "query": query}]
    q = query.lower()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        blob = f"{row.get('title','')} {row.get('text','')}".lower()
        if any(tok in blob for tok in q.split() if len(tok) > 3):
            hits.append({"chunk_id": row["chunk_id"], "title": row["title"], "modality": row["modality"]})
        if len(hits) >= 8:
            break
    return hits


def pdb_fetch(pdb_id: str) -> dict:
    try:
        meta = fetch_entry(pdb_id)
        meta.setdefault("files", {
            "pdb": f"https://files.rcsb.org/download/{meta['pdb_id']}.pdb",
            "cif": f"https://files.rcsb.org/download/{meta['pdb_id']}.cif",
        })
        return meta
    except RcsbError as exc:
        pid = pdb_id.upper()
        return {
            "pdb_id": pid,
            "error": str(exc),
            "rcsb": f"https://www.rcsb.org/structure/{pid}",
        }


def overlap_lookup(condition: str = "framecontact_joint_a0_pair_only") -> dict:
    return lookup_predicted_overlap(condition)


if mcp is not None:

    @mcp.tool()
    def search_literature(query: str) -> list[dict]:
        """Search indexed antibody literature chunks."""
        return literature_search(query)

    @mcp.tool()
    def fetch_pdb(pdb_id: str) -> dict:
        """Return cached RCSB metadata (live fetch only if ABCITE_RCSB_LIVE=1)."""
        return pdb_fetch(pdb_id)

    @mcp.tool()
    def lookup_overlap(condition: str = "framecontact_joint_a0_pair_only") -> dict:
        """Public FrameContact same-antigen predicted-overlap diagnostic numbers."""
        return overlap_lookup(condition)


def main() -> None:
    if mcp is None:
        raise SystemExit("mcp package not installed")
    mcp.run()


if __name__ == "__main__":
    main()
