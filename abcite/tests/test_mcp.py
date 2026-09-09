from mcp.server import literature_search, overlap_lookup, pdb_fetch


def test_mcp_pdb_and_overlap():
    meta = pdb_fetch("1n8z")
    assert meta["pdb_id"] == "1N8Z"
    assert "HER2" in (meta.get("title") or "")
    ov = overlap_lookup()
    assert ov["true_epitope_jaccard_mean"] < 0.1
    assert "not an official-test" in ov["claim_boundary"]


def test_mcp_literature(tmp_path, monkeypatch):
    hits = literature_search("trastuzumab HER2")
    assert any("1n8z" in h["chunk_id"] for h in hits)
