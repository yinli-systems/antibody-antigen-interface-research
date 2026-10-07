from pathlib import Path

from abcite_core.rcsb import ALLOWLIST, RcsbError, fetch_entry
from mcp.server import pdb_fetch


def test_rcsb_cache_offline():
    meta = fetch_entry("1n8z")
    assert meta["pdb_id"] == "1N8Z"
    assert meta["cache_hit"] is True
    assert meta["live"] is False
    assert "HER2" in (meta.get("title") or "")
    assert "1N8Z.cif.header" in (meta.get("cif_header_path") or "")


def test_rcsb_allowlist():
    try:
        fetch_entry("9ZZZ")
        raise AssertionError("should reject")
    except RcsbError as exc:
        assert "allowlist" in str(exc).lower()
    assert "1N8Z" in ALLOWLIST


def test_rcsb_live_mocked(tmp_path, monkeypatch):
    monkeypatch.setenv("ABCITE_RCSB_CACHE", str(tmp_path))

    class Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "rcsb_id": "1N8Z",
                "struct": {"title": "mock HER2-Herceptin"},
                "exptl": [{"method": "X-RAY DIFFRACTION"}],
                "refine": [{"ls_d_res_high": 2.4}],
                "rcsb_accession_info": {"initial_release_date": "2003-02-11"},
            }

        @property
        def text(self):
            return "data_1N8Z\n_entry.id 1N8Z\n"

    class Client:
        def get(self, url, headers=None):
            return Resp()

    meta = fetch_entry("1N8Z", live=True, client=Client())
    assert meta["live"] is True
    assert meta["title"] == "mock HER2-Herceptin"
    assert (tmp_path / "1N8Z.json").exists()


def test_mcp_uses_cache():
    meta = pdb_fetch("1N8Z")
    assert meta["pdb_id"] == "1N8Z"
    assert "error" not in meta
    assert "HER2" in (meta.get("title") or "")
