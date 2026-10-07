from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCREEN_PATH = (
    ROOT
    / "remote_tools/prospective_temporal_double_cold_v2"
    / "screen_sabdab2_release_metadata_v2.py"
)
FETCH_PATH = (
    ROOT
    / "remote_tools/prospective_temporal_double_cold_v2"
    / "fetch_rcsb_release_metadata_v2_1.py"
)


def load_screen():
    spec = importlib.util.spec_from_file_location("temporal_screen_v2", SCREEN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_fetcher():
    spec = importlib.util.spec_from_file_location("temporal_fetch_v2_1", FETCH_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_date_parser_accepts_official_formats() -> None:
    module = load_screen()
    assert module.parse_release_date("2026/09/02").isoformat() == "2026-09-02"
    assert module.parse_release_date("2026-09-02").isoformat() == "2026-09-02"
    assert module.parse_release_date("20260902").isoformat() == "2026-09-02"


def test_release_date_parser_keeps_missing_values_ineligible() -> None:
    module = load_screen()
    assert module.parse_release_date("NA") is None
    assert module.parse_release_date("") is None


def test_antigen_prefilter_is_protein_or_peptide_only() -> None:
    module = load_screen()
    assert module.has_amino_acid_antigen("PROTEIN")
    assert module.has_amino_acid_antigen("ION|PEPTIDE")
    assert not module.has_amino_acid_antigen("HAPTEN|ION")


def test_sabdab_canonical_id_maps_to_legacy_rcsb_id() -> None:
    module = load_fetcher()
    assert module.rcsb_query_id("pdb_000010bt") == "10BT"
    assert module.rcsb_query_id("10bt") == "10BT"


def test_true_extended_pdb_id_is_preserved() -> None:
    module = load_fetcher()
    assert module.rcsb_query_id("pdb_100010bt") == "PDB_100010BT"


def test_rcsb_release_timestamp_is_reduced_to_utc_date() -> None:
    module = load_fetcher()
    assert module.parse_initial_release_date("2026-06-10T00:00:00Z") == "2026-06-10"
