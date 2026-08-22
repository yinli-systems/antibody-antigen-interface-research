from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_string_v12_supplemental_cluster_pairs.py"


def module():
    spec = importlib.util.spec_from_file_location("string_supp_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def candidates():
    return [
        {
            "combined_maximum_exclusive": 700,
            "combined_minimum_inclusive": 650,
            "direct_channel_maximum_minimum_inclusive": 500,
            "selection_rank": 1,
        },
        {
            "combined_maximum_exclusive": 700,
            "combined_minimum_inclusive": 600,
            "direct_channel_maximum_minimum_inclusive": 300,
            "selection_rank": 2,
        },
    ]


def test_candidate_mask_respects_both_frozen_thresholds() -> None:
    value = module()
    assert value.candidate_mask(660, 550, candidates()) == 0b11
    assert value.candidate_mask(620, 400, candidates()) == 0b10
    assert value.candidate_mask(700, 900, candidates()) == 0


def test_audit_is_conservative_and_deduplicates_cluster_pairs(tmp_path: Path, monkeypatch) -> None:
    value = module()
    rows = [
        "protein1 protein2 homology experiments experiments_transferred database database_transferred textmining textmining_transferred combined_score\n",
        "a b 0 550 0 0 0 0 0 660\n",
        "a2 b 0 400 0 0 0 0 0 620\n",
        "a c 0 600 0 0 0 0 0 660\n",
        "a d 0 600 0 0 0 0 0 660\n",
        "missing b 0 600 0 0 0 0 0 660\n",
    ]
    monkeypatch.setattr(value, "zstd_lines", lambda _: iter(rows))
    counts, per_candidate = value.audit(
        Path("unused"),
        {"a": "A", "a2": "A", "b": "B", "c": "C", "d": "D"},
        {("A", "C")},
        {"D"},
        candidates(),
    )
    assert per_candidate == [1, 1]
    assert counts == {
        "candidate_edges": 5,
        "current_endpoint_edges": 4,
        "development_excluded_edges": 1,
        "missing_endpoint_edges": 1,
        "novel_cluster_pairs_in_broad_universe": 1,
        "self_cluster_edges": 0,
        "tier1_overlap_edges": 1,
    }
