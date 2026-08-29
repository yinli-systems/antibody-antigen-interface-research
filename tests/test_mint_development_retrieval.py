from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "run_mint_frozen_development_retrieval",
        ROOT / "scripts/run_mint_frozen_development_retrieval.py",
    )
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_endpoint_windows_are_deterministic_and_cover_both_ends() -> None:
    value = module()
    short = "A" * 510
    long = "A" * 510 + "C" * 10
    assert value.endpoint_windows(short) == [short]
    windows = value.endpoint_windows(long)
    assert len(windows) == 3
    assert windows[0] == long[:510]
    assert windows[-1] == long[-510:]
    assert all(len(window) == 510 for window in windows)


def test_average_tie_rank_matches_frozen_definition() -> None:
    value = module()
    assert value.average_tie_rank(0.5, [0.5, 0.4, 0.3]) == 1.0
    assert value.average_tie_rank(0.5, [0.5, 0.7, 0.5, 0.1]) == 2.5


def test_batch_planner_respects_batch_and_attention_limits() -> None:
    value = module()
    work = [(("ordered", str(i), str(i)), "A" * length, "C" * length) for i, length in enumerate((10, 20, 30, 40, 50))]
    batches = value.batch_work(work, maximum_batch_size=2, maximum_attention_cells=50_000)
    assert sum(map(len, batches)) == len(work)
    assert all(len(batch) <= 2 for batch in batches)
    for batch in batches:
        maximum = max(len(item[1]) + len(item[2]) + 4 for item in batch)
        assert maximum * maximum * len(batch) <= 50_000


def test_primary_metric_groups_duplicate_structure_instances() -> None:
    value = module()
    rows = [
        {"canonical_pair_sha256": "pair-a", "metric": 1.0},
        {"canonical_pair_sha256": "pair-a", "metric": 0.0},
        {"canonical_pair_sha256": "pair-b", "metric": 1.0},
    ]
    summary = value.summarize_anchor_metrics(rows, "metric")
    assert summary["canonical_exact_pair_macro"] == 0.75
    assert summary["instance_macro_secondary"] == 2 / 3
    assert summary["unique_canonical_exact_pair_count"] == 2


def test_smoke_anchor_pairs_can_reference_full_authority_candidates() -> None:
    value = module()
    anchor = {
        "instance_id": "anchor",
        "source_antibody_sequence_sha256": "ab-a",
        "source_antigen_sequence_sha256": "ag-a",
        "antibody_endpoint_sequence": "AC",
        "antigen_sequence": "DE",
        "canonical_pair_sha256": "pair-a",
        "wrong_antibody_instance_ids": ["candidate"],
        "wrong_antigen_instance_ids": ["candidate"],
    }
    candidate = {
        **anchor,
        "instance_id": "candidate",
        "source_antibody_sequence_sha256": "ab-b",
        "source_antigen_sequence_sha256": "ag-b",
        "antibody_endpoint_sequence": "FG",
        "antigen_sequence": "HI",
        "canonical_pair_sha256": "pair-b",
    }
    pairs, index, _ = value.retrieval_pairs([anchor], [anchor, candidate])
    assert set(index) == {"anchor", "candidate"}
    assert ("ordered", "ab-b", "ag-a") in pairs
    assert ("ordered", "ab-a", "ag-b") in pairs
