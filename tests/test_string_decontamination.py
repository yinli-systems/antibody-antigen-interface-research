from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "decontaminate_string_cluster_pairs",
        ROOT / "scripts/decontaminate_string_cluster_pairs.py",
    )
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_homology_hits_exclude_whole_endpoint_cluster(tmp_path: Path) -> None:
    value = module()
    hits = tmp_path / "hits.tsv"
    hits.write_text(
        "dev1\tmember2\t0.55\t100\t0.90\t0.85\t1e-20\t100\n"
        "dev2\tmember3\t0.70\t120\t0.95\t0.90\t1e-30\t120\n",
        encoding="utf-8",
    )
    excluded, counts = value.excluded_clusters(
        hits, {"member2": "clusterA", "member3": "clusterA"}
    )
    assert excluded == {"clusterA"}
    assert counts == {
        "development_queries_with_hits": 2,
        "endpoint_sequences_with_hits": 2,
        "excluded_endpoint_clusters": 1,
        "homology_hit_rows": 2,
    }


def test_homology_hits_fail_when_thresholds_are_not_enforced(tmp_path: Path) -> None:
    value = module()
    hits = tmp_path / "hits.tsv"
    hits.write_text(
        "dev1\tmember2\t0.49\t100\t0.90\t0.85\t1e-20\t100\n",
        encoding="utf-8",
    )
    try:
        value.excluded_clusters(hits, {"member2": "clusterA"})
    except ValueError as error:
        assert "frozen identity" in str(error)
    else:
        raise AssertionError("sub-threshold homology hit was accepted")
