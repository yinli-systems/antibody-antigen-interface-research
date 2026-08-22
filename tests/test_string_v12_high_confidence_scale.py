import importlib.util
import io
import json
from argparse import Namespace
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "collapse_string_v12_high_confidence_scale_v2.py"
)
SPEC = importlib.util.spec_from_file_location("high_confidence_scale", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_decorate_edges_filters_development_and_self_edges(tmp_path: Path) -> None:
    edges = tmp_path / "edges.txt"
    edges.write_text(
        " ".join(MODULE.EXPECTED_HEADER)
        + "\n"
        + "p1 p2 0 80 0 0 0 0 0 710\n"
        + "p2 p1 0 70 0 0 0 0 0 720\n"
        + "p1 p3 0 50 0 0 0 0 0 730\n"
        + "p4 p5 0 20 0 0 0 0 0 740\n",
        encoding="utf-8",
    )
    membership = {
        "p1": "c1",
        "p2": "c2",
        "p3": "c1",
        "p4": "c4",
        "p5": "c5",
    }
    output = io.StringIO()
    counts = MODULE.decorate_edges(edges, membership, {"c4"}, output)
    assert counts == {
        "development_excluded_edges": 1,
        "eligible_nonself_edges": 2,
        "missing_endpoint_memberships": 0,
        "self_cluster_edges_removed": 1,
        "source_edges": 4,
    }
    rows = output.getvalue().splitlines()
    assert rows[0].split("\t")[:4] == ["c1", "c2", "710", "80"]
    assert rows[1].split("\t")[:4] == ["c1", "c2", "720", "70"]


def test_decorate_edges_fails_on_missing_membership(tmp_path: Path) -> None:
    edges = tmp_path / "edges.txt"
    edges.write_text(
        " ".join(MODULE.EXPECTED_HEADER)
        + "\n"
        + "p1 missing 0 80 0 0 0 0 0 710\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing cluster memberships"):
        MODULE.decorate_edges(edges, {"p1": "c1"}, set(), io.StringIO())


def test_parse_stats_requires_exhaustive_accounting(tmp_path: Path) -> None:
    stats = tmp_path / "stats.txt"
    stats.write_text(
        "development_excluded_edges=3\n"
        "eligible_nonself_edges=10\n"
        "missing_endpoint_memberships=0\n"
        "output_cluster_pairs=7\n"
        "self_cluster_edges_removed=2\n"
        "source_edges=15\n",
        encoding="utf-8",
    )
    assert MODULE.parse_stats(stats)["output_cluster_pairs"] == 7
    stats.write_text(stats.read_text().replace("source_edges=15", "source_edges=16"))
    with pytest.raises(ValueError, match="not exhaustive"):
        MODULE.parse_stats(stats)


def test_protocol_keeps_production_training_closed() -> None:
    protocol = json.loads(
        (Path(__file__).parents[1] / "protocols" / "abpairfm_3b_scale_expansion_v2.json").read_text()
    )
    assert protocol["authorization"]["scale_pool_materialization"] is True
    assert protocol["authorization"]["production_pretraining"] is False
    assert (
        protocol["stage_a_scale_pool"]["eligible_pool_targets"][
            "minimum_endpoint_cluster_pairs"
        ]
        == 25_000_000
    )


def test_finalize_publishes_below_minimum_receipt_atomically(tmp_path: Path) -> None:
    edges = tmp_path / "edges.zst"
    clusters = tmp_path / "clusters.tsv.zst"
    hits = tmp_path / "development_hits.tsv"
    output = tmp_path / "staging.tsv.zst"
    for path, payload in (
        (edges, b"edges"),
        (clusters, b"clusters"),
        (hits, b"hits"),
        (output, b"output"),
    ):
        path.write_bytes(payload)
    edge_receipt = tmp_path / "edge.receipt.json"
    edge_receipt.write_text(
        json.dumps(
            {
                "counts": {"high_confidence_rows": 4},
                "format_version": "abpairfm_string_high_confidence_filter_v2",
                "output": {"sha256": MODULE.sha256_file(edges)},
                "status": "FILTERED_HIGH_CONFIDENCE_NOT_CLUSTERED_NOT_DECONTAMINATED",
                "test_access": False,
            }
        )
    )
    cluster_receipt = tmp_path / "cluster.receipt.json"
    cluster_receipt.write_text(
        json.dumps(
            {
                "cluster": {"membership_count": 5},
                "format_version": "abpairfm_string_high_confidence_cluster_assets_v1",
                "published_assets": [
                    {"name": clusters.name, "sha256": MODULE.sha256_file(clusters)},
                    {"name": hits.name, "sha256": MODULE.sha256_file(hits)},
                ],
                "status": "CLUSTERED_NOT_COLLAPSED_NOT_DECONTAMINATED",
                "test_access": False,
            }
        )
    )
    decorate_stats = tmp_path / "decorate.stats"
    decorate_stats.write_text(
        "cluster_count=3\n"
        "development_excluded_clusters=1\n"
        "development_excluded_edges=1\n"
        "eligible_nonself_edges=2\n"
        "membership_count=5\n"
        "missing_endpoint_memberships=0\n"
        "self_cluster_edges_removed=1\n"
        "source_edges=4\n"
    )
    final_stats = tmp_path / "final.stats"
    final_stats.write_text(
        "development_excluded_edges=1\n"
        "eligible_nonself_edges=2\n"
        "missing_endpoint_memberships=0\n"
        "self_cluster_edges_removed=1\n"
        "source_edges=4\n"
        "output_cluster_pairs=1\n"
    )
    receipt = tmp_path / "receipt.json"
    MODULE.finalize(
        Namespace(
            cluster_receipt=cluster_receipt,
            clusters=clusters,
            decorate_stats=decorate_stats,
            development_hits=hits,
            edge_receipt=edge_receipt,
            edges=edges,
            output=output,
            output_name="published.tsv.zst",
            protocol=Path(__file__).parents[1]
            / "protocols"
            / "abpairfm_3b_scale_expansion_v2.json",
            receipt=receipt,
            stats=final_stats,
        )
    )
    value = json.loads(receipt.read_text())
    assert value["output"]["name"] == "published.tsv.zst"
    assert value["scale_pool_gate"]["passed"] is False
    assert value["authorization"]["production_pretraining"] is False
