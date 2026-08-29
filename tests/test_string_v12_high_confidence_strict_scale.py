from __future__ import annotations

from argparse import Namespace
import importlib.util
import io
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "collapse_string_v12_high_confidence_scale_v3.py"
)
SPEC = importlib.util.spec_from_file_location("strict_string_scale", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_strict_hits_exclude_endpoint_clusters_at_30_percent(tmp_path: Path) -> None:
    hits = tmp_path / "hits.tsv"
    hits.write_text(
        "d1\tp1\t0.30\t80\t0.80\t0.80\t1e-5\t50\n"
        "d2\tp2\t0.45\t90\t0.90\t0.85\t1e-8\t70\n",
        encoding="utf-8",
    )
    membership = {"p1": "c1", "p2": "c1", "p3": "c3"}
    assert MODULE.read_strict_excluded_clusters(hits, membership) == {"c1"}


def test_strict_hits_fail_closed_below_identity_or_coverage(tmp_path: Path) -> None:
    hits = tmp_path / "hits.tsv"
    hits.write_text("d1\tp1\t0.29\t80\t0.80\t0.80\t1e-5\t50\n")
    with pytest.raises(ValueError, match="violates the frozen homology gate"):
        MODULE.read_strict_excluded_clusters(hits, {"p1": "c1"})
    hits.write_text("d1\tp1\t0.30\t80\t0.79\t0.80\t1e-5\t50\n")
    with pytest.raises(ValueError, match="violates the frozen homology gate"):
        MODULE.read_strict_excluded_clusters(hits, {"p1": "c1"})


def test_decorate_uses_strict_excluded_clusters(tmp_path: Path) -> None:
    edges = tmp_path / "edges.txt"
    edges.write_text(
        " ".join(MODULE.base.EXPECTED_HEADER)
        + "\n"
        + "p1 p2 0 80 0 0 0 0 0 710\n"
        + "p2 p3 0 70 0 0 0 0 0 720\n",
        encoding="utf-8",
    )
    membership = {"p1": "c1", "p2": "c2", "p3": "c3"}
    output = io.StringIO()
    counts = MODULE.base.decorate_edges(edges, membership, {"c1"}, output)
    assert counts["development_excluded_edges"] == 1
    assert counts["eligible_nonself_edges"] == 1


def test_finalize_authenticates_strict_hits_and_keeps_training_closed(
    tmp_path: Path,
) -> None:
    edges = tmp_path / "edges.zst"
    clusters = tmp_path / "clusters.tsv.zst"
    hits = tmp_path / "development_hits30.tsv.zst"
    output = tmp_path / "output.tsv.zst"
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
                "output": {"sha256": MODULE.base.sha256_file(edges)},
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
                    {
                        "name": clusters.name,
                        "sha256": MODULE.base.sha256_file(clusters),
                    }
                ],
                "status": "CLUSTERED_NOT_COLLAPSED_NOT_DECONTAMINATED",
                "test_access": False,
            }
        )
    )
    strict_receipt = tmp_path / "strict.receipt.json"
    strict_receipt.write_text(
        json.dumps(
            {
                "authorization": {
                    "pair_collapse_with_strict_hits": True,
                    "production_pretraining": False,
                },
                "counts": {"unique_string_endpoints_with_hits": 2},
                "format_version": "abpairfm_string_strict_development_hits_v1",
                "mmseqs": {
                    "minimum_bidirectional_coverage": 0.8,
                    "minimum_sequence_identity": 0.3,
                },
                "output": {"sha256": MODULE.base.sha256_file(hits)},
                "status": "STRICT_DEVELOPMENT_HITS_READY_NOT_COLLAPSED",
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
            / "abpairfm_3b_quality_frontier_v1.json",
            receipt=receipt,
            stats=final_stats,
            strict_hit_receipt=strict_receipt,
        )
    )
    value = json.loads(receipt.read_text())
    assert value["authorization"]["production_pretraining"] is False
    assert value["scale_pool_gate"]["passed"] is False
    assert value["strict_development_decontamination"] == {
        "exclude_any_pair_touching_a_hit_cluster": True,
        "minimum_bidirectional_coverage": 0.8,
        "minimum_sequence_identity": 0.3,
    }
