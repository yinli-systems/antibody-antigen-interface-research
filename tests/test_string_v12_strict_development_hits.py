from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "finalize_string_v12_strict_development_hits.py"
)
SPEC = importlib.util.spec_from_file_location("strict_string_hits", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _inputs(tmp_path: Path) -> Namespace:
    endpoint_fasta = tmp_path / "endpoint_sequences.fasta.zst"
    endpoint_fasta.write_bytes(b"endpoint-fasta")
    cluster_receipt = tmp_path / "cluster.receipt.json"
    cluster_receipt.write_text(
        json.dumps(
            {
                "format_version": "abpairfm_string_high_confidence_cluster_assets_v1",
                "published_assets": [
                    {
                        "name": endpoint_fasta.name,
                        "sha256": MODULE.sha256_file(endpoint_fasta),
                    }
                ],
                "status": "CLUSTERED_NOT_COLLAPSED_NOT_DECONTAMINATED",
                "test_access": False,
            }
        ),
        encoding="utf-8",
    )
    development_fasta = tmp_path / "development.fasta"
    development_fasta.write_text(">d1\nACDEFGHIKLMNPQRSTVWY\n", encoding="utf-8")
    development_receipt = tmp_path / "development.receipt.json"
    development_receipt.write_text(
        json.dumps(
            {
                "format_version": "abpairfm_development_sequence_panel_v1",
                "output_fasta": {"sha256": MODULE.sha256_file(development_fasta)},
                "status": "FROZEN_VALIDATION_SEQUENCE_PANEL",
                "test_access": False,
            }
        ),
        encoding="utf-8",
    )
    hits = tmp_path / "development_hits30.tsv.zst"
    return Namespace(
        cluster_receipt=cluster_receipt,
        development_fasta=development_fasta,
        development_receipt=development_receipt,
        endpoint_fasta=endpoint_fasta,
        hits=hits,
        mmseqs_version="test-version",
        output_name="development_hits30.tsv.zst",
    )


def test_audit_hits_enforces_identity_and_bidirectional_coverage(tmp_path: Path) -> None:
    hits = tmp_path / "hits.tsv"
    hits.write_text(
        "d1\tp1\t0.30\t80\t0.80\t0.80\t1e-5\t50\n"
        "d1\tp2\t0.55\t90\t0.90\t0.85\t1e-8\t70\n",
        encoding="utf-8",
    )
    assert MODULE.audit_hits(hits) == {
        "hit_rows": 2,
        "unique_development_queries_with_hits": 1,
        "unique_string_endpoints_with_hits": 2,
    }
    hits.write_text("d1\tp1\t0.29\t80\t0.80\t0.80\t1e-5\t50\n")
    with pytest.raises(ValueError, match="violates the gate"):
        MODULE.audit_hits(hits)


def test_build_receipt_authenticates_inputs_and_keeps_training_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _inputs(tmp_path)
    arguments.hits.write_bytes(b"compressed-placeholder")
    monkeypatch.setattr(
        MODULE,
        "audit_hits",
        lambda _path: {
            "hit_rows": 3,
            "unique_development_queries_with_hits": 2,
            "unique_string_endpoints_with_hits": 3,
        },
    )
    receipt = MODULE.build_receipt(arguments)
    assert receipt["authorization"]["production_pretraining"] is False
    assert receipt["authorization"]["pair_collapse_with_strict_hits"] is True
    assert receipt["mmseqs"]["minimum_sequence_identity"] == 0.3
    assert receipt["mmseqs"]["minimum_bidirectional_coverage"] == 0.8
    assert receipt["test_access"] is False
