"""Fail-closed helper tests for the one-shot AsEP official-test evaluator."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


def _module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / "evaluate_asep_frozen_test.py"
    spec = importlib.util.spec_from_file_location("asep_frozen_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_consumption_token_is_create_only_and_hash_bound(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "result"
    token = module._consume_once(
        tmp_path,
        opening_sha256="a" * 64,
        cache_manifest_sha256="b" * 64,
        output_root=output,
    )
    payload = json.loads(token.read_text())
    assert payload["state"] == "scoring_started_fail_closed"
    assert payload["opening_receipt_sha256"] == "a" * 64
    assert payload["test_cache_manifest_sha256"] == "b" * 64
    with pytest.raises(FileExistsError):
        module._consume_once(
            tmp_path,
            opening_sha256="a" * 64,
            cache_manifest_sha256="b" * 64,
            output_root=output,
        )


def test_bootstrap_uses_panel_clusters_not_cache_metadata() -> None:
    module = _module()
    panel = {
        "entries": [
            {
                "instance_id": "a",
                "antigen_cluster_id": "ag-large",
                "antibody_cluster_id": "ab-one",
            },
            {
                "instance_id": "b",
                "antigen_cluster_id": "ag-large",
                "antibody_cluster_id": "ab-two",
            },
            {
                "instance_id": "c",
                "antigen_cluster_id": "ag-small",
                "antibody_cluster_id": "ab-two",
            },
        ]
    }
    metrics = {
        "partner_retrieval_by_direction": {
            "fixed_antigen_wrong_antibody": {
                "cluster_margins": {"ag-large": 0.2, "ag-small": 0.1}
            },
            "fixed_antibody_wrong_antigen": {
                "cluster_margins": {"ab-one": 0.3, "ab-two": 0.0}
            },
        },
        "contact": {
            "per_complex_auprc_by_instance": {"a": 1.0, "b": 1.0, "c": 0.0}
        },
        "epitope": {
            "per_complex_auprc_by_instance": {"a": 1.0, "b": 1.0, "c": 0.0}
        },
        "paratope": {
            "per_complex_auprc_by_instance": {"a": 1.0, "b": 0.0, "c": 0.0}
        },
    }
    module._add_bootstrap(metrics, panel, replicates=100, seed=13)
    assert (
        metrics["contact"]["mean_per_complex_auprc_cluster_bootstrap_ci"][
            "estimate"
        ]
        == 0.5
    )
    assert (
        metrics["paratope"]["mean_per_complex_auprc_cluster_bootstrap_ci"][
            "group_count"
        ]
        == 2
    )


def test_cache_authentication_binds_graph_open_token(tmp_path: Path) -> None:
    module = _module()
    state = tmp_path / "state"
    state.mkdir()
    token = state / "ASEP_TEST_GRAPH_OPEN_CONSUMED.json"
    token.write_text('{"state":"opened"}', encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    manifest = cache / "ASEP_FROZEN_TEST_CACHE_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": module.FROZEN_TEST_CACHE_FORMAT,
                "authority": {
                    "opening_receipt_sha256": "a" * 64,
                    "selected_ensemble_result_sha256": "b" * 64,
                    "graph_open_consumption_token_sha256": _sha(token),
                },
                "checks": {
                    "official_test_graph_and_labels_opened_once": True,
                    "model_predictions_or_scores_read": False,
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = cache / "ASEP_FROZEN_TEST_CACHE_RECEIPT.json"
    receipt.write_text(
        json.dumps(
            {
                "format_version": module.FROZEN_TEST_CACHE_FORMAT,
                "manifest_sha256": _sha(manifest),
                "test_graph_or_label_open_count": 1,
                "model_predictions_or_scores_read": False,
                "graph_open_consumption_token_sha256": _sha(token),
            }
        ),
        encoding="utf-8",
    )
    opening = {
        "test_state_root": str(state),
        "graph_open_consumption_token_name": token.name,
    }
    module._authenticate_cache(
        cache,
        _sha(manifest),
        _sha(receipt),
        opening_sha256="a" * 64,
        selected_ensemble_sha256="b" * 64,
        opening=opening,
    )
    token.write_text('{"state":"tampered"}', encoding="utf-8")
    with pytest.raises(ValueError, match="consumption token"):
        module._authenticate_cache(
            cache,
            _sha(manifest),
            _sha(receipt),
            opening_sha256="a" * 64,
            selected_ensemble_sha256="b" * 64,
            opening=opening,
        )
