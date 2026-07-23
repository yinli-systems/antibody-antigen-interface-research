#!/usr/bin/env python3
"""Focused tests for deterministic asymmetric retrieval fold aggregation."""

from __future__ import annotations

import hashlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from aggregate_asymmetric_retrieval_folds import (
    FOLD_FORMAT_VERSION,
    HISTORY_FORMAT_VERSION,
    MANIFEST_FORMAT_VERSION,
    _selection_trace,
    aggregate_manifest,
    canonical_sha256,
    exact_random_best_positive_mrr,
    exact_trainable_parameter_count,
)


VARIANTS = (
    "antiberty_calm_mean",
    "antiberty_chain_balanced",
    "esm2_calm_mean",
)


def _digest(value: str) -> str:
    """Return a stable synthetic SHA-256 digest."""
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _file_digest(path: Path) -> str:
    """Return the digest of one synthetic input file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> str:
    """Write deterministic strict JSON and return its digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _file_digest(path)


def _model(variant: str) -> dict[str, object]:
    """Return a complete synthetic saved model configuration."""
    antiberty = variant != "esm2_calm_mean"
    return {
        "architecture": "AsymmetricPartnerRetrievalModel",
        "antibody_artifact_format": "antiberty" if antiberty else "esm_chunks",
        "antibody_encoder": (
            "AntiBERTy_md_smooth"
            if antiberty
            else "facebook/esm2_t33_650M_UR50D"
        ),
        "antibody_model_weights_sha256": _digest(
            "antiberty-weights" if antiberty else "esm-weights"
        ),
        "antigen_encoder": "facebook/esm2_t33_650M_UR50D",
        "antigen_model_weights_sha256": _digest("esm-weights"),
        "antibody_input_dim": 512 if antiberty else 1280,
        "antigen_input_dim": 1280,
        "embedding_dim": 128,
        "antibody_hidden_dim": 256,
        "antigen_hidden_dim": 256,
        "dropout": 0.1,
        "chain_balanced_antibody_pooling": variant == "antiberty_chain_balanced",
        "antibody_pooling": "mean",
        "antigen_pooling": "mean",
        "projection_order": "per_residue_ffn_then_length_pool_then_l2_normalize",
        "initial_temperature": 0.2,
        "minimum_temperature": 0.01,
        "maximum_temperature": 1.0,
    }


def _model_contract(variant: str) -> dict[str, object]:
    """Return variant-defining fields pinned in the aggregation manifest."""
    model = _model(variant)
    keys = (
        "architecture",
        "antibody_artifact_format",
        "antibody_encoder",
        "antibody_input_dim",
        "chain_balanced_antibody_pooling",
        "antibody_pooling",
        "antigen_encoder",
        "antigen_pooling",
        "projection_order",
    )
    return {key: model[key] for key in keys}


def _fold_payload(fold_index: int, authority_sha256: str) -> dict[str, object]:
    """Return one minimal valid train-only fold definition."""
    return {
        "format_version": FOLD_FORMAT_VERSION,
        "intended_use": "development_only",
        "stage": "train_only_antigen_cluster_model_selection",
        "fold_index": fold_index,
        "fold_count": 3,
        "seed": 42,
        "source_pool_split": "train",
        "coverage_mode": "exact",
        "isolation_policy": "antigen_exact",
        "splits": {"train": [f"train-{fold_index}"], "validation": [f"val-{fold_index}"]},
        "audit": {
            "valid": True,
            "coverage": {
                "mode": "exact",
                "sealed_id_intersection_count": 0,
            },
        },
        "provenance": {
            "authority": {
                "sha256": authority_sha256,
                "source_pool_split": "train",
                "sealed_splits": ["validation", "test"],
            },
            "labels_contacts_predictions_or_checkpoints_read": False,
            "sealed_sequences_read": False,
        },
    }


def _metrics(rank_pattern: tuple[int, int, int]) -> dict[str, object]:
    """Return internally consistent bidirectional full-library metrics."""
    antibody_keys = ["ab-0", "ab-1", "ab-2"]
    antigen_keys = ["ag-0", "ag-1", "ag-2"]
    pairs = {
        "ab-0": ["ag-0", "ag-1"],
        "ab-1": ["ag-1"],
        "ab-2": ["ag-2"],
    }
    reverse_pairs = {
        "ag-0": ["ab-0"],
        "ag-1": ["ab-0", "ab-1"],
        "ag-2": ["ab-2"],
    }

    def direction(
        anchors: list[str], candidates: list[str], positives: dict[str, list[str]]
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        records = []
        for anchor, rank in zip(anchors, rank_pattern):
            positive_keys = positives[anchor]
            records.append(
                {
                    "anchor_key": anchor,
                    "best_known_positive_rank": rank,
                    "reciprocal_rank": 1.0 / rank,
                    "known_positive_candidate_count": len(positive_keys),
                    "known_positive_candidate_keys": positive_keys,
                }
            )
        mrr = sum(1.0 / value for value in rank_pattern) / len(rank_pattern)
        return (
            {
                "anchor_count": len(anchors),
                "candidate_count": len(candidates),
                "known_positive_count": sum(len(positives[value]) for value in anchors),
                "mrr": mrr,
            },
            records,
        )

    ab_metrics, ab_records = direction(antibody_keys, antigen_keys, pairs)
    ag_metrics, ag_records = direction(antigen_keys, antibody_keys, reverse_pairs)
    return {
        "bidirectional_mrr": ab_metrics["mrr"],
        "antibody_to_antigen": ab_metrics,
        "antigen_to_antibody": ag_metrics,
        "library_keys": {"antibody": antibody_keys, "antigen": antigen_keys},
        "per_anchor_ranks": {
            "antibody_to_antigen": ab_records,
            "antigen_to_antibody": ag_records,
        },
    }


def _history_payload(
    variant: str,
    fold_sha256: str,
    source_sha256: dict[str, str],
    authority_sha256: str,
    rank_pattern: tuple[int, int, int],
) -> tuple[dict[str, object], float]:
    """Return one pinned training history and its selected MRR."""
    metrics = _metrics(rank_pattern)
    mrr = float(metrics["bidirectional_mrr"])
    config: dict[str, object] = {
        "format_version": HISTORY_FORMAT_VERSION,
        "model": _model(variant),
        "optimization": {"seed": 42, "batch_size": 1024},
        "dataset": {
            "split_manifest_sha256": fold_sha256,
            "split_format_version": FOLD_FORMAT_VERSION,
            "split_intended_use": "development_only",
            "loaded_splits": ["train", "validation"],
            "frozen_test_records_selected": False,
            "authority": {
                "sha256": authority_sha256,
                "authorized_source_pool_splits": ["train"],
                "sealed_source_splits": ["validation", "test"],
                "coverage_mode": "exact",
                "selected_sealed_overlap_count": 0,
            },
        },
        "source_sha256": source_sha256,
        "scientific_scope": {
            "frozen_test_used": False,
            "model_selection_data": "development validation only",
            "claims_allowed": "exploratory development-screen evidence only",
        },
    }
    config["config_sha256"] = canonical_sha256(config)
    history = {
        "format_version": HISTORY_FORMAT_VERSION,
        "config": config,
        "history": [
            {
                "epoch": 1,
                "validation": metrics,
                "selection_metric": "bidirectional_validation_mrr",
                "best_validation_mrr": mrr,
                "best_epoch": 1,
                "improved": True,
            }
        ],
    }
    return history, mrr


def _build_fixture(root: Path) -> tuple[Path, str, dict[str, object]]:
    """Build all 27 pinned synthetic input references and one manifest."""
    authority_sha256 = _digest("authority")
    source_sha256 = {
        "entrypoint": _digest("entrypoint"),
        "asymmetric_retrieval": _digest("asymmetric-retrieval"),
    }
    rank_patterns = {
        "antiberty_calm_mean": (1, 1, 1),
        "antiberty_chain_balanced": (1, 1, 2),
        "esm2_calm_mean": (1, 2, 3),
    }
    fold_references: dict[int, dict[str, str]] = {}
    for fold_index in range(3):
        fold_path = root / "folds" / f"fold_{fold_index}.json"
        fold_sha256 = _write_json(
            fold_path, _fold_payload(fold_index, authority_sha256)
        )
        fold_references[fold_index] = {
            "path": str(fold_path.relative_to(root)),
            "sha256": fold_sha256,
        }

    variants = []
    for variant in VARIANTS:
        runs = []
        for fold_index in range(3):
            fold_reference = fold_references[fold_index]
            history, mrr = _history_payload(
                variant,
                fold_reference["sha256"],
                source_sha256,
                authority_sha256,
                rank_patterns[variant],
            )
            run_root = root / "runs" / variant / f"fold_{fold_index}"
            history_path = run_root / "history.json"
            history_sha256 = _write_json(history_path, history)
            summary_path = run_root / "training_summary.json"
            summary_sha256 = _write_json(
                summary_path,
                {
                    "format_version": HISTORY_FORMAT_VERSION,
                    "stage": "train_complete",
                    "config_sha256": history["config"]["config_sha256"],
                    "best_validation_mrr": mrr,
                    "best_epoch": 1,
                    "completed_epoch": 1,
                    "frozen_test_used": False,
                },
            )
            runs.append(
                {
                    "fold_index": fold_index,
                    "fold_manifest": dict(fold_reference),
                    "history": {
                        "path": str(history_path.relative_to(root)),
                        "sha256": history_sha256,
                    },
                    "training_summary": {
                        "path": str(summary_path.relative_to(root)),
                        "sha256": summary_sha256,
                    },
                }
            )
        variants.append(
            {
                "name": variant,
                "model_contract": _model_contract(variant),
                "runs": runs,
            }
        )
    manifest: dict[str, object] = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "intended_use": "development_only",
        "frozen_test_used": False,
        "seed": 42,
        "fold_count": 3,
        "authority": {
            "sha256": authority_sha256,
            "source_pool_splits": ["train"],
            "sealed_source_splits": ["validation", "test"],
            "coverage_mode": "exact",
        },
        "expected_source_sha256": source_sha256,
        "shared_config_contract": {
            "format_version": HISTORY_FORMAT_VERSION,
            "optimization": {"batch_size": 1024, "seed": 42},
            "scientific_scope": {"frozen_test_used": False},
        },
        "variants": variants,
    }
    manifest_path = root / "aggregation_manifest.json"
    manifest_sha256 = _write_json(manifest_path, manifest)
    return manifest_path, manifest_sha256, manifest


class ExactChanceTests(unittest.TestCase):
    """Verify exact finite-library multi-positive chance calculations."""

    def test_formula_matches_complete_positive_position_enumeration(self) -> None:
        observed = exact_random_best_positive_mrr(6, 3)
        ranks = [
            min(positions)
            for positions in itertools.combinations(range(1, 7), 3)
        ]
        expected = sum((1.0 / value for value in ranks)) / len(ranks)
        self.assertAlmostEqual(float(observed), expected)


class ParameterAndSelectionTests(unittest.TestCase):
    """Verify exact parameter counting and the locked tiebreak sequence."""

    def test_parameter_count_matches_closed_form_and_breaks_metric_tie(self) -> None:
        anti_config = {"model": _model("antiberty_calm_mean")}
        esm_config = {"model": _model("esm2_calm_mean")}
        anti_count = exact_trainable_parameter_count(anti_config)
        esm_count = exact_trainable_parameter_count(esm_config)
        self.assertLess(anti_count, esm_count)
        variants = [
            {
                "name": "esm2_calm_mean",
                "macro_bidirectional_mrr": 0.5,
                "worst_fold_bidirectional_mrr": 0.4,
                "exact_trainable_parameter_count": esm_count,
            },
            {
                "name": "antiberty_calm_mean",
                "macro_bidirectional_mrr": 0.5,
                "worst_fold_bidirectional_mrr": 0.4,
                "exact_trainable_parameter_count": anti_count,
            },
        ]
        trace = _selection_trace(variants)
        self.assertEqual(trace["selected_variant"], "antiberty_calm_mean")
        self.assertFalse(trace["variant_name_tiebreak_used"])


class ManifestAggregationTests(unittest.TestCase):
    """Verify deterministic aggregation and adversarial input rejection."""

    def test_valid_manifest_aggregates_unweighted_folds_without_hardcoded_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, manifest_sha256, _ = _build_fixture(Path(directory))
            first = aggregate_manifest(manifest_path, manifest_sha256)
            second = aggregate_manifest(manifest_path, manifest_sha256)
            self.assertEqual(first, second)
            self.assertEqual(
                first["selection"]["selected_variant"], "antiberty_calm_mean"
            )
            variants = {value["name"]: value for value in first["variants"]}
            self.assertEqual(variants["antiberty_calm_mean"]["macro_bidirectional_mrr"], 1.0)
            for variant in variants.values():
                self.assertEqual(len(variant["folds"]), 3)
                self.assertGreater(
                    variant["macro_exact_random_chance"]["value"], 0.0
                )

    def test_manifest_and_nested_input_hash_tampering_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest_sha256, manifest = _build_fixture(root)
            with self.assertRaisesRegex(ValueError, "aggregation manifest SHA-256 mismatch"):
                aggregate_manifest(manifest_path, _digest("wrong-manifest"))

            history_reference = manifest["variants"][0]["runs"][0]["history"]
            history_path = root / history_reference["path"]
            history_path.write_bytes(history_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "history SHA-256 mismatch"):
                aggregate_manifest(manifest_path, manifest_sha256)

    def test_sealed_membership_is_rejected_even_when_all_hashes_are_re_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _, manifest = _build_fixture(root)
            fold_reference = manifest["variants"][0]["runs"][0]["fold_manifest"]
            fold_path = root / fold_reference["path"]
            fold = json.loads(fold_path.read_text(encoding="utf-8"))
            fold["audit"]["coverage"]["sealed_id_intersection_count"] = 1
            malicious_fold_sha256 = _write_json(fold_path, fold)
            for variant in manifest["variants"]:
                for run in variant["runs"]:
                    if run["fold_index"] == 0:
                        run["fold_manifest"]["sha256"] = malicious_fold_sha256
                        history_path = root / run["history"]["path"]
                        history = json.loads(history_path.read_text(encoding="utf-8"))
                        history["config"]["dataset"][
                            "split_manifest_sha256"
                        ] = malicious_fold_sha256
                        history["config"].pop("config_sha256")
                        history["config"]["config_sha256"] = canonical_sha256(
                            history["config"]
                        )
                        run["history"]["sha256"] = _write_json(history_path, history)
                        summary_path = root / run["training_summary"]["path"]
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        summary["config_sha256"] = history["config"]["config_sha256"]
                        run["training_summary"]["sha256"] = _write_json(
                            summary_path, summary
                        )
            malicious_manifest_sha256 = _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(PermissionError, "sealed source identifiers"):
                aggregate_manifest(manifest_path, malicious_manifest_sha256)


if __name__ == "__main__":
    unittest.main()
