#!/usr/bin/env python3
"""Focused tests for asymmetric retrieval score artifacts and statistics."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sabdab2_asymmetric_partner_retrieval import (
    FORMAT_VERSION,
    canonical_sha256,
    write_score_artifact,
)
from sabdab2_asymmetric_retrieval_statistics import (
    AUTHORITY_VALIDATION_BOUNDARY_FORMAT,
    AUTHORITY_VALIDATION_BOUNDARY_INTENDED_USE,
    CONTROL_LEDGER_FORMAT_VERSION,
    CONTROL_NAMES,
    CONTROL_SUITE_VERSION,
    STRICT_NW_CONSTRAINT_MODES,
    _best_positive_ranks,
    _classify_evaluation_scope,
    _expected_evaluation_context,
    _identifier_membership_sha256,
    _load_cluster_ids,
    _metric_vectors,
    _point_metrics,
    _stable_rank_matrix,
    _validate_locked_confirmation_parameters,
    _validate_v2_boundary_and_library,
    _verify_final_refit_one_shot_ledger,
    candidate_size_curves,
    exact_random_direction,
    fixed_degree_label_permutation_test,
    joint_bipartite_component_bootstrap,
    load_score_artifact,
    paired_effect_bootstrap,
    verify_evaluation_and_artifact,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direction_payload(
    scores: np.ndarray,
    mask: np.ndarray,
    anchor_keys: list[str],
    candidate_keys: list[str],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    ranks = _best_positive_ranks(_stable_rank_matrix(scores), mask)
    point = _point_metrics(_metric_vectors(ranks))
    records = []
    for index, (anchor, rank) in enumerate(zip(anchor_keys, ranks)):
        positives = [candidate_keys[int(value)] for value in np.flatnonzero(mask[index])]
        records.append(
            {
                "anchor_key": anchor,
                "best_known_positive_rank": int(rank),
                "known_positive_candidate_count": len(positives),
                "known_positive_candidate_keys": positives,
            }
        )
    return point, records


def _evaluation_payload(
    similarities: np.ndarray,
    positive_mask: np.ndarray,
    antibody_keys: list[str],
    antigen_keys: list[str],
) -> dict[str, object]:
    ab_metrics, ab_records = _direction_payload(
        similarities, positive_mask, antibody_keys, antigen_keys
    )
    ag_metrics, ag_records = _direction_payload(
        similarities.T, positive_mask.T, antigen_keys, antibody_keys
    )
    return {
        "format_version": FORMAT_VERSION,
        "stage": "full_library_development_validation",
        "split": "validation",
        "frozen_test_used": False,
        "checkpoint_sha256": _digest("checkpoint"),
        "config_sha256": _digest("config"),
        "checkpoint_run_mode": "fold_selection_with_validation",
        "one_shot_evaluation": False,
        "embedding_corpora": {
            "antibody": {"content_manifest_sha256": _digest("ab-corpus")},
            "antigen": {"content_manifest_sha256": _digest("ag-corpus")},
        },
        "dataset": {
            "instances_sha256": _digest("instances"),
            "split_manifest_sha256": _digest("split"),
            "split_format_version": "sabdab2_identity_isolated_development_v1",
            "split_intended_use": "development_only",
            "strict_nw_constraint_mode": None,
            "loaded_splits": ["train", "validation"],
            "frozen_test_records_selected": False,
            "authority": {
                "sha256": _digest("authority"),
                "authorized_source_pool_splits": ["train", "validation"],
                "sealed_source_splits": ["test"],
                "selected_sealed_overlap_count": 0,
            },
        },
        "metrics": {
            "antibody_to_antigen": ab_metrics,
            "antigen_to_antibody": ag_metrics,
            "bidirectional_mrr": 0.5 * (ab_metrics["mrr"] + ag_metrics["mrr"]),
            "exact_pair_count": int(positive_mask.sum()),
            "library_keys": {
                "antibody": antibody_keys,
                "antigen": antigen_keys,
            },
            "per_anchor_ranks": {
                "antibody_to_antigen": ab_records,
                "antigen_to_antibody": ag_records,
            },
        },
    }


class ExactRandomTests(unittest.TestCase):
    def test_multi_positive_mrr_matches_complete_enumeration(self) -> None:
        observed = exact_random_direction(5, [2])
        best_ranks = [
            min(positions) for positions in itertools.combinations(range(1, 6), 2)
        ]
        self.assertAlmostEqual(observed["mrr"], np.mean([1.0 / value for value in best_ranks]))
        self.assertAlmostEqual(observed["mean_rank"], np.mean(best_ranks))
        self.assertAlmostEqual(
            observed["recall_at_1"], np.mean([value <= 1 for value in best_ranks])
        )


class DeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.similarities = np.asarray(
            [[0.9, 0.2, 0.1], [0.1, 0.8, 0.3], [0.4, 0.5, 0.7]],
            dtype=np.float32,
        )
        self.mask = np.asarray(
            [[True, True, False], [False, True, False], [False, False, True]],
            dtype=bool,
        )

    def test_permutation_and_paired_bootstrap_are_deterministic(self) -> None:
        first = fixed_degree_label_permutation_test(
            self.similarities, self.mask, 200, seed=17
        )
        second = fixed_degree_label_permutation_test(
            self.similarities, self.mask, 200, seed=17
        )
        for direction in ("antibody_to_antigen", "antigen_to_antibody", "bidirectional"):
            for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20"):
                self.assertEqual(
                    first["results"][direction][metric],
                    second["results"][direction][metric],
                )
        effects = {
            "antibody_to_antigen": {
                metric: np.asarray([0.1, 0.2, -0.1])
                for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20")
            },
            "antigen_to_antibody": {
                metric: np.asarray([0.0, 0.3, 0.2])
                for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20")
            },
        }
        bootstrap_one = paired_effect_bootstrap(effects, 400, 23, 0.05)
        bootstrap_two = paired_effect_bootstrap(effects, 400, 23, 0.05)
        self.assertEqual(bootstrap_one, bootstrap_two)
        joint_one = joint_bipartite_component_bootstrap(
            effects, np.eye(3, dtype=bool), 400, 29, 0.05
        )
        joint_two = joint_bipartite_component_bootstrap(
            effects, np.eye(3, dtype=bool), 400, 29, 0.05
        )
        self.assertEqual(joint_one, joint_two)
        self.assertTrue(joint_one["available"])

    def test_score_npz_is_byte_deterministic(self) -> None:
        antibody_keys = sorted(_digest(f"ab-{index}") for index in range(3))
        antigen_keys = sorted(_digest(f"ag-{index}") for index in range(3))
        payload = {
            "similarity_matrix": self.similarities,
            "positive_mask": self.mask,
            "antibody_keys": antibody_keys,
            "antigen_keys": antigen_keys,
        }
        context = {
            "split": "validation",
            "checkpoint_sha256": _digest("checkpoint"),
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            first_reference = write_score_artifact(first, payload, context)
            second_reference = write_score_artifact(second, payload, context)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_reference["sha256"], second_reference["sha256"])

    def test_candidate_curve_omits_sizes_larger_than_each_native_library(self) -> None:
        curves = candidate_size_curves(
            self.similarities,
            self.mask,
            [2, 3, 5],
            sample_count=10,
            seed=42,
        )
        for direction in ("antibody_to_antigen", "antigen_to_antibody"):
            self.assertEqual(
                curves[direction]["omitted_sizes_larger_than_native_library"], [5]
            )
            self.assertNotIn("5", curves[direction]["curves"])


class AuthorityValidationBoundaryV2Tests(unittest.TestCase):
    def _fixture(
        self, root: Path
    ) -> tuple[
        dict[str, object],
        dict[str, tuple[str, str]],
        list[str],
        list[str],
        np.ndarray,
    ]:
        train = ["train-1"]
        validation = ["validation-1", "validation-2"]
        excluded = ["excluded-1"]
        test = ["test-1"]
        antibody_keys = sorted([_digest("validation-ab-1"), _digest("validation-ab-2")])
        antigen_keys = sorted([_digest("validation-ag-1"), _digest("validation-ag-2")])
        rows = {
            "validation-1": (antibody_keys[0], antigen_keys[0]),
            "validation-2": (antibody_keys[1], antigen_keys[1]),
            "excluded-1": (_digest("excluded-ab"), _digest("excluded-ag")),
        }
        boundary: dict[str, object] = {
            "format_version": AUTHORITY_VALIDATION_BOUNDARY_FORMAT,
            "intended_use": AUTHORITY_VALIDATION_BOUNDARY_INTENDED_USE,
            "stage": "build_authority_validation_nw_clean_boundary",
            "valid": True,
            "constraint_mode": "concat_antibody_and_antigen",
            "selection_influenced_by_labels_predictions_or_model_scores": False,
            "frozen_test_identifiers_read_for_exclusion_only": True,
            "frozen_test_sequence_fields_accessed_normalized_hashed_or_aligned": False,
            "splits": {
                "train": train,
                "validation": validation,
                "excluded_validation": excluded,
            },
            "membership_sha256": {
                "train": _identifier_membership_sha256(train),
                "validation": _identifier_membership_sha256(validation),
                "excluded_validation": _identifier_membership_sha256(excluded),
            },
        }
        boundary["scientific_payload_sha256"] = canonical_sha256(boundary)
        authority = {"splits": {"train": train, "validation": [*validation, *excluded], "test": test}}
        boundary_path = root / "boundary.json"
        authority_path = root / "authority.json"
        instances_path = root / "instances.parquet"
        boundary_path.write_text(
            json.dumps(boundary, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        authority_path.write_text(
            json.dumps(authority, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        instances_path.write_bytes(b"unit-test-parquet-placeholder")
        inventories = boundary["splits"]
        dataset: dict[str, object] = {
            "instances_path": str(instances_path.resolve()),
            "instances_sha256": _file_digest(instances_path),
            "split_manifest": str(boundary_path.resolve()),
            "split_manifest_sha256": _file_digest(boundary_path),
            "split_format_version": AUTHORITY_VALIDATION_BOUNDARY_FORMAT,
            "split_intended_use": AUTHORITY_VALIDATION_BOUNDARY_INTENDED_USE,
            "strict_nw_constraint_mode": "concat_antibody_and_antigen",
            "loaded_splits": ["train", "validation"],
            "unmaterialized_splits": [],
            "permanently_excluded_splits": ["excluded_validation"],
            "excluded_validation_rows_materialized": False,
            "raw_instance_counts": {
                name: len(values) for name, values in inventories.items()
            },
            "split_membership_sha256": {
                name: canonical_sha256({"instance_ids": sorted(values)})
                for name, values in inventories.items()
            },
            "parquet_row_filter": "train_and_validation_identifiers_only",
            "authority_accounting_splits": [
                "train",
                "validation",
                "excluded_validation",
            ],
            "frozen_test_records_selected": False,
            "authority": {
                "path": str(authority_path.resolve()),
                "sha256": _file_digest(authority_path),
                "authorized_source_pool_splits": ["train", "validation"],
                "authorized_source_pool_count": 4,
                "sealed_source_splits": ["test"],
                "sealed_source_counts": {"test": 1},
                "selected_sealed_overlap_count": 0,
                "coverage_mode": "exact",
                "selected_pool_coverage_count": 4,
            },
        }
        return dataset, rows, antibody_keys, antigen_keys, np.eye(2, dtype=bool)

    def test_valid_v2_reconstructs_the_validation_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, rows, antibody_keys, antigen_keys, mask = self._fixture(
                Path(directory)
            )
            with patch(
                "sabdab2_asymmetric_retrieval_statistics._load_instance_sequence_keys",
                return_value=rows,
            ) as load_rows:
                verified = _validate_v2_boundary_and_library(
                    dataset, antibody_keys, antigen_keys, mask
                )
            load_rows.assert_called_once_with(
                Path(str(dataset["instances_path"])),
                ["validation-1", "validation-2"],
            )
            self.assertEqual(verified["validation_count"], 2)
            self.assertEqual(verified["excluded_validation_count"], 1)
            self.assertFalse(
                verified["excluded_validation_rows_read_for_statistics"]
            )
            self.assertTrue(
                verified["validation_positive_graph_reconstructed_from_physical_rows"]
            )

    def test_full_control_evaluation_accepts_valid_v2_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, rows, antibody_keys, antigen_keys, mask = self._fixture(root)
            similarities = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
            evaluation = _evaluation_payload(
                similarities, mask, antibody_keys, antigen_keys
            )
            evaluation["dataset"] = dataset
            evaluation["checkpoint_run_mode"] = "train_only_control_bundle"
            evaluation["one_shot_evaluation"] = True
            evaluation["control_bundle_sha256"] = evaluation["checkpoint_sha256"]
            evaluation["one_shot_ledger"] = {
                "format_version": CONTROL_LEDGER_FORMAT_VERSION,
                "canonical_key": _digest("control-identity"),
                "canonical_path": str((root / "control-ledger.json").resolve()),
                "registry_dir": str(root.resolve()),
            }
            evaluation["locked_submission_evidence"] = {
                "control_protocol_v1": {
                    "path": str((root / "controls.md").resolve()),
                    "sha256": _digest("controls"),
                },
                "execution_protocol_v1_6": {
                    "path": str((root / "protocol.md").resolve()),
                    "sha256": _digest("protocol"),
                },
                "canonical_sequence_manifest": {
                    "path": str((root / "canonical.csv").resolve()),
                    "sha256": _digest("canonical"),
                },
                "training_embedding_content_manifest_sha256": {
                    "primary_antibody": _digest("primary-antibody"),
                    "shared_esm_antibody": _digest("shared-esm-antibody"),
                    "esm_antigen": _digest("esm-antigen"),
                },
                "required_boundary_modes": list(STRICT_NW_CONSTRAINT_MODES),
                "control_suite_completion_required": True,
            }
            evaluation["control"] = {
                "name": "raw_shared_esm_mean_cosine",
                "protocol_version": "unit-test",
                "bundle_config_sha256": _digest("bundle-config"),
            }
            score_path = root / "v2.scores.npz"
            evaluation["score_artifact"] = write_score_artifact(
                score_path,
                {
                    "similarity_matrix": similarities,
                    "positive_mask": mask,
                    "antibody_keys": antibody_keys,
                    "antigen_keys": antigen_keys,
                },
                _expected_evaluation_context(evaluation),
            )
            evaluation_path = root / "v2.evaluation.json"
            evaluation_path.write_text(
                json.dumps(evaluation, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            with patch(
                "sabdab2_asymmetric_retrieval_statistics._load_instance_sequence_keys",
                return_value=rows,
            ), patch(
                "sabdab2_asymmetric_retrieval_statistics._verify_final_refit_one_shot_ledger",
                return_value={"required": True, "status": "completed"},
            ):
                verified = verify_evaluation_and_artifact(
                    evaluation_path,
                    score_path,
                    _file_digest(evaluation_path),
                    _file_digest(score_path),
                )
            self.assertEqual(
                verified["evidence_scope"], "reused_development_confirmation"
            )
            self.assertEqual(
                verified["boundary_verification"]["validation_count"], 2
            )

    def test_physical_boundary_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, rows, antibody_keys, antigen_keys, mask = self._fixture(
                Path(directory)
            )
            boundary_path = Path(str(dataset["split_manifest"]))
            boundary_path.write_bytes(boundary_path.read_bytes() + b"\n")
            with patch(
                "sabdab2_asymmetric_retrieval_statistics._load_instance_sequence_keys",
                return_value=rows,
            ), self.assertRaisesRegex(ValueError, "physical SHA-256 mismatch"):
                _validate_v2_boundary_and_library(
                    dataset, antibody_keys, antigen_keys, mask
                )

    def test_wrong_intended_use_is_rejected_even_when_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, rows, antibody_keys, antigen_keys, mask = self._fixture(
                Path(directory)
            )
            boundary_path = Path(str(dataset["split_manifest"]))
            boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
            boundary["intended_use"] = "independent_test"
            boundary.pop("scientific_payload_sha256")
            boundary["scientific_payload_sha256"] = canonical_sha256(boundary)
            boundary_path.write_text(
                json.dumps(boundary, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            dataset["split_manifest_sha256"] = _file_digest(boundary_path)
            with patch(
                "sabdab2_asymmetric_retrieval_statistics._load_instance_sequence_keys",
                return_value=rows,
            ), self.assertRaisesRegex(PermissionError, "unapproved intended use"):
                _validate_v2_boundary_and_library(
                    dataset, antibody_keys, antigen_keys, mask
                )

    def test_self_consistent_unauthorized_accounting_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset, rows, antibody_keys, antigen_keys, mask = self._fixture(
                Path(directory)
            )
            boundary_path = Path(str(dataset["split_manifest"]))
            boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
            boundary["splits"]["validation"][1] = "unauthorized-validation"
            boundary["membership_sha256"]["validation"] = _identifier_membership_sha256(
                boundary["splits"]["validation"]
            )
            boundary.pop("scientific_payload_sha256")
            boundary["scientific_payload_sha256"] = canonical_sha256(boundary)
            boundary_path.write_text(
                json.dumps(boundary, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            dataset["split_manifest_sha256"] = _file_digest(boundary_path)
            dataset["split_membership_sha256"]["validation"] = canonical_sha256(
                {"instance_ids": sorted(boundary["splits"]["validation"])}
            )
            with patch(
                "sabdab2_asymmetric_retrieval_statistics._load_instance_sequence_keys",
                return_value=rows,
            ), self.assertRaisesRegex(ValueError, "does not equal authority validation"):
                _validate_v2_boundary_and_library(
                    dataset, antibody_keys, antigen_keys, mask
                )

    def test_scope_classification_is_explicit_and_fail_closed(self) -> None:
        self.assertEqual(
            _classify_evaluation_scope(
                {
                    "checkpoint_run_mode": "fold_selection_with_validation",
                    "one_shot_evaluation": False,
                }
            ),
            "exploratory_development",
        )
        self.assertEqual(
            _classify_evaluation_scope(
                {
                    "checkpoint_run_mode": "final_refit_fixed_epochs",
                    "one_shot_evaluation": True,
                }
            ),
            "reused_development_confirmation",
        )
        with self.assertRaisesRegex(PermissionError, "explicitly one-shot"):
            _classify_evaluation_scope(
                {
                    "checkpoint_run_mode": "final_refit_fixed_epochs",
                    "one_shot_evaluation": False,
                }
            )

    def test_v1_5_confirmation_parameters_are_immutable(self) -> None:
        arguments = argparse.Namespace(
            permutations=10_000,
            bootstrap_samples=10_000,
            seed=42,
            alpha=0.05,
            candidate_sizes=[10, 25, 50, 100],
            candidate_samples=1_000,
            anchor_clusters_json=None,
            expected_anchor_clusters_sha256=None,
        )
        contract = _validate_locked_confirmation_parameters(arguments)
        self.assertEqual(
            contract["protocol_sha256"],
            "43638fdd17b9ce3e4cac8b83e42c16261ecbf79c2e628d0b494e5dcbe02f9fc9",
        )
        arguments.seed = 43
        with self.assertRaisesRegex(PermissionError, "locked v1.5"):
            _validate_locked_confirmation_parameters(arguments)

    def test_final_refit_statistics_require_a_physical_one_shot_ledger(self) -> None:
        with self.assertRaisesRegex(PermissionError, "exact one-shot ledger reference"):
            _verify_final_refit_one_shot_ledger(
                {"checkpoint_run_mode": "final_refit_fixed_epochs"},
                Path("/tmp/evaluation.json"),
                _digest("evaluation"),
                {"sha256": _digest("artifact"), "path": Path("/tmp/scores.npz")},
            )

    def test_control_statistics_authenticate_the_completed_full_suite_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            output_dir = root / "suite"
            registry.mkdir()
            output_dir.mkdir()
            control_name = CONTROL_NAMES[0]
            evaluation_path = output_dir / f"{control_name}.evaluation.json"
            score_path = output_dir / f"{control_name}.scores.npz"
            score_path.write_bytes(b"current-score")
            checkpoint_sha256 = _digest("control-bundle")
            split_sha256 = _digest("boundary")
            authority_sha256 = _digest("authority")
            embedding_manifests = {
                "primary_antibody": _digest("primary-antibody"),
                "shared_esm_antibody": _digest("shared-esm-antibody"),
                "esm_antigen": _digest("esm-antigen"),
            }
            locked_evidence = {
                "control_protocol_v1": {
                    "path": "/synthetic/controls.md",
                    "sha256": _digest("controls-v1"),
                },
                "execution_protocol_v1_6": {
                    "path": "/synthetic/protocol.md",
                    "sha256": _digest("protocol-v1.6"),
                },
                "prepared_instances": {
                    "path": "/synthetic/instances.parquet",
                    "sha256": _digest("instances"),
                },
                "canonical_sequence_manifest": {
                    "path": "/synthetic/canonical.csv",
                    "sha256": _digest("canonical"),
                },
                "training_embedding_content_manifest_sha256": embedding_manifests,
            }
            contract = {
                "format_version": CONTROL_LEDGER_FORMAT_VERSION,
                "control_suite_version": CONTROL_SUITE_VERSION,
                "control_bundle_sha256": checkpoint_sha256,
                "split_manifest_sha256": split_sha256,
                "authority_split_sha256": authority_sha256,
                "control_protocol_v1_sha256": locked_evidence[
                    "control_protocol_v1"
                ]["sha256"],
                "execution_protocol_v1_6_sha256": locked_evidence[
                    "execution_protocol_v1_6"
                ]["sha256"],
                "prepared_instances_sha256": locked_evidence[
                    "prepared_instances"
                ]["sha256"],
                "canonical_sequence_manifest_sha256": locked_evidence[
                    "canonical_sequence_manifest"
                ]["sha256"],
                "training_embedding_content_manifest_sha256": embedding_manifests,
                "split": "validation",
                "controls": list(CONTROL_NAMES),
                "frozen_test_used": False,
            }
            identity = canonical_sha256(contract)
            ledger_path = registry / f"{identity}.json"
            evaluation = {
                "checkpoint_run_mode": "train_only_control_bundle",
                "checkpoint_sha256": checkpoint_sha256,
                "control_bundle_sha256": checkpoint_sha256,
                "control_protocol_v1": locked_evidence["control_protocol_v1"],
                "locked_submission_evidence": locked_evidence,
                "dataset": {
                    "split_manifest_sha256": split_sha256,
                    "authority": {"sha256": authority_sha256},
                },
                "control": {"name": control_name},
                "one_shot_ledger": {
                    "format_version": CONTROL_LEDGER_FORMAT_VERSION,
                    "canonical_key": identity,
                    "canonical_path": str(ledger_path),
                    "registry_dir": str(registry),
                },
            }
            evaluation_path.write_text(
                json.dumps(evaluation, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            outputs = {}
            for name in CONTROL_NAMES:
                current = name == control_name
                current_evaluation = (
                    evaluation_path
                    if current
                    else output_dir / f"{name}.evaluation.json"
                )
                current_score = score_path if current else output_dir / f"{name}.scores.npz"
                if not current:
                    current_evaluation.write_text(name, encoding="utf-8")
                    current_score.write_bytes(name.encode("ascii"))
                outputs[name] = {
                    "evaluation_json": str(current_evaluation),
                    "evaluation_json_sha256": _file_digest(current_evaluation),
                    "score_artifact": str(current_score),
                    "score_artifact_sha256": _file_digest(current_score),
                    "score_contract_sha256": _digest(f"{name}-contract"),
                    "bidirectional_mrr": 0.0,
                }
            ledger = {
                **contract,
                "evaluation_identity_sha256": identity,
                "status": "completed",
                "requested_output_dir": str(output_dir),
                "control_protocol_v1": locked_evidence["control_protocol_v1"],
                "execution_protocol_v1_6": locked_evidence[
                    "execution_protocol_v1_6"
                ],
                "locked_submission_evidence_sha256": canonical_sha256(
                    locked_evidence
                ),
                "control_outputs": outputs,
            }
            ledger_path.write_text(
                json.dumps(ledger, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            artifact = {
                "path": score_path.resolve(),
                "sha256": _file_digest(score_path),
                "metadata": {
                    "contract_sha256": outputs[control_name][
                        "score_contract_sha256"
                    ]
                },
            }
            with patch(
                "sabdab2_asymmetric_retrieval_statistics.CENTRAL_CONTROL_LEDGER_ROOT",
                registry,
            ), patch(
                "sabdab2_asymmetric_retrieval_statistics._verify_control_suite_completion",
                return_value={"required": True, "status": "completed"},
            ):
                verified = _verify_final_refit_one_shot_ledger(
                    evaluation,
                    evaluation_path.resolve(),
                    _file_digest(evaluation_path),
                    artifact,
                )
                self.assertEqual(verified["status"], "completed")
                ledger["status"] = "started_before_validation_materialization"
                ledger_path.write_text(
                    json.dumps(ledger, sort_keys=True, allow_nan=False),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "status"):
                    _verify_final_refit_one_shot_ledger(
                        evaluation,
                        evaluation_path.resolve(),
                        _file_digest(evaluation_path),
                        artifact,
                    )


class ContractSecurityTests(unittest.TestCase):
    def test_cluster_artifact_rejects_duplicate_keys_and_exponent_overflow(self) -> None:
        malformed = (
            ('{"directions":{},"directions":{}}\n', "duplicate JSON key"),
            ('{"directions":{"nested":[1e999]}}\n', "[Nn]on-finite JSON value"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (content, pattern) in enumerate(malformed):
                path = root / f"malformed-{index}.json"
                path.write_text(content, encoding="utf-8")
                with self.subTest(pattern=pattern), self.assertRaisesRegex(
                    ValueError, pattern
                ):
                    _load_cluster_ids(
                        path,
                        _file_digest(path),
                        _digest("evaluation"),
                        _digest("artifact"),
                        ["antibody"],
                        ["antigen"],
                    )

    def setUp(self) -> None:
        self.similarities = np.asarray(
            [[0.9, 0.2, 0.1], [0.1, 0.8, 0.3], [0.4, 0.5, 0.7]],
            dtype=np.float32,
        )
        self.mask = np.eye(3, dtype=bool)
        self.antibody_keys = sorted(_digest(f"ab-{index}") for index in range(3))
        self.antigen_keys = sorted(_digest(f"ag-{index}") for index in range(3))

    def _write_bound_inputs(self, root: Path) -> tuple[Path, Path, dict[str, object]]:
        evaluation = _evaluation_payload(
            self.similarities,
            self.mask,
            self.antibody_keys,
            self.antigen_keys,
        )
        score_path = root / "scores.npz"
        reference = write_score_artifact(
            score_path,
            {
                "similarity_matrix": self.similarities,
                "positive_mask": self.mask,
                "antibody_keys": self.antibody_keys,
                "antigen_keys": self.antigen_keys,
            },
            _expected_evaluation_context(evaluation),
        )
        evaluation["score_artifact"] = reference
        evaluation_path = root / "evaluation.json"
        evaluation_path.write_text(
            json.dumps(evaluation, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        return evaluation_path, score_path, evaluation

    def test_bound_artifact_verifies_and_key_tampering_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evaluation_path, score_path, evaluation = self._write_bound_inputs(
                Path(directory)
            )
            verified = verify_evaluation_and_artifact(
                evaluation_path,
                score_path,
                _file_digest(evaluation_path),
                _file_digest(score_path),
            )
            self.assertEqual(verified["artifact"]["positive_mask"].sum(), 3)
            evaluation["metrics"]["library_keys"]["antibody"][0] = _digest("malicious")
            evaluation_path.write_text(
                json.dumps(evaluation, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "library keys differ"):
                verify_evaluation_and_artifact(
                    evaluation_path,
                    score_path,
                    _file_digest(evaluation_path),
                    _file_digest(score_path),
                )

    def test_hash_mismatch_and_sealed_test_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evaluation_path, score_path, evaluation = self._write_bound_inputs(
                Path(directory)
            )
            with self.assertRaisesRegex(ValueError, "Score artifact SHA-256 mismatch"):
                verify_evaluation_and_artifact(
                    evaluation_path,
                    score_path,
                    _file_digest(evaluation_path),
                    _digest("wrong-artifact"),
                )
            evaluation["split"] = "test"
            evaluation["frozen_test_used"] = True
            evaluation_path.write_text(
                json.dumps(evaluation, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(PermissionError, "Frozen or non-validation"):
                verify_evaluation_and_artifact(
                    evaluation_path,
                    score_path,
                    _file_digest(evaluation_path),
                    _file_digest(score_path),
                )

    def test_duplicate_raw_npz_members_are_rejected_before_numpy_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.scores.npz"
            duplicate_path = root / "duplicate.scores.npz"
            write_score_artifact(
                valid_path,
                {
                    "similarity_matrix": np.eye(2, dtype=np.float32),
                    "positive_mask": np.eye(2, dtype=bool),
                    "antibody_keys": [_digest("a1"), _digest("a2")],
                    "antigen_keys": [_digest("g1"), _digest("g2")],
                },
                {"synthetic": True},
            )
            with zipfile.ZipFile(valid_path, mode="r") as source:
                members = [
                    (info, source.read(info.filename)) for info in source.infolist()
                ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(duplicate_path, mode="w") as target:
                    for info, payload in members:
                        target.writestr(info, payload)
                    target.writestr(members[0][0], members[0][1])
            with self.assertRaisesRegex(ValueError, "duplicate raw ZIP members"):
                load_score_artifact(duplicate_path)


if __name__ == "__main__":
    unittest.main()
