#!/usr/bin/env python3
"""Focused tests for authenticated partner-retrieval controls."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

from sabdab2_retrieval_controls import (
    CONTROL_NAMES,
    CONTROL_PROTOCOL_V1_SHA256,
    CONTROL_SUITE_COMPLETION_FORMAT,
    EXECUTION_PROTOCOL_V1_6_SHA256,
    LOCKED_AUTHORITY_SPLIT_SHA256,
    LOCKED_BOUNDARY_VERIFICATION_SHA256,
    LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256,
    LOCKED_PREPARED_INSTANCES_SHA256,
    LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256,
    _common_training_dataset_contract,
    _atomic_write_deterministic_npz,
    _create_exclusive_json,
    _dataset_from_args,
    _evaluation_context,
    _fit_controls_under_process_lock,
    _read_sequences,
    _validate_training_embedding_artifacts,
    _validate_locked_training_embedding_fingerprints,
    _verify_completed_control_boundary,
    _load_bundle,
    _write_bundle,
    _write_score_artifact_create_only,
    command_evaluate,
    command_fit,
    control_fit_identity,
    deterministic_endpoint_matched_negatives,
    exclusive_control_fit_process_lock,
    finalize_control_suite,
    fit_fixed_logistic,
    make_random_head_arrays,
    metrics_from_scores,
    pair_descriptor,
    random_head_pool,
    reserve_control_evaluation,
    sequence_descriptor,
    unary_score_matrices,
    validate_boundary_v2_accounting,
    validate_control_protocol_reference,
    validate_execution_protocol_reference,
)
from sabdab2_partner_retrieval import (
    PairRecord,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from sabdab2_asymmetric_partner_retrieval import write_score_artifact


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class StaticFirewallTests(unittest.TestCase):
    def test_control_roster_is_locked(self) -> None:
        self.assertEqual(
            CONTROL_NAMES,
            (
                "raw_shared_esm_mean_cosine",
                "random_untrained_calm_head",
                "antibody_only_unary",
                "antigen_only_unary",
                "length_composition_logistic",
            ),
        )
        self.assertEqual(
            EXECUTION_PROTOCOL_V1_6_SHA256,
            "b5b74452d3483d8b1fdf8c67387e3d38ad2be34a22b65e26a6cab71b0ca73078",
        )
        self.assertEqual(
            CONTROL_PROTOCOL_V1_SHA256,
            "21c3a0727a1b9d6543a022f6f9d216414ee8f0fc99590b4febca5dcb0d8cdc02",
        )
        self.assertEqual(
            LOCKED_PREPARED_INSTANCES_SHA256,
            "a9131d52323bae40572b7de81c5e41353ccc39a73309b7d7043485350facb009",
        )
        self.assertEqual(
            LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256,
            "548bcfab75923ec6621c1a3fef1aa6673c034f64a9bdc752e7ee15b41b4a4e63",
        )
        self.assertEqual(
            LOCKED_AUTHORITY_SPLIT_SHA256,
            "e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b",
        )
        self.assertEqual(
            LOCKED_BOUNDARY_VERIFICATION_SHA256,
            "3bd8186603d908a99e1fbf00ead3697dd276ccd798b4f91bcbdda81441bd80e8",
        )
        self.assertEqual(
            LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256,
            {
                "primary_antibody": "9eb10c64fd05897162548a81871841724d9471ba5ed1c800b3d10a2c9229a45e",
                "shared_esm_antibody": "03a9e0e4b4acf8b3b76cca651d65c4f278f5ac5a8985d3b12fe001da7809bf36",
                "esm_antigen": "d4157971ba2e977d1125e896fe13147371292495fa04a47f62b32ebe8f4ac97e",
            },
        )

    def test_execution_protocol_digest_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.md"
            path.write_text("substituted", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "final locked v1.6"):
                validate_execution_protocol_reference(path, _digest("wrong-protocol"))
            with self.assertRaisesRegex(ValueError, "retired v1.6 draft"):
                validate_execution_protocol_reference(
                    path,
                    "d9c3ca9bf9e07b498874b3324d3db4a2dfa2a5d0faf7de289311000f3fb2d87b",
                )

    def test_final_physical_execution_protocol_is_accepted(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        protocol = (
            repository
            / "outputs/interaction_first/research/"
            "PARTNER_RETRIEVAL_PROTOCOL_V1_6_EXECUTION_HARDENING.md"
        )
        reference = validate_execution_protocol_reference(
            protocol,
            EXECUTION_PROTOCOL_V1_6_SHA256,
        )
        self.assertEqual(reference["sha256"], EXECUTION_PROTOCOL_V1_6_SHA256)

    def test_final_physical_controls_protocol_is_accepted(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        protocol = (
            repository
            / "outputs/interaction_first/research/PARTNER_RETRIEVAL_CONTROLS_PROTOCOL_V1.md"
        )
        reference = validate_control_protocol_reference(
            protocol,
            CONTROL_PROTOCOL_V1_SHA256,
        )
        self.assertEqual(reference["sha256"], CONTROL_PROTOCOL_V1_SHA256)

    def test_execution_templates_pin_v1_6_and_offer_no_receipt_override(self) -> None:
        scripts = Path(__file__).resolve().parent
        for name in (
            "asymmetric_partner_retrieval_controls_fit.sbatch",
            "asymmetric_partner_retrieval_controls_evaluate.sbatch",
        ):
            text = (scripts / name).read_text(encoding="utf-8")
            self.assertIn(EXECUTION_PROTOCOL_V1_6_SHA256, text)
            self.assertIn(CONTROL_PROTOCOL_V1_SHA256, text)
            self.assertIn(LOCKED_PREPARED_INSTANCES_SHA256, text)
            self.assertIn(LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256, text)
            self.assertIn(LOCKED_AUTHORITY_SPLIT_SHA256, text)
            self.assertIn(LOCKED_BOUNDARY_VERIFICATION_SHA256, text)
            self.assertIn("RETRIEVAL_PROTOCOL_V1_6", text)
            self.assertNotIn("RETRIEVAL_PROTOCOL_V1_5", text)
        for name in (
            "asymmetric_partner_retrieval_controls_fit.sbatch",
            "asymmetric_partner_retrieval_controls_evaluate.sbatch",
            "asymmetric_partner_retrieval_controls_finalize_suite.sbatch",
        ):
            text = (scripts / name).read_text(encoding="utf-8")
            self.assertIn(
                "bb5742097c9a0b0f46419b0c3bebd9ccd378c69dc4137483f55d28219ca9a285",
                text,
            )
            self.assertIn("controls_v1_6_schema_fix_bb574209/sources", text)
        finalizer = (
            scripts / "asymmetric_partner_retrieval_controls_finalize_suite.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("finalize-suite", finalizer)
        self.assertIn("YES_BOTH_BOUNDARIES_REVIEWED", finalizer)
        statistics = (
            scripts / "asymmetric_partner_retrieval_controls_statistics.sbatch"
        ).read_text(encoding="utf-8")
        self.assertNotIn("SUITE_COMPLETION", statistics)
        self.assertNotIn("completion-receipt", statistics)

    def test_fit_explicitly_materializes_train_only(self) -> None:
        source = inspect.getsource(_fit_controls_under_process_lock)
        self.assertIn('_dataset_from_args(args, ("train",))', source)
        self.assertNotIn('("train", "validation")', source)
        self.assertIn('"validation_rows_materialized": False', source)
        wrapper = inspect.getsource(command_fit)
        self.assertIn("_validate_locked_fit_hyperparameters(args)", wrapper)
        self.assertIn("_validate_locked_model_contract(args)", wrapper)

    def test_fit_process_lock_precedes_all_dataset_and_embedding_access(self) -> None:
        source = inspect.getsource(command_fit)
        lock_index = source.index("exclusive_control_fit_process_lock")
        fit_index = source.index("_fit_controls_under_process_lock")
        self.assertLess(lock_index, fit_index)
        self.assertNotIn("_dataset_from_args", source)
        self.assertNotIn("_embedding_fingerprints", source)

    def test_fit_process_lock_is_path_independent_and_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fit-registry"
            authority = _digest("authority")
            verification = _digest("verification")
            first_identity, first_contract = control_fit_identity(authority, verification)
            second_identity, second_contract = control_fit_identity(authority, verification)
            self.assertEqual((first_identity, first_contract), (second_identity, second_contract))
            with exclusive_control_fit_process_lock(
                root, authority, verification, Path(directory) / "output-a"
            ):
                with self.assertRaisesRegex(RuntimeError, "already holds"):
                    with exclusive_control_fit_process_lock(
                        root,
                        authority,
                        verification,
                        Path(directory) / "output-b",
                    ):
                        self.fail("A concurrent duplicate fit unexpectedly acquired the lock")

    def test_shared_loader_requires_v2_boundary_mode(self) -> None:
        source = inspect.getsource(_dataset_from_args)
        self.assertIn("require_authority_validation_boundary_v2=True", source)
        self.assertIn("materialized_splits=materialized_splits", source)
        self.assertIn("expected_instances_sha256=args.prepared_instances_sha256", source)
        self.assertIn("canonical_sequence_manifest=args.canonical_sequence_manifest", source)

    def test_sequence_rows_are_reauthenticated_before_second_materialization(self) -> None:
        source = inspect.getsource(_read_sequences)
        first_hash_index = source.index("sha256_file(path)")
        csv_index = source.index("pd.read_csv")
        second_hash_index = source.index("sha256_file(path)", csv_index)
        self.assertLess(first_hash_index, csv_index)
        self.assertGreater(second_hash_index, csv_index)
        self.assertIn("LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256", source)

    @unittest.skipUnless(PANDAS_AVAILABLE, "pandas is unavailable")
    def test_sequence_reader_reconstructs_heavy_light_and_missing_light(self) -> None:
        rows = (
            "INSTANCE,vh_seq,vl_seq,ag_seq\n"
            "paired,VHSEQ,VLSEQ,ANTIGEN\n"
            "heavy_only,HEAVY,,TARGET\n"
        )
        records = [
            PairRecord(
                "paired",
                _digest("VHSEQVLSEQ"),
                _digest("ANTIGEN"),
                10,
                7,
            ),
            PairRecord(
                "heavy_only",
                _digest("HEAVY"),
                _digest("TARGET"),
                5,
                6,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.csv"
            path.write_text(rows, encoding="utf-8")
            with mock.patch(
                "sabdab2_retrieval_controls.LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256",
                sha256_file(path),
            ):
                antibodies, antigens = _read_sequences(path, records)
        self.assertEqual(
            antibodies,
            {_digest("VHSEQVLSEQ"): "VHSEQVLSEQ", _digest("HEAVY"): "HEAVY"},
        )
        self.assertEqual(
            antigens,
            {_digest("ANTIGEN"): "ANTIGEN", _digest("TARGET"): "TARGET"},
        )

    def test_raw_shared_esm_antibody_artifacts_are_mandatory(self) -> None:
        source = inspect.getsource(_validate_training_embedding_artifacts)
        self.assertIn('stores["shared_esm_antibody"].load', source)
        self.assertNotIn("except FileNotFoundError", source)
        self.assertNotIn("continue", source)

    def test_embedding_inventory_is_checked_around_physical_loading(self) -> None:
        fit_source = inspect.getsource(_fit_controls_under_process_lock)
        self.assertGreaterEqual(fit_source.count("_embedding_fingerprints"), 2)
        self.assertIn("changed during fit authentication", fit_source)
        evaluation_source = inspect.getsource(command_evaluate)
        self.assertGreaterEqual(evaluation_source.count("_embedding_fingerprints"), 3)
        self.assertIn("changed during authenticated loading", evaluation_source)

    def test_evaluation_ledger_precedes_dataset_materialization(self) -> None:
        source = inspect.getsource(command_evaluate)
        ledger_index = source.index("reserve_control_evaluation")
        dataset_index = source.index("_dataset_from_args")
        self.assertLess(ledger_index, dataset_index)
        self.assertIn('"frozen_test_used": False', source)

    def test_control_bundle_is_hashed_and_loaded_from_one_byte_buffer(self) -> None:
        source = inspect.getsource(_load_bundle)
        self.assertIn("bundle_bytes = target.read_bytes()", source)
        self.assertIn("hashlib.sha256(bundle_bytes).hexdigest()", source)
        self.assertIn("np.load(io.BytesIO(bundle_bytes)", source)
        self.assertNotIn("np.load(target", source)

    def test_locked_training_fingerprints_reject_any_inventory_change(self) -> None:
        exact = {
            name: {"content_manifest_sha256": digest}
            for name, digest in LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256.items()
        }
        self.assertEqual(
            _validate_locked_training_embedding_fingerprints(exact),
            LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256,
        )
        changed = {name: dict(value) for name, value in exact.items()}
        changed["shared_esm_antibody"]["content_manifest_sha256"] = _digest("changed")
        with self.assertRaisesRegex(ValueError, "final v1.6 lock"):
            _validate_locked_training_embedding_fingerprints(changed)

    def test_different_output_paths_collide_in_the_same_canonical_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "registry"
            bundle = _digest("bundle")
            split = _digest("split")
            authority = _digest("authority")
            first, identity, _ = reserve_control_evaluation(
                root,
                bundle,
                split,
                authority,
                Path(directory) / "output-a",
            )
            self.assertEqual(first.name, f"{identity}.json")
            with self.assertRaises(FileExistsError):
                reserve_control_evaluation(
                    root,
                    bundle,
                    split,
                    authority,
                    Path(directory) / "output-b",
                )

    def test_v2_accounting_excludes_rows_from_materialization(self) -> None:
        train = ["train-a", "train-b"]
        validation = ["validation-a"]
        excluded = ["validation-excluded"]

        def membership(values: list[str]) -> str:
            return hashlib.sha256(
                ("\n".join(sorted(values)) + "\n").encode("utf-8")
            ).hexdigest()

        boundary = {
            "format_version": "sabdab2_authority_validation_nw_boundary_v2",
            "intended_use": "reused_development_confirmation_only",
            "constraint_mode": "concat_antibody_and_antigen",
            "valid": True,
            "frozen_test_identifiers_read_for_exclusion_only": True,
            "frozen_test_sequence_fields_accessed_normalized_hashed_or_aligned": False,
            "splits": {
                "train": train,
                "validation": validation,
                "excluded_validation": excluded,
            },
            "membership_sha256": {
                "train": membership(train),
                "validation": membership(validation),
                "excluded_validation": membership(excluded),
            },
            "selection_rule": {
                "authority_train_moved_to_validation": False,
                "excluded_authority_validation_added_to_train": False,
                "size_target_or_random_subsampling_used": False,
            },
        }
        authority = {
            "splits": {
                "train": train,
                "validation": [*validation, *excluded],
                "test": ["sealed-test"],
            }
        }
        accounting = validate_boundary_v2_accounting(
            boundary,
            authority,
            "concat_antibody_and_antigen",
            ("train",),
        )
        self.assertEqual(accounting["excluded_validation_count"], 1)
        self.assertFalse(accounting["excluded_validation_materialized"])
        with self.assertRaisesRegex(PermissionError, "never materialized"):
            validate_boundary_v2_accounting(
                boundary,
                authority,
                "concat_antibody_and_antigen",
                ("train", "excluded_validation"),
            )

    def test_common_train_contract_is_independent_of_boundary_holdout(self) -> None:
        common = {
            "instances_sha256": _digest("instances"),
            "canonical_sequence_manifest": {
                "path": "/locked/canonical.csv",
                "sha256": _digest("canonical"),
            },
            "authority": {"sha256": _digest("authority")},
            "split_membership_sha256": {"train": _digest("train-membership")},
            "raw_instance_counts": {"train": 3229},
            "excluded_validation_accounting": {
                "excluded_validation_materialized": False
            },
        }
        concat = {
            **common,
            "split_manifest_sha256": _digest("concat-boundary"),
            "boundary_constraint_mode": "concat_antibody_and_antigen",
        }
        chain = {
            **common,
            "split_manifest_sha256": _digest("chain-boundary"),
            "boundary_constraint_mode": "concat_antibody_antigen_plus_vh_vl",
        }
        self.assertEqual(
            _common_training_dataset_contract(concat),
            _common_training_dataset_contract(chain),
        )


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is unavailable")
class NumericalControlTests(unittest.TestCase):
    def test_sequence_and_pair_descriptors_are_finite_and_fixed_width(self) -> None:
        import numpy as np

        antibody = sequence_descriptor("ACDEFGHIKLMNPQRSTVWYX")
        antigen = sequence_descriptor("AAAAACCCCCDDDDD")
        pair = pair_descriptor(antibody, antigen)
        self.assertEqual(antibody.shape, (28,))
        self.assertEqual(pair.shape, (100,))
        self.assertTrue(np.isfinite(pair).all())
        self.assertAlmostEqual(float(antibody[1:22].sum()), 1.0)

    def test_endpoint_permutation_is_deterministic_and_excludes_positives(self) -> None:
        positives = [
            (_digest("a1"), _digest("g1")),
            (_digest("a2"), _digest("g2")),
            (_digest("a3"), _digest("g3")),
            (_digest("a4"), _digest("g4")),
        ]
        first, first_summary = deterministic_endpoint_matched_negatives(positives, 5, 7)
        second, second_summary = deterministic_endpoint_matched_negatives(positives, 5, 7)
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        self.assertTrue(first)
        self.assertFalse(set(first) & set(positives))

    def test_logistic_fit_is_deterministic_and_separates_toy_data(self) -> None:
        import numpy as np

        positive = np.asarray([[2.0, 1.0], [3.0, 1.0], [4.0, 2.0]])
        negative = np.asarray([[-2.0, -1.0], [-3.0, -1.0], [-4.0, -2.0]])
        first, summary = fit_fixed_logistic(positive, negative, 0.01, 50, 1e-9)
        second, _ = fit_fixed_logistic(positive, negative, 0.01, 50, 1e-9)
        np.testing.assert_array_equal(first["coefficients"], second["coefficients"])
        self.assertTrue(summary["converged"])
        pos_design = np.column_stack(
            [np.ones(len(positive)), (positive - first["mean"]) / first["scale"]]
        )
        neg_design = np.column_stack(
            [np.ones(len(negative)), (negative - first["mean"]) / first["scale"]]
        )
        self.assertGreater(
            float((pos_design @ first["coefficients"]).mean()),
            float((neg_design @ first["coefficients"]).mean()),
        )

    def test_random_heads_are_reproducible_and_unit_normalized(self) -> None:
        import numpy as np

        first = make_random_head_arrays(5, 7, 11, 3, 13)
        second = make_random_head_arrays(5, 7, 11, 3, 13)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        vector = random_head_pool(np.arange(25, dtype=np.float32).reshape(5, 5), first, "random_ab")
        self.assertEqual(vector.shape, (3,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_metrics_use_stable_candidate_index_tie_breaking(self) -> None:
        import numpy as np

        antibody_keys = [_digest("a1"), _digest("a2")]
        antigen_keys = [_digest("g1"), _digest("g2"), _digest("g3")]
        scores = np.zeros((2, 3), dtype=np.float32)
        mask = np.asarray([[False, True, False], [True, False, True]])
        metrics = metrics_from_scores(scores, mask, antibody_keys, antigen_keys)
        ranks = metrics["per_anchor_ranks"]["antibody_to_antigen"]
        self.assertEqual([record["best_known_positive_rank"] for record in ranks], [2, 1])
        self.assertEqual(metrics["exact_pair_count"], 3)

    def test_unary_metrics_are_invariant_to_library_input_order(self) -> None:
        import numpy as np

        antibody_keys = [_digest("a1"), _digest("a2"), _digest("a3")]
        antigen_keys = [_digest("g1"), _digest("g2"), _digest("g3"), _digest("g4")]
        antibody_logits = np.asarray([0.2, 0.2, -1.0])
        antigen_logits = np.asarray([0.1, 0.5, 0.5, -0.2])
        mask = np.asarray(
            [
                [True, False, False, True],
                [False, True, False, False],
                [False, False, True, False],
            ]
        )
        original_scores = unary_score_matrices(
            antibody_logits, antigen_logits, antibody_keys, antigen_keys
        )
        original_metrics = [
            metrics_from_scores(scores, mask, antibody_keys, antigen_keys)
            for scores in original_scores
        ]
        antibody_order = np.asarray([2, 0, 1])
        antigen_order = np.asarray([3, 1, 0, 2])
        permuted_scores = unary_score_matrices(
            antibody_logits[antibody_order],
            antigen_logits[antigen_order],
            [antibody_keys[index] for index in antibody_order],
            [antigen_keys[index] for index in antigen_order],
        )
        permuted_mask = mask[np.ix_(antibody_order, antigen_order)]
        permuted_metrics = [
            metrics_from_scores(
                scores,
                permuted_mask,
                [antibody_keys[index] for index in antibody_order],
                [antigen_keys[index] for index in antigen_order],
            )
            for scores in permuted_scores
        ]
        for original, permuted in zip(original_metrics, permuted_metrics):
            for direction in ("antibody_to_antigen", "antigen_to_antibody"):
                for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10"):
                    self.assertEqual(original[direction][metric], permuted[direction][metric])

    def test_bundle_hashes_arrays_and_rejects_tampering(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controls.npz"
            arrays = {"weights": np.arange(6, dtype=np.float32).reshape(2, 3)}
            config = {"config_sha256": _digest("config")}
            receipt = _write_bundle(path, arrays, config)
            loaded = _load_bundle(path, receipt["sha256"])
            np.testing.assert_array_equal(loaded["arrays"]["weights"], arrays["weights"])
            with path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                _load_bundle(path, receipt["sha256"])

    def test_bundle_metadata_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        import numpy as np

        malformed = {
            "duplicate": b'{"format_version":"x","format_version":"x"}',
            "nonfinite": b'{"format_version":"x","value":NaN}',
            "overflow": b'{"format_version":"x","nested":[{"value":1e999}]}',
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, metadata in malformed.items():
                path = Path(directory) / f"{name}.npz"
                _atomic_write_deterministic_npz(
                    path,
                    {
                        "weights": np.asarray([1.0], dtype=np.float32),
                        "metadata_json_utf8": np.frombuffer(
                            metadata, dtype=np.uint8
                        ).copy(),
                    },
                )
                expected = "duplicate key" if name == "duplicate" else "non-finite"
                with self.assertRaisesRegex(ValueError, expected):
                    _load_bundle(path, sha256_file(path))

    def test_bundle_rejects_duplicate_raw_zip_members(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.npz"
            duplicate_path = root / "duplicate.npz"
            _write_bundle(
                valid_path,
                {"weights": np.asarray([1.0], dtype=np.float32)},
                {"config_sha256": _digest("config")},
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
                _load_bundle(duplicate_path, sha256_file(duplicate_path))

    def test_create_only_publication_never_clobbers_late_or_existing_targets(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_target = root / "late.bundle.npz"
            sentinel = b"late-created-bundle"

            def racing_writer(staged: Path, arrays: object) -> None:
                _atomic_write_deterministic_npz(staged, arrays)
                bundle_target.write_bytes(sentinel)

            with mock.patch(
                "sabdab2_retrieval_controls._atomic_write_deterministic_npz",
                side_effect=racing_writer,
            ):
                with self.assertRaises(FileExistsError):
                    _write_bundle(
                        bundle_target,
                        {"weights": np.asarray([1.0], dtype=np.float32)},
                        {"config_sha256": _digest("config")},
                    )
            self.assertEqual(bundle_target.read_bytes(), sentinel)

            json_target = root / "late.evaluation.json"
            json_target.write_bytes(b"late-created-json")
            with self.assertRaises(FileExistsError):
                _create_exclusive_json(json_target, {"replacement": True})
            self.assertEqual(json_target.read_bytes(), b"late-created-json")

            score_target = root / "late.scores.npz"
            score_target.write_bytes(b"late-created-score")
            with self.assertRaises(FileExistsError):
                _write_score_artifact_create_only(
                    score_target,
                    {
                        "similarity_matrix": np.eye(2, dtype=np.float32),
                        "positive_mask": np.eye(2, dtype=bool),
                        "antibody_keys": [_digest("race-a1"), _digest("race-a2")],
                        "antigen_keys": [_digest("race-g1"), _digest("race-g2")],
                    },
                    {"synthetic": True},
                )
            self.assertEqual(score_target.read_bytes(), b"late-created-score")

    def test_score_npz_is_accepted_by_statistics_loader(self) -> None:
        import numpy as np

        from sabdab2_asymmetric_retrieval_statistics import load_score_artifact

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.scores.npz"
            antibody_keys = sorted([_digest("a1"), _digest("a2")])
            antigen_keys = sorted([_digest("g1"), _digest("g2")])
            receipt = write_score_artifact(
                path,
                {
                    "similarity_matrix": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                    "positive_mask": np.eye(2, dtype=bool),
                    "antibody_keys": antibody_keys,
                    "antigen_keys": antigen_keys,
                },
                {"synthetic": True},
            )
            loaded = load_score_artifact(path)
            self.assertEqual(loaded["sha256"], receipt["sha256"])
            self.assertEqual(loaded["antibody_keys"], antibody_keys)
            self.assertEqual(loaded["antigen_keys"], antigen_keys)

    def test_two_boundary_finalizer_physically_authenticates_all_ten_outputs(self) -> None:
        import numpy as np

        repository = Path(__file__).resolve().parents[2]
        protocol = (
            repository
            / "outputs/interaction_first/research/"
            "PARTNER_RETRIEVAL_PROTOCOL_V1_6_EXECUTION_HARDENING.md"
        ).resolve()
        control_protocol = (
            repository
            / "outputs/interaction_first/research/PARTNER_RETRIEVAL_CONTROLS_PROTOCOL_V1.md"
        ).resolve()
        prepared_instances = (
            repository / "outputs/data_snapshot/instances.parquet"
        ).resolve()
        canonical_manifest = (
            repository / "outputs/interaction_first/remote/canonical_manifest.csv"
        ).resolve()
        boundary_verification = (
            repository
            / "outputs/interaction_first/research/authority_validation_nw_boundaries/"
            "independent_verification_v2.json"
        ).resolve()
        authority_split = (
            repository / "outputs/data_snapshot/double_remote_split.json"
        ).resolve()
        boundary_paths = {
            "concat_antibody_and_antigen": (
                repository
                / "outputs/interaction_first/research/authority_validation_nw_boundaries/"
                "concat_boundary_v2.json"
            ).resolve(),
            "concat_antibody_antigen_plus_vh_vl": (
                repository
                / "outputs/interaction_first/research/authority_validation_nw_boundaries/"
                "chain_boundary_v2.json"
            ).resolve(),
        }
        verification_payload = json.loads(boundary_verification.read_text(encoding="utf-8"))
        approved_modes = {
            mode: {
                "boundary_sha256": sha256_file(path),
                "boundary_scientific_payload_sha256": verification_payload["modes"][mode][
                    "boundary"
                ]["scientific_payload_sha256"],
            }
            for mode, path in boundary_paths.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            registry = root / "control-registry"
            registry.mkdir()
            config_without_hash = {
                "control_protocol_v1": {
                    "path": str(control_protocol),
                    "sha256": CONTROL_PROTOCOL_V1_SHA256,
                },
                "execution_protocol_v1_6": {
                    "path": str(protocol),
                    "sha256": EXECUTION_PROTOCOL_V1_6_SHA256,
                },
                "locked_training_embedding_content_manifest_sha256": dict(
                    LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256
                ),
                "approved_boundaries": {"modes": approved_modes},
            }
            config = {
                **config_without_hash,
                "config_sha256": canonical_sha256(config_without_hash),
            }
            bundle_path = root / "controls.bundle.npz"
            bundle_receipt = _write_bundle(
                bundle_path,
                {"synthetic": np.asarray([1.0], dtype=np.float32)},
                config,
            )
            antibody_keys = sorted([_digest("suite-a1"), _digest("suite-a2")])
            antigen_keys = sorted([_digest("suite-g1"), _digest("suite-g2")])
            scores = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            positive_mask = np.eye(2, dtype=bool)
            ledger_references: dict[str, dict[str, str]] = {}
            for mode, boundary_path in boundary_paths.items():
                boundary_sha256 = sha256_file(boundary_path)
                ledger_path, identity, ledger_contract = reserve_control_evaluation(
                    registry,
                    bundle_receipt["sha256"],
                    boundary_sha256,
                    sha256_file(authority_split),
                    root / f"{mode}.outputs",
                )
                output_dir = root / f"{mode}.outputs"
                output_dir.mkdir()
                selected_boundary = {
                    "constraint_mode": mode,
                    "path": str(boundary_path),
                    "sha256": boundary_sha256,
                    "scientific_payload_sha256": approved_modes[mode][
                        "boundary_scientific_payload_sha256"
                    ],
                }
                locked_evidence = {
                    "format_version": "sabdab2_control_locked_submission_evidence_v1",
                    "control_protocol_v1": {
                        "path": str(control_protocol),
                        "sha256": CONTROL_PROTOCOL_V1_SHA256,
                    },
                    "execution_protocol_v1_6": {
                        "path": str(protocol),
                        "sha256": EXECUTION_PROTOCOL_V1_6_SHA256,
                    },
                    "prepared_instances": {
                        "path": str(prepared_instances),
                        "sha256": LOCKED_PREPARED_INSTANCES_SHA256,
                    },
                    "canonical_sequence_manifest": {
                        "path": str(canonical_manifest),
                        "sha256": LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256,
                    },
                    "training_embedding_content_manifest_sha256": dict(
                        LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256
                    ),
                    "boundary_verification_v2": {
                        "path": str(boundary_verification),
                        "sha256": sha256_file(boundary_verification),
                    },
                    "authority_split": {
                        "path": str(authority_split),
                        "sha256": sha256_file(authority_split),
                    },
                    "selected_boundary": selected_boundary,
                    "required_boundary_modes": [
                        "concat_antibody_and_antigen",
                        "concat_antibody_antigen_plus_vh_vl",
                    ],
                    "control_suite_completion_required": True,
                    "frozen_test_used": False,
                }
                output_records: dict[str, dict[str, object]] = {}
                for control_name in CONTROL_NAMES:
                    control_spec = {
                        "protocol_version": "partner_retrieval_controls_protocol_v1",
                        "name": control_name,
                        "bundle_config_sha256": config["config_sha256"],
                    }
                    evaluation = {
                        "format_version": "sabdab2_asymmetric_partner_retrieval_v1",
                        "stage": "full_library_development_validation",
                        "split": "validation",
                        "frozen_test_used": False,
                        "checkpoint": str(bundle_path),
                        "checkpoint_sha256": bundle_receipt["sha256"],
                        "control_bundle_sha256": bundle_receipt["sha256"],
                        "config_sha256": canonical_sha256(control_spec),
                        "dataset": {
                            "instances_sha256": LOCKED_PREPARED_INSTANCES_SHA256,
                            "canonical_sequence_manifest": {
                                "path": str(canonical_manifest),
                                "sha256": LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256,
                            },
                            "split_manifest_sha256": boundary_sha256,
                            "split_format_version": (
                                "sabdab2_authority_validation_nw_boundary_v2"
                            ),
                            "split_intended_use": (
                                "reused_development_confirmation_only"
                            ),
                            "strict_nw_constraint_mode": mode,
                            "authority": {
                                "sha256": sha256_file(authority_split),
                                "authorized_source_pool_splits": ["train", "validation"],
                                "sealed_source_splits": ["test"],
                                "selected_sealed_overlap_count": 0,
                            },
                        },
                        "embedding_corpora": {
                            "training": {
                                name: {"content_manifest_sha256": digest}
                                for name, digest in (
                                    LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256.items()
                                )
                            },
                            "one_shot_validation": {"synthetic": True},
                        },
                        "metrics": metrics_from_scores(
                            scores,
                            positive_mask,
                            antibody_keys,
                            antigen_keys,
                        ),
                        "checkpoint_run_mode": "train_only_control_bundle",
                        "one_shot_evaluation": True,
                        "one_shot_ledger": {
                            "format_version": "sabdab2_control_one_shot_ledger_v3",
                            "canonical_key": identity,
                            "canonical_path": str(ledger_path),
                            "registry_dir": str(registry),
                        },
                        "control": control_spec,
                        "control_protocol_v1": locked_evidence[
                            "control_protocol_v1"
                        ],
                        "execution_protocol_v1_6": locked_evidence[
                            "execution_protocol_v1_6"
                        ],
                        "locked_submission_evidence": locked_evidence,
                    }
                    score_path = output_dir / f"{control_name}.scores.npz"
                    evaluation["score_artifact"] = write_score_artifact(
                        score_path,
                        {
                            "similarity_matrix": scores,
                            "positive_mask": positive_mask,
                            "antibody_keys": antibody_keys,
                            "antigen_keys": antigen_keys,
                        },
                        _evaluation_context(evaluation),
                    )
                    evaluation_path = output_dir / f"{control_name}.evaluation.json"
                    atomic_write_json(evaluation_path, evaluation)
                    output_records[control_name] = {
                        "evaluation_json": str(evaluation_path),
                        "evaluation_json_sha256": sha256_file(evaluation_path),
                        "score_artifact": str(score_path),
                        "score_artifact_sha256": evaluation["score_artifact"]["sha256"],
                        "score_contract_sha256": evaluation["score_artifact"][
                            "contract_sha256"
                        ],
                        "bidirectional_mrr": 1.0,
                    }
                atomic_write_json(
                    ledger_path,
                    {
                        **ledger_contract,
                        "evaluation_identity_sha256": identity,
                        "status": "completed",
                        "requested_output_dir": str(output_dir),
                        "control_protocol_v1": locked_evidence[
                            "control_protocol_v1"
                        ],
                        "execution_protocol_v1_6": locked_evidence[
                            "execution_protocol_v1_6"
                        ],
                        "locked_submission_evidence_sha256": canonical_sha256(
                            locked_evidence
                        ),
                        "control_outputs": output_records,
                    },
                )
                ledger_references[mode] = {
                    "path": str(ledger_path),
                    "sha256": sha256_file(ledger_path),
                }
            completion = finalize_control_suite(
                registry,
                bundle_path,
                bundle_receipt["sha256"],
                control_protocol,
                CONTROL_PROTOCOL_V1_SHA256,
                protocol,
                EXECUTION_PROTOCOL_V1_6_SHA256,
                ledger_references,
            )
            receipt = json.loads(Path(completion["path"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt["format_version"], CONTROL_SUITE_COMPLETION_FORMAT)
            self.assertEqual(set(receipt["boundaries"]), set(boundary_paths))
            self.assertTrue(
                all(
                    set(boundary["controls"]) == set(CONTROL_NAMES)
                    for boundary in receipt["boundaries"].values()
                )
            )
            identity_payload = dict(receipt)
            observed_identity = identity_payload.pop("completion_identity_sha256")
            self.assertEqual(observed_identity, canonical_sha256(identity_payload))
            tampered_score = Path(
                receipt["boundaries"]["concat_antibody_and_antigen"]["controls"][
                    CONTROL_NAMES[0]
                ]["score_artifact"]["path"]
            )
            with tampered_score.open("ab") as handle:
                handle.write(b"tamper")
            loaded_bundle = _load_bundle(bundle_path, bundle_receipt["sha256"])
            with self.assertRaisesRegex(ValueError, "physical SHA-256 mismatch"):
                _verify_completed_control_boundary(
                    registry,
                    "concat_antibody_and_antigen",
                    ledger_references["concat_antibody_and_antigen"],
                    loaded_bundle,
                    {
                        "path": str(protocol),
                        "sha256": EXECUTION_PROTOCOL_V1_6_SHA256,
                    },
                )


if __name__ == "__main__":
    unittest.main()
