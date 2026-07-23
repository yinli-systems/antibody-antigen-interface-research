#!/usr/bin/env python3
"""Tests for the development-only asymmetric partner retriever."""

from __future__ import annotations

import hashlib
import copy
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from sabdab2_asymmetric_partner_retrieval import (
    ANTIBERTY_FORMAT,
    ESM_FORMAT,
    FORMAT_VERSION,
    FINAL_REFIT_AUTHORITY_SHA256,
    FINAL_REFIT_ANTIBODY_EMBEDDING_MANIFEST_SHA256,
    FINAL_REFIT_ANTIGEN_EMBEDDING_MANIFEST_SHA256,
    FINAL_REFIT_BEST_EPOCHS,
    FINAL_REFIT_BOUNDARY_VERIFICATION_SCIENTIFIC_SHA256,
    FINAL_REFIT_BOUNDARY_VERIFICATION_SHA256,
    FINAL_REFIT_FIXED_EPOCHS,
    FINAL_REFIT_FOLD_SELECTION_SHA256,
    FINAL_REFIT_INDEPENDENT_AUDIT_SHA256,
    FINAL_REFIT_PROTOCOL_V1_3_SHA256,
    FINAL_REFIT_PROTOCOL_V1_4_SHA256,
    FINAL_REFIT_PROTOCOL_V1_5_SHA256,
    FINAL_REFIT_PROTOCOL_V1_6_SHA256,
    FINAL_REFIT_PREPARED_INSTANCES_SHA256,
    FINAL_REFIT_CANONICAL_SEQUENCE_MANIFEST_SHA256,
    FINAL_REFIT_SELECTED_VARIANT,
    FINAL_REFIT_SUBMISSION_FORMAT_VERSION,
    FrozenPartnerEmbeddingStore,
    PairRecord,
    _canonical_one_shot_ledger_contract,
    _atomic_write_deterministic_npz,
    _create_only_json,
    _create_only_torch_save,
    _create_one_shot_ledger,
    _expected_final_evaluation_contract,
    _expected_final_refit_model_contract,
    _expected_final_refit_optimization_contract,
    _exclusive_final_refit_process_lock,
    _final_refit_evaluation_contract,
    _ensure_checkpoint_suite_claim,
    _load_torch_checkpoint_once,
    _model_from_config,
    _optimizer,
    _provenance,
    _recover_final_refit_completion_if_present,
    _scheduler,
    _source_hashes,
    _strict_json_object,
    _validate_final_refit_resume_state,
    audit_isolation_policy,
    build_pair_masks,
    command_final_refit,
    _command_final_refit_locked,
    load_development_dataset,
    load_excluded_negative_pairs,
    side_embedding_fingerprint,
    validate_final_refit_submission_manifest,
    write_score_artifact,
)
from sabdab2_partner_retrieval import canonical_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _membership_digest(identifiers: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(identifiers)) + "\n").encode("utf-8")
    ).hexdigest()


def _synthetic_boundary_payload(
    authority_sha256: str,
    train: list[str],
    validation: list[str],
    excluded_validation: list[str],
    mode: str = "concat_antibody_and_antigen",
) -> dict[str, object]:
    views = (
        ["concat_antibody", "antigen"]
        if mode == "concat_antibody_and_antigen"
        else ["concat_antibody", "vh", "vl", "antigen"]
    )
    payload: dict[str, object] = {
        "format_version": "sabdab2_authority_validation_nw_boundary_v2",
        "stage": "build_authority_validation_nw_clean_boundary",
        "constraint_mode": mode,
        "declared_views": views,
        "intended_use": "reused_development_confirmation_only",
        "valid": True,
        "selection_influenced_by_labels_predictions_or_model_scores": False,
        "frozen_test_identifiers_read_for_exclusion_only": True,
        "frozen_test_sequence_fields_accessed_normalized_hashed_or_aligned": False,
        "splits": {
            "train": train,
            "validation": validation,
            "excluded_validation": excluded_validation,
        },
        "selection_rule": {
            "component_universe": (
                "pre-existing authoritative graph components assigned wholly to "
                "authority validation"
            ),
            "eligibility": (
                "retain every component with zero 80%-identity/80%-reciprocal-coverage "
                "exact global NW edge from any member to authority train in every "
                "declared view"
            ),
            "size_target_or_random_subsampling_used": False,
            "authority_train_moved_to_validation": False,
            "excluded_authority_validation_added_to_train": False,
        },
        "membership_sha256": {
            "train": _membership_digest(train),
            "validation": _membership_digest(validation),
            "excluded_validation": _membership_digest(excluded_validation),
        },
        "counts": {
            "authority_train": len(train),
            "authority_validation": len(validation) + len(excluded_validation),
            "sealed_test": 1,
            "eligible_validation_instances": len(validation),
            "excluded_validation_instances": len(excluded_validation),
        },
        "evidence": {
            "identity_threshold": 0.8,
            "reciprocal_coverage_threshold": 0.8,
            "authority_train_by_validation_exact_nw_artifacts": {
                view: {
                    "path": f"/synthetic/{view}.jsonl",
                    "sha256": _digest(f"{view}-edges"),
                    "edge_count": 0,
                    "validation_digest_count_with_train_edge": 0,
                }
                for view in views
            },
        },
        "provenance": {
            "authority_split": {
                "path": "/synthetic/authority.json",
                "sha256": authority_sha256,
            },
            "protocol_correction": {
                "path": "/synthetic/PARTNER_RETRIEVAL_PROTOCOL_V1_4_CORRECTION.md",
                "sha256": FINAL_REFIT_PROTOCOL_V1_4_SHA256,
            },
        },
        "chronology": {
            "authority_validation_previously_inspected_by_esm2_exploration": True,
            "clean_relative_to_antiberty_three_fold_architecture_selection": True,
            "globally_untouched_or_independent_test": False,
        },
        "limitations": [
            "This is a reused development confirmation subset, not an independent test.",
            "Eligibility uses the source authority graph components plus exhaustive exact-NW train-validation edges; it does not assert structural or epitope independence.",
            "External or prospective claims require separately locked external or temporal data.",
        ],
    }
    payload["scientific_payload_sha256"] = canonical_sha256(payload)
    return payload


def _submission_contract_fixture(
    root: Path,
) -> tuple[Path, Namespace, dict[str, object]]:
    """Create a physical archival contract without touching any dataset artifact."""
    script = Path(__file__).with_name("sabdab2_asymmetric_partner_retrieval.py").resolve()
    job_root = script.parent
    project_root = script.parents[2]
    archive = root / "read-only-source-snapshot"
    archive.mkdir(parents=True)
    live_sources = {
        "entrypoint": script,
        "asymmetric_retrieval": job_root / "interaction_first" / "asymmetric_retrieval.py",
        "shared_retrieval_protocol": job_root / "sabdab2_partner_retrieval.py",
        "runtime": job_root / "interaction_first" / "runtime.py",
    }
    source_files = {}
    for name, live_path in live_sources.items():
        archived = archive / f"{name}{live_path.suffix}"
        shutil.copyfile(live_path, archived)
        source_files[name] = {"path": str(archived), "sha256": _file_digest(archived)}
    live_sbatch = job_root / "asymmetric_partner_retrieval_clean_holdout_train.sbatch"
    archived_sbatch = archive / live_sbatch.name
    shutil.copyfile(live_sbatch, archived_sbatch)

    evidence_root = project_root / "outputs" / "interaction_first" / "research"
    fold_selection = (
        evidence_root
        / "asymmetric_retrieval_fold_aggregation"
        / "fold_selection.json"
    )
    independent_audit = (
        evidence_root
        / "asymmetric_retrieval_fold_screen_audit"
        / "audit.json"
    )
    protocol = evidence_root / "PARTNER_RETRIEVAL_PROTOCOL_V1_3_AMENDMENT.md"
    protocol_correction = evidence_root / "PARTNER_RETRIEVAL_PROTOCOL_V1_4_CORRECTION.md"
    protocol_execution = evidence_root / "PARTNER_RETRIEVAL_PROTOCOL_V1_5_EXECUTION_LOCK.md"
    protocol_hardening = (
        evidence_root / "PARTNER_RETRIEVAL_PROTOCOL_V1_6_EXECUTION_HARDENING.md"
    )
    boundary_verification = (
        evidence_root
        / "authority_validation_nw_boundaries"
        / "independent_verification_v2.json"
    )
    authority = project_root / "outputs" / "data_snapshot" / "double_remote_split.json"
    prepared_instances = project_root / "outputs" / "data_snapshot" / "instances.parquet"
    canonical_sequence_manifest = (
        project_root / "outputs" / "interaction_first" / "remote" / "canonical_manifest.csv"
    )
    boundary_root = evidence_root / "authority_validation_nw_boundaries"
    split = boundary_root / "concat_boundary_v2.json"
    chain_split = boundary_root / "chain_boundary_v2.json"
    output = root / "final-refit-output"
    registry = root / "central-one-shot-registry"
    common_train_sha256 = json.loads(split.read_text(encoding="utf-8"))[
        "membership_sha256"
    ]["train"]
    payload: dict[str, object] = {
        "format_version": FINAL_REFIT_SUBMISSION_FORMAT_VERSION,
        "intended_use": "development_only",
        "frozen_test_used": False,
        "selected_variant": FINAL_REFIT_SELECTED_VARIANT,
        "upstream_evidence": {
            "fold_selection": {
                "path": str(fold_selection),
                "sha256": FINAL_REFIT_FOLD_SELECTION_SHA256,
                "selected_variant": FINAL_REFIT_SELECTED_VARIANT,
                "best_epochs": list(FINAL_REFIT_BEST_EPOCHS),
                "median_best_epoch": FINAL_REFIT_FIXED_EPOCHS,
            },
            "independent_audit": {
                "path": str(independent_audit),
                "sha256": FINAL_REFIT_INDEPENDENT_AUDIT_SHA256,
            },
            "protocol_v1_3": {
                "path": str(protocol),
                "sha256": FINAL_REFIT_PROTOCOL_V1_3_SHA256,
            },
            "protocol_v1_4_correction": {
                "path": str(protocol_correction),
                "sha256": FINAL_REFIT_PROTOCOL_V1_4_SHA256,
            },
            "protocol_v1_5_execution_lock": {
                "path": str(protocol_execution),
                "sha256": FINAL_REFIT_PROTOCOL_V1_5_SHA256,
            },
            "protocol_v1_6_execution_hardening": {
                "path": str(protocol_hardening),
                "sha256": FINAL_REFIT_PROTOCOL_V1_6_SHA256,
            },
            "boundary_verification_v2": {
                "path": str(boundary_verification),
                "sha256": FINAL_REFIT_BOUNDARY_VERIFICATION_SHA256,
            },
        },
        "authority": {
            "split_manifest": {
                "path": str(authority),
                "sha256": FINAL_REFIT_AUTHORITY_SHA256,
            },
            "authorized_source_pool_splits": ["train", "validation"],
            "sealed_source_splits": ["test"],
            "coverage_mode": "exact",
        },
        "immutable_data_inputs": {
            "prepared_instances": {
                "path": str(prepared_instances),
                "sha256": FINAL_REFIT_PREPARED_INSTANCES_SHA256,
            },
            "canonical_sequence_manifest": {
                "path": str(canonical_sequence_manifest),
                "sha256": FINAL_REFIT_CANONICAL_SEQUENCE_MANIFEST_SHA256,
            },
            "training_embedding_content_manifest_sha256": {
                "antibody": FINAL_REFIT_ANTIBODY_EMBEDDING_MANIFEST_SHA256,
                "antigen": FINAL_REFIT_ANTIGEN_EMBEDDING_MANIFEST_SHA256,
            },
        },
        "model": _expected_final_refit_model_contract(),
        "fixed_refit": _expected_final_refit_optimization_contract(),
        "fixed_evaluation": _expected_final_evaluation_contract(),
        "common_train_membership_sha256": common_train_sha256,
        "canonical_refit_boundary": "repaired_concat_boundary",
        "refit_output_dir": str(output),
        "one_shot_ledger_registry_dir": str(registry),
        "boundaries": {
            "repaired_concat_boundary": {
                "split_manifest": {
                    "path": str(split),
                    "sha256": _file_digest(split),
                },
                "strict_nw_constraint_mode": "concat_antibody_and_antigen",
                "evaluation_output": str(root / "concat-evaluation.json"),
                "score_artifact": str(root / "concat-scores.npz"),
            },
            "repaired_chain_boundary": {
                "split_manifest": {
                    "path": str(chain_split),
                    "sha256": _file_digest(chain_split),
                },
                "strict_nw_constraint_mode": "concat_antibody_antigen_plus_vh_vl",
                "evaluation_output": str(root / "chain-evaluation.json"),
                "score_artifact": str(root / "chain-scores.npz"),
            },
        },
        "source_files": source_files,
        "submission_script": {
            "path": str(archived_sbatch),
            "sha256": _file_digest(archived_sbatch),
        },
    }
    manifest = root / "submission-manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model = _expected_final_refit_model_contract()
    fixed = _expected_final_refit_optimization_contract()
    optimizer = fixed["optimizer"]
    scheduler = fixed["scheduler"]
    args = Namespace(
        submission_manifest=manifest,
        submission_manifest_sha256=_file_digest(manifest),
        split_manifest=split,
        split_manifest_sha256=_file_digest(split),
        strict_nw_constraint_mode="concat_antibody_and_antigen",
        output_dir=output,
        prepare_dir=prepared_instances.parent,
        prepared_instances_sha256=FINAL_REFIT_PREPARED_INSTANCES_SHA256,
        canonical_sequence_manifest=canonical_sequence_manifest,
        canonical_sequence_manifest_sha256=(
            FINAL_REFIT_CANONICAL_SEQUENCE_MANIFEST_SHA256
        ),
        authority_split_manifest=authority,
        authority_split_sha256=FINAL_REFIT_AUTHORITY_SHA256,
        authorized_source_pool_splits=["train", "validation"],
        authority_sealed_splits=["test"],
        authority_coverage_mode="exact",
        antibody_artifact_format=model["antibody_artifact_format"],
        antibody_model_name=model["antibody_model_name"],
        antibody_model_weights_sha256=model["antibody_model_weights_sha256"],
        antigen_model_name=model["antigen_model_name"],
        antigen_model_weights_sha256=model["antigen_model_weights_sha256"],
        antibody_input_dim=model["antibody_input_dim"],
        antigen_input_dim=model["antigen_input_dim"],
        embedding_dim=model["embedding_dim"],
        antibody_hidden_dim=model["antibody_hidden_dim"],
        antigen_hidden_dim=model["antigen_hidden_dim"],
        dropout=model["dropout"],
        chain_balanced_antibody_pooling=model["chain_balanced_antibody_pooling"],
        antibody_pooling=model["antibody_pooling"],
        antigen_pooling=model["antigen_pooling"],
        initial_temperature=model["initial_temperature"],
        minimum_temperature=model["minimum_temperature"],
        maximum_temperature=model["maximum_temperature"],
        fixed_epochs=fixed["fixed_epochs"],
        seed=fixed["seed"],
        batch_size=fixed["batch_size"],
        bucket_width=fixed["bucket_width"],
        recall_ks=[1, 5, 10, 20],
        device=fixed["training_device"],
        allow_nondeterministic=False,
        isolation_policy=fixed["isolation_policy"],
        excluded_negative_pairs=None,
        learning_rate=optimizer["learning_rate"],
        minimum_learning_rate=optimizer["minimum_learning_rate"],
        weight_decay=optimizer["weight_decay"],
        gradient_clip=optimizer["gradient_clip"],
        schedule=scheduler["name"],
        warmup_fraction=scheduler["warmup_fraction"],
        restart_period_epochs=scheduler["restart_period_epochs"],
        restart_multiplier=scheduler["restart_multiplier"],
    )
    return manifest, args, payload


class SubmissionProvenanceTests(unittest.TestCase):
    def test_final_refit_template_uses_fragmentation_safe_cuda_allocator(self) -> None:
        script = (
            Path(__file__).resolve().parent
            / "asymmetric_partner_retrieval_clean_holdout_train.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            script,
        )
        self.assertIn('if [[ "$BATCH_SIZE" != "1024" ]]', script)

    def test_strict_json_rejects_nested_exponent_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overflow.json"
            path.write_text('{"outer":[{"value":1e999}]}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON value"):
                _strict_json_object(path, "synthetic manifest")

    def test_physical_submission_contract_is_checkpoint_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, args, _ = _submission_contract_fixture(Path(directory))
            provenance = validate_final_refit_submission_manifest(args)
            self.assertEqual(
                provenance["submission_manifest"]["sha256"], _file_digest(manifest)
            )
            self.assertEqual(
                provenance["selected_boundary"], "repaired_concat_boundary"
            )
            self.assertEqual(
                provenance["upstream_evidence"]["fold_selection"]["best_epochs"],
                FINAL_REFIT_BEST_EPOCHS,
            )
            verification = provenance["upstream_evidence"][
                "boundary_verification_v2"
            ]
            self.assertEqual(
                verification["scientific_payload_sha256"],
                FINAL_REFIT_BOUNDARY_VERIFICATION_SCIENTIFIC_SHA256,
            )
            self.assertEqual(
                set(verification["modes"]),
                {
                    "concat_antibody_and_antigen",
                    "concat_antibody_antigen_plus_vh_vl",
                },
            )
            self.assertTrue(provenance["validated_before_dataset_or_embedding_load"])
            self.assertEqual(
                {
                    name: reference["sha256"]
                    for name, reference in provenance["source_files"].items()
                },
                _source_hashes(),
            )

    def test_manifest_semantic_and_physical_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, args, payload = _submission_contract_fixture(root)
            payload["fixed_refit"]["seed"] = 43
            manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            args.submission_manifest_sha256 = _file_digest(manifest)
            with self.assertRaisesRegex(ValueError, "optimization differs"):
                validate_final_refit_submission_manifest(args)

            manifest, args, payload = _submission_contract_fixture(root / "second")
            entrypoint_snapshot = Path(payload["source_files"]["entrypoint"]["path"])
            with entrypoint_snapshot.open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "physical SHA-256 mismatch"):
                validate_final_refit_submission_manifest(args)

            manifest, args, payload = _submission_contract_fixture(root / "overlap")
            payload["boundaries"]["repaired_concat_boundary"][
                "evaluation_output"
            ] = str(Path(payload["refit_output_dir"]) / "nested-evaluation.json")
            manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args.submission_manifest_sha256 = _file_digest(manifest)
            with self.assertRaisesRegex(ValueError, "outside protected"):
                validate_final_refit_submission_manifest(args)

    def test_independent_verification_rejects_an_unreviewed_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, args, payload = _submission_contract_fixture(root)
            boundary_contract = payload["boundaries"]["repaired_concat_boundary"]
            reviewed_boundary = Path(boundary_contract["split_manifest"]["path"])
            changed_boundary = root / "unreviewed-concat-boundary.json"
            changed_payload = json.loads(reviewed_boundary.read_text(encoding="utf-8"))
            changed_payload["unreviewed_change"] = True
            changed_payload["scientific_payload_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in changed_payload.items()
                    if key != "scientific_payload_sha256"
                }
            )
            changed_boundary.write_text(
                json.dumps(changed_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            changed_sha256 = _file_digest(changed_boundary)
            boundary_contract["split_manifest"] = {
                "path": str(changed_boundary),
                "sha256": changed_sha256,
            }
            manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args.submission_manifest_sha256 = _file_digest(manifest)
            args.split_manifest = changed_boundary
            args.split_manifest_sha256 = changed_sha256
            with self.assertRaisesRegex(
                ValueError, "does not bind the submitted .* boundary"
            ):
                validate_final_refit_submission_manifest(args)

    def test_warmup_and_one_shot_metric_protocol_are_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, args, _ = _submission_contract_fixture(root / "warmup")
            args.warmup_fraction = 0.20
            with self.assertRaisesRegex(ValueError, "optimization arguments differ"):
                validate_final_refit_submission_manifest(args)

            manifest, args, payload = _submission_contract_fixture(root / "metrics")
            payload["fixed_evaluation"]["recall_ks"] = [1, 5]
            manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            args.submission_manifest_sha256 = _file_digest(manifest)
            with self.assertRaisesRegex(ValueError, "evaluation protocol differs"):
                validate_final_refit_submission_manifest(args)

    def test_cli_rejects_semantic_tamper_before_any_dataset_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, args, payload = _submission_contract_fixture(root)
            payload["fixed_refit"]["seed"] = 43
            manifest.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            args.submission_manifest_sha256 = _file_digest(manifest)
            script = Path(__file__).with_name("sabdab2_asymmetric_partner_retrieval.py")
            command = [
                sys.executable,
                str(script),
                "final-refit",
                "--submission-manifest",
                str(manifest),
                "--submission-manifest-sha256",
                args.submission_manifest_sha256,
                "--prepare-dir",
                str(root / "missing-prepared"),
                "--antibody-embedding-dir",
                str(root / "missing-antibody-embeddings"),
                "--antigen-embedding-dir",
                str(root / "missing-antigen-embeddings"),
                "--split-manifest",
                str(args.split_manifest),
                "--split-manifest-sha256",
                args.split_manifest_sha256,
                "--authority-split-manifest",
                str(args.authority_split_manifest),
                "--authority-split-sha256",
                args.authority_split_sha256,
                "--authorized-source-pool-splits",
                *args.authorized_source_pool_splits,
                "--authority-sealed-splits",
                *args.authority_sealed_splits,
                "--authority-coverage-mode",
                args.authority_coverage_mode,
                "--strict-nw-constraint-mode",
                args.strict_nw_constraint_mode,
                "--isolation-policy",
                args.isolation_policy,
                "--output-dir",
                str(args.output_dir),
                "--antibody-artifact-format",
                args.antibody_artifact_format,
                "--antibody-model-name",
                args.antibody_model_name,
                "--antibody-model-weights-sha256",
                args.antibody_model_weights_sha256,
                "--antigen-model-name",
                args.antigen_model_name,
                "--antigen-model-weights-sha256",
                args.antigen_model_weights_sha256,
                "--batch-size",
                str(args.batch_size),
                "--fixed-epochs",
                str(args.fixed_epochs),
                "--device",
                "cpu",
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("optimization differs", completed.stderr)
            self.assertNotIn("Prepared instances are missing", completed.stderr)

    def test_manifest_validation_precedes_final_refit_loader(self) -> None:
        sentinel = ValueError("submission contract rejected")
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "sabdab2_asymmetric_partner_retrieval.validate_final_refit_submission_manifest",
            side_effect=sentinel,
        ), mock.patch(
            "sabdab2_asymmetric_partner_retrieval._final_refit_dataset_from_args"
        ) as dataset_loader:
            with self.assertRaisesRegex(ValueError, "submission contract rejected"):
                command_final_refit(Namespace(output_dir=Path(directory) / "refit"))
            dataset_loader.assert_not_called()

    def test_final_refit_lock_rejects_concurrency_before_manifest_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "canonical-refit"
            with _exclusive_final_refit_process_lock(output_dir), mock.patch(
                "sabdab2_asymmetric_partner_retrieval.validate_final_refit_submission_manifest"
            ) as manifest_validator, mock.patch(
                "sabdab2_asymmetric_partner_retrieval._final_refit_dataset_from_args"
            ) as dataset_loader:
                with self.assertRaisesRegex(RuntimeError, "already holds"):
                    command_final_refit(Namespace(output_dir=output_dir))
                manifest_validator.assert_not_called()
                dataset_loader.assert_not_called()

    def test_canonical_ledger_identity_ignores_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory).resolve()
            contract = _canonical_one_shot_ledger_contract(
                registry,
                _digest("checkpoint"),
                _digest("split"),
                _digest("authority"),
            )
            _create_one_shot_ledger(
                Path(contract["path"]),
                {"canonical_key": contract["key"], "status": "started"},
            )
            same_identity = _canonical_one_shot_ledger_contract(
                registry,
                _digest("checkpoint"),
                _digest("split"),
                _digest("authority"),
            )
            self.assertEqual(contract, same_identity)
            with self.assertRaises(FileExistsError):
                _create_one_shot_ledger(
                    Path(same_identity["path"]),
                    {
                        "canonical_key": same_identity["key"],
                        "evaluation_json": str(registry / "different-output.json"),
                        "score_artifact": str(registry / "different-scores.npz"),
                    },
                )

    def test_checkpoint_suite_claim_is_immutable_for_one_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = (root / "registry").resolve()
            refit_dir = (root / "refit").resolve()
            refit_dir.mkdir()
            checkpoint = refit_dir / "final.pt"
            checkpoint.write_bytes(b"checkpoint-one")
            files = {}
            for name in (
                "submission_manifest",
                "protocol_v1_5_execution_lock",
                "protocol_v1_6_execution_hardening",
                "boundary_verification_v2",
                "concat_split",
                "chain_split",
            ):
                path = root / f"{name}.json"
                path.write_text(name, encoding="utf-8")
                files[name] = {
                    "path": str(path.resolve()),
                    "sha256": _file_digest(path),
                }
            provenance = {
                "format_version": FINAL_REFIT_SUBMISSION_FORMAT_VERSION,
                "submission_manifest": files["submission_manifest"],
                "refit_output_dir": str(refit_dir),
                "one_shot_ledger_registry_dir": str(registry),
                "selected_boundary": "concat",
                "fixed_evaluation": _expected_final_evaluation_contract(),
                "upstream_evidence": {
                    name: files[name]
                    for name in (
                        "protocol_v1_5_execution_lock",
                        "protocol_v1_6_execution_hardening",
                        "boundary_verification_v2",
                    )
                },
                "all_declared_boundaries": {
                    "concat": {
                        "split_manifest": files["concat_split"],
                        "strict_nw_constraint_mode": "concat_antibody_and_antigen",
                    },
                    "chain": {
                        "split_manifest": files["chain_split"],
                        "strict_nw_constraint_mode": (
                            "concat_antibody_antigen_plus_vh_vl"
                        ),
                    },
                },
            }
            config = {"submission_provenance": provenance}
            claim, evidence, context = _ensure_checkpoint_suite_claim(
                config,
                checkpoint,
                _file_digest(checkpoint),
                registry,
                "concat",
            )
            self.assertEqual(set(claim), {"path", "sha256"})
            self.assertEqual(evidence["selected_boundary"], "concat")
            self.assertEqual(
                context["checkpoint_suite_claim_sha256"], claim["sha256"]
            )
            checkpoint.write_bytes(b"checkpoint-two")
            with self.assertRaisesRegex(FileExistsError, "different immutable payload"):
                _ensure_checkpoint_suite_claim(
                    config,
                    checkpoint,
                    _file_digest(checkpoint),
                    registry,
                    "concat",
                )

    def test_create_only_publishers_preserve_late_race_targets(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_link = __import__("os").link

            def inject_late_target(source: str, target: str) -> None:
                Path(target).write_bytes(b"late-race-winner")
                real_link(source, target)

            json_path = root / "evaluation.json"
            with mock.patch(
                "sabdab2_asymmetric_partner_retrieval.os.link",
                side_effect=inject_late_target,
            ), self.assertRaises(FileExistsError):
                _create_only_json(json_path, {"status": "ours"})
            self.assertEqual(json_path.read_bytes(), b"late-race-winner")

            score_path = root / "scores.npz"
            with mock.patch(
                "sabdab2_asymmetric_partner_retrieval.os.link",
                side_effect=inject_late_target,
            ), self.assertRaises(FileExistsError):
                _atomic_write_deterministic_npz(
                    score_path, {"values": np.asarray([1.0], dtype=np.float32)}
                )
            self.assertEqual(score_path.read_bytes(), b"late-race-winner")

    def test_evaluation_registry_and_boundary_are_checkpoint_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            registry = root / "registry"
            split = root / "boundary.json"
            split.write_text("boundary", encoding="utf-8")
            authority = (
                Path(__file__).resolve().parents[2]
                / "outputs"
                / "data_snapshot"
                / "double_remote_split.json"
            )
            output = root / "evaluation.json"
            scores = root / "scores.npz"
            prepare = root / "prepared"
            prepare.mkdir()
            instances = prepare / "instances.parquet"
            instances.write_bytes(b"instances")
            canonical = root / "canonical.csv"
            canonical.write_bytes(b"canonical")
            train_sha256 = _digest("common-train")
            boundary = {
                "split_manifest": {
                    "path": str(split),
                    "sha256": _file_digest(split),
                },
                "strict_nw_constraint_mode": "concat_antibody_and_antigen",
                "train_membership_sha256": train_sha256,
                "evaluation_output": str(output),
                "score_artifact": str(scores),
            }
            config = {
                "submission_provenance": {
                    "format_version": FINAL_REFIT_SUBMISSION_FORMAT_VERSION,
                    "common_train_membership_sha256": train_sha256,
                    "one_shot_ledger_registry_dir": str(registry),
                    "immutable_data_inputs": {
                        "prepared_instances": {
                            "path": str(instances),
                            "sha256": _file_digest(instances),
                        },
                        "canonical_sequence_manifest": {
                            "path": str(canonical),
                            "sha256": _file_digest(canonical),
                        },
                    },
                    "fixed_evaluation": _expected_final_evaluation_contract(),
                    "authority": {
                        "split_manifest": {
                            "path": str(authority),
                            "sha256": FINAL_REFIT_AUTHORITY_SHA256,
                        },
                        "authorized_source_pool_splits": ["train", "validation"],
                        "sealed_source_splits": ["test"],
                        "coverage_mode": "exact",
                    },
                    "all_declared_boundaries": {
                        "concat": boundary,
                        "chain": {
                            **boundary,
                            "split_manifest": {
                                "path": str(root / "chain.json"),
                                "sha256": _digest("chain"),
                            },
                            "strict_nw_constraint_mode": (
                                "concat_antibody_antigen_plus_vh_vl"
                            ),
                            "evaluation_output": str(root / "chain-evaluation.json"),
                            "score_artifact": str(root / "chain-scores.npz"),
                        },
                    },
                }
            }
            args = Namespace(
                score_artifact=scores,
                prepare_dir=prepare,
                prepared_instances_sha256=_file_digest(instances),
                canonical_sequence_manifest=canonical,
                canonical_sequence_manifest_sha256=_file_digest(canonical),
                one_shot_ledger_registry_dir=registry,
                batch_size=256,
                bucket_width=128,
                recall_ks=[1, 5, 10, 20],
                seed=42,
                allow_nondeterministic=False,
                device="cuda",
                isolation_policy="double_exact",
                split_manifest=split,
                split_manifest_sha256=_digest("boundary"),
                strict_nw_constraint_mode="concat_antibody_and_antigen",
                output=output,
                authority_split_manifest=authority,
                authority_split_sha256=FINAL_REFIT_AUTHORITY_SHA256,
                authorized_source_pool_splits=["train", "validation"],
                authority_sealed_splits=["test"],
                authority_coverage_mode="exact",
            )
            name, _ = _final_refit_evaluation_contract(config, args)
            self.assertEqual(name, "concat")
            split.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "physical SHA-256 mismatch"):
                _final_refit_evaluation_contract(config, args)
            split.write_text("boundary", encoding="utf-8")
            args.recall_ks = [1, 5]
            with self.assertRaisesRegex(ValueError, "runtime arguments differ"):
                _final_refit_evaluation_contract(config, args)
            args.recall_ks = [1, 5, 10, 20]
            args.one_shot_ledger_registry_dir = root / "alternate-registry"
            with self.assertRaisesRegex(ValueError, "registry differs"):
                _final_refit_evaluation_contract(config, args)
            args.one_shot_ledger_registry_dir = registry
            args.split_manifest = root / "unlisted.json"
            args.split_manifest_sha256 = _digest("unlisted")
            with self.assertRaisesRegex(ValueError, "predeclared checkpoint boundary"):
                _final_refit_evaluation_contract(config, args)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class ModelTests(unittest.TestCase):
    def test_runtime_torch_version_is_serialized_as_plain_string(self) -> None:
        import torch

        config = {
            "config_sha256": _digest("config"),
            "source_sha256": {"entrypoint": _digest("source")},
            "scientific_scope": {"frozen_test_used": False},
        }
        args = Namespace(
            antibody_embedding_dir=Path("/synthetic/antibody"),
            antigen_embedding_dir=Path("/synthetic/antigen"),
        )
        runtime = _provenance(config, args, torch.device("cpu"))["runtime"]
        self.assertIs(type(runtime["torch"]), str)
        self.assertEqual(runtime["torch"], str(torch.__version__))

    def test_final_refit_resume_state_rejects_semantic_corruption(self) -> None:
        import torch

        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=4
        )
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        scheduler.step()
        config = {"contract": "synthetic"}
        config["config_sha256"] = canonical_sha256(config)
        submission = {"submission": "synthetic"}

        def provenance(timestamp: str, job_id: str | None) -> dict[str, object]:
            return {
                "format_version": FORMAT_VERSION,
                "created_at_utc": timestamp,
                "config_sha256": config["config_sha256"],
                "source_sha256": {"entrypoint": _digest("source")},
                "runtime": {
                    "python": sys.version,
                    "platform": "synthetic-platform",
                    "torch": str(torch.__version__),
                    "device": "cpu",
                    "cuda_available": False,
                    "cuda_version": None,
                    "gpu_name": None,
                    "slurm_job_id": job_id,
                },
                "embedding_locations": {
                    "antibody": "/synthetic/antibody",
                    "antigen": "/synthetic/antigen",
                },
                "scientific_scope": {"frozen_test_used": False},
            }

        previous = provenance("2026-01-01T00:00:00+00:00", "1")
        current = provenance("2026-01-02T00:00:00+00:00", "2")
        state = {
            "format_version": FORMAT_VERSION,
            "stage": "final_refit_in_progress",
            "config": config,
            "submission_provenance": submission,
            "execution_provenance_history": [previous],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": 1,
            "global_step": 1,
            "history": [
                {
                    "epoch": 1,
                    "global_step": 1,
                    "train": {"loss": 1.0},
                    "learning_rate": 0.001,
                    "temperature": 0.2,
                    "validation_metrics_computed": False,
                }
            ],
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": [],
        }
        history = _validate_final_refit_resume_state(
            state,
            config,
            submission,
            2,
            current,
            model,
            optimizer,
            scheduler,
        )
        self.assertEqual(len(history), 2)
        advanced = copy.deepcopy(state)
        advanced["epoch"] = 3
        with self.assertRaisesRegex(ValueError, "outside the locked range"):
            _validate_final_refit_resume_state(
                advanced,
                config,
                submission,
                2,
                current,
                model,
                optimizer,
                scheduler,
            )
        nonfinite = copy.deepcopy(state)
        first_key = next(iter(nonfinite["model_state_dict"]))
        nonfinite["model_state_dict"][first_key].reshape(-1)[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite tensor"):
            _validate_final_refit_resume_state(
                nonfinite,
                config,
                submission,
                2,
                current,
                model,
                optimizer,
                scheduler,
            )
        drifted = copy.deepcopy(state)
        drifted["execution_provenance_history"][0]["runtime"]["torch"] = "drifted"
        with self.assertRaisesRegex(ValueError, "runtime provenance changed"):
            _validate_final_refit_resume_state(
                drifted,
                config,
                submission,
                2,
                current,
                model,
                optimizer,
                scheduler,
            )

    def test_checkpoint_loader_hashes_the_deserialized_bytes_once(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"tensor": torch.arange(3)}, path)
            digest = _file_digest(path)
            checkpoint, observed = _load_torch_checkpoint_once(path, digest, "cpu")
            self.assertEqual(observed, digest)
            torch.testing.assert_close(checkpoint["tensor"], torch.arange(3))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                _load_torch_checkpoint_once(path, _digest("wrong"), "cpu")

    def test_final_checkpoint_publication_preserves_a_late_target(self) -> None:
        import os
        import torch

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "final.pt"
            real_link = os.link

            def inject_late_target(source: str, destination: str) -> None:
                Path(destination).write_bytes(b"late-final-checkpoint")
                real_link(source, destination)

            with mock.patch(
                "sabdab2_asymmetric_partner_retrieval.os.link",
                side_effect=inject_late_target,
            ), self.assertRaises(FileExistsError):
                _create_only_torch_save({"tensor": torch.ones(1)}, target)
            self.assertEqual(target.read_bytes(), b"late-final-checkpoint")

    def test_final_refit_recovers_only_summary_and_authenticates_noop(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "refit"
            output_dir.mkdir()
            submission = {
                "format_version": FINAL_REFIT_SUBMISSION_FORMAT_VERSION,
                "submission_manifest": {
                    "path": str((root / "submission.json").resolve()),
                    "sha256": _digest("submission"),
                },
                "canonical_refit_boundary": "concat",
                "fixed_refit": {"fixed_epochs": 1},
            }
            model_contract = _expected_final_refit_model_contract()
            config = {
                "format_version": FORMAT_VERSION,
                "run_mode": "final_refit_fixed_epochs",
                "model": {
                    "antibody_input_dim": model_contract["antibody_input_dim"],
                    "antigen_input_dim": model_contract["antigen_input_dim"],
                    "embedding_dim": model_contract["embedding_dim"],
                    "antibody_hidden_dim": model_contract["antibody_hidden_dim"],
                    "antigen_hidden_dim": model_contract["antigen_hidden_dim"],
                    "dropout": model_contract["dropout"],
                    "chain_balanced_antibody_pooling": model_contract[
                        "chain_balanced_antibody_pooling"
                    ],
                    "antibody_pooling": model_contract["antibody_pooling"],
                    "antigen_pooling": model_contract["antigen_pooling"],
                    "initial_temperature": model_contract["initial_temperature"],
                    "minimum_temperature": model_contract["minimum_temperature"],
                    "maximum_temperature": model_contract["maximum_temperature"],
                },
                "optimization": {
                    "learning_rate": 0.001,
                    "weight_decay": 0.2,
                    "train_batch_count": 1,
                    "fixed_epochs": 1,
                },
                "source_sha256": _source_hashes(),
                "submission_provenance": submission,
                "scientific_scope": {"frozen_test_used": False},
            }
            config["config_sha256"] = canonical_sha256(config)
            args = Namespace(
                schedule="cosine_warm_restarts",
                restart_period_epochs=10,
                restart_multiplier=2,
                minimum_learning_rate=0.000001,
                learning_rate=0.001,
                warmup_fraction=0.1,
                max_epochs=1,
                fixed_epochs=1,
                output_dir=output_dir,
                device="cpu",
                seed=42,
                allow_nondeterministic=False,
                antibody_embedding_dir=root / "missing-antibody-embeddings",
                antigen_embedding_dir=root / "missing-antigen-embeddings",
            )
            device = torch.device("cpu")
            model = _model_from_config(config, device)
            optimizer = _optimizer(model, 0.001, 0.2)
            scheduler = _scheduler(optimizer, 1, args)
            for parameter in model.parameters():
                parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            scheduler.step()
            history = [
                {
                    "epoch": 1,
                    "global_step": 1,
                    "train": {"loss": 1.0},
                    "learning_rate": 0.001,
                    "temperature": 0.2,
                    "validation_metrics_computed": False,
                }
            ]
            execution = _provenance(config, args, device)
            state = {
                "format_version": FORMAT_VERSION,
                "stage": "final_refit_in_progress",
                "config": config,
                "submission_provenance": submission,
                "execution_provenance_history": [execution],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": 1,
                "global_step": 1,
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": [],
            }
            final_checkpoint = {
                "format_version": FORMAT_VERSION,
                "stage": "final_refit_complete",
                "config": config,
                "submission_provenance": submission,
                "execution_provenance_history": [execution],
                "model_state_dict": model.state_dict(),
                "epoch": 1,
                "global_step": 1,
                "history": history,
                "validation_metrics_computed": False,
                "frozen_test_used": False,
            }
            torch.save(state, output_dir / "refit_state.pt")
            torch.save(final_checkpoint, output_dir / "final.pt")
            with mock.patch(
                "sabdab2_asymmetric_partner_retrieval._validate_numeric_arguments"
            ), mock.patch(
                "sabdab2_asymmetric_partner_retrieval._final_refit_dataset_from_args"
            ) as dataset_loader, mock.patch(
                "sabdab2_asymmetric_partner_retrieval._stores"
            ) as embedding_loader, mock.patch(
                "sabdab2_asymmetric_partner_retrieval._embedding_corpora"
            ) as embedding_fingerprint, mock.patch(
                "sabdab2_asymmetric_partner_retrieval.load_excluded_negative_pairs"
            ) as negative_loader:
                _command_final_refit_locked(args, submission)
                dataset_loader.assert_not_called()
                embedding_loader.assert_not_called()
                embedding_fingerprint.assert_not_called()
                negative_loader.assert_not_called()
            summary_path = output_dir / "final_summary.json"
            original_summary = summary_path.read_bytes()
            with mock.patch(
                "sabdab2_asymmetric_partner_retrieval._create_only_json"
            ) as publisher, mock.patch(
                "sabdab2_asymmetric_partner_retrieval._validate_numeric_arguments"
            ):
                _command_final_refit_locked(args, submission)
                publisher.assert_not_called()
            self.assertEqual(summary_path.read_bytes(), original_summary)
            summary_path.unlink()
            state["scheduler_state_dict"]["last_epoch"] = 2
            torch.save(state, output_dir / "refit_state.pt")
            with self.assertRaisesRegex(ValueError, "scheduler step differs"):
                _recover_final_refit_completion_if_present(
                    args, submission, output_dir, device
                )

    def test_mean_pooling_projects_residues_before_averaging(self) -> None:
        import torch

        from interaction_first.asymmetric_retrieval import CALMMeanPooler

        torch.manual_seed(3)
        pooler = CALMMeanPooler(4, 3, 5, dropout=0.0, pooling="mean")
        pooler.eval()
        features = torch.randn(2, 4, 4)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        observed, weights = pooler(features, mask)
        projected = pooler.residue_projection(
            torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))
        )
        manual = (projected * mask.unsqueeze(-1)).sum(dim=1)
        manual = manual / mask.sum(dim=1, keepdim=True)
        manual = manual / torch.linalg.vector_norm(manual, dim=1, keepdim=True)
        torch.testing.assert_close(observed, manual)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))

    def test_chain_balancing_assigns_equal_mass_to_present_chains(self) -> None:
        import torch

        from interaction_first.asymmetric_retrieval import CALMMeanPooler

        pooler = CALMMeanPooler(4, 3, 5, dropout=0.0, chain_balanced=True)
        features = torch.randn(2, 5, 4)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
        chain_ids = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, -1, -1]])
        _, weights = pooler(features, mask, chain_ids)
        self.assertAlmostEqual(float(weights[0, :3].sum()), 0.5)
        self.assertAlmostEqual(float(weights[0, 3:].sum()), 0.5)
        self.assertAlmostEqual(float(weights[1, :3].sum()), 1.0)

    def test_masked_info_nce_excludes_uncertain_negative(self) -> None:
        import torch

        from interaction_first.asymmetric_retrieval import (
            symmetric_masked_multi_positive_info_nce,
        )

        similarities = torch.tensor([[1.0, 0.9], [0.2, 1.0]], requires_grad=True)
        positives = torch.eye(2, dtype=torch.bool)
        full = symmetric_masked_multi_positive_info_nce(similarities, positives, 0.2)
        denominator = torch.ones_like(positives)
        denominator[0, 1] = False
        masked = symmetric_masked_multi_positive_info_nce(
            similarities, positives, 0.2, denominator
        )
        self.assertLess(float(masked), float(full))
        masked.backward()
        self.assertTrue(bool(torch.isfinite(similarities.grad).all()))
        invalid = denominator.clone()
        invalid[0, 0] = False
        with self.assertRaisesRegex(ValueError, "Known positives"):
            symmetric_masked_multi_positive_info_nce(
                similarities.detach(), positives, 0.2, invalid
            )


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class EmbeddingContractTests(unittest.TestCase):
    def _write_antibody(self, root: Path, key: str, dim: int = 5) -> Path:
        import torch

        path = root / key[:2] / f"{key}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": ANTIBERTY_FORMAT,
                "sequence_key": key,
                "source_sequence_sha256": key,
                "embedding": torch.randn(5, dim, dtype=torch.float32).contiguous(),
                "mask": torch.ones(5, dtype=torch.bool),
                "vh_length": 3,
                "vl_length": 2,
                "embedding_dim": dim,
                "model_name": "synthetic-ab",
                "model_weights_sha256": _digest("weights"),
            },
            path,
        )
        return path

    def _write_antigen(self, root: Path, key: str, dim: int = 7) -> Path:
        import torch

        path = root / key[:2] / f"{key}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        chunks = [torch.randn(4, dim), torch.randn(3, dim)]
        torch.save(
            {
                "format_version": ESM_FORMAT,
                "sequence_sha256": key,
                "sequence_length": 7,
                "model_name": "synthetic-ag",
                "model_weight_sha256": _digest("ag-weights"),
                "starts": [0, 4],
                "ends": [4, 7],
                "chunks": chunks,
            },
            path,
        )
        return path

    def test_separate_stores_validate_dimensions_models_and_chains(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            antibody_root = root / "antibody"
            antigen_root = root / "antigen"
            antibody_key = _digest("ab")
            antigen_key = _digest("ag")
            self._write_antibody(antibody_root, antibody_key)
            self._write_antigen(antigen_root, antigen_key)
            antibody_store = FrozenPartnerEmbeddingStore(
                antibody_root,
                "antibody",
                "antiberty",
                5,
                "synthetic-ab",
                _digest("weights"),
                0.0,
            )
            antigen_store = FrozenPartnerEmbeddingStore(
                antigen_root,
                "antigen",
                "esm_chunks",
                7,
                "synthetic-ag",
                _digest("ag-weights"),
                0.0,
            )
            antibody, chain_ids = antibody_store.load(antibody_key, 5, "cpu")
            antigen, no_chains = antigen_store.load(antigen_key, 7, "cpu")
            self.assertEqual(tuple(antibody.shape), (5, 5))
            self.assertEqual(tuple(antigen.shape), (7, 7))
            torch.testing.assert_close(chain_ids, torch.tensor([0, 0, 0, 1, 1]))
            self.assertIsNone(no_chains)
            esm_antibody_store = FrozenPartnerEmbeddingStore(
                antigen_root,
                "antibody",
                "esm_chunks",
                7,
                "synthetic-ag",
                _digest("ag-weights"),
                0.0,
            )
            esm_antibody, esm_chain_ids = esm_antibody_store.load(
                antigen_key, 7, "cpu"
            )
            self.assertEqual(tuple(esm_antibody.shape), (7, 7))
            self.assertIsNone(esm_chain_ids)
            wrong_model = FrozenPartnerEmbeddingStore(
                antibody_root,
                "antibody",
                "antiberty",
                5,
                "wrong",
                _digest("weights"),
                0.0,
            )
            with self.assertRaisesRegex(ValueError, "unexpected model"):
                wrong_model.load(antibody_key, 5, "cpu")
            wrong_weights = FrozenPartnerEmbeddingStore(
                antibody_root,
                "antibody",
                "antiberty",
                5,
                "synthetic-ab",
                _digest("different-weights"),
                0.0,
            )
            with self.assertRaisesRegex(ValueError, "weight digest"):
                wrong_weights.load(antibody_key, 5, "cpu")

    def test_side_fingerprint_detects_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = _digest("ab")
            path = self._write_antibody(root, key)
            original = side_embedding_fingerprint(root, [key], "antibody")
            with path.open("ab") as handle:
                handle.write(b"changed")
            changed = side_embedding_fingerprint(root, [key], "antibody")
            self.assertNotEqual(
                original["content_manifest_sha256"],
                changed["content_manifest_sha256"],
            )


@unittest.skipUnless(
    importlib.util.find_spec("pandas") and importlib.util.find_spec("pyarrow"),
    "pandas and PyArrow are unavailable",
)
class DevelopmentProtocolTests(unittest.TestCase):
    def test_one_shot_ledger_is_exclusive_and_durable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            _create_one_shot_ledger(path, {"status": "started"})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                _create_one_shot_ledger(path, {"status": "replacement"})
            self.assertEqual(path.read_bytes(), original)

    def test_final_refit_source_has_no_validation_scoring_path(self) -> None:
        source = inspect.getsource(_command_final_refit_locked)
        self.assertNotIn("evaluate_records(", source)
        self.assertNotIn("validation_records", source)
        self.assertIn("_final_refit_dataset_from_args", source)

    def test_antigen_exact_policy_reports_but_allows_antibody_overlap(self) -> None:
        shared_antibody = _digest("shared-ab")
        train = [PairRecord("train", shared_antibody, _digest("train-ag"), 5, 7)]
        validation = [
            PairRecord("validation", shared_antibody, _digest("validation-ag"), 5, 7)
        ]
        audit = audit_isolation_policy(train, validation, "antigen_exact")
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["antibody_sequence_overlap"], 1)
        with self.assertRaisesRegex(ValueError, "double_exact"):
            audit_isolation_policy(train, validation, "double_exact")

    def test_loader_selects_only_train_and_validation(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare = root / "prepared"
            prepare.mkdir()
            rows = []
            for name in ("train", "validation", "sealed"):
                rows.append(
                    {
                        "instance_id": name,
                        "ab_sequence_sha256": _digest(f"{name}-ab"),
                        "ag_sequence_sha256": _digest(f"{name}-ag"),
                        "ab_length": 5,
                        "ag_length": 7,
                    }
                )
            pd.DataFrame(rows).to_parquet(prepare / "instances.parquet", index=False)
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "format_version": "sabdab2_identity_isolated_development_v1",
                        "intended_use": "development_only",
                        "splits": {"train": ["train"], "validation": ["validation"]},
                    }
                ),
                encoding="utf-8",
            )
            authority = root / "authority.json"
            authority.write_text(
                json.dumps(
                    {
                        "format_version": "authoritative-source-v1",
                        "splits": {
                            "train": ["train"],
                            "validation": ["validation"],
                            "test": ["sealed"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            records, metadata = load_development_dataset(
                prepare,
                split,
                _file_digest(split),
                authority,
                _file_digest(authority),
                ["train", "validation"],
                ["test"],
                "exact",
            )
            self.assertEqual(records["train"][0].instance_id, "train")
            self.assertEqual(records["validation"][0].instance_id, "validation")
            self.assertTrue(metadata["frozen_test_records_selected"] is False)

            observed_filters = []
            original_read = pd.read_parquet

            def recording_read(*args, **kwargs):
                observed_filters.append(kwargs.get("filters"))
                return original_read(*args, **kwargs)

            with mock.patch("pandas.read_parquet", side_effect=recording_read):
                train_only_records, train_only_metadata = load_development_dataset(
                    prepare,
                    split,
                    _file_digest(split),
                    authority,
                    _file_digest(authority),
                    ["train", "validation"],
                    ["test"],
                    "exact",
                    materialized_splits=("train",),
                )
            self.assertEqual(set(train_only_records), {"train"})
            self.assertEqual(train_only_metadata["loaded_splits"], ["train"])
            self.assertEqual(train_only_metadata["unmaterialized_splits"], ["validation"])
            self.assertEqual(set(observed_filters[0][0][2]), {"train"})
            self.assertNotIn("validation", observed_filters[0][0][2])

            malicious = root / "malicious.json"
            malicious.write_text(
                json.dumps(
                    {
                        "format_version": "sabdab2_identity_isolated_development_v1",
                        "intended_use": "development_only",
                        "splits": {
                            "train": ["train", "sealed"],
                            "validation": ["validation"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PermissionError, "sealed source IDs"):
                load_development_dataset(
                    prepare,
                    malicious,
                    _file_digest(malicious),
                    authority,
                    _file_digest(authority),
                    ["train", "validation"],
                    ["test"],
                    "exact",
                )

            strict = root / "strict.json"
            strict_payload = {
                "format_version": "sabdab2_iterative_nw_strict_development_split_v1",
                "intended_use": "development_only",
                "valid": True,
                "constraint_mode": "concat_antibody_antigen_plus_vh_vl",
                "frozen_test_labels_or_predictions_read": False,
                "frozen_test_sequences_aligned": False,
                "sealed_identifiers_read_for_exclusion_only": True,
                "sealed_source_splits": ["test"],
                "splits": {"train": ["train"], "validation": ["validation"]},
                "final_graph_audit": {"valid": True},
                "evidence": {
                    "identity_threshold": 0.80,
                    "reciprocal_coverage_threshold": 0.80,
                    "exact_alignment_policy": {
                        "mode": "global",
                        "chain_constraints_included": True,
                    },
                },
                "provenance": {
                    "development_source_split": {"sha256": _file_digest(authority)}
                },
            }
            strict.write_text(json.dumps(strict_payload), encoding="utf-8")
            strict_records, _ = load_development_dataset(
                prepare,
                strict,
                _file_digest(strict),
                authority,
                _file_digest(authority),
                ["train", "validation"],
                ["test"],
                "exact",
                "concat_antibody_antigen_plus_vh_vl",
            )
            self.assertEqual(len(strict_records["validation"]), 1)
            with self.assertRaisesRegex(ValueError, "differs from the explicit"):
                load_development_dataset(
                    prepare,
                    strict,
                    _file_digest(strict),
                    authority,
                    _file_digest(authority),
                    ["train", "validation"],
                    ["test"],
                    "exact",
                    "concat_antibody_and_antigen",
                )
            with self.assertRaisesRegex(ValueError, "require an explicit"):
                load_development_dataset(
                    prepare,
                    strict,
                    _file_digest(strict),
                    authority,
                    _file_digest(authority),
                    ["train", "validation"],
                    ["test"],
                    "exact",
                )
            strict_payload["valid"] = False
            strict.write_text(json.dumps(strict_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid=true"):
                load_development_dataset(
                    prepare,
                    strict,
                    _file_digest(strict),
                    authority,
                    _file_digest(authority),
                    ["train", "validation"],
                    ["test"],
                    "exact",
                    "concat_antibody_antigen_plus_vh_vl",
                )

            strict_payload["valid"] = True
            strict_payload["constraint_mode"] = "concat_antibody_and_antigen"
            strict_payload["evidence"]["exact_alignment_policy"][
                "chain_constraints_included"
            ] = False
            strict.write_text(json.dumps(strict_payload), encoding="utf-8")
            concat_records, concat_metadata = load_development_dataset(
                prepare,
                strict,
                _file_digest(strict),
                authority,
                _file_digest(authority),
                ["train", "validation"],
                ["test"],
                "exact",
                "concat_antibody_and_antigen",
            )
            self.assertEqual(len(concat_records["validation"]), 1)
            self.assertEqual(
                concat_metadata["strict_nw_constraint_mode"],
                "concat_antibody_and_antigen",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                strict_payload["evidence"]["exact_alignment_policy"][
                    "chain_constraints_included"
                ] = True
                strict.write_text(json.dumps(strict_payload), encoding="utf-8")
                load_development_dataset(
                    prepare,
                    strict,
                    _file_digest(strict),
                    authority,
                    _file_digest(authority),
                    ["train", "validation"],
                    ["test"],
                    "exact",
                    "concat_antibody_and_antigen",
                )

    def test_uncertain_negative_artifact_cannot_mask_a_positive(self) -> None:
        antibody = _digest("ab")
        antigen = _digest("ag")
        records = [PairRecord("pair", antibody, antigen, 5, 7)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uncertain.json"
            path.write_text(
                json.dumps(
                    {
                        "format_version": "sabdab2_uncertain_negative_pairs_v1",
                        "intended_use": "development_training_only",
                        "pairs": [[antibody, antigen]],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Known training positives"):
                load_excluded_negative_pairs(path, records)

    def test_v2_boundary_accounts_for_but_never_materializes_excluded_validation(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare = root / "prepared"
            prepare.mkdir()
            identifiers = ["train-0", "train-1", "validation-0", "excluded-0", "sealed"]
            pd.DataFrame(
                [
                    {
                        "instance_id": name,
                        "ab_sequence_sha256": _digest(f"{name}-ab"),
                        "ag_sequence_sha256": _digest(f"{name}-ag"),
                        "ab_length": 5,
                        "ag_length": 7,
                    }
                    for name in identifiers
                ]
            ).to_parquet(prepare / "instances.parquet", index=False)
            canonical_manifest = root / "canonical-manifest.csv"
            canonical_manifest.write_text("synthetic\n", encoding="utf-8")
            pinned_data_arguments = {
                "expected_instances_sha256": _file_digest(
                    prepare / "instances.parquet"
                ),
                "canonical_sequence_manifest": canonical_manifest,
                "canonical_sequence_manifest_sha256": _file_digest(
                    canonical_manifest
                ),
            }
            authority = root / "authority.json"
            authority.write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": ["train-0", "train-1"],
                            "validation": ["validation-0", "excluded-0"],
                            "test": ["sealed"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            boundary = root / "boundary.json"
            boundary_payload = _synthetic_boundary_payload(
                _file_digest(authority),
                ["train-0", "train-1"],
                ["validation-0"],
                ["excluded-0"],
            )
            boundary.write_text(json.dumps(boundary_payload), encoding="utf-8")
            with self.assertRaisesRegex(PermissionError, "reserved"):
                load_development_dataset(
                    prepare,
                    boundary,
                    _file_digest(boundary),
                    authority,
                    _file_digest(authority),
                    ["train", "validation"],
                    ["test"],
                    "exact",
                    "concat_antibody_and_antigen",
                )
            observed_filters = []
            original_read = pd.read_parquet

            def recording_read(*args, **kwargs):
                observed_filters.append(kwargs.get("filters"))
                return original_read(*args, **kwargs)

            with mock.patch("pandas.read_parquet", side_effect=recording_read):
                records, metadata = load_development_dataset(
                    prepare,
                    boundary,
                    _file_digest(boundary),
                    authority,
                    _file_digest(authority),
                    ["train", "validation"],
                    ["test"],
                    "exact",
                    "concat_antibody_and_antigen",
                    materialized_splits=("train",),
                    require_authority_validation_boundary_v2=True,
                    **pinned_data_arguments,
                )
            self.assertEqual(set(records), {"train"})
            self.assertEqual(metadata["raw_instance_counts"]["excluded_validation"], 1)
            self.assertEqual(metadata["permanently_excluded_splits"], ["excluded_validation"])
            self.assertFalse(metadata["excluded_validation_rows_materialized"])
            self.assertEqual(set(observed_filters[0][0][2]), {"train-0", "train-1"})
            self.assertNotIn("excluded-0", observed_filters[0][0][2])
            with self.assertRaisesRegex(ValueError, "unique subset of train and validation"):
                load_development_dataset(
                    prepare,
                    boundary,
                    _file_digest(boundary),
                    authority,
                    _file_digest(authority),
                    ["train", "validation"],
                    ["test"],
                    "exact",
                    "concat_antibody_and_antigen",
                    materialized_splits=("excluded_validation",),
                    require_authority_validation_boundary_v2=True,
                    **pinned_data_arguments,
                )

            unsafe = dict(boundary_payload)
            unsafe["chronology"] = dict(boundary_payload["chronology"])
            unsafe["chronology"]["globally_untouched_or_independent_test"] = True
            unsafe["scientific_payload_sha256"] = canonical_sha256(
                {key: value for key, value in unsafe.items() if key != "scientific_payload_sha256"}
            )
            boundary.write_text(json.dumps(unsafe), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe chronology claim"):
                load_development_dataset(
                    prepare,
                    boundary,
                    _file_digest(boundary),
                    authority,
                    _file_digest(authority),
                    ["train", "validation"],
                    ["test"],
                    "exact",
                    "concat_antibody_and_antigen",
                    require_authority_validation_boundary_v2=True,
                    **pinned_data_arguments,
                )

            legacy = root / "legacy.json"
            legacy.write_text(
                json.dumps(
                    {
                        "format_version": "sabdab2_identity_isolated_development_v1",
                        "intended_use": "development_only",
                        "splits": {
                            "train": ["train-0", "train-1"],
                            "validation": ["validation-0", "excluded-0"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "boundary v2"):
                load_development_dataset(
                    prepare,
                    legacy,
                    _file_digest(legacy),
                    authority,
                    _file_digest(authority),
                    ["train", "validation"],
                    ["test"],
                    "exact",
                    require_authority_validation_boundary_v2=True,
                    **pinned_data_arguments,
                )


@unittest.skipUnless(
    importlib.util.find_spec("torch")
    and importlib.util.find_spec("pandas")
    and importlib.util.find_spec("pyarrow"),
    "PyTorch, pandas, and PyArrow are required",
)
class EndToEndTests(unittest.TestCase):
    def test_train_resume_evaluate_and_no_test_surface(self) -> None:
        import pandas as pd
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare = root / "prepared"
            antibody_root = root / "antibody"
            antigen_root = root / "antigen"
            output = root / "run"
            prepare.mkdir()
            rows = []
            splits = {"train": [], "validation": []}
            for split_name, count in (("train", 4), ("validation", 2)):
                for index in range(count):
                    instance = f"{split_name}-{index}"
                    antibody_key = _digest(f"{instance}-ab")
                    antigen_key = _digest(f"{instance}-ag")
                    rows.append(
                        {
                            "instance_id": instance,
                            "ab_sequence_sha256": antibody_key,
                            "ag_sequence_sha256": antigen_key,
                            "ab_length": 5,
                            "ag_length": 7,
                        }
                    )
                    splits[split_name].append(instance)
                    antibody_path = antibody_root / antibody_key[:2] / f"{antibody_key}.pt"
                    antibody_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "format_version": ANTIBERTY_FORMAT,
                            "sequence_key": antibody_key,
                            "source_sequence_sha256": antibody_key,
                            "embedding": torch.randn(5, 5).float().contiguous(),
                            "mask": torch.ones(5, dtype=torch.bool),
                            "vh_length": 3,
                            "vl_length": 2,
                            "embedding_dim": 5,
                            "model_name": "synthetic-ab",
                            "model_weights_sha256": _digest("weights"),
                        },
                        antibody_path,
                    )
                    antigen_path = antigen_root / antigen_key[:2] / f"{antigen_key}.pt"
                    antigen_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "format_version": ESM_FORMAT,
                            "sequence_sha256": antigen_key,
                            "sequence_length": 7,
                            "model_name": "synthetic-ag",
                            "model_weight_sha256": _digest("ag-weights"),
                            "starts": [0],
                            "ends": [7],
                            "chunks": [torch.randn(7, 7).float()],
                        },
                        antigen_path,
                    )
            rows.append(
                {
                    "instance_id": "sealed-test",
                    "ab_sequence_sha256": "not-selected-and-intentionally-invalid",
                    "ag_sequence_sha256": "not-selected-and-intentionally-invalid",
                    "ab_length": -1,
                    "ag_length": -1,
                }
            )
            pd.DataFrame(rows).to_parquet(prepare / "instances.parquet", index=False)
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "format_version": "sabdab2_identity_isolated_development_v1",
                        "intended_use": "development_only",
                        "splits": splits,
                    }
                ),
                encoding="utf-8",
            )
            authority_path = root / "authority.json"
            authority_path.write_text(
                json.dumps(
                    {
                        "format_version": "authoritative-source-v1",
                        "splits": {
                            "train": splits["train"],
                            "validation": splits["validation"],
                            "test": ["sealed-test"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            script = Path(__file__).with_name("sabdab2_asymmetric_partner_retrieval.py")
            train_command = [
                sys.executable,
                str(script),
                "train",
                "--prepare-dir",
                str(prepare),
                "--antibody-embedding-dir",
                str(antibody_root),
                "--antigen-embedding-dir",
                str(antigen_root),
                "--split-manifest",
                str(split_path),
                "--split-manifest-sha256",
                _file_digest(split_path),
                "--authority-split-manifest",
                str(authority_path),
                "--authority-split-sha256",
                _file_digest(authority_path),
                "--authorized-source-pool-splits",
                "train",
                "validation",
                "--authority-sealed-splits",
                "test",
                "--output-dir",
                str(output),
                "--antibody-model-name",
                "synthetic-ab",
                "--antibody-model-weights-sha256",
                _digest("weights"),
                "--antigen-model-name",
                "synthetic-ag",
                "--antigen-model-weights-sha256",
                _digest("ag-weights"),
                "--antibody-input-dim",
                "5",
                "--antigen-input-dim",
                "7",
                "--embedding-dim",
                "3",
                "--antibody-hidden-dim",
                "4",
                "--antigen-hidden-dim",
                "4",
                "--batch-size",
                "2",
                "--validation-batch-size",
                "2",
                "--max-epochs",
                "2",
                "--patience",
                "2",
                "--min-train-pairs",
                "2",
                "--min-validation-pairs",
                "2",
                "--antibody-cache-gib",
                "0",
                "--antigen-cache-gib",
                "0",
                "--bucket-width",
                "16",
                "--device",
                "cpu",
            ]
            subprocess.run(train_command, check=True, capture_output=True, text=True)
            subprocess.run(train_command, check=True, capture_output=True, text=True)
            evaluation_path = root / "evaluation.json"
            score_path = root / "scores.npz"
            evaluate_command = [
                sys.executable,
                str(script),
                "evaluate",
                "--prepare-dir",
                str(prepare),
                "--antibody-embedding-dir",
                str(antibody_root),
                "--antigen-embedding-dir",
                str(antigen_root),
                "--split-manifest",
                str(split_path),
                "--split-manifest-sha256",
                _file_digest(split_path),
                "--authority-split-manifest",
                str(authority_path),
                "--authority-split-sha256",
                _file_digest(authority_path),
                "--authorized-source-pool-splits",
                "train",
                "validation",
                "--authority-sealed-splits",
                "test",
                "--checkpoint",
                str(output / "best.pt"),
                "--checkpoint-sha256",
                _file_digest(output / "best.pt"),
                "--output",
                str(evaluation_path),
                "--score-artifact",
                str(score_path),
                "--antibody-model-name",
                "synthetic-ab",
                "--antibody-model-weights-sha256",
                _digest("weights"),
                "--antigen-model-name",
                "synthetic-ag",
                "--antigen-model-weights-sha256",
                _digest("ag-weights"),
                "--antibody-input-dim",
                "5",
                "--antigen-input-dim",
                "7",
                "--batch-size",
                "2",
                "--antibody-cache-gib",
                "0",
                "--antigen-cache-gib",
                "0",
                "--bucket-width",
                "16",
                "--device",
                "cpu",
            ]
            subprocess.run(evaluate_command, check=True, capture_output=True, text=True)
            payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["frozen_test_used"])
            self.assertIn("per_anchor_ranks", payload["metrics"])
            self.assertTrue(score_path.is_file())
            self.assertEqual(payload["score_artifact"]["sha256"], _file_digest(score_path))

            # Convert the synthetic checkpoint only to exercise the one-shot registry.
            # Final-refit training itself requires the locked physical submission contract.
            final_evaluation = root / "final-evaluation.json"
            final_scores = root / "final-scores.npz"
            final_registry = (root / "central-one-shot-registry").resolve()
            confirmation_split = root / "confirmation-boundary.json"
            confirmation_payload = _synthetic_boundary_payload(
                _file_digest(authority_path),
                list(splits["train"]),
                [splits["validation"][0]],
                [splits["validation"][1]],
            )
            confirmation_split.write_text(
                json.dumps(confirmation_payload), encoding="utf-8"
            )
            chain_split = root / "chain-confirmation-boundary.json"
            chain_payload = _synthetic_boundary_payload(
                _file_digest(authority_path),
                list(splits["train"]),
                [splits["validation"][0]],
                [splits["validation"][1]],
                "concat_antibody_antigen_plus_vh_vl",
            )
            chain_split.write_text(json.dumps(chain_payload), encoding="utf-8")
            common_train_sha256 = _membership_digest(list(splits["train"]))
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            config["run_mode"] = "final_refit_fixed_epochs"
            config["training_data"][
                "isolation_policy_to_audit_at_evaluation"
            ] = config["training_data"]["isolation_policy"]
            train_rows = {
                row["instance_id"]: row
                for row in rows
                if row["instance_id"] in set(splits["train"])
            }
            config["embedding_corpora"] = {
                "antibody": side_embedding_fingerprint(
                    antibody_root,
                    [train_rows[name]["ab_sequence_sha256"] for name in splits["train"]],
                    "antibody",
                ),
                "antigen": side_embedding_fingerprint(
                    antigen_root,
                    [train_rows[name]["ag_sequence_sha256"] for name in splits["train"]],
                    "antigen",
                ),
            }
            evidence_files = {}
            for name in (
                "protocol_v1_5_execution_lock",
                "protocol_v1_6_execution_hardening",
                "boundary_verification_v2",
            ):
                evidence_path = root / f"{name}.txt"
                evidence_path.write_text(name, encoding="utf-8")
                evidence_files[name] = {
                    "path": str(evidence_path.resolve()),
                    "sha256": _file_digest(evidence_path),
                }
            synthetic_manifest = root / "synthetic-submission-manifest.json"
            synthetic_manifest.write_text(
                json.dumps({"synthetic": True}), encoding="utf-8"
            )
            canonical_manifest = root / "canonical-manifest.csv"
            canonical_manifest.write_text("synthetic\n", encoding="utf-8")
            synthetic_refit_dir = root / "synthetic-final-refit"
            synthetic_refit_dir.mkdir()
            config["submission_provenance"] = {
                "format_version": FINAL_REFIT_SUBMISSION_FORMAT_VERSION,
                "submission_manifest": {
                    "path": str(synthetic_manifest.resolve()),
                    "sha256": _file_digest(synthetic_manifest),
                },
                "common_train_membership_sha256": common_train_sha256,
                "one_shot_ledger_registry_dir": str(final_registry),
                "refit_output_dir": str(synthetic_refit_dir.resolve()),
                "selected_boundary": "concat",
                "fixed_refit": {"fixed_epochs": 2},
                "immutable_data_inputs": {
                    "prepared_instances": {
                        "path": str((prepare / "instances.parquet").resolve()),
                        "sha256": _file_digest(prepare / "instances.parquet"),
                    },
                    "canonical_sequence_manifest": {
                        "path": str(canonical_manifest.resolve()),
                        "sha256": _file_digest(canonical_manifest),
                    },
                    "training_embedding_content_manifest_sha256": {
                        side: value["content_manifest_sha256"]
                        for side, value in config["embedding_corpora"].items()
                    },
                },
                "upstream_evidence": evidence_files,
                "fixed_evaluation": {
                    "split": "validation",
                    "batch_size": 2,
                    "bucket_width": 16,
                    "recall_ks": [1, 5, 10, 20],
                    "seed": 42,
                    "deterministic": True,
                    "device": "cpu",
                    "isolation_policy": "double_exact",
                    "score_artifact_required": True,
                },
                "authority": {
                    "split_manifest": {
                        "path": str(authority_path.resolve()),
                        "sha256": _file_digest(authority_path),
                    },
                    "authorized_source_pool_splits": ["train", "validation"],
                    "sealed_source_splits": ["test"],
                    "coverage_mode": "exact",
                },
                "all_declared_boundaries": {
                    "concat": {
                        "split_manifest": {
                            "path": str(confirmation_split.resolve()),
                            "sha256": _file_digest(confirmation_split),
                        },
                        "strict_nw_constraint_mode": "concat_antibody_and_antigen",
                        "train_membership_sha256": common_train_sha256,
                        "evaluation_output": str(final_evaluation.resolve()),
                        "score_artifact": str(final_scores.resolve()),
                    },
                    "chain": {
                        "split_manifest": {
                            "path": str(chain_split.resolve()),
                            "sha256": _file_digest(chain_split),
                        },
                        "strict_nw_constraint_mode": (
                            "concat_antibody_antigen_plus_vh_vl"
                        ),
                        "train_membership_sha256": common_train_sha256,
                        "evaluation_output": str((root / "chain-evaluation.json").resolve()),
                        "score_artifact": str((root / "chain-scores.npz").resolve()),
                    },
                },
            }
            config["optimization"]["fixed_epochs"] = 2
            config.pop("config_sha256")
            config["config_sha256"] = canonical_sha256(config)
            synthetic_history = [
                {
                    "epoch": epoch,
                    "global_step": epoch,
                    "train": {"loss": float(3 - epoch)},
                    "learning_rate": 0.001,
                    "temperature": 0.2,
                    "validation_metrics_computed": False,
                }
                for epoch in (1, 2)
            ]
            execution_provenance = {
                "format_version": checkpoint["format_version"],
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "config_sha256": config["config_sha256"],
                "source_sha256": config["source_sha256"],
                "runtime": {
                    "python": sys.version,
                    "platform": "synthetic",
                    "torch": str(torch.__version__),
                    "device": "cpu",
                    "cuda_available": False,
                    "cuda_version": None,
                    "gpu_name": None,
                    "slurm_job_id": None,
                },
                "embedding_locations": {
                    "antibody": str(antibody_root.resolve()),
                    "antigen": str(antigen_root.resolve()),
                },
                "scientific_scope": config["scientific_scope"],
            }
            checkpoint = {
                "format_version": checkpoint["format_version"],
                "stage": "final_refit_complete",
                "config": config,
                "submission_provenance": config["submission_provenance"],
                "execution_provenance_history": [execution_provenance],
                "model_state_dict": checkpoint["model_state_dict"],
                "epoch": 2,
                "global_step": 2,
                "history": synthetic_history,
                "validation_metrics_computed": False,
                "frozen_test_used": False,
            }
            synthetic_final = synthetic_refit_dir / "final.pt"
            torch.save(checkpoint, synthetic_final)
            final_evaluate_command = list(evaluate_command)
            final_evaluate_command[
                final_evaluate_command.index("--checkpoint") + 1
            ] = str(synthetic_final)
            final_evaluate_command[
                final_evaluate_command.index("--checkpoint-sha256") + 1
            ] = _file_digest(synthetic_final)
            final_evaluate_command[
                final_evaluate_command.index("--output") + 1
            ] = str(final_evaluation)
            final_evaluate_command[
                final_evaluate_command.index("--score-artifact") + 1
            ] = str(final_scores)
            final_evaluate_command[
                final_evaluate_command.index("--split-manifest") + 1
            ] = str(confirmation_split)
            final_evaluate_command[
                final_evaluate_command.index("--split-manifest-sha256") + 1
            ] = _file_digest(confirmation_split)
            final_evaluate_command.extend(
                [
                    "--strict-nw-constraint-mode",
                    "concat_antibody_and_antigen",
                ]
            )
            final_evaluate_command.extend(
                ["--one-shot-ledger-registry-dir", str(final_registry)]
            )
            final_evaluate_command.extend(
                [
                    "--prepared-instances-sha256",
                    _file_digest(prepare / "instances.parquet"),
                    "--canonical-sequence-manifest",
                    str(canonical_manifest),
                    "--canonical-sequence-manifest-sha256",
                    _file_digest(canonical_manifest),
                ]
            )
            subprocess.run(
                final_evaluate_command, check=True, capture_output=True, text=True
            )
            ledgers = list(final_registry.glob("validation-*.json"))
            self.assertEqual(len(ledgers), 1)
            ledger = json.loads(ledgers[0].read_text(encoding="utf-8"))
            self.assertEqual(ledger["status"], "completed")
            final_payload = json.loads(final_evaluation.read_text(encoding="utf-8"))
            self.assertEqual(
                final_payload["one_shot_ledger"]["canonical_key"],
                ledger["canonical_key"],
            )
            chain_evaluation = root / "chain-evaluation.json"
            chain_scores = root / "chain-scores.npz"
            chain_command = list(final_evaluate_command)
            chain_command[chain_command.index("--output") + 1] = str(
                chain_evaluation
            )
            chain_command[chain_command.index("--score-artifact") + 1] = str(
                chain_scores
            )
            chain_command[chain_command.index("--split-manifest") + 1] = str(
                chain_split
            )
            chain_command[
                chain_command.index("--split-manifest-sha256") + 1
            ] = _file_digest(chain_split)
            chain_command[
                chain_command.index("--strict-nw-constraint-mode") + 1
            ] = "concat_antibody_antigen_plus_vh_vl"
            subprocess.run(
                chain_command, check=True, capture_output=True, text=True
            )
            completion_path = (
                final_registry
                / "checkpoint-suite-completions"
                / f"{_file_digest(synthetic_manifest)}.json"
            )
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            self.assertEqual(completion["status"], "completed")
            self.assertEqual(set(completion["boundaries"]), {"concat", "chain"})
            second_command = list(final_evaluate_command)
            second_command[second_command.index("--output") + 1] = str(
                root / "different-final-evaluation.json"
            )
            second_command[second_command.index("--score-artifact") + 1] = str(
                root / "different-final-scores.npz"
            )
            repeated = subprocess.run(
                second_command, check=False, capture_output=True, text=True
            )
            self.assertNotEqual(repeated.returncode, 0)
            different_registry_command = list(final_evaluate_command)
            different_registry_command[
                different_registry_command.index("--one-shot-ledger-registry-dir") + 1
            ] = str((root / "different-registry").resolve())
            different_registry = subprocess.run(
                different_registry_command,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(different_registry.returncode, 0)
            self.assertIn("registry differs", different_registry.stderr)
            rejected = subprocess.run(
                [*evaluate_command, "--split", "test"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
