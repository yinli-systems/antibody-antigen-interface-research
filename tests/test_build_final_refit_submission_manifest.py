#!/usr/bin/env python3
"""Tests for the create-only Gate-A submission-manifest builder."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import build_final_refit_submission_manifest as builder
import sabdab2_asymmetric_partner_retrieval as retrieval_core
from build_final_refit_submission_manifest import (
    BOUNDARY_MODES,
    BuildConfig,
    BoundaryReference,
    FileReference,
    SOURCE_NAMES,
    build_and_preflight,
)
from sabdab2_asymmetric_partner_retrieval import (
    FINAL_REFIT_ANTIBODY_EMBEDDING_MANIFEST_SHA256,
    FINAL_REFIT_ANTIGEN_EMBEDDING_MANIFEST_SHA256,
    FINAL_REFIT_SUBMISSION_FORMAT_VERSION,
    _source_hashes,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _membership_digest(identifiers: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(identifiers)) + "\n").encode("utf-8")
    ).hexdigest()


def _write(path: Path, content: bytes) -> FileReference:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return FileReference(path=path.resolve(), sha256=_digest(path))


def _write_json(path: Path, payload: object) -> FileReference:
    return _write(
        path,
        (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )


def _boundary_payload(train: list[str]) -> dict[str, object]:
    return {
        "splits": {
            "train": train,
            "validation": ["validation-instance"],
            "excluded_validation": [],
        },
        "membership_sha256": {"train": _membership_digest(train)},
    }


def _synthetic_config(root: Path) -> BuildConfig:
    train = ["train-a", "train-b", "train-c"]
    common = _membership_digest(train)
    inputs = root / "inputs"
    sources = {
        name: _write(inputs / f"source-{name}.py", f"{name}\n".encode("ascii"))
        for name in SOURCE_NAMES
    }
    concat = _write_json(inputs / "concat-boundary.json", _boundary_payload(train))
    chain = _write_json(inputs / "chain-boundary.json", _boundary_payload(train))
    return BuildConfig(
        mode="preflight",
        manifest_output=(root / "submission-manifest.json").resolve(),
        fold_selection=_write_json(inputs / "fold-selection.json", {"value": 1}),
        independent_audit=_write_json(inputs / "audit.json", {"valid": True}),
        protocol_v1_3=_write(inputs / "protocol-v1.3.md", b"v1.3\n"),
        protocol_v1_4_correction=_write(
            inputs / "protocol-v1.4.md", b"v1.4\n"
        ),
        protocol_v1_5_execution_lock=_write(
            inputs / "protocol-v1.5.md", b"v1.5\n"
        ),
        protocol_v1_6_execution_hardening=_write(
            inputs / "protocol-v1.6.md", b"v1.6\n"
        ),
        boundary_verification_v2=_write_json(
            inputs / "boundary-verification.json", {"valid": True}
        ),
        authority_split_manifest=_write_json(
            inputs / "authority.json", {"splits": {}}
        ),
        prepared_instances=_write(inputs / "instances.parquet", b"parquet-bytes\n"),
        canonical_sequence_manifest=_write(
            inputs / "canonical-manifest.csv", b"sequence_key\n"
        ),
        antibody_embedding_content_manifest_sha256=(
            FINAL_REFIT_ANTIBODY_EMBEDDING_MANIFEST_SHA256
        ),
        antigen_embedding_content_manifest_sha256=(
            FINAL_REFIT_ANTIGEN_EMBEDDING_MANIFEST_SHA256
        ),
        common_train_membership_sha256=common,
        repaired_concat_boundary=BoundaryReference(
            split_manifest=concat,
            evaluation_output=(root / "concat-evaluation.json").resolve(),
            score_artifact=(root / "concat-scores.npz").resolve(),
        ),
        repaired_chain_boundary=BoundaryReference(
            split_manifest=chain,
            evaluation_output=(root / "chain-evaluation.json").resolve(),
            score_artifact=(root / "chain-scores.npz").resolve(),
        ),
        source_files=sources,
        submission_script=_write(inputs / "submit.sbatch", b"sbatch\n"),
        refit_output_dir=(root / "refit-output").resolve(),
        one_shot_ledger_registry_dir=(root / "one-shot-registry").resolve(),
    )


def _synthetic_locked_inputs(config: BuildConfig) -> dict[str, str]:
    return {
        "fold_selection": config.fold_selection.sha256,
        "independent_audit": config.independent_audit.sha256,
        "protocol_v1_3": config.protocol_v1_3.sha256,
        "protocol_v1_4_correction": config.protocol_v1_4_correction.sha256,
        "protocol_v1_5_execution_lock": config.protocol_v1_5_execution_lock.sha256,
        "protocol_v1_6_execution_hardening": (
            config.protocol_v1_6_execution_hardening.sha256
        ),
        "boundary_verification_v2": config.boundary_verification_v2.sha256,
        "authority_split_manifest": config.authority_split_manifest.sha256,
        "prepared_instances": config.prepared_instances.sha256,
        "canonical_sequence_manifest": config.canonical_sequence_manifest.sha256,
    }


def _replace_artifact_path(
    config: BuildConfig,
    label: str,
    path: Path,
) -> BuildConfig:
    """Return a config with one reserved evaluation or score path changed."""
    if label.startswith("concat_"):
        field = label.removeprefix("concat_")
        boundary = replace(config.repaired_concat_boundary, **{field: path})
        return replace(config, repaired_concat_boundary=boundary)
    field = label.removeprefix("chain_")
    boundary = replace(config.repaired_chain_boundary, **{field: path})
    return replace(config, repaired_chain_boundary=boundary)


def _artifact_paths(config: BuildConfig) -> dict[str, Path]:
    """Return the four reserved one-shot artifact paths."""
    return {
        "concat_evaluation_output": config.repaired_concat_boundary.evaluation_output,
        "concat_score_artifact": config.repaired_concat_boundary.score_artifact,
        "chain_evaluation_output": config.repaired_chain_boundary.evaluation_output,
        "chain_score_artifact": config.repaired_chain_boundary.score_artifact,
    }


@contextmanager
def _synthetic_contract(
    config: BuildConfig,
    *,
    source_hashes: dict[str, str] | None = None,
) -> Iterator[mock.Mock]:
    expected_sources = source_hashes or {
        name: config.source_files[name].sha256 for name in SOURCE_NAMES
    }
    validator = mock.Mock(return_value={"synthetic_core_preflight": True})
    with (
        mock.patch.object(
            builder,
            "_locked_input_sha256",
            return_value=_synthetic_locked_inputs(config),
        ),
        mock.patch.object(builder, "_source_hashes", return_value=expected_sources),
        mock.patch.object(
            builder,
            "LOCKED_COMMON_TRAIN_MEMBERSHIP_SHA256",
            config.common_train_membership_sha256,
        ),
        mock.patch.object(
            builder,
            "validate_final_refit_submission_manifest",
            validator,
        ),
    ):
        yield validator


def _real_config(root: Path) -> BuildConfig:
    project_root = Path(__file__).resolve().parents[2]
    job_root = Path(__file__).resolve().parent
    evidence_root = project_root / "outputs" / "interaction_first" / "research"
    boundary_root = evidence_root / "authority_validation_nw_boundaries"
    source_root = root / "source-snapshot"
    source_root.mkdir()
    live_sources = {
        "entrypoint": job_root / "sabdab2_asymmetric_partner_retrieval.py",
        "asymmetric_retrieval": job_root / "interaction_first" / "asymmetric_retrieval.py",
        "shared_retrieval_protocol": job_root / "sabdab2_partner_retrieval.py",
        "runtime": job_root / "interaction_first" / "runtime.py",
    }
    sources: dict[str, FileReference] = {}
    for name, live in live_sources.items():
        snapshot = source_root / f"{name}{live.suffix}"
        shutil.copyfile(live, snapshot)
        sources[name] = FileReference(snapshot.resolve(), _digest(snapshot))
    live_script = job_root / "asymmetric_partner_retrieval_clean_holdout_train.sbatch"
    script = source_root / live_script.name
    shutil.copyfile(live_script, script)

    def reference(path: Path) -> FileReference:
        return FileReference(path.resolve(), _digest(path))

    concat = boundary_root / "concat_boundary_v2.json"
    chain = boundary_root / "chain_boundary_v2.json"
    common = json.loads(concat.read_text(encoding="utf-8"))["membership_sha256"][
        "train"
    ]
    return BuildConfig(
        mode="preflight",
        manifest_output=(root / "submission-manifest.json").resolve(),
        fold_selection=reference(
            evidence_root / "asymmetric_retrieval_fold_aggregation" / "fold_selection.json"
        ),
        independent_audit=reference(
            evidence_root / "asymmetric_retrieval_fold_screen_audit" / "audit.json"
        ),
        protocol_v1_3=reference(
            evidence_root / "PARTNER_RETRIEVAL_PROTOCOL_V1_3_AMENDMENT.md"
        ),
        protocol_v1_4_correction=reference(
            evidence_root / "PARTNER_RETRIEVAL_PROTOCOL_V1_4_CORRECTION.md"
        ),
        protocol_v1_5_execution_lock=reference(
            evidence_root / "PARTNER_RETRIEVAL_PROTOCOL_V1_5_EXECUTION_LOCK.md"
        ),
        protocol_v1_6_execution_hardening=reference(
            evidence_root / "PARTNER_RETRIEVAL_PROTOCOL_V1_6_EXECUTION_HARDENING.md"
        ),
        boundary_verification_v2=reference(
            boundary_root / "independent_verification_v2.json"
        ),
        authority_split_manifest=reference(
            project_root / "outputs" / "data_snapshot" / "double_remote_split.json"
        ),
        prepared_instances=reference(
            project_root / "outputs" / "data_snapshot" / "instances.parquet"
        ),
        canonical_sequence_manifest=reference(
            project_root
            / "outputs"
            / "interaction_first"
            / "remote"
            / "canonical_manifest.csv"
        ),
        antibody_embedding_content_manifest_sha256=(
            FINAL_REFIT_ANTIBODY_EMBEDDING_MANIFEST_SHA256
        ),
        antigen_embedding_content_manifest_sha256=(
            FINAL_REFIT_ANTIGEN_EMBEDDING_MANIFEST_SHA256
        ),
        common_train_membership_sha256=common,
        repaired_concat_boundary=BoundaryReference(
            split_manifest=reference(concat),
            evaluation_output=(root / "concat-evaluation.json").resolve(),
            score_artifact=(root / "concat-scores.npz").resolve(),
        ),
        repaired_chain_boundary=BoundaryReference(
            split_manifest=reference(chain),
            evaluation_output=(root / "chain-evaluation.json").resolve(),
            score_artifact=(root / "chain-scores.npz").resolve(),
        ),
        source_files=sources,
        submission_script=reference(script),
        refit_output_dir=(root / "refit-output").resolve(),
        one_shot_ledger_registry_dir=(root / "one-shot-registry").resolve(),
    )


class SyntheticBuilderTests(unittest.TestCase):
    def test_exact_schema_and_two_preflight_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _synthetic_config(Path(directory))
            with _synthetic_contract(config) as validator:
                receipt = build_and_preflight(config)
            payload = json.loads(config.manifest_output.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload),
                {
                    "format_version",
                    "intended_use",
                    "frozen_test_used",
                    "selected_variant",
                    "upstream_evidence",
                    "authority",
                    "immutable_data_inputs",
                    "model",
                    "fixed_refit",
                    "fixed_evaluation",
                    "common_train_membership_sha256",
                    "canonical_refit_boundary",
                    "refit_output_dir",
                    "one_shot_ledger_registry_dir",
                    "boundaries",
                    "source_files",
                    "submission_script",
                },
            )
            self.assertEqual(
                payload["format_version"], FINAL_REFIT_SUBMISSION_FORMAT_VERSION
            )
            self.assertEqual(set(payload["boundaries"]), set(BOUNDARY_MODES))
            self.assertEqual(validator.call_count, 2)
            final_args = validator.call_args_list[-1].args[0]
            self.assertEqual(final_args.mode, "preflight")
            self.assertFalse(final_args.materialize_dataset_rows)
            self.assertFalse(final_args.materialize_embedding_rows)
            self.assertEqual(final_args.submission_manifest, config.manifest_output)
            self.assertTrue(receipt["published_create_only"])
            self.assertFalse(list(Path(directory).glob(".gate-a-manifest.*")))

    def test_physical_tampering_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _synthetic_config(Path(directory))
            config.fold_selection.path.write_bytes(b"tampered\n")
            with _synthetic_contract(config), self.assertRaisesRegex(
                ValueError, "physical SHA-256 mismatch"
            ):
                build_and_preflight(config)
            self.assertFalse(config.manifest_output.exists())

    def test_existing_manifest_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _synthetic_config(Path(directory))
            sentinel = b"preserve-me\n"
            config.manifest_output.write_bytes(sentinel)
            with _synthetic_contract(config), self.assertRaisesRegex(
                FileExistsError, "already exists"
            ):
                build_and_preflight(config)
            self.assertEqual(config.manifest_output.read_bytes(), sentinel)

    def test_output_path_collisions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _synthetic_config(Path(directory))
            changed_chain = replace(
                config.repaired_chain_boundary,
                score_artifact=config.repaired_concat_boundary.score_artifact,
            )
            config = replace(config, repaired_chain_boundary=changed_chain)
            with _synthetic_contract(config), self.assertRaisesRegex(
                ValueError, "Path collision"
            ):
                build_and_preflight(config)
            self.assertFalse(config.manifest_output.exists())

    def test_artifacts_cannot_overlap_protected_directory_trees(self) -> None:
        artifacts = (
            "concat_evaluation_output",
            "concat_score_artifact",
            "chain_evaluation_output",
            "chain_score_artifact",
        )
        for protected_name in (
            "refit_output_dir",
            "one_shot_ledger_registry_dir",
        ):
            for artifact_name in artifacts:
                for relation in ("equal", "descendant", "ancestor"):
                    with (
                        self.subTest(
                            protected=protected_name,
                            artifact=artifact_name,
                            relation=relation,
                        ),
                        tempfile.TemporaryDirectory() as directory,
                    ):
                        root = Path(directory)
                        config = _synthetic_config(root)
                        anchor = (root / "protected-overlap").resolve()
                        if relation == "equal":
                            protected = anchor
                            artifact = anchor
                        elif relation == "descendant":
                            protected = anchor
                            artifact = anchor / "artifact.bin"
                        else:
                            artifact = anchor
                            protected = anchor / "protected-child"
                        config = replace(config, **{protected_name: protected})
                        config = _replace_artifact_path(
                            config,
                            artifact_name,
                            artifact,
                        )
                        with _synthetic_contract(config), self.assertRaisesRegex(
                            ValueError, "overlaps protected directory tree"
                        ):
                            build_and_preflight(config)
                        self.assertFalse(config.manifest_output.exists())

    def test_input_hardlink_alias_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _synthetic_config(root)
            alias = root / "inputs" / "fold-selection-hardlink.json"
            os.link(config.fold_selection.path, alias)
            config = replace(
                config,
                independent_audit=FileReference(
                    path=alias.resolve(),
                    sha256=config.fold_selection.sha256,
                ),
            )
            with _synthetic_contract(config), self.assertRaisesRegex(
                ValueError, "Physical input collision"
            ):
                build_and_preflight(config)
            self.assertFalse(config.manifest_output.exists())

    def test_existing_evaluation_and_score_sentinels_are_preserved(self) -> None:
        for artifact_name in (
            "concat_evaluation_output",
            "concat_score_artifact",
            "chain_evaluation_output",
            "chain_score_artifact",
        ):
            with self.subTest(artifact=artifact_name), tempfile.TemporaryDirectory() as directory:
                config = _synthetic_config(Path(directory))
                artifact = _artifact_paths(config)[artifact_name]
                sentinel = f"preserve-{artifact_name}\n".encode("ascii")
                artifact.write_bytes(sentinel)
                with _synthetic_contract(config), self.assertRaisesRegex(
                    FileExistsError, "Reserved output already exists"
                ):
                    build_and_preflight(config)
                self.assertEqual(artifact.read_bytes(), sentinel)
                self.assertFalse(config.manifest_output.exists())

    def test_inode_replacement_survives_failed_post_publication_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _synthetic_config(root)
            replacement_identity: tuple[int, int] | None = None
            replacement_content: bytes | None = None

            def validate(args: Any) -> dict[str, bool]:
                nonlocal replacement_content, replacement_identity
                manifest = args.submission_manifest
                if manifest == config.manifest_output:
                    replacement_content = manifest.read_bytes()
                    replacement = root / "same-byte-replacement.json"
                    replacement.write_bytes(replacement_content)
                    replacement_identity = (
                        replacement.stat().st_dev,
                        replacement.stat().st_ino,
                    )
                    os.replace(replacement, manifest)
                    raise RuntimeError("synthetic post-publication failure")
                return {"synthetic_core_preflight": True}

            with _synthetic_contract(config) as validator:
                validator.side_effect = validate
                with self.assertRaisesRegex(RuntimeError, "post-publication failure"):
                    build_and_preflight(config)
            self.assertTrue(config.manifest_output.is_file())
            self.assertEqual(config.manifest_output.read_bytes(), replacement_content)
            self.assertEqual(
                (
                    config.manifest_output.stat().st_dev,
                    config.manifest_output.stat().st_ino,
                ),
                replacement_identity,
            )

    def test_source_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _synthetic_config(Path(directory))
            expected = {
                name: config.source_files[name].sha256 for name in SOURCE_NAMES
            }
            expected["entrypoint"] = hashlib.sha256(b"different").hexdigest()
            with _synthetic_contract(
                config, source_hashes=expected
            ), self.assertRaisesRegex(ValueError, "locked reviewed artifact"):
                build_and_preflight(config)
            self.assertFalse(config.manifest_output.exists())

    def test_duplicate_and_nonfinite_json_are_rejected(self) -> None:
        for content, pattern in (
            (b'{"value":1,"value":2}\n', "duplicate JSON key"),
            (b'{"value":NaN}\n', "non-finite JSON value"),
            (b'{"outer":[{"value":1e999}]}\n', "non-finite JSON value"),
        ):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
                config = _synthetic_config(Path(directory))
                config.fold_selection.path.write_bytes(content)
                changed = replace(
                    config.fold_selection,
                    sha256=_digest(config.fold_selection.path),
                )
                config = replace(config, fold_selection=changed)
                with _synthetic_contract(config), self.assertRaisesRegex(
                    ValueError, pattern
                ):
                    build_and_preflight(config)

    def test_relative_input_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _synthetic_config(Path(directory))
            changed = replace(config.fold_selection, path=Path("fold-selection.json"))
            config = replace(config, fold_selection=changed)
            with _synthetic_contract(config), self.assertRaisesRegex(
                ValueError, "explicit absolute path"
            ):
                build_and_preflight(config)


class LockedIntegrationTests(unittest.TestCase):
    def test_real_locked_evidence_passes_core_preflight_without_row_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _real_config(Path(directory))
            self.assertEqual(
                {name: reference.sha256 for name, reference in config.source_files.items()},
                _source_hashes(),
            )
            loader_names = (
                "load_development_dataset",
                "load_excluded_negative_pairs",
                "side_embedding_fingerprint",
                "FrozenPartnerEmbeddingStore",
                "_embedding_corpora",
            )
            with ExitStack() as stack:
                loader_spies = {
                    name: stack.enter_context(
                        mock.patch.object(
                            retrieval_core,
                            name,
                            side_effect=AssertionError(
                                f"row loader called during preflight: {name}"
                            ),
                        )
                    )
                    for name in loader_names
                }
                receipt = build_and_preflight(config)
            for spy in loader_spies.values():
                spy.assert_not_called()
            self.assertTrue(config.manifest_output.is_file())
            self.assertTrue(
                receipt[
                    "validated_before_dataset_or_embedding_row_materialization"
                ]
            )
            self.assertTrue(
                receipt["core_preflight"][
                    "validated_before_dataset_or_embedding_load"
                ]
            )


if __name__ == "__main__":
    unittest.main()
