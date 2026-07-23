#!/usr/bin/env python3
"""Build and preflight the immutable Gate-A final-refit submission manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sabdab2_asymmetric_partner_retrieval import (
    FINAL_REFIT_ANTIBODY_EMBEDDING_MANIFEST_SHA256,
    FINAL_REFIT_ANTIGEN_EMBEDDING_MANIFEST_SHA256,
    FINAL_REFIT_AUTHORITY_SHA256,
    FINAL_REFIT_BEST_EPOCHS,
    FINAL_REFIT_BOUNDARY_VERIFICATION_SHA256,
    FINAL_REFIT_CANONICAL_SEQUENCE_MANIFEST_SHA256,
    FINAL_REFIT_FIXED_EPOCHS,
    FINAL_REFIT_FOLD_SELECTION_SHA256,
    FINAL_REFIT_INDEPENDENT_AUDIT_SHA256,
    FINAL_REFIT_PREPARED_INSTANCES_SHA256,
    FINAL_REFIT_PROTOCOL_V1_3_SHA256,
    FINAL_REFIT_PROTOCOL_V1_4_SHA256,
    FINAL_REFIT_PROTOCOL_V1_5_SHA256,
    FINAL_REFIT_PROTOCOL_V1_6_SHA256,
    FINAL_REFIT_SELECTED_VARIANT,
    FINAL_REFIT_SUBMISSION_FORMAT_VERSION,
    _expected_final_evaluation_contract,
    _expected_final_refit_model_contract,
    _expected_final_refit_optimization_contract,
    _source_hashes,
    validate_final_refit_submission_manifest,
)


HEX_DIGEST_LENGTH = 64
LOCKED_COMMON_TRAIN_MEMBERSHIP_SHA256 = (
    "442095581ae9bf3c7440c1a818612298b5f81e5a62ef46f431e0867b9c8b466b"
)
BOUNDARY_MODES = {
    "repaired_concat_boundary": "concat_antibody_and_antigen",
    "repaired_chain_boundary": "concat_antibody_antigen_plus_vh_vl",
}
SOURCE_NAMES = (
    "entrypoint",
    "asymmetric_retrieval",
    "shared_retrieval_protocol",
    "runtime",
)


@dataclass(frozen=True)
class FileReference:
    """Describe one immutable physical input file."""

    path: Path
    sha256: str


@dataclass(frozen=True)
class BoundaryReference:
    """Describe one repaired boundary and its reserved one-shot outputs."""

    split_manifest: FileReference
    evaluation_output: Path
    score_artifact: Path


@dataclass(frozen=True)
class BuildConfig:
    """Collect every path and digest required to publish one manifest."""

    mode: str
    manifest_output: Path
    fold_selection: FileReference
    independent_audit: FileReference
    protocol_v1_3: FileReference
    protocol_v1_4_correction: FileReference
    protocol_v1_5_execution_lock: FileReference
    protocol_v1_6_execution_hardening: FileReference
    boundary_verification_v2: FileReference
    authority_split_manifest: FileReference
    prepared_instances: FileReference
    canonical_sequence_manifest: FileReference
    antibody_embedding_content_manifest_sha256: str
    antigen_embedding_content_manifest_sha256: str
    common_train_membership_sha256: str
    repaired_concat_boundary: BoundaryReference
    repaired_chain_boundary: BoundaryReference
    source_files: Mapping[str, FileReference]
    submission_script: FileReference
    refit_output_dir: Path
    one_shot_ledger_registry_dir: Path


def _is_sha256(value: Any) -> bool:
    """Return whether a value is a canonical lowercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == HEX_DIGEST_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    """Stream-hash a regular file without materializing its logical rows."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        descriptor_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise ValueError(f"Input is not a regular file: {path}")
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _strict_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read a finite JSON object while rejecting duplicate object keys."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains a non-finite JSON value: {value}")

    def require_finite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite JSON value: {value}")
        if isinstance(value, Mapping):
            for nested in value.values():
                require_finite(nested)
        elif isinstance(value, list):
            for nested in value:
                require_finite(nested)

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} as strict JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    require_finite(payload)
    return payload


def _absolute_path(value: Path | str, label: str) -> Path:
    """Resolve an explicitly absolute path without accepting home shortcuts."""
    raw = Path(value)
    if not raw.is_absolute():
        raise ValueError(f"{label} must be an explicit absolute path")
    return raw.resolve()


def _require_sha256(value: str, label: str) -> str:
    """Require one canonical SHA-256 value."""
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _locked_input_sha256() -> dict[str, str]:
    """Return the reviewed byte digests pinned by the Gate-A protocol."""
    return {
        "fold_selection": FINAL_REFIT_FOLD_SELECTION_SHA256,
        "independent_audit": FINAL_REFIT_INDEPENDENT_AUDIT_SHA256,
        "protocol_v1_3": FINAL_REFIT_PROTOCOL_V1_3_SHA256,
        "protocol_v1_4_correction": FINAL_REFIT_PROTOCOL_V1_4_SHA256,
        "protocol_v1_5_execution_lock": FINAL_REFIT_PROTOCOL_V1_5_SHA256,
        "protocol_v1_6_execution_hardening": FINAL_REFIT_PROTOCOL_V1_6_SHA256,
        "boundary_verification_v2": FINAL_REFIT_BOUNDARY_VERIFICATION_SHA256,
        "authority_split_manifest": FINAL_REFIT_AUTHORITY_SHA256,
        "prepared_instances": FINAL_REFIT_PREPARED_INSTANCES_SHA256,
        "canonical_sequence_manifest": (
            FINAL_REFIT_CANONICAL_SEQUENCE_MANIFEST_SHA256
        ),
    }


def _bind_file(
    reference: FileReference,
    label: str,
    *,
    locked_sha256: str | None = None,
) -> dict[str, str]:
    """Normalize and physically verify one explicit path-and-digest input."""
    path = _absolute_path(reference.path, label)
    expected = _require_sha256(reference.sha256, f"{label}.sha256")
    if locked_sha256 is not None and expected != locked_sha256:
        raise ValueError(
            f"{label}.sha256 differs from the locked reviewed artifact: "
            f"expected={locked_sha256}, observed={expected}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"{label} physical file is missing: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} physical SHA-256 mismatch: "
            f"expected={expected}, observed={observed}"
        )
    return {"path": str(path), "sha256": expected}


def _membership_sha256(identifiers: Sequence[str]) -> str:
    """Hash one sorted identifier inventory using the locked boundary contract."""
    return hashlib.sha256(
        ("\n".join(sorted(identifiers)) + "\n").encode("utf-8")
    ).hexdigest()


def _validate_boundary_membership(
    path: Path,
    label: str,
    expected_membership_sha256: str,
) -> tuple[str, ...]:
    """Verify a boundary's declared and physically recomputed train membership."""
    payload = _strict_json_object(path, label)
    splits = payload.get("splits")
    membership = payload.get("membership_sha256")
    if not isinstance(splits, Mapping) or not isinstance(membership, Mapping):
        raise ValueError(f"{label} lacks split membership evidence")
    train = splits.get("train")
    if (
        not isinstance(train, list)
        or not train
        or not all(isinstance(identifier, str) and identifier for identifier in train)
        or len(train) != len(set(train))
    ):
        raise ValueError(f"{label} has invalid train identifiers")
    observed = _membership_sha256(train)
    if (
        observed != expected_membership_sha256
        or membership.get("train") != expected_membership_sha256
    ):
        raise ValueError(
            f"{label} does not carry the locked common train membership"
        )
    return tuple(sorted(train))


def _verify_json_inputs(bound_inputs: Mapping[str, Mapping[str, str]]) -> None:
    """Apply strict JSON parsing to every JSON evidence input."""
    json_labels = (
        "fold_selection",
        "independent_audit",
        "boundary_verification_v2",
        "authority_split_manifest",
    )
    for label in json_labels:
        _strict_json_object(Path(bound_inputs[label]["path"]), label)


def _validate_distinct_paths(
    input_paths: Mapping[str, Path],
    output_paths: Mapping[str, Path],
) -> None:
    """Reject canonical path aliases, input hard-link aliases, and used outputs."""
    canonical: dict[str, str] = {}
    for label, path in {**input_paths, **output_paths}.items():
        key = str(path)
        if key in canonical:
            raise ValueError(
                f"Path collision between {canonical[key]} and {label}: {path}"
            )
        canonical[key] = label

    physical_inputs: dict[tuple[int, int], str] = {}
    for label, path in input_paths.items():
        value = path.stat()
        identity = (value.st_dev, value.st_ino)
        if identity in physical_inputs:
            raise ValueError(
                f"Physical input collision between {physical_inputs[identity]} "
                f"and {label}"
            )
        physical_inputs[identity] = label

    reserved_files = {
        label: path
        for label, path in output_paths.items()
        if label not in {"refit_output_dir", "one_shot_ledger_registry_dir"}
    }
    for label, path in reserved_files.items():
        if path.exists():
            raise FileExistsError(f"Reserved output already exists for {label}: {path}")
    for label in ("refit_output_dir", "one_shot_ledger_registry_dir"):
        path = output_paths[label]
        if path.exists() and not path.is_dir():
            raise ValueError(f"{label} exists but is not a directory: {path}")


def _paths_overlap_as_trees(first: Path, second: Path) -> bool:
    """Return whether either canonical path contains the other."""
    return first == second or first in second.parents or second in first.parents


def _validate_protected_output_trees(output_paths: Mapping[str, Path]) -> None:
    """Keep one-shot files completely outside checkpoint and registry trees."""
    artifact_labels = (
        "repaired_concat_boundary.evaluation_output",
        "repaired_concat_boundary.score_artifact",
        "repaired_chain_boundary.evaluation_output",
        "repaired_chain_boundary.score_artifact",
    )
    protected_labels = ("refit_output_dir", "one_shot_ledger_registry_dir")
    for artifact_label in artifact_labels:
        artifact = output_paths[artifact_label]
        for protected_label in protected_labels:
            protected = output_paths[protected_label]
            if _paths_overlap_as_trees(artifact, protected):
                raise ValueError(
                    f"{artifact_label} overlaps protected directory tree "
                    f"{protected_label}: artifact={artifact}, protected={protected}"
                )


def _build_namespace(
    manifest_path: Path,
    manifest_sha256: str,
    payload: Mapping[str, Any],
) -> Namespace:
    """Construct the data-free core-validator namespace for preflight mode."""
    model = _expected_final_refit_model_contract()
    fixed_refit = _expected_final_refit_optimization_contract()
    fixed_evaluation = _expected_final_evaluation_contract()
    optimizer = fixed_refit["optimizer"]
    scheduler = fixed_refit["scheduler"]
    canonical = payload["boundaries"][payload["canonical_refit_boundary"]]
    prepared_instances = Path(
        payload["immutable_data_inputs"]["prepared_instances"]["path"]
    )
    return Namespace(
        mode="preflight",
        command="preflight",
        preflight=True,
        materialize_dataset_rows=False,
        materialize_embedding_rows=False,
        submission_manifest=manifest_path,
        submission_manifest_sha256=manifest_sha256,
        split_manifest=Path(canonical["split_manifest"]["path"]),
        split_manifest_sha256=canonical["split_manifest"]["sha256"],
        strict_nw_constraint_mode=canonical["strict_nw_constraint_mode"],
        output_dir=Path(payload["refit_output_dir"]),
        prepare_dir=prepared_instances.parent,
        prepared_instances_sha256=payload["immutable_data_inputs"][
            "prepared_instances"
        ]["sha256"],
        canonical_sequence_manifest=Path(
            payload["immutable_data_inputs"]["canonical_sequence_manifest"]["path"]
        ),
        canonical_sequence_manifest_sha256=payload["immutable_data_inputs"][
            "canonical_sequence_manifest"
        ]["sha256"],
        authority_split_manifest=Path(payload["authority"]["split_manifest"]["path"]),
        authority_split_sha256=payload["authority"]["split_manifest"]["sha256"],
        authorized_source_pool_splits=list(
            payload["authority"]["authorized_source_pool_splits"]
        ),
        authority_sealed_splits=list(payload["authority"]["sealed_source_splits"]),
        authority_coverage_mode=payload["authority"]["coverage_mode"],
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
        chain_balanced_antibody_pooling=model[
            "chain_balanced_antibody_pooling"
        ],
        antibody_pooling=model["antibody_pooling"],
        antigen_pooling=model["antigen_pooling"],
        initial_temperature=model["initial_temperature"],
        minimum_temperature=model["minimum_temperature"],
        maximum_temperature=model["maximum_temperature"],
        fixed_epochs=fixed_refit["fixed_epochs"],
        seed=fixed_refit["seed"],
        batch_size=fixed_refit["batch_size"],
        bucket_width=fixed_refit["bucket_width"],
        recall_ks=list(fixed_evaluation["recall_ks"]),
        device=fixed_refit["training_device"],
        allow_nondeterministic=not fixed_refit["deterministic"],
        isolation_policy=fixed_refit["isolation_policy"],
        excluded_negative_pairs=fixed_refit["excluded_negative_pairs"],
        learning_rate=optimizer["learning_rate"],
        minimum_learning_rate=optimizer["minimum_learning_rate"],
        weight_decay=optimizer["weight_decay"],
        gradient_clip=optimizer["gradient_clip"],
        schedule=scheduler["name"],
        warmup_fraction=scheduler["warmup_fraction"],
        restart_period_epochs=scheduler["restart_period_epochs"],
        restart_multiplier=scheduler["restart_multiplier"],
    )


def _serialize_manifest(payload: Mapping[str, Any]) -> bytes:
    """Serialize the exact finite manifest schema deterministically."""
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    return text.encode("utf-8")


def _write_staged_file(parent: Path, content: bytes) -> Path:
    """Create and fsync one private same-filesystem staging file."""
    descriptor, name = tempfile.mkstemp(prefix=".gate-a-manifest.", dir=parent)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry when the platform supports directory fsync."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_identity(path: Path) -> tuple[int, int]:
    """Return one path entry's device and inode without following symlinks."""
    value = path.lstat()
    return value.st_dev, value.st_ino


def _unlink_if_identity(path: Path, expected_identity: tuple[int, int]) -> bool:
    """Unlink a path only while it still names the expected inode."""
    try:
        if _path_identity(path) != expected_identity:
            return False
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def _publish_hard_link(
    staged: Path,
    target: Path,
    staged_identity: tuple[int, int],
) -> tuple[int, int]:
    """Publish a staged manifest with a single no-clobber hard-link operation."""
    published_identity: tuple[int, int] | None = None
    try:
        os.link(staged, target, follow_symlinks=False)
        published_identity = staged_identity
        if _path_identity(target) != staged_identity:
            raise RuntimeError("Published manifest does not share the staged inode")
        _fsync_directory(target.parent)
    except FileExistsError as error:
        raise FileExistsError(f"Manifest output already exists: {target}") from error
    except BaseException:
        if published_identity is not None:
            _unlink_if_identity(target, published_identity)
        raise
    return published_identity


def _cleanup_failed_publication(
    target: Path,
    published_identity: tuple[int, int],
) -> None:
    """Remove only the exact invalid artifact created by this invocation."""
    try:
        if _unlink_if_identity(target, published_identity):
            _fsync_directory(target.parent)
    except OSError:
        pass


def build_and_preflight(config: BuildConfig) -> dict[str, Any]:
    """Build, publish, and core-validate one immutable Gate-A manifest."""
    if config.mode != "preflight":
        raise ValueError("Gate-A manifest construction is restricted to preflight mode")

    locked = _locked_input_sha256()
    references = {
        "fold_selection": config.fold_selection,
        "independent_audit": config.independent_audit,
        "protocol_v1_3": config.protocol_v1_3,
        "protocol_v1_4_correction": config.protocol_v1_4_correction,
        "protocol_v1_5_execution_lock": config.protocol_v1_5_execution_lock,
        "protocol_v1_6_execution_hardening": (
            config.protocol_v1_6_execution_hardening
        ),
        "boundary_verification_v2": config.boundary_verification_v2,
        "authority_split_manifest": config.authority_split_manifest,
        "prepared_instances": config.prepared_instances,
        "canonical_sequence_manifest": config.canonical_sequence_manifest,
    }
    bound_inputs = {
        label: _bind_file(reference, label, locked_sha256=locked[label])
        for label, reference in references.items()
    }
    _verify_json_inputs(bound_inputs)

    live_source_hashes = _source_hashes()
    if set(config.source_files) != set(SOURCE_NAMES) or set(live_source_hashes) != set(
        SOURCE_NAMES
    ):
        raise ValueError("Source files must contain exactly the four locked semantic sources")
    bound_sources = {
        name: _bind_file(
            config.source_files[name],
            f"source_files.{name}",
            locked_sha256=live_source_hashes[name],
        )
        for name in SOURCE_NAMES
    }
    bound_script = _bind_file(config.submission_script, "submission_script")
    bound_boundaries = {
        "repaired_concat_boundary": _bind_file(
            config.repaired_concat_boundary.split_manifest,
            "boundaries.repaired_concat_boundary.split_manifest",
        ),
        "repaired_chain_boundary": _bind_file(
            config.repaired_chain_boundary.split_manifest,
            "boundaries.repaired_chain_boundary.split_manifest",
        ),
    }

    common_membership = _require_sha256(
        config.common_train_membership_sha256,
        "common_train_membership_sha256",
    )
    if common_membership != LOCKED_COMMON_TRAIN_MEMBERSHIP_SHA256:
        raise ValueError(
            "common_train_membership_sha256 differs from the locked Gate-A inventory"
        )
    concat_train = _validate_boundary_membership(
        Path(bound_boundaries["repaired_concat_boundary"]["path"]),
        "repaired_concat_boundary",
        common_membership,
    )
    chain_train = _validate_boundary_membership(
        Path(bound_boundaries["repaired_chain_boundary"]["path"]),
        "repaired_chain_boundary",
        common_membership,
    )
    if concat_train != chain_train:
        raise ValueError("The two repaired boundaries have different train identifiers")

    antibody_embedding_sha256 = _require_sha256(
        config.antibody_embedding_content_manifest_sha256,
        "antibody_embedding_content_manifest_sha256",
    )
    antigen_embedding_sha256 = _require_sha256(
        config.antigen_embedding_content_manifest_sha256,
        "antigen_embedding_content_manifest_sha256",
    )
    if (
        antibody_embedding_sha256
        != FINAL_REFIT_ANTIBODY_EMBEDDING_MANIFEST_SHA256
        or antigen_embedding_sha256
        != FINAL_REFIT_ANTIGEN_EMBEDDING_MANIFEST_SHA256
    ):
        raise ValueError("Training embedding content digests differ from the locked audit")

    manifest_output = _absolute_path(config.manifest_output, "manifest_output")
    refit_output_dir = _absolute_path(config.refit_output_dir, "refit_output_dir")
    registry_dir = _absolute_path(
        config.one_shot_ledger_registry_dir,
        "one_shot_ledger_registry_dir",
    )
    concat_evaluation = _absolute_path(
        config.repaired_concat_boundary.evaluation_output,
        "repaired_concat_boundary.evaluation_output",
    )
    concat_score = _absolute_path(
        config.repaired_concat_boundary.score_artifact,
        "repaired_concat_boundary.score_artifact",
    )
    chain_evaluation = _absolute_path(
        config.repaired_chain_boundary.evaluation_output,
        "repaired_chain_boundary.evaluation_output",
    )
    chain_score = _absolute_path(
        config.repaired_chain_boundary.score_artifact,
        "repaired_chain_boundary.score_artifact",
    )
    if not manifest_output.parent.is_dir():
        raise FileNotFoundError(
            f"Manifest output parent is missing: {manifest_output.parent}"
        )

    input_paths = {
        **{label: Path(value["path"]) for label, value in bound_inputs.items()},
        **{
            f"source_files.{label}": Path(value["path"])
            for label, value in bound_sources.items()
        },
        "submission_script": Path(bound_script["path"]),
        **{
            f"boundaries.{label}.split_manifest": Path(value["path"])
            for label, value in bound_boundaries.items()
        },
    }
    output_paths = {
        "manifest_output": manifest_output,
        "refit_output_dir": refit_output_dir,
        "one_shot_ledger_registry_dir": registry_dir,
        "repaired_concat_boundary.evaluation_output": concat_evaluation,
        "repaired_concat_boundary.score_artifact": concat_score,
        "repaired_chain_boundary.evaluation_output": chain_evaluation,
        "repaired_chain_boundary.score_artifact": chain_score,
    }
    _validate_protected_output_trees(output_paths)
    _validate_distinct_paths(input_paths, output_paths)

    payload: dict[str, Any] = {
        "format_version": FINAL_REFIT_SUBMISSION_FORMAT_VERSION,
        "intended_use": "development_only",
        "frozen_test_used": False,
        "selected_variant": FINAL_REFIT_SELECTED_VARIANT,
        "upstream_evidence": {
            "fold_selection": {
                **bound_inputs["fold_selection"],
                "selected_variant": FINAL_REFIT_SELECTED_VARIANT,
                "best_epochs": list(FINAL_REFIT_BEST_EPOCHS),
                "median_best_epoch": FINAL_REFIT_FIXED_EPOCHS,
            },
            "independent_audit": bound_inputs["independent_audit"],
            "protocol_v1_3": bound_inputs["protocol_v1_3"],
            "protocol_v1_4_correction": bound_inputs[
                "protocol_v1_4_correction"
            ],
            "protocol_v1_5_execution_lock": bound_inputs[
                "protocol_v1_5_execution_lock"
            ],
            "protocol_v1_6_execution_hardening": bound_inputs[
                "protocol_v1_6_execution_hardening"
            ],
            "boundary_verification_v2": bound_inputs["boundary_verification_v2"],
        },
        "authority": {
            "split_manifest": bound_inputs["authority_split_manifest"],
            "authorized_source_pool_splits": ["train", "validation"],
            "sealed_source_splits": ["test"],
            "coverage_mode": "exact",
        },
        "immutable_data_inputs": {
            "prepared_instances": bound_inputs["prepared_instances"],
            "canonical_sequence_manifest": bound_inputs[
                "canonical_sequence_manifest"
            ],
            "training_embedding_content_manifest_sha256": {
                "antibody": antibody_embedding_sha256,
                "antigen": antigen_embedding_sha256,
            },
        },
        "model": _expected_final_refit_model_contract(),
        "fixed_refit": _expected_final_refit_optimization_contract(),
        "fixed_evaluation": _expected_final_evaluation_contract(),
        "common_train_membership_sha256": common_membership,
        "canonical_refit_boundary": "repaired_concat_boundary",
        "refit_output_dir": str(refit_output_dir),
        "one_shot_ledger_registry_dir": str(registry_dir),
        "boundaries": {
            "repaired_concat_boundary": {
                "split_manifest": bound_boundaries["repaired_concat_boundary"],
                "strict_nw_constraint_mode": BOUNDARY_MODES[
                    "repaired_concat_boundary"
                ],
                "evaluation_output": str(concat_evaluation),
                "score_artifact": str(concat_score),
            },
            "repaired_chain_boundary": {
                "split_manifest": bound_boundaries["repaired_chain_boundary"],
                "strict_nw_constraint_mode": BOUNDARY_MODES[
                    "repaired_chain_boundary"
                ],
                "evaluation_output": str(chain_evaluation),
                "score_artifact": str(chain_score),
            },
        },
        "source_files": bound_sources,
        "submission_script": bound_script,
    }
    content = _serialize_manifest(payload)
    manifest_sha256 = hashlib.sha256(content).hexdigest()
    staged = _write_staged_file(manifest_output.parent, content)
    staged_identity = _path_identity(staged)
    published_identity: tuple[int, int] | None = None
    try:
        validate_final_refit_submission_manifest(
            _build_namespace(staged, manifest_sha256, payload)
        )
        published_identity = _publish_hard_link(
            staged,
            manifest_output,
            staged_identity,
        )
        provenance = validate_final_refit_submission_manifest(
            _build_namespace(manifest_output, manifest_sha256, payload)
        )
    except BaseException:
        if published_identity is not None:
            _cleanup_failed_publication(manifest_output, published_identity)
        raise
    finally:
        staged.unlink(missing_ok=True)

    return {
        "format_version": "sabdab2_gate_a_manifest_builder_receipt_v1",
        "mode": "preflight",
        "manifest": {
            "path": str(manifest_output),
            "sha256": manifest_sha256,
        },
        "published_create_only": True,
        "validated_before_dataset_or_embedding_row_materialization": True,
        "common_train_membership_sha256": common_membership,
        "core_preflight": provenance,
    }


def _add_file_reference(parser: argparse.ArgumentParser, name: str) -> None:
    """Add one required explicit path-and-SHA argument pair."""
    parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(f"--{name}-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    """Build the standalone Gate-A command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight",), required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    for name in (
        "fold-selection",
        "independent-audit",
        "protocol-v1-3",
        "protocol-v1-4-correction",
        "protocol-v1-5-execution-lock",
        "protocol-v1-6-execution-hardening",
        "boundary-verification-v2",
        "authority-split-manifest",
        "prepared-instances",
        "canonical-sequence-manifest",
        "repaired-concat-boundary",
        "repaired-chain-boundary",
        "source-entrypoint",
        "source-asymmetric-retrieval",
        "source-shared-retrieval-protocol",
        "source-runtime",
        "submission-script",
    ):
        _add_file_reference(parser, name)
    parser.add_argument(
        "--antibody-embedding-content-manifest-sha256", required=True
    )
    parser.add_argument(
        "--antigen-embedding-content-manifest-sha256", required=True
    )
    parser.add_argument("--common-train-membership-sha256", required=True)
    parser.add_argument("--refit-output-dir", type=Path, required=True)
    parser.add_argument("--one-shot-ledger-registry-dir", type=Path, required=True)
    parser.add_argument("--concat-evaluation-output", type=Path, required=True)
    parser.add_argument("--concat-score-artifact", type=Path, required=True)
    parser.add_argument("--chain-evaluation-output", type=Path, required=True)
    parser.add_argument("--chain-score-artifact", type=Path, required=True)
    return parser


def _reference(args: Namespace, name: str) -> FileReference:
    """Extract one path-and-SHA pair from parsed arguments."""
    attribute = name.replace("-", "_")
    return FileReference(
        path=getattr(args, attribute),
        sha256=getattr(args, f"{attribute}_sha256"),
    )


def _config_from_args(args: Namespace) -> BuildConfig:
    """Convert parsed CLI values into the immutable builder configuration."""
    return BuildConfig(
        mode=args.mode,
        manifest_output=args.manifest_output,
        fold_selection=_reference(args, "fold-selection"),
        independent_audit=_reference(args, "independent-audit"),
        protocol_v1_3=_reference(args, "protocol-v1-3"),
        protocol_v1_4_correction=_reference(args, "protocol-v1-4-correction"),
        protocol_v1_5_execution_lock=_reference(
            args, "protocol-v1-5-execution-lock"
        ),
        protocol_v1_6_execution_hardening=_reference(
            args, "protocol-v1-6-execution-hardening"
        ),
        boundary_verification_v2=_reference(args, "boundary-verification-v2"),
        authority_split_manifest=_reference(args, "authority-split-manifest"),
        prepared_instances=_reference(args, "prepared-instances"),
        canonical_sequence_manifest=_reference(
            args, "canonical-sequence-manifest"
        ),
        antibody_embedding_content_manifest_sha256=(
            args.antibody_embedding_content_manifest_sha256
        ),
        antigen_embedding_content_manifest_sha256=(
            args.antigen_embedding_content_manifest_sha256
        ),
        common_train_membership_sha256=args.common_train_membership_sha256,
        repaired_concat_boundary=BoundaryReference(
            split_manifest=_reference(args, "repaired-concat-boundary"),
            evaluation_output=args.concat_evaluation_output,
            score_artifact=args.concat_score_artifact,
        ),
        repaired_chain_boundary=BoundaryReference(
            split_manifest=_reference(args, "repaired-chain-boundary"),
            evaluation_output=args.chain_evaluation_output,
            score_artifact=args.chain_score_artifact,
        ),
        source_files={
            "entrypoint": _reference(args, "source-entrypoint"),
            "asymmetric_retrieval": _reference(
                args, "source-asymmetric-retrieval"
            ),
            "shared_retrieval_protocol": _reference(
                args, "source-shared-retrieval-protocol"
            ),
            "runtime": _reference(args, "source-runtime"),
        },
        submission_script=_reference(args, "submission-script"),
        refit_output_dir=args.refit_output_dir,
        one_shot_ledger_registry_dir=args.one_shot_ledger_registry_dir,
    )


def main() -> None:
    """Run manifest construction and print its finite preflight receipt."""
    args = _parser().parse_args()
    receipt = build_and_preflight(_config_from_args(args))
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
