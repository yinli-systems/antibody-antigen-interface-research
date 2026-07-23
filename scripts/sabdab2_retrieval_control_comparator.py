#!/usr/bin/env python3
"""Compare one final retriever with the five locked controls on one v2 boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sabdab2_asymmetric_partner_retrieval as retrieval_runtime
import sabdab2_asymmetric_retrieval_statistics as statistics_runtime
import sabdab2_retrieval_controls as controls_runtime
from sabdab2_asymmetric_partner_retrieval import (
    _score_array_sha256,
    canonical_sha256,
    sha256_file,
)
from sabdab2_asymmetric_retrieval_statistics import (
    AUTHORITY_VALIDATION_BOUNDARY_FORMAT,
    AUTHORITY_VALIDATION_BOUNDARY_INTENDED_USE,
    CONTROL_NAMES,
    CONTROL_SUITE_VERSION,
    DIRECTIONS,
    METRIC_NAMES,
    _best_positive_ranks,
    _metric_vectors,
    _point_metrics,
    _stable_rank_matrix,
    _strict_json_object,
    joint_bipartite_component_bootstrap,
    paired_effect_bootstrap,
    verify_evaluation_and_artifact,
)
from sabdab2_retrieval_controls import _load_bundle, control_evaluation_identity


FORMAT_VERSION = "sabdab2_retrieval_control_comparison_v1"
EXECUTION_PROTOCOL_SHA256 = (
    "43638fdd17b9ce3e4cac8b83e42c16261ecbf79c2e628d0b494e5dcbe02f9fc9"
)
EXECUTION_HARDENING_PROTOCOL_SHA256 = (
    "b5b74452d3483d8b1fdf8c67387e3d38ad2be34a22b65e26a6cab71b0ca73078"
)
CONTROL_PROTOCOL_SHA256 = (
    "21c3a0727a1b9d6543a022f6f9d216414ee8f0fc99590b4febca5dcb0d8cdc02"
)
BOUNDARY_VERIFICATION_SHA256 = (
    "3bd8186603d908a99e1fbf00ead3697dd276ccd798b4f91bcbdda81441bd80e8"
)
LOCKED_BOOTSTRAP_COUNT = 10_000
LOCKED_BOOTSTRAP_SEED = 42
LOCKED_ALPHA = 0.05
GATE_NAME = "raw_random_head_length_composition_and_unary_controls_lower"
INFERENTIAL_METRICS = ("mrr", "recall_at_1")
CONTROL_COMPLETION_FORMAT_VERSION = (
    "sabdab2_control_two_boundary_suite_completion_v1"
)
CONTROL_BOUNDARY_MODES = (
    "concat_antibody_and_antigen",
    "concat_antibody_antigen_plus_vh_vl",
)
LOCKED_AUTHORITY_SHA256 = (
    "e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b"
)
LOCKED_PREPARED_INSTANCES_SHA256 = (
    "a9131d52323bae40572b7de81c5e41353ccc39a73309b7d7043485350facb009"
)
LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256 = (
    "548bcfab75923ec6621c1a3fef1aa6673c034f64a9bdc752e7ee15b41b4a4e63"
)
LOCKED_CONTROL_TRAINING_EMBEDDING_CONTENT_SHA256 = {
    "primary_antibody": (
        "9eb10c64fd05897162548a81871841724d9471ba5ed1c800b3d10a2c9229a45e"
    ),
    "shared_esm_antibody": (
        "03a9e0e4b4acf8b3b76cca651d65c4f278f5ac5a8985d3b12fe001da7809bf36"
    ),
    "esm_antigen": (
        "d4157971ba2e977d1125e896fe13147371292495fa04a47f62b32ebe8f4ac97e"
    ),
}
EXPECTED_FIXED_EVALUATION = {
    "split": "validation",
    "batch_size": 256,
    "bucket_width": 128,
    "recall_ks": [1, 5, 10, 20],
    "seed": 42,
    "deterministic": True,
    "device": "cuda",
    "isolation_policy": "double_exact",
    "score_artifact_required": True,
}


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one complete JSON file without ever replacing an existing target."""
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = target.parent / (
        f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            view = memoryview(serialized)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("Exclusive comparison output write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(temporary, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_json_bytes_object(data: bytes, label: str) -> dict[str, Any]:
    """Decode one finite JSON object while rejecting duplicate object keys."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains a non-finite JSON value: {value}")

    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    statistics_runtime._assert_finite_json(payload)
    return payload


def _read_strict_json_file(path: Path, label: str) -> tuple[dict[str, Any], str]:
    """Read, hash, and parse one JSON byte stream exactly once."""
    target = path.expanduser().resolve()
    try:
        data = target.read_bytes()
    except OSError as error:
        raise ValueError(f"Cannot read {label}: {target}") from error
    return _strict_json_bytes_object(data, label), hashlib.sha256(data).hexdigest()


def _authenticate_protocol(
    path: Path,
    supplied_sha256: str,
    locked_sha256: str,
    label: str,
) -> dict[str, str]:
    """Authenticate one immutable physical protocol document."""
    if supplied_sha256 != locked_sha256:
        raise ValueError(f"{label} SHA-256 differs from the locked digest")
    target = path.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"{label} is missing: {target}")
    observed = sha256_file(target)
    if observed != locked_sha256:
        raise ValueError(
            f"{label} physical SHA-256 mismatch: expected={locked_sha256}, "
            f"observed={observed}"
        )
    return {"path": str(target), "sha256": observed}


def _validate_control_arguments(
    raw_controls: Sequence[Sequence[str]],
) -> dict[str, tuple[Path, str, Path, str]]:
    """Require exactly one artifact pair for every locked control."""
    if len(raw_controls) != len(CONTROL_NAMES):
        raise ValueError(
            f"Exactly {len(CONTROL_NAMES)} --control records are required"
        )
    controls: dict[str, tuple[Path, str, Path, str]] = {}
    for raw in raw_controls:
        if len(raw) != 5:
            raise ValueError("Each --control record must contain exactly five values")
        name, evaluation, evaluation_sha256, score, score_sha256 = raw
        if name not in CONTROL_NAMES:
            raise ValueError(f"Unknown locked control: {name}")
        if name in controls:
            raise ValueError(f"Duplicate locked control: {name}")
        if not _is_sha256(evaluation_sha256) or not _is_sha256(score_sha256):
            raise ValueError(f"Control {name} requires exact SHA-256 values")
        controls[name] = (
            Path(evaluation).expanduser().resolve(),
            evaluation_sha256,
            Path(score).expanduser().resolve(),
            score_sha256,
        )
    if set(controls) != set(CONTROL_NAMES):
        missing = sorted(set(CONTROL_NAMES) - set(controls))
        raise ValueError(f"The locked control suite is incomplete: missing={missing}")
    return {name: controls[name] for name in CONTROL_NAMES}


def _authenticate_pair(
    evaluation_path: Path,
    evaluation_sha256: str,
    score_path: Path,
    score_sha256: str,
) -> dict[str, Any]:
    """Reject ambiguous JSON, then run the shared physical verification chain."""
    _strict_json_object(evaluation_path.expanduser().resolve(), "evaluation JSON")
    return verify_evaluation_and_artifact(
        evaluation_path,
        score_path,
        evaluation_sha256,
        score_sha256,
    )


def _directional_vectors(artifact: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Recompute standard per-anchor retrieval contributions in both directions."""
    similarities = np.asarray(artifact["similarities"], dtype=np.float32)
    positive_mask = np.asarray(artifact["positive_mask"], dtype=bool)
    directional = {
        "antibody_to_antigen": (similarities, positive_mask),
        "antigen_to_antibody": (similarities.T, positive_mask.T),
    }
    return {
        direction: _metric_vectors(
            _best_positive_ranks(_stable_rank_matrix(scores), mask)
        )
        for direction, (scores, mask) in directional.items()
    }


def _descriptive_metrics(
    vectors: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Summarize directional and equally weighted bidirectional point metrics."""
    result = {
        direction: _point_metrics(vectors[direction]) for direction in DIRECTIONS
    }
    result["bidirectional"] = {
        metric: 0.5
        * (
            result["antibody_to_antigen"][metric]
            + result["antigen_to_antibody"][metric]
        )
        for metric in METRIC_NAMES
    }
    return result


def _effect_vectors(
    primary: Mapping[str, Mapping[str, np.ndarray]],
    control: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Return paired primary-minus-control per-anchor contributions."""
    result: dict[str, dict[str, np.ndarray]] = {}
    for direction in DIRECTIONS:
        result[direction] = {}
        for metric in METRIC_NAMES:
            primary_values = np.asarray(primary[direction][metric], dtype=np.float64)
            control_values = np.asarray(control[direction][metric], dtype=np.float64)
            if primary_values.shape != control_values.shape:
                raise ValueError("Primary and control per-anchor vectors do not align")
            result[direction][metric] = primary_values - control_values
    return result


def _effect_vector_audit(
    effects: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Record compact hashes of paired vectors without emitting per-anchor scores."""
    return {
        direction: {
            metric: {
                "anchor_count": int(np.asarray(effects[direction][metric]).size),
                "dtype": "float64",
                "sha256": _score_array_sha256(
                    np.asarray(effects[direction][metric], dtype=np.float64)
                ),
            }
            for metric in METRIC_NAMES
        }
        for direction in DIRECTIONS
    }


def _descriptive_effect_points(
    effects: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Return point effects for every standard metric without extra inference."""
    result = {
        direction: {
            metric: float(np.asarray(effects[direction][metric]).mean())
            for metric in METRIC_NAMES
        }
        for direction in DIRECTIONS
    }
    result["bidirectional"] = {
        metric: 0.5
        * (
            result["antibody_to_antigen"][metric]
            + result["antigen_to_antibody"][metric]
        )
        for metric in METRIC_NAMES
    }
    return result


def _inferential_bootstrap_view(report: Mapping[str, Any]) -> dict[str, Any]:
    """Expose intervals only for the two predeclared paired-effect metrics."""
    result = {
        key: value
        for key, value in report.items()
        if key not in {*DIRECTIONS, "bidirectional"}
    }
    if report.get("available") is False:
        return result
    for direction in (*DIRECTIONS, "bidirectional"):
        result[direction] = {
            metric: report[direction][metric] for metric in INFERENTIAL_METRICS
        }
    return result


def _comparison_fingerprint(verified: Mapping[str, Any]) -> dict[str, Any]:
    """Extract every identity field that must match across all six evaluations."""
    evaluation = verified["evaluation"]
    dataset = evaluation["dataset"]
    authority = dataset["authority"]
    artifact = verified["artifact"]
    return {
        "split_manifest_sha256": dataset["split_manifest_sha256"],
        "split_format_version": dataset["split_format_version"],
        "split_intended_use": dataset["split_intended_use"],
        "strict_nw_constraint_mode": dataset["strict_nw_constraint_mode"],
        "instances_sha256": dataset["instances_sha256"],
        "split_membership_sha256": dataset["split_membership_sha256"],
        "raw_instance_counts": dataset["raw_instance_counts"],
        "authority_split_sha256": authority["sha256"],
        "antibody_keys": list(artifact["antibody_keys"]),
        "antigen_keys": list(artifact["antigen_keys"]),
        "positive_mask_sha256": artifact["array_sha256"]["positive_mask"],
        "positive_edge_count": int(np.asarray(artifact["positive_mask"]).sum()),
    }


def _assert_distinct_inputs(
    primary: tuple[Path, str, Path, str],
    controls: Mapping[str, tuple[Path, str, Path, str]],
) -> None:
    """Prevent one physical artifact pair from being counted more than once."""
    evaluation_paths = [primary[0], *(value[0] for value in controls.values())]
    score_paths = [primary[2], *(value[2] for value in controls.values())]
    if len(set(evaluation_paths)) != 1 + len(CONTROL_NAMES):
        raise ValueError("All six evaluation JSON paths must be distinct")
    if len(set(score_paths)) != 1 + len(CONTROL_NAMES):
        raise ValueError("All six score-artifact paths must be distinct")
    if set(evaluation_paths) & set(score_paths):
        raise ValueError("Evaluation and score-artifact paths must not overlap")


def _protocol_reference_matches(
    reference: Any,
    protocol: Mapping[str, str],
) -> bool:
    """Check one embedded physical protocol reference exactly."""
    return isinstance(reference, Mapping) and dict(reference) == dict(protocol)


def _hashed_reference(value: Any, label: str) -> dict[str, str]:
    """Physically authenticate one exact path-and-SHA reference."""
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path-and-SHA object")
    path_value = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path_value, str) or not Path(path_value).expanduser().is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    if not isinstance(digest, str) or not _is_sha256(digest):
        raise ValueError(f"{label}.sha256 must be an exact SHA-256")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != digest:
        raise ValueError(
            f"{label} physical SHA-256 mismatch: expected={digest}, observed={observed}"
        )
    return {"path": str(path), "sha256": digest}


def _hashed_json_reference(
    value: Any,
    label: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Authenticate and strictly parse the same referenced JSON byte stream."""
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path-and-SHA object")
    path_value = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path_value, str) or not Path(path_value).expanduser().is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    if not isinstance(digest, str) or not _is_sha256(digest):
        raise ValueError(f"{label}.sha256 must be an exact SHA-256")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload, observed = _read_strict_json_file(path, label)
    if observed != digest:
        raise ValueError(
            f"{label} physical SHA-256 mismatch: expected={digest}, observed={observed}"
        )
    return {"path": str(path), "sha256": digest}, payload


def _authenticate_checkpoint_suite_claim(
    verified: Mapping[str, Any],
    submission_manifest: Mapping[str, str],
    manifest: Mapping[str, Any],
    selected_boundary: str,
) -> dict[str, Any]:
    """Authenticate the immutable one-checkpoint, two-boundary suite claim."""
    evaluation = verified["evaluation"]
    claim_reference, claim = _hashed_json_reference(
        evaluation.get("checkpoint_suite_claim"), "Checkpoint-suite claim"
    )
    ledger_reference = evaluation.get("one_shot_ledger")
    if not isinstance(ledger_reference, Mapping):
        raise ValueError("Primary evaluation lacks its one-shot ledger reference")
    registry_value = ledger_reference.get("registry_dir")
    if not isinstance(registry_value, str) or not Path(registry_value).expanduser().is_absolute():
        raise ValueError("Primary one-shot ledger registry is not absolute")
    registry = Path(registry_value).expanduser().resolve()
    expected_claim_path = (
        registry
        / "checkpoint-suite-claims"
        / f"{submission_manifest['sha256']}.json"
    ).resolve()
    if Path(claim_reference["path"]) != expected_claim_path:
        raise ValueError("Checkpoint-suite claim path is not canonical")

    expected_keys = {
        "format_version",
        "stage",
        "intended_use",
        "status",
        "submission_manifest",
        "checkpoint",
        "one_shot_ledger_registry_dir",
        "boundaries",
    }
    if set(claim) != expected_keys:
        raise ValueError("Checkpoint-suite claim does not have the exact v1 schema")
    if (
        claim.get("format_version") != "sabdab2_final_checkpoint_suite_claim_v1"
        or claim.get("stage") != "final_checkpoint_suite_claim"
        or claim.get("intended_use")
        != "two_boundary_one_shot_development_confirmation"
        or claim.get("status") != "immutable"
        or claim.get("submission_manifest") != dict(submission_manifest)
        or claim.get("one_shot_ledger_registry_dir") != str(registry)
        or manifest.get("one_shot_ledger_registry_dir") != str(registry)
    ):
        raise ValueError("Checkpoint-suite claim violates its immutable identity")

    checkpoint = claim.get("checkpoint")
    evaluation_checkpoint = evaluation.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != {"path", "sha256"}
        or not isinstance(evaluation_checkpoint, str)
        or not Path(evaluation_checkpoint).expanduser().is_absolute()
        or Path(str(checkpoint.get("path", ""))).expanduser().resolve()
        != Path(evaluation_checkpoint).expanduser().resolve()
        or checkpoint.get("sha256") != evaluation.get("checkpoint_sha256")
        or not _is_sha256(str(checkpoint.get("sha256", "")))
    ):
        raise ValueError("Checkpoint-suite claim and primary checkpoint disagree")

    manifest_boundaries = manifest.get("boundaries")
    if not isinstance(manifest_boundaries, Mapping) or len(manifest_boundaries) != 2:
        raise ValueError("Submission manifest does not declare exactly two boundaries")
    expected_boundaries: dict[str, dict[str, str]] = {}
    for name, boundary in manifest_boundaries.items():
        split_reference = boundary.get("split_manifest") if isinstance(boundary, Mapping) else None
        mode = boundary.get("strict_nw_constraint_mode") if isinstance(boundary, Mapping) else None
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(split_reference, Mapping)
            or not _is_sha256(str(split_reference.get("sha256", "")))
            or not isinstance(mode, str)
            or not mode
        ):
            raise ValueError("Submission manifest has a malformed boundary contract")
        expected_boundaries[name] = {
            "split_manifest_sha256": str(split_reference["sha256"]),
            "strict_nw_constraint_mode": mode,
        }
    if claim.get("boundaries") != expected_boundaries:
        raise ValueError("Checkpoint-suite claim does not bind both manifest boundaries")
    dataset = evaluation["dataset"]
    if expected_boundaries.get(selected_boundary) != {
        "split_manifest_sha256": dataset.get("split_manifest_sha256"),
        "strict_nw_constraint_mode": dataset.get("strict_nw_constraint_mode"),
    }:
        raise ValueError("Checkpoint-suite claim does not bind the evaluated boundary")
    context = verified["artifact"].get("metadata", {}).get("contract", {}).get(
        "evaluation_context"
    )
    if not isinstance(context, Mapping) or context.get(
        "checkpoint_suite_claim_sha256"
    ) != claim_reference["sha256"]:
        raise ValueError("Primary score contract does not bind the checkpoint-suite claim")
    return {
        "path": claim_reference["path"],
        "sha256": claim_reference["sha256"],
        "format_version": claim["format_version"],
        "status": claim["status"],
        "checkpoint": dict(checkpoint),
        "one_shot_ledger_registry_dir": str(registry),
        "boundaries": expected_boundaries,
        "selected_boundary": selected_boundary,
        "checkpoint_bytes_loaded_by_comparator": False,
    }


def _authenticate_primary_suite_completion(
    verified: Mapping[str, Any],
    submission_manifest: Mapping[str, str],
    manifest: Mapping[str, Any],
    claim_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate completion of both mandatory primary boundary evaluations."""
    registry = Path(
        str(claim_verification["one_shot_ledger_registry_dir"])
    ).resolve()
    receipt_path = (
        registry
        / "checkpoint-suite-completions"
        / f"{submission_manifest['sha256']}.json"
    ).resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Primary suite-completion receipt is missing: {receipt_path}")
    receipt, receipt_sha256 = _read_strict_json_file(
        receipt_path,
        "primary suite-completion receipt",
    )
    expected_keys = {
        "format_version",
        "stage",
        "intended_use",
        "status",
        "checkpoint_suite_claim",
        "submission_manifest",
        "checkpoint",
        "one_shot_ledger_registry_dir",
        "boundaries",
    }
    if set(receipt) != expected_keys:
        raise ValueError("Primary suite-completion receipt has the wrong exact schema")
    expected_claim_reference = {
        "path": claim_verification["path"],
        "sha256": claim_verification["sha256"],
    }
    if (
        receipt.get("format_version")
        != "sabdab2_final_checkpoint_suite_completion_v1"
        or receipt.get("stage") != "final_checkpoint_suite_completion"
        or receipt.get("intended_use")
        != "two_boundary_one_shot_development_confirmation"
        or receipt.get("status") != "completed"
        or receipt.get("checkpoint_suite_claim") != expected_claim_reference
        or receipt.get("submission_manifest") != dict(submission_manifest)
        or receipt.get("checkpoint") != claim_verification.get("checkpoint")
        or receipt.get("one_shot_ledger_registry_dir") != str(registry)
    ):
        raise ValueError("Primary suite-completion receipt violates its identity")

    completion_boundaries = receipt.get("boundaries")
    claim_boundaries = claim_verification["boundaries"]
    manifest_boundaries = manifest["boundaries"]
    if (
        not isinstance(completion_boundaries, Mapping)
        or set(completion_boundaries) != set(claim_boundaries)
        or len(completion_boundaries) != 2
    ):
        raise ValueError("Primary completion does not contain both claimed boundaries")
    authenticated_boundaries: dict[str, Any] = {}
    current_match_count = 0
    for name in sorted(completion_boundaries):
        record = completion_boundaries[name]
        expected_record_keys = {
            "split_manifest_sha256",
            "strict_nw_constraint_mode",
            "one_shot_ledger",
            "evaluation_json",
            "score_artifact",
        }
        if not isinstance(record, Mapping) or set(record) != expected_record_keys:
            raise ValueError(f"Primary completion boundary {name!r} is malformed")
        if {
            "split_manifest_sha256": record["split_manifest_sha256"],
            "strict_nw_constraint_mode": record["strict_nw_constraint_mode"],
        } != claim_boundaries[name]:
            raise ValueError(f"Primary completion boundary {name!r} differs from its claim")
        ledger_reference, completed_ledger = _hashed_json_reference(
            record["one_shot_ledger"], f"Primary completion ledger {name}"
        )
        evaluation_reference, completed_evaluation = _hashed_json_reference(
            record["evaluation_json"], f"Primary completion evaluation {name}"
        )
        score_reference = _hashed_reference(
            record["score_artifact"], f"Primary completion score artifact {name}"
        )
        completed_score = statistics_runtime.load_score_artifact(
            Path(score_reference["path"])
        )
        manifest_boundary = manifest_boundaries[name]
        if (
            Path(evaluation_reference["path"])
            != Path(str(manifest_boundary["evaluation_output"])).expanduser().resolve()
            or Path(score_reference["path"])
            != Path(str(manifest_boundary["score_artifact"])).expanduser().resolve()
        ):
            raise ValueError(f"Primary completion boundary {name!r} changed output paths")
        completed_evaluation_ledger = completed_evaluation.get("one_shot_ledger")
        completed_score_reference = completed_evaluation.get("score_artifact")
        completed_score_context = completed_score.get("metadata", {}).get(
            "contract", {}
        ).get("evaluation_context")
        expected_score_context = {
            "submission_manifest_sha256": submission_manifest["sha256"],
            "protocol_v1_5_execution_lock_sha256": manifest["upstream_evidence"][
                "protocol_v1_5_execution_lock"
            ]["sha256"],
            "protocol_v1_6_execution_hardening_sha256": manifest[
                "upstream_evidence"
            ]["protocol_v1_6_execution_hardening"]["sha256"],
            "boundary_verification_v2_sha256": manifest["upstream_evidence"][
                "boundary_verification_v2"
            ]["sha256"],
            "fixed_evaluation_sha256": canonical_sha256(manifest["fixed_evaluation"]),
            "selected_boundary": name,
            "checkpoint_suite_claim_sha256": claim_verification["sha256"],
        }
        if (
            completed_evaluation.get("checkpoint_sha256")
            != claim_verification["checkpoint"]["sha256"]
            or completed_evaluation.get("checkpoint_suite_claim")
            != expected_claim_reference
            or completed_evaluation.get("predeclared_evaluation_boundary") != name
            or not isinstance(completed_evaluation_ledger, Mapping)
            or Path(
                str(completed_evaluation_ledger.get("canonical_path", ""))
            ).expanduser().resolve()
            != Path(ledger_reference["path"])
            or completed_score.get("sha256") != score_reference["sha256"]
            or not isinstance(completed_score_reference, Mapping)
            or completed_score_reference.get("sha256") != score_reference["sha256"]
            or completed_score_reference.get("contract_sha256")
            != completed_score.get("metadata", {}).get("contract_sha256")
            or not isinstance(completed_score_context, Mapping)
            or any(
                completed_score_context.get(key) != value
                for key, value in expected_score_context.items()
            )
            or completed_ledger.get("status") != "completed"
            or completed_ledger.get("checkpoint_suite_claim")
            != expected_claim_reference
            or completed_ledger.get("evaluation_json_sha256")
            != evaluation_reference["sha256"]
            or completed_ledger.get("score_artifact_sha256")
            != score_reference["sha256"]
        ):
            raise ValueError(f"Primary completion boundary {name!r} has inconsistent outputs")
        if (
            evaluation_reference["path"] == str(verified["evaluation_path"])
            and evaluation_reference["sha256"] == verified["evaluation_sha256"]
            and score_reference["path"] == str(verified["artifact"]["path"])
            and score_reference["sha256"] == verified["artifact"]["sha256"]
        ):
            current_ledger = verified.get("one_shot_verification")
            if (
                not isinstance(current_ledger, Mapping)
                or current_ledger.get("path") != ledger_reference["path"]
                or current_ledger.get("sha256") != ledger_reference["sha256"]
            ):
                raise ValueError("Primary completion changed the authenticated central ledger")
            current_match_count += 1
        authenticated_boundaries[name] = {
            "split_manifest_sha256": record["split_manifest_sha256"],
            "strict_nw_constraint_mode": record["strict_nw_constraint_mode"],
            "one_shot_ledger": ledger_reference,
            "evaluation_json": evaluation_reference,
            "score_artifact": score_reference,
        }
    if current_match_count != 1:
        raise ValueError("Primary comparator inputs do not match exactly one completed boundary")
    return {
        "path": str(receipt_path),
        "sha256": receipt_sha256,
        "format_version": receipt["format_version"],
        "status": receipt["status"],
        "boundaries": authenticated_boundaries,
        "both_mandatory_boundaries_completed": True,
    }


def _authenticate_control_bundle(
    bundle_reference: Mapping[str, str],
    bundle_config_sha256: str,
    control_protocol: Mapping[str, str],
    execution_hardening_protocol: Mapping[str, str],
) -> dict[str, Any]:
    """Parse one control bundle and authenticate its complete fit provenance."""
    loaded_bundle = _load_bundle(
        Path(bundle_reference["path"]),
        bundle_reference["sha256"],
    )
    bundle_config = loaded_bundle.get("config")
    if not isinstance(bundle_config, Mapping):
        raise ValueError("Completed control bundle lacks its configuration")
    unhashed_bundle_config = {
        key: value for key, value in bundle_config.items() if key != "config_sha256"
    }
    if (
        bundle_config.get("config_sha256") != bundle_config_sha256
        or bundle_config_sha256 != canonical_sha256(unhashed_bundle_config)
        or bundle_config.get("protocol_version") != CONTROL_SUITE_VERSION
        or bundle_config.get("control_protocol_v1") != dict(control_protocol)
        or bundle_config.get("execution_protocol_v1_6")
        != dict(execution_hardening_protocol)
        or bundle_config.get("source_sha256") != controls_runtime._source_hashes()
        or bundle_config.get("validation_rows_materialized") is not False
        or bundle_config.get("frozen_test_used") is not False
    ):
        raise ValueError("Completed control bundle violates its authenticated provenance")
    return {
        "path": str(loaded_bundle["path"]),
        "sha256": loaded_bundle["sha256"],
        "config_sha256": bundle_config_sha256,
        "control_protocol_v1": dict(control_protocol),
        "execution_protocol_v1_6": dict(execution_hardening_protocol),
        "source_sha256": dict(bundle_config["source_sha256"]),
    }


def _authenticate_control_suite_completion(
    verified_controls: Mapping[str, Mapping[str, Any]],
    execution_hardening_protocol: Mapping[str, str],
    control_protocol: Mapping[str, str],
) -> dict[str, Any]:
    """Authenticate completion of both mandatory five-control boundary suites."""
    first = verified_controls[CONTROL_NAMES[0]]
    first_evaluation = first["evaluation"]
    bundle_sha256 = str(first_evaluation.get("checkpoint_sha256", ""))
    if not _is_sha256(bundle_sha256):
        raise ValueError("Control suite lacks an exact bundle SHA-256")
    ledger_reference = first_evaluation.get("one_shot_ledger")
    if not isinstance(ledger_reference, Mapping):
        raise ValueError("Control suite lacks its central-ledger reference")
    registry_value = ledger_reference.get("registry_dir")
    if not isinstance(registry_value, str) or not Path(registry_value).expanduser().is_absolute():
        raise ValueError("Control suite ledger registry is not absolute")
    registry = Path(registry_value).expanduser().resolve()
    receipt_path = (registry / "suite-completions" / f"{bundle_sha256}.json").resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Control suite-completion receipt is missing: {receipt_path}")
    receipt, receipt_sha256 = _read_strict_json_file(
        receipt_path,
        "control suite-completion receipt",
    )
    expected_keys = {
        "format_version",
        "status",
        "control_suite_version",
        "completion_identity_sha256",
        "control_bundle",
        "control_protocol_v1",
        "execution_protocol_v1_6",
        "boundary_verification_v2",
        "authority_split",
        "immutable_inputs",
        "required_boundary_modes",
        "boundaries",
        "frozen_test_used",
    }
    if set(receipt) != expected_keys:
        raise ValueError("Control suite-completion receipt has the wrong exact schema")
    unhashed_receipt = dict(receipt)
    completion_identity = unhashed_receipt.pop("completion_identity_sha256", None)
    if (
        receipt.get("format_version") != CONTROL_COMPLETION_FORMAT_VERSION
        or receipt.get("status") != "completed"
        or receipt.get("control_suite_version") != CONTROL_SUITE_VERSION
        or receipt.get("frozen_test_used") is not False
        or completion_identity != canonical_sha256(unhashed_receipt)
        or receipt.get("required_boundary_modes") != list(CONTROL_BOUNDARY_MODES)
    ):
        raise ValueError("Control suite-completion receipt violates its identity")

    control_protocol_reference = _hashed_reference(
        receipt.get("control_protocol_v1"),
        "Control completion controls protocol v1",
    )
    if control_protocol_reference != dict(control_protocol):
        raise ValueError("Control completion uses a different controls protocol v1")

    bundle = receipt.get("control_bundle")
    if not isinstance(bundle, Mapping) or set(bundle) != {"path", "sha256", "config_sha256"}:
        raise ValueError("Control completion has a malformed bundle reference")
    bundle_reference = _hashed_reference(
        {"path": bundle.get("path"), "sha256": bundle.get("sha256")},
        "Completed control bundle",
    )
    bundle_config_sha256 = str(bundle.get("config_sha256", ""))
    if (
        bundle_reference["sha256"] != bundle_sha256
        or Path(bundle_reference["path"])
        != Path(str(first_evaluation.get("checkpoint", ""))).expanduser().resolve()
        or not _is_sha256(bundle_config_sha256)
    ):
        raise ValueError("Control completion bundle differs from comparator inputs")
    bundle_verification = _authenticate_control_bundle(
        bundle_reference,
        bundle_config_sha256,
        control_protocol_reference,
        execution_hardening_protocol,
    )
    for name in CONTROL_NAMES:
        evaluation = verified_controls[name]["evaluation"]
        control = evaluation.get("control")
        if (
            evaluation.get("checkpoint_sha256") != bundle_sha256
            or not isinstance(control, Mapping)
            or control.get("bundle_config_sha256") != bundle_config_sha256
        ):
            raise ValueError("Current controls do not share the completed control bundle")

    protocol_reference = _hashed_reference(
        receipt.get("execution_protocol_v1_6"), "Control completion protocol v1.6"
    )
    if protocol_reference != dict(execution_hardening_protocol):
        raise ValueError("Control completion uses a different protocol v1.6")
    boundary_verification = _hashed_reference(
        receipt.get("boundary_verification_v2"),
        "Control completion boundary verification",
    )
    authority = _hashed_reference(
        receipt.get("authority_split"), "Control completion authority split"
    )
    if (
        boundary_verification["sha256"] != BOUNDARY_VERIFICATION_SHA256
        or authority["sha256"] != LOCKED_AUTHORITY_SHA256
    ):
        raise ValueError("Control completion changed locked authority evidence")

    immutable = receipt.get("immutable_inputs")
    if not isinstance(immutable, Mapping) or set(immutable) != {
        "prepared_instances",
        "canonical_sequence_manifest",
        "training_embedding_content_manifest_sha256",
    }:
        raise ValueError("Control completion immutable-input contract is malformed")
    prepared = _hashed_reference(
        immutable["prepared_instances"], "Control completion prepared instances"
    )
    canonical_manifest = _hashed_reference(
        immutable["canonical_sequence_manifest"],
        "Control completion canonical sequence manifest",
    )
    if (
        prepared["sha256"] != LOCKED_PREPARED_INSTANCES_SHA256
        or canonical_manifest["sha256"]
        != LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256
        or immutable.get("training_embedding_content_manifest_sha256")
        != LOCKED_CONTROL_TRAINING_EMBEDDING_CONTENT_SHA256
    ):
        raise ValueError("Control completion changed immutable v1.6 inputs")

    boundaries = receipt.get("boundaries")
    if not isinstance(boundaries, Mapping) or set(boundaries) != set(CONTROL_BOUNDARY_MODES):
        raise ValueError("Control completion does not contain both mandatory modes")
    authenticated_boundaries: dict[str, Any] = {}
    current_mode_matches = 0
    seen_paths: set[str] = set()
    for mode in CONTROL_BOUNDARY_MODES:
        boundary = boundaries[mode]
        expected_boundary_keys = {
            "split_manifest",
            "boundary_scientific_payload_sha256",
            "ledger",
            "output_dir",
            "controls",
        }
        if not isinstance(boundary, Mapping) or set(boundary) != expected_boundary_keys:
            raise ValueError(f"Control completion boundary {mode!r} is malformed")
        split_reference, split_payload = _hashed_json_reference(
            boundary["split_manifest"], f"Control completion split {mode}"
        )
        if (
            split_payload.get("constraint_mode") != mode
            or split_payload.get("scientific_payload_sha256")
            != boundary.get("boundary_scientific_payload_sha256")
        ):
            raise ValueError(f"Control completion split {mode!r} is inconsistent")
        raw_ledger = boundary.get("ledger")
        if not isinstance(raw_ledger, Mapping) or set(raw_ledger) != {
            "path",
            "sha256",
            "canonical_key",
        }:
            raise ValueError(f"Control completion ledger {mode!r} is malformed")
        ledger, ledger_payload = _hashed_json_reference(
            {"path": raw_ledger["path"], "sha256": raw_ledger["sha256"]},
            f"Control completion ledger {mode}",
        )
        expected_ledger_identity, expected_ledger_contract = control_evaluation_identity(
            bundle_sha256,
            split_reference["sha256"],
            authority["sha256"],
        )
        expected_ledger_keys = {
            *expected_ledger_contract,
            "evaluation_identity_sha256",
            "status",
            "requested_output_dir",
            "control_protocol_v1",
            "execution_protocol_v1_6",
            "locked_submission_evidence_sha256",
            "control_outputs",
        }
        if (
            set(ledger_payload) != expected_ledger_keys
            or ledger_payload.get("status") != "completed"
            or ledger_payload.get("evaluation_identity_sha256")
            != expected_ledger_identity
            or raw_ledger["canonical_key"] != expected_ledger_identity
            or Path(ledger["path"]) != registry / f"{expected_ledger_identity}.json"
            or any(
                ledger_payload.get(key) != value
                for key, value in expected_ledger_contract.items()
            )
            or ledger_payload.get("control_bundle_sha256") != bundle_sha256
            or ledger_payload.get("split_manifest_sha256")
            != split_reference["sha256"]
            or ledger_payload.get("control_protocol_v1")
            != control_protocol_reference
            or ledger_payload.get("execution_protocol_v1_6") != protocol_reference
        ):
            raise ValueError(f"Control completion ledger {mode!r} is inconsistent")
        output_value = boundary.get("output_dir")
        if not isinstance(output_value, str) or not Path(output_value).expanduser().is_absolute():
            raise ValueError(f"Control completion output directory {mode!r} is not absolute")
        output_dir = Path(output_value).expanduser().resolve()
        if (
            not output_dir.is_dir()
            or Path(str(ledger_payload.get("requested_output_dir", ""))).expanduser().resolve()
            != output_dir
            or output_dir == registry
            or registry in output_dir.parents
        ):
            raise ValueError(f"Control completion output directory {mode!r} is inconsistent")
        raw_controls = boundary.get("controls")
        if not isinstance(raw_controls, Mapping) or set(raw_controls) != set(CONTROL_NAMES):
            raise ValueError(f"Control completion boundary {mode!r} lacks the full suite")
        ledger_outputs = ledger_payload.get("control_outputs")
        if not isinstance(ledger_outputs, Mapping) or set(ledger_outputs) != set(CONTROL_NAMES):
            raise ValueError(f"Completed control ledger {mode!r} lacks the full suite")
        authenticated_controls: dict[str, Any] = {}
        current_control_matches = 0
        for name in CONTROL_NAMES:
            control_record = raw_controls[name]
            if not isinstance(control_record, Mapping) or set(control_record) != {
                "evaluation_json",
                "score_artifact",
            }:
                raise ValueError(f"Control completion output {mode}/{name} is malformed")
            evaluation_reference, completed_evaluation = _hashed_json_reference(
                control_record["evaluation_json"],
                f"Control completion evaluation {mode}/{name}",
            )
            raw_score = control_record["score_artifact"]
            if not isinstance(raw_score, Mapping) or set(raw_score) != {
                "path",
                "sha256",
                "contract_sha256",
            }:
                raise ValueError(f"Control completion score {mode}/{name} is malformed")
            score_reference = _hashed_reference(
                {"path": raw_score["path"], "sha256": raw_score["sha256"]},
                f"Control completion score {mode}/{name}",
            )
            if (
                Path(evaluation_reference["path"]).parent != output_dir
                or Path(score_reference["path"]).parent != output_dir
                or evaluation_reference["path"] in seen_paths
                or score_reference["path"] in seen_paths
            ):
                raise ValueError("Control completion output paths overlap or escape their suite")
            seen_paths.update((evaluation_reference["path"], score_reference["path"]))
            completed_score = statistics_runtime.load_score_artifact(
                Path(score_reference["path"])
            )
            score_contract = completed_evaluation.get("score_artifact")
            control_identity = completed_evaluation.get("control")
            completed_ledger_reference = completed_evaluation.get("one_shot_ledger")
            locked_evidence = completed_evaluation.get("locked_submission_evidence")
            ledger_output = ledger_outputs[name]
            expected_locked_evidence_keys = {
                "format_version",
                "control_protocol_v1",
                "execution_protocol_v1_6",
                "prepared_instances",
                "canonical_sequence_manifest",
                "training_embedding_content_manifest_sha256",
                "boundary_verification_v2",
                "authority_split",
                "selected_boundary",
                "required_boundary_modes",
                "control_suite_completion_required",
                "frozen_test_used",
            }
            score_context = completed_score.get("metadata", {}).get(
                "contract", {}
            ).get("evaluation_context")
            if (
                completed_evaluation.get("checkpoint_sha256") != bundle_sha256
                or completed_evaluation.get("control_bundle_sha256") != bundle_sha256
                or completed_evaluation.get("control_protocol_v1")
                != control_protocol_reference
                or completed_evaluation.get("execution_protocol_v1_6")
                != protocol_reference
                or not isinstance(locked_evidence, Mapping)
                or set(locked_evidence) != expected_locked_evidence_keys
                or locked_evidence.get("control_protocol_v1")
                != control_protocol_reference
                or locked_evidence.get("execution_protocol_v1_6")
                != protocol_reference
                or locked_evidence.get("prepared_instances") != prepared
                or locked_evidence.get("canonical_sequence_manifest")
                != canonical_manifest
                or locked_evidence.get("boundary_verification_v2")
                != boundary_verification
                or locked_evidence.get("authority_split") != authority
                or locked_evidence.get(
                    "training_embedding_content_manifest_sha256"
                )
                != LOCKED_CONTROL_TRAINING_EMBEDDING_CONTENT_SHA256
                or locked_evidence.get("selected_boundary")
                != {
                    "constraint_mode": mode,
                    "path": split_reference["path"],
                    "sha256": split_reference["sha256"],
                    "scientific_payload_sha256": boundary[
                        "boundary_scientific_payload_sha256"
                    ],
                }
                or locked_evidence.get("required_boundary_modes")
                != list(CONTROL_BOUNDARY_MODES)
                or locked_evidence.get("control_suite_completion_required") is not True
                or locked_evidence.get("frozen_test_used") is not False
                or completed_evaluation.get("dataset", {}).get(
                    "split_manifest_sha256"
                )
                != split_reference["sha256"]
                or completed_evaluation.get("dataset", {}).get(
                    "strict_nw_constraint_mode"
                )
                != mode
                or not isinstance(control_identity, Mapping)
                or control_identity.get("name") != name
                or not isinstance(completed_ledger_reference, Mapping)
                or completed_ledger_reference.get("canonical_key")
                != raw_ledger["canonical_key"]
                or Path(str(completed_ledger_reference.get("canonical_path", ""))).expanduser().resolve()
                != Path(ledger["path"])
                or not isinstance(score_contract, Mapping)
                or score_contract.get("sha256") != score_reference["sha256"]
                or score_contract.get("contract_sha256")
                != raw_score["contract_sha256"]
                or completed_score.get("sha256") != score_reference["sha256"]
                or completed_score.get("metadata", {}).get("contract_sha256")
                != raw_score["contract_sha256"]
                or not isinstance(score_context, Mapping)
                or score_context.get("control_protocol_v1")
                != control_protocol_reference
                or score_context.get("control_protocol_v1_sha256")
                != control_protocol_reference["sha256"]
                or not isinstance(ledger_output, Mapping)
                or set(ledger_output)
                != {
                    "evaluation_json",
                    "evaluation_json_sha256",
                    "score_artifact",
                    "score_artifact_sha256",
                    "score_contract_sha256",
                    "bidirectional_mrr",
                }
                or Path(str(ledger_output.get("evaluation_json", ""))).expanduser().resolve()
                != Path(evaluation_reference["path"])
                or ledger_output.get("evaluation_json_sha256")
                != evaluation_reference["sha256"]
                or Path(str(ledger_output.get("score_artifact", ""))).expanduser().resolve()
                != Path(score_reference["path"])
                or ledger_output.get("score_artifact_sha256")
                != score_reference["sha256"]
                or ledger_output.get("score_contract_sha256")
                != raw_score["contract_sha256"]
            ):
                raise ValueError(f"Control completion output {mode}/{name} is inconsistent")
            current = verified_controls[name]
            if (
                evaluation_reference["path"] == str(current["evaluation_path"])
                and evaluation_reference["sha256"] == current["evaluation_sha256"]
                and score_reference["path"] == str(current["artifact"]["path"])
                and score_reference["sha256"] == current["artifact"]["sha256"]
            ):
                current_control_matches += 1
            authenticated_controls[name] = {
                "evaluation_json": evaluation_reference,
                "score_artifact": {
                    **score_reference,
                    "contract_sha256": raw_score["contract_sha256"],
                },
            }
        if current_control_matches == len(CONTROL_NAMES):
            for name in CONTROL_NAMES:
                authenticated_ledger = verified_controls[name].get(
                    "one_shot_verification"
                )
                if (
                    not isinstance(authenticated_ledger, Mapping)
                    or authenticated_ledger.get("path") != ledger["path"]
                    or authenticated_ledger.get("sha256") != ledger["sha256"]
                ):
                    raise ValueError(
                        "Control completion changed the authenticated central ledger"
                    )
            current_mode_matches += 1
        elif current_control_matches != 0:
            raise ValueError("Current controls span multiple completion boundaries")
        authenticated_boundaries[mode] = {
            "split_manifest": split_reference,
            "ledger": {**ledger, "canonical_key": raw_ledger["canonical_key"]},
            "output_dir": str(output_dir),
            "controls": authenticated_controls,
        }
    if current_mode_matches != 1:
        raise ValueError("Current controls do not match exactly one completed boundary suite")
    return {
        "path": str(receipt_path),
        "sha256": receipt_sha256,
        "format_version": receipt["format_version"],
        "status": receipt["status"],
        "completion_identity_sha256": completion_identity,
        "control_bundle": {
            **bundle_verification,
        },
        "control_protocol_v1": control_protocol_reference,
        "boundaries": authenticated_boundaries,
        "both_mandatory_boundaries_completed": True,
        "all_five_controls_completed_per_boundary": True,
    }


def _authenticate_locked_submission_evidence(
    verified: Mapping[str, Any],
    execution_protocol: Mapping[str, str],
) -> dict[str, Any]:
    """Bind the primary artifacts to the physically locked final-submission chain."""
    evaluation = verified["evaluation"]
    raw = evaluation.get("locked_submission_evidence")
    expected_keys = {
        "submission_manifest",
        "protocol_v1_5_execution_lock",
        "protocol_v1_6_execution_hardening",
        "boundary_verification_v2",
        "fixed_evaluation",
        "selected_boundary",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise PermissionError(
            "Primary evaluation lacks the exact locked-submission evidence contract"
        )
    submission_manifest, manifest = _hashed_json_reference(
        raw["submission_manifest"], "Final-refit submission manifest"
    )
    protocol = _hashed_reference(
        raw["protocol_v1_5_execution_lock"], "Protocol v1.5 submission evidence"
    )
    if protocol != dict(execution_protocol):
        raise ValueError("Primary submission evidence uses a different protocol v1.5")
    hardening_protocol = _hashed_reference(
        raw["protocol_v1_6_execution_hardening"],
        "Protocol v1.6 submission evidence",
    )
    if hardening_protocol["sha256"] != EXECUTION_HARDENING_PROTOCOL_SHA256:
        raise ValueError("Primary submission evidence uses a different protocol v1.6")
    boundary_verification = _hashed_reference(
        raw["boundary_verification_v2"], "Independent boundary verification v2"
    )
    if boundary_verification["sha256"] != BOUNDARY_VERIFICATION_SHA256:
        raise ValueError("Primary submission evidence uses an unlocked boundary verification")
    fixed_evaluation = raw["fixed_evaluation"]
    if fixed_evaluation != EXPECTED_FIXED_EVALUATION:
        raise ValueError("Primary submission evidence changed the fixed evaluation contract")
    selected_boundary = raw["selected_boundary"]
    if (
        not isinstance(selected_boundary, str)
        or not selected_boundary
        or evaluation.get("predeclared_evaluation_boundary") != selected_boundary
    ):
        raise ValueError("Primary selected-boundary evidence is missing or inconsistent")

    if (
        manifest.get("format_version")
        != "sabdab2_asymmetric_final_refit_submission_manifest_v1"
        or manifest.get("intended_use") != "development_only"
        or manifest.get("frozen_test_used") is not False
        or manifest.get("fixed_evaluation") != EXPECTED_FIXED_EVALUATION
        or manifest.get("canonical_refit_boundary") not in manifest.get("boundaries", {})
        or selected_boundary not in manifest.get("boundaries", {})
    ):
        raise ValueError("Physical submission manifest violates the final-refit contract")
    upstream = manifest.get("upstream_evidence")
    if (
        not isinstance(upstream, Mapping)
        or upstream.get("protocol_v1_5_execution_lock") != protocol
        or upstream.get("protocol_v1_6_execution_hardening") != hardening_protocol
        or upstream.get("boundary_verification_v2") != boundary_verification
    ):
        raise ValueError("Physical submission manifest and locked evidence disagree")
    boundary = manifest["boundaries"][selected_boundary]
    split_reference = boundary.get("split_manifest") if isinstance(boundary, Mapping) else None
    dataset = evaluation["dataset"]
    artifact = verified["artifact"]
    if (
        not isinstance(split_reference, Mapping)
        or split_reference.get("sha256") != dataset.get("split_manifest_sha256")
        or boundary.get("strict_nw_constraint_mode")
        != dataset.get("strict_nw_constraint_mode")
        or Path(str(boundary.get("evaluation_output", ""))).expanduser().resolve()
        != verified["evaluation_path"]
        or Path(str(boundary.get("score_artifact", ""))).expanduser().resolve()
        != artifact["path"]
    ):
        raise ValueError("Primary artifacts do not match the selected manifest boundary")

    checkpoint_suite_claim = _authenticate_checkpoint_suite_claim(
        verified,
        submission_manifest,
        manifest,
        selected_boundary,
    )
    primary_suite_completion = _authenticate_primary_suite_completion(
        verified,
        submission_manifest,
        manifest,
        checkpoint_suite_claim,
    )

    context = artifact.get("metadata", {}).get("contract", {}).get(
        "evaluation_context"
    )
    expected_context = {
        "submission_manifest_sha256": submission_manifest["sha256"],
        "protocol_v1_5_execution_lock_sha256": protocol["sha256"],
        "protocol_v1_6_execution_hardening_sha256": hardening_protocol["sha256"],
        "boundary_verification_v2_sha256": boundary_verification["sha256"],
        "fixed_evaluation_sha256": canonical_sha256(fixed_evaluation),
        "selected_boundary": selected_boundary,
        "checkpoint_suite_claim_sha256": checkpoint_suite_claim["sha256"],
    }
    if not isinstance(context, Mapping) or any(
        context.get(key) != value for key, value in expected_context.items()
    ):
        raise ValueError("Primary score contract lacks locked-submission evidence hashes")
    return {
        "submission_manifest": submission_manifest,
        "protocol_v1_5_execution_lock": protocol,
        "protocol_v1_6_execution_hardening": hardening_protocol,
        "boundary_verification_v2": boundary_verification,
        "fixed_evaluation_sha256": expected_context["fixed_evaluation_sha256"],
        "selected_boundary": selected_boundary,
        "score_contract_binding": expected_context,
        "checkpoint_suite_claim": checkpoint_suite_claim,
        "primary_suite_completion": primary_suite_completion,
    }


def _authenticate_suite(
    primary_spec: tuple[Path, str, Path, str],
    control_specs: Mapping[str, tuple[Path, str, Path, str]],
    execution_protocol: Mapping[str, str],
    control_protocol: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Authenticate all artifacts, ledgers, roles, and cross-result identities."""
    _assert_distinct_inputs(primary_spec, control_specs)
    primary = _authenticate_pair(*primary_spec)
    primary_evaluation = primary["evaluation"]
    if primary_evaluation.get("checkpoint_run_mode") != "final_refit_fixed_epochs":
        raise PermissionError("The primary evaluation is not the locked final refit")
    if primary_evaluation.get("one_shot_evaluation") is not True:
        raise PermissionError("The primary evaluation is not explicitly one-shot")
    if primary["evidence_scope"] != "reused_development_confirmation":
        raise PermissionError("The primary evaluation has an unauthorized evidence scope")
    primary_submission_evidence = _authenticate_locked_submission_evidence(
        primary, execution_protocol
    )
    execution_hardening_protocol = primary_submission_evidence[
        "protocol_v1_6_execution_hardening"
    ]
    primary_fingerprint = _comparison_fingerprint(primary)
    controls: dict[str, dict[str, Any]] = {}
    control_ledger_identity: tuple[str, str, str] | None = None
    control_bundle_sha256: str | None = None
    for name in CONTROL_NAMES:
        verified = _authenticate_pair(*control_specs[name])
        evaluation = verified["evaluation"]
        control = evaluation.get("control")
        if evaluation.get("checkpoint_run_mode") != "train_only_control_bundle":
            raise PermissionError(f"Control {name} is not a train-only control evaluation")
        if evaluation.get("one_shot_evaluation") is not True:
            raise PermissionError(f"Control {name} is not explicitly one-shot")
        if verified["evidence_scope"] != "reused_development_confirmation":
            raise PermissionError(f"Control {name} has an unauthorized evidence scope")
        if not isinstance(control, Mapping) or control.get("name") != name:
            raise ValueError(f"Control identity mismatch for {name}")
        if control.get("protocol_version") != CONTROL_SUITE_VERSION:
            raise ValueError(f"Control protocol version mismatch for {name}")
        if not _protocol_reference_matches(
            evaluation.get("execution_protocol_v1_6"), execution_hardening_protocol
        ):
            raise ValueError(f"Control {name} is not bound to the supplied protocol v1.6")
        if not _protocol_reference_matches(
            evaluation.get("control_protocol_v1"), control_protocol
        ):
            raise ValueError(
                f"Control {name} is not bound to the supplied controls protocol v1"
            )
        if _comparison_fingerprint(verified) != primary_fingerprint:
            raise ValueError(
                f"Control {name} does not share the primary boundary, authority, "
                "candidate libraries, and positive graph"
            )
        ledger = verified["one_shot_verification"]
        current_ledger = (
            str(ledger.get("canonical_key", "")),
            str(ledger.get("path", "")),
            str(ledger.get("sha256", "")),
        )
        if control_ledger_identity is None:
            control_ledger_identity = current_ledger
        elif current_ledger != control_ledger_identity:
            raise ValueError("The five controls do not share one completed central ledger")
        current_bundle = str(evaluation.get("checkpoint_sha256", ""))
        if control_bundle_sha256 is None:
            control_bundle_sha256 = current_bundle
        elif current_bundle != control_bundle_sha256:
            raise ValueError("The five controls do not share one train-only bundle")
        controls[name] = verified
    if control_ledger_identity is None or control_bundle_sha256 is None:
        raise AssertionError("The locked control suite was not authenticated")
    control_suite_completion = _authenticate_control_suite_completion(
        controls,
        execution_hardening_protocol,
        control_protocol,
    )
    return primary, controls, {
        "primary_one_shot_ledger": primary["one_shot_verification"],
        "primary_locked_submission_evidence": primary_submission_evidence,
        "control_suite_one_shot_ledger": controls[CONTROL_NAMES[0]][
            "one_shot_verification"
        ],
        "control_bundle_sha256": control_bundle_sha256,
        "control_suite_completion": control_suite_completion,
        "common_identity_sha256": canonical_sha256(primary_fingerprint),
    }


def _paired_comparison(
    primary_vectors: Mapping[str, Mapping[str, np.ndarray]],
    control_vectors: Mapping[str, Mapping[str, np.ndarray]],
    positive_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute locked paired effects and prefer graph-component resampling."""
    effects = _effect_vectors(primary_vectors, control_vectors)
    joint = joint_bipartite_component_bootstrap(
        effects,
        positive_mask,
        LOCKED_BOOTSTRAP_COUNT,
        LOCKED_BOOTSTRAP_SEED,
        LOCKED_ALPHA,
    )
    anchor = paired_effect_bootstrap(
        effects,
        LOCKED_BOOTSTRAP_COUNT,
        LOCKED_BOOTSTRAP_SEED,
        LOCKED_ALPHA,
    )
    if joint.get("available"):
        selected = joint
        selected_source = "joint_bipartite_component_bootstrap"
    else:
        selected = anchor
        selected_source = "paired_anchor_bootstrap_independent_directions"
    bidirectional_mrr = selected["bidirectional"]["mrr"]
    gate_passed = (
        float(bidirectional_mrr["estimate"]) > 0.0
        and float(bidirectional_mrr["lower"]) > 0.0
    )
    return {
        "effect_definition": "primary_minus_control",
        "per_anchor_effect_vector_audit": _effect_vector_audit(effects),
        "descriptive_point_effects": _descriptive_effect_points(effects),
        "selected_interval_source": selected_source,
        "selected_paired_intervals": _inferential_bootstrap_view(selected),
        "joint_bipartite_component_bootstrap": _inferential_bootstrap_view(joint),
        "paired_anchor_bootstrap_sensitivity": _inferential_bootstrap_view(anchor),
        "gate_relevant_effects": {
            direction: {
                metric: selected[direction][metric]
                for metric in INFERENTIAL_METRICS
            }
            for direction in (*DIRECTIONS, "bidirectional")
        },
        "primary_bidirectional_mrr_exceeds_control": float(
            bidirectional_mrr["estimate"]
        )
        > 0.0,
        "bidirectional_mrr_paired_95_percent_lower_ci_positive": float(
            bidirectional_mrr["lower"]
        )
        > 0.0,
        "gate_status": "passed" if gate_passed else "failed",
    }


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Authenticate one six-model suite and evaluate the locked control gate."""
    if (
        controls_runtime.CONTROL_PROTOCOL_V1_SHA256 != CONTROL_PROTOCOL_SHA256
        or controls_runtime.CONTROL_PROTOCOL_VERSION != CONTROL_SUITE_VERSION
    ):
        raise RuntimeError("Comparator and controls runtime protocol locks disagree")
    execution_protocol = _authenticate_protocol(
        args.execution_protocol,
        args.execution_protocol_sha256,
        EXECUTION_PROTOCOL_SHA256,
        "Partner-retrieval protocol v1.5",
    )
    control_protocol = _authenticate_protocol(
        args.control_protocol,
        args.control_protocol_sha256,
        CONTROL_PROTOCOL_SHA256,
        "Partner-retrieval controls protocol v1",
    )
    if not _is_sha256(args.primary_evaluation_sha256) or not _is_sha256(
        args.primary_score_artifact_sha256
    ):
        raise ValueError("The primary input requires exact SHA-256 values")
    primary_spec = (
        args.primary_evaluation_json.expanduser().resolve(),
        args.primary_evaluation_sha256,
        args.primary_score_artifact.expanduser().resolve(),
        args.primary_score_artifact_sha256,
    )
    controls = _validate_control_arguments(args.control)
    primary, verified_controls, ledger_authentication = _authenticate_suite(
        primary_spec,
        controls,
        execution_protocol,
        control_protocol,
    )
    primary_artifact = primary["artifact"]
    positive_mask = np.asarray(primary_artifact["positive_mask"], dtype=bool)
    primary_vectors = _directional_vectors(primary_artifact)
    primary_observed = _descriptive_metrics(primary_vectors)
    comparisons: dict[str, Any] = {}
    control_observed: dict[str, Any] = {}
    for name in CONTROL_NAMES:
        control_vectors = _directional_vectors(verified_controls[name]["artifact"])
        control_observed[name] = _descriptive_metrics(control_vectors)
        comparisons[name] = _paired_comparison(
            primary_vectors,
            control_vectors,
            positive_mask,
        )
    all_controls_passed = all(
        comparison["gate_status"] == "passed"
        for comparison in comparisons.values()
    )
    minimum_lower = min(
        float(comparison["gate_relevant_effects"]["bidirectional"]["mrr"]["lower"])
        for comparison in comparisons.values()
    )
    primary_evaluation = primary["evaluation"]
    dataset = primary_evaluation["dataset"]
    return {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": {
            "split": "validation",
            "intended_use": "reused_development_confirmation_control_comparison_only",
            "frozen_test_used": False,
            "claims_allowed": (
                "reused development confirmation evidence only; not an independent test"
            ),
        },
        "protocols": {
            "execution_lock_v1_5": execution_protocol,
            "execution_hardening_v1_6": ledger_authentication[
                "primary_locked_submission_evidence"
            ]["protocol_v1_6_execution_hardening"],
            "controls_v1": control_protocol,
        },
        "parameters": {
            "bootstrap_count": LOCKED_BOOTSTRAP_COUNT,
            "seed": LOCKED_BOOTSTRAP_SEED,
            "confidence_level": 1.0 - LOCKED_ALPHA,
            "interval": "two-sided_percentile",
            "joint_component_bootstrap_preferred_when_available": True,
            "multiplicity_adjustment": None,
            "gate_metric": "bidirectional_mrr",
            "paired_inferential_metrics": list(INFERENTIAL_METRICS),
            "all_other_metrics_descriptive_only": True,
        },
        "runtime_environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "inputs": {
            "primary": {
                "evaluation_json": str(primary["evaluation_path"]),
                "evaluation_json_sha256": primary["evaluation_sha256"],
                "score_artifact": str(primary_artifact["path"]),
                "score_artifact_sha256": primary_artifact["sha256"],
                "checkpoint_sha256": primary_evaluation["checkpoint_sha256"],
            },
            "controls": {
                name: {
                    "evaluation_json": str(verified_controls[name]["evaluation_path"]),
                    "evaluation_json_sha256": verified_controls[name][
                        "evaluation_sha256"
                    ],
                    "score_artifact": str(verified_controls[name]["artifact"]["path"]),
                    "score_artifact_sha256": verified_controls[name]["artifact"][
                        "sha256"
                    ],
                }
                for name in CONTROL_NAMES
            },
            "source_files": {
                "entrypoint": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "shared_statistics": {
                    "path": str(Path(statistics_runtime.__file__).resolve()),
                    "sha256": sha256_file(Path(statistics_runtime.__file__).resolve()),
                },
                "retrieval_runtime": {
                    "path": str(Path(retrieval_runtime.__file__).resolve()),
                    "sha256": sha256_file(Path(retrieval_runtime.__file__).resolve()),
                },
                "controls_runtime": {
                    "path": str(Path(controls_runtime.__file__).resolve()),
                    "sha256": sha256_file(Path(controls_runtime.__file__).resolve()),
                },
            },
        },
        "authentication": {
            **ledger_authentication,
            "boundary_verification": primary["boundary_verification"],
            "all_evaluation_and_score_hashes_physically_verified": True,
            "all_candidate_libraries_and_positive_graphs_identical": True,
            "excluded_validation_rows_read": False,
        },
        "library": {
            "split_manifest_sha256": dataset["split_manifest_sha256"],
            "split_format_version": AUTHORITY_VALIDATION_BOUNDARY_FORMAT,
            "split_intended_use": AUTHORITY_VALIDATION_BOUNDARY_INTENDED_USE,
            "strict_nw_constraint_mode": dataset["strict_nw_constraint_mode"],
            "authority_split_sha256": dataset["authority"]["sha256"],
            "antibody_count": int(positive_mask.shape[0]),
            "antigen_count": int(positive_mask.shape[1]),
            "positive_edge_count": int(positive_mask.sum()),
            "antibody_keys_sha256": primary_artifact["array_sha256"][
                "antibody_keys"
            ],
            "antigen_keys_sha256": primary_artifact["array_sha256"]["antigen_keys"],
            "positive_mask_sha256": primary_artifact["array_sha256"][
                "positive_mask"
            ],
        },
        "descriptive_metrics": {
            "primary": primary_observed,
            "controls": control_observed,
        },
        "paired_primary_minus_control": comparisons,
        "preregistered_gate_update": {
            "gate": GATE_NAME,
            "decision_rule": (
                "For every locked control, pass only when the primary bidirectional "
                "MRR point estimate is higher and the paired 95% lower confidence "
                "bound for that bidirectional-MRR difference is strictly positive. "
                "Directional MRR and R@1 effects are not gate criteria."
            ),
            "per_control_status": {
                name: comparison["gate_status"]
                for name, comparison in comparisons.items()
            },
            "minimum_bidirectional_mrr_lower_ci_across_controls": minimum_lower,
            "status": "passed" if all_controls_passed else "failed",
        },
        "limitations": [
            "This authority-validation pool was previously used for development exploration.",
            "The five paired 95% intervals are not multiplicity adjusted.",
            "Unobserved SAbDab pairings are not experimentally verified non-binders.",
            "Protein-language-model pretraining exposure cannot be fully audited here.",
            "No frozen-test, external, prospective, clinical, or world-leading claim is authorized.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-evaluation-json", type=Path, required=True)
    parser.add_argument("--primary-evaluation-sha256", required=True)
    parser.add_argument("--primary-score-artifact", type=Path, required=True)
    parser.add_argument("--primary-score-artifact-sha256", required=True)
    parser.add_argument(
        "--control",
        action="append",
        nargs=5,
        metavar=("NAME", "EVALUATION_JSON", "EVALUATION_SHA256", "SCORES_NPZ", "SCORES_SHA256"),
        required=True,
    )
    parser.add_argument("--execution-protocol", type=Path, required=True)
    parser.add_argument("--execution-protocol-sha256", required=True)
    parser.add_argument("--control-protocol", type=Path, required=True)
    parser.add_argument("--control-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison output: {output}")
    protected = {
        args.primary_evaluation_json.expanduser().resolve(),
        args.primary_score_artifact.expanduser().resolve(),
        *(Path(raw[1]).expanduser().resolve() for raw in args.control),
        *(Path(raw[3]).expanduser().resolve() for raw in args.control),
        args.execution_protocol.expanduser().resolve(),
        args.control_protocol.expanduser().resolve(),
    }
    if output in protected:
        raise ValueError("Comparison output cannot overwrite an authenticated input")
    report = run_comparison(args)
    _write_exclusive_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
