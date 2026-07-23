#!/usr/bin/env python3
"""Fit, evaluate, and finalize authenticated partner-retrieval controls."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sabdab2_asymmetric_partner_retrieval import (
    FORMAT_VERSION as EVALUATION_FORMAT_VERSION,
    FrozenPartnerEmbeddingStore,
    _atomic_write_deterministic_npz,
    _score_array_sha256,
    audit_isolation_policy,
    load_development_dataset,
    side_embedding_fingerprint,
    write_score_artifact,
)
from sabdab2_partner_retrieval import (
    PairRecord,
    SequenceRecord,
    atomic_write_json,
    canonical_sha256,
    deduplicate_exact_pairs,
    sha256_file,
    unique_sequences,
)


CONTROL_BUNDLE_FORMAT = "sabdab2_partner_retrieval_controls_v1"
CONTROL_PROTOCOL_VERSION = "partner_retrieval_controls_protocol_v1"
CONTROL_PROTOCOL_V1_SHA256 = (
    "21c3a0727a1b9d6543a022f6f9d216414ee8f0fc99590b4febca5dcb0d8cdc02"
)
EXECUTION_PROTOCOL_V1_6_SHA256 = (
    "b5b74452d3483d8b1fdf8c67387e3d38ad2be34a22b65e26a6cab71b0ca73078"
)
RETIRED_EXECUTION_PROTOCOL_SHA256 = frozenset(
    {"d9c3ca9bf9e07b498874b3324d3db4a2dfa2a5d0faf7de289311000f3fb2d87b"}
)
RETIRED_EXECUTION_PROTOCOL_SHA256_PREFIXES = ("6d354",)
LOCKED_PREPARED_INSTANCES_SHA256 = (
    "a9131d52323bae40572b7de81c5e41353ccc39a73309b7d7043485350facb009"
)
LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256 = (
    "548bcfab75923ec6621c1a3fef1aa6673c034f64a9bdc752e7ee15b41b4a4e63"
)
LOCKED_AUTHORITY_SPLIT_SHA256 = (
    "e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b"
)
LOCKED_BOUNDARY_VERIFICATION_SHA256 = (
    "3bd8186603d908a99e1fbf00ead3697dd276ccd798b4f91bcbdda81441bd80e8"
)
LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256 = {
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
AUTHORITY_VALIDATION_BOUNDARY_FORMAT = "sabdab2_authority_validation_nw_boundary_v2"
CONTROL_LEDGER_FORMAT = "sabdab2_control_one_shot_ledger_v3"
CONTROL_FIT_LOCK_FORMAT = "sabdab2_control_fit_process_lock_v1"
CONTROL_SUITE_COMPLETION_FORMAT = "sabdab2_control_two_boundary_suite_completion_v1"
CENTRAL_CONTROL_LEDGER_ROOT = Path(
    "/cluster/project/model_deployment/results/sabdab2/"
    "asymmetric_partner_retrieval/control_evaluation_registry"
)
CENTRAL_CONTROL_FIT_LOCK_ROOT = Path(
    "/cluster/project/model_deployment/results/sabdab2/"
    "asymmetric_partner_retrieval/control_fit_registry"
)
CONTROL_NAMES = (
    "raw_shared_esm_mean_cosine",
    "random_untrained_calm_head",
    "antibody_only_unary",
    "antigen_only_unary",
    "length_composition_logistic",
)
APPROVED_BOUNDARY_MODES = (
    "concat_antibody_and_antigen",
    "concat_antibody_antigen_plus_vh_vl",
)
LOCKED_PRIMARY_ANTIBODY_MODEL = {
    "artifact_format": "antiberty",
    "model_name": "AntiBERTy_md_smooth",
    "model_weights_sha256": "f1ae33eac8cc8784a7d4be5a600141d2fa7bc7d8d5b3f5324d64a6a63bd0f137",
    "input_dim": 512,
}
LOCKED_ESM_MODEL = {
    "artifact_format": "esm_chunks",
    "model_name": "facebook/esm2_t33_650M_UR50D",
    "model_weights_sha256": "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0",
    "input_dim": 1280,
}
LOCKED_FIT_HYPERPARAMETERS = {
    "negative_replicas": 5,
    "negative_seed": 42017,
    "logistic_l2": 0.01,
    "logistic_max_iterations": 50,
    "logistic_tolerance": 1e-9,
    "random_head_seed": 20260723,
    "random_head_hidden_dim": 256,
    "random_head_output_dim": 128,
}
UNARY_TIE_POLICY = {
    "policy": "domain_separated_sha256_key_rank_jitter",
    "label_or_positive_graph_used": False,
    "input_order_used": False,
    "jitter_amplitude": 0.25,
    "base_transform": "strict_logit_order_preserving_integer_rank",
    "minimum_gap_between_distinct_base_ranks": 1.0,
    "maximum_jitter_range_exclusive": 0.25,
    "reason": "the_score_schema_requires_a_deterministic_point_order",
}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
COMPOSITION_GROUPS = {
    "hydrophobic": frozenset("AILMFWVY"),
    "positive": frozenset("KRH"),
    "negative": frozenset("DE"),
    "polar": frozenset("STNQ"),
    "aromatic": frozenset("FWY"),
}
RECALL_KS = (1, 5, 10, 20)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.strip().upper().encode("ascii")).hexdigest()


def _identifier_membership_sha256(identifiers: Iterable[str]) -> str:
    """Hash an ordered-independent identifier inventory using the v2 boundary contract."""
    return hashlib.sha256(
        ("\n".join(sorted(str(value) for value in identifiers)) + "\n").encode("utf-8")
    ).hexdigest()


def _source_hashes() -> dict[str, str]:
    """Hash the control code and shared contracts that define its semantics."""
    script = Path(__file__).resolve()
    return {
        "control_entrypoint": sha256_file(script),
        "asymmetric_entrypoint": sha256_file(
            script.parent / "sabdab2_asymmetric_partner_retrieval.py"
        ),
        "retrieval_data_contract": sha256_file(
            script.parent / "sabdab2_partner_retrieval.py"
        ),
    }


def load_approved_boundary_registry(
    path: Path,
    expected_sha256: str,
    authority_split_sha256: str,
) -> dict[str, Any]:
    """Authenticate the independent verifier that predeclares both v2 boundaries."""
    if expected_sha256 != LOCKED_BOUNDARY_VERIFICATION_SHA256:
        raise ValueError("Boundary-verification digest differs from final v1.6")
    if authority_split_sha256 != LOCKED_AUTHORITY_SPLIT_SHA256:
        raise ValueError("Authority split digest differs from final v1.6")
    target = path.expanduser().resolve()
    if not _is_sha256(expected_sha256) or sha256_file(target) != expected_sha256:
        raise ValueError("Boundary-verification manifest SHA-256 mismatch")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if (
        payload.get("format_version")
        != "sabdab2_authority_validation_nw_boundary_verification_v2"
        or payload.get("intended_use") != "reused_development_confirmation_only"
        or payload.get("valid") is not True
        or payload.get("labels_predictions_embeddings_or_checkpoints_read") is not False
        or payload.get("frozen_test_identifiers_read_for_exclusion_only") is not True
        or payload.get(
            "frozen_test_sequence_fields_accessed_normalized_hashed_or_aligned"
        )
        is not False
    ):
        raise PermissionError("Independent v2 boundary verification is invalid")
    authority_reference = payload.get("provenance", {}).get("authority_split", {})
    if authority_reference.get("sha256") != authority_split_sha256:
        raise ValueError("Boundary verification references a different authority split")
    modes = payload.get("modes")
    if not isinstance(modes, Mapping) or set(modes) != set(APPROVED_BOUNDARY_MODES):
        raise ValueError("Boundary verification must predeclare exactly the two locked modes")
    approved_modes = {}
    for mode in APPROVED_BOUNDARY_MODES:
        record = modes[mode]
        boundary = record.get("boundary", {}) if isinstance(record, Mapping) else {}
        boundary_sha256 = str(boundary.get("sha256", ""))
        if record.get("valid") is not True or not _is_sha256(boundary_sha256):
            raise ValueError(f"Boundary mode {mode!r} lacks a valid verified artifact")
        approved_modes[mode] = {
            "boundary_sha256": boundary_sha256,
            "boundary_scientific_payload_sha256": boundary.get(
                "scientific_payload_sha256"
            ),
            "authority_train_instance_count": int(
                record["authority_train_instance_count"]
            ),
            "eligible_validation_instance_count": int(
                record["eligible_validation_instance_count"]
            ),
            "excluded_validation_instance_count": int(
                record["excluded_validation_instance_count"]
            ),
        }
    return {
        "format_version": payload["format_version"],
        "sha256": expected_sha256,
        "scientific_payload_sha256": payload.get("scientific_payload_sha256"),
        "authority_split_sha256": authority_split_sha256,
        "modes": approved_modes,
    }


def validate_execution_protocol_reference(
    path: Path,
    expected_sha256: str,
) -> dict[str, str]:
    """Bind the final physical protocol v1.6 file and reject retired drafts."""
    target = path.expanduser().resolve()
    if not isinstance(expected_sha256, str):
        raise ValueError("Execution protocol requires an exact SHA-256 string")
    if expected_sha256 in RETIRED_EXECUTION_PROTOCOL_SHA256 or expected_sha256.startswith(
        RETIRED_EXECUTION_PROTOCOL_SHA256_PREFIXES
    ):
        raise ValueError("Execution protocol SHA-256 identifies a retired v1.6 draft")
    if expected_sha256 != EXECUTION_PROTOCOL_V1_6_SHA256:
        raise ValueError("Execution protocol SHA-256 differs from the final locked v1.6 digest")
    if not target.is_file() or sha256_file(target) != EXECUTION_PROTOCOL_V1_6_SHA256:
        raise ValueError("Physical execution protocol v1.6 is missing or changed")
    return {
        "path": str(target),
        "sha256": EXECUTION_PROTOCOL_V1_6_SHA256,
    }


def validate_control_protocol_reference(
    path: Path,
    expected_sha256: str,
) -> dict[str, str]:
    """Bind the frozen physical controls protocol rather than a version label alone."""
    target = path.expanduser().resolve()
    if expected_sha256 != CONTROL_PROTOCOL_V1_SHA256:
        raise ValueError("Controls protocol SHA-256 differs from the final locked digest")
    if not target.is_file() or sha256_file(target) != CONTROL_PROTOCOL_V1_SHA256:
        raise ValueError("Physical controls protocol v1 is missing or changed")
    return {"path": str(target), "sha256": CONTROL_PROTOCOL_V1_SHA256}


def _locked_immutable_input_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """Validate immutable v1.6 digests without reading dataset or embedding bytes."""
    observed = {
        "prepared_instances_sha256": args.prepared_instances_sha256,
        "canonical_sequence_manifest_sha256": args.canonical_sequence_manifest_sha256,
    }
    expected = {
        "prepared_instances_sha256": LOCKED_PREPARED_INSTANCES_SHA256,
        "canonical_sequence_manifest_sha256": (
            LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256
        ),
    }
    if observed != expected:
        raise ValueError(
            "Immutable data-input digests differ from the final v1.6 contract: "
            f"expected={expected}, observed={observed}"
        )
    if (
        args.authority_split_sha256 != LOCKED_AUTHORITY_SPLIT_SHA256
        or args.boundary_verification_sha256
        != LOCKED_BOUNDARY_VERIFICATION_SHA256
    ):
        raise ValueError("Authority or boundary verification differs from final v1.6")
    return {
        "prepared_instances": {
            "path": str(
                (args.prepare_dir.expanduser().resolve() / "instances.parquet")
            ),
            "sha256": LOCKED_PREPARED_INSTANCES_SHA256,
        },
        "canonical_sequence_manifest": {
            "path": str(args.canonical_sequence_manifest.expanduser().resolve()),
            "sha256": LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256,
        },
        "training_embedding_content_manifest_sha256": dict(
            LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256
        ),
    }


def _validate_locked_training_embedding_fingerprints(
    fingerprints: Mapping[str, Any],
) -> dict[str, str]:
    """Require exact reviewed training-universe embedding inventories."""
    observed = {
        name: str(fingerprints.get(name, {}).get("content_manifest_sha256", ""))
        for name in LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256
    }
    if set(fingerprints) != set(LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256):
        raise ValueError("Training embedding fingerprint roster is not exact")
    if observed != LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256:
        raise ValueError(
            "Training embedding content fingerprints differ from the final v1.6 lock: "
            f"expected={LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256}, "
            f"observed={observed}"
        )
    return observed


def control_fit_identity(
    authority_split_sha256: str,
    boundary_verification_sha256: str,
) -> tuple[str, dict[str, Any]]:
    """Return one path-independent identity for the common authority-train control fit."""
    if not _is_sha256(authority_split_sha256) or not _is_sha256(
        boundary_verification_sha256
    ):
        raise ValueError("Control-fit identity requires exact authority and audit hashes")
    contract = {
        "format_version": CONTROL_FIT_LOCK_FORMAT,
        "control_suite_version": CONTROL_PROTOCOL_VERSION,
        "control_protocol_v1_sha256": CONTROL_PROTOCOL_V1_SHA256,
        "execution_protocol_v1_6_sha256": EXECUTION_PROTOCOL_V1_6_SHA256,
        "authority_split_sha256": authority_split_sha256,
        "boundary_verification_sha256": boundary_verification_sha256,
        "prepared_instances_sha256": LOCKED_PREPARED_INSTANCES_SHA256,
        "canonical_sequence_manifest_sha256": (
            LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256
        ),
        "training_embedding_content_manifest_sha256": dict(
            LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256
        ),
        "controls": list(CONTROL_NAMES),
        "fit_split": "common_authority_train",
        "frozen_test_used": False,
    }
    return canonical_sha256(contract), contract


@contextmanager
def exclusive_control_fit_process_lock(
    registry_root: Path,
    authority_split_sha256: str,
    boundary_verification_sha256: str,
    requested_output_dir: Path,
) -> Iterable[dict[str, Any]]:
    """Hold a nonblocking process-lifetime lock for the single common control fit."""
    identity, contract = control_fit_identity(
        authority_split_sha256,
        boundary_verification_sha256,
    )
    registry = registry_root.expanduser().resolve()
    registry.mkdir(parents=True, exist_ok=True)
    lock_path = registry / f"{identity}.process.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another process already holds the common control-fit lock: "
                f"{lock_path}"
            ) from error
        payload = {
            **contract,
            "fit_identity_sha256": identity,
            "status": "held_for_process_lifetime",
            "requested_output_dir": str(requested_output_dir.expanduser().resolve()),
        }
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, encoded)
        os.fsync(descriptor)
        yield {
            "path": str(lock_path),
            "identity_sha256": identity,
            "contract": contract,
        }
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _create_exclusive_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Create an immutable-start ledger with O_EXCL before holdout loading."""
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _temporary_sibling(path: Path, suffix: str) -> Path:
    """Reserve a unique absent sibling path for create-only staging."""
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    staged = Path(name)
    staged.unlink()
    return staged


def _write_npz_create_only(path: Path, arrays: Mapping[str, Any]) -> Path:
    """Stage deterministic NPZ bytes and publish by an atomic no-clobber link."""
    target = path.expanduser().resolve()
    staged = _temporary_sibling(target, ".staged.npz")
    try:
        _atomic_write_deterministic_npz(staged, arrays)
        os.link(staged, target)
    finally:
        try:
            staged.unlink()
        except FileNotFoundError:
            pass
    return target


def _write_score_artifact_create_only(
    path: Path,
    score_payload: Mapping[str, Any],
    evaluation_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish score bytes create-only while preserving their final-path receipt."""
    target = path.expanduser().resolve()
    staged = _temporary_sibling(target, ".staged.scores.npz")
    try:
        receipt = write_score_artifact(staged, score_payload, evaluation_context)
        os.link(staged, target)
    finally:
        try:
            staged.unlink()
        except FileNotFoundError:
            pass
    return {
        **receipt,
        "path_at_creation": str(target),
        "sha256": sha256_file(target),
    }


def _strict_json_object(path: Path, label: str) -> dict[str, Any]:
    """Read one finite JSON object while rejecting duplicate keys."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read {label} as UTF-8 JSON: {path}") from error
    return _strict_json_text_object(text, label)


def _strict_json_text_object(text: str, label: str) -> dict[str, Any]:
    """Decode one finite JSON object while rejecting duplicate keys."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON value {value}")

    def require_finite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} contains non-finite JSON value {value}")
        if isinstance(value, Mapping):
            for nested in value.values():
                require_finite(nested)
        elif isinstance(value, list):
            for nested in value:
                require_finite(nested)

    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Cannot decode {label} as strict JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    require_finite(payload)
    return payload


def _raw_npz_member_names(payload: bytes, label: str) -> tuple[str, ...]:
    """Authenticate raw NPZ members before NumPy normalizes their names."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            raw_names = tuple(info.filename for info in archive.infolist())
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"{label} is not a valid NPZ archive") from error
    if not raw_names:
        raise ValueError(f"{label} contains no array members")
    if len(raw_names) != len(set(raw_names)):
        raise ValueError(f"{label} contains duplicate raw ZIP members")
    if any(
        not name.endswith(".npy")
        or not name[:-4]
        or "/" in name
        or "\\" in name
        for name in raw_names
    ):
        raise ValueError(f"{label} contains a non-canonical NPZ member name")
    logical_names = tuple(name[:-4] for name in raw_names)
    if len(logical_names) != len(set(logical_names)):
        raise ValueError(f"{label} contains duplicate logical array members")
    return logical_names


def _strict_hashed_json_reference(
    reference: Any,
    label: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Hash and parse one JSON file from the same immutable byte buffer."""
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path_value = reference.get("path")
    expected_sha256 = str(reference.get("sha256", ""))
    if not isinstance(path_value, str) or not Path(path_value).expanduser().is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file() or not _is_sha256(expected_sha256):
        raise ValueError(f"{label} is missing or lacks an exact SHA-256")
    payload_bytes = path.read_bytes()
    observed_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{label} physical SHA-256 mismatch: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    try:
        payload_text = payload_bytes.decode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    payload = _strict_json_text_object(payload_text, label)
    return {"path": str(path), "sha256": expected_sha256}, payload


def _physical_hashed_reference(
    reference: Any,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    """Resolve and hash one exact physical file reference."""
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path_value = reference.get("path")
    digest = str(reference.get("sha256", ""))
    if not isinstance(path_value, str) or not Path(path_value).expanduser().is_absolute():
        raise ValueError(f"{label} path must be absolute")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file() or not _is_sha256(digest):
        raise ValueError(f"{label} is missing or lacks an exact SHA-256")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"{label} digest differs from the locked value")
    observed = sha256_file(path)
    if observed != digest:
        raise ValueError(
            f"{label} physical SHA-256 mismatch: expected={digest}, observed={observed}"
        )
    return {"path": str(path), "sha256": digest}


def control_evaluation_identity(
    control_bundle_sha256: str,
    split_manifest_sha256: str,
    authority_split_sha256: str,
) -> tuple[str, dict[str, Any]]:
    """Return the path-independent identity of the one allowed suite evaluation."""
    digests = (
        control_bundle_sha256,
        split_manifest_sha256,
        authority_split_sha256,
    )
    if not all(_is_sha256(value) for value in digests):
        raise ValueError("Canonical control-evaluation identity requires exact SHA-256 values")
    contract = {
        "format_version": CONTROL_LEDGER_FORMAT,
        "control_suite_version": CONTROL_PROTOCOL_VERSION,
        "control_bundle_sha256": control_bundle_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "authority_split_sha256": authority_split_sha256,
        "control_protocol_v1_sha256": CONTROL_PROTOCOL_V1_SHA256,
        "execution_protocol_v1_6_sha256": EXECUTION_PROTOCOL_V1_6_SHA256,
        "prepared_instances_sha256": LOCKED_PREPARED_INSTANCES_SHA256,
        "canonical_sequence_manifest_sha256": (
            LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256
        ),
        "training_embedding_content_manifest_sha256": dict(
            LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256
        ),
        "split": "validation",
        "controls": list(CONTROL_NAMES),
        "frozen_test_used": False,
    }
    return canonical_sha256(contract), contract


def reserve_control_evaluation(
    registry_root: Path,
    control_bundle_sha256: str,
    split_manifest_sha256: str,
    authority_split_sha256: str,
    requested_output_dir: Path,
) -> tuple[Path, str, dict[str, Any]]:
    """Reserve one canonical suite identity independently of requested output paths."""
    registry = registry_root.expanduser().resolve()
    identity, contract = control_evaluation_identity(
        control_bundle_sha256,
        split_manifest_sha256,
        authority_split_sha256,
    )
    ledger = registry / f"{identity}.json"
    _create_exclusive_json(
        ledger,
        {
            **contract,
            "evaluation_identity_sha256": identity,
            "status": "started_before_validation_materialization",
            "requested_output_dir": str(requested_output_dir.expanduser().resolve()),
        },
    )
    return ledger, identity, contract


def sequence_descriptor(sequence: str) -> Any:
    """Return fixed length, composition, entropy, and chemistry descriptors."""
    import numpy as np

    normalized = sequence.strip().upper()
    if not normalized or any(not (character.isalpha() or character == "*") for character in normalized):
        raise ValueError("Protein sequences must be non-empty alphabetic strings")
    length = len(normalized)
    fractions = np.asarray(
        [normalized.count(character) / length for character in AMINO_ACIDS],
        dtype=np.float64,
    )
    other_fraction = min(1.0, max(0.0, 1.0 - float(fractions.sum())))
    composition = np.concatenate([fractions, np.asarray([other_fraction])])
    positive = composition[composition > 0.0]
    entropy = float(-(positive * np.log(positive)).sum())
    group_fractions = np.asarray(
        [
            sum(character in members for character in normalized) / length
            for members in COMPOSITION_GROUPS.values()
        ],
        dtype=np.float64,
    )
    result = np.concatenate(
        [np.asarray([math.log1p(length)], dtype=np.float64), composition, [entropy], group_fractions]
    )
    if result.shape != (28,) or not np.isfinite(result).all():
        raise AssertionError("Sequence descriptor contract changed unexpectedly")
    return result


def pair_descriptor(antibody: Any, antigen: Any) -> Any:
    """Combine marginal descriptors with low-capacity composition interactions."""
    import numpy as np

    antibody = np.asarray(antibody, dtype=np.float64)
    antigen = np.asarray(antigen, dtype=np.float64)
    if antibody.shape != (28,) or antigen.shape != (28,):
        raise ValueError("Pair descriptors require two 28-dimensional sequence vectors")
    antibody_composition = antibody[1:22]
    antigen_composition = antigen[1:22]
    return np.concatenate(
        [
            antibody,
            antigen,
            np.abs(antibody_composition - antigen_composition),
            antibody_composition * antigen_composition,
            np.asarray(
                [
                    abs(float(antibody[0] - antigen[0])),
                    float(antibody[0] * antigen[0]),
                ]
            ),
        ]
    )


def deterministic_endpoint_matched_negatives(
    positive_pairs: Sequence[tuple[str, str]],
    replicas: int,
    seed: int,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Generate train-only pseudo-negatives by permuting observed antigen endpoints."""
    import numpy as np

    positives = sorted(set((str(left), str(right)) for left, right in positive_pairs))
    if not positives or replicas <= 0:
        raise ValueError("Positive pairs and a positive replica count are required")
    positive_set = set(positives)
    antibodies = [left for left, _ in positives]
    antigens = np.asarray([right for _, right in positives])
    generator = np.random.default_rng(seed)
    negatives: list[tuple[str, str]] = []
    collision_count = 0
    for _ in range(replicas):
        best: list[tuple[str, str]] = []
        best_collisions = len(positives) + 1
        for _attempt in range(128):
            shuffled = antigens[generator.permutation(len(antigens))]
            proposed = list(zip(antibodies, shuffled.tolist()))
            valid = [pair for pair in proposed if pair not in positive_set]
            collisions = len(proposed) - len(valid)
            if collisions < best_collisions:
                best = valid
                best_collisions = collisions
            if collisions == 0:
                break
        negatives.extend(best)
        collision_count += best_collisions
    if not negatives or any(pair in positive_set for pair in negatives):
        raise AssertionError("Pseudo-negative construction produced an invalid edge")
    return negatives, {
        "policy": "train_only_antigen_endpoint_permutation_with_positive_rejection",
        "replicas": int(replicas),
        "seed": int(seed),
        "positive_edge_count": len(positives),
        "negative_observation_count": len(negatives),
        "known_positive_collisions_rejected": collision_count,
        "unobserved_pairs_are_verified_nonbinders": False,
    }


def fit_fixed_logistic(
    positive_features: Any,
    negative_features: Any,
    l2_penalty: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a class-balanced L2 logistic model by deterministic Newton updates."""
    import numpy as np

    positive = np.asarray(positive_features, dtype=np.float64)
    negative = np.asarray(negative_features, dtype=np.float64)
    if (
        positive.ndim != 2
        or negative.ndim != 2
        or positive.shape[1] != negative.shape[1]
        or positive.shape[0] == 0
        or negative.shape[0] == 0
        or not np.isfinite(positive).all()
        or not np.isfinite(negative).all()
    ):
        raise ValueError("Positive and negative design matrices must be finite and aligned")
    if l2_penalty <= 0.0 or max_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("Logistic optimization constants must be positive")
    features = np.vstack([positive, negative])
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(features)), standardized])
    sample_weights = np.concatenate(
        [
            np.full(len(positive), len(features) / (2.0 * len(positive))),
            np.full(len(negative), len(features) / (2.0 * len(negative))),
        ]
    )
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    converged = False
    step_norm = math.inf
    for iteration in range(1, max_iterations + 1):
        logits = np.clip(design @ coefficients, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (sample_weights * (probabilities - labels))
        gradient = gradient / len(features) + l2_penalty * (penalty @ coefficients)
        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian = (design.T * curvature) @ design / len(features)
        hessian = hessian + l2_penalty * penalty + np.eye(design.shape[1]) * 1e-10
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        step_norm = float(np.linalg.norm(step))
        if step_norm <= tolerance:
            converged = True
            break
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("Logistic control fit produced non-finite coefficients")
    model = {
        "mean": mean.astype(np.float64),
        "scale": scale.astype(np.float64),
        "coefficients": coefficients.astype(np.float64),
    }
    summary = {
        "optimizer": "deterministic_class_balanced_newton_irls",
        "positive_count": int(len(positive)),
        "negative_count": int(len(negative)),
        "feature_count": int(positive.shape[1]),
        "l2_penalty": float(l2_penalty),
        "maximum_iterations": int(max_iterations),
        "iterations_completed": int(iteration),
        "tolerance": float(tolerance),
        "final_step_l2_norm": step_norm,
        "converged": converged,
    }
    return model, summary


def apply_fixed_logistic(features: Any, model: Mapping[str, Any]) -> Any:
    """Apply one immutable standardized logistic scoring function."""
    import numpy as np

    values = np.asarray(features, dtype=np.float64)
    standardized = (values - model["mean"]) / model["scale"]
    design = np.column_stack([np.ones(len(values)), standardized])
    logits = design @ model["coefficients"]
    if not np.isfinite(logits).all():
        raise FloatingPointError("Logistic control produced non-finite scores")
    return logits


def _hash_rank(keys: Sequence[str], domain: str) -> Any:
    """Return input-order-invariant integer ranks from domain-separated key hashes."""
    import numpy as np

    values = [str(key) for key in keys]
    if not values or len(values) != len(set(values)):
        raise ValueError("Tie-breaking keys must be non-empty and unique")
    hashed = [
        hashlib.sha256(f"{domain}\0{key}".encode("ascii")).hexdigest()
        for key in values
    ]
    ordered = sorted(range(len(values)), key=lambda index: hashed[index])
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[ordered] = np.arange(len(values), dtype=np.float64)
    return ranks


def _rank_preserving_unary_scores(
    values: Any,
    keys: Sequence[str],
    domain: str,
) -> Any:
    """Map unary logits to unique ranks while preserving every strict logit order."""
    import numpy as np

    logits = np.asarray(values, dtype=np.float64)
    if logits.shape != (len(keys),) or not np.isfinite(logits).all():
        raise ValueError("Unary logits must align with finite unique-key libraries")
    tie_ranks = _hash_rank(keys, f"{domain}:equal_logit")
    ordered = sorted(
        range(len(keys)),
        key=lambda index: (float(logits[index]), float(tie_ranks[index])),
    )
    ranks = np.empty(len(keys), dtype=np.float64)
    ranks[ordered] = np.arange(len(keys), dtype=np.float64)
    return ranks


def unary_score_matrices(
    antibody_logits: Any,
    antigen_logits: Any,
    antibody_keys: Sequence[str],
    antigen_keys: Sequence[str],
) -> tuple[Any, Any]:
    """Resolve structurally tied unary directions with label-blind key-hash jitter."""
    import numpy as np

    antibody_order = _rank_preserving_unary_scores(
        antibody_logits, antibody_keys, "antibody_only:antibody"
    )
    antigen_order = _rank_preserving_unary_scores(
        antigen_logits, antigen_keys, "antigen_only:antigen"
    )
    antibody_jitter = (
        _hash_rank(antibody_keys, "antigen_only:antibody_tie") + 1.0
    ) / (len(antibody_keys) + 1.0)
    antigen_jitter = (
        _hash_rank(antigen_keys, "antibody_only:antigen_tie") + 1.0
    ) / (len(antigen_keys) + 1.0)
    jitter_amplitude = 0.25
    antibody_only = antibody_order[:, None] + jitter_amplitude * antigen_jitter[None, :]
    antigen_only = antigen_order[None, :] + jitter_amplitude * antibody_jitter[:, None]
    antibody_only = antibody_only.astype(np.float32)
    antigen_only = antigen_only.astype(np.float32)
    if (
        any(len(set(row.tolist())) != len(row) for row in antibody_only)
        or any(len(set(row.tolist())) != len(row) for row in antigen_only.T)
    ):
        raise FloatingPointError("Float32 unary tie-breaking scores are not unique")
    return antibody_only, antigen_only


def make_random_head_arrays(
    antibody_input_dim: int,
    antigen_input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int,
) -> dict[str, Any]:
    """Create fixed independent CALM-shaped random heads without fitting labels."""
    import numpy as np

    if min(antibody_input_dim, antigen_input_dim, hidden_dim, output_dim) <= 0:
        raise ValueError("Random-head dimensions must be positive")
    generator = np.random.default_rng(seed)

    def side(prefix: str, input_dim: int) -> dict[str, Any]:
        return {
            f"{prefix}_w1": generator.normal(
                0.0, 1.0 / math.sqrt(input_dim), size=(input_dim, hidden_dim)
            ).astype(np.float32),
            f"{prefix}_b1": np.zeros(hidden_dim, dtype=np.float32),
            f"{prefix}_w2": generator.normal(
                0.0, 1.0 / math.sqrt(hidden_dim), size=(hidden_dim, output_dim)
            ).astype(np.float32),
            f"{prefix}_b2": np.zeros(output_dim, dtype=np.float32),
        }

    return {**side("random_ab", antibody_input_dim), **side("random_ag", antigen_input_dim)}


def random_head_pool(residue_features: Any, arrays: Mapping[str, Any], prefix: str) -> Any:
    """Apply fixed layer normalization, two random layers, GELU, and mean pooling."""
    import numpy as np

    features = np.asarray(residue_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError("Residue features must be a finite non-empty matrix")
    mean = features.mean(axis=1, keepdims=True)
    variance = ((features - mean) ** 2).mean(axis=1, keepdims=True)
    normalized = (features - mean) / np.sqrt(variance + 1e-5)
    hidden = normalized @ arrays[f"{prefix}_w1"] + arrays[f"{prefix}_b1"]
    hidden = 0.5 * hidden * (
        1.0
        + np.tanh(
            math.sqrt(2.0 / math.pi) * (hidden + 0.044715 * hidden**3)
        )
    )
    pooled = (hidden @ arrays[f"{prefix}_w2"] + arrays[f"{prefix}_b2"]).mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Random head produced a zero or non-finite pooled vector")
    return pooled / norm


def _cosine_matrix(left: Any, right: Any) -> Any:
    import numpy as np

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[1]
        or np.any(left_norm <= 1e-12)
        or np.any(right_norm <= 1e-12)
    ):
        raise ValueError("Cosine libraries must be non-zero matrices with a shared dimension")
    scores = (left / left_norm) @ (right / right_norm).T
    return scores.astype(np.float32)


def _stable_best_ranks(scores: Any, mask: Any) -> Any:
    import numpy as np

    values = np.asarray(scores)
    positives = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or values.shape != positives.shape or not positives.any(axis=1).all():
        raise ValueError("Scores and positive masks must align with a positive per anchor")
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int64)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, values.shape[1] + 1)
    return np.where(positives, ranks, values.shape[1] + 1).min(axis=1)


def _direction_metrics(
    scores: Any,
    mask: Any,
    anchor_keys: Sequence[str],
    candidate_keys: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np

    ranks = _stable_best_ranks(scores, mask)
    positive_mask = np.asarray(mask, dtype=bool)
    per_anchor = []
    for index, (anchor_key, rank) in enumerate(zip(anchor_keys, ranks)):
        positive_indices = np.flatnonzero(positive_mask[index])
        positive_keys = [candidate_keys[int(value)] for value in positive_indices]
        candidate_count = len(candidate_keys)
        per_anchor.append(
            {
                "anchor_key": anchor_key,
                "best_known_positive_rank": int(rank),
                "reciprocal_rank": 1.0 / int(rank),
                "normalized_rank_zero_one": (
                    (int(rank) - 1) / (candidate_count - 1) if candidate_count > 1 else 0.0
                ),
                "rank_fraction": int(rank) / candidate_count,
                "known_positive_candidate_count": len(positive_keys),
                "known_positive_candidate_keys": positive_keys,
            }
        )
    metrics = {
        "anchor_count": len(anchor_keys),
        "candidate_count": len(candidate_keys),
        "known_positive_count": int(positive_mask.sum()),
        "mean_positives_per_anchor": float(positive_mask.sum(axis=1).mean()),
        "mrr": float(math.fsum(1.0 / int(rank) for rank in ranks) / len(ranks)),
        "mean_rank": float(sum(int(rank) for rank in ranks) / len(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_normalized_rank_zero_one": float(
            math.fsum(value["normalized_rank_zero_one"] for value in per_anchor)
            / len(per_anchor)
        ),
        "median_normalized_rank_zero_one": float(
            np.median([value["normalized_rank_zero_one"] for value in per_anchor])
        ),
        "normalized_rank_definition": "(best_positive_rank-1)/(candidate_count-1); lower_is_better",
    }
    for cutoff in RECALL_KS:
        value = float(sum(int(rank) <= cutoff for rank in ranks) / len(ranks))
        metrics[f"recall_at_{cutoff}"] = value
        metrics[f"hit_rate_at_{cutoff}"] = value
    return metrics, per_anchor


def metrics_from_scores(
    scores: Any,
    positive_mask: Any,
    antibody_keys: Sequence[str],
    antigen_keys: Sequence[str],
) -> dict[str, Any]:
    """Compute statistics-tool-compatible full-library retrieval metrics."""
    import numpy as np

    values = np.asarray(scores, dtype=np.float32)
    mask = np.asarray(positive_mask, dtype=bool)
    antibody_metrics, antibody_ranks = _direction_metrics(
        values, mask, antibody_keys, antigen_keys
    )
    antigen_metrics, antigen_ranks = _direction_metrics(
        values.T, mask.T, antigen_keys, antibody_keys
    )
    directional = [antibody_metrics["mrr"], antigen_metrics["mrr"]]
    return {
        "bidirectional_mrr": float(sum(directional) / 2.0),
        "harmonic_directional_mrr": float(
            2.0 / sum(1.0 / value for value in directional)
            if all(value > 0.0 for value in directional)
            else 0.0
        ),
        "minimum_directional_mrr": float(min(directional)),
        "antibody_to_antigen": antibody_metrics,
        "antigen_to_antibody": antigen_metrics,
        "library_keys": {"antibody": list(antibody_keys), "antigen": list(antigen_keys)},
        "per_anchor_ranks": {
            "antibody_to_antigen": antibody_ranks,
            "antigen_to_antibody": antigen_ranks,
        },
        "exact_pair_count": int(mask.sum()),
        "unique_antibody_count": len(antibody_keys),
        "unique_antigen_count": len(antigen_keys),
    }


def _read_sequences(
    canonical_sequence_manifest: Path,
    records: Sequence[PairRecord],
) -> tuple[dict[str, str], dict[str, str]]:
    """Read only selected rows and authenticate every normalized sequence."""
    import pandas as pd

    identifiers = sorted({record.instance_id for record in records})
    path = canonical_sequence_manifest.expanduser().resolve()
    if (
        not path.is_file()
        or sha256_file(path) != LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256
    ):
        raise ValueError(
            "Canonical sequence manifest changed before sequence row materialization"
        )
    frame = pd.read_csv(
        path,
        usecols=["INSTANCE", "vh_seq", "vl_seq", "ag_seq"],
        dtype=str,
        keep_default_na=False,
    )
    if sha256_file(path) != LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256:
        raise ValueError("Canonical sequence manifest changed during row materialization")
    frame["INSTANCE"] = frame["INSTANCE"].astype(str)
    frame = frame.loc[frame["INSTANCE"].isin(identifiers)].copy()
    if set(frame["INSTANCE"]) != set(identifiers) or frame["INSTANCE"].duplicated().any():
        raise ValueError("Selected sequence rows do not match the authorized identifiers")
    by_identifier = frame.set_index("INSTANCE")
    antibodies: dict[str, str] = {}
    antigens: dict[str, str] = {}
    for record in records:
        row = by_identifier.loc[record.instance_id]
        heavy = str(row["vh_seq"]).strip().upper()
        light = str(row["vl_seq"]).strip().upper()
        antibody = heavy + light
        antigen = str(row["ag_seq"]).strip().upper()
        if not heavy or not antibody or not antigen:
            raise ValueError(f"Canonical sequence is missing for {record.instance_id}")
        if (
            _sequence_sha256(antibody) != record.antibody_key
            or _sequence_sha256(antigen) != record.antigen_key
            or len(antibody) != record.antibody_length
            or len(antigen) != record.antigen_length
        ):
            raise ValueError(f"Sequence authentication failed for {record.instance_id}")
        if record.antibody_key in antibodies and antibodies[record.antibody_key] != antibody:
            raise ValueError("One antibody digest maps to multiple sequences")
        if record.antigen_key in antigens and antigens[record.antigen_key] != antigen:
            raise ValueError("One antigen digest maps to multiple sequences")
        antibodies[record.antibody_key] = antibody
        antigens[record.antigen_key] = antigen
    return antibodies, antigens


def _dataset_from_args(
    args: argparse.Namespace,
    materialized_splits: Sequence[str],
) -> tuple[dict[str, list[PairRecord]], dict[str, Any]]:
    _locked_immutable_input_arguments(args)
    if (
        list(args.authorized_source_pool_splits) != ["train", "validation"]
        or list(args.authority_sealed_splits) != ["test"]
        or args.authority_coverage_mode != "exact"
    ):
        raise ValueError(
            "V2 controls require exact authority train+validation accounting with test sealed"
        )
    splits, metadata = load_development_dataset(
        prepare_dir=args.prepare_dir,
        split_manifest=args.split_manifest,
        split_manifest_sha256=args.split_manifest_sha256,
        authority_split_manifest=args.authority_split_manifest,
        authority_split_sha256=args.authority_split_sha256,
        authorized_source_pool_splits=args.authorized_source_pool_splits,
        authority_sealed_splits=args.authority_sealed_splits,
        authority_coverage_mode=args.authority_coverage_mode,
        strict_nw_constraint_mode=args.strict_nw_constraint_mode,
        materialized_splits=materialized_splits,
        require_authority_validation_boundary_v2=True,
        expected_instances_sha256=args.prepared_instances_sha256,
        canonical_sequence_manifest=args.canonical_sequence_manifest,
        canonical_sequence_manifest_sha256=(
            args.canonical_sequence_manifest_sha256
        ),
    )
    boundary = json.loads(args.split_manifest.expanduser().resolve().read_text(encoding="utf-8"))
    authority = json.loads(
        args.authority_split_manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    accounting = validate_boundary_v2_accounting(
        boundary,
        authority,
        args.strict_nw_constraint_mode,
        materialized_splits,
    )
    metadata["boundary_constraint_mode"] = args.strict_nw_constraint_mode
    metadata["excluded_validation_accounting"] = accounting
    return splits, metadata


def validate_boundary_v2_accounting(
    boundary: Mapping[str, Any],
    authority: Mapping[str, Any],
    expected_constraint_mode: str,
    materialized_splits: Sequence[str],
) -> dict[str, Any]:
    """Verify v2 boundary coverage while proving excluded rows are not materialized."""
    if boundary.get("format_version") != AUTHORITY_VALIDATION_BOUNDARY_FORMAT:
        raise ValueError("Controls require an authority-validation boundary v2 manifest")
    if boundary.get("intended_use") != "reused_development_confirmation_only":
        raise ValueError("Boundary intended use is incompatible with control confirmation")
    if boundary.get("constraint_mode") != expected_constraint_mode:
        raise ValueError("Boundary constraint mode differs from the explicit control mode")
    if boundary.get("valid") is not True:
        raise ValueError("Boundary does not carry a valid=true verdict")
    if boundary.get("frozen_test_identifiers_read_for_exclusion_only") is not True:
        raise PermissionError("Boundary lacks sealed-test exclusion evidence")
    if boundary.get("frozen_test_sequence_fields_accessed_normalized_hashed_or_aligned") is not False:
        raise PermissionError("Boundary reports forbidden frozen-test sequence access")
    raw_splits = boundary.get("splits")
    authority_splits = authority.get("splits")
    if not isinstance(raw_splits, Mapping) or set(raw_splits) != {
        "train",
        "validation",
        "excluded_validation",
    }:
        raise ValueError("Boundary must contain exactly three accounting splits")
    if not isinstance(authority_splits, Mapping):
        raise ValueError("Authority split manifest lacks a splits object")
    inventories = {name: [str(value) for value in raw_splits[name]] for name in raw_splits}
    if any(not values or len(values) != len(set(values)) for values in inventories.values()):
        raise ValueError("Every v2 boundary inventory must be non-empty and unique")
    train = set(inventories["train"])
    validation = set(inventories["validation"])
    excluded = set(inventories["excluded_validation"])
    if train & validation or train & excluded or validation & excluded:
        raise ValueError("V2 boundary accounting inventories must be disjoint")
    authority_train = {str(value) for value in authority_splits.get("train", [])}
    authority_validation = {str(value) for value in authority_splits.get("validation", [])}
    authority_test = {str(value) for value in authority_splits.get("test", [])}
    if train != authority_train:
        raise ValueError("V2 boundary training inventory must equal authority train")
    if validation | excluded != authority_validation:
        raise ValueError(
            "V2 validation plus excluded_validation must equal authority validation"
        )
    if (train | validation | excluded) & authority_test:
        raise PermissionError("V2 boundary overlaps the sealed authority test")
    membership = boundary.get("membership_sha256")
    if not isinstance(membership, Mapping):
        raise ValueError("V2 boundary lacks membership hashes")
    observed_membership = {
        name: _identifier_membership_sha256(values)
        for name, values in inventories.items()
    }
    if dict(membership) != observed_membership:
        raise ValueError("V2 boundary membership hash mismatch")
    selection = boundary.get("selection_rule")
    if (
        not isinstance(selection, Mapping)
        or selection.get("authority_train_moved_to_validation") is not False
        or selection.get("excluded_authority_validation_added_to_train") is not False
        or selection.get("size_target_or_random_subsampling_used") is not False
    ):
        raise ValueError("V2 boundary selection-rule firewall is incomplete")
    materialized = [str(value) for value in materialized_splits]
    if "excluded_validation" in materialized or not set(materialized).issubset(
        {"train", "validation"}
    ):
        raise PermissionError("excluded_validation can be accounted for but never materialized")
    return {
        "excluded_validation_count": len(excluded),
        "excluded_validation_membership_sha256": observed_membership[
            "excluded_validation"
        ],
        "authority_validation_accounted_count": len(validation | excluded),
        "excluded_validation_materialized": False,
        "materialized_splits": materialized,
    }


def _embedding_fingerprints(
    args: argparse.Namespace,
    records: Sequence[PairRecord],
) -> dict[str, Any]:
    return {
        "primary_antibody": side_embedding_fingerprint(
            args.antibody_embedding_dir,
            (record.antibody_key for record in records),
            "antibody",
        ),
        "shared_esm_antibody": side_embedding_fingerprint(
            args.shared_esm_embedding_dir,
            (record.antibody_key for record in records),
            "antibody",
        ),
        "esm_antigen": side_embedding_fingerprint(
            args.antigen_embedding_dir,
            (record.antigen_key for record in records),
            "antigen",
        ),
    }


def _data_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "immutable_inputs": _locked_immutable_input_arguments(args),
        "authority_split_sha256": args.authority_split_sha256,
        "authorized_source_pool_splits": list(args.authorized_source_pool_splits),
        "authority_sealed_splits": list(args.authority_sealed_splits),
        "authority_coverage_mode": args.authority_coverage_mode,
        "isolation_policy": args.isolation_policy,
        "boundary_verification_sha256": args.boundary_verification_sha256,
    }


def _common_training_dataset_contract(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Remove boundary-specific holdout fields from the common authority-train contract."""
    accounting = dataset.get("excluded_validation_accounting")
    if not isinstance(accounting, Mapping) or accounting.get(
        "excluded_validation_materialized"
    ) is not False:
        raise ValueError("Training dataset lacks excluded-validation non-materialization proof")
    return {
        "instances_sha256": dataset["instances_sha256"],
        "canonical_sequence_manifest": dataset["canonical_sequence_manifest"],
        "authority_split_sha256": dataset["authority"]["sha256"],
        "authority_train_membership_sha256": dataset["split_membership_sha256"][
            "train"
        ],
        "authority_train_raw_instance_count": dataset["raw_instance_counts"]["train"],
        "loaded_splits": ["train"],
        "validation_rows_materialized": False,
        "excluded_validation_rows_materialized": False,
        "frozen_test_records_selected": False,
    }


def _model_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "primary_antibody": {
            "artifact_format": args.antibody_artifact_format,
            "model_name": args.antibody_model_name,
            "model_weights_sha256": args.antibody_model_weights_sha256,
            "input_dim": args.antibody_input_dim,
        },
        "shared_esm": {
            "artifact_format": "esm_chunks",
            "model_name": args.esm_model_name,
            "model_weights_sha256": args.esm_model_weights_sha256,
            "input_dim": args.esm_input_dim,
        },
    }


def _validate_locked_model_contract(args: argparse.Namespace) -> None:
    """Reject model substitutions under the versioned control protocol."""
    observed = _model_contract(args)
    expected = {
        "primary_antibody": LOCKED_PRIMARY_ANTIBODY_MODEL,
        "shared_esm": LOCKED_ESM_MODEL,
    }
    if observed != expected:
        raise ValueError("Embedding model arguments differ from control protocol v1")


def _validate_locked_fit_hyperparameters(args: argparse.Namespace) -> None:
    """Reject post-registration changes to every scientific fit hyperparameter."""
    observed = {
        name: getattr(args, name) for name in LOCKED_FIT_HYPERPARAMETERS
    }
    if observed != LOCKED_FIT_HYPERPARAMETERS:
        raise ValueError("Control fit hyperparameters differ from protocol v1")


def _array_contract(arrays: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    return {
        "array_sha256": {name: _score_array_sha256(value) for name, value in arrays.items()},
        "array_dtypes": {name: str(np.asarray(value).dtype) for name, value in arrays.items()},
        "array_shapes": {name: list(np.asarray(value).shape) for name, value in arrays.items()},
    }


def _write_bundle(path: Path, arrays: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    target = path.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Control bundle already exists: {target}")
    array_contract = _array_contract(arrays)
    contract = {
        "format_version": CONTROL_BUNDLE_FORMAT,
        "scientific_scope": {
            "fit_split": "train",
            "validation_materialized": False,
            "frozen_test_used": False,
            "contains_raw_sequences": False,
        },
        "config": dict(config),
        **array_contract,
    }
    metadata = {
        "format_version": CONTROL_BUNDLE_FORMAT,
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
    }
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    _write_npz_create_only(
        target,
        {
            **arrays,
            "metadata_json_utf8": np.frombuffer(metadata_bytes, dtype=np.uint8).copy(),
        },
    )
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "contract_sha256": metadata["contract_sha256"],
        **array_contract,
    }


def _load_bundle(path: Path, expected_sha256: str) -> dict[str, Any]:
    import numpy as np

    target = path.expanduser().resolve()
    if not _is_sha256(expected_sha256) or not target.is_file():
        raise ValueError("Control bundle SHA-256 mismatch")
    bundle_bytes = target.read_bytes()
    if hashlib.sha256(bundle_bytes).hexdigest() != expected_sha256:
        raise ValueError("Control bundle SHA-256 mismatch")
    raw_members = _raw_npz_member_names(bundle_bytes, "control bundle")
    if "metadata_json_utf8" not in raw_members:
        raise ValueError("Control bundle metadata member is missing")
    with np.load(io.BytesIO(bundle_bytes), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Control bundle contains duplicate array members")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    metadata_array = arrays.pop("metadata_json_utf8", None)
    if metadata_array is None or metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
        raise ValueError("Control bundle metadata is missing or malformed")
    try:
        metadata_text = metadata_array.tobytes().decode("utf-8")
    except UnicodeError as error:
        raise ValueError("Control bundle metadata is not valid UTF-8") from error
    metadata = _strict_json_text_object(metadata_text, "control bundle metadata")
    if set(metadata) != {"format_version", "contract", "contract_sha256"}:
        raise ValueError("Control bundle metadata schema is not exact")
    if metadata.get("format_version") != CONTROL_BUNDLE_FORMAT:
        raise ValueError("Control bundle format is incompatible")
    contract = metadata.get("contract")
    expected_contract_keys = {
        "format_version",
        "scientific_scope",
        "config",
        "array_sha256",
        "array_dtypes",
        "array_shapes",
    }
    if not isinstance(contract, Mapping) or set(contract) != expected_contract_keys:
        raise ValueError("Control bundle contract schema is not exact")
    if metadata.get("contract_sha256") != canonical_sha256(contract):
        raise ValueError("Control bundle contract hash mismatch")
    if contract.get("scientific_scope") != {
        "fit_split": "train",
        "validation_materialized": False,
        "frozen_test_used": False,
        "contains_raw_sequences": False,
    }:
        raise PermissionError("Control bundle was not fitted under the train-only contract")
    observed = _array_contract(arrays)
    for key in ("array_sha256", "array_dtypes", "array_shapes"):
        if contract.get(key) != observed[key]:
            raise ValueError(f"Control bundle {key} mismatch")
    return {
        "path": target,
        "sha256": expected_sha256,
        "metadata": metadata,
        "config": contract["config"],
        "arrays": arrays,
    }


def command_fit(args: argparse.Namespace) -> None:
    """Fit low-capacity controls using authorized training rows only."""
    import numpy as np

    bundle_path = args.output_bundle.expanduser().resolve()
    summary_path = args.output_summary.expanduser().resolve()
    if bundle_path == summary_path:
        raise ValueError("Control bundle and summary paths must differ")
    if bundle_path.exists() or summary_path.exists():
        raise FileExistsError("Control fit outputs must both be new paths")
    _validate_locked_model_contract(args)
    _validate_locked_fit_hyperparameters(args)
    execution_protocol = validate_execution_protocol_reference(
        args.execution_protocol,
        args.execution_protocol_sha256,
    )
    control_protocol = validate_control_protocol_reference(
        args.control_protocol,
        args.control_protocol_sha256,
    )
    approved_boundaries = load_approved_boundary_registry(
        args.boundary_verification_manifest,
        args.boundary_verification_sha256,
        args.authority_split_sha256,
    )
    selected_approval = approved_boundaries["modes"].get(
        args.strict_nw_constraint_mode
    )
    if (
        not isinstance(selected_approval, Mapping)
        or selected_approval.get("boundary_sha256") != args.split_manifest_sha256
    ):
        raise PermissionError("Fit boundary is not in the independently approved registry")
    _locked_immutable_input_arguments(args)
    with exclusive_control_fit_process_lock(
        CENTRAL_CONTROL_FIT_LOCK_ROOT,
        args.authority_split_sha256,
        args.boundary_verification_sha256,
        bundle_path.parent,
    ) as fit_lock:
        _fit_controls_under_process_lock(
            args,
            bundle_path,
            summary_path,
            execution_protocol,
            control_protocol,
            approved_boundaries,
            fit_lock,
        )


def _fit_controls_under_process_lock(
    args: argparse.Namespace,
    bundle_path: Path,
    summary_path: Path,
    execution_protocol: Mapping[str, str],
    control_protocol: Mapping[str, str],
    approved_boundaries: Mapping[str, Any],
    fit_lock: Mapping[str, Any],
) -> None:
    """Execute the fit while the caller holds the common process-lifetime lock."""
    import numpy as np

    splits, dataset = _dataset_from_args(args, ("train",))
    train_records, duplicates_removed = deduplicate_exact_pairs(splits["train"])
    if len(train_records) < 32:
        raise ValueError("At least 32 unique training pairs are required")
    training_fingerprints = _embedding_fingerprints(args, train_records)
    locked_fingerprint_digests = _validate_locked_training_embedding_fingerprints(
        training_fingerprints
    )
    antibodies, antigens = _read_sequences(
        args.canonical_sequence_manifest, train_records
    )
    artifact_validation = _validate_training_embedding_artifacts(args, train_records)
    if _embedding_fingerprints(args, train_records) != training_fingerprints:
        raise ValueError("Training embedding artifacts changed during fit authentication")
    antibody_descriptors = {key: sequence_descriptor(value) for key, value in antibodies.items()}
    antigen_descriptors = {key: sequence_descriptor(value) for key, value in antigens.items()}
    positive_pairs = sorted({(record.antibody_key, record.antigen_key) for record in train_records})
    negative_pairs, negative_summary = deterministic_endpoint_matched_negatives(
        positive_pairs, args.negative_replicas, args.negative_seed
    )

    def matrices(builder: Any) -> tuple[Any, Any]:
        return (
            np.vstack([builder(left, right) for left, right in positive_pairs]),
            np.vstack([builder(left, right) for left, right in negative_pairs]),
        )

    feature_builders = {
        "antibody_only": lambda left, _right: antibody_descriptors[left],
        "antigen_only": lambda _left, right: antigen_descriptors[right],
        "length_composition": lambda left, right: pair_descriptor(
            antibody_descriptors[left], antigen_descriptors[right]
        ),
    }
    arrays: dict[str, Any] = make_random_head_arrays(
        args.antibody_input_dim,
        args.esm_input_dim,
        args.random_head_hidden_dim,
        args.random_head_output_dim,
        args.random_head_seed,
    )
    fit_summaries: dict[str, Any] = {}
    for name, builder in feature_builders.items():
        positive, negative = matrices(builder)
        model, summary = fit_fixed_logistic(
            positive,
            negative,
            args.logistic_l2,
            args.logistic_max_iterations,
            args.logistic_tolerance,
        )
        arrays[f"{name}_mean"] = model["mean"]
        arrays[f"{name}_scale"] = model["scale"]
        arrays[f"{name}_coefficients"] = model["coefficients"]
        fit_summaries[name] = summary
    config_without_hash = {
        "protocol_version": CONTROL_PROTOCOL_VERSION,
        "controls": list(CONTROL_NAMES),
        "data_contract": _data_contract(args),
        "approved_boundaries": approved_boundaries,
        "control_protocol_v1": control_protocol,
        "execution_protocol_v1_6": execution_protocol,
        "training_dataset": _common_training_dataset_contract(dataset),
        "model_contract": _model_contract(args),
        "training_embedding_corpora": training_fingerprints,
        "locked_training_embedding_content_manifest_sha256": (
            locked_fingerprint_digests
        ),
        "fit_process_lock": dict(fit_lock),
        "training": {
            "deduplicated_pair_count": len(train_records),
            "duplicates_removed": duplicates_removed,
            "negative_construction": negative_summary,
            "logistic_l2": args.logistic_l2,
            "logistic_max_iterations": args.logistic_max_iterations,
            "logistic_tolerance": args.logistic_tolerance,
            "random_head": {
                "seed": args.random_head_seed,
                "hidden_dim": args.random_head_hidden_dim,
                "output_dim": args.random_head_output_dim,
                "layer_norm_epsilon": 1e-5,
                "gelu": "tanh_approximation",
                "dropout": 0.0,
                "parameter_updates": 0,
            },
            "unary_tie_policy": UNARY_TIE_POLICY,
            "fit_summaries": fit_summaries,
            "embedding_artifact_validation": artifact_validation,
        },
        "source_sha256": _source_hashes(),
        "frozen_test_used": False,
        "validation_rows_materialized": False,
    }
    config = {
        **config_without_hash,
        "config_sha256": canonical_sha256(config_without_hash),
    }
    bundle = _write_bundle(bundle_path, arrays, config)
    summary = {
        "format_version": CONTROL_BUNDLE_FORMAT,
        "stage": "train_only_controls_fit_complete",
        "bundle": bundle,
        "config_sha256": config["config_sha256"],
        "validation_rows_materialized": False,
        "validation_metrics_computed": False,
        "frozen_test_used": False,
    }
    _create_exclusive_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


def _stores(args: argparse.Namespace) -> dict[str, FrozenPartnerEmbeddingStore]:
    return {
        "primary_antibody": FrozenPartnerEmbeddingStore(
            args.antibody_embedding_dir,
            "antibody",
            args.antibody_artifact_format,
            args.antibody_input_dim,
            args.antibody_model_name,
            args.antibody_model_weights_sha256,
            args.cache_gib,
        ),
        "shared_esm_antibody": FrozenPartnerEmbeddingStore(
            args.shared_esm_embedding_dir,
            "antibody",
            "esm_chunks",
            args.esm_input_dim,
            args.esm_model_name,
            args.esm_model_weights_sha256,
            args.cache_gib,
        ),
        "esm_antigen": FrozenPartnerEmbeddingStore(
            args.antigen_embedding_dir,
            "antigen",
            "esm_chunks",
            args.esm_input_dim,
            args.esm_model_name,
            args.esm_model_weights_sha256,
            args.cache_gib,
        ),
    }


def _validate_training_embedding_artifacts(
    args: argparse.Namespace,
    records: Sequence[PairRecord],
) -> dict[str, int]:
    """Load every train artifact once so model metadata and tensor shapes are verified."""
    stores = _stores(args)
    antibody_library = unique_sequences(records, "antibody")
    antigen_library = unique_sequences(records, "antigen")
    for record in antibody_library:
        stores["primary_antibody"].load(record.key, record.length, "cpu")
        stores["shared_esm_antibody"].load(record.key, record.length, "cpu")
    for record in antigen_library:
        stores["esm_antigen"].load(record.key, record.length, "cpu")
    return {
        "primary_antibody_artifacts_validated": len(antibody_library),
        "shared_esm_antibody_artifacts_validated": len(antibody_library),
        "esm_antigen_artifacts_validated": len(antigen_library),
    }


def _load_residue_library(
    records: Sequence[SequenceRecord],
    store: FrozenPartnerEmbeddingStore,
) -> dict[str, Any]:
    values = {}
    for record in records:
        tensor, _ = store.load(record.key, record.length, "cpu")
        values[record.key] = tensor.detach().to(dtype=tensor.dtype, device="cpu").numpy()
    return values


def _positive_mask(
    antibody_keys: Sequence[str],
    antigen_keys: Sequence[str],
    records: Sequence[PairRecord],
) -> Any:
    import numpy as np

    positives = {(record.antibody_key, record.antigen_key) for record in records}
    mask = np.asarray(
        [[(left, right) in positives for right in antigen_keys] for left in antibody_keys],
        dtype=bool,
    )
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        raise ValueError("Every validation anchor must have a known positive")
    return mask


def _control_scores(
    arrays: Mapping[str, Any],
    antibody_keys: Sequence[str],
    antigen_keys: Sequence[str],
    antibody_sequences: Mapping[str, str],
    antigen_sequences: Mapping[str, str],
    primary_antibody_residues: Mapping[str, Any],
    esm_antibody_residues: Mapping[str, Any],
    esm_antigen_residues: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    raw_antibody = np.vstack(
        [np.asarray(esm_antibody_residues[key], dtype=np.float64).mean(axis=0) for key in antibody_keys]
    )
    raw_antigen = np.vstack(
        [np.asarray(esm_antigen_residues[key], dtype=np.float64).mean(axis=0) for key in antigen_keys]
    )
    random_antibody = np.vstack(
        [random_head_pool(primary_antibody_residues[key], arrays, "random_ab") for key in antibody_keys]
    )
    random_antigen = np.vstack(
        [random_head_pool(esm_antigen_residues[key], arrays, "random_ag") for key in antigen_keys]
    )
    antibody_descriptors = np.vstack(
        [sequence_descriptor(antibody_sequences[key]) for key in antibody_keys]
    )
    antigen_descriptors = np.vstack(
        [sequence_descriptor(antigen_sequences[key]) for key in antigen_keys]
    )
    antibody_model = {
        key: arrays[f"antibody_only_{key}"] for key in ("mean", "scale", "coefficients")
    }
    antigen_model = {
        key: arrays[f"antigen_only_{key}"] for key in ("mean", "scale", "coefficients")
    }
    pair_model = {
        key: arrays[f"length_composition_{key}"] for key in ("mean", "scale", "coefficients")
    }
    antibody_logits = apply_fixed_logistic(antibody_descriptors, antibody_model)
    antigen_logits = apply_fixed_logistic(antigen_descriptors, antigen_model)
    antibody_only_scores, antigen_only_scores = unary_score_matrices(
        antibody_logits,
        antigen_logits,
        antibody_keys,
        antigen_keys,
    )
    pair_features = np.vstack(
        [
            pair_descriptor(antibody_descriptors[i], antigen_descriptors[j])
            for i in range(len(antibody_keys))
            for j in range(len(antigen_keys))
        ]
    )
    pair_logits = apply_fixed_logistic(pair_features, pair_model).reshape(
        len(antibody_keys), len(antigen_keys)
    )
    return {
        "raw_shared_esm_mean_cosine": _cosine_matrix(raw_antibody, raw_antigen),
        "random_untrained_calm_head": _cosine_matrix(random_antibody, random_antigen),
        "antibody_only_unary": antibody_only_scores,
        "antigen_only_unary": antigen_only_scores,
        "length_composition_logistic": pair_logits.astype(np.float32),
    }


def _verify_runtime_contract(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    approved_boundaries: Mapping[str, Any],
    execution_protocol: Mapping[str, str],
    control_protocol: Mapping[str, str],
) -> None:
    unhashed = {key: value for key, value in config.items() if key != "config_sha256"}
    if config.get("config_sha256") != canonical_sha256(unhashed):
        raise ValueError("Control configuration hash mismatch")
    if config.get("source_sha256") != _source_hashes():
        raise ValueError("Control source code differs from fit provenance")
    if config.get("data_contract") != _data_contract(args):
        raise ValueError("Evaluation data contract differs from the fitted control bundle")
    if config.get("approved_boundaries") != approved_boundaries:
        raise ValueError("Approved boundary registry differs from control-fit provenance")
    if config.get("execution_protocol_v1_6") != execution_protocol:
        raise ValueError("Physical execution protocol differs from control-fit provenance")
    if config.get("control_protocol_v1") != control_protocol:
        raise ValueError("Physical controls protocol differs from control-fit provenance")
    immutable_inputs = _locked_immutable_input_arguments(args)
    if config.get("data_contract", {}).get("immutable_inputs") != immutable_inputs:
        raise ValueError("Immutable v1.6 inputs differ from control-fit provenance")
    selected = approved_boundaries.get("modes", {}).get(
        args.strict_nw_constraint_mode
    )
    if (
        not isinstance(selected, Mapping)
        or selected.get("boundary_sha256") != args.split_manifest_sha256
    ):
        raise PermissionError("Evaluation boundary is not independently approved")
    if config.get("model_contract") != _model_contract(args):
        raise ValueError("Evaluation embedding-model contract differs from fit")
    if config.get("locked_training_embedding_content_manifest_sha256") != dict(
        LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256
    ):
        raise ValueError("Control bundle lacks the exact final v1.6 embedding lock")
    if config.get("validation_rows_materialized") is not False or config.get("frozen_test_used") is not False:
        raise PermissionError("Control bundle violates the train-only fit firewall")


def _locked_submission_evidence(
    args: argparse.Namespace,
    dataset: Mapping[str, Any],
    config: Mapping[str, Any],
    execution_protocol: Mapping[str, str],
    control_protocol: Mapping[str, str],
    approved_boundaries: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose the physically reauthenticated v1.6 evidence used by one evaluation."""
    selected = approved_boundaries["modes"][args.strict_nw_constraint_mode]
    immutable_inputs = _locked_immutable_input_arguments(args)
    if dataset.get("instances_sha256") != immutable_inputs["prepared_instances"]["sha256"]:
        raise ValueError("Evaluation dataset is not the locked prepared instance table")
    if dataset.get("canonical_sequence_manifest") != immutable_inputs[
        "canonical_sequence_manifest"
    ]:
        raise ValueError("Evaluation dataset is not bound to the locked canonical manifest")
    return {
        "format_version": "sabdab2_control_locked_submission_evidence_v1",
        "control_protocol_v1": dict(control_protocol),
        "execution_protocol_v1_6": dict(execution_protocol),
        "prepared_instances": immutable_inputs["prepared_instances"],
        "canonical_sequence_manifest": immutable_inputs[
            "canonical_sequence_manifest"
        ],
        "training_embedding_content_manifest_sha256": dict(
            config["locked_training_embedding_content_manifest_sha256"]
        ),
        "boundary_verification_v2": {
            "path": str(
                args.boundary_verification_manifest.expanduser().resolve()
            ),
            "sha256": args.boundary_verification_sha256,
        },
        "authority_split": {
            "path": dataset["authority"]["path"],
            "sha256": dataset["authority"]["sha256"],
        },
        "selected_boundary": {
            "constraint_mode": args.strict_nw_constraint_mode,
            "path": dataset["split_manifest"],
            "sha256": dataset["split_manifest_sha256"],
            "scientific_payload_sha256": selected.get(
                "boundary_scientific_payload_sha256"
            ),
        },
        "required_boundary_modes": list(APPROVED_BOUNDARY_MODES),
        "control_suite_completion_required": True,
        "frozen_test_used": False,
    }


def _evaluation_context(
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = evaluation["dataset"]
    authority = dataset["authority"]
    evidence = evaluation["locked_submission_evidence"]
    ledger = evaluation["one_shot_ledger"]
    return {
        "evaluation_format_version": evaluation["format_version"],
        "stage": evaluation["stage"],
        "split": evaluation["split"],
        "frozen_test_used": evaluation["frozen_test_used"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "config_sha256": evaluation["config_sha256"],
        "instances_sha256": dataset["instances_sha256"],
        "split_manifest_sha256": dataset["split_manifest_sha256"],
        "split_format_version": dataset["split_format_version"],
        "split_intended_use": dataset["split_intended_use"],
        "strict_nw_constraint_mode": dataset.get("strict_nw_constraint_mode"),
        "authority_split_sha256": authority["sha256"],
        "authorized_source_pool_splits": authority["authorized_source_pool_splits"],
        "authority_sealed_splits": authority["sealed_source_splits"],
        "authority_selected_sealed_overlap_count": authority["selected_sealed_overlap_count"],
        "evaluation_embedding_corpora_sha256": canonical_sha256(evaluation["embedding_corpora"]),
        "one_shot_ledger_key": ledger["canonical_key"],
        "one_shot_ledger_path": ledger["canonical_path"],
        "control_bundle_sha256": evaluation["control_bundle_sha256"],
        "control_evaluation_identity_sha256": ledger["canonical_key"],
        "locked_submission_evidence_sha256": canonical_sha256(evidence),
        "control_protocol_v1": dict(evidence["control_protocol_v1"]),
        "control_protocol_v1_sha256": evidence["control_protocol_v1"]["sha256"],
        "execution_protocol_v1_6_sha256": evidence[
            "execution_protocol_v1_6"
        ]["sha256"],
        "canonical_sequence_manifest_sha256": evidence[
            "canonical_sequence_manifest"
        ]["sha256"],
        "training_embedding_content_manifest_sha256": canonical_sha256(
            evidence["training_embedding_content_manifest_sha256"]
        ),
        "required_boundary_modes": list(evidence["required_boundary_modes"]),
        "control_suite_completion_required": True,
    }


def command_evaluate(args: argparse.Namespace) -> None:
    """Evaluate every locked control in one development-only holdout read."""
    _validate_locked_model_contract(args)
    _locked_immutable_input_arguments(args)
    execution_protocol = validate_execution_protocol_reference(
        args.execution_protocol,
        args.execution_protocol_sha256,
    )
    control_protocol = validate_control_protocol_reference(
        args.control_protocol,
        args.control_protocol_sha256,
    )
    bundle = _load_bundle(args.control_bundle, args.control_bundle_sha256)
    config = bundle["config"]
    approved_boundaries = load_approved_boundary_registry(
        args.boundary_verification_manifest,
        args.boundary_verification_sha256,
        args.authority_split_sha256,
    )
    _verify_runtime_contract(
        args,
        config,
        approved_boundaries,
        execution_protocol,
        control_protocol,
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"One-shot control output directory already exists: {output_dir}")
    canonical_registry = CENTRAL_CONTROL_LEDGER_ROOT.expanduser().resolve()
    if output_dir == canonical_registry or canonical_registry in output_dir.parents:
        raise ValueError("Control outputs cannot overlap the central ledger registry")
    ledger, evaluation_identity, ledger_contract = reserve_control_evaluation(
        CENTRAL_CONTROL_LEDGER_ROOT,
        bundle["sha256"],
        args.split_manifest_sha256,
        args.authority_split_sha256,
        output_dir,
    )
    splits, dataset = _dataset_from_args(args, ("train", "validation"))
    train_records, train_removed = deduplicate_exact_pairs(splits["train"])
    validation_records, validation_removed = deduplicate_exact_pairs(splits["validation"])
    isolation = audit_isolation_policy(train_records, validation_records, args.isolation_policy)
    training_dataset = config["training_dataset"]
    if dataset["instances_sha256"] != training_dataset["instances_sha256"]:
        raise ValueError("Prepared instances changed after control fitting")
    if (
        dataset["split_membership_sha256"]["train"]
        != training_dataset["authority_train_membership_sha256"]
        or dataset["raw_instance_counts"]["train"]
        != training_dataset["authority_train_raw_instance_count"]
    ):
        raise ValueError("Authority-train membership changed after control fitting")
    training_fingerprints = _embedding_fingerprints(args, train_records)
    _validate_locked_training_embedding_fingerprints(training_fingerprints)
    if training_fingerprints != config["training_embedding_corpora"]:
        raise ValueError("Training embedding corpora changed after control fitting")
    validation_fingerprints = _embedding_fingerprints(args, validation_records)
    antibody_sequences, antigen_sequences = _read_sequences(
        args.canonical_sequence_manifest, validation_records
    )
    antibody_library = unique_sequences(validation_records, "antibody")
    antigen_library = unique_sequences(validation_records, "antigen")
    antibody_keys = [record.key for record in antibody_library]
    antigen_keys = [record.key for record in antigen_library]
    positive_mask = _positive_mask(antibody_keys, antigen_keys, validation_records)
    stores = _stores(args)
    primary_antibody = _load_residue_library(antibody_library, stores["primary_antibody"])
    esm_antibody = _load_residue_library(antibody_library, stores["shared_esm_antibody"])
    esm_antigen = _load_residue_library(antigen_library, stores["esm_antigen"])
    if _embedding_fingerprints(args, validation_records) != validation_fingerprints:
        raise ValueError("Validation embedding artifacts changed during authenticated loading")
    scores_by_control = _control_scores(
        bundle["arrays"],
        antibody_keys,
        antigen_keys,
        antibody_sequences,
        antigen_sequences,
        primary_antibody,
        esm_antibody,
        esm_antigen,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    output_records: dict[str, Any] = {}
    embedding_corpora = {
        "training": training_fingerprints,
        "one_shot_validation": validation_fingerprints,
    }
    locked_evidence = _locked_submission_evidence(
        args,
        dataset,
        config,
        execution_protocol,
        control_protocol,
        approved_boundaries,
    )
    for control_name in CONTROL_NAMES:
        control_spec = {
            "protocol_version": CONTROL_PROTOCOL_VERSION,
            "name": control_name,
            "bundle_config_sha256": config["config_sha256"],
        }
        if control_name in {"antibody_only_unary", "antigen_only_unary"}:
            control_spec["tie_policy"] = UNARY_TIE_POLICY
        control_config_sha256 = canonical_sha256(control_spec)
        metrics = metrics_from_scores(
            scores_by_control[control_name], positive_mask, antibody_keys, antigen_keys
        )
        evaluation: dict[str, Any] = {
            "format_version": EVALUATION_FORMAT_VERSION,
            "stage": "full_library_development_validation",
            "split": "validation",
            "frozen_test_used": False,
            "checkpoint": str(bundle["path"]),
            "checkpoint_sha256": bundle["sha256"],
            "control_bundle_sha256": bundle["sha256"],
            "config_sha256": control_config_sha256,
            "dataset": dataset,
            "embedding_corpora": embedding_corpora,
            "exact_duplicate_pairs_removed": validation_removed,
            "train_exact_duplicate_pairs_removed": train_removed,
            "train_validation_isolation": isolation,
            "metrics": metrics,
            "checkpoint_run_mode": "train_only_control_bundle",
            "one_shot_evaluation": True,
            "one_shot_ledger": {
                "format_version": CONTROL_LEDGER_FORMAT,
                "canonical_key": evaluation_identity,
                "canonical_path": str(ledger),
                "registry_dir": str(ledger.parent),
            },
            "control": control_spec,
            "control_protocol_v1": control_protocol,
            "execution_protocol_v1_6": execution_protocol,
            "locked_submission_evidence": locked_evidence,
        }
        score_path = output_dir / f"{control_name}.scores.npz"
        evaluation_path = output_dir / f"{control_name}.evaluation.json"
        evaluation["score_artifact"] = _write_score_artifact_create_only(
            score_path,
            {
                "similarity_matrix": scores_by_control[control_name],
                "positive_mask": positive_mask,
                "antibody_keys": antibody_keys,
                "antigen_keys": antigen_keys,
            },
            _evaluation_context(evaluation),
        )
        _create_exclusive_json(evaluation_path, evaluation)
        output_records[control_name] = {
            "evaluation_json": str(evaluation_path),
            "evaluation_json_sha256": sha256_file(evaluation_path),
            "score_artifact": str(score_path),
            "score_artifact_sha256": evaluation["score_artifact"]["sha256"],
            "score_contract_sha256": evaluation["score_artifact"]["contract_sha256"],
            "bidirectional_mrr": metrics["bidirectional_mrr"],
        }
    atomic_write_json(
        ledger,
        {
            **ledger_contract,
            "evaluation_identity_sha256": evaluation_identity,
            "status": "completed",
            "requested_output_dir": str(output_dir),
            "control_protocol_v1": control_protocol,
            "execution_protocol_v1_6": execution_protocol,
            "locked_submission_evidence_sha256": canonical_sha256(locked_evidence),
            "control_outputs": output_records,
        },
    )
    result = {
        "evaluation_identity_sha256": evaluation_identity,
        "central_ledger": str(ledger),
        "central_ledger_sha256": sha256_file(ledger),
        "control_outputs": output_records,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


def _verify_control_evaluation_output(
    evaluation_reference: Mapping[str, Any],
    score_reference: Mapping[str, Any],
    control_name: str,
    mode: str,
    bundle: Mapping[str, Any],
    ledger_path: Path,
    ledger_identity: str,
    expected_evidence_sha256: str,
    expected_output_dir: Path,
) -> dict[str, Any]:
    """Physically authenticate one control evaluation and its score contract."""
    from sabdab2_asymmetric_retrieval_statistics import load_score_artifact

    evaluation_physical, evaluation = _strict_hashed_json_reference(
        evaluation_reference,
        f"{mode}.{control_name}.evaluation_json",
    )
    score_core_reference = {
        "path": score_reference.get("path"),
        "sha256": score_reference.get("sha256"),
    }
    score_physical = _physical_hashed_reference(
        score_core_reference,
        f"{mode}.{control_name}.score_artifact",
    )
    evaluation_path = Path(evaluation_physical["path"])
    score_path = Path(score_physical["path"])
    if evaluation_path.parent != expected_output_dir or score_path.parent != expected_output_dir:
        raise ValueError("Control artifacts are outside their declared boundary output directory")
    if (
        evaluation.get("format_version") != EVALUATION_FORMAT_VERSION
        or evaluation.get("stage") != "full_library_development_validation"
        or evaluation.get("split") != "validation"
        or evaluation.get("frozen_test_used") is not False
        or evaluation.get("checkpoint_run_mode") != "train_only_control_bundle"
        or evaluation.get("one_shot_evaluation") is not True
        or evaluation.get("control_bundle_sha256") != bundle["sha256"]
        or evaluation.get("checkpoint_sha256") != bundle["sha256"]
    ):
        raise PermissionError("Control evaluation violates the locked confirmation scope")
    if evaluation.get("control", {}).get("name") != control_name:
        raise ValueError("Control evaluation name differs from its ledger slot")
    control_spec = evaluation["control"]
    if (
        control_spec.get("bundle_config_sha256") != bundle["config"]["config_sha256"]
        or evaluation.get("config_sha256") != canonical_sha256(control_spec)
        or Path(str(evaluation.get("checkpoint", ""))).expanduser().resolve()
        != Path(bundle["path"]).expanduser().resolve()
    ):
        raise ValueError("Control evaluation configuration or bundle path is inconsistent")
    dataset = evaluation.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get(
        "strict_nw_constraint_mode"
    ) != mode:
        raise ValueError("Control evaluation uses a different boundary mode")
    ledger = evaluation.get("one_shot_ledger")
    if (
        not isinstance(ledger, Mapping)
        or ledger.get("format_version") != CONTROL_LEDGER_FORMAT
        or ledger.get("canonical_key") != ledger_identity
        or Path(str(ledger.get("canonical_path", ""))).expanduser().resolve()
        != ledger_path
    ):
        raise ValueError("Control evaluation is not bound to its completed central ledger")
    evidence = evaluation.get("locked_submission_evidence")
    if not isinstance(evidence, Mapping) or canonical_sha256(evidence) != expected_evidence_sha256:
        raise ValueError("Control evaluation locked-submission evidence hash mismatch")
    expected_evidence_keys = {
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
    if set(evidence) != expected_evidence_keys:
        raise ValueError("Control evaluation locked evidence schema is not exact")
    _physical_hashed_reference(
        evidence["control_protocol_v1"],
        "controls protocol v1",
        CONTROL_PROTOCOL_V1_SHA256,
    )
    _physical_hashed_reference(
        evidence["execution_protocol_v1_6"],
        "execution protocol v1.6",
        EXECUTION_PROTOCOL_V1_6_SHA256,
    )
    _physical_hashed_reference(
        evidence["prepared_instances"],
        "prepared instances",
        LOCKED_PREPARED_INSTANCES_SHA256,
    )
    _physical_hashed_reference(
        evidence["canonical_sequence_manifest"],
        "canonical sequence manifest",
        LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256,
    )
    _physical_hashed_reference(
        evidence["boundary_verification_v2"],
        "boundary verification v2",
        LOCKED_BOUNDARY_VERIFICATION_SHA256,
    )
    _physical_hashed_reference(
        evidence["authority_split"],
        "authority split",
        LOCKED_AUTHORITY_SPLIT_SHA256,
    )
    selected_boundary = evidence["selected_boundary"]
    if not isinstance(selected_boundary, Mapping) or set(selected_boundary) != {
        "constraint_mode",
        "path",
        "sha256",
        "scientific_payload_sha256",
    }:
        raise ValueError("Selected control boundary reference schema is not exact")
    _physical_hashed_reference(
        {
            "path": selected_boundary["path"],
            "sha256": selected_boundary["sha256"],
        },
        "selected control boundary",
    )
    if (
        selected_boundary["constraint_mode"] != mode
        or dataset.get("split_manifest_sha256") != selected_boundary["sha256"]
        or dataset.get("instances_sha256")
        != evidence["prepared_instances"]["sha256"]
        or dataset.get("canonical_sequence_manifest")
        != evidence["canonical_sequence_manifest"]
        or dataset.get("authority", {}).get("sha256")
        != evidence["authority_split"]["sha256"]
        or evaluation.get("execution_protocol_v1_6")
        != evidence["execution_protocol_v1_6"]
        or evaluation.get("control_protocol_v1")
        != evidence["control_protocol_v1"]
        or evidence["required_boundary_modes"] != list(APPROVED_BOUNDARY_MODES)
        or evidence["control_suite_completion_required"] is not True
        or evidence["frozen_test_used"] is not False
        or evidence["training_embedding_content_manifest_sha256"]
        != LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256
    ):
        raise PermissionError("Control evaluation locked evidence violates v1.6")
    score = load_score_artifact(score_path)
    if score["sha256"] != score_physical["sha256"]:
        raise ValueError("Loaded control score artifact SHA-256 mismatch")
    if score_reference.get("contract_sha256") != score["metadata"]["contract_sha256"]:
        raise ValueError("Control score contract SHA-256 mismatch")
    evaluation_score = evaluation.get("score_artifact")
    if (
        not isinstance(evaluation_score, Mapping)
        or evaluation_score.get("sha256") != score["sha256"]
        or evaluation_score.get("contract_sha256")
        != score["metadata"]["contract_sha256"]
        or Path(str(evaluation_score.get("path_at_creation", ""))).expanduser().resolve()
        != score_path
    ):
        raise ValueError("Control evaluation references a different score artifact")
    if score["metadata"]["contract"].get("evaluation_context") != _evaluation_context(
        evaluation
    ):
        raise ValueError("Control score context does not authenticate v1.6 evaluation")
    observed_metrics = metrics_from_scores(
        score["similarities"],
        score["positive_mask"],
        score["antibody_keys"],
        score["antigen_keys"],
    )
    if evaluation.get("metrics") != observed_metrics:
        raise ValueError("Control evaluation metrics do not reproduce from score bytes")
    return {
        "evaluation_json": evaluation_physical,
        "score_artifact": {
            **score_physical,
            "contract_sha256": score["metadata"]["contract_sha256"],
        },
        "evaluation": evaluation,
    }


def _verify_completed_control_boundary(
    registry_root: Path,
    mode: str,
    ledger_reference: Mapping[str, Any],
    bundle: Mapping[str, Any],
    execution_protocol: Mapping[str, str],
) -> dict[str, Any]:
    """Verify one completed all-five-control boundary ledger and every output byte."""
    ledger_physical, ledger = _strict_hashed_json_reference(
        ledger_reference,
        f"{mode} control ledger",
    )
    ledger_path = Path(ledger_physical["path"])
    registry = registry_root.expanduser().resolve()
    if ledger_path.parent != registry:
        raise ValueError("Control boundary ledger is outside the canonical registry")
    expected_keys = {
        "format_version",
        "control_suite_version",
        "control_bundle_sha256",
        "split_manifest_sha256",
        "authority_split_sha256",
        "control_protocol_v1_sha256",
        "execution_protocol_v1_6_sha256",
        "prepared_instances_sha256",
        "canonical_sequence_manifest_sha256",
        "training_embedding_content_manifest_sha256",
        "split",
        "controls",
        "frozen_test_used",
        "evaluation_identity_sha256",
        "status",
        "requested_output_dir",
        "control_protocol_v1",
        "execution_protocol_v1_6",
        "locked_submission_evidence_sha256",
        "control_outputs",
    }
    if set(ledger) != expected_keys:
        raise ValueError("Completed control ledger schema is not exact")
    split_sha256 = str(ledger.get("split_manifest_sha256", ""))
    identity, identity_contract = control_evaluation_identity(
        bundle["sha256"],
        split_sha256,
        str(ledger.get("authority_split_sha256", "")),
    )
    if (
        ledger.get("status") != "completed"
        or ledger.get("evaluation_identity_sha256") != identity
        or ledger_path != registry / f"{identity}.json"
        or any(ledger.get(key) != value for key, value in identity_contract.items())
        or ledger.get("control_protocol_v1")
        != bundle["config"]["control_protocol_v1"]
        or ledger.get("execution_protocol_v1_6") != execution_protocol
    ):
        raise ValueError("Completed control ledger identity or v1.6 contract mismatch")
    approved = bundle["config"]["approved_boundaries"]["modes"][mode]
    if split_sha256 != approved["boundary_sha256"]:
        raise ValueError("Completed control ledger uses the wrong approved boundary")
    output_dir_value = ledger.get("requested_output_dir")
    if not isinstance(output_dir_value, str) or not Path(
        output_dir_value
    ).expanduser().is_absolute():
        raise ValueError("Completed control ledger output_dir must be absolute")
    output_dir = Path(output_dir_value).expanduser().resolve()
    if not output_dir.is_dir() or output_dir == registry or registry in output_dir.parents:
        raise ValueError("Control output directory is missing or overlaps the registry")
    outputs = ledger.get("control_outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != set(CONTROL_NAMES):
        raise ValueError("Completed control ledger must contain exactly five controls")
    normalized_controls: dict[str, Any] = {}
    representative_evidence: Mapping[str, Any] | None = None
    for control_name in CONTROL_NAMES:
        record = outputs[control_name]
        if not isinstance(record, Mapping) or set(record) != {
            "evaluation_json",
            "evaluation_json_sha256",
            "score_artifact",
            "score_artifact_sha256",
            "score_contract_sha256",
            "bidirectional_mrr",
        }:
            raise ValueError("Completed control output record schema is not exact")
        verified = _verify_control_evaluation_output(
            {
                "path": record["evaluation_json"],
                "sha256": record["evaluation_json_sha256"],
            },
            {
                "path": record["score_artifact"],
                "sha256": record["score_artifact_sha256"],
                "contract_sha256": record["score_contract_sha256"],
            },
            control_name,
            mode,
            bundle,
            ledger_path,
            identity,
            str(ledger["locked_submission_evidence_sha256"]),
            output_dir,
        )
        reported_mrr = verified["evaluation"]["metrics"]["bidirectional_mrr"]
        if float(record["bidirectional_mrr"]) != float(reported_mrr):
            raise ValueError("Control ledger bidirectional MRR differs from evaluation")
        normalized_controls[control_name] = {
            "evaluation_json": verified["evaluation_json"],
            "score_artifact": verified["score_artifact"],
        }
        if representative_evidence is None:
            representative_evidence = verified["evaluation"][
                "locked_submission_evidence"
            ]
    if representative_evidence is None:
        raise AssertionError("Completed control suite unexpectedly contains no controls")
    evidence = representative_evidence
    selected = evidence["selected_boundary"]
    return {
        "split_manifest": {
            "path": str(Path(selected["path"]).expanduser().resolve()),
            "sha256": selected["sha256"],
        },
        "boundary_scientific_payload_sha256": selected[
            "scientific_payload_sha256"
        ],
        "ledger": {
            **ledger_physical,
            "canonical_key": identity,
        },
        "output_dir": str(output_dir),
        "controls": normalized_controls,
        "representative_evidence": evidence,
    }


def finalize_control_suite(
    registry_root: Path,
    control_bundle: Path,
    control_bundle_sha256: str,
    control_protocol_path: Path,
    control_protocol_sha256: str,
    execution_protocol_path: Path,
    execution_protocol_sha256: str,
    ledger_references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Create one central receipt only after both complete control boundaries verify."""
    if set(ledger_references) != set(APPROVED_BOUNDARY_MODES):
        raise ValueError("Control suite finalization requires exactly both boundary modes")
    control_protocol = validate_control_protocol_reference(
        control_protocol_path,
        control_protocol_sha256,
    )
    execution_protocol = validate_execution_protocol_reference(
        execution_protocol_path,
        execution_protocol_sha256,
    )
    bundle = _load_bundle(control_bundle, control_bundle_sha256)
    config = bundle["config"]
    unhashed_config = {key: value for key, value in config.items() if key != "config_sha256"}
    if config.get("config_sha256") != canonical_sha256(unhashed_config):
        raise ValueError("Control bundle configuration SHA-256 mismatch")
    if config.get("execution_protocol_v1_6") != execution_protocol:
        raise ValueError("Control bundle does not bind the supplied final v1.6 protocol")
    if config.get("control_protocol_v1") != control_protocol:
        raise ValueError("Control bundle does not bind the supplied controls protocol")
    if config.get("locked_training_embedding_content_manifest_sha256") != dict(
        LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256
    ):
        raise ValueError("Control bundle lacks exact locked embedding fingerprints")
    boundaries = {
        mode: _verify_completed_control_boundary(
            registry_root,
            mode,
            ledger_references[mode],
            bundle,
            execution_protocol,
        )
        for mode in APPROVED_BOUNDARY_MODES
    }
    evidence_values = [value.pop("representative_evidence") for value in boundaries.values()]
    first_evidence, second_evidence = evidence_values
    for key in (
        "control_protocol_v1",
        "execution_protocol_v1_6",
        "prepared_instances",
        "canonical_sequence_manifest",
        "training_embedding_content_manifest_sha256",
        "boundary_verification_v2",
        "authority_split",
        "required_boundary_modes",
        "control_suite_completion_required",
        "frozen_test_used",
    ):
        if first_evidence[key] != second_evidence[key]:
            raise ValueError(f"Control boundaries disagree on common evidence field {key}")
    artifact_paths: list[str] = []
    for boundary in boundaries.values():
        for record in boundary["controls"].values():
            artifact_paths.extend(
                [record["evaluation_json"]["path"], record["score_artifact"]["path"]]
            )
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ValueError("All two-boundary control evaluation and score paths must be distinct")
    bundle_path = Path(bundle["path"]).expanduser().resolve()
    if str(bundle_path) in artifact_paths:
        raise ValueError("A control artifact path cannot overlap the fitted bundle")
    output_directories = [Path(value["output_dir"]) for value in boundaries.values()]
    if (
        len(set(output_directories)) != len(APPROVED_BOUNDARY_MODES)
        or bundle_path.parent in output_directories
    ):
        raise ValueError("Boundary output directories must be distinct from each other and fit")
    immutable_inputs = {
        "prepared_instances": _physical_hashed_reference(
            first_evidence["prepared_instances"],
            "suite prepared instances",
            LOCKED_PREPARED_INSTANCES_SHA256,
        ),
        "canonical_sequence_manifest": _physical_hashed_reference(
            first_evidence["canonical_sequence_manifest"],
            "suite canonical sequence manifest",
            LOCKED_CANONICAL_SEQUENCE_MANIFEST_SHA256,
        ),
        "training_embedding_content_manifest_sha256": dict(
            LOCKED_TRAINING_EMBEDDING_CONTENT_MANIFEST_SHA256
        ),
    }
    receipt_without_identity = {
        "format_version": CONTROL_SUITE_COMPLETION_FORMAT,
        "status": "completed",
        "control_suite_version": CONTROL_PROTOCOL_VERSION,
        "control_bundle": {
            "path": str(bundle_path),
            "sha256": bundle["sha256"],
            "config_sha256": config["config_sha256"],
        },
        "control_protocol_v1": _physical_hashed_reference(
            first_evidence["control_protocol_v1"],
            "suite controls protocol v1",
            CONTROL_PROTOCOL_V1_SHA256,
        ),
        "execution_protocol_v1_6": _physical_hashed_reference(
            first_evidence["execution_protocol_v1_6"],
            "suite execution protocol v1.6",
            EXECUTION_PROTOCOL_V1_6_SHA256,
        ),
        "boundary_verification_v2": _physical_hashed_reference(
            first_evidence["boundary_verification_v2"],
            "suite boundary verification v2",
            LOCKED_BOUNDARY_VERIFICATION_SHA256,
        ),
        "authority_split": _physical_hashed_reference(
            first_evidence["authority_split"],
            "suite authority split",
            LOCKED_AUTHORITY_SPLIT_SHA256,
        ),
        "immutable_inputs": immutable_inputs,
        "required_boundary_modes": list(APPROVED_BOUNDARY_MODES),
        "boundaries": boundaries,
        "frozen_test_used": False,
    }
    completion_identity = canonical_sha256(receipt_without_identity)
    receipt = {
        **receipt_without_identity,
        "completion_identity_sha256": completion_identity,
    }
    completion_path = (
        registry_root.expanduser().resolve()
        / "suite-completions"
        / f"{bundle['sha256']}.json"
    )
    _create_exclusive_json(completion_path, receipt)
    return {
        "path": str(completion_path),
        "sha256": sha256_file(completion_path),
        "completion_identity_sha256": completion_identity,
    }


def command_finalize_suite(args: argparse.Namespace) -> None:
    """Verify both boundary suites and emit their canonical completion receipt."""
    ledger_references = {
        "concat_antibody_and_antigen": {
            "path": str(args.concat_ledger.expanduser().resolve()),
            "sha256": args.concat_ledger_sha256,
        },
        "concat_antibody_antigen_plus_vh_vl": {
            "path": str(args.concat_vh_vl_ledger.expanduser().resolve()),
            "sha256": args.concat_vh_vl_ledger_sha256,
        },
    }
    result = finalize_control_suite(
        CENTRAL_CONTROL_LEDGER_ROOT,
        args.control_bundle,
        args.control_bundle_sha256,
        args.control_protocol,
        args.control_protocol_sha256,
        args.execution_protocol,
        args.execution_protocol_sha256,
        ledger_references,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--prepared-instances-sha256", required=True)
    parser.add_argument("--canonical-sequence-manifest", type=Path, required=True)
    parser.add_argument("--canonical-sequence-manifest-sha256", required=True)
    parser.add_argument("--antibody-embedding-dir", type=Path, required=True)
    parser.add_argument("--shared-esm-embedding-dir", type=Path, required=True)
    parser.add_argument("--antigen-embedding-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--authority-split-manifest", type=Path, required=True)
    parser.add_argument("--authority-split-sha256", required=True)
    parser.add_argument("--boundary-verification-manifest", type=Path, required=True)
    parser.add_argument("--boundary-verification-sha256", required=True)
    parser.add_argument("--control-protocol", type=Path, required=True)
    parser.add_argument("--control-protocol-sha256", required=True)
    parser.add_argument("--execution-protocol", type=Path, required=True)
    parser.add_argument("--execution-protocol-sha256", required=True)
    parser.add_argument("--authorized-source-pool-splits", nargs="+", required=True)
    parser.add_argument("--authority-sealed-splits", nargs="+", required=True)
    parser.add_argument("--authority-coverage-mode", choices=("exact", "subset"), default="exact")
    parser.add_argument("--strict-nw-constraint-mode", required=True)
    parser.add_argument("--isolation-policy", choices=("double_exact", "antigen_exact"), default="double_exact")
    parser.add_argument("--antibody-artifact-format", choices=("antiberty", "esm_chunks"), default="antiberty")
    parser.add_argument("--antibody-model-name", required=True)
    parser.add_argument("--antibody-model-weights-sha256", required=True)
    parser.add_argument("--antibody-input-dim", type=int, default=512)
    parser.add_argument("--esm-model-name", required=True)
    parser.add_argument("--esm-model-weights-sha256", required=True)
    parser.add_argument("--esm-input-dim", type=int, default=1280)
    parser.add_argument("--cache-gib", type=float, default=4.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    _add_data_arguments(fit)
    fit.add_argument("--output-bundle", type=Path, required=True)
    fit.add_argument("--output-summary", type=Path, required=True)
    fit.add_argument("--negative-replicas", type=int, default=5)
    fit.add_argument("--negative-seed", type=int, default=42017)
    fit.add_argument("--logistic-l2", type=float, default=0.01)
    fit.add_argument("--logistic-max-iterations", type=int, default=50)
    fit.add_argument("--logistic-tolerance", type=float, default=1e-9)
    fit.add_argument("--random-head-seed", type=int, default=20260723)
    fit.add_argument("--random-head-hidden-dim", type=int, default=256)
    fit.add_argument("--random-head-output-dim", type=int, default=128)
    fit.set_defaults(handler=command_fit)
    evaluate = commands.add_parser("evaluate")
    _add_data_arguments(evaluate)
    evaluate.add_argument("--control-bundle", type=Path, required=True)
    evaluate.add_argument("--control-bundle-sha256", required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.set_defaults(handler=command_evaluate)
    finalize = commands.add_parser("finalize-suite")
    finalize.add_argument("--control-bundle", type=Path, required=True)
    finalize.add_argument("--control-bundle-sha256", required=True)
    finalize.add_argument("--control-protocol", type=Path, required=True)
    finalize.add_argument("--control-protocol-sha256", required=True)
    finalize.add_argument("--execution-protocol", type=Path, required=True)
    finalize.add_argument("--execution-protocol-sha256", required=True)
    finalize.add_argument("--concat-ledger", type=Path, required=True)
    finalize.add_argument("--concat-ledger-sha256", required=True)
    finalize.add_argument("--concat-vh-vl-ledger", type=Path, required=True)
    finalize.add_argument("--concat-vh-vl-ledger-sha256", required=True)
    finalize.set_defaults(handler=command_finalize_suite)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
