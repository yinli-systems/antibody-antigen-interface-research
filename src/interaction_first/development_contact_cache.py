"""Materialize an isolated train-only contact cache for PAIRScale development.

The controller reads only an exact authorized subset of a train-only embedding
cache and the matching sparse contact artifacts. Tensor artifacts are copied as
authenticated bytes and are never deserialized by this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Any, Mapping, Sequence
import uuid

from .contacts import FORMAT_VERSION as CONTACT_FORMAT_VERSION
from .contacts import load_contact_map
from .development_authority import (
    CLAIM_BOUNDARY,
    PERMISSION_NAME,
    canonical_json_bytes,
    canonical_sha256,
)
from .development_subset import (
    DevelopmentSubsetError,
    parse_train_authorization,
)


CACHE_FORMAT = "pairscale_development_contact_cache_v1"
RECEIPT_FORMAT = "pairscale_development_contact_cache_receipt_v1"
PARENT_CACHE_FORMAT = "frontier_fold_local_cache_manifest_v1"
_PARENT_INSTANCE_FIELDS = {
    "instance_id",
    "antibody_id",
    "antigen_id",
    "artifact_path",
    "artifact_sha256",
    "artifact_size_bytes",
    "antibody_length",
    "antigen_length",
    "contact_supervision",
}
_DECISION_FIELDS = {
    "schema_version",
    "permission_name",
    "policy_sha256",
    "request_sha256",
    "run_id",
    "purpose",
    "subset_manifest_sha256",
    "subset_membership_sha256",
    "subset_instance_count",
    "evaluated_at_utc",
    "authorized",
    "blocking_reasons",
    "sealed_data_access_authorized",
    "validation_data_access_authorized",
    "test_data_access_authorized",
    "model_selection_authorized",
    "checkpoint_promotion_authorized",
    "confirmatory_claim_authorized",
    "claim_boundary",
    "implementation_source_sha256",
}
_SUBSET_FIELDS = {
    "format_version",
    "permission_name",
    "purpose",
    "claim_boundary",
    "source_split",
    "parent_authorization_sha256",
    "policy_sha256",
    "selection",
    "authorization_manifest_sha256",
    "instance_count",
    "instance_membership_sha256",
    "antibody_count",
    "antigen_count",
    "positive_pair_count",
    "excluded_pair_count",
    "instances",
}


class DevelopmentContactCacheError(DevelopmentSubsetError):
    """Raised when a contact-cache publication fails closed."""


def materialize_development_contact_cache(
    *,
    parent_cache_root: Path,
    parent_cache_manifest_sha256: str,
    subset_authorization_bytes: bytes,
    subset_authorization_sha256: str,
    subset_manifest_bytes: bytes,
    subset_manifest_sha256: str,
    decision_bytes: bytes,
    decision_sha256: str,
    expected_policy_sha256: str,
    contact_root: Path,
    contact_summary_sha256: str,
    output_root: Path,
    implementation_source_sha256: Mapping[str, str],
) -> Mapping[str, object]:
    """Verify all boundaries and atomically publish one isolated cache."""

    for label, digest in (
        ("parent_cache_manifest_sha256", parent_cache_manifest_sha256),
        ("subset_authorization_sha256", subset_authorization_sha256),
        ("subset_manifest_sha256", subset_manifest_sha256),
        ("decision_sha256", decision_sha256),
        ("expected_policy_sha256", expected_policy_sha256),
        ("contact_summary_sha256", contact_summary_sha256),
    ):
        _sha256(digest, label)
    _verify_source_map(implementation_source_sha256)

    parent_cache_root = _absolute(parent_cache_root)
    contact_root = _absolute(contact_root)
    output_root = _absolute(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise DevelopmentContactCacheError(
            "output_root already exists; publication is create-only"
        )
    parent_manifest_path = parent_cache_root / "FOLD_LOCAL_CACHE_MANIFEST.json"
    parent_manifest_bytes = _read_stable_regular(parent_manifest_path)
    if (
        hashlib.sha256(parent_manifest_bytes).hexdigest()
        != parent_cache_manifest_sha256
    ):
        raise DevelopmentContactCacheError("parent cache manifest digest mismatch")
    if hashlib.sha256(subset_authorization_bytes).hexdigest() != (
        subset_authorization_sha256
    ):
        raise DevelopmentContactCacheError("subset authorization digest mismatch")
    if hashlib.sha256(subset_manifest_bytes).hexdigest() != subset_manifest_sha256:
        raise DevelopmentContactCacheError("subset manifest digest mismatch")

    try:
        subset_authorization = parse_train_authorization(
            subset_authorization_bytes
        )
    except DevelopmentSubsetError as error:
        raise DevelopmentContactCacheError(
            "subset authorization is invalid"
        ) from error
    parent_manifest = _parse_parent_manifest(parent_manifest_bytes)
    subset_manifest = _strict_json_object(subset_manifest_bytes, "subset manifest")
    _exact_keys(subset_manifest, _SUBSET_FIELDS, "subset manifest")
    decision = _strict_json_object(decision_bytes, "development decision")
    _exact_keys(decision, _DECISION_FIELDS, "development decision")
    if canonical_sha256(decision) != decision_sha256:
        raise DevelopmentContactCacheError("development decision digest mismatch")
    _validate_subset_and_decision(
        subset_manifest=subset_manifest,
        subset_manifest_sha256=subset_manifest_sha256,
        subset_authorization_sha256=subset_authorization_sha256,
        subset_instance_ids=subset_authorization.instance_ids,
        decision=decision,
        decision_sha256=decision_sha256,
        expected_policy_sha256=expected_policy_sha256,
    )

    parent_by_id = {
        record["instance_id"]: record for record in parent_manifest["instances"]
    }
    requested_by_id = {
        record.instance_id: record for record in subset_authorization.instances
    }
    if not set(requested_by_id).issubset(parent_by_id):
        raise DevelopmentContactCacheError(
            "subset contains an instance absent from the parent cache"
        )
    for instance_id, requested in requested_by_id.items():
        parent = parent_by_id[instance_id]
        if (
            parent["antibody_id"] != requested.antibody_id
            or parent["antigen_id"] != requested.antigen_id
        ):
            raise DevelopmentContactCacheError(
                "subset endpoints differ from the parent cache"
            )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(
        f".{output_root.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    staging.mkdir(mode=0o700)
    embedding_root = staging / "embeddings"
    contact_output_root = staging / "contacts"
    embedding_root.mkdir(mode=0o700)
    contact_output_root.mkdir(mode=0o700)
    records: list[dict[str, object]] = []
    total_embedding_bytes = 0
    total_contact_bytes = 0
    total_candidates = 0
    total_positives = 0
    try:
        for instance_id in sorted(requested_by_id):
            parent = parent_by_id[instance_id]
            relative = _relative_artifact_path(parent["artifact_path"])
            source_embedding = parent_cache_root / relative
            embedding_bytes = _read_stable_regular(source_embedding)
            if len(embedding_bytes) != parent["artifact_size_bytes"]:
                raise DevelopmentContactCacheError(
                    f"embedding size mismatch for {instance_id}"
                )
            if hashlib.sha256(embedding_bytes).hexdigest() != parent["artifact_sha256"]:
                raise DevelopmentContactCacheError(
                    f"embedding digest mismatch for {instance_id}"
                )
            embedding_relative = PurePosixPath(
                "embeddings", f"{instance_id}.pt"
            )
            _write_create_only(staging / embedding_relative, embedding_bytes)

            source_contact = contact_root / f"{instance_id}.npz"
            contact_bytes = _read_stable_regular(source_contact)
            contact_sha256 = hashlib.sha256(contact_bytes).hexdigest()
            contact_relative = PurePosixPath("contacts", f"{instance_id}.npz")
            contact_output_path = staging / contact_relative
            _write_create_only(contact_output_path, contact_bytes)
            contact = load_contact_map(
                contact_output_path,
                expected_instance_id=instance_id,
            )
            if (
                contact.antibody_length != parent["antibody_length"]
                or contact.antigen_length != parent["antigen_length"]
            ):
                raise DevelopmentContactCacheError(
                    f"contact and embedding lengths differ for {instance_id}"
                )
            records.append(
                {
                    "instance_id": instance_id,
                    "antibody_id": parent["antibody_id"],
                    "antigen_id": parent["antigen_id"],
                    "antibody_length": parent["antibody_length"],
                    "antigen_length": parent["antigen_length"],
                    "embedding_artifact_path": str(embedding_relative),
                    "embedding_artifact_sha256": parent["artifact_sha256"],
                    "embedding_artifact_size_bytes": len(embedding_bytes),
                    "contact_artifact_path": str(contact_relative),
                    "contact_artifact_sha256": contact_sha256,
                    "contact_artifact_size_bytes": len(contact_bytes),
                    "contact_format_version": CONTACT_FORMAT_VERSION,
                    "antibody_candidate_residue_count": len(
                        contact.antibody_indices
                    ),
                    "antigen_candidate_residue_count": len(contact.antigen_indices),
                    "candidate_pair_count": contact.candidate_count,
                    "positive_pair_count": contact.positive_count,
                }
            )
            total_embedding_bytes += len(embedding_bytes)
            total_contact_bytes += len(contact_bytes)
            total_candidates += contact.candidate_count
            total_positives += contact.positive_count

        _write_create_only(
            staging / "AUTHORIZED_FOLD_INVENTORY.json",
            subset_authorization_bytes,
        )
        _write_create_only(
            staging / "PAIRSCALE_DEVELOPMENT_SUBSET.json",
            subset_manifest_bytes,
        )
        _write_create_only(
            staging / "DEVELOPMENT_AUTHORITY_DECISION.json",
            decision_bytes,
        )
        manifest: dict[str, object] = {
            "format_version": CACHE_FORMAT,
            "permission_name": PERMISSION_NAME,
            "claim_boundary": CLAIM_BOUNDARY,
            "source_split": "train",
            "whole_pool_materialized": False,
            "policy_sha256": expected_policy_sha256,
            "decision_sha256": decision_sha256,
            "subset_manifest_sha256": subset_manifest_sha256,
            "authorization_manifest_sha256": subset_authorization_sha256,
            "parent_cache_manifest_sha256": parent_cache_manifest_sha256,
            "contact_summary_sha256": contact_summary_sha256,
            "antibody_input_dim": parent_manifest["antibody_input_dim"],
            "antigen_input_dim": parent_manifest["antigen_input_dim"],
            "instance_count": len(records),
            "total_embedding_bytes": total_embedding_bytes,
            "total_contact_bytes": total_contact_bytes,
            "candidate_pair_count": total_candidates,
            "positive_pair_count": total_positives,
            "instances": records,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        _write_create_only(
            staging / "PAIRSCALE_CONTACT_CACHE_MANIFEST.json",
            manifest_bytes,
        )
        receipt: dict[str, object] = {
            "format_version": RECEIPT_FORMAT,
            "permission_name": PERMISSION_NAME,
            "claim_boundary": CLAIM_BOUNDARY,
            "source_split": "train",
            "whole_pool_materialized": False,
            "manifest_sha256": manifest_sha256,
            "policy_sha256": expected_policy_sha256,
            "decision_sha256": decision_sha256,
            "subset_manifest_sha256": subset_manifest_sha256,
            "authorization_manifest_sha256": subset_authorization_sha256,
            "parent_cache_manifest_sha256": parent_cache_manifest_sha256,
            "contact_summary_sha256": contact_summary_sha256,
            "instance_count": len(records),
            "candidate_pair_count": total_candidates,
            "positive_pair_count": total_positives,
            "implementation_source_sha256": dict(
                sorted(implementation_source_sha256.items())
            ),
            "tensor_artifacts_deserialized_by_materializer": False,
            "validation_or_test_data_accessed": False,
            "confirmatory_authority_granted": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _write_create_only(
            staging / "CONTACT_CACHE_BUILD_RECEIPT.json",
            canonical_json_bytes(receipt),
        )
        for directory in (embedding_root, contact_output_root, staging):
            directory.chmod(0o550)
        os.rename(staging, output_root)
        return receipt
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _parse_parent_manifest(payload_bytes: bytes) -> Mapping[str, Any]:
    payload = _strict_json_object(payload_bytes, "parent cache manifest")
    _exact_keys(
        payload,
        {
            "format_version",
            "data_access_scope",
            "split_role",
            "fold_id",
            "whole_pool_materialized",
            "authorization_manifest_sha256",
            "antibody_input_dim",
            "antigen_input_dim",
            "instances",
            "positive_pairs",
            "excluded_pairs",
        },
        "parent cache manifest",
    )
    if payload["format_version"] != PARENT_CACHE_FORMAT:
        raise DevelopmentContactCacheError("unsupported parent cache format")
    if payload["data_access_scope"] != "train_only_fold_local":
        raise DevelopmentContactCacheError("parent cache is not train-only")
    if payload["split_role"] != "train":
        raise DevelopmentContactCacheError("parent cache split_role is not train")
    if payload["whole_pool_materialized"] is not False:
        raise DevelopmentContactCacheError(
            "parent cache whole_pool_materialized must be false"
        )
    instances = payload["instances"]
    if type(instances) is not list or not instances:
        raise DevelopmentContactCacheError("parent instances must be nonempty")
    identifiers: list[str] = []
    for index, record in enumerate(instances):
        if type(record) is not dict:
            raise DevelopmentContactCacheError(
                f"parent instances[{index}] must be an object"
            )
        _exact_keys(record, _PARENT_INSTANCE_FIELDS, f"parent instances[{index}]")
        identifiers.append(_identifier(record["instance_id"], "instance_id"))
        _identifier(record["antibody_id"], "antibody_id")
        _identifier(record["antigen_id"], "antigen_id")
        _relative_artifact_path(record["artifact_path"])
        _sha256(record["artifact_sha256"], "artifact_sha256")
        for name in (
            "artifact_size_bytes",
            "antibody_length",
            "antigen_length",
        ):
            _positive_integer(record[name], name)
        if not isinstance(record["contact_supervision"], bool):
            raise DevelopmentContactCacheError(
                "parent contact_supervision must be boolean"
            )
    if len(set(identifiers)) != len(identifiers) or identifiers != sorted(identifiers):
        raise DevelopmentContactCacheError(
            "parent instance identifiers must be sorted and unique"
        )
    _positive_integer(payload["antibody_input_dim"], "antibody_input_dim")
    _positive_integer(payload["antigen_input_dim"], "antigen_input_dim")
    _sha256(payload["authorization_manifest_sha256"], "authorization sha256")
    return payload


def _validate_subset_and_decision(
    *,
    subset_manifest: Mapping[str, Any],
    subset_manifest_sha256: str,
    subset_authorization_sha256: str,
    subset_instance_ids: Sequence[str],
    decision: Mapping[str, Any],
    decision_sha256: str,
    expected_policy_sha256: str,
) -> None:
    if subset_manifest.get("format_version") != "pairscale_development_subset_v1":
        raise DevelopmentContactCacheError("unsupported subset manifest")
    if subset_manifest.get("permission_name") != PERMISSION_NAME:
        raise DevelopmentContactCacheError("subset permission is not locked")
    if subset_manifest.get("purpose") != "overfit_sanity":
        raise DevelopmentContactCacheError("subset purpose is not overfit_sanity")
    if subset_manifest.get("source_split") != "train":
        raise DevelopmentContactCacheError("subset source split is not train")
    if subset_manifest.get("claim_boundary") != CLAIM_BOUNDARY:
        raise DevelopmentContactCacheError("subset claim boundary is not locked")
    if subset_manifest.get("policy_sha256") != expected_policy_sha256:
        raise DevelopmentContactCacheError("subset policy digest mismatch")
    if (
        subset_manifest.get("authorization_manifest_sha256")
        != subset_authorization_sha256
    ):
        raise DevelopmentContactCacheError(
            "subset does not bind the authorization manifest"
        )
    manifest_ids = [
        record.get("instance_id")
        for record in subset_manifest.get("instances", [])
        if type(record) is dict
    ]
    if manifest_ids != list(subset_instance_ids):
        raise DevelopmentContactCacheError(
            "subset and authorization instance memberships differ"
        )

    if decision.get("schema_version") != "pairscale_development_decision_v1":
        raise DevelopmentContactCacheError("unsupported development decision")
    if decision.get("permission_name") != PERMISSION_NAME:
        raise DevelopmentContactCacheError("decision permission is not locked")
    if decision.get("policy_sha256") != expected_policy_sha256:
        raise DevelopmentContactCacheError("decision policy digest mismatch")
    if decision.get("subset_manifest_sha256") != subset_manifest_sha256:
        raise DevelopmentContactCacheError("decision subset digest mismatch")
    if decision.get("purpose") != "overfit_sanity":
        raise DevelopmentContactCacheError("decision purpose is not overfit_sanity")
    if decision.get("authorized") is not True or decision.get("blocking_reasons") != []:
        raise DevelopmentContactCacheError("development decision is not authorized")
    for name in (
        "sealed_data_access_authorized",
        "validation_data_access_authorized",
        "test_data_access_authorized",
        "model_selection_authorized",
        "checkpoint_promotion_authorized",
        "confirmatory_claim_authorized",
    ):
        if decision.get(name) is not False:
            raise DevelopmentContactCacheError(f"decision grants forbidden {name}")
    if decision.get("claim_boundary") != CLAIM_BOUNDARY:
        raise DevelopmentContactCacheError("decision claim boundary is not locked")
    if canonical_sha256(decision) != decision_sha256:
        raise DevelopmentContactCacheError("decision digest changed during validation")


def _read_stable_regular(path: Path) -> bytes:
    path = _absolute(path)
    if path.is_symlink():
        raise DevelopmentContactCacheError(f"symlink is forbidden: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise DevelopmentContactCacheError(f"not a regular file: {path}")
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise DevelopmentContactCacheError(f"file ended early: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise DevelopmentContactCacheError(f"file grew while reading: {path}")
        after = os.fstat(descriptor)
        fingerprint = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if fingerprint(before) != fingerprint(after):
            raise DevelopmentContactCacheError(f"file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o440)


def _strict_json_object(payload_bytes: bytes, label: str) -> dict[str, Any]:
    if not payload_bytes or payload_bytes.startswith(b"\xef\xbb\xbf"):
        raise DevelopmentContactCacheError(f"{label} is empty or contains a BOM")
    try:
        text = payload_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise DevelopmentContactCacheError(f"{label} must be ASCII JSON") from error

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DevelopmentContactCacheError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DevelopmentContactCacheError(
                    f"{label} contains non-finite JSON constant {token}"
                )
            ),
        )
    except json.JSONDecodeError as error:
        raise DevelopmentContactCacheError(f"{label} is not valid JSON") from error
    if type(value) is not dict:
        raise DevelopmentContactCacheError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if type(value) is not dict:
        raise DevelopmentContactCacheError(f"{label} must be a plain object")
    actual = set(value)
    if actual != expected:
        raise DevelopmentContactCacheError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _relative_artifact_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or not value.isascii():
        raise DevelopmentContactCacheError("artifact_path must be nonempty ASCII")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
        raise DevelopmentContactCacheError("artifact_path is not a safe relative path")
    if path.parts[0] != "examples" or path.suffix != ".pt":
        raise DevelopmentContactCacheError("artifact_path has an unexpected layout")
    return path


def _identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or any(character.isspace() for character in value)
        or len(value) > 512
    ):
        raise DevelopmentContactCacheError(f"{label} is not canonical ASCII")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DevelopmentContactCacheError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DevelopmentContactCacheError(f"{label} must be a positive integer")
    return value


def _verify_source_map(value: Mapping[str, str]) -> None:
    expected = {
        "scripts/materialize_pairscale_development_contact_cache.py",
        "src/interaction_first/contacts.py",
        "src/interaction_first/development_contact_cache.py",
    }
    if set(value) != expected:
        raise DevelopmentContactCacheError(
            "implementation source keys differ from the locked set"
        )
    for name, digest in value.items():
        _sha256(digest, f"implementation_source_sha256[{name}]")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))
