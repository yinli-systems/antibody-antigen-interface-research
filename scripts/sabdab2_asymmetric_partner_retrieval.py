#!/usr/bin/env python3
"""Train a development-only AntiBERTy--ESM-2 partner retriever."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sabdab2_partner_retrieval import (
    PairRecord,
    SequenceRecord,
    atomic_write_json,
    canonical_sha256,
    deduplicate_exact_pairs,
    length_bucket_batches,
    sequence_bucket_batches,
    sha256_file,
    unique_sequences,
    validate_config_integrity,
)


FORMAT_VERSION = "sabdab2_asymmetric_partner_retrieval_v1"
SCORE_ARTIFACT_FORMAT_VERSION = "sabdab2_asymmetric_retrieval_scores_v1"
ANTIBERTY_FORMAT = "sabdab2_antiberty_embedding_v1"
ESM_FORMAT = "sppider_seq_embedding_v1"
AUTHORITY_VALIDATION_BOUNDARY_FORMAT = "sabdab2_authority_validation_nw_boundary_v2"
ACCEPTED_SPLIT_FORMATS = {
    "sabdab2_interaction_first_v1",
    "sabdab2_identity_isolated_development_v1",
    "sabdab2_antigen_cluster_development_fold_v1",
    "sabdab2_iterative_nw_strict_development_split_v1",
    AUTHORITY_VALIDATION_BOUNDARY_FORMAT,
}
HEX_DIGEST_LENGTH = 64
STRICT_NW_CONSTRAINT_MODES = {
    "concat_antibody_and_antigen": False,
    "concat_antibody_antigen_plus_vh_vl": True,
}
FINAL_REFIT_SUBMISSION_FORMAT_VERSION = (
    "sabdab2_asymmetric_final_refit_submission_manifest_v1"
)
FINAL_REFIT_SELECTED_VARIANT = "antiberty_calm_mean"
FINAL_REFIT_FOLD_SELECTION_SHA256 = (
    "4b26d10b8b54e75a2599da0aa5e6cb531d33f0c297c32615a64249b7080178ca"
)
FINAL_REFIT_INDEPENDENT_AUDIT_SHA256 = (
    "91d94092fafbc5fd4110e52de8f75712638b114371fe6468a08af8e963c62a3e"
)
FINAL_REFIT_PROTOCOL_V1_3_SHA256 = (
    "6e38014e390c3f39943557ff3b96feb449b818f03e99da70797efaf634d963a5"
)
FINAL_REFIT_PROTOCOL_V1_4_SHA256 = (
    "34c3bbe82baf431c87fae0d54da965819e91c9762f000487dd076ba36b76eaf5"
)
FINAL_REFIT_PROTOCOL_V1_5_SHA256 = (
    "43638fdd17b9ce3e4cac8b83e42c16261ecbf79c2e628d0b494e5dcbe02f9fc9"
)
FINAL_REFIT_PROTOCOL_V1_6_SHA256 = (
    "b5b74452d3483d8b1fdf8c67387e3d38ad2be34a22b65e26a6cab71b0ca73078"
)
FINAL_REFIT_BOUNDARY_VERIFICATION_SHA256 = (
    "3bd8186603d908a99e1fbf00ead3697dd276ccd798b4f91bcbdda81441bd80e8"
)
FINAL_REFIT_BOUNDARY_VERIFICATION_SCIENTIFIC_SHA256 = (
    "44c6b763419087a0b21faae4b55e03aa0904380a71f4cec6d8df3945b48f6a29"
)
FINAL_REFIT_AUTHORITY_SHA256 = (
    "e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b"
)
FINAL_REFIT_PREPARED_INSTANCES_SHA256 = (
    "a9131d52323bae40572b7de81c5e41353ccc39a73309b7d7043485350facb009"
)
FINAL_REFIT_CANONICAL_SEQUENCE_MANIFEST_SHA256 = (
    "548bcfab75923ec6621c1a3fef1aa6673c034f64a9bdc752e7ee15b41b4a4e63"
)
FINAL_REFIT_ANTIBODY_EMBEDDING_MANIFEST_SHA256 = (
    "9eb10c64fd05897162548a81871841724d9471ba5ed1c800b3d10a2c9229a45e"
)
FINAL_REFIT_ANTIGEN_EMBEDDING_MANIFEST_SHA256 = (
    "d4157971ba2e977d1125e896fe13147371292495fa04a47f62b32ebe8f4ac97e"
)
FINAL_REFIT_ANTIBODY_WEIGHTS_SHA256 = (
    "f1ae33eac8cc8784a7d4be5a600141d2fa7bc7d8d5b3f5324d64a6a63bd0f137"
)
FINAL_REFIT_ANTIGEN_WEIGHTS_SHA256 = (
    "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"
)
FINAL_REFIT_BEST_EPOCHS = [58, 60, 44]
FINAL_REFIT_FIXED_EPOCHS = 58
RETIRED_CONTAMINATED_BOUNDARY_SHA256 = {
    "c586fbd1ec83effe1208746d64950feddef280a95a938cf241293dda0b396b64",
    "5d92edd03e3b96edfa727823b2a5bbdf49e80614ef759fe0c6b14cf6c2b6b92b",
}
ONE_SHOT_LEDGER_FORMAT_VERSION = "sabdab2_one_shot_evaluation_ledger_v2"
CHECKPOINT_SUITE_CLAIM_FORMAT_VERSION = "sabdab2_final_checkpoint_suite_claim_v1"
CHECKPOINT_SUITE_COMPLETION_FORMAT_VERSION = (
    "sabdab2_final_checkpoint_suite_completion_v1"
)


def _is_sha256(value: str) -> bool:
    return len(value) == HEX_DIGEST_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _pair_set(records: Iterable[PairRecord]) -> set[tuple[str, str]]:
    return {(record.antibody_key, record.antigen_key) for record in records}


def _identifier_membership_sha256(identifiers: Iterable[str]) -> str:
    """Hash one sorted identifier inventory using the boundary-builder contract."""
    return hashlib.sha256(
        ("\n".join(sorted(identifiers)) + "\n").encode("utf-8")
    ).hexdigest()


def audit_isolation_policy(
    train_records: Sequence[PairRecord],
    holdout_records: Sequence[PairRecord],
    policy: str,
) -> dict[str, Any]:
    """Audit exact sequence isolation under a declared double- or antigen-remote policy."""
    if policy not in {"double_exact", "antigen_exact"}:
        raise ValueError("isolation policy must be 'double_exact' or 'antigen_exact'")
    train_antibodies = {record.antibody_key for record in train_records}
    train_antigens = {record.antigen_key for record in train_records}
    holdout_antibodies = {record.antibody_key for record in holdout_records}
    holdout_antigens = {record.antigen_key for record in holdout_records}
    audit = {
        "policy": policy,
        "antibody_sequence_overlap": len(train_antibodies & holdout_antibodies),
        "antigen_sequence_overlap": len(train_antigens & holdout_antigens),
        "exact_pair_overlap": len(_pair_set(train_records) & _pair_set(holdout_records)),
    }
    forbidden = ["antigen_sequence_overlap", "exact_pair_overlap"]
    if policy == "double_exact":
        forbidden.insert(0, "antibody_sequence_overlap")
    violations = {name: audit[name] for name in forbidden if audit[name]}
    if violations:
        raise ValueError(f"Train/validation violates {policy} isolation: {violations}")
    audit["valid"] = True
    return audit


def _records_from_frame(frame: Any) -> list[PairRecord]:
    records = []
    for row in frame.itertuples(index=False):
        record = PairRecord(
            instance_id=str(row.instance_id),
            antibody_key=str(row.ab_sequence_sha256),
            antigen_key=str(row.ag_sequence_sha256),
            antibody_length=int(row.ab_length),
            antigen_length=int(row.ag_length),
        )
        if not record.instance_id:
            raise ValueError("Prepared instances contain an empty identifier")
        if not _is_sha256(record.antibody_key) or not _is_sha256(record.antigen_key):
            raise ValueError(f"Invalid sequence digest for {record.instance_id}")
        if record.antibody_length <= 0 or record.antigen_length <= 0:
            raise ValueError(f"Invalid sequence length for {record.instance_id}")
        records.append(record)
    return records


def load_development_dataset(
    prepare_dir: Path,
    split_manifest: Path,
    split_manifest_sha256: str,
    authority_split_manifest: Path,
    authority_split_sha256: str,
    authorized_source_pool_splits: Sequence[str],
    authority_sealed_splits: Sequence[str],
    authority_coverage_mode: str,
    strict_nw_constraint_mode: str | None = None,
    materialized_splits: Sequence[str] = ("train", "validation"),
    require_authority_validation_boundary_v2: bool = False,
    expected_instances_sha256: str | None = None,
    canonical_sequence_manifest: Path | None = None,
    canonical_sequence_manifest_sha256: str | None = None,
) -> tuple[dict[str, list[PairRecord]], dict[str, Any]]:
    """Authorize a development split against a pinned source split before reading rows."""
    import pandas as pd

    prepare_dir = prepare_dir.expanduser().resolve()
    instances_path = prepare_dir / "instances.parquet"
    split_manifest = split_manifest.expanduser().resolve()
    authority_split_manifest = authority_split_manifest.expanduser().resolve()
    if not instances_path.is_file():
        raise FileNotFoundError(f"Prepared instances are missing: {instances_path}")
    observed_instances_sha256 = sha256_file(instances_path)
    if expected_instances_sha256 is not None:
        if not _is_sha256(expected_instances_sha256):
            raise ValueError("An exact prepared-instances SHA-256 is required")
        if observed_instances_sha256 != expected_instances_sha256:
            raise ValueError(
                "Prepared instances physical SHA-256 mismatch: "
                f"expected={expected_instances_sha256}, "
                f"observed={observed_instances_sha256}"
            )
    if (canonical_sequence_manifest is None) != (
        canonical_sequence_manifest_sha256 is None
    ):
        raise ValueError(
            "Canonical sequence manifest path and SHA-256 must be supplied together"
        )
    canonical_sequence_reference = None
    if canonical_sequence_manifest is not None:
        canonical_path = canonical_sequence_manifest.expanduser().resolve()
        if not canonical_path.is_file():
            raise FileNotFoundError(
                f"Canonical sequence manifest is missing: {canonical_path}"
            )
        if not isinstance(canonical_sequence_manifest_sha256, str) or not _is_sha256(
            canonical_sequence_manifest_sha256
        ):
            raise ValueError("An exact canonical sequence-manifest SHA-256 is required")
        observed_canonical_sha256 = sha256_file(canonical_path)
        if observed_canonical_sha256 != canonical_sequence_manifest_sha256:
            raise ValueError(
                "Canonical sequence manifest physical SHA-256 mismatch: "
                f"expected={canonical_sequence_manifest_sha256}, "
                f"observed={observed_canonical_sha256}"
            )
        canonical_sequence_reference = {
            "path": str(canonical_path),
            "sha256": observed_canonical_sha256,
        }
    if not split_manifest.is_file():
        raise FileNotFoundError(f"Split manifest is missing: {split_manifest}")
    if not authority_split_manifest.is_file():
        raise FileNotFoundError(
            f"Authoritative source split is missing: {authority_split_manifest}"
        )
    if not _is_sha256(split_manifest_sha256):
        raise ValueError("An exact development split SHA-256 is required")
    observed_split_sha256 = sha256_file(split_manifest)
    if observed_split_sha256 != split_manifest_sha256:
        raise ValueError(
            "Development split hash mismatch: "
            f"expected={split_manifest_sha256}, observed={observed_split_sha256}"
        )
    if not _is_sha256(authority_split_sha256):
        raise ValueError("An exact authoritative source split SHA-256 is required")
    observed_authority_sha256 = sha256_file(authority_split_manifest)
    if observed_authority_sha256 != authority_split_sha256:
        raise ValueError(
            "Authoritative source split hash mismatch: "
            f"expected={authority_split_sha256}, observed={observed_authority_sha256}"
        )

    payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    split_format = payload.get("format_version")
    if split_format not in ACCEPTED_SPLIT_FORMATS:
        raise ValueError("Split manifest format is not approved for this experiment")
    if require_authority_validation_boundary_v2 and split_format != AUTHORITY_VALIDATION_BOUNDARY_FORMAT:
        raise ValueError(
            "Final-refit and one-shot confirmation require an authority-validation "
            "boundary v2 manifest"
        )
    if require_authority_validation_boundary_v2 and (
        expected_instances_sha256 is None or canonical_sequence_reference is None
    ):
        raise ValueError(
            "Final-refit confirmation requires pinned prepared and canonical inputs"
        )
    if split_format == AUTHORITY_VALIDATION_BOUNDARY_FORMAT and not require_authority_validation_boundary_v2:
        raise PermissionError(
            "Authority-validation boundary v2 is reserved for explicitly authorized "
            "fixed confirmation commands"
        )
    expected_intended_use = (
        "reused_development_confirmation_only"
        if split_format == AUTHORITY_VALIDATION_BOUNDARY_FORMAT
        else "development_only"
    )
    if payload.get("intended_use") != expected_intended_use:
        raise ValueError("This entrypoint accepts development-only splits exclusively")
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("Split manifest does not contain a splits object")
    accounting_split_names = (
        ("train", "validation", "excluded_validation")
        if split_format == AUTHORITY_VALIDATION_BOUNDARY_FORMAT
        else ("train", "validation")
    )
    if split_format == AUTHORITY_VALIDATION_BOUNDARY_FORMAT and set(raw_splits) != set(
        accounting_split_names
    ):
        raise ValueError(
            "Authority-validation boundary v2 must contain exactly train, validation, "
            "and excluded_validation"
        )
    selected_ids: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for split_name in accounting_split_names:
        raw_values = raw_splits.get(split_name)
        if not isinstance(raw_values, list):
            raise ValueError(f"Split {split_name!r} must be a list")
        if split_format == AUTHORITY_VALIDATION_BOUNDARY_FORMAT and not all(
            isinstance(value, str) and value for value in raw_values
        ):
            raise ValueError(f"Boundary split {split_name!r} must contain string identifiers")
        values = [str(value) for value in raw_values]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Split {split_name!r} must contain unique identifiers")
        overlap = assigned.intersection(values)
        if overlap:
            raise ValueError(f"Development identifiers cross splits: {sorted(overlap)[:5]}")
        assigned.update(values)
        selected_ids[split_name] = values
    materialized_names = [str(value) for value in materialized_splits]
    if (
        not materialized_names
        or len(materialized_names) != len(set(materialized_names))
        or not set(materialized_names).issubset({"train", "validation"})
    ):
        raise ValueError("materialized_splits must be a unique subset of train and validation")

    authority_payload = json.loads(
        authority_split_manifest.read_text(encoding="utf-8")
    )
    authority_raw_splits = authority_payload.get("splits")
    if not isinstance(authority_raw_splits, dict):
        raise ValueError("Authoritative source split does not contain a splits object")
    authority_splits: dict[str, set[str]] = {}
    authority_observed: set[str] = set()
    for raw_name, raw_values in authority_raw_splits.items():
        name = str(raw_name)
        if not isinstance(raw_values, list):
            raise ValueError(f"Authoritative split {name!r} is not a list")
        values = [str(value) for value in raw_values]
        if len(values) != len(set(values)):
            raise ValueError(f"Authoritative split {name!r} contains duplicates")
        overlap = authority_observed.intersection(values)
        if overlap:
            raise ValueError(
                f"Authoritative identifiers cross source splits: {sorted(overlap)[:5]}"
            )
        authority_observed.update(values)
        authority_splits[name] = set(values)
    pool_names = [str(value) for value in authorized_source_pool_splits]
    sealed_names = [str(value) for value in authority_sealed_splits]
    if not pool_names or not sealed_names:
        raise ValueError("Authorized pool and sealed split names must both be explicit")
    if len(pool_names) != len(set(pool_names)) or len(sealed_names) != len(set(sealed_names)):
        raise ValueError("Authority split-name arguments cannot contain duplicates")
    if set(pool_names) & set(sealed_names):
        raise ValueError("Authorized pool and sealed source split names cannot overlap")
    unknown_names = (set(pool_names) | set(sealed_names)) - set(authority_splits)
    if unknown_names:
        raise ValueError(f"Unknown authoritative source split names: {sorted(unknown_names)}")
    allowed_pool = set().union(*(authority_splits[name] for name in pool_names))
    sealed = set().union(*(authority_splits[name] for name in sealed_names))
    sealed_overlap = assigned & sealed
    if sealed_overlap:
        raise PermissionError(
            f"Development split contains sealed source IDs: {sorted(sealed_overlap)[:5]}"
        )
    unauthorized = assigned - allowed_pool
    if unauthorized:
        raise PermissionError(
            f"Development split contains IDs outside the authorized pool: {sorted(unauthorized)[:5]}"
        )
    if authority_coverage_mode not in {"exact", "subset"}:
        raise ValueError("authority_coverage_mode must be 'exact' or 'subset'")
    if authority_coverage_mode == "exact" and assigned != allowed_pool:
        missing_from_development = allowed_pool - assigned
        raise ValueError(
            "Development split does not exactly cover the authorized source pool: "
            f"missing={sorted(missing_from_development)[:5]}"
        )
    if split_format == AUTHORITY_VALIDATION_BOUNDARY_FORMAT:
        if set(pool_names) != {"train", "validation"} or authority_coverage_mode != "exact":
            raise ValueError(
                "Authority-validation boundary v2 requires exact authority train+validation coverage"
            )
        if "train" not in authority_splits or "validation" not in authority_splits:
            raise ValueError("Authority lacks the required train and validation origins")
        boundary_train = set(selected_ids["train"])
        boundary_validation = set(selected_ids["validation"])
        boundary_excluded = set(selected_ids["excluded_validation"])
        if boundary_train != authority_splits["train"]:
            raise ValueError("Boundary train must equal authority train exactly")
        if boundary_validation | boundary_excluded != authority_splits["validation"]:
            raise ValueError(
                "Boundary validation plus excluded_validation must equal authority validation"
            )
        if boundary_validation & boundary_excluded:
            raise ValueError("Boundary validation and excluded_validation overlap")
        if strict_nw_constraint_mode not in STRICT_NW_CONSTRAINT_MODES:
            raise ValueError(
                "Authority-validation boundary v2 requires an explicit approved constraint mode"
            )
        if payload.get("constraint_mode") != strict_nw_constraint_mode:
            raise ValueError(
                "Authority-validation boundary constraint mode differs from the command line"
            )
        if (
            payload.get("stage") != "build_authority_validation_nw_clean_boundary"
            or payload.get("valid") is not True
            or payload.get("selection_influenced_by_labels_predictions_or_model_scores")
            is not False
            or payload.get("frozen_test_identifiers_read_for_exclusion_only") is not True
            or payload.get(
                "frozen_test_sequence_fields_accessed_normalized_hashed_or_aligned"
            )
            is not False
        ):
            raise ValueError("Authority-validation boundary v2 violates its firewall scope")
        expected_views = (
            ["concat_antibody", "antigen"]
            if strict_nw_constraint_mode == "concat_antibody_and_antigen"
            else ["concat_antibody", "vh", "vl", "antigen"]
        )
        if payload.get("declared_views") != expected_views:
            raise ValueError("Authority-validation boundary declares unexpected sequence views")
        chronology = payload.get("chronology")
        if chronology != {
            "authority_validation_previously_inspected_by_esm2_exploration": True,
            "clean_relative_to_antiberty_three_fold_architecture_selection": True,
            "globally_untouched_or_independent_test": False,
        }:
            raise ValueError("Authority-validation boundary has an unsafe chronology claim")
        expected_limitations = [
            "This is a reused development confirmation subset, not an independent test.",
            "Eligibility uses the source authority graph components plus exhaustive exact-NW train-validation edges; it does not assert structural or epitope independence.",
            "External or prospective claims require separately locked external or temporal data.",
        ]
        if payload.get("limitations") != expected_limitations:
            raise ValueError("Authority-validation boundary has an unsafe claim scope")
        selection_rule = payload.get("selection_rule")
        if selection_rule != {
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
        }:
            raise ValueError("Authority-validation boundary selection rule changed")
        evidence = payload.get("evidence")
        if (
            not isinstance(evidence, Mapping)
            or float(evidence.get("identity_threshold", -math.inf)) != 0.80
            or float(evidence.get("reciprocal_coverage_threshold", -math.inf)) != 0.80
            or not isinstance(
                evidence.get("authority_train_by_validation_exact_nw_artifacts"),
                Mapping,
            )
        ):
            raise ValueError("Authority-validation boundary lacks exact 80/80 NW evidence")
        edge_artifacts = evidence["authority_train_by_validation_exact_nw_artifacts"]
        if set(edge_artifacts) != set(expected_views):
            raise ValueError("Authority-validation boundary NW evidence has unexpected views")
        for view in expected_views:
            artifact = edge_artifacts[view]
            if (
                not isinstance(artifact, Mapping)
                or not isinstance(artifact.get("path"), str)
                or not isinstance(artifact.get("sha256"), str)
                or not _is_sha256(artifact["sha256"])
                or not isinstance(artifact.get("edge_count"), int)
                or artifact["edge_count"] < 0
                or not isinstance(
                    artifact.get("validation_digest_count_with_train_edge"), int
                )
                or artifact["validation_digest_count_with_train_edge"] < 0
            ):
                raise ValueError(
                    f"Authority-validation boundary has invalid NW evidence for {view}"
                )
        provenance = payload.get("provenance")
        authority_provenance = (
            provenance.get("authority_split", {})
            if isinstance(provenance, Mapping)
            else {}
        )
        if authority_provenance.get("sha256") != observed_authority_sha256:
            raise ValueError("Authority-validation boundary was built from another authority")
        protocol_correction = (
            provenance.get("protocol_correction", {})
            if isinstance(provenance, Mapping)
            else {}
        )
        if protocol_correction.get("sha256") != FINAL_REFIT_PROTOCOL_V1_4_SHA256:
            raise ValueError("Authority-validation boundary lacks the locked v1.4 correction")
        membership = payload.get("membership_sha256")
        if not isinstance(membership, Mapping) or set(membership) != set(
            accounting_split_names
        ):
            raise ValueError("Authority-validation boundary lacks membership hashes")
        for name in accounting_split_names:
            if membership.get(name) != _identifier_membership_sha256(selected_ids[name]):
                raise ValueError(
                    f"Authority-validation boundary membership hash changed for {name}"
                )
        counts = payload.get("counts")
        expected_counts = {
            "authority_train": len(authority_splits["train"]),
            "authority_validation": len(authority_splits["validation"]),
            "sealed_test": len(sealed),
            "eligible_validation_instances": len(boundary_validation),
            "excluded_validation_instances": len(boundary_excluded),
        }
        if not isinstance(counts, Mapping) or any(
            counts.get(name) != value for name, value in expected_counts.items()
        ):
            raise ValueError("Authority-validation boundary count accounting changed")
        scientific_payload_sha256 = payload.get("scientific_payload_sha256")
        scientific_payload = dict(payload)
        scientific_payload.pop("scientific_payload_sha256", None)
        if (
            not isinstance(scientific_payload_sha256, str)
            or canonical_sha256(scientific_payload) != scientific_payload_sha256
        ):
            raise ValueError("Authority-validation boundary scientific payload hash changed")
    elif split_format == "sabdab2_iterative_nw_strict_development_split_v1":
        if strict_nw_constraint_mode not in STRICT_NW_CONSTRAINT_MODES:
            raise ValueError(
                "Strict NW manifests require an explicit approved constraint mode"
            )
        if payload.get("constraint_mode") != strict_nw_constraint_mode:
            raise ValueError(
                "Strict NW constraint mode differs from the explicit command-line mode"
            )
        if payload.get("valid") is not True:
            raise ValueError("Strict NW split does not carry a valid=true verdict")
        if payload.get("frozen_test_labels_or_predictions_read") is not False:
            raise ValueError("Strict NW split reports access to frozen test targets")
        if payload.get("frozen_test_sequences_aligned") is not False:
            raise ValueError("Strict NW split reports alignment against frozen test sequences")
        if payload.get("sealed_identifiers_read_for_exclusion_only") is not True:
            raise ValueError("Strict NW split lacks sealed-identifier exclusion evidence")
        if [str(value) for value in payload.get("sealed_source_splits", [])] != sealed_names:
            raise ValueError("Strict NW split sealed-source declaration differs from authority")
        graph_audit = payload.get("final_graph_audit")
        if not isinstance(graph_audit, Mapping) or graph_audit.get("valid") is not True:
            raise ValueError("Strict NW split lacks a valid final graph audit")
        evidence = payload.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("Strict NW split lacks exact-alignment evidence")
        identity_threshold = float(evidence.get("identity_threshold", -math.inf))
        coverage_threshold = float(
            evidence.get("reciprocal_coverage_threshold", -math.inf)
        )
        exact_policy = evidence.get("exact_alignment_policy")
        expected_chain_constraints = STRICT_NW_CONSTRAINT_MODES[
            strict_nw_constraint_mode
        ]
        if (
            identity_threshold < 0.80
            or coverage_threshold < 0.80
            or not isinstance(exact_policy, Mapping)
            or exact_policy.get("mode") != "global"
            or exact_policy.get("chain_constraints_included")
            is not expected_chain_constraints
        ):
            raise ValueError(
                "Strict NW split evidence does not match the explicit 80/80 global constraint mode"
            )
        development_source = (
            payload.get("provenance", {}).get("development_source_split", {})
            if isinstance(payload.get("provenance"), Mapping)
            else {}
        )
        if development_source.get("sha256") != observed_authority_sha256:
            raise ValueError("Strict NW split was built from a different authoritative split")
    elif strict_nw_constraint_mode is not None:
        raise ValueError(
            "strict_nw_constraint_mode is only valid for strict NW split manifests"
        )

    columns = [
        "instance_id",
        "ab_sequence_sha256",
        "ag_sequence_sha256",
        "ab_length",
        "ag_length",
    ]
    materialized_ids = set().union(
        *(set(selected_ids[name]) for name in materialized_names)
    )
    frame = pd.read_parquet(
        instances_path,
        columns=columns,
        filters=[("instance_id", "in", sorted(materialized_ids))],
    )
    if frame.empty or frame["instance_id"].isna().any():
        raise ValueError("Prepared instances cannot be empty or contain null IDs")
    frame["instance_id"] = frame["instance_id"].astype(str)
    if frame["instance_id"].duplicated().any():
        raise ValueError("Prepared instances contain duplicate instance IDs")
    selected_available = set(frame["instance_id"])
    missing = materialized_ids - selected_available
    if missing:
        raise ValueError(f"Split identifiers are missing from metadata: {sorted(missing)[:5]}")
    frame = frame.loc[frame["instance_id"].isin(materialized_ids)].copy()
    assignment = {
        identifier: split_name
        for split_name, identifiers in selected_ids.items()
        for identifier in identifiers
    }
    frame["retrieval_split"] = frame["instance_id"].map(assignment)
    split_records = {
        split_name: _records_from_frame(
            frame.loc[frame["retrieval_split"] == split_name, columns].sort_values(
                "instance_id", kind="stable"
            )
        )
        for split_name in materialized_names
    }
    metadata = {
        "instances_path": str(instances_path),
        "instances_sha256": observed_instances_sha256,
        "canonical_sequence_manifest": canonical_sequence_reference,
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": observed_split_sha256,
        "split_format_version": payload["format_version"],
        "split_intended_use": payload["intended_use"],
        "strict_nw_constraint_mode": strict_nw_constraint_mode,
        "loaded_splits": materialized_names,
        "unmaterialized_splits": [
            name for name in ("train", "validation") if name not in materialized_names
        ],
        "permanently_excluded_splits": (
            ["excluded_validation"]
            if split_format == AUTHORITY_VALIDATION_BOUNDARY_FORMAT
            else []
        ),
        "excluded_validation_rows_materialized": False,
        "raw_instance_counts": {
            name: len(values) for name, values in selected_ids.items()
        },
        "split_membership_sha256": {
            name: canonical_sha256({"instance_ids": sorted(values)})
            for name, values in selected_ids.items()
        },
        "declared_identifier_membership_sha256": (
            dict(payload["membership_sha256"])
            if split_format == AUTHORITY_VALIDATION_BOUNDARY_FORMAT
            else None
        ),
        "parquet_row_filter": "_and_".join(materialized_names) + "_identifiers_only",
        "authority_accounting_splits": list(accounting_split_names),
        "frozen_test_records_selected": False,
        "authority": {
            "path": str(authority_split_manifest),
            "sha256": observed_authority_sha256,
            "format_version": authority_payload.get("format_version"),
            "authorized_source_pool_splits": pool_names,
            "authorized_source_pool_count": len(allowed_pool),
            "sealed_source_splits": sealed_names,
            "sealed_source_counts": {
                name: len(authority_splits[name]) for name in sealed_names
            },
            "selected_sealed_overlap_count": 0,
            "coverage_mode": authority_coverage_mode,
            "selected_pool_coverage_count": len(assigned),
        },
    }
    return split_records, metadata


def side_embedding_fingerprint(
    root: Path,
    sequence_keys: Iterable[str],
    side: str,
) -> dict[str, Any]:
    """Hash every immutable embedding artifact used on one model side."""
    root = root.expanduser().resolve()
    keys = sorted(set(sequence_keys))
    if side not in {"antibody", "antigen"} or not keys:
        raise ValueError("A non-empty antibody or antigen fingerprint is required")
    manifest_digest = hashlib.sha256()
    total_bytes = 0
    for sequence_key in keys:
        if not _is_sha256(sequence_key):
            raise ValueError(f"Invalid {side} sequence key: {sequence_key!r}")
        path = root / sequence_key[:2] / f"{sequence_key}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Frozen {side} embedding is missing: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        manifest_digest.update(f"{sequence_key}\t{size}\t{digest}\n".encode("ascii"))
        total_bytes += size
    return {
        "side": side,
        "sequence_count": len(keys),
        "total_bytes": total_bytes,
        "manifest_algorithm": "sha256(sequence_key,size,file_sha256)",
        "content_manifest_sha256": manifest_digest.hexdigest(),
    }


class FrozenPartnerEmbeddingStore:
    """Validate and cache one side of the asymmetric frozen embedding corpus."""

    def __init__(
        self,
        root: Path,
        side: str,
        artifact_format: str,
        expected_dim: int,
        expected_model_name: str,
        expected_model_weights_sha256: str,
        max_cache_gib: float,
    ) -> None:
        if side not in {"antibody", "antigen"}:
            raise ValueError("side must be 'antibody' or 'antigen'")
        if expected_dim <= 0 or not expected_model_name:
            raise ValueError("Embedding dimension and model name must be specified")
        if artifact_format not in {"antiberty", "esm_chunks"}:
            raise ValueError("artifact_format must be 'antiberty' or 'esm_chunks'")
        if side == "antigen" and artifact_format != "esm_chunks":
            raise ValueError("Antigen embeddings must use the ESM chunk format")
        if not _is_sha256(expected_model_weights_sha256):
            raise ValueError("An exact expected model weight SHA-256 is required")
        if not math.isfinite(max_cache_gib) or max_cache_gib < 0.0:
            raise ValueError("max_cache_gib must be finite and non-negative")
        self.root = root.expanduser().resolve()
        self.side = side
        self.artifact_format = artifact_format
        self.expected_dim = expected_dim
        self.expected_model_name = expected_model_name
        self.expected_model_weights_sha256 = expected_model_weights_sha256
        self.max_cache_bytes = int(max_cache_gib * 1024**3)
        self.cache: OrderedDict[str, tuple[Any, Any | None, int]] = OrderedDict()
        self.cache_bytes = 0
        self.hits = 0
        self.misses = 0

    def _load_antibody(self, artifact: Mapping[str, Any], key: str) -> tuple[Any, Any]:
        import torch

        required = {
            "format_version",
            "sequence_key",
            "source_sequence_sha256",
            "embedding",
            "mask",
            "vh_length",
            "vl_length",
            "embedding_dim",
            "model_name",
            "model_weights_sha256",
        }
        missing = required - set(artifact)
        if missing:
            raise ValueError(f"AntiBERTy artifact is missing keys: {sorted(missing)}")
        if artifact["format_version"] != ANTIBERTY_FORMAT:
            raise ValueError(f"Unsupported antibody format: {artifact['format_version']!r}")
        if artifact["sequence_key"] != key or artifact["source_sequence_sha256"] != key:
            raise ValueError("AntiBERTy artifact digest does not match its requested key")
        if artifact["model_name"] != self.expected_model_name:
            raise ValueError("AntiBERTy artifact was produced by an unexpected model")
        if artifact["model_weights_sha256"] != self.expected_model_weights_sha256:
            raise ValueError("AntiBERTy artifact model weight digest is unexpected")
        embedding = artifact["embedding"]
        mask = artifact["mask"]
        vh_length = int(artifact["vh_length"])
        vl_length = int(artifact["vl_length"])
        full_length = vh_length + vl_length
        if vh_length <= 0 or vl_length < 0 or full_length <= 0:
            raise ValueError("AntiBERTy chain lengths are invalid")
        if int(artifact["embedding_dim"]) != self.expected_dim:
            raise ValueError("AntiBERTy metadata dimension is incompatible")
        if (
            not torch.is_tensor(embedding)
            or embedding.dtype != torch.float32
            or tuple(embedding.shape) != (full_length, self.expected_dim)
            or not embedding.is_contiguous()
            or not bool(torch.isfinite(embedding).all())
        ):
            raise ValueError("AntiBERTy embedding must be finite contiguous float32 [L, D]")
        if (
            not torch.is_tensor(mask)
            or mask.dtype != torch.bool
            or tuple(mask.shape) != (full_length,)
            or not bool(mask.all())
        ):
            raise ValueError("AntiBERTy mask must mark every stored residue as valid")
        chain_ids = torch.empty(full_length, dtype=torch.long)
        chain_ids[:vh_length] = 0
        chain_ids[vh_length:] = 1
        return embedding, chain_ids

    def _load_esm_chunks(self, artifact: Mapping[str, Any], key: str) -> tuple[Any, None]:
        import torch

        required = {
            "format_version",
            "sequence_sha256",
            "sequence_length",
            "model_name",
            "model_weight_sha256",
            "starts",
            "ends",
            "chunks",
        }
        missing = required - set(artifact)
        if missing:
            raise ValueError(f"ESM-2 artifact is missing keys: {sorted(missing)}")
        if artifact["format_version"] != ESM_FORMAT:
            raise ValueError(f"Unsupported antigen format: {artifact['format_version']!r}")
        if artifact["sequence_sha256"] != key:
            raise ValueError("ESM-2 artifact digest does not match its requested key")
        if artifact["model_name"] != self.expected_model_name:
            raise ValueError("ESM-2 artifact was produced by an unexpected model")
        if artifact["model_weight_sha256"] != self.expected_model_weights_sha256:
            raise ValueError("ESM-2 artifact model weight digest is unexpected")
        starts = [int(value) for value in artifact["starts"]]
        ends = [int(value) for value in artifact["ends"]]
        chunks = list(artifact["chunks"])
        full_length = int(artifact["sequence_length"])
        if full_length <= 0 or not chunks or len(chunks) != len(starts) or len(chunks) != len(ends):
            raise ValueError("ESM-2 chunk metadata is empty or inconsistent")
        merged = torch.zeros(full_length, self.expected_dim, dtype=torch.float32)
        counts = torch.zeros(full_length, 1, dtype=torch.float32)
        for start, end, chunk in zip(starts, ends, chunks):
            if start < 0 or end <= start or end > full_length:
                raise ValueError(f"Invalid ESM-2 interval [{start}, {end})")
            if (
                not torch.is_tensor(chunk)
                or not torch.is_floating_point(chunk)
                or tuple(chunk.shape) != (end - start, self.expected_dim)
                or not bool(torch.isfinite(chunk).all())
            ):
                raise ValueError("ESM-2 chunk shape, type, or values are invalid")
            merged[start:end] += chunk.to(dtype=torch.float32)
            counts[start:end] += 1.0
        if bool((counts == 0).any()):
            raise ValueError("ESM-2 chunks do not cover the complete antigen")
        return (merged / counts).contiguous(), None

    def load(self, key: str, expected_length: int, device: Any) -> tuple[Any, Any | None]:
        """Load one validated embedding and optional antibody chain IDs."""
        from interaction_first.runtime import safe_torch_load

        if key in self.cache:
            embedding, chain_ids, size = self.cache.pop(key)
            self.cache[key] = (embedding, chain_ids, size)
            self.hits += 1
        else:
            path = self.root / key[:2] / f"{key}.pt"
            artifact = safe_torch_load(path, map_location="cpu")
            if not isinstance(artifact, Mapping):
                raise ValueError(f"Frozen {self.side} artifact must be a mapping")
            if self.artifact_format == "antiberty":
                embedding, chain_ids = self._load_antibody(artifact, key)
            else:
                embedding, chain_ids = self._load_esm_chunks(artifact, key)
            size = int(embedding.numel() * embedding.element_size())
            if chain_ids is not None:
                size += int(chain_ids.numel() * chain_ids.element_size())
            self.misses += 1
            if size <= self.max_cache_bytes:
                while self.cache and self.cache_bytes + size > self.max_cache_bytes:
                    _, (_, _, evicted_size) = self.cache.popitem(last=False)
                    self.cache_bytes -= evicted_size
                self.cache[key] = (embedding, chain_ids, size)
                self.cache_bytes += size
        if int(embedding.shape[0]) != expected_length:
            raise ValueError(
                f"{self.side} length mismatch for {key}: "
                f"metadata={expected_length}, artifact={int(embedding.shape[0])}"
            )
        return (
            embedding.to(device=device, non_blocking=False),
            chain_ids.to(device=device) if chain_ids is not None else None,
        )

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe cache and validation settings."""
        return {
            "side": self.side,
            "artifact_format": self.artifact_format,
            "root": str(self.root),
            "expected_dim": self.expected_dim,
            "expected_model_name": self.expected_model_name,
            "expected_model_weights_sha256": self.expected_model_weights_sha256,
            "max_cache_bytes": self.max_cache_bytes,
            "cache_bytes": self.cache_bytes,
            "cached_sequence_count": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
        }


def _pad_sequences(
    records: Sequence[SequenceRecord],
    store: FrozenPartnerEmbeddingStore,
    device: Any,
) -> tuple[Any, Any, Any | None]:
    import torch

    if not records:
        raise ValueError("Cannot pad an empty sequence batch")
    features = torch.zeros(
        len(records),
        max(record.length for record in records),
        store.expected_dim,
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros(features.shape[:2], dtype=torch.bool, device=device)
    chain_ids = (
        torch.full(features.shape[:2], -1, dtype=torch.long, device=device)
        if store.side == "antibody" and store.artifact_format == "antiberty"
        else None
    )
    for index, record in enumerate(records):
        embedding, record_chain_ids = store.load(record.key, record.length, device)
        features[index, : record.length] = embedding
        mask[index, : record.length] = True
        if chain_ids is not None:
            if record_chain_ids is None:
                raise ValueError("Antibody artifact lacks required chain metadata")
            chain_ids[index, : record.length] = record_chain_ids
    return features, mask, chain_ids


def build_pair_masks(
    antibody_keys: Sequence[str],
    antigen_keys: Sequence[str],
    known_positive_pairs: set[tuple[str, str]],
    excluded_negative_pairs: set[tuple[str, str]],
    device: Any,
) -> tuple[Any, Any]:
    """Build known-positive and allowed-denominator matrices for one batch."""
    import torch

    positives = torch.tensor(
        [
            [(antibody, antigen) in known_positive_pairs for antigen in antigen_keys]
            for antibody in antibody_keys
        ],
        dtype=torch.bool,
        device=device,
    )
    if positives.numel() == 0 or not bool(positives.any(dim=1).all()) or not bool(
        positives.any(dim=0).all()
    ):
        raise ValueError("Every batch candidate must participate in a known positive")
    denominator = torch.tensor(
        [
            [
                (antibody, antigen) not in excluded_negative_pairs
                or (antibody, antigen) in known_positive_pairs
                for antigen in antigen_keys
            ]
            for antibody in antibody_keys
        ],
        dtype=torch.bool,
        device=device,
    )
    return positives, denominator


def prepare_pair_batch(
    records: Sequence[PairRecord],
    known_positive_pairs: set[tuple[str, str]],
    excluded_negative_pairs: set[tuple[str, str]],
    antibody_store: FrozenPartnerEmbeddingStore,
    antigen_store: FrozenPartnerEmbeddingStore,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Materialize unique partner axes and supervision masks for one batch."""
    antibodies = unique_sequences(records, "antibody")
    antigens = unique_sequences(records, "antigen")
    antibody_features, antibody_mask, antibody_chain_ids = _pad_sequences(
        antibodies, antibody_store, device
    )
    antigen_features, antigen_mask, _ = _pad_sequences(antigens, antigen_store, device)
    positive_mask, denominator_mask = build_pair_masks(
        [record.key for record in antibodies],
        [record.key for record in antigens],
        known_positive_pairs,
        excluded_negative_pairs,
        device,
    )
    return (
        antibody_features,
        antigen_features,
        antibody_mask,
        antigen_mask,
        antibody_chain_ids,
        positive_mask,
        denominator_mask,
    )


def load_excluded_negative_pairs(
    path: Path | None,
    train_records: Sequence[PairRecord],
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    """Load optional homology-uncertain pair exclusions with strict scope checks."""
    if path is None:
        return set(), {
            "enabled": False,
            "policy": "all_unobserved_training_pairs_are_negatives",
            "pair_count": 0,
        }
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != "sabdab2_uncertain_negative_pairs_v1":
        raise ValueError("Uncertain-negative artifact format is incompatible")
    if payload.get("intended_use") != "development_training_only":
        raise ValueError("Uncertain-negative artifact is not development-training-only")
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("Uncertain-negative artifact must contain a pairs list")
    pairs: set[tuple[str, str]] = set()
    for value in raw_pairs:
        if isinstance(value, Mapping):
            pair = (str(value.get("antibody_key", "")), str(value.get("antigen_key", "")))
        elif isinstance(value, list) and len(value) == 2:
            pair = (str(value[0]), str(value[1]))
        else:
            raise ValueError("Every uncertain-negative pair must contain two sequence keys")
        if not all(_is_sha256(key) for key in pair):
            raise ValueError("Uncertain-negative artifact contains an invalid digest")
        pairs.add(pair)
    known_pairs = _pair_set(train_records)
    overlap = pairs & known_pairs
    if overlap:
        raise ValueError("Known training positives cannot be marked as uncertain negatives")
    antibodies = {record.antibody_key for record in train_records}
    antigens = {record.antigen_key for record in train_records}
    outside = [pair for pair in pairs if pair[0] not in antibodies or pair[1] not in antigens]
    if outside:
        raise ValueError(f"Uncertain-negative pairs reference non-training sequences: {outside[:3]}")
    return pairs, {
        "enabled": True,
        "policy": "exclude_declared_homology_uncertain_edges_from_info_nce_denominators",
        "path": str(path),
        "sha256": sha256_file(path),
        "pair_count": len(pairs),
    }


def _model_from_config(config: Mapping[str, Any], device: Any) -> Any:
    from interaction_first.asymmetric_retrieval import AsymmetricPartnerRetrievalModel

    architecture = config["model"]
    return AsymmetricPartnerRetrievalModel(
        antibody_input_dim=int(architecture["antibody_input_dim"]),
        antigen_input_dim=int(architecture["antigen_input_dim"]),
        embedding_dim=int(architecture["embedding_dim"]),
        antibody_hidden_dim=int(architecture["antibody_hidden_dim"]),
        antigen_hidden_dim=int(architecture["antigen_hidden_dim"]),
        dropout=float(architecture["dropout"]),
        chain_balanced_antibody_pooling=bool(
            architecture["chain_balanced_antibody_pooling"]
        ),
        antibody_pooling=str(architecture["antibody_pooling"]),
        antigen_pooling=str(architecture["antigen_pooling"]),
        initial_temperature=float(architecture["initial_temperature"]),
        minimum_temperature=float(architecture["minimum_temperature"]),
        maximum_temperature=float(architecture["maximum_temperature"]),
    ).to(device)


def _encode_library(
    model: Any,
    records: Sequence[SequenceRecord],
    side: str,
    store: FrozenPartnerEmbeddingStore,
    batch_size: int,
    bucket_width: int,
    device: Any,
) -> tuple[list[str], Any]:
    import torch

    model.eval()
    encoded: dict[str, Any] = {}
    with torch.no_grad():
        for batch in sequence_bucket_batches(records, batch_size, bucket_width):
            features, mask, chain_ids = _pad_sequences(batch, store, device)
            if side == "antibody":
                embeddings, _ = model.encode_antibody(features, mask, chain_ids)
            elif side == "antigen":
                embeddings, _ = model.encode_antigen(features, mask)
            else:
                raise ValueError("side must be 'antibody' or 'antigen'")
            for record, embedding in zip(batch, embeddings):
                encoded[record.key] = embedding.detach().to(device="cpu")
    keys = sorted(encoded)
    if keys != sorted(record.key for record in records):
        raise AssertionError("Library encoding did not preserve every unique sequence")
    return keys, torch.stack([encoded[key] for key in keys])


def _score_array_sha256(array: Any) -> str:
    """Hash an array together with its exact dtype and shape."""
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _npy_bytes(array: Any) -> bytes:
    """Serialize one non-object NumPy array deterministically."""
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject:
        raise ValueError("Score artifacts cannot contain object arrays")
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        contiguous,
        version=(1, 0),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _fsync_directory(path: Path) -> None:
    """Durably publish a newly linked artifact in its parent directory."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish strict JSON atomically while refusing every existing target."""
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
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_name, target)
        os.unlink(temporary_name)
        temporary_name = None
        _fsync_directory(target.parent)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _create_only_torch_save(payload: Any, path: Path) -> None:
    """Stage a torch artifact and atomically publish it without clobbering."""
    import torch

    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        torch.save(payload, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary_name, target)
        os.unlink(temporary_name)
        temporary_name = None
        _fsync_directory(target.parent)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _atomic_write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, Any],
) -> None:
    """Write a byte-deterministic compressed NPZ with fixed member metadata."""
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    member_names = sorted(arrays)
    if not member_names or any(not name.isidentifier() for name in member_names):
        raise ValueError("NPZ member names must be non-empty Python identifiers")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        with zipfile.ZipFile(
            temporary_name,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for name in member_names:
                member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.create_system = 3
                member.external_attr = 0o600 << 16
                archive.writestr(member, _npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary_name, target)
        os.unlink(temporary_name)
        temporary_name = None
        _fsync_directory(target.parent)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def write_score_artifact(
    path: Path,
    score_payload: Mapping[str, Any],
    evaluation_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a deterministic, sequence-free full-library score artifact."""
    import numpy as np

    similarities = np.asarray(score_payload["similarity_matrix"], dtype=np.float32)
    positive_mask = np.asarray(score_payload["positive_mask"], dtype=bool)
    antibody_keys = [str(value) for value in score_payload["antibody_keys"]]
    antigen_keys = [str(value) for value in score_payload["antigen_keys"]]
    if similarities.ndim != 2 or similarities.size == 0:
        raise ValueError("Score similarity matrix must be non-empty and two-dimensional")
    if positive_mask.shape != similarities.shape or not np.isfinite(similarities).all():
        raise ValueError("Score and positive matrices must be aligned and finite")
    if len(antibody_keys) != similarities.shape[0] or len(antigen_keys) != similarities.shape[1]:
        raise ValueError("Ordered score keys do not match matrix axes")
    if len(set(antibody_keys)) != len(antibody_keys) or len(set(antigen_keys)) != len(antigen_keys):
        raise ValueError("Ordered score keys must be unique")
    if not all(_is_sha256(value) for value in [*antibody_keys, *antigen_keys]):
        raise ValueError("Score artifacts may contain only SHA-256 sequence identifiers")
    if not positive_mask.any(axis=1).all() or not positive_mask.any(axis=0).all():
        raise ValueError("Every score-artifact anchor must have a known positive")
    antibody_array = np.asarray([value.encode("ascii") for value in antibody_keys], dtype="S64")
    antigen_array = np.asarray([value.encode("ascii") for value in antigen_keys], dtype="S64")
    arrays_without_metadata = {
        "similarity_matrix": np.ascontiguousarray(similarities),
        "positive_mask": np.ascontiguousarray(positive_mask),
        "antibody_keys": antibody_array,
        "antigen_keys": antigen_array,
    }
    array_sha256 = {
        name: _score_array_sha256(value)
        for name, value in arrays_without_metadata.items()
    }
    contract = {
        "format_version": SCORE_ARTIFACT_FORMAT_VERSION,
        "scientific_scope": {
            "split": "validation",
            "intended_use": "development_statistics_only",
            "frozen_test_used": False,
            "contains_raw_sequences": False,
        },
        "evaluation_context": dict(evaluation_context),
        "matrix_shape": [int(value) for value in similarities.shape],
        "array_dtypes": {
            name: str(np.asarray(value).dtype)
            for name, value in arrays_without_metadata.items()
        },
        "array_sha256": array_sha256,
        "positive_edge_count": int(positive_mask.sum()),
    }
    contract_sha256 = canonical_sha256(contract)
    metadata = {
        "format_version": SCORE_ARTIFACT_FORMAT_VERSION,
        "contract": contract,
        "contract_sha256": contract_sha256,
    }
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    arrays = {
        **arrays_without_metadata,
        "metadata_json_utf8": np.frombuffer(metadata_bytes, dtype=np.uint8).copy(),
    }
    target = path.expanduser().resolve()
    _atomic_write_deterministic_npz(target, arrays)
    return {
        "format_version": SCORE_ARTIFACT_FORMAT_VERSION,
        "path_at_creation": str(target),
        "sha256": sha256_file(target),
        "contract_sha256": contract_sha256,
        "array_sha256": array_sha256,
        "matrix_shape": contract["matrix_shape"],
        "positive_edge_count": contract["positive_edge_count"],
        "contains_raw_sequences": False,
    }


def detailed_directional_metrics(
    similarities: Any,
    positive_mask: Any,
    anchor_keys: Sequence[str],
    candidate_keys: Sequence[str],
    recall_ks: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return aggregate retrieval metrics and auditable per-anchor ranks."""
    import torch

    if similarities.ndim != 2 or tuple(positive_mask.shape) != tuple(similarities.shape):
        raise ValueError("Retrieval matrices must be aligned and two-dimensional")
    if positive_mask.dtype != torch.bool or not bool(torch.isfinite(similarities).all()):
        raise ValueError("Retrieval matrices contain invalid types or values")
    if len(anchor_keys) != similarities.shape[0] or len(candidate_keys) != similarities.shape[1]:
        raise ValueError("Retrieval keys must align with their matrix axes")
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("Every retrieval anchor must have a known positive")
    ks = sorted({int(value) for value in recall_ks})
    if not ks or ks[0] <= 0:
        raise ValueError("recall_ks must contain positive integers")
    candidate_count = int(similarities.shape[1])
    per_anchor = []
    ranks = []
    for row_index, anchor_key in enumerate(anchor_keys):
        try:
            order = torch.argsort(similarities[row_index], descending=True, stable=True)
        except TypeError:
            order = torch.argsort(similarities[row_index], descending=True)
        ordered_positive = positive_mask[row_index].index_select(0, order)
        positions = torch.nonzero(ordered_positive, as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError("Retrieval anchor has no known positive candidate")
        rank = int(positions.min().item()) + 1
        positive_indices = torch.nonzero(positive_mask[row_index], as_tuple=False).flatten()
        positive_keys = [candidate_keys[int(index)] for index in positive_indices]
        normalized_rank = (
            (rank - 1) / (candidate_count - 1) if candidate_count > 1 else 0.0
        )
        ranks.append(rank)
        per_anchor.append(
            {
                "anchor_key": anchor_key,
                "best_known_positive_rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "normalized_rank_zero_one": normalized_rank,
                "rank_fraction": rank / candidate_count,
                "known_positive_candidate_count": len(positive_keys),
                "known_positive_candidate_keys": positive_keys,
            }
        )
    metrics: dict[str, Any] = {
        "anchor_count": len(ranks),
        "candidate_count": candidate_count,
        "known_positive_count": int(positive_mask.sum().item()),
        "mean_positives_per_anchor": float(positive_mask.sum(dim=1).float().mean().item()),
        "mrr": float(sum(1.0 / rank for rank in ranks) / len(ranks)),
        "mean_rank": float(sum(ranks) / len(ranks)),
        "median_rank": float(statistics.median(ranks)),
        "mean_normalized_rank_zero_one": float(
            sum(value["normalized_rank_zero_one"] for value in per_anchor) / len(per_anchor)
        ),
        "median_normalized_rank_zero_one": float(
            statistics.median(value["normalized_rank_zero_one"] for value in per_anchor)
        ),
        "normalized_rank_definition": "(best_positive_rank-1)/(candidate_count-1); lower_is_better",
    }
    for value in ks:
        hit_rate = float(sum(rank <= value for rank in ranks) / len(ranks))
        metrics[f"recall_at_{value}"] = hit_rate
        metrics[f"hit_rate_at_{value}"] = hit_rate
    return metrics, per_anchor


def full_library_metrics(
    antibody_embeddings: Any,
    antigen_embeddings: Any,
    positive_mask: Any,
    temperature: Any,
    recall_ks: Sequence[int],
    antibody_keys: Sequence[str],
    antigen_keys: Sequence[str],
) -> dict[str, Any]:
    """Compute complete-library metrics in both retrieval directions."""
    import torch

    from interaction_first.asymmetric_retrieval import (
        cosine_similarity_matrix,
        symmetric_masked_multi_positive_info_nce,
    )

    antibody_embeddings = antibody_embeddings.to(dtype=torch.float32, device="cpu")
    antigen_embeddings = antigen_embeddings.to(dtype=torch.float32, device="cpu")
    positive_mask = positive_mask.to(device="cpu")
    similarities = cosine_similarity_matrix(antibody_embeddings, antigen_embeddings)
    loss = symmetric_masked_multi_positive_info_nce(
        similarities,
        positive_mask,
        temperature=temperature.detach().to(dtype=torch.float32, device="cpu"),
    )
    antibody_to_antigen, antibody_to_antigen_ranks = detailed_directional_metrics(
        similarities,
        positive_mask,
        antibody_keys,
        antigen_keys,
        recall_ks,
    )
    antigen_to_antibody, antigen_to_antibody_ranks = detailed_directional_metrics(
        similarities.transpose(0, 1),
        positive_mask.transpose(0, 1),
        antigen_keys,
        antibody_keys,
        recall_ks,
    )
    directional_mrrs = [
        float(antibody_to_antigen["mrr"]),
        float(antigen_to_antibody["mrr"]),
    ]
    bidirectional_mrr = sum(directional_mrrs) / 2.0
    harmonic_mrr = (
        2.0 / sum(1.0 / value for value in directional_mrrs)
        if all(value > 0.0 for value in directional_mrrs)
        else 0.0
    )
    return {
        "loss": float(loss.item()),
        "bidirectional_mrr": bidirectional_mrr,
        "harmonic_directional_mrr": harmonic_mrr,
        "minimum_directional_mrr": min(directional_mrrs),
        "antibody_to_antigen": antibody_to_antigen,
        "antigen_to_antibody": antigen_to_antibody,
        "library_keys": {
            "antibody": list(antibody_keys),
            "antigen": list(antigen_keys),
        },
        "per_anchor_ranks": {
            "antibody_to_antigen": antibody_to_antigen_ranks,
            "antigen_to_antibody": antigen_to_antibody_ranks,
        },
    }


def evaluate_records_with_scores(
    model: Any,
    records: Sequence[PairRecord],
    antibody_store: FrozenPartnerEmbeddingStore,
    antigen_store: FrozenPartnerEmbeddingStore,
    batch_size: int,
    bucket_width: int,
    device: Any,
    recall_ks: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score every known pair and return metrics plus a sequence-free score payload."""
    from interaction_first.asymmetric_retrieval import cosine_similarity_matrix

    antibodies = unique_sequences(records, "antibody")
    antigens = unique_sequences(records, "antigen")
    antibody_keys, antibody_embeddings = _encode_library(
        model, antibodies, "antibody", antibody_store, batch_size, bucket_width, device
    )
    antigen_keys, antigen_embeddings = _encode_library(
        model, antigens, "antigen", antigen_store, batch_size, bucket_width, device
    )
    positive_mask, _ = build_pair_masks(
        antibody_keys,
        antigen_keys,
        _pair_set(records),
        set(),
        device="cpu",
    )
    metrics = full_library_metrics(
        antibody_embeddings,
        antigen_embeddings,
        positive_mask,
        model.temperature_head(),
        recall_ks,
        antibody_keys,
        antigen_keys,
    )
    metrics.update(
        {
            "exact_pair_count": len(_pair_set(records)),
            "unique_antibody_count": len(antibody_keys),
            "unique_antigen_count": len(antigen_keys),
        }
    )
    similarities = cosine_similarity_matrix(
        antibody_embeddings.to(device="cpu"),
        antigen_embeddings.to(device="cpu"),
    )
    score_payload = {
        "similarity_matrix": similarities.detach().to(device="cpu").numpy(),
        "positive_mask": positive_mask.detach().to(device="cpu").numpy(),
        "antibody_keys": antibody_keys,
        "antigen_keys": antigen_keys,
    }
    return metrics, score_payload


def evaluate_records(
    model: Any,
    records: Sequence[PairRecord],
    antibody_store: FrozenPartnerEmbeddingStore,
    antigen_store: FrozenPartnerEmbeddingStore,
    batch_size: int,
    bucket_width: int,
    device: Any,
    recall_ks: Sequence[int],
) -> dict[str, Any]:
    """Score every known pair against both complete validation libraries."""
    metrics, _ = evaluate_records_with_scores(
        model,
        records,
        antibody_store,
        antigen_store,
        batch_size,
        bucket_width,
        device,
        recall_ks,
    )
    return metrics


def _optimizer(model: Any, learning_rate: float, weight_decay: float) -> Any:
    import torch

    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "temperature" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay or not no_decay:
        raise ValueError("Optimizer parameter partition is unexpectedly empty")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
    )


def _scheduler(optimizer: Any, steps_per_epoch: int, args: argparse.Namespace) -> Any:
    import torch

    if args.schedule == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_epoch * args.restart_period_epochs,
            T_mult=args.restart_multiplier,
            eta_min=args.minimum_learning_rate,
        )
    total_steps = steps_per_epoch * args.max_epochs
    warmup_steps = min(total_steps - 1, max(1, round(total_steps * args.warmup_fraction)))

    def multiplier(step: int) -> float:
        minimum_ratio = args.minimum_learning_rate / args.learning_rate
        if step < warmup_steps:
            return max(minimum_ratio, (step + 1) / warmup_steps)
        progress = min(
            1.0,
            (step - warmup_steps) / max(1, total_steps - warmup_steps - 1),
        )
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _train_epoch(
    model: Any,
    records: Sequence[PairRecord],
    known_positive_pairs: set[tuple[str, str]],
    excluded_negative_pairs: set[tuple[str, str]],
    antibody_store: FrozenPartnerEmbeddingStore,
    antigen_store: FrozenPartnerEmbeddingStore,
    optimizer: Any,
    scheduler: Any,
    device: Any,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, Any], int]:
    import torch

    from interaction_first.asymmetric_retrieval import (
        symmetric_masked_multi_positive_info_nce,
    )

    model.train()
    batches = length_bucket_batches(
        records, args.batch_size, args.bucket_width, args.seed, epoch
    )
    loss_total = 0.0
    anchor_total = 0
    optimized_batches = 0
    skipped_batches = 0
    excluded_matrix_edges = 0
    maximum_gradient_norm = 0.0
    for batch_index, batch in enumerate(batches, 1):
        (
            antibody_features,
            antigen_features,
            antibody_mask,
            antigen_mask,
            antibody_chain_ids,
            positive_mask,
            denominator_mask,
        ) = prepare_pair_batch(
            batch,
            known_positive_pairs,
            excluded_negative_pairs,
            antibody_store,
            antigen_store,
            device,
        )
        if bool(denominator_mask.equal(positive_mask)):
            skipped_batches += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            antibody_features,
            antigen_features,
            antibody_mask,
            antigen_mask,
            antibody_chain_ids,
        )
        loss = symmetric_masked_multi_positive_info_nce(
            outputs["similarity_matrix"],
            positive_mask,
            temperature=outputs["temperature"],
            denominator_mask=denominator_mask,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite contrastive loss in epoch {epoch}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"Non-finite gradient norm in epoch {epoch}")
        optimizer.step()
        scheduler.step()
        global_step += 1
        anchors = int(positive_mask.shape[0] + positive_mask.shape[1])
        loss_total += float(loss.detach().item()) * anchors
        anchor_total += anchors
        optimized_batches += 1
        excluded_matrix_edges += int((~denominator_mask).sum().item())
        maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm.item()))
        if batch_index % args.log_every == 0 or batch_index == len(batches):
            print(
                f"epoch={epoch} batch={batch_index}/{len(batches)} "
                f"contrastive_loss={loss_total / max(1, anchor_total):.8f}",
                flush=True,
            )
    if optimized_batches == 0 or anchor_total == 0:
        raise ValueError("Every training batch lacked eligible negative candidates")
    return (
        {
            "loss": loss_total / anchor_total,
            "batch_count": len(batches),
            "optimized_batch_count": optimized_batches,
            "skipped_without_eligible_negatives": skipped_batches,
            "anchor_count": anchor_total,
            "excluded_uncertain_matrix_edge_count": excluded_matrix_edges,
            "maximum_preclip_gradient_norm": maximum_gradient_norm,
        },
        global_step,
    )


def _validate_numeric_arguments(args: argparse.Namespace) -> None:
    integer_positive = {
        "antibody_input_dim": args.antibody_input_dim,
        "antigen_input_dim": args.antigen_input_dim,
        "embedding_dim": args.embedding_dim,
        "antibody_hidden_dim": args.antibody_hidden_dim,
        "antigen_hidden_dim": args.antigen_hidden_dim,
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "bucket_width": args.bucket_width,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "restart_period_epochs": args.restart_period_epochs,
        "restart_multiplier": args.restart_multiplier,
    }
    if any(value <= 0 for value in integer_positive.values()) or args.batch_size < 2:
        raise ValueError(f"Dimensions and training limits must be positive: {integer_positive}")
    if args.min_train_pairs < 2 or args.min_validation_pairs < 2:
        raise ValueError("Minimum split sizes must be at least two")
    finite_values = (
        args.dropout,
        args.initial_temperature,
        args.minimum_temperature,
        args.maximum_temperature,
        args.learning_rate,
        args.minimum_learning_rate,
        args.weight_decay,
        args.warmup_fraction,
        args.gradient_clip,
        args.min_delta,
        args.antibody_cache_gib,
        args.antigen_cache_gib,
    )
    if not all(math.isfinite(float(value)) for value in finite_values):
        raise ValueError("Numeric training arguments must be finite")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not 0.0 < args.minimum_temperature < args.initial_temperature < args.maximum_temperature:
        raise ValueError("Temperature bounds are invalid")
    if not 0.0 < args.minimum_learning_rate <= args.learning_rate:
        raise ValueError("Learning-rate bounds are invalid")
    if args.weight_decay < 0.0 or args.gradient_clip <= 0.0 or args.min_delta < 0.0:
        raise ValueError("Regularization and stopping values are invalid")
    if not 0.0 < args.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in (0, 1)")
    if args.antibody_cache_gib < 0.0 or args.antigen_cache_gib < 0.0:
        raise ValueError("Cache sizes cannot be negative")
    if not _is_sha256(args.antibody_model_weights_sha256) or not _is_sha256(
        args.antigen_model_weights_sha256
    ):
        raise ValueError("Exact model weight SHA-256 values are required for both sides")
    if args.antibody_artifact_format == "esm_chunks" and args.chain_balanced_antibody_pooling:
        raise ValueError(
            "Chain-balanced pooling requires chain-aware AntiBERTy artifacts; "
            "use --no-chain-balanced-antibody-pooling for ESM controls"
        )


def _source_hashes() -> dict[str, str]:
    """Return hashes for every source file that defines checkpoint semantics."""
    script = Path(__file__).resolve()
    package = script.parent / "interaction_first"
    return {
        "entrypoint": sha256_file(script),
        "asymmetric_retrieval": sha256_file(package / "asymmetric_retrieval.py"),
        "shared_retrieval_protocol": sha256_file(
            script.parent / "sabdab2_partner_retrieval.py"
        ),
        "runtime": sha256_file(package / "runtime.py"),
    }


def _strict_json_object(path: Path, label: str) -> dict[str, Any]:
    """Load a finite JSON object while rejecting duplicate object keys."""

    def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
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
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} as strict JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    require_finite(payload)
    return payload


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    """Reject missing or unreviewed fields in a versioned provenance contract."""
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} keys differ from the v1 contract: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _absolute_path(value: Any, label: str) -> Path:
    """Normalize one required absolute path without requiring it to exist."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    return path.resolve()


def _path_trees_overlap(left: Path, right: Path) -> bool:
    """Return whether either resolved path is the other path or its ancestor."""
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _manifest_reference(
    value: Any,
    label: str,
    *,
    expected_sha256: str | None = None,
    verify_physical_file: bool,
) -> dict[str, str]:
    """Validate a path-and-SHA reference, optionally hashing its physical file."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a path-and-SHA object")
    _require_exact_keys(value, {"path", "sha256"}, label)
    path = _absolute_path(value["path"], label)
    digest = value["sha256"]
    if not isinstance(digest, str) or not _is_sha256(digest):
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"{label}.sha256 differs from the locked reviewed artifact: "
            f"expected={expected_sha256}, observed={digest}"
        )
    if verify_physical_file:
        if not path.is_file():
            raise FileNotFoundError(f"{label} physical file is missing: {path}")
        observed = sha256_file(path)
        if observed != digest:
            raise ValueError(
                f"{label} physical SHA-256 mismatch: expected={digest}, observed={observed}"
            )
    return {"path": str(path), "sha256": digest}


def _validate_fold_selection_evidence(reference: Any) -> dict[str, Any]:
    """Verify the locked winner and refit epochs from the physical fold JSON."""
    if not isinstance(reference, Mapping):
        raise ValueError("upstream_evidence.fold_selection must be an object")
    _require_exact_keys(
        reference,
        {"path", "sha256", "selected_variant", "best_epochs", "median_best_epoch"},
        "upstream_evidence.fold_selection",
    )
    physical = _manifest_reference(
        {"path": reference["path"], "sha256": reference["sha256"]},
        "upstream_evidence.fold_selection",
        expected_sha256=FINAL_REFIT_FOLD_SELECTION_SHA256,
        verify_physical_file=True,
    )
    payload = _strict_json_object(Path(physical["path"]), "fold-selection evidence")
    if (
        payload.get("format_version") != "sabdab2_asymmetric_fold_aggregation_v1"
        or payload.get("intended_use") != "development_model_selection_only"
        or payload.get("frozen_test_used") is not False
        or payload.get("seed") != 42
        or payload.get("fold_count") != 3
    ):
        raise ValueError("Fold-selection evidence violates its locked scientific scope")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping) or selection.get(
        "selected_variant"
    ) != FINAL_REFIT_SELECTED_VARIANT:
        raise ValueError("Fold-selection evidence does not select the locked variant")
    variants = payload.get("variants")
    if not isinstance(variants, list):
        raise ValueError("Fold-selection evidence lacks variant results")
    selected = [
        value
        for value in variants
        if isinstance(value, Mapping)
        and value.get("name") == FINAL_REFIT_SELECTED_VARIANT
    ]
    if len(selected) != 1 or not isinstance(selected[0].get("folds"), list):
        raise ValueError("Fold-selection evidence has an ambiguous selected variant")
    fold_rows = selected[0]["folds"]
    try:
        ordered = sorted(fold_rows, key=lambda value: int(value["fold_index"]))
        indices = [int(value["fold_index"]) for value in ordered]
        epochs = [int(value["best_epoch"]) for value in ordered]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Fold-selection evidence has invalid fold epoch rows") from error
    if indices != [0, 1, 2] or epochs != FINAL_REFIT_BEST_EPOCHS:
        raise ValueError(
            "Fold-selection evidence differs from the reviewed best epochs: "
            f"indices={indices}, epochs={epochs}"
        )
    median_epoch = int(statistics.median(epochs))
    if (
        reference["selected_variant"] != FINAL_REFIT_SELECTED_VARIANT
        or reference["best_epochs"] != FINAL_REFIT_BEST_EPOCHS
        or reference["median_best_epoch"] != FINAL_REFIT_FIXED_EPOCHS
        or median_epoch != FINAL_REFIT_FIXED_EPOCHS
    ):
        raise ValueError("Submission manifest misstates the physical fold-selection result")
    if payload.get("authority", {}).get("sha256") != FINAL_REFIT_AUTHORITY_SHA256:
        raise ValueError("Fold-selection evidence used a different authority split")
    return {
        **physical,
        "selected_variant": FINAL_REFIT_SELECTED_VARIANT,
        "best_epochs": list(epochs),
        "median_best_epoch": median_epoch,
    }


def _validate_independent_audit_evidence(reference: Any) -> dict[str, Any]:
    """Verify the independent nine-run audit and its selected fold epochs."""
    physical = _manifest_reference(
        reference,
        "upstream_evidence.independent_audit",
        expected_sha256=FINAL_REFIT_INDEPENDENT_AUDIT_SHA256,
        verify_physical_file=True,
    )
    payload = _strict_json_object(Path(physical["path"]), "independent fold audit")
    if (
        payload.get("format_version")
        != "sabdab2_asymmetric_retrieval_fold_screen_audit_v1"
        or payload.get("valid") is not True
        or payload.get("all_runs_valid") is not True
        or payload.get("frozen_test_used") is not False
        or payload.get("fixed_holdout_evaluated") is not False
        or payload.get("top_level_failures") != []
        or payload.get("run_count") != 9
    ):
        raise ValueError("Independent fold audit does not carry a clean pass verdict")
    selection = payload.get("architecture_selection")
    if not isinstance(selection, Mapping) or selection.get(
        "winner"
    ) != FINAL_REFIT_SELECTED_VARIANT:
        raise ValueError("Independent fold audit does not confirm the locked winner")
    validity = payload.get("selection_validity_assessment")
    if not isinstance(validity, Mapping) or validity.get(
        "metric_or_selection_invalidated_by_caveats"
    ) is not False:
        raise ValueError("Independent fold audit does not preserve selection validity")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Independent fold audit lacks individual run evidence")
    selected_runs = [
        value
        for value in runs
        if isinstance(value, Mapping)
        and value.get("variant") == FINAL_REFIT_SELECTED_VARIANT
    ]
    try:
        selected_runs.sort(key=lambda value: int(value["fold_index"]))
        indices = [int(value["fold_index"]) for value in selected_runs]
        epochs = [int(value["epochs"]["selected"]) for value in selected_runs]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Independent fold audit contains invalid selected runs") from error
    if indices != [0, 1, 2] or epochs != FINAL_REFIT_BEST_EPOCHS:
        raise ValueError("Independent fold audit disagrees with the selected best epochs")
    expected_embedding_manifests = {
        "antibody": FINAL_REFIT_ANTIBODY_EMBEDDING_MANIFEST_SHA256,
        "antigen": FINAL_REFIT_ANTIGEN_EMBEDDING_MANIFEST_SHA256,
    }
    for run in selected_runs:
        if run.get("valid") is not True:
            raise ValueError("Independent fold audit marks a selected run invalid")
        weights = run.get("model_weight_sha256")
        if not isinstance(weights, Mapping) or weights.get(
            "antibody"
        ) != FINAL_REFIT_ANTIBODY_WEIGHTS_SHA256 or weights.get(
            "antigen"
        ) != FINAL_REFIT_ANTIGEN_WEIGHTS_SHA256:
            raise ValueError("Independent fold audit has unexpected encoder weight hashes")
        if run.get("authority_sha256") != FINAL_REFIT_AUTHORITY_SHA256:
            raise ValueError("Independent fold audit has an unexpected authority hash")
        if run.get("embedding_content_manifest_sha256") != expected_embedding_manifests:
            raise ValueError(
                "Independent fold audit has unexpected training embedding fingerprints"
            )
    return {
        **physical,
        "valid": True,
        "all_runs_valid": True,
        "winner": FINAL_REFIT_SELECTED_VARIANT,
        "selected_best_epochs": list(epochs),
        "fixed_holdout_evaluated": False,
        "frozen_test_used": False,
        "training_embedding_content_manifest_sha256": expected_embedding_manifests,
    }


def _validate_boundary_verification_evidence(
    reference: Any,
    normalized_boundaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind both submitted boundaries to the locked independent NW verification."""
    physical = _manifest_reference(
        reference,
        "upstream_evidence.boundary_verification_v2",
        expected_sha256=FINAL_REFIT_BOUNDARY_VERIFICATION_SHA256,
        verify_physical_file=True,
    )
    payload = _strict_json_object(
        Path(physical["path"]), "independent boundary verification"
    )
    scientific_digest = payload.get("scientific_payload_sha256")
    scientific_payload = dict(payload)
    scientific_payload.pop("scientific_payload_sha256", None)
    if (
        payload.get("format_version")
        != "sabdab2_authority_validation_nw_boundary_verification_v2"
        or payload.get("stage")
        != "independent_authority_validation_nw_boundary_verification"
        or payload.get("intended_use")
        != "reused_development_confirmation_only"
        or payload.get("valid") is not True
        or scientific_digest
        != FINAL_REFIT_BOUNDARY_VERIFICATION_SCIENTIFIC_SHA256
        or canonical_sha256(scientific_payload) != scientific_digest
        or payload.get("frozen_test_identifiers_read_for_exclusion_only") is not True
        or payload.get(
            "frozen_test_sequence_fields_accessed_normalized_hashed_or_aligned"
        )
        is not False
        or payload.get("labels_predictions_embeddings_or_checkpoints_read") is not False
    ):
        raise ValueError(
            "Independent boundary verification violates its locked scientific scope"
        )
    provenance = payload.get("provenance")
    authority_reference = (
        provenance.get("authority_split")
        if isinstance(provenance, Mapping)
        else None
    )
    if (
        not isinstance(authority_reference, Mapping)
        or authority_reference.get("sha256") != FINAL_REFIT_AUTHORITY_SHA256
    ):
        raise ValueError(
            "Independent boundary verification used a different authority split"
        )

    boundary_by_mode: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for name, boundary in normalized_boundaries.items():
        mode = boundary.get("strict_nw_constraint_mode")
        if not isinstance(mode, str) or mode in boundary_by_mode:
            raise ValueError("Submitted boundaries have an ambiguous constraint mode")
        boundary_by_mode[mode] = (name, boundary)
    if set(boundary_by_mode) != set(STRICT_NW_CONSTRAINT_MODES):
        raise ValueError("Submitted boundaries do not cover both verified modes")

    modes = payload.get("modes")
    if not isinstance(modes, Mapping) or set(modes) != set(STRICT_NW_CONSTRAINT_MODES):
        raise ValueError(
            "Independent boundary verification does not cover both submitted modes"
        )
    normalized_modes: dict[str, Any] = {}
    for mode, expected_chain_views in STRICT_NW_CONSTRAINT_MODES.items():
        row = modes[mode]
        if not isinstance(row, Mapping) or row.get("valid") is not True:
            raise ValueError(f"Independent boundary verification failed for {mode}")
        firewall_counts = (
            row.get("authority_train_intersection_with_validation"),
            row.get("sealed_intersection_with_all_boundary_partitions"),
        )
        if any(type(value) is not int or value != 0 for value in firewall_counts):
            raise ValueError(
                f"Independent boundary verification found a firewall crossing for {mode}"
            )
        boundary_reference = row.get("boundary")
        if not isinstance(boundary_reference, Mapping):
            raise ValueError(
                f"Independent boundary verification lacks a boundary reference for {mode}"
            )
        submitted_name, submitted_boundary = boundary_by_mode[mode]
        submitted_split = submitted_boundary.get("split_manifest")
        if not isinstance(submitted_split, Mapping) or (
            boundary_reference.get("sha256") != submitted_split.get("sha256")
            or boundary_reference.get("scientific_payload_sha256")
            != submitted_boundary.get("scientific_payload_sha256")
        ):
            raise ValueError(
                f"Independent verification does not bind the submitted {mode} boundary"
            )
        final_crossings = row.get(
            "final_train_validation_exact_nw_crossing_count_by_role"
        )
        expected_roles = (
            {"antigen", "concat_antibody", "vh", "vl"}
            if expected_chain_views
            else {"antigen", "concat_antibody"}
        )
        if (
            not isinstance(final_crossings, Mapping)
            or set(final_crossings) != expected_roles
            or any(type(value) is not int or value != 0 for value in final_crossings.values())
        ):
            raise ValueError(
                f"Independent verification has a final NW crossing for {mode}"
            )
        normalized_modes[mode] = {
            "submission_boundary": submitted_name,
            "boundary_sha256": submitted_split["sha256"],
            "boundary_scientific_payload_sha256": submitted_boundary[
                "scientific_payload_sha256"
            ],
            "authority_train_intersection_with_validation": 0,
            "sealed_intersection_with_all_boundary_partitions": 0,
            "final_train_validation_exact_nw_crossing_count_by_role": {
                role: 0 for role in sorted(expected_roles)
            },
            "valid": True,
        }
    return {
        **physical,
        "format_version": payload["format_version"],
        "scientific_payload_sha256": scientific_digest,
        "valid": True,
        "frozen_test_identifiers_read_for_exclusion_only": True,
        "frozen_test_sequence_fields_accessed_normalized_hashed_or_aligned": False,
        "labels_predictions_embeddings_or_checkpoints_read": False,
        "authority_sha256": FINAL_REFIT_AUTHORITY_SHA256,
        "modes": normalized_modes,
    }


def _expected_final_refit_model_contract() -> dict[str, Any]:
    """Return the exact architecture selected by the reviewed fold experiment."""
    return {
        "architecture": "AsymmetricPartnerRetrievalModel",
        "antibody_artifact_format": "antiberty",
        "antibody_model_name": "AntiBERTy_md_smooth",
        "antibody_model_weights_sha256": FINAL_REFIT_ANTIBODY_WEIGHTS_SHA256,
        "antigen_model_name": "facebook/esm2_t33_650M_UR50D",
        "antigen_model_weights_sha256": FINAL_REFIT_ANTIGEN_WEIGHTS_SHA256,
        "antibody_input_dim": 512,
        "antigen_input_dim": 1280,
        "embedding_dim": 128,
        "antibody_hidden_dim": 256,
        "antigen_hidden_dim": 256,
        "dropout": 0.1,
        "chain_balanced_antibody_pooling": False,
        "antibody_pooling": "mean",
        "antigen_pooling": "mean",
        "initial_temperature": 0.2,
        "minimum_temperature": 0.01,
        "maximum_temperature": 1.0,
    }


def _expected_final_refit_optimization_contract() -> dict[str, Any]:
    """Return the preselected fixed-epoch optimization contract."""
    return {
        "fixed_epochs": FINAL_REFIT_FIXED_EPOCHS,
        "seed": 42,
        "batch_size": 1024,
        "bucket_width": 128,
        "training_device": "cuda",
        "deterministic": True,
        "isolation_policy": "double_exact",
        "excluded_negative_pairs": None,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.001,
            "minimum_learning_rate": 0.000001,
            "weight_decay": 0.2,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "gradient_clip": 1.0,
        },
        "scheduler": {
            "name": "cosine_warm_restarts",
            "warmup_fraction": 0.10,
            "warmup_fraction_used": False,
            "restart_period_epochs": 10,
            "restart_multiplier": 2,
        },
    }


def _expected_final_evaluation_contract() -> dict[str, Any]:
    """Return the one-shot evaluation settings fixed before final refit."""
    return {
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


def validate_final_refit_submission_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the immutable final-refit submission contract before data access."""
    manifest_path = args.submission_manifest.expanduser().resolve()
    if not _is_sha256(args.submission_manifest_sha256):
        raise ValueError("An exact submission-manifest SHA-256 is required")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Submission manifest is missing: {manifest_path}")
    observed_manifest_sha256 = sha256_file(manifest_path)
    if observed_manifest_sha256 != args.submission_manifest_sha256:
        raise ValueError(
            "Submission manifest hash mismatch: "
            f"expected={args.submission_manifest_sha256}, "
            f"observed={observed_manifest_sha256}"
        )
    payload = _strict_json_object(manifest_path, "final-refit submission manifest")
    _require_exact_keys(
        payload,
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
        "final-refit submission manifest",
    )
    if (
        payload["format_version"] != FINAL_REFIT_SUBMISSION_FORMAT_VERSION
        or payload["intended_use"] != "development_only"
        or payload["frozen_test_used"] is not False
        or payload["selected_variant"] != FINAL_REFIT_SELECTED_VARIANT
    ):
        raise ValueError("Submission manifest violates the locked development-only scope")

    upstream = payload["upstream_evidence"]
    if not isinstance(upstream, Mapping):
        raise ValueError("upstream_evidence must be an object")
    _require_exact_keys(
        upstream,
        {
            "fold_selection",
            "independent_audit",
            "protocol_v1_3",
            "protocol_v1_4_correction",
            "protocol_v1_5_execution_lock",
            "protocol_v1_6_execution_hardening",
            "boundary_verification_v2",
        },
        "upstream_evidence",
    )
    fold_evidence = _validate_fold_selection_evidence(upstream["fold_selection"])
    audit_evidence = _validate_independent_audit_evidence(
        upstream["independent_audit"]
    )
    protocol_evidence = _manifest_reference(
        upstream["protocol_v1_3"],
        "upstream_evidence.protocol_v1_3",
        expected_sha256=FINAL_REFIT_PROTOCOL_V1_3_SHA256,
        verify_physical_file=True,
    )
    protocol_correction_evidence = _manifest_reference(
        upstream["protocol_v1_4_correction"],
        "upstream_evidence.protocol_v1_4_correction",
        expected_sha256=FINAL_REFIT_PROTOCOL_V1_4_SHA256,
        verify_physical_file=True,
    )
    protocol_execution_evidence = _manifest_reference(
        upstream["protocol_v1_5_execution_lock"],
        "upstream_evidence.protocol_v1_5_execution_lock",
        expected_sha256=FINAL_REFIT_PROTOCOL_V1_5_SHA256,
        verify_physical_file=True,
    )
    protocol_hardening_evidence = _manifest_reference(
        upstream["protocol_v1_6_execution_hardening"],
        "upstream_evidence.protocol_v1_6_execution_hardening",
        expected_sha256=FINAL_REFIT_PROTOCOL_V1_6_SHA256,
        verify_physical_file=True,
    )

    expected_model = _expected_final_refit_model_contract()
    if payload["model"] != expected_model:
        raise ValueError("Submission manifest model differs from the selected architecture")
    observed_model = {
        "architecture": "AsymmetricPartnerRetrievalModel",
        "antibody_artifact_format": args.antibody_artifact_format,
        "antibody_model_name": args.antibody_model_name,
        "antibody_model_weights_sha256": args.antibody_model_weights_sha256,
        "antigen_model_name": args.antigen_model_name,
        "antigen_model_weights_sha256": args.antigen_model_weights_sha256,
        "antibody_input_dim": args.antibody_input_dim,
        "antigen_input_dim": args.antigen_input_dim,
        "embedding_dim": args.embedding_dim,
        "antibody_hidden_dim": args.antibody_hidden_dim,
        "antigen_hidden_dim": args.antigen_hidden_dim,
        "dropout": args.dropout,
        "chain_balanced_antibody_pooling": args.chain_balanced_antibody_pooling,
        "antibody_pooling": args.antibody_pooling,
        "antigen_pooling": args.antigen_pooling,
        "initial_temperature": args.initial_temperature,
        "minimum_temperature": args.minimum_temperature,
        "maximum_temperature": args.maximum_temperature,
    }
    if observed_model != expected_model:
        raise ValueError("Final-refit model arguments differ from the submission manifest")

    expected_refit = _expected_final_refit_optimization_contract()
    if payload["fixed_refit"] != expected_refit:
        raise ValueError("Submission manifest optimization differs from the locked refit")
    observed_refit = {
        "fixed_epochs": args.fixed_epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "bucket_width": args.bucket_width,
        "training_device": args.device,
        "deterministic": not args.allow_nondeterministic,
        "isolation_policy": args.isolation_policy,
        "excluded_negative_pairs": (
            None
            if args.excluded_negative_pairs is None
            else str(args.excluded_negative_pairs.expanduser().resolve())
        ),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "weight_decay": args.weight_decay,
            "betas": [0.9, 0.98],
            "eps": 1e-8,
            "gradient_clip": args.gradient_clip,
        },
        "scheduler": {
            "name": args.schedule,
            "warmup_fraction": args.warmup_fraction,
            "warmup_fraction_used": args.schedule == "warmup_cosine_decay",
            "restart_period_epochs": args.restart_period_epochs,
            "restart_multiplier": args.restart_multiplier,
        },
    }
    if observed_refit != expected_refit:
        raise ValueError("Final-refit optimization arguments differ from the submission manifest")
    expected_evaluation = _expected_final_evaluation_contract()
    if payload["fixed_evaluation"] != expected_evaluation:
        raise ValueError(
            "Submission manifest evaluation protocol differs from the locked one-shot contract"
        )
    if list(args.recall_ks) != expected_evaluation["recall_ks"]:
        raise ValueError(
            "Final-refit recall_ks must match the predeclared one-shot metric protocol"
        )

    immutable_inputs = payload["immutable_data_inputs"]
    if not isinstance(immutable_inputs, Mapping):
        raise ValueError("immutable_data_inputs must be an object")
    _require_exact_keys(
        immutable_inputs,
        {
            "prepared_instances",
            "canonical_sequence_manifest",
            "training_embedding_content_manifest_sha256",
        },
        "immutable_data_inputs",
    )
    prepared_instances = _manifest_reference(
        immutable_inputs["prepared_instances"],
        "immutable_data_inputs.prepared_instances",
        expected_sha256=FINAL_REFIT_PREPARED_INSTANCES_SHA256,
        verify_physical_file=True,
    )
    canonical_sequence_manifest = _manifest_reference(
        immutable_inputs["canonical_sequence_manifest"],
        "immutable_data_inputs.canonical_sequence_manifest",
        expected_sha256=FINAL_REFIT_CANONICAL_SEQUENCE_MANIFEST_SHA256,
        verify_physical_file=True,
    )
    expected_embedding_manifests = {
        "antibody": FINAL_REFIT_ANTIBODY_EMBEDDING_MANIFEST_SHA256,
        "antigen": FINAL_REFIT_ANTIGEN_EMBEDDING_MANIFEST_SHA256,
    }
    if (
        immutable_inputs["training_embedding_content_manifest_sha256"]
        != expected_embedding_manifests
        or audit_evidence["training_embedding_content_manifest_sha256"]
        != expected_embedding_manifests
    ):
        raise ValueError(
            "Immutable embedding inputs differ from the independent selected-run audit"
        )
    if (
        args.prepare_dir.expanduser().resolve() / "instances.parquet"
        != Path(prepared_instances["path"])
        or args.prepared_instances_sha256 != prepared_instances["sha256"]
        or args.canonical_sequence_manifest.expanduser().resolve()
        != Path(canonical_sequence_manifest["path"])
        or args.canonical_sequence_manifest_sha256
        != canonical_sequence_manifest["sha256"]
    ):
        raise ValueError("Final-refit immutable data arguments differ from the manifest")

    authority = payload["authority"]
    if not isinstance(authority, Mapping):
        raise ValueError("authority must be an object")
    _require_exact_keys(
        authority,
        {
            "split_manifest",
            "authorized_source_pool_splits",
            "sealed_source_splits",
            "coverage_mode",
        },
        "authority",
    )
    authority_reference = _manifest_reference(
        authority["split_manifest"],
        "authority.split_manifest",
        expected_sha256=FINAL_REFIT_AUTHORITY_SHA256,
        verify_physical_file=True,
    )
    pool_names = authority["authorized_source_pool_splits"]
    sealed_names = authority["sealed_source_splits"]
    if (
        not isinstance(pool_names, list)
        or not pool_names
        or not all(isinstance(value, str) and value for value in pool_names)
        or not isinstance(sealed_names, list)
        or not sealed_names
        or not all(isinstance(value, str) and value for value in sealed_names)
    ):
        raise ValueError("Submission authority policy is incomplete or unsafe")
    if (
        len(pool_names) != len(set(pool_names))
        or len(sealed_names) != len(set(sealed_names))
        or set(pool_names) & set(sealed_names)
        or "test" not in sealed_names
        or set(pool_names) != {"train", "validation"}
        or authority["coverage_mode"] != "exact"
    ):
        raise ValueError("Submission authority policy is incomplete or unsafe")
    if (
        args.authority_split_manifest.expanduser().resolve()
        != Path(authority_reference["path"])
        or args.authority_split_sha256 != authority_reference["sha256"]
        or list(args.authorized_source_pool_splits) != pool_names
        or list(args.authority_sealed_splits) != sealed_names
        or args.authority_coverage_mode != authority["coverage_mode"]
    ):
        raise ValueError("Final-refit authority arguments differ from the submission manifest")

    boundaries = payload["boundaries"]
    if not isinstance(boundaries, Mapping) or len(boundaries) != 2:
        raise ValueError("Submission manifest must contain exactly two repaired boundaries")
    common_train_membership_sha256 = payload["common_train_membership_sha256"]
    if not isinstance(common_train_membership_sha256, str) or not _is_sha256(
        common_train_membership_sha256
    ):
        raise ValueError("common_train_membership_sha256 must be a SHA-256 digest")
    canonical_refit_boundary = payload["canonical_refit_boundary"]
    if not isinstance(canonical_refit_boundary, str) or not canonical_refit_boundary:
        raise ValueError("canonical_refit_boundary must name one declared boundary")
    refit_output_dir = _absolute_path(payload["refit_output_dir"], "refit_output_dir")
    one_shot_registry = _absolute_path(
        payload["one_shot_ledger_registry_dir"], "one_shot_ledger_registry_dir"
    )
    if _path_trees_overlap(refit_output_dir, one_shot_registry):
        raise ValueError("Refit output and one-shot registry path trees must be disjoint")
    normalized_boundaries: dict[str, Any] = {}
    evaluation_output_paths: set[str] = set()
    score_paths: set[str] = set()
    all_artifact_paths: set[str] = set()
    split_contracts: set[tuple[str, str, str]] = set()
    common_train_ids: tuple[str, ...] | None = None
    for name, raw_boundary in boundaries.items():
        if (
            not isinstance(name, str)
            or not name
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name)
            or not isinstance(raw_boundary, Mapping)
        ):
            raise ValueError("Boundary names and objects must follow the v1 contract")
        _require_exact_keys(
            raw_boundary,
            {
                "split_manifest",
                "strict_nw_constraint_mode",
                "evaluation_output",
                "score_artifact",
            },
            f"boundaries.{name}",
        )
        split_reference = _manifest_reference(
            raw_boundary["split_manifest"],
            f"boundaries.{name}.split_manifest",
            verify_physical_file=True,
        )
        if split_reference["sha256"] in RETIRED_CONTAMINATED_BOUNDARY_SHA256:
            raise ValueError(
                f"boundaries.{name} references a retired post-selection-contaminated split"
            )
        mode = raw_boundary["strict_nw_constraint_mode"]
        if not isinstance(mode, str) or mode not in STRICT_NW_CONSTRAINT_MODES:
            raise ValueError(f"boundaries.{name} has an unsupported NW constraint mode")
        boundary_payload = _strict_json_object(
            Path(split_reference["path"]), f"boundaries.{name}.split_manifest"
        )
        if (
            boundary_payload.get("format_version") != AUTHORITY_VALIDATION_BOUNDARY_FORMAT
            or boundary_payload.get("intended_use")
            != "reused_development_confirmation_only"
            or boundary_payload.get("valid") is not True
            or boundary_payload.get("constraint_mode") != mode
        ):
            raise ValueError(f"boundaries.{name} is not an approved repaired v2 boundary")
        boundary_scientific_sha256 = boundary_payload.get(
            "scientific_payload_sha256"
        )
        boundary_scientific_payload = dict(boundary_payload)
        boundary_scientific_payload.pop("scientific_payload_sha256", None)
        if (
            not isinstance(boundary_scientific_sha256, str)
            or not _is_sha256(boundary_scientific_sha256)
            or canonical_sha256(boundary_scientific_payload)
            != boundary_scientific_sha256
        ):
            raise ValueError(
                f"boundaries.{name} has an invalid scientific-payload digest"
            )
        raw_boundary_splits = boundary_payload.get("splits")
        if not isinstance(raw_boundary_splits, Mapping) or set(raw_boundary_splits) != {
            "train",
            "validation",
            "excluded_validation",
        }:
            raise ValueError(f"boundaries.{name} lacks exact three-way accounting")
        raw_train_ids = raw_boundary_splits.get("train")
        if (
            not isinstance(raw_train_ids, list)
            or not raw_train_ids
            or not all(isinstance(value, str) and value for value in raw_train_ids)
            or len(raw_train_ids) != len(set(raw_train_ids))
        ):
            raise ValueError(f"boundaries.{name} has invalid train membership")
        train_ids = tuple(sorted(raw_train_ids))
        observed_train_membership_sha256 = _identifier_membership_sha256(train_ids)
        boundary_membership = boundary_payload.get("membership_sha256")
        if (
            observed_train_membership_sha256 != common_train_membership_sha256
            or not isinstance(boundary_membership, Mapping)
            or boundary_membership.get("train") != common_train_membership_sha256
        ):
            raise ValueError(f"boundaries.{name} does not share the locked train membership")
        if common_train_ids is None:
            common_train_ids = train_ids
        elif train_ids != common_train_ids:
            raise ValueError("Repaired boundaries do not have identical train identifiers")
        boundary_provenance = boundary_payload.get("provenance")
        correction = (
            boundary_provenance.get("protocol_correction", {})
            if isinstance(boundary_provenance, Mapping)
            else {}
        )
        if correction.get("sha256") != FINAL_REFIT_PROTOCOL_V1_4_SHA256:
            raise ValueError(f"boundaries.{name} lacks the locked v1.4 correction")
        evaluation_output = _absolute_path(
            raw_boundary["evaluation_output"], f"boundaries.{name}.evaluation_output"
        )
        score_artifact = _absolute_path(
            raw_boundary["score_artifact"], f"boundaries.{name}.score_artifact"
        )
        if any(
            _path_trees_overlap(artifact, protected)
            for artifact in (evaluation_output, score_artifact)
            for protected in (refit_output_dir, one_shot_registry)
        ):
            raise ValueError(
                "Evaluation and score artifacts must be outside protected refit/registry trees"
            )
        split_contract = (split_reference["path"], split_reference["sha256"], mode)
        if (
            str(evaluation_output) in evaluation_output_paths
            or str(score_artifact) in score_paths
            or evaluation_output == score_artifact
            or split_contract in split_contracts
            or str(evaluation_output) in all_artifact_paths
            or str(score_artifact) in all_artifact_paths
        ):
            raise ValueError(
                "Submission boundaries must have globally distinct artifact paths"
            )
        evaluation_output_paths.add(str(evaluation_output))
        score_paths.add(str(score_artifact))
        all_artifact_paths.update((str(evaluation_output), str(score_artifact)))
        split_contracts.add(split_contract)
        normalized_boundaries[name] = {
            "split_manifest": split_reference,
            "strict_nw_constraint_mode": mode,
            "scientific_payload_sha256": boundary_scientific_sha256,
            "train_membership_sha256": observed_train_membership_sha256,
            "evaluation_output": str(evaluation_output),
            "score_artifact": str(score_artifact),
        }
    if set(boundary["strict_nw_constraint_mode"] for boundary in normalized_boundaries.values()) != set(
        STRICT_NW_CONSTRAINT_MODES
    ):
        raise ValueError("Submission manifest must bind both approved repaired constraint modes")
    if canonical_refit_boundary not in normalized_boundaries:
        raise ValueError("canonical_refit_boundary is not declared")
    boundary_verification_evidence = _validate_boundary_verification_evidence(
        upstream["boundary_verification_v2"], normalized_boundaries
    )
    requested_contract = (
        str(args.split_manifest.expanduser().resolve()),
        args.split_manifest_sha256,
        args.strict_nw_constraint_mode,
    )
    canonical_boundary = normalized_boundaries[canonical_refit_boundary]
    canonical_contract = (
        canonical_boundary["split_manifest"]["path"],
        canonical_boundary["split_manifest"]["sha256"],
        canonical_boundary["strict_nw_constraint_mode"],
    )
    if requested_contract != canonical_contract or args.output_dir.expanduser().resolve() != refit_output_dir:
        raise ValueError(
            "Final-refit arguments do not match the canonical boundary and common output"
        )
    selected_boundary = canonical_refit_boundary

    source_files = payload["source_files"]
    current_source_sha256 = _source_hashes()
    if not isinstance(source_files, Mapping):
        raise ValueError("source_files must be an object")
    _require_exact_keys(source_files, set(current_source_sha256), "source_files")
    normalized_sources = {
        name: _manifest_reference(
            source_files[name],
            f"source_files.{name}",
            expected_sha256=current_source_sha256[name],
            verify_physical_file=True,
        )
        for name in sorted(current_source_sha256)
    }
    submission_script = _manifest_reference(
        payload["submission_script"],
        "submission_script",
        verify_physical_file=True,
    )
    script_text = Path(submission_script["path"]).read_text(encoding="utf-8")
    required_tokens = (
        "final-refit",
        "RETRIEVAL_SUBMISSION_MANIFEST",
        "RETRIEVAL_SUBMISSION_MANIFEST_SHA256",
        '--submission-manifest "$SUBMISSION_MANIFEST"',
        '--submission-manifest-sha256 "$SUBMISSION_MANIFEST_SHA256"',
    )
    if any(token not in script_text for token in required_tokens):
        raise ValueError("Submission-script snapshot lacks required manifest enforcement")

    return {
        "format_version": FINAL_REFIT_SUBMISSION_FORMAT_VERSION,
        "submission_manifest": {
            "path": str(manifest_path),
            "sha256": observed_manifest_sha256,
        },
        "validated_before_dataset_or_embedding_load": True,
        "intended_use": "development_only",
        "frozen_test_used": False,
        "selected_variant": FINAL_REFIT_SELECTED_VARIANT,
        "selected_boundary": selected_boundary,
        "canonical_refit_boundary": selected_boundary,
        "boundary": normalized_boundaries[selected_boundary],
        "all_declared_boundaries": normalized_boundaries,
        "common_train_membership_sha256": common_train_membership_sha256,
        "refit_output_dir": str(refit_output_dir),
        "one_shot_ledger_registry_dir": str(one_shot_registry),
        "upstream_evidence": {
            "fold_selection": fold_evidence,
            "independent_audit": audit_evidence,
            "protocol_v1_3": protocol_evidence,
            "protocol_v1_4_correction": protocol_correction_evidence,
            "protocol_v1_5_execution_lock": protocol_execution_evidence,
            "protocol_v1_6_execution_hardening": protocol_hardening_evidence,
            "boundary_verification_v2": boundary_verification_evidence,
        },
        "authority": {
            "split_manifest": authority_reference,
            "authorized_source_pool_splits": list(pool_names),
            "sealed_source_splits": list(sealed_names),
            "coverage_mode": authority["coverage_mode"],
        },
        "immutable_data_inputs": {
            "prepared_instances": prepared_instances,
            "canonical_sequence_manifest": canonical_sequence_manifest,
            "training_embedding_content_manifest_sha256": expected_embedding_manifests,
        },
        "model": expected_model,
        "fixed_refit": expected_refit,
        "fixed_evaluation": expected_evaluation,
        "source_files": normalized_sources,
        "source_sha256": current_source_sha256,
        "submission_script": submission_script,
    }


def _config(
    args: argparse.Namespace,
    dataset_metadata: Mapping[str, Any],
    embedding_corpora: Mapping[str, Any],
    train_records: Sequence[PairRecord],
    validation_records: Sequence[PairRecord],
    duplicates: Mapping[str, int],
    isolation: Mapping[str, int],
    negative_policy: Mapping[str, Any],
    train_batch_count: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model": {
            "architecture": "AsymmetricPartnerRetrievalModel",
            "antibody_artifact_format": args.antibody_artifact_format,
            "antibody_encoder": args.antibody_model_name,
            "antibody_model_weights_sha256": args.antibody_model_weights_sha256,
            "antigen_encoder": args.antigen_model_name,
            "antigen_model_weights_sha256": args.antigen_model_weights_sha256,
            "antibody_input_dim": args.antibody_input_dim,
            "antigen_input_dim": args.antigen_input_dim,
            "embedding_dim": args.embedding_dim,
            "antibody_hidden_dim": args.antibody_hidden_dim,
            "antigen_hidden_dim": args.antigen_hidden_dim,
            "dropout": args.dropout,
            "chain_balanced_antibody_pooling": args.chain_balanced_antibody_pooling,
            "antibody_pooling": args.antibody_pooling,
            "antigen_pooling": args.antigen_pooling,
            "projection_order": "per_residue_ffn_then_length_pool_then_l2_normalize",
            "initial_temperature": args.initial_temperature,
            "minimum_temperature": args.minimum_temperature,
            "maximum_temperature": args.maximum_temperature,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "weight_decay": args.weight_decay,
            "schedule": args.schedule,
            "warmup_fraction": args.warmup_fraction,
            "restart_period_epochs": args.restart_period_epochs,
            "restart_multiplier": args.restart_multiplier,
            "gradient_clip": args.gradient_clip,
            "batch_size": args.batch_size,
            "bucket_width": args.bucket_width,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "seed": args.seed,
            "deterministic": not args.allow_nondeterministic,
            "train_batch_count": train_batch_count,
        },
        "validation": {
            "protocol": "fixed_full_library_bidirectional_development_validation",
            "checkpoint_selection": "bidirectional_validation_mrr",
            "recall_ks": sorted(set(args.recall_ks)),
            "validation_batch_size": args.validation_batch_size,
            "pair_count": len(validation_records),
            "exact_pair_sha256": canonical_sha256(
                {"pairs": sorted([list(value) for value in _pair_set(validation_records)])}
            ),
        },
        "training_data": {
            "pair_count_after_exact_deduplication": len(train_records),
            "exact_pair_sha256": canonical_sha256(
                {"pairs": sorted([list(value) for value in _pair_set(train_records)])}
            ),
            "duplicate_counts": dict(duplicates),
            "train_validation_isolation": dict(isolation),
            "isolation_policy": args.isolation_policy,
            "positive_mask_policy": "all_observed_exact_pairs_within_training_split",
            "negative_policy": dict(negative_policy),
        },
        "dataset": dict(dataset_metadata),
        "embedding_corpora": dict(embedding_corpora),
        "source_sha256": _source_hashes(),
        "scientific_scope": {
            "frozen_test_used": False,
            "model_selection_data": "development validation only",
            "claims_allowed": "exploratory development-screen evidence only",
        },
    }
    config["config_sha256"] = canonical_sha256(config)
    return config


def _final_refit_config(
    args: argparse.Namespace,
    submission_provenance: Mapping[str, Any],
    dataset_metadata: Mapping[str, Any],
    embedding_corpora: Mapping[str, Any],
    train_records: Sequence[PairRecord],
    duplicates_removed: int,
    negative_policy: Mapping[str, Any],
    train_batch_count: int,
) -> dict[str, Any]:
    """Build a fixed-epoch refit contract without holdout pair information."""
    config: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "run_mode": "final_refit_fixed_epochs",
        "model": {
            "architecture": "AsymmetricPartnerRetrievalModel",
            "antibody_artifact_format": args.antibody_artifact_format,
            "antibody_encoder": args.antibody_model_name,
            "antibody_model_weights_sha256": args.antibody_model_weights_sha256,
            "antigen_encoder": args.antigen_model_name,
            "antigen_model_weights_sha256": args.antigen_model_weights_sha256,
            "antibody_input_dim": args.antibody_input_dim,
            "antigen_input_dim": args.antigen_input_dim,
            "embedding_dim": args.embedding_dim,
            "antibody_hidden_dim": args.antibody_hidden_dim,
            "antigen_hidden_dim": args.antigen_hidden_dim,
            "dropout": args.dropout,
            "chain_balanced_antibody_pooling": args.chain_balanced_antibody_pooling,
            "antibody_pooling": args.antibody_pooling,
            "antigen_pooling": args.antigen_pooling,
            "projection_order": "per_residue_ffn_then_length_pool_then_l2_normalize",
            "initial_temperature": args.initial_temperature,
            "minimum_temperature": args.minimum_temperature,
            "maximum_temperature": args.maximum_temperature,
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "weight_decay": args.weight_decay,
            "schedule": args.schedule,
            "warmup_fraction": args.warmup_fraction,
            "restart_period_epochs": args.restart_period_epochs,
            "restart_multiplier": args.restart_multiplier,
            "gradient_clip": args.gradient_clip,
            "batch_size": args.batch_size,
            "bucket_width": args.bucket_width,
            "fixed_epochs": args.fixed_epochs,
            "seed": args.seed,
            "deterministic": not args.allow_nondeterministic,
            "train_batch_count": train_batch_count,
            "checkpoint_selection": None,
            "early_stopping": False,
        },
        "fixed_holdout_firewall": {
            "validation_pair_rows_materialized_during_refit": False,
            "validation_embeddings_loaded_during_refit": False,
            "validation_metrics_computed_during_refit": False,
            "validation_membership_count": dataset_metadata["raw_instance_counts"][
                "validation"
            ],
            "validation_membership_sha256": dataset_metadata[
                "split_membership_sha256"
            ]["validation"],
            "excluded_validation_pair_rows_materialized_during_refit": False,
            "excluded_validation_embeddings_loaded_during_refit": False,
            "excluded_validation_metrics_computed_during_refit": False,
            "excluded_validation_membership_count": dataset_metadata[
                "raw_instance_counts"
            ]["excluded_validation"],
            "excluded_validation_membership_sha256": dataset_metadata[
                "split_membership_sha256"
            ]["excluded_validation"],
            "evaluation_policy": "one_shot_after_fixed_epoch_refit",
        },
        "training_data": {
            "pair_count_after_exact_deduplication": len(train_records),
            "exact_pair_sha256": canonical_sha256(
                {"pairs": sorted([list(value) for value in _pair_set(train_records)])}
            ),
            "exact_duplicate_pairs_removed": duplicates_removed,
            "isolation_policy_to_audit_at_evaluation": args.isolation_policy,
            "positive_mask_policy": "all_observed_exact_pairs_within_training_split",
            "negative_policy": dict(negative_policy),
        },
        "dataset": dict(dataset_metadata),
        "embedding_corpora": dict(embedding_corpora),
        "source_sha256": _source_hashes(),
        "submission_provenance": dict(submission_provenance),
        "scientific_scope": {
            "frozen_test_used": False,
            "model_selection_data": "train-only folds completed before final refit",
            "fixed_holdout_used_during_refit": False,
            "claims_allowed": "no claim before one-shot development evaluation",
        },
    }
    config["config_sha256"] = canonical_sha256(config)
    return config


def _provenance(config: Mapping[str, Any], args: argparse.Namespace, device: Any) -> dict[str, Any]:
    import torch

    return {
        "format_version": FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config["config_sha256"],
        "source_sha256": config["source_sha256"],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "embedding_locations": {
            "antibody": str(args.antibody_embedding_dir.expanduser().resolve()),
            "antigen": str(args.antigen_embedding_dir.expanduser().resolve()),
        },
        "scientific_scope": config["scientific_scope"],
    }


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    epoch: int,
    global_step: int,
    best_validation_mrr: float,
    best_epoch: int,
    epochs_without_improvement: int,
) -> dict[str, Any]:
    import torch

    return {
        "format_version": FORMAT_VERSION,
        "config": dict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_mrr": best_validation_mrr,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "history": list(history),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _embedding_corpora(
    args: argparse.Namespace,
    records: Sequence[PairRecord],
) -> dict[str, Any]:
    return {
        "antibody": side_embedding_fingerprint(
            args.antibody_embedding_dir,
            (record.antibody_key for record in records),
            "antibody",
        ),
        "antigen": side_embedding_fingerprint(
            args.antigen_embedding_dir,
            (record.antigen_key for record in records),
            "antigen",
        ),
    }


def _stores(args: argparse.Namespace) -> tuple[FrozenPartnerEmbeddingStore, FrozenPartnerEmbeddingStore]:
    return (
        FrozenPartnerEmbeddingStore(
            args.antibody_embedding_dir,
            "antibody",
            args.antibody_artifact_format,
            args.antibody_input_dim,
            args.antibody_model_name,
            args.antibody_model_weights_sha256,
            args.antibody_cache_gib,
        ),
        FrozenPartnerEmbeddingStore(
            args.antigen_embedding_dir,
            "antigen",
            "esm_chunks",
            args.antigen_input_dim,
            args.antigen_model_name,
            args.antigen_model_weights_sha256,
            args.antigen_cache_gib,
        ),
    )


def _dataset_from_args(
    args: argparse.Namespace,
    require_authority_validation_boundary_v2: bool = False,
) -> tuple[dict[str, list[PairRecord]], dict[str, Any]]:
    """Load an authority-checked development dataset from command arguments."""
    return load_development_dataset(
        args.prepare_dir,
        args.split_manifest,
        args.split_manifest_sha256,
        args.authority_split_manifest,
        args.authority_split_sha256,
        args.authorized_source_pool_splits,
        args.authority_sealed_splits,
        args.authority_coverage_mode,
        args.strict_nw_constraint_mode,
        require_authority_validation_boundary_v2=require_authority_validation_boundary_v2,
        expected_instances_sha256=getattr(args, "prepared_instances_sha256", None),
        canonical_sequence_manifest=getattr(
            args, "canonical_sequence_manifest", None
        ),
        canonical_sequence_manifest_sha256=getattr(
            args, "canonical_sequence_manifest_sha256", None
        ),
    )


def _final_refit_dataset_from_args(
    args: argparse.Namespace,
) -> tuple[dict[str, list[PairRecord]], dict[str, Any]]:
    """Load training rows while keeping fixed-holdout pair columns unmaterialized."""
    return load_development_dataset(
        args.prepare_dir,
        args.split_manifest,
        args.split_manifest_sha256,
        args.authority_split_manifest,
        args.authority_split_sha256,
        args.authorized_source_pool_splits,
        args.authority_sealed_splits,
        args.authority_coverage_mode,
        args.strict_nw_constraint_mode,
        materialized_splits=("train",),
        require_authority_validation_boundary_v2=True,
        expected_instances_sha256=args.prepared_instances_sha256,
        canonical_sequence_manifest=args.canonical_sequence_manifest,
        canonical_sequence_manifest_sha256=args.canonical_sequence_manifest_sha256,
    )


def command_train(args: argparse.Namespace) -> None:
    """Train and select checkpoints without exposing any frozen test split."""
    import torch

    from interaction_first.runtime import atomic_torch_save, safe_torch_load, set_reproducible_seed

    _validate_numeric_arguments(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    set_reproducible_seed(args.seed, deterministic=not args.allow_nondeterministic)

    splits, dataset_metadata = _dataset_from_args(args)
    train_records, train_duplicates = deduplicate_exact_pairs(splits["train"])
    validation_records, validation_duplicates = deduplicate_exact_pairs(splits["validation"])
    if len(train_records) < args.min_train_pairs:
        raise ValueError(f"Training has {len(train_records)} pairs; minimum={args.min_train_pairs}")
    if len(validation_records) < args.min_validation_pairs:
        raise ValueError(
            f"Validation has {len(validation_records)} pairs; minimum={args.min_validation_pairs}"
        )
    isolation = audit_isolation_policy(
        train_records, validation_records, args.isolation_policy
    )
    train_batches = length_bucket_batches(
        train_records, args.batch_size, args.bucket_width, args.seed, epoch=1
    )
    excluded_pairs, negative_policy = load_excluded_negative_pairs(
        args.excluded_negative_pairs, train_records
    )
    corpora = _embedding_corpora(args, [*train_records, *validation_records])
    config = _config(
        args,
        dataset_metadata,
        corpora,
        train_records,
        validation_records,
        {"train_removed": train_duplicates, "validation_removed": validation_duplicates},
        isolation,
        negative_policy,
        len(train_batches),
    )

    model = _model_from_config(config, device)
    optimizer = _optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = _scheduler(optimizer, len(train_batches), args)
    antibody_store, antigen_store = _stores(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.json"
    provenance_path = output_dir / "provenance.json"
    existing = [path for path in (last_path, best_path, history_path) if path.exists()]
    if not args.resume and existing:
        raise FileExistsError(f"Refusing to overwrite an existing run: {[str(path) for path in existing]}")
    if args.resume and not last_path.is_file() and existing:
        raise ValueError("Run artifacts exist but last.pt is missing; resume is unsafe")

    provenance = _provenance(config, args, device)
    if provenance_path.is_file():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config["config_sha256"]:
            raise ValueError("Existing provenance belongs to a different configuration")
    else:
        atomic_write_json(provenance_path, provenance)

    start_epoch = 1
    global_step = 0
    best_validation_mrr = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if args.resume and last_path.is_file():
        checkpoint = safe_torch_load(last_path, map_location=device)
        if checkpoint.get("format_version") != FORMAT_VERSION:
            raise ValueError("Checkpoint format is incompatible")
        previous_config = checkpoint.get("config", {})
        validate_config_integrity(previous_config)
        if previous_config.get("config_sha256") != config["config_sha256"]:
            raise ValueError("Checkpoint configuration, source, or input digests changed")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_validation_mrr = float(checkpoint["best_validation_mrr"])
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history = list(checkpoint.get("history", []))
        torch.set_rng_state(checkpoint["torch_rng_state"].to(device="cpu"))
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(
                [state.to(device="cpu") for state in checkpoint["cuda_rng_state_all"]]
            )
    if epochs_without_improvement >= args.patience:
        print("The resumed checkpoint already satisfies early stopping.", flush=True)
        start_epoch = args.max_epochs + 1

    known_positive_pairs = _pair_set(train_records)
    for epoch in range(start_epoch, args.max_epochs + 1):
        started = time.time()
        train_metrics, global_step = _train_epoch(
            model,
            train_records,
            known_positive_pairs,
            excluded_pairs,
            antibody_store,
            antigen_store,
            optimizer,
            scheduler,
            device,
            args,
            epoch,
            global_step,
        )
        validation_metrics = evaluate_records(
            model,
            validation_records,
            antibody_store,
            antigen_store,
            args.validation_batch_size,
            args.bucket_width,
            device,
            args.recall_ks,
        )
        validation_mrr = float(validation_metrics["bidirectional_mrr"])
        if not math.isfinite(validation_mrr):
            raise FloatingPointError("Validation MRR is non-finite")
        improved = validation_mrr > best_validation_mrr + args.min_delta
        if improved:
            best_validation_mrr = validation_mrr
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_metric": "bidirectional_validation_mrr",
            "best_validation_mrr": best_validation_mrr,
            "best_epoch": best_epoch,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "temperature": float(model.temperature_head().detach().cpu().item()),
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        checkpoint_payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            config,
            history,
            epoch,
            global_step,
            best_validation_mrr,
            best_epoch,
            epochs_without_improvement,
        )
        if improved:
            atomic_torch_save(checkpoint_payload, best_path)
        atomic_torch_save(checkpoint_payload, last_path)
        atomic_write_json(
            history_path,
            {
                "format_version": FORMAT_VERSION,
                "config": config,
                "history": history,
                "embedding_caches": {
                    "antibody": antibody_store.summary(),
                    "antigen": antigen_store.summary(),
                },
            },
        )
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError("Training did not produce a best checkpoint")
    summary = {
        "format_version": FORMAT_VERSION,
        "stage": "train_complete",
        "config_sha256": config["config_sha256"],
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "best_validation_mrr": best_validation_mrr,
        "best_epoch": best_epoch,
        "completed_epoch": history[-1]["epoch"] if history else start_epoch - 1,
        "frozen_test_used": False,
    }
    atomic_write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


def _assert_finite_tensor_tree(value: Any, label: str) -> None:
    """Reject non-finite tensors nested in a trusted checkpoint structure."""
    import torch

    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise ValueError(f"{label} contains a non-finite tensor")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_finite_tensor_tree(nested, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_finite_tensor_tree(nested, f"{label}[{index}]")


def _validate_execution_provenance_history(
    history: Any,
    current: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate runtime provenance, allowing only timestamp and job-ID drift."""
    expected_keys = {
        "format_version",
        "created_at_utc",
        "config_sha256",
        "source_sha256",
        "runtime",
        "embedding_locations",
        "scientific_scope",
    }
    runtime_keys = {
        "python",
        "platform",
        "torch",
        "device",
        "cuda_available",
        "cuda_version",
        "gpu_name",
        "slurm_job_id",
    }
    if not isinstance(history, list) or not history:
        raise ValueError("Execution provenance history must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    baseline_semantics = None
    for index, raw in enumerate(history):
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise ValueError(f"Execution provenance entry {index} has unexpected fields")
        runtime = raw.get("runtime")
        if not isinstance(runtime, Mapping) or set(runtime) != runtime_keys:
            raise ValueError(f"Execution provenance entry {index} has invalid runtime fields")
        semantics = {
            **{key: raw[key] for key in expected_keys - {"created_at_utc", "runtime"}},
            "runtime": {
                key: runtime[key] for key in runtime_keys - {"slurm_job_id"}
            },
        }
        if baseline_semantics is None:
            baseline_semantics = semantics
        elif semantics != baseline_semantics:
            raise ValueError("Scientific runtime provenance changed across resume jobs")
        normalized.append(dict(raw))
    if current is not None:
        combined = [*normalized, dict(current)]
        _validate_execution_provenance_history(combined)
    return normalized


def _validate_final_refit_resume_state(
    state: Any,
    config: Mapping[str, Any],
    submission_provenance: Mapping[str, Any],
    fixed_epochs: int,
    current_execution_provenance: Mapping[str, Any],
    model: Any,
    optimizer: Any,
    scheduler: Any,
) -> list[dict[str, Any]]:
    """Fail closed on malformed, advanced, or scientifically drifting resume state."""
    import torch

    expected_keys = {
        "format_version",
        "stage",
        "config",
        "submission_provenance",
        "execution_provenance_history",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
        "global_step",
        "history",
        "torch_rng_state",
        "cuda_rng_state_all",
    }
    if not isinstance(state, Mapping) or set(state) != expected_keys:
        raise ValueError("Final-refit state has missing or unreviewed top-level fields")
    if (
        state.get("format_version") != FORMAT_VERSION
        or state.get("stage") != "final_refit_in_progress"
    ):
        raise ValueError("Final-refit state format is incompatible")
    previous_config = state.get("config")
    if not isinstance(previous_config, Mapping):
        raise ValueError("Final-refit state config is malformed")
    validate_config_integrity(previous_config)
    if previous_config.get("config_sha256") != config["config_sha256"]:
        raise ValueError("Final-refit state belongs to a different fixed contract")
    if state.get("submission_provenance") != submission_provenance:
        raise ValueError("Final-refit state submission provenance changed")
    epoch = state.get("epoch")
    global_step = state.get("global_step")
    if type(epoch) is not int or not 1 <= epoch <= fixed_epochs:
        raise ValueError("Final-refit state epoch is outside the locked range")
    if type(global_step) is not int or global_step < 0:
        raise ValueError("Final-refit state global_step is invalid")
    history = state.get("history")
    if not isinstance(history, list) or len(history) != epoch:
        raise ValueError("Final-refit state history length does not match its epoch")
    expected_history_keys = {
        "epoch",
        "global_step",
        "train",
        "learning_rate",
        "temperature",
        "validation_metrics_computed",
    }
    previous_step = -1
    for expected_epoch, record in enumerate(history, 1):
        if not isinstance(record, Mapping) or set(record) != expected_history_keys:
            raise ValueError("Final-refit state history has unexpected fields")
        if (
            record.get("epoch") != expected_epoch
            or type(record.get("global_step")) is not int
            or record["global_step"] <= previous_step
            or record.get("validation_metrics_computed") is not False
        ):
            raise ValueError("Final-refit state history is noncontiguous or inconsistent")
        previous_step = record["global_step"]
    if history[-1]["epoch"] != epoch or history[-1]["global_step"] != global_step:
        raise ValueError("Final-refit state history tail differs from the saved state")
    try:
        json.dumps(history, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Final-refit state history contains non-finite values") from error
    model_state = state.get("model_state_dict")
    expected_model_keys = set(model.state_dict())
    if not isinstance(model_state, Mapping) or set(model_state) != expected_model_keys:
        raise ValueError("Final-refit state model structure changed")
    _assert_finite_tensor_tree(model_state, "model_state_dict")
    optimizer_state = state.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping) or set(optimizer_state) != {
        "state",
        "param_groups",
    }:
        raise ValueError("Final-refit optimizer state is malformed")
    serialized_groups = optimizer_state.get("param_groups")
    if (
        not isinstance(serialized_groups, list)
        or len(serialized_groups) != len(optimizer.param_groups)
        or any(
            not isinstance(group, Mapping)
            or not isinstance(group.get("params"), list)
            or len(group["params"]) != len(optimizer.param_groups[index]["params"])
            for index, group in enumerate(serialized_groups)
        )
    ):
        raise ValueError("Final-refit optimizer parameter groups changed")
    _assert_finite_tensor_tree(optimizer_state, "optimizer_state_dict")
    scheduler_state = state.get("scheduler_state_dict")
    expected_scheduler_keys = set(scheduler.state_dict())
    if not isinstance(scheduler_state, Mapping) or set(
        scheduler_state
    ) != expected_scheduler_keys:
        raise ValueError("Final-refit scheduler state is malformed")
    _assert_finite_tensor_tree(scheduler_state, "scheduler_state_dict")
    if scheduler_state.get("last_epoch") != global_step:
        raise ValueError("Final-refit scheduler step differs from global_step")
    torch_rng_state = state.get("torch_rng_state")
    cuda_rng_state_all = state.get("cuda_rng_state_all")
    if (
        not torch.is_tensor(torch_rng_state)
        or torch_rng_state.dtype != torch.uint8
        or torch_rng_state.ndim != 1
        or not isinstance(cuda_rng_state_all, list)
        or any(
            not torch.is_tensor(value)
            or value.dtype != torch.uint8
            or value.ndim != 1
            for value in cuda_rng_state_all
        )
    ):
        raise ValueError("Final-refit RNG state is malformed")
    previous_execution_history = _validate_execution_provenance_history(
        state.get("execution_provenance_history"), current_execution_provenance
    )
    return [*previous_execution_history, dict(current_execution_provenance)]


def _validate_final_refit_checkpoint(checkpoint: Any) -> dict[str, Any]:
    """Validate a completed final-refit checkpoint before holdout access."""
    expected_keys = {
        "format_version",
        "stage",
        "config",
        "submission_provenance",
        "execution_provenance_history",
        "model_state_dict",
        "epoch",
        "global_step",
        "history",
        "validation_metrics_computed",
        "frozen_test_used",
    }
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != expected_keys:
        raise ValueError("Final checkpoint has missing or unreviewed top-level fields")
    if (
        checkpoint.get("format_version") != FORMAT_VERSION
        or checkpoint.get("stage") != "final_refit_complete"
        or checkpoint.get("validation_metrics_computed") is not False
        or checkpoint.get("frozen_test_used") is not False
    ):
        raise ValueError("Final checkpoint is incomplete or violates the holdout firewall")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Final checkpoint config is malformed")
    validate_config_integrity(config)
    if config.get("run_mode") != "final_refit_fixed_epochs":
        raise ValueError("Final checkpoint does not use the locked final-refit mode")
    submission_provenance = checkpoint.get("submission_provenance")
    if (
        not isinstance(submission_provenance, Mapping)
        or submission_provenance != config.get("submission_provenance")
        or submission_provenance.get("format_version")
        != FINAL_REFIT_SUBMISSION_FORMAT_VERSION
    ):
        raise ValueError("Final checkpoint submission provenance is inconsistent")
    optimization = config.get("optimization")
    fixed_refit = submission_provenance.get("fixed_refit")
    if not isinstance(optimization, Mapping) or not isinstance(
        fixed_refit, Mapping
    ):
        raise ValueError("Final checkpoint fixed-refit contract is malformed")
    fixed_epochs = optimization.get("fixed_epochs")
    epoch = checkpoint.get("epoch")
    global_step = checkpoint.get("global_step")
    if (
        type(fixed_epochs) is not int
        or type(epoch) is not int
        or epoch != fixed_epochs
        or epoch != fixed_refit.get("fixed_epochs")
        or epoch <= 0
    ):
        raise ValueError("Final checkpoint did not reach the locked epoch count")
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("Final checkpoint global_step is invalid")
    history = checkpoint.get("history")
    expected_history_keys = {
        "epoch",
        "global_step",
        "train",
        "learning_rate",
        "temperature",
        "validation_metrics_computed",
    }
    if not isinstance(history, list) or len(history) != epoch:
        raise ValueError("Final checkpoint history length differs from its epoch")
    previous_step = 0
    for expected_epoch, record in enumerate(history, 1):
        if not isinstance(record, Mapping) or set(record) != expected_history_keys:
            raise ValueError("Final checkpoint history has unexpected fields")
        if (
            record.get("epoch") != expected_epoch
            or type(record.get("global_step")) is not int
            or record["global_step"] <= previous_step
            or record.get("validation_metrics_computed") is not False
        ):
            raise ValueError("Final checkpoint history is noncontiguous or inconsistent")
        previous_step = record["global_step"]
    if history[-1]["global_step"] != global_step:
        raise ValueError("Final checkpoint history tail differs from global_step")
    try:
        json.dumps(history, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Final checkpoint history contains non-finite values") from error
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, Mapping) or not model_state:
        raise ValueError("Final checkpoint model state is empty or malformed")
    _assert_finite_tensor_tree(model_state, "model_state_dict")
    execution_history = _validate_execution_provenance_history(
        checkpoint.get("execution_provenance_history")
    )
    for entry in execution_history:
        if (
            entry["format_version"] != FORMAT_VERSION
            or entry["config_sha256"] != config["config_sha256"]
            or entry["source_sha256"] != config.get("source_sha256")
            or entry["scientific_scope"] != config.get("scientific_scope")
        ):
            raise ValueError("Final checkpoint execution provenance is not config-bound")
    return dict(config)


def _read_torch_checkpoint_once(
    path: Path, map_location: Any
) -> tuple[Any, str]:
    """Deserialize and hash the same stable checkpoint byte string once."""
    from interaction_first.runtime import safe_torch_load

    checkpoint_path = path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint_path}")
    with checkpoint_path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        checkpoint_bytes = handle.read()
        after = os.fstat(handle.fileno())
    current = checkpoint_path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_current = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    )
    if identity_before != identity_after or identity_after != identity_current:
        raise RuntimeError("Checkpoint path changed while its immutable bytes were acquired")
    observed_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = safe_torch_load(io.BytesIO(checkpoint_bytes), map_location=map_location)
    return checkpoint, observed_sha256


def _load_torch_checkpoint_once(
    path: Path, expected_sha256: str, map_location: Any
) -> tuple[Any, str]:
    """Load one checkpoint only when its exact reviewed digest matches."""
    if not _is_sha256(expected_sha256):
        raise ValueError("An exact checkpoint SHA-256 is required")
    checkpoint, observed_sha256 = _read_torch_checkpoint_once(path, map_location)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Checkpoint SHA-256 mismatch: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    return checkpoint, observed_sha256


def _final_refit_summary_payload(
    config: Mapping[str, Any],
    submission_provenance: Mapping[str, Any],
    final_path: Path,
    final_sha256: str,
    completed_epoch: int,
) -> dict[str, Any]:
    """Build the deterministic completion summary for one authenticated refit."""
    return {
        "format_version": FORMAT_VERSION,
        "stage": "final_refit_complete",
        "config_sha256": config["config_sha256"],
        "submission_manifest": submission_provenance["submission_manifest"],
        "canonical_refit_boundary": submission_provenance[
            "canonical_refit_boundary"
        ],
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": final_sha256,
        "fixed_epochs": completed_epoch,
        "completed_epoch": completed_epoch,
        "checkpoint_selection": None,
        "validation_pair_rows_materialized": False,
        "validation_embeddings_loaded": False,
        "validation_metrics_computed": False,
        "excluded_validation_pair_rows_materialized": False,
        "excluded_validation_embeddings_loaded": False,
        "excluded_validation_metrics_computed": False,
        "frozen_test_used": False,
    }


def _state_dicts_are_identical(left: Any, right: Any) -> bool:
    """Return whether two exact tensor-only model state mappings are identical."""
    import torch

    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    if set(left) != set(right):
        return False
    return all(
        torch.is_tensor(left[name])
        and torch.is_tensor(right[name])
        and left[name].dtype == right[name].dtype
        and tuple(left[name].shape) == tuple(right[name].shape)
        and bool(torch.equal(left[name], right[name]))
        for name in left
    )


def _recover_final_refit_completion_if_present(
    args: argparse.Namespace,
    submission_provenance: Mapping[str, Any],
    output_dir: Path,
    device: Any,
) -> bool:
    """Authenticate a published final checkpoint and recover only its summary."""
    final_path = output_dir / "final.pt"
    summary_path = output_dir / "final_summary.json"
    state_path = output_dir / "refit_state.pt"
    if summary_path.exists() and not final_path.is_file():
        raise ValueError("Final-refit summary exists without its final checkpoint")
    if not final_path.exists():
        return False
    if not final_path.is_file():
        raise ValueError("Final-refit final.pt is not a regular file")
    checkpoint, checkpoint_sha256 = _read_torch_checkpoint_once(final_path, device)
    config = _validate_final_refit_checkpoint(checkpoint)
    if (
        checkpoint["submission_provenance"] != submission_provenance
        or config.get("submission_provenance") != submission_provenance
        or config.get("source_sha256") != _source_hashes()
    ):
        raise ValueError("Published final checkpoint differs from the current submission")
    model = _model_from_config(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if not state_path.is_file():
        raise ValueError("Published final checkpoint lacks its authenticated refit state")
    state, _ = _read_torch_checkpoint_once(state_path, device)
    optimization = config.get("optimization")
    if not isinstance(optimization, Mapping):
        raise ValueError("Published final checkpoint optimization is malformed")
    optimizer = _optimizer(
        model,
        float(optimization["learning_rate"]),
        float(optimization["weight_decay"]),
    )
    scheduler = _scheduler(
        optimizer, int(optimization["train_batch_count"]), args
    )
    current_execution_provenance = _provenance(config, args, device)
    _validate_final_refit_resume_state(
        state,
        config,
        submission_provenance,
        int(checkpoint["epoch"]),
        current_execution_provenance,
        model,
        optimizer,
        scheduler,
    )
    if (
        state["epoch"] != checkpoint["epoch"]
        or state["global_step"] != checkpoint["global_step"]
        or state["history"] != checkpoint["history"]
        or state["execution_provenance_history"]
        != checkpoint["execution_provenance_history"]
        or not _state_dicts_are_identical(
            state["model_state_dict"], checkpoint["model_state_dict"]
        )
    ):
        raise ValueError("Published final checkpoint and refit state are inconsistent")
    summary = _final_refit_summary_payload(
        config,
        submission_provenance,
        final_path,
        checkpoint_sha256,
        int(checkpoint["epoch"]),
    )
    if summary_path.exists():
        if not summary_path.is_file() or _strict_json_object(
            summary_path, "final-refit summary"
        ) != summary:
            raise ValueError("Existing final-refit summary failed authentication")
    else:
        _create_only_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return True


@contextmanager
def _exclusive_final_refit_process_lock(output_dir: Path) -> Iterable[Path]:
    """Hold a nonblocking process lock for one canonical final-refit output."""
    import fcntl

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".final-refit.process.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another final-refit process already holds the canonical lock: {lock_path}"
            ) from error
        payload = (
            json.dumps(
                {
                    "format_version": "sabdab2_final_refit_process_lock_v1",
                    "pid": os.getpid(),
                    "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
                    "output_dir": str(output_dir),
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.write(descriptor, payload)
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def command_final_refit(args: argparse.Namespace) -> None:
    """Validate and exclusively execute one canonical fixed-epoch refit."""
    output_dir = args.output_dir.expanduser().resolve()
    with _exclusive_final_refit_process_lock(output_dir):
        submission_provenance = validate_final_refit_submission_manifest(args)
        _command_final_refit_locked(args, submission_provenance)


def _command_final_refit_locked(
    args: argparse.Namespace, submission_provenance: Mapping[str, Any]
) -> None:
    """Refit while the caller holds the canonical process-lifetime lock."""

    import torch

    from interaction_first.runtime import (
        atomic_torch_save,
        safe_torch_load,
        set_reproducible_seed,
    )

    args.max_epochs = args.fixed_epochs
    args.validation_batch_size = 1
    args.patience = 1
    args.min_validation_pairs = 2
    args.min_delta = 0.0
    _validate_numeric_arguments(args)
    if args.fixed_epochs <= 0:
        raise ValueError("fixed_epochs must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    set_reproducible_seed(args.seed, deterministic=not args.allow_nondeterministic)
    output_dir = args.output_dir.expanduser().resolve()
    if _recover_final_refit_completion_if_present(
        args, submission_provenance, output_dir, device
    ):
        return

    splits, dataset_metadata = _final_refit_dataset_from_args(args)
    if set(splits) != {"train"} or dataset_metadata["loaded_splits"] != ["train"]:
        raise PermissionError("Final refit must materialize training rows exclusively")
    if dataset_metadata["unmaterialized_splits"] != ["validation"]:
        raise PermissionError("Final refit did not keep validation rows unmaterialized")
    if (
        dataset_metadata.get("permanently_excluded_splits") != ["excluded_validation"]
        or dataset_metadata.get("excluded_validation_rows_materialized") is not False
    ):
        raise PermissionError("Final refit did not preserve excluded_validation isolation")
    if dataset_metadata.get("declared_identifier_membership_sha256", {}).get(
        "train"
    ) != submission_provenance["common_train_membership_sha256"]:
        raise ValueError("Canonical refit boundary changed its common train membership")
    train_records, train_duplicates = deduplicate_exact_pairs(splits["train"])
    if len(train_records) < args.min_train_pairs:
        raise ValueError(
            f"Training has {len(train_records)} pairs; minimum={args.min_train_pairs}"
        )
    train_batches = length_bucket_batches(
        train_records, args.batch_size, args.bucket_width, args.seed, epoch=1
    )
    excluded_pairs, negative_policy = load_excluded_negative_pairs(
        args.excluded_negative_pairs, train_records
    )
    corpora = _embedding_corpora(args, train_records)
    observed_embedding_manifests = {
        side: value["content_manifest_sha256"] for side, value in corpora.items()
    }
    expected_embedding_manifests = submission_provenance[
        "immutable_data_inputs"
    ]["training_embedding_content_manifest_sha256"]
    if observed_embedding_manifests != expected_embedding_manifests:
        raise ValueError(
            "Final-refit training embeddings differ from the independently audited corpus"
        )
    config = _final_refit_config(
        args,
        submission_provenance,
        dataset_metadata,
        corpora,
        train_records,
        train_duplicates,
        negative_policy,
        len(train_batches),
    )
    current_execution_provenance = _provenance(config, args, device)
    model = _model_from_config(config, device)
    optimizer = _optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = _scheduler(optimizer, len(train_batches), args)
    antibody_store, antigen_store = _stores(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "refit_state.pt"
    final_path = output_dir / "final.pt"
    history_path = output_dir / "refit_history.json"
    summary_path = output_dir / "final_summary.json"
    existing = [
        path
        for path in (state_path, final_path, history_path, summary_path)
        if path.exists()
    ]
    if not args.resume and existing:
        raise FileExistsError(
            f"Refusing to overwrite an existing final-refit run: {[str(path) for path in existing]}"
        )
    if args.resume and not state_path.is_file() and existing:
        raise ValueError("Final-refit artifacts exist but refit_state.pt is missing")

    start_epoch = 1
    global_step = 0
    history: list[dict[str, Any]] = []
    execution_provenance_history: list[dict[str, Any]] = [
        current_execution_provenance
    ]
    if args.resume and state_path.is_file():
        state = safe_torch_load(state_path, map_location=device)
        execution_provenance_history = _validate_final_refit_resume_state(
            state,
            config,
            submission_provenance,
            args.fixed_epochs,
            current_execution_provenance,
            model,
            optimizer,
            scheduler,
        )
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        history = list(state["history"])
        torch.set_rng_state(state["torch_rng_state"].to(device="cpu"))
        if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(
                [value.to(device="cpu") for value in state["cuda_rng_state_all"]]
            )
    known_positive_pairs = _pair_set(train_records)
    for epoch in range(start_epoch, args.fixed_epochs + 1):
        train_metrics, global_step = _train_epoch(
            model,
            train_records,
            known_positive_pairs,
            excluded_pairs,
            antibody_store,
            antigen_store,
            optimizer,
            scheduler,
            device,
            args,
            epoch,
            global_step,
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "temperature": float(model.temperature_head().detach().cpu().item()),
            "validation_metrics_computed": False,
        }
        history.append(record)
        state = {
            "format_version": FORMAT_VERSION,
            "stage": "final_refit_in_progress",
            "config": config,
            "submission_provenance": submission_provenance,
            "execution_provenance_history": execution_provenance_history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        }
        atomic_torch_save(state, state_path)
        atomic_write_json(
            history_path,
            {
                "format_version": FORMAT_VERSION,
                "stage": "final_refit_fixed_epochs",
                "config": config,
                "history": history,
                "validation_metrics_computed": False,
            },
        )
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)

    if not history or int(history[-1]["epoch"]) != args.fixed_epochs:
        raise RuntimeError("Final refit did not reach its locked epoch count")
    final_checkpoint = {
        "format_version": FORMAT_VERSION,
        "stage": "final_refit_complete",
        "config": config,
        "submission_provenance": submission_provenance,
        "execution_provenance_history": execution_provenance_history,
        "model_state_dict": model.state_dict(),
        "epoch": args.fixed_epochs,
        "global_step": global_step,
        "history": history,
        "validation_metrics_computed": False,
        "frozen_test_used": False,
    }
    _create_only_torch_save(final_checkpoint, final_path)
    summary = _final_refit_summary_payload(
        config,
        submission_provenance,
        final_path,
        sha256_file(final_path),
        int(history[-1]["epoch"]),
    )
    _create_only_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


def _create_one_shot_ledger(path: Path, payload: Mapping[str, Any]) -> Path:
    """Create an exclusive evaluation ledger before any holdout row is loaded."""
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
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return target


def _canonical_one_shot_ledger_contract(
    registry_dir: Path,
    checkpoint_sha256: str,
    split_manifest_sha256: str,
    authority_split_sha256: str,
) -> dict[str, str]:
    """Derive the sole ledger path for one checkpoint and development boundary."""
    digests = {
        "checkpoint_sha256": checkpoint_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "authority_split_sha256": authority_split_sha256,
    }
    invalid = [name for name, value in digests.items() if not _is_sha256(value)]
    if invalid:
        raise ValueError(f"Canonical one-shot ledger received invalid hashes: {invalid}")
    raw_registry = registry_dir.expanduser()
    if not raw_registry.is_absolute():
        raise ValueError("one_shot_ledger_registry_dir must be an absolute central path")
    registry = raw_registry.resolve()
    identity = {**digests, "split": "validation"}
    key = canonical_sha256(identity)
    return {
        "key": key,
        "registry_dir": str(registry),
        "path": str(registry / f"validation-{key}.json"),
        **identity,
    }


def _create_or_validate_immutable_json(
    path: Path, payload: Mapping[str, Any], label: str
) -> dict[str, str]:
    """Create one canonical JSON object exclusively or validate its exact payload."""
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = dict(payload)
    encoded = (
        json.dumps(
            normalized,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = _strict_json_object(target, label)
        if existing != normalized:
            raise FileExistsError(
                f"{label} already exists with a different immutable payload: {target}"
            )
    else:
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
    return {"path": str(target), "sha256": sha256_file(target)}


def _physical_exact_reference(value: Any, label: str) -> dict[str, str]:
    """Authenticate an exact ``{path, sha256}`` physical-file reference."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    _require_exact_keys(value, {"path", "sha256"}, label)
    path = _absolute_path(value["path"], f"{label}.path")
    digest = value["sha256"]
    if not isinstance(digest, str) or not _is_sha256(digest):
        raise ValueError(f"{label}.sha256 must be an exact SHA-256")
    if not path.is_file():
        raise FileNotFoundError(f"{label} physical file is missing: {path}")
    observed = sha256_file(path)
    if observed != digest:
        raise ValueError(
            f"{label} physical SHA-256 mismatch: expected={digest}, observed={observed}"
        )
    return {"path": str(path), "sha256": digest}


def _locked_submission_evidence(
    submission_provenance: Mapping[str, Any],
    selected_boundary: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reauthenticate the immutable evidence inherited by every suite artifact."""
    upstream = submission_provenance.get("upstream_evidence")
    if not isinstance(upstream, Mapping):
        raise ValueError("Checkpoint submission provenance lacks upstream evidence")
    references = {
        "submission_manifest": submission_provenance.get("submission_manifest"),
        "protocol_v1_5_execution_lock": upstream.get(
            "protocol_v1_5_execution_lock"
        ),
        "protocol_v1_6_execution_hardening": upstream.get(
            "protocol_v1_6_execution_hardening"
        ),
        "boundary_verification_v2": upstream.get("boundary_verification_v2"),
    }
    normalized = {
        name: _physical_exact_reference(
            {"path": value.get("path"), "sha256": value.get("sha256")}
            if isinstance(value, Mapping)
            else value,
            f"locked_submission_evidence.{name}",
        )
        for name, value in references.items()
    }
    fixed_evaluation = submission_provenance.get("fixed_evaluation")
    if not isinstance(fixed_evaluation, Mapping):
        raise ValueError("Checkpoint submission provenance lacks fixed_evaluation")
    _require_exact_keys(
        fixed_evaluation,
        set(_expected_final_evaluation_contract()),
        "locked_submission_evidence.fixed_evaluation",
    )
    if selected_boundary is None:
        selected_boundary = submission_provenance.get("selected_boundary")
    if not isinstance(selected_boundary, str) or not selected_boundary:
        raise ValueError("Checkpoint submission provenance lacks selected_boundary")
    declared_boundaries = submission_provenance.get("all_declared_boundaries")
    if not isinstance(declared_boundaries, Mapping) or selected_boundary not in declared_boundaries:
        raise ValueError("selected_boundary is not declared by the submission manifest")
    evidence = {
        **normalized,
        "fixed_evaluation": dict(fixed_evaluation),
        "selected_boundary": selected_boundary,
    }
    score_context = {
        "submission_manifest_sha256": normalized["submission_manifest"]["sha256"],
        "protocol_v1_5_execution_lock_sha256": normalized[
            "protocol_v1_5_execution_lock"
        ]["sha256"],
        "protocol_v1_6_execution_hardening_sha256": normalized[
            "protocol_v1_6_execution_hardening"
        ]["sha256"],
        "boundary_verification_v2_sha256": normalized[
            "boundary_verification_v2"
        ]["sha256"],
        "fixed_evaluation_sha256": canonical_sha256(dict(fixed_evaluation)),
        "selected_boundary": selected_boundary,
    }
    return evidence, score_context


def _ensure_checkpoint_suite_claim(
    config: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    registry_dir: Path,
    selected_boundary: str | None = None,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    """Create or validate the immutable two-boundary checkpoint-suite claim."""
    provenance = config.get("submission_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Final checkpoint lacks submission provenance")
    registry = registry_dir.expanduser()
    if not registry.is_absolute():
        raise ValueError("one_shot_ledger_registry_dir must be absolute")
    registry = registry.resolve()
    if str(registry) != provenance.get("one_shot_ledger_registry_dir"):
        raise ValueError("Checkpoint-suite registry differs from submission provenance")
    expected_checkpoint_path = (
        Path(str(provenance.get("refit_output_dir", ""))).expanduser().resolve()
        / "final.pt"
    )
    if checkpoint_path.expanduser().resolve() != expected_checkpoint_path:
        raise ValueError(
            "Final-refit evaluation checkpoint must be exactly "
            "<manifest refit_output_dir>/final.pt"
        )
    if not _is_sha256(checkpoint_sha256):
        raise ValueError("Checkpoint-suite claim requires an exact checkpoint SHA-256")
    locked_evidence, score_context = _locked_submission_evidence(
        provenance, selected_boundary
    )
    manifest_reference = locked_evidence["submission_manifest"]
    boundaries = provenance.get("all_declared_boundaries")
    if not isinstance(boundaries, Mapping) or len(boundaries) != 2:
        raise ValueError("Checkpoint-suite claim requires exactly two boundaries")
    normalized_boundaries: dict[str, Any] = {}
    for name, raw in boundaries.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise ValueError("Checkpoint-suite boundary provenance is malformed")
        split_reference = _physical_exact_reference(
            raw.get("split_manifest"), f"checkpoint_suite_claim.boundaries.{name}"
        )
        mode = raw.get("strict_nw_constraint_mode")
        if mode not in STRICT_NW_CONSTRAINT_MODES:
            raise ValueError("Checkpoint-suite boundary has an unsupported NW mode")
        normalized_boundaries[name] = {
            "split_manifest_sha256": split_reference["sha256"],
            "strict_nw_constraint_mode": mode,
        }
    claim = {
        "format_version": CHECKPOINT_SUITE_CLAIM_FORMAT_VERSION,
        "stage": "final_checkpoint_suite_claim",
        "intended_use": "two_boundary_one_shot_development_confirmation",
        "status": "immutable",
        "submission_manifest": manifest_reference,
        "checkpoint": {
            "path": str(expected_checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "one_shot_ledger_registry_dir": str(registry),
        "boundaries": normalized_boundaries,
    }
    claim_path = (
        registry
        / "checkpoint-suite-claims"
        / f"{manifest_reference['sha256']}.json"
    )
    claim_reference = _create_or_validate_immutable_json(
        claim_path, claim, "checkpoint-suite claim"
    )
    score_context["checkpoint_suite_claim_sha256"] = claim_reference["sha256"]
    return claim_reference, locked_evidence, score_context


def _finalize_checkpoint_suite_if_complete(
    config: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    registry_dir: Path,
    claim_reference: Mapping[str, Any],
    require_complete: bool = False,
) -> dict[str, str] | None:
    """Seal a suite receipt only after both predeclared ledgers are complete."""
    provenance = config.get("submission_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Final checkpoint lacks submission provenance")
    registry = registry_dir.expanduser().resolve()
    normalized_claim = _physical_exact_reference(
        claim_reference, "checkpoint_suite_claim"
    )
    boundaries = provenance.get("all_declared_boundaries")
    authority = provenance.get("authority")
    if (
        not isinstance(boundaries, Mapping)
        or len(boundaries) != 2
        or not isinstance(authority, Mapping)
        or not isinstance(authority.get("split_manifest"), Mapping)
    ):
        raise ValueError("Checkpoint suite provenance is incomplete")
    authority_sha256 = str(authority["split_manifest"].get("sha256", ""))
    completed_boundaries: dict[str, Any] = {}
    for name, boundary in boundaries.items():
        if not isinstance(boundary, Mapping):
            raise ValueError("Checkpoint suite boundary is malformed")
        split_reference = boundary.get("split_manifest")
        if not isinstance(split_reference, Mapping):
            raise ValueError("Checkpoint suite split reference is malformed")
        split_sha256 = str(split_reference.get("sha256", ""))
        ledger_contract = _canonical_one_shot_ledger_contract(
            registry, checkpoint_sha256, split_sha256, authority_sha256
        )
        ledger_path = Path(ledger_contract["path"])
        if not ledger_path.is_file():
            if require_complete:
                raise RuntimeError(
                    f"Checkpoint suite is incomplete; missing boundary ledger: {name}"
                )
            return None
        ledger = _strict_json_object(ledger_path, f"completed ledger {name}")
        if ledger.get("status") != "completed":
            if require_complete:
                raise RuntimeError(
                    f"Checkpoint suite is incomplete; boundary ledger is not complete: {name}"
                )
            return None
        expected_fields = {
            "format_version",
            "status",
            "canonical_key",
            "canonical_path",
            "registry_dir",
            "checkpoint_suite_claim",
            "locked_submission_evidence",
            "boundary_name",
            "strict_nw_constraint_mode",
            "checkpoint_sha256",
            "config_sha256",
            "split_manifest_sha256",
            "authority_split_sha256",
            "evaluation_json",
            "evaluation_json_sha256",
            "score_artifact",
            "score_artifact_sha256",
            "split",
            "frozen_test_used",
        }
        _require_exact_keys(ledger, expected_fields, f"completed ledger {name}")
        if (
            ledger["format_version"] != ONE_SHOT_LEDGER_FORMAT_VERSION
            or ledger["canonical_key"] != ledger_contract["key"]
            or ledger["canonical_path"] != ledger_contract["path"]
            or ledger["registry_dir"] != ledger_contract["registry_dir"]
            or ledger["checkpoint_suite_claim"] != normalized_claim
            or ledger["boundary_name"] != name
            or ledger["strict_nw_constraint_mode"]
            != boundary.get("strict_nw_constraint_mode")
            or ledger["checkpoint_sha256"] != checkpoint_sha256
            or ledger["config_sha256"] != config["config_sha256"]
            or ledger["split_manifest_sha256"] != split_sha256
            or ledger["authority_split_sha256"] != authority_sha256
            or ledger["split"] != "validation"
            or ledger["frozen_test_used"] is not False
        ):
            raise ValueError(f"Completed ledger {name} is inconsistent")
        evaluation_reference = _physical_exact_reference(
            {
                "path": ledger["evaluation_json"],
                "sha256": ledger["evaluation_json_sha256"],
            },
            f"completed ledger {name}.evaluation_json",
        )
        score_reference = _physical_exact_reference(
            {
                "path": ledger["score_artifact"],
                "sha256": ledger["score_artifact_sha256"],
            },
            f"completed ledger {name}.score_artifact",
        )
        expected_output = Path(str(boundary.get("evaluation_output", ""))).resolve()
        expected_score = Path(str(boundary.get("score_artifact", ""))).resolve()
        if (
            Path(evaluation_reference["path"]) != expected_output
            or Path(score_reference["path"]) != expected_score
        ):
            raise ValueError(f"Completed ledger {name} points to unpredeclared artifacts")
        evaluation = _strict_json_object(
            Path(evaluation_reference["path"]), f"evaluation JSON {name}"
        )
        if (
            evaluation.get("checkpoint_sha256") != checkpoint_sha256
            or evaluation.get("checkpoint_suite_claim") != normalized_claim
            or evaluation.get("predeclared_evaluation_boundary") != name
        ):
            raise ValueError(f"Evaluation JSON {name} is not suite-bound")
        completed_boundaries[name] = {
            "split_manifest_sha256": split_sha256,
            "strict_nw_constraint_mode": boundary["strict_nw_constraint_mode"],
            "one_shot_ledger": {
                "path": str(ledger_path),
                "sha256": sha256_file(ledger_path),
            },
            "evaluation_json": evaluation_reference,
            "score_artifact": score_reference,
        }
    submission_reference = _physical_exact_reference(
        provenance.get("submission_manifest"), "submission_manifest"
    )
    receipt = {
        "format_version": CHECKPOINT_SUITE_COMPLETION_FORMAT_VERSION,
        "stage": "final_checkpoint_suite_completion",
        "intended_use": "two_boundary_one_shot_development_confirmation",
        "status": "completed",
        "checkpoint_suite_claim": normalized_claim,
        "submission_manifest": submission_reference,
        "checkpoint": {
            "path": str(checkpoint_path.expanduser().resolve()),
            "sha256": checkpoint_sha256,
        },
        "one_shot_ledger_registry_dir": str(registry),
        "boundaries": completed_boundaries,
    }
    receipt_path = (
        registry
        / "checkpoint-suite-completions"
        / f"{submission_reference['sha256']}.json"
    )
    return _create_or_validate_immutable_json(
        receipt_path, receipt, "checkpoint-suite completion"
    )


def _final_refit_evaluation_contract(
    config: Mapping[str, Any], args: argparse.Namespace
) -> tuple[str, dict[str, Any]]:
    """Bind one evaluation command to a predeclared checkpoint boundary."""
    provenance = config.get("submission_provenance")
    if not isinstance(provenance, Mapping) or provenance.get(
        "format_version"
    ) != FINAL_REFIT_SUBMISSION_FORMAT_VERSION:
        raise ValueError("Final-refit checkpoint lacks submission provenance")
    boundaries = provenance.get("all_declared_boundaries")
    if not isinstance(boundaries, Mapping) or len(boundaries) != 2:
        raise ValueError("Final-refit checkpoint lacks both repaired boundaries")
    if args.score_artifact is None or args.one_shot_ledger_registry_dir is None:
        raise ValueError(
            "Final-refit evaluation requires a score artifact and central ledger registry"
        )
    fixed_evaluation = provenance.get("fixed_evaluation")
    expected_evaluation_keys = set(_expected_final_evaluation_contract())
    if not isinstance(fixed_evaluation, Mapping):
        raise ValueError("Final-refit checkpoint lacks a fixed evaluation protocol")
    _require_exact_keys(
        fixed_evaluation, expected_evaluation_keys, "fixed_evaluation"
    )
    observed_evaluation = {
        "split": "validation",
        "batch_size": args.batch_size,
        "bucket_width": args.bucket_width,
        "recall_ks": list(args.recall_ks),
        "seed": args.seed,
        "deterministic": not args.allow_nondeterministic,
        "device": args.device,
        "isolation_policy": args.isolation_policy,
        "score_artifact_required": args.score_artifact is not None,
    }
    if (
        fixed_evaluation.get("split") != "validation"
        or fixed_evaluation.get("score_artifact_required") is not True
        or observed_evaluation != dict(fixed_evaluation)
    ):
        raise ValueError(
            "Evaluation runtime arguments differ from the checkpoint-bound one-shot protocol"
        )
    registry = args.one_shot_ledger_registry_dir.expanduser().resolve()
    if str(registry) != provenance.get("one_shot_ledger_registry_dir"):
        raise ValueError("One-shot ledger registry differs from checkpoint provenance")
    requested = (
        str(args.split_manifest.expanduser().resolve()),
        args.split_manifest_sha256,
        args.strict_nw_constraint_mode,
        str(args.output.expanduser().resolve()),
        str(args.score_artifact.expanduser().resolve()),
    )
    matches: list[tuple[str, dict[str, Any]]] = []
    for name, raw_boundary in boundaries.items():
        if not isinstance(name, str) or not isinstance(raw_boundary, Mapping):
            raise ValueError("Checkpoint boundary provenance is malformed")
        split_reference = raw_boundary.get("split_manifest")
        if not isinstance(split_reference, Mapping):
            raise ValueError("Checkpoint boundary split provenance is malformed")
        candidate = (
            split_reference.get("path"),
            split_reference.get("sha256"),
            raw_boundary.get("strict_nw_constraint_mode"),
            raw_boundary.get("evaluation_output"),
            raw_boundary.get("score_artifact"),
        )
        if candidate == requested:
            matches.append((name, dict(raw_boundary)))
    if len(matches) != 1:
        raise ValueError(
            "Evaluation split, mode, output, and score artifact do not match one "
            "predeclared checkpoint boundary"
        )
    authority = provenance.get("authority")
    if not isinstance(authority, Mapping) or not isinstance(
        authority.get("split_manifest"), Mapping
    ):
        raise ValueError("Checkpoint authority provenance is malformed")
    authority_reference = authority["split_manifest"]
    if (
        str(args.authority_split_manifest.expanduser().resolve())
        != authority_reference.get("path")
        or args.authority_split_sha256 != authority_reference.get("sha256")
        or list(args.authorized_source_pool_splits)
        != authority.get("authorized_source_pool_splits")
        or list(args.authority_sealed_splits) != authority.get("sealed_source_splits")
        or args.authority_coverage_mode != authority.get("coverage_mode")
    ):
        raise ValueError("Evaluation authority differs from checkpoint provenance")
    if provenance.get("common_train_membership_sha256") != matches[0][1].get(
        "train_membership_sha256"
    ):
        raise ValueError("Evaluation boundary does not share the checkpoint train membership")
    immutable_inputs = provenance.get("immutable_data_inputs")
    if not isinstance(immutable_inputs, Mapping):
        raise ValueError("Checkpoint provenance lacks immutable data inputs")
    prepared_reference = immutable_inputs.get("prepared_instances")
    canonical_reference = immutable_inputs.get("canonical_sequence_manifest")
    if (
        not isinstance(prepared_reference, Mapping)
        or not isinstance(canonical_reference, Mapping)
        or args.prepare_dir.expanduser().resolve() / "instances.parquet"
        != Path(str(prepared_reference.get("path", ""))).expanduser().resolve()
        or args.prepared_instances_sha256 != prepared_reference.get("sha256")
        or args.canonical_sequence_manifest.expanduser().resolve()
        != Path(str(canonical_reference.get("path", ""))).expanduser().resolve()
        or args.canonical_sequence_manifest_sha256
        != canonical_reference.get("sha256")
    ):
        raise ValueError("Evaluation immutable data inputs differ from checkpoint provenance")
    physical_references = {
        "evaluation split": (
            args.split_manifest.expanduser().resolve(),
            args.split_manifest_sha256,
        ),
        "evaluation authority": (
            args.authority_split_manifest.expanduser().resolve(),
            args.authority_split_sha256,
        ),
    }
    for label, (path, expected_sha256) in physical_references.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} physical file is missing: {path}")
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"{label} physical SHA-256 mismatch: "
                f"expected={expected_sha256}, observed={observed_sha256}"
            )
    return matches[0]


def command_evaluate(args: argparse.Namespace) -> None:
    """Replay full-library development validation with strict provenance checks."""
    import torch

    from interaction_first.runtime import set_reproducible_seed

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    set_reproducible_seed(args.seed, deterministic=not args.allow_nondeterministic)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint, checkpoint_sha256 = _load_torch_checkpoint_once(
        checkpoint_path, args.checkpoint_sha256, device
    )
    if not isinstance(checkpoint, Mapping) or checkpoint.get(
        "format_version"
    ) != FORMAT_VERSION:
        raise ValueError("Checkpoint format is incompatible")
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint config is malformed")
    run_mode = raw_config.get("run_mode", "fold_selection_with_validation")
    final_refit = run_mode == "final_refit_fixed_epochs"
    config = (
        _validate_final_refit_checkpoint(checkpoint)
        if final_refit
        else dict(raw_config)
    )
    if not final_refit:
        validate_config_integrity(config)
    model = _model_from_config(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    selected_evaluation_boundary = None
    selected_evaluation_contract = None
    if final_refit:
        (
            selected_evaluation_boundary,
            selected_evaluation_contract,
        ) = _final_refit_evaluation_contract(config, args)
        if config.get("source_sha256") != _source_hashes():
            raise ValueError("Evaluation source code differs from checkpoint provenance")
    if not final_refit and args.one_shot_ledger_registry_dir is not None:
        raise ValueError(
            "one_shot_ledger_registry_dir is reserved for final-refit evaluation"
        )
    ledger_path = None
    ledger_contract = None
    claim_reference = None
    locked_submission_evidence = None
    locked_score_context = None
    output_path = args.output.expanduser().resolve()
    if final_refit:
        if args.one_shot_ledger_registry_dir is None or args.score_artifact is None:
            raise ValueError(
                "Final-refit evaluation requires a central one-shot ledger registry "
                "and score artifact"
            )
        score_path = args.score_artifact.expanduser().resolve()
        preexisting = [path for path in (output_path, score_path) if path.exists()]
        if preexisting:
            raise FileExistsError(
                f"One-shot evaluation outputs already exist: {[str(path) for path in preexisting]}"
            )
        claim_reference, locked_submission_evidence, locked_score_context = (
            _ensure_checkpoint_suite_claim(
                config,
                checkpoint_path,
                checkpoint_sha256,
                args.one_shot_ledger_registry_dir,
                selected_evaluation_boundary,
            )
        )
        ledger_contract = _canonical_one_shot_ledger_contract(
            args.one_shot_ledger_registry_dir,
            checkpoint_sha256,
            args.split_manifest_sha256,
            args.authority_split_sha256,
        )
        ledger_path = _create_one_shot_ledger(
            Path(ledger_contract["path"]),
            {
                "format_version": ONE_SHOT_LEDGER_FORMAT_VERSION,
                "status": "started_before_holdout_materialization",
                "canonical_key": ledger_contract["key"],
                "canonical_path": ledger_contract["path"],
                "registry_dir": ledger_contract["registry_dir"],
                "checkpoint_suite_claim": claim_reference,
                "locked_submission_evidence": locked_submission_evidence,
                "boundary_name": selected_evaluation_boundary,
                "strict_nw_constraint_mode": args.strict_nw_constraint_mode,
                "checkpoint_sha256": checkpoint_sha256,
                "config_sha256": config["config_sha256"],
                "split_manifest_sha256": args.split_manifest_sha256,
                "authority_split_sha256": args.authority_split_sha256,
                "split": "validation",
                "frozen_test_used": False,
            },
        )
    splits, dataset_metadata = _dataset_from_args(
        args, require_authority_validation_boundary_v2=final_refit
    )
    train_records, _ = deduplicate_exact_pairs(splits["train"])
    validation_records, removed = deduplicate_exact_pairs(splits["validation"])
    configured_isolation_policy = (
        config["training_data"].get("isolation_policy_to_audit_at_evaluation")
        if final_refit
        else config["training_data"].get("isolation_policy")
    )
    if configured_isolation_policy != args.isolation_policy:
        raise ValueError("Isolation policy differs from checkpoint configuration")
    isolation = audit_isolation_policy(
        train_records, validation_records, args.isolation_policy
    )
    if not final_refit and config.get("source_sha256") != _source_hashes():
        raise ValueError("Evaluation source code differs from checkpoint provenance")
    if config["dataset"]["instances_sha256"] != dataset_metadata["instances_sha256"]:
        raise ValueError("Prepared instances changed after checkpoint training")
    if not final_refit and config["dataset"]["split_manifest_sha256"] != dataset_metadata["split_manifest_sha256"]:
        raise ValueError("Split manifest changed after checkpoint training")
    if final_refit:
        common_train_sha256 = config["submission_provenance"].get(
            "common_train_membership_sha256"
        )
        if (
            dataset_metadata.get("declared_identifier_membership_sha256", {}).get(
                "train"
            )
            != common_train_sha256
            or selected_evaluation_contract is None
            or selected_evaluation_contract.get("train_membership_sha256")
            != common_train_sha256
        ):
            raise ValueError("Evaluation boundary changed the checkpoint train membership")
    fingerprint_records = train_records if final_refit else [*train_records, *validation_records]
    observed_corpora = _embedding_corpora(args, fingerprint_records)
    if config.get("embedding_corpora") != observed_corpora:
        raise ValueError("Frozen embedding corpora changed after checkpoint training")
    validation_corpora = (
        _embedding_corpora(args, validation_records) if final_refit else None
    )
    if str(config["model"]["antibody_encoder"]) != args.antibody_model_name:
        raise ValueError("Antibody encoder name differs from checkpoint configuration")
    if str(config["model"]["antibody_artifact_format"]) != args.antibody_artifact_format:
        raise ValueError("Antibody artifact format differs from checkpoint configuration")
    if (
        str(config["model"]["antibody_model_weights_sha256"])
        != args.antibody_model_weights_sha256
    ):
        raise ValueError("Antibody model weight hash differs from checkpoint configuration")
    if str(config["model"]["antigen_encoder"]) != args.antigen_model_name:
        raise ValueError("Antigen encoder name differs from checkpoint configuration")
    if (
        str(config["model"]["antigen_model_weights_sha256"])
        != args.antigen_model_weights_sha256
    ):
        raise ValueError("Antigen model weight hash differs from checkpoint configuration")
    if int(config["model"]["antibody_input_dim"]) != args.antibody_input_dim:
        raise ValueError("Antibody input dimension differs from checkpoint configuration")
    if int(config["model"]["antigen_input_dim"]) != args.antigen_input_dim:
        raise ValueError("Antigen input dimension differs from checkpoint configuration")

    args.antibody_input_dim = int(config["model"]["antibody_input_dim"])
    args.antigen_input_dim = int(config["model"]["antigen_input_dim"])
    antibody_store, antigen_store = _stores(args)
    score_payload = None
    if args.score_artifact is None:
        metrics = evaluate_records(
            model,
            validation_records,
            antibody_store,
            antigen_store,
            args.batch_size,
            args.bucket_width,
            device,
            args.recall_ks,
        )
    else:
        metrics, score_payload = evaluate_records_with_scores(
            model,
            validation_records,
            antibody_store,
            antigen_store,
            args.batch_size,
            args.bucket_width,
            device,
            args.recall_ks,
        )
    output = {
        "format_version": FORMAT_VERSION,
        "stage": "full_library_development_validation",
        "split": "validation",
        "frozen_test_used": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config["config_sha256"],
        "dataset": dataset_metadata,
        "embedding_corpora": (
            {
                "training": observed_corpora,
                "one_shot_validation": validation_corpora,
            }
            if final_refit
            else observed_corpora
        ),
        "exact_duplicate_pairs_removed": removed,
        "train_validation_isolation": isolation,
        "metrics": metrics,
        "embedding_caches": {
            "antibody": antibody_store.summary(),
            "antigen": antigen_store.summary(),
        },
        "checkpoint_run_mode": run_mode,
        "one_shot_evaluation": final_refit,
    }
    if final_refit:
        output["predeclared_evaluation_boundary"] = selected_evaluation_boundary
        output["checkpoint_suite_claim"] = claim_reference
        output["locked_submission_evidence"] = locked_submission_evidence
    if ledger_contract is not None:
        output["one_shot_ledger"] = {
            "format_version": ONE_SHOT_LEDGER_FORMAT_VERSION,
            "canonical_key": ledger_contract["key"],
            "canonical_path": ledger_contract["path"],
            "registry_dir": ledger_contract["registry_dir"],
        }
    if args.score_artifact is not None:
        score_path = args.score_artifact.expanduser().resolve()
        if score_path == output_path:
            raise ValueError("Evaluation JSON and score artifact paths must differ")
        if score_payload is None:
            raise AssertionError("Score export was requested without a score payload")
        authority = dataset_metadata["authority"]
        context = {
            "evaluation_format_version": FORMAT_VERSION,
            "stage": output["stage"],
            "split": output["split"],
            "frozen_test_used": output["frozen_test_used"],
            "checkpoint_sha256": output["checkpoint_sha256"],
            "config_sha256": output["config_sha256"],
            "instances_sha256": dataset_metadata["instances_sha256"],
            "split_manifest_sha256": dataset_metadata["split_manifest_sha256"],
            "split_format_version": dataset_metadata["split_format_version"],
            "split_intended_use": dataset_metadata["split_intended_use"],
            "strict_nw_constraint_mode": dataset_metadata[
                "strict_nw_constraint_mode"
            ],
            "authority_split_sha256": authority["sha256"],
            "authorized_source_pool_splits": authority[
                "authorized_source_pool_splits"
            ],
            "authority_sealed_splits": authority["sealed_source_splits"],
            "authority_selected_sealed_overlap_count": authority[
                "selected_sealed_overlap_count"
            ],
            "evaluation_embedding_corpora_sha256": canonical_sha256(
                output["embedding_corpora"]
            ),
        }
        if ledger_contract is not None:
            context["one_shot_ledger_key"] = ledger_contract["key"]
            context["one_shot_ledger_path"] = ledger_contract["path"]
        if locked_score_context is not None:
            context.update(locked_score_context)
        output["score_artifact"] = write_score_artifact(
            score_path,
            score_payload,
            context,
        )
    if final_refit:
        _create_only_json(output_path, output)
    else:
        atomic_write_json(output_path, output)
    if ledger_path is not None and ledger_contract is not None:
        atomic_write_json(
            ledger_path,
            {
                "format_version": ONE_SHOT_LEDGER_FORMAT_VERSION,
                "status": "completed",
                "canonical_key": ledger_contract["key"],
                "canonical_path": ledger_contract["path"],
                "registry_dir": ledger_contract["registry_dir"],
                "checkpoint_suite_claim": claim_reference,
                "locked_submission_evidence": locked_submission_evidence,
                "boundary_name": selected_evaluation_boundary,
                "strict_nw_constraint_mode": args.strict_nw_constraint_mode,
                "checkpoint_sha256": output["checkpoint_sha256"],
                "config_sha256": output["config_sha256"],
                "split_manifest_sha256": dataset_metadata["split_manifest_sha256"],
                "authority_split_sha256": dataset_metadata["authority"]["sha256"],
                "evaluation_json": str(output_path),
                "evaluation_json_sha256": sha256_file(output_path),
                "score_artifact": str(args.score_artifact.expanduser().resolve()),
                "score_artifact_sha256": output["score_artifact"]["sha256"],
                "split": "validation",
                "frozen_test_used": False,
            },
        )
        suite_completion = _finalize_checkpoint_suite_if_complete(
            config,
            checkpoint_path,
            checkpoint_sha256,
            args.one_shot_ledger_registry_dir,
            claim_reference,
        )
        if suite_completion is not None:
            print(
                json.dumps(
                    {"checkpoint_suite_completion": suite_completion},
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


def command_finalize_evaluation_suite(args: argparse.Namespace) -> None:
    """Recoverably seal the two-boundary suite after both ledgers complete."""
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint, checkpoint_sha256 = _load_torch_checkpoint_once(
        checkpoint_path, args.checkpoint_sha256, "cpu"
    )
    config = _validate_final_refit_checkpoint(checkpoint)
    claim_reference, _, _ = _ensure_checkpoint_suite_claim(
        config,
        checkpoint_path,
        checkpoint_sha256,
        args.one_shot_ledger_registry_dir,
    )
    completion = _finalize_checkpoint_suite_if_complete(
        config,
        checkpoint_path,
        checkpoint_sha256,
        args.one_shot_ledger_registry_dir,
        claim_reference,
        require_complete=True,
    )
    if completion is None:
        raise AssertionError("Required suite completion was not produced")
    print(
        json.dumps(
            {"checkpoint_suite_completion": completion},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def command_smoke(args: argparse.Namespace) -> None:
    """Run a synthetic asymmetric forward, loss, and retrieval metric check."""
    import torch

    from interaction_first.asymmetric_retrieval import (
        AsymmetricPartnerRetrievalModel,
        symmetric_masked_multi_positive_info_nce,
    )

    torch.manual_seed(args.seed)
    model = AsymmetricPartnerRetrievalModel(
        antibody_input_dim=8,
        antigen_input_dim=12,
        embedding_dim=6,
        antibody_hidden_dim=10,
        antigen_hidden_dim=10,
        dropout=0.0,
        chain_balanced_antibody_pooling=True,
    )
    antibody = torch.randn(4, 5, 8)
    antigen = torch.randn(4, 7, 12)
    antibody_mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
        dtype=torch.bool,
    )
    antigen_mask = torch.tensor(
        [[1] * 7, [1] * 6 + [0], [1] * 5 + [0, 0], [1] * 7], dtype=torch.bool
    )
    chain_ids = torch.tensor(
        [[0, 0, 0, 1, 1], [0, 0, 1, 1, -1], [0, 0, 0, 1, 1], [0, 0, 1, -1, -1]],
        dtype=torch.long,
    )
    outputs = model(antibody, antigen, antibody_mask, antigen_mask, chain_ids)
    positives = torch.eye(4, dtype=torch.bool)
    positives[0, 1] = True
    denominator = torch.ones_like(positives)
    denominator[1, 2] = False
    loss = symmetric_masked_multi_positive_info_nce(
        outputs["similarity_matrix"], positives, outputs["temperature"], denominator
    )
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    metrics = full_library_metrics(
        outputs["antibody_embeddings"].detach(),
        outputs["antigen_embeddings"].detach(),
        positives,
        outputs["temperature"].detach(),
        (1, 2, 4),
        [f"ab-{index}" for index in range(4)],
        [f"ag-{index}" for index in range(4)],
    )
    summary = {
        "format_version": FORMAT_VERSION,
        "stage": "synthetic_smoke",
        "loss": float(loss.item()),
        "finite_gradients": finite_gradients,
        "metrics": metrics,
    }
    if not finite_gradients:
        raise FloatingPointError("Synthetic smoke test produced invalid gradients")
    if args.output is not None:
        atomic_write_json(args.output.expanduser().resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--prepared-instances-sha256")
    parser.add_argument("--canonical-sequence-manifest", type=Path)
    parser.add_argument("--canonical-sequence-manifest-sha256")
    parser.add_argument("--antibody-embedding-dir", type=Path, required=True)
    parser.add_argument("--antigen-embedding-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--authority-split-manifest", type=Path, required=True)
    parser.add_argument("--authority-split-sha256", required=True)
    parser.add_argument("--authorized-source-pool-splits", nargs="+", required=True)
    parser.add_argument("--authority-sealed-splits", nargs="+", required=True)
    parser.add_argument(
        "--authority-coverage-mode", choices=("exact", "subset"), default="exact"
    )
    parser.add_argument(
        "--strict-nw-constraint-mode",
        choices=tuple(STRICT_NW_CONSTRAINT_MODES),
        default=None,
        help="Required only for strict NW split manifests; must match the manifest.",
    )
    parser.add_argument(
        "--isolation-policy",
        choices=("double_exact", "antigen_exact"),
        default="double_exact",
    )
    parser.add_argument(
        "--antibody-artifact-format",
        choices=("antiberty", "esm_chunks"),
        default="antiberty",
    )
    parser.add_argument("--antibody-model-name", required=True)
    parser.add_argument("--antibody-model-weights-sha256", required=True)
    parser.add_argument("--antigen-model-name", required=True)
    parser.add_argument("--antigen-model-weights-sha256", required=True)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument("--antibody-cache-gib", type=float, default=8.0)
    parser.add_argument("--antigen-cache-gib", type=float, default=24.0)
    parser.add_argument("--bucket-width", type=int, default=128)
    parser.add_argument("--recall-ks", type=int, nargs="+", default=[1, 5, 10, 20])


def build_parser() -> argparse.ArgumentParser:
    """Build the train-only asymmetric retrieval command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    _add_data_arguments(train)
    _add_runtime_arguments(train)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--excluded-negative-pairs", type=Path)
    train.add_argument("--antibody-input-dim", type=int, default=512)
    train.add_argument("--antigen-input-dim", type=int, default=1280)
    train.add_argument("--embedding-dim", type=int, default=128)
    train.add_argument("--antibody-hidden-dim", type=int, default=256)
    train.add_argument("--antigen-hidden-dim", type=int, default=256)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--chain-balanced-antibody-pooling", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument("--antibody-pooling", choices=("mean", "attention"), default="mean")
    train.add_argument("--antigen-pooling", choices=("mean", "attention"), default="mean")
    train.add_argument("--initial-temperature", type=float, default=0.2)
    train.add_argument("--minimum-temperature", type=float, default=0.01)
    train.add_argument("--maximum-temperature", type=float, default=1.0)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--validation-batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    train.add_argument("--weight-decay", type=float, default=0.2)
    train.add_argument("--schedule", choices=("cosine_warm_restarts", "warmup_cosine_decay"), default="cosine_warm_restarts")
    train.add_argument("--warmup-fraction", type=float, default=0.10)
    train.add_argument("--restart-period-epochs", type=int, default=10)
    train.add_argument("--restart-multiplier", type=int, default=2)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--max-epochs", type=int, default=100)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--min-delta", type=float, default=1e-4)
    train.add_argument("--min-train-pairs", type=int, default=32)
    train.add_argument("--min-validation-pairs", type=int, default=32)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    train.set_defaults(handler=command_train)

    final_refit = subparsers.add_parser("final-refit")
    _add_data_arguments(final_refit)
    _add_runtime_arguments(final_refit)
    final_refit.add_argument("--submission-manifest", type=Path, required=True)
    final_refit.add_argument("--submission-manifest-sha256", required=True)
    final_refit.add_argument("--output-dir", type=Path, required=True)
    final_refit.add_argument("--excluded-negative-pairs", type=Path)
    final_refit.add_argument("--antibody-input-dim", type=int, default=512)
    final_refit.add_argument("--antigen-input-dim", type=int, default=1280)
    final_refit.add_argument("--embedding-dim", type=int, default=128)
    final_refit.add_argument("--antibody-hidden-dim", type=int, default=256)
    final_refit.add_argument("--antigen-hidden-dim", type=int, default=256)
    final_refit.add_argument("--dropout", type=float, default=0.1)
    final_refit.add_argument(
        "--chain-balanced-antibody-pooling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    final_refit.add_argument(
        "--antibody-pooling", choices=("mean", "attention"), default="mean"
    )
    final_refit.add_argument(
        "--antigen-pooling", choices=("mean", "attention"), default="mean"
    )
    final_refit.add_argument("--initial-temperature", type=float, default=0.2)
    final_refit.add_argument("--minimum-temperature", type=float, default=0.01)
    final_refit.add_argument("--maximum-temperature", type=float, default=1.0)
    final_refit.add_argument("--batch-size", type=int, default=512)
    final_refit.add_argument("--learning-rate", type=float, default=1e-3)
    final_refit.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    final_refit.add_argument("--weight-decay", type=float, default=0.2)
    final_refit.add_argument(
        "--schedule",
        choices=("cosine_warm_restarts", "warmup_cosine_decay"),
        default="cosine_warm_restarts",
    )
    final_refit.add_argument("--warmup-fraction", type=float, default=0.10)
    final_refit.add_argument("--restart-period-epochs", type=int, default=10)
    final_refit.add_argument("--restart-multiplier", type=int, default=2)
    final_refit.add_argument("--gradient-clip", type=float, default=1.0)
    final_refit.add_argument("--fixed-epochs", type=int, required=True)
    final_refit.add_argument("--min-train-pairs", type=int, default=32)
    final_refit.add_argument("--log-every", type=int, default=10)
    final_refit.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    final_refit.set_defaults(handler=command_final_refit)

    evaluate = subparsers.add_parser("evaluate")
    _add_data_arguments(evaluate)
    _add_runtime_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--checkpoint-sha256", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--score-artifact", type=Path)
    evaluate.add_argument("--one-shot-ledger-registry-dir", type=Path)
    evaluate.add_argument("--antibody-input-dim", type=int, default=512)
    evaluate.add_argument("--antigen-input-dim", type=int, default=1280)
    evaluate.add_argument("--batch-size", type=int, default=128)
    evaluate.set_defaults(handler=command_evaluate)

    finalize_suite = subparsers.add_parser("finalize-evaluation-suite")
    finalize_suite.add_argument("--checkpoint", type=Path, required=True)
    finalize_suite.add_argument("--checkpoint-sha256", required=True)
    finalize_suite.add_argument(
        "--one-shot-ledger-registry-dir", type=Path, required=True
    )
    finalize_suite.set_defaults(handler=command_finalize_evaluation_suite)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--output", type=Path)
    smoke.set_defaults(handler=command_smoke)
    return parser


def main() -> None:
    """Dispatch one asymmetric partner-retrieval stage."""
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
