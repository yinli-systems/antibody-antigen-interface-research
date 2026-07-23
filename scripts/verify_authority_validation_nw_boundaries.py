#!/usr/bin/env python3
"""Independently verify authority-validation-only exact-NW boundaries.

This verifier recomputes exhaustive authority-train-by-authority-validation
global alignments in every declared view. It does not trust the edge artifacts
used by the boundary builder and does not load labels, predictions, embeddings,
checkpoints, or sealed-test sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from build_authority_validation_nw_boundary import (
    AUTHORITY_SHA256,
    CONSTRAINT_ROLES,
    canonical_sha256,
    load_authority_validation_components,
    load_digest_views,
    load_json,
    require_split_lists,
)
from build_identity_isolated_development_split import atomic_write_json, sha256_file
from needleman_wunsch_development_audit import (
    AuditPolicy,
    _unique_chain_sequences,
    _unique_sequences,
    audit_role,
    load_development_chain_records,
    load_development_records,
    write_primary_edge_artifact,
)


FORMAT_VERSION = "sabdab2_authority_validation_nw_boundary_verification_v2"


def _parse_named_paths(values: list[str]) -> dict[str, Path]:
    """Parse repeatable constraint-mode=path arguments."""
    result: dict[str, Path] = {}
    for value in values:
        mode, separator, raw_path = value.partition("=")
        if not separator or mode not in CONSTRAINT_ROLES or not raw_path:
            raise ValueError(f"Invalid boundary argument: {value!r}")
        if mode in result:
            raise ValueError(f"Duplicate boundary mode: {mode}")
        result[mode] = Path(raw_path).expanduser().resolve()
    return result


def _strip_internal(result: dict[str, object]) -> dict[str, object]:
    """Return an audit summary without process-local dataclass objects."""
    return {key: value for key, value in result.items() if not key.startswith("_")}


def _verify_static_accounting(
    boundary: Mapping[str, object],
    mode: str,
    authority_train: set[str],
    authority_validation: set[str],
    sealed_ids: set[str],
) -> tuple[set[str], set[str], set[str]]:
    """Verify origin, exclusion, and exact accounting of one boundary."""
    if boundary.get("constraint_mode") != mode:
        raise ValueError(f"Boundary mode mismatch for {mode}")
    if boundary.get("intended_use") != "reused_development_confirmation_only":
        raise ValueError(f"Boundary {mode} has an unsafe intended-use claim")
    raw_splits = boundary.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError(f"Boundary {mode} has no splits object")
    required = {"train", "validation", "excluded_validation"}
    if required - set(raw_splits):
        raise ValueError(f"Boundary {mode} is missing accounting partitions")
    train = set(map(str, raw_splits["train"]))
    validation = set(map(str, raw_splits["validation"]))
    excluded = set(map(str, raw_splits["excluded_validation"]))
    if train != authority_train:
        raise ValueError(f"Boundary {mode} does not retain exactly authority train")
    if validation & excluded or train & validation or train & excluded:
        raise ValueError(f"Boundary {mode} partitions overlap")
    if validation | excluded != authority_validation:
        raise ValueError(f"Boundary {mode} does not exactly account for authority validation")
    if (train | validation | excluded) & sealed_ids:
        raise ValueError(f"Boundary {mode} includes sealed identifiers")
    provenance_sha = boundary.get("provenance", {}).get("authority_split", {}).get(
        "sha256"
    )
    if provenance_sha != AUTHORITY_SHA256:
        raise ValueError(f"Boundary {mode} authority provenance changed")
    return train, validation, excluded


def run(args: argparse.Namespace) -> dict[str, object]:
    """Recompute the exact edge inventory and verify all supplied boundaries."""
    boundary_paths = _parse_named_paths(args.boundary)
    if set(boundary_paths) != set(CONSTRAINT_ROLES):
        raise ValueError("Both declared constraint modes must be verified together")
    authority_path = args.authority_split.expanduser().resolve()
    authority_sha = sha256_file(authority_path)
    if authority_sha != args.expected_authority_sha256:
        raise ValueError("Authority SHA-256 mismatch")
    authority = load_json(authority_path)
    splits = require_split_lists(authority)
    authority_train = set(splits[args.train_split])
    authority_validation = set(splits[args.validation_split])
    sealed_ids = {
        identifier for name in args.sealed_splits for identifier in splits[name]
    }
    if not authority_train or not authority_validation or not sealed_ids:
        raise ValueError("Authority train, validation, and sealed partitions must be non-empty")
    components = load_authority_validation_components(
        authority, authority_train, authority_validation, sealed_ids
    )
    boundary_payloads = {
        mode: load_json(path) for mode, path in boundary_paths.items()
    }
    static_partitions = {
        mode: _verify_static_accounting(
            boundary_payloads[mode],
            mode,
            authority_train,
            authority_validation,
            sealed_ids,
        )
        for mode in CONSTRAINT_ROLES
    }
    development_ids = authority_train | authority_validation
    manifest_path = args.sequence_manifest.expanduser().resolve()
    records = load_development_records(
        manifest_path,
        development_ids,
        args.id_column,
        args.heavy_column,
        args.light_column,
        args.antigen_column,
    )
    chain_records = load_development_chain_records(
        manifest_path,
        development_ids,
        args.id_column,
        args.heavy_column,
        args.light_column,
    )
    digest_views, access_summary = load_digest_views(
        manifest_path,
        development_ids,
        sealed_ids,
        args.id_column,
        args.heavy_column,
        args.light_column,
        args.antigen_column,
    )
    policy = AuditPolicy(
        identity_threshold=0.80,
        coverage_threshold=0.80,
        calm_scaled_threshold=0.80,
        gap_open=10.0,
        gap_extend=1.0,
        top_pair_count=args.top_pair_count,
    )
    roles = ("concat_antibody", "vh", "vl", "antigen")
    exact_edges: dict[str, list[object]] = {}
    role_reports: dict[str, object] = {}
    edge_dir = args.edge_output_dir.expanduser().resolve()
    for role in roles:
        if role == "concat_antibody":
            train_sequences = _unique_sequences(records, authority_train, "antibody")
            validation_sequences = _unique_sequences(
                records, authority_validation, "antibody"
            )
        elif role == "antigen":
            train_sequences = _unique_sequences(records, authority_train, "antigen")
            validation_sequences = _unique_sequences(
                records, authority_validation, "antigen"
            )
        else:
            train_sequences = _unique_chain_sequences(
                chain_records, authority_train, role
            )
            validation_sequences = _unique_chain_sequences(
                chain_records, authority_validation, role
            )
        role_result = audit_role(
            train_sequences,
            validation_sequences,
            policy,
            workers=args.workers,
            chunk_size=args.chunk_size,
        )
        exact_edges[role] = list(role_result["_primary_crossing_edges"])
        artifact = write_primary_edge_artifact(
            role_result,
            edge_dir,
            "authority_validation_independent",
            role,
        )
        role_reports[role] = {
            **_strip_internal(role_result),
            "identity_and_reciprocal_coverage_edge_artifact": artifact,
        }
    mode_reports: dict[str, object] = {}
    for mode, declared_roles in CONSTRAINT_ROLES.items():
        _, observed_validation, observed_excluded = static_partitions[mode]
        crossing_digests = {
            role: {str(edge.validation_digest) for edge in exact_edges[role]}
            for role in declared_roles
        }
        crossing_ids_by_role = {
            role: {
                identifier
                for identifier in authority_validation
                if digest_views[identifier][role] in crossing_digests[role]
            }
            for role in declared_roles
        }
        crossing_ids = set().union(*crossing_ids_by_role.values())
        expected_validation: set[str] = set()
        expected_excluded: set[str] = set()
        eligible_component_count = 0
        excluded_component_count = 0
        for component in components:
            members = set(map(str, component["members"]))
            if members & crossing_ids:
                expected_excluded.update(members)
                excluded_component_count += 1
            else:
                expected_validation.update(members)
                eligible_component_count += 1
        if observed_validation != expected_validation:
            raise ValueError(
                f"Boundary {mode} did not select every and only pure validation component"
            )
        if observed_excluded != expected_excluded:
            raise ValueError(f"Boundary {mode} excluded-component inventory changed")
        final_crossing_counts = {
            role: sum(
                digest_views[identifier][role] in crossing_digests[role]
                for identifier in observed_validation
            )
            for role in declared_roles
        }
        if any(final_crossing_counts.values()):
            raise AssertionError(f"Boundary {mode} retains an exact-NW train crossing")
        mode_reports[mode] = {
            "valid": True,
            "boundary": {
                "path": str(boundary_paths[mode]),
                "sha256": sha256_file(boundary_paths[mode]),
                "scientific_payload_sha256": boundary_payloads[mode].get(
                    "scientific_payload_sha256"
                ),
            },
            "authority_train_instance_count": len(authority_train),
            "eligible_validation_instance_count": len(observed_validation),
            "excluded_validation_instance_count": len(observed_excluded),
            "eligible_validation_component_count": eligible_component_count,
            "excluded_validation_component_count": excluded_component_count,
            "direct_crossing_validation_instance_count_by_role": {
                role: len(crossing_ids_by_role[role]) for role in declared_roles
            },
            "final_train_validation_exact_nw_crossing_count_by_role": (
                final_crossing_counts
            ),
            "authority_train_intersection_with_validation": len(
                authority_train & observed_validation
            ),
            "sealed_intersection_with_all_boundary_partitions": len(
                sealed_ids
                & (authority_train | observed_validation | observed_excluded)
            ),
        }
    source_path = Path(__file__).resolve()
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "stage": "independent_authority_validation_nw_boundary_verification",
        "intended_use": "reused_development_confirmation_only",
        "valid": all(bool(value["valid"]) for value in mode_reports.values()),
        "frozen_test_identifiers_read_for_exclusion_only": True,
        "frozen_test_sequence_fields_accessed_normalized_hashed_or_aligned": False,
        "labels_predictions_embeddings_or_checkpoints_read": False,
        "policy": {
            "backend": "Biopython PairwiseAligner",
            "mode": "global",
            "substitution_matrix": "BLOSUM62",
            "gap_open_penalty": 10.0,
            "gap_extend_penalty": 1.0,
            "identity_threshold": 0.80,
            "reciprocal_coverage_threshold": 0.80,
            "enumeration": (
                "all unique authority-train x authority-validation sequence pairs "
                "in concat-antibody, VH, non-empty VL, and antigen views"
            ),
        },
        "authority_counts": {
            "train": len(authority_train),
            "validation": len(authority_validation),
            "sealed_test": len(sealed_ids),
        },
        "role_recomputation": role_reports,
        "modes": mode_reports,
        "sequence_access_summary": access_summary,
        "provenance": {
            "authority_split": {"path": str(authority_path), "sha256": authority_sha},
            "sequence_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
        },
        "chronology_guardrail": (
            "These subsets were previously exposed through exploratory ESM2/ESM2 "
            "development work and are not independent test sets."
        ),
    }
    payload["scientific_payload_sha256"] = canonical_sha256(payload)
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "valid": payload["valid"],
                "modes": mode_reports,
                "scientific_payload_sha256": payload["scientific_payload_sha256"],
                "file_sha256": sha256_file(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-split", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", default=AUTHORITY_SHA256)
    parser.add_argument("--sequence-manifest", type=Path, required=True)
    parser.add_argument("--boundary", action="append", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--sealed-splits", nargs="+", default=["test"])
    parser.add_argument("--id-column", default="INSTANCE")
    parser.add_argument("--heavy-column", default="vh_seq")
    parser.add_argument("--light-column", default="vl_seq")
    parser.add_argument("--antigen-column", default="ag_seq")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-pair-count", type=int, default=20)
    parser.add_argument("--edge-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Run the independent verifier."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
