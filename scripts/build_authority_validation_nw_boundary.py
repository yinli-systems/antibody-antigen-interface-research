#!/usr/bin/env python3
"""Build a post-selection boundary strictly inside authority validation.

The builder consumes the authoritative component assignment and a previously
completed exhaustive authority-train-by-authority-validation exact-NW audit. It
does not read labels, predictions, embeddings, or checkpoints. Sequence fields
are accessed only for authority train and authority validation identifiers;
sealed identifiers are used only for exclusion checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from build_identity_isolated_development_split import (
    atomic_write_json,
    normalize_sequence,
    sequence_digest,
    sha256_file,
)


FORMAT_VERSION = "sabdab2_authority_validation_nw_boundary_v2"
AUTHORITY_SHA256 = "e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b"
CONSTRAINT_ROLES = {
    "concat_antibody_and_antigen": ("concat_antibody", "antigen"),
    "concat_antibody_antigen_plus_vh_vl": (
        "concat_antibody",
        "vh",
        "vl",
        "antigen",
    ),
}


def canonical_sha256(payload: object) -> str:
    """Hash one JSON-compatible object using canonical serialization."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    """Load one JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def require_split_lists(payload: Mapping[str, object]) -> dict[str, list[str]]:
    """Validate and return disjoint split lists."""
    raw = payload.get("splits")
    if not isinstance(raw, dict):
        raise ValueError("Authority does not contain a splits object")
    result: dict[str, list[str]] = {}
    observed: set[str] = set()
    for name, values in raw.items():
        if not isinstance(values, list):
            raise TypeError(f"Split {name!r} is not a list")
        normalized = [str(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Split {name!r} contains duplicate identifiers")
        overlap = observed.intersection(normalized)
        if overlap:
            raise ValueError(f"Authority splits overlap: {sorted(overlap)[:5]}")
        observed.update(normalized)
        result[str(name)] = normalized
    return result


def load_digest_views(
    path: Path,
    development_ids: set[str],
    sealed_ids: set[str],
    id_column: str,
    heavy_column: str,
    light_column: str,
    antigen_column: str,
) -> tuple[dict[str, dict[str, str | None]], dict[str, int]]:
    """Load digest views only for authorized development identifiers.

    The identifier gate is evaluated before any sequence column is accessed.
    Rows belonging to sealed identifiers contribute only to the exclusion
    counter and their sequence fields are never indexed, normalized, or hashed.
    """
    views: dict[str, dict[str, str | None]] = {}
    sealed_identifier_rows_skipped = 0
    unrelated_rows_skipped = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Sequence manifest is empty") from exc
        positions = {name: index for index, name in enumerate(header)}
        required = {id_column, heavy_column, light_column, antigen_column}
        missing = required - set(positions)
        if missing:
            raise ValueError(f"Sequence manifest is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Malformed manifest row {row_number}")
            identifier = str(row[positions[id_column]])
            if identifier in sealed_ids:
                sealed_identifier_rows_skipped += 1
                continue
            if identifier not in development_ids:
                unrelated_rows_skipped += 1
                continue
            if identifier in views:
                raise ValueError(f"Duplicate development identifier: {identifier}")
            heavy = normalize_sequence(
                row[positions[heavy_column]], f"{identifier}:{heavy_column}"
            )
            light = normalize_sequence(
                row[positions[light_column]],
                f"{identifier}:{light_column}",
                allow_empty=True,
            )
            antigen = normalize_sequence(
                row[positions[antigen_column]], f"{identifier}:{antigen_column}"
            )
            views[identifier] = {
                "concat_antibody": sequence_digest(heavy + light),
                "vh": sequence_digest(heavy),
                "vl": sequence_digest(light) if light else None,
                "antigen": sequence_digest(antigen),
            }
    missing_ids = development_ids - set(views)
    if missing_ids:
        raise ValueError(
            f"Sequence manifest is missing development IDs: {sorted(missing_ids)[:10]}"
        )
    return views, {
        "development_rows_materialized": len(views),
        "sealed_identifier_rows_skipped_before_sequence_access": (
            sealed_identifier_rows_skipped
        ),
        "unrelated_rows_skipped_before_sequence_access": unrelated_rows_skipped,
    }


def load_authority_validation_components(
    authority: Mapping[str, object],
    authority_train: set[str],
    authority_validation: set[str],
    sealed_ids: set[str],
) -> list[dict[str, object]]:
    """Return every source component assigned wholly to authority validation."""
    raw_components = authority.get("components")
    if not isinstance(raw_components, list):
        raise ValueError("Authority does not expose source graph components")
    observed: set[str] = set()
    validation_components: list[dict[str, object]] = []
    all_authority_ids = authority_train | authority_validation | sealed_ids
    for index, raw_component in enumerate(raw_components):
        if not isinstance(raw_component, dict):
            raise TypeError(f"Authority component {index} is not an object")
        component_id = str(raw_component.get("component_id", f"component_{index:05d}"))
        raw_members = raw_component.get("members")
        if not isinstance(raw_members, list):
            raise TypeError(f"Authority component {component_id} has no member list")
        members = {str(value) for value in raw_members}
        if not members:
            raise ValueError(f"Authority component {component_id} is empty")
        if members - all_authority_ids:
            raise ValueError(f"Authority component {component_id} has unknown members")
        overlap = observed & members
        if overlap:
            raise ValueError(f"Authority components overlap: {sorted(overlap)[:5]}")
        observed.update(members)
        occupied = [
            name
            for name, identifiers in (
                ("train", authority_train),
                ("validation", authority_validation),
                ("test", sealed_ids),
            )
            if members & identifiers
        ]
        if len(occupied) != 1:
            raise ValueError(
                f"Authority component {component_id} spans source splits: {occupied}"
            )
        if occupied[0] == "validation":
            validation_components.append(
                {"component_id": component_id, "members": sorted(members)}
            )
    if observed != all_authority_ids:
        raise ValueError("Authority components do not exactly cover authority identifiers")
    covered_validation = {
        identifier
        for component in validation_components
        for identifier in component["members"]
    }
    if covered_validation != authority_validation:
        raise AssertionError("Validation components do not exactly cover authority validation")
    return sorted(validation_components, key=lambda item: str(item["component_id"]))


def _artifact_from_report(
    report: Mapping[str, object], role: str
) -> Mapping[str, object]:
    """Resolve one active-boundary exact-NW artifact descriptor."""
    if role in {"concat_antibody", "antigen"}:
        result_role = "antibody" if role == "concat_antibody" else role
        return report["results"]["active"]["roles"][result_role][
            "identity_and_reciprocal_coverage_edge_artifact"
        ]
    return report["supplementary_sensitivity_analyses"]["antibody_chain_aware"][
        "splits"
    ]["active"]["chains"][role][
        "identity_and_reciprocal_coverage_edge_artifact"
    ]


def resolve_artifact_path(descriptor: Mapping[str, object], edge_dir: Path) -> Path:
    """Resolve an artifact by basename under an explicitly trusted directory."""
    declared = Path(str(descriptor["path"]))
    candidate = edge_dir.expanduser().resolve() / declared.name
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def load_crossing_validation_digests(
    report: Mapping[str, object],
    edge_dir: Path,
    roles: Sequence[str],
    train_digests: Mapping[str, set[str]],
    validation_digests: Mapping[str, set[str]],
) -> tuple[dict[str, set[str]], dict[str, object]]:
    """Load and validate all relevant exhaustive cross-boundary edge artifacts."""
    result: dict[str, set[str]] = {}
    provenance: dict[str, object] = {}
    for role in roles:
        descriptor = _artifact_from_report(report, role)
        path = resolve_artifact_path(descriptor, edge_dir)
        observed_sha = sha256_file(path)
        if observed_sha != str(descriptor["sha256"]):
            raise ValueError(f"Exact-NW artifact hash changed: {path}")
        crossing_validation: set[str] = set()
        edge_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                edge = json.loads(line)
                train_digest = str(edge["train_digest"])
                validation_digest = str(edge["validation_digest"])
                if train_digest not in train_digests[role]:
                    raise ValueError(
                        f"Artifact {role} row {line_number} has a non-train digest"
                    )
                if validation_digest not in validation_digests[role]:
                    raise ValueError(
                        f"Artifact {role} row {line_number} has a non-validation digest"
                    )
                if not bool(edge["violates_identity_and_coverage_rule"]):
                    raise ValueError(f"Artifact {role} contains a non-primary edge")
                crossing_validation.add(validation_digest)
                edge_count += 1
        if edge_count != int(descriptor["edge_count"]):
            raise ValueError(f"Exact-NW artifact count changed: {path}")
        result[role] = crossing_validation
        provenance[role] = {
            "path": str(path),
            "sha256": observed_sha,
            "edge_count": edge_count,
            "validation_digest_count_with_train_edge": len(crossing_validation),
        }
    return result, provenance


def _membership_sha256(identifiers: Iterable[str]) -> str:
    """Hash a sorted identifier inventory."""
    return hashlib.sha256(
        ("\n".join(sorted(identifiers)) + "\n").encode("utf-8")
    ).hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    """Build and persist one corrected authority-validation boundary."""
    authority_path = args.authority_split.expanduser().resolve()
    authority_sha = sha256_file(authority_path)
    if authority_sha != args.expected_authority_sha256:
        raise ValueError(
            f"Authority SHA-256 mismatch: expected {args.expected_authority_sha256}, "
            f"observed {authority_sha}"
        )
    authority = load_json(authority_path)
    splits = require_split_lists(authority)
    required_splits = {args.train_split, args.validation_split, *args.sealed_splits}
    if required_splits - set(splits):
        raise ValueError(f"Authority is missing splits: {sorted(required_splits-set(splits))}")
    authority_train = set(splits[args.train_split])
    authority_validation = set(splits[args.validation_split])
    sealed_ids = {
        identifier for name in args.sealed_splits for identifier in splits[name]
    }
    if not authority_train or not authority_validation or not sealed_ids:
        raise ValueError("Train, validation, and sealed authority partitions must be non-empty")
    development_ids = authority_train | authority_validation
    validation_components = load_authority_validation_components(
        authority, authority_train, authority_validation, sealed_ids
    )
    views, access_summary = load_digest_views(
        args.sequence_manifest.expanduser().resolve(),
        development_ids,
        sealed_ids,
        args.id_column,
        args.heavy_column,
        args.light_column,
        args.antigen_column,
    )
    roles = CONSTRAINT_ROLES[args.constraint_mode]
    train_digests = {
        role: {
            digest
            for identifier in authority_train
            if (digest := views[identifier][role]) is not None
        }
        for role in roles
    }
    validation_digests = {
        role: {
            digest
            for identifier in authority_validation
            if (digest := views[identifier][role]) is not None
        }
        for role in roles
    }
    report_path = args.authority_nw_audit.expanduser().resolve()
    report = load_json(report_path)
    source = report.get("provenance", {}).get("development_source_split", {})
    if str(source.get("sha256")) != authority_sha:
        raise ValueError("Exact-NW audit was not run against the declared authority")
    if report.get("frozen_test_sequences_aligned") is not False:
        raise ValueError("Exact-NW audit does not prove the sealed sequences stayed closed")
    crossing_digests, edge_provenance = load_crossing_validation_digests(
        report,
        args.edge_dir,
        roles,
        train_digests,
        validation_digests,
    )
    direct_crossing_ids_by_role = {
        role: sorted(
            identifier
            for identifier in authority_validation
            if views[identifier][role] in crossing_digests[role]
        )
        for role in roles
    }
    any_direct_crossing = {
        identifier
        for role in roles
        for identifier in direct_crossing_ids_by_role[role]
    }
    eligible_components: list[dict[str, object]] = []
    excluded_components: list[dict[str, object]] = []
    for component in validation_components:
        members = list(component["members"])
        crossing_by_role = {
            role: sorted(set(members) & set(direct_crossing_ids_by_role[role]))
            for role in roles
        }
        record = {
            "component_id": component["component_id"],
            "members": members,
            "member_count": len(members),
            "direct_train_crossing_member_ids_by_role": crossing_by_role,
        }
        if set(members) & any_direct_crossing:
            excluded_components.append(record)
        else:
            eligible_components.append(record)
    validation_ids = sorted(
        identifier
        for component in eligible_components
        for identifier in component["members"]
    )
    excluded_ids = sorted(
        identifier
        for component in excluded_components
        for identifier in component["members"]
    )
    train_ids = sorted(authority_train)
    if set(validation_ids) | set(excluded_ids) != authority_validation:
        raise AssertionError("Corrected boundary does not account for authority validation")
    if set(validation_ids) & set(excluded_ids):
        raise AssertionError("Validation and excluded-validation overlap")
    if set(train_ids) & (set(validation_ids) | set(excluded_ids)):
        raise AssertionError("Authority train leaked into a validation partition")
    if (set(train_ids) | set(validation_ids) | set(excluded_ids)) & sealed_ids:
        raise AssertionError("Sealed identifier entered the corrected boundary")
    source_path = Path(__file__).resolve()
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "stage": "build_authority_validation_nw_clean_boundary",
        "constraint_mode": args.constraint_mode,
        "declared_views": list(roles),
        "intended_use": "reused_development_confirmation_only",
        "valid": True,
        "selection_influenced_by_labels_predictions_or_model_scores": False,
        "frozen_test_identifiers_read_for_exclusion_only": True,
        "frozen_test_sequence_fields_accessed_normalized_hashed_or_aligned": False,
        "splits": {
            "train": train_ids,
            "validation": validation_ids,
            "excluded_validation": excluded_ids,
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
        "counts": {
            "authority_train": len(authority_train),
            "authority_validation": len(authority_validation),
            "sealed_test": len(sealed_ids),
            "source_authority_validation_components": len(validation_components),
            "eligible_validation_components": len(eligible_components),
            "excluded_validation_components": len(excluded_components),
            "eligible_validation_instances": len(validation_ids),
            "excluded_validation_instances": len(excluded_ids),
            "direct_crossing_validation_instances": len(any_direct_crossing),
            "direct_crossing_validation_instances_by_role": {
                role: len(values)
                for role, values in direct_crossing_ids_by_role.items()
            },
        },
        "membership_sha256": {
            "train": _membership_sha256(train_ids),
            "validation": _membership_sha256(validation_ids),
            "excluded_validation": _membership_sha256(excluded_ids),
        },
        "eligible_components": eligible_components,
        "excluded_components": excluded_components,
        "evidence": {
            "identity_threshold": 0.80,
            "reciprocal_coverage_threshold": 0.80,
            "authority_train_by_validation_exact_nw_artifacts": edge_provenance,
            "sequence_access_summary": access_summary,
        },
        "provenance": {
            "authority_split": {"path": str(authority_path), "sha256": authority_sha},
            "sequence_manifest": {
                "path": str(args.sequence_manifest.expanduser().resolve()),
                "sha256": sha256_file(args.sequence_manifest),
            },
            "authority_nw_audit": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
                "scientific_payload_sha256": report.get("scientific_payload_sha256"),
            },
            "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "protocol_correction": {
                "path": str(args.protocol.expanduser().resolve()),
                "sha256": sha256_file(args.protocol),
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
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "constraint_mode": args.constraint_mode,
                "counts": payload["counts"],
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
    parser.add_argument(
        "--expected-authority-sha256", default=AUTHORITY_SHA256
    )
    parser.add_argument("--sequence-manifest", type=Path, required=True)
    parser.add_argument("--authority-nw-audit", type=Path, required=True)
    parser.add_argument("--edge-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--constraint-mode", choices=tuple(CONSTRAINT_ROLES), required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--sealed-splits", nargs="+", default=["test"])
    parser.add_argument("--id-column", default="INSTANCE")
    parser.add_argument("--heavy-column", default="vh_seq")
    parser.add_argument("--light-column", default="vl_seq")
    parser.add_argument("--antigen-column", default="ag_seq")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Run the boundary builder."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
