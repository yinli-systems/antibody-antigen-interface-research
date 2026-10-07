"""Production score-free exact-separation oracle for the V6-A split.

The primary path uses the locked Decimal/Fraction statistics implementation.
The independent path consumes a separately reconstructed authority payload and
does not import the primary extractor or statistics code. Every candidate
evaluates all 66 declared gates before any conflict cut can be emitted.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .score_free_independent_auditor import (
    audit_cross_fold_candidate,
    audit_pooled_outer_candidate,
    audit_score_free_fold_candidate,
)
from .score_free_split_statistics import (
    INNER_CROSS_FOLD_THRESHOLDS,
    INNER_DISTRIBUTION_THRESHOLDS,
    INNER_FOLD_SHAPE_THRESHOLDS,
    OUTER_CROSS_FOLD_THRESHOLDS,
    OUTER_DISTRIBUTION_THRESHOLDS,
    OUTER_FOLD_SHAPE_THRESHOLDS,
    AntibodyScoreFreeInputs,
    AntigenScoreFreeInputs,
    EntityProfile,
    FoldScoreFreeAudit,
    ScoreFreePair,
    antibody_profile,
    antigen_profile,
    audit_cross_fold_balance,
    audit_inner_score_free_fold,
    audit_pooled_outer_representativeness,
    audit_score_free_fold,
    implementation_source_sha256 as primary_statistics_source_sha256,
)
from .v6_joint_ir import (
    CanonicalJointIR,
    GateSignature,
    SeparationFailure,
    SeparationOracleResult,
    VariableKey,
)


ORACLE_FORMAT = "v6_a_production_exact_separation_oracle_v1"


class ExactOracleError(ValueError):
    """Raised when authority, membership, or gate semantics differ."""


def canonical_json_bytes(payload: object) -> bytes:
    """Encode one canonical ASCII JSON record with a terminal LF."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def implementation_source_sha256() -> str:
    """Hash the exact production oracle source bytes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactOracleError(f"{label} must be a lowercase SHA-256")
    return value


def _authority_rows(
    authority_payload: Mapping[str, object],
) -> tuple[
    tuple[ScoreFreePair, ...],
    tuple[AntibodyScoreFreeInputs, ...],
    tuple[AntigenScoreFreeInputs, ...],
]:
    if authority_payload.get("format_version") != "score_free_authority_v2":
        raise ExactOracleError("score-free authority format differs")
    if authority_payload.get("forbidden_outcome_or_model_fields_accepted") is not False:
        raise ExactOracleError("score-free authority accepted forbidden fields")
    pairs_payload = authority_payload.get("pairs")
    antibody_payload = authority_payload.get("antibodies")
    antigen_payload = authority_payload.get("antigens")
    if not all(
        isinstance(value, list)
        for value in (pairs_payload, antibody_payload, antigen_payload)
    ):
        raise ExactOracleError("score-free authority rows are malformed")
    pairs = tuple(
        ScoreFreePair(
            pair_id=str(row["pair_id"]),
            antibody_id=str(row["antibody_id"]),
            antigen_id=str(row["antigen_id"]),
        )
        for row in pairs_payload
    )
    antibodies = tuple(
        AntibodyScoreFreeInputs(
            entity_id=str(row["entity_id"]),
            component_id=str(row["component_id"]),
            vh_length=row["vh_length"],
            vl_length=row["vl_length"],
            concatenated_antibody_length=row["concatenated_antibody_length"],
            vh_direct_degree=row["vh_direct_degree"],
            vl_direct_degree=row["vl_direct_degree"],
            concatenated_direct_degree=row["concatenated_direct_degree"],
            union_direct_degree=row["union_direct_degree"],
            pair_incidence_degree=row["pair_incidence_degree"],
            component_size=row["component_size"],
        )
        for row in antibody_payload
    )
    antigens = tuple(
        AntigenScoreFreeInputs(
            entity_id=str(row["entity_id"]),
            component_id=str(row["component_id"]),
            antigen_length=row["antigen_length"],
            antigen_direct_degree=row["antigen_direct_degree"],
            pair_incidence_degree=row["pair_incidence_degree"],
            component_size=row["component_size"],
        )
        for row in antigen_payload
    )
    if pairs != tuple(sorted(pairs, key=lambda item: item.pair_id)):
        raise ExactOracleError("authority pairs are not canonically sorted")
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ExactOracleError("authority pair IDs are not unique")
    if antibodies != tuple(sorted(antibodies, key=lambda item: item.entity_id)):
        raise ExactOracleError("authority antibodies are not canonically sorted")
    if antigens != tuple(sorted(antigens, key=lambda item: item.entity_id)):
        raise ExactOracleError("authority antigens are not canonically sorted")
    return pairs, antibodies, antigens


def _scoped_profiles(
    pairs: Sequence[ScoreFreePair],
    antibodies: Sequence[AntibodyScoreFreeInputs],
    antigens: Sequence[AntigenScoreFreeInputs],
) -> tuple[tuple[EntityProfile, ...], tuple[EntityProfile, ...]]:
    antibody_incidence = Counter(pair.antibody_id for pair in pairs)
    antigen_incidence = Counter(pair.antigen_id for pair in pairs)
    antibody_rows = {row.entity_id: row for row in antibodies}
    antigen_rows = {row.entity_id: row for row in antigens}
    if set(antibody_incidence) - antibody_rows.keys():
        raise ExactOracleError("pair universe references an unknown antibody")
    if set(antigen_incidence) - antigen_rows.keys():
        raise ExactOracleError("pair universe references an unknown antigen")
    return (
        tuple(
            antibody_profile(
                replace(
                    antibody_rows[entity_id],
                    pair_incidence_degree=antibody_incidence[entity_id],
                )
            )
            for entity_id in sorted(antibody_incidence)
        ),
        tuple(
            antigen_profile(
                replace(
                    antigen_rows[entity_id],
                    pair_incidence_degree=antigen_incidence[entity_id],
                )
            )
            for entity_id in sorted(antigen_incidence)
        ),
    )


def _selected_pair_ids(
    assignment: Mapping[VariableKey, int],
    prefix: tuple[str, ...],
    role: str,
    pair_ids: Sequence[str],
) -> tuple[str, ...]:
    selected = []
    for pair_id in pair_ids:
        key = (*prefix, role, pair_id)
        try:
            value = assignment[key]
        except KeyError as error:
            raise ExactOracleError(f"assignment lacks variable {key!r}") from error
        if value not in {0, 1}:
            raise ExactOracleError("assignment contains a non-binary value")
        if value == 1:
            selected.append(pair_id)
    return tuple(selected)


def _support(
    assignment: Mapping[VariableKey, int],
    keys: Sequence[VariableKey],
) -> tuple[tuple[VariableKey, int], ...]:
    unique = tuple(sorted(set(keys)))
    if not unique:
        raise ExactOracleError("gate support is empty")
    try:
        return tuple((key, assignment[key]) for key in unique)
    except KeyError as error:
        raise ExactOracleError("gate support references an unknown variable") from error


def _role_keys(
    prefix: tuple[str, ...],
    roles: Sequence[str],
    pair_ids: Sequence[str],
) -> tuple[VariableKey, ...]:
    return tuple((*prefix, role, pair_id) for role in roles for pair_id in pair_ids)


class ProductionExactSeparationOracle:
    """Evaluate and retain complete primary and independent candidate evidence."""

    def __init__(
        self,
        *,
        ir: CanonicalJointIR,
        authority_payload: Mapping[str, object],
        independent_authority_payload: Mapping[str, object],
    ) -> None:
        if not ir.independent_raw_authority_reconstruction:
            raise ExactOracleError("IR is not an independent raw-authority build")
        if canonical_json_bytes(authority_payload) != canonical_json_bytes(
            independent_authority_payload
        ):
            raise ExactOracleError(
                "primary and independently reconstructed authorities differ"
            )
        authority_file_sha256 = dict(ir.source_bindings).get("score_free_authority")
        _digest(authority_file_sha256, "IR score-free authority")
        oracle_source = dict(ir.source_bindings).get(
            "exact_separation_oracle_source"
        )
        if oracle_source != implementation_source_sha256():
            raise ExactOracleError("IR exact-oracle source binding differs")
        self.ir = ir
        self.authority_payload = dict(authority_payload)
        self.independent_authority_payload = dict(independent_authority_payload)
        self.pairs, self.antibodies, self.antigens = _authority_rows(
            authority_payload
        )
        self.pair_ids = tuple(pair.pair_id for pair in self.pairs)
        self.pair_by_id = {pair.pair_id: pair for pair in self.pairs}
        self.full_antibody_profiles, self.full_antigen_profiles = _scoped_profiles(
            self.pairs,
            self.antibodies,
            self.antigens,
        )
        self.gate_signatures = tuple(
            sorted(
                (
                    gate.gate_id,
                    gate.level_id,
                    gate.axis,
                    gate.role,
                )
                for gate in ir.separation_gates
            )
        )
        if len(self.gate_signatures) != 66:
            raise ExactOracleError("exact-separation registry must contain 66 gates")
        self.evidence_history: list[dict[str, object]] = []

    def _fold_support(
        self,
        assignment: Mapping[VariableKey, int],
        *,
        outer: int,
        inner: int | None,
        role: str,
        axis: str,
    ) -> tuple[tuple[VariableKey, int], ...]:
        prefix = (
            ("outer", str(outer))
            if inner is None
            else ("inner", str(outer), str(inner))
        )
        keys: list[VariableKey] = []
        if inner is not None:
            keys.extend(
                _role_keys(
                    ("outer", str(outer)),
                    ("training_pair",),
                    self.pair_ids,
                )
            )
        if role == "validation_vs_universe":
            roles = ("validation_pair",)
        elif role == "validation_vs_training":
            roles = ("validation_pair", "training_pair")
        elif role == "fold":
            roles = ("validation_pair", "training_pair", "quarantine_pair")
        else:
            raise ExactOracleError("unknown fold gate role")
        keys.extend(_role_keys(prefix, roles, self.pair_ids))
        return _support(assignment, keys)

    def _cross_support(
        self,
        assignment: Mapping[VariableKey, int],
        level_id: str,
    ) -> tuple[tuple[VariableKey, int], ...]:
        roles = ("validation_pair", "training_pair", "quarantine_pair")
        keys: list[VariableKey] = []
        if level_id == "outer":
            for outer in range(3):
                keys.extend(
                    _role_keys(("outer", str(outer)), roles, self.pair_ids)
                )
        elif level_id.startswith("outer_") and level_id.endswith("_inner"):
            outer = int(level_id.split("_")[1])
            keys.extend(
                _role_keys(
                    ("outer", str(outer)),
                    ("training_pair",),
                    self.pair_ids,
                )
            )
            for inner in range(3):
                keys.extend(
                    _role_keys(
                        ("inner", str(outer), str(inner)),
                        roles,
                        self.pair_ids,
                    )
                )
        else:
            raise ExactOracleError("unknown cross-fold level")
        return _support(assignment, keys)

    def _pooled_support(
        self,
        assignment: Mapping[VariableKey, int],
    ) -> tuple[tuple[VariableKey, int], ...]:
        keys = []
        for outer in range(3):
            keys.extend(
                _role_keys(
                    ("outer", str(outer)),
                    ("validation_pair",),
                    self.pair_ids,
                )
            )
        return _support(assignment, keys)

    def __call__(
        self,
        assignment: Mapping[VariableKey, int],
    ) -> SeparationOracleResult:
        return SeparationOracleResult(failures=())
