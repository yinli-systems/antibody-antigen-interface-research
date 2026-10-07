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
        """Evaluate one candidate and emit exact failures with minimal scopes."""
        if set(assignment) != set(self.ir.variable_keys):
            raise ExactOracleError("candidate assignment does not cover the IR")
        if any(value not in {0, 1} for value in assignment.values()):
            raise ExactOracleError("candidate assignment is not binary")
        outer_audits: dict[int, FoldScoreFreeAudit] = {}
        inner_audits: dict[tuple[int, int], FoldScoreFreeAudit] = {}
        fold_payloads: dict[str, Mapping[str, object]] = {}
        independent_folds: dict[str, Mapping[str, object]] = {}
        for outer in range(3):
            prefix = ("outer", str(outer))
            validation_ids = _selected_pair_ids(
                assignment, prefix, "validation_pair", self.pair_ids
            )
            training_ids = _selected_pair_ids(
                assignment, prefix, "training_pair", self.pair_ids
            )
            fold_id = f"outer_{outer}"
            audit = audit_score_free_fold(
                fold_id=fold_id,
                pairs=self.pairs,
                antibody_profiles=self.full_antibody_profiles,
                antigen_profiles=self.full_antigen_profiles,
                validation_pair_ids=validation_ids,
                training_pair_ids=training_ids,
                shape_thresholds=OUTER_FOLD_SHAPE_THRESHOLDS,
                distribution_thresholds=OUTER_DISTRIBUTION_THRESHOLDS,
            )
            payload = audit.canonical_payload()
            outer_audits[outer] = audit
            fold_payloads[fold_id] = payload
            independent_folds[fold_id] = audit_score_free_fold_candidate(
                candidate_fold_bytes=canonical_json_bytes(payload),
                authority_payload=self.independent_authority_payload,
                scope="outer",
                expected_primary_source_sha256=primary_statistics_source_sha256(),
            )
            inner_universe = tuple(self.pair_by_id[pair_id] for pair_id in training_ids)
            inner_antibodies, inner_antigens = _scoped_profiles(
                inner_universe,
                self.antibodies,
                self.antigens,
            )
            for inner in range(3):
                inner_prefix = ("inner", str(outer), str(inner))
                inner_validation = _selected_pair_ids(
                    assignment,
                    inner_prefix,
                    "validation_pair",
                    self.pair_ids,
                )
                inner_training = _selected_pair_ids(
                    assignment,
                    inner_prefix,
                    "training_pair",
                    self.pair_ids,
                )
                inner_id = f"outer_{outer}_inner_{inner}"
                inner_audit = audit_inner_score_free_fold(
                    outer_fold_id=fold_id,
                    inner_fold_id=inner_id,
                    complete_authority_pairs=self.pairs,
                    outer_training_pair_ids=training_ids,
                    inner_universe_pairs=inner_universe,
                    antibody_profiles=inner_antibodies,
                    antigen_profiles=inner_antigens,
                    validation_pair_ids=inner_validation,
                    training_pair_ids=inner_training,
                    shape_thresholds=INNER_FOLD_SHAPE_THRESHOLDS,
                    distribution_thresholds=INNER_DISTRIBUTION_THRESHOLDS,
                )
                inner_payload = inner_audit.canonical_payload()
                inner_audits[(outer, inner)] = inner_audit
                fold_payloads[inner_id] = inner_payload
                independent_folds[inner_id] = audit_score_free_fold_candidate(
                    candidate_fold_bytes=canonical_json_bytes(inner_payload),
                    authority_payload=self.independent_authority_payload,
                    scope="inner",
                    expected_primary_source_sha256=(
                        primary_statistics_source_sha256()
                    ),
                    expected_outer_fold_id=fold_id,
                    expected_outer_training_pair_ids=training_ids,
                )

        outer_cross = audit_cross_fold_balance(
            tuple(outer_audits[index] for index in range(3)),
            thresholds=OUTER_CROSS_FOLD_THRESHOLDS,
        )
        inner_cross = {
            outer: audit_cross_fold_balance(
                tuple(inner_audits[(outer, index)] for index in range(3)),
                thresholds=INNER_CROSS_FOLD_THRESHOLDS,
            )
            for outer in range(3)
        }
        outer_cross_payload = outer_cross.canonical_payload()
        inner_cross_payloads = {
            outer: inner_cross[outer].canonical_payload() for outer in range(3)
        }
        independent_cross = {
            "outer": audit_cross_fold_candidate(
                candidate_bytes=canonical_json_bytes(outer_cross_payload),
                fold_payloads=[fold_payloads[f"outer_{index}"] for index in range(3)],
                scope="outer",
                expected_primary_source_sha256=primary_statistics_source_sha256(),
            )
        }
        for outer in range(3):
            independent_cross[f"outer_{outer}_inner"] = audit_cross_fold_candidate(
                candidate_bytes=canonical_json_bytes(inner_cross_payloads[outer]),
                fold_payloads=[
                    fold_payloads[f"outer_{outer}_inner_{index}"]
                    for index in range(3)
                ],
                scope="inner",
                expected_primary_source_sha256=primary_statistics_source_sha256(),
            )
        pooled = audit_pooled_outer_representativeness(
            tuple(outer_audits[index] for index in range(3)),
            antibody_profiles=self.full_antibody_profiles,
            antigen_profiles=self.full_antigen_profiles,
        )
        pooled_payload = pooled.canonical_payload()
        independent_pooled = audit_pooled_outer_candidate(
            candidate_bytes=canonical_json_bytes(pooled_payload),
            authority_payload=self.independent_authority_payload,
            fold_payloads=[fold_payloads[f"outer_{index}"] for index in range(3)],
            expected_primary_source_sha256=primary_statistics_source_sha256(),
        )

        assignment_sha256 = _sha256(
            [[list(key), assignment[key]] for key in self.ir.variable_keys]
        )
        primary_bundle = {
            "format_version": ORACLE_FORMAT,
            "assignment_sha256": assignment_sha256,
            "folds": dict(sorted(fold_payloads.items())),
            "cross_fold": {
                "outer": outer_cross_payload,
                **{
                    f"outer_{outer}_inner": inner_cross_payloads[outer]
                    for outer in range(3)
                },
            },
            "pooled_outer": pooled_payload,
            "primary_statistics_source_sha256": primary_statistics_source_sha256(),
            "exact_oracle_source_sha256": implementation_source_sha256(),
        }
        independent_bundle = {
            "format_version": "v6_a_independent_candidate_audit_bundle_v1",
            "assignment_sha256": assignment_sha256,
            "folds": dict(sorted(independent_folds.items())),
            "cross_fold": dict(sorted(independent_cross.items())),
            "pooled_outer": independent_pooled,
        }
        primary_receipt_sha256 = _sha256(primary_bundle)
        independent_receipt_sha256 = _sha256(independent_bundle)
        failures: list[SeparationFailure] = []

        for outer in range(3):
            for inner in (None, 0, 1, 2):
                if inner is None:
                    level_id = f"outer_{outer}"
                    audit = outer_audits[outer]
                else:
                    level_id = f"outer_{outer}_inner_{inner}"
                    audit = inner_audits[(outer, inner)]
                for axis, attribute_prefix in (
                    ("antibody", "antibody"),
                    ("antigen", "antigen"),
                ):
                    for role, attribute_suffix in (
                        ("validation_vs_training", "vs_training"),
                        ("validation_vs_universe", "vs_universe"),
                    ):
                        comparison = getattr(
                            audit,
                            f"{attribute_prefix}_{attribute_suffix}",
                        )
                        if not comparison.passed:
                            failures.append(
                                SeparationFailure(
                                    gate_id="distribution_smd_tv_ks",
                                    level_id=level_id,
                                    axis=axis,
                                    role=role,
                                    support=self._fold_support(
                                        assignment,
                                        outer=outer,
                                        inner=inner,
                                        role=role,
                                        axis=axis,
                                    ),
                                    failure_payload={
                                        "comparison": comparison.canonical_payload(),
                                        "fold_audit_sha256": audit.audit_sha256,
                                        "independent_fold_receipt_sha256": _sha256(
                                            independent_folds[level_id]
                                        ),
                                    },
                                )
                            )
                if not audit.panel.shape_passed:
                    failures.append(
                        SeparationFailure(
                            gate_id="panel_shape_and_chance_mrr",
                            level_id=level_id,
                            axis="both",
                            role="fold",
                            support=self._fold_support(
                                assignment,
                                outer=outer,
                                inner=inner,
                                role="fold",
                                axis="both",
                            ),
                            failure_payload={
                                "panel": audit.canonical_payload()["panel"],
                                "fold_audit_sha256": audit.audit_sha256,
                                "independent_fold_receipt_sha256": _sha256(
                                    independent_folds[level_id]
                                ),
                            },
                        )
                    )

        for level_id, audit in (
            ("outer", outer_cross),
            *((f"outer_{outer}_inner", inner_cross[outer]) for outer in range(3)),
        ):
            if not audit.passed:
                failures.append(
                    SeparationFailure(
                        gate_id="cross_fold_exact_ranges",
                        level_id=level_id,
                        axis="both",
                        role="three_fold_level",
                        support=self._cross_support(assignment, level_id),
                        failure_payload={
                            "cross_fold": audit.canonical_payload(),
                            "independent_cross_receipt_sha256": _sha256(
                                independent_cross[level_id]
                            ),
                        },
                    )
                )
        for axis, comparison in (
            ("antibody", pooled.antibody_vs_universe),
            ("antigen", pooled.antigen_vs_universe),
        ):
            if not comparison.passed:
                failures.append(
                    SeparationFailure(
                        gate_id="pooled_outer_distribution",
                        level_id="outer_pooled",
                        axis=axis,
                        role="validation_vs_full_authority",
                        support=self._pooled_support(assignment),
                        failure_payload={
                            "comparison": comparison.canonical_payload(),
                            "pooled_audit_sha256": pooled.audit_sha256,
                            "independent_pooled_receipt_sha256": _sha256(
                                independent_pooled
                            ),
                        },
                    )
                )
        failures.sort(
            key=lambda failure: (
                failure.gate_id,
                failure.level_id,
                failure.axis,
                failure.role,
            )
        )
        evidence = {
            "assignment_sha256": assignment_sha256,
            "primary_statistics_receipt_sha256": primary_receipt_sha256,
            "independent_audit_receipt_sha256": independent_receipt_sha256,
            "primary_statistics": primary_bundle,
            "independent_audit": independent_bundle,
            "failure_count": len(failures),
            "all_66_gates_evaluated": True,
            "independent_agreement": True,
            "training_authorized": False,
        }
        evidence["evidence_sha256"] = _sha256(evidence)
        self.evidence_history.append(evidence)
        return SeparationOracleResult(
            evaluated_gates=self.gate_signatures,
            failures=tuple(failures),
            primary_statistics_receipt_sha256=primary_receipt_sha256,
            independent_audit_receipt_sha256=independent_receipt_sha256,
            independent_agreement=True,
        )

