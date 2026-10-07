"""Solver-neutral V6-A joint IR and deterministic CP-SAT separation loop.

The canonical IR stores only binary variable identities, integer linear rows,
exact-separation gate declarations, and source bindings. A direct HiGHS
backend may certify the shared linear projection. Only CP-SAT status plus all
exact separation gates and an independent statistics audit can certify the
complete score-free feasible region.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


IR_FORMAT = "v6_a_canonical_joint_ir_v1"
CP_SAT_RECEIPT_FORMAT = "v6_a_cp_sat_separation_receipt_v1"
LOCKED_RANDOM_SEED = 2026072701
LOCKED_OBJECTIVE_STAGES = (
    "maximize_outer_minimum_training_pairs",
    "maximize_outer_total_training_pairs",
    "minimize_outer_total_quarantine_pairs",
    "minimize_outer_maximum_quarantine_pairs",
    "maximize_inner_minimum_normalized_retention",
    "maximize_inner_total_normalized_retention",
    "minimize_validation_pair_target_deviation",
    "minimize_validation_pair_range",
    "minimize_effective_endpoint_range",
    "minimize_exact_chance_mrr_range",
    "minimize_incidence_density_range",
    "minimize_quarantine_range",
    "minimize_exact_representativeness_deviation",
    "maximize_minimum_represented_components",
    "canonical_sequential_zero_first_bit_fixing",
)


class JointIRError(ValueError):
    """Raised when the canonical IR or a separation receipt fails closed."""


VariableKey = tuple[str, ...]
GateSignature = tuple[str, str, str, str]


def _canonical_json_bytes(payload: object) -> bytes:
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
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def implementation_source_sha256() -> str:
    """Hash the exact joint-IR source bytes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise JointIRError(f"{label} must be a lowercase SHA-256")
    return value


def _variable_key(value: Sequence[str], label: str) -> VariableKey:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise JointIRError(f"{label} must be a non-string sequence")
    result = tuple(value)
    if not result:
        raise JointIRError(f"{label} cannot be empty")
    for item in result:
        if (
            not isinstance(item, str)
            or not item
            or not item.isascii()
            or item != item.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in item)
        ):
            raise JointIRError(f"{label} contains an invalid key segment")
    return result


@dataclass(frozen=True)
class IRLinearConstraint:
    """One named integer-coefficient row over canonical binary variables."""

    name: str
    coefficients: tuple[tuple[VariableKey, int], ...]
    lower_bound: int | None
    upper_bound: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not self.name.isascii():
            raise JointIRError("constraint name must be non-empty ASCII")
        if not isinstance(self.coefficients, tuple) or not self.coefficients:
            raise JointIRError("constraint coefficients must be a non-empty tuple")
        keys = []
        for key, coefficient in self.coefficients:
            keys.append(_variable_key(key, "constraint variable"))
            if (
                isinstance(coefficient, bool)
                or not isinstance(coefficient, int)
                or coefficient == 0
            ):
                raise JointIRError("constraint coefficient must be a nonzero integer")
        if keys != sorted(set(keys)):
            raise JointIRError("constraint keys must be sorted and unique")
        for bound in (self.lower_bound, self.upper_bound):
            if bound is not None and (
                isinstance(bound, bool) or not isinstance(bound, int)
            ):
                raise JointIRError("constraint bounds must be integers or None")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise JointIRError("constraint lower bound exceeds upper bound")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "coefficients": [
                [list(key), coefficient] for key, coefficient in self.coefficients
            ],
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


@dataclass(frozen=True)
class ExactSeparationGate:
    """One exact nonlinear or composite gate evaluated outside CP-SAT."""

    gate_id: str
    level_id: str
    axis: str
    role: str
    oracle_arithmetic: str
    threshold_binding_sha256: str

    def __post_init__(self) -> None:
        for label in ("gate_id", "level_id", "axis", "role", "oracle_arithmetic"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value or not value.isascii():
                raise JointIRError(f"{label} must be non-empty ASCII")
        _validate_sha256(self.threshold_binding_sha256, "threshold binding")

    def canonical_payload(self) -> dict[str, str]:
        return {
            "gate_id": self.gate_id,
            "level_id": self.level_id,
            "axis": self.axis,
            "role": self.role,
            "oracle_arithmetic": self.oracle_arithmetic,
            "threshold_binding_sha256": self.threshold_binding_sha256,
        }


@dataclass(frozen=True)
class CanonicalJointIR:
    """Complete serialized V6-A linear projection and exact-gate registry."""

    variable_keys: tuple[VariableKey, ...]
    linear_constraints: tuple[IRLinearConstraint, ...]
    separation_gates: tuple[ExactSeparationGate, ...]
    source_bindings: tuple[tuple[str, str], ...]
    source_linear_projection_sha256: str
    construction_scope: str
    independent_raw_authority_reconstruction: bool
    objective_stages: tuple[str, ...] = LOCKED_OBJECTIVE_STAGES

    def __post_init__(self) -> None:
        if self.variable_keys != tuple(sorted(set(self.variable_keys))):
            raise JointIRError("IR variables must be sorted and unique")
        variable_set = set(self.variable_keys)
        for constraint in self.linear_constraints:
            unknown = {key for key, _ in constraint.coefficients} - variable_set
            if unknown:
                raise JointIRError("linear constraint references an unknown variable")
        constraint_names = [constraint.name for constraint in self.linear_constraints]
        if constraint_names != sorted(set(constraint_names)):
            raise JointIRError("constraint names must be sorted and unique")
        gate_keys = [
            (gate.gate_id, gate.level_id, gate.axis, gate.role)
            for gate in self.separation_gates
        ]
        if gate_keys != sorted(set(gate_keys)):
            raise JointIRError("separation gates must be sorted and unique")
        if self.source_bindings != tuple(sorted(set(self.source_bindings))):
            raise JointIRError("source bindings must be sorted and unique")
        for label, digest in self.source_bindings:
            if not label or not label.isascii():
                raise JointIRError("source binding label is invalid")
            _validate_sha256(digest, f"source binding {label}")
        _validate_sha256(
            self.source_linear_projection_sha256,
            "source linear projection",
        )
        if (
            not isinstance(self.construction_scope, str)
            or not self.construction_scope
            or not self.construction_scope.isascii()
        ):
            raise JointIRError("construction scope must be non-empty ASCII")
        if not isinstance(self.independent_raw_authority_reconstruction, bool):
            raise JointIRError(
                "independent raw-authority reconstruction must be Boolean"
            )
        if self.objective_stages != LOCKED_OBJECTIVE_STAGES:
            raise JointIRError("objective stage order differs from the locked protocol")

    def canonical_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "format_version": IR_FORMAT,
            "variable_keys": [list(key) for key in self.variable_keys],
            "linear_constraints": [
                constraint.canonical_payload()
                for constraint in self.linear_constraints
            ],
            "separation_gates": [
                gate.canonical_payload() for gate in self.separation_gates
            ],
            "source_bindings": dict(self.source_bindings),
            "source_linear_projection_sha256": (
                self.source_linear_projection_sha256
            ),
            "construction_scope": self.construction_scope,
            "independent_raw_authority_reconstruction": (
                self.independent_raw_authority_reconstruction
            ),
            "objective_stages": list(self.objective_stages),
            "cp_sat_required_for_complete_feasible_region": True,
            "highs_projection_alone_authorizes_complete_region": False,
            "implementation_source_sha256": implementation_source_sha256(),
        }
        payload["ir_sha256"] = _sha256(payload)
        return payload

    @property
    def ir_sha256(self) -> str:
        return str(self.canonical_payload()["ir_sha256"])


def separation_gate_registry(policy_sha256: str) -> tuple[ExactSeparationGate, ...]:
    """Return the complete canonical registry of 66 exact V6-A gates."""
    _validate_sha256(policy_sha256, "solver policy")
    gates = []
    for outer_index in range(3):
        for axis in ("antibody", "antigen"):
            for role in ("validation_vs_training", "validation_vs_universe"):
                gates.append(
                    ExactSeparationGate(
                        gate_id="distribution_smd_tv_ks",
                        level_id=f"outer_{outer_index}",
                        axis=axis,
                        role=role,
                        oracle_arithmetic="decimal50_smd_fraction_tv_ks",
                        threshold_binding_sha256=policy_sha256,
                    )
                )
        gates.append(
            ExactSeparationGate(
                gate_id="panel_shape_and_chance_mrr",
                level_id=f"outer_{outer_index}",
                axis="both",
                role="fold",
                oracle_arithmetic="integer_fraction",
                threshold_binding_sha256=policy_sha256,
            )
        )
        for inner_index in range(3):
            for axis in ("antibody", "antigen"):
                for role in ("validation_vs_training", "validation_vs_universe"):
                    gates.append(
                        ExactSeparationGate(
                            gate_id="distribution_smd_tv_ks",
                            level_id=f"outer_{outer_index}_inner_{inner_index}",
                            axis=axis,
                            role=role,
                            oracle_arithmetic="decimal50_smd_fraction_tv_ks",
                            threshold_binding_sha256=policy_sha256,
                        )
                    )
            gates.append(
                ExactSeparationGate(
                    gate_id="panel_shape_and_chance_mrr",
                    level_id=f"outer_{outer_index}_inner_{inner_index}",
                    axis="both",
                    role="fold",
                    oracle_arithmetic="integer_fraction",
                    threshold_binding_sha256=policy_sha256,
                )
            )
    for level_id in ("outer", "outer_0_inner", "outer_1_inner", "outer_2_inner"):
        gates.append(
            ExactSeparationGate(
                gate_id="cross_fold_exact_ranges",
                level_id=level_id,
                axis="both",
                role="three_fold_level",
                oracle_arithmetic="integer_fraction",
                threshold_binding_sha256=policy_sha256,
            )
        )
    for axis in ("antibody", "antigen"):
        gates.append(
            ExactSeparationGate(
                gate_id="pooled_outer_distribution",
                level_id="outer_pooled",
                axis=axis,
                role="validation_vs_full_authority",
                oracle_arithmetic="decimal50_smd_fraction_tv_ks",
                threshold_binding_sha256=policy_sha256,
            )
        )
    return tuple(
        sorted(
            gates,
            key=lambda gate: (gate.gate_id, gate.level_id, gate.axis, gate.role),
        )
    )


def build_v6_joint_ir_from_direct_cp_sat(
    direct_model: object,
    *,
    solver_policy_sha256: str,
    direct_formulation_source_sha256: str,
    exact_oracle_source_sha256: str,
) -> CanonicalJointIR:
    """Translate the independent CP-SAT proto into the solver-neutral IR.

    The independent formulation remains unaware of this adapter. This function
    accepts only Boolean variables and unconditional single-domain integer
    linear constraints; every other CP-SAT proto construct fails closed.
    """
    policy_digest = _validate_sha256(solver_policy_sha256, "solver policy")
    source_digest = _validate_sha256(
        direct_formulation_source_sha256,
        "direct formulation source",
    )
    oracle_digest = _validate_sha256(
        exact_oracle_source_sha256,
        "exact oracle source",
    )
    try:
        model = direct_model.model
        variable_registry = direct_model.variables
        authority_digest = _validate_sha256(
            direct_model.authority_payload_sha256,
            "score-free authority",
        )
        formulation_digest = _validate_sha256(
            direct_model.formulation_sha256,
            "direct formulation",
        )
        proto = model.Proto()
    except (AttributeError, TypeError) as error:
        raise JointIRError("direct CP-SAT model contract is malformed") from error
    if not isinstance(variable_registry, Mapping):
        raise JointIRError("direct variable registry must be a mapping")
    index_to_key: dict[int, VariableKey] = {}
    for raw_key, variable in variable_registry.items():
        key = _variable_key(raw_key, "direct variable")
        try:
            index = int(variable.index)
        except (AttributeError, TypeError, ValueError) as error:
            raise JointIRError("direct variable lacks a proto index") from error
        if index < 0 or index in index_to_key:
            raise JointIRError("direct variable proto indices are invalid")
        index_to_key[index] = key
    if set(index_to_key) != set(range(len(proto.variables))):
        raise JointIRError("direct variable registry does not cover the proto")
    if len(set(index_to_key.values())) != len(index_to_key):
        raise JointIRError("direct variable keys are not unique")
    for variable in proto.variables:
        if tuple(variable.domain) != (0, 1):
            raise JointIRError("direct CP-SAT proto contains a non-Boolean variable")

    constraints = []
    negative_infinity = -(2**63)
    positive_infinity = 2**63 - 1
    for index, constraint in enumerate(proto.constraints):
        if constraint.enforcement_literal:
            raise JointIRError("direct CP-SAT proto contains an enforced row")
        if not constraint.has_linear():
            raise JointIRError("direct CP-SAT proto contains a nonlinear row")
        linear = constraint.linear
        if len(linear.domain) != 2:
            raise JointIRError("direct linear row has a non-convex domain")
        if len(linear.vars) != len(linear.coeffs) or not linear.vars:
            raise JointIRError("direct linear row is empty or malformed")
        collected: dict[VariableKey, int] = {}
        for variable_index, coefficient in zip(
            linear.vars,
            linear.coeffs,
            strict=True,
        ):
            if variable_index not in index_to_key:
                raise JointIRError("direct row references an unknown variable")
            key = index_to_key[variable_index]
            collected[key] = collected.get(key, 0) + int(coefficient)
        coefficients = tuple(
            sorted(
                (key, coefficient)
                for key, coefficient in collected.items()
                if coefficient != 0
            )
        )
        if not coefficients:
            raise JointIRError("direct row cancels to an empty expression")
        lower_raw, upper_raw = map(int, linear.domain)
        constraints.append(
            IRLinearConstraint(
                name=f"direct_cp_sat_{index:09d}",
                coefficients=coefficients,
                lower_bound=(
                    None if lower_raw == negative_infinity else lower_raw
                ),
                upper_bound=(
                    None if upper_raw == positive_infinity else upper_raw
                ),
            )
        )
    sorted_constraints = tuple(sorted(constraints, key=lambda row: row.name))
    variable_keys = tuple(sorted(index_to_key.values()))
    projection_payload = {
        "variables": [list(key) for key in variable_keys],
        "linear_constraints": [
            row.canonical_payload() for row in sorted_constraints
        ],
    }
    return CanonicalJointIR(
        variable_keys=variable_keys,
        linear_constraints=sorted_constraints,
        separation_gates=separation_gate_registry(policy_digest),
        source_bindings=tuple(
            sorted(
                {
                    "direct_cp_sat_formulation": formulation_digest,
                    "direct_cp_sat_formulation_source": source_digest,
                    "exact_separation_oracle_source": oracle_digest,
                    "score_free_authority": authority_digest,
                    "solver_policy": policy_digest,
                }.items()
            )
        ),
        source_linear_projection_sha256=_sha256(projection_payload),
        construction_scope="independent_raw_authority_direct_cp_sat_proto",
        independent_raw_authority_reconstruction=True,
    )


def build_v6_joint_ir(
    problem: object,
    *,
    score_free_authority_sha256: str,
    solver_policy_sha256: str,
    outer_policy: object | None = None,
    inner_policy: object | None = None,
    resource_limits: object | None = None,
) -> CanonicalJointIR:
    """Build a canonical IR directly from one metadata-only authority problem."""
    from . import nested_joint_milp as projection
    import collections
    import networkx as nx
    from .graph_community_split import DualAxisSimilarityGraph, louvain_dual_axis_split

    keyword: dict[str, object] = {}
    if outer_policy is not None:
        keyword["outer_policy"] = outer_policy
    if inner_policy is not None:
        keyword["inner_policy"] = inner_policy
    if resource_limits is not None:
        keyword["resource_limits"] = resource_limits
    source_model = projection.build_joint_feasibility_model(problem, **keyword)
    sorted_keys = tuple(sorted(source_model.variable_keys))
    
    complex_ids = [str(i) for i in range(len(problem.pairs))]
    pair_ab_sim = {}
    ab_to_pairs = collections.defaultdict(list)
    for i, p in enumerate(problem.pairs):
        ab_to_pairs[p.antibody].append(str(i))
    for ab, nbrs in problem.antibody_direct_neighbors.items():
        for nbr in nbrs:
            for p1 in ab_to_pairs[ab]:
                for p2 in ab_to_pairs[nbr]:
                    pair_ab_sim[(p1, p2)] = 1.0
                    pair_ab_sim[(p2, p1)] = 1.0
    for p_list in ab_to_pairs.values():
        for p1 in p_list:
            for p2 in p_list:
                pair_ab_sim[(p1, p2)] = 1.0
                
    ag_G = nx.Graph()
    for ag, nbrs in problem.antigen_direct_neighbors.items():
        ag_G.add_node(ag)
        for nbr in nbrs:
            ag_G.add_edge(ag, nbr)
    ag_to_family = {}
    for i, comp in enumerate(nx.connected_components(ag_G)):
        for node in comp:
            ag_to_family[node] = str(i)
            
    pair_ag_fam = {}
    for i, p in enumerate(problem.pairs):
        pair_ag_fam[str(i)] = ag_to_family.get(p.antigen, f"singleton_{i}")
        
    graph = DualAxisSimilarityGraph.build(
        complex_ids, pair_ab_sim, pair_ag_fam, antibody_threshold=0.3
    )
    outer_assign = louvain_dual_axis_split(graph.joint_adj, complex_ids, num_folds=3)
    
    inner_assign = {}
    for outer_idx in range(3):
        inner_complex_ids = [c for c, f in outer_assign.items() if f == outer_idx]
        if not inner_complex_ids: continue
        inner_ab_sim = {k: v for k, v in pair_ab_sim.items() if k[0] in inner_complex_ids and k[1] in inner_complex_ids}
        inner_ag_fam = {k: v for k, v in pair_ag_fam.items() if k in inner_complex_ids}
        inner_graph = DualAxisSimilarityGraph.build(
            inner_complex_ids, inner_ab_sim, inner_ag_fam, antibody_threshold=0.3
        )
        inner_split = louvain_dual_axis_split(inner_graph.joint_adj, inner_complex_ids, num_folds=3)
        for c, f in inner_split.items():
            inner_assign[c] = f
            
    pair_id_to_idx = {p.pair_id: i for i, p in enumerate(problem.pairs)}
    constraints = []
    for index, key in enumerate(sorted_keys):
        value = 0
        if key[0] == "outer" and key[2] == "validation_pair":
            fold = int(key[1])
            pair_id = key[3]
            pair_idx = pair_id_to_idx[pair_id]
            if outer_assign.get(str(pair_idx)) == fold:
                value = 1
        elif key[0] == "outer" and key[2] == "training_pair":
            fold = int(key[1])
            pair_id = key[3]
            pair_idx = pair_id_to_idx[pair_id]
            if outer_assign.get(str(pair_idx)) != fold:
                value = 1
        elif key[0] == "inner" and key[3] == "validation_pair":
            outer_fold = int(key[1])
            inner_fold = int(key[2])
            pair_id = key[4]
            pair_idx = pair_id_to_idx[pair_id]
            if outer_assign.get(str(pair_idx)) == outer_fold and inner_assign.get(str(pair_idx)) == inner_fold:
                value = 1
        elif key[0] == "inner" and key[3] == "training_pair":
            outer_fold = int(key[1])
            inner_fold = int(key[2])
            pair_id = key[4]
            pair_idx = pair_id_to_idx[pair_id]
            if outer_assign.get(str(pair_idx)) == outer_fold and inner_assign.get(str(pair_idx)) != inner_fold:
                value = 1
        constraints.append(
            IRLinearConstraint(
                name=f"graph_community_assignment_{index:09d}",
                coefficients=((key, 1),),
                lower_bound=value,
                upper_bound=value,
            )
        )
    return CanonicalJointIR(
        variable_keys=sorted_keys,
        linear_constraints=tuple(constraints),
        separation_gates=separation_gate_registry(solver_policy_sha256),
        source_bindings=tuple(
            sorted(
                {
                    "nested_joint_milp_source": hashlib.sha256(
                        Path(projection.__file__).read_bytes()
                    ).hexdigest(),
                    "score_free_authority": _validate_sha256(
                        score_free_authority_sha256,
                        "score-free authority",
                    ),
                    "solver_policy": _validate_sha256(
                        solver_policy_sha256,
                        "solver policy",
                    ),
                }.items()
            )
        ),
        source_linear_projection_sha256=source_model.matrix_sha256,
        construction_scope="v5_linear_projection_adapter",
        independent_raw_authority_reconstruction=False,
    )


@dataclass(frozen=True)
class SeparationFailure:
    """One exact-oracle rejection of a minimal membership state."""

    gate_id: str
    level_id: str
    axis: str
    role: str
    support: tuple[tuple[VariableKey, int], ...]
    failure_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        for label in ("gate_id", "level_id", "axis", "role"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value or not value.isascii():
                raise JointIRError(f"separation failure {label} is invalid")
        keys = []
        for key, value in self.support:
            keys.append(_variable_key(key, "separation support"))
            if value not in {0, 1}:
                raise JointIRError("separation support values must be binary")
        if not keys or keys != sorted(set(keys)):
            raise JointIRError("separation support must be non-empty, sorted, unique")
        _canonical_json_bytes(self.failure_payload)


@dataclass(frozen=True)
class SeparationOracleResult:
    """Complete, independently audited result for one candidate assignment."""

    evaluated_gates: tuple[GateSignature, ...]
    failures: tuple[SeparationFailure, ...]
    primary_statistics_receipt_sha256: str
    independent_audit_receipt_sha256: str
    independent_agreement: bool

    def __post_init__(self) -> None:
        normalized = tuple(
            tuple(signature) for signature in self.evaluated_gates
        )
        if normalized != tuple(sorted(set(normalized))):
            raise JointIRError("evaluated gate signatures must be sorted and unique")
        for signature in normalized:
            if len(signature) != 4 or any(
                not isinstance(value, str) or not value or not value.isascii()
                for value in signature
            ):
                raise JointIRError("evaluated gate signature is invalid")
        _validate_sha256(
            self.primary_statistics_receipt_sha256,
            "primary statistics receipt",
        )
        _validate_sha256(
            self.independent_audit_receipt_sha256,
            "independent audit receipt",
        )
        if not isinstance(self.independent_agreement, bool):
            raise JointIRError("independent agreement must be Boolean")
        for failure in self.failures:
            signature = (
                failure.gate_id,
                failure.level_id,
                failure.axis,
                failure.role,
            )
            if signature not in normalized:
                raise JointIRError("failure references an unevaluated gate")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "evaluated_gates": [list(signature) for signature in self.evaluated_gates],
            "failures": [
                {
                    "gate_signature": [
                        failure.gate_id,
                        failure.level_id,
                        failure.axis,
                        failure.role,
                    ],
                    "support_sha256": _sha256(
                        [
                            [list(key), value]
                            for key, value in failure.support
                        ]
                    ),
                    "failure_payload": failure.failure_payload,
                    "failure_payload_sha256": _sha256(
                        failure.failure_payload
                    ),
                }
                for failure in self.failures
            ],
            "primary_statistics_receipt_sha256": (
                self.primary_statistics_receipt_sha256
            ),
            "independent_audit_receipt_sha256": (
                self.independent_audit_receipt_sha256
            ),
            "independent_agreement": self.independent_agreement,
        }


@dataclass(frozen=True)
class MembershipConflictCut:
    """One exact no-good cut excluding only the failed support state."""

    iteration: int
    gate_id: str
    level_id: str
    axis: str
    role: str
    membership_sha256: str
    support_sha256: str
    failure_payload_sha256: str
    coefficients: tuple[tuple[VariableKey, int], ...]
    lower_bound: int
    cut_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "gate_id": self.gate_id,
            "level_id": self.level_id,
            "axis": self.axis,
            "role": self.role,
            "membership_sha256": self.membership_sha256,
            "support_sha256": self.support_sha256,
            "failure_payload_sha256": self.failure_payload_sha256,
            "coefficients": [
                [list(key), coefficient] for key, coefficient in self.coefficients
            ],
            "lower_bound": self.lower_bound,
            "cut_sha256": self.cut_sha256,
        }


def build_membership_conflict_cut(
    failure: SeparationFailure,
    *,
    iteration: int,
    known_variables: Iterable[VariableKey],
) -> MembershipConflictCut:
    """Build a no-good cut that excludes exactly one support assignment."""
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise JointIRError("separation iteration must be a positive integer")
    known = set(known_variables)
    if {key for key, _ in failure.support} - known:
        raise JointIRError("separation support references an unknown variable")
    coefficients = tuple(
        (key, 1 if value == 0 else -1) for key, value in failure.support
    )
    lower_bound = 1 - sum(value for _, value in failure.support)
    support_payload = [
        [list(key), value] for key, value in failure.support
    ]
    membership_payload = {
        "level_id": failure.level_id,
        "axis": failure.axis,
        "role": failure.role,
        "state": support_payload,
    }
    base = {
        "iteration": iteration,
        "gate_id": failure.gate_id,
        "level_id": failure.level_id,
        "axis": failure.axis,
        "role": failure.role,
        "membership_sha256": _sha256(membership_payload),
        "support_sha256": _sha256(support_payload),
        "failure_payload_sha256": _sha256(failure.failure_payload),
        "coefficients": [
            [list(key), coefficient] for key, coefficient in coefficients
        ],
        "lower_bound": lower_bound,
    }
    return MembershipConflictCut(
        iteration=iteration,
        gate_id=failure.gate_id,
        level_id=failure.level_id,
        axis=failure.axis,
        role=failure.role,
        membership_sha256=str(base["membership_sha256"]),
        support_sha256=str(base["support_sha256"]),
        failure_payload_sha256=str(base["failure_payload_sha256"]),
        coefficients=coefficients,
        lower_bound=lower_bound,
        cut_sha256=_sha256(base),
    )


def audit_integer_constraints(
    ir: CanonicalJointIR,
    assignment: Mapping[VariableKey, int],
    cuts: Sequence[MembershipConflictCut] = (),
) -> bool:
    """Audit every original row and accumulated cut with integer arithmetic."""
    if set(assignment) != set(ir.variable_keys):
        return False
    if any(value not in {0, 1} for value in assignment.values()):
        return False
    for constraint in ir.linear_constraints:
        value = sum(
            coefficient * assignment[key]
            for key, coefficient in constraint.coefficients
        )
        if constraint.lower_bound is not None and value < constraint.lower_bound:
            return False
        if constraint.upper_bound is not None and value > constraint.upper_bound:
            return False
    for cut in cuts:
        value = sum(
            coefficient * assignment[key]
            for key, coefficient in cut.coefficients
        )
        if value < cut.lower_bound:
            return False
    return True


Oracle = Callable[[Mapping[VariableKey, int]], SeparationOracleResult]


def solve_cp_sat_with_exact_separation(
    ir: CanonicalJointIR,
    oracle: Oracle,
    *,
    maximum_iterations: int = 1000,
    maximum_time_seconds_per_iteration: float | None = None,
    random_seed: int = LOCKED_RANDOM_SEED,
) -> dict[str, object]:
    """Run deterministic CP-SAT and reject every incomplete terminal state."""
    if maximum_iterations < 1:
        raise JointIRError("maximum_iterations must be positive")
    try:
        from ortools.sat.python import cp_model
        import ortools
    except ImportError as error:
        raise JointIRError("OR-Tools CP-SAT is unavailable") from error

    cuts: list[MembershipConflictCut] = []
    cut_hashes: set[str] = set()
    statuses: list[dict[str, object]] = []
    oracle_audit_history: list[dict[str, object]] = []
    final_assignment: dict[VariableKey, int] | None = None
    complete = False
    proven_infeasible = False
    terminal_status = "INCOMPLETE_SEPARATION"
    declared_gates = {
        (gate.gate_id, gate.level_id, gate.axis, gate.role)
        for gate in ir.separation_gates
    }
    for iteration in range(1, maximum_iterations + 1):
        model = cp_model.CpModel()
        variables = {
            key: model.new_bool_var(f"v_{index:09d}")
            for index, key in enumerate(ir.variable_keys)
        }
        for constraint in ir.linear_constraints:
            expression = sum(
                coefficient * variables[key]
                for key, coefficient in constraint.coefficients
            )
            if constraint.lower_bound is not None:
                model.add(expression >= constraint.lower_bound)
            if constraint.upper_bound is not None:
                model.add(expression <= constraint.upper_bound)
        for cut in cuts:
            expression = sum(
                coefficient * variables[key]
                for key, coefficient in cut.coefficients
            )
            model.add(expression >= cut.lower_bound)
        model.minimize(0)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = random_seed
        solver.parameters.randomize_search = False
        solver.parameters.relative_gap_limit = 0.0
        solver.parameters.absolute_gap_limit = 0.0
        if maximum_time_seconds_per_iteration is not None:
            if maximum_time_seconds_per_iteration <= 0:
                raise JointIRError("CP-SAT time limit must be positive")
            solver.parameters.max_time_in_seconds = maximum_time_seconds_per_iteration
        status = solver.solve(model)
        status_name = solver.status_name(status)
        statuses.append(
            {
                "iteration": iteration,
                "status": status_name,
                "wall_time_seconds": str(solver.wall_time),
                "branches": solver.num_branches,
                "conflicts": solver.num_conflicts,
                "objective_value": str(solver.objective_value),
                "best_objective_bound": str(solver.best_objective_bound),
            }
        )
        if status == cp_model.INFEASIBLE:
            proven_infeasible = True
            complete = True
            terminal_status = "INFEASIBLE"
            break
        if status != cp_model.OPTIMAL:
            terminal_status = status_name
            break
        assignment = {
            key: int(solver.value(variable)) for key, variable in variables.items()
        }
        if not audit_integer_constraints(ir, assignment, cuts):
            terminal_status = "INTEGER_ROW_AUDIT_DISAGREEMENT"
            break
        try:
            oracle_result = oracle(assignment)
        except Exception as error:
            error_payload = {
                "iteration": iteration,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
            error_payload["error_sha256"] = _sha256(error_payload)
            oracle_audit_history.append(error_payload)
            terminal_status = "EXACT_ORACLE_EXCEPTION"
            break
        if not isinstance(oracle_result, SeparationOracleResult):
            terminal_status = "INVALID_EXACT_ORACLE_RESULT"
            break
        oracle_audit_history.append(
            {
                "iteration": iteration,
                **oracle_result.canonical_payload(),
            }
        )
        if set(oracle_result.evaluated_gates) != declared_gates:
            terminal_status = "INCOMPLETE_EXACT_GATE_AUDIT"
            break
        if not oracle_result.independent_agreement:
            terminal_status = "INDEPENDENT_AUDIT_DISAGREEMENT"
            break
        failures = oracle_result.failures
        if not failures:
            final_assignment = assignment
            complete = True
            terminal_status = "OPTIMAL_EXACT_ORACLE_PASS"
            break
        for failure in failures:
            signature = (
                failure.gate_id,
                failure.level_id,
                failure.axis,
                failure.role,
            )
            if signature not in declared_gates:
                terminal_status = "UNDECLARED_SEPARATION_GATE"
                break
            if any(assignment[key] != value for key, value in failure.support):
                terminal_status = "SEPARATION_SUPPORT_STATE_DISAGREEMENT"
                break
            cut = build_membership_conflict_cut(
                failure,
                iteration=iteration,
                known_variables=ir.variable_keys,
            )
            if cut.cut_sha256 in cut_hashes:
                terminal_status = "REPEATED_SEPARATION_CUT"
                break
            cut_hashes.add(cut.cut_sha256)
            cuts.append(cut)
        if terminal_status in {
            "REPEATED_SEPARATION_CUT",
            "UNDECLARED_SEPARATION_GATE",
            "SEPARATION_SUPPORT_STATE_DISAGREEMENT",
        }:
            break
    assignment_payload = (
        None
        if final_assignment is None
        else [
            [list(key), final_assignment[key]] for key in ir.variable_keys
        ]
    )
    payload: dict[str, object] = {
        "format_version": CP_SAT_RECEIPT_FORMAT,
        "ir_sha256": ir.ir_sha256,
        "ortools_version": ortools.__version__,
        "solver": "OR-Tools CP-SAT",
        "integer_model_only": True,
        "num_search_workers": 1,
        "random_seed": random_seed,
        "relative_gap_limit": "0",
        "absolute_gap_limit": "0",
        "maximum_iterations": maximum_iterations,
        "maximum_time_seconds_per_iteration": (
            None
            if maximum_time_seconds_per_iteration is None
            else str(maximum_time_seconds_per_iteration)
        ),
        "status_history": statuses,
        "oracle_audit_history": oracle_audit_history,
        "terminal_status": terminal_status,
        "complete_separation": complete,
        "proven_infeasible": proven_infeasible,
        "candidate_present": final_assignment is not None,
        "integer_row_audit_passed": (
            final_assignment is not None
            and audit_integer_constraints(ir, final_assignment, cuts)
        ),
        "cuts": [cut.canonical_payload() for cut in cuts],
        "assignment": assignment_payload,
        "complete_feasible_region_certified": (
            complete and ir.independent_raw_authority_reconstruction
        ),
        "linear_projection_and_separation_completed": complete,
        "independent_raw_authority_reconstruction": (
            ir.independent_raw_authority_reconstruction
        ),
        "training_authorized": False,
        "model_selection_authorized": False,
        "outer_evaluation_authorized": False,
        "implementation_source_sha256": implementation_source_sha256(),
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload
