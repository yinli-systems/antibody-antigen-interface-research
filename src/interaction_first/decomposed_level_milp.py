"""Exact one-level 3-fold MILP used to decompose the nested retrieval split.

Scientific policy is imported verbatim from :mod:`nested_joint_milp`.  This module
changes only optimization factorization: one outer 3-fold problem is solved first,
then each realized outer training universe can be solved as an independent inner
3-fold problem.  No model score, label, validation/test sequence, or embedding is
accepted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from interaction_first import nested_joint_milp as njm


@dataclass(frozen=True)
class LevelSolve:
    """One exact 3-fold level solve plus held-out endpoint sets."""

    result: njm.JointSolveResult
    held_out_antibodies: tuple[frozenset[str], frozenset[str], frozenset[str]] | None
    held_out_antigens: tuple[frozenset[str], frozenset[str], frozenset[str]] | None


def _edge_count(graph: Mapping[str, frozenset[str]]) -> int:
    seen: set[tuple[str, str]] = set()
    for left, values in graph.items():
        for right in values:
            if left == right:
                continue
            seen.add(tuple(sorted((left, right))))
    return len(seen)


def build_level_feasibility_model(
    problem: njm.JointSplitProblem,
    *,
    policy: njm.JointLevelPolicy,
    level_label: str,
) -> njm.JointMILPModel:
    """Build the exact three-fold feasibility model for one fixed pair universe."""
    (
        antibodies,
        antigens,
        antibody_closed,
        antigen_closed,
        antibody_components,
        antigen_components,
    ) = njm._validate_problem(problem)
    njm._check_component_gate_capacity(
        policy,
        antibody_component_count=len(antibody_components),
        antigen_component_count=len(antigen_components),
        level=level_label,
    )
    pairs = tuple(sorted(problem.pairs, key=lambda value: value.pair_id))
    lower, upper = njm.validation_pair_interval(policy, len(pairs))
    if lower > upper:
        raise njm.PolicyArithmeticError(
            f"{level_label} validation interval infeasible: lower={lower}, upper={upper}"
        )

    incident_by_antibody = {
        entity: tuple(i for i, pair in enumerate(pairs) if pair.antibody_id == entity)
        for entity in antibodies
    }
    incident_by_antigen = {
        entity: tuple(i for i, pair in enumerate(pairs) if pair.antigen_id == entity)
        for entity in antigens
    }
    builder = njm._ModelBuilder()

    def v(*key: object) -> int:
        return builder.variable(level_label, *key)

    for fold in range(3):
        for entity in antibodies:
            for role in ("holdout_a", "closed_a", "train_present_a"):
                v(fold, role, entity)
        for entity in antigens:
            for role in ("holdout_g", "closed_g", "train_present_g"):
                v(fold, role, entity)
        for component in antibody_components:
            v(fold, "validation_component_a", component[0])
        for component in antigen_components:
            v(fold, "validation_component_g", component[0])
        for pair in pairs:
            for role in ("validation_pair", "training_pair", "quarantine_pair"):
                v(fold, role, pair.pair_id)

    validation_pairs: list[list[int]] = []
    training_pairs: list[list[int]] = []
    quarantine_pairs: list[list[int]] = []
    holdout_antibodies: list[list[int]] = []
    holdout_antigens: list[list[int]] = []
    validation_antibody_components: list[list[int]] = []
    validation_antigen_components: list[list[int]] = []
    training_antibodies: list[list[int]] = []
    training_antigens: list[list[int]] = []

    for fold in range(3):
        for entity in antibodies:
            njm._add_exact_or(
                builder,
                v(fold, "closed_a", entity),
                [v(fold, "holdout_a", source) for source in antibody_closed[entity]],
            )
        for entity in antigens:
            njm._add_exact_or(
                builder,
                v(fold, "closed_g", entity),
                [v(fold, "holdout_g", source) for source in antigen_closed[entity]],
            )

        for pair in pairs:
            valid = v(fold, "validation_pair", pair.pair_id)
            train = v(fold, "training_pair", pair.pair_id)
            quarantine = v(fold, "quarantine_pair", pair.pair_id)
            njm._add_exact_and_two(
                builder,
                valid,
                v(fold, "holdout_a", pair.antibody_id),
                v(fold, "holdout_g", pair.antigen_id),
            )
            njm._add_training_and(
                builder,
                train,
                v(fold, "closed_a", pair.antibody_id),
                v(fold, "closed_g", pair.antigen_id),
            )
            builder.add_row(
                ((valid, 1), (train, 1), (quarantine, 1)),
                lower_bound=1,
                upper_bound=1,
            )

        for entity in antibodies:
            incident_train = [
                v(fold, "training_pair", pairs[i].pair_id)
                for i in incident_by_antibody[entity]
            ]
            njm._add_endpoint_presence(
                builder, v(fold, "train_present_a", entity), incident_train
            )
            incident_valid = [
                v(fold, "validation_pair", pairs[i].pair_id)
                for i in incident_by_antibody[entity]
            ]
            builder.add_row(
                [
                    (v(fold, "holdout_a", entity), 1),
                    *((value, -1) for value in incident_valid),
                ],
                upper_bound=0,
            )
        for entity in antigens:
            incident_train = [
                v(fold, "training_pair", pairs[i].pair_id)
                for i in incident_by_antigen[entity]
            ]
            njm._add_endpoint_presence(
                builder, v(fold, "train_present_g", entity), incident_train
            )
            incident_valid = [
                v(fold, "validation_pair", pairs[i].pair_id)
                for i in incident_by_antigen[entity]
            ]
            builder.add_row(
                [
                    (v(fold, "holdout_g", entity), 1),
                    *((value, -1) for value in incident_valid),
                ],
                upper_bound=0,
            )

        component_vars_a: list[int] = []
        for component in antibody_components:
            presence = v(fold, "validation_component_a", component[0])
            njm._add_exact_or(
                builder,
                presence,
                [v(fold, "holdout_a", endpoint) for endpoint in component],
            )
            component_vars_a.append(presence)
        component_vars_g: list[int] = []
        for component in antigen_components:
            presence = v(fold, "validation_component_g", component[0])
            njm._add_exact_or(
                builder,
                presence,
                [v(fold, "holdout_g", endpoint) for endpoint in component],
            )
            component_vars_g.append(presence)

        validation_pairs.append([v(fold, "validation_pair", p.pair_id) for p in pairs])
        training_pairs.append([v(fold, "training_pair", p.pair_id) for p in pairs])
        quarantine_pairs.append([v(fold, "quarantine_pair", p.pair_id) for p in pairs])
        holdout_antibodies.append([v(fold, "holdout_a", x) for x in antibodies])
        holdout_antigens.append([v(fold, "holdout_g", x) for x in antigens])
        validation_antibody_components.append(component_vars_a)
        validation_antigen_components.append(component_vars_g)
        training_antibodies.append([v(fold, "train_present_a", x) for x in antibodies])
        training_antigens.append([v(fold, "train_present_g", x) for x in antigens])

    # A held-out entity may belong to at most one validation fold at a level.
    for entity in antibodies:
        builder.add_row(
            ((v(fold, "holdout_a", entity), 1) for fold in range(3)),
            upper_bound=1,
        )
    for entity in antigens:
        builder.add_row(
            ((v(fold, "holdout_g", entity), 1) for fold in range(3)),
            upper_bound=1,
        )

    njm._add_fixed_level_policy(
        builder,
        policy=policy,
        validation_pairs=validation_pairs,
        validation_antibodies=holdout_antibodies,
        validation_antigens=holdout_antigens,
        validation_antibody_components=validation_antibody_components,
        validation_antigen_components=validation_antigen_components,
        training_pairs=training_pairs,
        training_antibodies=training_antibodies,
        training_antigens=training_antigens,
        quarantine_pairs=quarantine_pairs,
        pair_count=len(pairs),
        antibody_count=len(antibodies),
        antigen_count=len(antigens),
    )

    rows = tuple(builder.rows)
    keys = tuple(builder.variable_keys)
    nonzeros = sum(len(row.coefficients) for row in rows)
    directed_closed = sum(len(v) for v in antibody_closed.values()) + sum(
        len(v) for v in antigen_closed.values()
    )
    estimate = njm.ModelSizeEstimate(
        pair_count=len(pairs),
        antibody_count=len(antibodies),
        antigen_count=len(antigens),
        antibody_component_count=len(antibody_components),
        antigen_component_count=len(antigen_components),
        antibody_undirected_edge_count=_edge_count(problem.antibody_direct_neighbors),
        antigen_undirected_edge_count=_edge_count(problem.antigen_direct_neighbors),
        directed_closed_relation_count=directed_closed,
        binary_variable_count=len(keys),
        linear_constraint_count=len(rows),
        nonzero_coefficient_count=nonzeros,
    )
    return njm.JointMILPModel(
        variable_keys=keys,
        rows=rows,
        estimate=estimate,
        matrix_sha256=njm._matrix_sha256(keys, rows),
    )


def solve_level(
    problem: njm.JointSplitProblem,
    *,
    policy: njm.JointLevelPolicy,
    level_label: str,
    time_limit_seconds: float,
) -> LevelSolve:
    """Solve one level and return exact held-out endpoint sets if certified."""
    model = build_level_feasibility_model(problem, policy=policy, level_label=level_label)
    result = njm.solve_joint_feasibility(model, time_limit_seconds=time_limit_seconds)
    if not result.feasible_for_unscored_split_construction or result.assignment is None:
        return LevelSolve(result, None, None)
    held_a = []
    held_g = []
    for fold in range(3):
        held_a.append(
            frozenset(
                key[-1]
                for key, value in result.assignment.items()
                if value == 1
                and len(key) == 4
                and key[:3] == (level_label, str(fold), "holdout_a")
            )
        )
        held_g.append(
            frozenset(
                key[-1]
                for key, value in result.assignment.items()
                if value == 1
                and len(key) == 4
                and key[:3] == (level_label, str(fold), "holdout_g")
            )
        )
    return LevelSolve(
        result,
        tuple(held_a),  # type: ignore[arg-type]
        tuple(held_g),  # type: ignore[arg-type]
    )
