from __future__ import annotations

from interaction_first.decomposed_level_milp import build_level_feasibility_model, solve_level
from interaction_first.nested_joint_milp import CanonicalPair, JointLevelPolicy, JointSplitProblem, Ratio


def policy(*, min_pairs=1, min_components=1):
    return JointLevelPolicy(
        minimum_validation_pairs=min_pairs,
        minimum_validation_endpoints_per_axis=1,
        minimum_validation_components_per_axis=min_components,
        validation_pair_lower_fraction=Ratio(1, 9),
        validation_pair_upper_fraction=Ratio(1, 3),
        minimum_training_pair_fraction=Ratio(1, 2),
        minimum_training_endpoint_fraction=Ratio(1, 2),
        maximum_quarantine_pair_fraction=Ratio(1, 2),
        maximum_validation_pair_relative_range=Ratio(1, 1),
        maximum_other_count_relative_range=Ratio(1, 1),
    )


def independent(n=9):
    return JointSplitProblem(
        pairs=tuple(CanonicalPair(f'p{i}',f'a{i}',f'g{i}') for i in range(n)),
        antibody_direct_neighbors={f'a{i}':frozenset() for i in range(n)},
        antigen_direct_neighbors={f'g{i}':frozenset() for i in range(n)},
    )


def test_model_is_order_stable():
    p=independent()
    m1=build_level_feasibility_model(p,policy=policy(),level_label='outer')
    p2=JointSplitProblem(tuple(reversed(p.pairs)),dict(reversed(tuple(p.antibody_direct_neighbors.items()))),dict(reversed(tuple(p.antigen_direct_neighbors.items()))))
    m2=build_level_feasibility_model(p2,policy=policy(),level_label='outer')
    assert m1.variable_keys==m2.variable_keys
    assert m1.matrix_sha256==m2.matrix_sha256
    assert m1.estimate.binary_variable_count < 1188  # strictly smaller than 3+9 joint model


def test_feasible_level_returns_disjoint_holdouts():
    solved=solve_level(independent(),policy=policy(),level_label='outer',time_limit_seconds=30)
    assert solved.result.feasible_for_unscored_split_construction
    assert solved.held_out_antibodies is not None and solved.held_out_antigens is not None
    assert all(solved.held_out_antibodies) and all(solved.held_out_antigens)
    assert sum(map(len,solved.held_out_antibodies))==len(set().union(*solved.held_out_antibodies))
    assert sum(map(len,solved.held_out_antigens))==len(set().union(*solved.held_out_antigens))


def test_component_gate_can_make_problem_infeasible():
    # One connected component on each axis cannot supply one distinct component to every fold.
    n=9
    graph_a={f'a{i}':frozenset({f'a{j}' for j in range(n) if j!=i}) for i in range(n)}
    graph_g={f'g{i}':frozenset({f'g{j}' for j in range(n) if j!=i}) for i in range(n)}
    p=JointSplitProblem(tuple(CanonicalPair(f'p{i}',f'a{i}',f'g{i}') for i in range(n)),graph_a,graph_g)
    solved=solve_level(p,policy=policy(min_components=1),level_label='outer',time_limit_seconds=30)
    assert not solved.result.feasible_for_unscored_split_construction
    assert solved.held_out_antibodies is None
