"""Regression test for the sensitivity bootstrap in the evaluate path.

``component_by_instance`` is built from the whole split file while the deltas
cover only the evaluated instances, so the cluster mapping is a strict superset.
``cluster_bootstrap_mean_ci`` requires an exact match, so passing it unrestricted
raised and failed the evaluation *after* the prediction parquets had been
written -- which is how both arms of the decoder ablation exited non-zero while
still producing complete, usable outputs.
"""

from __future__ import annotations

import pytest

from interaction_first.metrics import cluster_bootstrap_mean_ci


def _restricted(deltas: dict[str, float], components: dict[str, str]):
    """The fix applied in command_evaluate, isolated for testing."""
    evaluated = set(deltas)
    missing = sorted(evaluated - set(components))
    if missing:
        return None, f"{len(missing)} evaluated instance(s) have no component assignment"
    return {instance: components[instance] for instance in evaluated}, None


def test_a_superset_cluster_mapping_is_rejected_unrestricted() -> None:
    """The failure this regression test exists for."""
    deltas = {"a": 0.1, "b": -0.2}
    components = {"a": "c1", "b": "c1", "c": "c2", "d": "c3"}  # covers the whole split

    with pytest.raises(ValueError, match="must exactly match"):
        cluster_bootstrap_mean_ci(deltas, components, iterations=32, seed=0)


def test_restricting_to_the_evaluated_instances_succeeds() -> None:
    deltas = {"a": 0.1, "b": -0.2, "e": 0.3}
    components = {"a": "c1", "b": "c1", "c": "c2", "d": "c3", "e": "c2"}

    restricted, skipped = _restricted(deltas, components)
    result = cluster_bootstrap_mean_ci(deltas, restricted, iterations=32, seed=0)

    assert skipped is None
    assert set(restricted) == set(deltas)
    assert isinstance(result, dict) and result


def test_restriction_preserves_every_evaluated_instance() -> None:
    deltas = {f"i{n}": 0.01 * n for n in range(50)}
    components = {f"i{n}": f"c{n % 7}" for n in range(200)}  # far larger split

    restricted, skipped = _restricted(deltas, components)

    assert skipped is None
    assert set(restricted) == set(deltas)
    assert all(restricted[key] == components[key] for key in restricted)


def test_an_evaluated_instance_without_a_component_skips_rather_than_raises() -> None:
    """A genuine gap must be reported, not silently dropped and not fatal."""
    deltas = {"a": 0.1, "orphan": 0.2}
    components = {"a": "c1", "c": "c2"}

    restricted, skipped = _restricted(deltas, components)

    assert restricted is None
    assert "1 evaluated instance(s) have no component assignment" == skipped


def test_the_bootstrap_still_runs_on_the_restricted_mapping() -> None:
    deltas = {"a": 0.5, "b": 0.4, "c": -0.1, "d": 0.2}
    components = {"a": "c1", "b": "c1", "c": "c2", "d": "c3", "unused": "c9"}

    restricted, _ = _restricted(deltas, components)
    result = cluster_bootstrap_mean_ci(deltas, restricted, iterations=64, seed=1)

    assert isinstance(result, dict)
    assert result
