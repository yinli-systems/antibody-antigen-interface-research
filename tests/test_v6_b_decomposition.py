"""Tests for V6-B IR partitioning and telemetry-only row shards."""

from __future__ import annotations

from interaction_first.v6_b_decomposition import (
    V6BDecompositionError,
    build_row_shard_telemetry,
    partition_joint_ir,
    selected_row_hashes,
)
from interaction_first.v6_joint_ir import (
    CanonicalJointIR,
    IRLinearConstraint,
    separation_gate_registry,
)


def _row(name: str, keys: tuple[tuple[str, ...], ...]) -> IRLinearConstraint:
    return IRLinearConstraint(
        name=name,
        coefficients=tuple((key, 1) for key in sorted(keys)),
        lower_bound=0,
        upper_bound=1,
    )


def _ir() -> CanonicalJointIR:
    outer = tuple(("outer", str(group), "training_pair", f"p{group}") for group in range(3))
    inner = tuple(
        ("inner", str(group), str(fold), "training_pair", f"p{group}")
        for group in range(3)
        for fold in range(3)
    )
    rows = [_row("000_outer_cross", outer)]
    for group in range(3):
        for fold in range(3):
            rows.append(
                _row(
                    f"100_inner_{group}_{fold}",
                    (
                        outer[group],
                        ("inner", str(group), str(fold), "training_pair", f"p{group}"),
                    ),
                )
            )
    return CanonicalJointIR(
        variable_keys=tuple(sorted(outer + inner)),
        linear_constraints=tuple(sorted(rows, key=lambda row: row.name)),
        separation_gates=separation_gate_registry("0" * 64),
        source_bindings=(("synthetic", "1" * 64),),
        source_linear_projection_sha256="2" * 64,
        construction_scope="synthetic_v6_b_partition_test",
        independent_raw_authority_reconstruction=True,
    )


def test_partition_is_lossless_and_has_frozen_gate_counts() -> None:
    ir = _ir()
    partition = partition_joint_ir(ir)
    assert len(partition.master_rows) == 1
    assert len(partition.master_gates) == 18
    assert [len(rows) for _, rows in partition.inner_rows] == [3, 3, 3]
    assert [len(gates) for _, gates in partition.inner_gates] == [16, 16, 16]
    reconstructed = [row.name for row in partition.master_rows]
    reconstructed.extend(
        row.name for _, rows in partition.inner_rows for row in rows
    )
    assert sorted(reconstructed) == [row.name for row in ir.linear_constraints]
    assert partition.canonical_payload()["training_authorized"] is False


def test_partition_rejects_a_row_that_mixes_inner_groups() -> None:
    ir = _ir()
    bad = _row(
        "999_mixed",
        (
            ("inner", "0", "0", "training_pair", "p0"),
            ("inner", "1", "0", "training_pair", "p1"),
        ),
    )
    malformed = CanonicalJointIR(
        variable_keys=ir.variable_keys,
        linear_constraints=tuple(sorted(ir.linear_constraints + (bad,), key=lambda row: row.name)),
        separation_gates=ir.separation_gates,
        source_bindings=ir.source_bindings,
        source_linear_projection_sha256=ir.source_linear_projection_sha256,
        construction_scope=ir.construction_scope,
        independent_raw_authority_reconstruction=True,
    )
    try:
        partition_joint_ir(malformed)
    except V6BDecompositionError as error:
        assert "mixes independent inner groups" in str(error)
    else:
        raise AssertionError("mixed inner groups must fail closed")


def test_row_shard_is_deterministic_and_never_authorizes_science() -> None:
    ir = _ir()
    first = build_row_shard_telemetry(
        ir,
        serialized_full_model_bytes=b"full-model-proto",
        available_memory_bytes=10_000_000,
        observed_peak_rss_bytes=100_000,
    )
    second = build_row_shard_telemetry(
        ir,
        serialized_full_model_bytes=b"full-model-proto",
        available_memory_bytes=10_000_000,
        observed_peak_rss_bytes=100_000,
    )
    assert first == second
    assert selected_row_hashes(first) == selected_row_hashes(second)
    assert first["full_variable_count"] == len(ir.variable_keys)
    assert first["full_row_count"] == len(ir.linear_constraints)
    assert first["vertices_or_pairs_sampled"] is False
    assert first["solver_invoked"] is False
    assert first["feasibility_authorized"] is False
    assert first["training_authorized"] is False
    assert first["resource_headroom_passed"] is True


def test_row_shard_ceil_selects_at_least_one_row_per_family() -> None:
    telemetry = build_row_shard_telemetry(
        _ir(),
        serialized_full_model_bytes=b"full-model-proto",
        available_memory_bytes=10_000_000,
        observed_peak_rss_bytes=100_000,
    )
    families = telemetry["row_selection"]["families"]
    assert families
    assert all(family["selected_row_count"] >= 1 for family in families)


def test_resource_projection_requires_twenty_five_percent_headroom() -> None:
    telemetry = build_row_shard_telemetry(
        _ir(),
        serialized_full_model_bytes=b"x" * 100,
        available_memory_bytes=120_000,
        observed_peak_rss_bytes=100_000,
    )
    assert telemetry["projected_peak_bytes"] == 100_000
    assert telemetry["required_available_bytes_with_headroom"] == 133_334
    assert telemetry["resource_headroom_passed"] is False
