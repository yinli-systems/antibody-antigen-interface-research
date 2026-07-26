"""Tests for V6-B IR partitioning and telemetry-only row shards."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from ortools.sat.python import cp_model

from interaction_first.v6_b_decomposition import (
    SignedSupportLiteral,
    V6BDecompositionError,
    build_exact_no_good_cuts,
    build_replayed_master_no_good_cut,
    build_row_shard_telemetry,
    partition_joint_ir,
    replay_inner_infeasibility_support,
    run_v6_b_exact_feasibility,
    selected_row_hashes,
    solve_inner_linear_projection,
    solve_outer_master_linear_projection,
)
from interaction_first.v6_joint_ir import (
    CanonicalJointIR,
    IRLinearConstraint,
    SeparationFailure,
    SeparationOracleResult,
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


def _infeasible_inner_ir() -> CanonicalJointIR:
    outer = tuple(("outer", str(group), "training_pair", f"p{group}") for group in range(3))
    inner = tuple(
        ("inner", str(group), "0", "training_pair", f"p{group}")
        for group in range(3)
    )
    rows = [_row("000_master", outer)]
    for group in range(3):
        x = outer[group]
        y = inner[group]
        rows.append(
            IRLinearConstraint(
                name=f"100_link_{group}",
                coefficients=tuple(sorted(((x, -1), (y, 1)))),
                lower_bound=0,
                upper_bound=None,
            )
        )
        rows.append(
            IRLinearConstraint(
                name=f"101_zero_{group}",
                coefficients=((y, 1),),
                lower_bound=None,
                upper_bound=0,
            )
        )
    return CanonicalJointIR(
        variable_keys=tuple(sorted(outer + inner)),
        linear_constraints=tuple(sorted(rows, key=lambda row: row.name)),
        separation_gates=separation_gate_registry("0" * 64),
        source_bindings=(("synthetic", "1" * 64),),
        source_linear_projection_sha256="2" * 64,
        construction_scope="synthetic_v6_b_core_replay_test",
        independent_raw_authority_reconstruction=True,
    )


def _assignment(partition, values: tuple[int, int, int]) -> dict[tuple[str, ...], int]:
    return {
        key: values[int(key[1])] for key in partition.master_variables
    }


def _independent_receipt(support: tuple[SignedSupportLiteral, ...]) -> dict[str, object]:
    payload: dict[str, object] = {
        "format_version": "v6_b_independent_inner_core_replay_v1",
        "outer_fold": 0,
        "support": [literal.canonical_payload() for literal in support],
        "status": "INFEASIBLE",
        "replay_infeasible": True,
        "accepted_terminal_status": True,
        "v6_b_decomposition_imported": False,
        "historical_milp_or_highs_artifact_consumed": False,
        "training_authorized": False,
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    payload["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def test_inner_infeasibility_core_replays_and_generates_exact_no_good() -> None:
    partition = partition_joint_ir(_infeasible_inner_ir())
    first = solve_inner_linear_projection(
        partition,
        outer_fold=0,
        outer_assignment=_assignment(partition, (1, 0, 0)),
    )
    assert first["status"] == "INFEASIBLE"
    assert first["sufficient_assumption_support"] == [
        [["outer", "0", "training_pair", "p0"], 1]
    ]
    support = (SignedSupportLiteral(("outer", "0", "training_pair", "p0"), 1),)
    replay = replay_inner_infeasibility_support(
        partition,
        outer_fold=0,
        support=support,
    )
    assert replay["status"] == "INFEASIBLE"
    assert replay["replay_infeasible"] is True
    cut = build_replayed_master_no_good_cut(
        outer_fold=0,
        support=support,
        first_response_sha256=first["response"]["response_sha256"],
        replay_response_sha256=replay["response"]["response_sha256"],
        independent_replay_receipt=_independent_receipt(support),
    )
    master = solve_outer_master_linear_projection(partition, cuts=(cut,))
    assert master["status"] == "OPTIMAL"
    solved = {tuple(key): value for key, value in master["assignment"]}
    assert solved[("outer", "0", "training_pair", "p0")] == 0


def test_inner_projection_is_optimal_when_boundary_state_is_feasible() -> None:
    partition = partition_joint_ir(_infeasible_inner_ir())
    result = solve_inner_linear_projection(
        partition,
        outer_fold=0,
        outer_assignment=_assignment(partition, (0, 0, 0)),
    )
    assert result["status"] == "OPTIMAL"
    assert result["candidate_present"] is True
    assert result["sufficient_assumption_support"] == []
    assert result["complete_score_free_feasible_region_authorized"] is False


def test_support_cut_differs_only_on_its_signed_literal() -> None:
    partition = partition_joint_ir(_infeasible_inner_ir())
    support = (SignedSupportLiteral(("outer", "0", "training_pair", "p0"), 1),)
    cut = build_replayed_master_no_good_cut(
        outer_fold=0,
        support=support,
        first_response_sha256="1" * 64,
        replay_response_sha256="2" * 64,
        independent_replay_receipt=_independent_receipt(support),
    )
    assert cut.support == support
    assert cut.cut_sha256 == build_replayed_master_no_good_cut(
        outer_fold=0,
        support=support,
        first_response_sha256="1" * 64,
        replay_response_sha256="2" * 64,
        independent_replay_receipt=_independent_receipt(support),
    ).cut_sha256


def test_master_cut_rejects_a_tampered_independent_replay() -> None:
    support = (SignedSupportLiteral(("outer", "0", "training_pair", "p0"), 1),)
    independent = _independent_receipt(support)
    independent["replay_infeasible"] = False
    try:
        build_replayed_master_no_good_cut(
            outer_fold=0,
            support=support,
            first_response_sha256="1" * 64,
            replay_response_sha256="2" * 64,
            independent_replay_receipt=independent,
        )
    except V6BDecompositionError as error:
        assert "cannot support" in str(error)
    else:
        raise AssertionError("tampered independent replay must fail closed")


def test_all_eight_outer_states_match_the_three_inner_linear_truths() -> None:
    partition = partition_joint_ir(_infeasible_inner_ir())
    observed = []
    for bits in (
        (a, b, c)
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    ):
        statuses = tuple(
            solve_inner_linear_projection(
                partition,
                outer_fold=group,
                outer_assignment=_assignment(partition, bits),
            )["status"]
            for group in range(3)
        )
        expected = tuple("OPTIMAL" if bit == 0 else "INFEASIBLE" for bit in bits)
        assert statuses == expected
        observed.append((bits, statuses))
    assert len(observed) == 8


def test_exact_oracle_failures_are_partitioned_without_scope_leakage() -> None:
    outer_key = ("outer", "0", "training_pair", "p0")
    inner_key = ("inner", "0", "0", "training_pair", "p0")
    failures = (
        SeparationFailure(
            gate_id="outer_gate",
            level_id="outer_0",
            axis="both",
            role="fold",
            support=((outer_key, 0),),
            failure_payload={"passed": False},
        ),
        SeparationFailure(
            gate_id="inner_gate",
            level_id="outer_0_inner_0",
            axis="both",
            role="fold",
            support=tuple(sorted(((outer_key, 0), (inner_key, 0)))),
            failure_payload={"passed": False},
        ),
    )
    signatures = tuple(
        sorted((item.gate_id, item.level_id, item.axis, item.role) for item in failures)
    )
    result = SeparationOracleResult(
        evaluated_gates=signatures,
        failures=failures,
        primary_statistics_receipt_sha256="1" * 64,
        independent_audit_receipt_sha256="2" * 64,
        independent_agreement=True,
    )
    cuts = build_exact_no_good_cuts(result)
    assert [(cut.scope, cut.outer_fold) for cut in cuts] == [
        ("outer", None),
        ("inner", 0),
    ]
    partition = partition_joint_ir(_infeasible_inner_ir())
    inner_result = solve_inner_linear_projection(
        partition,
        outer_fold=0,
        outer_assignment=_assignment(partition, (0, 0, 0)),
        exact_cuts=(cuts[1],),
    )
    assert inner_result["status"] == "INFEASIBLE"
    assert inner_result["exact_cut_sha256"] == [cuts[1].cut_sha256]


def test_exact_oracle_disagreement_cannot_create_a_cut() -> None:
    result = SeparationOracleResult(
        evaluated_gates=(),
        failures=(),
        primary_statistics_receipt_sha256="1" * 64,
        independent_audit_receipt_sha256="2" * 64,
        independent_agreement=False,
    )
    try:
        build_exact_no_good_cuts(result)
    except V6BDecompositionError as error:
        assert "disagreement" in str(error)
    else:
        raise AssertionError("oracle disagreement must fail closed")


def _all_gate_signatures(ir: CanonicalJointIR) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (gate.gate_id, gate.level_id, gate.axis, gate.role)
        for gate in ir.separation_gates
    )


def test_complete_decomposition_accepts_only_all_66_exact_gates() -> None:
    ir = _ir()
    partition = partition_joint_ir(ir)

    def oracle(assignment):
        assert set(assignment) == set(ir.variable_keys)
        return SeparationOracleResult(
            evaluated_gates=_all_gate_signatures(ir),
            failures=(),
            primary_statistics_receipt_sha256="1" * 64,
            independent_audit_receipt_sha256="2" * 64,
            independent_agreement=True,
        )

    receipt = run_v6_b_exact_feasibility(
        partition,
        object(),
        oracle,
        maximum_master_iterations=2,
        maximum_combined_candidates=2,
    )
    assert receipt["terminal_status"] == "OPTIMAL_EXACT_ORACLE_PASS"
    assert receipt["complete"] is True
    assert receipt["candidate_present"] is True
    assert receipt["all_66_exact_gates_passed"] is True
    assert receipt["split_gate0_authorized"] is False
    assert receipt["training_authorized"] is False


def test_outer_exact_failure_is_cut_before_second_candidate_passes() -> None:
    ir = _ir()
    partition = partition_joint_ir(ir)
    calls = []

    def oracle(assignment):
        calls.append(dict(assignment))
        failures = ()
        if len(calls) == 1:
            key = ("outer", "0", "training_pair", "p0")
            failures = (
                SeparationFailure(
                    gate_id="panel_shape_and_chance_mrr",
                    level_id="outer_0",
                    axis="both",
                    role="fold",
                    support=((key, assignment[key]),),
                    failure_payload={"passed": False},
                ),
            )
        return SeparationOracleResult(
            evaluated_gates=_all_gate_signatures(ir),
            failures=failures,
            primary_statistics_receipt_sha256="1" * 64,
            independent_audit_receipt_sha256="2" * 64,
            independent_agreement=True,
        )

    receipt = run_v6_b_exact_feasibility(
        partition,
        object(),
        oracle,
        maximum_master_iterations=3,
        maximum_combined_candidates=3,
    )
    assert receipt["terminal_status"] == "OPTIMAL_EXACT_ORACLE_PASS"
    assert len(calls) == 2
    key = ("outer", "0", "training_pair", "p0")
    assert calls[0][key] != calls[1][key]
    assert len(receipt["master_cuts"]) == 1


def test_incomplete_exact_audit_stops_without_candidate() -> None:
    ir = _ir()
    partition = partition_joint_ir(ir)

    def oracle(_assignment):
        return SeparationOracleResult(
            evaluated_gates=(),
            failures=(),
            primary_statistics_receipt_sha256="1" * 64,
            independent_audit_receipt_sha256="2" * 64,
            independent_agreement=True,
        )

    receipt = run_v6_b_exact_feasibility(
        partition,
        object(),
        oracle,
        maximum_master_iterations=2,
        maximum_combined_candidates=2,
    )
    assert receipt["terminal_status"] == "INCOMPLETE_EXACT_GATE_AUDIT"
    assert receipt["complete"] is False
    assert receipt["candidate_present"] is False


def test_inner_exact_failure_is_locally_cut_without_changing_outer_state() -> None:
    ir = _ir()
    partition = partition_joint_ir(ir)
    calls = []

    def oracle(assignment):
        calls.append(dict(assignment))
        failures = ()
        if len(calls) == 1:
            keys = (
                ("inner", "0", "0", "training_pair", "p0"),
                ("outer", "0", "training_pair", "p0"),
            )
            failures = (
                SeparationFailure(
                    gate_id="panel_shape_and_chance_mrr",
                    level_id="outer_0_inner_0",
                    axis="both",
                    role="fold",
                    support=tuple((key, assignment[key]) for key in keys),
                    failure_payload={"passed": False},
                ),
            )
        return SeparationOracleResult(
            evaluated_gates=_all_gate_signatures(ir),
            failures=failures,
            primary_statistics_receipt_sha256="1" * 64,
            independent_audit_receipt_sha256="2" * 64,
            independent_agreement=True,
        )

    receipt = run_v6_b_exact_feasibility(
        partition,
        object(),
        oracle,
        maximum_master_iterations=2,
        maximum_combined_candidates=3,
    )
    assert receipt["terminal_status"] == "OPTIMAL_EXACT_ORACLE_PASS"
    assert len(calls) == 2
    outer_key = ("outer", "0", "training_pair", "p0")
    inner_key = ("inner", "0", "0", "training_pair", "p0")
    assert calls[0][outer_key] == calls[1][outer_key]
    assert calls[0][inner_key] != calls[1][inner_key]
    assert len(receipt["inner_exact_cuts"]["0"]) == 1
    assert receipt["master_cuts"] == []


def _direct_core_model() -> object:
    model = cp_model.CpModel()
    variables = {}
    for group in range(3):
        x_key = ("outer", str(group), "training_pair", f"p{group}")
        y_key = ("inner", str(group), "0", "training_pair", f"p{group}")
        x = model.new_bool_var(f"x{group}")
        y = model.new_bool_var(f"y{group}")
        variables[x_key] = x
        variables[y_key] = y
        model.add(y >= x)
        model.add(y <= 0)
    return SimpleNamespace(
        model=model,
        variables=variables,
        formulation_sha256="1" * 64,
        authority_payload_sha256="2" * 64,
    )


def test_replayed_inner_linear_core_can_prove_master_infeasible() -> None:
    base = _infeasible_inner_ir()
    forced = IRLinearConstraint(
        name="001_force_outer_zero_group_to_one",
        coefficients=((('outer', '0', 'training_pair', 'p0'), 1),),
        lower_bound=1,
        upper_bound=1,
    )
    ir = CanonicalJointIR(
        variable_keys=base.variable_keys,
        linear_constraints=tuple(
            sorted(base.linear_constraints + (forced,), key=lambda row: row.name)
        ),
        separation_gates=base.separation_gates,
        source_bindings=base.source_bindings,
        source_linear_projection_sha256=base.source_linear_projection_sha256,
        construction_scope=base.construction_scope,
        independent_raw_authority_reconstruction=True,
    )

    def oracle(_assignment):
        raise AssertionError("linear infeasibility must be cut before exact audit")

    receipt = run_v6_b_exact_feasibility(
        partition_joint_ir(ir),
        _direct_core_model(),
        oracle,
        maximum_master_iterations=3,
        maximum_combined_candidates=3,
    )
    assert receipt["terminal_status"] == "PROVEN_INFEASIBLE"
    assert receipt["complete"] is True
    assert receipt["proven_infeasible"] is True
    assert receipt["candidate_present"] is False
    assert len(receipt["master_cuts"]) == 1
    replay_events = [event for event in receipt["inner_history"] if "master_cut" in event]
    assert len(replay_events) == 1
    assert replay_events[0]["primary_replay"]["replay_infeasible"] is True
    assert replay_events[0]["independent_replay"]["replay_infeasible"] is True
