#!/usr/bin/env python3
"""Tests for the train-only approximate S2 MILP diagnostic."""

from __future__ import annotations

import unittest

from optimize_train_only_s2_feasibility import (
    build_problem,
    optimize,
    summarize_solution,
)


class TrainOnlyS2OptimizerTests(unittest.TestCase):
    """Verify lexicographic retention and two-axis balance on small graphs."""

    def test_three_disjoint_pairs_are_retained_and_balanced(self) -> None:
        records = [
            ("r0", "a0", "g0"),
            ("r1", "a1", "g1"),
            ("r2", "a2", "g2"),
        ]
        problem = build_problem(
            records,
            {"a0": "ac0", "a1": "ac1", "a2": "ac2"},
            {"g0": "gc0", "g1": "gc1", "g2": "gc2"},
        )
        solution, stages = optimize(problem, 3, 0.0, 30.0)
        summary = summarize_solution(problem, solution, 3, 0.0)
        self.assertEqual(summary["retained_unique_exact_pair_count"], 3)
        self.assertEqual(summary["discarded_unique_exact_pair_count"], 0)
        self.assertEqual(summary["retained_unique_exact_pair_count_range"], 0)
        self.assertEqual(summary["retained_source_row_count_range"], 0)
        self.assertEqual(stages["stage_1"]["mip_gap"], 0.0)

    def test_duplicate_rows_do_not_change_primary_pair_objective(self) -> None:
        records = [
            ("r0", "a0", "g0"),
            ("r0_dup", "a0", "g0"),
            ("r1", "a1", "g1"),
            ("r2", "a2", "g2"),
        ]
        problem = build_problem(
            records,
            {"a0": "ac0", "a1": "ac1", "a2": "ac2"},
            {"g0": "gc0", "g1": "gc1", "g2": "gc2"},
        )
        self.assertEqual(problem.total_unique_pairs, 3)
        self.assertEqual(problem.total_source_rows, 4)
        solution, _ = optimize(problem, 3, 0.0, 30.0)
        summary = summarize_solution(problem, solution, 3, 0.0)
        self.assertEqual(summary["retained_unique_exact_pair_count"], 3)
        self.assertEqual(summary["retained_source_row_count"], 4)


if __name__ == "__main__":
    unittest.main()
