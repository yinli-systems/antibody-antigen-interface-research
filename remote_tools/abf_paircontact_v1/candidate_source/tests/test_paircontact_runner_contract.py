from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path

import torch

from antibody_pretraining.stage_a1_batch import build_stage_a1_microbatch


def _runner_module():
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_abf_paircontact_residual_canary.py"
    )
    spec = importlib.util.spec_from_file_location("paircontact_runner", runner)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_microbatch_calls_match_frozen_batch_api() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_abf_paircontact_residual_canary.py"
    )
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    accepted = set(inspect.signature(build_stage_a1_microbatch).parameters)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_stage_a1_microbatch"
    ]
    assert calls
    for call in calls:
        supplied = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
        assert supplied <= accepted


def test_average_precision_is_tie_aware_and_permutation_invariant() -> None:
    runner = _runner_module()
    constant = runner._binary_average_precision(
        torch.zeros(5), torch.tensor([True, False, True, False, False])
    )
    assert constant == 0.4

    scores = torch.tensor([0.9, 0.5, 0.5, 0.1])
    first = runner._binary_average_precision(
        scores, torch.tensor([True, True, False, False])
    )
    second = runner._binary_average_precision(
        scores, torch.tensor([True, False, True, False])
    )
    assert first == second
    assert abs(first - (5.0 / 6.0)) < 1e-12


def test_holdout_can_exclude_multiple_prior_pdb_inventories() -> None:
    runner = _runner_module()
    records = [
        {"example_id": f"{pdb}_pair", "pilot_record_index": index}
        for index, pdb in enumerate(("1aaa", "2bbb", "3ccc", "4ddd", "5eee", "6fff"))
    ]

    def inventory(seed: int, salt: str, candidates: list[str]) -> tuple[str, str]:
        selected = min(
            candidates,
            key=lambda pdb: hashlib.sha256(
                f"{seed}\0{salt}\0{pdb}".encode("utf-8")
            ).digest(),
        )
        digest = hashlib.sha256(f"{selected}\n".encode("utf-8")).hexdigest()
        return selected, digest

    candidates = [record["example_id"].split("_", 1)[0] for record in records]
    first, first_sha = inventory(11, "first", candidates)
    second_seed = next(
        seed
        for seed in range(13, 100)
        if inventory(seed, "second", candidates)[0] != first
    )
    second, second_sha = inventory(second_seed, "second", candidates)
    remaining = [p for p in candidates if p not in {first, second}]
    selected, selected_sha = inventory(17, "current", remaining)
    protocol = {
        "internal_holdout": {
            "excluded_pdb_sets": [
                {
                    "expected_pdb_count": 1,
                    "ordering_salt": "first",
                    "pdb_inventory_sha256": first_sha,
                    "selection_seed": 11,
                    "target_records": 1,
                },
                {
                    "expected_pdb_count": 1,
                    "ordering_salt": "second",
                    "pdb_inventory_sha256": second_sha,
                    "selection_seed": second_seed,
                    "target_records": 1,
                },
            ],
            "expected_pdb_count": 1,
            "grouping": "lowercase_pdb_prefix",
            "ordering_salt": "current",
            "pdb_inventory_sha256": selected_sha,
            "selection_seed": 17,
            "target_records": 1,
            "test_access": False,
        }
    }
    training, holdout, receipt = runner._geometry_ranking_holdout(records, protocol)
    assert {row["example_id"].split("_", 1)[0] for row in holdout} == {selected}
    assert len(training) == len(records) - 1
    assert receipt["excluded_pdb_count"] == 2


def test_later_excluded_inventory_is_built_after_prior_exclusions() -> None:
    runner = _runner_module()
    records = [
        {"example_id": f"{pdb}_pair", "pilot_record_index": index}
        for index, pdb in enumerate(("1aaa", "2bbb", "3ccc", "4ddd", "5eee", "6fff"))
    ]

    def ordered(seed: int, salt: str, candidates: list[str]) -> list[str]:
        return sorted(
            candidates,
            key=lambda pdb: hashlib.sha256(
                f"{seed}\0{salt}\0{pdb}".encode("utf-8")
            ).digest(),
        )

    candidates = [record["example_id"].split("_", 1)[0] for record in records]
    first = ordered(11, "first", candidates)[0]
    first_sha = hashlib.sha256(f"{first}\n".encode("utf-8")).hexdigest()
    remaining_after_first = [pdb for pdb in candidates if pdb != first]
    second = ordered(13, "second", remaining_after_first)[0]
    second_sha = hashlib.sha256(f"{second}\n".encode("utf-8")).hexdigest()
    remaining = [pdb for pdb in candidates if pdb not in {first, second}]
    selected = ordered(17, "current", remaining)[0]
    selected_sha = hashlib.sha256(f"{selected}\n".encode("utf-8")).hexdigest()
    protocol = {
        "internal_holdout": {
            "excluded_pdb_sets": [
                {
                    "expected_pdb_count": 1,
                    "ordering_salt": "first",
                    "pdb_inventory_sha256": first_sha,
                    "selection_seed": 11,
                    "target_records": 1,
                },
                {
                    "expected_pdb_count": 1,
                    "ordering_salt": "second",
                    "pdb_inventory_sha256": second_sha,
                    "selection_seed": 13,
                    "target_records": 1,
                },
            ],
            "expected_pdb_count": 1,
            "grouping": "lowercase_pdb_prefix",
            "ordering_salt": "current",
            "pdb_inventory_sha256": selected_sha,
            "selection_seed": 17,
            "target_records": 1,
            "test_access": False,
        }
    }
    _, holdout, receipt = runner._geometry_ranking_holdout(records, protocol)
    assert {row["example_id"].split("_", 1)[0] for row in holdout} == {selected}
    assert receipt["excluded_pdb_count"] == 2
