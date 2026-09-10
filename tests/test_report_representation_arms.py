"""Comparing several representation arms against a shared baseline arm.

The comparison that decides the headline is not the one against the narrow arm.
An arm with extra columns carries more parameters in its input projections, so a
gain over `esm2` cannot separate the content of those columns from their width.
Only the width-matched control isolates the content, and these tests hold the
arithmetic that makes both comparisons readable side by side.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
for extra in ("src", "scripts"):
    if str(REPOSITORY_ROOT / extra) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT / extra))
_SPEC = importlib.util.spec_from_file_location(
    "report_representation_arms_under_test",
    REPOSITORY_ROOT / "scripts" / "report_representation_arms.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

arm_receipts = _MODULE.arm_receipts
compare = _MODULE.compare
main = _MODULE.main


def _write_arm(root: Path, arm: str, values: dict[int, float], metric: str = "auprc") -> None:
    for seed, value in values.items():
        directory = root / arm / f"seed_{seed}" / "validation"
        directory.mkdir(parents=True)
        (directory / "evaluation.json").write_text(
            json.dumps({"pair": {"macro": {metric: value, "auroc": 0.9 + value}}}),
            encoding="utf-8",
        )


def test_receipts_are_read_per_seed(tmp_path: Path) -> None:
    _write_arm(tmp_path, "esm2", {1: 0.02, 2: 0.021, 7: 0.019})

    receipts = arm_receipts(tmp_path, "esm2")

    assert sorted(receipts) == [1, 2, 7]
    assert receipts[7]["auprc"] == 0.019


def test_a_seed_without_a_receipt_is_skipped(tmp_path: Path) -> None:
    """A run that failed at evaluate has a directory and no receipt."""
    _write_arm(tmp_path, "esm2", {1: 0.02, 2: 0.021})
    (tmp_path / "esm2" / "seed_3" / "validation").mkdir(parents=True)

    assert sorted(arm_receipts(tmp_path, "esm2")) == [1, 2]


def test_only_shared_seeds_are_paired(tmp_path: Path) -> None:
    base = {seed: 0.020 for seed in range(1, 6)}
    arm = {seed: 0.022 for seed in range(3, 9)}

    result = compare(
        {s: {"auprc": v} for s, v in base.items()},
        {s: {"auprc": v} for s, v in arm.items()},
        "auprc",
    )

    assert result["seed_count"] == 3
    assert [entry["seed"] for entry in result["per_seed"]] == [3, 4, 5]


def test_a_consistent_gain_is_decided(tmp_path: Path) -> None:
    base = {s: {"auprc": 0.020} for s in range(1, 11)}
    arm = {s: {"auprc": 0.020 + 0.002 + 0.0001 * s} for s in range(1, 11)}

    result = compare(base, arm, "auprc")

    assert result["favourable_seed_count"] == 10
    assert result["sign_lower_bound"] == pytest.approx(0.05 ** 0.1, rel=1e-6)
    assert result["decided"] is True
    assert result["bootstrap_ci95"][0] > 0


def test_a_null_arm_is_not_decided(tmp_path: Path) -> None:
    """The width-matched control is expected to land here if width does nothing."""
    base = {s: {"auprc": 0.020} for s in range(1, 11)}
    arm = {s: {"auprc": 0.020 + (0.0004 if s % 2 else -0.0004)} for s in range(1, 11)}

    result = compare(base, arm, "auprc")

    assert result["decided"] is False


def test_a_regression_is_decided_too(tmp_path: Path) -> None:
    """An arm that loses must be reported as decided, not as no decision."""
    base = {s: {"auprc": 0.020} for s in range(1, 11)}
    arm = {s: {"auprc": 0.020 - 0.002 - 0.0001 * s} for s in range(1, 11)}

    result = compare(base, arm, "auprc")

    assert result["decided"] is True
    assert result["bootstrap_ci95"][1] < 0
    assert result["favourable_seed_count"] == 0


def test_fewer_than_two_shared_seeds_yields_nothing(tmp_path: Path) -> None:
    result = compare({1: {"auprc": 0.02}}, {1: {"auprc": 0.03}}, "auprc")

    assert result is None


def test_a_metric_missing_from_one_arm_is_skipped(tmp_path: Path) -> None:
    base = {s: {"auprc": 0.02} for s in range(1, 5)}
    arm = {s: {"auprc": 0.03, "mcc": 0.1} for s in range(1, 5)}

    assert compare(base, arm, "mcc") is None
    assert compare(base, arm, "auprc")["seed_count"] == 4


def test_every_arm_present_is_compared_by_default(tmp_path: Path) -> None:
    _write_arm(tmp_path, "esm2", {s: 0.020 for s in range(1, 6)})
    _write_arm(tmp_path, "random_control", {s: 0.0201 for s in range(1, 6)})
    _write_arm(tmp_path, "antiberty_only", {s: 0.025 for s in range(1, 6)})
    output = tmp_path / "report.json"

    assert main(["--run-root", str(tmp_path), "--json", str(output)]) == 0

    report = json.loads(output.read_text())
    assert sorted(report["comparisons"]) == ["antiberty_only", "random_control"]


def test_the_relative_difference_is_against_the_baseline_arm(tmp_path: Path) -> None:
    base = {s: {"auprc": 0.020} for s in range(1, 6)}
    arm = {s: {"auprc": 0.025} for s in range(1, 6)}

    assert compare(base, arm, "auprc")["relative_difference"] == pytest.approx(0.25)


def test_a_missing_baseline_arm_refuses(tmp_path: Path) -> None:
    _write_arm(tmp_path, "random_control", {s: 0.02 for s in range(1, 6)})

    assert main(["--run-root", str(tmp_path), "--baseline", "esm2"]) == 2
