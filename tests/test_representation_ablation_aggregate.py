"""The paired arithmetic behind the representation ablation.

Both arms share their seeds, and differencing within a seed is the whole point:
the ten-seed replication measured a macro-AUPRC sd of 0.00168 against a mean of
0.02246, so an unpaired comparison at this n would be swamped by variation that
is common to both arms.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
if str(REPOSITORY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "aggregate_representation_ablation_under_test",
    REPOSITORY_ROOT / "scripts" / "aggregate_representation_ablation.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

main = _MODULE.main
clopper_pearson_lower_bound = _MODULE.clopper_pearson_lower_bound
bootstrap_mean_interval = _MODULE.bootstrap_mean_interval


def _run(monkeypatch, control: dict[int, float], treatment: dict[int, float], tmp_path: Path):
    monkeypatch.setattr(
        _MODULE,
        "arm_scores",
        lambda root, metric, scope: control if root.name == "esm2" else treatment,
    )
    output = tmp_path / "report.json"
    code = main(["--run-root", str(tmp_path), "--json", str(output)])
    import json

    return code, json.loads(output.read_text())


def test_only_seeds_present_in_both_arms_are_compared(monkeypatch, tmp_path: Path) -> None:
    """A seed that failed in one arm cannot contribute a paired difference."""
    _, report = _run(
        monkeypatch, {1: 0.02, 2: 0.02, 3: 0.02}, {1: 0.03, 2: 0.03}, tmp_path
    )

    assert report["seed_count"] == 2
    assert report["unpaired_seeds"] == [3]


def test_the_difference_is_treatment_minus_control(monkeypatch, tmp_path: Path) -> None:
    _, report = _run(monkeypatch, {1: 0.02, 2: 0.02}, {1: 0.03, 2: 0.01}, tmp_path)

    assert [entry["difference"] for entry in report["per_seed"]] == pytest.approx([0.01, -0.01])
    assert report["mean_difference"] == pytest.approx(0.0)


def test_a_consistent_improvement_is_decided(monkeypatch, tmp_path: Path) -> None:
    control = {seed: 0.020 for seed in range(1, 11)}
    treatment = {seed: 0.020 + 0.002 + 0.0001 * seed for seed in range(1, 11)}

    _, report = _run(monkeypatch, control, treatment, tmp_path)

    assert report["favourable_seed_count"] == 10
    assert report["bootstrap_ci95"][0] > 0
    assert report["decided"] is True
    assert report["verdict"] == "treatment better"


def test_a_consistent_regression_is_also_decided(monkeypatch, tmp_path: Path) -> None:
    """A treatment that loses must be reported as decided, not as "no decision"."""
    control = {seed: 0.020 for seed in range(1, 11)}
    treatment = {seed: 0.020 - 0.002 - 0.0001 * seed for seed in range(1, 11)}

    _, report = _run(monkeypatch, control, treatment, tmp_path)

    assert report["favourable_seed_count"] == 0
    assert report["bootstrap_ci95"][1] < 0
    assert report["verdict"] == "control better"


def test_noise_around_zero_is_not_decided(monkeypatch, tmp_path: Path) -> None:
    """Not worse is not better; the interval has to exclude zero to decide anything."""
    control = {seed: 0.020 for seed in range(1, 11)}
    treatment = {seed: 0.020 + (0.0005 if seed % 2 else -0.0005) for seed in range(1, 11)}

    _, report = _run(monkeypatch, control, treatment, tmp_path)

    assert report["decided"] is False
    assert report["verdict"] == "no decision"


def test_the_relative_difference_is_against_the_control(monkeypatch, tmp_path: Path) -> None:
    control = {seed: 0.020 for seed in range(1, 6)}
    treatment = {seed: 0.022 for seed in range(1, 6)}

    _, report = _run(monkeypatch, control, treatment, tmp_path)

    assert report["relative_difference"] == pytest.approx(0.10)


def test_fewer_than_two_paired_seeds_refuses(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        _MODULE, "arm_scores", lambda root, metric, scope: {1: 0.02} if root.name == "esm2" else {}
    )

    assert main(["--run-root", str(tmp_path)]) == 2


def test_the_sign_bound_cannot_clear_one_half_below_five_seeds() -> None:
    """Inherited from the margin aggregator, and load-bearing for small runs."""
    assert clopper_pearson_lower_bound(4, 4) < 0.5
    assert clopper_pearson_lower_bound(5, 5) > 0.5


def test_pairing_detects_an_effect_that_is_invisible_unpaired() -> None:
    """The reason both arms share seeds rather than being sampled independently."""
    seed_effect = [0.0, 0.002, -0.0018, 0.0015, -0.001, 0.0022, -0.002, 0.0009, -0.0012, 0.0016]
    control = [0.0225 + value for value in seed_effect]
    treatment = [value + 0.0008 for value in control]

    paired_low, _ = bootstrap_mean_interval([t - c for t, c in zip(treatment, control)])
    unpaired_low, unpaired_high = bootstrap_mean_interval(
        [t - c for t, c in zip(treatment, reversed(control))]
    )

    assert paired_low > 0
    assert unpaired_low < 0 < unpaired_high
