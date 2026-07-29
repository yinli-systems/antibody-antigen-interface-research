"""Tests for the sealed AB03 A100 replay verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import statistics
import tarfile

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_ab03_a100_geometry_replay.py"
)
SPEC = importlib.util.spec_from_file_location(
    "verify_ab03_a100_geometry_replay",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def _statistics(values: list[float]) -> dict[str, object]:
    return {
        "values": values,
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
    }


def _multiseed_payload() -> dict[str, object]:
    seeds = [17, 42, 73]
    metrics: dict[str, object] = {}
    for mode_index, mode in enumerate(
        (
            "dense_sequence",
            "topology_only",
            "distance",
            "distance_direction",
            "full_pose",
        )
    ):
        metrics[mode] = {
            metric: _statistics(
                [
                    0.10 + 0.01 * mode_index + 0.001 * seed_index
                    for seed_index in range(len(seeds))
                ]
            )
            for metric in ("macro_auprc", "paratope_auprc", "epitope_auprc")
        }
    return {
        "format_version": "sabdab2_self_attention_geometry_multiseed_v1",
        "seeds": seeds,
        "seed_count": len(seeds),
        "metrics": metrics,
    }


def test_multiseed_summary_recomputes_all_statistics() -> None:
    result = SCRIPT._multiseed_summary(_multiseed_payload())
    assert result["seed_count"] == 3
    assert result["recomputation_check_count"] == 60
    comparison = result["comparisons_vs_topology_only"][
        "distance_direction"
    ]["macro_auprc"]
    assert comparison["positive_seed_count"] == 3
    assert comparison["mean_difference"] == pytest.approx(0.02)


def test_multiseed_summary_rejects_changed_reported_mean() -> None:
    payload = _multiseed_payload()
    payload["metrics"]["distance_direction"]["macro_auprc"]["mean"] += 0.001
    with pytest.raises(ValueError, match="recomputation differs"):
        SCRIPT._multiseed_summary(payload)


def test_safe_member_rejects_path_traversal_and_links() -> None:
    traversal = tarfile.TarInfo("../receipt.json")
    traversal.type = tarfile.REGTYPE
    with pytest.raises(ValueError, match="Unsafe archive member"):
        SCRIPT._safe_member(traversal)

    link = tarfile.TarInfo("receipt-link")
    link.type = tarfile.SYMTYPE
    link.linkname = "receipt.json"
    with pytest.raises(ValueError, match="Unsafe archive member"):
        SCRIPT._safe_member(link)


def test_manifest_rejects_duplicate_path() -> None:
    digest = "a" * 64
    payload = f"{digest}  result.json\n{digest}  result.json\n".encode("ascii")
    with pytest.raises(ValueError, match="duplicate path"):
        SCRIPT._manifest(payload, "manifest")

