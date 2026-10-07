from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "remote_tools"
    / "abpairfm_distributed_contact_pretraining_v1"
    / "analyze_pinder_bipartite_decoder_canary_v1.py"
)
SPEC = importlib.util.spec_from_file_location("bipartite_decoder_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _systems(*, contact_gain: float, mrr_gain: float) -> tuple[list[dict], list[dict]]:
    before = []
    after = []
    for index in range(16):
        base = {
            "contact_ap": 0.01 + index / 10_000,
            "contact_auroc": 0.6,
            "contact_mrr": 0.1,
            "contact_precision_at_l_over_10": 0.05,
            "contact_recall_at_1": 0.0,
            "contact_recall_at_l_over_10": 0.01,
            "epitope_ap": 0.2,
            "example_id": f"example-{index:02d}",
            "matched_minus_swapped_mean_logit": 0.01,
            "paratope_ap": 0.4,
            "pdb_id": f"p{index:03d}",
            "prevalence": 0.001,
        }
        updated = dict(base)
        updated["contact_ap"] += contact_gain
        updated["contact_mrr"] += mrr_gain
        if contact_gain > 0.015:
            updated["contact_recall_at_1"] = 1.0
        before.append(base)
        after.append(updated)
    return before, after


def _write_arm(root: Path, arm: str, *, contact_gain: float, mrr_gain: float) -> None:
    before, after = _systems(contact_gain=contact_gain, mrr_gain=mrr_gain)
    expected = MODULE.EXPECTED[arm]
    value = {
        "checkpoint": {"content_sha256": "c" * 64, "restore_content_sha256": "c" * 64},
        "evaluation": {
            "after": {"contact_prevalence": 0.001, "per_system": after},
            "before": {"contact_prevalence": 0.001, "per_system": before},
        },
        "input_checkpoint": {"content_sha256": MODULE.EXPECTED_CHECKPOINT},
        "promotion_checks": {"finite_all_update_metrics": True},
        "provenance": {
            "protocol": expected["protocol"],
            "source": MODULE.EXPECTED_SOURCE,
        },
        "status": expected["status"],
        "test_access": False,
    }
    path = root / arm / "RESULT.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value), encoding="ascii")


def test_matched_analysis_promotes_positive_edge_refinement(tmp_path: Path, monkeypatch) -> None:
    _write_arm(tmp_path, "low_rank_control", contact_gain=0.01, mrr_gain=0.01)
    _write_arm(
        tmp_path,
        "bipartite_edge_refinement",
        contact_gain=0.02,
        mrr_gain=0.02,
    )
    output = tmp_path / "ANALYSIS.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--campaign-root", str(tmp_path), "--output", str(output)],
    )
    MODULE.main()
    result = json.loads(output.read_text(encoding="ascii"))
    assert result["status"] == "PASS_BIPARTITE_EDGE_REFINEMENT_RANKING_GATE"
    assert result["test_access"] is False
    assert result["delta_of_deltas"]["contact_ap"]["mean"] > 0.0
