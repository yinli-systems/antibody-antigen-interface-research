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
    / "analyze_pinder_pair_loss_canary_v1.py"
)
SPEC = importlib.util.spec_from_file_location("pair_loss_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_arm(root: Path, arm: str, *, gain: float) -> None:
    expected = MODULE.EXPECTED[arm]
    before = []
    after = []
    for index in range(16):
        base = {
            "contact_ap": 0.01,
            "contact_auroc": 0.6,
            "contact_mrr": 0.1,
            "contact_precision_at_l_over_10": 0.0,
            "contact_recall_at_1": 0.0,
            "contact_recall_at_l_over_10": 0.0,
            "epitope_ap": 0.2,
            "example_id": f"example-{index:02d}",
            "matched_minus_swapped_mean_logit": 0.01,
            "paratope_ap": 0.4,
            "pdb_id": f"p{index:03d}",
            "prevalence": 0.001,
        }
        updated = dict(base)
        updated["contact_ap"] += gain
        updated["contact_mrr"] += gain
        before.append(base)
        after.append(updated)
    value = {
        "checkpoint": {"content_sha256": "c" * 64, "restore_content_sha256": "c" * 64},
        "evaluation": {
            "after": {"contact_prevalence": 0.001, "per_system": after},
            "before": {"contact_prevalence": 0.001, "per_system": before},
        },
        "input_checkpoint": {"content_sha256": MODULE.EXPECTED_CHECKPOINT},
        "objective": {"pair_loss_mode": expected["mode"]},
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


def test_pair_loss_analysis_promotes_focal_gain(tmp_path: Path, monkeypatch) -> None:
    _write_arm(tmp_path, "adaptive_balanced_control", gain=0.01)
    _write_arm(tmp_path, "balanced_focal", gain=0.02)
    output = tmp_path / "ANALYSIS.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--campaign-root", str(tmp_path), "--output", str(output)],
    )
    MODULE.main()
    result = json.loads(output.read_text(encoding="ascii"))
    assert result["status"] == "PASS_BALANCED_FOCAL_PAIR_RANKING_GATE"
    assert result["delta_of_deltas"]["contact_ap"]["mean"] > 0.0
    assert result["test_access"] is False
