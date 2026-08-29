import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "results/cluster/CLUSTER_ACTIVITY_SNAPSHOT_20260821.json"


def test_snapshot_preserves_transient_status_boundary() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    assert snapshot["status"] == "point_in_time_scheduler_snapshot"
    assert "not a terminal training result" in snapshot["claim_boundary"]
    mipe = snapshot["active_work"]["mipe_paper_appendix_lr1e4"]
    assert mipe["running_fold_jobs"] == 5
    assert mipe["terminal_metrics_available"] is False


def test_snapshot_records_only_audited_running_work() -> None:
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    ablation = snapshot["active_work"]["partner_negative_ablation"]
    assert ablation["running_tasks"] == 10
    assert ablation["test_accessed"] is False
    assert snapshot["pretraining"]["active_scheduler_jobs"] == 1
    assert snapshot["pretraining"]["exact_recovery_passed"] is True
    assert snapshot["pretraining"]["terminal_metrics_available"] is False


def test_snapshot_excludes_cluster_identifiers() -> None:
    serialized = SNAPSHOT_PATH.read_text()
    assert not any(
        marker in serialized
        for marker in ("scxi253", "BSCC", "ssh.cn", "/ssd/", "JobId")
    )
