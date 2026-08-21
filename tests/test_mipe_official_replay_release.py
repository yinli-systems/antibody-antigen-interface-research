import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results/mipe/MIPE_OFFICIAL_REPLAY_20260821.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_binds_exact_scripts_and_protocol() -> None:
    result = json.loads(RESULT_PATH.read_text())
    receipts = result["artifact_receipts"]
    expected = {
        "mipe_independent_test_script_sha256": ROOT
        / "scripts/mipe_independent_test.py",
        "mipe_oof_threshold_calibration_script_sha256": ROOT
        / "scripts/mipe_oof_threshold_calibration.py",
        "mipe_bound_geometry_diagnostic_script_sha256": ROOT
        / "scripts/mipe_bound_geometry_diagnostic.py",
        "freeze_mipe_selections_script_sha256": ROOT
        / "scripts/freeze_mipe_selections.py",
        "paper_appendix_protocol_sha256": ROOT
        / "protocols/mipe_paper_appendix_lr1e4_seed42.json",
    }
    for receipt_key, path in expected.items():
        assert receipts[receipt_key] == sha256(path)


def test_release_preserves_claim_and_access_boundaries() -> None:
    result = json.loads(RESULT_PATH.read_text())
    serialized = json.dumps(result, sort_keys=True)
    assert "not a pristine blind test" in result["claim_boundary"]
    assert "not a deployable unbound or sequence-only predictor" in result[
        "bound_geometry_oracle_diagnostic"
    ]["claim_boundary"]
    assert result["paper_appendix_lr1e4_replay"]["terminal_metrics"] is None
    assert result["paper_appendix_lr1e4_replay"]["status"].startswith("running_as_of_")
    assert not any(
        marker in serialized
        for marker in ("scxi253", "BSCC", "ssh.cn", "slurm_job_id")
    )


def test_pair_metrics_are_not_conflated_with_the_oracle() -> None:
    result = json.loads(RESULT_PATH.read_text())
    replay = result["provided_test_five_fold_ensemble"][
        "secondary_validation_best"
    ]
    oracle = result["bound_geometry_oracle_diagnostic"]["provided_test"][
        "pair_contact"
    ]
    assert replay["native_mean_bidirectional_attention"]["recall_at_1"] == 0.0
    assert replay["predicted_residue_outer_product"]["macro_auprc"] < oracle[
        "macro_auprc"
    ]
