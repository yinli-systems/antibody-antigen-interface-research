from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_abpairfm_stage_a1_geometry_ranking_canary.py"


def test_geometry_ranking_audit_passes_all_frozen_gates(tmp_path: Path) -> None:
    loss = {
        "active_update_count": 80,
        "first_20_active_update_mean": 1.0,
        "final_20_active_update_mean": 0.75,
    }
    before = {
        "balanced_contact_bce": 0.70,
        "complex_count": 512,
        "distance_cross_entropy": 2.0,
        "macro_sampled_contact_auprc": 0.20,
        "mean_negative_logit": 0.0,
        "mean_positive_logit": 0.0,
        "mean_sampled_positive_prevalence": 0.20,
        "pair_count": 2560,
        "pairwise_positive_over_negative_accuracy": 0.50,
        "ranking_softplus": 0.70,
    }
    after = {
        **before,
        "balanced_contact_bce": 0.65,
        "macro_sampled_contact_auprc": 0.24,
        "pairwise_positive_over_negative_accuracy": 0.54,
        "ranking_softplus": 0.65,
    }
    result = tmp_path / "RESULT.json"
    result.write_text(
        json.dumps(
            {
                "authorization": {"development_evaluation": False},
                "checkpoint": {
                    "content_sha256": "a" * 64,
                    "provenance_hashes": {
                        "protocol": "c" * 64,
                        "source": "b" * 64,
                    },
                    "restore_content_sha256": "a" * 64,
                },
                "internal_holdout": {
                    "after_training": after,
                    "before_training": before,
                    "receipt": {
                        "holdout_pdb_count": 511,
                        "holdout_record_count": 512,
                        "pdb_inventory_sha256": (
                            "0166ecee25b61f333fe36197ca9625ed3a134221946a38e023f6022b4cc49028"
                        ),
                        "test_access": False,
                        "training_record_count": 4455,
                    },
                    "test_access": False,
                },
                "loss": {
                    "contact_loss": loss,
                    "distance_loss": loss,
                    "ranking_loss": loss,
                },
                "source_sha256": "b" * 64,
                "status": "PASS_STAGE_A1_EFFICIENCY_GATE_CHECKPOINT_VERIFIED",
                "test_access": False,
                "training_protocol_format_version": (
                    "abpairfm_stage_a1_geometry_ranking_canary_v17"
                ),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "AUDIT.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--result",
            str(result),
            "--expected-protocol-sha256",
            "c" * 64,
            "--expected-source-sha256",
            "b" * 64,
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS_STAGE_A1_GEOMETRY_RANKING_CANARY_GATE"
    assert payload["authorization"] == {
        "development_evaluation": True,
        "test_access": False,
    }
