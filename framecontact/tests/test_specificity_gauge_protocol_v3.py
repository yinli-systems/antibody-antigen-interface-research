"""Execution contracts for the post-preflight specificity-gauge protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/framecontact_specificity_gauge_exploratory_v3.json"
RECEIPT = ROOT / "protocols/FRAMECONTACT_SPECIFICITY_GAUGE_EXPLORATORY_V3_RECEIPT.json"
SLURM = ROOT / "scripts/slurm_train_framecontact_specificity_gauge_exploratory_v3.sbatch"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v3_restores_the_exact_runtime_authority_schema() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["v2_execution_failure"]["slurm_job"] == "1349889"
    assert protocol["evidence_boundary_at_freeze"][
        "v2_valid_training_epoch_completed"
    ] is False
    panel = protocol["data"]["reciprocal_partner_panel"]
    assert panel["format_version"] == "framecontact_reciprocal_partner_panel_v2"
    assert panel["size_per_query"] == 31
    assert panel["train_sha256"].startswith("fd64b540")
    assert panel["validation_sha256"].startswith("222a22e1")
    assert protocol["implementation"]["attention_kernel"] == "reference"
    weights = protocol["training_objective"]["loss_weights_by_condition"]
    assert weights["framecontact_gauge_hybrid_contrastive"][
        "residue_partner_contrastive"
    ] == 0.25


def test_v3_receipt_matches_protocol_and_v2_parents() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["protocol_file_sha256"] == _sha(PROTOCOL)
    for key, metadata in protocol["parents"].items():
        assert _sha(ROOT / metadata["path"]) == metadata["sha256"]
        assert receipt["parent_sha256"][key] == metadata["sha256"]
    assert receipt["reciprocal_panel_contract_validated"] is True
    assert receipt["implementation_contract_validated"] is True


def test_v3_slurm_is_the_frozen_development_matrix_only() -> None:
    text = SLURM.read_text()
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    assert "#SBATCH --array=0-14%5" in text
    assert "specificity-gauge-source-v3" in text
    assert "specificity-gauge-exploratory-v3" in text
    assert "framecontact_specificity_gauge_exploratory_v3.json" in text
    assert "FRAMECONTACT_SPECIFICITY_GAUGE_EXPLORATORY_V3_RECEIPT.json" in text
    assert "GAUGE_V3_SOURCE_MANIFEST_SHA256" in text
    assert "GAUGE_V3_PROTOCOL_RECEIPT_SHA256" in text
    assert "frozen_test" not in text.lower()
    assert "test.csv" not in text


def test_v3_freezer_reproduces_committed_receipt(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/freeze_framecontact_specificity_gauge_protocol_v3.py",
            "--protocol",
            "protocols/framecontact_specificity_gauge_exploratory_v3.json",
            "--v2-protocol",
            "protocols/framecontact_specificity_gauge_exploratory_v2.json",
            "--v2-receipt",
            "protocols/FRAMECONTACT_SPECIFICITY_GAUGE_EXPLORATORY_V2_RECEIPT.json",
            "--receipt",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert json.loads(output.read_text()) == json.loads(RECEIPT.read_text())
