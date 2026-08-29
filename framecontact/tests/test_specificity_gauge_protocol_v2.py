"""Contracts for the repaired centered-evidence gauge protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "framecontact_specificity_gauge_exploratory_v2.json"
RECEIPT = (
    ROOT
    / "protocols"
    / "FRAMECONTACT_SPECIFICITY_GAUGE_EXPLORATORY_V2_RECEIPT.json"
)
SLURM = (
    ROOT
    / "scripts"
    / "slurm_train_framecontact_specificity_gauge_exploratory_v2.sbatch"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_freezes_the_saturation_repair_and_failure_boundary() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["v1_failure"]["slurm_job"] == "1349055"
    assert protocol["v1_failure"]["observed_tasks"] == [0, 1, 2, 3, 4]
    assert "cannot enter model selection" in protocol["v1_failure"][
        "evidence_policy"
    ]
    architecture = protocol["controlled_architecture"]
    assert "log-mean-exp" in architecture["gauge_aggregation"]
    assert "not a Bernoulli" in architecture["gauge_aggregation_semantics"]
    assert architecture["raw_contact_aggregation"].startswith("masked noisy-OR")
    assert protocol["data"]["test_role_forbidden"] is True
    assert protocol["model_selection"]["seeds"] == [17, 42, 73, 101, 137]


def test_v2_receipt_matches_protocol_and_all_parents() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    receipt = json.loads(RECEIPT.read_text())
    assert receipt["protocol_file_sha256"] == _sha(PROTOCOL)
    assert receipt["centered_logmeanexp_repair_validated"] is True
    for key, metadata in protocol["parents"].items():
        parent = ROOT / metadata["path"]
        assert _sha(parent) == metadata["sha256"]
        assert receipt["parent_sha256"][key] == metadata["sha256"]


def test_v2_slurm_entry_is_bounded_to_the_repaired_development_matrix() -> None:
    text = SLURM.read_text()
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    assert "#SBATCH --array=0-14%5" in text
    assert "specificity-gauge-source-v2" in text
    assert "specificity-gauge-exploratory-v2" in text
    assert "specificity-gauge-source-v2-r2" in text
    assert "specificity-gauge-exploratory-v2-r2" in text
    assert "framecontact_specificity_gauge_exploratory_v2.json" in text
    assert "FRAMECONTACT_SPECIFICITY_GAUGE_EXPLORATORY_V2_RECEIPT.json" in text
    assert "GAUGE_V2_SOURCE_MANIFEST_SHA256" in text
    assert "GAUGE_V2_PROTOCOL_RECEIPT_SHA256" in text
    assert "antigen_fold.csv" in text
    assert "frozen_test" not in text.lower()
    assert "test.csv" not in text


def test_v2_freezer_reproduces_committed_receipt(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/freeze_framecontact_specificity_gauge_protocol_v2.py",
            "--protocol",
            "protocols/framecontact_specificity_gauge_exploratory_v2.json",
            "--v1-protocol",
            "protocols/framecontact_specificity_gauge_exploratory_v1.json",
            "--v1-receipt",
            "protocols/FRAMECONTACT_SPECIFICITY_GAUGE_EXPLORATORY_V1_RECEIPT.json",
            "--residue-head-parent",
            "protocols/framecontact_residue_head_exploratory_v1.json",
            "--split-parent",
            "protocols/framecontact_sabdab2_antigen_fold_v3.json",
            "--receipt",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    reproduced = json.loads(output.read_text())
    committed = json.loads(RECEIPT.read_text())
    assert reproduced == committed
