from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_abpairfm_stage_a1_geometry_canary.py"


def test_geometry_canary_audit_passes_learning_gate(tmp_path: Path) -> None:
    loss = {
        "active_update_count": 20,
        "first_10_active_update_mean": 1.0,
        "final_10_active_update_mean": 0.8,
    }
    result = tmp_path / "RESULT.json"
    result.write_text(
        json.dumps(
            {
                "checkpoint": {
                    "content_sha256": "a" * 64,
                    "restore_content_sha256": "a" * 64,
                },
                "loss": {"contact_loss": loss, "distance_loss": loss},
                "source_sha256": "b" * 64,
                "status": "PASS_STAGE_A1_EFFICIENCY_GATE_CHECKPOINT_VERIFIED",
                "test_access": False,
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
    assert payload["status"] == "PASS_STAGE_A1_GEOMETRY_CANARY_LEARNING_GATE"
    assert payload["authorization"]["development_evaluation"] is True
    assert payload["test_access"] is False
