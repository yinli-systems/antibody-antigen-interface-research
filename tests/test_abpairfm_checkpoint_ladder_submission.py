from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import subprocess

import scripts.maybe_submit_abpairfm_checkpoint_ladder_probe as submission
from scripts.maybe_submit_abpairfm_checkpoint_ladder_probe import selected_target


def test_selected_target_uses_first_checkpoint_with_bounded_overshoot() -> None:
    assert selected_target(249_999_999) is None
    assert selected_target(250_000_000) == 250_000_000
    assert selected_target(252_633_502) == 250_000_000
    assert selected_target(270_000_000) == 250_000_000
    assert selected_target(270_000_001) is None
    assert selected_target(499_999_999) is None
    assert selected_target(500_000_000) == 500_000_000
    assert selected_target(520_000_000) == 500_000_000
    assert selected_target(520_000_001) is None
    assert selected_target(1_000_000_000) == 1_000_000_000
    assert selected_target(1_020_000_000) == 1_000_000_000


def test_submission_binds_the_immutable_project_root(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "immutable-candidate"
    project_root.mkdir()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "checkpoint_ladder": {
                    "maximum_target_overshoot_residue_tokens": 20_000_000,
                    "targets": [250_000_000, 500_000_000, 1_000_000_000],
                },
                "status": "FROZEN_BEFORE_GPU_EVALUATION",
                "test_access": False,
            }
        )
    )
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "checkpoint": {
                    "content_sha256": "checkpoint-content",
                    "generation": "segment-000031",
                    "root": "/immutable/checkpoints",
                },
                "status": "PASS_CHECKPOINT_VERIFIED",
                "test_access": False,
                "training_state": {"consumed_residue_tokens": 500_000_000},
            }
        )
    )
    expected_protocol_sha256 = hashlib.sha256(protocol.read_bytes()).hexdigest()
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="1462000\n", stderr="")

    monkeypatch.setattr(
        submission,
        "_arguments",
        lambda: Namespace(
            expected_protocol_sha256=expected_protocol_sha256,
            project_root=project_root,
            protocol=protocol,
            result=result,
            submission_root=tmp_path / "submissions",
        ),
    )
    monkeypatch.setattr(submission, "_source_sha256", lambda _path: "source-hash")
    monkeypatch.setattr(submission.subprocess, "run", fake_run)

    submission.main()

    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert environment["ABPAIRFM_PROBE_PROJECT"] == str(project_root)
    assert observed["command"][-1] == str(
        project_root / "scripts/submit_abpairfm_checkpoint_transfer_probe.sbatch"
    )
