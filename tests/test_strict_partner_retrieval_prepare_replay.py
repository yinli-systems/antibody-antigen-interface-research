from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "extensions/strict_partner_retrieval_v1/prepare_pinned_replay.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> list[str]:
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "src/interaction_first").mkdir(parents=True)
    files = {
        "scripts/build_train_only_nested_neighborhood_folds.py": b"builder",
        "scripts/audit_train_only_nested_neighborhood_folds.py": b"auditor",
        "src/interaction_first/nested_joint_milp.py": b"joint",
    }
    rows = []
    for relative, payload in files.items():
        path = source / relative
        path.write_bytes(payload)
        rows.append(f"{_sha(path)}  {relative}")
    manifest = source / "SOURCE_MANIFEST.sha256"
    manifest.write_text("\n".join(sorted(rows)) + "\n")
    authority = tmp_path / "authority.json"
    canonical = tmp_path / "canonical.csv"
    python = tmp_path / "python"
    authority.write_bytes(b"authority")
    canonical.write_bytes(b"canonical")
    python.write_bytes(b"python")
    anchor = tmp_path / "anchor.json"
    anchor.write_text(
        json.dumps(
            {
                "format_version": "strict_partner_retrieval_membership_anchor_v1",
                "status": "FROZEN_FOR_SEPARATE_PINNED_REPLAY",
                "provisional_attempt_id": "provisional_001",
                "selected_binary_membership_sha256": "b" * 64,
                "exact_joint_policy_sha256": "a" * 64,
                "replay_authorized": True,
                "training_authorized": False,
                "model_selection_authorized": False,
                "outer_evaluation_authorized": False,
                "test_access": False,
                "ood_access": False,
            }
        )
    )
    return [
        sys.executable,
        str(SCRIPT),
        "--anchor",
        str(anchor),
        "--source-root",
        str(source),
        "--expected-source-manifest-sha256",
        _sha(manifest),
        "--train-authority-allowlist",
        str(authority),
        "--expected-train-authority-sha256",
        _sha(authority),
        "--train-canonical-manifest",
        str(canonical),
        "--expected-train-manifest-sha256",
        _sha(canonical),
        "--expected-joint-policy-sha256",
        "a" * 64,
        "--pinned-replay-id",
        "replay_001",
        "--workers",
        "6",
        "--python-bin",
        str(python),
        "--output",
        str(tmp_path / "intent.json"),
    ]


def test_valid_replay_intent_is_non_authorizing(tmp_path: Path) -> None:
    command = _fixture(tmp_path)
    subprocess.run(command, check=True, capture_output=True, text=True)
    value = json.loads((tmp_path / "intent.json").read_text())
    assert value["status"] == "FROZEN_REPLAY_INTENT"
    assert value["replay_authorized"] is True
    assert value["training_authorized"] is False
    assert value["environment"]["FRONTIER_JOINT_ASSIGNMENT_MODE"] == "externally_pinned_replay"


def test_replay_id_cannot_equal_provisional_id(tmp_path: Path) -> None:
    command = _fixture(tmp_path)
    index = command.index("replay_001")
    command[index] = "provisional_001"
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "differ" in result.stderr
