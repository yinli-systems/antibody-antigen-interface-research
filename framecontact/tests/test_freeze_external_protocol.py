from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_external_protocol.py"
SPEC = importlib.util.spec_from_file_location("freeze_external_protocol", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def protocol_payload() -> dict:
    return json.loads(
        (ROOT / "evaluation" / "chimera_zero_shot_protocol_v1.json").read_text()
    )


def test_protocol_is_valid_without_remote_checkpoint_access(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol_payload()))
    observed = MODULE.load_and_validate(path, verify_checkpoints=False)
    assert observed["evaluation"]["primary_comparison"].startswith("spatial_topology")


def test_protocol_rejects_candidate_drift(tmp_path: Path) -> None:
    payload = protocol_payload()
    payload["model"]["candidates"]["distance"] = payload["model"]["candidates"][
        "spatial_topology"
    ]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="candidate set differs"):
        MODULE.load_and_validate(path, verify_checkpoints=False)


def test_protocol_rejects_split_reordering(tmp_path: Path) -> None:
    payload = protocol_payload()
    payload["evaluation"]["primary_splits"].reverse()
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="primary split order differs"):
        MODULE.load_and_validate(path, verify_checkpoints=False)
