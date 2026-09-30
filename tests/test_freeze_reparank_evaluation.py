from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from extensions.frontier_multimodel_v1.freeze_reparank_evaluation import freeze, sha256


def write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_bytes(payload)
    return path


def fixture(tmp_path: Path, mode: str = "sequence") -> SimpleNamespace:
    checkpoint = write(tmp_path / "model.ckpt", b"checkpoint")
    source = write(tmp_path / "evaluate.py", b"source")
    split = write(tmp_path / "split.json", {
        "splits": {"train": ["t"], "validation": [f"v{i}" for i in range(217)], "test": ["z"]}
    })
    canonical = write(tmp_path / "canonical.csv", b"INSTANCE,vh_seq,vl_seq,ag_seq\n")
    contact = tmp_path / "contacts"
    contact.mkdir()
    for index in range(3663):
        (contact / f"{index}.npz").write_bytes(b"x")
    audit = write(tmp_path / "audit.json", {
        "status": "PASS_CHECKPOINT_AUDIT",
        "checkpoint_sha256": sha256(checkpoint),
        "mode": mode,
        "model_state_loaded_strictly": True,
    })
    canary_status = "PASS_SEQUENCE_CANARY_REPEAT" if mode == "sequence" else "PASS_STRUCTURE_CANARY"
    canary = write(tmp_path / "canary.json", {"status": canary_status, "test_labels_read": False})
    structure = None
    if mode == "sequence_plus_structure":
        base = tmp_path / "structures"
        pdb = write(base / "pdb/a.pdb", b"ATOM\n")
        structure = write(base / "manifest.json", {
            "format_version": "frontier_structure_units_v2",
            "contact_labels_loaded": False,
            "test_loaded": False,
            "coordinate_policy": "predicted chain local",
            "counts_by_side": {"antibody": 1, "antigen": 0},
            "records": [{
                "sequence_sha256": "a" * 64,
                "pdb_path": "pdb/a.pdb",
                "pdb_sha256": "b" * 64,
            }],
        })
    return SimpleNamespace(
        mode=mode, checkpoint=checkpoint, checkpoint_audit=audit,
        canary_result=canary, source_script=source, source_commit="commit",
        official_source_commit="official", split_manifest=split,
        canonical_manifest=canonical, contact_dir=contact,
        structure_manifest=structure,
    )


def test_sequence_plan_freezes_without_structure(tmp_path: Path) -> None:
    payload = freeze(fixture(tmp_path))
    assert payload["mode"] == "sequence"
    assert payload["structure_authority"] is None
    assert payload["validation_instance_count"] == 217
    assert payload["contact_inventory"]["file_count"] == 3663
    assert payload["test_authorized"] is False


def test_structure_plan_requires_and_binds_structure(tmp_path: Path) -> None:
    payload = freeze(fixture(tmp_path, "sequence_plus_structure"))
    assert payload["structure_authority"]["record_count"] == 1
    assert payload["structure_authority"]["contact_labels_loaded"] is False


def test_wrong_checkpoint_hash_fails(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    data = json.loads(args.checkpoint_audit.read_text())
    data["checkpoint_sha256"] = "0" * 64
    args.checkpoint_audit.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not bind"):
        freeze(args)


def test_canary_gate_fails_closed(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    args.canary_result.write_text(json.dumps({"status": "FAILED", "test_labels_read": False}))
    with pytest.raises(ValueError, match="canary"):
        freeze(args)


def test_overlap_fails_closed(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    split = json.loads(args.split_manifest.read_text())
    split["splits"]["test"].append("v1")
    args.split_manifest.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="split overlap"):
        freeze(args)


def test_sequence_mode_rejects_structure_argument(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    args.structure_manifest = tmp_path / "irrelevant.json"
    with pytest.raises(ValueError, match="must not name"):
        freeze(args)
