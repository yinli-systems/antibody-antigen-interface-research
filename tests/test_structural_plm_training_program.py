from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from extensions.frontier_multimodel_v1.freeze_structural_training import (
    REAL_LAYERS,
    SEEDS,
    discover_specs,
)
from extensions.frontier_multimodel_v1.run_structural_training import (
    evaluation_negative_count,
    representation_args,
    run_checked,
)


def _spec(path: Path, family: str, layer: int, side: str, dim: int) -> Path:
    payload = {
        "format_version": "generic_representation_source_v1",
        "name": f"{family}_l{layer:02d}_{'ab' if side == 'antibody' else 'ag'}",
        "root": str((path.parent / "features" / family / str(layer) / side).resolve()),
        "dim": dim,
        "sides": [side],
        "antibody_storage": "parts",
        "model_name": f"{family}_model_layer_{layer}",
        "model_revision": "revision",
        "evidence_role": "development-only",
        "source_commit": "commit",
        "model_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "structure_authority_sha256": "c" * 64,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _complete_spec_tree(tmp_path: Path) -> Path:
    root = tmp_path / "source_specs"
    root.mkdir()
    for family, layers in REAL_LAYERS.items():
        dim = 1280 if family == "saprot" else 768
        for layer in layers:
            for side in ("antibody", "antigen"):
                _spec(root / f"{family}_layer_{layer:02d}_{side}.json", family, layer, side, dim)
    return root


def test_discover_specs_requires_exact_frozen_grid(tmp_path: Path) -> None:
    groups = discover_specs(_complete_spec_tree(tmp_path))
    assert set(groups) == {
        (family, layer) for family, layers in REAL_LAYERS.items() for layer in layers
    }
    assert all(len(paths) == 2 for paths in groups.values())


def test_discover_specs_rejects_missing_layer(tmp_path: Path) -> None:
    root = _complete_spec_tree(tmp_path)
    (root / "saprot_layer_16_antigen.json").unlink()
    with pytest.raises(ValueError, match="source-spec groups differ|requires antibody"):
        discover_specs(root)


def test_discover_specs_rejects_extra_file(tmp_path: Path) -> None:
    root = _complete_spec_tree(tmp_path)
    (root / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected source specification"):
        discover_specs(root)


def test_real_representation_arguments_are_two_explicit_sources() -> None:
    task = {
        "specs": ["/abs/antibody.json", "/abs/antigen.json"],
        "random_projection_control": False,
    }
    plan = {
        "paths": {"canonical_manifest": "/abs/canonical.csv"},
        "training": {"random_projection_seed": 20260930},
    }
    assert representation_args(task, plan) == [
        "--representation-manifest", "/abs/canonical.csv",
        "--generic-representation-source", "/abs/antibody.json",
        "--generic-representation-source", "/abs/antigen.json",
    ]


def test_control_reuses_source_contract_and_adds_fixed_projection() -> None:
    task = {
        "specs": ["/abs/antibody.json", "/abs/antigen.json"],
        "random_projection_control": True,
    }
    plan = {
        "paths": {"canonical_manifest": "/abs/canonical.csv"},
        "training": {"random_projection_seed": 20260930},
    }
    arguments = representation_args(task, plan)
    assert arguments[:6] == [
        "--representation-manifest", "/abs/canonical.csv",
        "--generic-representation-source", "/abs/antibody.json",
        "--generic-representation-source", "/abs/antigen.json",
    ]
    assert arguments[-3:] == [
        "--random-projection-control", "--random-projection-seed", "20260930"
    ]


def test_frozen_grid_has_ten_real_layers_thirty_real_tasks_and_six_controls() -> None:
    assert sum(len(layers) for layers in REAL_LAYERS.values()) == 10
    assert len(SEEDS) == 3
    assert 10 * len(SEEDS) == 30
    assert len(REAL_LAYERS) * len(SEEDS) == 6


def test_protocol_forbids_closed_test_and_smoke_as_science() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "extensions/frontier_multimodel_v1/STRUCTURAL_PLM_TRAINING_PROTOCOL.md"
    ).read_text(encoding="utf-8")
    assert "No test or rejected OOD label is read" in text
    assert "Smoke metrics are never scientific results" in text
    assert "No SOTA" in text


def test_runner_prepends_source_src_to_pythonpath(tmp_path: Path, monkeypatch) -> None:
    observed = {}

    def fake_run(command, cwd, check, env):
        observed.update({"command": command, "cwd": cwd, "check": check, "env": env})

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/existing")
    commands = tmp_path / "commands.jsonl"
    (tmp_path / "src").mkdir()
    run_checked(["python", "script.py"], tmp_path, commands)
    assert observed["check"] is True
    assert observed["cwd"] == tmp_path
    assert observed["env"]["PYTHONPATH"].split(":") == [str((tmp_path / "src").resolve()), "/existing"]
    assert json.loads(commands.read_text().strip()) == ["python", "script.py"]


def test_smoke_uses_one_partner_distractor_but_formal_keeps_nine() -> None:
    assert evaluation_negative_count(True) == 1
    assert evaluation_negative_count(False) == 9
