"""Validate production recovery gate freezing after independent corpus audit."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts/freeze_antibody_production_recovery_gate.py"


def _load(name: str, path: Path):
    for item in (ROOT / "scripts", ROOT / "src"):
        value = str(item)
        if value not in sys.path:
            sys.path.insert(0, value)
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_gate_freezer_is_ascii_and_syntax_valid() -> None:
    source = FREEZER.read_text(encoding="ascii")
    ast.parse(source)
    assert "load_production_corpus(" in source
    assert "Audited production training split is empty" in source
    assert "Implementation input is invalid" in source
    assert '"unique_train_residue_tokens"' in source
    assert '"decontamination_excluded_train_residue_tokens"' in source
    assert "Audited corpus decontamination accounting differs" in source
    assert "corpus.close()" in source
    assert "os.O_EXCL" in source


def test_gate_freezer_locks_exact_recovery_roles_and_no_scale() -> None:
    source = FREEZER.read_text(encoding="ascii")
    assert '"continuous": run(steps=4, checkpoint=False, resume=False)' in source
    assert '"stage1": run(steps=2, checkpoint=True, resume=False)' in source
    assert '"resumed": run(steps=2, checkpoint=False, resume=True)' in source
    assert '"preset": "abf_650m"' in source
    assert '"world_size": 1' in source
    assert '"grouped_update_read": True' in source
    assert '"maximum_open_arrays": 1024' in source
    assert '"minimum_soft_open_file_limit": 4096' in source
    assert '"production_pretraining": False' in source
    assert '"scale_out": False' in source
    assert '"scientific_claim": False' in source


def test_gate_uses_exact_decontaminated_counts_and_closes_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freezer = _load("production_gate_freezer_counts_test", FREEZER)

    class FakeCorpus:
        closed = False
        manifest = {
            "counts": {
                "decontamination_excluded_train_records": 3,
                "decontamination_excluded_train_residue_tokens": 30,
                "decontamination_excluded_validation_records": 2,
                "decontamination_excluded_validation_residue_tokens": 20,
                "decontamination_source_train_records": 103,
                "decontamination_source_train_residue_tokens": 1_030,
                "decontamination_source_validation_records": 12,
                "decontamination_source_validation_residue_tokens": 120,
                "unique_train_records": 100,
                "unique_train_residue_tokens": 1_000,
                "unique_validation_records": 10,
                "unique_validation_residue_tokens": 100,
            }
        }

        def __len__(self) -> int:
            return 100

        def close(self) -> None:
            self.closed = True

    corpus = FakeCorpus()
    monkeypatch.setattr(freezer, "load_production_corpus", lambda *_, **__: corpus)
    required = set(freezer.IMPLEMENTATION_FILES)
    required.update(
        {
            "protocols/antibody_efficiency_gate_v1.json",
            "protocols/antibody_model_family_v1.json",
            "protocols/antibody_training_v1.json",
            "scripts/audit_antibody_production_loader_recovery.py",
            "scripts/freeze_antibody_production_recovery_gate.py",
        }
    )
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="ascii")

    gate = freezer.freeze_gate(
        source_root=tmp_path,
        corpus_root=tmp_path,
        manifest_file_sha256="1" * 64,
        manifest_payload_sha256="2" * 64,
        corpus_protocol_file_sha256="3" * 64,
        independent_audit_path=tmp_path,
        independent_audit_file_sha256="4" * 64,
        independent_audit_payload_sha256="5" * 64,
    )

    assert corpus.closed is True
    assert gate["corpus"]["train_record_count"] == 100
    assert gate["corpus"]["train_residue_token_count"] == 1_000
    assert gate["corpus"]["validation_record_count"] == 10
    assert gate["corpus"]["validation_residue_token_count"] == 100
