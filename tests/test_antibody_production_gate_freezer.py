"""Validate production recovery gate freezing after independent corpus audit."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts/freeze_antibody_production_recovery_gate.py"


def test_gate_freezer_is_ascii_and_syntax_valid() -> None:
    source = FREEZER.read_text(encoding="ascii")
    ast.parse(source)
    assert "load_production_corpus(" in source
    assert "Audited production training split is empty" in source
    assert "Implementation input is invalid" in source
    assert "os.O_EXCL" in source


def test_gate_freezer_locks_exact_recovery_roles_and_no_scale() -> None:
    source = FREEZER.read_text(encoding="ascii")
    assert '"continuous": run(steps=4, checkpoint=False, resume=False)' in source
    assert '"stage1": run(steps=2, checkpoint=True, resume=False)' in source
    assert '"resumed": run(steps=2, checkpoint=False, resume=True)' in source
    assert '"preset": "abf_650m"' in source
    assert '"world_size": 1' in source
    assert '"production_pretraining": False' in source
    assert '"scale_out": False' in source
    assert '"scientific_claim": False' in source
