"""Contract tests for the pinned public MINT checkpoint auditor."""

from __future__ import annotations

import importlib.util
from math import prod
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_mint_650m_checkpoint.py"


def _module():
    specification = importlib.util.spec_from_file_location("mint_audit", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_expected_mint_schema_matches_checkpoint_inventory() -> None:
    module = _module()
    shapes = module.expected_shapes()
    assert len(shapes) == 768
    assert sum(".multimer_attn." in name for name in shapes) == 198
    assert shapes["embed_tokens.weight"] == (33, 1280)
    assert shapes["layers.32.multimer_attn.q_proj.weight"] == (1280, 1280)
    assert shapes["layers.0.self_attn.rot_emb.inv_freq"] == (32,)
    state_elements = sum(prod(shape) for shape in shapes.values())
    assert state_elements == 813_414_209
    assert state_elements - prod(shapes["lm_head.weight"]) == 813_371_969


def test_normalize_state_requires_exact_model_prefix() -> None:
    module = _module()
    tensor = torch.zeros(1)
    assert module.normalize_state({"model.value": tensor}) == {"value": tensor}
    with pytest.raises(ValueError, match="exact model. prefix"):
        module.normalize_state({"value": tensor})


def test_mint_audit_receipt_is_create_only(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "receipt.json"
    module.publish_create_only(output, {"status": "PASS"})
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.publish_create_only(output, {"status": "PASS"})
