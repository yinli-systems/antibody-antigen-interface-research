"""Test protocol-bound useful-token approximate 6N MFU calculations."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_antibody_approximate_mfu.py"
PROTOCOL = ROOT / "protocols/antibody_foundation_scale_v2.json"


def _load_module():
    """Load the standalone MFU auditor from source."""
    specification = importlib.util.spec_from_file_location("mfu_auditor", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _receipt(tmp_path: Path, *, world_size: int = 8) -> Path:
    parameter_count = 1000
    trace = []
    consumed = 0
    for update in range(1, 5):
        consumed += 200
        trace.append(
            {
                "consumed_residue_tokens": consumed,
                "residue_tokens": 200,
                "seconds": 0.5,
                "update": update,
            }
        )
    value = {
        "ddp_logging_data": {"total_parameter_size_bytes": parameter_count * 4},
        "device": "NVIDIA GeForce RTX 5090",
        "optimization": {"ddp_communication_dtype": "float32"},
        "parameter_moments": {"count": parameter_count},
        "trace": trace,
        "world_size": world_size,
    }
    path = tmp_path / f"receipt-{world_size}.json"
    path.write_text(_canonical(value), encoding="ascii")
    return path


def test_auditor_replays_exact_useful_token_mfu_and_pooling(tmp_path: Path) -> None:
    """Use measured non-padding tokens and the frozen dense BF16 denominator."""
    module = _load_module()
    receipt = _receipt(tmp_path)
    result = module.audit_mfu(
        protocol_path=PROTOCOL,
        runs={"run-a": receipt, "run-b": receipt},
        warmup_updates=1,
    )
    summary = result["runs"]["run-a"]["summary"]
    expected = 6 * 1000 * 600 / (8 * 209.5e12 * 1.5)
    assert math.isclose(summary["approximate_6n_mfu"], expected)
    assert summary["nonpadding_residue_tokens"] == 600
    assert summary["measured_updates"] == 3
    assert result["pooled_identical_domain"]["residue_tokens_per_second"] == 400
    assert result["authorization"]["configuration_promotion"] is False
    assert "profiler FLOPs" in result["claim_boundary"]


def test_auditor_does_not_pool_different_world_sizes(tmp_path: Path) -> None:
    """Avoid a meaningless pooled MFU across different denominator domains."""
    module = _load_module()
    result = module.audit_mfu(
        protocol_path=PROTOCOL,
        runs={
            "eight": _receipt(tmp_path, world_size=8),
            "four": _receipt(tmp_path, world_size=4),
        },
        warmup_updates=0,
    )
    assert result["pooled_identical_domain"] is None


def test_auditor_rejects_parameter_and_token_accounting_tamper(
    tmp_path: Path,
) -> None:
    """Fail closed when the trace or DDP parameter size disagrees."""
    module = _load_module()
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text(encoding="ascii"))
    value["ddp_logging_data"]["total_parameter_size_bytes"] += 4
    receipt.write_text(_canonical(value), encoding="ascii")
    with pytest.raises(module.MfuAuditError, match="parameter bytes differ"):
        module.audit_mfu(
            protocol_path=PROTOCOL,
            runs={"tampered": receipt},
            warmup_updates=0,
        )
