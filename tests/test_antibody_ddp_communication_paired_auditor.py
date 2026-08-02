import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_ddp_communication_paired_gate.py"


def _module():
    spec = importlib.util.spec_from_file_location("paired_comm_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auditor_is_ascii_and_syntax_valid() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert "len(receipt.get(\"trace\", [])) != 500" in source
    assert '"allocation": True' in source
    assert '"membership": membership_equal' in source
    assert '"numerical_correctness"' in source
    assert '"rank_parity": True' in source
    assert '"replicate_exact"' in source
    assert '"replicate_throughput_range"' in source
    assert '"stability"' in source
    assert '"throughput_each_round"' in source
    assert '"throughput_pooled"' in source
    assert "os.O_EXCL" in source


def test_relative_range_uses_conservative_minimum_denominator() -> None:
    module = _module()
    assert module._relative_range_over_minimum([100.0, 104.0]) == 0.04
    assert module._relative_range_over_minimum([104.0, 100.0]) == 0.04
    with pytest.raises(RuntimeError):
        module._relative_range_over_minimum([0.0, 1.0])


def test_auditor_preserves_failed_gate_without_scale_authorization() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert '"gate_passed": gate_passed' in source
    assert '"exact_recovery": gate_passed' in source
    assert '"pilot_16_gpu": False' in source
    assert '"pilot_32_gpu": False' in source
    assert '"production_pretraining": False' in source
    assert '"scientific_claim": False' in source
    assert "raise RuntimeError(f\"Paired gate failed" not in source


def test_auditor_requires_counterbalanced_order_and_same_job() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert 'paired["benchmark"].get("run_order") != list(RUNS)' in source
    assert "set(job_ids.values()) != {job_id}" in source
    assert 'allocation.get("protocol_sha256") != paired_sha256' in source
    assert 'receipt.get("allocation") != expected_receipt_allocation' in source
