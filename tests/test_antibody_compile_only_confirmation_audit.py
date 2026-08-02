"""Test the independent compile-only confirmation auditor contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_antibody_compile_only_confirmation.py"


def _load_module():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    specification = importlib.util.spec_from_file_location(
        "compile_only_confirmation_audit",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_compile_only_contract_excludes_failed_fused_and_data_factors() -> None:
    module = _load_module()
    observed = module._expected_optimization("fp32_compiled")
    assert observed["compile_blocks"] is True
    assert observed["compile_prewarm_buckets"] is True
    assert observed["fused_adamw"] is False
    assert observed["pinned_staging"] is False
    assert observed["bucketed_padding"] is False
    assert observed["ddp_communication_dtype"] == "float32"


def test_compile_only_auditor_is_ascii_and_fail_closed() -> None:
    source = SCRIPT.read_text(encoding="ascii")
    assert "confirmation_auditor_sha256" in source
    assert "parent._expected_optimization = _expected_optimization" in source
    assert "finally:" in source
    assert 'result["confirmed_candidate"] = "compile_only"' in source
    assert 'return 0 if result["gate_passed"] else 1' in source
