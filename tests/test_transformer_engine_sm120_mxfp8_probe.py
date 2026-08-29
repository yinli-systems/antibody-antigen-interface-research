"""Test the non-authorizing RTX 5090 MXFP8 capability observation."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/probe_transformer_engine_sm120_mxfp8.py"
SLURM = ROOT / "slurm/transformer_engine_sm120_mxfp8_probe.sbatch"
SPECIFICATION = importlib.util.spec_from_file_location("sm120_mxfp8_probe", PROBE)
assert SPECIFICATION is not None and SPECIFICATION.loader is not None
MODULE = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(MODULE)


def test_relative_l2_reports_raw_observation_without_threshold() -> None:
    reference = torch.tensor([3.0, 4.0])
    observed = torch.tensor([0.0, 4.0])
    assert MODULE._relative_l2(observed, reference) == pytest.approx(0.6)


def test_probe_is_ascii_syntax_valid_and_fail_closed() -> None:
    source = PROBE.read_text(encoding="ascii")
    ast.parse(source)
    assert "MXFP8BlockScaling(fp8_format=Format.E4M3)" in source
    assert "is_mxfp8_available(return_reason=True)" in source
    assert '"mxfp8_parity": False' in source
    assert '"four_gpu_training": False' in source
    assert '"production_pretraining": False' in source
    assert '"scientific_claim": False' in source
    assert "no tolerance was selected" in source
    assert "not an attention, numerical-parity, throughput, MFU" in source


def test_job_is_bash_valid_source_bound_and_replays_overlay_hashes() -> None:
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:1" in source
    assert "SOURCE_COMMIT" in source
    assert "BASE_PYTHON_SHA256" in source
    assert "OVERLAY_FILE_MANIFEST_SHA256" in source
    assert "PREREQUISITE_RECEIPT_SHA256" in source
    assert "CONTROLLER_SHA256" in source
    assert "overlay entry SHA-256 differs" in source
    assert '[[ ! -e "${OUTPUT_ROOT}"' in source
    assert '[[ ! -e "${PARTIAL}"' in source
    assert 'mv "${PARTIAL}" "${OUTPUT_ROOT}"' in source
    assert "probe requires one scheduler GPU" in source
    assert "transformer_engine_sm120_mxfp8_probe=PASS" in source
