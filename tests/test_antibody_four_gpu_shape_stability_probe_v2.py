"""Test the non-authorizing four-GPU hybrid-shape probe."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts/freeze_antibody_four_gpu_shape_stability_probe_v2.py"
AUDITOR = ROOT / "scripts/audit_antibody_four_gpu_shape_stability_probe_v2.py"
PIPELINE = ROOT / "slurm/antibody_four_gpu_shape_stability_probe_v2.sbatch"


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


def _base_gate() -> dict[str, object]:
    return {
        "execution": {
            "backend": "ddp",
            "length_bucket_boundaries": "96,128,160,194",
            "optimization": {
                "bucketed_padding": True,
                "compile_blocks": True,
                "compile_prewarm_buckets": True,
                "ddp_bucket_cap_mb": 25,
                "deterministic_algorithms": True,
                "disable_compiled_cudagraphs": True,
                "fused_adamw": True,
                "length_bucketing": True,
                "pinned_staging": True,
            },
            "world_size": 4,
        },
        "runs": {},
    }


def _config(rate: float, *, eligible: bool = True) -> dict[str, object]:
    return {
        "eligible": eligible,
        "pooled": {"residue_tokens_per_second": rate},
    }


def test_sources_are_ascii_syntax_valid_and_pipeline_is_valid() -> None:
    for path in (FREEZER, AUDITOR):
        ast.parse(path.read_text(encoding="ascii"))
    subprocess.run(["bash", "-n", str(PIPELINE)], check=True)


def test_candidate_gates_change_only_compiled_shape_boundaries() -> None:
    freezer = _load("shape_probe_v2_freezer_gate_test", FREEZER)
    expected = {
        "static_194": "194",
        "hybrid_160_194": "160,194",
        "hybrid_128_194": "128,194",
    }
    for candidate, boundaries in expected.items():
        gate = freezer._candidate_gate(_base_gate(), candidate)
        assert gate["execution"]["length_bucket_boundaries"] == boundaries
        assert gate["runs"]["continuous"] == {
            "batch_quantum_sequences": 256,
            "checkpoint": False,
            "micro_batch_per_gpu": 16,
            "preset": "abf_650m",
            "resume": False,
            "run_seed": 17,
            "steps": 300,
            "target_residue_tokens": 131072,
        }


def test_v2_preserves_original_stability_thresholds_and_five_percent_gate() -> None:
    freezer = _load("shape_probe_v2_freezer_threshold_test", FREEZER)
    assert freezer.ACCEPTANCE["minimum_approximate_6n_mfu"] == 0.35
    assert freezer.ACCEPTANCE["maximum_p95_over_p50_step_time"] == 1.10
    assert freezer.ACCEPTANCE["maximum_memory_fraction"] == 0.80
    assert freezer.ACCEPTANCE["maximum_replicate_throughput_relative_range"] == 0.05
    assert (
        freezer.ACCEPTANCE["minimum_candidate_throughput_relative_to_static_194"]
        == 1.05
    )
    assert not any(freezer.TEST_ACCESS.values())
    assert freezer.AUTHORIZATION["production_pretraining"] is False


def test_selection_requires_eligibility_and_five_percent_improvement() -> None:
    auditor = _load("shape_probe_v2_auditor_selection_test", AUDITOR)
    configurations = {
        "static_194": _config(100.0),
        "hybrid_160_194": _config(106.0),
        "hybrid_128_194": _config(108.0),
    }
    selected, decisions = auditor._select_candidate(
        configurations,
        minimum_relative_throughput=1.05,
    )
    assert selected == "hybrid_128_194"
    assert decisions["hybrid_160_194"]["passed_relative_throughput"] is True
    assert decisions["hybrid_128_194"]["throughput_relative_to_static_194"] == 1.08

    configurations["hybrid_128_194"] = _config(120.0, eligible=False)
    configurations["hybrid_160_194"] = _config(104.9)
    selected, _ = auditor._select_candidate(
        configurations,
        minimum_relative_throughput=1.05,
    )
    assert selected is None


def test_pipeline_is_symmetric_six_run_real_oas_probe() -> None:
    source = PIPELINE.read_text(encoding="ascii")
    expected = [
        "run_candidate static_194_r1 static_194 194",
        "run_candidate hybrid_160_194_r1 hybrid_160_194 160,194",
        "run_candidate hybrid_128_194_r1 hybrid_128_194 128,194",
        "run_candidate hybrid_128_194_r2 hybrid_128_194 128,194",
        "run_candidate hybrid_160_194_r2 hybrid_160_194 160,194",
        "run_candidate static_194_r2 static_194 194",
    ]
    assert "#SBATCH --gres=gpu:4" in source
    assert [source.index(item) for item in expected] == sorted(
        source.index(item) for item in expected
    )
    assert "--steps 300" in source
    assert "--skip-checkpoint" in source
    assert "evaluate_antibody_production_checkpoint.py" not in source
    assert "ABF_RECOVERY_AUDIT" in source


def test_source_closure_binds_v2_and_all_imported_execution_files() -> None:
    freezer = _load("shape_probe_v2_freezer_source_test", FREEZER)
    assert set(freezer.SOURCE_FILES) == {
        "scripts/audit_antibody_four_gpu_efficiency_gate.py",
        "scripts/audit_antibody_four_gpu_shape_stability_probe.py",
        "scripts/audit_antibody_four_gpu_shape_stability_probe_v2.py",
        "scripts/audit_antibody_production_batch_efficiency.py",
        "scripts/freeze_antibody_four_gpu_efficiency_gate.py",
        "scripts/freeze_antibody_four_gpu_recovery_gate.py",
        "scripts/freeze_antibody_four_gpu_shape_stability_probe.py",
        "scripts/freeze_antibody_four_gpu_shape_stability_probe_v2.py",
        "scripts/run_antibody_pretraining_production_gate.py",
        "slurm/antibody_four_gpu_shape_stability_probe_v2.sbatch",
    }


def test_result_scope_cannot_authorize_training() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert '"configuration_promotion": False' in source
    assert '"production_pretraining": False' in source
    assert '"scientific_claim": False' in source
    assert '"full_efficiency_gate_candidate": passed' in source
    assert '"selected_candidate_for_full_efficiency_gate"' in source
