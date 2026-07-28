import ast
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_antibody_efficiency_benchmark.py"
ATTEMPT_ONE = (
    ROOT
    / "results/pretraining/antibody_efficiency_benchmark_attempt_001_v1.json"
)
ATTEMPT_TWO = (
    ROOT
    / "results/pretraining/antibody_efficiency_benchmark_attempt_002_v1.json"
)
ATTEMPT_THREE = (
    ROOT
    / "results/pretraining/antibody_efficiency_benchmark_attempt_003_v1.json"
)
ATTEMPT_FOUR = (
    ROOT
    / "results/pretraining/antibody_efficiency_benchmark_attempt_004_v1.json"
)
CUDAGRAPH_DISABLED = (
    ROOT
    / "results/pretraining/antibody_efficiency_benchmark_cudagraph_disabled_v1.json"
)
BUCKET144_BENCHMARK = (
    ROOT / "results/pretraining/antibody_efficiency_benchmark_bucket144_v1.json"
)
DETERMINISTIC_BENCHMARK = (
    ROOT
    / "results/pretraining/antibody_efficiency_benchmark_deterministic_v1.json"
)
PAIRED_BENCHMARK = (
    ROOT / "results/pretraining/antibody_efficiency_benchmark_paired_v1.json"
)
EFFICIENCY_PROTOCOL = ROOT / "protocols/antibody_efficiency_gate_v1.json"
PAIRED_PROTOCOL = ROOT / "protocols/antibody_paired_efficiency_v1.json"


def test_efficiency_benchmark_auditor_is_ascii_and_syntax_valid() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    ast.parse(source)
    assert "len(trace) != 500" in source
    assert '"memory"' in source
    assert '"membership"' in source
    assert '"stability"' in source
    assert '"throughput"' in source
    assert '"--require-disabled-compiled-cudagraphs"' in source
    assert '"--require-enabled-compiled-cudagraphs"' in source
    assert '"--expected-length-bucket-boundaries"' in source
    assert '"--require-deterministic-algorithms"' in source
    assert '"--allocation-receipt"' in source
    assert '"--paired-protocol"' in source
    assert '"paired_allocation"' in source
    assert '"disable_compiled_cudagraphs": (' in source
    assert "require_disabled_compiled_cudagraphs" in source


def test_efficiency_benchmark_auditor_preserves_failed_results() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert '"gate_passed": passed' in source
    assert '"production_pretraining": False' in source
    assert '"recovery_required_before_scale": True' in source
    assert '"scale_16_gpu": False' in source
    assert "os.O_EXCL" in source
    assert "raise RuntimeError(f\"Efficiency benchmark failed" not in source


def test_variable_width_attempt_preserves_stability_failure() -> None:
    import json

    result = json.loads(ATTEMPT_ONE.read_text(encoding="ascii"))
    assert result["label"] == "variable_width_compile_attempt"
    assert result["gate_passed"] is False
    assert result["gates"] == {
        "membership": True,
        "memory": True,
        "stability": False,
        "throughput": True,
    }
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] > 1.29
    assert result["summaries"]["compiled"]["p95_over_p50"] > 1.24
    assert result["authorization"]["scale_16_gpu"] is False


def test_bucketed_lazy_compile_attempt_preserves_late_compile_failure() -> None:
    import json

    result = json.loads(ATTEMPT_TWO.read_text(encoding="ascii"))
    assert result["label"] == "bucketed_lazy_compile_attempt"
    assert result["gate_passed"] is False
    assert result["gates"] == {
        "membership": True,
        "memory": True,
        "stability": False,
        "throughput": False,
    }
    assert result["summaries"]["compiled"]["maximum_step_seconds"] > 7.0
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] < 0.81
    assert result["authorization"]["scale_16_gpu"] is False


def test_prewarm_attempt_preserves_near_threshold_failure() -> None:
    import json

    result = json.loads(ATTEMPT_THREE.read_text(encoding="ascii"))
    assert result["label"] == "prewarmed_bucket_attempt"
    assert result["gate_passed"] is False
    assert result["gates"] == {
        "membership": True,
        "memory": True,
        "stability": False,
        "throughput": False,
    }
    assert 1.24 < result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] < 1.25
    assert result["summaries"]["compiled"]["p95_over_p50"] > 1.15
    assert result["authorization"]["scale_16_gpu"] is False


def test_low_frequency_telemetry_attempt_preserves_exact_failure() -> None:
    import json

    result = json.loads(ATTEMPT_FOUR.read_text(encoding="ascii"))
    assert result["label"] == "low_frequency_telemetry_attempt"
    assert result["gate_passed"] is False
    assert result["gates"] == {
        "membership": True,
        "memory": True,
        "stability": False,
        "throughput": True,
    }
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] > 1.25
    assert 1.10 < result["summaries"]["compiled"][
        "p95_over_p50"
    ] < 1.101
    assert result["authorization"]["scale_16_gpu"] is False


def test_cudagraph_disabled_candidate_preserves_stability_failure() -> None:
    import json

    result = json.loads(CUDAGRAPH_DISABLED.read_text(encoding="ascii"))
    assert result["label"] == "compiled_cudagraph_disabled_candidate"
    assert result["required_disabled_compiled_cudagraphs"] is True
    assert result["gate_passed"] is False
    assert result["gates"] == {
        "membership": True,
        "memory": True,
        "stability": False,
        "throughput": True,
    }
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] > 1.28
    assert result["summaries"]["compiled"]["p95_over_p50"] > 1.12
    assert result["authorization"]["scale_16_gpu"] is False


def test_bucket144_candidate_passes_all_frozen_eight_gpu_gates() -> None:
    import json

    result = json.loads(BUCKET144_BENCHMARK.read_text(encoding="ascii"))
    assert result["label"] == "narrow_bucket_compiled_candidate"
    assert result["expected_length_bucket_boundaries"] == [96, 128, 144, 194]
    assert result["required_disabled_compiled_cudagraphs"] is True
    assert result["gate_passed"] is True
    assert all(result["gates"].values())
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] > 1.30
    assert result["summaries"]["compiled"]["p95_over_p50"] < 1.051
    assert result["summaries"]["compiled"]["maximum_memory_fraction"] < 0.60
    assert result["authorization"] == {
        "production_pretraining": False,
        "recovery_required_before_scale": True,
        "scale_16_gpu": False,
        "scale_32_gpu": False,
    }


def test_deterministic_long_benchmark_preserves_throughput_failure() -> None:
    result = json.loads(DETERMINISTIC_BENCHMARK.read_text(encoding="ascii"))
    assert result["label"] == "deterministic_cudagraph_candidate"
    assert result["required_enabled_compiled_cudagraphs"] is True
    assert result["required_deterministic_algorithms"] is True
    assert result["gate_passed"] is False
    assert result["gates"] == {
        "membership": True,
        "memory": True,
        "stability": True,
        "throughput": False,
    }
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] < 1.0
    assert result["authorization"]["scale_16_gpu"] is False


def test_paired_allocation_requires_same_job_and_hardware(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("efficiency_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    allocation = {
        "format_version": "antibody_paired_allocation_v1",
        "gpu_count": 8,
        "gpu_inventory": [{"uuid": f"GPU-{index}"} for index in range(8)],
        "hostname": "node-a",
        "pair_order": "eager,compiled",
        "slurm": {
            "job_id": "123",
            "job_nodelist": "node-a",
            "node_count": 1,
        },
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "topology": "GPU0 GPU1",
    }
    allocation_path = tmp_path / "allocation.json"
    allocation_path.write_text(json.dumps(allocation), encoding="ascii")
    receipt_allocation = {
        "hostname": "node-a",
        "slurm_job_id": "123",
        "slurm_job_nodelist": "node-a",
    }
    receipts = {
        "eager": {"allocation": receipt_allocation},
        "compiled": {"allocation": receipt_allocation},
    }
    protocol_sha256 = module._sha256(EFFICIENCY_PROTOCOL)
    result = module._paired_allocation(
        allocation_path,
        PAIRED_PROTOCOL,
        receipts,
        {"eager": "123", "compiled": "123"},
        protocol_sha256,
    )
    assert result["hostname"] == "node-a"
    assert result["pair_order"] == "eager,compiled"
    with pytest.raises(RuntimeError, match="share"):
        module._paired_allocation(
            allocation_path,
            PAIRED_PROTOCOL,
            receipts,
            {"eager": "123", "compiled": "124"},
            protocol_sha256,
        )


def test_paired_deterministic_candidate_passes_all_efficiency_gates() -> None:
    result = json.loads(PAIRED_BENCHMARK.read_text(encoding="ascii"))
    assert result["label"] == (
        "paired_same_allocation_deterministic_cudagraph_candidate"
    )
    assert result["gate_passed"] is True
    assert all(result["gates"].values())
    assert result["runs"]["eager"]["job_id"] == "1336684"
    assert result["runs"]["compiled"]["job_id"] == "1336684"
    assert result["paired_allocation"]["hostname"] == "wqd10nah10g2"
    assert len(result["paired_allocation"]["gpu_uuids"]) == 8
    assert result["summaries"]["compiled"][
        "throughput_relative_to_eager"
    ] > 1.33
    assert result["summaries"]["compiled"]["p95_over_p50"] < 1.09
    assert result["summaries"]["compiled"]["maximum_memory_fraction"] < 0.60
    assert result["authorization"] == {
        "production_pretraining": False,
        "recovery_required_before_scale": True,
        "scale_16_gpu": False,
        "scale_32_gpu": False,
    }
