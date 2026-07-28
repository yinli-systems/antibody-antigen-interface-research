import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_efficiency_gate_requires_ordered_fixed_work_scaling() -> None:
    protocol = json.loads(
        (ROOT / "protocols/antibody_efficiency_gate_v1.json").read_text(
            encoding="ascii"
        )
    )
    assert protocol["format_version"] == "antibody_efficiency_gate_v1"
    assert protocol["benchmark"]["reference_micro_batch_per_gpu"] == 16
    assert protocol["benchmark"]["fixed_membership_fields"] == [
        "consumed_residue_tokens",
        "residue_tokens",
        "selected_targets",
        "update",
    ]
    assert protocol["scale_gate"] == {
        "16_gpu_minimum_efficiency": 0.85,
        "32_gpu_minimum_efficiency": 0.8,
        "64_gpu_minimum_efficiency": 0.7,
        "fixed_global_work_required": True,
        "maximum_p95_over_p50_step_time": 1.2,
        "required_sequence": [8, 16, 32, 64],
        "scale_only_after_previous_stage_passes": True,
    }


def test_efficiency_gate_does_not_authorize_unverified_scale_or_science() -> None:
    protocol = json.loads(
        (ROOT / "protocols/antibody_efficiency_gate_v1.json").read_text(
            encoding="ascii"
        )
    )
    assert protocol["authorization"] == {
        "benchmark_16_gpu": True,
        "benchmark_32_gpu": True,
        "benchmark_64_gpu": False,
        "production_pretraining": False,
        "systems_only_1_3b_pilot": False,
    }
    assert protocol["correctness_gate"]["downstream_labels_permitted"] is False
    assert protocol["scientific_gate"]["test_set_model_selection_permitted"] is False
    assert "state-of-the-art" in protocol["claim_boundary"]


def test_research_report_discloses_exploratory_limits() -> None:
    report = (
        ROOT / "docs/ANTIBODY_PRETRAINING_EFFICIENCY_RESEARCH_V1.md"
    ).read_text(encoding="ascii")
    assert "132,873.24" in report
    assert "fails the frozen 25% headroom" in report
    assert "cannot authorize production" in report
    assert "not a verified\n32-GPU allocation" in report
