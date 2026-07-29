import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/antibody_ddp_communication_paired_gate_v1.json"
CAPTURE = ROOT / "scripts/capture_antibody_communication_allocation.py"
LAUNCHER = ROOT / "slurm/antibody_ddp_communication_paired_gate.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_binds_selected_candidate_and_parent_evidence() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert protocol["format_version"] == (
        "antibody_ddp_communication_paired_gate_v1"
    )
    assert protocol["parent_communication_grid_result_sha256"] == _sha256(
        ROOT
        / "results/pretraining/"
        "antibody_ddp_communication_ablation_attempt_012_v1.json"
    )
    assert protocol["parent_communication_protocol_sha256"] == _sha256(
        ROOT / "protocols/antibody_ddp_communication_optimization_v1.json"
    )
    assert protocol["parent_efficiency_protocol_sha256"] == _sha256(
        ROOT / "protocols/antibody_efficiency_gate_v1.json"
    )
    assert protocol["selected_candidate"] == {
        "batched_grad_copy": True,
        "communication_dtype": "bfloat16",
        "compile_blocks": True,
        "label": "bf16_compress_batched_copy",
    }


def test_protocol_freezes_counterbalanced_replicates_and_fail_closed_gates() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert protocol["benchmark"] == {
        "candidate_warmup_updates": 50,
        "deterministic_algorithms": True,
        "isolated_compile_cache_each_run": True,
        "replicates_per_mode": 2,
        "run_order": [
            "safe_eager_r1",
            "fp32_compiled_r1",
            "bf16_selected_r1",
            "bf16_selected_r2",
            "fp32_compiled_r2",
            "safe_eager_r2",
        ],
        "same_membership_required": True,
        "updates_per_run": 500,
    }
    acceptance = protocol["acceptance"]
    assert acceptance["minimum_selected_throughput_relative_to_safe_reference_each_round"] == 1.25
    assert acceptance["minimum_selected_throughput_relative_to_safe_reference_pooled"] == 1.25
    assert acceptance["maximum_p95_over_p50_step_time_each_run"] == 1.1
    assert acceptance["maximum_replicate_throughput_relative_range"] == 0.05
    assert protocol["authorization"] == {
        "pilot_16_gpu": False,
        "pilot_32_gpu": False,
        "production_pretraining": False,
        "scientific_claim": False,
    }


def test_allocation_capture_is_create_only_and_protocol_bound() -> None:
    source = CAPTURE.read_text(encoding="ascii")
    ast.parse(source)
    assert "os.O_EXCL" in source
    assert "exactly eight visible GPUs" in source
    assert '"protocol_sha256": args.protocol_sha256' in source
    assert '"run_order": list(run_order)' in source
    assert '"nvidia-smi", "topo", "-m"' in source


def test_launcher_runs_all_modes_in_one_allocation() -> None:
    source = LAUNCHER.read_text(encoding="ascii")
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --gres=gpu:8" in source
    assert "capture_antibody_communication_allocation.py" in source
    assert "--steps 500" in source
    assert "--audit-gradient-rank-parity" in source
    assert "--deterministic-algorithms" in source
    assert "--ddp-communication-dtype bfloat16" in source
    assert "--ddp-batched-grad-copy" in source
    assert "--compile-blocks" in source
    assert "--nproc-per-node=8" in source
    assert 'for label in "${run_labels[@]}"' in source
    assert source.count("run_mode \"$label\"") == 1
