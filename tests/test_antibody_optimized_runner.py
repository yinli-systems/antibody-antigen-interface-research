import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_antibody_pretraining_optimized.py"
SLURM = ROOT / "slurm/antibody_pretraining_efficiency_pilot.sbatch"
PAIRED_SLURM = ROOT / "slurm/antibody_pretraining_paired_efficiency.sbatch"
ALLOCATION_CAPTURE = ROOT / "scripts/capture_antibody_allocation.py"
PAIRED_PROTOCOL = ROOT / "protocols/antibody_paired_efficiency_v1.json"


def test_optimized_runner_is_ascii_and_syntax_valid() -> None:
    source = RUNNER.read_text(encoding="ascii")
    ast.parse(source)
    assert 'options = {"triton.cudagraphs": False}' in source
    assert "block.compile(backend=\"inductor\", fullgraph=True, options=options)" in source
    assert "gradient_as_bucket_view=True" in source
    assert "static_graph=True" in source
    assert "fused=fused" in source


def test_optimized_receipt_binds_every_efficiency_switch() -> None:
    source = RUNNER.read_text(encoding="ascii")
    for field in (
        "bucketed_padding",
        "compile_blocks",
        "compile_prewarm_buckets",
        "disable_compiled_cudagraphs",
        "ddp_bucket_cap_mb",
        "deterministic_algorithms",
        "fsdp_output_dtype",
        "fused_adamw",
        "length_bucketing",
        "pinned_staging",
    ):
        assert f'"{field}"' in source
    assert '"launch_argv"' in source
    assert '"compile_prewarm_trace"' in source
    assert '"membership_sha256"' in source
    assert '"peak_allocated_memory_mib"' in source
    assert '"telemetry_interval_seconds"' in source
    assert '"production_pretraining": False' in source
    assert '"scientific_claim": False' in source
    assert '"slurm_job_id"' in source
    assert '"slurm_job_nodelist"' in source
    assert "socket.gethostname()" in source


def test_resume_state_binds_batch_and_bucket_geometry() -> None:
    source = RUNNER.read_text(encoding="ascii")
    expected_start = source.index("expected = {")
    expected = source[
        expected_start : source.index("trace: list", expected_start)
    ]
    trainer_start = source.index("trainer_state = {")
    trainer = source[
        trainer_start : source.index("checkpoint_format", trainer_start)
    ]
    for field in (
        "batch_quantum",
        "length_bucket_boundaries",
        "micro_batch_per_gpu",
        "target_residue_tokens",
    ):
        assert f'"{field}"' in expected
        assert f'"{field}"' in trainer


def test_deterministic_candidate_fails_on_unsupported_algorithms() -> None:
    source = RUNNER.read_text(encoding="ascii")
    assert "torch.use_deterministic_algorithms(" in source
    assert "warn_only=False" in source
    assert "torch.backends.cudnn.deterministic = args.deterministic_algorithms" in source
    assert "torch.backends.cudnn.benchmark = False" in source
    assert "random.getstate()" in source
    assert "np.random.get_state()" in source
    assert "torch.get_rng_state()" in source
    assert "torch.cuda.get_rng_state(device)" in source
    assert "_restore_rng_checkpoint(" in source
    assert '"checkpoint_rng_manifest_sha256"' in source
    assert '"resume_loaded_model_sha256"' in source
    assert '"resume_loaded_optimizer_sha256"' in source
    assert '"checkpoint_payload_model_sha256"' in source
    assert '"checkpoint_payload_optimizer_sha256"' in source
    assert '"resume_payload_model_sha256"' in source
    assert '"resume_payload_optimizer_sha256"' in source
    assert '"resume_rng_manifest_sha256"' in source
    main_start = source.index("def main()")
    assert source.index(") = _load_authenticated_dcp(", main_start) < source.index(
        "compile_prewarm_trace:", main_start
    )


def test_length_bucketing_preserves_membership_by_construction() -> None:
    source = RUNNER.read_text(encoding="ascii")
    assert "np.lexsort" in source
    assert "Length bucketing changed update membership" in source
    assert "indices=plan.indices" in source
    assert "args.bucketed_padding or args.length_bucketing" in source


def test_efficiency_slurm_exposes_pinned_optimization_flags() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "run_antibody_pretraining_optimized.py" in source
    assert "ABF_COMPILE_BLOCKS" in source
    assert "ABF_DISABLE_COMPILED_CUDAGRAPHS" in source
    assert "ABF_DETERMINISTIC_ALGORITHMS" in source
    assert "ABF_COMPILE_PREWARM_BUCKETS" in source
    assert "ABF_FUSED_ADAMW" in source
    assert "ABF_PINNED_STAGING" in source
    assert "ABF_BUCKETED_PADDING" in source
    assert "ABF_LENGTH_BUCKETING" in source
    assert "ABF_LENGTH_BUCKET_BOUNDARIES" in source
    assert 'LENGTH_BUCKET_BOUNDARIES:-96,128,160,194' in source
    assert '--length-bucket-boundaries "$LENGTH_BUCKET_BOUNDARIES"' in source
    assert "ABF_RESUME_ROOT" in source
    assert "ABF_TELEMETRY_INTERVAL_SECONDS" in source
    assert 'TELEMETRY_INTERVAL_SECONDS:-30' in source
    assert "TORCHINDUCTOR_CACHE_DIR" in source
    protocol = json.loads(
        (ROOT / "protocols/antibody_efficiency_gate_v1.json").read_text(
            encoding="ascii"
        )
    )
    assert protocol["candidate_optimizations"]["compile"].startswith("regional")


def test_paired_efficiency_launcher_uses_one_allocation_and_frozen_modes() -> None:
    source = PAIRED_SLURM.read_text(encoding="ascii")
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --gres=gpu:8" in source
    assert 'PAIR_ORDER=${ABF_PAIR_ORDER:-eager,compiled}' in source
    assert "eager,compiled|compiled,eager" in source
    assert "capture_antibody_allocation.py" in source
    assert '--pair-order "$PAIR_ORDER"' in source
    assert "--steps 500" in source
    assert "--deterministic-algorithms" in source
    assert "--length-bucket-boundaries 96,128,144,194" in source
    assert "--compile-blocks" in source
    assert "--compile-prewarm-buckets" in source
    assert "--bucketed-padding" in source
    assert source.count("--nproc-per-node=8") == 1


def test_allocation_capture_and_protocol_are_fail_closed() -> None:
    source = ALLOCATION_CAPTURE.read_text(encoding="ascii")
    ast.parse(source)
    assert "os.O_EXCL" in source
    assert "exactly eight visible GPUs" in source
    assert '"nvidia-smi", "topo", "-m"' in source
    assert 'choices=("eager,compiled", "compiled,eager")' in source
    protocol = json.loads(PAIRED_PROTOCOL.read_text(encoding="ascii"))
    assert protocol["allocation"] == {
        "gpu_count": 8,
        "hardware_identity_required": True,
        "node_count": 1,
        "same_slurm_job_required": True,
    }
    assert protocol["benchmark"]["candidate_updates"] == 500
    assert protocol["benchmark"]["deterministic_algorithms"] is True
