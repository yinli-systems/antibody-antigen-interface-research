import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_antibody_pretraining_optimized.py"
SLURM = ROOT / "slurm/antibody_pretraining_efficiency_pilot.sbatch"


def test_optimized_runner_is_ascii_and_syntax_valid() -> None:
    source = RUNNER.read_text(encoding="ascii")
    ast.parse(source)
    assert "block.compile(backend=\"inductor\", fullgraph=True)" in source
    assert "gradient_as_bucket_view=True" in source
    assert "static_graph=True" in source
    assert "fused=fused" in source


def test_optimized_receipt_binds_every_efficiency_switch() -> None:
    source = RUNNER.read_text(encoding="ascii")
    for field in (
        "bucketed_padding",
        "compile_blocks",
        "compile_prewarm_buckets",
        "ddp_bucket_cap_mb",
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
    assert '"production_pretraining": False' in source
    assert '"scientific_claim": False' in source


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
    assert "ABF_COMPILE_PREWARM_BUCKETS" in source
    assert "ABF_FUSED_ADAMW" in source
    assert "ABF_PINNED_STAGING" in source
    assert "ABF_BUCKETED_PADDING" in source
    assert "ABF_LENGTH_BUCKETING" in source
    assert "TORCHINDUCTOR_CACHE_DIR" in source
    protocol = json.loads(
        (ROOT / "protocols/antibody_efficiency_gate_v1.json").read_text(
            encoding="ascii"
        )
    )
    assert protocol["candidate_optimizations"]["compile"].startswith("regional")
