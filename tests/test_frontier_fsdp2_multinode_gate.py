"""Validate the heterogeneous-capacity 4x2 FSDP2 gate launcher."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/frontier_fsdp2_multinode_gate.sbatch"


def test_multinode_gate_binds_runtime_on_every_node_and_uses_world_eight() -> None:
    source = LAUNCHER.read_text(encoding="ascii")
    for marker in (
        "#SBATCH --partition=gpu_4090",
        "#SBATCH --nodes=4",
        "#SBATCH --gres=gpu:2",
        "WORLD_SIZE=8",
        "NPROC_PER_NODE=2",
        "runtime_by_node",
        "len(files) != 4",
        '--node-rank="$SLURM_PROCID"',
        '--rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT"',
        "NCCL_SOCKET_IFNAME=${FRONTIER_NCCL_SOCKET_IFNAME:-=bond0}",
        "NCCL_IB_DISABLE=${FRONTIER_NCCL_IB_DISABLE:-1}",
        "NCCL_DEBUG=${FRONTIER_NCCL_DEBUG:-INFO}",
        "NCCL_DEBUG_SUBSYS=${FRONTIER_NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING=1",
        'echo "nccl_socket_ifname=$NCCL_SOCKET_IFNAME"',
        'echo "nccl_ib_disable=$NCCL_IB_DISABLE"',
        "--gpus-per-node 2",
        "--processes-per-node 2",
        '--world-size "$WORLD_SIZE"',
    ):
        assert marker in source


def test_multinode_gate_preserves_scientific_and_performance_boundaries() -> None:
    source = LAUNCHER.read_text(encoding="ascii")
    assert "dcp_smoke|reshard_save|reshard_load|profile" in source
    assert "--warmup-steps 50" in source
    assert "--measured-steps 500" in source
    assert "--batch-size-per-rank 8" in source
    assert "--synthetic-only" in source
    assert "no biological metric or scaling claim" in source
    assert "testdata" not in source


def test_every_multinode_entrypoint_binds_the_launcher_source() -> None:
    for relative in (
        "scripts/frontier_fsdp2_dcp_smoke.py",
        "scripts/frontier_fsdp2_reshard.py",
        "scripts/frontier_fsdp2_profile.py",
    ):
        source = (ROOT / relative).read_text(encoding="ascii")
        assert '"slurm/frontier_fsdp2_multinode_gate.sbatch"' in source
