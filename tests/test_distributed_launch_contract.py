#!/usr/bin/env python3
"""Tests for fail-closed synthetic distributed-launch contracts."""

from __future__ import annotations

import json
import importlib
from pathlib import Path
import sys

import pytest
import torch

from distributed_launch_contract import (
    LaunchContractError,
    build_launch_manifest,
    main as launch_manifest_main,
    publish_create_only_json,
    require_safe_label,
    require_sha256,
    safe_environment_snapshot,
    source_tree_digest,
    validate_distributed_identity,
    validate_training_mode,
    verify_file_sha256,
    verify_launch_manifest,
)
from nccl_message_size_sweep import _payload_sizes
from nccl_topology_diagnostic import _tensor_digest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_r2_smoke_reduces_contact_loss_by_valid_complex_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = importlib.import_module("frontier_ddp_smoke")
    captured: dict[str, object] = {}

    def fake_exact_mean(
        local_sum: torch.Tensor,
        local_count: int,
        group: object = None,
    ) -> torch.Tensor:
        captured["local_sum"] = local_sum
        captured["local_count"] = local_count
        captured["group"] = group
        return local_sum / local_count

    monkeypatch.setattr(smoke, "exact_distributed_mean", fake_exact_mean)
    logits = torch.zeros(3, 2, 2)
    targets = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    mask = torch.ones_like(targets, dtype=torch.bool)
    observed, details = smoke._exact_balanced_contact_mean(
        logits,
        targets,
        mask,
    )
    assert captured["local_sum"] is details["loss_sum"]
    assert captured["local_count"] == 2
    torch.testing.assert_close(observed, details["loss"])


def test_external_file_digest_is_mandatory_and_exact(tmp_path: Path) -> None:
    artifact = tmp_path / "runtime.json"
    artifact.write_bytes(b'{"locked":true}\n')
    digest = __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()
    assert verify_file_sha256(
        artifact,
        digest,
        field="runtime_manifest_sha256",
    ) == digest
    with pytest.raises(LaunchContractError, match="mismatch"):
        verify_file_sha256(
            artifact,
            "0" * 64,
            field="runtime_manifest_sha256",
        )
    with pytest.raises(LaunchContractError, match="64-character"):
        require_sha256("not-a-digest", field="source_sha256")


def test_source_tree_digest_binds_names_and_bytes(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    (root / "a.py").write_text("value = 1\n", encoding="utf-8")
    (root / "b.py").write_text("value = 2\n", encoding="utf-8")
    first = source_tree_digest(root, ("a.py", "b.py"))
    (root / "b.py").write_text("value = 3\n", encoding="utf-8")
    second = source_tree_digest(root, ("a.py", "b.py"))
    assert first != second
    with pytest.raises(LaunchContractError, match="Invalid source-relative"):
        source_tree_digest(root, ("../outside.py",))
    with pytest.raises(LaunchContractError, match="unique"):
        source_tree_digest(root, ("a.py", "a.py"))


def test_rank_identity_accepts_reviewed_64_gpu_topology() -> None:
    receipt = validate_distributed_identity(
        rank=63,
        local_rank=7,
        world_size=64,
        visible_device_count=8,
        minimum_world_size=2,
        expected_world_size=64,
        environment={
            "SLURM_JOB_NUM_NODES": "8",
            "LOCAL_WORLD_SIZE": "8",
        },
    )
    assert receipt["qos_max_nodes"] == 8
    assert receipt["qos_max_gpus"] == 64
    assert receipt["slurm_node_count"] == 8


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"world_size": 65, "expected_world_size": 65}, "QoS limit"),
        ({"rank": 8}, "RANK"),
        ({"local_rank": 8}, "LOCAL_RANK"),
        ({"visible_device_count": 9}, "per-node limit"),
        ({"expected_world_size": 7}, "externally declared"),
    ],
)
def test_rank_identity_fails_closed(
    overrides: dict[str, int],
    message: str,
) -> None:
    arguments = {
        "rank": 0,
        "local_rank": 0,
        "world_size": 8,
        "visible_device_count": 8,
        "minimum_world_size": 2,
        "expected_world_size": 8,
        "environment": {
            "SLURM_JOB_NUM_NODES": "1",
            "LOCAL_WORLD_SIZE": "8",
        },
    }
    arguments.update(overrides)
    with pytest.raises(LaunchContractError, match=message):
        validate_distributed_identity(**arguments)


def test_rank_identity_rejects_scheduler_product_mismatch() -> None:
    with pytest.raises(LaunchContractError, match="nodes times"):
        validate_distributed_identity(
            rank=0,
            local_rank=0,
            world_size=8,
            visible_device_count=8,
            minimum_world_size=2,
            expected_world_size=8,
            environment={
                "SLURM_JOB_NUM_NODES": "2",
                "LOCAL_WORLD_SIZE": "8",
            },
        )


def test_training_mode_is_explicit_and_never_implies_fsdp2() -> None:
    receipt = validate_training_mode(
        parallelism="ddp",
        precision="bfloat16",
        cuda_available=True,
        bf16_supported=True,
        require_cuda=True,
    )
    assert receipt["parallelism"] == "ddp"
    assert receipt["gradient_reduction"] == "ddp_average"
    assert receipt["metric_claim"] == "forbidden"
    assert receipt["efficiency_claim"] == "forbidden"
    with pytest.raises(LaunchContractError, match="FSDP2"):
        validate_training_mode(
            parallelism="fsdp2",
            precision="bfloat16",
            cuda_available=True,
            bf16_supported=True,
            require_cuda=True,
        )
    with pytest.raises(LaunchContractError, match="does not support bfloat16"):
        validate_training_mode(
            parallelism="ddp",
            precision="bfloat16",
            cuda_available=True,
            bf16_supported=False,
            require_cuda=True,
        )


def test_safe_environment_snapshot_uses_an_allowlist() -> None:
    snapshot = safe_environment_snapshot(
        {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_PRIVATE_TOKEN": "must-not-leak",
            "AWS_SECRET_ACCESS_KEY": "must-not-leak",
        }
    )
    assert snapshot["NCCL_P2P_DISABLE"] == "1"
    assert "NCCL_PRIVATE_TOKEN" not in snapshot
    assert "AWS_SECRET_ACCESS_KEY" not in snapshot
    assert "must-not-leak" not in json.dumps(snapshot)


def test_create_only_publication_preserves_existing_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    publish_create_only_json(output, {"generation": 1})
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_create_only_json(output, {"generation": 2})
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".result.json.tmp-*"))


def test_launch_manifest_binds_runtime_source_topology_and_mode(
    tmp_path: Path,
) -> None:
    source_sha256 = "1" * 64
    runtime_sha256 = "2" * 64
    payload = build_launch_manifest(
        source_sha256=source_sha256,
        runtime_manifest_sha256=runtime_sha256,
        runtime_verification_sha256="3" * 64,
        nodes=8,
        gpus_per_node=8,
        processes_per_node=8,
        world_size=64,
        backend="nccl",
        parallelism="ddp",
        precision="bfloat16",
        scope="synthetic_ddp_smoke",
    )
    path = tmp_path / "launch-manifest.json"
    publish_create_only_json(path, payload)
    path.chmod(0o444)
    file_sha256 = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    verified = verify_launch_manifest(
        path,
        expected_file_sha256=file_sha256,
        expected_source_sha256=source_sha256,
        expected_world_size=64,
        expected_backend="nccl",
        expected_parallelism="ddp",
        expected_precision="bfloat16",
    )
    assert verified["topology"]["qos_max_nodes"] == 8
    assert verified["topology"]["qos_max_gpus"] == 64
    with pytest.raises(LaunchContractError, match="topology differs"):
        verify_launch_manifest(
            path,
            expected_file_sha256=file_sha256,
            expected_source_sha256=source_sha256,
            expected_world_size=63,
            expected_backend="nccl",
            expected_parallelism="ddp",
            expected_precision="bfloat16",
        )
    with pytest.raises(LaunchContractError, match="execution mode differs"):
        verify_launch_manifest(
            path,
            expected_file_sha256=file_sha256,
            expected_source_sha256=source_sha256,
            expected_world_size=64,
            expected_backend="nccl",
            expected_parallelism="ddp",
            expected_precision="float32",
        )
    link = tmp_path / "launch-manifest-link.json"
    link.symlink_to(path.name)
    with pytest.raises(LaunchContractError, match="symlink traversal"):
        verify_launch_manifest(
            link,
            expected_file_sha256=file_sha256,
            expected_source_sha256=source_sha256,
            expected_world_size=64,
            expected_backend="nccl",
            expected_parallelism="ddp",
            expected_precision="bfloat16",
        )


def test_launch_manifest_explicitly_supports_fsdp2_engineering() -> None:
    payload = build_launch_manifest(
        source_sha256="1" * 64,
        runtime_manifest_sha256="2" * 64,
        runtime_verification_sha256="3" * 64,
        nodes=1,
        gpus_per_node=8,
        processes_per_node=8,
        world_size=8,
        backend="nccl",
        parallelism="fsdp2",
        precision="bfloat16",
        scope="synthetic_fsdp2_dcp_smoke",
    )
    assert payload["execution"] == {
        "backend": "nccl",
        "parallelism": "fsdp2",
        "precision": "bfloat16",
    }
    assert "Synthetic correctness" in payload["claim_boundary"]


def test_launch_manifest_rejects_unreviewed_96_gpu_plan() -> None:
    with pytest.raises(LaunchContractError, match="nodes"):
        build_launch_manifest(
            source_sha256="1" * 64,
            runtime_manifest_sha256="2" * 64,
            runtime_verification_sha256="3" * 64,
            nodes=12,
            gpus_per_node=8,
            processes_per_node=8,
            world_size=96,
            backend="nccl",
            parallelism="ddp",
            precision="bfloat16",
            scope="synthetic_ddp_smoke",
        )


def test_launch_manifest_cli_requires_passed_runtime_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_sha256 = "2" * 64
    verification = tmp_path / "runtime-verification.json"
    verification.write_text(
        json.dumps({"valid": True, "file_sha256": runtime_sha256}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "launch-manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "distributed_launch_contract.py",
            "--output",
            str(output),
            "--source-sha256",
            "1" * 64,
            "--runtime-manifest-sha256",
            runtime_sha256,
            "--runtime-verification",
            str(verification),
            "--nodes",
            "1",
            "--gpus-per-node",
            "8",
            "--processes-per-node",
            "8",
            "--world-size",
            "8",
            "--backend",
            "nccl",
            "--parallelism",
            "ddp",
            "--precision",
            "bfloat16",
            "--scope",
            "synthetic_ddp_smoke",
        ],
    )
    assert launch_manifest_main() == 0
    assert output.is_file()
    assert output.stat().st_mode & 0o222 == 0

    failed_receipt = tmp_path / "failed-runtime-verification.json"
    failed_receipt.write_text(
        json.dumps({"valid": False, "file_sha256": runtime_sha256}) + "\n",
        encoding="utf-8",
    )
    output_index = sys.argv.index("--output") + 1
    sys.argv[output_index] = str(tmp_path / "second-launch-manifest.json")
    runtime_index = sys.argv.index("--runtime-verification") + 1
    sys.argv[runtime_index] = str(failed_receipt)
    with pytest.raises(LaunchContractError, match="not valid"):
        launch_manifest_main()


@pytest.mark.parametrize(
    "label",
    ("../escape", "/absolute", "UPPERCASE", "", "contains space"),
)
def test_diagnostic_labels_are_path_safe(label: str) -> None:
    with pytest.raises(LaunchContractError, match="must match"):
        require_safe_label(label, field="mode")


def test_message_sweep_caps_single_payload_at_one_gibibyte() -> None:
    assert _payload_sizes(30, 30) == [1 << 30]
    with pytest.raises(ValueError, match="cannot exceed 30"):
        _payload_sizes(10, 31)


def test_topology_state_digest_binds_tensor_name_shape_dtype_and_bytes() -> None:
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    original = _tensor_digest([("weight", tensor)])
    assert original == _tensor_digest([("weight", tensor.clone())])
    assert original != _tensor_digest([("bias", tensor)])
    assert original != _tensor_digest([("weight", tensor.to(torch.float64))])
    changed = tensor.clone()
    changed[0, 1] = 3.0
    assert original != _tensor_digest([("weight", changed)])


def test_slurm_launchers_are_fail_closed_and_do_not_dump_all_nccl_values() -> None:
    launchers = (
        "slurm/frontier_ddp_smoke.sbatch",
        "slurm/frontier_r2_smoke.sbatch",
        "slurm/nccl_backward_isolation.sbatch",
        "slurm/nccl_forward_collective_ddp_isolation.sbatch",
        "slurm/nccl_message_size_sweep.sbatch",
        "slurm/nccl_topology_diagnostic.sbatch",
    )
    for relative in launchers:
        content = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        assert "set -euo pipefail" in content
        assert "umask 077" in content
        assert "FRONTIER_SOURCE_SHA256" in content
        assert "FRONTIER_RUNTIME_MANIFEST_SHA256" in content
        assert "--expected-file-sha256" in content
        assert "--expected-world-size" in content
        assert "--expected-source-sha256" in content
        assert "--expected-launch-manifest-sha256" in content
        assert "launch_manifest.json" in content
        assert "SLURM_JOB_NUM_NODES" in content
        assert "chmod -R a-w" in content
        assert "env |" not in content
        assert "grep '^NCCL_'" not in content
