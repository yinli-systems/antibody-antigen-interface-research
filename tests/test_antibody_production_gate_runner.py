"""Validate the isolated production-corpus recovery runner contract."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_antibody_pretraining_production_gate.py"


def _load_runner():
    for item in (ROOT / "scripts", ROOT / "src"):
        value = str(item)
        if value not in sys.path:
            sys.path.insert(0, value)
    specification = importlib.util.spec_from_file_location(
        "production_gate_runner_test", RUNNER
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _load_mode_contract() -> dict[str, object]:
    tree = ast.parse(RUNNER.read_text(encoding="ascii"))
    required_assignments = {
        "RECOVERY_RUN_FORMAT",
        "RECOVERY_GATE_FORMAT",
        "PRETRAINING_RUN_FORMAT",
        "PRETRAINING_GATE_FORMAT",
        "RECOVERY_AUTHORIZATION",
        "PRETRAINING_AUTHORIZATION",
    }
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in required_assignments
            for target in node.targets
        ):
            body.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_mode_contract":
            body.append(node)
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace


def test_production_gate_runner_is_ascii_and_syntax_valid() -> None:
    source = RUNNER.read_text(encoding="ascii")
    ast.parse(source)
    assert "load_production_corpus(" in source
    assert "build_indexed_update_plan(" in source


def test_production_gate_is_fail_closed_and_recovery_bound() -> None:
    source = RUNNER.read_text(encoding="ascii")
    assert 'world_size != local_world_size' in source
    assert 'partition_update_indices(' in source
    assert 'output.loss_sum * world_size / target_count' in source
    assert 'verify_array_file_hashes=True' in source
    assert 'verify_array_file_hashes=False' in source
    assert 'dist.all_gather_object(rank_hosts, socket.gethostname())' in source
    assert '"array_file_hashes_verified_by_local_rank_zero": True' in source
    assert '"maximum_open_arrays": args.maximum_open_arrays' in source
    assert '"minimum_soft_open_file_limit": (' in source
    assert "Soft open-file limit is below the frozen minimum" in source
    assert "Open-file limit does not preserve descriptor headroom" in source
    assert '"production_pretraining": False' in source
    assert '"scale_out": False' in source
    assert '"scientific_claim": False' in source
    assert "Gate protocol identity or authorization differs" in source
    assert "Launch input evidence differs from the gate protocol" in source
    assert "Resume provenance differs from this gate" in source
    assert "_load_authenticated_dcp(" in source
    assert "_save_authenticated_dcp(" in source
    assert "_restore_rng_checkpoint(" in source
    assert "_save_rng_checkpoint(" in source
    assert "torch.use_deterministic_algorithms(True, warn_only=False)" in source
    assert "Audited OAS loader and exact-recovery gate only" in source


def test_production_gate_streams_only_training_fields() -> None:
    source = RUNNER.read_text(encoding="ascii")
    for field in (
        '"chain_types.npy"',
        '"lengths.npy"',
        '"sample_keys.npy"',
        '"tokens.npy"',
    ):
        assert field in source
    assert '"split": "train"' in source
    assert '"sabdab2_heldout_accessed": False' in source
    assert '"downstream_labels_used": False' in source
    assert '"grouped_update_read": True' in source


def test_authorization_modes_are_exact_and_non_overlapping() -> None:
    namespace = _load_mode_contract()
    mode_contract = namespace["_mode_contract"]

    recovery = mode_contract("recovery_gate")
    production = mode_contract("production_pretraining")

    assert recovery[:3] == (
        "antibody_production_loader_recovery_gate_v1",
        "antibody_production_loader_recovery_run_v1",
        {
            "loader_recovery_gate": True,
            "production_pretraining": False,
            "scale_out": False,
            "scientific_claim": False,
        },
    )
    assert production[:3] == (
        "antibody_four_gpu_pretraining_segment_gate_v1",
        "antibody_four_gpu_pretraining_segment_run_v1",
        {
            "configuration_promotion": True,
            "downstream_transfer": False,
            "production_pretraining": True,
            "scientific_claim": False,
        },
    )
    assert set(recovery[2]) != set(production[2])
    assert "official-test" in production[3]
    assert "SOTA claim" in production[3]


def test_unknown_authorization_mode_fails_closed() -> None:
    mode_contract = _load_mode_contract()["_mode_contract"]
    with pytest.raises(ValueError, match="Unknown authorization mode"):
        mode_contract("production")


def test_canonical_resume_evidence_rejects_reformatted_json(tmp_path: Path) -> None:
    runner = _load_runner()
    path = tmp_path / "receipt.json"
    path.write_text('{"b": 2, "a": 1}\n', encoding="ascii")
    with pytest.raises(RuntimeError, match="canonical JSON object"):
        runner._load_canonical_json(path, "Resume receipt")

    path.write_bytes(runner._canonical_json_bytes({"b": 2, "a": 1}))
    value, digest = runner._load_canonical_json(path, "Resume receipt")
    assert value == {"a": 1, "b": 2}
    assert len(digest) == 64


def test_campaign_stop_is_bound_to_accumulated_residue_tokens() -> None:
    source = RUNNER.read_text(encoding="ascii")
    assert "consumed >= args.campaign_token_limit" in source
    assert '"executed_steps": executed_steps' in source
    assert '"next_update": start_update + executed_steps' in source
    assert "Segment would execute zero optimizer updates" in source
    assert "Resume receipt differs from this gate" in source
    assert "Resume checkpoint differs from the authenticated receipt" in source
    assert "Resume RNG state differs from the authenticated receipt" in source
