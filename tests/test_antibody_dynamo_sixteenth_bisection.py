from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/antibody_dynamo_sixteenth_bisection_v1.json"
PREDECESSOR_PROTOCOL_PATH = (
    ROOT / "protocols/antibody_dynamo_eighth_bisection_v1.json"
)
PREDECESSOR_AUDIT_PATH = (
    ROOT
    / "results/pretraining/antibody_dynamo_eighth_bisection_v1_attempt_001_audit.json"
)
WRAPPER_PATH = ROOT / "scripts/run_antibody_dynamo_sixteenth_bisection.py"
AUDITOR_PATH = ROOT / "scripts/audit_antibody_dynamo_sixteenth_bisection.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WRAPPER = _module(WRAPPER_PATH, "dynamo_sixteenth_bisection_wrapper")
AUDITOR = _module(AUDITOR_PATH, "dynamo_sixteenth_bisection_auditor")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_binds_predecessor_and_every_implementation() -> None:
    protocol = _load(PROTOCOL_PATH)
    predecessor_audit = _load(PREDECESSOR_AUDIT_PATH)
    bindings = {
        "audit_launcher_sha256": (
            ROOT / "slurm/antibody_dynamo_sixteenth_bisection_audit.sbatch"
        ),
        "auditor_sha256": AUDITOR_PATH,
        "base_runner_sha256": ROOT / "scripts/run_antibody_pretraining_optimized.py",
        "communication_protocol_sha256": (
            ROOT / "protocols/antibody_ddp_communication_optimization_v1.json"
        ),
        "efficiency_protocol_sha256": (
            ROOT / "protocols/antibody_efficiency_gate_v1.json"
        ),
        "launcher_sha256": (
            ROOT / "slurm/antibody_dynamo_sixteenth_bisection.sbatch"
        ),
        "model_family_sha256": ROOT / "protocols/antibody_model_family_v1.json",
        "parent_auditor_sha256": (
            ROOT / "scripts/audit_antibody_dynamo_region_bisection.py"
        ),
        "parent_wrapper_sha256": (
            ROOT / "scripts/run_antibody_dynamo_region_bisection.py"
        ),
        "predecessor_auditor_sha256": (
            ROOT / "scripts/audit_antibody_dynamo_eighth_bisection.py"
        ),
        "predecessor_wrapper_sha256": (
            ROOT / "scripts/run_antibody_dynamo_eighth_bisection.py"
        ),
        "parity_evaluation_indices_sha256": (
            ROOT
            / "results/pretraining/antibody_ddp_parity_split_v2/evaluation_indices.npy"
        ),
        "parity_protocol_sha256": (
            ROOT / "protocols/antibody_ddp_scientific_parity_gate_v2.json"
        ),
        "parity_split_receipt_file_sha256": (
            ROOT / "results/pretraining/antibody_ddp_parity_split_v2/receipt.json"
        ),
        "parity_training_indices_sha256": (
            ROOT
            / "results/pretraining/antibody_ddp_parity_split_v2/training_indices.npy"
        ),
        "predecessor_audit_file_sha256": PREDECESSOR_AUDIT_PATH,
        "predecessor_protocol_file_sha256": PREDECESSOR_PROTOCOL_PATH,
        "training_protocol_sha256": ROOT / "protocols/antibody_training_v1.json",
        "wrapper_sha256": WRAPPER_PATH,
    }
    for label, path in bindings.items():
        assert protocol["inputs"][label] == _sha256(path)
    assert protocol["predecessor_bisection"] == {
        "audit_file_sha256": _sha256(PREDECESSOR_AUDIT_PATH),
        "audit_job_id": 1348351,
        "audit_receipt_sha256": predecessor_audit["receipt_sha256"],
        "diagnostic_valid": True,
        "next_disposition": "bisect_failed_eighths_into_sixteenths",
        "next_registered_regions": AUDITOR.EXPECTED_SIXTEENTHS,
        "protocol_file_sha256": _sha256(PREDECESSOR_PROTOCOL_PATH),
        "training_job_id": 1348350,
    }
    assert protocol["measurement"]["promotable_runs"] == []
    assert protocol["selection"]["candidate_promotion_permitted"] is False
    assert set(protocol["authorization"].values()) == {False}


def test_registered_modes_are_exact_sixteenths_and_full_control() -> None:
    protocol = _load(PROTOCOL_PATH)
    assert protocol["modes"] == WRAPPER.MODES
    assert [
        protocol["modes"][f"dynamo_s{index + 1}_gc_off"]["block_indices"]
        for index in range(16)
    ] == AUDITOR.EXPECTED_SIXTEENTHS
    assert protocol["modes"]["dynamo_full_control_gc_off"][
        "block_indices"
    ] == list(range(32))
    assert len(protocol["measurement"]["run_order"]) == 19


def test_wrapper_compiles_only_the_registered_sixteenth(tmp_path: Path) -> None:
    protocol = _load(PROTOCOL_PATH)
    output, flags = WRAPPER._validate_launch(
        protocol,
        mode="dynamo_s11_gc_off",
        arguments=[
            "--output",
            str(tmp_path / "run"),
            "--compile-blocks",
            "--compile-prewarm-buckets",
        ],
        scripts_root=ROOT / "scripts",
    )
    assert output == tmp_path / "run"
    assert flags["block_indices"] == [20, 21]

    class Block:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def compile(self, **kwargs) -> None:
            self.calls.append(kwargs)

    class Model:
        def __init__(self) -> None:
            self.blocks = [Block() for _ in range(32)]

    model = Model()
    WRAPPER._compile_registered_blocks(model, block_indices=flags["block_indices"])
    assert all(block.calls == [] for block in model.blocks[:20])
    assert all(
        block.calls == [{"backend": "eager", "fullgraph": True}]
        for block in model.blocks[20:22]
    )
    assert all(block.calls == [] for block in model.blocks[22:])


def test_wrapper_rejects_unregistered_sixteenth_drift(tmp_path: Path) -> None:
    protocol = _load(PROTOCOL_PATH)
    changed = copy.deepcopy(protocol)
    changed["modes"]["dynamo_s1_gc_off"]["block_indices"] = [0]
    with pytest.raises(
        WRAPPER.parent.DynamoRegionBisectionError,
        match="mode differs",
    ):
        WRAPPER._validate_launch(
            changed,
            mode="dynamo_s1_gc_off",
            arguments=[
                "--output",
                str(tmp_path / "run"),
                "--compile-blocks",
                "--compile-prewarm-buckets",
            ],
            scripts_root=ROOT / "scripts",
        )


def _result(failed: set[str], *, full_failed: bool = True) -> dict:
    comparisons = {
        name: {"gates": {"evaluation_noninferiority": name not in failed}}
        for name in AUDITOR.SIXTEENTH_RUNS
    }
    comparisons["dynamo_full_control"] = {
        "gates": {"evaluation_noninferiority": not full_failed}
    }
    return {"comparisons": comparisons}


def test_auditor_registers_only_blocks_inside_failed_sixteenths() -> None:
    localization = AUDITOR._localization(_result({"dynamo_s2", "dynamo_s16"}))
    assert localization == {
        "failed_sixteenth_runs": ["dynamo_s2", "dynamo_s16"],
        "full_control_reproduced": True,
        "next_disposition": "bisect_failed_sixteenths_into_blocks",
        "next_registered_regions": [[2], [3], [30], [31]],
    }


def test_auditor_stops_when_full_control_does_not_reproduce() -> None:
    localization = AUDITOR._localization(
        _result({"dynamo_s1"}, full_failed=False)
    )
    assert localization["next_disposition"] == "full_control_did_not_reproduce_stop"
    assert localization["next_registered_regions"] == []


def test_auditor_accepts_only_the_exact_valid_predecessor() -> None:
    protocol = _load(PROTOCOL_PATH)
    observed = AUDITOR._validate_predecessor(
        protocol,
        PREDECESSOR_PROTOCOL_PATH,
        PREDECESSOR_AUDIT_PATH,
    )
    assert observed == {
        "predecessor_audit_file_sha256": _sha256(PREDECESSOR_AUDIT_PATH),
        "predecessor_audit_receipt_sha256": _load(PREDECESSOR_AUDIT_PATH)[
            "receipt_sha256"
        ],
        "predecessor_protocol_file_sha256": _sha256(
            PREDECESSOR_PROTOCOL_PATH
        ),
    }


def test_scripts_are_ascii_and_launchers_fail_closed() -> None:
    for path in (WRAPPER_PATH, AUDITOR_PATH):
        path.read_bytes().decode("ascii")
        source = path.read_text(encoding="ascii")
        assert "allow_abbrev=False" in source or path == WRAPPER_PATH
    for path in (
        ROOT / "slurm/antibody_dynamo_sixteenth_bisection.sbatch",
        ROOT / "slurm/antibody_dynamo_sixteenth_bisection_audit.sbatch",
    ):
        source = path.read_text(encoding="ascii")
        assert "set -euo pipefail" in source
        assert "SAbDab2" not in source
