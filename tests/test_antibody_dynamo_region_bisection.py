from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/antibody_dynamo_region_bisection_v1.json"
MINIFIER_PROTOCOL_PATH = (
    ROOT / "protocols/antibody_compile_accuracy_minifier_v1.json"
)
MINIFIER_AUDIT_PATH = (
    ROOT
    / "results/pretraining/antibody_compile_accuracy_minifier_v1_attempt_001_audit.json"
)
WRAPPER_PATH = ROOT / "scripts/run_antibody_dynamo_region_bisection.py"
AUDITOR_PATH = ROOT / "scripts/audit_antibody_dynamo_region_bisection.py"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WRAPPER = _module(WRAPPER_PATH, "dynamo_region_bisection_wrapper")
AUDITOR = _module(AUDITOR_PATH, "dynamo_region_bisection_auditor")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_protocol_binds_every_implementation_and_failed_minifier() -> None:
    protocol = _load(PROTOCOL_PATH)
    minifier_audit = _load(MINIFIER_AUDIT_PATH)
    bindings = {
        "audit_launcher_sha256": (
            ROOT / "slurm/antibody_dynamo_region_bisection_audit.sbatch"
        ),
        "auditor_sha256": AUDITOR_PATH,
        "base_runner_sha256": (
            ROOT / "scripts/run_antibody_pretraining_optimized.py"
        ),
        "communication_protocol_sha256": (
            ROOT / "protocols/antibody_ddp_communication_optimization_v1.json"
        ),
        "efficiency_protocol_sha256": (
            ROOT / "protocols/antibody_efficiency_gate_v1.json"
        ),
        "launcher_sha256": (
            ROOT / "slurm/antibody_dynamo_region_bisection.sbatch"
        ),
        "minifier_audit_file_sha256": MINIFIER_AUDIT_PATH,
        "minifier_protocol_file_sha256": MINIFIER_PROTOCOL_PATH,
        "model_family_sha256": ROOT / "protocols/antibody_model_family_v1.json",
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
        "training_protocol_sha256": ROOT / "protocols/antibody_training_v1.json",
        "wrapper_sha256": WRAPPER_PATH,
    }
    for label, path in bindings.items():
        assert protocol["inputs"][label] == _sha256(path)
    assert protocol["failed_minifier"] == {
        "audit_file_sha256": _sha256(MINIFIER_AUDIT_PATH),
        "audit_receipt_sha256": minifier_audit["receipt_sha256"],
        "audit_job_id": 1346021,
        "boundary": "dynamo",
        "boundary_localized": False,
        "minifier_artifacts": [],
        "protocol_file_sha256": _sha256(MINIFIER_PROTOCOL_PATH),
        "training_job_id": 1346020,
    }
    assert protocol["measurement"]["promotable_runs"] == []
    assert protocol["selection"]["candidate_promotion_permitted"] is False
    assert set(protocol["authorization"].values()) == {False}


def test_registered_modes_cover_only_the_two_halves_and_full_control() -> None:
    protocol = _load(PROTOCOL_PATH)
    assert protocol["modes"] == WRAPPER.MODES
    assert protocol["modes"]["dynamo_first_half_gc_off"][
        "block_indices"
    ] == list(range(16))
    assert protocol["modes"]["dynamo_second_half_gc_off"][
        "block_indices"
    ] == list(range(16, 32))
    assert protocol["modes"]["dynamo_full_control_gc_off"][
        "block_indices"
    ] == list(range(32))


def test_wrapper_validates_flags_and_compiles_exact_registered_membership(
    tmp_path: Path,
) -> None:
    protocol = _load(PROTOCOL_PATH)
    arguments = [
        "--output",
        str(tmp_path / "run"),
        "--compile-blocks",
        "--compile-prewarm-buckets",
    ]
    output, flags = WRAPPER._validate_launch(
        protocol,
        mode="dynamo_first_half_gc_off",
        arguments=arguments,
        scripts_root=ROOT / "scripts",
    )
    assert output == tmp_path / "run"
    assert flags["block_indices"] == list(range(16))

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
    assert all(
        block.calls == [{"backend": "eager", "fullgraph": True}]
        for block in model.blocks[:16]
    )
    assert all(block.calls == [] for block in model.blocks[16:])


def test_wrapper_rejects_drift_and_unregistered_optimizations(
    tmp_path: Path,
) -> None:
    protocol = _load(PROTOCOL_PATH)
    changed = copy.deepcopy(protocol)
    changed["modes"]["dynamo_first_half_gc_off"]["block_indices"] = [0]
    with pytest.raises(WRAPPER.DynamoRegionBisectionError, match="mode differs"):
        WRAPPER._validate_launch(
            changed,
            mode="dynamo_first_half_gc_off",
            arguments=[
                "--output",
                str(tmp_path / "run"),
                "--compile-blocks",
                "--compile-prewarm-buckets",
            ],
            scripts_root=ROOT / "scripts",
        )
    with pytest.raises(
        WRAPPER.DynamoRegionBisectionError,
        match="unregistered optimization",
    ):
        WRAPPER._validate_launch(
            protocol,
            mode="dynamo_first_half_gc_off",
            arguments=[
                "--output",
                str(tmp_path / "run"),
                "--compile-blocks",
                "--compile-prewarm-buckets",
                "--fused-adamw",
            ],
            scripts_root=ROOT / "scripts",
        )


def _result(first_passed: bool, second_passed: bool, full_passed: bool) -> dict:
    return {
        "comparisons": {
            "dynamo_first_half": {
                "gates": {"evaluation_noninferiority": first_passed}
            },
            "dynamo_second_half": {
                "gates": {"evaluation_noninferiority": second_passed}
            },
            "dynamo_full_control": {
                "gates": {"evaluation_noninferiority": full_passed}
            },
        }
    }


@pytest.mark.parametrize(
    ("passed", "disposition", "regions"),
    [
        (
            (False, True, False),
            "bisect_first_half",
            [list(range(0, 8)), list(range(8, 16))],
        ),
        (
            (True, False, False),
            "bisect_second_half",
            [list(range(16, 24)), list(range(24, 32))],
        ),
        (
            (False, False, False),
            "both_halves_fail_bisect_each_independently",
            [
                list(range(0, 8)),
                list(range(8, 16)),
                list(range(16, 24)),
                list(range(24, 32)),
            ],
        ),
        ((True, True, False), "cross_half_interaction_required", []),
        ((False, False, True), "full_control_did_not_reproduce_stop", []),
    ],
)
def test_auditor_fail_closed_localization(
    passed: tuple[bool, bool, bool],
    disposition: str,
    regions: list[list[int]],
) -> None:
    localization = AUDITOR._localization(_result(*passed))
    assert localization["next_disposition"] == disposition
    assert localization["next_registered_regions"] == regions
    assert localization["full_control_reproduced"] is (not passed[2])


def test_auditor_accepts_only_the_exact_failed_minifier_predecessor() -> None:
    protocol = _load(PROTOCOL_PATH)
    observed = AUDITOR._validate_minifier_predecessor(
        protocol,
        MINIFIER_PROTOCOL_PATH,
        MINIFIER_AUDIT_PATH,
    )
    assert observed == {
        "minifier_audit_file_sha256": _sha256(MINIFIER_AUDIT_PATH),
        "minifier_audit_receipt_sha256": _load(MINIFIER_AUDIT_PATH)[
            "receipt_sha256"
        ],
        "minifier_protocol_file_sha256": _sha256(MINIFIER_PROTOCOL_PATH),
    }


def test_auditor_requires_exact_full_dynamo_replay(tmp_path: Path) -> None:
    protocol = _load(PROTOCOL_PATH)
    expected = protocol["expected_replay"]
    for name, _ in AUDITOR.RUNS:
        receipt = {
            "final_model_sha256": "partial-arm-model",
            "final_optimizer_sha256": "partial-arm-optimizer",
            "membership_sha256": expected["membership_sha256"],
            "preset": expected["preset"],
            "run_seed": expected["run_seed"],
            "world_size": expected["world_size"],
        }
        if name == "dynamo_full_control":
            receipt["final_model_sha256"] = expected[
                "full_dynamo_final_model_sha256"
            ]
            receipt["final_optimizer_sha256"] = expected[
                "full_dynamo_final_optimizer_sha256"
            ]
        destination = tmp_path / name / "run/receipt.json"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(AUDITOR.artifacts._canonical_json_bytes(receipt))
    replay = AUDITOR._validate_exact_replay(tmp_path, protocol)
    assert replay["passed"] is True

    full_path = tmp_path / "dynamo_full_control/run/receipt.json"
    full_receipt = _load(full_path)
    full_receipt["final_model_sha256"] = "different"
    full_path.write_bytes(AUDITOR.artifacts._canonical_json_bytes(full_receipt))
    with pytest.raises(
        AUDITOR.DynamoRegionBisectionAuditError,
        match="does not exactly replay",
    ):
        AUDITOR._validate_exact_replay(tmp_path, protocol)


def test_scripts_are_ascii_and_disable_long_option_abbreviation() -> None:
    for path in (WRAPPER_PATH, AUDITOR_PATH):
        path.read_bytes().decode("ascii")
        source = path.read_text(encoding="ascii")
        assert "allow_abbrev=False" in source
        assert "sabdab2_heldout_accessed" in source
    for path in (
        ROOT / "slurm/antibody_dynamo_region_bisection.sbatch",
        ROOT / "slurm/antibody_dynamo_region_bisection_audit.sbatch",
    ):
        path.read_bytes().decode("ascii")
        assert "set -euo pipefail" in path.read_text(encoding="ascii")
