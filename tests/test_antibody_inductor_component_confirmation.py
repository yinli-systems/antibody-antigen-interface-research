"""Test strict confirmation of one audited Inductor component scope."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCALIZATION_FREEZER = (
    ROOT / "scripts/freeze_antibody_inductor_component_localization.py"
)
FREEZER = ROOT / "scripts/freeze_antibody_inductor_component_confirmation.py"
WRAPPER = ROOT / "scripts/run_antibody_inductor_component_confirmation.py"
AUDITOR = ROOT / "scripts/audit_antibody_inductor_component_confirmation.py"
SLURM = ROOT / "slurm/antibody_inductor_component_confirmation.sbatch"
AUDIT_SLURM = (
    ROOT / "slurm/antibody_inductor_component_confirmation_audit.sbatch"
)
ATTRIBUTION_PROTOCOL = (
    ROOT / "protocols/antibody_compile_backend_attribution_v1.json"
)


def _load(name: str, path: Path):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _self_hash(value: dict[str, object]) -> dict[str, object]:
    value["receipt_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def _parent_audit(protocol_sha256: str) -> dict[str, object]:
    protocol = json.loads(ATTRIBUTION_PROTOCOL.read_text(encoding="ascii"))
    passing = {
        "evaluation_noninferiority": True,
        "memory": True,
        "short_horizon_numerical": True,
        "step_stability": True,
        "throughput": True,
    }
    return _self_hash(
        {
            "authorization": protocol["authorization"],
            "candidate_found": False,
            "comparisons": {
                "aot_eager": {
                    "backend_layer": "aot_eager",
                    "gates": dict(passing),
                },
                "dynamo_eager": {
                    "backend_layer": "eager",
                    "gates": dict(passing),
                },
                "inductor_no_cudagraphs": {
                    "backend_layer": "inductor",
                    "gates": {
                        **passing,
                        "evaluation_noninferiority": False,
                    },
                },
            },
            "diagnostic_valid": True,
            "format_version": "antibody_compile_backend_attribution_result_v1",
            "inputs": {
                "auditor_sha256": protocol["inputs"]["auditor_sha256"],
                "protocol_file_sha256": protocol_sha256,
            },
            "selected_candidate_for_strict_confirmation": None,
            "test_access": protocol["test_access"],
        }
    )


def _localization_evidence(
    tmp_path: Path,
    *,
    candidate: str,
    failed_gate: str | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    localization_freezer = _load(
        f"localization_freezer_{candidate}_{failed_gate}",
        LOCALIZATION_FREEZER,
    )
    attribution_protocol_sha256 = hashlib.sha256(
        ATTRIBUTION_PROTOCOL.read_bytes()
    ).hexdigest()
    parent_audit = _parent_audit(attribution_protocol_sha256)
    parent_audit_path = tmp_path / f"parent-{candidate}-{failed_gate}.json"
    parent_audit_path.write_bytes(_canonical(parent_audit))
    localization_protocol = localization_freezer.freeze_localization(
        source_root=ROOT,
        attribution_protocol_path=ATTRIBUTION_PROTOCOL,
        attribution_protocol_file_sha256=attribution_protocol_sha256,
        attribution_audit_path=parent_audit_path,
        attribution_audit_file_sha256=hashlib.sha256(
            parent_audit_path.read_bytes()
        ).hexdigest(),
    )
    localization_protocol_path = tmp_path / f"protocol-{candidate}-{failed_gate}.json"
    localization_protocol_path.write_bytes(_canonical(localization_protocol))
    gates = {
        "evaluation_noninferiority": True,
        "memory": True,
        "short_horizon_numerical": True,
        "step_stability": True,
        "throughput": True,
    }
    if failed_gate is not None:
        gates[failed_gate] = False
    modes = {
        "inductor_attention_only": "inductor_attention_only_gc_off",
        "inductor_ffn_only": "inductor_ffn_only_gc_off",
        "inductor_full_block": "inductor_full_block_gc_off",
    }
    mode = modes[candidate]
    localization_audit = _self_hash(
        {
            "authorization": localization_protocol["authorization"],
            "candidate_found": True,
            "comparisons": {
                candidate: {
                    "compile_scope": localization_protocol["modes"][mode][
                        "compile_scope"
                    ],
                    "eligible_for_strict_confirmation": all(gates.values()),
                    "gates": gates,
                    "promotable_backend": candidate
                    in {
                        "inductor_attention_only",
                        "inductor_ffn_only",
                    },
                }
            },
            "diagnostic_valid": True,
            "format_version": "antibody_inductor_component_localization_result_v1",
            "inputs": {
                "auditor_sha256": localization_protocol["inputs"][
                    "auditor_sha256"
                ],
                "protocol_file_sha256": hashlib.sha256(
                    localization_protocol_path.read_bytes()
                ).hexdigest(),
            },
            "selected_candidate_for_strict_confirmation": candidate,
            "selected_component_candidate": candidate,
            "test_access": localization_protocol["test_access"],
        }
    )
    localization_audit_path = tmp_path / f"audit-{candidate}-{failed_gate}.json"
    localization_audit_path.write_bytes(_canonical(localization_audit))
    return localization_protocol_path, localization_audit_path, localization_protocol


def _freeze(tmp_path: Path, *, candidate: str):
    freezer = _load(f"component_confirmation_freezer_{candidate}", FREEZER)
    protocol_path, audit_path, _ = _localization_evidence(
        tmp_path,
        candidate=candidate,
    )
    frozen = freezer.freeze_confirmation(
        source_root=ROOT,
        localization_protocol_path=protocol_path,
        localization_protocol_file_sha256=hashlib.sha256(
            protocol_path.read_bytes()
        ).hexdigest(),
        localization_audit_path=audit_path,
        localization_audit_file_sha256=hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest(),
    )
    return freezer, frozen, protocol_path, audit_path


def _base_arguments(output: Path, *, compiled: bool) -> list[str]:
    arguments = ["--output", str(output), "--steps", "600", "--backend", "ddp"]
    if compiled:
        arguments.extend(
            (
                "--compile-blocks",
                "--compile-prewarm-buckets",
                "--disable-compiled-cudagraphs",
            )
        )
    return arguments


def test_sources_are_ascii_and_syntax_valid() -> None:
    for path in (FREEZER, WRAPPER, AUDITOR):
        source = path.read_text(encoding="ascii")
        ast.parse(source)
        assert "os.O_EXCL" in source or path == AUDITOR
    for path in (SLURM, AUDIT_SLURM):
        path.read_text(encoding="ascii")


@pytest.mark.parametrize(
    ("candidate", "scope"),
    (
        ("inductor_attention_only", "attention"),
        ("inductor_ffn_only", "ffn"),
    ),
)
def test_freezer_preserves_selected_scope_and_unchanged_gates(
    tmp_path: Path,
    candidate: str,
    scope: str,
) -> None:
    _, frozen, _, _ = _freeze(tmp_path, candidate=candidate)
    selection = frozen["component_confirmation_selection"]
    assert selection["selected_candidate"] == candidate
    assert selection["selected_mode_flags"]["compile_scope"] == scope
    assert frozen["batch"]["updates_per_run"] == 600
    assert frozen["acceptance"][
        "maximum_masked_cross_entropy_relative_increase"
    ] == 0.01
    assert frozen["acceptance"]["compiled_relative_to_eager_pooled"] == 1.05
    assert frozen["optimization"]["execution_controls"] == {
        "compile_backend": "inductor",
        "compile_scope": scope,
        "disable_compiled_cudagraphs": True,
        "disable_python_gc": True,
    }
    assert frozen["authorization"]["production_pretraining"] is False
    assert frozen["measurement"]["crossover_run_order"] == [
        "safe_eager_r1",
        "component_compiled_r1",
        "component_compiled_r2",
        "safe_eager_r2",
    ]
    for key, path in (
        ("confirmation_audit_launcher_sha256", AUDIT_SLURM),
        ("confirmation_auditor_sha256", AUDITOR),
        ("confirmation_freezer_sha256", FREEZER),
        ("confirmation_launcher_sha256", SLURM),
        ("confirmation_wrapper_sha256", WRAPPER),
    ):
        assert frozen["inputs"][key] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("candidate", "failed_gate"),
    (
        ("inductor_attention_only", "evaluation_noninferiority"),
        ("inductor_ffn_only", "throughput"),
        ("inductor_full_block", None),
    ),
)
def test_freezer_rejects_failed_or_nonpromotable_selection(
    tmp_path: Path,
    candidate: str,
    failed_gate: str | None,
) -> None:
    freezer = _load(f"component_confirmation_reject_{candidate}", FREEZER)
    protocol_path, audit_path, _ = _localization_evidence(
        tmp_path,
        candidate=candidate,
        failed_gate=failed_gate,
    )
    with pytest.raises(
        freezer.ComponentConfirmationFreezeError,
        match="does not authorize",
    ):
        freezer.freeze_confirmation(
            source_root=ROOT,
            localization_protocol_path=protocol_path,
            localization_protocol_file_sha256=hashlib.sha256(
                protocol_path.read_bytes()
            ).hexdigest(),
            localization_audit_path=audit_path,
            localization_audit_file_sha256=hashlib.sha256(
                audit_path.read_bytes()
            ).hexdigest(),
        )


@pytest.mark.parametrize(
    ("candidate", "scope"),
    (
        ("inductor_attention_only", "attention"),
        ("inductor_ffn_only", "ffn"),
    ),
)
def test_wrapper_compiles_only_the_selected_component(
    tmp_path: Path,
    candidate: str,
    scope: str,
) -> None:
    _, protocol, _, _ = _freeze(tmp_path, candidate=candidate)
    wrapper = _load(f"component_confirmation_wrapper_{candidate}", WRAPPER)
    output, flags = wrapper._validate_launch(
        protocol,
        mode=candidate,
        arguments=_base_arguments(tmp_path / candidate, compiled=True),
    )
    assert output == tmp_path / candidate
    assert flags["compile_scope"] == scope
    eager_output, eager = wrapper._validate_launch(
        protocol,
        mode="safe_eager",
        arguments=_base_arguments(tmp_path / f"eager-{candidate}", compiled=False),
    )
    assert eager_output == tmp_path / f"eager-{candidate}"
    assert eager["compile_scope"] == "none"
    with pytest.raises(wrapper.ComponentConfirmationError):
        wrapper._validate_launch(
            protocol,
            mode=(
                "inductor_ffn_only"
                if candidate == "inductor_attention_only"
                else "inductor_attention_only"
            ),
            arguments=_base_arguments(tmp_path / "wrong", compiled=True),
        )


def test_independent_auditor_replays_localization_selection(tmp_path: Path) -> None:
    _, protocol, protocol_path, audit_path = _freeze(
        tmp_path,
        candidate="inductor_attention_only",
    )
    auditor = _load("component_confirmation_auditor", AUDITOR)
    evidence, candidate, flags = auditor._validate_localization(
        protocol=protocol,
        localization_protocol_path=protocol_path,
        localization_audit_path=audit_path,
    )
    assert candidate == "inductor_attention_only"
    assert flags["compile_scope"] == "attention"
    assert evidence["localization_audit_file_sha256"] == hashlib.sha256(
        audit_path.read_bytes()
    ).hexdigest()
    tampered = json.loads(audit_path.read_text(encoding="ascii"))
    tampered["candidate_found"] = False
    tampered.pop("receipt_sha256")
    _self_hash(tampered)
    tampered_path = tmp_path / "tampered-localization.json"
    tampered_path.write_bytes(_canonical(tampered))
    with pytest.raises(
        auditor.ComponentConfirmationAuditError,
        match="selection evidence differs",
    ):
        auditor._validate_localization(
            protocol=protocol,
            localization_protocol_path=protocol_path,
            localization_audit_path=tampered_path,
        )


def test_auditor_run_mappings_preserve_two_replicates() -> None:
    auditor = _load("component_confirmation_run_mapping", AUDITOR)
    assert auditor.CONFIRMATION_RUNS == (
        ("safe_eager_r1", "safe_eager"),
        ("component_compiled_r1", "candidate"),
        ("component_compiled_r2", "candidate"),
        ("safe_eager_r2", "safe_eager"),
    )
    assert auditor.PARENT_RUNS[1][1] == "fp32_compiled"
    assert auditor.PARENT_RUNS[2][1] == "fp32_compiled"


def test_launchers_freeze_four_runs_and_independent_audit() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "vip_gpu_5090_scxi253" not in source
    assert "#SBATCH --gres=gpu:8" in source
    assert "--nproc-per-node=8" in source
    assert source.count("run_mode ") == 4
    assert "run_mode component_compiled_r1" in source
    assert "run_mode component_compiled_r2" in source
    assert source.count("run_mode safe_eager_") == 2
    assert "--steps 600" in source
    assert "--disable-compiled-cudagraphs" in source
    audit_source = AUDIT_SLURM.read_text(encoding="ascii")
    assert "CUDA_VISIBLE_DEVICES=" in audit_source
    assert "audit_antibody_inductor_component_confirmation.py" in audit_source
    assert '--localization-protocol "${LOCALIZATION_PROTOCOL}"' in audit_source
    assert '--localization-audit "${LOCALIZATION_AUDIT}"' in audit_source
