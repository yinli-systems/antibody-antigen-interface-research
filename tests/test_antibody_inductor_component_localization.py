"""Test the fail-closed Inductor component-localization chain."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "scripts/freeze_antibody_inductor_component_localization.py"
WRAPPER = ROOT / "scripts/run_antibody_inductor_component_localization.py"
AUDITOR = ROOT / "scripts/audit_antibody_inductor_component_localization.py"
SLURM = ROOT / "slurm/antibody_inductor_component_localization.sbatch"
AUDIT_SLURM = (
    ROOT / "slurm/antibody_inductor_component_localization_audit.sbatch"
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


def _parent_audit(protocol_sha256: str) -> dict[str, object]:
    protocol = json.loads(ATTRIBUTION_PROTOCOL.read_text(encoding="ascii"))
    passing = {
        "evaluation_noninferiority": True,
        "memory": True,
        "short_horizon_numerical": True,
        "step_stability": True,
        "throughput": True,
    }
    value: dict[str, object] = {
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
    value["receipt_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return value


def _write_parent_audit(
    tmp_path: Path,
    *,
    mutate=None,
) -> tuple[Path, str, dict[str, object]]:
    protocol_sha256 = hashlib.sha256(ATTRIBUTION_PROTOCOL.read_bytes()).hexdigest()
    value = _parent_audit(protocol_sha256)
    if mutate is not None:
        mutate(value)
        value.pop("receipt_sha256")
        value["receipt_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    path = tmp_path / f"parent-{len(list(tmp_path.iterdir()))}.json"
    path.write_bytes(_canonical(value))
    return path, protocol_sha256, value


def _freeze(tmp_path: Path):
    freezer = _load("inductor_component_freezer", FREEZER)
    audit_path, protocol_sha256, _ = _write_parent_audit(tmp_path)
    protocol = freezer.freeze_localization(
        source_root=ROOT,
        attribution_protocol_path=ATTRIBUTION_PROTOCOL,
        attribution_protocol_file_sha256=protocol_sha256,
        attribution_audit_path=audit_path,
        attribution_audit_file_sha256=hashlib.sha256(
            audit_path.read_bytes()
        ).hexdigest(),
    )
    return freezer, protocol, audit_path


def _base_arguments(output: Path, *, compiled: bool) -> list[str]:
    arguments = ["--output", str(output), "--steps", "500", "--backend", "ddp"]
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
        assert "os.O_EXCL" in source
    for path in (SLURM, AUDIT_SLURM):
        path.read_text(encoding="ascii")


def test_freezer_accepts_only_an_inductor_quality_failure(tmp_path: Path) -> None:
    freezer, protocol, _ = _freeze(tmp_path)
    assert protocol["parent_attribution"]["earliest_failing_layer"] == "inductor"
    assert protocol["measurement"]["promotable_runs"] == [
        "inductor_attention_only",
        "inductor_ffn_only",
    ]
    assert protocol["measurement"]["run_order"] == [
        "safe_eager_r1",
        "inductor_attention_only",
        "inductor_ffn_only",
        "inductor_full_block",
        "safe_eager_r2",
    ]
    assert protocol["batch"]["updates_per_run"] == 500
    assert (
        protocol["acceptance"]["maximum_masked_cross_entropy_relative_increase"]
        == 0.01
    )
    assert protocol["authorization"]["production_pretraining"] is False
    for key, path in (
        ("audit_launcher_sha256", AUDIT_SLURM),
        ("auditor_sha256", AUDITOR),
        ("freezer_sha256", FREEZER),
        ("launcher_sha256", SLURM),
        ("wrapper_sha256", WRAPPER),
    ):
        assert protocol["inputs"][key] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert freezer.PROMOTABLE_RUNS == (
        "inductor_attention_only",
        "inductor_ffn_only",
    )


@pytest.mark.parametrize("case", ("candidate", "aot", "inductor"))
def test_freezer_rejects_adversarial_parent_states(
    tmp_path: Path,
    case: str,
) -> None:
    freezer = _load(f"inductor_component_freezer_{case}", FREEZER)

    def mutate(value: dict[str, object]) -> None:
        if case == "candidate":
            value["candidate_found"] = True
            value["selected_candidate_for_strict_confirmation"] = (
                "inductor_no_cudagraphs"
            )
        elif case == "aot":
            value["comparisons"]["aot_eager"]["gates"][
                "evaluation_noninferiority"
            ] = False
        else:
            value["comparisons"]["inductor_no_cudagraphs"]["gates"][
                "evaluation_noninferiority"
            ] = True

    audit_path, protocol_sha256, _ = _write_parent_audit(
        tmp_path,
        mutate=mutate,
    )
    with pytest.raises(
        freezer.ComponentLocalizationFreezeError,
        match="Inductor-only quality failure",
    ):
        freezer.freeze_localization(
            source_root=ROOT,
            attribution_protocol_path=ATTRIBUTION_PROTOCOL,
            attribution_protocol_file_sha256=protocol_sha256,
            attribution_audit_path=audit_path,
            attribution_audit_file_sha256=hashlib.sha256(
                audit_path.read_bytes()
            ).hexdigest(),
        )


def test_wrapper_accepts_only_registered_scopes(tmp_path: Path) -> None:
    _, protocol, _ = _freeze(tmp_path)
    wrapper = _load("inductor_component_wrapper", WRAPPER)
    expected = {
        "safe_eager": (False, "none"),
        "inductor_attention_only_gc_off": (True, "attention"),
        "inductor_ffn_only_gc_off": (True, "ffn"),
        "inductor_full_block_gc_off": (True, "full_block"),
    }
    for mode, (compiled, scope) in expected.items():
        output = tmp_path / mode
        observed_output, flags = wrapper._validate_launch(
            protocol,
            mode=mode,
            arguments=_base_arguments(output, compiled=compiled),
            scripts_root=ROOT / "scripts",
        )
        assert observed_output == output
        assert flags["compile_scope"] == scope
    with pytest.raises(
        wrapper.ComponentLocalizationError,
        match="Compile launch controls differ",
    ):
        wrapper._validate_launch(
            protocol,
            mode="inductor_attention_only_gc_off",
            arguments=_base_arguments(tmp_path / "bad", compiled=False),
            scripts_root=ROOT / "scripts",
        )
    with pytest.raises(
        wrapper.ComponentLocalizationError,
        match="unregistered optimization",
    ):
        wrapper._validate_launch(
            protocol,
            mode="inductor_ffn_only_gc_off",
            arguments=[
                *_base_arguments(tmp_path / "bad-optimization", compiled=True),
                "--fused-adamw",
            ],
            scripts_root=ROOT / "scripts",
        )


class _CompileRecorder:
    def __init__(self, label: str, calls: list[tuple[object, ...]]) -> None:
        self.label = label
        self.calls = calls

    def compile(self, **kwargs: object) -> None:
        self.calls.append((self.label, kwargs))


def test_wrapper_compiles_exactly_one_registered_region() -> None:
    wrapper = _load("inductor_component_compile_scope", WRAPPER)
    for scope, expected in (
        ("attention", {"attention-0", "attention-1"}),
        ("ffn", {"ffn-0", "ffn-1"}),
        ("full_block", {"block-0", "block-1"}),
    ):
        calls: list[tuple[object, ...]] = []
        blocks = []
        for index in range(2):
            block = _CompileRecorder(f"block-{index}", calls)
            block.attention = _CompileRecorder(f"attention-{index}", calls)
            block.ffn = _CompileRecorder(f"ffn-{index}", calls)
            blocks.append(block)
        wrapper._compile_scoped(SimpleNamespace(blocks=blocks), scope=scope)
        assert {call[0] for call in calls} == expected
        assert all(
            call[1]
            == {
                "backend": "inductor",
                "fullgraph": True,
                "options": {"triton.cudagraphs": False},
            }
            for call in calls
        )


def test_independent_auditor_replays_parent_trigger(tmp_path: Path) -> None:
    _, protocol, audit_path = _freeze(tmp_path)
    protocol_path = tmp_path / "localization.json"
    protocol_path.write_bytes(_canonical(protocol))
    auditor = _load("inductor_component_auditor", AUDITOR)
    evidence = auditor._validate_parent_attribution(
        protocol=protocol,
        attribution_protocol_path=ATTRIBUTION_PROTOCOL,
        attribution_audit_path=audit_path,
    )
    assert evidence["attribution_protocol_file_sha256"] == hashlib.sha256(
        ATTRIBUTION_PROTOCOL.read_bytes()
    ).hexdigest()
    tampered = json.loads(audit_path.read_text(encoding="ascii"))
    tampered["candidate_found"] = True
    tampered["selected_candidate_for_strict_confirmation"] = (
        "inductor_no_cudagraphs"
    )
    tampered.pop("receipt_sha256")
    tampered["receipt_sha256"] = hashlib.sha256(_canonical(tampered)).hexdigest()
    tampered_path = tmp_path / "tampered-parent.json"
    tampered_path.write_bytes(_canonical(tampered))
    with pytest.raises(
        auditor.ComponentLocalizationAuditError,
        match="Inductor-only quality failure",
    ):
        auditor._validate_parent_attribution(
            protocol=protocol,
            attribution_protocol_path=ATTRIBUTION_PROTOCOL,
            attribution_audit_path=tampered_path,
        )


def test_launchers_freeze_five_arms_and_independent_audit() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "vip_gpu_5090_scxi253" not in source
    assert "#SBATCH --gres=gpu:8" in source
    assert "--nproc-per-node=8" in source
    assert source.count("run_mode ") == 5
    assert (
        "run_mode inductor_attention_only "
        "inductor_attention_only_gc_off true"
    ) in source
    assert "run_mode inductor_ffn_only inductor_ffn_only_gc_off true" in source
    assert "run_mode inductor_full_block inductor_full_block_gc_off true" in source
    assert source.count("run_mode safe_eager_") == 2
    assert "--steps 500" in source
    assert "--disable-compiled-cudagraphs" in source
    audit_source = AUDIT_SLURM.read_text(encoding="ascii")
    assert "CUDA_VISIBLE_DEVICES=" in audit_source
    assert "audit_antibody_inductor_component_localization.py" in audit_source
    assert '--attribution-protocol "${ATTRIBUTION_PROTOCOL}"' in audit_source
    assert '--attribution-audit "${ATTRIBUTION_AUDIT}"' in audit_source
