#!/usr/bin/env python3
"""Freeze component localization after an Inductor-only quality failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import run_antibody_inductor_component_localization as runner


ATTRIBUTION_PROTOCOL_FORMAT = "antibody_compile_backend_attribution_protocol_v1"
ATTRIBUTION_AUDIT_FORMAT = "antibody_compile_backend_attribution_result_v1"
OUTPUT_FORMAT = "antibody_inductor_component_localization_protocol_v1"
AUTHORIZATION = runner.AUTHORIZATION
TEST_ACCESS = runner.TEST_ACCESS
RUNS = (
    ("safe_eager_r1", "safe_eager"),
    ("inductor_attention_only", "inductor_attention_only_gc_off"),
    ("inductor_ffn_only", "inductor_ffn_only_gc_off"),
    ("inductor_full_block", "inductor_full_block_gc_off"),
    ("safe_eager_r2", "safe_eager"),
)
PROMOTABLE_RUNS = ("inductor_attention_only", "inductor_ffn_only")
COMMON_BINDINGS = {
    "base_runner_sha256": "scripts/run_antibody_pretraining_optimized.py",
    "communication_protocol_sha256": (
        "protocols/antibody_ddp_communication_optimization_v1.json"
    ),
    "efficiency_protocol_sha256": "protocols/antibody_efficiency_gate_v1.json",
    "model_family_sha256": "protocols/antibody_model_family_v1.json",
    "parity_evaluation_indices_sha256": (
        "results/pretraining/antibody_ddp_parity_split_v2/evaluation_indices.npy"
    ),
    "parity_protocol_sha256": (
        "protocols/antibody_ddp_scientific_parity_gate_v2.json"
    ),
    "parity_split_receipt_file_sha256": (
        "results/pretraining/antibody_ddp_parity_split_v2/receipt.json"
    ),
    "parity_training_indices_sha256": (
        "results/pretraining/antibody_ddp_parity_split_v2/training_indices.npy"
    ),
    "training_protocol_sha256": "protocols/antibody_training_v1.json",
}
NEW_BINDINGS = {
    "audit_launcher_sha256": (
        "slurm/antibody_inductor_component_localization_audit.sbatch"
    ),
    "auditor_sha256": (
        "scripts/audit_antibody_inductor_component_localization.py"
    ),
    "freezer_sha256": (
        "scripts/freeze_antibody_inductor_component_localization.py"
    ),
    "launcher_sha256": "slurm/antibody_inductor_component_localization.sbatch",
    "wrapper_sha256": (
        "scripts/run_antibody_inductor_component_localization.py"
    ),
}


class ComponentLocalizationFreezeError(RuntimeError):
    """Report parent-evidence or source-closure disagreement."""


def _canonical_json_bytes(value: object) -> bytes:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> tuple[dict[str, Any], str]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ComponentLocalizationFreezeError(
            f"Input is not a regular file: {path}"
        )
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComponentLocalizationFreezeError(
            f"Input is not strict ASCII JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ComponentLocalizationFreezeError(f"Input is not an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise ComponentLocalizationFreezeError("Attribution audit self-hash differs")
    return observed


def _gate(comparison: Mapping[str, Any], name: str) -> bool:
    return comparison.get("gates", {}).get(name) is True


def _validate_inductor_only_trigger(
    protocol: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    protocol_sha256: str,
) -> None:
    comparisons = audit.get("comparisons", {})
    dynamo = comparisons.get("dynamo_eager", {})
    aot = comparisons.get("aot_eager", {})
    inductor = comparisons.get("inductor_no_cudagraphs", {})
    stable_gates = (
        "evaluation_noninferiority",
        "memory",
        "short_horizon_numerical",
        "step_stability",
    )
    if (
        protocol.get("format_version") != ATTRIBUTION_PROTOCOL_FORMAT
        or audit.get("format_version") != ATTRIBUTION_AUDIT_FORMAT
        or protocol.get("authorization") != AUTHORIZATION
        or audit.get("authorization") != AUTHORIZATION
        or protocol.get("test_access") != TEST_ACCESS
        or audit.get("test_access") != TEST_ACCESS
        or audit.get("inputs", {}).get("protocol_file_sha256")
        != protocol_sha256
        or audit.get("inputs", {}).get("auditor_sha256")
        != protocol.get("inputs", {}).get("auditor_sha256")
        or audit.get("diagnostic_valid") is not True
        or audit.get("candidate_found") is not False
        or audit.get("selected_candidate_for_strict_confirmation") is not None
        or not all(_gate(dynamo, gate) for gate in stable_gates)
        or not all(_gate(aot, gate) for gate in stable_gates)
        or _gate(inductor, "evaluation_noninferiority")
        or not all(
            _gate(inductor, gate)
            for gate in (
                "memory",
                "short_horizon_numerical",
                "step_stability",
                "throughput",
            )
        )
        or dynamo.get("backend_layer") != "eager"
        or aot.get("backend_layer") != "aot_eager"
        or inductor.get("backend_layer") != "inductor"
    ):
        raise ComponentLocalizationFreezeError(
            "Attribution does not prove an Inductor-only quality failure"
        )


def _bind_sources(
    source_root: Path,
    parent_protocol: Mapping[str, Any],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, relative in COMMON_BINDINGS.items():
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise ComponentLocalizationFreezeError(
                f"Common source input is invalid: {relative}"
            )
        observed = _sha256(path)
        if observed != parent_protocol["inputs"][key]:
            raise ComponentLocalizationFreezeError(
                f"Attribution source closure has drifted: {relative}"
            )
        result[key] = observed
    for key, relative in NEW_BINDINGS.items():
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise ComponentLocalizationFreezeError(
                f"Localization source input is invalid: {relative}"
            )
        result[key] = _sha256(path)
    return result


def freeze_localization(
    *,
    source_root: Path,
    attribution_protocol_path: Path,
    attribution_protocol_file_sha256: str,
    attribution_audit_path: Path,
    attribution_audit_file_sha256: str,
) -> dict[str, Any]:
    """Freeze a component crossover after exact predecessor validation."""
    protocol, observed_protocol_sha256 = _load(attribution_protocol_path)
    audit, observed_audit_sha256 = _load(attribution_audit_path)
    if observed_protocol_sha256 != attribution_protocol_file_sha256:
        raise ComponentLocalizationFreezeError("Attribution protocol anchor differs")
    if observed_audit_sha256 != attribution_audit_file_sha256:
        raise ComponentLocalizationFreezeError("Attribution audit anchor differs")
    audit_receipt_sha256 = _self_hash(audit)
    _validate_inductor_only_trigger(
        protocol,
        audit,
        protocol_sha256=attribution_protocol_file_sha256,
    )
    inputs = {
        **_bind_sources(source_root, protocol),
        "attribution_audit_file_sha256": attribution_audit_file_sha256,
        "attribution_protocol_file_sha256": attribution_protocol_file_sha256,
        "dense_manifest_payload_sha256": protocol["inputs"][
            "dense_manifest_payload_sha256"
        ],
    }
    acceptance = dict(protocol["acceptance"])
    acceptance["bootstrap_seed"] = int(acceptance["bootstrap_seed"]) + 1000
    return {
        "acceptance": acceptance,
        "authorization": AUTHORIZATION,
        "batch": dict(protocol["batch"]),
        "claim_boundary": (
            "Training-excluded same-allocation localization of an independently "
            "proven Inductor-only quality regression across attention, FFN, and "
            "full-block compile scopes; no configuration promotion, production "
            "training, biological metric, multi-node scale, or SOTA claim"
        ),
        "format_version": OUTPUT_FORMAT,
        "hardware": dict(protocol["hardware"]),
        "inputs": inputs,
        "measurement": {
            "promotable_runs": list(PROMOTABLE_RUNS),
            "run_order": [name for name, _ in RUNS],
            "run_seed": int(protocol["measurement"]["run_seed"]),
            "same_allocation_required": True,
            "training_evaluation_panel": protocol["measurement"][
                "training_evaluation_panel"
            ],
        },
        "modes": runner.MODES,
        "parent_attribution": {
            "audit_file_sha256": attribution_audit_file_sha256,
            "audit_receipt_sha256": audit_receipt_sha256,
            "candidate_found": False,
            "earliest_failing_layer": "inductor",
            "protocol_file_sha256": attribution_protocol_file_sha256,
        },
        "research_basis": [
            {
                "consequence": (
                    "Attention and FFN are compiled separately before any "
                    "operator-level fallback or precision change."
                ),
                "title": "PyTorch compiler troubleshooting",
                "url": (
                    "https://docs.pytorch.org/docs/main/user_guide/"
                    "torch_compiler/torch.compiler_troubleshooting.html"
                ),
            },
            {
                "consequence": (
                    "Regional compilation preserves safe compiled regions while "
                    "localizing one problematic repeated Transformer component."
                ),
                "title": "PyTorch regional compilation",
                "url": (
                    "https://docs.pytorch.org/tutorials/recipes/"
                    "regional_compilation.html"
                ),
            },
        ],
        "selection": {
            "ranking": (
                "Highest-throughput attention-only or FFN-only candidate among "
                "candidates passing every unchanged gate"
            ),
            "required_successor": (
                "new source-bound 600-update repeated crossover confirmation"
            ),
        },
        "test_access": TEST_ACCESS,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ComponentLocalizationFreezeError("Protocol must be create-only")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--attribution-protocol", type=Path, required=True)
    parser.add_argument("--attribution-protocol-file-sha256", required=True)
    parser.add_argument("--attribution-audit", type=Path, required=True)
    parser.add_argument("--attribution-audit-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    value = freeze_localization(
        source_root=arguments.source_root,
        attribution_protocol_path=arguments.attribution_protocol,
        attribution_protocol_file_sha256=(
            arguments.attribution_protocol_file_sha256
        ),
        attribution_audit_path=arguments.attribution_audit,
        attribution_audit_file_sha256=arguments.attribution_audit_file_sha256,
    )
    _publish(arguments.output, value)


if __name__ == "__main__":
    main()
