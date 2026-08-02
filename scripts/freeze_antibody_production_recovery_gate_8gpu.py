#!/usr/bin/env python3
"""Freeze the eight-GPU real-OAS recovery gate for a confirmed candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from antibody_pretraining.production_corpus import load_production_corpus


FORMAT_VERSION = "antibody_production_loader_recovery_gate_v1"
CONFIRMATION_FORMATS = {
    "antibody_compile_candidate_confirmation_result_v1",
    "antibody_compile_candidate_confirmation_result_v2",
    "antibody_inductor_component_confirmation_result_v1",
}
AUTHORIZATION = {
    "loader_recovery_gate": True,
    "production_pretraining": False,
    "scale_out": False,
    "scientific_claim": False,
}
CONFIRMATION_AUTHORIZATION = {
    "configuration_promotion": True,
    "production_pretraining": False,
    "scale_16_gpu": False,
    "scale_32_gpu": False,
    "scientific_claim": False,
}
TEST_ACCESS = {
    "downstream_labels_used": False,
    "sabdab2_heldout_accessed": False,
}
BACKEND_CONTROLS = {
    "compile_backend": "inductor",
    "disable_compiled_cudagraphs": True,
    "disable_python_gc": True,
}
COMPONENT_SCOPES = {"attention", "ffn"}
COMPONENT_CANDIDATE_SCOPES = {
    "inductor_attention_only": "attention",
    "inductor_ffn_only": "ffn",
}
IMPLEMENTATION_FILES = (
    "scripts/run_antibody_pretraining.py",
    "scripts/run_antibody_pretraining_optimized.py",
    "scripts/run_antibody_pretraining_production_gate.py",
    "scripts/run_antibody_inductor_component_localization.py",
    "src/antibody_pretraining/dense_shard.py",
    "src/antibody_pretraining/masking.py",
    "src/antibody_pretraining/model.py",
    "src/antibody_pretraining/production_corpus.py",
    "src/antibody_pretraining/production_update_plan.py",
    "src/antibody_pretraining/tokenizer.py",
    "src/antibody_pretraining/training_contract.py",
)


class EightGpuGateFreezeError(RuntimeError):
    """Report an invalid promotion, corpus, or source closure."""


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
        raise EightGpuGateFreezeError(f"Input is not a regular file: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EightGpuGateFreezeError(
            f"Input is not strict ASCII JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise EightGpuGateFreezeError(f"Input is not a JSON object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _self_hash(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != observed:
        raise EightGpuGateFreezeError("Confirmation audit self-hash differs")
    return observed


def _component_backend_controls(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value)
        == {
            "compile_backend",
            "compile_scope",
            "disable_compiled_cudagraphs",
            "disable_python_gc",
        }
        and value["compile_backend"] == "inductor"
        and value["compile_scope"] in COMPONENT_SCOPES
        and value["disable_compiled_cudagraphs"] is True
        and value["disable_python_gc"] is True
    )


def freeze_gate(
    *,
    source_root: Path,
    corpus_root: Path,
    manifest_file_sha256: str,
    manifest_payload_sha256: str,
    corpus_protocol_file_sha256: str,
    independent_audit_path: Path,
    independent_audit_file_sha256: str,
    independent_audit_payload_sha256: str,
    confirmation_protocol_path: Path,
    confirmation_protocol_file_sha256: str,
    confirmation_audit_path: Path,
    confirmation_audit_file_sha256: str,
) -> dict[str, Any]:
    """Freeze one eight-GPU recovery protocol after promotion evidence passes."""
    confirmation_protocol, observed_protocol_sha256 = _load(
        confirmation_protocol_path
    )
    confirmation_audit, observed_audit_sha256 = _load(confirmation_audit_path)
    if observed_protocol_sha256 != confirmation_protocol_file_sha256:
        raise EightGpuGateFreezeError("Confirmation protocol anchor differs")
    if observed_audit_sha256 != confirmation_audit_file_sha256:
        raise EightGpuGateFreezeError("Confirmation audit anchor differs")
    confirmation_receipt_sha256 = _self_hash(confirmation_audit)
    confirmation_version = confirmation_audit.get("format_version")
    component_confirmation = (
        confirmation_version
        == "antibody_inductor_component_confirmation_result_v1"
    )
    selection_key = (
        "component_confirmation_selection"
        if component_confirmation
        else "confirmation_selection"
    )
    confirmed_key = (
        "confirmed_component_candidate"
        if component_confirmation
        else "confirmed_candidate"
    )
    selected = confirmation_protocol.get(selection_key, {}).get(
        "selected_candidate"
    )
    controls = confirmation_protocol.get("optimization", {}).get(
        "execution_controls"
    )
    if component_confirmation:
        controls_valid = (
            _component_backend_controls(controls)
            and COMPONENT_CANDIDATE_SCOPES.get(selected)
            == controls["compile_scope"]
        )
    else:
        controls_valid = controls in (
            {
                "accuracy_first_bf16_reduction": False,
                "disable_python_gc": False,
            },
            {
                "accuracy_first_bf16_reduction": False,
                "disable_python_gc": True,
            },
            {
                "accuracy_first_bf16_reduction": True,
                "disable_python_gc": False,
            },
            {
                "accuracy_first_bf16_reduction": True,
                "disable_python_gc": True,
            },
            BACKEND_CONTROLS,
        )
    if (
        confirmation_audit.get("format_version") not in CONFIRMATION_FORMATS
        or confirmation_audit.get("gate_passed") is not True
        or confirmation_audit.get("authorization") != CONFIRMATION_AUTHORIZATION
        or confirmation_audit.get("test_access") != TEST_ACCESS
        or confirmation_audit.get(confirmed_key) != selected
        or confirmation_audit.get("inputs", {}).get("protocol_file_sha256")
        != confirmation_protocol_file_sha256
        or not controls_valid
    ):
        raise EightGpuGateFreezeError(
            "Compile confirmation does not authorize a recovery gate"
        )
    if component_confirmation:
        confirmation_sources = {
            "confirmation_auditor_sha256": (
                "scripts/audit_antibody_inductor_component_confirmation.py"
            ),
            "confirmation_wrapper_sha256": (
                "scripts/run_antibody_inductor_component_confirmation.py"
            ),
        }
    else:
        confirmation_suffix = (
            "_v2" if confirmation_version.endswith("_v2") else ""
        )
        confirmation_sources = {
            "confirmation_auditor_sha256": (
                "scripts/audit_antibody_compile_candidate_confirmation"
                f"{confirmation_suffix}.py"
            ),
            "confirmation_wrapper_sha256": (
                "scripts/run_antibody_compile_candidate_confirmation"
                f"{confirmation_suffix}.py"
            ),
        }
    expected_sources = {
        **confirmation_sources,
        "runner_sha256": "scripts/run_antibody_pretraining_optimized.py",
        "runner_wrapper_sha256": (
            "scripts/run_antibody_pretraining_parity_extension.py"
        ),
    }
    if any(
        _sha256(source_root / relative)
        != confirmation_protocol["inputs"][key]
        for key, relative in expected_sources.items()
    ):
        raise EightGpuGateFreezeError("Confirmation source closure has drifted")

    corpus = load_production_corpus(
        corpus_root,
        split="train",
        expected_manifest_file_sha256=manifest_file_sha256,
        expected_manifest_payload_sha256=manifest_payload_sha256,
        expected_protocol_file_sha256=corpus_protocol_file_sha256,
        independent_audit_path=independent_audit_path,
        expected_independent_audit_file_sha256=independent_audit_file_sha256,
        expected_independent_audit_payload_sha256=(
            independent_audit_payload_sha256
        ),
    )
    if len(corpus) <= 0:
        raise EightGpuGateFreezeError("Audited production training split is empty")
    implementation = {}
    for relative in IMPLEMENTATION_FILES:
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise EightGpuGateFreezeError(
                f"Implementation input is invalid: {relative}"
            )
        implementation[relative] = _sha256(path)
    model_family = source_root / "protocols/antibody_model_family_v1.json"
    training_protocol = source_root / "protocols/antibody_training_v1.json"
    efficiency_protocol = source_root / "protocols/antibody_efficiency_gate_v1.json"
    auditor = source_root / "scripts/audit_antibody_production_recovery_gate_8gpu.py"
    wrapper = (
        source_root
        / "scripts/run_antibody_pretraining_production_candidate_gate.py"
    )
    inputs = {
        "efficiency_protocol_file_sha256": _sha256(efficiency_protocol),
        "independent_audit_file_sha256": independent_audit_file_sha256,
        "independent_audit_payload_sha256": independent_audit_payload_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_payload_sha256": manifest_payload_sha256,
        "model_family_file_sha256": _sha256(model_family),
        "protocol_file_sha256": corpus_protocol_file_sha256,
        "split": "train",
        "training_protocol_file_sha256": _sha256(training_protocol),
    }
    optimization = {
        "bucketed_padding": False,
        "compile_blocks": True,
        "compile_prewarm_buckets": True,
        "ddp_bucket_cap_mb": 25,
        "deterministic_algorithms": True,
        "disable_compiled_cudagraphs": (
            controls == BACKEND_CONTROLS
            or _component_backend_controls(controls)
        ),
        "fused_adamw": False,
        "length_bucketing": False,
        "pinned_staging": False,
    }

    def run(*, steps: int, checkpoint: bool, resume: bool) -> dict[str, object]:
        return {
            "batch_quantum_sequences": 512,
            "checkpoint": checkpoint,
            "micro_batch_per_gpu": 16,
            "preset": "abf_650m",
            "resume": resume,
            "run_seed": 17,
            "steps": steps,
            "target_residue_tokens": 262_144,
        }

    return {
        "auditor_sha256": _sha256(auditor),
        "authorization": AUTHORIZATION,
        "candidate_selection": {
            "base_runner_sha256": _sha256(
                source_root / "scripts/run_antibody_pretraining_production_gate.py"
            ),
            "configuration_promotion_verified": True,
            "confirmation_audit_file_sha256": confirmation_audit_file_sha256,
            "confirmation_audit_receipt_sha256": confirmation_receipt_sha256,
            "confirmation_protocol_file_sha256": (
                confirmation_protocol_file_sha256
            ),
            "execution_controls": controls,
            "confirmation_format": confirmation_version,
            "selected_candidate": selected,
            "wrapper_sha256": _sha256(wrapper),
        },
        "claim_boundary": (
            "Authorizes only an eight-GPU real-OAS loader and exact-recovery "
            "gate for a confirmed execution candidate; no production training, "
            "scale, biological, downstream, or SOTA claim"
        ),
        "corpus": {
            "train_record_count": len(corpus),
            "validation_record_count": corpus.manifest["counts"][
                "unique_validation_records"
            ],
        },
        "execution": {
            "backend": "ddp",
            "data_loader": {
                "grouped_update_read": True,
                "maximum_open_arrays": 1024,
                "minimum_soft_open_file_limit": 4096,
            },
            "length_bucket_boundaries": "96,128,160,194",
            "optimization": optimization,
            "world_size": 8,
        },
        "format_version": FORMAT_VERSION,
        "implementation": implementation,
        "inputs": inputs,
        "runs": {
            "continuous": run(steps=4, checkpoint=False, resume=False),
            "resumed": run(steps=2, checkpoint=False, resume=True),
            "stage1": run(steps=2, checkpoint=True, resume=False),
        },
        "test_access": TEST_ACCESS,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise EightGpuGateFreezeError("Gate protocol must be create-only")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--manifest-payload-sha256", required=True)
    parser.add_argument("--corpus-protocol-file-sha256", required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--independent-audit-file-sha256", required=True)
    parser.add_argument("--independent-audit-payload-sha256", required=True)
    parser.add_argument("--confirmation-protocol", type=Path, required=True)
    parser.add_argument("--confirmation-protocol-file-sha256", required=True)
    parser.add_argument("--confirmation-audit", type=Path, required=True)
    parser.add_argument("--confirmation-audit-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    value = freeze_gate(
        source_root=arguments.source_root,
        corpus_root=arguments.corpus_root,
        manifest_file_sha256=arguments.manifest_file_sha256,
        manifest_payload_sha256=arguments.manifest_payload_sha256,
        corpus_protocol_file_sha256=arguments.corpus_protocol_file_sha256,
        independent_audit_path=arguments.independent_audit,
        independent_audit_file_sha256=arguments.independent_audit_file_sha256,
        independent_audit_payload_sha256=(
            arguments.independent_audit_payload_sha256
        ),
        confirmation_protocol_path=arguments.confirmation_protocol,
        confirmation_protocol_file_sha256=(
            arguments.confirmation_protocol_file_sha256
        ),
        confirmation_audit_path=arguments.confirmation_audit,
        confirmation_audit_file_sha256=arguments.confirmation_audit_file_sha256,
    )
    _publish(arguments.output, value)


if __name__ == "__main__":
    main()
