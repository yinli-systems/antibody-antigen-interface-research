"""Validate the registered real-OAS elastic recovery gate implementation."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

import audit_antibody_elastic_recovery as auditor
import freeze_antibody_elastic_recovery_gate as freezer
import run_antibody_pretraining_production_gate as runner


ROOT = Path(__file__).resolve().parents[1]


class _FakeCorpus:
    class contract:
        array_file_hashes_verified = True

    manifest = {"counts": {"unique_validation_records": 123}}

    def __len__(self) -> int:
        return 456


def test_freezer_locks_the_bidirectional_real_oas_smoke(monkeypatch) -> None:
    monkeypatch.setattr(freezer, "load_production_corpus", lambda *a, **k: _FakeCorpus())
    payload = freezer.freeze_gate(
        source_root=ROOT,
        corpus_root=ROOT / "ignored-corpus",
        manifest_file_sha256="a" * 64,
        manifest_payload_sha256="b" * 64,
        corpus_protocol_file_sha256="c" * 64,
        independent_audit_path=ROOT / "ignored-audit.json",
        independent_audit_file_sha256="d" * 64,
        independent_audit_payload_sha256="e" * 64,
    )
    assert payload["format_version"] == "antibody_elastic_recovery_gate_v1"
    assert payload["topology"] == {
        "allowed_transitions": [[4, 2], [2, 4]],
        "allowed_world_sizes": [2, 4],
        "gpu_model": "NVIDIA GeForce RTX 4090",
        "node_count": 1,
    }
    assert [
        payload["runs"][key]["world_size"]
        for key in ("continuous_4", "stage1_4", "stage2_2", "stage3_4")
    ] == [4, 4, 2, 4]
    assert payload["runs"]["stage2_2"]["source_world_size"] == 4
    assert payload["runs"]["stage3_4"]["source_world_size"] == 2
    assert payload["runs"]["continuous_4"]["preset"] == "abf_35m"
    assert payload["corpus"]["array_file_hashes_verified_during_freeze"] is True
    assert (
        payload["corpus"]["stage_revalidation"]
        == "immutable_tree_manifest_size_dtype_shape"
    )
    assert payload["execution"]["optimization"]["compile_blocks"] is False
    assert "src/antibody_pretraining/elastic_resume.py" in payload["implementation"]
    assert payload["numerical_gate"]["absolute_tolerance"] == 0.005
    assert payload["numerical_gate"]["relative_tolerance"] == 0.02
    assert payload["numerical_gate"]["same_world_prefix_exact"] is True
    assert payload["numerical_gate"]["terminal_coordinate_exact"] is True
    assert payload["numerical_gate"]["optimizer_state"] == {
        "exp_avg": {
            "maximum_absolute_difference": 2.0**-29,
            "maximum_relative_difference": 0.001,
            "maximum_ulp_distance": 4096,
        },
        "exp_avg_sq": {
            "maximum_absolute_difference": 2.0**-36,
            "maximum_relative_difference": 1.0e-5,
            "maximum_ulp_distance": 32,
        },
        "param_groups_exact": True,
        "step_exact": True,
        "terminal_model_exact": True,
    }
    assert len(payload["optimizer_diagnostic_sha256"]) == 64


def test_runner_loads_only_canonical_registered_elastic_protocol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(freezer, "load_production_corpus", lambda *a, **k: _FakeCorpus())
    payload = freezer.freeze_gate(
        source_root=ROOT,
        corpus_root=ROOT / "ignored-corpus",
        manifest_file_sha256="a" * 64,
        manifest_payload_sha256="b" * 64,
        corpus_protocol_file_sha256="c" * 64,
        independent_audit_path=ROOT / "ignored-audit.json",
        independent_audit_file_sha256="d" * 64,
        independent_audit_payload_sha256="e" * 64,
    )
    path = tmp_path / "gate.json"
    path.write_bytes(runner._canonical_json_bytes(payload))
    loaded, digest, elastic = runner._load_gate_protocol(path, "stage2_2")
    assert loaded == payload
    assert len(digest) == 64
    assert elastic is True
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")
    with pytest.raises(RuntimeError, match="canonical"):
        runner._load_gate_protocol(path, "stage2_2")


def test_elastic_runner_and_auditor_are_syntax_valid_and_fail_closed() -> None:
    runner_source = (
        ROOT / "scripts/run_antibody_pretraining_production_gate.py"
    ).read_text(encoding="ascii")
    auditor_source = (
        ROOT / "scripts/audit_antibody_elastic_recovery.py"
    ).read_text(encoding="ascii")
    ast.parse(runner_source)
    ast.parse(auditor_source)
    for required in (
        "authenticate_source_rng_manifest(",
        "destination_rank_seed(",
        "semantic_trainer_state_sha256(",
        "Elastic DCP payload differs from its source receipt",
    ):
        assert required in runner_source
    for required in (
        "same_world_prefix_exact",
        "trace_numerical_noninferiority",
        "source_rng_inventories_authenticated",
        "terminal_coordinate_exact",
        "terminal_model_exact",
        "terminal_optimizer_moments_within_gate",
    ):
        assert required in auditor_source


def test_scaled_error_uses_the_frozen_elementwise_rule() -> None:
    assert auditor._scaled_error(
        1.01,
        1.0,
        absolute_tolerance=0.005,
        relative_tolerance=0.02,
    ) == pytest.approx(0.4)


def test_checkpoint_chain_compares_optimizer_hashes_within_their_domain() -> None:
    source = {
        "receipt": {
            "checkpoint_model_sha256": "a" * 64,
            "final_model_sha256": "a" * 64,
            "checkpoint_optimizer_sha256": "b" * 64,
            "final_optimizer_sha256": "c" * 64,
            "final_semantic_optimizer_sha256": "9" * 64,
            "checkpoint_rng_manifest_sha256": "d" * 64,
            "checkpoint_state_hash_domains": {
                "model": "torch_distributed_checkpoint_canonical_state_dict_v1",
                "optimizer": (
                    "torch_distributed_checkpoint_canonical_optimizer_state_dict_v1"
                ),
                "rng": "authenticated_rank_rng_manifest_v1",
            },
            "final_state_hash_domains": {
                "model": "unwrapped_module_state_dict_v1",
                "optimizer": "native_optimizer_state_dict_v1",
            },
            "elastic_resume": {"destination_semantic_state_sha256": "e" * 64},
        },
        "state": {"world_size": 4},
    }
    destination = {
        "receipt": {
            "resume_payload_model_sha256": "a" * 64,
            "resume_loaded_model_sha256": "a" * 64,
            "resume_payload_optimizer_sha256": "b" * 64,
            "resume_loaded_optimizer_sha256": "c" * 64,
            "resume_loaded_semantic_optimizer_sha256": "9" * 64,
            "resume_rng_manifest_sha256": "d" * 64,
            "elastic_resume": {
                "cross_world": True,
                "rng_policy": "authenticate_source_then_topology_reseed",
                "source_semantic_state_sha256": "e" * 64,
                "source_world_size": 4,
                "source_rng_authentication": {
                    "manifest_sha256": "d" * 64,
                    "source_rank_state_sha256": ["f" * 64] * 4,
                    "source_world_size": 4,
                },
            },
        }
    }
    auditor._validate_checkpoint_link(source, destination)
    destination["receipt"]["resume_loaded_optimizer_sha256"] = "0" * 64
    with pytest.raises(auditor.ElasticRecoveryAuditError, match="chain"):
        auditor._validate_checkpoint_link(source, destination)


def test_optimizer_diagnostic_uses_frozen_moment_limits(tmp_path: Path) -> None:
    continuous = tmp_path / "continuous"
    stage3 = tmp_path / "stage3"
    (continuous / "checkpoint").mkdir(parents=True)
    (stage3 / "checkpoint").mkdir(parents=True)
    implementation = ROOT / "scripts/diagnose_antibody_optimizer_state.py"
    files = [{"bytes": 1, "path": ".metadata", "sha256": "a" * 64}]
    field = {
        "different_elements": 1,
        "different_tensors": 1,
        "maximum_absolute_difference": 1.0e-10,
        "maximum_relative_difference": 1.0e-4,
        "maximum_ulp_distance": 8,
        "tensor_count": 2,
    }
    payload = {
        "authorization": {
            "diagnostic_only": True,
            "production_pretraining": False,
            "scientific_claim": False,
        },
        "checkpoint_inventories": {
            key: {
                "files": files,
                "inventory_sha256": hashlib.sha256(
                    auditor._canonical(files)
                ).hexdigest(),
            }
            for key in ("left", "right")
        },
        "checkpoint_roots": {
            "left": str((continuous / "checkpoint").resolve()),
            "right": str((stage3 / "checkpoint").resolve()),
        },
        "field_summaries": {
            "exp_avg": dict(field),
            "exp_avg_sq": {
                **field,
                "maximum_absolute_difference": 1.0e-12,
                "maximum_relative_difference": 1.0e-6,
                "maximum_ulp_distance": 4,
            },
            "step": {
                "different_elements": 0,
                "different_tensors": 0,
                "maximum_absolute_difference": 0.0,
                "maximum_relative_difference": 0.0,
                "maximum_ulp_distance": 0,
                "tensor_count": 2,
            },
        },
        "format_version": "antibody_optimizer_state_diagnostic_v1",
        "implementation_sha256": hashlib.sha256(
            implementation.read_bytes()
        ).hexdigest(),
        "model": {
            "different_tensors": 0,
            "exact": True,
            "tensor_differences": {},
        },
        "optimizer": {
            "exact": False,
            "named_parameter_count": 2,
            "non_tensor_differences": [],
            "param_groups_equal": True,
            "top_tensor_differences": [],
        },
        "preset": "abf_35m",
        "test_access": {
            "downstream_labels_used": False,
            "sabdab2_heldout_accessed": False,
        },
    }
    payload["diagnostic_sha256"] = hashlib.sha256(
        auditor._canonical(payload)
    ).hexdigest()
    path = tmp_path / "diagnostic.json"
    path.write_bytes(auditor._canonical(payload))
    protocol = {
        "optimizer_diagnostic_sha256": payload["implementation_sha256"],
        "numerical_gate": {
            "optimizer_state": {
                "exp_avg": {
                    "maximum_absolute_difference": 2.0**-29,
                    "maximum_relative_difference": 0.001,
                    "maximum_ulp_distance": 4096,
                },
                "exp_avg_sq": {
                    "maximum_absolute_difference": 2.0**-36,
                    "maximum_relative_difference": 1.0e-5,
                    "maximum_ulp_distance": 32,
                },
                "param_groups_exact": True,
                "step_exact": True,
                "terminal_model_exact": True,
            }
        },
    }
    loaded, digest = auditor._validate_optimizer_diagnostic(
        path,
        protocol=protocol,
        continuous_root=continuous,
        stage3_root=stage3,
    )
    assert loaded == payload
    assert len(digest) == 64


def test_slurm_gate_uses_only_public_4090_and_one_allocation() -> None:
    source = (ROOT / "slurm/antibody_elastic_recovery_gate.sbatch").read_text(
        encoding="ascii"
    )
    assert "#SBATCH --partition=gpu_4090" in source
    assert "#SBATCH --gres=gpu:4" in source
    assert "--mem" not in source
    assert "hp_4090" not in source
    assert "/ssd/scxi253/model_deployment/envs/contact/bin/python3.10" in source
    assert "FIRST_TWO_VISIBLE" in source
    assert "continuous_4" in source
    assert "stage1_4" in source
    assert "stage2_2" in source
    assert "stage3_4" in source
