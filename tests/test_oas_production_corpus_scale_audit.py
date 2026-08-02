"""Validate exact OAS corpus-scale measurement and its non-authorizing boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/oas_production_corpus_scale_audit_v1.json"
AUDITOR = ROOT / "scripts/audit_oas_production_corpus_scale.py"
SLURM = ROOT / "slurm/oas_production_corpus_scale_audit.sbatch"


def _load_auditor():
    specification = importlib.util.spec_from_file_location("oas_scale_audit", AUDITOR)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_protocol_binds_corpus_and_large_model_threshold() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    assert value["format_version"] == "oas_production_corpus_scale_audit_protocol_v1"
    assert set(value["authorization"].values()) == {False}
    assert value["test_access"]["sabdab2_heldout_accessed"] is False
    bindings = {
        "corpus_finalize_receipt_file_sha256": "results/pretraining/oas_production_materialization_v3_corpus_finalize_receipt.json",
        "corpus_manifest_file_sha256": "results/pretraining/oas_production_materialization_v3_corpus_manifest.json",
        "efficiency_protocol_file_sha256": "protocols/antibody_efficiency_gate_v1.json",
        "large_model_protocol_file_sha256": "protocols/antibody_large_model_pilot_v1.json",
        "materialization_protocol_file_sha256": "protocols/oas_production_materialization_v3.json",
    }
    for key, relative in bindings.items():
        assert value["inputs"][key] == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    threshold = value["acceptance"][
        "minimum_large_model_train_unique_residue_tokens"
    ]
    large_model = json.loads(
        (ROOT / "protocols/antibody_large_model_pilot_v1.json").read_text(
            encoding="ascii"
        )
    )
    efficiency = json.loads(
        (ROOT / "protocols/antibody_efficiency_gate_v1.json").read_text(
            encoding="ascii"
        )
    )
    assert threshold == 30_000_000_000
    assert large_model["production_gate"]["minimum_admitted_unique_residue_tokens"] == threshold
    assert efficiency["large_model_gate"]["minimum_admitted_unique_residue_tokens"] == threshold


def test_protocol_binds_manifest_payloads_and_exact_split_counts() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="ascii"))
    manifest = json.loads(
        (ROOT / "results/pretraining/oas_production_materialization_v3_corpus_manifest.json").read_text(
            encoding="ascii"
        )
    )
    finalizer = json.loads(
        (ROOT / "results/pretraining/oas_production_materialization_v3_corpus_finalize_receipt.json").read_text(
            encoding="ascii"
        )
    )
    assert value["inputs"]["corpus_manifest_payload_sha256"] == manifest["manifest_sha256"]
    assert value["inputs"]["corpus_finalize_receipt_payload_sha256"] == finalizer["receipt_sha256"]
    assert value["acceptance"]["train_record_count"] == manifest["counts"]["unique_train_records"]
    assert value["acceptance"]["validation_record_count"] == manifest["counts"]["unique_validation_records"]


def test_array_histogram_returns_exact_counts_and_sum() -> None:
    module = _load_auditor()
    array = np.asarray([80, 80, 81, 192], dtype=np.uint16)
    histogram, total = module._array_histogram(
        array,
        bin_count=193,
        minimum=80,
        maximum=192,
    )
    assert histogram[80] == 2
    assert histogram[81] == 1
    assert histogram[192] == 1
    assert sum(histogram) == 4
    assert total == 433


def test_array_histogram_rejects_out_of_domain_value() -> None:
    module = _load_auditor()
    with pytest.raises(module.CorpusScaleAuditError, match="outside its domain"):
        module._array_histogram(
            np.asarray([79, 80], dtype=np.uint16),
            bin_count=193,
            minimum=80,
            maximum=192,
        )


def test_shard_auditor_rejects_noncanonical_relative_root(tmp_path: Path) -> None:
    module = _load_auditor()
    with pytest.raises(module.CorpusScaleAuditError, match="relative root differs"):
        module._audit_shard(
            corpus_root=tmp_path,
            bucket=0,
            split="train",
            reference={"relative_root": "../escape"},
            study_count=90,
            minimum_length=80,
            maximum_length=192,
        )


def test_auditor_uses_only_composition_arrays_and_is_create_only() -> None:
    source = AUDITOR.read_text(encoding="ascii")
    assert '"sequence_lengths.npy"' in source
    assert '"chain_types.npy"' in source
    assert '"tokens.npy"' not in source
    assert "ThreadPoolExecutor" in source
    assert "os.O_EXCL" in source
    assert "mmap_mode=\"r\"" in source


def test_slurm_launcher_hides_scheduler_required_gpu_and_binds_runtime() -> None:
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=8" in source
    assert "export CUDA_VISIBLE_DEVICES=" in source
    assert "OAS_SCALE_INDEPENDENT_AUDIT_FILE_SHA256" in source
    assert "OAS_SCALE_INDEPENDENT_AUDIT_PAYLOAD_SHA256" in source
    assert "OAS_SCALE_PROTOCOL_SHA256" in source
    assert "OAS_SCALE_AUDITOR_SHA256" in source
    assert 'test ! -e "${OUTPUT}"' in source
    assert "generate_runtime_manifest.py\" verify" in source
    assert 'exec "${PYTHON_BIN}" "${AUDITOR}"' in source
    assert "|| true" not in source
