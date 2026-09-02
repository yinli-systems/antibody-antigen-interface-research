from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "remote_tools" / "sable_external_v1"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_measurement_source_manifest_binds_every_frozen_file() -> None:
    manifest_path = TOOLS / "SABLE_V0_2_0_MEASUREMENT_SOURCE_MANIFEST_V1.json"
    receipt_path = (
        TOOLS / "SABLE_V0_2_0_MEASUREMENT_SOURCE_MANIFEST_V1_RECEIPT.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["manifest_sha256"] == _sha256(manifest_path)
    assert manifest["status"] == "FROZEN_MEASUREMENT_SOURCE_BEFORE_BOUND_LABEL_OPENING"
    assert len(manifest["records"]) == 8
    for record in manifest["records"]:
        assert _sha256(TOOLS / record["relative_path"]) == record["sha256"]


def test_contact_definition_includes_exact_4_5_angstrom_boundary() -> None:
    labels = _load_module(
        "sable_labels_once", "materialize_sable_v0_2_0_bound_labels_once_v1.py"
    )
    antibody = [
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[20.0, 0.0, 0.0]], dtype=np.float32),
    ]
    antigen = [
        np.asarray([[4.5, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[30.0, 0.0, 0.0]], dtype=np.float32),
    ]

    assert labels.positive_contacts(antibody, antigen).tolist() == [0]


def test_recovery_normalizes_only_audited_modified_residues() -> None:
    labels = _load_module(
        "sable_labels_recovery_v2",
        "materialize_sable_v0_2_0_bound_labels_once_recovery_v2.py",
    )

    assert labels.MODIFIED_AMINO_ACID_PARENT == {
        "CSO": "C",
        "HYP": "P",
        "MSE": "M",
        "PCA": "Q",
        "SEP": "S",
        "TYS": "Y",
    }


def test_recovery_maps_repeated_physical_chain_once() -> None:
    labels = _load_module(
        "sable_labels_repeated_chain_v2",
        "materialize_sable_v0_2_0_bound_labels_once_recovery_v2.py",
    )
    observed = [
        {"amino_acid": amino_acid, "heavy_atoms": np.asarray([[index, 0, 0]])}
        for index, amino_acid in enumerate("ABCD")
    ]
    observed_calls: list[str] = []
    mapping_calls: list[tuple[str, str]] = []

    def fake_observed_chain(_model, chain_id):
        observed_calls.append(chain_id)
        return observed

    def fake_mapping(_observed, target, label):
        mapping_calls.append((target, label))
        return {index: index for index in range(len(target))}

    labels.observed_chain = fake_observed_chain
    labels.exact_order_preserving_mapping = fake_mapping
    sequence, mask, atoms = labels.partner_axis(
        object(), ("A", "A"), ("AB", "CD"), "scFv antibody"
    )

    assert sequence == "ABCD"
    assert mask.tolist() == [True, True, True, True]
    assert [int(row[0, 0]) for row in atoms] == [0, 1, 2, 3]
    assert observed_calls == ["A"]
    assert mapping_calls == [("ABCD", "scFv antibody chain A")]


def test_pdb_equal_weighted_bootstrap_is_deterministic() -> None:
    analysis = _load_module(
        "sable_analysis_once", "analyze_sable_v0_2_0_external_once_v1.py"
    )
    per_seed = {
        17: {"a": 0.1, "b": 0.3, "c": 0.9},
        29: {"a": 0.2, "b": 0.4, "c": 0.8},
        43: {"a": 0.3, "b": 0.5, "c": 0.7},
    }
    pdb_by_entry = {"a": "pdb-1", "b": "pdb-1", "c": "pdb-2"}

    first = analysis.cluster_weighted_estimate(per_seed, pdb_by_entry)
    second = analysis.cluster_weighted_estimate(per_seed, pdb_by_entry)

    assert first == second
    assert first["point_estimate"] == np.mean([0.3, 0.8])
    assert first["eligible_entry_count"] == 3
    assert first["unique_pdb_id_count"] == 2
