"""Tests for authenticated SAbDab2 structure batches."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


pytestmark = pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in ("Bio", "pandas", "pyarrow", "torch")
    ),
    reason="SAbDab2 structure test dependencies are unavailable",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pdb_atom(
    serial: int,
    atom: str,
    residue: str,
    chain: str,
    residue_number: int,
    xyz: tuple[float, float, float],
) -> str:
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain:1s}"
        f"{residue_number:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        "  1.00 20.00           "
        f"{atom[0]:>2s}\n"
    )


def _cif_payload(tmp_path: Path) -> bytes:
    from Bio.PDB import MMCIFIO, PDBParser

    residues = (
        ("H", 1, "ALA", 0.0, False),
        ("H", 2, "CYS", 4.0, False),
        ("L", 1, "ASP", 8.0, False),
        ("A", 1, "GLU", 12.0, False),
        ("A", 2, "PHE", 16.0, True),
    )
    lines: list[str] = []
    serial = 1
    for chain, number, name, offset, omit_c in residues:
        atoms = (
            ("N", (offset, 1.0, 0.0)),
            ("CA", (offset, 0.0, 0.0)),
            ("C", (offset + 1.0, 0.0, 0.0)),
        )
        for atom, xyz in atoms:
            if omit_c and atom == "C":
                continue
            lines.append(_pdb_atom(serial, atom, name, chain, number, xyz))
            serial += 1
        lines.append("TER\n")
    lines.append("END\n")
    structure = PDBParser(QUIET=True).get_structure("synthetic", io.StringIO("".join(lines)))
    path = tmp_path / "synthetic.cif"
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(str(path))
    return path.read_bytes()


def _numbering(sequence: str) -> str:
    return json.dumps([[amino_acid, [index, " "]] for index, amino_acid in enumerate(sequence, 1)])


def _metadata(instance_id: str) -> dict[str, str]:
    return {
        "INSTANCE": instance_id,
        "Hchain": "H",
        "Lchain": "L",
        "agchains": "A",
        "VH_numbering_list": _numbering("AC"),
        "VL_numbering_list": _numbering("D"),
        "ag_index_list": _numbering("EF"),
        "ab_ag_split": "train",
    }


def _canonical(instance_id: str) -> dict[str, str]:
    return {"INSTANCE": instance_id, "vh_seq": "AC", "vl_seq": "D", "ag_seq": "EF"}


def _prepared(instance_id: str) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "ab_sequence_sha256": hashlib.sha256(b"ACD").hexdigest(),
        "ag_sequence_sha256": hashlib.sha256(b"EF").hexdigest(),
        "ab_length": 3,
        "ag_length": 2,
        "paratope_positive_count": 1,
        "epitope_positive_count": 1,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_tensor_artifact_preserves_missing_frames_and_chain_ids(tmp_path: Path) -> None:
    import torch

    from interaction_first.sabdab2_structure import _tensor_artifact

    artifact = _tensor_artifact(
        "train_case",
        "train",
        _metadata("train_case"),
        _canonical("train_case"),
        _prepared("train_case"),
        _cif_payload(tmp_path),
    )
    assert artifact["antibody_backbone"].shape == (3, 3, 3)
    assert artifact["antigen_backbone"].shape == (2, 3, 3)
    assert torch.equal(artifact["antibody_chain_ids"], torch.tensor([0, 0, 1]))
    assert torch.equal(artifact["antigen_chain_ids"], torch.tensor([0, 0]))
    assert torch.equal(
        artifact["antigen_frame_valid_mask"], torch.tensor([True, False])
    )
    assert torch.equal(
        artifact["antigen_missing_frame_mask"], torch.tensor([False, True])
    )
    assert torch.isnan(artifact["antigen_backbone"][1, 2]).all()


def test_materialized_cache_joins_embeddings_labels_and_batches(tmp_path: Path) -> None:
    import numpy as np
    import pandas as pd
    import torch

    from interaction_first.sabdab2_structure import (
        Sabdab2StructureBatchCollator,
        Sabdab2StructureEmbeddingDataset,
        materialize_structure_cache,
    )

    instance_ids = ("train_case", "validation_case")
    native_test_id = "native_test_case"
    payload = _cif_payload(tmp_path)
    native_test_metadata = _metadata(native_test_id)
    native_test_metadata["ab_ag_split"] = "test"
    metadata_rows = [
        *[_metadata(identifier) for identifier in instance_ids],
        native_test_metadata,
    ]
    archive_path = tmp_path / "splits.tar.gz"
    metadata_buffer = io.StringIO()
    writer = csv.DictWriter(metadata_buffer, fieldnames=list(metadata_rows[0]))
    writer.writeheader()
    writer.writerows(metadata_rows)
    with tarfile.open(archive_path, "w:gz") as archive:
        metadata_bytes = metadata_buffer.getvalue().encode()
        info = tarfile.TarInfo("splits_final/abag_split.csv")
        info.size = len(metadata_bytes)
        archive.addfile(info, io.BytesIO(metadata_bytes))
        for identifier in instance_ids:
            info = tarfile.TarInfo(f"splits_final/{identifier}.cif")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        invalid_test_payload = b"this native-test member must not be parsed"
        info = tarfile.TarInfo(f"splits_final/{native_test_id}.cif")
        info.size = len(invalid_test_payload)
        archive.addfile(info, io.BytesIO(invalid_test_payload))

    split_path = tmp_path / "split.json"
    split_payload = {
        "format_version": "test",
        "intended_use": "development_only",
        "valid": True,
        "frozen_test_labels_or_predictions_read": False,
        "frozen_test_sequences_aligned": False,
        "scientific_payload_sha256": "0" * 64,
        "splits": {
            "train": [instance_ids[0]],
            "validation": [instance_ids[1], native_test_id],
        },
    }
    split_path.write_text(json.dumps(split_payload), encoding="utf-8")
    split_sha256 = _sha256(split_path)
    verification_path = tmp_path / "verification.json"
    zero_audit = {
        "calm_scaled_similarity_violation_count": 0,
        "identity_and_reciprocal_coverage_violation_count": 0,
        "exact_sequence_cross_split_overlap_count": 0,
    }
    verification_path.write_text(
        json.dumps(
            {
                "format_version": "test_audit",
                "intended_use": "development_only",
                "development_instance_count": 3,
                "frozen_test_labels_or_predictions_read": False,
                "frozen_test_sequences_aligned": False,
                "provenance": {
                    "audited_splits": {"nw_chain_strict": {"sha256": split_sha256}}
                },
                "results": {
                    "nw_chain_strict": {
                        "roles": {"antibody": zero_audit, "antigen": zero_audit}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    canonical_path = tmp_path / "canonical.csv"
    _write_csv(canonical_path, [_canonical(identifier) for identifier in instance_ids])
    prepared_path = tmp_path / "instances.parquet"
    pd.DataFrame([_prepared(identifier) for identifier in instance_ids]).to_parquet(
        prepared_path, index=False
    )
    cache_root = tmp_path / "cache"
    receipt = materialize_structure_cache(
        archive_path=archive_path,
        archive_sha256=_sha256(archive_path),
        split_path=split_path,
        split_sha256=split_sha256,
        split_verification_path=verification_path,
        split_verification_sha256=_sha256(verification_path),
        canonical_manifest_path=canonical_path,
        canonical_manifest_sha256=_sha256(canonical_path),
        prepared_instances_path=prepared_path,
        prepared_instances_sha256=_sha256(prepared_path),
        output_root=cache_root,
    )
    assert receipt["role_counts"] == {"train": 1, "validation": 1}
    manifest = json.loads(
        (cache_root / "STRUCTURE_CACHE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["authority"]["sealed_test_cif_members_read"] is False
    assert {
        "instance_id": native_test_id,
        "reason": "new_release_native_role_not_train",
    } in manifest["exclusions"]

    embedding_root = tmp_path / "embeddings"
    label_root = tmp_path / "labels"
    label_root.mkdir()
    for identifier in instance_ids:
        prepared = _prepared(identifier)
        for partner, sequence, length in (("ab", "ACD", 3), ("ag", "EF", 2)):
            digest = str(prepared[f"{partner}_sequence_sha256"])
            path = embedding_root / digest[:2] / f"{digest}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "format_version": "sppider_seq_embedding_v1",
                    "sequence_sha256": digest,
                    "sequence_length": length,
                    "starts": [0],
                    "ends": [length],
                    "chunks": [torch.arange(length * 8).view(length, 8).float()],
                },
                path,
            )
        np.savez(
            label_root / f"{identifier}.npz",
            ab_labels=np.array([1, 0, 0], dtype=np.float32),
            ag_labels=np.array([0, 1], dtype=np.float32),
        )
    train = Sabdab2StructureEmbeddingDataset(
        cache_root, embedding_root, label_root, split_role="train"
    )
    validation = Sabdab2StructureEmbeddingDataset(
        cache_root, embedding_root, label_root, split_role="validation"
    )
    batch = Sabdab2StructureBatchCollator()([train[0], validation[0]])
    assert batch["antibody_features"].shape == (2, 3, 8)
    assert batch["antigen_features"].shape == (2, 2, 8)
    assert torch.equal(
        batch["antigen_missing_frame_mask"],
        torch.tensor([[False, True], [False, True]]),
    )
    assert torch.equal(
        batch["antigen_frame_valid_mask"] | batch["antigen_missing_frame_mask"],
        batch["antigen_mask"],
    )

    output_root = tmp_path / "ablation"
    subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "train_sabdab2_structure_ablation.py"
            ),
            "--cache-root",
            str(cache_root),
            "--embedding-root",
            str(embedding_root),
            "--label-root",
            str(label_root),
            "--output-root",
            str(output_root),
            "--expected-cache-manifest-sha256",
            _sha256(cache_root / "STRUCTURE_CACHE_MANIFEST.json"),
            "--modes",
            "none",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--max-antigen-length",
            "10",
            "--d-model",
            "16",
            "--num-heads",
            "4",
            "--num-rbf",
            "8",
            "--dropout",
            "0",
            "--precision",
            "fp32",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(
        (output_root / "SABDAB2_SELF_ATTENTION_ABLATION.json").read_text()
    )
    assert result["data"]["train_instance_count"] == 1
    assert result["data"]["validation_instance_count"] == 1
    assert result["modes"][0]["selection_rule"].startswith("fixed final epoch")


def test_four_classifier_modes_have_equal_parameters_and_finite_outputs() -> None:
    import torch

    from interaction_first.sabdab2_structure import IntraChainResidueClassifier

    antibody_backbone = torch.tensor(
        [[[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]] * 3]
    )
    antigen_backbone = antibody_backbone[:, :2].clone()
    batch = {
        "antibody_features": torch.randn(1, 3, 8),
        "antigen_features": torch.randn(1, 2, 8),
        "antibody_backbone": antibody_backbone,
        "antigen_backbone": antigen_backbone,
        "antibody_mask": torch.ones(1, 3, dtype=torch.bool),
        "antigen_mask": torch.ones(1, 2, dtype=torch.bool),
        "antibody_chain_ids": torch.tensor([[0, 0, 1]]),
        "antigen_chain_ids": torch.tensor([[0, 0]]),
        "antibody_frame_valid_mask": torch.ones(1, 3, dtype=torch.bool),
        "antigen_frame_valid_mask": torch.ones(1, 2, dtype=torch.bool),
        "antibody_missing_frame_mask": torch.zeros(1, 3, dtype=torch.bool),
        "antigen_missing_frame_mask": torch.zeros(1, 2, dtype=torch.bool),
    }
    counts = []
    for mode in ("none", "distance", "distance_direction", "full_pose"):
        model = IntraChainResidueClassifier(
            antibody_input_dim=8,
            antigen_input_dim=8,
            d_model=16,
            num_heads=4,
            geometry_feature_mode=mode,
            num_rbf=8,
            dropout=0.0,
        ).eval()
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        output = model(batch)
        assert torch.isfinite(output["paratope_logits"]).all()
        assert torch.isfinite(output["epitope_logits"]).all()
    assert len(set(counts)) == 1
