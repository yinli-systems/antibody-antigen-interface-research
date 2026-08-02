"""Contracts for label-free, receipt-gated AsEP test preparation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest


_AA1_TO_3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pdb_payload() -> bytes:
    sequences = {"H": "AC", "L": "DE", "a": "FGHI"}
    lines: list[str] = []
    for chain, sequence in sequences.items():
        tokens = " ".join(_AA1_TO_3[letter] for letter in sequence)
        lines.append(f"SEQRES {1:3d} {chain} {len(sequence):4d}  {tokens}")
    serial = 1
    residue = 1
    for chain, sequence in sequences.items():
        for letter in sequence:
            for atom_index, atom in enumerate(("N", "CA", "C")):
                x = float(residue * 10 + atom_index)
                lines.append(
                    f"ATOM  {serial:5d} {atom:>4s} {_AA1_TO_3[letter]:>3s} "
                    f"{chain}{residue:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
                    f"{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}"
                )
                serial += 1
            residue += 1
    return ("\n".join((*lines, "TER", "END")) + "\n").encode("ascii")


def _write_selection(root: Path) -> Path:
    ensemble = root / "ensemble"
    ensemble.mkdir(parents=True)
    ensemble_result_path = (
        ensemble / "FRAMECONTACT_ASEP_DEVELOPMENT_ENSEMBLE.json"
    )
    ensemble_result_path.write_text(
        json.dumps(
            {
                "format_version": "framecontact_asep_development_ensemble_v1",
                "experiment": "framecontact_interleaved",
                "seeds": [17, 42, 73, 101, 137],
                "ensemble_rule": {"member_weighting": "uniform 1/5"},
                "authority": {
                    "dataset_backend": "asep",
                    "members": [{"seed": seed} for seed in (17, 42, 73, 101, 137)],
                    "test_loaded": False,
                    "external_test_loaded": False,
                },
                "validation": {
                    "selection_metric": 0.1,
                    "selection_metric_rule": "minimum_direction_cluster_weighted_margin",
                    "specificity_score_source": "interaction_only",
                    "contact": {"mean_per_complex_auprc": 0.2},
                },
            }
        )
    )
    ensemble_receipt_path = ensemble / "ENSEMBLE_RECEIPT.json"
    ensemble_receipt_path.write_text(
        json.dumps(
            {
                "format_version": "framecontact_asep_development_ensemble_v1",
                "result_sha256": _sha256(ensemble_result_path),
                "test_loaded": False,
                "external_test_loaded": False,
            }
        )
    )
    selection = root / "selection"
    selection.mkdir()
    selection_result_path = selection / "ASEP_DEVELOPMENT_SELECTION.json"
    selection_result_path.write_text(
        json.dumps(
            {
                "format_version": "framecontact_asep_development_selection_v1",
                "selected_experiment": "framecontact_interleaved",
                "selection_rule": {
                    "primary": "maximize minimum_direction_cluster_weighted_margin",
                    "first_tie_break": "maximize mean_per_complex_contact_auprc",
                    "test_role_forbidden": True,
                },
                "selected_ensemble": {
                    "root": str(ensemble),
                    "result_sha256": _sha256(ensemble_result_path),
                    "receipt_sha256": _sha256(ensemble_receipt_path),
                    "selection_metric": 0.1,
                    "contact_mean_per_complex_auprc": 0.2,
                },
                "all_candidates": [
                    {
                        "experiment": experiment,
                        "root": str(ensemble) if index == 0 else f"unused-{index}",
                        "result_sha256": (
                            _sha256(ensemble_result_path) if index == 0 else f"{index}" * 64
                        ),
                        "receipt_sha256": (
                            _sha256(ensemble_receipt_path) if index == 0 else f"{index}" * 64
                        ),
                        "selection_metric": 0.1 if index == 0 else -float(index),
                        "contact_mean_per_complex_auprc": 0.2 if index == 0 else 0.0,
                    }
                    for index, experiment in enumerate(
                        (
                            "framecontact_interleaved",
                            "sequence_dense",
                            "spatial_topology",
                            "distance",
                            "distance_direction",
                            "full_pose",
                            "frozen_frame_ipa",
                            "framecontact",
                        )
                    )
                ],
                "authority": {
                    "test_loaded": False,
                    "external_test_loaded": False,
                },
            }
        )
    )
    selection_receipt = selection / "SELECTION_RECEIPT.json"
    selection_receipt.write_text(
        json.dumps(
            {
                "format_version": "framecontact_asep_development_selection_v1",
                "result_sha256": _sha256(selection_result_path),
                "selected_experiment": "framecontact_interleaved",
                "selected_ensemble_result_sha256": _sha256(ensemble_result_path),
                "test_loaded": False,
                "external_test_loaded": False,
            }
        )
    )
    return selection


def _write_authorities(root: Path) -> tuple[Path, Path, Path]:
    from interaction_first.asep_authority import (
        INNER_AUDIT_FORMAT,
        SPLIT_MANIFEST_FORMAT,
    )

    authority = root / "authority"
    payload = authority / "payload/asepv1.2.0"
    payload.mkdir(parents=True)
    structure_archive = payload / "structures.tar.gz"
    with tarfile.open(structure_archive, "w:gz") as archive:
        directory = tarfile.TarInfo("structures")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for identifier in ("1abc_0P", "2def_0P", "3ghi_0P"):
            raw = _pdb_payload()
            member = tarfile.TarInfo(f"structures/{identifier}.pdb")
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    inner = root / "inner.json"
    inner.write_text(
        json.dumps(
            {
                "format_version": INNER_AUDIT_FORMAT,
                "authority": {
                    "task_labels_read": False,
                    "model_predictions_or_scores_read": False,
                },
                "checks": {"split_payload_opened": False},
                "structure_archive": {"sha256": _sha256(structure_archive)},
            }
        )
    )
    split_root = root / "splits"
    split_root.mkdir()
    split_manifest = split_root / "epitope_group.csv"
    with split_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("dataset_index", "instance_id", "role")
        )
        writer.writeheader()
        writer.writerows(
            [
                {"dataset_index": 0, "instance_id": "1abc_0P", "role": "train"},
                {"dataset_index": 1, "instance_id": "2def_0P", "role": "val"},
                {"dataset_index": 2, "instance_id": "3ghi_0P", "role": "test"},
            ]
        )
    split_receipt = split_root / "ASEP_SPLIT_MANIFEST_RECEIPT.json"
    split_receipt.write_text(
        json.dumps(
            {
                "format_version": SPLIT_MANIFEST_FORMAT,
                "authority": {
                    "task_labels_read": False,
                    "model_predictions_or_scores_read": False,
                },
                "instance_count": 3,
                "manifests": {
                    "epitope_group": {
                        "path": split_manifest.name,
                        "sha256": _sha256(split_manifest),
                        "role_counts": {"train": 1, "val": 1, "test": 1},
                    }
                },
            }
        )
    )
    return authority, inner, split_root


def test_label_free_test_materialization_requires_frozen_selection(
    tmp_path: Path,
) -> None:
    from interaction_first.asep_frozen_test import (
        materialize_asep_label_free_test_metadata,
    )

    selection = _write_selection(tmp_path)
    authority, inner, split_root = _write_authorities(tmp_path)
    output = tmp_path / "test_metadata"
    result = materialize_asep_label_free_test_metadata(
        authority_root=authority,
        inner_audit_path=inner,
        inner_audit_sha256=_sha256(inner),
        split_root=split_root,
        split_receipt_sha256=_sha256(
            split_root / "ASEP_SPLIT_MANIFEST_RECEIPT.json"
        ),
        split_method="epitope_group",
        selection_root=selection,
        selection_receipt_sha256=_sha256(
            selection / "SELECTION_RECEIPT.json"
        ),
        output_root=output,
    )
    assert result["test_instance_count"] == 1
    assert result["test_label_open_count"] == 0
    assert result["checks"]["official_test_graph_tensors_opened"] is False
    rows = list(
        csv.DictReader((output / "ASEP_LABEL_FREE_TEST_METADATA.csv").open())
    )
    assert [row["instance_id"] for row in rows] == ["3ghi_0P"]
    antigen_path = output / rows[0]["antigen_structure_path"]
    assert antigen_path.is_file()
    assert len(rows[0]["foldseek_structure_id"]) == 64
    assert antigen_path.stem == rows[0]["foldseek_structure_id"]
    assert {line[21] for line in antigen_path.read_text().splitlines() if line.startswith("ATOM")} == {"a"}
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_asep_label_free_test_metadata(
            authority_root=authority,
            inner_audit_path=inner,
            inner_audit_sha256=_sha256(inner),
            split_root=split_root,
            split_receipt_sha256=_sha256(
                split_root / "ASEP_SPLIT_MANIFEST_RECEIPT.json"
            ),
            split_method="epitope_group",
            selection_root=selection,
            selection_receipt_sha256=_sha256(
                selection / "SELECTION_RECEIPT.json"
            ),
            output_root=output,
        )


def test_label_free_test_rejects_selection_that_opened_test(tmp_path: Path) -> None:
    from interaction_first.asep_frozen_test import (
        AsepExternalError,
        authenticate_asep_development_selection,
    )

    selection = _write_selection(tmp_path)
    receipt_path = selection / "SELECTION_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["test_loaded"] = True
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(AsepExternalError, match="pre-opening eligible"):
        authenticate_asep_development_selection(
            selection, selection_receipt_sha256=_sha256(receipt_path)
        )
