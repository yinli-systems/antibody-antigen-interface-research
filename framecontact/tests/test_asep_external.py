"""Restricted-loader and sealed-test tests for the AsEP graph adapter."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import torch


def _sparse(shape: tuple[int, int], edges: tuple[tuple[int, int], ...]) -> torch.Tensor:
    indices = torch.tensor(edges, dtype=torch.long).T.contiguous()
    values = torch.ones(len(edges), dtype=torch.int8)
    return torch.sparse_coo_tensor(
        indices, values, shape, check_invariants=True
    ).coalesce()


def _graph(identifier: str = "1abc_0P") -> dict:
    return {
        "Nb": np.int64(2),
        "Ng": np.int64(3),
        "abdbid": identifier,
        "seqres": {"ab": {"H": "AC", "L": "DE"}, "ag": {"a": "FGHI"}},
        "mapping": {
            "ab": {"seqres2cdr": np.asarray([1, 0, 1, 0], dtype=np.int64)},
            "ag": {
                "seqres2surf": np.asarray([1, 1, 0, 1], dtype=np.int64),
                "seqres2epitope": np.asarray([0, 1, 0, 1], dtype=np.int64),
            },
        },
        "embedding": {
            "ab": {
                "esm2": torch.arange(4 * 480, dtype=torch.float32).view(4, 480),
                "igfold": torch.arange(4 * 512, dtype=torch.float32).view(4, 512),
            },
            "ag": {
                "esm2": torch.arange(4 * 480, dtype=torch.float32).view(4, 480)
            },
        },
        "edges": {
            "ab": _sparse((2, 2), ((0, 0), (1, 1))),
            "ag": _sparse((3, 3), ((0, 0), (1, 1), (2, 2))),
            "bipartite": _sparse((2, 3), ((0, 1), (1, 2))),
        },
        "stats": {
            "cdr": np.int64(2),
            "surf": np.int64(3),
            "epitope": np.int64(2),
            "epitope2surf_ratio": np.float64(2 / 3),
        },
    }


def _serialized(graph: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(graph, buffer)
    return buffer.getvalue()


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


def _pdb_payload(sequences: dict[str, str]) -> bytes:
    lines: list[str] = []
    for chain, sequence in sequences.items():
        tokens = " ".join(_AA1_TO_3[letter] for letter in sequence)
        lines.append(f"SEQRES {1:3d} {chain} {len(sequence):4d}  {tokens}")
    atom_serial = 1
    residue_number = 1
    for chain, sequence in sequences.items():
        for letter in sequence:
            for atom_index, atom in enumerate(("N", "CA", "C")):
                x = float(residue_number * 10 + atom_index)
                lines.append(
                    f"ATOM  {atom_serial:5d} {atom:>4s} {_AA1_TO_3[letter]:>3s} "
                    f"{chain}{residue_number:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
                    f"{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}"
                )
                atom_serial += 1
            residue_number += 1
    lines.extend(("TER", "END"))
    return ("\n".join(lines) + "\n").encode("ascii")


def _with_nonprotein_seqres(payload: bytes) -> bytes:
    lines = payload.decode("ascii").splitlines()
    lines.insert(0, "SEQRES   1 Z    1  PO4")
    lines.insert(0, "SEQRES   1 L    1  NAG")
    return ("\n".join(lines) + "\n").encode("ascii")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_restricted_graph_loader_normalizes_contact_first_axes() -> None:
    from interaction_first.asep_external import load_asep_graph_bytes

    result = load_asep_graph_bytes(
        _serialized(_graph()), expected_instance_id="1abc_0P"
    )
    assert result["antibody_sequence"] == "AD"
    assert result["antigen_sequence"] == "FGI"
    assert result["antibody_chain_ids"].tolist() == [0, 1]
    assert result["antibody_esm2_35m"].shape == (2, 480)
    assert result["antigen_esm2_35m"].shape == (3, 480)
    assert result["pair_positive_indices"].tolist() == [[0, 1], [1, 2]]
    assert result["antibody_labels"].tolist() == [1.0, 1.0]
    assert result["antigen_labels"].tolist() == [0.0, 1.0, 1.0]
    assert all(name.startswith("numpy") for name in result["restricted_pickle_globals"])


def test_label_free_structure_metadata_and_antigen_export_open_no_graph() -> None:
    from interaction_first.asep_external import (
        extract_asep_antigen_pdb_bytes,
        extract_asep_label_free_structure_metadata,
    )

    payload = _pdb_payload({"H": "AC", "L": "DE", "a": "FGHI"})
    metadata = extract_asep_label_free_structure_metadata(
        payload, expected_instance_id="1abc_0P"
    )
    assert metadata["pdb_id"] == "1abc"
    assert metadata["antigen_chain"] == "a"
    assert metadata["antigen_sequence"] == "FGHI"
    assert metadata["antibody_pdb_seqres_length"] == 4
    antigen = extract_asep_antigen_pdb_bytes(payload, antigen_chain="a")
    atom_lines = [line for line in antigen.decode().splitlines() if line.startswith("ATOM")]
    assert len(atom_lines) == 12
    assert {line[21] for line in atom_lines} == {"a"}


def test_label_free_structure_metadata_rejects_ambiguous_antigen_axis() -> None:
    from interaction_first.asep_external import (
        AsepExternalError,
        extract_asep_label_free_structure_metadata,
    )

    payload = _pdb_payload({"H": "AC", "L": "DE", "a": "FG", "b": "HI"})
    with pytest.raises(AsepExternalError, match="one protein antigen"):
        extract_asep_label_free_structure_metadata(
            payload, expected_instance_id="1abc_0P"
        )


class _BlockedPickleGlobal:
    pass


def test_restricted_graph_loader_rejects_unknown_pickle_global() -> None:
    from interaction_first.asep_external import AsepExternalError, load_asep_graph_bytes

    with pytest.raises(AsepExternalError, match="blocked pickle globals"):
        load_asep_graph_bytes(
            _serialized({"payload": _BlockedPickleGlobal()}),
            expected_instance_id="1abc_0P",
        )


def test_graph_validator_rejects_contact_marginal_disagreement() -> None:
    from interaction_first.asep_external import AsepExternalError, load_asep_graph_bytes

    graph = _graph()
    graph["mapping"]["ag"]["seqres2epitope"] = np.asarray(
        [1, 1, 0, 1], dtype=np.int64
    )
    with pytest.raises(AsepExternalError, match="reproduce epitope"):
        load_asep_graph_bytes(_serialized(graph), expected_instance_id="1abc_0P")


def test_structure_adapter_maps_n_ca_c_to_reduced_graph_axes() -> None:
    from interaction_first.asep_external import (
        adapt_asep_structure_bytes,
        load_asep_graph_bytes,
    )

    graph = load_asep_graph_bytes(
        _serialized(_graph()), expected_instance_id="1abc_0P"
    )
    result = adapt_asep_structure_bytes(
        _with_nonprotein_seqres(
            _pdb_payload({"H": "AC", "L": "DE", "a": "FGHI"})
        ),
        graph,
    )
    assert result["antibody_backbone"].shape == (2, 3, 3)
    assert result["antigen_backbone"].shape == (3, 3, 3)
    assert result["antibody_frame_valid_mask"].tolist() == [True, True]
    assert result["antigen_frame_valid_mask"].tolist() == [True, True, True]
    assert not result["antibody_missing_frame_mask"].any()
    assert result["alignment"]["H"]["optimal_alignment_count"] == 1


def test_structure_adapter_rejects_pdb_seqres_drift() -> None:
    from interaction_first.asep_external import (
        AsepExternalError,
        adapt_asep_structure_bytes,
        load_asep_graph_bytes,
    )

    graph = load_asep_graph_bytes(
        _serialized(_graph()), expected_instance_id="1abc_0P"
    )
    with pytest.raises(AsepExternalError, match="SEQRES differs"):
        adapt_asep_structure_bytes(
            _pdb_payload({"H": "AA", "L": "DE", "a": "FGHI"}), graph
        )


def _write_split_authority(root: Path) -> tuple[Path, str]:
    root.mkdir(parents=True)
    manifest = root / "epitope_group.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset_index", "instance_id", "role"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            (
                {"dataset_index": 0, "instance_id": "1abc_0P", "role": "train"},
                {"dataset_index": 1, "instance_id": "2def_0P", "role": "val"},
                {"dataset_index": 2, "instance_id": "3ghi_0P", "role": "test"},
            )
        )
    receipt = root / "ASEP_SPLIT_MANIFEST_RECEIPT.json"
    receipt.write_text(
        json.dumps(
            {
                "format_version": "framecontact_asep_split_manifest_authority_v1",
                "authority": {
                    "task_labels_read": False,
                    "model_predictions_or_scores_read": False,
                },
                "instance_count": 3,
                "manifests": {
                    "epitope_group": {
                        "path": manifest.name,
                        "sha256": _sha(manifest),
                        "role_counts": {"train": 1, "val": 1, "test": 1},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return receipt, _sha(receipt)


def test_development_audit_never_deserializes_test_member(tmp_path: Path) -> None:
    from interaction_first.asep_external import (
        AsepDevelopmentDataset,
        audit_asep_development_graphs,
        audit_asep_development_structures,
        materialize_asep_development_cache,
        materialize_asep_partner_metadata,
    )

    authority = tmp_path / "authority"
    payload_root = authority / "payload/asepv1.2.0"
    payload_root.mkdir(parents=True)
    graph_archive = payload_root / "asepv1_interim_graphs.tar.gz"
    with tarfile.open(graph_archive, "w:gz") as archive:
        directory = tarfile.TarInfo("asepv1_interim_graphs")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for identifier, payload in (
            ("1abc_0P", _serialized(_graph("1abc_0P"))),
            ("2def_0P", _serialized(_graph("2def_0P"))),
            ("3ghi_0P", b"deliberately-not-a-torch-serialization"),
        ):
            member = tarfile.TarInfo(f"asepv1_interim_graphs/{identifier}.pt")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    structure_archive = payload_root / "structures.tar.gz"
    with tarfile.open(structure_archive, "w:gz") as archive:
        directory = tarfile.TarInfo("structures")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for identifier, payload in (
            (
                "1abc_0P",
                _pdb_payload({"H": "AC", "L": "DE", "a": "FGHI"}),
            ),
            (
                "2def_0P",
                _pdb_payload({"H": "AC", "L": "DE", "a": "FGHI"}),
            ),
            ("3ghi_0P", b"deliberately-invalid-test-pdb"),
        ):
            member = tarfile.TarInfo(f"structures/{identifier}.pdb")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    inner = tmp_path / "inner.json"
    inner.write_text(
        json.dumps(
            {
                "format_version": "framecontact_asep_inner_archive_audit_v1",
                "authority": {"task_labels_read": False},
                "checks": {"split_payload_opened": False},
                "graph_archive": {"sha256": _sha(graph_archive)},
                "structure_archive": {"sha256": _sha(structure_archive)},
            }
        ),
        encoding="utf-8",
    )
    split_receipt, split_sha = _write_split_authority(tmp_path / "split")
    output = tmp_path / "audit"
    result = audit_asep_development_graphs(
        authority_root=authority,
        inner_audit_path=inner,
        inner_audit_sha256=_sha(inner),
        split_root=split_receipt.parent,
        split_receipt_sha256=split_sha,
        split_method="epitope_group",
        output_root=output,
    )
    assert result["role_counts"] == {"train": 1, "val": 1}
    assert result["checks"]["official_test_graph_tensors_opened"] is False
    with (output / "ASEP_DEVELOPMENT_GRAPH_MANIFEST.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["instance_id"] for row in rows] == ["1abc_0P", "2def_0P"]
    structure_output = tmp_path / "structure-audit"
    structure_result = audit_asep_development_structures(
        authority_root=authority,
        inner_audit_path=inner,
        inner_audit_sha256=_sha(inner),
        graph_audit_root=output,
        graph_audit_receipt_sha256=_sha(
            output / "ASEP_DEVELOPMENT_GRAPH_AUDIT.json"
        ),
        output_root=structure_output,
    )
    assert structure_result["development_instance_count"] == 2
    assert structure_result["totals"]["antibody_valid_frame_count"] == 4
    assert structure_result["totals"]["antigen_valid_frame_count"] == 6
    assert structure_result["checks"]["official_test_pdb_members_opened"] is False
    cache = tmp_path / "cache"
    cache_result = materialize_asep_development_cache(
        authority_root=authority,
        inner_audit_path=inner,
        inner_audit_sha256=_sha(inner),
        graph_audit_root=output,
        graph_audit_receipt_sha256=_sha(
            output / "ASEP_DEVELOPMENT_GRAPH_AUDIT.json"
        ),
        structure_audit_root=structure_output,
        structure_audit_receipt_sha256=_sha(
            structure_output / "ASEP_DEVELOPMENT_STRUCTURE_AUDIT.json"
        ),
        output_root=cache,
    )
    assert cache_result["role_counts"] == {"train": 1, "val": 1}
    assert cache_result["official_test_members_materialized"] is False
    dataset = AsepDevelopmentDataset(
        cache,
        manifest_sha256=_sha(cache / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"),
        split_role="val",
    )
    assert len(dataset) == 1
    item = dataset[0]
    assert item["instance_id"] == "2def_0P"
    assert item["pair_labels"].shape == (2, 3)
    assert int(item["pair_labels"].sum()) == 2
    from interaction_first.sabdab2_contacts import Sabdab2ContactBatchCollator

    batch = Sabdab2ContactBatchCollator()([item])
    assert batch["pair_labels"].shape == (1, 2, 3)
    metadata_root = tmp_path / "partner-metadata"
    metadata_result = materialize_asep_partner_metadata(
        cache_root=cache,
        cache_manifest_sha256=_sha(cache / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"),
        output_root=metadata_root,
    )
    assert metadata_result["task_label_or_contact_keys_accessed_by_metadata_builder"] is False
    with (metadata_root / "ASEP_PARTNER_METADATA.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        metadata_rows = list(csv.DictReader(handle))
    assert [row["role"] for row in metadata_rows] == ["train", "validation"]
    assert not any(
        fragment in column.casefold()
        for column in metadata_rows[0]
        for fragment in ("label", "contact", "epitope", "paratope", "prediction")
    )


def test_frozen_test_cache_opens_graph_only_after_hash_bound_receipt(
    tmp_path: Path,
) -> None:
    import interaction_first.asep_frozen_test as frozen_test_module
    from interaction_first.asep_frozen_test import (
        AsepFrozenTestDataset,
        OPENING_FORMAT,
        materialize_asep_frozen_test_cache,
    )

    authority = tmp_path / "authority"
    payload_root = authority / "payload/asepv1.2.0"
    payload_root.mkdir(parents=True)
    graph_archive = payload_root / "asepv1_interim_graphs.tar.gz"
    with tarfile.open(graph_archive, "w:gz") as archive:
        directory = tarfile.TarInfo("asepv1_interim_graphs")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for identifier in ("1abc_0P", "2def_0P", "3ghi_0P"):
            payload = _serialized(_graph(identifier))
            member = tarfile.TarInfo(f"asepv1_interim_graphs/{identifier}.pt")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    structure_archive = payload_root / "structures.tar.gz"
    with tarfile.open(structure_archive, "w:gz") as archive:
        directory = tarfile.TarInfo("structures")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        for identifier in ("1abc_0P", "2def_0P", "3ghi_0P"):
            payload = _pdb_payload({"H": "AC", "L": "DE", "a": "FGHI"})
            member = tarfile.TarInfo(f"structures/{identifier}.pdb")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    inner = tmp_path / "inner.json"
    inner.write_text(
        json.dumps(
            {
                "format_version": "framecontact_asep_inner_archive_audit_v1",
                "authority": {
                    "task_labels_read": False,
                    "model_predictions_or_scores_read": False,
                },
                "checks": {"split_payload_opened": False},
                "graph_archive": {"sha256": _sha(graph_archive)},
                "structure_archive": {"sha256": _sha(structure_archive)},
            }
        )
    )
    split_receipt, split_sha = _write_split_authority(tmp_path / "split")
    test_state_root = tmp_path / "test-state"
    test_state_root.mkdir()
    opening = tmp_path / "opening.json"
    opening.write_text(
        json.dumps(
            {
                "format_version": OPENING_FORMAT,
                "eligible_to_open": True,
                "test_feature_open_count_before": 1,
                "test_graph_or_label_open_count_before": 0,
                "test_labels_read_at_freeze": False,
                "model_predictions_or_scores_read_at_freeze": False,
                "test_adapter_source_sha256": _sha(
                    Path(frozen_test_module.__file__)
                ),
                "test_evaluator_source_sha256": frozen_test_module.asep_evaluation_source_hashes()[
                    "scripts/evaluate_asep_frozen_test.py"
                ],
                "evaluation_source_dependencies": (
                    frozen_test_module.asep_evaluation_source_hashes()
                ),
                "test_state_root": str(test_state_root),
                "graph_open_consumption_token_name": (
                    "ASEP_TEST_GRAPH_OPEN_CONSUMED.json"
                ),
                "score_consumption_token_name": "ASEP_TEST_SCORE_CONSUMED.json",
                "inner_audit_sha256": _sha(inner),
                "split_receipt_sha256": split_sha,
                "split_method": "epitope_group",
                "official_test_instance_count": 1,
                "graph_archive_sha256": _sha(graph_archive),
                "structure_archive_sha256": _sha(structure_archive),
                "selected_ensemble_result_sha256": "a" * 64,
            }
        )
    )
    output = tmp_path / "test-cache"
    result = materialize_asep_frozen_test_cache(
        authority_root=authority,
        inner_audit_path=inner,
        inner_audit_sha256=_sha(inner),
        split_root=split_receipt.parent,
        split_receipt_sha256=split_sha,
        split_method="epitope_group",
        opening_receipt_path=opening,
        opening_receipt_sha256=_sha(opening),
        output_root=output,
    )
    assert result["record_count"] == 1
    assert result["test_graph_or_label_open_count"] == 1
    dataset = AsepFrozenTestDataset(
        output,
        manifest_sha256=_sha(output / "ASEP_FROZEN_TEST_CACHE_MANIFEST.json"),
    )
    assert len(dataset) == 1
    assert dataset[0]["instance_id"] == "3ghi_0P"
    assert int(dataset[0]["pair_labels"].sum()) == 2
    graph_open_token = test_state_root / "ASEP_TEST_GRAPH_OPEN_CONSUMED.json"
    assert graph_open_token.is_file()
    assert json.loads(graph_open_token.read_text())["state"] == (
        "official_graph_label_open_started_fail_closed"
    )
    with pytest.raises(FileExistsError):
        materialize_asep_frozen_test_cache(
            authority_root=authority,
            inner_audit_path=inner,
            inner_audit_sha256=_sha(inner),
            split_root=split_receipt.parent,
            split_receipt_sha256=split_sha,
            split_method="epitope_group",
            opening_receipt_path=opening,
            opening_receipt_sha256=_sha(opening),
            output_root=tmp_path / "reopened-cache",
        )

    opening_payload = json.loads(opening.read_text())
    opening_payload["test_adapter_source_sha256"] = "0" * 64
    bad_opening = tmp_path / "bad-opening.json"
    bad_opening.write_text(json.dumps(opening_payload))
    with pytest.raises(ValueError, match="opening receipt is ineligible"):
        materialize_asep_frozen_test_cache(
            authority_root=authority,
            inner_audit_path=inner,
            inner_audit_sha256=_sha(inner),
            split_root=split_receipt.parent,
            split_receipt_sha256=split_sha,
            split_method="epitope_group",
            opening_receipt_path=bad_opening,
            opening_receipt_sha256=_sha(bad_opening),
            output_root=tmp_path / "bad-cache",
        )

    dependency_tamper = json.loads(opening.read_text())
    dependency_tamper["evaluation_source_dependencies"][
        "scripts/train_framecontact.py"
    ] = "0" * 64
    bad_dependency_opening = tmp_path / "bad-dependency-opening.json"
    bad_dependency_opening.write_text(json.dumps(dependency_tamper))
    with pytest.raises(ValueError, match="opening receipt is ineligible"):
        materialize_asep_frozen_test_cache(
            authority_root=authority,
            inner_audit_path=inner,
            inner_audit_sha256=_sha(inner),
            split_root=split_receipt.parent,
            split_receipt_sha256=split_sha,
            split_method="epitope_group",
            opening_receipt_path=bad_dependency_opening,
            opening_receipt_sha256=_sha(bad_dependency_opening),
            output_root=tmp_path / "bad-dependency-cache",
        )
