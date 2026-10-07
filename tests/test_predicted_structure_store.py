from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path

import numpy as np
import pytest

from antibody_pretraining.predicted_structure_store import (
    EndpointStructure,
    FEATURE_DIMENSION,
    PredictedStructureStore,
    THREEDI_INDEX,
    prepare_pair_structure,
)


def _sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _write_assets(tmp_path: Path, *, usable: bool = True) -> dict[str, Path]:
    predicted = "ACD"
    holo = "AC"
    member = "af__one.pdb"
    alignment = tmp_path / "alignment.tsv.gz"
    with gzip.open(alignment, "wt", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "member",
                "holo_sequence_sha256",
                "predicted_sequence_sha256",
                "holo_length",
                "ambiguous_optimum",
                "usable_structure_modality",
                "holo_to_predicted_zero_based",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "member": member,
                "holo_sequence_sha256": _sha(holo),
                "predicted_sequence_sha256": _sha(predicted),
                "holo_length": len(holo),
                "ambiguous_optimum": "false",
                "usable_structure_modality": "true" if usable else "false",
                "holo_to_predicted_zero_based": "0,2",
            }
        )
    three_di = tmp_path / "3di.tsv.gz"
    with gzip.open(three_di, "wt", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["member", "sequence_sha256", "three_di_tokens"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "member": member,
                "sequence_sha256": _sha(predicted),
                "three_di_tokens": "DYQ",
            }
        )
    feature_index = tmp_path / "features.tsv"
    with feature_index.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "member",
                "sequence_sha256",
                "residue_offset",
                "residue_count",
                "feature_dimension",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "member": member,
                "sequence_sha256": _sha(predicted),
                "residue_offset": 0,
                "residue_count": len(predicted),
                "feature_dimension": FEATURE_DIMENSION,
            }
        )
    features = np.zeros((3, FEATURE_DIMENSION), dtype="<f4")
    features[:, -1] = [0.9, 0.8, 0.6]
    feature_binary = tmp_path / "features.f32"
    feature_binary.write_bytes(features.tobytes())
    return {
        "alignment_manifest": alignment,
        "three_di_manifest": three_di,
        "feature_index": feature_index,
        "feature_binary": feature_binary,
    }


def test_endpoint_maps_only_confident_sequence_aligned_residues(tmp_path: Path) -> None:
    store = PredictedStructureStore(**_write_assets(tmp_path))
    endpoint = store.endpoint(member="af__one.pdb", holo_sequence="AC")
    assert endpoint.token_indices.tolist() == [THREEDI_INDEX["D"], 0]
    assert endpoint.observed_mask.tolist() == [True, False]
    assert endpoint.features[0, -1] == pytest.approx(0.9)
    assert not endpoint.features[1].any()


def test_unusable_alignment_retains_sequence_with_missing_structure(tmp_path: Path) -> None:
    store = PredictedStructureStore(**_write_assets(tmp_path, usable=False))
    endpoint = store.endpoint(member="af__one.pdb", holo_sequence="AC")
    assert not endpoint.token_indices.any()
    assert not endpoint.features.any()
    assert not endpoint.observed_mask.any()


def test_store_rejects_feature_binary_size_mismatch(tmp_path: Path) -> None:
    paths = _write_assets(tmp_path)
    paths["feature_binary"].write_bytes(b"\0\0\0\0")
    with pytest.raises(ValueError, match="binary size differs"):
        PredictedStructureStore(**paths)


def test_pair_corruption_is_deterministic_and_special_token_aligned() -> None:
    features = np.ones((10, FEATURE_DIMENSION), dtype=np.float32)
    endpoint = EndpointStructure(
        token_indices=np.full(10, 3, dtype=np.int64),
        features=features,
        observed_mask=np.ones(10, dtype=np.bool_),
    )
    first = prepare_pair_structure(
        endpoint,
        endpoint,
        seed=91,
        training=True,
        chain_dropout_probability=0.0,
        system_dropout_probability=0.0,
    )
    second = prepare_pair_structure(
        endpoint,
        endpoint,
        seed=91,
        training=True,
        chain_dropout_probability=0.0,
        system_dropout_probability=0.0,
    )
    assert np.array_equal(first.input_token_indices, second.input_token_indices)
    assert len(first.input_token_indices) == 24
    assert first.feature_target_mask.sum() == 4
    assert (first.token_labels != -100).sum() == 4
    assert first.input_token_indices[[0, 11, 12, 23]].tolist() == [0, 0, 0, 0]
    assert not first.input_feature_mask[first.feature_target_mask].any()


def test_pair_corruption_drops_structure_without_targets() -> None:
    endpoint = EndpointStructure(
        token_indices=np.full(3, 3, dtype=np.int64),
        features=np.ones((3, FEATURE_DIMENSION), dtype=np.float32),
        observed_mask=np.ones(3, dtype=np.bool_),
    )
    prepared = prepare_pair_structure(
        endpoint,
        endpoint,
        seed=7,
        training=True,
        chain_dropout_probability=1.0,
        system_dropout_probability=0.0,
    )
    assert prepared.dropped_chains == (True, True)
    assert set(prepared.input_token_indices) == {0, 2}
    assert not prepared.input_features.any()
    assert not prepared.input_feature_mask.any()
    assert not prepared.feature_target_mask.any()
    assert (prepared.token_labels == -100).all()
