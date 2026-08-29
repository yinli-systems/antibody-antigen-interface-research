"""Tests for fail-closed authentic residue-pair supervision."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("numpy", "pyarrow", "torch")),
    reason="contact adapter dependencies are unavailable",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_structure_manifest(
    path: Path,
    identifiers=("train_case", "validation_case"),
    *,
    cohort: bool = False,
):
    from interaction_first.sabdab2_structure import (
        STRUCTURE_CACHE_FORMAT,
        STRUCTURE_COHORT_CACHE_FORMAT,
        canonical_json_bytes,
    )

    records = []
    for index, identifier in enumerate(identifiers):
        record = {
                "instance_id": identifier,
                "antibody_length": 2,
                "antigen_length": 3,
            }
        if not cohort:
            record["split_role"] = "train" if index == 0 else "validation"
        records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        canonical_json_bytes(
            {
                "format_version": (
                    STRUCTURE_COHORT_CACHE_FORMAT if cohort else STRUCTURE_CACHE_FORMAT
                ),
                "records": records,
            }
        )
    )
    return records


def _write_pairs(path: Path, *, incomplete: bool = False) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("instance_id", pa.string()),
            ("ab_res_idx", pa.int64()),
            ("ag_res_idx", pa.int64()),
            ("label", pa.int64()),
        ]
    )
    with pq.ParquetWriter(path, schema) as writer:
        for identifier, positives in (
            ("train_case", {1, 3}),
            ("validation_case", {0, 5}),
            # Invalid labels prove unselected label columns are never read.
            ("native_test_case", {0, 1, 2, 3, 4, 5}),
        ):
            count = 5 if incomplete and identifier == "train_case" else 6
            writer.write_table(
                pa.table(
                    {
                        "instance_id": [identifier] * count,
                        "ab_res_idx": [flat // 3 for flat in range(count)],
                        "ag_res_idx": [flat % 3 for flat in range(count)],
                        "label": [
                            7 if identifier == "native_test_case" else int(flat in positives)
                            for flat in range(count)
                        ],
                    },
                    schema=schema,
                )
            )


def _base_item(identifier: str, positives: set[int]):
    import torch

    ca_ab = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    ca_ag = torch.tensor(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]]
    )

    def backbone(ca):
        return torch.stack(
            (ca + torch.tensor([0.0, 1.0, 0.0]), ca, ca + torch.tensor([1.0, 0.0, 0.0])),
            dim=-2,
        )

    pair = torch.zeros(6, dtype=torch.bool)
    pair[list(positives)] = True
    pair = pair.view(2, 3)
    return {
        "instance_id": identifier,
        "antibody_features": torch.randn(2, 4),
        "antigen_features": torch.randn(3, 5),
        "antibody_backbone": backbone(ca_ab),
        "antigen_backbone": backbone(ca_ag),
        "antibody_chain_ids": torch.zeros(2, dtype=torch.long),
        "antigen_chain_ids": torch.zeros(3, dtype=torch.long),
        "antibody_frame_valid_mask": torch.ones(2, dtype=torch.bool),
        "antigen_frame_valid_mask": torch.ones(3, dtype=torch.bool),
        "antibody_missing_frame_mask": torch.zeros(2, dtype=torch.bool),
        "antigen_missing_frame_mask": torch.zeros(3, dtype=torch.bool),
        "antibody_labels": pair.any(dim=1).float(),
        "antigen_labels": pair.any(dim=0).float(),
    }


class _FakeStructureDataset:
    def __init__(self, cache_root: Path, records):
        self.cache_root = cache_root
        self.records = tuple(records)
        self.items = (
            _base_item("train_case", {1, 3}),
            _base_item("validation_case", {0, 5}),
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_contact_cache_authenticates_complete_grids_and_batches(tmp_path: Path) -> None:
    import torch

    from interaction_first.sabdab2_contacts import (
        Sabdab2ContactBatchCollator,
        Sabdab2ContactDataset,
        materialize_contact_cache,
    )

    structure_root = tmp_path / "structure"
    records = _write_structure_manifest(
        structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    )
    pairs_path = tmp_path / "pairs.parquet"
    _write_pairs(pairs_path)
    output_root = tmp_path / "contacts"
    receipt = materialize_contact_cache(
        pairs_path=pairs_path,
        pairs_sha256=_sha256(pairs_path),
        structure_cache_root=structure_root,
        structure_manifest_sha256=_sha256(
            structure_root / "STRUCTURE_CACHE_MANIFEST.json"
        ),
        output_root=output_root,
    )
    assert receipt["record_count"] == 2
    assert receipt["frozen_test_labels_read"] is False
    manifest = json.loads(
        (output_root / "CONTACT_CACHE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["authority"]["unselected_pair_label_columns_read"] is False
    assert manifest["counts"]["skipped_unselected_row_groups"] == 1
    assert manifest["counts"]["candidate_pairs"] == 12
    assert manifest["counts"]["positive_pairs"] == 4

    dataset = Sabdab2ContactDataset(
        _FakeStructureDataset(structure_root, records), output_root
    )
    train = dataset[0]
    validation = dataset[1]
    assert torch.equal(
        train["pair_labels"],
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    batch = Sabdab2ContactBatchCollator()([train, validation])
    assert batch["pair_labels"].shape == (2, 2, 3)
    assert batch["pair_candidate_mask"].all()


def test_boolean_pair_labels_match_the_published_arrow_schema() -> None:
    import numpy as np

    from interaction_first.sabdab2_contacts import _safe_numpy_binary

    values = _safe_numpy_binary(np.asarray([True, False, True]), "label")
    np.testing.assert_array_equal(values, np.asarray([1, 0, 1], dtype=np.int64))


def test_cohort_contact_cache_derives_case_safe_residue_marginals(
    tmp_path: Path,
) -> None:
    from interaction_first.sabdab2_contacts import (
        CONTACT_COHORT_CACHE_FORMAT,
        Sabdab2ContactDataset,
        materialize_contact_cache,
    )

    structure_root = tmp_path / "structure"
    records = _write_structure_manifest(
        structure_root / "STRUCTURE_CACHE_MANIFEST.json", cohort=True
    )
    pairs_path = tmp_path / "pairs.parquet"
    _write_pairs(pairs_path)
    output_root = tmp_path / "contacts"
    receipt = materialize_contact_cache(
        pairs_path=pairs_path,
        pairs_sha256=_sha256(pairs_path),
        structure_cache_root=structure_root,
        structure_manifest_sha256=_sha256(
            structure_root / "STRUCTURE_CACHE_MANIFEST.json"
        ),
        output_root=output_root,
    )
    manifest = json.loads(
        (output_root / "CONTACT_CACHE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["format_version"] == CONTACT_COHORT_CACHE_FORMAT
    assert receipt["split_membership_read"] is False
    base = _FakeStructureDataset(structure_root, records)
    for item in base.items:
        item.pop("antibody_labels")
        item.pop("antigen_labels")
    item = Sabdab2ContactDataset(base, output_root)[0]
    assert item["antibody_labels"].tolist() == [1.0, 1.0]
    assert item["antigen_labels"].tolist() == [1.0, 1.0, 0.0]


def test_incomplete_or_duplicate_pair_grid_fails_without_partial_cache(
    tmp_path: Path,
) -> None:
    from interaction_first.sabdab2_contacts import materialize_contact_cache
    from interaction_first.sabdab2_structure import StructureCacheError

    structure_root = tmp_path / "structure"
    _write_structure_manifest(structure_root / "STRUCTURE_CACHE_MANIFEST.json")
    pairs_path = tmp_path / "pairs.parquet"
    _write_pairs(pairs_path, incomplete=True)
    output_root = tmp_path / "contacts"
    with pytest.raises(StructureCacheError, match="complete residue Cartesian grid"):
        materialize_contact_cache(
            pairs_path=pairs_path,
            pairs_sha256=_sha256(pairs_path),
            structure_cache_root=structure_root,
            structure_manifest_sha256=_sha256(
                structure_root / "STRUCTURE_CACHE_MANIFEST.json"
            ),
            output_root=output_root,
        )
    assert not output_root.exists()
