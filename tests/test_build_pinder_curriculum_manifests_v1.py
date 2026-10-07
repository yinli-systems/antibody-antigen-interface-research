import csv
import gzip
from pathlib import Path

import pytest

from remote_tools.abpairfm_distributed_contact_pretraining_v1.build_pinder_curriculum_manifests_v1 import (
    cluster_balanced_order,
    enrich_and_weight,
    read_train_rows,
)


def test_cluster_balanced_order_is_deterministic_and_front_loads_clusters() -> None:
    rows = [
        {"id": "a1", "cluster_id": "a"},
        {"id": "a2", "cluster_id": "a"},
        {"id": "b1", "cluster_id": "b"},
        {"id": "b2", "cluster_id": "b"},
        {"id": "c1", "cluster_id": "c"},
    ]
    first = cluster_balanced_order([row.copy() for row in rows], seed=17)
    second = cluster_balanced_order([row.copy() for row in reversed(rows)], seed=17)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len({row["cluster_id"] for row in first[:3]}) == 3


def test_enrichment_uses_exact_sequences_and_normalizes_weights() -> None:
    rows = [
        {
            "id": "x1",
            "pdb_id": "1abc",
            "cluster_id": "c1",
            "holo_R_pdb": "r1.pdb",
            "holo_L_pdb": "l1.pdb",
        },
        {
            "id": "x2",
            "pdb_id": "1abc",
            "cluster_id": "c1",
            "holo_R_pdb": "r2.pdb",
            "holo_L_pdb": "l2.pdb",
        },
        {
            "id": "x3",
            "pdb_id": "2abc",
            "cluster_id": "c2",
            "holo_R_pdb": "r3.pdb",
            "holo_L_pdb": "l3.pdb",
        },
    ]
    sequences = {
        "r1": "AAAA",
        "l1": "CCCC",
        "r2": "CCCC",
        "l2": "AAAA",
        "r3": "DDDD",
        "l3": "EEEE",
    }
    enriched = enrich_and_weight(rows, sequences)
    assert enriched[0]["exact_sequence_pair_sha256"] == enriched[1][
        "exact_sequence_pair_sha256"
    ]
    assert enriched[0]["exact_sequence_pair_multiplicity"] == "2"
    assert sum(float(row["system_weight"]) for row in enriched) / 3 == pytest.approx(1)


def test_read_train_rows_rejects_non_train_input(tmp_path: Path) -> None:
    path = tmp_path / "rows.tsv.gz"
    fields = [
        "split",
        "id",
        "pdb_id",
        "cluster_id",
        "holo_R_pdb",
        "holo_L_pdb",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "split": "test",
                "id": "x",
                "pdb_id": "1abc",
                "cluster_id": "c",
                "holo_R_pdb": "r",
                "holo_L_pdb": "l",
            }
        )
    with pytest.raises(ValueError, match="train-only"):
        read_train_rows(path)
