from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path

from scripts import audit_pinder_unique_sequence_pairs as audit


def test_pair_inventory_separates_views_from_exact_pairs(tmp_path: Path) -> None:
    sequence_a = hashlib.sha256(b"AAAA").digest()
    sequence_b = hashlib.sha256(b"BBBB").digest()
    sequences = {"a-R": sequence_a, "b-L": sequence_b, "c-R": sequence_a}
    systems = tmp_path / "systems.tsv.gz"
    fields = (
        "split",
        "id",
        "cluster_id",
        "holo_R_pdb",
        "holo_L_pdb",
        "uniprot_R",
        "uniprot_L",
    )
    with gzip.open(systems, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(dict(zip(fields, ("train", "one", "c1", "a-R.pdb", "b-L.pdb", "U1", "U2"))))
        writer.writerow(dict(zip(fields, ("train", "two", "c2", "c-R.pdb", "b-L.pdb", "U1", "U2"))))
    assert audit.pair_inventory(systems, sequences) == {
        "exact_self_pair_systems": 0,
        "post_decontamination_systems": 2,
        "repeated_structural_views": 1,
        "unique_exact_endpoint_sequences": 2,
        "unique_exact_unordered_sequence_pairs": 1,
        "unique_interface_clusters": 2,
        "unique_unordered_uniprot_pairs": 1,
    }
