from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/deduplicate_ppiref50k_against_pinder.py"


def module():
    spec = importlib.util.spec_from_file_location("ppiref_pinder_dedup", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_pinder_inventory_and_order_invariant_reasons(tmp_path: Path) -> None:
    value = module()
    systems = tmp_path / "systems.tsv.gz"
    fields = ("split", "pdb_id", "holo_R_pdb", "holo_L_pdb")
    with gzip.open(systems, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(dict(zip(fields, ("train", "1ABC", "a-R.pdb", "b-L.pdb"))))
    digest_a = hashlib.sha256(b"AAAA").digest()
    digest_b = hashlib.sha256(b"BBBB").digest()
    pairs, pdb_ids, count = value.pinder_priority_inventory(
        systems, {"a-R": digest_a, "b-L": digest_b}
    )
    assert count == 1
    assert pdb_ids == {"1abc"}
    assert value.deduplication_reasons(
        {"pdb_id": "9xyz", "chain1_full_sequence": "BBBB", "chain2_full_sequence": "AAAA"},
        pairs,
        pdb_ids,
    ) == {"exact_sequence_pair"}
    assert value.deduplication_reasons(
        {"pdb_id": "1ABC", "chain1_full_sequence": "CCCC", "chain2_full_sequence": "DDDD"},
        pairs,
        pdb_ids,
    ) == {"same_pdb_entry"}


def test_missing_ppiref_deduplication_fields_fail_closed() -> None:
    value = module()
    try:
        value.deduplication_reasons({"pdb_id": "1abc"}, set(), set())
    except ValueError as error:
        assert "lacks" in str(error)
    else:
        raise AssertionError("Malformed PPIRef record was accepted")
