from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_materializer_reads_only_sequence_and_identity_columns() -> None:
    path = ROOT / "scripts" / "materialize_abpairfm_development_contact_sequences.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    assert tree
    assert '["instance_id", "ab_seq", "ag_seq"]' in text
    assert '"contact_labels_read": False' in text
    assert '"task_labels_read": False' in text
    assert '"primary_sealed_role_ids_retained": False' in text
    assert '"test_access": False' in text


def test_materializer_filters_both_parquet_authorities_to_selected_ids() -> None:
    text = (
        ROOT / "scripts" / "materialize_abpairfm_development_contact_sequences.py"
    ).read_text(encoding="utf-8")
    assert text.count('filters=[("instance_id", "in", selected)]') == 2
    assert "EXPECTED_TRANSFER_STATUS" in text
