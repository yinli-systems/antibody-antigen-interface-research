"""Adversarial tests for the cap-audited MMseqs2 gap-aware helpers."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from antibody_pretraining.mmseqs_gap_aware import (
    GLOBAL_TARGET_FORMAT,
    MMseqsGapAwareError,
    audit_alignment_rows,
    audit_prefilter_database,
    export_authenticated_target_task,
    parse_target_identifier,
    target_identifier,
)
from antibody_pretraining.near_identity_export import publish_export_receipt
from antibody_pretraining.oas_imgt_export import (
    IMGT_TARGET_RECORD,
    UNUSUAL_TARGET_RECORD,
    classify_oas_target,
    pack_imgt_target,
    pack_unusual_target,
)
from antibody_pretraining.production_materialization import ProductionMapRecord


def _production_record(sequence: str, *, ordinal: int, split: str) -> ProductionMapRecord:
    return ProductionMapRecord(
        authority_ordinal=ordinal,
        chain_type="heavy",
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        split=split,
        status_flags=(),
        stratum="human_heavy",
        study_id=1,
    )


def _numbering(sequence: str) -> str:
    return "{" + ", ".join(
        f"'{index} ': '{residue}'"
        for index, residue in enumerate(sequence, start=1)
    ) + "}"


def _file_inventory(path: Path, *, width: int, records: int) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "record_bytes": width,
        "records": records,
    }


def test_target_identifier_is_canonical_and_shard_bound() -> None:
    value = target_identifier(
        task_id=7,
        split="validation",
        chain_type="lambda",
        authority_ordinal=123,
    )
    assert value == "oasgap_0007_v_l_000000000000007b"
    assert parse_target_identifier(value) == {
        "authority_ordinal": 123,
        "chain_type": "lambda",
        "split": "validation",
        "task_id": 7,
    }
    with pytest.raises(MMseqsGapAwareError, match="not canonical"):
        parse_target_identifier("oasgap_7_v_l_7b")


def test_authenticated_target_export_preserves_both_binary_streams(tmp_path: Path) -> None:
    root = tmp_path / "targets"
    task_root = root / "imgt-map/mapper-0000"
    task_root.mkdir(parents=True)
    sequence = "A" * 100
    representable = classify_oas_target(
        _numbering(sequence), _production_record(sequence, ordinal=17, split="train")
    ).representable
    unusual = classify_oas_target(
        "not a numbering",
        _production_record("C" * 100, ordinal=19, split="validation"),
    ).unusual
    assert representable is not None and unusual is not None
    representable_path = task_root / "representable.bin"
    unusual_path = task_root / "unusual.bin"
    representable_path.write_bytes(pack_imgt_target(representable))
    unusual_path.write_bytes(pack_unusual_target(unusual))
    task = {
        "counts": {
            "accepted_records": 2,
            "chain:heavy": 2,
            "representable_records": 1,
            "split:train": 1,
            "split:validation": 1,
            "unusual_records": 1,
        },
        "output_inventory": {
            "representable": _file_inventory(
                representable_path, width=IMGT_TARGET_RECORD.size, records=1
            ),
            "unusual": _file_inventory(
                unusual_path, width=UNUSUAL_TARGET_RECORD.size, records=1
            ),
        },
        "task_id": 0,
    }
    tasks = [task] + [{"task_id": value} for value in range(1, 32)]
    publish_export_receipt(
        root / "global_receipt.json",
        {
            "claim_boundary": {
                "accepted_oas_domain_partition_complete": True,
                "global_task_set_complete": True,
                "near_identity_search_complete": False,
            },
            "format_version": GLOBAL_TARGET_FORMAT,
            "task_count": 32,
            "task_receipts": tasks,
        },
    )
    fasta = tmp_path / "task.fasta"
    result = export_authenticated_target_task(
        target_output_root=root, task_id=0, output_fasta=fasta
    )
    assert result["counts"] == {
        "chain:heavy": 2,
        "records": 2,
        "split:train": 1,
        "split:validation": 1,
    }
    assert fasta.read_text(encoding="ascii") == (
        ">oasgap_0000_t_h_0000000000000011\n" + "A" * 100 + "\n"
        ">oasgap_0000_v_h_0000000000000013\n" + "C" * 100 + "\n"
    )


def _write_prefilter(prefix: Path, records: list[bytes]) -> None:
    offset = 0
    rows = []
    payload = bytearray()
    for key, body in enumerate(records):
        record = body + b"\0"
        rows.append(f"{key}\t{offset}\t{len(record)}\n")
        payload.extend(record)
        offset += len(record)
    prefix.write_bytes(bytes(payload))
    Path(str(prefix) + ".index").write_text("".join(rows), encoding="ascii")


def test_prefilter_audit_rejects_cap_saturation(tmp_path: Path) -> None:
    prefix = tmp_path / "prefilter"
    _write_prefilter(prefix, [b"1\t2\t3\n", b"1\t2\t3\n4\t5\t6\n"])
    result = audit_prefilter_database(
        prefix=prefix, expected_query_count=2, maximum_sequences=3
    )
    assert result["hit_count_total"] == 3
    assert result["hit_count_maximum"] == 2
    with pytest.raises(MMseqsGapAwareError, match="truncation is possible"):
        audit_prefilter_database(
            prefix=prefix, expected_query_count=2, maximum_sequences=2
        )


def test_alignment_audit_replays_exact_metrics_and_threshold() -> None:
    target = target_identifier(
        task_id=3, split="train", chain_type="heavy", authority_ordinal=7
    )
    row = "\t".join(
        [
            "asep_" + "a" * 64,
            target,
            "90.000",
            "1.000",
            "1.000",
            "10",
            "9",
            "1",
            "10",
            "10",
            "1",
            "10",
            "10",
            "ACDEFGHIKL",
            "ACDEFGHIKA",
        ]
    ) + "\n"
    output = io.StringIO()
    result = audit_alignment_rows(
        [row],
        query_lengths={"asep_" + "a" * 64: 10},
        expected_task_id=3,
        hits_output=output,
    )
    assert result["alignment_rows"] == 1
    assert result["split_hit_counts"] == {"train": 1, "validation": 0}
    assert "0.900000000000\t1.000000000000\t1.000000000000" in output.getvalue()


def test_alignment_audit_rejects_rounded_metric_disagreement() -> None:
    target = target_identifier(
        task_id=0, split="train", chain_type="heavy", authority_ordinal=7
    )
    row = "\t".join(
        [
            "asep_" + "b" * 64,
            target,
            "0.950",
            "1.000",
            "1.000",
            "10",
            "9",
            "1",
            "10",
            "10",
            "1",
            "10",
            "10",
            "ACDEFGHIKL",
            "ACDEFGHIKA",
        ]
    ) + "\n"
    with pytest.raises(MMseqsGapAwareError, match="metric replay differs"):
        audit_alignment_rows(
            [row],
            query_lengths={"asep_" + "b" * 64: 10},
            expected_task_id=0,
        )
