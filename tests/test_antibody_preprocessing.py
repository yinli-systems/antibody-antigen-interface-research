import hashlib
from pathlib import Path

import numpy as np

from antibody_pretraining.dense_shard import (
    DenseShardRecord,
    load_dense_shard,
    write_dense_shard,
)
from antibody_pretraining.preprocessing import (
    CompiledUnpairedRowEvaluator,
    evaluate_unpaired_row,
    load_filter_policy,
    selected_by_plan_fraction,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "protocols/oas_sequence_filter_v1.json"
SEQUENCE = "ACDEFGHIKLMNPQRSTVWY" * 4


def _row(**updates: str) -> dict[str, str]:
    row = {
        "ANARCI_status": "||||Shorter than IMGT defined: fw4|",
        "locus": "IGH",
        "productive": "T",
        "sequence_alignment_aa": SEQUENCE,
        "stop_codon": "F",
        "vj_in_frame": "T",
    }
    row.update(updates)
    return row


def test_row_filter_accepts_audited_truncation_warning_and_rejects_fatal() -> None:
    policy, _ = load_filter_policy(POLICY_PATH)
    accepted = evaluate_unpaired_row(_row(), authority_chain="Heavy", policy=policy)
    fatal = evaluate_unpaired_row(
        _row(ANARCI_status="Missing Conserved Cysteine"),
        authority_chain="Heavy",
        policy=policy,
    )
    wrong_locus = evaluate_unpaired_row(
        _row(locus="IGK"),
        authority_chain="Heavy",
        policy=policy,
    )

    assert accepted.accepted is not None
    assert accepted.accepted.status_flags == ("shorter",)
    assert fatal.rejection_reason == "fatal_anarci_status"
    assert wrong_locus.rejection_reason == "authority_locus_disagreement"


def test_compiled_filter_matches_public_filter_on_adversarial_rows() -> None:
    """Preserve exact decisions while moving validation outside the hot loop."""
    policy, _ = load_filter_policy(POLICY_PATH)
    compiled = CompiledUnpairedRowEvaluator.from_policy(
        policy,
        authority_chain="Heavy",
    )
    rows = (
        _row(),
        _row(productive="F"),
        _row(stop_codon="T"),
        _row(vj_in_frame="invalid"),
        _row(sequence_alignment_aa=""),
        _row(sequence_alignment_aa="A" * 79),
        _row(sequence_alignment_aa="A" * 80 + "Z"),
        _row(locus="IGK"),
        _row(ANARCI_status="ANARCI FAILED"),
    )
    for row in rows:
        assert compiled.evaluate(row) == evaluate_unpaired_row(
            row,
            authority_chain="Heavy",
            policy=policy,
        )


def test_plan_fraction_selection_is_exact_and_replays() -> None:
    namespace = hashlib.sha256(b"namespace").hexdigest()
    first = selected_by_plan_fraction(
        SEQUENCE,
        namespace_sha256=namespace,
        numerator=1,
        denominator=3,
    )
    replay = selected_by_plan_fraction(
        SEQUENCE,
        namespace_sha256=namespace,
        numerator=1,
        denominator=3,
    )

    assert first is replay
    assert selected_by_plan_fraction(
        SEQUENCE,
        namespace_sha256=namespace,
        numerator=1,
        denominator=1,
    )


def test_dense_shard_round_trip_and_hash_validation(tmp_path: Path) -> None:
    output = tmp_path / "shard"
    records = [
        DenseShardRecord("heavy", 11, SEQUENCE),
        DenseShardRecord("kappa", 13, SEQUENCE[::-1]),
    ]
    manifest = write_dense_shard(
        output,
        records,
        maximum_sequence_tokens=96,
        provenance={"pilot": True},
    )
    loaded = load_dense_shard(
        output,
        expected_manifest_sha256=manifest["manifest_sha256"],
    )

    assert len(loaded) == 2
    assert loaded.tokens.shape == (2, 96)
    assert loaded.tokens.dtype == np.uint8
    assert loaded.lengths.tolist() == [82, 82]
    assert loaded.sample_keys.tolist() == [11, 13]
