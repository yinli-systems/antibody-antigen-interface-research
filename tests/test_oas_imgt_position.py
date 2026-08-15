"""Test the KA-Search-compatible fixed IMGT-position representation."""

from __future__ import annotations

import pytest

from antibody_pretraining.imgt_position import (
    CANONICAL_LENGTH,
    IMGTPositionError,
    KASEARCH_CANONICAL_ALIGNMENT_SHA256,
    KASEARCH_SOURCE_COMMIT,
    MISSING_END,
    encode_numbered_residues,
    parse_oas_anarci_numbering,
    position_identity,
)


def _numbered(sequence: str, *, start: int = 1):
    return [((index, ""), residue) for index, residue in enumerate(sequence, start)]


def test_kasearch_source_is_immutable_and_vocabulary_has_200_positions() -> None:
    assert KASEARCH_SOURCE_COMMIT == "952445c813ea183b353d2f3aa9015df5a8587461"
    assert KASEARCH_CANONICAL_ALIGNMENT_SHA256 == (
        "edc0d734a666e65f2e845bd599d6631ee8c63e30648b992ac949a05f0f6b99ec"
    )
    assert CANONICAL_LENGTH == 200


def test_oas_numbering_parser_is_fixed_width_and_does_not_execute_text() -> None:
    alignment = parse_oas_anarci_numbering(
        "{'1 ': 'Q', '2 ': 'V', '3 ': 'Q', '3A': 'L', '4 ': '-'}"
    )
    assert len(alignment.encoded) == CANONICAL_LENGTH
    assert alignment.residue_count == 4
    assert alignment.encoded[:4] == b"QVQL"
    assert set(alignment.encoded[4:]) == {MISSING_END}
    with pytest.raises(IMGTPositionError, match="no residue entries"):
        parse_oas_anarci_numbering("__import__('os').system('false')")


def test_rare_insertion_and_duplicate_position_fail_closed() -> None:
    with pytest.raises(IMGTPositionError, match="outside the canonical 200"):
        parse_oas_anarci_numbering("{'111Z': 'A'}")
    with pytest.raises(IMGTPositionError, match="repeats a position"):
        parse_oas_anarci_numbering("{'1 ': 'A', \"1 \" : \"C\"}")


def test_exact_and_single_substitution_controls() -> None:
    query = encode_numbered_residues(_numbered("A" * 100))
    exact = position_identity(query, query)
    assert exact.identity == 1.0
    assert exact.query_coverage == 1.0
    assert exact.target_coverage == 1.0
    assert exact.qualifies()

    substitution_rows = _numbered("A" * 100)
    substitution_rows[49] = ((50, ""), "C")
    substitution = position_identity(query, encode_numbered_residues(substitution_rows))
    assert substitution.identity == pytest.approx(0.99)
    assert substitution.qualifies()


def test_internal_and_terminal_indel_controls_use_union_identity() -> None:
    query = encode_numbered_residues(_numbered("A" * 100))
    internal_rows = [row for row in _numbered("A" * 100) if row[0][0] != 50]
    internal = position_identity(query, encode_numbered_residues(internal_rows))
    assert internal.identity == pytest.approx(0.99)
    assert internal.query_coverage == pytest.approx(0.99)
    assert internal.target_coverage == 1.0
    assert internal.qualifies()

    terminal_rows = _numbered("A" * 95)
    terminal = position_identity(query, encode_numbered_residues(terminal_rows))
    assert terminal.identity == pytest.approx(0.95)
    assert terminal.query_coverage == pytest.approx(0.95)
    assert terminal.target_coverage == 1.0
    assert terminal.qualifies()


def test_reciprocal_coverage_blocks_short_fragment_false_positive() -> None:
    query = encode_numbered_residues(_numbered("A" * 100))
    short = encode_numbered_residues(_numbered("A" * 80))
    metrics = position_identity(query, short)
    assert metrics.identity == pytest.approx(0.80)
    assert metrics.query_coverage == pytest.approx(0.80)
    assert not metrics.qualifies()
