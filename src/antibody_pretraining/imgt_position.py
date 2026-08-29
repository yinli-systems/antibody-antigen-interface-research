"""Fixed-width IMGT-position representation for exhaustive antibody search.

The 200-position vocabulary is the canonical alignment published by KA-Search
(Olsen et al., Scientific Reports 2023).  OAS already records ANARCI numberings
for every source row, so production-corpus audit jobs can reuse those numberings
instead of renumbering hundreds of millions of admitted sequences.

This module deliberately does not hide rare insertions.  A numbering containing
an IMGT position outside the registered 200-position vocabulary is reported as
unrepresentable and must remain covered by the independent gap-aware track.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable


# Source-pinned against the official oxpig/kasearch implementation.  These
# anchors make later 226M-record export receipts auditable without trusting an
# unversioned copy of the 200-position vocabulary.
KASEARCH_PAPER_DOI = "10.1038/s41598-023-38108-7"
KASEARCH_SOURCE_COMMIT = "952445c813ea183b353d2f3aa9015df5a8587461"
KASEARCH_CANONICAL_ALIGNMENT_SHA256 = (
    "edc0d734a666e65f2e845bd599d6631ee8c63e30648b992ac949a05f0f6b99ec"
)
KASEARCH_IDENTITY_CALCULATIONS_SHA256 = (
    "09e8771e6935a615a3fcc09cd12575c62f4563616a2c55877f98990e5f4b9aad"
)
KASEARCH_PREPARE_OASDB_SHA256 = (
    "7c37e016fe485696c9a01f1a3ebd13243bac61f0b3d2f2ad90c2e10bba204874"
)

CANONICAL_NUMBERING = (
    "1 ", "2 ", "3 ", "3A", "4 ", "5 ", "6 ", "7 ", "8 ", "9 ",
    "10 ", "11 ", "12 ", "13 ", "14 ", "15 ", "16 ", "17 ", "18 ",
    "19 ", "20 ", "21 ", "22 ", "23 ", "24 ", "25 ", "26 ", "27 ",
    "28 ", "29 ", "30 ", "31 ", "32 ", "32A", "32B", "33C", "33B",
    "33A", "33 ", "34 ", "35 ", "36 ", "37 ", "38 ", "39 ", "40 ",
    "40A", "41 ", "42 ", "43 ", "44 ", "44A", "45 ", "45A", "46 ",
    "46A", "47 ", "47A", "48 ", "48A", "48B", "49 ", "49A", "50 ",
    "51 ", "51A", "52 ", "53 ", "54 ", "55 ", "56 ", "57 ", "58 ",
    "59 ", "60 ", "60A", "60B", "60C", "60D", "61E", "61D", "61C",
    "61B", "61A", "61 ", "62 ", "63 ", "64 ", "65 ", "66 ", "67 ",
    "67A", "67B", "68 ", "68A", "68B", "69 ", "69A", "69B", "70 ",
    "71 ", "71A", "71B", "72 ", "73 ", "73A", "73B", "74 ", "75 ",
    "76 ", "77 ", "78 ", "79 ", "80 ", "80A", "81 ", "81A", "81B",
    "81C", "82 ", "82A", "83 ", "83A", "83B", "84 ", "85 ", "85A",
    "85B", "85C", "85D", "86 ", "86A", "87 ", "88 ", "89 ", "90 ",
    "91 ", "92 ", "93 ", "94 ", "95 ", "96 ", "96A", "97 ", "98 ",
    "99 ", "100 ", "101 ", "102 ", "103 ", "104 ", "105 ", "106 ",
    "107 ", "108 ", "109 ", "110 ", "111 ", "111A", "111B", "111C",
    "111D", "111E", "111F", "111G", "111H", "111I", "111J", "111K",
    "111L", "112L", "112K", "112J", "112I", "112H", "112G", "112F",
    "112E", "112D", "112C", "112B", "112A", "112 ", "113 ", "114 ",
    "115 ", "116 ", "117 ", "118 ", "119 ", "119A", "120 ", "121 ",
    "122 ", "123 ", "124 ", "125 ", "126 ", "127 ", "128 ",
)
CANONICAL_INDEX = {position: index for index, position in enumerate(CANONICAL_NUMBERING)}
CANONICAL_LENGTH = 200
GAP = 0
MISSING_END = ord("|")
_OAS_NUMBERING_ENTRY = re.compile(
    r"['\"](?P<number>[1-9][0-9]{0,2})(?P<insertion>[A-Z ]?)['\"]\s*"
    r":\s*['\"](?P<residue>[A-Z-])['\"]"
)


class IMGTPositionError(ValueError):
    """Report malformed or out-of-vocabulary IMGT-position evidence."""


@dataclass(frozen=True)
class CanonicalAlignment:
    """One KA-Search-compatible fixed-width antibody representation."""

    encoded: bytes
    residue_count: int

    def validate(self) -> None:
        if len(self.encoded) != CANONICAL_LENGTH:
            raise IMGTPositionError("Canonical alignment width differs")
        if not 0 < self.residue_count <= CANONICAL_LENGTH:
            raise IMGTPositionError("Canonical alignment residue count is invalid")
        observed = sum(value not in {GAP, MISSING_END} for value in self.encoded)
        if observed != self.residue_count:
            raise IMGTPositionError("Canonical alignment residue count differs")


@dataclass(frozen=True)
class PositionIdentity:
    """Identity and reciprocal coverage on fixed IMGT positions."""

    identity: float
    query_coverage: float
    target_coverage: float
    identical_positions: int
    overlapping_positions: int
    query_positions: int
    target_positions: int
    union_positions: int

    def qualifies(
        self,
        *,
        identity: float = 0.90,
        query_coverage: float = 0.90,
        target_coverage: float = 0.90,
    ) -> bool:
        thresholds = (identity, query_coverage, target_coverage)
        if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in thresholds):
            raise IMGTPositionError("Near-identity threshold is invalid")
        tolerance = 1e-12
        return (
            self.identity + tolerance >= identity
            and self.query_coverage + tolerance >= query_coverage
            and self.target_coverage + tolerance >= target_coverage
        )


def _normalize_position(number: str, insertion: str) -> str:
    value = f"{int(number)}{insertion or ' '}"
    if value not in CANONICAL_INDEX:
        raise IMGTPositionError(f"IMGT position {value!r} is outside the canonical 200")
    return value


def parse_oas_anarci_numbering(value: str) -> CanonicalAlignment:
    """Parse one OAS ``ANARCI_numbering`` cell without executing its text."""
    if not isinstance(value, str) or not value.strip():
        raise IMGTPositionError("OAS ANARCI numbering is empty")
    entries = list(_OAS_NUMBERING_ENTRY.finditer(value))
    if not entries:
        raise IMGTPositionError("OAS ANARCI numbering has no residue entries")
    encoded = bytearray(CANONICAL_LENGTH)
    seen: set[str] = set()
    first: int | None = None
    last: int | None = None
    residue_count = 0
    for match in entries:
        position = _normalize_position(match.group("number"), match.group("insertion"))
        if position in seen:
            raise IMGTPositionError("OAS ANARCI numbering repeats a position")
        seen.add(position)
        index = CANONICAL_INDEX[position]
        residue = match.group("residue")
        if residue == "-":
            continue
        encoded[index] = ord(residue)
        residue_count += 1
        first = index if first is None else min(first, index)
        last = index if last is None else max(last, index)
    if first is None or last is None:
        raise IMGTPositionError("OAS ANARCI numbering contains no residues")
    encoded[:first] = bytes([MISSING_END]) * first
    suffix = CANONICAL_LENGTH - last - 1
    encoded[last + 1 :] = bytes([MISSING_END]) * suffix
    result = CanonicalAlignment(encoded=bytes(encoded), residue_count=residue_count)
    result.validate()
    return result


def encode_numbered_residues(
    residues: Iterable[tuple[tuple[int, str], str]],
) -> CanonicalAlignment:
    """Encode ANARCI/ANARCII-style ``((number, insertion), residue)`` rows."""
    parts = []
    for (number, insertion), residue in residues:
        if not isinstance(number, int) or not isinstance(insertion, str):
            raise IMGTPositionError("Numbered residue position is malformed")
        if not isinstance(residue, str) or len(residue) != 1 or not re.fullmatch(
            r"[A-Z-]", residue
        ):
            raise IMGTPositionError("Numbered residue value is malformed")
        key = f"{number}{insertion or ' '}"
        parts.append(f"'{key}': '{residue}'")
    return parse_oas_anarci_numbering("{" + ", ".join(parts) + "}")


def position_identity(query: CanonicalAlignment, target: CanonicalAlignment) -> PositionIdentity:
    """Compute KA-Search whole-domain identity plus reciprocal position coverage."""
    query.validate()
    target.validate()
    identical = overlapping = query_count = target_count = union = 0
    for query_value, target_value in zip(query.encoded, target.encoded):
        query_exists = query_value not in {GAP, MISSING_END}
        target_exists = target_value not in {GAP, MISSING_END}
        query_count += query_exists
        target_count += target_exists
        overlapping += query_exists and target_exists
        union += query_exists or target_exists
        identical += query_exists and target_exists and query_value == target_value
    if not query_count or not target_count or not union:
        raise IMGTPositionError("Position identity has an empty residue domain")
    return PositionIdentity(
        identity=identical / union,
        query_coverage=overlapping / query_count,
        target_coverage=overlapping / target_count,
        identical_positions=identical,
        overlapping_positions=overlapping,
        query_positions=query_count,
        target_positions=target_count,
        union_positions=union,
    )


if len(CANONICAL_NUMBERING) != CANONICAL_LENGTH or len(CANONICAL_INDEX) != CANONICAL_LENGTH:
    raise RuntimeError("KA-Search canonical IMGT-position vocabulary is not exactly 200")


__all__ = [
    "CANONICAL_INDEX",
    "CANONICAL_LENGTH",
    "CANONICAL_NUMBERING",
    "CanonicalAlignment",
    "IMGTPositionError",
    "KASEARCH_CANONICAL_ALIGNMENT_SHA256",
    "KASEARCH_IDENTITY_CALCULATIONS_SHA256",
    "KASEARCH_PAPER_DOI",
    "KASEARCH_PREPARE_OASDB_SHA256",
    "KASEARCH_SOURCE_COMMIT",
    "MISSING_END",
    "PositionIdentity",
    "encode_numbered_residues",
    "parse_oas_anarci_numbering",
    "position_identity",
]
