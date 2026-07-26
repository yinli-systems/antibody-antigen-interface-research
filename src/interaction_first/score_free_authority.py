"""Derive score-free authority data from authenticated train-only raw bytes.

The extractor accepts an exact train allowlist, an exact four-column sequence
manifest, and four lifted direct-edge ledgers. It derives every permitted
sequence and topology summary rather than accepting caller-provided counts.
Outcome labels and model-derived fields have no input channel in this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .score_free_split_statistics import (
    AntibodyScoreFreeInputs,
    AntigenScoreFreeInputs,
    QuantileScheme,
    ScoreFreePair,
    antibody_profile,
    antigen_profile,
    build_quantile_schemes,
    implementation_source_sha256 as statistics_implementation_source_sha256,
)


AUTHORITY_FORMAT_VERSION = "gate0_physical_train_only_allowlist_v1"
EDGE_LEDGER_FORMAT_VERSION = "score_free_direct_entity_edges_v1"
BUNDLE_FORMAT_VERSION = "score_free_authority_v2"
MANIFEST_COLUMNS = ("INSTANCE", "vh_seq", "vl_seq", "ag_seq")
REQUIRED_VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class ScoreFreeAuthorityError(ValueError):
    """Raised when authenticated score-free inputs fail closed."""


def implementation_source_sha256() -> str:
    """Hash the exact authority-extractor source bytes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class AuthorityCounts:
    """Expected inventory counts bound before raw derivation."""

    source_rows: int
    exact_pairs: int
    antibody_entities: int
    antigen_entities: int

    def __post_init__(self) -> None:
        for name in (
            "source_rows",
            "exact_pairs",
            "antibody_entities",
            "antigen_entities",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ScoreFreeAuthorityError(f"{name} must be a positive integer")


LOCKED_AUTHORITY_COUNTS = AuthorityCounts(
    source_rows=3229,
    exact_pairs=1830,
    antibody_entities=1383,
    antigen_entities=1172,
)


@dataclass(frozen=True)
class SourceRecord:
    """One normalized source row from the exact train-only projection."""

    instance_id: str
    vh: str
    vl: str
    antigen: str


@dataclass(frozen=True)
class AntibodyEntity:
    """One framed antibody entity."""

    entity_id: str
    vh: str
    vl: str


@dataclass(frozen=True)
class AntigenEntity:
    """One framed antigen entity."""

    entity_id: str
    sequence: str


@dataclass(frozen=True)
class AuthorityPair:
    """One unique observed pair and all source rows that realize it."""

    pair_id: str
    antibody_id: str
    antigen_id: str
    source_instance_ids: tuple[str, ...]

    def score_free_pair(self) -> ScoreFreePair:
        """Return the outcome-free incidence record used by the oracle."""
        return ScoreFreePair(
            pair_id=self.pair_id,
            antibody_id=self.antibody_id,
            antigen_id=self.antigen_id,
        )


@dataclass(frozen=True)
class DirectEdgeLedger:
    """One exact lifted direct-edge relation."""

    view: str
    edges: tuple[tuple[str, str], ...]
    source_sha256: str


@dataclass(frozen=True)
class AuthorityQuantileScheme:
    """One universe-fixed quantile scheme with an explicit entity axis."""

    axis: str
    scheme: QuantileScheme


@dataclass(frozen=True)
class ScoreFreeAuthorityBundle:
    """Canonical score-free authority bundle for solver and audit consumers."""

    input_sha256: tuple[tuple[str, str], ...]
    authority_membership_sha256: str
    antibodies: tuple[AntibodyScoreFreeInputs, ...]
    antigens: tuple[AntigenScoreFreeInputs, ...]
    pairs: tuple[AuthorityPair, ...]
    quantile_schemes: tuple[AuthorityQuantileScheme, ...]

    def canonical_payload(self) -> dict[str, object]:
        """Return a byte-canonical payload with a self-verifying digest."""
        payload: dict[str, object] = {
            "format_version": BUNDLE_FORMAT_VERSION,
            "input_sha256": dict(self.input_sha256),
            "authority_membership_sha256": self.authority_membership_sha256,
            "implementation_source_sha256": {
                "score_free_authority": implementation_source_sha256(),
                "score_free_split_statistics": (
                    statistics_implementation_source_sha256()
                ),
            },
            "counts": {
                "source_rows": sum(
                    len(pair.source_instance_ids) for pair in self.pairs
                ),
                "exact_pairs": len(self.pairs),
                "antibody_entities": len(self.antibodies),
                "antigen_entities": len(self.antigens),
            },
            "antibodies": [asdict(value) for value in self.antibodies],
            "antigens": [asdict(value) for value in self.antigens],
            "pairs": [
                {
                    "pair_id": value.pair_id,
                    "antibody_id": value.antibody_id,
                    "antigen_id": value.antigen_id,
                    "source_instance_ids": list(value.source_instance_ids),
                }
                for value in self.pairs
            ],
            "quantile_schemes": [
                {
                    "axis": value.axis,
                    "feature_name": value.scheme.feature_name,
                    "cuts_decimal": [str(cut) for cut in value.scheme.cuts],
                    "optional_missing_bin": value.scheme.optional_missing_bin,
                    "categorical_binary": value.scheme.categorical_binary,
                    "informative": value.scheme.informative,
                }
                for value in self.quantile_schemes
            ],
            "forbidden_outcome_or_model_fields_accepted": False,
        }
        payload["scientific_payload_sha256"] = canonical_payload_sha256(payload)
        return payload

    @property
    def scientific_payload_sha256(self) -> str:
        """Return the digest of the payload before its digest field is added."""
        return str(self.canonical_payload()["scientific_payload_sha256"])


def canonical_json_bytes(payload: object) -> bytes:
    """Encode one canonical ASCII JSON document."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def canonical_payload_sha256(payload: object) -> str:
    """Hash one canonical payload."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def membership_sha256(values: Iterable[str]) -> str:
    """Hash a sorted unique ASCII membership with length framing."""
    unique = sorted(set(values))
    digest = hashlib.sha256()
    for value in unique:
        if not isinstance(value, str) or not value or not value.isascii():
            raise ScoreFreeAuthorityError("membership values must be non-empty ASCII")
        encoded = value.encode("ascii")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    return digest.hexdigest()


def _strict_json_object(payload_bytes: bytes, label: str) -> dict[str, Any]:
    if not payload_bytes or payload_bytes.startswith(b"\xef\xbb\xbf"):
        raise ScoreFreeAuthorityError(f"{label} is empty or contains a BOM")
    try:
        text = payload_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise ScoreFreeAuthorityError(f"{label} must be ASCII JSON") from error

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ScoreFreeAuthorityError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ScoreFreeAuthorityError(
                    f"{label} contains non-finite JSON constant {token}"
                )
            ),
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ScoreFreeAuthorityError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ScoreFreeAuthorityError(f"{label} must be a JSON object")
    return value


def _validate_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScoreFreeAuthorityError(f"{label} must be a non-empty string")
    if value != value.strip() or not value.isascii():
        raise ScoreFreeAuthorityError(
            f"{label} must be exact ASCII without edge whitespace"
        )
    if any(character in value for character in (",", '"', "\r", "\n", "\0")):
        raise ScoreFreeAuthorityError(f"{label} contains a forbidden character")
    return value


def _normalize_sequence(value: str, label: str, *, allow_empty: bool = False) -> str:
    sequence = "".join(character for character in value if not character.isspace())
    sequence = sequence.upper()
    try:
        sequence.encode("ascii")
    except UnicodeEncodeError as error:
        raise ScoreFreeAuthorityError(f"{label} is not ASCII") from error
    if not sequence and allow_empty:
        return ""
    if not sequence:
        raise ScoreFreeAuthorityError(f"{label} is empty")
    unexpected = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if unexpected:
        raise ScoreFreeAuthorityError(
            f"{label} contains invalid residue symbols: {unexpected}"
        )
    return sequence


def _physical_csv_rows(payload_bytes: bytes) -> list[list[str]]:
    label = "physical train-only canonical manifest"
    if not payload_bytes or payload_bytes.startswith(b"\xef\xbb\xbf"):
        raise ScoreFreeAuthorityError(f"{label} is empty or contains a BOM")
    if b"\0" in payload_bytes or b"\r" in payload_bytes:
        raise ScoreFreeAuthorityError(f"{label} contains forbidden NUL or CR bytes")
    if not payload_bytes.endswith(b"\n"):
        raise ScoreFreeAuthorityError(f"{label} must end with one LF")
    physical_lines = payload_bytes.split(b"\n")[:-1]
    if not physical_lines or any(not line for line in physical_lines):
        raise ScoreFreeAuthorityError(f"{label} contains an empty physical line")
    rows: list[list[str]] = []
    for line_number, raw_line in enumerate(physical_lines, start=1):
        try:
            line = raw_line.decode("utf-8")
            parsed = list(csv.reader([line], strict=True))
        except (UnicodeDecodeError, csv.Error) as error:
            raise ScoreFreeAuthorityError(
                f"{label} line {line_number} is not one UTF-8 CSV record"
            ) from error
        if len(parsed) != 1:
            raise ScoreFreeAuthorityError(
                f"{label} line {line_number} is not one complete record"
            )
        rows.append(parsed[0])
    return rows


def _load_authority(payload_bytes: bytes) -> tuple[set[str], str]:
    payload = _strict_json_object(payload_bytes, "train authority allowlist")
    if set(payload) != {"format_version", "authorized_split", "instance_ids"}:
        raise ScoreFreeAuthorityError(
            "train authority schema contains forbidden or missing fields"
        )
    if payload["format_version"] != AUTHORITY_FORMAT_VERSION:
        raise ScoreFreeAuthorityError("unsupported train authority format_version")
    if payload["authorized_split"] != "train":
        raise ScoreFreeAuthorityError("train authority must authorize only train")
    raw_ids = payload["instance_ids"]
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ScoreFreeAuthorityError("train authority instance_ids must be non-empty")
    identifiers = [
        _validate_identifier(value, "train authority identifier") for value in raw_ids
    ]
    if len(identifiers) != len(set(identifiers)):
        raise ScoreFreeAuthorityError("train authority contains duplicate identifiers")
    membership = set(identifiers)
    return membership, membership_sha256(membership)


def _load_records(payload_bytes: bytes, authority_ids: set[str]) -> dict[str, SourceRecord]:
    rows = _physical_csv_rows(payload_bytes)
    if tuple(rows[0]) != MANIFEST_COLUMNS:
        raise ScoreFreeAuthorityError(
            "manifest must contain exactly INSTANCE,vh_seq,vl_seq,ag_seq"
        )
    records: dict[str, SourceRecord] = {}
    for line_number, fields in enumerate(rows[1:], start=2):
        if len(fields) != len(MANIFEST_COLUMNS):
            raise ScoreFreeAuthorityError(f"malformed manifest row {line_number}")
        instance_id = _validate_identifier(fields[0], f"row {line_number} identifier")
        if instance_id not in authority_ids:
            raise ScoreFreeAuthorityError("manifest row is outside the train authority")
        if instance_id in records:
            raise ScoreFreeAuthorityError("manifest contains a duplicate source row")
        records[instance_id] = SourceRecord(
            instance_id=instance_id,
            vh=_normalize_sequence(fields[1], f"{instance_id}:VH"),
            vl=_normalize_sequence(
                fields[2], f"{instance_id}:VL", allow_empty=True
            ),
            antigen=_normalize_sequence(fields[3], f"{instance_id}:antigen"),
        )
    if set(records) != authority_ids:
        raise ScoreFreeAuthorityError(
            "manifest source membership does not exactly equal train authority"
        )
    return records


def _frame(sequence: str) -> bytes:
    encoded = sequence.encode("ascii")
    return str(len(encoded)).encode("ascii") + b":" + encoded


def _antibody_id(vh: str, vl: str) -> str:
    return hashlib.sha256(b"antibody-v1\0" + _frame(vh) + _frame(vl)).hexdigest()


def _antigen_id(sequence: str) -> str:
    return hashlib.sha256(b"antigen-v1\0" + _frame(sequence)).hexdigest()


def _pair_id(antibody_id: str, antigen_id: str) -> str:
    return hashlib.sha256(
        b"pair-v1\0"
        + antibody_id.encode("ascii")
        + b"\0"
        + antigen_id.encode("ascii")
    ).hexdigest()


def _canonicalize(
    records: Mapping[str, SourceRecord],
) -> tuple[
    dict[str, AntibodyEntity],
    dict[str, AntigenEntity],
    tuple[AuthorityPair, ...],
]:
    antibodies: dict[str, AntibodyEntity] = {}
    antigens: dict[str, AntigenEntity] = {}
    pair_sources: dict[tuple[str, str], list[str]] = defaultdict(list)
    for instance_id in sorted(records):
        record = records[instance_id]
        antibody_id = _antibody_id(record.vh, record.vl)
        antigen_id = _antigen_id(record.antigen)
        antibody = AntibodyEntity(antibody_id, record.vh, record.vl)
        antigen = AntigenEntity(antigen_id, record.antigen)
        if antibodies.setdefault(antibody_id, antibody) != antibody:
            raise ScoreFreeAuthorityError("antibody digest collision")
        if antigens.setdefault(antigen_id, antigen) != antigen:
            raise ScoreFreeAuthorityError("antigen digest collision")
        pair_sources[(antibody_id, antigen_id)].append(instance_id)
    pairs = tuple(
        AuthorityPair(
            pair_id=_pair_id(antibody_id, antigen_id),
            antibody_id=antibody_id,
            antigen_id=antigen_id,
            source_instance_ids=tuple(sorted(source_ids)),
        )
        for (antibody_id, antigen_id), source_ids in sorted(pair_sources.items())
    )
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ScoreFreeAuthorityError("pair digest collision")
    return antibodies, antigens, pairs


def _load_edge_ledger(
    payload_bytes: bytes,
    *,
    expected_view: str,
    allowed_entities: set[str],
    antibodies: Mapping[str, AntibodyEntity],
) -> DirectEdgeLedger:
    payload = _strict_json_object(payload_bytes, f"{expected_view} edge ledger")
    if set(payload) != {"format_version", "view", "edges"}:
        raise ScoreFreeAuthorityError(
            f"{expected_view} edge ledger schema contains forbidden fields"
        )
    if payload["format_version"] != EDGE_LEDGER_FORMAT_VERSION:
        raise ScoreFreeAuthorityError("unsupported edge ledger format_version")
    if payload["view"] != expected_view:
        raise ScoreFreeAuthorityError("edge ledger view does not match its binding")
    raw_edges = payload["edges"]
    if not isinstance(raw_edges, list):
        raise ScoreFreeAuthorityError("edge ledger edges must be a list")
    edges: list[tuple[str, str]] = []
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise ScoreFreeAuthorityError("each edge must contain exactly two IDs")
        left = _validate_identifier(raw_edge[0], "edge left entity")
        right = _validate_identifier(raw_edge[1], "edge right entity")
        if left >= right:
            raise ScoreFreeAuthorityError("edge endpoints must be canonical and distinct")
        if left not in allowed_entities or right not in allowed_entities:
            raise ScoreFreeAuthorityError("edge references an unknown entity")
        if expected_view == "vl" and (
            not antibodies[left].vl or not antibodies[right].vl
        ):
            raise ScoreFreeAuthorityError("VL edge references a missing VL chain")
        edges.append((left, right))
    if len(edges) != len(set(edges)) or tuple(edges) != tuple(sorted(edges)):
        raise ScoreFreeAuthorityError("edge ledger must be sorted and duplicate-free")
    return DirectEdgeLedger(
        view=expected_view,
        edges=tuple(edges),
        source_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def _adjacency(
    entities: Iterable[str], edges: Iterable[tuple[str, str]]
) -> dict[str, set[str]]:
    graph = {entity: set() for entity in entities}
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)
    return graph


def _components(graph: Mapping[str, set[str]]) -> tuple[dict[str, str], dict[str, int]]:
    component_by_entity: dict[str, str] = {}
    size_by_component: dict[str, int] = {}
    for start in sorted(graph):
        if start in component_by_entity:
            continue
        stack = [start]
        members: set[str] = set()
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            stack.extend(sorted(graph[current] - members, reverse=True))
        component_id = min(members)
        size_by_component[component_id] = len(members)
        for member in members:
            component_by_entity[member] = component_id
    return component_by_entity, size_by_component


def _check_counts(
    records: Mapping[str, SourceRecord],
    antibodies: Mapping[str, AntibodyEntity],
    antigens: Mapping[str, AntigenEntity],
    pairs: Sequence[AuthorityPair],
    expected: AuthorityCounts,
) -> None:
    observed = AuthorityCounts(
        source_rows=len(records),
        exact_pairs=len(pairs),
        antibody_entities=len(antibodies),
        antigen_entities=len(antigens),
    )
    if observed != expected:
        raise ScoreFreeAuthorityError(
            f"authority inventory differs: expected={expected}, observed={observed}"
        )


def build_score_free_authority(
    *,
    authority_bytes: bytes,
    manifest_bytes: bytes,
    edge_ledger_bytes: Mapping[str, bytes],
    expected_counts: AuthorityCounts = LOCKED_AUTHORITY_COUNTS,
) -> ScoreFreeAuthorityBundle:
    """Derive the complete score-free authority bundle from raw bytes."""
    if set(edge_ledger_bytes) != set(REQUIRED_VIEWS):
        raise ScoreFreeAuthorityError("exactly four named edge ledgers are required")
    authority_ids, authority_membership_hash = _load_authority(authority_bytes)
    records = _load_records(manifest_bytes, authority_ids)
    antibodies, antigens, pairs = _canonicalize(records)
    _check_counts(records, antibodies, antigens, pairs, expected_counts)

    ledgers: dict[str, DirectEdgeLedger] = {}
    for view in REQUIRED_VIEWS:
        allowed = set(antigens) if view == "antigen" else set(antibodies)
        ledgers[view] = _load_edge_ledger(
            edge_ledger_bytes[view],
            expected_view=view,
            allowed_entities=allowed,
            antibodies=antibodies,
        )

    antibody_view_graphs = {
        view: _adjacency(antibodies, ledgers[view].edges)
        for view in ("vh", "vl", "concatenated_antibody")
    }
    union_edges = set().union(
        ledgers["vh"].edges,
        ledgers["vl"].edges,
        ledgers["concatenated_antibody"].edges,
    )
    antibody_union_graph = _adjacency(antibodies, union_edges)
    antigen_graph = _adjacency(antigens, ledgers["antigen"].edges)
    antibody_components, antibody_component_sizes = _components(
        antibody_union_graph
    )
    antigen_components, antigen_component_sizes = _components(antigen_graph)

    antibody_pair_degree: dict[str, int] = defaultdict(int)
    antigen_pair_degree: dict[str, int] = defaultdict(int)
    for pair in pairs:
        antibody_pair_degree[pair.antibody_id] += 1
        antigen_pair_degree[pair.antigen_id] += 1

    antibody_inputs = tuple(
        AntibodyScoreFreeInputs(
            entity_id=entity_id,
            component_id=antibody_components[entity_id],
            vh_length=len(entity.vh),
            vl_length=None if not entity.vl else len(entity.vl),
            concatenated_antibody_length=len(entity.vh) + len(entity.vl),
            vh_direct_degree=len(antibody_view_graphs["vh"][entity_id]),
            vl_direct_degree=(
                None
                if not entity.vl
                else len(antibody_view_graphs["vl"][entity_id])
            ),
            concatenated_direct_degree=len(
                antibody_view_graphs["concatenated_antibody"][entity_id]
            ),
            union_direct_degree=len(antibody_union_graph[entity_id]),
            pair_incidence_degree=antibody_pair_degree[entity_id],
            component_size=antibody_component_sizes[
                antibody_components[entity_id]
            ],
        )
        for entity_id, entity in sorted(antibodies.items())
    )
    antigen_inputs = tuple(
        AntigenScoreFreeInputs(
            entity_id=entity_id,
            component_id=antigen_components[entity_id],
            antigen_length=len(entity.sequence),
            antigen_direct_degree=len(antigen_graph[entity_id]),
            pair_incidence_degree=antigen_pair_degree[entity_id],
            component_size=antigen_component_sizes[antigen_components[entity_id]],
        )
        for entity_id, entity in sorted(antigens.items())
    )
    antibody_profiles = tuple(
        antibody_profile(value) for value in antibody_inputs
    )
    antigen_profiles = tuple(antigen_profile(value) for value in antigen_inputs)
    schemes = tuple(
        AuthorityQuantileScheme("antibody", scheme)
        for scheme in build_quantile_schemes(antibody_profiles)
    ) + tuple(
        AuthorityQuantileScheme("antigen", scheme)
        for scheme in build_quantile_schemes(antigen_profiles)
    )
    input_hashes = {
        "authority_allowlist": hashlib.sha256(authority_bytes).hexdigest(),
        "canonical_manifest": hashlib.sha256(manifest_bytes).hexdigest(),
        **{
            f"edge_ledger_{view}": ledgers[view].source_sha256
            for view in REQUIRED_VIEWS
        },
    }
    return ScoreFreeAuthorityBundle(
        input_sha256=tuple(sorted(input_hashes.items())),
        authority_membership_sha256=authority_membership_hash,
        antibodies=antibody_inputs,
        antigens=antigen_inputs,
        pairs=pairs,
        quantile_schemes=tuple(
            sorted(
                schemes,
                key=lambda value: (
                    value.axis,
                    value.scheme.feature_name,
                    value.scheme.cuts,
                ),
            )
        ),
    )
