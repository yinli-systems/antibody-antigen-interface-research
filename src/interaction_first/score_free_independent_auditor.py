"""Independent raw-byte reconstruction for V6-A score-free authority.

This module intentionally does not import the primary authority extractor, the
primary statistics oracle, or any solver formulation. It repeats parsing,
canonicalization, graph construction, integer covariate derivation, and
quantile construction from authenticated byte strings.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUTHORITY_FORMAT = "gate0_physical_train_only_allowlist_v1"
EDGE_FORMAT = "score_free_direct_entity_edges_v1"
BUNDLE_FORMAT = "score_free_authority_v2"
AUDIT_FORMAT = "score_free_independent_authority_audit_v1"
MANIFEST_HEADER = ("INSTANCE", "vh_seq", "vl_seq", "ag_seq")
VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
DECIMAL_PRECISION = 50
EXPECTED_COUNTS = {
    "source_rows": 3229,
    "exact_pairs": 1830,
    "antibody_entities": 1383,
    "antigen_entities": 1172,
}
FOLD_SHAPE_THRESHOLDS = {
    "outer": {
        "minimum_validation_pairs": 120,
        "minimum_validation_endpoints_per_axis": 60,
        "minimum_validation_components_per_axis": 25,
        "validation_pair_lower_fraction": [7, 100],
        "validation_pair_upper_fraction": [9, 100],
        "minimum_training_pair_fraction": [3, 5],
        "minimum_training_endpoint_fraction": [3, 5],
        "maximum_quarantine_pair_fraction": [33, 100],
    },
    "inner": {
        "minimum_validation_pairs": 80,
        "minimum_validation_endpoints_per_axis": 40,
        "minimum_validation_components_per_axis": 20,
        "validation_pair_lower_fraction": [7, 100],
        "validation_pair_upper_fraction": [9, 100],
        "minimum_training_pair_fraction": [11, 20],
        "minimum_training_endpoint_fraction": [11, 20],
        "maximum_quarantine_pair_fraction": [19, 50],
    },
}
DISTRIBUTION_THRESHOLDS = {
    "outer": {
        "maximum_smd_decimal": "0.35",
        "maximum_tv_exact": [1, 5],
        "maximum_ks_exact": [1, 5],
    },
    "inner": {
        "maximum_smd_decimal": "0.40",
        "maximum_tv_exact": [1, 4],
        "maximum_ks_exact": [1, 4],
    },
    "pooled_outer": {
        "maximum_smd_decimal": "0.25",
        "maximum_tv_exact": [3, 20],
        "maximum_ks_exact": [1, 5],
    },
}
CROSS_FOLD_THRESHOLDS = {
    "outer": {
        "maximum_validation_pair_relative_range": [3, 20],
        "maximum_other_relative_range": [1, 5],
    },
    "inner": {
        "maximum_validation_pair_relative_range": [1, 5],
        "maximum_other_relative_range": [1, 4],
    },
}
FEATURES = {
    "antibody": (
        ("vh_length", True, True, True, False, False),
        ("vl_length", True, True, True, True, False),
        ("concatenated_antibody_length", True, True, True, False, False),
        ("log1p_vh_direct_degree", True, True, True, False, False),
        ("log1p_vl_direct_degree", True, True, True, True, False),
        ("log1p_concatenated_direct_degree", True, True, True, False, False),
        ("log1p_union_direct_degree", True, True, True, False, False),
        ("log1p_pair_incidence_degree", True, True, True, False, False),
        ("component_size", True, True, True, False, False),
        ("missing_vl", False, False, True, False, True),
    ),
    "antigen": (
        ("antigen_length", True, True, True, False, False),
        ("log1p_antigen_direct_degree", True, True, True, False, False),
        ("log1p_pair_incidence_degree", True, True, True, False, False),
        ("component_size", True, True, True, False, False),
    ),
}


class IndependentScoreFreeAuditError(ValueError):
    """Raised when independent raw reconstruction fails closed."""


def canonical_json_bytes(payload: object) -> bytes:
    """Encode canonical ASCII JSON with one terminal LF."""
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


def _compact_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def implementation_source_sha256() -> str:
    """Hash the independent auditor source bytes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _strict_object(payload_bytes: bytes, label: str) -> dict[str, Any]:
    if not payload_bytes or payload_bytes.startswith(b"\xef\xbb\xbf"):
        raise IndependentScoreFreeAuditError(f"{label} is empty or contains a BOM")
    try:
        text = payload_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise IndependentScoreFreeAuditError(f"{label} must be ASCII JSON") from error

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise IndependentScoreFreeAuditError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        result = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                IndependentScoreFreeAuditError(
                    f"{label} contains non-finite constant {token}"
                )
            ),
        )
    except json.JSONDecodeError as error:
        raise IndependentScoreFreeAuditError(f"{label} is not strict JSON") from error
    if not isinstance(result, dict):
        raise IndependentScoreFreeAuditError(f"{label} must be a JSON object")
    return result


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise IndependentScoreFreeAuditError(f"{label} must be non-empty text")
    if value != value.strip() or not value.isascii():
        raise IndependentScoreFreeAuditError(
            f"{label} must be exact ASCII without edge whitespace"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise IndependentScoreFreeAuditError(f"{label} contains a control character")
    if any(character in value for character in (",", '"')):
        raise IndependentScoreFreeAuditError(f"{label} contains a CSV delimiter")
    return value


def _sequence(value: str, label: str, *, optional: bool = False) -> str:
    normalized = "".join(character for character in value if not character.isspace())
    normalized = normalized.upper()
    try:
        normalized.encode("ascii")
    except UnicodeEncodeError as error:
        raise IndependentScoreFreeAuditError(f"{label} is not ASCII") from error
    if not normalized and optional:
        return ""
    if not normalized:
        raise IndependentScoreFreeAuditError(f"{label} is empty")
    invalid = sorted(set(normalized) - AMINO_ACIDS)
    if invalid:
        raise IndependentScoreFreeAuditError(
            f"{label} contains invalid residue symbols: {invalid}"
        )
    return normalized


def _membership_sha256(values: Iterable[str]) -> str:
    members = sorted(set(values))
    digest = hashlib.sha256()
    for member in members:
        checked = _identifier(member, "membership identifier")
        encoded = checked.encode("ascii")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
    return digest.hexdigest()


def _load_authority(payload_bytes: bytes) -> tuple[set[str], str]:
    payload = _strict_object(payload_bytes, "authority allowlist")
    if set(payload) != {"format_version", "authorized_split", "instance_ids"}:
        raise IndependentScoreFreeAuditError("authority schema differs")
    if payload["format_version"] != AUTHORITY_FORMAT:
        raise IndependentScoreFreeAuditError("authority format differs")
    if payload["authorized_split"] != "train":
        raise IndependentScoreFreeAuditError("authority split is not train")
    raw_ids = payload["instance_ids"]
    if not isinstance(raw_ids, list) or not raw_ids:
        raise IndependentScoreFreeAuditError("authority membership is empty")
    identifiers = [_identifier(item, "authority identifier") for item in raw_ids]
    if len(identifiers) != len(set(identifiers)):
        raise IndependentScoreFreeAuditError("authority membership contains duplicates")
    return set(identifiers), _membership_sha256(identifiers)


def _manifest_rows(payload_bytes: bytes) -> list[list[str]]:
    label = "canonical manifest"
    if not payload_bytes or payload_bytes.startswith(b"\xef\xbb\xbf"):
        raise IndependentScoreFreeAuditError(f"{label} is empty or contains a BOM")
    if b"\0" in payload_bytes or b"\r" in payload_bytes:
        raise IndependentScoreFreeAuditError(f"{label} contains NUL or CR")
    if not payload_bytes.endswith(b"\n"):
        raise IndependentScoreFreeAuditError(f"{label} lacks terminal LF")
    lines = payload_bytes.split(b"\n")[:-1]
    if not lines or any(not line for line in lines):
        raise IndependentScoreFreeAuditError(f"{label} contains an empty line")
    result: list[list[str]] = []
    for number, raw_line in enumerate(lines, 1):
        try:
            line = raw_line.decode("utf-8")
            rows = list(csv.reader([line], strict=True))
        except (UnicodeDecodeError, csv.Error) as error:
            raise IndependentScoreFreeAuditError(
                f"{label} line {number} is invalid"
            ) from error
        if len(rows) != 1:
            raise IndependentScoreFreeAuditError(
                f"{label} line {number} spans records"
            )
        result.append(rows[0])
    return result


def _load_records(
    payload_bytes: bytes,
    authority_ids: set[str],
) -> dict[str, tuple[str, str, str]]:
    rows = _manifest_rows(payload_bytes)
    if tuple(rows[0]) != MANIFEST_HEADER:
        raise IndependentScoreFreeAuditError("manifest header differs")
    records: dict[str, tuple[str, str, str]] = {}
    for number, row in enumerate(rows[1:], 2):
        if len(row) != 4:
            raise IndependentScoreFreeAuditError(f"manifest row {number} differs")
        instance_id = _identifier(row[0], f"manifest row {number} identifier")
        if instance_id not in authority_ids or instance_id in records:
            raise IndependentScoreFreeAuditError(
                "manifest membership is outside authority or duplicated"
            )
        records[instance_id] = (
            _sequence(row[1], f"{instance_id}:VH"),
            _sequence(row[2], f"{instance_id}:VL", optional=True),
            _sequence(row[3], f"{instance_id}:antigen"),
        )
    if set(records) != authority_ids:
        raise IndependentScoreFreeAuditError(
            "manifest membership does not equal authority"
        )
    return records


def _frame(value: str) -> bytes:
    encoded = value.encode("ascii")
    return str(len(encoded)).encode("ascii") + b":" + encoded


def _antibody_id(vh: str, vl: str) -> str:
    return hashlib.sha256(b"antibody-v1\0" + _frame(vh) + _frame(vl)).hexdigest()


def _antigen_id(sequence: str) -> str:
    return hashlib.sha256(b"antigen-v1\0" + _frame(sequence)).hexdigest()


def _pair_id(antibody_id: str, antigen_id: str) -> str:
    material = (
        b"pair-v1\0"
        + antibody_id.encode("ascii")
        + b"\0"
        + antigen_id.encode("ascii")
    )
    return hashlib.sha256(material).hexdigest()


def _canonical_entities(
    records: Mapping[str, tuple[str, str, str]],
) -> tuple[
    dict[str, tuple[str, str]],
    dict[str, str],
    list[dict[str, object]],
]:
    antibodies: dict[str, tuple[str, str]] = {}
    antigens: dict[str, str] = {}
    pair_sources: dict[tuple[str, str], list[str]] = defaultdict(list)
    for instance_id in sorted(records):
        vh, vl, antigen = records[instance_id]
        antibody_id = _antibody_id(vh, vl)
        antigen_id = _antigen_id(antigen)
        if antibodies.setdefault(antibody_id, (vh, vl)) != (vh, vl):
            raise IndependentScoreFreeAuditError("antibody digest collision")
        if antigens.setdefault(antigen_id, antigen) != antigen:
            raise IndependentScoreFreeAuditError("antigen digest collision")
        pair_sources[(antibody_id, antigen_id)].append(instance_id)
    pairs = [
        {
            "pair_id": _pair_id(antibody_id, antigen_id),
            "antibody_id": antibody_id,
            "antigen_id": antigen_id,
            "source_instance_ids": sorted(source_ids),
        }
        for (antibody_id, antigen_id), source_ids in sorted(pair_sources.items())
    ]
    if len({str(pair["pair_id"]) for pair in pairs}) != len(pairs):
        raise IndependentScoreFreeAuditError("pair digest collision")
    return antibodies, antigens, pairs


def _edge_ledger(
    payload_bytes: bytes,
    view: str,
    allowed: set[str],
    antibodies: Mapping[str, tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    payload = _strict_object(payload_bytes, f"{view} edge ledger")
    if set(payload) != {"format_version", "view", "edges"}:
        raise IndependentScoreFreeAuditError(f"{view} edge schema differs")
    if payload["format_version"] != EDGE_FORMAT or payload["view"] != view:
        raise IndependentScoreFreeAuditError(f"{view} edge binding differs")
    if not isinstance(payload["edges"], list):
        raise IndependentScoreFreeAuditError(f"{view} edges are not a list")
    edges: list[tuple[str, str]] = []
    for raw_edge in payload["edges"]:
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise IndependentScoreFreeAuditError(f"{view} edge shape differs")
        left = _identifier(raw_edge[0], f"{view} edge left")
        right = _identifier(raw_edge[1], f"{view} edge right")
        if left >= right or left not in allowed or right not in allowed:
            raise IndependentScoreFreeAuditError(f"{view} edge is noncanonical")
        if view == "vl" and (not antibodies[left][1] or not antibodies[right][1]):
            raise IndependentScoreFreeAuditError("VL edge uses a missing chain")
        edges.append((left, right))
    if tuple(edges) != tuple(sorted(set(edges))):
        raise IndependentScoreFreeAuditError(f"{view} edges are unsorted or repeated")
    return tuple(edges)


def _graph(
    entities: Iterable[str],
    edges: Iterable[tuple[str, str]],
) -> dict[str, set[str]]:
    result = {entity: set() for entity in entities}
    for left, right in edges:
        result[left].add(right)
        result[right].add(left)
    return result


def _components(graph: Mapping[str, set[str]]) -> tuple[dict[str, str], dict[str, int]]:
    component: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for start in sorted(graph):
        if start in component:
            continue
        pending = [start]
        members: set[str] = set()
        while pending:
            current = pending.pop()
            if current in members:
                continue
            members.add(current)
            pending.extend(sorted(graph[current] - members, reverse=True))
        identity = min(members)
        sizes[identity] = len(members)
        component.update({member: identity for member in members})
    return component, sizes


def _log1p(value: int) -> Decimal:
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        return +(Decimal(value) + 1).ln()


def _feature_rows(
    antibodies: Sequence[dict[str, object]],
    antigens: Sequence[dict[str, object]],
) -> dict[str, list[tuple[str, bool, bool, list[Decimal | None]]]]:
    antibody_features = [
        ("vh_length", False, False, [Decimal(row["vh_length"]) for row in antibodies]),
        (
            "vl_length",
            True,
            False,
            [None if row["vl_length"] is None else Decimal(row["vl_length"]) for row in antibodies],
        ),
        (
            "concatenated_antibody_length",
            False,
            False,
            [Decimal(row["concatenated_antibody_length"]) for row in antibodies],
        ),
        (
            "log1p_vh_direct_degree",
            False,
            False,
            [_log1p(int(row["vh_direct_degree"])) for row in antibodies],
        ),
        (
            "log1p_vl_direct_degree",
            True,
            False,
            [None if row["vl_direct_degree"] is None else _log1p(int(row["vl_direct_degree"])) for row in antibodies],
        ),
        (
            "log1p_concatenated_direct_degree",
            False,
            False,
            [_log1p(int(row["concatenated_direct_degree"])) for row in antibodies],
        ),
        (
            "log1p_union_direct_degree",
            False,
            False,
            [_log1p(int(row["union_direct_degree"])) for row in antibodies],
        ),
        (
            "log1p_pair_incidence_degree",
            False,
            False,
            [_log1p(int(row["pair_incidence_degree"])) for row in antibodies],
        ),
        (
            "component_size",
            False,
            False,
            [Decimal(row["component_size"]) for row in antibodies],
        ),
        (
            "missing_vl",
            False,
            True,
            [Decimal(int(row["vl_length"] is None)) for row in antibodies],
        ),
    ]
    antigen_features = [
        ("antigen_length", False, False, [Decimal(row["antigen_length"]) for row in antigens]),
        (
            "log1p_antigen_direct_degree",
            False,
            False,
            [_log1p(int(row["antigen_direct_degree"])) for row in antigens],
        ),
        (
            "log1p_pair_incidence_degree",
            False,
            False,
            [_log1p(int(row["pair_incidence_degree"])) for row in antigens],
        ),
        (
            "component_size",
            False,
            False,
            [Decimal(row["component_size"]) for row in antigens],
        ),
    ]
    return {"antibody": antibody_features, "antigen": antigen_features}


def _quantile_schemes(
    antibodies: Sequence[dict[str, object]],
    antigens: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for axis, features in _feature_rows(antibodies, antigens).items():
        for name, optional, categorical, raw_values in features:
            if categorical:
                observed = set(raw_values)
                result.append(
                    {
                        "axis": axis,
                        "feature_name": name,
                        "cuts_decimal": [],
                        "optional_missing_bin": False,
                        "categorical_binary": True,
                        "informative": len(observed) >= 2,
                    }
                )
                continue
            values = sorted(value for value in raw_values if value is not None)
            if not values:
                raise IndependentScoreFreeAuditError(f"no values for feature {name}")
            tentative = []
            for numerator in range(1, 5):
                one_based = (numerator * len(values) + 4) // 5
                tentative.append(values[one_based - 1])
            cuts = tuple(sorted({value for value in tentative if value < values[-1]}))

            def bin_index(value: Decimal | None) -> int:
                if value is None:
                    if not optional:
                        raise IndependentScoreFreeAuditError(
                            f"unexpected missing value for {name}"
                        )
                    return 0
                offset = int(optional)
                return offset + sum(cut < value for cut in cuts)

            observed_bins = {bin_index(value) for value in raw_values}
            result.append(
                {
                    "axis": axis,
                    "feature_name": name,
                    "cuts_decimal": [str(value) for value in cuts],
                    "optional_missing_bin": optional,
                    "categorical_binary": False,
                    "informative": len(observed_bins) >= 2,
                }
            )
    return sorted(
        result,
        key=lambda row: (
            str(row["axis"]),
            str(row["feature_name"]),
            tuple(str(value) for value in row["cuts_decimal"]),
        ),
    )


def reconstruct_score_free_authority(
    *,
    authority_bytes: bytes,
    manifest_bytes: bytes,
    edge_ledger_bytes: Mapping[str, bytes],
    expected_primary_implementation_sha256: Mapping[str, str],
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
) -> dict[str, object]:
    """Reconstruct the primary scientific payload without importing it."""
    if set(edge_ledger_bytes) != set(VIEWS):
        raise IndependentScoreFreeAuditError("exactly four edge ledgers are required")
    if set(expected_primary_implementation_sha256) != {
        "score_free_authority",
        "score_free_split_statistics",
    }:
        raise IndependentScoreFreeAuditError(
            "primary implementation anchors are incomplete"
        )
    for label, value in expected_primary_implementation_sha256.items():
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise IndependentScoreFreeAuditError(
                f"primary implementation anchor {label} is malformed"
            )
    authority_ids, authority_membership_hash = _load_authority(authority_bytes)
    records = _load_records(manifest_bytes, authority_ids)
    antibody_sequences, antigen_sequences, pairs = _canonical_entities(records)
    counts = {
        "source_rows": len(records),
        "exact_pairs": len(pairs),
        "antibody_entities": len(antibody_sequences),
        "antigen_entities": len(antigen_sequences),
    }
    if counts != dict(expected_counts):
        raise IndependentScoreFreeAuditError(
            f"authority inventory differs: expected={dict(expected_counts)}, observed={counts}"
        )
    ledgers = {
        view: _edge_ledger(
            edge_ledger_bytes[view],
            view,
            set(antigen_sequences) if view == "antigen" else set(antibody_sequences),
            antibody_sequences,
        )
        for view in VIEWS
    }
    antibody_graphs = {
        view: _graph(antibody_sequences, ledgers[view])
        for view in ("vh", "vl", "concatenated_antibody")
    }
    antibody_union_edges = set().union(
        ledgers["vh"], ledgers["vl"], ledgers["concatenated_antibody"]
    )
    antibody_union = _graph(antibody_sequences, antibody_union_edges)
    antigen_graph = _graph(antigen_sequences, ledgers["antigen"])
    antibody_component, antibody_component_sizes = _components(antibody_union)
    antigen_component, antigen_component_sizes = _components(antigen_graph)
    antibody_incidence = Counter(str(pair["antibody_id"]) for pair in pairs)
    antigen_incidence = Counter(str(pair["antigen_id"]) for pair in pairs)
    antibodies = [
        {
            "entity_id": entity_id,
            "component_id": antibody_component[entity_id],
            "vh_length": len(vh),
            "vl_length": None if not vl else len(vl),
            "concatenated_antibody_length": len(vh) + len(vl),
            "vh_direct_degree": len(antibody_graphs["vh"][entity_id]),
            "vl_direct_degree": None if not vl else len(antibody_graphs["vl"][entity_id]),
            "concatenated_direct_degree": len(
                antibody_graphs["concatenated_antibody"][entity_id]
            ),
            "union_direct_degree": len(antibody_union[entity_id]),
            "pair_incidence_degree": antibody_incidence[entity_id],
            "component_size": antibody_component_sizes[antibody_component[entity_id]],
        }
        for entity_id, (vh, vl) in sorted(antibody_sequences.items())
    ]
    antigens = [
        {
            "entity_id": entity_id,
            "component_id": antigen_component[entity_id],
            "antigen_length": len(sequence),
            "antigen_direct_degree": len(antigen_graph[entity_id]),
            "pair_incidence_degree": antigen_incidence[entity_id],
            "component_size": antigen_component_sizes[antigen_component[entity_id]],
        }
        for entity_id, sequence in sorted(antigen_sequences.items())
    ]
    payload: dict[str, object] = {
        "format_version": BUNDLE_FORMAT,
        "input_sha256": {
            "authority_allowlist": hashlib.sha256(authority_bytes).hexdigest(),
            "canonical_manifest": hashlib.sha256(manifest_bytes).hexdigest(),
            **{
                f"edge_ledger_{view}": hashlib.sha256(
                    edge_ledger_bytes[view]
                ).hexdigest()
                for view in VIEWS
            },
        },
        "authority_membership_sha256": authority_membership_hash,
        "implementation_source_sha256": dict(
            sorted(expected_primary_implementation_sha256.items())
        ),
        "counts": counts,
        "antibodies": antibodies,
        "antigens": antigens,
        "pairs": pairs,
        "quantile_schemes": _quantile_schemes(antibodies, antigens),
        "forbidden_outcome_or_model_fields_accepted": False,
    }
    payload["scientific_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _fraction(payload: Sequence[int]) -> Fraction:
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in payload)
        or payload[1] <= 0
    ):
        raise IndependentScoreFreeAuditError("exact fraction payload is malformed")
    return Fraction(payload[0], payload[1])


def _fraction_payload(value: Fraction | None) -> list[int] | None:
    if value is None:
        return None
    return [value.numerator, value.denominator]


def _profile_values(row: Mapping[str, object], axis: str) -> dict[str, Decimal | None]:
    if axis == "antibody":
        missing_vl = row["vl_length"] is None
        return {
            "vh_length": Decimal(int(row["vh_length"])),
            "vl_length": None if missing_vl else Decimal(int(row["vl_length"])),
            "concatenated_antibody_length": Decimal(
                int(row["concatenated_antibody_length"])
            ),
            "log1p_vh_direct_degree": _log1p(int(row["vh_direct_degree"])),
            "log1p_vl_direct_degree": (
                None
                if row["vl_direct_degree"] is None
                else _log1p(int(row["vl_direct_degree"]))
            ),
            "log1p_concatenated_direct_degree": _log1p(
                int(row["concatenated_direct_degree"])
            ),
            "log1p_union_direct_degree": _log1p(
                int(row["union_direct_degree"])
            ),
            "log1p_pair_incidence_degree": _log1p(
                int(row["pair_incidence_degree"])
            ),
            "component_size": Decimal(int(row["component_size"])),
            "missing_vl": Decimal(int(missing_vl)),
        }
    return {
        "antigen_length": Decimal(int(row["antigen_length"])),
        "log1p_antigen_direct_degree": _log1p(
            int(row["antigen_direct_degree"])
        ),
        "log1p_pair_incidence_degree": _log1p(
            int(row["pair_incidence_degree"])
        ),
        "component_size": Decimal(int(row["component_size"])),
    }


def _profiles(
    authority_payload: Mapping[str, object],
    axis: str,
    required_ids: set[str],
) -> dict[str, dict[str, object]]:
    key = "antibodies" if axis == "antibody" else "antigens"
    rows = authority_payload[key]
    if not isinstance(rows, list):
        raise IndependentScoreFreeAuditError(f"authority {key} are malformed")
    available = {str(row["entity_id"]): row for row in rows}
    if required_ids - available.keys():
        raise IndependentScoreFreeAuditError(f"fold references an unknown {axis}")
    return {
        entity_id: {
            "entity_id": entity_id,
            "component_id": str(available[entity_id]["component_id"]),
            "axis": axis,
            "values": _profile_values(available[entity_id], axis),
        }
        for entity_id in sorted(required_ids)
    }


def _profile_digest(profiles: Mapping[str, Mapping[str, object]]) -> str:
    payload = []
    for entity_id in sorted(profiles):
        profile = profiles[entity_id]
        values = profile["values"]
        assert isinstance(values, dict)
        payload.append(
            {
                "entity_id": entity_id,
                "component_id": profile["component_id"],
                "axis": profile["axis"],
                "values_decimal": [
                    [name, None if values[name] is None else str(values[name])]
                    for name, *_ in FEATURES[str(profile["axis"])]
                ],
            }
        )
    return _compact_json_sha256(payload)


def _pair_digest(pairs: Sequence[Mapping[str, object]]) -> str:
    payload = [
        {
            "pair_id": pair["pair_id"],
            "antibody_id": pair["antibody_id"],
            "antigen_id": pair["antigen_id"],
        }
        for pair in sorted(pairs, key=lambda row: str(row["pair_id"]))
    ]
    return _compact_json_sha256(payload)


def _compact_membership_sha256(values: Iterable[str]) -> str:
    members = sorted(set(values))
    return _compact_json_sha256(members)


def _scheme(
    profiles: Mapping[str, Mapping[str, object]],
    feature: tuple[str, bool, bool, bool, bool, bool],
) -> dict[str, object]:
    name, _, _, _, optional, categorical = feature
    raw = [profile["values"][name] for profile in profiles.values()]
    if categorical:
        observed = set(raw)
        return {
            "cuts": (),
            "optional": False,
            "categorical": True,
            "informative": len(observed) >= 2,
            "bin_count": 2,
        }
    values = sorted(value for value in raw if value is not None)
    if not values:
        raise IndependentScoreFreeAuditError(f"no values for fold feature {name}")
    tentative = []
    for numerator in range(1, 5):
        one_based = (numerator * len(values) + 4) // 5
        tentative.append(values[one_based - 1])
    cuts = tuple(sorted({value for value in tentative if value < values[-1]}))

    def bin_index(value: Decimal | None) -> int:
        if value is None:
            if not optional:
                raise IndependentScoreFreeAuditError(f"unexpected missing {name}")
            return 0
        return int(optional) + sum(cut < value for cut in cuts)

    observed_bins = {bin_index(value) for value in raw}
    return {
        "cuts": cuts,
        "optional": optional,
        "categorical": False,
        "informative": len(observed_bins) >= 2,
        "bin_count": len(cuts) + 1 + int(optional),
    }


def _bin_index(scheme: Mapping[str, object], value: Decimal | None) -> int:
    if scheme["categorical"]:
        if value not in {Decimal(0), Decimal(1)}:
            raise IndependentScoreFreeAuditError("categorical value is not binary")
        return int(value)
    if value is None:
        if not scheme["optional"]:
            raise IndependentScoreFreeAuditError("unexpected missing fold value")
        return 0
    cuts = scheme["cuts"]
    assert isinstance(cuts, tuple)
    return int(bool(scheme["optional"])) + sum(cut < value for cut in cuts)


def _smd(
    left: Sequence[Decimal],
    right: Sequence[Decimal],
) -> tuple[Decimal | None, str | None]:
    if len(left) < 2 or len(right) < 2:
        return None, "smd_requires_at_least_two_nonmissing_values_per_group"
    left_exact = tuple(Fraction(value) for value in left)
    right_exact = tuple(Fraction(value) for value in right)
    left_mean = sum(left_exact, Fraction(0)) / len(left_exact)
    right_mean = sum(right_exact, Fraction(0)) / len(right_exact)
    left_variance = sum(
        ((value - left_mean) ** 2 for value in left_exact), Fraction(0)
    ) / (len(left_exact) - 1)
    right_variance = sum(
        ((value - right_mean) ** 2 for value in right_exact), Fraction(0)
    ) / (len(right_exact) - 1)
    pooled = (left_variance + right_variance) / 2
    difference = abs(left_mean - right_mean)
    if pooled == 0:
        if difference == 0:
            return Decimal(0), None
        return None, "smd_zero_pooled_variance_with_unequal_means"
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        context.rounding = ROUND_HALF_EVEN
        decimal_difference = Decimal(difference.numerator) / Decimal(
            difference.denominator
        )
        decimal_pooled = Decimal(pooled.numerator) / Decimal(pooled.denominator)
        return +(decimal_difference / decimal_pooled.sqrt()), None


def _ks(left: Sequence[Decimal], right: Sequence[Decimal]) -> Fraction:
    left_counts = Counter(left)
    right_counts = Counter(right)
    left_seen = 0
    right_seen = 0
    result = Fraction(0)
    for value in sorted(set(left_counts) | set(right_counts)):
        left_seen += left_counts[value]
        right_seen += right_counts[value]
        result = max(
            result,
            abs(
                Fraction(left_seen, len(left))
                - Fraction(right_seen, len(right))
            ),
        )
    return result


def _representativeness(
    *,
    left_ids: set[str],
    right_ids: set[str],
    profiles: Mapping[str, Mapping[str, object]],
    axis: str,
    reference_name: str,
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    maximum_smd = Decimal(str(thresholds["maximum_smd_decimal"]))
    maximum_tv = _fraction(thresholds["maximum_tv_exact"])
    maximum_ks = _fraction(thresholds["maximum_ks_exact"])
    feature_payloads = []
    for feature in FEATURES[axis]:
        name, gate_smd, gate_ks, gate_tv, _, _ = feature
        scheme = _scheme(profiles, feature)
        left_profiles = [profiles[item] for item in sorted(left_ids)]
        right_profiles = [profiles[item] for item in sorted(right_ids)]
        left_values = [
            value
            for profile in left_profiles
            if (value := profile["values"][name]) is not None
        ]
        right_values = [
            value
            for profile in right_profiles
            if (value := profile["values"][name]) is not None
        ]
        failures = []
        smd_value: Decimal | None = None
        ks_value: Fraction | None = None
        if gate_smd:
            smd_value, failure = _smd(left_values, right_values)
            if failure is not None:
                failures.append(failure)
            elif smd_value is not None and smd_value > maximum_smd:
                failures.append("smd_exceeds_locked_threshold")
        if gate_ks:
            if not left_values or not right_values:
                failures.append("ks_requires_nonmissing_values_in_both_groups")
            else:
                ks_value = _ks(left_values, right_values)
                if ks_value > maximum_ks:
                    failures.append("ks_exceeds_locked_threshold")
        left_bins = Counter(
            _bin_index(scheme, profile["values"][name])
            for profile in left_profiles
        )
        right_bins = Counter(
            _bin_index(scheme, profile["values"][name])
            for profile in right_profiles
        )
        tv_value = sum(
            (
                abs(
                    Fraction(left_bins[index], len(left_profiles))
                    - Fraction(right_bins[index], len(right_profiles))
                )
                for index in range(int(scheme["bin_count"]))
            ),
            Fraction(0),
        ) / 2
        if gate_tv and scheme["informative"] and tv_value > maximum_tv:
            failures.append("tv_exceeds_locked_threshold")
        feature_payloads.append(
            {
                "feature_name": name,
                "left_count": len(left_profiles),
                "right_count": len(right_profiles),
                "nonmissing_left_count": len(left_values),
                "nonmissing_right_count": len(right_values),
                "smd_decimal": None if smd_value is None else str(smd_value),
                "tv_exact": _fraction_payload(tv_value),
                "ks_exact": _fraction_payload(ks_value),
                "quantile_cuts_decimal": [
                    str(value) for value in scheme["cuts"]
                ],
                "optional_missing_bin": scheme["optional"],
                "categorical_binary": scheme["categorical"],
                "histogram_informative": scheme["informative"],
                "failures": failures,
                "passed": not failures,
            }
        )
    return {
        "axis": axis,
        "reference_name": reference_name,
        "features": feature_payloads,
        "passed": all(feature["passed"] for feature in feature_payloads),
    }


def _chance_mrr(library_size: int, positives: int) -> Fraction:
    denominator = comb(library_size, positives)
    return sum(
        Fraction(comb(library_size - rank, positives - 1), denominator * rank)
        for rank in range(1, library_size - positives + 2)
    )


def _ceil(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _floor(value: Fraction) -> int:
    return value.numerator // value.denominator


def _panel(
    *,
    fold_id: str,
    pairs: Sequence[Mapping[str, object]],
    validation_ids: set[str],
    training_ids: set[str],
    antibody_profiles: Mapping[str, Mapping[str, object]],
    antigen_profiles: Mapping[str, Mapping[str, object]],
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    validation_pairs = [pair for pair in pairs if pair["pair_id"] in validation_ids]
    training_pairs = [pair for pair in pairs if pair["pair_id"] in training_ids]
    universe_antibodies = {str(pair["antibody_id"]) for pair in pairs}
    universe_antigens = {str(pair["antigen_id"]) for pair in pairs}
    validation_antibodies = {
        str(pair["antibody_id"]) for pair in validation_pairs
    }
    validation_antigens = {str(pair["antigen_id"]) for pair in validation_pairs}
    training_antibodies = {str(pair["antibody_id"]) for pair in training_pairs}
    training_antigens = {str(pair["antigen_id"]) for pair in training_pairs}
    antibody_degrees = Counter(str(pair["antibody_id"]) for pair in validation_pairs)
    antigen_degrees = Counter(str(pair["antigen_id"]) for pair in validation_pairs)
    antibody_singletons = Fraction(
        sum(degree == 1 for degree in antibody_degrees.values()),
        len(validation_antibodies),
    )
    antigen_singletons = Fraction(
        sum(degree == 1 for degree in antigen_degrees.values()),
        len(validation_antigens),
    )
    density = Fraction(
        len(validation_pairs),
        len(validation_antibodies) * len(validation_antigens),
    )
    antibody_chance = sum(
        (
            _chance_mrr(len(validation_antigens), degree)
            for degree in antibody_degrees.values()
        ),
        Fraction(0),
    ) / len(validation_antibodies)
    antigen_chance = sum(
        (
            _chance_mrr(len(validation_antibodies), degree)
            for degree in antigen_degrees.values()
        ),
        Fraction(0),
    ) / len(validation_antigens)
    pair_count = len(pairs)
    lower = max(
        int(thresholds["minimum_validation_pairs"]),
        _ceil(_fraction(thresholds["validation_pair_lower_fraction"]) * pair_count),
    )
    upper = _floor(
        _fraction(thresholds["validation_pair_upper_fraction"]) * pair_count
    )
    failures = []
    if not lower <= len(validation_pairs) <= upper:
        failures.append("validation_pair_count_outside_locked_interval")
    endpoint_minimum = int(thresholds["minimum_validation_endpoints_per_axis"])
    if len(validation_antibodies) < endpoint_minimum:
        failures.append("validation_antibody_count_below_locked_minimum")
    if len(validation_antigens) < endpoint_minimum:
        failures.append("validation_antigen_count_below_locked_minimum")
    antibody_components = len(
        {
            str(antibody_profiles[item]["component_id"])
            for item in validation_antibodies
        }
    )
    antigen_components = len(
        {
            str(antigen_profiles[item]["component_id"])
            for item in validation_antigens
        }
    )
    component_minimum = int(thresholds["minimum_validation_components_per_axis"])
    if antibody_components < component_minimum:
        failures.append("validation_antibody_component_count_below_locked_minimum")
    if antigen_components < component_minimum:
        failures.append("validation_antigen_component_count_below_locked_minimum")
    if len(training_pairs) < _ceil(
        _fraction(thresholds["minimum_training_pair_fraction"]) * pair_count
    ):
        failures.append("training_pair_retention_below_locked_minimum")
    endpoint_retention = _fraction(thresholds["minimum_training_endpoint_fraction"])
    if len(training_antibodies) < _ceil(endpoint_retention * len(universe_antibodies)):
        failures.append("training_antibody_retention_below_locked_minimum")
    if len(training_antigens) < _ceil(endpoint_retention * len(universe_antigens)):
        failures.append("training_antigen_retention_below_locked_minimum")
    quarantine = pair_count - len(validation_pairs) - len(training_pairs)
    if quarantine > _floor(
        _fraction(thresholds["maximum_quarantine_pair_fraction"]) * pair_count
    ):
        failures.append("quarantine_pair_count_above_locked_maximum")
    return {
        "universe_pair_count": pair_count,
        "universe_antibody_count": len(universe_antibodies),
        "universe_antigen_count": len(universe_antigens),
        "validation_pair_count": len(validation_pairs),
        "validation_antibody_count": len(validation_antibodies),
        "validation_antigen_count": len(validation_antigens),
        "validation_antibody_component_count": antibody_components,
        "validation_antigen_component_count": antigen_components,
        "training_pair_count": len(training_pairs),
        "training_antibody_count": len(training_antibodies),
        "training_antigen_count": len(training_antigens),
        "quarantine_pair_count": quarantine,
        "antibody_singleton_fraction": _fraction_payload(antibody_singletons),
        "antigen_singleton_fraction": _fraction_payload(antigen_singletons),
        "bipartite_density": _fraction_payload(density),
        "antibody_to_antigen_chance_mrr": _fraction_payload(antibody_chance),
        "antigen_to_antibody_chance_mrr": _fraction_payload(antigen_chance),
        "bidirectional_chance_mrr": _fraction_payload(
            (antibody_chance + antigen_chance) / 2
        ),
        "shape_failures": failures,
        "shape_passed": not failures,
    }


def reconstruct_score_free_fold_payload(
    *,
    authority_payload: Mapping[str, object],
    candidate_payload: Mapping[str, object],
    scope: str,
    expected_primary_source_sha256: str,
    expected_outer_fold_id: str | None = None,
    expected_outer_training_pair_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    """Recompute one primary fold payload from independent authority values."""
    if scope not in {"outer", "inner"}:
        raise IndependentScoreFreeAuditError("fold scope must be outer or inner")
    if candidate_payload.get("schema_version") != "score_free_fold_audit_v2":
        raise IndependentScoreFreeAuditError("candidate fold schema differs")
    shape_thresholds = FOLD_SHAPE_THRESHOLDS[scope]
    distribution_thresholds = DISTRIBUTION_THRESHOLDS[scope]
    if candidate_payload.get("shape_thresholds") != shape_thresholds:
        raise IndependentScoreFreeAuditError("candidate shape thresholds differ")
    if candidate_payload.get("distribution_thresholds") != distribution_thresholds:
        raise IndependentScoreFreeAuditError("candidate distribution thresholds differ")
    all_pairs = authority_payload["pairs"]
    if not isinstance(all_pairs, list):
        raise IndependentScoreFreeAuditError("authority pairs are malformed")
    pair_by_id = {str(pair["pair_id"]): pair for pair in all_pairs}
    universe_ids = candidate_payload.get("universe_pair_ids")
    validation_ids = candidate_payload.get("validation_pair_ids")
    training_ids = candidate_payload.get("training_pair_ids")
    for label, values in (
        ("universe_pair_ids", universe_ids),
        ("validation_pair_ids", validation_ids),
        ("training_pair_ids", training_ids),
    ):
        if (
            not isinstance(values, list)
            or not values
            or values != sorted(set(values))
            or any(value not in pair_by_id for value in values)
        ):
            raise IndependentScoreFreeAuditError(f"candidate {label} differs")
    universe_set = set(universe_ids)
    validation_set = set(validation_ids)
    training_set = set(training_ids)
    if not validation_set <= universe_set or not training_set <= universe_set:
        raise IndependentScoreFreeAuditError("fold role membership leaves universe")
    if validation_set & training_set:
        raise IndependentScoreFreeAuditError("fold roles overlap")
    pairs = [pair_by_id[pair_id] for pair_id in universe_ids]
    antibody_ids = {str(pair["antibody_id"]) for pair in pairs}
    antigen_ids = {str(pair["antigen_id"]) for pair in pairs}
    antibody_profiles = _profiles(authority_payload, "antibody", antibody_ids)
    antigen_profiles = _profiles(authority_payload, "antigen", antigen_ids)
    validation_pairs = [pair for pair in pairs if pair["pair_id"] in validation_set]
    training_pairs = [pair for pair in pairs if pair["pair_id"] in training_set]
    validation_antibodies = {str(pair["antibody_id"]) for pair in validation_pairs}
    validation_antigens = {str(pair["antigen_id"]) for pair in validation_pairs}
    training_antibodies = {str(pair["antibody_id"]) for pair in training_pairs}
    training_antigens = {str(pair["antigen_id"]) for pair in training_pairs}
    representativeness = {
        "antibody_vs_universe": _representativeness(
            left_ids=validation_antibodies,
            right_ids=antibody_ids,
            profiles=antibody_profiles,
            axis="antibody",
            reference_name="universe",
            thresholds=distribution_thresholds,
        ),
        "antibody_vs_training": _representativeness(
            left_ids=validation_antibodies,
            right_ids=training_antibodies,
            profiles=antibody_profiles,
            axis="antibody",
            reference_name="training",
            thresholds=distribution_thresholds,
        ),
        "antigen_vs_universe": _representativeness(
            left_ids=validation_antigens,
            right_ids=antigen_ids,
            profiles=antigen_profiles,
            axis="antigen",
            reference_name="universe",
            thresholds=distribution_thresholds,
        ),
        "antigen_vs_training": _representativeness(
            left_ids=validation_antigens,
            right_ids=training_antigens,
            profiles=antigen_profiles,
            axis="antigen",
            reference_name="training",
            thresholds=distribution_thresholds,
        ),
    }
    source_hash = _identifier(expected_primary_source_sha256, "primary source SHA-256")
    if len(source_hash) != 64 or any(character not in "0123456789abcdef" for character in source_hash):
        raise IndependentScoreFreeAuditError("primary source SHA-256 is malformed")
    inner_scope: dict[str, object] | None = None
    if scope == "inner":
        if expected_outer_fold_id is None or expected_outer_training_pair_ids is None:
            raise IndependentScoreFreeAuditError("inner audit lacks outer scope anchor")
        outer_training = sorted(set(expected_outer_training_pair_ids))
        if outer_training != universe_ids:
            raise IndependentScoreFreeAuditError(
                "inner universe differs from externally anchored outer training"
            )
        inner_scope = {
            "schema_version": "score_free_inner_universe_scope_receipt_v1",
            "outer_fold_id": expected_outer_fold_id,
            "inner_fold_id": candidate_payload["fold_id"],
            "complete_authority_pair_membership_sha256": (
                _compact_membership_sha256(pair_by_id)
            ),
            "outer_training_pair_membership_sha256": (
                _compact_membership_sha256(outer_training)
            ),
            "inner_universe_pair_membership_sha256": (
                _compact_membership_sha256(universe_ids)
            ),
            "outer_training_pair_count": len(outer_training),
            "inner_universe_pair_count": len(universe_ids),
            "implementation_source_sha256": source_hash,
            "passed": True,
        }
    panel = _panel(
        fold_id=str(candidate_payload["fold_id"]),
        pairs=pairs,
        validation_ids=validation_set,
        training_ids=training_set,
        antibody_profiles=antibody_profiles,
        antigen_profiles=antigen_profiles,
        thresholds=shape_thresholds,
    )
    passed = bool(panel["shape_passed"]) and all(
        comparison["passed"] for comparison in representativeness.values()
    )
    return {
        "schema_version": "score_free_fold_audit_v2",
        "fold_id": candidate_payload["fold_id"],
        "shape_thresholds": shape_thresholds,
        "distribution_thresholds": distribution_thresholds,
        "panel": panel,
        "representativeness": representativeness,
        "universe_pair_ids": universe_ids,
        "validation_pair_ids": validation_ids,
        "training_pair_ids": training_ids,
        "validation_antibody_ids": sorted(validation_antibodies),
        "validation_antigen_ids": sorted(validation_antigens),
        "input_bindings": {
            "pair_inventory_sha256": _pair_digest(pairs),
            "antibody_profile_sha256": _profile_digest(antibody_profiles),
            "antigen_profile_sha256": _profile_digest(antigen_profiles),
            "implementation_source_sha256": source_hash,
        },
        "inner_universe_scope_receipt": inner_scope,
        "passed": passed,
    }


def audit_score_free_fold_candidate(
    *,
    candidate_fold_bytes: bytes,
    authority_payload: Mapping[str, object],
    scope: str,
    expected_primary_source_sha256: str,
    expected_outer_fold_id: str | None = None,
    expected_outer_training_pair_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    """Compare all fold-level score-free statistics field by field."""
    candidate = _strict_object(candidate_fold_bytes, "candidate fold audit")
    expected = reconstruct_score_free_fold_payload(
        authority_payload=authority_payload,
        candidate_payload=candidate,
        scope=scope,
        expected_primary_source_sha256=expected_primary_source_sha256,
        expected_outer_fold_id=expected_outer_fold_id,
        expected_outer_training_pair_ids=expected_outer_training_pair_ids,
    )
    difference = _first_difference(expected, candidate)
    if difference is not None:
        raise IndependentScoreFreeAuditError(
            f"independent fold disagreement at {difference}"
        )
    receipt: dict[str, object] = {
        "format_version": "score_free_independent_fold_audit_v1",
        "scope": scope,
        "fold_id": candidate["fold_id"],
        "candidate_fold_sha256": hashlib.sha256(candidate_fold_bytes).hexdigest(),
        "independent_fold_sha256": _compact_json_sha256(expected),
        "independent_auditor_source_sha256": implementation_source_sha256(),
        "field_by_field_agreement": True,
        "candidate_gate_passed": candidate["passed"],
        "training_authorized": False,
        "passed": True,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    return receipt


def _relative_range(values: Sequence[int | Fraction]) -> Fraction:
    exact = tuple(Fraction(value) for value in values)
    if not exact or any(value < 0 for value in exact):
        raise IndependentScoreFreeAuditError("relative-range vector is invalid")
    total = sum(exact, Fraction(0))
    if total == 0:
        return Fraction(0)
    return Fraction(len(exact)) * (max(exact) - min(exact)) / total


def reconstruct_cross_fold_payload(
    *,
    fold_payloads: Sequence[Mapping[str, object]],
    scope: str,
    expected_primary_source_sha256: str,
) -> dict[str, object]:
    """Recompute one three-fold balance payload from independently audited folds."""
    if scope not in CROSS_FOLD_THRESHOLDS:
        raise IndependentScoreFreeAuditError("cross-fold scope differs")
    if len(fold_payloads) != 3:
        raise IndependentScoreFreeAuditError("cross-fold audit requires three folds")
    folds = sorted(fold_payloads, key=lambda payload: str(payload["fold_id"]))
    if len({str(payload["fold_id"]) for payload in folds}) != 3:
        raise IndependentScoreFreeAuditError("cross-fold IDs are not unique")
    if len({tuple(payload["universe_pair_ids"]) for payload in folds}) != 1:
        raise IndependentScoreFreeAuditError("cross-fold universes differ")
    panels = [payload["panel"] for payload in folds]
    metrics: dict[str, Sequence[int | Fraction]] = {
        "validation_pair_count": [panel["validation_pair_count"] for panel in panels],
        "validation_antibody_count": [
            panel["validation_antibody_count"] for panel in panels
        ],
        "validation_antigen_count": [
            panel["validation_antigen_count"] for panel in panels
        ],
        "training_pair_retention": [
            Fraction(panel["training_pair_count"], panel["universe_pair_count"])
            for panel in panels
        ],
        "quarantine_pair_count": [panel["quarantine_pair_count"] for panel in panels],
        "antibody_singleton_fraction": [
            _fraction(panel["antibody_singleton_fraction"]) for panel in panels
        ],
        "antigen_singleton_fraction": [
            _fraction(panel["antigen_singleton_fraction"]) for panel in panels
        ],
        "bipartite_density": [
            _fraction(panel["bipartite_density"]) for panel in panels
        ],
        "bidirectional_chance_mrr": [
            _fraction(panel["bidirectional_chance_mrr"]) for panel in panels
        ],
    }
    order = (
        "validation_pair_count",
        "validation_antibody_count",
        "validation_antigen_count",
        "training_pair_retention",
        "quarantine_pair_count",
        "antibody_singleton_fraction",
        "antigen_singleton_fraction",
        "bipartite_density",
        "bidirectional_chance_mrr",
    )
    ranges = {name: _relative_range(metrics[name]) for name in order}
    thresholds = CROSS_FOLD_THRESHOLDS[scope]
    failures = []
    for name in order:
        maximum = _fraction(
            thresholds[
                "maximum_validation_pair_relative_range"
                if name == "validation_pair_count"
                else "maximum_other_relative_range"
            ]
        )
        if ranges[name] > maximum:
            failures.append(f"{name}_relative_range_exceeds_locked_threshold")
    return {
        "schema_version": "score_free_cross_fold_balance_v2",
        "thresholds": thresholds,
        "relative_ranges": {
            name: _fraction_payload(ranges[name]) for name in order
        },
        "failures": failures,
        "input_bindings": {
            "fold_audit_sha256": [
                _compact_json_sha256(payload) for payload in folds
            ],
            "implementation_source_sha256": expected_primary_source_sha256,
        },
        "passed": not failures,
    }


def audit_cross_fold_candidate(
    *,
    candidate_bytes: bytes,
    fold_payloads: Sequence[Mapping[str, object]],
    scope: str,
    expected_primary_source_sha256: str,
) -> dict[str, object]:
    """Independently compare every exact cross-fold balance field."""
    candidate = _strict_object(candidate_bytes, "candidate cross-fold audit")
    expected = reconstruct_cross_fold_payload(
        fold_payloads=fold_payloads,
        scope=scope,
        expected_primary_source_sha256=expected_primary_source_sha256,
    )
    difference = _first_difference(expected, candidate)
    if difference is not None:
        raise IndependentScoreFreeAuditError(
            f"independent cross-fold disagreement at {difference}"
        )
    return {
        "format_version": "score_free_independent_cross_fold_audit_v1",
        "scope": scope,
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "independent_payload_sha256": _compact_json_sha256(expected),
        "field_by_field_agreement": True,
        "candidate_gate_passed": candidate["passed"],
        "training_authorized": False,
        "passed": True,
    }


def reconstruct_pooled_outer_payload(
    *,
    authority_payload: Mapping[str, object],
    fold_payloads: Sequence[Mapping[str, object]],
    expected_primary_source_sha256: str,
) -> dict[str, object]:
    """Recompute pooled entity-disjoint outer representativeness."""
    if len(fold_payloads) != 3:
        raise IndependentScoreFreeAuditError("pooled outer requires three folds")
    folds = sorted(fold_payloads, key=lambda payload: str(payload["fold_id"]))
    if len({tuple(payload["universe_pair_ids"]) for payload in folds}) != 1:
        raise IndependentScoreFreeAuditError("pooled outer universes differ")
    full_antibody_ids = {
        str(row["entity_id"]) for row in authority_payload["antibodies"]
    }
    full_antigen_ids = {
        str(row["entity_id"]) for row in authority_payload["antigens"]
    }
    full_antibodies = _profiles(
        authority_payload, "antibody", full_antibody_ids
    )
    full_antigens = _profiles(authority_payload, "antigen", full_antigen_ids)
    pooled_antibodies: set[str] = set()
    pooled_antigens: set[str] = set()
    for fold in folds:
        current_antibodies = set(fold["validation_antibody_ids"])
        current_antigens = set(fold["validation_antigen_ids"])
        if pooled_antibodies & current_antibodies:
            raise IndependentScoreFreeAuditError(
                "pooled outer antibody endpoints are not disjoint"
            )
        if pooled_antigens & current_antigens:
            raise IndependentScoreFreeAuditError(
                "pooled outer antigen endpoints are not disjoint"
            )
        pooled_antibodies.update(current_antibodies)
        pooled_antigens.update(current_antigens)
    thresholds = DISTRIBUTION_THRESHOLDS["pooled_outer"]
    antibody_comparison = _representativeness(
        left_ids=pooled_antibodies,
        right_ids=full_antibody_ids,
        profiles=full_antibodies,
        axis="antibody",
        reference_name="full_authority_pool",
        thresholds=thresholds,
    )
    antigen_comparison = _representativeness(
        left_ids=pooled_antigens,
        right_ids=full_antigen_ids,
        profiles=full_antigens,
        axis="antigen",
        reference_name="full_authority_pool",
        thresholds=thresholds,
    )
    return {
        "schema_version": "score_free_pooled_outer_audit_v2",
        "thresholds": thresholds,
        "antibody_vs_universe": antibody_comparison,
        "antigen_vs_universe": antigen_comparison,
        "pooled_antibody_ids": sorted(pooled_antibodies),
        "pooled_antigen_ids": sorted(pooled_antigens),
        "input_bindings": {
            "fold_audit_sha256": [
                _compact_json_sha256(payload) for payload in folds
            ],
            "antibody_profile_sha256": _profile_digest(full_antibodies),
            "antigen_profile_sha256": _profile_digest(full_antigens),
            "implementation_source_sha256": expected_primary_source_sha256,
        },
        "passed": antibody_comparison["passed"] and antigen_comparison["passed"],
    }


def audit_pooled_outer_candidate(
    *,
    candidate_bytes: bytes,
    authority_payload: Mapping[str, object],
    fold_payloads: Sequence[Mapping[str, object]],
    expected_primary_source_sha256: str,
) -> dict[str, object]:
    """Independently compare every pooled-outer score-free field."""
    candidate = _strict_object(candidate_bytes, "candidate pooled outer audit")
    expected = reconstruct_pooled_outer_payload(
        authority_payload=authority_payload,
        fold_payloads=fold_payloads,
        expected_primary_source_sha256=expected_primary_source_sha256,
    )
    difference = _first_difference(expected, candidate)
    if difference is not None:
        raise IndependentScoreFreeAuditError(
            f"independent pooled disagreement at {difference}"
        )
    return {
        "format_version": "score_free_independent_pooled_outer_audit_v1",
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "independent_payload_sha256": _compact_json_sha256(expected),
        "field_by_field_agreement": True,
        "candidate_gate_passed": candidate["passed"],
        "training_authorized": False,
        "passed": True,
    }


def _first_difference(expected: object, observed: object, path: str = "$") -> str | None:
    if type(expected) is not type(observed):
        return f"{path}: type {type(expected).__name__} != {type(observed).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        observed_keys = set(observed)
        if expected_keys != observed_keys:
            return f"{path}: keys {sorted(expected_keys ^ observed_keys)} differ"
        for key in sorted(expected):
            difference = _first_difference(expected[key], observed[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path}: length {len(expected)} != {len(observed)}"
        for index, (left, right) in enumerate(zip(expected, observed)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != observed:
        return f"{path}: {expected!r} != {observed!r}"
    return None


def audit_score_free_authority_candidate(
    *,
    candidate_bytes: bytes,
    authority_bytes: bytes,
    manifest_bytes: bytes,
    edge_ledger_bytes: Mapping[str, bytes],
    expected_primary_implementation_sha256: Mapping[str, str],
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
) -> dict[str, object]:
    """Compare an independently rebuilt authority payload field by field."""
    candidate = _strict_object(candidate_bytes, "candidate authority bundle")
    independent = reconstruct_score_free_authority(
        authority_bytes=authority_bytes,
        manifest_bytes=manifest_bytes,
        edge_ledger_bytes=edge_ledger_bytes,
        expected_primary_implementation_sha256=(
            expected_primary_implementation_sha256
        ),
        expected_counts=expected_counts,
    )
    allowed = set(independent) | {"verified_file_receipts", "publication_sha256"}
    if set(candidate) - allowed:
        raise IndependentScoreFreeAuditError(
            f"candidate has unknown fields: {sorted(set(candidate) - allowed)}"
        )
    missing = set(independent) - set(candidate)
    if missing:
        raise IndependentScoreFreeAuditError(
            f"candidate lacks scientific fields: {sorted(missing)}"
        )
    candidate_scientific = {key: candidate[key] for key in independent}
    difference = _first_difference(independent, candidate_scientific)
    if difference is not None:
        raise IndependentScoreFreeAuditError(
            f"independent authority disagreement at {difference}"
        )
    if "publication_sha256" in candidate:
        without_publication = dict(candidate)
        claimed_publication = without_publication.pop("publication_sha256")
        expected_publication = hashlib.sha256(
            canonical_json_bytes(without_publication)
        ).hexdigest()
        if claimed_publication != expected_publication:
            raise IndependentScoreFreeAuditError("publication SHA-256 disagrees")
    receipt: dict[str, object] = {
        "format_version": AUDIT_FORMAT,
        "candidate_file_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_scientific_payload_sha256": candidate[
            "scientific_payload_sha256"
        ],
        "independent_scientific_payload_sha256": independent[
            "scientific_payload_sha256"
        ],
        "independent_auditor_source_sha256": implementation_source_sha256(),
        "raw_input_sha256": independent["input_sha256"],
        "field_by_field_agreement": True,
        "forbidden_outcome_or_model_fields_observed": False,
        "training_authorized": False,
        "passed": True,
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    return receipt


def authority_audit_markdown(receipt: Mapping[str, object]) -> str:
    """Render Markdown only from one machine receipt payload."""
    if receipt.get("format_version") != AUDIT_FORMAT or receipt.get("passed") is not True:
        raise IndependentScoreFreeAuditError("cannot render an invalid receipt")
    lines = [
        "# V6-A Independent Score-Free Authority Audit",
        "",
        f"- Passed: `{str(receipt['passed']).lower()}`",
        f"- Candidate SHA-256: `{receipt['candidate_file_sha256']}`",
        f"- Scientific payload SHA-256: `{receipt['independent_scientific_payload_sha256']}`",
        f"- Auditor source SHA-256: `{receipt['independent_auditor_source_sha256']}`",
        "- Field-by-field agreement: `true`",
        "- Training authorized: `false`",
        "",
    ]
    return "\n".join(lines)
