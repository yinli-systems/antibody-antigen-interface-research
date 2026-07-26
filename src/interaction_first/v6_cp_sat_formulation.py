"""Independent raw-authority CP-SAT formulation for the V6-A split.

This module intentionally does not import the historical MILP builder, the
solver-neutral projection adapter, or the HiGHS backend. It authenticates the
score-free authority payload and direct-edge ledgers, reconstructs graph
semantics, and emits CP-SAT constraints directly from canonical identifiers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DIRECT_FORMULATION_FORMAT = "v6_a_independent_cp_sat_formulation_v1"
DIRECT_PROJECTION_RECEIPT_FORMAT = "v6_a_independent_cp_sat_projection_receipt_v1"
AUTHORITY_FORMAT = "score_free_authority_v2"
EDGE_FORMAT = "score_free_direct_entity_edges_v1"
VIEWS = ("vh", "vl", "concatenated_antibody", "antigen")
LOCKED_RANDOM_SEED = 2026072701


class DirectCPSatError(ValueError):
    """Raised when direct authority reconstruction or CP-SAT setup fails closed."""


VariableKey = tuple[str, ...]


def _canonical_json_bytes(payload: object) -> bytes:
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


def _sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def implementation_source_sha256() -> str:
    """Hash the exact independent formulation source bytes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DirectCPSatError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.isascii()
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise DirectCPSatError(f"{label} must be exact non-empty ASCII")
    return value


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DirectCPSatError(f"{label} must be an integer")
    if value < int(positive):
        qualifier = "positive" if positive else "non-negative"
        raise DirectCPSatError(f"{label} must be {qualifier}")
    return value


def _strict_object(payload_bytes: bytes, label: str) -> dict[str, Any]:
    if not payload_bytes or payload_bytes.startswith(b"\xef\xbb\xbf"):
        raise DirectCPSatError(f"{label} is empty or contains a BOM")
    try:
        text = payload_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise DirectCPSatError(f"{label} must be ASCII JSON") from error

    def reject_duplicates(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise DirectCPSatError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        result = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DirectCPSatError(f"{label} contains non-finite {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise DirectCPSatError(f"{label} is not strict JSON") from error
    if not isinstance(result, dict):
        raise DirectCPSatError(f"{label} must be a JSON object")
    return result


@dataclass(frozen=True)
class DirectPair:
    """One canonical positive pair used by the independent formulation."""

    pair_id: str
    antibody_id: str
    antigen_id: str

    def __post_init__(self) -> None:
        _identifier(self.pair_id, "pair identifier")
        _identifier(self.antibody_id, "antibody identifier")
        _identifier(self.antigen_id, "antigen identifier")


@dataclass(frozen=True)
class DirectAuthorityProblem:
    """Authenticated identifier, incidence, graph, and component semantics."""

    authority_payload_sha256: str
    pairs: tuple[DirectPair, ...]
    antibody_neighbors: tuple[tuple[str, tuple[str, ...]], ...]
    antigen_neighbors: tuple[tuple[str, tuple[str, ...]], ...]
    antibody_components: tuple[tuple[str, ...], ...]
    antigen_components: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        _digest(self.authority_payload_sha256, "authority payload")
        if not self.pairs or self.pairs != tuple(
            sorted(self.pairs, key=lambda pair: pair.pair_id)
        ):
            raise DirectCPSatError("pairs must be non-empty and canonically sorted")
        if len({pair.pair_id for pair in self.pairs}) != len(self.pairs):
            raise DirectCPSatError("pair identifiers are not unique")
        for label, rows in (
            ("antibody", self.antibody_neighbors),
            ("antigen", self.antigen_neighbors),
        ):
            keys = [key for key, _ in rows]
            if keys != sorted(set(keys)):
                raise DirectCPSatError(f"{label} graph keys are not canonical")
            universe = set(keys)
            graph = dict(rows)
            for entity, neighbors in rows:
                if neighbors != tuple(sorted(set(neighbors))):
                    raise DirectCPSatError(f"{label} neighbors are not canonical")
                if entity in neighbors or set(neighbors) - universe:
                    raise DirectCPSatError(f"{label} graph contains an invalid edge")
                if any(entity not in graph[neighbor] for neighbor in neighbors):
                    raise DirectCPSatError(f"{label} graph is not symmetric")
        for label, components, rows in (
            ("antibody", self.antibody_components, self.antibody_neighbors),
            ("antigen", self.antigen_components, self.antigen_neighbors),
        ):
            flattened = tuple(member for component in components for member in component)
            if any(component != tuple(sorted(set(component))) for component in components):
                raise DirectCPSatError(f"{label} components are not canonical")
            if components != tuple(sorted(set(components))):
                raise DirectCPSatError(f"{label} components are not sorted and unique")
            if tuple(sorted(flattened)) != tuple(key for key, _ in rows):
                raise DirectCPSatError(f"{label} components do not partition entities")


@dataclass(frozen=True)
class DirectLevelPolicy:
    """Exact integer and rational gates for one independent three-fold level."""

    minimum_validation_pairs: int
    minimum_validation_endpoints_per_axis: int
    minimum_validation_components_per_axis: int
    validation_pair_lower_fraction: Fraction
    validation_pair_upper_fraction: Fraction
    minimum_training_pair_fraction: Fraction
    minimum_training_endpoint_fraction: Fraction
    maximum_quarantine_pair_fraction: Fraction
    maximum_validation_pair_relative_range: Fraction
    maximum_other_count_relative_range: Fraction

    def __post_init__(self) -> None:
        for name in (
            "minimum_validation_pairs",
            "minimum_validation_endpoints_per_axis",
            "minimum_validation_components_per_axis",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DirectCPSatError(f"{name} must be a non-negative integer")
        for name in (
            "validation_pair_lower_fraction",
            "validation_pair_upper_fraction",
            "minimum_training_pair_fraction",
            "minimum_training_endpoint_fraction",
            "maximum_quarantine_pair_fraction",
            "maximum_validation_pair_relative_range",
            "maximum_other_count_relative_range",
        ):
            value = getattr(self, name)
            if not isinstance(value, Fraction) or not 0 <= value <= 1:
                raise DirectCPSatError(f"{name} must be an exact fraction in [0, 1]")
        if self.validation_pair_lower_fraction > self.validation_pair_upper_fraction:
            raise DirectCPSatError("validation lower fraction exceeds upper fraction")


LOCKED_OUTER_POLICY = DirectLevelPolicy(
    120,
    60,
    25,
    Fraction(7, 100),
    Fraction(9, 100),
    Fraction(3, 5),
    Fraction(3, 5),
    Fraction(33, 100),
    Fraction(3, 20),
    Fraction(1, 5),
)
LOCKED_INNER_POLICY = DirectLevelPolicy(
    80,
    40,
    20,
    Fraction(7, 100),
    Fraction(9, 100),
    Fraction(11, 20),
    Fraction(11, 20),
    Fraction(19, 50),
    Fraction(1, 5),
    Fraction(1, 4),
)


def _edge_ledger(
    payload_bytes: bytes,
    view: str,
    allowed_entities: set[str],
) -> tuple[tuple[str, str], ...]:
    payload = _strict_object(payload_bytes, f"{view} edge ledger")
    if set(payload) != {"format_version", "view", "edges"}:
        raise DirectCPSatError(f"{view} edge-ledger schema differs")
    if payload["format_version"] != EDGE_FORMAT or payload["view"] != view:
        raise DirectCPSatError(f"{view} edge-ledger binding differs")
    if not isinstance(payload["edges"], list):
        raise DirectCPSatError(f"{view} edge list is malformed")
    edges = []
    for item in payload["edges"]:
        if not isinstance(item, list) or len(item) != 2:
            raise DirectCPSatError(f"{view} edge shape differs")
        left = _identifier(item[0], f"{view} edge left")
        right = _identifier(item[1], f"{view} edge right")
        if left >= right or {left, right} - allowed_entities:
            raise DirectCPSatError(f"{view} edge is noncanonical")
        edges.append((left, right))
    result = tuple(edges)
    if result != tuple(sorted(set(result))):
        raise DirectCPSatError(f"{view} edges are unsorted or repeated")
    return result


def _graph(
    entities: Iterable[str],
    edges: Iterable[tuple[str, str]],
) -> dict[str, set[str]]:
    result = {entity: set() for entity in entities}
    for left, right in edges:
        result[left].add(right)
        result[right].add(left)
    return result


def _components(graph: Mapping[str, set[str]]) -> tuple[tuple[str, ...], ...]:
    remaining = set(graph)
    result = []
    for root in sorted(graph):
        if root not in remaining:
            continue
        remaining.remove(root)
        pending = [root]
        members = []
        while pending:
            entity = pending.pop()
            members.append(entity)
            reached = sorted(remaining.intersection(graph[entity]), reverse=True)
            for neighbor in reached:
                remaining.remove(neighbor)
                pending.append(neighbor)
        result.append(tuple(sorted(members)))
    return tuple(sorted(result))


def direct_problem_from_authority_payload(
    authority_payload_bytes: bytes,
    edge_ledger_bytes: Mapping[str, bytes],
    *,
    expected_authority_payload_sha256: str,
    expected_counts: Mapping[str, int],
) -> DirectAuthorityProblem:
    """Authenticate one canonical authority payload and rebuild graph semantics."""
    expected_digest = _digest(
        expected_authority_payload_sha256,
        "expected authority payload",
    )
    if hashlib.sha256(authority_payload_bytes).hexdigest() != expected_digest:
        raise DirectCPSatError("authority payload byte hash differs from anchor")
    authority = _strict_object(authority_payload_bytes, "score-free authority")
    required_top_level = {
        "format_version",
        "input_sha256",
        "authority_membership_sha256",
        "implementation_source_sha256",
        "counts",
        "antibodies",
        "antigens",
        "pairs",
        "quantile_schemes",
        "forbidden_outcome_or_model_fields_accepted",
        "scientific_payload_sha256",
    }
    if set(authority) != required_top_level:
        raise DirectCPSatError("authority top-level schema differs")
    if authority["format_version"] != AUTHORITY_FORMAT:
        raise DirectCPSatError("authority format differs")
    if authority["forbidden_outcome_or_model_fields_accepted"] is not False:
        raise DirectCPSatError("authority admits forbidden outcome or model fields")
    scientific_digest = authority.pop("scientific_payload_sha256")
    if _sha256(authority) != scientific_digest:
        raise DirectCPSatError("authority scientific payload digest differs")
    authority["scientific_payload_sha256"] = scientific_digest
    if authority["counts"] != dict(expected_counts):
        raise DirectCPSatError("authority inventory differs from the external anchor")
    if set(expected_counts) != {
        "source_rows",
        "exact_pairs",
        "antibody_entities",
        "antigen_entities",
    }:
        raise DirectCPSatError("external count anchor schema differs")
    input_hashes = authority["input_sha256"]
    expected_input_hash_keys = {
        "authority_allowlist",
        "canonical_manifest",
        *(f"edge_ledger_{view}" for view in VIEWS),
    }
    if not isinstance(input_hashes, dict) or set(input_hashes) != expected_input_hash_keys:
        raise DirectCPSatError("authority input hashes are malformed")
    for label, value in input_hashes.items():
        _digest(value, f"authority input {label}")
    implementation_hashes = authority["implementation_source_sha256"]
    if not isinstance(implementation_hashes, dict) or set(implementation_hashes) != {
        "score_free_authority",
        "score_free_split_statistics",
    }:
        raise DirectCPSatError("authority implementation binding schema differs")
    for label, value in implementation_hashes.items():
        _digest(value, f"authority implementation {label}")
    if set(edge_ledger_bytes) != set(VIEWS):
        raise DirectCPSatError("exactly four edge ledgers are required")
    for view in VIEWS:
        key = f"edge_ledger_{view}"
        if input_hashes.get(key) != hashlib.sha256(edge_ledger_bytes[view]).hexdigest():
            raise DirectCPSatError(f"{view} edge ledger differs from authority binding")

    antibody_rows = authority["antibodies"]
    antigen_rows = authority["antigens"]
    pair_rows = authority["pairs"]
    if not all(isinstance(rows, list) for rows in (antibody_rows, antigen_rows, pair_rows)):
        raise DirectCPSatError("authority entity or pair rows are malformed")
    antibody_fields = {
        "entity_id",
        "component_id",
        "vh_length",
        "vl_length",
        "concatenated_antibody_length",
        "vh_direct_degree",
        "vl_direct_degree",
        "concatenated_direct_degree",
        "union_direct_degree",
        "pair_incidence_degree",
        "component_size",
    }
    antigen_fields = {
        "entity_id",
        "component_id",
        "antigen_length",
        "antigen_direct_degree",
        "pair_incidence_degree",
        "component_size",
    }
    pair_fields = {
        "pair_id",
        "antibody_id",
        "antigen_id",
        "source_instance_ids",
    }
    if any(not isinstance(row, dict) or set(row) != antibody_fields for row in antibody_rows):
        raise DirectCPSatError("authority antibody row schema differs")
    if any(not isinstance(row, dict) or set(row) != antigen_fields for row in antigen_rows):
        raise DirectCPSatError("authority antigen row schema differs")
    if any(not isinstance(row, dict) or set(row) != pair_fields for row in pair_rows):
        raise DirectCPSatError("authority pair row schema differs")
    quantile_rows = authority["quantile_schemes"]
    quantile_fields = {
        "axis",
        "feature_name",
        "cuts_decimal",
        "optional_missing_bin",
        "categorical_binary",
        "informative",
    }
    if not isinstance(quantile_rows, list) or any(
        not isinstance(row, dict) or set(row) != quantile_fields
        for row in quantile_rows
    ):
        raise DirectCPSatError("authority quantile row schema differs")
    antibodies = tuple(_identifier(row["entity_id"], "antibody") for row in antibody_rows)
    antigens = tuple(_identifier(row["entity_id"], "antigen") for row in antigen_rows)
    if antibodies != tuple(sorted(set(antibodies))):
        raise DirectCPSatError("authority antibodies are not canonical")
    if antigens != tuple(sorted(set(antigens))):
        raise DirectCPSatError("authority antigens are not canonical")
    for row in antibody_rows:
        _identifier(row["component_id"], "antibody component")
        vh_length = _integer(row["vh_length"], "VH length", positive=True)
        vl_length = row["vl_length"]
        vl_degree = row["vl_direct_degree"]
        if vl_length is None:
            if vl_degree is not None:
                raise DirectCPSatError("missing VL has a direct degree")
            normalized_vl_length = 0
        else:
            normalized_vl_length = _integer(vl_length, "VL length", positive=True)
            _integer(vl_degree, "VL direct degree")
        if row["concatenated_antibody_length"] != vh_length + normalized_vl_length:
            raise DirectCPSatError("concatenated antibody length differs")
        for name in (
            "vh_direct_degree",
            "concatenated_direct_degree",
            "union_direct_degree",
        ):
            _integer(row[name], name)
        _integer(row["pair_incidence_degree"], "antibody incidence", positive=True)
        _integer(row["component_size"], "antibody component size", positive=True)
    for row in antigen_rows:
        _identifier(row["component_id"], "antigen component")
        _integer(row["antigen_length"], "antigen length", positive=True)
        _integer(row["antigen_direct_degree"], "antigen direct degree")
        _integer(row["pair_incidence_degree"], "antigen incidence", positive=True)
        _integer(row["component_size"], "antigen component size", positive=True)
    pairs = tuple(
        DirectPair(
            _identifier(row["pair_id"], "pair"),
            _identifier(row["antibody_id"], "pair antibody"),
            _identifier(row["antigen_id"], "pair antigen"),
        )
        for row in pair_rows
    )
    if pairs != tuple(sorted(pairs, key=lambda pair: pair.pair_id)):
        raise DirectCPSatError("authority pairs are not canonical")
    if any(pair.antibody_id not in antibodies or pair.antigen_id not in antigens for pair in pairs):
        raise DirectCPSatError("authority pair references an unknown endpoint")
    source_instances = []
    for pair, row in zip(pairs, pair_rows, strict=True):
        raw_sources = row["source_instance_ids"]
        if not isinstance(raw_sources, list) or not raw_sources:
            raise DirectCPSatError("pair source-instance membership is malformed")
        sources = tuple(
            _identifier(source, "pair source instance") for source in raw_sources
        )
        if sources != tuple(sorted(set(sources))):
            raise DirectCPSatError("pair source-instance membership is noncanonical")
        source_instances.extend(sources)
        material = (
            b"pair-v1\0"
            + pair.antibody_id.encode("ascii")
            + b"\0"
            + pair.antigen_id.encode("ascii")
        )
        if pair.pair_id != hashlib.sha256(material).hexdigest():
            raise DirectCPSatError("pair identifier differs from canonical endpoints")
    if len(source_instances) != len(set(source_instances)):
        raise DirectCPSatError("source instance occurs in more than one canonical pair")
    observed_counts = {
        "source_rows": len(source_instances),
        "exact_pairs": len(pairs),
        "antibody_entities": len(antibodies),
        "antigen_entities": len(antigens),
    }
    if observed_counts != dict(expected_counts):
        raise DirectCPSatError("authority rows do not reproduce external counts")

    ledgers = {
        view: _edge_ledger(
            edge_ledger_bytes[view],
            view,
            set(antigens) if view == "antigen" else set(antibodies),
        )
        for view in VIEWS
    }
    union_edges = set().union(
        ledgers["vh"],
        ledgers["vl"],
        ledgers["concatenated_antibody"],
    )
    antibody_graph = _graph(antibodies, union_edges)
    antigen_graph = _graph(antigens, ledgers["antigen"])
    antibody_view_graphs = {
        view: _graph(antibodies, ledgers[view])
        for view in ("vh", "vl", "concatenated_antibody")
    }
    antibody_by_id = {str(row["entity_id"]): row for row in antibody_rows}
    for left, right in ledgers["vl"]:
        if antibody_by_id[left]["vl_length"] is None or antibody_by_id[right]["vl_length"] is None:
            raise DirectCPSatError("VL edge references a missing chain")
    antibody_components = _components(antibody_graph)
    antigen_components = _components(antigen_graph)
    antibody_component_by_id = {
        member: component[0]
        for component in antibody_components
        for member in component
    }
    antigen_component_by_id = {
        member: component[0]
        for component in antigen_components
        for member in component
    }
    antibody_component_sizes = {
        member: len(component)
        for component in antibody_components
        for member in component
    }
    antigen_component_sizes = {
        member: len(component)
        for component in antigen_components
        for member in component
    }
    antibody_incidence = {entity: 0 for entity in antibodies}
    antigen_incidence = {entity: 0 for entity in antigens}
    for pair in pairs:
        antibody_incidence[pair.antibody_id] += 1
        antigen_incidence[pair.antigen_id] += 1
    for row in antibody_rows:
        entity = str(row["entity_id"])
        if row.get("component_id") != antibody_component_by_id[entity]:
            raise DirectCPSatError("antibody component identity differs")
        if row.get("union_direct_degree") != len(antibody_graph[entity]):
            raise DirectCPSatError("antibody union degree differs")
        expected_view_degrees = {
            "vh_direct_degree": len(antibody_view_graphs["vh"][entity]),
            "vl_direct_degree": (
                None
                if row.get("vl_length") is None
                else len(antibody_view_graphs["vl"][entity])
            ),
            "concatenated_direct_degree": len(
                antibody_view_graphs["concatenated_antibody"][entity]
            ),
        }
        if any(row.get(name) != value for name, value in expected_view_degrees.items()):
            raise DirectCPSatError("antibody view degree differs")
        if row.get("pair_incidence_degree") != antibody_incidence[entity]:
            raise DirectCPSatError("antibody pair-incidence degree differs")
        if row.get("component_size") != antibody_component_sizes[entity]:
            raise DirectCPSatError("antibody component size differs")
    for row in antigen_rows:
        entity = str(row["entity_id"])
        if row.get("component_id") != antigen_component_by_id[entity]:
            raise DirectCPSatError("antigen component identity differs")
        if row.get("antigen_direct_degree") != len(antigen_graph[entity]):
            raise DirectCPSatError("antigen direct degree differs")
        if row.get("pair_incidence_degree") != antigen_incidence[entity]:
            raise DirectCPSatError("antigen pair-incidence degree differs")
        if row.get("component_size") != antigen_component_sizes[entity]:
            raise DirectCPSatError("antigen component size differs")
    return DirectAuthorityProblem(
        authority_payload_sha256=expected_digest,
        pairs=pairs,
        antibody_neighbors=tuple(
            (entity, tuple(sorted(neighbors)))
            for entity, neighbors in sorted(antibody_graph.items())
        ),
        antigen_neighbors=tuple(
            (entity, tuple(sorted(neighbors)))
            for entity, neighbors in sorted(antigen_graph.items())
        ),
        antibody_components=antibody_components,
        antigen_components=antigen_components,
    )


class _DirectBuilder:
    """Direct CP-SAT variable and constraint-family builder."""

    def __init__(self, cp_model: object) -> None:
        self.cp_model = cp_model
        self.model = cp_model.CpModel()
        self.variables: dict[VariableKey, object] = {}
        self.constraint_family_counts: dict[str, int] = {}

    def variable(self, *parts: object) -> object:
        key = tuple(str(part) for part in parts)
        if key not in self.variables:
            self.variables[key] = self.model.new_bool_var(
                f"v_{len(self.variables):09d}"
            )
        return self.variables[key]

    def add(self, expression: object, family: str) -> None:
        self.model.add(expression)
        self.constraint_family_counts[family] = (
            self.constraint_family_counts.get(family, 0) + 1
        )

    def exact_or(self, output: object, inputs: Sequence[object], family: str) -> None:
        for input_variable in inputs:
            self.add(output >= input_variable, family)
        self.add(output <= sum(inputs), family)

    def exact_and(self, output: object, inputs: Sequence[object], family: str) -> None:
        for input_variable in inputs:
            self.add(output <= input_variable, family)
        self.add(output >= sum(inputs) - (len(inputs) - 1), family)

    def endpoint_presence(
        self,
        output: object,
        incident: Sequence[object],
        family: str,
    ) -> None:
        for pair_variable in incident:
            self.add(pair_variable <= output, family)
        self.add(output <= sum(incident), family)


@dataclass(frozen=True)
class DirectCPModel:
    """One directly built CP-SAT model and its canonical variable registry."""

    model: object
    variables: Mapping[VariableKey, object]
    authority_payload_sha256: str
    constraint_family_counts: tuple[tuple[str, int], ...]
    formulation_sha256: str


def _add_balance(
    builder: _DirectBuilder,
    folds: Sequence[Sequence[object]],
    maximum_relative_range: Fraction,
    family: str,
) -> None:
    denominator = maximum_relative_range.denominator
    numerator = maximum_relative_range.numerator
    fold_count = len(folds)
    for left in range(fold_count):
        for right in range(left + 1, fold_count):
            totals = [sum(values) for values in folds]
            grand_total = sum(totals)
            builder.add(
                fold_count * denominator * (totals[left] - totals[right])
                <= numerator * grand_total,
                family,
            )
            builder.add(
                fold_count * denominator * (totals[right] - totals[left])
                <= numerator * grand_total,
                family,
            )


def _add_level_policy(
    builder: _DirectBuilder,
    policy: DirectLevelPolicy,
    validation_pairs: Sequence[Sequence[object]],
    validation_antibodies: Sequence[Sequence[object]],
    validation_antigens: Sequence[Sequence[object]],
    validation_antibody_components: Sequence[Sequence[object]],
    validation_antigen_components: Sequence[Sequence[object]],
    training_pairs: Sequence[Sequence[object]],
    training_antibodies: Sequence[Sequence[object]],
    training_antigens: Sequence[Sequence[object]],
    quarantine_pairs: Sequence[Sequence[object]],
    universe_pairs: Sequence[object] | int,
    universe_antibodies: Sequence[object] | int,
    universe_antigens: Sequence[object] | int,
    family_prefix: str,
) -> None:
    for fold_index in range(3):
        validation_pair_count = sum(validation_pairs[fold_index])
        pair_universe_count = (
            universe_pairs if isinstance(universe_pairs, int) else sum(universe_pairs)
        )
        antibody_universe_count = (
            universe_antibodies
            if isinstance(universe_antibodies, int)
            else sum(universe_antibodies)
        )
        antigen_universe_count = (
            universe_antigens
            if isinstance(universe_antigens, int)
            else sum(universe_antigens)
        )
        family = f"{family_prefix}_shape"
        builder.add(validation_pair_count >= policy.minimum_validation_pairs, family)
        ratio = policy.validation_pair_lower_fraction
        builder.add(
            ratio.denominator * validation_pair_count
            >= ratio.numerator * pair_universe_count,
            family,
        )
        ratio = policy.validation_pair_upper_fraction
        builder.add(
            ratio.denominator * validation_pair_count
            <= ratio.numerator * pair_universe_count,
            family,
        )
        builder.add(
            sum(validation_antibodies[fold_index])
            >= policy.minimum_validation_endpoints_per_axis,
            family,
        )
        builder.add(
            sum(validation_antigens[fold_index])
            >= policy.minimum_validation_endpoints_per_axis,
            family,
        )
        builder.add(
            sum(validation_antibody_components[fold_index])
            >= policy.minimum_validation_components_per_axis,
            family,
        )
        builder.add(
            sum(validation_antigen_components[fold_index])
            >= policy.minimum_validation_components_per_axis,
            family,
        )
        ratio = policy.minimum_training_pair_fraction
        builder.add(
            ratio.denominator * sum(training_pairs[fold_index])
            >= ratio.numerator * pair_universe_count,
            family,
        )
        ratio = policy.minimum_training_endpoint_fraction
        builder.add(
            ratio.denominator * sum(training_antibodies[fold_index])
            >= ratio.numerator * antibody_universe_count,
            family,
        )
        builder.add(
            ratio.denominator * sum(training_antigens[fold_index])
            >= ratio.numerator * antigen_universe_count,
            family,
        )
        ratio = policy.maximum_quarantine_pair_fraction
        builder.add(
            ratio.denominator * sum(quarantine_pairs[fold_index])
            <= ratio.numerator * pair_universe_count,
            family,
        )
    _add_balance(
        builder,
        validation_pairs,
        policy.maximum_validation_pair_relative_range,
        f"{family_prefix}_validation_pair_balance",
    )
    for label, values in (
        ("validation_antibody", validation_antibodies),
        ("validation_antigen", validation_antigens),
        ("training_pair", training_pairs),
        ("quarantine_pair", quarantine_pairs),
    ):
        _add_balance(
            builder,
            values,
            policy.maximum_other_count_relative_range,
            f"{family_prefix}_{label}_balance",
        )


def build_independent_cp_sat_formulation(
    problem: DirectAuthorityProblem,
    *,
    outer_policy: DirectLevelPolicy = LOCKED_OUTER_POLICY,
    inner_policy: DirectLevelPolicy = LOCKED_INNER_POLICY,
) -> DirectCPModel:
    """Build all 3 outer and 9 dependent inner constraints directly."""
    try:
        import ortools
        from ortools.sat.python import cp_model
    except ImportError as error:
        raise DirectCPSatError("OR-Tools CP-SAT is unavailable") from error
    builder = _DirectBuilder(cp_model)
    pairs = problem.pairs
    antibodies = tuple(entity for entity, _ in problem.antibody_neighbors)
    antigens = tuple(entity for entity, _ in problem.antigen_neighbors)
    antibody_graph = {
        entity: (entity, *neighbors) for entity, neighbors in problem.antibody_neighbors
    }
    antigen_graph = {
        entity: (entity, *neighbors) for entity, neighbors in problem.antigen_neighbors
    }
    incident_a = {
        entity: tuple(pair for pair in pairs if pair.antibody_id == entity)
        for entity in antibodies
    }
    incident_g = {
        entity: tuple(pair for pair in pairs if pair.antigen_id == entity)
        for entity in antigens
    }
    v = builder.variable

    for outer in range(3):
        for entity in antibodies:
            for role in ("holdout_a", "closed_a", "train_present_a"):
                v("outer", outer, role, entity)
        for entity in antigens:
            for role in ("holdout_g", "closed_g", "train_present_g"):
                v("outer", outer, role, entity)
        for component in problem.antibody_components:
            v("outer", outer, "validation_component_a", component[0])
        for component in problem.antigen_components:
            v("outer", outer, "validation_component_g", component[0])
        for pair in pairs:
            for role in ("validation_pair", "training_pair", "quarantine_pair"):
                v("outer", outer, role, pair.pair_id)
        for inner in range(3):
            for entity in antibodies:
                for role in ("holdout_a", "closed_a", "train_present_a"):
                    v("inner", outer, inner, role, entity)
            for entity in antigens:
                for role in ("holdout_g", "closed_g", "train_present_g"):
                    v("inner", outer, inner, role, entity)
            for component in problem.antibody_components:
                v("inner", outer, inner, "validation_component_a", component[0])
            for component in problem.antigen_components:
                v("inner", outer, inner, "validation_component_g", component[0])
            for pair in pairs:
                for role in ("validation_pair", "training_pair", "quarantine_pair"):
                    v("inner", outer, inner, role, pair.pair_id)

    outer_values: dict[str, list[list[object]]] = {
        name: []
        for name in (
            "validation_pairs",
            "training_pairs",
            "quarantine_pairs",
            "holdout_antibodies",
            "holdout_antigens",
            "antibody_components",
            "antigen_components",
            "training_antibodies",
            "training_antigens",
        )
    }
    for outer in range(3):
        for entity in antibodies:
            builder.exact_or(
                v("outer", outer, "closed_a", entity),
                [v("outer", outer, "holdout_a", source) for source in antibody_graph[entity]],
                "outer_antibody_closure",
            )
        for entity in antigens:
            builder.exact_or(
                v("outer", outer, "closed_g", entity),
                [v("outer", outer, "holdout_g", source) for source in antigen_graph[entity]],
                "outer_antigen_closure",
            )
        for pair in pairs:
            validation = v("outer", outer, "validation_pair", pair.pair_id)
            training = v("outer", outer, "training_pair", pair.pair_id)
            quarantine = v("outer", outer, "quarantine_pair", pair.pair_id)
            builder.exact_and(
                validation,
                [
                    v("outer", outer, "holdout_a", pair.antibody_id),
                    v("outer", outer, "holdout_g", pair.antigen_id),
                ],
                "outer_validation_pair",
            )
            closed_a = v("outer", outer, "closed_a", pair.antibody_id)
            closed_g = v("outer", outer, "closed_g", pair.antigen_id)
            builder.add(training <= 1 - closed_a, "outer_training_pair")
            builder.add(training <= 1 - closed_g, "outer_training_pair")
            builder.add(training >= 1 - closed_a - closed_g, "outer_training_pair")
            builder.add(validation + training + quarantine == 1, "outer_partition")
        for entity in antibodies:
            builder.endpoint_presence(
                v("outer", outer, "train_present_a", entity),
                [v("outer", outer, "training_pair", pair.pair_id) for pair in incident_a[entity]],
                "outer_training_antibody_presence",
            )
            builder.add(
                v("outer", outer, "holdout_a", entity)
                <= sum(v("outer", outer, "validation_pair", pair.pair_id) for pair in incident_a[entity]),
                "outer_holdout_antibody_support",
            )
        for entity in antigens:
            builder.endpoint_presence(
                v("outer", outer, "train_present_g", entity),
                [v("outer", outer, "training_pair", pair.pair_id) for pair in incident_g[entity]],
                "outer_training_antigen_presence",
            )
            builder.add(
                v("outer", outer, "holdout_g", entity)
                <= sum(v("outer", outer, "validation_pair", pair.pair_id) for pair in incident_g[entity]),
                "outer_holdout_antigen_support",
            )
        antibody_component_variables = []
        for component in problem.antibody_components:
            output = v("outer", outer, "validation_component_a", component[0])
            builder.exact_or(
                output,
                [v("outer", outer, "holdout_a", entity) for entity in component],
                "outer_antibody_component_presence",
            )
            antibody_component_variables.append(output)
        antigen_component_variables = []
        for component in problem.antigen_components:
            output = v("outer", outer, "validation_component_g", component[0])
            builder.exact_or(
                output,
                [v("outer", outer, "holdout_g", entity) for entity in component],
                "outer_antigen_component_presence",
            )
            antigen_component_variables.append(output)
        outer_values["validation_pairs"].append(
            [v("outer", outer, "validation_pair", pair.pair_id) for pair in pairs]
        )
        outer_values["training_pairs"].append(
            [v("outer", outer, "training_pair", pair.pair_id) for pair in pairs]
        )
        outer_values["quarantine_pairs"].append(
            [v("outer", outer, "quarantine_pair", pair.pair_id) for pair in pairs]
        )
        outer_values["holdout_antibodies"].append(
            [v("outer", outer, "holdout_a", entity) for entity in antibodies]
        )
        outer_values["holdout_antigens"].append(
            [v("outer", outer, "holdout_g", entity) for entity in antigens]
        )
        outer_values["antibody_components"].append(antibody_component_variables)
        outer_values["antigen_components"].append(antigen_component_variables)
        outer_values["training_antibodies"].append(
            [v("outer", outer, "train_present_a", entity) for entity in antibodies]
        )
        outer_values["training_antigens"].append(
            [v("outer", outer, "train_present_g", entity) for entity in antigens]
        )
    for entity in antibodies:
        builder.add(
            sum(v("outer", outer, "holdout_a", entity) for outer in range(3)) <= 1,
            "outer_antibody_disjointness",
        )
    for entity in antigens:
        builder.add(
            sum(v("outer", outer, "holdout_g", entity) for outer in range(3)) <= 1,
            "outer_antigen_disjointness",
        )
    _add_level_policy(
        builder,
        outer_policy,
        outer_values["validation_pairs"],
        outer_values["holdout_antibodies"],
        outer_values["holdout_antigens"],
        outer_values["antibody_components"],
        outer_values["antigen_components"],
        outer_values["training_pairs"],
        outer_values["training_antibodies"],
        outer_values["training_antigens"],
        outer_values["quarantine_pairs"],
        len(pairs),
        len(antibodies),
        len(antigens),
        "outer",
    )

    for outer in range(3):
        inner_values: dict[str, list[list[object]]] = {
            name: [] for name in outer_values
        }
        for inner in range(3):
            for axis, entities, graph, suffix in (
                ("antibody", antibodies, antibody_graph, "a"),
                ("antigen", antigens, antigen_graph, "g"),
            ):
                for entity in entities:
                    holdout = v("inner", outer, inner, f"holdout_{suffix}", entity)
                    outer_present = v("outer", outer, f"train_present_{suffix}", entity)
                    closed = v("inner", outer, inner, f"closed_{suffix}", entity)
                    builder.add(holdout <= outer_present, f"inner_{axis}_universe")
                    builder.add(closed <= outer_present, f"inner_{axis}_closure")
                    selected = [
                        v("inner", outer, inner, f"holdout_{suffix}", source)
                        for source in graph[entity]
                    ]
                    builder.add(closed <= sum(selected), f"inner_{axis}_closure")
                    for selected_value in selected:
                        builder.add(
                            closed >= outer_present + selected_value - 1,
                            f"inner_{axis}_closure",
                        )
            for pair in pairs:
                outer_training = v("outer", outer, "training_pair", pair.pair_id)
                validation = v("inner", outer, inner, "validation_pair", pair.pair_id)
                holdout_a = v("inner", outer, inner, "holdout_a", pair.antibody_id)
                holdout_g = v("inner", outer, inner, "holdout_g", pair.antigen_id)
                builder.exact_and(
                    validation,
                    [outer_training, holdout_a, holdout_g],
                    "inner_validation_pair",
                )
                training = v("inner", outer, inner, "training_pair", pair.pair_id)
                closed_a = v("inner", outer, inner, "closed_a", pair.antibody_id)
                closed_g = v("inner", outer, inner, "closed_g", pair.antigen_id)
                builder.add(training <= outer_training, "inner_training_pair")
                builder.add(training <= 1 - closed_a, "inner_training_pair")
                builder.add(training <= 1 - closed_g, "inner_training_pair")
                builder.add(
                    training >= outer_training - closed_a - closed_g,
                    "inner_training_pair",
                )
                quarantine = v("inner", outer, inner, "quarantine_pair", pair.pair_id)
                builder.add(
                    validation + training + quarantine == outer_training,
                    "inner_partition",
                )
            for entity in antibodies:
                builder.endpoint_presence(
                    v("inner", outer, inner, "train_present_a", entity),
                    [v("inner", outer, inner, "training_pair", pair.pair_id) for pair in incident_a[entity]],
                    "inner_training_antibody_presence",
                )
                builder.add(
                    v("inner", outer, inner, "holdout_a", entity)
                    <= sum(v("inner", outer, inner, "validation_pair", pair.pair_id) for pair in incident_a[entity]),
                    "inner_holdout_antibody_support",
                )
            for entity in antigens:
                builder.endpoint_presence(
                    v("inner", outer, inner, "train_present_g", entity),
                    [v("inner", outer, inner, "training_pair", pair.pair_id) for pair in incident_g[entity]],
                    "inner_training_antigen_presence",
                )
                builder.add(
                    v("inner", outer, inner, "holdout_g", entity)
                    <= sum(v("inner", outer, inner, "validation_pair", pair.pair_id) for pair in incident_g[entity]),
                    "inner_holdout_antigen_support",
                )
            antibody_component_variables = []
            for component in problem.antibody_components:
                output = v("inner", outer, inner, "validation_component_a", component[0])
                builder.exact_or(
                    output,
                    [v("inner", outer, inner, "holdout_a", entity) for entity in component],
                    "inner_antibody_component_presence",
                )
                antibody_component_variables.append(output)
            antigen_component_variables = []
            for component in problem.antigen_components:
                output = v("inner", outer, inner, "validation_component_g", component[0])
                builder.exact_or(
                    output,
                    [v("inner", outer, inner, "holdout_g", entity) for entity in component],
                    "inner_antigen_component_presence",
                )
                antigen_component_variables.append(output)
            inner_values["validation_pairs"].append(
                [v("inner", outer, inner, "validation_pair", pair.pair_id) for pair in pairs]
            )
            inner_values["training_pairs"].append(
                [v("inner", outer, inner, "training_pair", pair.pair_id) for pair in pairs]
            )
            inner_values["quarantine_pairs"].append(
                [v("inner", outer, inner, "quarantine_pair", pair.pair_id) for pair in pairs]
            )
            inner_values["holdout_antibodies"].append(
                [v("inner", outer, inner, "holdout_a", entity) for entity in antibodies]
            )
            inner_values["holdout_antigens"].append(
                [v("inner", outer, inner, "holdout_g", entity) for entity in antigens]
            )
            inner_values["antibody_components"].append(antibody_component_variables)
            inner_values["antigen_components"].append(antigen_component_variables)
            inner_values["training_antibodies"].append(
                [v("inner", outer, inner, "train_present_a", entity) for entity in antibodies]
            )
            inner_values["training_antigens"].append(
                [v("inner", outer, inner, "train_present_g", entity) for entity in antigens]
            )
        for entity in antibodies:
            builder.add(
                sum(v("inner", outer, inner, "holdout_a", entity) for inner in range(3)) <= 1,
                "inner_antibody_disjointness",
            )
        for entity in antigens:
            builder.add(
                sum(v("inner", outer, inner, "holdout_g", entity) for inner in range(3)) <= 1,
                "inner_antigen_disjointness",
            )
        _add_level_policy(
            builder,
            inner_policy,
            inner_values["validation_pairs"],
            inner_values["holdout_antibodies"],
            inner_values["holdout_antigens"],
            inner_values["antibody_components"],
            inner_values["antigen_components"],
            inner_values["training_pairs"],
            inner_values["training_antibodies"],
            inner_values["training_antigens"],
            inner_values["quarantine_pairs"],
            outer_values["training_pairs"][outer],
            outer_values["training_antibodies"][outer],
            outer_values["training_antigens"][outer],
            "inner",
        )

    variable_keys = tuple(builder.variables)
    model_bytes = str(builder.model.Proto()).encode("ascii")
    formulation_payload = {
        "format_version": DIRECT_FORMULATION_FORMAT,
        "authority_payload_sha256": problem.authority_payload_sha256,
        "variable_keys": [list(key) for key in variable_keys],
        "constraint_family_counts": dict(sorted(builder.constraint_family_counts.items())),
        "cp_model_proto_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "ortools_version": ortools.__version__,
        "implementation_source_sha256": implementation_source_sha256(),
        "historical_milp_matrix_consumed": False,
        "highs_artifact_consumed": False,
    }
    return DirectCPModel(
        model=builder.model,
        variables=dict(builder.variables),
        authority_payload_sha256=problem.authority_payload_sha256,
        constraint_family_counts=tuple(sorted(builder.constraint_family_counts.items())),
        formulation_sha256=_sha256(formulation_payload),
    )


def solve_direct_linear_projection(
    direct_model: DirectCPModel,
    *,
    fixed_assignment: Mapping[VariableKey, int] | None = None,
    maximum_time_seconds: float | None = None,
) -> dict[str, object]:
    """Solve one direct linear projection without authorizing exact score-free gates."""
    try:
        import ortools
        from ortools.sat.python import cp_model
    except ImportError as error:
        raise DirectCPSatError("OR-Tools CP-SAT is unavailable") from error
    working_model = direct_model.model.clone()
    if fixed_assignment:
        unknown = set(fixed_assignment) - set(direct_model.variables)
        if unknown:
            raise DirectCPSatError("fixed assignment references unknown variables")
        for key, value in fixed_assignment.items():
            if value not in {0, 1}:
                raise DirectCPSatError("fixed assignment values must be binary")
            variable_index = direct_model.variables[key].index
            working_model.add(working_model.get_bool_var_from_proto_index(variable_index) == value)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = LOCKED_RANDOM_SEED
    solver.parameters.randomize_search = False
    solver.parameters.relative_gap_limit = 0.0
    solver.parameters.absolute_gap_limit = 0.0
    if maximum_time_seconds is not None:
        if maximum_time_seconds <= 0:
            raise DirectCPSatError("time limit must be positive")
        solver.parameters.max_time_in_seconds = maximum_time_seconds
    status = solver.solve(working_model)
    status_name = solver.status_name(status)
    accepted_terminal = status in {cp_model.OPTIMAL, cp_model.INFEASIBLE}
    payload: dict[str, object] = {
        "format_version": DIRECT_PROJECTION_RECEIPT_FORMAT,
        "formulation_sha256": direct_model.formulation_sha256,
        "authority_payload_sha256": direct_model.authority_payload_sha256,
        "ortools_version": ortools.__version__,
        "status": status_name,
        "accepted_projection_terminal_status": accepted_terminal,
        "historical_milp_matrix_consumed": False,
        "highs_artifact_consumed": False,
        "exact_score_free_separation_completed": False,
        "independent_statistics_audit_completed": False,
        "complete_score_free_feasible_region_certified": False,
        "training_authorized": False,
        "model_selection_authorized": False,
        "outer_evaluation_authorized": False,
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload
