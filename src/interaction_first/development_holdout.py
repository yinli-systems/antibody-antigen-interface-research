"""Score-free component-disjoint development holdout construction.

The builder consumes only identifiers from a previously authorized train-only
development subset. It does not accept labels, structures, embeddings, model
outputs, ranks, losses, or any interface-derived covariate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from .development_authority import (
    CLAIM_BOUNDARY,
    canonical_json_bytes,
    canonical_sha256,
    membership_sha256,
)
from .development_subset import AuthorizedInstance, DevelopmentSubsetError


SOURCE_FORMAT = "pairscale_development_subset_v1"
HOLDOUT_FORMAT = "pairscale_development_component_holdout_v2"
RECEIPT_FORMAT = "pairscale_development_component_holdout_receipt_v2"
SELECTION_DOMAIN = "pairscale-development-component-holdout-v1"
FORBIDDEN_FIELDS = frozenset(
    {
        "affinity",
        "contact",
        "contact_density",
        "embedding",
        "epitope",
        "interface",
        "label",
        "loss",
        "paratope",
        "prediction",
        "rank",
        "score",
    }
)
REQUIRED_IMPLEMENTATION_SOURCES = frozenset(
    {
        "scripts/build_pairscale_development_holdout.py",
        "src/interaction_first/development_authority.py",
        "src/interaction_first/development_holdout.py",
        "src/interaction_first/development_subset.py",
    }
)


class DevelopmentHoldoutError(DevelopmentSubsetError):
    """Raised when a development holdout artifact fails closed."""


@dataclass(frozen=True)
class ScoreFreeSubset:
    """The allowed identifier-only view of one development subset."""

    instances: tuple[AuthorizedInstance, ...]
    subset_sha256: str
    policy_sha256: str
    permission_name: str
    source_split: str


@dataclass(frozen=True)
class ComponentHoldoutAssignment:
    """One authenticated train/holdout assignment for development only."""

    assignment_file_sha256: str
    assignment_content_sha256: str
    source_subset_sha256: str
    policy_sha256: str
    train_instances: tuple[AuthorizedInstance, ...]
    holdout_instances: tuple[AuthorizedInstance, ...]


@dataclass(frozen=True)
class EndpointComponent:
    """One connected component of the observed antibody--antigen graph."""

    antibody_ids: tuple[str, ...]
    antigen_ids: tuple[str, ...]
    instances: tuple[AuthorizedInstance, ...]
    component_sha256: str

    @property
    def pair_count(self) -> int:
        """Return the number of observed instance rows in this component."""

        return len(self.instances)


def parse_score_free_subset(
    payload_bytes: bytes,
    *,
    expected_subset_sha256: str,
) -> ScoreFreeSubset:
    """Parse the exact allowlisted fields from an authenticated subset."""

    _sha256(expected_subset_sha256, "expected_subset_sha256")
    actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha256 != expected_subset_sha256:
        raise DevelopmentHoldoutError("subset SHA-256 mismatch")
    payload = _strict_json_object(payload_bytes, "subset")
    _reject_forbidden_keys(payload)
    required = {
        "format_version",
        "permission_name",
        "purpose",
        "claim_boundary",
        "source_split",
        "parent_authorization_sha256",
        "policy_sha256",
        "selection",
        "authorization_manifest_sha256",
        "instance_count",
        "instance_membership_sha256",
        "antibody_count",
        "antigen_count",
        "positive_pair_count",
        "excluded_pair_count",
        "instances",
    }
    if set(payload) != required:
        missing = sorted(required.difference(payload))
        extra = sorted(set(payload).difference(required))
        raise DevelopmentHoldoutError(
            f"subset keys differ; missing={missing}, extra={extra}"
        )
    if payload["format_version"] != SOURCE_FORMAT:
        raise DevelopmentHoldoutError("unsupported subset format")
    if payload["source_split"] != "train":
        raise DevelopmentHoldoutError("source subset is not train-only")
    if payload["claim_boundary"] != CLAIM_BOUNDARY:
        raise DevelopmentHoldoutError("claim boundary differs from policy")
    raw_instances = payload["instances"]
    if type(raw_instances) is not list or not raw_instances:
        raise DevelopmentHoldoutError("instances must be a nonempty list")
    instances: list[AuthorizedInstance] = []
    for index, raw in enumerate(raw_instances):
        if type(raw) is not dict:
            raise DevelopmentHoldoutError(f"instances[{index}] must be an object")
        if set(raw) != {"instance_id", "antibody_id", "antigen_id"}:
            raise DevelopmentHoldoutError(
                f"instances[{index}] contains non-allowlisted fields"
            )
        instances.append(
            AuthorizedInstance(
                instance_id=raw["instance_id"],
                antibody_id=raw["antibody_id"],
                antigen_id=raw["antigen_id"],
            )
        )
    canonical_instances = tuple(instances)
    if canonical_instances != tuple(
        sorted(canonical_instances, key=lambda value: value.instance_id)
    ):
        raise DevelopmentHoldoutError("instances must be sorted by instance ID")
    if len({value.instance_id for value in canonical_instances}) != len(
        canonical_instances
    ):
        raise DevelopmentHoldoutError("instances contain duplicate IDs")
    if payload["instance_count"] != len(canonical_instances):
        raise DevelopmentHoldoutError("instance count mismatch")
    if payload["instance_membership_sha256"] != membership_sha256(
        value.instance_id for value in canonical_instances
    ):
        raise DevelopmentHoldoutError("instance membership digest mismatch")
    if payload["antibody_count"] != len(
        {value.antibody_id for value in canonical_instances}
    ):
        raise DevelopmentHoldoutError("antibody count mismatch")
    if payload["antigen_count"] != len(
        {value.antigen_id for value in canonical_instances}
    ):
        raise DevelopmentHoldoutError("antigen count mismatch")
    return ScoreFreeSubset(
        instances=canonical_instances,
        subset_sha256=actual_sha256,
        policy_sha256=_sha256(payload["policy_sha256"], "policy_sha256"),
        permission_name=str(payload["permission_name"]),
        source_split=str(payload["source_split"]),
    )


def parse_component_holdout_assignment(
    assignment_bytes: bytes,
    receipt_bytes: bytes,
    *,
    expected_assignment_file_sha256: str,
    expected_assignment_content_sha256: str,
    expected_receipt_file_sha256: str,
    expected_source_subset_sha256: str,
    expected_policy_sha256: str,
) -> ComponentHoldoutAssignment:
    """Authenticate and validate one published component holdout and receipt."""

    for value, label in (
        (expected_assignment_file_sha256, "expected_assignment_file_sha256"),
        (expected_assignment_content_sha256, "expected_assignment_content_sha256"),
        (expected_receipt_file_sha256, "expected_receipt_file_sha256"),
        (expected_source_subset_sha256, "expected_source_subset_sha256"),
        (expected_policy_sha256, "expected_policy_sha256"),
    ):
        _sha256(value, label)
    assignment_file_sha256 = hashlib.sha256(assignment_bytes).hexdigest()
    if assignment_file_sha256 != expected_assignment_file_sha256:
        raise DevelopmentHoldoutError("assignment file SHA-256 mismatch")
    if hashlib.sha256(receipt_bytes).hexdigest() != expected_receipt_file_sha256:
        raise DevelopmentHoldoutError("receipt file SHA-256 mismatch")
    payload = _strict_json_object(assignment_bytes, "assignment")
    expected_top_fields = {
        "format_version",
        "permission_name",
        "source_split",
        "scope",
        "claim_boundary",
        "source_subset_sha256",
        "policy_sha256",
        "selection",
        "source",
        "train",
        "holdout",
        "separation_audit",
        "assignment_content_sha256",
    }
    _exact_fields(payload, expected_top_fields, "assignment")
    if payload["format_version"] != HOLDOUT_FORMAT:
        raise DevelopmentHoldoutError("unsupported assignment format")
    if payload["source_split"] != "train":
        raise DevelopmentHoldoutError("assignment source is not train-only")
    if payload["scope"] != "authority_train_only_development_double_cold":
        raise DevelopmentHoldoutError("assignment scope differs")
    if payload["claim_boundary"] != CLAIM_BOUNDARY:
        raise DevelopmentHoldoutError("assignment claim boundary differs")
    if payload["source_subset_sha256"] != expected_source_subset_sha256:
        raise DevelopmentHoldoutError("assignment source subset digest differs")
    if payload["policy_sha256"] != expected_policy_sha256:
        raise DevelopmentHoldoutError("assignment policy digest differs")
    claimed_content_sha256 = _sha256(
        payload["assignment_content_sha256"],
        "assignment_content_sha256",
    )
    if claimed_content_sha256 != expected_assignment_content_sha256:
        raise DevelopmentHoldoutError("assignment content digest anchor differs")
    content_payload = dict(payload)
    content_payload.pop("assignment_content_sha256")
    if canonical_sha256(content_payload) != claimed_content_sha256:
        raise DevelopmentHoldoutError("assignment content digest mismatch")
    selection = _object(payload["selection"], "selection")
    _exact_fields(
        selection,
        {
            "algorithm",
            "domain_separator",
            "seed",
            "requested_holdout_pair_count",
        },
        "selection",
    )
    if selection["algorithm"] != (
        "sha256_ordered_component_subset_sum_exact_pair_count"
    ):
        raise DevelopmentHoldoutError("assignment algorithm differs")
    if selection["domain_separator"] != SELECTION_DOMAIN:
        raise DevelopmentHoldoutError("assignment domain separator differs")
    if type(selection["seed"]) is not int or selection["seed"] < 0:
        raise DevelopmentHoldoutError("assignment seed is invalid")
    requested_holdout_count = _positive_integer(
        selection["requested_holdout_pair_count"],
        "requested_holdout_pair_count",
    )
    source = _object(payload["source"], "source")
    _exact_fields(
        source,
        {"instance_count", "component_count", "instance_membership_sha256"},
        "source",
    )
    source_count = _positive_integer(source["instance_count"], "source count")
    _positive_integer(source["component_count"], "source component count")
    source_membership_sha256 = _sha256(
        source["instance_membership_sha256"],
        "source instance membership",
    )
    train_instances = _parse_partition(payload["train"], "train")
    holdout_instances = _parse_partition(payload["holdout"], "holdout")
    if len(holdout_instances) != requested_holdout_count:
        raise DevelopmentHoldoutError("holdout count differs from selection")
    if len(train_instances) + len(holdout_instances) != source_count:
        raise DevelopmentHoldoutError("partition counts do not cover the source")
    all_instances = (*train_instances, *holdout_instances)
    if membership_sha256(value.instance_id for value in all_instances) != (
        source_membership_sha256
    ):
        raise DevelopmentHoldoutError("partition membership differs from source")
    train_antibodies = {value.antibody_id for value in train_instances}
    train_antigens = {value.antigen_id for value in train_instances}
    holdout_antibodies = {value.antibody_id for value in holdout_instances}
    holdout_antigens = {value.antigen_id for value in holdout_instances}
    if train_antibodies.intersection(holdout_antibodies):
        raise DevelopmentHoldoutError("antibody endpoints cross the holdout")
    if train_antigens.intersection(holdout_antigens):
        raise DevelopmentHoldoutError("antigen endpoints cross the holdout")
    separation = _object(payload["separation_audit"], "separation_audit")
    _exact_fields(
        separation,
        {
            "instance_overlap_count",
            "antibody_overlap_count",
            "antigen_overlap_count",
            "contact_or_score_fields_accessed",
            "validation_or_test_data_accessed",
        },
        "separation_audit",
    )
    if separation != {
        "instance_overlap_count": 0,
        "antibody_overlap_count": 0,
        "antigen_overlap_count": 0,
        "contact_or_score_fields_accessed": False,
        "validation_or_test_data_accessed": False,
    }:
        raise DevelopmentHoldoutError("assignment separation audit did not pass")
    receipt = _strict_json_object(receipt_bytes, "receipt")
    _exact_fields(
        receipt,
        {
            "format_version",
            "assignment_content_sha256",
            "assignment_file_sha256",
            "source_subset_sha256",
            "implementation_source_sha256",
            "score_or_label_fields_accessed",
            "validation_or_test_data_accessed",
            "confirmatory_authority_granted",
            "receipt_sha256",
        },
        "receipt",
    )
    if receipt["format_version"] != RECEIPT_FORMAT:
        raise DevelopmentHoldoutError("unsupported receipt format")
    if receipt["assignment_content_sha256"] != claimed_content_sha256:
        raise DevelopmentHoldoutError("receipt content digest differs")
    if receipt["assignment_file_sha256"] != assignment_file_sha256:
        raise DevelopmentHoldoutError("receipt assignment file digest differs")
    if receipt["source_subset_sha256"] != expected_source_subset_sha256:
        raise DevelopmentHoldoutError("receipt source subset digest differs")
    if any(
        receipt[field] is not False
        for field in (
            "score_or_label_fields_accessed",
            "validation_or_test_data_accessed",
            "confirmatory_authority_granted",
        )
    ):
        raise DevelopmentHoldoutError("receipt grants forbidden authority")
    claimed_receipt_sha256 = _sha256(
        receipt["receipt_sha256"],
        "receipt_sha256",
    )
    receipt_payload = dict(receipt)
    receipt_payload.pop("receipt_sha256")
    if canonical_sha256(receipt_payload) != claimed_receipt_sha256:
        raise DevelopmentHoldoutError("receipt content digest mismatch")
    implementation_sources = _object(
        receipt["implementation_source_sha256"],
        "implementation_source_sha256",
    )
    if set(implementation_sources) != REQUIRED_IMPLEMENTATION_SOURCES:
        raise DevelopmentHoldoutError("receipt source inventory differs")
    for name, digest in implementation_sources.items():
        _sha256(digest, f"implementation_source_sha256[{name}]")
    return ComponentHoldoutAssignment(
        assignment_file_sha256=assignment_file_sha256,
        assignment_content_sha256=claimed_content_sha256,
        source_subset_sha256=expected_source_subset_sha256,
        policy_sha256=expected_policy_sha256,
        train_instances=train_instances,
        holdout_instances=holdout_instances,
    )


def connected_components(
    instances: Sequence[AuthorizedInstance],
) -> tuple[EndpointComponent, ...]:
    """Build canonical bipartite components without reading score fields."""

    if not instances:
        raise DevelopmentHoldoutError("instances must not be empty")
    by_antibody: dict[str, set[str]] = {}
    by_antigen: dict[str, set[str]] = {}
    for value in instances:
        by_antibody.setdefault(value.antibody_id, set()).add(value.antigen_id)
        by_antigen.setdefault(value.antigen_id, set()).add(value.antibody_id)
    unseen_antibodies = set(by_antibody)
    components: list[EndpointComponent] = []
    while unseen_antibodies:
        antibody_frontier = [min(unseen_antibodies)]
        component_antibodies: set[str] = set()
        component_antigens: set[str] = set()
        while antibody_frontier:
            antibody_id = antibody_frontier.pop()
            if antibody_id in component_antibodies:
                continue
            component_antibodies.add(antibody_id)
            unseen_antibodies.discard(antibody_id)
            for antigen_id in sorted(by_antibody[antibody_id]):
                if antigen_id in component_antigens:
                    continue
                component_antigens.add(antigen_id)
                antibody_frontier.extend(sorted(by_antigen[antigen_id], reverse=True))
        component_instances = tuple(
            sorted(
                (
                    value
                    for value in instances
                    if value.antibody_id in component_antibodies
                ),
                key=lambda value: value.instance_id,
            )
        )
        identity_payload = {
            "antibody_ids": sorted(component_antibodies),
            "antigen_ids": sorted(component_antigens),
            "instance_ids": [value.instance_id for value in component_instances],
        }
        components.append(
            EndpointComponent(
                antibody_ids=tuple(sorted(component_antibodies)),
                antigen_ids=tuple(sorted(component_antigens)),
                instances=component_instances,
                component_sha256=canonical_sha256(identity_payload),
            )
        )
    return tuple(sorted(components, key=lambda value: value.component_sha256))


def component_selection_key(
    component_sha256: str,
    *,
    seed: int,
) -> tuple[bytes, str]:
    """Return the deterministic score-free ordering for one component."""

    _sha256(component_sha256, "component_sha256")
    if type(seed) is not int or seed < 0:
        raise DevelopmentHoldoutError("seed must be a nonnegative integer")
    material = (
        SELECTION_DOMAIN.encode("ascii")
        + b"\0"
        + str(seed).encode("ascii")
        + b"\0"
        + component_sha256.encode("ascii")
    )
    return hashlib.sha256(material).digest(), component_sha256


def select_exact_holdout_components(
    components: Sequence[EndpointComponent],
    *,
    holdout_pair_count: int,
    seed: int,
) -> tuple[EndpointComponent, ...]:
    """Select an exact pair count using canonical component-level subset sum."""

    if type(holdout_pair_count) is not int or holdout_pair_count <= 0:
        raise DevelopmentHoldoutError("holdout_pair_count must be positive")
    total = sum(value.pair_count for value in components)
    if holdout_pair_count >= total:
        raise DevelopmentHoldoutError("holdout must be smaller than the source")
    ranked = tuple(
        sorted(
            components,
            key=lambda value: component_selection_key(
                value.component_sha256,
                seed=seed,
            ),
        )
    )
    states: dict[int, tuple[int, ...]] = {0: ()}
    for index, component in enumerate(ranked):
        additions: dict[int, tuple[int, ...]] = {}
        for current, selected in sorted(states.items(), reverse=True):
            candidate_total = current + component.pair_count
            if candidate_total > holdout_pair_count:
                continue
            candidate = (*selected, index)
            previous = states.get(candidate_total)
            pending = additions.get(candidate_total)
            best = min(
                (value for value in (previous, pending, candidate) if value is not None),
                default=candidate,
            )
            additions[candidate_total] = best
        for candidate_total, candidate in additions.items():
            previous = states.get(candidate_total)
            if previous is None or candidate < previous:
                states[candidate_total] = candidate
    if holdout_pair_count not in states:
        raise DevelopmentHoldoutError(
            "component partition cannot realize the requested holdout size"
        )
    return tuple(ranked[index] for index in states[holdout_pair_count])


def build_component_holdout(
    subset: ScoreFreeSubset,
    *,
    holdout_pair_count: int,
    seed: int,
) -> Mapping[str, object]:
    """Build one endpoint-disjoint train/holdout assignment receipt."""

    components = connected_components(subset.instances)
    holdout_components = select_exact_holdout_components(
        components,
        holdout_pair_count=holdout_pair_count,
        seed=seed,
    )
    holdout_component_ids = {
        value.component_sha256 for value in holdout_components
    }
    train_components = tuple(
        value
        for value in components
        if value.component_sha256 not in holdout_component_ids
    )
    holdout_instances = tuple(
        sorted(
            (
                instance
                for component in holdout_components
                for instance in component.instances
            ),
            key=lambda value: value.instance_id,
        )
    )
    train_instances = tuple(
        sorted(
            (
                instance
                for component in train_components
                for instance in component.instances
            ),
            key=lambda value: value.instance_id,
        )
    )
    train_antibodies = {value.antibody_id for value in train_instances}
    train_antigens = {value.antigen_id for value in train_instances}
    holdout_antibodies = {value.antibody_id for value in holdout_instances}
    holdout_antigens = {value.antigen_id for value in holdout_instances}
    if train_antibodies.intersection(holdout_antibodies):
        raise AssertionError("antibody endpoints cross the component split")
    if train_antigens.intersection(holdout_antigens):
        raise AssertionError("antigen endpoints cross the component split")
    if {value.instance_id for value in train_instances}.intersection(
        value.instance_id for value in holdout_instances
    ):
        raise AssertionError("instance IDs cross the component split")
    payload: dict[str, object] = {
        "format_version": HOLDOUT_FORMAT,
        "permission_name": subset.permission_name,
        "source_split": subset.source_split,
        "scope": "authority_train_only_development_double_cold",
        "claim_boundary": CLAIM_BOUNDARY,
        "source_subset_sha256": subset.subset_sha256,
        "policy_sha256": subset.policy_sha256,
        "selection": {
            "algorithm": "sha256_ordered_component_subset_sum_exact_pair_count",
            "domain_separator": SELECTION_DOMAIN,
            "seed": seed,
            "requested_holdout_pair_count": holdout_pair_count,
        },
        "source": {
            "instance_count": len(subset.instances),
            "component_count": len(components),
            "instance_membership_sha256": membership_sha256(
                value.instance_id for value in subset.instances
            ),
        },
        "train": _partition_payload(train_instances, train_components),
        "holdout": _partition_payload(holdout_instances, holdout_components),
        "separation_audit": {
            "instance_overlap_count": 0,
            "antibody_overlap_count": 0,
            "antigen_overlap_count": 0,
            "contact_or_score_fields_accessed": False,
            "validation_or_test_data_accessed": False,
        },
    }
    payload["assignment_content_sha256"] = canonical_sha256(payload)
    return payload


def build_holdout_receipt(
    assignment: Mapping[str, object],
    *,
    implementation_source_sha256: Mapping[str, str],
) -> Mapping[str, object]:
    """Bind a holdout assignment to the complete implementation source set."""

    if set(implementation_source_sha256) != REQUIRED_IMPLEMENTATION_SOURCES:
        raise DevelopmentHoldoutError(
            "implementation source keys differ from the locked set"
        )
    for name, digest in implementation_source_sha256.items():
        _sha256(digest, f"implementation_source_sha256[{name}]")
    receipt: dict[str, object] = {
        "format_version": RECEIPT_FORMAT,
        "assignment_content_sha256": assignment["assignment_content_sha256"],
        "assignment_file_sha256": canonical_sha256(assignment),
        "source_subset_sha256": assignment["source_subset_sha256"],
        "implementation_source_sha256": dict(
            sorted(implementation_source_sha256.items())
        ),
        "score_or_label_fields_accessed": False,
        "validation_or_test_data_accessed": False,
        "confirmatory_authority_granted": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _partition_payload(
    instances: Sequence[AuthorizedInstance],
    components: Sequence[EndpointComponent],
) -> Mapping[str, object]:
    return {
        "instance_count": len(instances),
        "antibody_count": len({value.antibody_id for value in instances}),
        "antigen_count": len({value.antigen_id for value in instances}),
        "component_count": len(components),
        "instance_membership_sha256": membership_sha256(
            value.instance_id for value in instances
        ),
        "component_membership_sha256": membership_sha256(
            value.component_sha256 for value in components
        ),
        "instances": [value.canonical_payload() for value in instances],
        "component_sha256": sorted(value.component_sha256 for value in components),
    }


def _parse_partition(
    value: object,
    name: str,
) -> tuple[AuthorizedInstance, ...]:
    partition = _object(value, name)
    _exact_fields(
        partition,
        {
            "instance_count",
            "antibody_count",
            "antigen_count",
            "component_count",
            "instance_membership_sha256",
            "component_membership_sha256",
            "instances",
            "component_sha256",
        },
        name,
    )
    raw_instances = partition["instances"]
    if type(raw_instances) is not list or not raw_instances:
        raise DevelopmentHoldoutError(f"{name}.instances must be nonempty")
    instances: list[AuthorizedInstance] = []
    for index, raw in enumerate(raw_instances):
        if type(raw) is not dict or set(raw) != {
            "instance_id",
            "antibody_id",
            "antigen_id",
        }:
            raise DevelopmentHoldoutError(
                f"{name}.instances[{index}] contains non-allowlisted fields"
            )
        instances.append(
            AuthorizedInstance(
                instance_id=raw["instance_id"],
                antibody_id=raw["antibody_id"],
                antigen_id=raw["antigen_id"],
            )
        )
    result = tuple(instances)
    if result != tuple(sorted(result, key=lambda item: item.instance_id)):
        raise DevelopmentHoldoutError(f"{name}.instances are not sorted")
    if len({item.instance_id for item in result}) != len(result):
        raise DevelopmentHoldoutError(f"{name}.instances contain duplicates")
    if _positive_integer(partition["instance_count"], f"{name} instance count") != len(
        result
    ):
        raise DevelopmentHoldoutError(f"{name} instance count mismatch")
    if _positive_integer(partition["antibody_count"], f"{name} antibody count") != len(
        {item.antibody_id for item in result}
    ):
        raise DevelopmentHoldoutError(f"{name} antibody count mismatch")
    if _positive_integer(partition["antigen_count"], f"{name} antigen count") != len(
        {item.antigen_id for item in result}
    ):
        raise DevelopmentHoldoutError(f"{name} antigen count mismatch")
    component_count = _positive_integer(
        partition["component_count"],
        f"{name} component count",
    )
    component_sha256 = partition["component_sha256"]
    if type(component_sha256) is not list or len(component_sha256) != component_count:
        raise DevelopmentHoldoutError(f"{name} component inventory count mismatch")
    if component_sha256 != sorted(component_sha256):
        raise DevelopmentHoldoutError(f"{name} component inventory is not sorted")
    if len(set(component_sha256)) != len(component_sha256):
        raise DevelopmentHoldoutError(f"{name} component inventory has duplicates")
    for digest in component_sha256:
        _sha256(digest, f"{name} component digest")
    if partition["instance_membership_sha256"] != membership_sha256(
        item.instance_id for item in result
    ):
        raise DevelopmentHoldoutError(f"{name} instance membership mismatch")
    if partition["component_membership_sha256"] != membership_sha256(
        component_sha256
    ):
        raise DevelopmentHoldoutError(f"{name} component membership mismatch")
    return result


def _strict_json_object(raw: bytes, name: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise DevelopmentHoldoutError(f"non-finite JSON constant in {name}: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DevelopmentHoldoutError(f"duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentHoldoutError(f"{name} is not valid JSON") from exc
    if type(value) is not dict:
        raise DevelopmentHoldoutError(f"{name} must be a JSON object")
    return value


def _object(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise DevelopmentHoldoutError(f"{name} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise DevelopmentHoldoutError(
            f"{name} fields differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise DevelopmentHoldoutError(f"{name} must be a positive integer")
    return value


def _reject_forbidden_keys(value: object, path: str = "$") -> None:
    if type(value) is dict:
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_FIELDS):
                raise DevelopmentHoldoutError(
                    f"forbidden score-derived field at {path}.{key}"
                )
            _reject_forbidden_keys(child, f"{path}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _reject_forbidden_keys(child, f"{path}[{index}]")


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise DevelopmentHoldoutError(f"{label} must be a lowercase SHA-256")
    return value
