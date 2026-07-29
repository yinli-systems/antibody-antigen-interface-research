"""Deterministic score-free subsets for nonconfirmatory development.

Selection consumes only the exact train-only authorization inventory and uses
identifier hashes. It never reads contact labels, tensor artifacts, model
outputs, losses, ranks, or held-out data.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .development_authority import (
    CLAIM_BOUNDARY,
    PERMISSION_NAME,
    DevelopmentAuthorityError,
    DevelopmentDecision,
    DevelopmentPolicy,
    DevelopmentPurpose,
    DevelopmentRequest,
    canonical_json_bytes,
    canonical_sha256,
    decide_development_authority,
    membership_sha256,
)


AUTHORIZATION_FORMAT = "frontier_authorized_fold_inventory_v1"
DATA_ACCESS_SCOPE = "train_only_fold_local"
SUBSET_FORMAT = "pairscale_development_subset_v1"
RECEIPT_FORMAT = "pairscale_development_subset_receipt_v1"
SELECTION_DOMAIN = "pairscale-development-subset-v1"
REQUIRED_IMPLEMENTATION_SOURCES = frozenset(
    {
        "scripts/build_pairscale_development_subsets.py",
        "src/interaction_first/development_authority.py",
        "src/interaction_first/development_subset.py",
    }
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/@-]{0,511}$")


class DevelopmentSubsetError(DevelopmentAuthorityError):
    """Raised when a train-only subset artifact fails closed."""


@dataclass(frozen=True, order=True)
class AuthorizedInstance:
    """One score-free instance record from the train authorization."""

    instance_id: str
    antibody_id: str
    antigen_id: str

    def __post_init__(self) -> None:
        for name in ("instance_id", "antibody_id", "antigen_id"):
            _identifier(getattr(self, name), name)

    def canonical_payload(self) -> Mapping[str, str]:
        """Return the strict frontier authorization record."""

        return {
            "instance_id": self.instance_id,
            "antibody_id": self.antibody_id,
            "antigen_id": self.antigen_id,
        }


@dataclass(frozen=True)
class TrainAuthorizationInventory:
    """One complete train-only authorization inventory."""

    fold_id: str
    instances: tuple[AuthorizedInstance, ...]
    positive_pairs: tuple[tuple[str, str], ...]
    excluded_pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _identifier(self.fold_id, "fold_id")
        if not isinstance(self.instances, tuple) or not self.instances:
            raise DevelopmentSubsetError("instances must be a nonempty tuple")
        if not all(isinstance(value, AuthorizedInstance) for value in self.instances):
            raise DevelopmentSubsetError("instances contain an invalid record")
        instance_ids = [value.instance_id for value in self.instances]
        if len(set(instance_ids)) != len(instance_ids):
            raise DevelopmentSubsetError("instances contain duplicate instance IDs")
        if instance_ids != sorted(instance_ids):
            raise DevelopmentSubsetError("instances must be sorted by instance ID")
        for name, pairs in (
            ("positive_pairs", self.positive_pairs),
            ("excluded_pairs", self.excluded_pairs),
        ):
            if not isinstance(pairs, tuple):
                raise DevelopmentSubsetError(f"{name} must be an immutable tuple")
            for pair in pairs:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise DevelopmentSubsetError(f"{name} contains an invalid pair")
                _identifier(pair[0], f"{name} antibody ID")
                _identifier(pair[1], f"{name} antigen ID")
            if len(set(pairs)) != len(pairs):
                raise DevelopmentSubsetError(f"{name} contains duplicates")
            if list(pairs) != sorted(pairs):
                raise DevelopmentSubsetError(f"{name} must be sorted")
        if set(self.positive_pairs).intersection(self.excluded_pairs):
            raise DevelopmentSubsetError(
                "positive_pairs and excluded_pairs must be disjoint"
            )
        antibody_ids = {value.antibody_id for value in self.instances}
        antigen_ids = {value.antigen_id for value in self.instances}
        for pair in (*self.positive_pairs, *self.excluded_pairs):
            if pair[0] not in antibody_ids or pair[1] not in antigen_ids:
                raise DevelopmentSubsetError(
                    "pair endpoints must exist in the authorization inventory"
                )

    @property
    def instance_ids(self) -> tuple[str, ...]:
        """Return the canonical instance identifier order."""

        return tuple(value.instance_id for value in self.instances)


@dataclass(frozen=True)
class DevelopmentSubset:
    """One deterministic nested subset and its authorization decision."""

    requested_size: int
    seed: int
    parent_authorization_sha256: str
    policy: DevelopmentPolicy
    instances: tuple[AuthorizedInstance, ...]
    positive_pairs: tuple[tuple[str, str], ...]
    excluded_pairs: tuple[tuple[str, str], ...]
    authorization_payload: Mapping[str, object]
    subset_payload: Mapping[str, object]
    decision: DevelopmentDecision

    @property
    def authorization_sha256(self) -> str:
        """Return the exact compatible authorization-manifest digest."""

        return hashlib.sha256(
            canonical_json_bytes(self.authorization_payload)
        ).hexdigest()

    @property
    def subset_sha256(self) -> str:
        """Return the exact research subset-manifest digest."""

        return hashlib.sha256(canonical_json_bytes(self.subset_payload)).hexdigest()


def parse_train_authorization(payload_bytes: bytes) -> TrainAuthorizationInventory:
    """Parse an exact train-only frontier authorization from strict JSON."""

    payload = _strict_json_object(payload_bytes, "train authorization")
    _exact_keys(
        payload,
        {
            "format_version",
            "data_access_scope",
            "split_role",
            "fold_id",
            "instances",
            "positive_pairs",
            "excluded_pairs",
        },
        "train authorization",
    )
    if payload["format_version"] != AUTHORIZATION_FORMAT:
        raise DevelopmentSubsetError("unsupported authorization format")
    if payload["data_access_scope"] != DATA_ACCESS_SCOPE:
        raise DevelopmentSubsetError("authorization scope is not train-only")
    if payload["split_role"] != "train":
        raise DevelopmentSubsetError("authorization split_role is not train")
    raw_instances = _plain_list(payload["instances"], "instances")
    instances: list[AuthorizedInstance] = []
    for index, raw in enumerate(raw_instances):
        if type(raw) is not dict:
            raise DevelopmentSubsetError(f"instances[{index}] must be an object")
        _exact_keys(
            raw,
            {"instance_id", "antibody_id", "antigen_id"},
            f"instances[{index}]",
        )
        instances.append(
            AuthorizedInstance(
                instance_id=raw["instance_id"],
                antibody_id=raw["antibody_id"],
                antigen_id=raw["antigen_id"],
            )
        )
    return TrainAuthorizationInventory(
        fold_id=_identifier(payload["fold_id"], "fold_id"),
        instances=tuple(instances),
        positive_pairs=_pairs(payload["positive_pairs"], "positive_pairs"),
        excluded_pairs=_pairs(payload["excluded_pairs"], "excluded_pairs"),
    )


def selection_key(instance_id: str, *, seed: int) -> tuple[bytes, str]:
    """Return a cross-runtime deterministic, score-free selection key."""

    _identifier(instance_id, "instance_id")
    _nonnegative_integer(seed, "seed")
    material = (
        SELECTION_DOMAIN.encode("ascii")
        + b"\0"
        + str(seed).encode("ascii")
        + b"\0"
        + instance_id.encode("ascii")
    )
    return hashlib.sha256(material).digest(), instance_id


def build_development_subsets(
    *,
    inventory: TrainAuthorizationInventory,
    parent_authorization_sha256: str,
    policy: DevelopmentPolicy,
    expected_policy_sha256: str,
    sizes: Sequence[int],
    seed: int,
    requested_at_utc: str,
    evaluated_at_utc: str,
) -> tuple[DevelopmentSubset, ...]:
    """Build nested score-free subsets and authorization decisions."""

    _sha256(parent_authorization_sha256, "parent_authorization_sha256")
    _sha256(expected_policy_sha256, "expected_policy_sha256")
    if parent_authorization_sha256 != policy.cache_authorization_manifest_sha256:
        raise DevelopmentSubsetError(
            "parent authorization digest differs from the frozen policy"
        )
    canonical_sizes = _sizes(sizes, maximum=len(inventory.instances))
    _nonnegative_integer(seed, "seed")
    if len(inventory.instances) != policy.counts.exact_pairs:
        raise DevelopmentSubsetError(
            "authorization inventory count differs from the frozen policy"
        )
    if membership_sha256(inventory.instance_ids) != policy.authority_membership_sha256:
        raise DevelopmentSubsetError(
            "authorization inventory membership differs from the frozen policy"
        )
    ranked = sorted(
        inventory.instances,
        key=lambda value: selection_key(value.instance_id, seed=seed),
    )
    results: list[DevelopmentSubset] = []
    for size in canonical_sizes:
        chosen = tuple(sorted(ranked[:size], key=lambda value: value.instance_id))
        antibody_ids = {value.antibody_id for value in chosen}
        antigen_ids = {value.antigen_id for value in chosen}
        positive_pairs = tuple(
            pair
            for pair in inventory.positive_pairs
            if pair[0] in antibody_ids and pair[1] in antigen_ids
        )
        excluded_pairs = tuple(
            pair
            for pair in inventory.excluded_pairs
            if pair[0] in antibody_ids and pair[1] in antigen_ids
        )
        fold_id = f"pairscale-dev-seed{seed}-n{size}"
        authorization_payload: Mapping[str, object] = {
            "format_version": AUTHORIZATION_FORMAT,
            "data_access_scope": DATA_ACCESS_SCOPE,
            "split_role": "train",
            "fold_id": fold_id,
            "instances": [value.canonical_payload() for value in chosen],
            "positive_pairs": [list(pair) for pair in positive_pairs],
            "excluded_pairs": [list(pair) for pair in excluded_pairs],
        }
        authorization_sha256 = hashlib.sha256(
            canonical_json_bytes(authorization_payload)
        ).hexdigest()
        subset_payload: Mapping[str, object] = {
            "format_version": SUBSET_FORMAT,
            "permission_name": PERMISSION_NAME,
            "purpose": DevelopmentPurpose.OVERFIT_SANITY.value,
            "claim_boundary": CLAIM_BOUNDARY,
            "source_split": "train",
            "parent_authorization_sha256": parent_authorization_sha256,
            "policy_sha256": policy.policy_sha256,
            "selection": {
                "algorithm": "sha256_ascending_then_instance_id",
                "domain_separator": SELECTION_DOMAIN,
                "seed": seed,
                "requested_size": size,
            },
            "authorization_manifest_sha256": authorization_sha256,
            "instance_count": len(chosen),
            "instance_membership_sha256": membership_sha256(
                value.instance_id for value in chosen
            ),
            "antibody_count": len(antibody_ids),
            "antigen_count": len(antigen_ids),
            "positive_pair_count": len(positive_pairs),
            "excluded_pair_count": len(excluded_pairs),
            "instances": [value.canonical_payload() for value in chosen],
        }
        subset_sha256 = hashlib.sha256(
            canonical_json_bytes(subset_payload)
        ).hexdigest()
        request = DevelopmentRequest(
            run_id=fold_id,
            purpose=DevelopmentPurpose.OVERFIT_SANITY,
            subset_instance_ids=tuple(value.instance_id for value in chosen),
            subset_manifest_sha256=subset_sha256,
            requested_at_utc=requested_at_utc,
        )
        decision = decide_development_authority(
            policy=policy,
            request=request,
            authority_instance_ids=inventory.instance_ids,
            expected_policy_sha256=expected_policy_sha256,
            evaluated_at_utc=evaluated_at_utc,
        )
        if not decision.authorized:
            raise DevelopmentSubsetError(
                "development decision unexpectedly blocked: "
                + ",".join(decision.blocking_reasons)
            )
        results.append(
            DevelopmentSubset(
                requested_size=size,
                seed=seed,
                parent_authorization_sha256=parent_authorization_sha256,
                policy=policy,
                instances=chosen,
                positive_pairs=positive_pairs,
                excluded_pairs=excluded_pairs,
                authorization_payload=authorization_payload,
                subset_payload=subset_payload,
                decision=decision,
            )
        )
    for smaller, larger in zip(results, results[1:]):
        if not {value.instance_id for value in smaller.instances}.issubset(
            value.instance_id for value in larger.instances
        ):
            raise AssertionError("deterministic subsets are not nested")
    return tuple(results)


def build_subset_receipt(
    subsets: Sequence[DevelopmentSubset],
    *,
    implementation_source_sha256: Mapping[str, str],
) -> Mapping[str, object]:
    """Build a deterministic receipt over one complete nested subset family."""

    if set(implementation_source_sha256) != REQUIRED_IMPLEMENTATION_SOURCES:
        raise DevelopmentSubsetError(
            "implementation source keys differ from the locked set"
        )
    for name, digest in implementation_source_sha256.items():
        _sha256(digest, f"implementation_source_sha256[{name}]")
    if not subsets:
        raise DevelopmentSubsetError("at least one subset is required")
    parent_hashes = {value.parent_authorization_sha256 for value in subsets}
    policy_hashes = {value.policy.policy_sha256 for value in subsets}
    seeds = {value.seed for value in subsets}
    if len(parent_hashes) != 1 or len(policy_hashes) != 1 or len(seeds) != 1:
        raise DevelopmentSubsetError("subset family has inconsistent bindings")
    payload: dict[str, object] = {
        "format_version": RECEIPT_FORMAT,
        "permission_name": PERMISSION_NAME,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_split": "train",
        "parent_authorization_sha256": next(iter(parent_hashes)),
        "policy_sha256": next(iter(policy_hashes)),
        "seed": next(iter(seeds)),
        "selection_domain": SELECTION_DOMAIN,
        "implementation_source_sha256": dict(
            sorted(implementation_source_sha256.items())
        ),
        "subsets": [
            {
                "requested_size": value.requested_size,
                "authorization_manifest_sha256": value.authorization_sha256,
                "subset_manifest_sha256": value.subset_sha256,
                "decision_sha256": value.decision.decision_sha256,
                "instance_membership_sha256": membership_sha256(
                    item.instance_id for item in value.instances
                ),
            }
            for value in subsets
        ],
        "score_or_label_fields_accessed": False,
        "validation_or_test_data_accessed": False,
        "confirmatory_authority_granted": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _strict_json_object(payload_bytes: bytes, label: str) -> dict[str, Any]:
    if not isinstance(payload_bytes, bytes) or not payload_bytes:
        raise DevelopmentSubsetError(f"{label} must be nonempty bytes")
    if payload_bytes.startswith(b"\xef\xbb\xbf"):
        raise DevelopmentSubsetError(f"{label} must not contain a BOM")
    try:
        text = payload_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise DevelopmentSubsetError(f"{label} must be ASCII JSON") from error

    def reject_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DevelopmentSubsetError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DevelopmentSubsetError(
                    f"{label} contains non-finite JSON constant {token}"
                )
            ),
        )
    except json.JSONDecodeError as error:
        raise DevelopmentSubsetError(f"{label} is not valid JSON") from error
    if type(value) is not dict:
        raise DevelopmentSubsetError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if type(value) is not dict:
        raise DevelopmentSubsetError(f"{label} must be a plain object")
    actual = set(value)
    if actual != expected:
        raise DevelopmentSubsetError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _plain_list(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        raise DevelopmentSubsetError(f"{label} must be a JSON array")
    return value


def _pairs(value: Any, label: str) -> tuple[tuple[str, str], ...]:
    raw = _plain_list(value, label)
    pairs: list[tuple[str, str]] = []
    for index, pair in enumerate(raw):
        if type(pair) is not list or len(pair) != 2:
            raise DevelopmentSubsetError(f"{label}[{index}] must be a pair")
        pairs.append(
            (
                _identifier(pair[0], f"{label}[{index}][0]"),
                _identifier(pair[1], f"{label}[{index}][1]"),
            )
        )
    return tuple(pairs)


def _identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or _IDENTIFIER_RE.fullmatch(value) is None
    ):
        raise DevelopmentSubsetError(f"{label} is not a canonical identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DevelopmentSubsetError(f"{label} must be a lowercase SHA-256")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DevelopmentSubsetError(f"{label} must be a nonnegative integer")
    if value > (1 << 53) - 1:
        raise DevelopmentSubsetError(f"{label} exceeds the interoperable range")
    return value


def _sizes(values: Sequence[int], *, maximum: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise DevelopmentSubsetError("sizes must be a nonempty integer sequence")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DevelopmentSubsetError("sizes must contain positive integers")
        if value > maximum:
            raise DevelopmentSubsetError("requested subset exceeds the inventory")
        result.append(value)
    if len(set(result)) != len(result):
        raise DevelopmentSubsetError("sizes contains duplicates")
    return tuple(sorted(result))
