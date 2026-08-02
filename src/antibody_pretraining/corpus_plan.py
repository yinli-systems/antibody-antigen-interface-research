"""Deterministic, loss-independent planning for the OAS pretraining corpus."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .oas_authority import canonical_json_bytes


PLAN_FORMAT = "oas_unpaired_corpus_plan_v1"
SPEC_FORMAT = "oas_unpaired_corpus_plan_spec_v1"
STRATA = (
    "human_heavy",
    "human_light",
    "nonhuman_heavy",
    "nonhuman_light",
)


class CorpusPlanError(RuntimeError):
    """Report a fail-closed corpus-planning error."""


def file_sha256(path: Path) -> str:
    """Hash a regular file without following a symbolic link."""
    if not path.is_file() or path.is_symlink():
        raise CorpusPlanError("Input must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan_spec(path: Path) -> tuple[dict[str, Any], str]:
    """Load and validate an ASCII corpus-plan specification."""
    payload = path.read_bytes()
    try:
        spec = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusPlanError("Corpus-plan specification must be ASCII JSON") from error
    validate_plan_spec(spec)
    return spec, hashlib.sha256(payload).hexdigest()


def _positive_integer(value: Any, label: str) -> int:
    """Return a positive integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorpusPlanError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    """Return a non-negative integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusPlanError(f"{label} must be a non-negative integer")
    return value


def validate_plan_spec(spec: Mapping[str, Any]) -> None:
    """Validate the fields that authorize deterministic corpus selection."""
    if spec.get("format_version") != SPEC_FORMAT:
        raise CorpusPlanError("Unexpected corpus-plan specification version")
    if not isinstance(spec.get("namespace"), str) or not spec["namespace"]:
        raise CorpusPlanError("Corpus-plan namespace must be nonempty text")
    authority = spec.get("authority")
    if not isinstance(authority, Mapping) or authority.get("source_class") != "unpaired":
        raise CorpusPlanError("The corpus plan requires an unpaired OAS authority")
    for field in (
        "expected_data_unit_count",
        "expected_study_count",
        "expected_unique_sequence_count",
    ):
        _positive_integer(authority.get(field), f"authority.{field}")
    for field in ("manifest_file_sha256", "manifest_payload_sha256"):
        value = authority.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise CorpusPlanError(f"authority.{field} must be a SHA-256")
    targets = spec.get("training_expected_prefilter_records")
    if not isinstance(targets, Mapping) or tuple(sorted(targets)) != tuple(sorted(STRATA)):
        raise CorpusPlanError("Training targets must define exactly the four strata")
    for stratum in STRATA:
        _positive_integer(targets[stratum], f"training target {stratum}")
    allocation = spec.get("study_allocation")
    if not isinstance(allocation, Mapping):
        raise CorpusPlanError("study_allocation must be an object")
    numerator = _positive_integer(
        allocation.get("maximum_share_numerator"),
        "maximum_share_numerator",
    )
    denominator = _positive_integer(
        allocation.get("maximum_share_denominator"),
        "maximum_share_denominator",
    )
    if numerator >= denominator:
        raise CorpusPlanError("Maximum study share must be strictly below one")
    _nonnegative_integer(
        allocation.get("minimum_study_cap_records"),
        "minimum_study_cap_records",
    )
    overrides = allocation.get("stratum_maximum_share_overrides", {})
    if not isinstance(overrides, Mapping):
        raise CorpusPlanError("stratum_maximum_share_overrides must be an object")
    for stratum, override in overrides.items():
        if stratum not in STRATA or not isinstance(override, Mapping):
            raise CorpusPlanError("Invalid stratum maximum-share override")
        override_numerator = _positive_integer(
            override.get("numerator"), f"{stratum} override numerator"
        )
        override_denominator = _positive_integer(
            override.get("denominator"), f"{stratum} override denominator"
        )
        if override_numerator >= override_denominator:
            raise CorpusPlanError("Stratum maximum-share override must be below one")
        if not isinstance(override.get("reason"), str) or not override["reason"]:
            raise CorpusPlanError("Stratum maximum-share override needs a reason")
    exclusions = spec.get("excluded_authority_studies", [])
    if not isinstance(exclusions, list):
        raise CorpusPlanError("excluded_authority_studies must be a list")
    excluded_names: set[str] = set()
    for exclusion in exclusions:
        if not isinstance(exclusion, Mapping):
            raise CorpusPlanError("Excluded study entry must be an object")
        study = exclusion.get("study")
        reason = exclusion.get("reason")
        if (
            not isinstance(study, str)
            or not study
            or study in excluded_names
            or not isinstance(reason, str)
            or not reason
        ):
            raise CorpusPlanError("Excluded studies require unique names and reasons")
        excluded_names.add(study)
    validation = spec.get("validation")
    if not isinstance(validation, Mapping):
        raise CorpusPlanError("validation must be an object")
    minimum = _positive_integer(
        validation.get("minimum_eligible_study_records"),
        "minimum_eligible_study_records",
    )
    maximum = _positive_integer(
        validation.get("maximum_eligible_study_records"),
        "maximum_eligible_study_records",
    )
    if minimum > maximum:
        raise CorpusPlanError("Validation study bounds are reversed")
    _positive_integer(
        validation.get("sample_records_per_study"),
        "sample_records_per_study",
    )
    cohorts = validation.get("cohorts")
    if not isinstance(cohorts, list) or not cohorts:
        raise CorpusPlanError("At least one validation cohort is required")
    names: set[str] = set()
    for cohort in cohorts:
        if not isinstance(cohort, Mapping):
            raise CorpusPlanError("Validation cohort must be an object")
        name = cohort.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise CorpusPlanError("Validation cohort names must be unique")
        names.add(name)
        _positive_integer(cohort.get("count"), f"cohort {name} count")
        requirements = cohort.get("minimum_stratum_records")
        if not isinstance(requirements, Mapping) or not requirements:
            raise CorpusPlanError(f"cohort {name} has no stratum requirements")
        for stratum, count in requirements.items():
            if stratum not in STRATA:
                raise CorpusPlanError(f"Unknown validation stratum {stratum}")
            _positive_integer(count, f"cohort {name} requirement {stratum}")
    if spec.get("test_access") != {
        "downstream_heldout_accessed": False,
        "downstream_metrics_used": False,
        "sabdab2_heldout_accessed": False,
    }:
        raise CorpusPlanError("The corpus-plan test-access boundary is not closed")


def classify_stratum(entry: Mapping[str, Any]) -> str:
    """Map an OAS authority entry to one species-by-chain stratum."""
    metadata = entry.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CorpusPlanError("Authority entry metadata must be an object")
    organism = metadata.get("Organism")
    chain = metadata.get("Chain")
    if not isinstance(organism, str) or not organism:
        raise CorpusPlanError("Authority entry has no organism")
    if chain not in {"Heavy", "Light"}:
        raise CorpusPlanError(f"Unsupported chain class: {chain!r}")
    species = "human" if organism == "human" else "nonhuman"
    return f"{species}_{chain.lower()}"


def _digest(namespace: str, *parts: str) -> str:
    """Return a domain-separated deterministic ordering digest."""
    framed = "\x00".join((namespace, *parts)).encode("utf-8")
    return hashlib.sha256(framed).hexdigest()


def _authority_counts(
    data_units: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Aggregate study-by-stratum capacities from raw manifest entries."""
    study_strata: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    seen_units: set[str] = set()
    for entry in data_units:
        relative_unit = entry.get("relative_unit")
        study = entry.get("study")
        count = entry.get("unique_sequences")
        if not isinstance(relative_unit, str) or not relative_unit:
            raise CorpusPlanError("Authority entry has no relative_unit")
        if relative_unit in seen_units:
            raise CorpusPlanError(f"Duplicate authority unit: {relative_unit}")
        seen_units.add(relative_unit)
        if not isinstance(study, str) or not study:
            raise CorpusPlanError("Authority entry has no study")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise CorpusPlanError("Authority unique_sequences must be non-negative")
        study_strata[study][classify_stratum(entry)] += count
    study_totals = {
        study: sum(strata.values()) for study, strata in study_strata.items()
    }
    return {study: dict(values) for study, values in study_strata.items()}, study_totals


def select_validation_studies(
    data_units: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    *,
    authority_manifest_sha256: str,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Select whole validation studies by frozen metadata constraints and hashes."""
    study_strata, study_totals = _authority_counts(data_units)
    validation = spec["validation"]
    lower = validation["minimum_eligible_study_records"]
    upper = validation["maximum_eligible_study_records"]
    selected: dict[str, str] = {}
    for cohort in validation["cohorts"]:
        name = cohort["name"]
        requirements = cohort["minimum_stratum_records"]
        candidates = []
        for study, total in study_totals.items():
            if study in selected or not lower <= total <= upper:
                continue
            if any(
                study_strata[study].get(stratum, 0) < required
                for stratum, required in requirements.items()
            ):
                continue
            candidates.append(study)
        ordered = sorted(
            candidates,
            key=lambda study: (
                _digest(
                    spec["namespace"],
                    authority_manifest_sha256,
                    "validation_study",
                    name,
                    study,
                ),
                study,
            ),
        )
        required_count = cohort["count"]
        if len(ordered) < required_count:
            raise CorpusPlanError(
                f"Validation cohort {name} has only {len(ordered)} eligible studies"
            )
        for study in ordered[:required_count]:
            selected[study] = name
    return selected, study_strata


def allocate_capped_sqrt(
    capacities: Mapping[str, int],
    target: int,
    *,
    maximum_share: Fraction,
    minimum_cap: int,
    namespace: str,
) -> dict[str, int]:
    """Allocate an exact target by square-root weights under deterministic caps."""
    _positive_integer(target, "allocation target")
    if maximum_share <= 0 or maximum_share > 1:
        raise CorpusPlanError("maximum_share must be in (0, 1]")
    if isinstance(minimum_cap, bool) or not isinstance(minimum_cap, int) or minimum_cap < 0:
        raise CorpusPlanError("minimum_cap must be a non-negative integer")
    normalized: dict[str, int] = {}
    for key, value in capacities.items():
        if not isinstance(key, str) or not key:
            raise CorpusPlanError("Allocation keys must be nonempty text")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CorpusPlanError("Allocation capacities must be non-negative")
        if value:
            normalized[key] = value
    if target > sum(normalized.values()):
        raise CorpusPlanError("Allocation target exceeds available records")
    share_cap = (target * maximum_share.numerator) // maximum_share.denominator
    common_cap = max(share_cap, minimum_cap)
    hard_caps = {key: min(value, common_cap) for key, value in normalized.items()}
    if target > sum(hard_caps.values()):
        raise CorpusPlanError("Study contribution cap makes the allocation infeasible")

    allocations = {key: 0 for key in normalized}
    remaining = target
    active = set(normalized)
    with localcontext() as context:
        context.prec = 60
        weights = {key: Decimal(normalized[key]).sqrt() for key in normalized}
        while remaining:
            if not active:
                raise CorpusPlanError("Allocation exhausted active studies")
            weight_sum = sum(weights[key] for key in active)
            ideals = {
                key: Decimal(remaining) * weights[key] / weight_sum for key in active
            }
            saturated = [
                key
                for key in active
                if ideals[key] >= hard_caps[key] - allocations[key]
            ]
            if saturated:
                for key in sorted(saturated):
                    addition = hard_caps[key] - allocations[key]
                    allocations[key] += addition
                    remaining -= addition
                    active.remove(key)
                continue
            floors = {key: int(ideals[key]) for key in active}
            floor_total = sum(floors.values())
            for key, value in floors.items():
                allocations[key] += value
            remaining -= floor_total
            if remaining:
                remainder_order = sorted(
                    active,
                    key=lambda key: (
                        -(ideals[key] - floors[key]),
                        _digest(namespace, "largest_remainder", key),
                        key,
                    ),
                )
                for key in remainder_order[:remaining]:
                    allocations[key] += 1
                remaining = 0
    if sum(allocations.values()) != target:
        raise CorpusPlanError("Allocation does not sum to the requested target")
    for key, value in allocations.items():
        if value > hard_caps[key] or value > normalized[key]:
            raise CorpusPlanError("Allocation exceeded a hard capacity")
    return {key: value for key, value in allocations.items() if value}


def _select_units(
    entries: Iterable[Mapping[str, Any]],
    requested_records: int,
    *,
    namespace: str,
    split: str,
    stratum: str,
    study: str,
) -> list[dict[str, Any]]:
    """Choose full data units and at most one exact-fraction terminal unit."""
    ordered = sorted(
        entries,
        key=lambda entry: (
            _digest(namespace, "data_unit", entry["relative_unit"]),
            entry["relative_unit"],
        ),
    )
    remaining = requested_records
    selected: list[dict[str, Any]] = []
    for entry in ordered:
        if not remaining:
            break
        available = entry["unique_sequences"]
        if not available:
            continue
        numerator = min(remaining, available)
        denominator = available
        fraction = Fraction(numerator, denominator)
        selected.append(
            {
                "canonical_url": entry["canonical_url"],
                "expected_selected_records": numerator,
                "relative_unit": entry["relative_unit"],
                "selection_fraction_denominator": fraction.denominator,
                "selection_fraction_numerator": fraction.numerator,
                "selection_hash_namespace": _digest(
                    namespace,
                    "record_filter",
                    split,
                    stratum,
                    study,
                    entry["relative_unit"],
                ),
                "source_records": available,
                "split": split,
                "stratum": stratum,
                "study": study,
            }
        )
        remaining -= numerator
    if remaining:
        raise CorpusPlanError(
            f"Study {study} lacks {remaining} records for {split}/{stratum}"
        )
    return selected


def build_corpus_plan(
    authority_manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    spec_file_sha256: str,
    planner_module_sha256: str,
    planner_source_sha256: str,
) -> dict[str, Any]:
    """Build the canonical data-unit and expected-record pretraining plan."""
    validate_plan_spec(spec)
    data_units = authority_manifest.get("data_units")
    if not isinstance(data_units, list):
        raise CorpusPlanError("Authority manifest has no data_units list")
    authority = spec["authority"]
    summary = authority_manifest.get("summary")
    if not isinstance(summary, Mapping):
        raise CorpusPlanError("Authority manifest has no summary")
    observed = {
        "expected_data_unit_count": len(data_units),
        "expected_study_count": len(set(entry["study"] for entry in data_units)),
        "expected_unique_sequence_count": sum(
            entry["unique_sequences"] for entry in data_units
        ),
    }
    for field, value in observed.items():
        if authority[field] != value:
            raise CorpusPlanError(f"Authority count differs for {field}")
    manifest_sha256 = authority_manifest.get("manifest_sha256")
    if manifest_sha256 != authority["manifest_payload_sha256"]:
        raise CorpusPlanError("Authority payload digest differs from the frozen spec")

    excluded_studies = {
        entry["study"] for entry in spec.get("excluded_authority_studies", [])
    }
    observed_studies = {entry["study"] for entry in data_units}
    if not excluded_studies.issubset(observed_studies):
        raise CorpusPlanError("An excluded study is absent from the authority manifest")
    planning_data_units = [
        entry for entry in data_units if entry["study"] not in excluded_studies
    ]
    validation_studies, study_strata = select_validation_studies(
        planning_data_units,
        spec,
        authority_manifest_sha256=manifest_sha256,
    )
    entries_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for entry in planning_data_units:
        entries_by_key[(entry["study"], classify_stratum(entry))].append(entry)

    maximum_share = Fraction(
        spec["study_allocation"]["maximum_share_numerator"],
        spec["study_allocation"]["maximum_share_denominator"],
    )
    minimum_cap = spec["study_allocation"]["minimum_study_cap_records"]
    maximum_share_overrides = spec["study_allocation"].get(
        "stratum_maximum_share_overrides", {}
    )
    training_allocations: dict[str, dict[str, int]] = {}
    selected_units: list[dict[str, Any]] = []
    for stratum in STRATA:
        capacities = {
            study: values.get(stratum, 0)
            for study, values in study_strata.items()
            if study not in validation_studies
        }
        target = spec["training_expected_prefilter_records"][stratum]
        stratum_maximum_share = maximum_share
        if stratum in maximum_share_overrides:
            override = maximum_share_overrides[stratum]
            stratum_maximum_share = Fraction(
                override["numerator"], override["denominator"]
            )
        try:
            allocations = allocate_capped_sqrt(
                capacities,
                target,
                maximum_share=stratum_maximum_share,
                minimum_cap=minimum_cap,
                namespace=f"{spec['namespace']}:{manifest_sha256}:train:{stratum}",
            )
        except CorpusPlanError as error:
            raise CorpusPlanError(
                f"Training allocation failed for {stratum}: {error}"
            ) from error
        training_allocations[stratum] = dict(sorted(allocations.items()))
        for study, requested in sorted(allocations.items()):
            selected_units.extend(
                _select_units(
                    entries_by_key[(study, stratum)],
                    requested,
                    namespace=f"{spec['namespace']}:{manifest_sha256}",
                    split="train",
                    stratum=stratum,
                    study=study,
                )
            )

    validation_allocations: dict[str, dict[str, int]] = {}
    sample_per_study = spec["validation"]["sample_records_per_study"]
    for study, cohort in sorted(validation_studies.items()):
        capacities = study_strata[study]
        target = min(sample_per_study, sum(capacities.values()))
        allocations = allocate_capped_sqrt(
            capacities,
            target,
            maximum_share=Fraction(1, 1),
            minimum_cap=target,
            namespace=f"{spec['namespace']}:{manifest_sha256}:validation:{study}",
        )
        validation_allocations[study] = dict(sorted(allocations.items()))
        for stratum, requested in sorted(allocations.items()):
            selected_units.extend(
                _select_units(
                    entries_by_key[(study, stratum)],
                    requested,
                    namespace=f"{spec['namespace']}:{manifest_sha256}",
                    split="validation",
                    stratum=stratum,
                    study=study,
                )
            )

    selected_units.sort(key=lambda unit: (unit["split"], unit["relative_unit"]))
    plan = {
        "authority": {
            "data_unit_count": len(data_units),
            "manifest_file_sha256": authority["manifest_file_sha256"],
            "manifest_payload_sha256": manifest_sha256,
            "study_count": observed["expected_study_count"],
            "unique_sequence_count": observed["expected_unique_sequence_count"],
        },
        "claim_boundary": {
            "downstream_heldout_accessed": False,
            "downstream_metrics_used": False,
            "expected_prefilter_counts_are_not_admitted_counts": True,
            "sabdab2_heldout_accessed": False,
            "training_started": False,
        },
        "format_version": PLAN_FORMAT,
        "excluded_authority_studies": spec.get("excluded_authority_studies", []),
        "namespace": spec["namespace"],
        "planner_module_sha256": planner_module_sha256,
        "planner_source_sha256": planner_source_sha256,
        "record_selection": spec["record_selection"],
        "selected_data_units": selected_units,
        "spec_file_sha256": spec_file_sha256,
        "training_expected_prefilter_records": spec[
            "training_expected_prefilter_records"
        ],
        "training_study_allocations": training_allocations,
        "validation_studies": [
            {"cohort": cohort, "study": study}
            for study, cohort in sorted(validation_studies.items())
        ],
        "validation_study_allocations": validation_allocations,
    }
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    validate_corpus_plan(plan)
    return plan


def validate_corpus_plan(plan: Mapping[str, Any]) -> None:
    """Validate split isolation, fractions, expected counts, and payload digest."""
    if plan.get("format_version") != PLAN_FORMAT:
        raise CorpusPlanError("Unexpected corpus-plan version")
    expected_digest = plan.get("plan_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise CorpusPlanError("Corpus plan has no valid payload digest")
    unsigned = dict(plan)
    del unsigned["plan_sha256"]
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != expected_digest:
        raise CorpusPlanError("Corpus-plan payload digest differs")
    validation_studies = {
        entry["study"] for entry in plan.get("validation_studies", [])
    }
    seen_units: set[str] = set()
    observed_training = defaultdict(int)
    for unit in plan.get("selected_data_units", []):
        relative_unit = unit.get("relative_unit")
        if not isinstance(relative_unit, str) or relative_unit in seen_units:
            raise CorpusPlanError("Corpus plan has a missing or duplicate data unit")
        seen_units.add(relative_unit)
        numerator = unit.get("selection_fraction_numerator")
        denominator = unit.get("selection_fraction_denominator")
        if (
            isinstance(numerator, bool)
            or isinstance(denominator, bool)
            or not isinstance(numerator, int)
            or not isinstance(denominator, int)
            or numerator <= 0
            or numerator > denominator
        ):
            raise CorpusPlanError("Corpus plan contains an invalid selection fraction")
        if unit["split"] == "train":
            if unit["study"] in validation_studies:
                raise CorpusPlanError("A validation study appears in training")
            observed_training[unit["stratum"]] += unit[
                "expected_selected_records"
            ]
        elif unit["split"] == "validation":
            if unit["study"] not in validation_studies:
                raise CorpusPlanError("A validation unit has no held-out study")
        else:
            raise CorpusPlanError("Corpus-plan split must be train or validation")
    if dict(observed_training) != plan["training_expected_prefilter_records"]:
        raise CorpusPlanError("Training unit expectations do not match frozen targets")
