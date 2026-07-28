"""Independent standard-library audit of an OAS corpus plan."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AUDIT_FORMAT = "oas_unpaired_corpus_plan_independent_audit_v1"
PLAN_FORMAT = "oas_unpaired_corpus_plan_v1"
SPEC_FORMAT = "oas_unpaired_corpus_plan_spec_v1"
STRATA = (
    "human_heavy",
    "human_light",
    "nonhuman_heavy",
    "nonhuman_light",
)


class IndependentCorpusAuditError(RuntimeError):
    """Report a fail-closed disagreement in the independent audit."""


def canonical_json_bytes(payload: object) -> bytes:
    """Encode finite, sorted ASCII JSON without importing the planner."""
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise IndependentCorpusAuditError("Payload is not canonical JSON") from error
    return (encoded + "\n").encode("ascii")


def file_sha256(path: Path) -> str:
    """Hash a regular non-symlink file."""
    if not path.is_file() or path.is_symlink():
        raise IndependentCorpusAuditError("Audit input must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ascii_json(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    """Load an exact byte-anchored ASCII JSON document."""
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_file_sha256:
        raise IndependentCorpusAuditError("Audit input file digest differs")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndependentCorpusAuditError("Audit input is not ASCII JSON") from error
    if not isinstance(value, dict):
        raise IndependentCorpusAuditError("Audit input root must be an object")
    return value


def _digest(namespace: str, *parts: str) -> str:
    """Recompute one domain-separated SHA-256 ordering key."""
    return hashlib.sha256("\x00".join((namespace, *parts)).encode("utf-8")).hexdigest()


def _stratum(entry: Mapping[str, Any]) -> str:
    """Independently classify species and chain from authority metadata."""
    metadata = entry.get("metadata")
    if not isinstance(metadata, Mapping):
        raise IndependentCorpusAuditError("Authority entry metadata is invalid")
    organism = metadata.get("Organism")
    chain = metadata.get("Chain")
    if not isinstance(organism, str) or chain not in {"Heavy", "Light"}:
        raise IndependentCorpusAuditError("Authority organism or chain is invalid")
    species = "human" if organism == "human" else "nonhuman"
    return f"{species}_{chain.lower()}"


def _study_counts(
    data_units: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Independently aggregate study capacities."""
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen: set[str] = set()
    for unit in data_units:
        relative = unit.get("relative_unit")
        study = unit.get("study")
        count = unit.get("unique_sequences")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or not isinstance(study, str)
            or not study
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise IndependentCorpusAuditError("Authority data-unit identity is invalid")
        seen.add(relative)
        result[study][_stratum(unit)] += count
    plain = {study: dict(strata) for study, strata in result.items()}
    return plain, {study: sum(strata.values()) for study, strata in plain.items()}


def _validation_studies(
    data_units: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    authority_payload_sha256: str,
) -> dict[str, str]:
    """Recompute the cohort-constrained whole-study validation selection."""
    counts, totals = _study_counts(data_units)
    validation = spec["validation"]
    selected: dict[str, str] = {}
    for cohort in validation["cohorts"]:
        candidates: list[str] = []
        for study, total in totals.items():
            if study in selected:
                continue
            if not (
                validation["minimum_eligible_study_records"]
                <= total
                <= validation["maximum_eligible_study_records"]
            ):
                continue
            requirements = cohort["minimum_stratum_records"]
            if all(counts[study].get(key, 0) >= value for key, value in requirements.items()):
                candidates.append(study)
        candidates.sort(
            key=lambda study: (
                _digest(
                    spec["namespace"],
                    authority_payload_sha256,
                    "validation_study",
                    cohort["name"],
                    study,
                ),
                study,
            )
        )
        if len(candidates) < cohort["count"]:
            raise IndependentCorpusAuditError("Validation cohort is infeasible")
        for study in candidates[: cohort["count"]]:
            selected[study] = cohort["name"]
    return selected


def _allocate(
    capacities: Mapping[str, int],
    target: int,
    maximum_share: Fraction,
    minimum_cap: int,
    namespace: str,
) -> dict[str, int]:
    """Recompute capped square-root allocation in an independent code path."""
    available = {key: value for key, value in capacities.items() if value > 0}
    common_cap = max(
        target * maximum_share.numerator // maximum_share.denominator,
        minimum_cap,
    )
    caps = {key: min(value, common_cap) for key, value in available.items()}
    if target > sum(caps.values()):
        raise IndependentCorpusAuditError("Independent allocation is infeasible")
    output = {key: 0 for key in available}
    live = set(available)
    unassigned = target
    with localcontext() as context:
        context.prec = 60
        roots = {key: Decimal(value).sqrt() for key, value in available.items()}
        while unassigned:
            denominator = sum(roots[key] for key in live)
            quotas = {
                key: Decimal(unassigned) * roots[key] / denominator for key in live
            }
            at_cap = [
                key for key in live if quotas[key] >= caps[key] - output[key]
            ]
            if at_cap:
                for key in sorted(at_cap):
                    delta = caps[key] - output[key]
                    output[key] += delta
                    unassigned -= delta
                    live.remove(key)
                continue
            integer_parts = {key: int(quotas[key]) for key in live}
            for key, value in integer_parts.items():
                output[key] += value
                unassigned -= value
            if unassigned:
                ordered = sorted(
                    live,
                    key=lambda key: (
                        -(quotas[key] - integer_parts[key]),
                        _digest(namespace, "largest_remainder", key),
                        key,
                    ),
                )
                for key in ordered[:unassigned]:
                    output[key] += 1
                unassigned = 0
    return {key: value for key, value in output.items() if value}


def _unit_selection(
    entries: Iterable[Mapping[str, Any]],
    requested: int,
    *,
    namespace: str,
    split: str,
    stratum: str,
    study: str,
) -> list[dict[str, Any]]:
    """Independently reconstruct full and terminal partial data-unit selection."""
    candidates = sorted(
        entries,
        key=lambda unit: (
            _digest(namespace, "data_unit", unit["relative_unit"]),
            unit["relative_unit"],
        ),
    )
    output: list[dict[str, Any]] = []
    remaining = requested
    for unit in candidates:
        if not remaining:
            break
        available = unit["unique_sequences"]
        if not available:
            continue
        selected = min(remaining, available)
        fraction = Fraction(selected, available)
        output.append(
            {
                "canonical_url": unit["canonical_url"],
                "expected_selected_records": selected,
                "relative_unit": unit["relative_unit"],
                "selection_fraction_denominator": fraction.denominator,
                "selection_fraction_numerator": fraction.numerator,
                "selection_hash_namespace": _digest(
                    namespace,
                    "record_filter",
                    split,
                    stratum,
                    study,
                    unit["relative_unit"],
                ),
                "source_records": available,
                "split": split,
                "stratum": stratum,
                "study": study,
            }
        )
        remaining -= selected
    if remaining:
        raise IndependentCorpusAuditError("Unit selection did not meet its allocation")
    return output


def audit_corpus_plan(
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    manifest_file_sha256: str,
    plan_file_sha256: str,
    spec_file_sha256: str,
    auditor_module_sha256: str,
    auditor_source_sha256: str,
    planner_module_sha256: str,
    planner_source_sha256: str,
) -> dict[str, Any]:
    """Rebuild all memberships and return a canonical audit receipt."""
    if spec.get("format_version") != SPEC_FORMAT or plan.get("format_version") != PLAN_FORMAT:
        raise IndependentCorpusAuditError("Plan or specification version differs")
    unsigned_plan = dict(plan)
    recorded_plan_payload_sha256 = unsigned_plan.pop("plan_sha256", None)
    observed_plan_payload_sha256 = hashlib.sha256(
        canonical_json_bytes(unsigned_plan)
    ).hexdigest()
    if recorded_plan_payload_sha256 != observed_plan_payload_sha256:
        raise IndependentCorpusAuditError("Plan payload digest differs")
    if plan.get("spec_file_sha256") != spec_file_sha256:
        raise IndependentCorpusAuditError("Plan does not bind the specification bytes")
    if plan.get("planner_module_sha256") != planner_module_sha256:
        raise IndependentCorpusAuditError("Plan does not bind the planner module")
    if plan.get("planner_source_sha256") != planner_source_sha256:
        raise IndependentCorpusAuditError("Plan does not bind the planner controller")
    authority_spec = spec["authority"]
    if manifest.get("manifest_sha256") != authority_spec["manifest_payload_sha256"]:
        raise IndependentCorpusAuditError("Authority payload digest differs")
    data_units = manifest.get("data_units")
    if not isinstance(data_units, list):
        raise IndependentCorpusAuditError("Authority data_units are absent")
    authority_studies = {unit["study"] for unit in data_units}
    if (
        len(data_units) != authority_spec["expected_data_unit_count"]
        or len(authority_studies) != authority_spec["expected_study_count"]
        or sum(unit["unique_sequences"] for unit in data_units)
        != authority_spec["expected_unique_sequence_count"]
    ):
        raise IndependentCorpusAuditError("Authority count receipt differs")

    exclusions = {entry["study"] for entry in spec.get("excluded_authority_studies", [])}
    if not exclusions.issubset(authority_studies):
        raise IndependentCorpusAuditError("Excluded study is not authoritative")
    planning_units = [unit for unit in data_units if unit["study"] not in exclusions]
    independently_selected_validation = _validation_studies(
        planning_units,
        spec,
        manifest["manifest_sha256"],
    )
    recorded_validation = {
        entry["study"]: entry["cohort"] for entry in plan["validation_studies"]
    }
    if independently_selected_validation != recorded_validation:
        raise IndependentCorpusAuditError("Validation study selection differs")

    study_strata, _ = _study_counts(planning_units)
    units_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for unit in planning_units:
        units_by_key[(unit["study"], _stratum(unit))].append(unit)
    default_share = Fraction(
        spec["study_allocation"]["maximum_share_numerator"],
        spec["study_allocation"]["maximum_share_denominator"],
    )
    minimum_cap = spec["study_allocation"]["minimum_study_cap_records"]
    overrides = spec["study_allocation"].get("stratum_maximum_share_overrides", {})
    expected_units: list[dict[str, Any]] = []
    expected_training_allocations: dict[str, dict[str, int]] = {}
    maximum_observed_study_share: dict[str, dict[str, int]] = {}
    for stratum in STRATA:
        target = spec["training_expected_prefilter_records"][stratum]
        share = default_share
        if stratum in overrides:
            share = Fraction(
                overrides[stratum]["numerator"],
                overrides[stratum]["denominator"],
            )
        capacities = {
            study: values.get(stratum, 0)
            for study, values in study_strata.items()
            if study not in independently_selected_validation
        }
        allocations = _allocate(
            capacities,
            target,
            share,
            minimum_cap,
            f"{spec['namespace']}:{manifest['manifest_sha256']}:train:{stratum}",
        )
        expected_training_allocations[stratum] = dict(sorted(allocations.items()))
        maximum_value = max(allocations.values())
        maximum_observed_study_share[stratum] = {
            "denominator": target,
            "numerator": maximum_value,
        }
        for study, requested in sorted(allocations.items()):
            expected_units.extend(
                _unit_selection(
                    units_by_key[(study, stratum)],
                    requested,
                    namespace=f"{spec['namespace']}:{manifest['manifest_sha256']}",
                    split="train",
                    stratum=stratum,
                    study=study,
                )
            )
    if plan.get("training_study_allocations") != expected_training_allocations:
        raise IndependentCorpusAuditError("Training study allocation differs")

    expected_validation_allocations: dict[str, dict[str, int]] = {}
    validation_target = spec["validation"]["sample_records_per_study"]
    for study in sorted(independently_selected_validation):
        capacities = study_strata[study]
        target = min(validation_target, sum(capacities.values()))
        allocations = _allocate(
            capacities,
            target,
            Fraction(1, 1),
            target,
            f"{spec['namespace']}:{manifest['manifest_sha256']}:validation:{study}",
        )
        expected_validation_allocations[study] = dict(sorted(allocations.items()))
        for stratum, requested in sorted(allocations.items()):
            expected_units.extend(
                _unit_selection(
                    units_by_key[(study, stratum)],
                    requested,
                    namespace=f"{spec['namespace']}:{manifest['manifest_sha256']}",
                    split="validation",
                    stratum=stratum,
                    study=study,
                )
            )
    if plan.get("validation_study_allocations") != expected_validation_allocations:
        raise IndependentCorpusAuditError("Validation allocation differs")
    expected_units.sort(key=lambda unit: (unit["split"], unit["relative_unit"]))
    if plan.get("selected_data_units") != expected_units:
        raise IndependentCorpusAuditError("Selected data-unit membership differs")
    if plan.get("record_selection") != spec.get("record_selection"):
        raise IndependentCorpusAuditError("Record-selection semantics differ")
    if plan.get("excluded_authority_studies") != spec.get("excluded_authority_studies"):
        raise IndependentCorpusAuditError("Excluded-study receipt differs")
    if plan.get("claim_boundary") != {
        "downstream_heldout_accessed": False,
        "downstream_metrics_used": False,
        "expected_prefilter_counts_are_not_admitted_counts": True,
        "sabdab2_heldout_accessed": False,
        "training_started": False,
    }:
        raise IndependentCorpusAuditError("Plan claim boundary is not closed")

    train_expected = sum(
        unit["expected_selected_records"]
        for unit in expected_units
        if unit["split"] == "train"
    )
    validation_expected = sum(
        unit["expected_selected_records"]
        for unit in expected_units
        if unit["split"] == "validation"
    )
    receipt = {
        "audit_result": "passed",
        "auditor_module_sha256": auditor_module_sha256,
        "auditor_source_sha256": auditor_source_sha256,
        "authority_data_unit_count": len(data_units),
        "authority_manifest_file_sha256": manifest_file_sha256,
        "authority_manifest_payload_sha256": manifest["manifest_sha256"],
        "authority_study_count": len(authority_studies),
        "authority_unique_sequence_count": sum(
            unit["unique_sequences"] for unit in data_units
        ),
        "excluded_studies": sorted(exclusions),
        "format_version": AUDIT_FORMAT,
        "maximum_observed_training_study_share": maximum_observed_study_share,
        "plan_file_sha256": plan_file_sha256,
        "plan_payload_sha256": observed_plan_payload_sha256,
        "planner_module_sha256": planner_module_sha256,
        "planner_source_sha256": planner_source_sha256,
        "selected_data_unit_count": len(expected_units),
        "source_records_requiring_download": sum(
            unit["source_records"] for unit in expected_units
        ),
        "spec_file_sha256": spec_file_sha256,
        "test_access": {
            "downstream_heldout_accessed": False,
            "downstream_metrics_used": False,
            "sabdab2_heldout_accessed": False,
        },
        "training_expected_prefilter_records": train_expected,
        "validation_expected_prefilter_records": validation_expected,
        "validation_studies": sorted(independently_selected_validation),
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    return receipt
