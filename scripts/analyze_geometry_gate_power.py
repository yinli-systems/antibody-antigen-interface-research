#!/usr/bin/env python3
"""Report the power and decision-rule properties of the geometry superiority gate.

This is a design analysis of already-audited evidence. It reads three committed
receipts, binds them by SHA-256, and reports what the gate could and could not
have detected. It does not retrain, re-score, re-evaluate, or revise any
performance number, and it does not authorize model selection or test access.

    python3 scripts/analyze_geometry_gate_power.py
    python3 scripts/analyze_geometry_gate_power.py --json > receipt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.geometry_gate_power import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_TARGET_POWER,
    SIGN_CRITERION_THRESHOLD,
    ReplicateSummary,
    clopper_pearson_lower_bound,
    holm_adjusted,
    linear_dependence_residual,
    minimum_detectable_effect,
    minimum_unanimous_replicates,
    one_sided_t_power,
    replicates_for_power,
)

FORMAT_VERSION = "sabdab2_geometry_gate_power_analysis_v1"
RESULTS = REPOSITORY_ROOT / "results" / "sabdab2_structure_self_attention"
MULTISEED = RESULTS / "ab03_a100_replay_verification.json"
A100 = RESULTS / "a100_checkpoint_paired_statistics_summary.json"
RTX5090 = RESULTS / "paired_statistics_summary.json"
CANDIDATE = "distance_direction"
REFERENCE = "topology_only"
ENDPOINTS = ("macro_auprc", "paratope_auprc", "epitope_auprc")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _adjustment_audit() -> list[dict]:
    """Check each recorded ``holm_adjusted_one_sided_p`` against Holm and Bonferroni.

    Holm is uniformly at least as powerful as Bonferroni, so a receipt that names
    Holm while carrying a Bonferroni value understates the evidence it reports.
    """
    audit = []
    for path, label in ((A100, "AB03 A100 final checkpoints"),
                        (RTX5090, "RTX 5090 final checkpoints")):
        block = _load(path)["comparisons"]["distance_direction_minus_topology_only"]
        raw = {key: entry["one_sided_component_sign_flip_p"] for key, entry in block.items()}
        recorded = block["joint_macro_auprc"].get("holm_adjusted_one_sided_p")
        if recorded is None:
            continue
        holm = holm_adjusted(raw)["joint_macro_auprc"]
        bonferroni = min(1.0, len(raw) * raw["joint_macro_auprc"])
        audit.append({
            "receipt": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "label": label,
            "field": "comparisons.distance_direction_minus_topology_only."
                     "joint_macro_auprc.holm_adjusted_one_sided_p",
            "recorded_value": recorded,
            "holm_step_down": holm,
            "bonferroni": bonferroni,
            "matches_holm": abs(recorded - holm) < 1e-12,
            "matches_bonferroni": abs(recorded - bonferroni) < 1e-12,
        })
    return audit


def build_report() -> dict:
    """Assemble the complete design analysis from the committed receipts."""
    multiseed = _load(MULTISEED)["multiseed"]
    a100 = _load(A100)
    rtx = _load(RTX5090)

    comparisons = multiseed["comparisons_vs_topology_only"][CANDIDATE]

    # --- Replicate-level analysis: what three seeds can support -------------
    seeds: dict[str, object] = {}
    for endpoint in ENDPOINTS:
        record = comparisons[endpoint]
        summary = ReplicateSummary.from_differences(record["differences"])
        bound = clopper_pearson_lower_bound(summary.positive_count, summary.count)
        entry: dict[str, object] = {
            "differences": list(summary.differences),
            "mean_difference": summary.mean,
            "sample_sd": summary.sample_sd,
            "positive_count": summary.positive_count,
            "replicate_count": summary.count,
            "observed_effect_size": summary.effect_size,
            "sign_criterion": {
                "point_estimate": summary.positive_count / summary.count,
                "exact_lower_bound": bound,
                "threshold": SIGN_CRITERION_THRESHOLD,
                "significant": bound > SIGN_CRITERION_THRESHOLD,
            },
        }
        if summary.effect_size is not None and summary.effect_size > 0.0:
            entry["power"] = {
                "at_observed_replicates": one_sided_t_power(summary.effect_size, summary.count),
                "at_five_replicates": one_sided_t_power(summary.effect_size, 5),
                "replicates_for_target_power": replicates_for_power(summary.effect_size),
            }
        seeds[endpoint] = entry

    # --- Design properties, independent of the observed effect -------------
    design = {
        "alpha": DEFAULT_ALPHA,
        "target_power": DEFAULT_TARGET_POWER,
        "minimum_unanimous_replicates_for_sign_criterion": minimum_unanimous_replicates(),
        "minimum_detectable_effect_size": {
            str(count): minimum_detectable_effect(count) for count in (3, 5, 8, 10, 20)
        },
    }

    # --- Multiplicity family ------------------------------------------------
    unadjusted = {
        endpoint: a100["comparisons"]["distance_direction_minus_topology_only"][
            {"macro_auprc": "joint_macro_auprc",
             "paratope_auprc": "paratope_auprc",
             "epitope_auprc": "epitope_auprc"}[endpoint]
        ]["one_sided_component_sign_flip_p"]
        for endpoint in ENDPOINTS
    }
    modes = a100["modes"]
    residual = linear_dependence_residual(
        {name: values["joint_macro_auprc"] for name, values in modes.items()},
        [
            {name: values["paratope_auprc"] for name, values in modes.items()},
            {name: values["epitope_auprc"] for name, values in modes.items()},
        ],
    )
    family = {
        "unadjusted_one_sided_p": unadjusted,
        "holm_over_all_three_endpoints": holm_adjusted(unadjusted),
        "holm_over_components_only": holm_adjusted(
            {k: v for k, v in unadjusted.items() if k != "macro_auprc"}
        ),
        "composite_linear_dependence_residual": residual,
        "composite_is_determined_by_components": residual < 1e-12,
        "note": (
            "macro_auprc is the arithmetic mean of paratope_auprc and epitope_auprc. "
            "A family containing all three charges for a hypothesis that cannot vary "
            "independently of the other two."
        ),
        "recorded_adjustment_audit": _adjustment_audit(),
    }

    # --- Two checkpoint audits of the same comparison ------------------------
    def _checkpoint(document: dict, label: str) -> dict:
        block = document["comparisons"]["distance_direction_minus_topology_only"]
        resampling = document.get("resampling", {})
        out = {"label": label, "endpoints": {}}
        for key in ("joint_macro_auprc", "paratope_auprc", "epitope_auprc"):
            record = block[key]
            fraction = record["positive_complex_fraction"]
            instances = resampling.get("validation_instance_count")
            components = resampling.get("independent_endpoint_component_count")
            entry = {
                "mean_difference": record["mean_difference"],
                "bootstrap_ci95": record["bootstrap_ci95"],
                "one_sided_p": record["one_sided_component_sign_flip_p"],
                "positive_complex_fraction": fraction,
            }
            if instances and components:
                entry["win_fraction_lower_bound"] = {
                    "as_reported_over_instances": clopper_pearson_lower_bound(
                        round(fraction * instances), instances
                    ),
                    "over_independent_components": clopper_pearson_lower_bound(
                        round(fraction * components), components
                    ),
                    "instances": instances,
                    "independent_components": components,
                }
            out["endpoints"][key] = entry
        return out

    checkpoints = [_checkpoint(a100, "AB03 A100 final checkpoints"),
                   _checkpoint(rtx, "RTX 5090 final checkpoints")]

    return {
        "format_version": FORMAT_VERSION,
        "claim_boundary": (
            "Design analysis of already-audited development evidence. It reports the "
            "power and decision-rule properties of the intra-chain geometry superiority "
            "gate. It revises no performance number, does not re-run the gate, does not "
            "reverse its recorded outcome, and authorizes no model selection or test "
            "access. The preregistered gate remains failed."
        ),
        "inputs": {
            path.relative_to(REPOSITORY_ROOT).as_posix(): _sha256(path)
            for path in (MULTISEED, A100, RTX5090)
        },
        "comparison": {"candidate": CANDIDATE, "reference": REFERENCE},
        "replicate_analysis": seeds,
        "design_properties": design,
        "multiplicity_family": family,
        "checkpoint_audits": checkpoints,
    }


def _render(report: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"Comparison: {report['comparison']['candidate']} minus {report['comparison']['reference']}")
    add("")

    add("Replicate-level analysis (three A100 seeds)")
    add(f"  {'endpoint':16s} {'mean':>10s} {'sd':>9s} {'d':>7s} {'pos':>5s} "
        f"{'P(A>B) LB':>10s} {'sig':>4s} {'power':>7s} {'n@0.80':>7s}")
    for endpoint, entry in report["replicate_analysis"].items():
        sign = entry["sign_criterion"]
        power = entry.get("power", {})
        effect = entry["observed_effect_size"]
        add(
            f"  {endpoint:16s} {entry['mean_difference']:+10.6f} {entry['sample_sd']:9.6f} "
            f"{(effect if effect is not None else float('nan')):7.3f} "
            f"{entry['positive_count']:>3d}/{entry['replicate_count']:<1d} "
            f"{sign['exact_lower_bound']:10.4f} {'yes' if sign['significant'] else 'no':>4s} "
            f"{power.get('at_observed_replicates', float('nan')):7.3f} "
            f"{str(power.get('replicates_for_target_power', '-')):>7s}"
        )
    add("")

    design = report["design_properties"]
    add("Design properties")
    add(f"  Unanimous replicates needed before the sign criterion is reachable at all: "
        f"{design['minimum_unanimous_replicates_for_sign_criterion']}")
    add(f"  Minimum detectable standardised effect at alpha={design['alpha']}, "
        f"power={design['target_power']}:")
    for count, effect in design["minimum_detectable_effect_size"].items():
        add(f"    n={count:>2s}  d >= {effect:.3f}")
    add("")

    family = report["multiplicity_family"]
    add("Multiplicity family")
    add(f"  composite determined by its components: {family['composite_is_determined_by_components']} "
        f"(max residual {family['composite_linear_dependence_residual']:.2e})")
    add(f"  {'endpoint':16s} {'unadjusted':>11s} {'holm(3)':>9s} {'holm(2)':>9s}")
    for endpoint, value in family["unadjusted_one_sided_p"].items():
        holm3 = family["holm_over_all_three_endpoints"][endpoint]
        holm2 = family["holm_over_components_only"].get(endpoint)
        add(f"  {endpoint:16s} {value:11.6f} {holm3:9.6f} "
            f"{(f'{holm2:.6f}' if holm2 is not None else '-'):>9s}")
    add("")

    audit = family["recorded_adjustment_audit"]
    if audit:
        add("Recorded adjustment audit")
        for entry in audit:
            verdict = ("Holm" if entry["matches_holm"]
                       else "Bonferroni" if entry["matches_bonferroni"] else "neither")
            add(f"  {entry['label']}")
            add(f"    field named Holm holds {entry['recorded_value']:.10f}; "
                f"Holm is {entry['holm_step_down']:.10f}, Bonferroni is {entry['bonferroni']:.10f} "
                f"-> value is {verdict}")
        add("")

    add("Checkpoint audits of the same comparison")
    for checkpoint in report["checkpoint_audits"]:
        add(f"  {checkpoint['label']}")
        for endpoint, entry in checkpoint["endpoints"].items():
            bound = entry.get("win_fraction_lower_bound", {})
            add(f"    {endpoint:20s} mean {entry['mean_difference']:+.6f}  "
                f"p {entry['one_sided_p']:.6f}  win {entry['positive_complex_fraction']:.4f}  "
                f"LB(inst) {bound.get('as_reported_over_instances', float('nan')):.4f}  "
                f"LB(comp) {bound.get('over_independent_components', float('nan')):.4f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit the machine-readable receipt")
    arguments = parser.parse_args(argv)

    report = build_report()
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
