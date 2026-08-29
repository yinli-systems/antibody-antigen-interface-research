#!/usr/bin/env python3
"""Produce the FrameContact decision deliverable: one table, three figures, one verdict.

The report answers the three architecture questions in order and refuses to
attribute a difference to a factor when more than one factor changed.  It reads
either in-flight Slurm stdout logs or finished development result JSON, so it
can be run while the arrays are still going.

Example, reading the two live arrays::

    python scripts/framecontact_decision_report.py \\
        --live fc-res-head:1345042:framecontact_unary_only,framecontact_contact_marginal,framecontact_contact_hybrid \\
        --live fc-distogram:1345108:framecontact_distogram,framecontact_distogram_triangle \\
        --log-root /ssd/.../slurm-logs \\
        --seeds 17 42 73 101 137 \\
        --output-root reports/decision-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from interaction_first.decision_matrix import (  # noqa: E402
    ARM_FACTORS,
    DECISION_CHAIN,
    EPOCH_NOISE_WINDOW,
    MAIN_TABLE_METRICS,
    ComparisonError,
    SeedRun,
    build_main_table,
    epoch_noise,
    evaluate_phase,
    load_completed_run,
    load_live_log,
    matched_epoch,
    summarize_arm,
    trajectory,
)


REPORT_FORMAT = "framecontact_decision_report_v1"

# The three figures the ablation is meant to produce.
TRAJECTORY_FIGURES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "contact_precision_recall",
        "Contact precision and recall at a fixed raw-logit threshold",
        ("contact_precision", "contact_recall"),
    ),
    (
        "sparsity_versus_ranking",
        "Predicted-positive ratio against threshold-free contact AUPRC",
        ("contact_predicted_positive_ratio", "contact_pooled_auprc"),
    ),
    (
        "partner_margin",
        "Worst-direction partner-specificity margin",
        ("partner_margin_worst_direction",),
    ),
)


def _parse_live_specification(value: str) -> tuple[str, str, tuple[str, ...]]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "live specification must be PREFIX:JOB_ID:ARM[,ARM...]"
        )
    prefix, job_id, arms = parts
    arm_names = tuple(name for name in arms.split(",") if name)
    if not arm_names:
        raise argparse.ArgumentTypeError("live specification lists no arms")
    return prefix, job_id, arm_names


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="append",
        type=_parse_live_specification,
        default=[],
        help=(
            "In-flight array as PREFIX:JOB_ID:ARM[,ARM...]. Arms are listed in "
            "the same order as the sbatch experiments array, which fixes the "
            "task-index mapping."
        ),
    )
    parser.add_argument("--log-root", type=Path, help="Directory of Slurm stdout logs")
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=[],
        help="Directory holding {experiment}/seed-{seed}/FRAMECONTACT_DEVELOPMENT_RESULT.json",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[17, 42, 73, 101, 137]
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Pin the comparison epoch; defaults to the latest epoch all arms reached",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure rendering when matplotlib is unavailable",
    )
    return parser.parse_args()


def _collect_live(
    log_root: Path,
    specifications: Sequence[tuple[str, str, tuple[str, ...]]],
    seeds: Sequence[int],
) -> tuple[dict[str, dict[int, SeedRun]], list[str]]:
    runs_by_arm: dict[str, dict[int, SeedRun]] = {}
    warnings: list[str] = []
    for prefix, job_id, arms in specifications:
        for arm_index, arm in enumerate(arms):
            for seed_index, seed in enumerate(seeds):
                task_index = arm_index * len(seeds) + seed_index
                path = log_root / f"{prefix}-{job_id}_{task_index}.out"
                if not path.exists():
                    warnings.append(f"missing log {path}")
                    continue
                try:
                    run = load_live_log(path, experiment=arm, seed=seed)
                except ComparisonError as error:
                    warnings.append(f"{path}: {error}")
                    continue
                runs_by_arm.setdefault(arm, {})[seed] = run
    return runs_by_arm, warnings


def _collect_completed(
    run_roots: Sequence[Path], seeds: Sequence[int]
) -> tuple[dict[str, dict[int, SeedRun]], list[str]]:
    runs_by_arm: dict[str, dict[int, SeedRun]] = {}
    warnings: list[str] = []
    for run_root in run_roots:
        if not run_root.exists():
            warnings.append(f"missing run root {run_root}")
            continue
        for experiment_directory in sorted(run_root.iterdir()):
            if not experiment_directory.is_dir():
                continue
            for seed in seeds:
                result = (
                    experiment_directory
                    / f"seed-{seed}"
                    / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
                )
                if not result.exists():
                    continue
                try:
                    run = load_completed_run(result)
                except (ComparisonError, KeyError, json.JSONDecodeError) as error:
                    warnings.append(f"{result}: {error}")
                    continue
                runs_by_arm.setdefault(run.experiment, {})[run.seed] = run
    return runs_by_arm, warnings


def _format_value(name: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) < 0.001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _main_table_markdown(table: Mapping[str, Any]) -> str:
    metrics = list(table["metrics"])
    header = (
        "| Arm | Residue head | Pair target | Triangle | Gauge | Contrastive | Seeds | "
        + " | ".join(metrics)
        + " |"
    )
    separator = "|" + "---|" * (7 + len(metrics))
    lines = [header, separator]
    for row in table["rows"]:
        factors = row["factors"] or {}
        cells = [
            f"`{row['experiment']}`",
            factors.get("residue_aggregation", "?"),
            factors.get("pair_target", "?"),
            factors.get("triangle", "?"),
            factors.get("pair_gauge", "?"),
            factors.get("residue_contrastive", "?"),
            str(len(row["seeds"])),
        ]
        for metric in metrics:
            entry = row["metrics"].get(metric)
            cells.append(_format_value(metric, None if entry is None else entry["mean"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _phase_markdown(phase_result: Mapping[str, Any]) -> str:
    lines = [f"### {phase_result['phase']}", "", phase_result["question"], ""]
    if phase_result["status"] == "unavailable":
        lines.append(f"Not evaluable: {phase_result['reason']}")
        return "\n".join(lines) + "\n"
    lines.append(f"Status: {phase_result['status']}. {phase_result['claim_boundary']}")
    for caveat in phase_result.get("caveats", []):
        lines.append("")
        lines.append(f"> Caveat: {caveat}")
    if phase_result["missing_candidates"]:
        lines.append(
            "Still missing: "
            + ", ".join(f"`{name}`" for name in phase_result["missing_candidates"])
            + "."
        )
    lines.append("")
    lines.append(
        "| Candidate | Changed factor | Epoch | Seeds | Primary delta | Up/Down | "
        "Sign-test p | Epoch noise | Beats noise | Contact verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for comparison in phase_result["comparisons"]:
        primary = comparison["primary"]
        sign = primary["sign_test"] if primary else None
        stability = comparison.get("epoch_stability") or {}
        beats = stability.get("exceeds_noise")
        lines.append(
            "| `{candidate}` | {factor} | {epoch} | {seeds} | {delta} | {up}/{down} | "
            "{p} | {noise} | {beats} | {verdict} |".format(
                candidate=comparison["candidate"],
                factor=comparison["factor"],
                epoch=comparison["epoch"],
                seeds=len(comparison["seeds"]),
                delta="n/a" if primary is None else f"{primary['mean_delta']:+.6f}",
                up="?" if sign is None else sign["positive"],
                down="?" if sign is None else sign["negative"],
                p="n/a" if sign is None else f"{sign['p_value']:.4f}",
                noise=(
                    "n/a"
                    if stability.get("noise") is None
                    else f"{stability['noise']:.6f}"
                ),
                beats={True: "yes", False: "NO", None: "n/a"}[beats],
                verdict=comparison["sparsity_verdict"]["verdict"],
            )
        )
    lines.append("")
    for comparison in phase_result["comparisons"]:
        verdict = comparison["sparsity_verdict"]
        lines.append(
            f"- `{comparison['candidate']}`: {verdict['explanation']}"
        )
        stability = comparison.get("epoch_stability") or {}
        if stability.get("exceeds_noise") is False:
            lines.append(
                f"  The primary metric moves by {stability['noise']:.6f} between "
                f"neighbouring epochs of one run, which is larger than the "
                f"{abs(stability['mean_delta']):.6f} difference between arms. "
                "Any verdict here would be a property of the pinned epoch."
            )
        retention = comparison.get("recall_retention")
        if retention and retention.get("minimum") is not None:
            lines.append(
                f"  Recall retention minimum {retention['minimum']:.4f} against a "
                f"{retention['floor']:.2f} floor: "
                + ("met." if retention["meets_floor"] else "NOT met.")
            )
    lines.append("")
    lines.append(f"**Decision.** {phase_result['decision']['statement']}")
    lines.append("")
    return "\n".join(lines)


def _stability_audit(
    runs_by_arm: Mapping[str, Mapping[int, SeedRun]], *, epoch: int
) -> dict[str, Any]:
    """How far each headline metric drifts between neighbouring epochs.

    This runs even when no comparison is admissible yet, because a metric that
    wanders more between epochs than between arms cannot support any verdict,
    and that is worth knowing before the remaining arms finish.
    """

    audited = (
        "partner_margin_worst_direction",
        "contact_pooled_auprc",
        "contact_p_at_lmin_over_10",
        "epitope_pooled_auprc",
        "paratope_pooled_auprc",
    )
    rows: dict[str, Any] = {}
    for arm, runs in sorted(runs_by_arm.items()):
        entry: dict[str, Any] = {}
        for metric in audited:
            noise = epoch_noise(runs, metric, epoch=epoch)
            summary = summarize_arm(runs, epoch=epoch, metrics=(metric,))
            value = summary["metrics"].get(metric)
            entry[metric] = {
                "mean_at_epoch": None if value is None else value["mean"],
                "epoch_noise": noise,
                "noise_exceeds_value": (
                    None
                    if noise is None or value is None
                    else noise > abs(value["mean"])
                ),
            }
        rows[arm] = entry
    return {
        "definition": (
            "epoch_noise is the mean within-run standard deviation of the metric "
            f"over the trailing {EPOCH_NOISE_WINDOW} epochs. When it exceeds the "
            "metric's own value, single-epoch readings of that metric are not "
            "interpretable."
        ),
        "window": EPOCH_NOISE_WINDOW,
        "epoch": epoch,
        "by_arm": rows,
    }


def _write_trajectory_data(
    runs_by_arm: Mapping[str, Mapping[int, SeedRun]], output_root: Path
) -> str:
    """Persist the figure data so plots can be rebuilt without matplotlib."""
    destination = output_root / "TRAJECTORIES.csv"
    metrics = sorted({metric for _, _, group in TRAJECTORY_FIGURES for metric in group})
    lines = ["arm,metric,epoch,mean,seed_count"]
    for arm, runs in sorted(runs_by_arm.items()):
        for metric in metrics:
            series = trajectory(runs, metric)
            for index, epoch in enumerate(series["epochs"]):
                lines.append(
                    f"{arm},{metric},{epoch},{series['mean'][index]!r},"
                    f"{series['seed_count'][index]}"
                )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination.name


def _render_figures(
    runs_by_arm: Mapping[str, Mapping[int, SeedRun]], output_root: Path
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError:
        return []

    written: list[str] = []
    for key, title, metrics in TRAJECTORY_FIGURES:
        figure, axes = pyplot.subplots(
            1, len(metrics), figsize=(6.5 * len(metrics), 4.2), squeeze=False
        )
        drew_anything = False
        for column, metric in enumerate(metrics):
            axis = axes[0][column]
            for arm, runs in sorted(runs_by_arm.items()):
                series = trajectory(runs, metric)
                if not series["epochs"]:
                    continue
                axis.plot(
                    series["epochs"], series["mean"], marker="o", markersize=2.5, label=arm
                )
                drew_anything = True
            axis.set_xlabel("epoch")
            axis.set_ylabel(metric)
            axis.grid(alpha=0.3)
            axis.legend(fontsize=7)
        if not drew_anything:
            pyplot.close(figure)
            continue
        figure.suptitle(title)
        figure.tight_layout()
        destination = output_root / f"{key}.png"
        figure.savefig(destination, dpi=160)
        pyplot.close(figure)
        written.append(destination.name)
    return written


def main() -> None:
    arguments = _arguments()
    if not arguments.live and not arguments.run_root:
        raise SystemExit("supply at least one --live specification or --run-root")
    if arguments.live and arguments.log_root is None:
        raise SystemExit("--live requires --log-root")

    runs_by_arm: dict[str, dict[int, SeedRun]] = {}
    warnings: list[str] = []
    if arguments.run_root:
        completed, completed_warnings = _collect_completed(
            arguments.run_root, arguments.seeds
        )
        runs_by_arm.update(completed)
        warnings.extend(completed_warnings)
    if arguments.live:
        live, live_warnings = _collect_live(
            arguments.log_root, arguments.live, arguments.seeds
        )
        warnings.extend(live_warnings)
        # A finished run supersedes the same arm/seed read from a live log.
        for arm, runs in live.items():
            for seed, run in runs.items():
                runs_by_arm.setdefault(arm, {}).setdefault(seed, run)

    if not runs_by_arm:
        raise SystemExit("no runs were loaded; check --log-root and --run-root")

    unregistered = sorted(set(runs_by_arm) - set(ARM_FACTORS))
    if unregistered:
        raise SystemExit(
            "unregistered arms cannot be placed in the factor space: "
            + ", ".join(unregistered)
        )

    flattened = [run for runs in runs_by_arm.values() for run in runs.values()]
    epoch = arguments.epoch if arguments.epoch is not None else matched_epoch(flattened)
    table = build_main_table(runs_by_arm, epoch=epoch)
    phases = [evaluate_phase(phase, runs_by_arm, epoch=epoch) for phase in DECISION_CHAIN]

    output_root = arguments.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    figures = [] if arguments.no_figures else _render_figures(runs_by_arm, output_root)
    trajectory_data = _write_trajectory_data(runs_by_arm, output_root)
    stability = _stability_audit(runs_by_arm, epoch=epoch)

    payload = {
        "format_version": REPORT_FORMAT,
        "claim_boundary": (
            "Development validation only. No SAbDab2 frozen test or AsEP official "
            "test was read. Comparisons are pinned to a common epoch and are "
            "refused unless exactly one registered factor differs."
        ),
        "matched_epoch": epoch,
        "seeds": list(arguments.seeds),
        "arms": {
            arm: {
                "factors": ARM_FACTORS[arm].as_mapping(),
                "seeds": sorted(runs),
                "completed_seeds": sorted(
                    seed for seed, run in runs.items() if run.completed
                ),
                "last_epoch_by_seed": {
                    str(seed): run.last_epoch for seed, run in sorted(runs.items())
                },
            }
            for arm, runs in sorted(runs_by_arm.items())
        },
        "main_table": table,
        "decision_chain": phases,
        "metric_stability_audit": stability,
        "figures": figures,
        "trajectory_data": trajectory_data,
        "warnings": warnings,
    }
    (output_root / "DECISION_REPORT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    document = [
        "# FrameContact architecture decision report",
        "",
        f"Matched epoch: {epoch}. Seeds: {', '.join(str(s) for s in arguments.seeds)}.",
        "",
        payload["claim_boundary"],
        "",
        "## Main table",
        "",
        _main_table_markdown(table),
        "",
        "## Decision chain",
        "",
    ]
    document.extend(_phase_markdown(phase) for phase in phases)
    document.append("## Metric stability audit")
    document.append("")
    document.append(stability["definition"])
    document.append("")
    document.append("| Arm | Metric | Value at epoch | Epoch noise | Noise exceeds value |")
    document.append("|---|---|---|---|---|")
    for arm, metrics in stability["by_arm"].items():
        for metric, entry in metrics.items():
            document.append(
                "| `{arm}` | {metric} | {value} | {noise} | {flag} |".format(
                    arm=arm,
                    metric=metric,
                    value=(
                        "n/a"
                        if entry["mean_at_epoch"] is None
                        else f"{entry['mean_at_epoch']:+.6f}"
                    ),
                    noise=(
                        "n/a"
                        if entry["epoch_noise"] is None
                        else f"{entry['epoch_noise']:.6f}"
                    ),
                    flag={True: "YES", False: "no", None: "n/a"}[
                        entry["noise_exceeds_value"]
                    ],
                )
            )
    document.append("")
    if figures:
        document.append("## Figures")
        document.append("")
        document.extend(f"![{name}]({name})" for name in figures)
        document.append("")
    if warnings:
        document.append("## Warnings")
        document.append("")
        document.extend(f"- {warning}" for warning in warnings)
        document.append("")
    (output_root / "DECISION_REPORT.md").write_text(
        "\n".join(document), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "matched_epoch": epoch,
                "arms": sorted(runs_by_arm),
                "figures": figures,
                "warning_count": len(warnings),
                "decisions": {
                    phase["phase"]: phase.get("decision", {}).get(
                        "statement", phase.get("reason")
                    )
                    for phase in phases
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
