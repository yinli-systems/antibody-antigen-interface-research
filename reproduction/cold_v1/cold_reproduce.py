#!/usr/bin/env python3
"""Regenerate every headline number from frozen inputs, in a clean process.

    python3 reproduction/cold_v1/cold_reproduce.py --runs /path/to/runs \
        --out reproduction/cold_v1/output

This does no science. It answers one question: can a process that has never seen
this project's prose rebuild its numbers from the artifacts and configs alone?

**What it may read**: run directories, evaluation receipts, training histories,
prediction parquets, extraction receipts, split manifests, and the analysis
scripts in `scripts/`.

**What it may not read**: any README, `EVIDENCE_INVENTORY.md`,
`CLAIMS_LEDGER.md`, any hand-written table, or any generated figure. Those are
where a number could have been transcribed, and a reproduction that consults them
proves nothing. The comparison targets are the frozen **JSON receipts**, which
are outputs of the same scripts and not prose.

**Pass rule, fixed before running**: every numeric field must agree with the
frozen receipt to within 1e-10, or exactly where the value is integral. Every
bootstrap interval must agree exactly, which is only possible because every
bootstrap in this project takes a fixed seed. A single mismatch is a FAIL and is
reported with both values; it is not rounded away and not explained here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

TOLERANCE = 1e-10

REPOSITORY = Path(__file__).resolve().parent.parent.parent
FORBIDDEN = ("README.md", "EVIDENCE_INVENTORY.md", "CLAIMS_LEDGER.md",
             "REPRODUCIBILITY_NOTES.md", "BENCHMARK.md", "STAGE_THREE.md",
             "PART_TWO.md", "RUN_CONTRACT.md")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_digest(root: Path) -> str:
    """Content digest of a directory, independent of filesystem order."""
    if not root.exists():
        return "absent"
    outer = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            outer.update(str(path.relative_to(root)).encode())
            outer.update(bytes.fromhex(digest(path)))
            count += 1
    return f"{outer.hexdigest()}:{count}"


def run(script: str, arguments: list[str], out: Path, name: str) -> dict:
    """Run one analysis script and capture what it wrote."""
    target = out / f"{name}.json"
    command = [sys.executable, str(REPOSITORY / "scripts" / script), *arguments,
               "--json", str(target)]
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True,
                            cwd=str(REPOSITORY), env=None)
    record = {
        "script": script,
        "script_sha256": digest(REPOSITORY / "scripts" / script),
        "arguments": arguments,
        "returncode": result.returncode,
        "seconds": round(time.time() - started, 1),
        "output": str(target),
        "output_sha256": digest(target) if target.is_file() else None,
        "stderr_tail": result.stderr.strip().splitlines()[-3:] if result.stderr else [],
    }
    return record


def compare(cold: Path, frozen: Path, path: str = "") -> list[dict]:
    """Every numeric leaf, compared. Returns the mismatches."""
    if not cold.is_file():
        return [{"field": path or "<file>", "cold": "missing", "frozen": "present"}]
    if not frozen.is_file():
        return [{"field": path or "<file>", "cold": "present", "frozen": "missing"}]
    return _walk(json.loads(cold.read_text()), json.loads(frozen.read_text()), "")


def _walk(a, b, path: str) -> list[dict]:
    bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                # A new field in the cold run is not a mismatch of a number; it is
                # reported so the receipt shows the shape changed.
                bad.append({"field": f"{path}.{key}", "cold": key in a, "frozen": key in b,
                            "kind": "field present in only one"})
                continue
            bad += _walk(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            bad.append({"field": path, "cold": len(a), "frozen": len(b),
                        "kind": "length"})
        else:
            for index, (x, y) in enumerate(zip(a, b)):
                bad += _walk(x, y, f"{path}[{index}]")
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            bad.append({"field": path, "cold": a, "frozen": b, "kind": "bool"})
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(float(a) - float(b)) > TOLERANCE:
            bad.append({"field": path, "cold": a, "frozen": b,
                        "delta": float(a) - float(b), "kind": "numeric"})
    elif a != b:
        bad.append({"field": path, "cold": a, "frozen": b, "kind": "value"})
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True,
                        help="root holding the frozen run directories")
    parser.add_argument("--asep", type=Path, required=True,
                        help="root holding the AsEP-side runs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=None)
    arguments = parser.parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)

    runs, asep = arguments.runs, arguments.asep
    results = Path(REPOSITORY / "results")

    # name -> (script, args, frozen receipt to compare against)
    plan = [
        ("baseline_margin", "aggregate_baseline_margin.py",
         ["--run-root", str(runs / "esm_if1_only")],
         results / "baseline_margin_replication_v1" / "report.json"),
        ("attribution_aligned_vs_shuffled", "aggregate_representation_ablation.py",
         ["--run-root", str(runs), "--control", "esm_if1_shuffled",
          "--treatment", "esm_if1_only", "--metric", "auprc", "--scope", "macro"],
         None),
        ("corruption_esm_if1", "aggregate_corruption_curve.py",
         ["--aligned-root", str(runs / "esm_if1_only"),
          "--corrupt-root", str(asep / "corrupt" / "esm_if1"), "--label", "esm_if1"],
         results / "correspondence_corruption_curve_v1" / "curve_esm_if1.json"),
        ("corruption_antiberty", "aggregate_corruption_curve.py",
         ["--aligned-root", str(asep / "antiberty_ladder" / "k0"),
          "--corrupt-root", str(asep / "antiberty_ladder"),
          "--label", "antiberty_sabdab2", "--aligned-subdir", ""],
         results / "correspondence_corruption_curve_v1" / "curve_antiberty_sabdab2.json"),
        ("gradient_step_trace", "aggregate_step_trace.py",
         ["--cohort", f"asep={asep}/steptrace/asep_seed*",
          "--cohort", f"sabdab2={asep}/steptrace/sabdab2_seed*"],
         results / "gradient_alignment_dynamics_v1" / "step_trace_summary.json"),
        ("selection_stage_two", "aggregate_selection_metric_confirmation.py",
         ["--arm", f"128={asep}/seltotal/d128", "--arm", f"256={asep}/seltotal/d256"],
         results / "checkpoint_selection_confirmation_v1" / "stage_two_group_validation.json"),
        ("ratio_test_confirmation", "ratio_test_confirmation.py",
         ["--run-root", str(asep / "ratio_seltotal" / "d256")],
         results / "checkpoint_selection_confirmation_v1" / "ratio_test_confirmation.json"),
        ("complex_heterogeneity", "paired_complex_variance.py",
         ["--arm", f"aligned={runs}/esm_if1_only",
          "--arm", f"shuffled={runs}/esm_if1_shuffled"],
         results / "representation_attribution_v1" / "variance_aligned_vs_shuffled.json"),
    ]

    steps, mismatches = [], {}
    for name, script, args, frozen in plan:
        print(f"[{name}] {script}", flush=True)
        record = run(script, args, arguments.out, name)
        if frozen is not None:
            bad = compare(arguments.out / f"{name}.json", frozen)
            record["compared_against"] = str(frozen.relative_to(REPOSITORY))
            record["compared_against_sha256"] = digest(frozen) if frozen.is_file() else None
            record["mismatches"] = len(bad)
            if bad:
                mismatches[name] = bad[:40]
        else:
            record["compared_against"] = None
            record["mismatches"] = None
        steps.append(record)
        status = ("no frozen receipt" if record["mismatches"] is None
                  else "MATCH" if record["mismatches"] == 0
                  else f"MISMATCH ({record['mismatches']})")
        print(f"    rc={record['returncode']}  {record['seconds']}s  {status}", flush=True)

    failures = [s for s in steps if s["returncode"] != 0] + \
               [s for s in steps if s["mismatches"]]
    receipt = {
        "format_version": "cold_reproduction_receipt_v1",
        "verdict": "PASS" if not failures else "FAIL",
        "tolerance": TOLERANCE,
        "pass_rule": (
            "every numeric leaf within 1e-10 of the frozen receipt, every "
            "bootstrap interval exact, every script exiting zero"
        ),
        "forbidden_inputs": list(FORBIDDEN),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "analysis_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPOSITORY),
            capture_output=True, text=True).stdout.strip(),
        "analysis_tree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(REPOSITORY),
            capture_output=True, text=True).stdout.strip()),
        "scripts_digest": tree_digest(REPOSITORY / "scripts"),
        "src_digest": tree_digest(REPOSITORY / "src"),
        "output_digest": tree_digest(arguments.out),
        "steps": steps,
        "mismatches": mismatches,
    }
    target = arguments.receipt or (arguments.out.parent / "cold_reproduction_receipt.json")
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print()
    print(f"VERDICT: {receipt['verdict']}")
    for name, bad in mismatches.items():
        print(f"  {name}: {len(bad)} mismatching fields")
        for entry in bad[:5]:
            print(f"    {entry}")
    print(f"receipt: {target}")
    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
