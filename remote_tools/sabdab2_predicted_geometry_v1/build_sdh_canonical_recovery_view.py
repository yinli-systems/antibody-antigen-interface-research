#!/usr/bin/env python3
"""Build an immutable canonical view over completed SD-H AF3 outputs.

The view never edits source predictions.  It resolves each frozen plan record to
exactly one complete AF3 result directory, normalizes only filenames required by
the downstream materializer, and freezes source paths plus content hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> str:
    payload = canonical_bytes(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def only(paths: list[Path], description: str, directory: Path) -> Path:
    if len(paths) != 1:
        raise ValueError(
            f"expected one {description} in {directory}, observed {len(paths)}"
        )
    return paths[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdh-plan", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--failed-prepare-stderr", type=Path, required=True)
    parser.add_argument("--failed-prepare-job-id", required=True)
    parser.add_argument("--af3-recovery-job-id", required=True)
    parser.add_argument("--expected-record-count", type=int, default=332)
    parser.add_argument("--expected-recovered-count", type=int, default=1)
    return parser.parse_args()


def complete_candidate(directory: Path) -> dict[str, Any] | None:
    models = sorted(directory.glob("*_model.cif"))
    data_files = sorted(directory.glob("*_data.json"))
    ranking_files = sorted(directory.glob("*ranking_scores.csv"))
    if len(models) != 1 or len(data_files) != 1 or len(ranking_files) != 1:
        return None

    with ranking_files[0].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"seed", "sample", "ranking_score"}.issubset(rows[0]):
        return None
    try:
        selected = max(rows, key=lambda row: float(row["ranking_score"]))
        seed = int(selected["seed"])
        sample = int(selected["sample"])
        ranking_score = float(selected["ranking_score"])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ranking_score):
        return None

    selected_dir = directory / f"seed-{seed}_sample-{sample}"
    selected_model = sorted(selected_dir.glob("*model.cif"))
    if len(selected_model) != 1:
        return None
    model_sha256 = sha256_file(models[0])
    selected_model_sha256 = sha256_file(selected_model[0])
    seed_directories = sorted(
        path for path in directory.glob("seed-*_sample-*") if path.is_dir()
    )
    if len(seed_directories) != 25:
        return None
    return {
        "directory": directory,
        "model": models[0],
        "data": data_files[0],
        "ranking": ranking_files[0],
        "seed_directories": seed_directories,
        "selected_seed": seed,
        "selected_sample": sample,
        "selected_ranking_score": ranking_score,
        "model_sha256": model_sha256,
        "selected_model_sha256": selected_model_sha256,
    }


def symlink(source: Path, target: Path) -> None:
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    args = parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    plan_sha256 = sha256_file(args.sdh_plan)
    plan = json.loads(args.sdh_plan.read_text(encoding="utf-8"))
    if (
        plan.get("allowed_roles") != ["train", "validation"]
        or plan.get("test_access") is not False
        or plan.get("temporal_holdout_access") is not False
        or plan.get("task_labels_loaded") is not False
        or len(plan.get("records", [])) != args.expected_record_count
    ):
        raise ValueError("SD-H input plan authority differs")

    args.output_root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    exact_count = 0
    recovered_count = 0
    for record in plan["records"]:
        batch_index = int(record["batch_index"])
        name = record["name"]
        source_results = (
            args.source_root / "batches" / f"batch-{batch_index:03d}" / "results"
        )
        candidates = sorted(
            path
            for path in source_results.iterdir()
            if path.is_dir()
            and (path.name == name or path.name.startswith(name + "_"))
        )
        complete = [
            result
            for path in candidates
            if (result := complete_candidate(path)) is not None
        ]
        if len(complete) != 1:
            raise ValueError(
                f"{name} resolves to {len(complete)} complete candidates: "
                f"{[path.name for path in candidates]}"
            )
        selected = complete[0]
        selected_directory = selected["directory"]
        source_kind = (
            "exact_canonical_name"
            if selected_directory.name == name
            else "timestamp_suffixed_recovery"
        )
        if source_kind == "exact_canonical_name":
            exact_count += 1
        else:
            recovered_count += 1

        target = (
            args.output_root
            / "batches"
            / f"batch-{batch_index:03d}"
            / "results"
            / name
        )
        target.mkdir(parents=True)
        symlink(selected["model"], target / selected["model"].name)
        symlink(selected["data"], target / selected["data"].name)
        symlink(selected["ranking"], target / "ranking_scores.csv")
        for seed_directory in selected["seed_directories"]:
            symlink(seed_directory, target / seed_directory.name)

        records.append(
            {
                "batch_index": batch_index,
                "name": name,
                "sequence_sha256": record["sequence_sha256"],
                "source_kind": source_kind,
                "source_directory": str(selected_directory.resolve()),
                "canonical_directory": str(target),
                "model_path": str(selected["model"].resolve()),
                "model_sha256": selected["model_sha256"],
                "data_path": str(selected["data"].resolve()),
                "data_sha256": sha256_file(selected["data"]),
                "ranking_path": str(selected["ranking"].resolve()),
                "ranking_sha256": sha256_file(selected["ranking"]),
                "selected_seed": selected["selected_seed"],
                "selected_sample": selected["selected_sample"],
                "selected_ranking_score": selected["selected_ranking_score"],
                "selected_model_sha256": selected["selected_model_sha256"],
            }
        )

    if recovered_count != args.expected_recovered_count:
        raise ValueError(
            f"expected {args.expected_recovered_count} recovered records, "
            f"observed {recovered_count}"
        )
    manifest = {
        "format_version": "sabdab2_sdh_canonical_recovery_view_v1",
        "status": "PASS_CANONICAL_SDH_RECOVERY_VIEW",
        "selection_policy": (
            "exactly one complete candidate per frozen plan record; source content "
            "is unchanged and exposed through read-only canonical symlinks"
        ),
        "sdh_plan_path": str(args.sdh_plan),
        "sdh_plan_sha256": plan_sha256,
        "source_root": str(args.source_root),
        "canonical_root": str(args.output_root),
        "record_count": len(records),
        "exact_canonical_name_count": exact_count,
        "timestamp_suffixed_recovery_count": recovered_count,
        "failed_prepare_job_id": args.failed_prepare_job_id,
        "failed_prepare_stderr_path": str(args.failed_prepare_stderr),
        "failed_prepare_stderr_sha256": sha256_file(args.failed_prepare_stderr),
        "af3_recovery_job_id": args.af3_recovery_job_id,
        "experimental_bound_complex_coordinates_read": False,
        "cross_partner_geometry_used": False,
        "task_labels_loaded": False,
        "test_access": False,
        "temporal_holdout_access": False,
        "records": records,
    }
    manifest_path = args.output_root / "CANONICAL_RECOVERY_VIEW_MANIFEST.json"
    manifest_sha256 = write_json(manifest_path, manifest)
    receipt = {
        "format_version": "sabdab2_sdh_canonical_recovery_view_receipt_v1",
        "status": "PASS_CANONICAL_SDH_RECOVERY_VIEW",
        "manifest_sha256": manifest_sha256,
        "record_count": len(records),
        "exact_canonical_name_count": exact_count,
        "timestamp_suffixed_recovery_count": recovered_count,
        "source_predictions_modified": False,
        "test_access": False,
        "temporal_holdout_access": False,
    }
    receipt_path = args.output_root / "CANONICAL_RECOVERY_VIEW_RECEIPT.json"
    receipt_sha256 = write_json(receipt_path, receipt)
    for directory, directory_names, file_names in os.walk(args.output_root):
        for filename in file_names:
            path = Path(directory) / filename
            if not path.is_symlink():
                path.chmod(0o444)
        Path(directory).chmod(0o555)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "manifest_sha256": manifest_sha256,
                "receipt_sha256": receipt_sha256,
                "record_count": len(records),
                "exact_count": exact_count,
                "recovered_count": recovered_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
