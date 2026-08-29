#!/usr/bin/env python3
"""Freeze a case-exact development-only view over paired-antibody AF3 outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any


ALLOWED_ROLES = frozenset({"train", "validation"})
TIMESTAMP_SUFFIX = re.compile(r"_[0-9]{8}_[0-9]{6}$")


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


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


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
    parser.add_argument("--instance-manifest-csv", type=Path, required=True)
    parser.add_argument("--source-split-csv", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sdh-recovery-manifest", type=Path, required=True)
    parser.add_argument("--sdh-recovery-receipt", type=Path, required=True)
    parser.add_argument("--failed-prepare-receipt", type=Path, required=True)
    parser.add_argument("--expected-development-count", type=int, default=3114)
    parser.add_argument("--expected-paired-count", type=int, default=2224)
    parser.add_argument("--expected-sdh-count", type=int, default=890)
    parser.add_argument("--expected-collision-group-count", type=int, default=7)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def prediction_schema(directory: Path) -> dict[str, Any]:
    model = only(sorted(directory.glob("*_model.cif")), "top model", directory)
    data = only(sorted(directory.glob("*_data.json")), "input JSON", directory)
    ranking = only(
        sorted(directory.glob("*ranking_scores.csv")), "ranking CSV", directory
    )
    with ranking.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"seed", "sample", "ranking_score"}.issubset(rows[0]):
        raise ValueError(f"invalid ranking CSV: {ranking}")
    selected = max(rows, key=lambda row: float(row["ranking_score"]))
    seed = int(selected["seed"])
    sample = int(selected["sample"])
    score = float(selected["ranking_score"])
    if not math.isfinite(score):
        raise ValueError(f"non-finite ranking score: {ranking}")
    seed_directories = sorted(
        path for path in directory.glob("seed-*_sample-*") if path.is_dir()
    )
    if len(seed_directories) != 25:
        raise ValueError(
            f"expected 25 seed/sample directories in {directory}, "
            f"observed {len(seed_directories)}"
        )
    selected_model = only(
        sorted((directory / f"seed-{seed}_sample-{sample}").glob("*model.cif")),
        "selected model",
        directory,
    )
    return {
        "model": model,
        "data": data,
        "ranking": ranking,
        "selected_seed": seed,
        "selected_sample": sample,
        "selected_ranking_score": score,
        "selected_model": selected_model,
        "seed_directories": seed_directories,
    }


def input_identity(data_path: Path) -> tuple[str, tuple[str, ...]]:
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    name = payload.get("name")
    proteins = [entry.get("protein") for entry in payload.get("sequences", [])]
    if not isinstance(name, str) or any(not isinstance(item, dict) for item in proteins):
        raise ValueError(f"invalid AF3 input identity: {data_path}")
    return name, tuple(str(item.get("sequence", "")) for item in proteins)


def symlink(source: Path, target: Path) -> None:
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main() -> None:
    args = parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    split_fields, split_rows = read_csv(args.source_split_csv)
    if not {"instance_id", "role"}.issubset(split_fields):
        raise ValueError("split CSV lacks instance_id/role")
    development_rows = [
        row for row in split_rows if row["role"].strip() in ALLOWED_ROLES
    ]
    role_counts = {
        role: sum(row["role"].strip() == role for row in development_rows)
        for role in sorted(ALLOWED_ROLES)
    }
    if (
        len(development_rows) != args.expected_development_count
        or role_counts != {"train": 2565, "validation": 549}
        or len({row["instance_id"] for row in development_rows})
        != args.expected_development_count
    ):
        raise ValueError("development split authority differs")
    selected_ids = {row["instance_id"] for row in development_rows}

    manifest_fields, manifest_rows = read_csv(args.instance_manifest_csv)
    required = {"instance_id", "type", "vh_seq", "vl_seq"}
    if not required.issubset(manifest_fields):
        raise ValueError("instance manifest fields differ")
    selected_manifest = [
        row for row in manifest_rows if row["instance_id"] in selected_ids
    ]
    if (
        len(selected_manifest) != args.expected_development_count
        or {row["instance_id"] for row in selected_manifest} != selected_ids
    ):
        raise ValueError("instance manifest/development split differ")
    paired_rows = [row for row in selected_manifest if row["type"] != "SD-H"]
    sdh_count = len(selected_manifest) - len(paired_rows)
    if len(paired_rows) != args.expected_paired_count or sdh_count != args.expected_sdh_count:
        raise ValueError("paired/SD-H development counts differ")

    source_index: dict[str, list[Path]] = {}
    for directory in args.source_root.iterdir():
        if not directory.is_dir():
            continue
        base = TIMESTAMP_SUFFIX.sub("", directory.name).lower()
        source_index.setdefault(base, []).append(directory)

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.tmp-", dir=args.output_root.parent
        )
    )
    records: list[dict[str, Any]] = []
    canonical_count = 0
    timestamped_count = 0
    collision_keys: set[str] = set()
    selected_sources: set[Path] = set()
    try:
        for row in sorted(paired_rows, key=lambda item: item["instance_id"]):
            instance_id = row["instance_id"]
            antibody_type = row["type"]
            target_name = f"{instance_id}_{antibody_type}"
            key = target_name.lower()
            candidates = sorted(source_index.get(key, []))
            if not candidates:
                raise ValueError(f"no paired AF3 candidates for {target_name}")

            expected_sequences = (
                row["vh_seq"].strip().upper(),
                row["vl_seq"].strip().upper(),
            )
            if not all(expected_sequences):
                raise ValueError(f"empty paired sequence for {instance_id}")

            candidate_schemas = [prediction_schema(path) for path in candidates]
            if len(candidates) == 1:
                selected_index = 0
                source_json_name = None
            else:
                collision_keys.add(key)
                identities = [input_identity(item["data"]) for item in candidate_schemas]
                matching = [
                    index
                    for index, (name, sequences) in enumerate(identities)
                    if name == target_name and sequences == expected_sequences
                ]
                if len(matching) != 1:
                    raise ValueError(
                        f"case-exact AF3 identity for {target_name} resolves to "
                        f"{len(matching)} candidates"
                    )
                selected_index = matching[0]
                source_json_name = identities[selected_index][0]

            selected_directory = candidates[selected_index]
            selected = candidate_schemas[selected_index]
            if selected_directory in selected_sources:
                raise ValueError(f"paired AF3 source reused: {selected_directory}")
            selected_sources.add(selected_directory)

            target = temporary_root / target_name
            target.mkdir()
            symlink(selected["model"], target / selected["model"].name)
            symlink(selected["data"], target / selected["data"].name)
            symlink(selected["ranking"], target / "ranking_scores.csv")
            for seed_directory in selected["seed_directories"]:
                symlink(seed_directory, target / seed_directory.name)

            timestamped = TIMESTAMP_SUFFIX.search(selected_directory.name) is not None
            if timestamped:
                timestamped_count += 1
            else:
                canonical_count += 1
            records.append(
                {
                    "instance_id": instance_id,
                    "antibody_type": antibody_type,
                    "target_name": target_name,
                    "canonical_directory": str(args.output_root / target_name),
                    "source_directory": str(selected_directory.resolve()),
                    "source_kind": (
                        "timestamped_case_collision_recovery"
                        if timestamped
                        else "canonical_output_directory"
                    ),
                    "candidate_count": len(candidates),
                    "source_json_name": source_json_name,
                    "antibody_sequence_sha256": sequence_sha256(
                        "".join(expected_sequences)
                    ),
                    "model_path": str(selected["model"].resolve()),
                    "model_sha256": sha256_file(selected["model"]),
                    "data_path": str(selected["data"].resolve()),
                    "data_size_bytes": selected["data"].stat().st_size,
                    "ranking_path": str(selected["ranking"].resolve()),
                    "ranking_sha256": sha256_file(selected["ranking"]),
                    "selected_seed": selected["selected_seed"],
                    "selected_sample": selected["selected_sample"],
                    "selected_ranking_score": selected["selected_ranking_score"],
                    "selected_model_sha256": sha256_file(selected["selected_model"]),
                }
            )

        if (
            len(collision_keys) != args.expected_collision_group_count
            or timestamped_count != args.expected_collision_group_count
            or canonical_count + timestamped_count != args.expected_paired_count
            or len(selected_sources) != args.expected_paired_count
        ):
            raise ValueError("case-collision recovery counts differ")

        manifest = {
            "format_version": "sabdab2_paired_case_exact_recovery_view_v2",
            "status": "PASS_PAIRED_CASE_EXACT_RECOVERY_VIEW",
            "selection_policy": (
                "development instance identity is matched to the case-exact AF3 "
                "input JSON name only when lower-cased output names collide; no "
                "metric, label, ranking score, or downstream result selects a source"
            ),
            "instance_manifest_path": str(args.instance_manifest_csv),
            "instance_manifest_sha256": sha256_file(args.instance_manifest_csv),
            "source_split_path": str(args.source_split_csv),
            "source_split_sha256": sha256_file(args.source_split_csv),
            "source_root": str(args.source_root),
            "canonical_root": str(args.output_root),
            "development_count": args.expected_development_count,
            "role_counts": role_counts,
            "paired_instance_count": len(records),
            "sdh_instance_count": sdh_count,
            "case_collision_group_count": len(collision_keys),
            "canonical_source_count": canonical_count,
            "timestamped_case_collision_recovery_count": timestamped_count,
            "source_predictions_modified": False,
            "task_labels_loaded": False,
            "test_access": False,
            "temporal_holdout_access": False,
            "records": records,
        }
        paired_manifest_path = temporary_root / "PAIRED_CASE_EXACT_VIEW_MANIFEST.json"
        paired_manifest_sha256 = write_json(paired_manifest_path, manifest)
        paired_receipt = {
            "format_version": "sabdab2_paired_case_exact_recovery_receipt_v2",
            "status": manifest["status"],
            "manifest_sha256": paired_manifest_sha256,
            "paired_instance_count": len(records),
            "case_collision_group_count": len(collision_keys),
            "timestamped_case_collision_recovery_count": timestamped_count,
            "all_target_directories_unique": len(records)
            == len({record["target_name"] for record in records}),
            "all_source_directories_unique": len(records) == len(selected_sources),
            "source_predictions_modified": False,
            "test_access": False,
            "temporal_holdout_access": False,
        }
        paired_receipt_path = temporary_root / "PAIRED_CASE_EXACT_VIEW_RECEIPT.json"
        paired_receipt_sha256 = write_json(paired_receipt_path, paired_receipt)

        sdh_manifest_sha256 = sha256_file(args.sdh_recovery_manifest)
        sdh_receipt_sha256 = sha256_file(args.sdh_recovery_receipt)
        failed_receipt_sha256 = sha256_file(args.failed_prepare_receipt)
        sdh_manifest = json.loads(args.sdh_recovery_manifest.read_text(encoding="utf-8"))
        sdh_receipt = json.loads(args.sdh_recovery_receipt.read_text(encoding="utf-8"))
        failed_receipt = json.loads(args.failed_prepare_receipt.read_text(encoding="utf-8"))
        if (
            sdh_manifest.get("status") != "PASS_CANONICAL_SDH_RECOVERY_VIEW"
            or sdh_manifest.get("record_count") != 332
            or sdh_manifest.get("exact_canonical_name_count") != 331
            or sdh_manifest.get("timestamp_suffixed_recovery_count") != 1
            or sdh_receipt.get("status") != "PASS_CANONICAL_SDH_RECOVERY_VIEW"
            or sdh_receipt.get("manifest_sha256") != sdh_manifest_sha256
            or failed_receipt.get("job", {}).get("job_id") != "1483067"
            or failed_receipt.get("effects", {}).get("automatic_retry_authorized")
            is not False
        ):
            raise ValueError("prior recovery/failure authority differs")

        combined = {
            "format_version": "sabdab2_predicted_geometry_combined_recovery_v2",
            "status": "PASS_CANONICAL_SDH_RECOVERY_VIEW",
            "record_count": 332,
            "exact_canonical_name_count": 331,
            "timestamp_suffixed_recovery_count": 1,
            "sdh_recovery_manifest_sha256": sdh_manifest_sha256,
            "sdh_recovery_receipt_sha256": sdh_receipt_sha256,
            "paired_case_exact_manifest_sha256": paired_manifest_sha256,
            "paired_case_exact_receipt_sha256": paired_receipt_sha256,
            "failed_prepare_receipt_sha256": failed_receipt_sha256,
            "paired_instance_count": len(records),
            "paired_case_collision_group_count": len(collision_keys),
            "paired_timestamped_recovery_count": timestamped_count,
            "user_authorized_recovery_after_failure": True,
            "authorization_scope": (
                "restore instance-identity mapping only; model, split, labels, "
                "training seeds, loss, and checkpoint selection remain unchanged"
            ),
            "experimental_bound_complex_coordinates_read": False,
            "cross_partner_geometry_used": False,
            "task_labels_loaded": False,
            "test_access": False,
            "temporal_holdout_access": False,
            "source_predictions_modified": False,
        }
        combined_path = temporary_root / "COMBINED_RECOVERY_V2_MANIFEST.json"
        combined_sha256 = write_json(combined_path, combined)
        combined_receipt = {
            "format_version": "sabdab2_predicted_geometry_combined_recovery_receipt_v2",
            "status": combined["status"],
            "manifest_sha256": combined_sha256,
            "record_count": 332,
            "source_predictions_modified": False,
            "test_access": False,
            "temporal_holdout_access": False,
        }
        combined_receipt_path = temporary_root / "COMBINED_RECOVERY_V2_RECEIPT.json"
        combined_receipt_sha256 = write_json(combined_receipt_path, combined_receipt)

        temporary_root.rename(args.output_root)
        for directory, _, _ in os.walk(args.output_root):
            os.chmod(directory, 0o555)
        for path in args.output_root.glob("*.json"):
            path.chmod(0o444)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": "PASS_PAIRED_CASE_EXACT_RECOVERY_VIEW",
                "paired_instance_count": len(records),
                "collision_group_count": len(collision_keys),
                "paired_manifest_sha256": paired_manifest_sha256,
                "paired_receipt_sha256": paired_receipt_sha256,
                "combined_manifest_sha256": combined_sha256,
                "combined_receipt_sha256": combined_receipt_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
