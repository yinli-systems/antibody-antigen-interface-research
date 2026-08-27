#!/usr/bin/env python3
"""Freeze an offline authority binding for the official DiscoTope-3.0 release."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path


EXPECTED_REVISION = "35d9f2e55f97eaba2a7acefbc394db58fb9670bc"
EXPECTED_MODELS_ZIP_SHA256 = "fa627cd66cb9d3fc7892eabd5962b324ffbfa3b68a54718aae34cc3e839b4f4d"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_sha = sha256_file(archive)
    if archive_sha != args.archive_sha256:
        raise ValueError(f"archive SHA mismatch: {archive_sha} != {args.archive_sha256}")

    with tarfile.open(archive, "r:gz") as tar:
        members = [item for item in tar.getmembers() if item.isfile()]
        models_members = [item for item in members if item.name.endswith("/models.zip")]
        if len(models_members) != 1:
            raise ValueError(f"expected one models.zip, got {[item.name for item in models_members]}")
        handle = tar.extractfile(models_members[0])
        if handle is None:
            raise ValueError("cannot read models.zip from source archive")
        models_zip = handle.read()
        requirements_members = [item for item in members if item.name.endswith("/requirements.txt")]
        if len(requirements_members) != 1:
            raise ValueError("expected exactly one requirements.txt")
        requirements_handle = tar.extractfile(requirements_members[0])
        if requirements_handle is None:
            raise ValueError("cannot read requirements.txt")
        requirements = requirements_handle.read().decode("utf-8")

    models_zip_sha = sha256_bytes(models_zip)
    if models_zip_sha != EXPECTED_MODELS_ZIP_SHA256:
        raise ValueError(
            f"models.zip SHA mismatch: {models_zip_sha} != {EXPECTED_MODELS_ZIP_SHA256}"
        )
    with zipfile.ZipFile(io.BytesIO(models_zip)) as zf:
        names = sorted(name for name in zf.namelist() if not name.endswith("/"))
        xgb_names = sorted(
            name for name in names if name.startswith("models/XGB_") and name.endswith("_of_100.json")
        )
        expected_names = {f"models/XGB_{index}_of_100.json" for index in range(1, 101)}
        if set(xgb_names) != expected_names:
            raise ValueError("models.zip does not contain the exact 100-member XGBoost ensemble")
        args_json = json.loads(zf.read("models/args.json"))

    payload = {
        "schema_version": "discotope3-official-binding-v1",
        "status": "PASS_OFFICIAL_SOURCE_AND_100_MODEL_ENSEMBLE_BOUND",
        "official_repository": "https://github.com/Magnushhoie/DiscoTope-3.0",
        "official_paper_doi": "10.3389/fimmu.2024.1322712",
        "revision": EXPECTED_REVISION,
        "source_archive": {
            "path": str(archive),
            "sha256": archive_sha,
            "bytes": archive.stat().st_size,
            "regular_file_count": len(members),
        },
        "models_zip": {
            "member_path": models_members[0].name,
            "sha256": models_zip_sha,
            "bytes": len(models_zip),
            "xgboost_member_count": len(xgb_names),
            "member_names": xgb_names,
            "args_json": args_json,
        },
        "requirements": {
            "member_path": requirements_members[0].name,
            "content_sha256": sha256_bytes(requirements.encode("utf-8")),
            "content": requirements.splitlines(),
        },
        "training_curve_policy": {
            "applicable": False,
            "reason_code": "RELEASED_FITTED_XGBOOST_ENSEMBLE_NO_NEURAL_EPOCH_TRAJECTORY",
        },
        "execution": {
            "dependencies_installed": False,
            "model_loaded": False,
            "inference_performed": False,
            "raw_predictions_read": False,
            "raw_labels_read": False,
            "old_sabdab2_test_opened": False,
            "temporal_holdout_opened": False,
        },
    }
    output = args.output_root.resolve()
    result = output / "DISCOTOPE3_OFFICIAL_BINDING_V1.json"
    receipt = output / "DISCOTOPE3_OFFICIAL_BINDING_V1_RECEIPT.json"
    write_json(result, payload)
    write_json(
        receipt,
        {
            "schema_version": "discotope3-official-binding-v1-receipt",
            "status": "PASS",
            "generator_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "bytes": Path(__file__).resolve().stat().st_size,
            },
            "result": {
                "path": str(result),
                "sha256": sha256_file(result),
                "bytes": result.stat().st_size,
            },
            "source_archive_sha256": archive_sha,
            "models_zip_sha256": models_zip_sha,
            "old_sabdab2_test_opened": False,
            "temporal_holdout_opened": False,
        },
    )
    print(receipt.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
