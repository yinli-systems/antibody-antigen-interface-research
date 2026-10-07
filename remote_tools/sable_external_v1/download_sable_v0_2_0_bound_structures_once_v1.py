#!/usr/bin/env python3
"""Download and extract the authorized SABLE v0.2.0 bound structures once."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from typing import Any
from urllib.request import Request, urlopen


ZENODO_RECORD_ID = "21759053"
EXPECTED_EXTERNAL_PLAN_SHA256 = (
    "d97f0b94863fe9227d80c5675e944091809245b8fab22a89cda81148a8de172a"
)
EXPECTED_INPUT_PLAN_SHA256 = (
    "c82c12f08c87b42d21aee31cc555373ffbc4211415d9aff302444baefc971ad6"
)
EXPECTED_METADATA_SHA256 = (
    "d6b16885c7ab44bbe2a38b3fc02ab085adff73479c1a119e3786831f464b20b7"
)
ARCHIVES = (
    (
        "sable_v0.2.0_structures_pdb_000.tar.gz",
        502_497_553,
        "7e5a9a65d74fd4a957454c10553e298f274dc7309a11e8b4c10428e6165d8b7a",
    ),
    (
        "sable_v0.2.0_structures_pdb_001.tar.gz",
        476_067_632,
        "50139a56034db8dc58ceac6aacee6598c28d88e119870e34755eaafbc920a614",
    ),
    (
        "sable_v0.2.0_structures_pdb_002.tar.gz",
        465_950_634,
        "102dd2fa9fcc75183f7ded44385b730d2d321f00239ab1866738d8370cc29d17",
    ),
    (
        "sable_v0.2.0_structures_pdb_003.tar.gz",
        478_526_026,
        "74f6c54bcfcb032227ab9bf71cc7430bfc2aba10fe7f699096b1ca66ffa9d0b7",
    ),
    (
        "sable_v0.2.0_structures_pdb_004.tar.gz",
        131_889_323,
        "2f0e09ae08f0596ea108a96da19f395a5ac22c792073779e1b0a4bd313516e12",
    ),
)
EXPECTED_ENTRIES = 97
EXPECTED_PDB_IDS = 88


class DownloadError(ValueError):
    """Raised when an authorization or downloaded artifact differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DownloadError(f"JSON root is not an object: {path}")
    return value


def verify_authorization(path: Path, receipt_path: Path) -> tuple[str, str]:
    authorization = load_json(path)
    receipt = load_json(receipt_path)
    authorization_sha256 = sha256_file(path)
    if (
        authorization.get("status") != "AUTHORIZED_ONE_TIME_BOUND_LABEL_OPENING"
        or authorization.get("external_execution_plan_sha256")
        != EXPECTED_EXTERNAL_PLAN_SHA256
        or authorization.get("eligible_entries") != EXPECTED_ENTRIES
        or authorization.get("eligible_unique_pdb_ids") != EXPECTED_PDB_IDS
        or authorization.get("maximum_label_openings") != 1
        or authorization.get("bound_structure_download_authorized") is not True
        or authorization.get("bound_label_materialization_authorized") is not True
        or authorization.get("one_time_evaluation_authorized") is not True
        or authorization.get("feedback_to_model_or_protocol") is not False
        or authorization.get("performance_threshold_applied") is not False
        or authorization.get("task_labels_read_by_this_freezer") is not False
        or authorization.get("sable_bound_structures_downloaded_by_this_freezer")
        is not False
        or authorization.get("old_sabdab2_test_loaded") is not False
        or authorization.get("temporal_holdout_loaded") is not False
        or authorization.get("selection_or_tuning") is not False
        or receipt.get("status") != authorization.get("status")
        or receipt.get("authorization_sha256") != authorization_sha256
        or receipt.get("all_label_free_gates_verified") is not True
        or receipt.get("maximum_label_openings") != 1
    ):
        raise DownloadError("one-time bound-label authorization differs")
    return authorization_sha256, sha256_file(receipt_path)


def download_archive(cache_root: Path, name: str, size: int, digest: str) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / name
    if target.is_file() and target.stat().st_size == size and sha256_file(target) == digest:
        return target
    if target.exists() or target.is_symlink():
        raise DownloadError(f"archive cache contains a differing target: {target}")
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > size:
        raise DownloadError(f"partial archive is larger than expected: {partial}")
    headers = {"User-Agent": "FrameContact-SABLE-authorized-download/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{name}?download=1"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=120) as response:
        status = int(getattr(response, "status", 200))
        if offset and status != 206:
            raise DownloadError("server did not honor the range request")
        with partial.open("ab" if offset else "xb") as handle:
            shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    if partial.stat().st_size != size or sha256_file(partial) != digest:
        raise DownloadError(f"downloaded archive differs: {name}")
    os.replace(partial, target)
    return target


def selected_structure_paths(metadata: Path, input_plan: Path) -> dict[str, str]:
    if sha256_file(metadata) != EXPECTED_METADATA_SHA256:
        raise DownloadError("SABLE metadata SHA256 differs")
    if sha256_file(input_plan) != EXPECTED_INPUT_PLAN_SHA256:
        raise DownloadError("SABLE label-free AF3 input plan SHA256 differs")
    plan = load_json(input_plan)
    selected = {str(row["entry_id"]) for row in plan.get("entry_bindings", [])}
    if len(selected) != EXPECTED_ENTRIES:
        raise DownloadError("selected SABLE entry cardinality differs")
    paths: dict[str, str] = {}
    with metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            entry_id = str(row["entry_id"])
            if entry_id not in selected:
                continue
            relative = PurePosixPath(str(row["structure_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise DownloadError(f"unsafe structure path for {entry_id}")
            paths[entry_id] = relative.as_posix()
    if set(paths) != selected or len({Path(value).name for value in paths.values()}) != len(paths):
        raise DownloadError("metadata does not uniquely bind every selected structure")
    return paths


def extract_selected(
    archives: list[Path], selected: dict[str, str], temporary_root: Path
) -> list[dict[str, Any]]:
    wanted = set(selected.values())
    extracted: dict[str, Path] = {}
    for archive in archives:
        with tarfile.open(archive, mode="r:gz") as handle:
            for member in handle:
                name = PurePosixPath(member.name).as_posix().lstrip("./")
                if name not in wanted:
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise DownloadError(f"selected archive member is not a regular file: {name}")
                destination = temporary_root / name
                if destination.exists() or destination.is_symlink():
                    raise DownloadError(f"selected structure occurs more than once: {name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise DownloadError(f"cannot open selected archive member: {name}")
                with source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                extracted[name] = destination
    if set(extracted) != wanted:
        missing = sorted(wanted - set(extracted))
        raise DownloadError(f"archives lack {len(missing)} selected structures")
    by_path = {value: key for key, value in selected.items()}
    return [
        {
            "entry_id": by_path[name],
            "structure_path": name,
            "structure_sha256": sha256_file(extracted[name]),
            "structure_size_bytes": extracted[name].stat().st_size,
        }
        for name in sorted(extracted)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input-plan", type=Path, required=True)
    parser.add_argument("--archive-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    authorization_sha256, authorization_receipt_sha256 = verify_authorization(
        args.authorization, args.authorization_receipt
    )
    selected = selected_structure_paths(args.metadata, args.input_plan)
    archive_paths = [
        download_archive(args.archive_cache, name, size, digest)
        for name, size, digest in ARCHIVES
    ]
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.tmp-", dir=args.output_root.parent)
    )
    try:
        records = extract_selected(archive_paths, selected, temporary_root)
        payload = {
            "format_version": "sable_v0_2_0_authorized_bound_structure_download_v1",
            "status": "PASS_AUTHORIZED_BOUND_STRUCTURES_DOWNLOADED_ONCE",
            "authority": {
                "authorization_sha256": authorization_sha256,
                "authorization_receipt_sha256": authorization_receipt_sha256,
                "external_execution_plan_sha256": EXPECTED_EXTERNAL_PLAN_SHA256,
                "input_plan_sha256": EXPECTED_INPUT_PLAN_SHA256,
                "metadata_sha256": EXPECTED_METADATA_SHA256,
                "zenodo_record_id": ZENODO_RECORD_ID,
                "doi": "10.5281/zenodo.21759053",
                "label_opening_count_after_download": 1,
                "labels_or_performance_read_during_download": False,
                "old_sabdab2_test_loaded": False,
                "temporal_holdout_loaded": False,
                "selection_or_tuning": False,
            },
            "archives": [
                {"name": name, "size_bytes": size, "sha256": digest}
                for name, size, digest in ARCHIVES
            ],
            "counts": {
                "structures": len(records),
                "unique_pdb_ids": len({row["entry_id"].split("_", 1)[0] for row in records}),
            },
            "records": records,
        }
        if (
            payload["counts"]["structures"] != EXPECTED_ENTRIES
            or payload["counts"]["unique_pdb_ids"] != EXPECTED_PDB_IDS
        ):
            raise DownloadError("extracted SABLE cohort cardinality differs")
        manifest_path = temporary_root / "AUTHORIZED_BOUND_STRUCTURE_MANIFEST.json"
        manifest_path.write_bytes(canonical_bytes(payload))
        receipt = {
            "format_version": "sable_v0_2_0_authorized_bound_structure_download_receipt_v1",
            "status": payload["status"],
            "manifest_sha256": sha256_file(manifest_path),
            "authorization_sha256": authorization_sha256,
            "all_archive_sha256_verified": True,
            "all_selected_structure_sha256_recorded": True,
            "structure_count": EXPECTED_ENTRIES,
            "opening_count": 1,
            "selection_or_tuning": False,
        }
        receipt_path = temporary_root / "AUTHORIZED_BOUND_STRUCTURE_RECEIPT.json"
        receipt_path.write_bytes(canonical_bytes(receipt))
        os.replace(temporary_root, args.output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(
        json.dumps(
            {**receipt, "receipt_sha256": sha256_file(args.output_root / receipt_path.name)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
