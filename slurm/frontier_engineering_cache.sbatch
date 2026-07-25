#!/bin/bash
#SBATCH --job-name=frontier-cache
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --exclude=wqd10nah09g2
#SBATCH --output=/tmp/frontier-cache-%j.out
#SBATCH --error=/tmp/frontier-cache-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${FRONTIER_PYTHON_BIN:?FRONTIER_PYTHON_BIN is required}
SOURCE_ROOT=${FRONTIER_SOURCE_ROOT:?FRONTIER_SOURCE_ROOT must identify sealed source}
EXPECTED_BUILDER_SHA256=${FRONTIER_EXPECTED_BUILDER_SHA256:?FRONTIER_EXPECTED_BUILDER_SHA256 is required}
AUTHORITY_MANIFEST=${FRONTIER_AUTHORITY_MANIFEST:?FRONTIER_AUTHORITY_MANIFEST must be train-only}
CANONICAL_MANIFEST=${FRONTIER_CANONICAL_MANIFEST:?FRONTIER_CANONICAL_MANIFEST is required}
ANTIBODY_EMBEDDING_ROOT=${FRONTIER_ANTIBODY_EMBEDDING_ROOT:?FRONTIER_ANTIBODY_EMBEDDING_ROOT is required}
ANTIGEN_EMBEDDING_ROOT=${FRONTIER_ANTIGEN_EMBEDDING_ROOT:?FRONTIER_ANTIGEN_EMBEDDING_ROOT is required}
ANTIBODY_STORE_MANIFEST=${FRONTIER_ANTIBODY_STORE_MANIFEST:?FRONTIER_ANTIBODY_STORE_MANIFEST is required}
ANTIGEN_STORE_MANIFEST=${FRONTIER_ANTIGEN_STORE_MANIFEST:?FRONTIER_ANTIGEN_STORE_MANIFEST is required}
EXPECTED_AUTHORITY_SHA256=${FRONTIER_EXPECTED_AUTHORITY_SHA256:?FRONTIER_EXPECTED_AUTHORITY_SHA256 is required}
EXPECTED_CANONICAL_SHA256=${FRONTIER_EXPECTED_CANONICAL_SHA256:?FRONTIER_EXPECTED_CANONICAL_SHA256 is required}
EXPECTED_ANTIBODY_STORE_SHA256=${FRONTIER_EXPECTED_ANTIBODY_STORE_SHA256:?FRONTIER_EXPECTED_ANTIBODY_STORE_SHA256 is required}
EXPECTED_ANTIGEN_STORE_SHA256=${FRONTIER_EXPECTED_ANTIGEN_STORE_SHA256:?FRONTIER_EXPECTED_ANTIGEN_STORE_SHA256 is required}
OUTPUT_ROOT=${FRONTIER_OUTPUT_ROOT:?FRONTIER_OUTPUT_ROOT must be create-only}
BUILDER=$SOURCE_ROOT/scripts/build_frontier_engineering_cache.py

if [[ ! -x "$PYTHON_BIN" || -L "$PYTHON_BIN" ]]; then
  echo "Python runtime must be an executable non-symlink file" >&2
  exit 2
fi
if [[ ! -d "$SOURCE_ROOT" || -L "$SOURCE_ROOT" ]]; then
  echo "Source root must be a real directory" >&2
  exit 2
fi
if [[ ! -f "$BUILDER" || -L "$BUILDER" ]]; then
  echo "Builder must be a regular non-symlink file" >&2
  exit 2
fi
for digest in \
  "$EXPECTED_BUILDER_SHA256" \
  "$EXPECTED_AUTHORITY_SHA256" \
  "$EXPECTED_CANONICAL_SHA256" \
  "$EXPECTED_ANTIBODY_STORE_SHA256" \
  "$EXPECTED_ANTIGEN_STORE_SHA256"
do
  if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Every external anchor must be a lowercase SHA-256 digest" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "$BUILDER" | cut -d ' ' -f 1)" != "$EXPECTED_BUILDER_SHA256" ]]; then
  echo "Builder SHA-256 differs from the submitted lock" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -L "$OUTPUT_ROOT" ]]; then
  echo "Refusing to reuse output root" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" "$BUILDER" \
  --authority-manifest "$AUTHORITY_MANIFEST" \
  --canonical-manifest "$CANONICAL_MANIFEST" \
  --antibody-embedding-root "$ANTIBODY_EMBEDDING_ROOT" \
  --antigen-embedding-root "$ANTIGEN_EMBEDDING_ROOT" \
  --antibody-store-manifest "$ANTIBODY_STORE_MANIFEST" \
  --antigen-store-manifest "$ANTIGEN_STORE_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --expected-builder-sha256 "$EXPECTED_BUILDER_SHA256" \
  --expected-authority-sha256 "$EXPECTED_AUTHORITY_SHA256" \
  --expected-canonical-manifest-sha256 "$EXPECTED_CANONICAL_SHA256" \
  --expected-antibody-store-manifest-sha256 "$EXPECTED_ANTIBODY_STORE_SHA256" \
  --expected-antigen-store-manifest-sha256 "$EXPECTED_ANTIGEN_STORE_SHA256"

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys


def reject_constant(value):
    raise RuntimeError(f"Non-finite JSON constant is forbidden: {value}")


def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"Duplicate JSON key is forbidden: {key!r}")
        value[key] = item
    return value


def acquire(path, label):
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        chunks = []
        remaining = int(before.st_size)
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"{label} ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} grew during verification")
        after = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if tuple(getattr(before, field) for field in fields) != tuple(
            getattr(after, field) for field in fields
        ):
            raise RuntimeError(f"{label} changed during verification")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def parse_json(raw, label):
    return json.loads(
        raw.decode("utf-8", errors="strict"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


root = Path(os.path.abspath(sys.argv[1]))
if root.is_symlink() or not root.is_dir():
    raise RuntimeError("Published cache root is not a real directory")
receipt_bytes = acquire(root / "ENGINEERING_CACHE_BUILD_RECEIPT.json", "receipt")
receipt = parse_json(receipt_bytes, "receipt")
if receipt.get("status") != "complete":
    raise RuntimeError("Published cache receipt is not complete")
boundary = receipt.get("claim_boundary", {})
if boundary.get("engineering_data_path_verified") is not True or any(
    boundary.get(field) is not False
    for field in (
        "scientific_training_evidence",
        "model_selection_evidence",
        "validation_or_test_evidence",
        "biological_performance_evidence",
        "state_of_the_art_evidence",
    )
):
    raise RuntimeError("Published cache claim boundary is unsafe")
expected_examples = int(receipt["selection"]["selected_instance_count"])
if expected_examples != int(receipt["selection"]["authorized_instance_count"]):
    raise RuntimeError("Published cache is not exact authorization-equal")
if expected_examples != int(receipt["artifacts"]["example_artifact_count"]):
    raise RuntimeError("Published cache example counts disagree")

authorization_bytes = acquire(
    root / "AUTHORIZED_FOLD_INVENTORY.json",
    "authorization manifest",
)
cache_manifest_bytes = acquire(
    root / "fold_cache" / "FOLD_LOCAL_CACHE_MANIFEST.json",
    "cache manifest",
)
if hashlib.sha256(authorization_bytes).hexdigest() != receipt["artifacts"][
    "authorization_manifest_sha256"
]:
    raise RuntimeError("Authorization manifest hash differs")
if hashlib.sha256(cache_manifest_bytes).hexdigest() != receipt["artifacts"][
    "cache_manifest_sha256"
]:
    raise RuntimeError("Cache manifest hash differs")
authorization = parse_json(authorization_bytes, "authorization manifest")
cache_manifest = parse_json(cache_manifest_bytes, "cache manifest")
authorized_ids = [value["instance_id"] for value in authorization["instances"]]
cached_ids = [value["instance_id"] for value in cache_manifest["instances"]]
if authorized_ids != cached_ids or len(cached_ids) != expected_examples:
    raise RuntimeError("Published cache instance inventory differs from authorization")

expected_files = {
    PurePosixPath("AUTHORIZED_FOLD_INVENTORY.json"),
    PurePosixPath("ENGINEERING_CACHE_BUILD_RECEIPT.json"),
    PurePosixPath("fold_cache/FOLD_LOCAL_CACHE_MANIFEST.json"),
}
for record in cache_manifest["instances"]:
    relative = PurePosixPath("fold_cache") / PurePosixPath(record["artifact_path"])
    expected_files.add(relative)
    artifact_bytes = acquire(root.joinpath(*relative.parts), "example artifact")
    if len(artifact_bytes) != int(record["artifact_size_bytes"]):
        raise RuntimeError("Example artifact size differs")
    if hashlib.sha256(artifact_bytes).hexdigest() != record["artifact_sha256"]:
        raise RuntimeError("Example artifact hash differs")

actual_files = set()
actual_directories = set()
for path in root.rglob("*"):
    relative = PurePosixPath(path.relative_to(root).as_posix())
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode):
        raise RuntimeError("Published cache contains a symlink")
    if stat.S_ISDIR(status.st_mode):
        actual_directories.add(relative)
        expected_mode = 0o555
    elif stat.S_ISREG(status.st_mode):
        actual_files.add(relative)
        expected_mode = 0o444
    else:
        raise RuntimeError("Published cache contains a special file")
    if stat.S_IMODE(status.st_mode) != expected_mode:
        raise RuntimeError("Published cache mode differs")
if stat.S_IMODE(root.lstat().st_mode) != 0o555:
    raise RuntimeError("Published cache root mode differs")
if actual_files != expected_files:
    raise RuntimeError("Published cache file inventory differs")
if len(actual_directories) + 1 != int(
    receipt["immutability_policy"]["expected_directory_count"]
):
    raise RuntimeError("Published cache directory count differs")
if len(actual_files) != int(
    receipt["immutability_policy"]["expected_regular_file_count"]
):
    raise RuntimeError("Published cache regular-file count differs")

print(
    json.dumps(
        {
            "fold_id": receipt["fold_id"],
            "selected_instance_count": expected_examples,
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "status": "independently_verified_train_only_engineering_cache",
        },
        sort_keys=True,
    )
)
PY
