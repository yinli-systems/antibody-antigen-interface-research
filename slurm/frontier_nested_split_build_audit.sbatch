#!/bin/bash
#SBATCH --job-name=frontier-split-audit
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=96
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=wqd10nah09g2
#SBATCH --output=/tmp/frontier-split-audit-%j.out
#SBATCH --error=/tmp/frontier-split-audit-%j.err

set -euo pipefail
umask 077

# The scheduler-started /bin/bash and anything it sourced before this line are
# external trust roots and cannot be self-attested by an already running shell.
# Remove inherited command-resolution hooks before invoking any child process.
PATH=/usr/bin:/bin
export PATH
unset BASH_ENV ENV CDPATH GLOBIGNORE
builtin trap - DEBUG RETURN ERR EXIT
builtin unalias -a 2>/dev/null || :
builtin shopt -u expand_aliases
REMOVED_INHERITED_FUNCTION_COUNT=0
while IFS= builtin read -r inherited_function; do
  builtin unset -f "$inherited_function"
  REMOVED_INHERITED_FUNCTION_COUNT=$((REMOVED_INHERITED_FUNCTION_COUNT + 1))
done < <(builtin compgen -A function)
unset inherited_function
builtin hash -r

DEPLOY_ROOT=/ssd/scxi253/model_deployment
PYTHON_BIN=${FRONTIER_SPLIT_PYTHON_BIN:-$DEPLOY_ROOT/envs/epiformer/bin/python3.10}
if [[ "$PYTHON_BIN" != /* || "$PYTHON_BIN" == *$'\r'* || "$PYTHON_BIN" == *$'\n'* ]]; then
  echo "FRONTIER_SPLIT_PYTHON_BIN must be an absolute external trust-root path" >&2
  exit 2
fi
SOURCE_ROOT=${FRONTIER_SOURCE_ROOT:?FRONTIER_SOURCE_ROOT must identify externally inventoried source}
OUTPUT_ROOT=${FRONTIER_OUTPUT_ROOT:?FRONTIER_OUTPUT_ROOT must already exist}
TRAIN_AUTHORITY_ALLOWLIST=${FRONTIER_TRAIN_AUTHORITY_ALLOWLIST:?FRONTIER_TRAIN_AUTHORITY_ALLOWLIST must be a physical train-only allowlist}
TRAIN_CANONICAL_MANIFEST=${FRONTIER_TRAIN_CANONICAL_MANIFEST:?FRONTIER_TRAIN_CANONICAL_MANIFEST must be a physical train-only four-column manifest}
EXPECTED_TRAIN_AUTHORITY_SHA256=${FRONTIER_EXPECTED_TRAIN_AUTHORITY_ALLOWLIST_SHA256:?FRONTIER_EXPECTED_TRAIN_AUTHORITY_ALLOWLIST_SHA256 must be externally pinned}
EXPECTED_TRAIN_MANIFEST_SHA256=${FRONTIER_EXPECTED_TRAIN_CANONICAL_MANIFEST_SHA256:?FRONTIER_EXPECTED_TRAIN_CANONICAL_MANIFEST_SHA256 must be externally pinned}
EXPECTED_SOURCE_MANIFEST_SHA256=${FRONTIER_EXPECTED_SOURCE_MANIFEST_SHA256:?FRONTIER_EXPECTED_SOURCE_MANIFEST_SHA256 must be externally pinned}
EXPECTED_JOINT_POLICY_SHA256=${FRONTIER_EXPECTED_JOINT_POLICY_SHA256:?FRONTIER_EXPECTED_JOINT_POLICY_SHA256 must be externally pinned}
EXPECTED_BUILDER_SHA256=${FRONTIER_EXPECTED_BUILDER_SOURCE_SHA256:?FRONTIER_EXPECTED_BUILDER_SOURCE_SHA256 must be externally pinned}
EXPECTED_AUDITOR_SHA256=${FRONTIER_EXPECTED_AUDITOR_SOURCE_SHA256:?FRONTIER_EXPECTED_AUDITOR_SOURCE_SHA256 must be externally pinned}
EXPECTED_JOINT_MILP_SHA256=${FRONTIER_EXPECTED_JOINT_MILP_SOURCE_SHA256:?FRONTIER_EXPECTED_JOINT_MILP_SOURCE_SHA256 must be externally pinned}
JOINT_ASSIGNMENT_MODE=${FRONTIER_JOINT_ASSIGNMENT_MODE:?FRONTIER_JOINT_ASSIGNMENT_MODE must be provisional_discovery or externally_pinned_replay}
EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256=${FRONTIER_EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256-}
PROVISIONAL_ATTEMPT_ID=${FRONTIER_PROVISIONAL_ATTEMPT_ID-}
PINNED_REPLAY_ID=${FRONTIER_PINNED_REPLAY_ID-}
SOURCE_PROVISIONAL_ATTEMPT_ID=${FRONTIER_SOURCE_PROVISIONAL_ATTEMPT_ID-}
WORKERS=${FRONTIER_SPLIT_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}
CHUNK_SIZE=${FRONTIER_SPLIT_CHUNK_SIZE:-256}
SEED=${FRONTIER_SPLIT_SEED:-2026072401}
JOINT_TIME_LIMIT_SECONDS=${FRONTIER_JOINT_TIME_LIMIT_SECONDS:-43200}
RUN_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
RUN_ROOT=$OUTPUT_ROOT/slurm-$RUN_ID
STAGE_ROOT=$RUN_ROOT/stage
STAGED_SOURCE_ROOT=$STAGE_ROOT/source
STAGED_TRAIN_AUTHORITY=$STAGE_ROOT/inputs/train_authority_allowlist.json
STAGED_TRAIN_MANIFEST=$STAGE_ROOT/inputs/train_canonical_manifest.csv
STAGED_BUILDER=$STAGED_SOURCE_ROOT/scripts/build_train_only_nested_neighborhood_folds.py
STAGED_AUDITOR=$STAGED_SOURCE_ROOT/scripts/audit_train_only_nested_neighborhood_folds.py
STAGED_JOINT_MILP=$STAGED_SOURCE_ROOT/src/interaction_first/nested_joint_milp.py
ARTIFACT_ROOT=$RUN_ROOT/artifacts
ASSIGNMENT=$ARTIFACT_ROOT/nested_assignment.json
AUDIT=$ARTIFACT_ROOT/core_feasibility_precursor_audit.json
PROVISIONAL_LEDGER_ENTRY=$ARTIFACT_ROOT/provisional_attempt_ledger_entry.json

validate_workflow_identifier() {
  local name=$1
  local value=$2
  if [[ -z "$value" \
    || ${#value} -gt 128 \
    || "$value" =~ ^[[:space:]] \
    || "$value" =~ [[:space:]]$ \
    || "$value" == *","* \
    || "$value" == *\"* \
    || "$value" == *$'\r'* \
    || "$value" == *$'\n'* ]]; then
    echo "$name is not a valid nonempty workflow identifier" >&2
    exit 2
  fi
}

validate_sha256_identifier() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$name must be one lowercase SHA-256 digest" >&2
    exit 2
  fi
}

validate_sha256_identifier \
  FRONTIER_EXPECTED_TRAIN_AUTHORITY_ALLOWLIST_SHA256 \
  "$EXPECTED_TRAIN_AUTHORITY_SHA256"
validate_sha256_identifier \
  FRONTIER_EXPECTED_TRAIN_CANONICAL_MANIFEST_SHA256 \
  "$EXPECTED_TRAIN_MANIFEST_SHA256"
validate_sha256_identifier \
  FRONTIER_EXPECTED_SOURCE_MANIFEST_SHA256 \
  "$EXPECTED_SOURCE_MANIFEST_SHA256"
validate_sha256_identifier \
  FRONTIER_EXPECTED_JOINT_POLICY_SHA256 \
  "$EXPECTED_JOINT_POLICY_SHA256"
validate_sha256_identifier \
  FRONTIER_EXPECTED_BUILDER_SOURCE_SHA256 "$EXPECTED_BUILDER_SHA256"
validate_sha256_identifier \
  FRONTIER_EXPECTED_AUDITOR_SOURCE_SHA256 "$EXPECTED_AUDITOR_SHA256"
validate_sha256_identifier \
  FRONTIER_EXPECTED_JOINT_MILP_SOURCE_SHA256 "$EXPECTED_JOINT_MILP_SHA256"

case "$JOINT_ASSIGNMENT_MODE" in
  provisional_discovery)
    if [[ -n "$EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256" ]]; then
      echo "provisional_discovery forbids an in-band selected-membership anchor" >&2
      exit 2
    fi
    if [[ -z "$PROVISIONAL_ATTEMPT_ID" ]]; then
      echo "provisional_discovery requires FRONTIER_PROVISIONAL_ATTEMPT_ID" >&2
      exit 2
    fi
    if [[ -n "$PINNED_REPLAY_ID" || -n "$SOURCE_PROVISIONAL_ATTEMPT_ID" ]]; then
      echo "provisional_discovery forbids pinned replay identifiers" >&2
      exit 2
    fi
    validate_workflow_identifier \
      FRONTIER_PROVISIONAL_ATTEMPT_ID "$PROVISIONAL_ATTEMPT_ID"
    ;;
  externally_pinned_replay)
    if [[ -z "$EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256" ]]; then
      echo "externally_pinned_replay requires FRONTIER_EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256" >&2
      exit 2
    fi
    validate_sha256_identifier \
      FRONTIER_EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256 \
      "$EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256"
    if [[ -n "$PROVISIONAL_ATTEMPT_ID" ]]; then
      echo "externally_pinned_replay forbids provisional attempt options" >&2
      exit 2
    fi
    if [[ -z "$PINNED_REPLAY_ID" || -z "$SOURCE_PROVISIONAL_ATTEMPT_ID" ]]; then
      echo "externally_pinned_replay requires FRONTIER_PINNED_REPLAY_ID and FRONTIER_SOURCE_PROVISIONAL_ATTEMPT_ID" >&2
      exit 2
    fi
    validate_workflow_identifier \
      FRONTIER_PINNED_REPLAY_ID "$PINNED_REPLAY_ID"
    validate_workflow_identifier \
      FRONTIER_SOURCE_PROVISIONAL_ATTEMPT_ID \
      "$SOURCE_PROVISIONAL_ATTEMPT_ID"
    if [[ "$PINNED_REPLAY_ID" == "$SOURCE_PROVISIONAL_ATTEMPT_ID" ]]; then
      echo "Replay ID must differ from source provisional attempt ID" >&2
      exit 2
    fi
    ;;
  *)
    echo "Unsupported FRONTIER_JOINT_ASSIGNMENT_MODE: $JOINT_ASSIGNMENT_MODE" >&2
    exit 2
    ;;
esac

export SOURCE_ROOT OUTPUT_ROOT RUN_ID
export TRAIN_AUTHORITY_ALLOWLIST TRAIN_CANONICAL_MANIFEST
export EXPECTED_TRAIN_AUTHORITY_SHA256 EXPECTED_TRAIN_MANIFEST_SHA256
export EXPECTED_SOURCE_MANIFEST_SHA256 EXPECTED_BUILDER_SHA256
export EXPECTED_AUDITOR_SHA256 EXPECTED_JOINT_MILP_SHA256
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

# Trust boundary: this launcher, its already-running Bash/OS, PYTHON_BIN,
# Python's standard library, Biopython, SciPy/HiGHS, and native dependencies
# are externally provisioned and are not a three-file runtime closure. The
# bootstrap authenticates the exact three-file staged application-source
# closure plus both physical train-only inputs. No original pathname is
# executed after staging.
"$PYTHON_BIN" -I -S -B - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath


HEX = frozenset("0123456789abcdef")
CHUNK = 8 * 1024 * 1024
SOURCE_MANIFEST_RELATIVE = PurePosixPath("SOURCE_MANIFEST.sha256")
BUILDER_RELATIVE = PurePosixPath(
    "scripts/build_train_only_nested_neighborhood_folds.py"
)
AUDITOR_RELATIVE = PurePosixPath(
    "scripts/audit_train_only_nested_neighborhood_folds.py"
)
JOINT_RELATIVE = PurePosixPath("src/interaction_first/nested_joint_milp.py")
REQUIRED_OPEN_FLAGS = (
    "O_NOFOLLOW",
    "O_DIRECTORY",
    "O_CLOEXEC",
    "O_NONBLOCK",
)
MISSING_OPEN_FLAGS = [
    name
    for name in REQUIRED_OPEN_FLAGS
    if not isinstance(getattr(os, name, None), int)
]
OPEN_DIR_FD_SUPPORTED = os.open in getattr(os, "supports_dir_fd", ())
if MISSING_OPEN_FLAGS or not OPEN_DIR_FD_SUPPORTED:
    raise SystemExit(
        "Secure staging capabilities are unavailable: "
        f"missing_flags={MISSING_OPEN_FLAGS}, "
        f"open_dir_fd={OPEN_DIR_FD_SUPPORTED}"
    )
SECURE_OPEN_CAPABILITIES = {
    **{name: True for name in REQUIRED_OPEN_FLAGS},
    "open_dir_fd": True,
    "component_wise_parent_traversal": True,
}


def digest(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in HEX for character in value):
        raise SystemExit(f"{label} must be one lowercase SHA-256 digest")
    return value


def lexical_absolute(value: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(value)))


def fingerprint(status: os.stat_result) -> dict[str, int]:
    return {
        "device": int(status.st_dev),
        "inode": int(status.st_ino),
        "mode": int(status.st_mode),
        "link_count": int(status.st_nlink),
        "uid": int(status.st_uid),
        "gid": int(status.st_gid),
        "size_bytes": int(status.st_size),
        "mtime_ns": int(status.st_mtime_ns),
        "ctime_ns": int(status.st_ctime_ns),
    }


def open_directory_nofollow(path: Path) -> int:
    absolute = lexical_absolute(os.fspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component,
                flags | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            status = os.fstat(next_descriptor)
            if not stat.S_ISDIR(status.st_mode):
                os.close(next_descriptor)
                raise SystemExit(f"Non-directory path component: {absolute}")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def read_open_regular(
    descriptor: int,
    label: str,
) -> tuple[bytes, dict[str, int], dict[str, int]]:
    before_status = os.fstat(descriptor)
    if not stat.S_ISREG(before_status.st_mode):
        raise SystemExit(f"{label} must be a regular non-symlink file")
    before = fingerprint(before_status)
    chunks: list[bytes] = []
    remaining = int(before_status.st_size)
    while remaining:
        chunk = os.read(descriptor, min(CHUNK, remaining))
        if not chunk:
            raise SystemExit(f"{label} ended before its authenticated size")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise SystemExit(f"{label} grew during authenticated acquisition")
    after = fingerprint(os.fstat(descriptor))
    if before != after:
        raise SystemExit(f"{label} changed during authenticated acquisition")
    payload = b"".join(chunks)
    if len(payload) != before["size_bytes"]:
        raise SystemExit(f"{label} byte count differs from fd metadata")
    return payload, before, after


def acquire_absolute(path: str, expected: str, label: str) -> tuple[bytes, dict[str, object]]:
    expected_digest = digest(expected, f"{label} expected digest")
    absolute = lexical_absolute(path)
    parent = open_directory_nofollow(absolute.parent)
    descriptor = -1
    try:
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent,
        )
        payload, before, after = read_open_regular(descriptor, label)
    except OSError as error:
        raise SystemExit(f"Cannot securely acquire {label}: {absolute}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_digest:
        raise SystemExit(
            f"{label} digest mismatch: expected={expected_digest}, observed={observed}"
        )
    return payload, {
        "path": str(absolute),
        "sha256": observed,
        "size_bytes": len(payload),
        "stat_fingerprint_before": before,
        "stat_fingerprint_after": after,
        "single_verified_fd_read": True,
        "symlink_followed": False,
    }


def open_relative_regular(root_descriptor: int, relative: PurePosixPath) -> int:
    current = os.dup(root_descriptor)
    try:
        for component in relative.parts[:-1]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=current,
            )
            status = os.fstat(next_descriptor)
            if not stat.S_ISDIR(status.st_mode):
                os.close(next_descriptor)
                raise SystemExit(f"Source component is not a directory: {relative}")
            os.close(current)
            current = next_descriptor
        descriptor = os.open(
            relative.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=current,
        )
        os.close(current)
        return descriptor
    except BaseException:
        os.close(current)
        raise


def acquire_relative(
    root_descriptor: int,
    root_path: Path,
    relative: PurePosixPath,
    expected: str,
    label: str,
) -> tuple[bytes, dict[str, object]]:
    descriptor = -1
    try:
        descriptor = open_relative_regular(root_descriptor, relative)
        payload, before, after = read_open_regular(descriptor, label)
    except OSError as error:
        raise SystemExit(f"Cannot securely acquire {label}: {relative}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != digest(expected, f"{label} manifest digest"):
        raise SystemExit(f"{label} differs from SOURCE_MANIFEST.sha256")
    return payload, {
        "path": str(root_path / Path(*relative.parts)),
        "sha256": observed,
        "size_bytes": len(payload),
        "stat_fingerprint_before": before,
        "stat_fingerprint_after": after,
        "single_verified_fd_read": True,
        "symlink_followed": False,
    }


def parse_manifest(payload: bytes) -> dict[PurePosixPath, str]:
    if (
        not payload
        or not payload.endswith(b"\n")
        or b"\r" in payload
        or b"\0" in payload
        or payload.startswith(b"\xef\xbb\xbf")
    ):
        raise SystemExit("SOURCE_MANIFEST.sha256 is not strict LF-delimited text")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise SystemExit("SOURCE_MANIFEST.sha256 is not UTF-8") from error
    result: dict[PurePosixPath, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if len(line) < 67 or line[64:66] not in {"  ", " *"}:
            raise SystemExit(f"Malformed source-manifest line {line_number}")
        expected = digest(line[:64], f"source-manifest line {line_number}")
        raw_path = line[66:]
        relative = PurePosixPath(raw_path)
        if (
            not raw_path
            or "\\" in raw_path
            or relative.is_absolute()
            or raw_path != relative.as_posix()
            or raw_path != raw_path.strip()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise SystemExit(f"Unsafe source-manifest path on line {line_number}")
        if relative in result:
            raise SystemExit(f"Duplicate source-manifest path: {relative}")
        result[relative] = expected
    if not result:
        raise SystemExit("SOURCE_MANIFEST.sha256 is empty")
    return result


def ensure_relative_directory(root_descriptor: int, parts: tuple[str, ...]) -> int:
    current = os.dup(root_descriptor)
    try:
        for component in parts:
            try:
                os.mkdir(component, 0o700, dir_fd=current)
            except FileExistsError:
                pass
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=current,
            )
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise SystemExit("Private stage contains a non-directory component")
            os.close(current)
            current = next_descriptor
        return current
    except BaseException:
        os.close(current)
        raise


def create_relative_file(
    root_descriptor: int,
    relative: PurePosixPath,
    payload: bytes,
) -> dict[str, object]:
    parent = ensure_relative_directory(root_descriptor, relative.parts[:-1])
    descriptor = -1
    try:
        descriptor = os.open(
            relative.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o400,
            dir_fd=parent,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SystemExit(f"Short write while staging {relative}")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        status = os.fstat(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)
    return {
        "relative_path": relative.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "mode": int(status.st_mode),
        "create_only": True,
    }


def seal_tree(root: Path) -> None:
    directories: list[Path] = []
    for current, names, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories.append(current_path)
        for name in [*names, *files]:
            child = current_path / name
            status = os.lstat(child)
            if stat.S_ISLNK(status.st_mode):
                raise SystemExit(f"Symlink appeared in private stage: {child}")
        for filename in files:
            os.chmod(current_path / filename, 0o400, follow_symlinks=False)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        os.chmod(directory, 0o500, follow_symlinks=False)


source_root = lexical_absolute(os.environ["SOURCE_ROOT"])
output_root = lexical_absolute(os.environ["OUTPUT_ROOT"])
run_id = os.environ["RUN_ID"]
if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", run_id) is None:
    raise SystemExit("Unsafe SLURM_JOB_ID")
run_name = f"slurm-{run_id}"

output_descriptor = open_directory_nofollow(output_root)
try:
    os.mkdir(run_name, 0o700, dir_fd=output_descriptor)
    run_descriptor = os.open(
        run_name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=output_descriptor,
    )
finally:
    os.close(output_descriptor)
try:
    os.mkdir("stage", 0o700, dir_fd=run_descriptor)
    os.mkdir("artifacts", 0o700, dir_fd=run_descriptor)
    stage_descriptor = os.open(
        "stage",
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=run_descriptor,
    )
finally:
    os.close(run_descriptor)

source_descriptor = open_directory_nofollow(source_root)
try:
    manifest_descriptor = open_relative_regular(
        source_descriptor,
        SOURCE_MANIFEST_RELATIVE,
    )
    try:
        manifest_payload, manifest_before, manifest_after = read_open_regular(
            manifest_descriptor,
            "source manifest",
        )
    finally:
        os.close(manifest_descriptor)
    manifest_observed = hashlib.sha256(manifest_payload).hexdigest()
    manifest_expected = digest(
        os.environ["EXPECTED_SOURCE_MANIFEST_SHA256"],
        "expected source-manifest digest",
    )
    if manifest_observed != manifest_expected:
        raise SystemExit("Source-manifest digest differs from external anchor")
    entries = parse_manifest(manifest_payload)
    required_external = {
        BUILDER_RELATIVE: digest(
            os.environ["EXPECTED_BUILDER_SHA256"],
            "expected builder digest",
        ),
        AUDITOR_RELATIVE: digest(
            os.environ["EXPECTED_AUDITOR_SHA256"],
            "expected auditor digest",
        ),
        JOINT_RELATIVE: digest(
            os.environ["EXPECTED_JOINT_MILP_SHA256"],
            "expected joint-MILP digest",
        ),
    }
    if set(entries) != set(required_external):
        unexpected = sorted(
            relative.as_posix()
            for relative in set(entries).difference(required_external)
        )
        missing = sorted(
            relative.as_posix()
            for relative in set(required_external).difference(entries)
        )
        raise SystemExit(
            "Source manifest must contain exactly the three-file staged "
            "application-source closure: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative, externally_expected in required_external.items():
        if entries.get(relative) != externally_expected:
            raise SystemExit(
                f"Source manifest does not bind the external digest for {relative}"
            )

    source_inventory: dict[str, object] = {}
    staged_inventory: dict[str, object] = {}
    for relative, expected in sorted(
        entries.items(),
        key=lambda item: item[0].as_posix(),
    ):
        payload, reference = acquire_relative(
            source_descriptor,
            source_root,
            relative,
            expected,
            f"source entry {relative}",
        )
        source_inventory[relative.as_posix()] = reference
        staged_inventory[f"source/{relative.as_posix()}"] = create_relative_file(
            stage_descriptor,
            PurePosixPath("source") / relative,
            payload,
        )
finally:
    os.close(source_descriptor)

authority_payload, authority_reference = acquire_absolute(
    os.environ["TRAIN_AUTHORITY_ALLOWLIST"],
    os.environ["EXPECTED_TRAIN_AUTHORITY_SHA256"],
    "physical train-only authority allowlist",
)
manifest_data_payload, manifest_data_reference = acquire_absolute(
    os.environ["TRAIN_CANONICAL_MANIFEST"],
    os.environ["EXPECTED_TRAIN_MANIFEST_SHA256"],
    "physical train-only canonical manifest",
)
staged_inventory["inputs/train_authority_allowlist.json"] = create_relative_file(
    stage_descriptor,
    PurePosixPath("inputs/train_authority_allowlist.json"),
    authority_payload,
)
staged_inventory["inputs/train_canonical_manifest.csv"] = create_relative_file(
    stage_descriptor,
    PurePosixPath("inputs/train_canonical_manifest.csv"),
    manifest_data_payload,
)
staged_inventory["inventory/SOURCE_MANIFEST.sha256"] = create_relative_file(
    stage_descriptor,
    PurePosixPath("inventory/SOURCE_MANIFEST.sha256"),
    manifest_payload,
)

inventory = {
    "format_version": "frontier_private_nofollow_stage_inventory_v1",
    "source_manifest": {
        "path": str(source_root / "SOURCE_MANIFEST.sha256"),
        "sha256": manifest_observed,
        "size_bytes": len(manifest_payload),
        "stat_fingerprint_before": manifest_before,
        "stat_fingerprint_after": manifest_after,
        "single_verified_fd_read": True,
        "symlink_followed": False,
    },
    "source_entries": source_inventory,
    "data_inputs": {
        "train_authority_allowlist": authority_reference,
        "train_canonical_manifest": manifest_data_reference,
    },
    "staged_files": staged_inventory,
    "all_originals_acquired_by_component_wise_nofollow": True,
    "all_original_bytes_hashed_and_copied_from_same_fd": True,
    "exact_three_file_staged_application_source_closure": True,
    "full_runtime_closure_attested": False,
    "external_runtime_trust_roots": [
        "launcher_bytes_and_preexecution_shell_state",
        "bash_and_operating_system",
        "python_interpreter_and_standard_library",
        "biopython_scipy_highs_and_native_dependencies",
    ],
    "secure_open_capabilities": SECURE_OPEN_CAPABILITIES,
    "full_split_or_sealed_manifest_input_accepted": False,
}
inventory_payload = (
    json.dumps(
        inventory,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    + "\n"
).encode("utf-8")
create_relative_file(
    stage_descriptor,
    PurePosixPath("inventory/stage_inventory.json"),
    inventory_payload,
)
os.fsync(stage_descriptor)
os.close(stage_descriptor)
seal_tree(output_root / run_name / "stage")
PY

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
set -o noclobber

"$PYTHON_BIN" -I -B - \
  "$ARTIFACT_ROOT/environment.json" \
  "$RUN_ID" "$RUN_ROOT" "$WORKERS" "$CHUNK_SIZE" "$SEED" \
  "$JOINT_ASSIGNMENT_MODE" "$PROVISIONAL_ATTEMPT_ID" \
  "$PINNED_REPLAY_ID" "$SOURCE_PROVISIONAL_ATTEMPT_ID" \
  "$JOINT_TIME_LIMIT_SECONDS" "$EXPECTED_TRAIN_AUTHORITY_SHA256" \
  "$EXPECTED_TRAIN_MANIFEST_SHA256" "$EXPECTED_BUILDER_SHA256" \
  "$EXPECTED_AUDITOR_SHA256" "$EXPECTED_JOINT_MILP_SHA256" \
  "$EXPECTED_SOURCE_MANIFEST_SHA256" "$EXPECTED_JOINT_POLICY_SHA256" \
  "$EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256" \
  "$REMOVED_INHERITED_FUNCTION_COUNT" "$BASH_VERSION" <<'PY'
from __future__ import annotations

import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import Bio
import scipy


target = Path(sys.argv[1])
payload = {
    "format_version": "frontier_nested_launcher_environment_v1",
    "job_id": sys.argv[2],
    "node": socket.gethostname(),
    "run_root": sys.argv[3],
    "workers": int(sys.argv[4]),
    "chunk_size": int(sys.argv[5]),
    "selection_seed": int(sys.argv[6]),
    "joint_assignment_mode": sys.argv[7],
    "provisional_attempt_id": sys.argv[8] or None,
    "pinned_replay_id": sys.argv[9] or None,
    "source_provisional_attempt_id": sys.argv[10] or None,
    "joint_time_limit_seconds": float(sys.argv[11]),
    "expected_train_authority_sha256": sys.argv[12],
    "expected_train_manifest_sha256": sys.argv[13],
    "expected_builder_sha256": sys.argv[14],
    "expected_auditor_sha256": sys.argv[15],
    "expected_joint_milp_sha256": sys.argv[16],
    "expected_source_manifest_sha256": sys.argv[17],
    "expected_joint_policy_sha256": sys.argv[18],
    "expected_joint_selected_membership_sha256": sys.argv[19] or None,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "python_executable": sys.executable,
    "biopython_version": Bio.__version__,
    "scipy_version": scipy.__version__,
    "bash_version": sys.argv[21],
    "trusted_command_path": "/usr/bin:/bin",
    "removed_inherited_shell_function_count": int(sys.argv[20]),
    "launcher_preexecution_shell_state_attested": False,
    "launcher_bytes_in_application_source_manifest": False,
    "full_runtime_closure_attested": False,
    "exact_three_file_staged_application_source_closure": True,
    "successful_completion_required_for_final_artifact_inventory_and_tree_seal": (
        True
    ),
    "artifact_inventory_is_conservative_until_post_seal_verification": True,
    "post_seal_launcher_success_receipt_required": True,
    "launcher_success_receipt_path": str(
        Path(sys.argv[3]) / "launcher_success.json"
    ),
    "failed_launch_artifact_tree_inventory_or_seal_guaranteed": False,
    "external_runtime_trust_roots": [
        "launcher_bytes_and_preexecution_shell_state",
        "bash_and_operating_system",
        "python_interpreter_and_standard_library",
        "biopython_scipy_highs_and_native_dependencies",
    ],
}
encoded = (
    json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    )
    + "\n"
).encode("utf-8")
descriptor = os.open(
    target,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
    0o400,
)
try:
    view = memoryview(encoded)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("Short launcher-environment write")
        view = view[written:]
    os.fchmod(descriptor, 0o400)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY

BUILDER_COMMAND=(
  "$PYTHON_BIN" -I -B "$STAGED_BUILDER"
  --train-authority-allowlist "$STAGED_TRAIN_AUTHORITY"
  --train-authority-allowlist-sha256 "$EXPECTED_TRAIN_AUTHORITY_SHA256"
  --train-canonical-manifest "$STAGED_TRAIN_MANIFEST"
  --train-canonical-manifest-sha256 "$EXPECTED_TRAIN_MANIFEST_SHA256"
  --expected-builder-source-sha256 "$EXPECTED_BUILDER_SHA256"
  --joint-milp-source "$STAGED_JOINT_MILP"
  --expected-joint-milp-source-sha256 "$EXPECTED_JOINT_MILP_SHA256"
  --expected-joint-policy-sha256 "$EXPECTED_JOINT_POLICY_SHA256"
  --joint-assignment-mode "$JOINT_ASSIGNMENT_MODE"
  --output "$ASSIGNMENT"
  --fold-count 3
  --selection-method joint_core_milp
  --selection-seed "$SEED"
  --joint-time-limit-seconds "$JOINT_TIME_LIMIT_SECONDS"
  --workers "$WORKERS"
  --chunk-size "$CHUNK_SIZE"
)
if [[ "$JOINT_ASSIGNMENT_MODE" == "provisional_discovery" ]]; then
  BUILDER_COMMAND+=(
    --provisional-attempt-id "$PROVISIONAL_ATTEMPT_ID"
    --provisional-ledger-entry "$PROVISIONAL_LEDGER_ENTRY"
  )
else
  BUILDER_COMMAND+=(
    --expected-joint-selected-membership-sha256 "$EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256"
    --pinned-replay-id "$PINNED_REPLAY_ID"
    --source-provisional-attempt-id "$SOURCE_PROVISIONAL_ATTEMPT_ID"
  )
fi
"${BUILDER_COMMAND[@]}" > "$ARTIFACT_ROOT/builder.log" 2>&1

if [[ "$JOINT_ASSIGNMENT_MODE" == "provisional_discovery" ]]; then
  printf '%s\n' \
    "Provisional attempt preserved. Freeze its digest externally, then launch a separate externally_pinned_replay." \
    > "$ARTIFACT_ROOT/provisional_next_step.txt"
else
  # Copy the generated assignment into a new sealed audit-input directory.
  # The digest and scientific anchor are extracted from the same acquired bytes.
  read -r ASSIGNMENT_SHA256 ASSIGNMENT_SCIENTIFIC_SHA256 < <(
    "$PYTHON_BIN" -I -S -B - "$ASSIGNMENT" "$RUN_ROOT" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def absolute(value: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(value)))


def open_dir(path: Path) -> int:
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for part in absolute(os.fspath(path)).parts[1:]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                os.close(next_descriptor)
                raise RuntimeError("Non-directory path component")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


assignment = absolute(sys.argv[1])
parent = open_dir(assignment.parent)
descriptor = os.open(
    assignment.name,
    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
    dir_fd=parent,
)
os.close(parent)
try:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("Generated assignment is not a regular file")
    chunks: list[bytes] = []
    remaining = int(before.st_size)
    while remaining:
        chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise RuntimeError("Generated assignment was truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise RuntimeError("Generated assignment grew during acquisition")
    after = os.fstat(descriptor)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if tuple(getattr(before, field) for field in fields) != tuple(
        getattr(after, field) for field in fields
    ):
        raise RuntimeError("Generated assignment changed during acquisition")
    payload = b"".join(chunks)
finally:
    os.close(descriptor)

if payload.startswith(b"\xef\xbb\xbf"):
    raise RuntimeError("Generated assignment contains a BOM")
parsed = json.loads(
    payload.decode("utf-8"),
    object_pairs_hook=reject_duplicates,
    parse_constant=reject_constant,
)
if not isinstance(parsed, dict):
    raise RuntimeError("Generated assignment is not a JSON object")
scientific = parsed.get("scientific_payload_sha256")
if (
    not isinstance(scientific, str)
    or len(scientific) != 64
    or any(character not in "0123456789abcdef" for character in scientific)
):
    raise RuntimeError("Generated assignment has no valid scientific digest")

run_root = absolute(sys.argv[2])
run_descriptor = open_dir(run_root)
try:
    os.mkdir("audit_input", 0o700, dir_fd=run_descriptor)
    input_descriptor = os.open(
        "audit_input",
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=run_descriptor,
    )
finally:
    os.close(run_descriptor)
try:
    output = os.open(
        "assignment.json",
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | os.O_NOFOLLOW,
        0o400,
        dir_fd=input_descriptor,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(output, view)
            if written <= 0:
                raise RuntimeError("Short assignment staging write")
            view = view[written:]
        os.fchmod(output, 0o400)
        os.fsync(output)
    finally:
        os.close(output)
    os.fchmod(input_descriptor, 0o500)
    os.fsync(input_descriptor)
finally:
    os.close(input_descriptor)
print(hashlib.sha256(payload).hexdigest(), scientific)
PY
  )
  SEALED_ASSIGNMENT=$RUN_ROOT/audit_input/assignment.json
  "$PYTHON_BIN" -I -B "$STAGED_AUDITOR" \
    --assignment "$SEALED_ASSIGNMENT" \
    --assignment-sha256 "$ASSIGNMENT_SHA256" \
    --assignment-scientific-sha256 "$ASSIGNMENT_SCIENTIFIC_SHA256" \
    --train-authority-allowlist "$STAGED_TRAIN_AUTHORITY" \
    --train-authority-allowlist-sha256 "$EXPECTED_TRAIN_AUTHORITY_SHA256" \
    --train-canonical-manifest "$STAGED_TRAIN_MANIFEST" \
    --train-canonical-manifest-sha256 "$EXPECTED_TRAIN_MANIFEST_SHA256" \
    --expected-auditor-source-sha256 "$EXPECTED_AUDITOR_SHA256" \
    --builder-source "$STAGED_BUILDER" \
    --expected-builder-source-sha256 "$EXPECTED_BUILDER_SHA256" \
    --joint-milp-source "$STAGED_JOINT_MILP" \
    --expected-joint-milp-source-sha256 "$EXPECTED_JOINT_MILP_SHA256" \
    --expected-joint-policy-sha256 "$EXPECTED_JOINT_POLICY_SHA256" \
    --expected-joint-selected-membership-sha256 "$EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256" \
    --expected-pinned-replay-id "$PINNED_REPLAY_ID" \
    --expected-source-provisional-attempt-id "$SOURCE_PROVISIONAL_ATTEMPT_ID" \
    --output "$AUDIT" \
    --workers "$WORKERS" \
    --chunk-size "$CHUNK_SIZE" \
    > "$ARTIFACT_ROOT/auditor.log" 2>&1
fi

# This begins the successful-workflow finalizer. Because set -e exits before
# this point on earlier failures, failed workflows do not publish an artifact
# inventory. The inventory itself remains conservative about the subsequent
# seal; only a separate post-seal receipt can claim final launcher success.
# Individual JSON evidence is create-only and read-only; partial logs are
# outside that guarantee.
"$PYTHON_BIN" -I -S -B - \
  "$ARTIFACT_ROOT/workflow_steps_completion.json" <<'PY'
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


target = Path(sys.argv[1])
encoded = (
    json.dumps(
        {
            "format_version": "frontier_nested_workflow_steps_completion_v1",
            "workflow_steps_completed_at_utc": (
                datetime.now(timezone.utc).isoformat()
            ),
            "workflow_steps_completed_before_finalization": True,
            "launcher_success_attested": False,
            "post_seal_launcher_success_receipt_required": True,
        },
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    + "\n"
).encode("utf-8")
descriptor = os.open(
    target,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
    0o400,
)
try:
    view = memoryview(encoded)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("Short launcher-completion write")
        view = view[written:]
    os.fchmod(descriptor, 0o400)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY

"$PYTHON_BIN" -I -S -B - \
  "$ARTIFACT_ROOT" "$RUN_ROOT/launcher_success.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path


root = Path(os.path.abspath(sys.argv[1]))
success_target = Path(os.path.abspath(sys.argv[2]))
if (
    success_target.parent != root.parent
    or success_target.name != "launcher_success.json"
):
    raise RuntimeError("Launcher-success receipt must be the fixed run-root sibling")
inventory_name = "artifact_sha256.json"
entries: dict[str, dict[str, object]] = {}
for current, directories, files in os.walk(root, topdown=True, followlinks=False):
    current_path = Path(current)
    for name in [*directories, *files]:
        child = current_path / name
        if stat.S_ISLNK(os.lstat(child).st_mode):
            raise RuntimeError(f"Artifact tree contains a symlink: {child}")
    for filename in files:
        path = current_path / filename
        relative = path.relative_to(root).as_posix()
        if relative == inventory_name:
            raise RuntimeError("Create-only artifact inventory already exists")
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(f"Artifact is not regular: {relative}")
            digest = hashlib.sha256()
            remaining = int(before.st_size)
            while remaining:
                chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError(f"Artifact was truncated: {relative}")
                digest.update(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise RuntimeError(f"Artifact grew: {relative}")
            after = os.fstat(descriptor)
            fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if tuple(getattr(before, field) for field in fields) != tuple(
                getattr(after, field) for field in fields
            ):
                raise RuntimeError(f"Artifact changed: {relative}")
        finally:
            os.close(descriptor)
        entries[relative] = {
            "sha256": digest.hexdigest(),
            "size_bytes": int(before.st_size),
        }
payload = (
    json.dumps(
        {
            "format_version": "frontier_nested_artifact_inventory_v1",
            "artifacts": entries,
            "successful_launcher_completion": False,
            "artifact_tree_sealed_after_inventory": False,
            "artifact_tree_seal_pending_at_inventory_publication": True,
            "separate_post_seal_success_receipt_required": True,
            "inventory_file_self_excluded": True,
            "sealed_file_mode": "0400",
            "sealed_directory_mode": "0500",
        },
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    + "\n"
).encode("utf-8")
descriptor = os.open(
    root / inventory_name,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
    0o400,
)
try:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("Short artifact-inventory write")
        view = view[written:]
    os.fchmod(descriptor, 0o400)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
directory_descriptor = os.open(
    root,
    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
)
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)

for current, directories, files in os.walk(
    root,
    topdown=False,
    followlinks=False,
):
    current_path = Path(current)
    for filename in files:
        os.chmod(
            current_path / filename,
            0o400,
            follow_symlinks=False,
        )
    for directory in directories:
        os.chmod(
            current_path / directory,
            0o500,
            follow_symlinks=False,
        )
    os.chmod(current_path, 0o500, follow_symlinks=False)

expected_files = set(entries) | {inventory_name}
observed_files: set[str] = set()
sealed_directories: list[Path] = []
inventory_sha256 = hashlib.sha256(payload).hexdigest()
for current, directories, files in os.walk(
    root,
    topdown=True,
    followlinks=False,
):
    current_path = Path(current)
    sealed_directories.append(current_path)
    current_status = os.lstat(current_path)
    if (
        not stat.S_ISDIR(current_status.st_mode)
        or stat.S_IMODE(current_status.st_mode) != 0o500
    ):
        raise RuntimeError(f"Artifact directory was not sealed: {current_path}")
    for directory in directories:
        child = current_path / directory
        child_status = os.lstat(child)
        if (
            not stat.S_ISDIR(child_status.st_mode)
            or stat.S_IMODE(child_status.st_mode) != 0o500
        ):
            raise RuntimeError(f"Artifact directory was not sealed: {child}")
    for filename in files:
        path = current_path / filename
        relative = path.relative_to(root).as_posix()
        observed_files.add(relative)
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o400
        ):
            raise RuntimeError(f"Artifact file was not sealed: {relative}")
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        try:
            digest = hashlib.sha256()
            remaining = int(before.st_size)
            while remaining:
                chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError(
                        f"Sealed artifact was truncated: {relative}"
                    )
                digest.update(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise RuntimeError(f"Sealed artifact grew: {relative}")
            os.fsync(descriptor)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if tuple(getattr(before, field) for field in fields) != tuple(
            getattr(after, field) for field in fields
        ):
            raise RuntimeError(f"Sealed artifact changed: {relative}")
        expected = (
            {
                "sha256": inventory_sha256,
                "size_bytes": len(payload),
            }
            if relative == inventory_name
            else entries.get(relative)
        )
        observed = {
            "sha256": digest.hexdigest(),
            "size_bytes": int(before.st_size),
        }
        if expected != observed:
            raise RuntimeError(
                f"Sealed artifact differs from its inventory: {relative}"
            )
if observed_files != expected_files:
    raise RuntimeError(
        "Post-seal artifact inventory differs: "
        f"missing={sorted(expected_files - observed_files)}, "
        f"unexpected={sorted(observed_files - expected_files)}"
    )
for directory in sorted(
    sealed_directories,
    key=lambda value: len(value.parts),
    reverse=True,
):
    directory_descriptor = os.open(
        directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)

success_payload = (
    json.dumps(
        {
            "format_version": "frontier_nested_launcher_success_v1",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "successful_launcher_completion": True,
            "successful_workflow_steps_completed": True,
            "artifact_tree_sealed_after_inventory": True,
            "artifact_tree_post_seal_hash_and_mode_verification_passed": True,
            "artifact_tree_post_seal_fsync_completed": True,
            "artifact_inventory_path": str(root / inventory_name),
            "artifact_inventory_sha256": inventory_sha256,
            "artifact_inventory_size_bytes": len(payload),
            "artifact_inventory_conservatively_records_seal_pending": True,
            "success_receipt_is_outside_artifact_inventory": True,
            "success_receipt_is_not_self_inventoried": True,
            "full_run_root_tree_sealed": False,
            "sealed_file_mode": "0400",
            "sealed_directory_mode": "0500",
        },
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    + "\n"
).encode("utf-8")
success_descriptor = os.open(
    success_target,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
    0o400,
)
try:
    view = memoryview(success_payload)
    while view:
        written = os.write(success_descriptor, view)
        if written <= 0:
            raise RuntimeError("Short launcher-success write")
        view = view[written:]
    os.fchmod(success_descriptor, 0o400)
    os.fsync(success_descriptor)
finally:
    os.close(success_descriptor)
success_parent_descriptor = os.open(
    success_target.parent,
    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
)
try:
    os.fsync(success_parent_descriptor)
finally:
    os.close(success_parent_descriptor)
PY
