"""Authenticated production control for V6-B exact feasibility.

The controller in this module is deliberately non-authorizing. A successful CP-SAT
and exact-oracle candidate still requires the separately locked direct-HiGHS fixed
assignment audit before Gate 0 can pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import (
    score_free_independent_auditor,
    score_free_split_statistics,
    v6_b_decomposition,
    v6_b_independent_core_auditor,
    v6_cp_sat_formulation,
    v6_exact_separation_oracle,
    v6_joint_ir,
)


REGISTRATION_FORMAT = "v6_b_production_feasibility_registration_v2"
SUPPORTED_REGISTRATION_FORMATS = {
    "v6_b_production_feasibility_registration_v1",
    REGISTRATION_FORMAT,
}
ATTEMPT_FORMAT = "v6_b_production_feasibility_attempt_v1"
JOURNAL_FORMAT = "v6_b_production_hash_chain_event_v1"
FINAL_FORMAT = "v6_b_production_feasibility_final_receipt_v1"
EXPECTED_COUNTS = {
    "source_rows": 3229,
    "exact_pairs": 1830,
    "antibody_entities": 1383,
    "antigen_entities": 1172,
}
EXPECTED_SOURCE_FILES = {
    "score_free_independent_auditor": score_free_independent_auditor,
    "score_free_split_statistics": score_free_split_statistics,
    "v6_b_decomposition": v6_b_decomposition,
    "v6_b_independent_core_auditor": v6_b_independent_core_auditor,
    "v6_cp_sat_formulation": v6_cp_sat_formulation,
    "v6_exact_separation_oracle": v6_exact_separation_oracle,
    "v6_joint_ir": v6_joint_ir,
    "v6_b_production": sys.modules[__name__],
}


class V6BProductionError(ValueError):
    """Raised when production evidence or execution fails closed."""


class V6BTerminationRequested(RuntimeError):
    """Raised by the bounded Slurm termination handler."""


def canonical_json_bytes(payload: object) -> bytes:
    """Encode canonical ASCII JSON with one terminal line feed."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_payload(payload: object) -> str:
    return _sha256_bytes(canonical_json_bytes(payload))


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V6BProductionError(f"{label} must be a lowercase SHA-256")
    return value


def _git_object_id(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V6BProductionError(f"{label} must be a lowercase Git object ID")
    return value


def strict_json(payload: bytes, label: str) -> dict[str, Any]:
    """Parse one strict ASCII JSON object and reject duplicate fields."""
    if not payload or payload.startswith(b"\xef\xbb\xbf"):
        raise V6BProductionError(f"{label} is empty or contains a BOM")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise V6BProductionError(f"{label} must be ASCII JSON") from error

    def no_duplicates(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise V6BProductionError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        result = json.loads(
            text,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                V6BProductionError(f"{label} contains non-finite {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise V6BProductionError(f"{label} is invalid JSON") from error
    if not isinstance(result, dict):
        raise V6BProductionError(f"{label} must be a JSON object")
    return result


def _absolute_regular_file(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    current = Path(absolute.parts[0])
    for part in absolute.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise V6BProductionError(f"{label} path contains a symbolic link")
    if not absolute.is_file():
        raise V6BProductionError(f"{label} is not a regular file")
    return absolute


def read_verified(path: Path, expected_sha256: str, label: str) -> bytes:
    """Read and hash bytes through the same stable no-follow descriptor."""
    expected = _digest(expected_sha256, label)
    absolute = _absolute_regular_file(path, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    fingerprint = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if fingerprint(before) != fingerprint(after) or len(payload) != before.st_size:
        raise V6BProductionError(f"{label} changed during authentication")
    if _sha256_bytes(payload) != expected:
        raise V6BProductionError(f"{label} SHA-256 differs")
    return payload


def extract_scientific_authority(
    published_bytes: bytes,
) -> tuple[bytes, dict[str, object]]:
    """Authenticate the publication envelope and return its scientific fields."""
    published = strict_json(published_bytes, "published authority")
    scientific_keys = {
        "format_version",
        "input_sha256",
        "authority_membership_sha256",
        "implementation_source_sha256",
        "counts",
        "antibodies",
        "antigens",
        "pairs",
        "quantile_schemes",
        "forbidden_outcome_or_model_fields_accepted",
        "scientific_payload_sha256",
    }
    envelope_keys = {
        "verified_file_receipts",
        "publication_controller_sha256",
        "publication_sha256",
    }
    if set(published) != scientific_keys | envelope_keys:
        raise V6BProductionError("published authority envelope schema differs")
    publication_sha256 = _digest(
        published.pop("publication_sha256"), "authority publication"
    )
    if _sha256_payload(published) != publication_sha256:
        raise V6BProductionError("authority publication digest differs")
    controller_sha256 = _digest(
        published.pop("publication_controller_sha256"),
        "authority publication controller",
    )
    verified = published.pop("verified_file_receipts")
    expected_receipts = {
        "train_authority",
        "canonical_manifest",
        *(f"edge_ledger_{view}" for view in v6_cp_sat_formulation.VIEWS),
    }
    if not isinstance(verified, Mapping) or set(verified) != expected_receipts:
        raise V6BProductionError("authority verified-file receipts differ")
    scientific_sha256 = _digest(
        published.get("scientific_payload_sha256"),
        "scientific authority payload",
    )
    scientific_core = dict(published)
    scientific_core.pop("scientific_payload_sha256")
    if _sha256_payload(scientific_core) != scientific_sha256:
        raise V6BProductionError("scientific authority digest differs")
    scientific_bytes = canonical_json_bytes(published)
    return scientific_bytes, {
        "publication_sha256": publication_sha256,
        "publication_controller_sha256": controller_sha256,
        "verified_file_receipts_sha256": _sha256_payload(verified),
        "scientific_payload_sha256": scientific_sha256,
        "scientific_file_sha256": _sha256_bytes(scientific_bytes),
    }


def normalize_scientific_pair_order(
    scientific_bytes: bytes,
) -> tuple[bytes, dict[str, object]]:
    """Sort complete pair records by pair ID after proving endpoint order."""
    scientific = strict_json(scientific_bytes, "scientific authority")
    pairs = scientific.get("pairs")
    required = {"pair_id", "antibody_id", "antigen_id", "source_instance_ids"}
    if (
        not isinstance(pairs, list)
        or not pairs
        or any(not isinstance(row, dict) or set(row) != required for row in pairs)
    ):
        raise V6BProductionError("scientific authority pair schema differs")
    endpoint_order = sorted(
        pairs,
        key=lambda row: (
            str(row["antibody_id"]),
            str(row["antigen_id"]),
            str(row["pair_id"]),
        ),
    )
    if pairs != endpoint_order:
        raise V6BProductionError("published pairs are not in canonical endpoint order")
    pair_ids = [str(row["pair_id"]) for row in pairs]
    endpoints = [
        (str(row["antibody_id"]), str(row["antigen_id"])) for row in pairs
    ]
    if len(pair_ids) != len(set(pair_ids)) or len(endpoints) != len(set(endpoints)):
        raise V6BProductionError("published pair IDs or endpoints are duplicated")
    row_hashes = sorted(_sha256_payload(row) for row in pairs)
    ordered = sorted(pairs, key=lambda row: str(row["pair_id"]))
    scientific["pairs"] = ordered
    scientific.pop("scientific_payload_sha256", None)
    scientific["scientific_payload_sha256"] = _sha256_payload(scientific)
    normalized = canonical_json_bytes(scientific)
    if row_hashes != sorted(_sha256_payload(row) for row in ordered):
        raise V6BProductionError("pair normalization changed a complete record")
    return normalized, {
        "operation": "whole_record_sort_by_lowercase_pair_id",
        "pair_count": len(pairs),
        "record_multiset_sha256": _sha256_payload(row_hashes),
        "original_order_sha256": _sha256_payload(pair_ids),
        "normalized_order_sha256": _sha256_payload(
            [str(row["pair_id"]) for row in ordered]
        ),
        "normalized_scientific_payload_sha256": scientific[
            "scientific_payload_sha256"
        ],
        "normalized_scientific_file_sha256": _sha256_bytes(normalized),
        "semantic_fields_changed": False,
    }


def validate_registration(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the frozen production registration without supplying defaults."""
    expected_keys = {
        "format_version",
        "parent_evidence",
        "authority",
        "raw_inputs",
        "runtime",
        "source",
        "limits",
        "slurm",
        "authorization",
    }
    if set(payload) != expected_keys:
        raise V6BProductionError("production registration schema differs")
    registration_format = payload.get("format_version")
    if (
        not isinstance(registration_format, str)
        or registration_format not in SUPPORTED_REGISTRATION_FORMATS
    ):
        raise V6BProductionError("production registration format differs")
    parent = payload.get("parent_evidence")
    if not isinstance(parent, Mapping) or set(parent) != {
        "v6_b_policy",
        "v6_b_protocol",
        "row_shard_telemetry",
    }:
        raise V6BProductionError("registration parent-evidence schema differs")
    for label in ("v6_b_policy", "v6_b_protocol"):
        reference = parent[label]
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise V6BProductionError(f"registered parent {label} is malformed")
        _digest(reference.get("sha256"), f"registered parent {label}")
    telemetry = parent["row_shard_telemetry"]
    if not isinstance(telemetry, Mapping) or set(telemetry) != {
        "path",
        "file_sha256",
        "receipt_sha256",
        "terminal_status",
    }:
        raise V6BProductionError("registered row-shard telemetry is malformed")
    _digest(telemetry.get("file_sha256"), "registered telemetry file")
    _digest(telemetry.get("receipt_sha256"), "registered telemetry receipt")
    if telemetry.get("terminal_status") != "TELEMETRY_HEADROOM_PASSED":
        raise V6BProductionError("registered telemetry did not pass headroom")
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping) or set(authorization) != {
        "split_gate0_authorized",
        "training_authorized",
        "test_access_authorized",
    }:
        raise V6BProductionError("registration authorization schema differs")
    if any(value is not False for value in authorization.values()):
        raise V6BProductionError("registration grants forbidden authority")
    limits = payload.get("limits")
    if not isinstance(limits, Mapping) or set(limits) != {
        "maximum_master_iterations",
        "maximum_combined_candidates",
        "maximum_time_seconds_per_solve",
    }:
        raise V6BProductionError("registration limits schema differs")
    for name in (
        "maximum_master_iterations",
        "maximum_combined_candidates",
        "maximum_time_seconds_per_solve",
    ):
        value = limits.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise V6BProductionError(f"registration {name} must be positive")
    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or set(authority) != {
        "path",
        "sha256",
        "scientific_payload_sha256",
        "counts",
    }:
        raise V6BProductionError("registration authority schema differs")
    if authority.get("counts") != EXPECTED_COUNTS:
        raise V6BProductionError("registration authority counts differ")
    _digest(authority.get("sha256"), "registered authority")
    _digest(
        authority.get("scientific_payload_sha256"),
        "registered authority scientific payload",
    )
    raw_inputs = payload.get("raw_inputs")
    required_raw = {
        "train_authority",
        "canonical_manifest",
        *(f"edge_ledger_{view}" for view in v6_cp_sat_formulation.VIEWS),
    }
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != required_raw:
        raise V6BProductionError("registration raw-input schema differs")
    for label, reference in raw_inputs.items():
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise V6BProductionError(f"registered raw input {label} is malformed")
        _digest(reference.get("sha256"), f"registered raw input {label}")
    source = payload.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "implementation_commit",
        "source_root",
        "files",
    }:
        raise V6BProductionError("registration source schema differs")
    _git_object_id(source.get("implementation_commit"), "implementation commit")
    files = source.get("files")
    required_source = set(EXPECTED_SOURCE_FILES) | {
        "production_controller",
        "slurm_launcher",
    }
    if not isinstance(files, Mapping) or set(files) != required_source:
        raise V6BProductionError("registration source-file set differs")
    for label, reference in files.items():
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise V6BProductionError(f"registered source {label} is malformed")
        _digest(reference.get("sha256"), f"registered source {label}")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "python_executable",
        "python_version",
        "ortools_version",
        "platform",
        "cp_sat_runtime_receipt_path",
        "cp_sat_runtime_receipt_sha256",
    }:
        raise V6BProductionError("registration runtime schema differs")
    _digest(
        runtime.get("cp_sat_runtime_receipt_sha256"),
        "registered CP-SAT runtime receipt",
    )
    slurm = payload.get("slurm")
    common_slurm_fields = {
        "partition",
        "nodes",
        "tasks",
        "cpus_per_task",
        "memory_gib",
        "wall_time",
        "gpus_requested",
        "solver_workers",
    }
    expected_slurm_fields = (
        common_slurm_fields
        if registration_format == "v6_b_production_feasibility_registration_v1"
        else common_slurm_fields
        | {"accelerator_visible_to_solver", "allocated_gpu_purpose"}
    )
    if not isinstance(slurm, Mapping) or set(slurm) != expected_slurm_fields:
        raise V6BProductionError("registration Slurm schema differs")
    if slurm.get("nodes") != 1 or slurm.get("tasks") != 1:
        raise V6BProductionError("production registration must use one task")
    if slurm.get("solver_workers") != 1:
        raise V6BProductionError("production registration must use one solver worker")
    if registration_format == "v6_b_production_feasibility_registration_v1":
        if slurm.get("gpus_requested") != 0:
            raise V6BProductionError("V1 registration must retain its zero-GPU request")
    else:
        if slurm.get("gpus_requested") != 1:
            raise V6BProductionError(
                "scheduler-compatible production must request one GPU"
            )
        if slurm.get("accelerator_visible_to_solver") is not False:
            raise V6BProductionError("the allocated scheduler GPU must be hidden")
        if slurm.get("allocated_gpu_purpose") != "site_scheduler_requirement_only":
            raise V6BProductionError("the allocated GPU purpose differs")
    return dict(payload)


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass
class ProgressJournal:
    """Append canonical hash-chained events and fsync every completed record."""

    path: Path
    registration_sha256: str
    previous_event_sha256: str | None = None
    event_count: int = 0
    exact_oracle: object | None = None

    def append(self, event_type: str, evidence: Mapping[str, object]) -> dict[str, object]:
        self.event_count += 1
        core: dict[str, object] = {
            "format_version": JOURNAL_FORMAT,
            "event_index": self.event_count,
            "event_type": event_type,
            "previous_event_sha256": self.previous_event_sha256,
            "registration_sha256": self.registration_sha256,
            "evidence": dict(evidence),
            "split_gate0_authorized": False,
            "training_authorized": False,
        }
        core["event_sha256"] = _sha256_payload(core)
        payload = canonical_json_bytes(core)
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_APPEND | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.previous_event_sha256 = str(core["event_sha256"])
        return core

    def solver_event(self, event: Mapping[str, object]) -> None:
        """Persist one controller event and the full exact evidence when available."""
        self.append("DECOMPOSITION_EVENT", event)
        if (
            event.get("event_type") == "EXACT_ORACLE_EVALUATED"
            and self.exact_oracle is not None
        ):
            history = getattr(self.exact_oracle, "evidence_history", None)
            if not isinstance(history, list) or not history:
                raise V6BProductionError("exact oracle evidence history is absent")
            self.append("FULL_EXACT_ORACLE_EVIDENCE", history[-1])


def create_attempt_directory(
    output_directory: Path,
    registration_sha256: str,
    registration: Mapping[str, object],
) -> ProgressJournal:
    """Create a non-overwriting attempt root and durable start evidence."""
    target = Path(os.path.abspath(os.path.expanduser(str(output_directory))))
    if target.exists() or target.is_symlink():
        raise V6BProductionError("output directory already exists")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise V6BProductionError("output parent is absent or symbolic")
    os.mkdir(target, 0o700)
    started = {
        "format_version": ATTEMPT_FORMAT,
        "registration_sha256": registration_sha256,
        "registration_payload_sha256": _sha256_payload(registration),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "process_id": os.getpid(),
        "started_unix_ns": time.time_ns(),
        "split_gate0_authorized": False,
        "training_authorized": False,
    }
    started["receipt_sha256"] = _sha256_payload(started)
    _write_exclusive(target / "started.json", canonical_json_bytes(started))
    _write_exclusive(target / "progress.jsonl", b"")
    journal = ProgressJournal(target / "progress.jsonl", registration_sha256)
    journal.append("ATTEMPT_STARTED", {"started_receipt_sha256": started["receipt_sha256"]})
    return journal


def _source_receipts(registration: Mapping[str, object]) -> dict[str, object]:
    source = registration["source"]
    assert isinstance(source, Mapping)
    source_root = Path(str(source["source_root"]))
    files = source["files"]
    assert isinstance(files, Mapping)
    receipts: dict[str, object] = {}
    for label, module in EXPECTED_SOURCE_FILES.items():
        reference = files[label]
        assert isinstance(reference, Mapping)
        registered_path = source_root / str(reference["path"])
        module_path = Path(str(module.__file__)).resolve()
        if registered_path.resolve() != module_path:
            raise V6BProductionError(f"imported source path differs for {label}")
        payload = read_verified(
            registered_path,
            str(reference["sha256"]),
            f"source {label}",
        )
        receipts[label] = {
            "path": str(registered_path),
            "sha256": _sha256_bytes(payload),
            "byte_size": len(payload),
        }
    launcher_reference = files["slurm_launcher"]
    assert isinstance(launcher_reference, Mapping)
    launcher_path = source_root / str(launcher_reference["path"])
    launcher_bytes = read_verified(
        launcher_path,
        str(launcher_reference["sha256"]),
        "source slurm_launcher",
    )
    receipts["slurm_launcher"] = {
        "path": str(launcher_path),
        "sha256": _sha256_bytes(launcher_bytes),
        "byte_size": len(launcher_bytes),
    }
    return receipts


def _parent_evidence_receipts(
    registration: Mapping[str, object],
) -> dict[str, object]:
    """Authenticate the frozen V6-B policy, protocol, and telemetry receipt."""
    parent = registration["parent_evidence"]
    assert isinstance(parent, Mapping)
    policy_reference = parent["v6_b_policy"]
    protocol_reference = parent["v6_b_protocol"]
    telemetry_reference = parent["row_shard_telemetry"]
    assert isinstance(policy_reference, Mapping)
    assert isinstance(protocol_reference, Mapping)
    assert isinstance(telemetry_reference, Mapping)
    policy_bytes = read_verified(
        Path(str(policy_reference["path"])),
        str(policy_reference["sha256"]),
        "V6-B policy",
    )
    protocol_bytes = read_verified(
        Path(str(protocol_reference["path"])),
        str(protocol_reference["sha256"]),
        "V6-B protocol",
    )
    telemetry_bytes = read_verified(
        Path(str(telemetry_reference["path"])),
        str(telemetry_reference["file_sha256"]),
        "V6-B row-shard telemetry",
    )
    policy = strict_json(policy_bytes, "V6-B policy")
    if policy.get("format_version") != (
        "train_only_v6_b_logic_based_decomposition_policy_v1"
    ):
        raise V6BProductionError("V6-B policy format differs")
    policy_authorization = policy.get("authorization")
    if (
        not isinstance(policy_authorization, Mapping)
        or policy_authorization.get("split_gate0_authorized") is not False
        or policy_authorization.get("contact_training_authorized") is not False
        or policy_authorization.get("test_access_authorized") is not False
    ):
        raise V6BProductionError("V6-B policy grants forbidden authority")
    registered_authority = registration["authority"]
    assert isinstance(registered_authority, Mapping)
    policy_authority = policy.get("authority_train")
    if (
        not isinstance(policy_authority, Mapping)
        or policy_authority.get("score_free_authority_file_sha256")
        != registered_authority["sha256"]
        or policy_authority.get("score_free_authority_scientific_sha256")
        != registered_authority["scientific_payload_sha256"]
    ):
        raise V6BProductionError("V6-B policy authority binding differs")
    source_files = registration["source"]["files"]
    policy_sources = policy.get("source_bindings")
    if (
        not isinstance(policy_sources, Mapping)
        or policy_sources.get("independent_auditor_sha256")
        != source_files["score_free_independent_auditor"]["sha256"]
        or policy_sources.get("v6_a_exact_oracle_sha256")
        != source_files["v6_exact_separation_oracle"]["sha256"]
    ):
        raise V6BProductionError("V6-B policy source binding differs")
    telemetry = strict_json(telemetry_bytes, "V6-B row-shard telemetry")
    claimed_telemetry = telemetry.pop("receipt_sha256", None)
    if (
        claimed_telemetry != telemetry_reference["receipt_sha256"]
        or claimed_telemetry != _sha256_payload(telemetry)
        or telemetry.get("terminal_status") != "TELEMETRY_HEADROOM_PASSED"
        or telemetry.get("solver_invoked") is not False
        or telemetry.get("training_authorized") is not False
    ):
        raise V6BProductionError("V6-B telemetry receipt differs or is authorizing")
    return {
        "v6_b_policy_sha256": _sha256_bytes(policy_bytes),
        "v6_b_protocol_sha256": _sha256_bytes(protocol_bytes),
        "row_shard_telemetry_file_sha256": _sha256_bytes(telemetry_bytes),
        "row_shard_telemetry_receipt_sha256": claimed_telemetry,
    }


def _runtime_receipt(registration: Mapping[str, object]) -> dict[str, object]:
    runtime = registration["runtime"]
    assert isinstance(runtime, Mapping)
    executable = Path(sys.executable).resolve()
    registered_executable = Path(str(runtime["python_executable"]))
    if registered_executable.resolve() != executable:
        raise V6BProductionError("Python executable differs from registration")
    observed_python = platform.python_version()
    observed_platform = platform.platform()
    try:
        import ortools
    except ImportError as error:
        raise V6BProductionError("OR-Tools is unavailable") from error
    if observed_python != runtime["python_version"]:
        raise V6BProductionError("Python version differs from registration")
    if ortools.__version__ != runtime["ortools_version"]:
        raise V6BProductionError("OR-Tools version differs from registration")
    if observed_platform != runtime["platform"]:
        raise V6BProductionError("runtime platform differs from registration")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible != "":
        raise V6BProductionError("a GPU is visible to the score-free solver process")
    locked = read_verified(
        Path(str(runtime["cp_sat_runtime_receipt_path"])),
        str(runtime["cp_sat_runtime_receipt_sha256"]),
        "CP-SAT runtime receipt",
    )
    return {
        "python_executable": str(registered_executable),
        "python_version": observed_python,
        "ortools_version": ortools.__version__,
        "platform": observed_platform,
        "cuda_visible_devices": cuda_visible,
        "accelerator_visible_to_solver": False,
        "cp_sat_runtime_receipt_sha256": _sha256_bytes(locked),
        "cp_sat_runtime_receipt_byte_size": len(locked),
    }


def _registered_input_bytes(
    registration: Mapping[str, object],
) -> tuple[bytes, bytes, dict[str, bytes], dict[str, object]]:
    raw = registration["raw_inputs"]
    assert isinstance(raw, Mapping)
    consumed: dict[str, bytes] = {}
    receipts: dict[str, object] = {}
    for label, reference in raw.items():
        assert isinstance(reference, Mapping)
        payload = read_verified(
            Path(str(reference["path"])),
            str(reference["sha256"]),
            f"raw input {label}",
        )
        consumed[label] = payload
        receipts[label] = {
            "path": str(reference["path"]),
            "sha256": _sha256_bytes(payload),
            "byte_size": len(payload),
        }
    edges = {
        view: consumed[f"edge_ledger_{view}"]
        for view in v6_cp_sat_formulation.VIEWS
    }
    return (
        consumed["train_authority"],
        consumed["canonical_manifest"],
        edges,
        receipts,
    )


def _final_markdown(receipt: Mapping[str, object]) -> bytes:
    result = receipt.get("decomposition_result")
    result_map = result if isinstance(result, Mapping) else {}
    lines = [
        "# V6-B production feasibility result",
        "",
        f"Terminal status: `{receipt['terminal_status']}`",
        f"Decomposition status: `{result_map.get('terminal_status', 'unavailable')}`",
        f"Candidate present: `{str(bool(result_map.get('candidate_present'))).lower()}`",
        f"All 66 exact gates passed: `{str(bool(result_map.get('all_66_exact_gates_passed'))).lower()}`",
        f"Journal events: {receipt['journal_event_count']}",
        "",
        "This receipt does not authorize Gate 0 or training. Even an exact CP-SAT",
        "candidate must pass the separately locked direct-HiGHS fixed-assignment",
        "linear projection audit before the split can enter the freeze workflow.",
        "",
    ]
    return "\n".join(lines).encode("ascii")


def publish_final(
    output_directory: Path,
    receipt: dict[str, object],
) -> tuple[Path, Path]:
    """Write one final receipt and report, then make completed evidence read-only."""
    target = Path(os.path.abspath(os.path.expanduser(str(output_directory))))
    if not target.is_dir() or target.is_symlink():
        raise V6BProductionError("attempt directory is absent or symbolic")
    report = _final_markdown(receipt)
    receipt["report_sha256"] = _sha256_bytes(report)
    receipt["receipt_sha256"] = _sha256_payload(receipt)
    receipt_path = target / "final_receipt.json"
    report_path = target / "final_report.md"
    if receipt_path.exists() or report_path.exists():
        raise V6BProductionError("final evidence already exists")
    _write_exclusive(receipt_path, canonical_json_bytes(receipt))
    _write_exclusive(report_path, report)
    for path in target.iterdir():
        if path.is_file() and not path.is_symlink():
            os.chmod(path, 0o400)
    os.chmod(target, 0o500)
    return receipt_path, report_path


def execute_registered_feasibility(
    *,
    registration: Mapping[str, object],
    registration_sha256: str,
    controller_path: Path,
    controller_sha256: str,
    output_directory: Path,
) -> dict[str, object]:
    """Authenticate, reconstruct, solve, and publish one V6-B production attempt."""
    checked = validate_registration(registration)
    if _sha256_payload(checked) != registration_sha256:
        raise V6BProductionError("registration payload digest differs")
    source = checked["source"]
    assert isinstance(source, Mapping)
    files = source["files"]
    assert isinstance(files, Mapping)
    controller_reference = files["production_controller"]
    assert isinstance(controller_reference, Mapping)
    expected_controller = _digest(
        controller_reference["sha256"], "registered production controller"
    )
    registered_controller_path = Path(str(source["source_root"])) / str(
        controller_reference["path"]
    )
    if registered_controller_path.resolve() != controller_path.resolve():
        raise V6BProductionError("controller path differs from registration")
    if controller_sha256 != expected_controller:
        raise V6BProductionError("controller command-line hash differs")
    controller_bytes = read_verified(
        controller_path,
        expected_controller,
        "production controller",
    )
    journal = create_attempt_directory(
        output_directory,
        registration_sha256,
        checked,
    )
    started_ns = time.monotonic_ns()
    try:
        runtime_receipt = _runtime_receipt(checked)
        source_receipts = _source_receipts(checked)
        parent_receipts = _parent_evidence_receipts(checked)
        journal.append(
            "RUNTIME_SOURCE_AND_PARENT_AUTHENTICATED",
            {
                "runtime": runtime_receipt,
                "source": source_receipts,
                "parent_evidence": parent_receipts,
                "controller_sha256": _sha256_bytes(controller_bytes),
            },
        )
        authority_reference = checked["authority"]
        assert isinstance(authority_reference, Mapping)
        published_authority = read_verified(
            Path(str(authority_reference["path"])),
            str(authority_reference["sha256"]),
            "published authority",
        )
        scientific_bytes, publication_receipt = extract_scientific_authority(
            published_authority
        )
        if (
            publication_receipt["scientific_payload_sha256"]
            != authority_reference["scientific_payload_sha256"]
        ):
            raise V6BProductionError("registered scientific authority hash differs")
        normalized_primary_bytes, primary_normalization = (
            normalize_scientific_pair_order(scientific_bytes)
        )
        authority_bytes, manifest_bytes, edge_bytes, raw_receipts = (
            _registered_input_bytes(checked)
        )
        primary_payload = strict_json(normalized_primary_bytes, "normalized authority")
        implementation_hashes = primary_payload.get("implementation_source_sha256")
        if not isinstance(implementation_hashes, Mapping):
            raise V6BProductionError("authority implementation anchors are absent")
        independent_payload = (
            score_free_independent_auditor.reconstruct_score_free_authority(
                authority_bytes=authority_bytes,
                manifest_bytes=manifest_bytes,
                edge_ledger_bytes=edge_bytes,
                expected_primary_implementation_sha256=implementation_hashes,
                expected_counts=EXPECTED_COUNTS,
            )
        )
        independent_bytes, independent_normalization = (
            normalize_scientific_pair_order(canonical_json_bytes(independent_payload))
        )
        if independent_bytes != normalized_primary_bytes:
            raise V6BProductionError(
                "primary and independent normalized authority payloads differ"
            )
        normalized_independent = strict_json(
            independent_bytes, "independent normalized authority"
        )
        journal.append(
            "RAW_AUTHORITY_RECONSTRUCTED",
            {
                "published_authority_sha256": _sha256_bytes(published_authority),
                "publication": publication_receipt,
                "primary_normalization": primary_normalization,
                "independent_normalization": independent_normalization,
                "normalized_authority_sha256": _sha256_bytes(normalized_primary_bytes),
                "primary_independent_byte_agreement": True,
                "raw_inputs": raw_receipts,
            },
        )
        problem = v6_cp_sat_formulation.direct_problem_from_authority_payload(
            normalized_primary_bytes,
            edge_bytes,
            expected_authority_payload_sha256=_sha256_bytes(normalized_primary_bytes),
            expected_counts=EXPECTED_COUNTS,
        )
        direct = v6_cp_sat_formulation.build_independent_cp_sat_formulation(problem)
        policy_sha256 = str(parent_receipts["v6_b_policy_sha256"])
        direct_source_sha256 = str(
            files["v6_cp_sat_formulation"]["sha256"]
        )
        exact_source_sha256 = str(
            files["v6_exact_separation_oracle"]["sha256"]
        )
        ir = v6_joint_ir.build_v6_joint_ir_from_direct_cp_sat(
            direct,
            solver_policy_sha256=policy_sha256,
            direct_formulation_source_sha256=direct_source_sha256,
            exact_oracle_source_sha256=exact_source_sha256,
        )
        partition = v6_b_decomposition.partition_joint_ir(ir)
        partition_payload = partition.canonical_payload()
        journal.append(
            "FULL_IR_PARTITIONED",
            {
                "direct_formulation_sha256": direct.formulation_sha256,
                "ir_sha256": ir.ir_sha256,
                "partition_sha256": partition_payload["partition_sha256"],
                "variable_count": len(ir.variable_keys),
                "row_count": len(ir.linear_constraints),
                "gate_count": len(ir.separation_gates),
            },
        )
        oracle = v6_exact_separation_oracle.ProductionExactSeparationOracle(
            ir=ir,
            authority_payload=primary_payload,
            independent_authority_payload=normalized_independent,
        )
        journal.exact_oracle = oracle
        limits = checked["limits"]
        assert isinstance(limits, Mapping)
        result = v6_b_decomposition.run_v6_b_exact_feasibility(
            partition,
            direct,
            oracle,
            maximum_master_iterations=int(limits["maximum_master_iterations"]),
            maximum_combined_candidates=int(limits["maximum_combined_candidates"]),
            maximum_time_seconds_per_solve=float(
                limits["maximum_time_seconds_per_solve"]
            ),
            event_callback=journal.solver_event,
        )
        terminal = (
            "CP_SAT_EXACT_CANDIDATE_PENDING_HIGHS"
            if result["terminal_status"] == "OPTIMAL_EXACT_ORACLE_PASS"
            else str(result["terminal_status"])
        )
        final: dict[str, object] = {
            "format_version": FINAL_FORMAT,
            "terminal_status": terminal,
            "registration_sha256": registration_sha256,
            "implementation_commit": source["implementation_commit"],
            "elapsed_ns": time.monotonic_ns() - started_ns,
            "runtime_receipt": runtime_receipt,
            "source_receipts": source_receipts,
            "parent_evidence_receipts": parent_receipts,
            "authority_publication": publication_receipt,
            "primary_normalization": primary_normalization,
            "independent_normalization": independent_normalization,
            "raw_input_receipts": raw_receipts,
            "direct_formulation_sha256": direct.formulation_sha256,
            "ir_sha256": ir.ir_sha256,
            "partition_sha256": partition_payload["partition_sha256"],
            "decomposition_result": result,
            "exact_oracle_evidence": oracle.evidence_history,
            "journal_event_count": journal.event_count,
            "journal_terminal_event_sha256": journal.previous_event_sha256,
            "direct_highs_fixed_assignment_audit_completed": False,
            "split_gate0_authorized": False,
            "training_authorized": False,
        }
        publish_final(output_directory, final)
        return final
    except BaseException as error:
        try:
            journal.append(
                "ATTEMPT_EXCEPTION",
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "error_sha256": _sha256_payload(
                        [type(error).__name__, str(error)]
                    ),
                },
            )
            failure: dict[str, object] = {
                "format_version": FINAL_FORMAT,
                "terminal_status": "PRODUCTION_CONTROLLER_EXCEPTION",
                "registration_sha256": registration_sha256,
                "implementation_commit": source["implementation_commit"],
                "elapsed_ns": time.monotonic_ns() - started_ns,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_sha256": _sha256_payload(
                    [type(error).__name__, str(error)]
                ),
                "decomposition_result": None,
                "journal_event_count": journal.event_count,
                "journal_terminal_event_sha256": journal.previous_event_sha256,
                "direct_highs_fixed_assignment_audit_completed": False,
                "split_gate0_authorized": False,
                "training_authorized": False,
            }
            publish_final(output_directory, failure)
        except BaseException:
            pass
        raise


def install_termination_handlers() -> None:
    """Convert scheduler termination into a receiptable Python exception."""

    def terminate(signum: int, _frame: object) -> None:
        raise V6BTerminationRequested(f"received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
