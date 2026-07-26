#!/usr/bin/env python3
"""Build the single V6-B full-variable, telemetry-only production row shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import stat
import time
from pathlib import Path
from typing import Mapping, Sequence

from interaction_first.v6_b_decomposition import (
    build_row_shard_telemetry,
    canonical_json_bytes,
    partition_joint_ir,
)
from interaction_first.v6_cp_sat_formulation import (
    VIEWS,
    build_independent_cp_sat_formulation,
    direct_problem_from_authority_payload,
)
from interaction_first.v6_joint_ir import build_v6_joint_ir_from_direct_cp_sat


FORMAT_VERSION = "v6_b_row_shard_production_receipt_v1"
EXPECTED_COUNTS = {
    "source_rows": 3229,
    "exact_pairs": 1830,
    "antibody_entities": 1383,
    "antigen_entities": 1172,
}


class V6BRowShardError(ValueError):
    """Raised when authenticated construction or create-only publication fails."""


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V6BRowShardError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_json(payload: bytes, label: str) -> dict[str, object]:
    if not payload or payload.startswith(b"\xef\xbb\xbf"):
        raise V6BRowShardError(f"{label} is empty or contains a BOM")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise V6BRowShardError(f"{label} must be ASCII JSON") from error

    def reject_duplicates(items: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise V6BRowShardError(f"{label} repeats key {key!r}")
            result[key] = value
        return result

    try:
        result = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                V6BRowShardError(f"{label} contains non-finite {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise V6BRowShardError(f"{label} is invalid JSON") from error
    if not isinstance(result, dict):
        raise V6BRowShardError(f"{label} must be a JSON object")
    return result


def extract_scientific_authority(
    published_bytes: bytes,
) -> tuple[bytes, dict[str, object]]:
    """Authenticate the publication envelope and return exact scientific JSON."""
    published = _strict_json(published_bytes, "published authority")
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
        raise V6BRowShardError("published authority envelope schema differs")
    publication_sha256 = _digest(
        published.pop("publication_sha256"),
        "authority publication",
    )
    if hashlib.sha256(canonical_json_bytes(published)).hexdigest() != publication_sha256:
        raise V6BRowShardError("authority publication digest differs")
    controller_sha256 = _digest(
        published.pop("publication_controller_sha256"),
        "authority publication controller",
    )
    verified = published.pop("verified_file_receipts")
    if not isinstance(verified, Mapping) or set(verified) != {
        "train_authority",
        "canonical_manifest",
        *(f"edge_ledger_{view}" for view in VIEWS),
    }:
        raise V6BRowShardError("authority verified-file receipts differ")
    for label, receipt in verified.items():
        if not isinstance(receipt, Mapping):
            raise V6BRowShardError(f"authority {label} file receipt is malformed")
        canonical_json_bytes(receipt)
    scientific_sha256 = _digest(
        published.get("scientific_payload_sha256"),
        "scientific authority payload",
    )
    scientific_without_digest = dict(published)
    scientific_without_digest.pop("scientific_payload_sha256")
    if (
        hashlib.sha256(canonical_json_bytes(scientific_without_digest)).hexdigest()
        != scientific_sha256
    ):
        raise V6BRowShardError("scientific authority digest differs")
    scientific_bytes = canonical_json_bytes(published)
    return scientific_bytes, {
        "publication_sha256": publication_sha256,
        "publication_controller_sha256": controller_sha256,
        "scientific_payload_sha256": scientific_sha256,
        "scientific_file_sha256": hashlib.sha256(scientific_bytes).hexdigest(),
        "verified_file_receipts_sha256": hashlib.sha256(
            canonical_json_bytes(verified)
        ).hexdigest(),
    }


def read_verified(path: Path, expected_sha256: str, label: str) -> bytes:
    """Read one regular file through a stable descriptor and verify its bytes."""
    expected = _digest(expected_sha256, label)
    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    current = Path(absolute.parts[0])
    for part in absolute.parts[1:]:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise V6BRowShardError(f"{label} path contains a symbolic link")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(absolute, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise V6BRowShardError(f"{label} is not a regular file")
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
        raise V6BRowShardError(f"{label} changed during authentication")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise V6BRowShardError(f"{label} SHA-256 differs")
    return payload


def available_memory_bytes() -> int:
    """Return the current host's available memory without allocating it."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) == 3 and parts[2] == "kB":
                    return int(parts[1]) * 1024
        raise V6BRowShardError("Linux MemAvailable is absent")
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    result = int(page_size) * int(available_pages)
    if result < 1:
        raise V6BRowShardError("available memory is not positive")
    return result


def peak_rss_bytes() -> int:
    """Normalize ru_maxrss to bytes on Linux and macOS."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _markdown(receipt_core: Mapping[str, object]) -> bytes:
    telemetry = receipt_core["telemetry"]
    partition = receipt_core["partition"]
    lines = [
        "# V6-B full-variable row-shard telemetry",
        "",
        f"Terminal status: `{receipt_core['terminal_status']}`",
        f"Full variables: {telemetry['full_variable_count']}",
        f"Full rows: {telemetry['full_row_count']}",
        f"Full nonzeros: {telemetry['full_nonzero_count']}",
        f"Selected telemetry rows: {telemetry['row_selection']['selected_row_count']}",
        f"Serialized full model bytes: {telemetry['serialized_full_model_bytes']}",
        f"Observed peak RSS bytes: {telemetry['observed_peak_rss_bytes']}",
        f"Projected peak bytes: {telemetry['projected_peak_bytes']}",
        f"Available memory bytes: {telemetry['available_memory_bytes']}",
        f"At least 25 percent headroom: {telemetry['resource_headroom_passed']}",
        f"Outer rows: {partition['master']['row_count']}",
        f"Inner rows: {partition['inner_row_counts']}",
        "",
        "This receipt sampled no entities or pairs, invoked no solver, and does not",
        "authorize feasibility, Gate 0, training, model selection, or evaluation.",
        "",
    ]
    return "\n".join(lines).encode("ascii")


def build_receipt(
    *,
    authority_bytes: bytes,
    edge_bytes: Mapping[str, bytes],
    policy_bytes: bytes,
    authority_sha256: str,
    policy_sha256: str,
    direct_source_sha256: str,
    exact_oracle_source_sha256: str,
    controller_source_sha256: str,
) -> tuple[dict[str, object], bytes]:
    """Build all full-authority variables and rows, then derive telemetry only."""
    policy = _strict_json(policy_bytes, "V6-B policy")
    if policy.get("format_version") != (
        "train_only_v6_b_logic_based_decomposition_policy_v1"
    ):
        raise V6BRowShardError("V6-B policy format differs")
    if policy.get("authorization", {}).get("contact_training_authorized") is not False:
        raise V6BRowShardError("V6-B policy unexpectedly authorizes training")
    expected = policy.get("authority_train")
    if not isinstance(expected, Mapping):
        raise V6BRowShardError("V6-B authority anchor is malformed")
    if expected.get("score_free_authority_file_sha256") != authority_sha256:
        raise V6BRowShardError("authority file hash differs from V6-B policy")
    if {
        "source_rows": expected.get("source_rows"),
        "exact_pairs": expected.get("exact_pairs"),
        "antibody_entities": expected.get("antibody_entities"),
        "antigen_entities": expected.get("antigen_entities"),
    } != EXPECTED_COUNTS:
        raise V6BRowShardError("V6-B authority inventory differs")

    scientific_authority, publication = extract_scientific_authority(
        authority_bytes
    )
    if (
        publication["scientific_payload_sha256"]
        != expected.get("score_free_authority_scientific_sha256")
    ):
        raise V6BRowShardError("scientific authority hash differs from V6-B policy")
    problem = direct_problem_from_authority_payload(
        scientific_authority,
        edge_bytes,
        expected_authority_payload_sha256=str(
            publication["scientific_file_sha256"]
        ),
        expected_counts=EXPECTED_COUNTS,
    )
    start_ns = time.monotonic_ns()
    direct = build_independent_cp_sat_formulation(problem)
    ir = build_v6_joint_ir_from_direct_cp_sat(
        direct,
        solver_policy_sha256=policy_sha256,
        direct_formulation_source_sha256=direct_source_sha256,
        exact_oracle_source_sha256=exact_oracle_source_sha256,
    )
    partition = partition_joint_ir(ir)
    model_bytes = direct.model.Proto().SerializeToString(deterministic=True)
    telemetry = build_row_shard_telemetry(
        ir,
        serialized_full_model_bytes=model_bytes,
        available_memory_bytes=available_memory_bytes(),
        observed_peak_rss_bytes=peak_rss_bytes(),
    )
    elapsed_ns = time.monotonic_ns() - start_ns
    partition_payload = partition.canonical_payload()
    partition_summary = {
        "partition_sha256": partition_payload["partition_sha256"],
        "source_ir_sha256": partition.source_ir_sha256,
        "master": {
            "variable_count": len(partition.master_variables),
            "row_count": len(partition.master_rows),
            "gate_count": len(partition.master_gates),
        },
        "inner_variable_counts": [
            len(variables) for _, variables in partition.inner_variables
        ],
        "inner_row_counts": [len(rows) for _, rows in partition.inner_rows],
        "inner_gate_counts": [len(gates) for _, gates in partition.inner_gates],
    }
    terminal = (
        "TELEMETRY_HEADROOM_PASSED"
        if telemetry["resource_headroom_passed"]
        else "TELEMETRY_HEADROOM_FAILED"
    )
    core: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "terminal_status": terminal,
        "construction_elapsed_ns": elapsed_ns,
        "input_sha256": {
            "published_authority_file": authority_sha256,
            "scientific_authority_file": publication["scientific_file_sha256"],
            "scientific_authority_payload": publication[
                "scientific_payload_sha256"
            ],
            "policy": policy_sha256,
            **{
                f"edge_ledger_{view}": hashlib.sha256(edge_bytes[view]).hexdigest()
                for view in VIEWS
            },
        },
        "authority_publication": publication,
        "source_sha256": {
            "controller": controller_source_sha256,
            "direct_cp_sat_formulation": direct_source_sha256,
            "exact_separation_oracle": exact_oracle_source_sha256,
        },
        "direct_formulation_sha256": direct.formulation_sha256,
        "partition": partition_summary,
        "telemetry": telemetry,
        "candidate_assignment_created": False,
        "solver_invoked": False,
        "feasibility_authorized": False,
        "split_gate0_authorized": False,
        "training_authorized": False,
    }
    report = _markdown(core)
    core["report_sha256"] = hashlib.sha256(report).hexdigest()
    core["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    return core, report


def publish_create_only(
    output_directory: Path,
    receipt: Mapping[str, object],
    report: bytes,
) -> tuple[Path, Path]:
    """Publish immutable JSON and Markdown into one absent final directory."""
    target = Path(os.path.abspath(os.path.expanduser(str(output_directory))))
    if target.exists() or target.is_symlink():
        raise V6BRowShardError("output directory already exists")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise V6BRowShardError("output parent is absent or symbolic")
    os.mkdir(target, 0o700)
    receipt_path = target / "receipt.json"
    report_path = target / "report.md"
    try:
        for path, payload in (
            (receipt_path, canonical_json_bytes(receipt)),
            (report_path, report),
        ):
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.chmod(path, 0o400)
        os.chmod(target, 0o500)
    except Exception:
        for path in (receipt_path, report_path):
            if path.exists():
                os.chmod(path, 0o600)
                path.unlink()
        os.chmod(target, 0o700)
        target.rmdir()
        raise
    return receipt_path, report_path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    for view in VIEWS:
        parser.add_argument(f"--edge-{view.replace('_', '-')}", type=Path, required=True)
        parser.add_argument(f"--edge-{view.replace('_', '-')}-sha256", required=True)
    parser.add_argument("--direct-source", type=Path, required=True)
    parser.add_argument("--direct-source-sha256", required=True)
    parser.add_argument("--exact-oracle-source", type=Path, required=True)
    parser.add_argument("--exact-oracle-source-sha256", required=True)
    parser.add_argument("--controller-sha256", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Authenticate, build, measure, and publish one non-authorizing receipt."""
    args = _arguments()
    controller_sha256 = _digest(args.controller_sha256, "controller")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != controller_sha256:
        raise V6BRowShardError("controller source hash differs")
    authority = read_verified(args.authority, args.authority_sha256, "authority")
    policy = read_verified(args.policy, args.policy_sha256, "policy")
    direct_source = read_verified(
        args.direct_source,
        args.direct_source_sha256,
        "direct formulation source",
    )
    exact_source = read_verified(
        args.exact_oracle_source,
        args.exact_oracle_source_sha256,
        "exact oracle source",
    )
    edge_bytes = {}
    for view in VIEWS:
        key = view.replace("_", "-")
        edge_bytes[view] = read_verified(
            getattr(args, f"edge_{view}"),
            getattr(args, f"edge_{view}_sha256"),
            f"{key} edge ledger",
        )
    receipt, report = build_receipt(
        authority_bytes=authority,
        edge_bytes=edge_bytes,
        policy_bytes=policy,
        authority_sha256=args.authority_sha256,
        policy_sha256=args.policy_sha256,
        direct_source_sha256=args.direct_source_sha256,
        exact_oracle_source_sha256=args.exact_oracle_source_sha256,
        controller_source_sha256=controller_sha256,
    )
    receipt_path, report_path = publish_create_only(
        args.output_directory,
        receipt,
        report,
    )
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "report": str(report_path),
                "terminal_status": receipt["terminal_status"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["terminal_status"] == "TELEMETRY_HEADROOM_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
