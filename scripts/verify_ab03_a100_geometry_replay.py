#!/usr/bin/env python3
"""Verify the sealed AB03 A100 geometry replay archive without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import PurePosixPath, Path
import statistics
import tarfile
from typing import Any, Mapping


FORMAT = "ab03_a100_geometry_replay_verification_v1"
RUN_PREFIX = "runs/exact_replay_20260729_09"
SEAL_PREFIX = "sealed/exact_replay_20260729_09"
EVIDENCE_PREFIX = "evidence/exact_replay_20260729_09"
AUTHORITY_PREFIX = "authority/ab03_case_sensitive_data_v1"
RESULT_MEMBER = f"{RUN_PREFIX}/result/SABDAB2_SELF_ATTENTION_ABLATION.json"
RUN_RECEIPT_MEMBER = f"{RUN_PREFIX}/result/RUN_RECEIPT.json"
SCIENTIFIC_RECEIPT_MEMBER = f"{RUN_PREFIX}/A100_REPLAY_RECEIPT.json"
SEAL_RECEIPT_MEMBER = f"{SEAL_PREFIX}/SEAL_RECEIPT.json"
ORIGINAL_MANIFEST_MEMBER = f"{SEAL_PREFIX}/ORIGINAL_RUN_SHA256SUMS.txt"
SEAL_MANIFEST_MEMBER = f"{SEAL_PREFIX}/SHA256SUMS.txt"
EVIDENCE_MANIFEST_MEMBER = f"{EVIDENCE_PREFIX}/EVIDENCE_SHA256SUMS.txt"
MULTISEED_MEMBER = f"{EVIDENCE_PREFIX}/SABDAB2_SELF_ATTENTION_MULTISEED.json"
DATA_MANIFEST_MEMBER = f"{AUTHORITY_PREFIX}/DATA_SHA256SUMS.txt"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    return value


def _safe_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or member.issym()
        or member.islnk()
        or member.isdev()
    ):
        raise ValueError(f"Unsafe archive member: {member.name}")
    if not member.isfile() and not member.isdir():
        raise ValueError(f"Unsupported archive member type: {member.name}")


def _archive_files(archive: tarfile.TarFile) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for member in archive.getmembers():
        _safe_member(member)
        if not member.isfile():
            continue
        if member.name in files:
            raise ValueError(f"Duplicate archive member: {member.name}")
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"Cannot read archive member: {member.name}")
        files[member.name] = handle.read()
    return files


def _json(files: Mapping[str, bytes], member: str) -> dict[str, Any]:
    if member not in files:
        raise ValueError(f"Missing required archive member: {member}")
    try:
        value = json.loads(files[member].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON member: {member}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON member is not an object: {member}")
    return value


def _manifest(payload: bytes, name: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise ValueError(f"{name} is not UTF-8") from error
    for line in lines:
        if not line:
            continue
        digest, separator, relative_path = line.partition("  ")
        if separator != "  " or not relative_path:
            raise ValueError(f"{name} contains a malformed line")
        _digest(digest, f"{name} digest")
        if relative_path in entries:
            raise ValueError(f"{name} contains a duplicate path")
        entries[relative_path] = digest
    if not entries:
        raise ValueError(f"{name} is empty")
    return entries


def _verify_manifest(
    files: Mapping[str, bytes],
    *,
    manifest_member: str,
    target_prefix: str,
) -> dict[str, Any]:
    entries = _manifest(files[manifest_member], manifest_member)
    checks: list[dict[str, Any]] = []
    for relative_path, expected in entries.items():
        normalized = relative_path[2:] if relative_path.startswith("./") else relative_path
        member = f"{target_prefix}/{normalized}"
        if member not in files:
            raise ValueError(f"Manifest target is absent: {member}")
        observed = _sha256_bytes(files[member])
        checks.append(
            {
                "member": member,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )
    if not all(check["passed"] for check in checks):
        raise ValueError(f"Manifest verification failed: {manifest_member}")
    return {
        "manifest_member": manifest_member,
        "manifest_sha256": _sha256_bytes(files[manifest_member]),
        "file_count": len(checks),
        "passed": True,
    }


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _multiseed_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("format_version") != "sabdab2_self_attention_geometry_multiseed_v1":
        raise ValueError("Unexpected multi-seed format")
    seeds = payload.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(seeds) != payload.get("seed_count")
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("Multi-seed membership is invalid")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Multi-seed metrics are absent")
    required_modes = {
        "dense_sequence",
        "topology_only",
        "distance",
        "distance_direction",
        "full_pose",
    }
    if set(metrics) != required_modes:
        raise ValueError("Multi-seed modes differ from the frozen contract")
    contract = payload.get("scientific_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Multi-seed scientific contract is absent")
    configuration = contract.get("configuration_without_seed")
    if not isinstance(configuration, Mapping):
        raise ValueError("Multi-seed configuration is absent")
    configured_modes = configuration.get("modes")
    mode_order = contract.get("mode_order")
    if (
        not isinstance(configured_modes, list)
        or set(configured_modes) != required_modes
        or not isinstance(mode_order, list)
        or mode_order != configured_modes
    ):
        raise ValueError("Multi-seed mode order differs from the frozen contract")
    runtime_contract = contract.get("runtime_contract")
    data_contract = contract.get("data")
    if not isinstance(runtime_contract, Mapping):
        raise ValueError("Multi-seed runtime contract is absent")
    if (
        not isinstance(data_contract, Mapping)
        or data_contract.get("native_release_test_cif_read") is not False
    ):
        raise ValueError("Multi-seed data contract crosses the native-test boundary")
    model_parameter_count = contract.get("model_parameter_count")
    if (
        isinstance(model_parameter_count, bool)
        or not isinstance(model_parameter_count, int)
        or model_parameter_count <= 0
    ):
        raise ValueError("Multi-seed model parameter count is invalid")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != len(seeds):
        raise ValueError("Multi-seed input receipts are invalid")
    for input_receipt in inputs:
        if not isinstance(input_receipt, Mapping):
            raise ValueError("Multi-seed input receipt is invalid")
        _digest(input_receipt.get("sha256"), "multi-seed input SHA-256")

    recomputation_checks = 0
    values_by_mode: dict[str, dict[str, list[float]]] = {}
    for mode in sorted(required_modes):
        raw_mode = metrics[mode]
        if not isinstance(raw_mode, Mapping):
            raise ValueError(f"Invalid multi-seed mode: {mode}")
        values_by_mode[mode] = {}
        for metric, raw_statistics in sorted(raw_mode.items()):
            if not isinstance(raw_statistics, Mapping):
                raise ValueError(f"Invalid statistics for {mode}/{metric}")
            raw_values = raw_statistics.get("values")
            if not isinstance(raw_values, list) or len(raw_values) != len(seeds):
                raise ValueError(f"Invalid values for {mode}/{metric}")
            values = [
                _finite_number(value, f"{mode}/{metric} value")
                for value in raw_values
            ]
            expected = {
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
            }
            for field, observed in expected.items():
                reported = _finite_number(
                    raw_statistics.get(field),
                    f"{mode}/{metric}/{field}",
                )
                if not math.isclose(reported, observed, rel_tol=0.0, abs_tol=1e-15):
                    raise ValueError(f"Multi-seed recomputation differs for {mode}/{metric}")
                recomputation_checks += 1
            values_by_mode[mode][metric] = values

    comparisons: dict[str, dict[str, Any]] = {}
    for candidate in ("distance", "distance_direction", "full_pose"):
        comparisons[candidate] = {}
        for metric in ("macro_auprc", "paratope_auprc", "epitope_auprc"):
            differences = [
                candidate_value - control_value
                for candidate_value, control_value in zip(
                    values_by_mode[candidate][metric],
                    values_by_mode["topology_only"][metric],
                )
            ]
            comparisons[candidate][metric] = {
                "differences": differences,
                "mean_difference": statistics.fmean(differences),
                "positive_seed_count": sum(value > 0.0 for value in differences),
                "seed_count": len(differences),
            }
    return {
        "seeds": seeds,
        "seed_count": len(seeds),
        "source_input_count": len(inputs),
        "model_parameter_count": model_parameter_count,
        "data_contract": dict(data_contract),
        "runtime_contract": dict(runtime_contract),
        "recomputation_check_count": recomputation_checks,
        "comparisons_vs_topology_only": comparisons,
    }


def verify_archive(path: Path, expected_archive_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("archive must be a regular non-symlink file")
    expected_archive_sha256 = _digest(
        expected_archive_sha256,
        "expected archive SHA-256",
    )
    archive_sha256 = _sha256_file(path)
    if archive_sha256 != expected_archive_sha256:
        raise ValueError("archive SHA-256 differs from the frozen digest")
    with tarfile.open(path, mode="r:gz") as archive:
        files = _archive_files(archive)

    seal_manifest = _verify_manifest(
        files,
        manifest_member=SEAL_MANIFEST_MEMBER,
        target_prefix=SEAL_PREFIX,
    )
    original_manifest = _verify_manifest(
        files,
        manifest_member=ORIGINAL_MANIFEST_MEMBER,
        target_prefix=RUN_PREFIX,
    )
    evidence_manifest = _verify_manifest(
        files,
        manifest_member=EVIDENCE_MANIFEST_MEMBER,
        target_prefix=EVIDENCE_PREFIX,
    )
    seal = _json(files, SEAL_RECEIPT_MEMBER)
    scientific = _json(files, SCIENTIFIC_RECEIPT_MEMBER)
    result = _json(files, RESULT_MEMBER)
    run_receipt = _json(files, RUN_RECEIPT_MEMBER)
    multiseed = _json(files, MULTISEED_MEMBER)

    if seal.get("status") != "passed" or seal.get("scientific_gate_count") != 48:
        raise ValueError("External A100 seal did not pass the frozen gates")
    if scientific.get("status") != "passed":
        raise ValueError("A100 scientific replay receipt did not pass")
    gates = scientific.get("gates")
    if (
        not isinstance(gates, list)
        or len(gates) != 48
        or any(
            not isinstance(gate, Mapping) or gate.get("passed") is not True
            for gate in gates
        )
    ):
        raise ValueError("A100 scientific gate count differs")
    if result.get("data", {}).get("native_release_test_cif_read") is not False:
        raise ValueError("A100 result does not retain the native-test boundary")
    if seal.get("replay_result_sha256") != _sha256_bytes(files[RESULT_MEMBER]):
        raise ValueError("Seal does not authenticate the replay result")
    if scientific.get("replay_result_sha256") != _sha256_bytes(files[RESULT_MEMBER]):
        raise ValueError("Scientific receipt does not authenticate the replay result")
    if run_receipt.get("result_sha256") != _sha256_bytes(files[RESULT_MEMBER]):
        raise ValueError("Run receipt does not authenticate the replay result")
    if seal.get("scientific_receipt_sha256") != _sha256_bytes(
        files[SCIENTIFIC_RECEIPT_MEMBER]
    ):
        raise ValueError("Seal does not authenticate the scientific receipt")
    if seal.get("original_run_manifest_sha256") != _sha256_bytes(
        files[ORIGINAL_MANIFEST_MEMBER]
    ):
        raise ValueError("Seal does not authenticate the original run manifest")

    data_manifest_sha256 = _sha256_bytes(files[DATA_MANIFEST_MEMBER])
    runtime_provenance = scientific.get("environment", {}).get(
        "runtime_provenance", {}
    )
    if runtime_provenance.get("REPLAY_DATA_MANIFEST_SHA256") != data_manifest_sha256:
        raise ValueError("Scientific receipt does not authenticate the data manifest")
    data_entries = _manifest(files[DATA_MANIFEST_MEMBER], DATA_MANIFEST_MEMBER)
    if (
        len(data_entries) != seal.get("post_training_data_file_count")
        or str(len(data_entries))
        != runtime_provenance.get("REPLAY_DATA_FILE_COUNT")
    ):
        raise ValueError("Data authority file count differs")

    multiseed_summary = _multiseed_summary(multiseed)
    if multiseed_summary["data_contract"] != result.get("data"):
        raise ValueError("Multi-seed and exact-replay data contracts differ")
    if (
        multiseed_summary["model_parameter_count"]
        != run_receipt.get("model_parameter_count")
    ):
        raise ValueError("Multi-seed and exact-replay model sizes differ")
    environment = scientific.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("A100 environment receipt is absent")
    return {
        "format_version": FORMAT,
        "status": "passed",
        "archive": {
            "sha256": archive_sha256,
            "bytes": path.stat().st_size,
            "regular_file_count": len(files),
        },
        "manifests": {
            "seal": seal_manifest,
            "original_run": original_manifest,
            "evidence": evidence_manifest,
            "data_authority": {
                "manifest_sha256": data_manifest_sha256,
                "file_count": len(data_entries),
                "passed": True,
            },
        },
        "runtime": {
            "gpu_name": environment.get("gpu_name"),
            "gpu_count": environment.get("gpu_count"),
            "torch": environment.get("torch"),
            "cuda": environment.get("cuda_runtime"),
            "parent_image": runtime_provenance.get("REPLAY_BASE_IMAGE_ID"),
            "runtime_image": runtime_provenance.get("REPLAY_RUNTIME_IMAGE_ID"),
            "shared_memory": runtime_provenance.get("REPLAY_SHM_SIZE"),
        },
        "scientific_replay": {
            "gate_count": seal.get("scientific_gate_count"),
            "result_sha256": seal.get("replay_result_sha256"),
            "scientific_receipt_sha256": seal.get("scientific_receipt_sha256"),
            "fixed_final_epoch": True,
            "native_test_read": False,
        },
        "multiseed": multiseed_summary,
        "claim_boundary": (
            "Development-only fixed-final-epoch A100 replay and multi-seed "
            "aggregation. This verification does not authorize a frozen-test, "
            "partner-specific, external-generalization, or SOTA claim."
        ),
    }


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise ValueError("output already exists")
    payload = verify_archive(arguments.archive, arguments.expected_archive_sha256)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical_json_bytes(payload))
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
