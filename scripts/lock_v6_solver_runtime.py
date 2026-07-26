#!/usr/bin/env python3
"""Publish a create-only receipt for one pinned V6-A solver runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import stat
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Sequence


FORMAT_VERSION = "v6_a_solver_runtime_lock_v1"
LOCKED_DIRECT_VERSIONS = {
    "highspy": "1.15.1",
    "ortools": "9.15.6755",
}
LOCKED_HIGHS_OPTIONS = {
    "threads": 1,
    "parallel": "off",
    "random_seed": 2026072701,
    "mip_rel_gap": 0.0,
    "mip_abs_gap": 0.0,
    "mip_feasibility_tolerance": 1e-7,
    "primal_feasibility_tolerance": 1e-7,
    "dual_feasibility_tolerance": 1e-7,
}


class RuntimeLockError(RuntimeError):
    """Raised when a wheel, environment, or solver probe disagrees."""


def _canonical_json_bytes(payload: object) -> bytes:
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


def _normalize_project_name(value: str) -> str:
    return "-".join(value.lower().replace("_", "-").split("-"))


def _read_regular_nofollow(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeLockError(f"wheel is not a regular file: {path.name}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        fingerprint = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if fingerprint(before) != fingerprint(after):
            raise RuntimeLockError(f"wheel changed during read: {path.name}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise RuntimeLockError(f"wheel size changed during read: {path.name}")
        return payload
    finally:
        os.close(descriptor)


def _wheel_metadata(payload: bytes, filename: str) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise RuntimeLockError(
                    f"wheel must contain one METADATA file: {filename}"
                )
            metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeLockError(f"invalid wheel archive: {filename}") from error
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version or not name.isascii() or not version.isascii():
        raise RuntimeLockError(f"wheel metadata lacks an ASCII name/version: {filename}")
    return {
        "project": _normalize_project_name(name),
        "version": version,
        "requires_dist": sorted(metadata.get_all("Requires-Dist", [])),
    }


def wheelhouse_receipt(wheelhouse: Path) -> list[dict[str, object]]:
    """Authenticate every wheel and require one installed matching distribution."""
    root = Path(os.path.abspath(os.path.expanduser(str(wheelhouse))))
    metadata = os.lstat(root)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeLockError("wheelhouse must be a physical directory")
    wheel_paths = sorted(root.glob("*.whl"), key=lambda path: path.name)
    if not wheel_paths:
        raise RuntimeLockError("wheelhouse contains no wheels")
    records = []
    projects = set()
    for path in wheel_paths:
        if not path.name.isascii() or path.name != path.name.strip():
            raise RuntimeLockError("wheel filename must be exact ASCII")
        payload = _read_regular_nofollow(path)
        parsed = _wheel_metadata(payload, path.name)
        project = str(parsed["project"])
        if project in projects:
            raise RuntimeLockError(f"wheelhouse repeats project {project}")
        projects.add(project)
        try:
            installed_version = importlib.metadata.version(project)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeLockError(f"wheel project is not installed: {project}") from error
        if installed_version != parsed["version"]:
            raise RuntimeLockError(f"installed version differs for {project}")
        records.append(
            {
                "filename": path.name,
                "byte_size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                **parsed,
                "installed_version": installed_version,
            }
        )
    for project, version in LOCKED_DIRECT_VERSIONS.items():
        if importlib.metadata.version(project) != version:
            raise RuntimeLockError(f"locked direct version differs for {project}")
        if _normalize_project_name(project) not in projects:
            raise RuntimeLockError(f"wheelhouse omits locked project {project}")
    return records


def solver_smoke_receipt() -> dict[str, object]:
    """Execute each native solver in a separate clean Python process."""
    cp_source = """
import json
import ortools
from ortools.sat.python import cp_model
model = cp_model.CpModel()
value = model.new_bool_var("locked_probe")
model.maximize(value)
solver = cp_model.CpSolver()
solver.parameters.num_search_workers = 1
solver.parameters.random_seed = 2026072701
solver.parameters.randomize_search = False
solver.parameters.relative_gap_limit = 0.0
solver.parameters.absolute_gap_limit = 0.0
status = solver.solve(model)
if status != cp_model.OPTIMAL or solver.value(value) != 1:
    raise SystemExit(7)
print(json.dumps({"ortools_version": ortools.__version__, "status": "OPTIMAL", "objective": "1", "num_search_workers": 1, "random_seed": 2026072701, "relative_gap_limit": "0", "absolute_gap_limit": "0"}, sort_keys=True))
"""
    highs_source = f"""
import json
import highspy
options = {LOCKED_HIGHS_OPTIONS!r}
solver = highspy.Highs()
for name, value in options.items():
    if solver.setOptionValue(name, value) != highspy.HighsStatus.kOk:
        raise SystemExit(8)
if solver.setOptionValue("output_flag", False) != highspy.HighsStatus.kOk:
    raise SystemExit(9)
if solver.addVar(0.0, 1.0) != highspy.HighsStatus.kOk:
    raise SystemExit(10)
if solver.changeColIntegrality(0, highspy.HighsVarType.kInteger) != highspy.HighsStatus.kOk:
    raise SystemExit(11)
if solver.changeColCost(0, 1.0) != highspy.HighsStatus.kOk:
    raise SystemExit(12)
solver.setMaximize()
if solver.run() != highspy.HighsStatus.kOk:
    raise SystemExit(13)
info = solver.getInfo()
if solver.getModelStatus() != highspy.HighsModelStatus.kOptimal or info.mip_gap != 0.0 or solver.getSolution().col_value != [1.0]:
    raise SystemExit(14)
print(json.dumps({{"highs_version": solver.version(), "status": "kOptimal", "objective": "1.0", "mip_gap": "0.0", "locked_options": options}}, sort_keys=True))
"""

    def probe(source: str, label: str) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, "-I", "-c", source],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 or result.stderr:
            raise RuntimeLockError(
                f"{label} isolated smoke probe failed with code {result.returncode}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeLockError(f"{label} smoke receipt is malformed") from error
        if not isinstance(payload, dict):
            raise RuntimeLockError(f"{label} smoke receipt is not an object")
        return payload

    return {
        "native_library_process_isolation": True,
        "cp_sat": probe(cp_source, "CP-SAT"),
        "highs": probe(highs_source, "HiGHS"),
    }


def build_receipt(wheelhouse: Path) -> dict[str, object]:
    """Build the canonical runtime receipt without granting model authority."""
    os_release = Path("/etc/os-release")
    os_release_bytes = os_release.read_bytes() if os_release.is_file() else b""
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "abi_flags": sys.abiflags,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
            "os_release_sha256": hashlib.sha256(os_release_bytes).hexdigest(),
        },
        "wheels": wheelhouse_receipt(wheelhouse),
        "solver_smoke": solver_smoke_receipt(),
        "implementation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "complete_score_free_feasible_region_certified": False,
        "training_authorized": False,
        "model_selection_authorized": False,
        "outer_evaluation_authorized": False,
    }
    payload["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(os.path.abspath(os.path.expanduser(str(args.output))))
    if not output.parent.is_dir() or output.exists() or output.is_symlink():
        raise FileExistsError("runtime receipt output must be new in an existing directory")
    receipt = build_receipt(args.wheelhouse)
    _write_exclusive(output, _canonical_json_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
