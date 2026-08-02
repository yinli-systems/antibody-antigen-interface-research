#!/usr/bin/env python3
"""Create and verify an immutable Python/PyTorch runtime manifest.

The manifest has two complementary integrity layers:

* ``content_sha256`` authenticates every byte-significant JSON field except the
  digest object itself.
* ``runtime_state_sha256`` identifies the reproducible software and stable ABI
  state while excluding collection time, host name, exact kernel build, and
  processor model.

The command performs no network access and does not inspect model or dataset
artifacts. Run it with the interpreter belonging to the environment being
recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import secrets
import site
import stat
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "frontier.runtime-manifest.v3"
GENERATOR_VERSION = "1.2.0"
HASH_ALGORITHM = "sha256"
CANONICALIZATION = "UTF-8 JSON; sort_keys=true; separators=(',', ':')"
DEFAULT_PIP_CHECK_TIMEOUT_SECONDS = 120
READ_ONLY_MODE = 0o444
_UNSET = object()

CLAIM_BOUNDARY = {
    "attests": (
        "The recorded software metadata, stable platform ABI, and file hashes "
        "describe the runtime state visible to this interpreter at collection "
        "time."
    ),
    "does_not_attest": (
        "This manifest does not establish hardware equivalence, deterministic "
        "training, scientific validity, benchmark comparability, biological "
        "performance, or state-of-the-art performance."
    ),
    "data_access": (
        "The generator does not read model artifacts, training data, "
        "validation data, held-out data, or test data."
    ),
}


class ManifestError(ValueError):
    """Report a malformed, inconsistent, or unverifiable manifest."""


@dataclass(frozen=True)
class PytorchProbe:
    """Hold deterministic PyTorch metadata and candidate shared libraries."""

    metadata: Mapping[str, Any]
    library_candidates: tuple[tuple[str, Path], ...] = ()
    collection_context: Mapping[str, Any] = field(default_factory=dict)


_PYTORCH_RUNTIME_CONFIGURATION_PREFIXES = (
    "  - CPU capability usage:",
    "  - CUDA Runtime ",
    "  - NVCC architecture flags:",
    "  - CuDNN ",
    "  - Magma ",
)


def _split_pytorch_build_configuration(
    configuration: str | None,
) -> tuple[str | None, list[str]]:
    """Separate stable build settings from node-dependent capability telemetry."""
    if configuration is None:
        return None, []
    stable_lines: list[str] = []
    runtime_lines: list[str] = []
    for line in configuration.splitlines():
        if line.startswith(_PYTORCH_RUNTIME_CONFIGURATION_PREFIXES):
            runtime_lines.append(line)
        else:
            stable_lines.append(line)
    stable = "\n".join(stable_lines)
    if configuration.endswith("\n"):
        stable += "\n"
    return stable, runtime_lines


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize one finite JSON value using the locked canonical form."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ManifestError("Value cannot be represented as strict JSON") from error


def _sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    """Hash a regular file through an open descriptor to reduce race exposure."""
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise ManifestError(f"Artifact is not a regular file: {path}")
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                digest.update(block)
    except OSError as error:
        raise ManifestError(f"Cannot hash artifact: {path}") from error
    return digest.hexdigest(), size


def _lexical_absolute(path: Path | str) -> Path:
    """Return an absolute path while preserving its final symbolic-link name."""
    return Path(os.path.abspath(os.fspath(path)))


def _capture_file(path: Path | str, *, role: str) -> dict[str, Any]:
    """Capture path, link, mode, byte-size, and content-hash evidence."""
    logical_path = _lexical_absolute(path)
    try:
        link_status = logical_path.lstat()
    except OSError as error:
        raise ManifestError(f"Required {role} is missing: {logical_path}") from error
    resolved = logical_path.resolve(strict=True)
    digest, size = _sha256_file(logical_path)
    return {
        "role": role,
        "path": str(logical_path),
        "resolved_path": str(resolved),
        "is_symlink": stat.S_ISLNK(link_status.st_mode),
        "symlink_target": (
            os.readlink(logical_path) if stat.S_ISLNK(link_status.st_mode) else None
        ),
        "mode": f"{stat.S_IMODE(link_status.st_mode):04o}",
        "size_bytes": size,
        "sha256": digest,
    }


def _parse_pyvenv_cfg(path: Path) -> dict[str, str]:
    """Parse the simple key-value grammar used by ``pyvenv.cfg``."""
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ManifestError(f"Cannot read pyvenv.cfg: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        normalized_key = key.strip().lower().replace("-", "_")
        if not separator or not normalized_key:
            raise ManifestError(f"Malformed pyvenv.cfg line {line_number}: {path}")
        if normalized_key in result:
            raise ManifestError(f"Duplicate pyvenv.cfg key {normalized_key!r}: {path}")
        result[normalized_key] = value.strip()
    return result


def _parse_boolean(value: str | None) -> bool | None:
    """Parse a configuration boolean without silently guessing."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ManifestError(f"Invalid boolean value in pyvenv.cfg: {value!r}")


def _normalized_distribution_name(name: str) -> str:
    """Normalize a distribution name according to PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is equal to or below a resolved root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _distribution_metadata_path(distribution: Any) -> Path | None:
    """Return the concrete dist-info or egg-info path when exposed."""
    raw_path = getattr(distribution, "_path", None)
    if raw_path is None:
        return None
    path = Path(raw_path)
    return path if path.exists() else None


def _capture_optional_metadata_file(
    metadata_path: Path | None,
    filename: str,
    role: str,
) -> dict[str, Any] | None:
    """Hash an installed-distribution evidence file when it exists."""
    if metadata_path is None:
        return None
    candidate = metadata_path / filename
    return _capture_file(candidate, role=role) if candidate.is_file() else None


def _collect_distributions(
    distributions: Iterable[Any],
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Collect a deterministically sorted installed-distribution inventory."""
    collected: list[dict[str, Any]] = []
    locations: list[Path] = []
    for distribution in distributions:
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ManifestError("Installed distribution is missing its Name metadata")
        version = str(distribution.version)
        location = Path(distribution.locate_file("")).resolve()
        metadata_path = _distribution_metadata_path(distribution)
        locations.append(location)
        normalized_name = _normalized_distribution_name(raw_name)
        evidence = {
            "record": _capture_optional_metadata_file(
                metadata_path,
                "RECORD",
                f"distribution:{normalized_name}:RECORD",
            ),
            "direct_url": _capture_optional_metadata_file(
                metadata_path,
                "direct_url.json",
                f"distribution:{normalized_name}:direct_url.json",
            ),
        }
        collected.append(
            {
                "name": raw_name,
                "normalized_name": normalized_name,
                "version": version,
                "location": str(location),
                "metadata_directory": (
                    str(metadata_path.resolve()) if metadata_path is not None else None
                ),
                "evidence": evidence,
            }
        )
    collected.sort(
        key=lambda item: (
            item["normalized_name"],
            item["version"],
            item["location"],
            item["metadata_directory"] or "",
        )
    )
    return collected, locations


def _run_pip_check(
    executable: str,
    *,
    timeout_seconds: int = DEFAULT_PIP_CHECK_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run ``pip check`` without user-site or locale-dependent output."""
    command = [executable, "-m", "pip", "check"]
    command_environment = os.environ.copy()
    command_environment["PYTHONNOUSERSITE"] = "1"
    command_environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            env=command_environment,
        )
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        return {
            "command": command,
            "timeout_seconds": timeout_seconds,
            "exit_code": completed.returncode,
            "ok": completed.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "command": command,
            "timeout_seconds": timeout_seconds,
            "exit_code": None,
            "ok": False,
            "stdout": (error.stdout or b"").decode("utf-8", errors="replace").strip(),
            "stderr": (error.stderr or b"").decode("utf-8", errors="replace").strip(),
            "failure": "timeout",
        }
    except OSError as error:
        return {
            "command": command,
            "timeout_seconds": timeout_seconds,
            "exit_code": None,
            "ok": False,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
            "failure": "execution_error",
        }


def _version_sequence(value: Any) -> list[int] | None:
    """Convert a backend version tuple to strict JSON integers."""
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            return [int(component) for component in value]
        except (TypeError, ValueError):
            return None
    return None


def _probe_pytorch() -> PytorchProbe:
    """Read PyTorch build metadata without requiring CUDA device access."""
    try:
        import torch
    except Exception as error:  # noqa: BLE001 - framework import failures are evidence.
        return PytorchProbe(
            metadata={
                "available": False,
                "import_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )

    module_artifact = _capture_file(torch.__file__, role="pytorch_module")
    candidates: list[tuple[str, Path]] = []
    torch_root = Path(torch.__file__).resolve().parent
    torch_library_root = torch_root / "lib"
    if torch_library_root.is_dir():
        for candidate in sorted(torch_library_root.glob("*")):
            if candidate.is_file() and (
                ".so" in candidate.name
                or candidate.suffix.lower() in {".dylib", ".dll"}
            ):
                candidates.append(("pytorch_shared_library", candidate))

    extension_artifact: dict[str, Any] | None = None
    extension_path = getattr(getattr(torch, "_C", None), "__file__", None)
    if extension_path:
        extension_artifact = _capture_file(
            extension_path, role="pytorch_native_extension"
        )
        candidates.append(("pytorch_native_extension", Path(extension_path)))

    def safe_call(function: Callable[[], Any]) -> Any:
        try:
            return function()
        except Exception:  # noqa: BLE001 - optional framework probes must be bounded.
            return None

    nccl_raw = safe_call(lambda: torch.cuda.nccl.version())
    cudnn_raw = safe_call(lambda: torch.backends.cudnn.version())
    cuda_compiled_raw = safe_call(lambda: torch._C._cuda_getCompiledVersion())
    configuration = safe_call(lambda: torch.__config__.show())
    stable_configuration, runtime_configuration = _split_pytorch_build_configuration(
        str(configuration) if configuration is not None else None
    )
    metadata = {
        "available": True,
        "version": str(torch.__version__),
        "git_version": getattr(torch.version, "git_version", None),
        "debug_build": bool(getattr(torch.version, "debug", False)),
        "module": module_artifact,
        "native_extension": extension_artifact,
        "cuda": {
            "build_version": getattr(torch.version, "cuda", None),
            "compiled_version_integer": (
                int(cuda_compiled_raw) if cuda_compiled_raw is not None else None
            ),
        },
        "cudnn": {"version_integer": int(cudnn_raw) if cudnn_raw is not None else None},
        "nccl": {
            "version_components": _version_sequence(nccl_raw),
        },
        "cxx11_abi": safe_call(lambda: bool(torch.compiled_with_cxx11_abi())),
        "build_configuration": stable_configuration,
    }
    return PytorchProbe(
        metadata=metadata,
        library_candidates=tuple(candidates),
        collection_context={
            "runtime_build_capability_lines": runtime_configuration,
        },
    )


def _python_shared_library_candidates() -> list[tuple[str, Path]]:
    """Locate the shared Python library exposed by sysconfig."""
    candidates: list[tuple[str, Path]] = []
    library_name = sysconfig.get_config_var("LDLIBRARY")
    for directory_key in ("LIBDIR", "LIBPL"):
        directory = sysconfig.get_config_var(directory_key)
        if directory and library_name:
            candidate = Path(directory) / str(library_name)
            if candidate.is_file():
                candidates.append(("python_shared_library", candidate))
    return candidates


def _cuda_library_candidates(
    distribution_locations: Iterable[Path],
) -> list[tuple[str, Path]]:
    """Locate key CUDA, cuDNN, and NCCL shared libraries in site-packages."""
    patterns = {
        "cuda_runtime_shared_library": "nvidia/cuda_runtime/lib/libcudart.so*",
        "cudnn_shared_library": "nvidia/cudnn/lib/libcudnn*.so*",
        "nccl_shared_library": "nvidia/nccl/lib/libnccl.so*",
    }
    candidates: list[tuple[str, Path]] = []
    for location in sorted({path.resolve() for path in distribution_locations}):
        for role, pattern in patterns.items():
            for candidate in sorted(location.glob(pattern)):
                if candidate.is_file():
                    candidates.append((role, candidate))
    return candidates


def _collect_shared_libraries(
    candidates: Iterable[tuple[str, Path]],
) -> list[dict[str, Any]]:
    """Hash a deduplicated inventory of relevant shared libraries."""
    unique: dict[str, tuple[str, Path]] = {}
    for role, path in candidates:
        logical = _lexical_absolute(path)
        key = str(logical)
        previous = unique.get(key)
        if previous is None or role < previous[0]:
            unique[key] = (role, logical)
    captured = [
        _capture_file(path, role=role)
        for role, path in sorted(unique.values(), key=lambda item: str(item[1]))
    ]
    captured.sort(key=lambda item: (item["role"], item["path"]))
    return captured


def _platform_observation() -> dict[str, Any]:
    """Capture stable ABI fields plus node-specific platform telemetry."""
    libc_name, libc_version = platform.libc_ver()
    architecture_bits, architecture_linkage = platform.architecture()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture": {
            "bits": architecture_bits,
            "linkage": architecture_linkage,
        },
        "libc": {"name": libc_name or None, "version": libc_version or None},
    }


def _stable_platform_abi(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Keep binary-compatibility fields while excluding node-varying telemetry."""
    required = {"system", "machine", "architecture", "libc"}
    if not required.issubset(observation):
        raise ManifestError("Platform observation lacks stable ABI fields")
    return {
        "system": observation["system"],
        "machine": observation["machine"],
        "architecture": observation["architecture"],
        "libc": observation["libc"],
        "kernel_release_bound": False,
        "processor_model_bound": False,
    }


def _python_state() -> dict[str, Any]:
    """Capture interpreter identity and ABI metadata."""
    executable = _capture_file(sys.executable, role="python_interpreter")
    return {
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:5]),
        "implementation": platform.python_implementation(),
        "implementation_version": platform.python_version(),
        "compiler": platform.python_compiler(),
        "build": list(platform.python_build()),
        "cache_tag": getattr(sys.implementation, "cache_tag", None),
        "hexversion": sys.hexversion,
        "abi": {
            "soabi": sysconfig.get_config_var("SOABI"),
            "abiflags": getattr(sys, "abiflags", ""),
            "platform": sysconfig.get_platform(),
        },
        "executable": executable,
    }


def _environment_state(
    environment_root: Path,
    distributions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe environment isolation and installed-package scope."""
    root = environment_root.resolve(strict=True)
    pyvenv_path = root / "pyvenv.cfg"
    pyvenv_artifact = (
        _capture_file(pyvenv_path, role="pyvenv_cfg") if pyvenv_path.is_file() else None
    )
    pyvenv_values = _parse_pyvenv_cfg(pyvenv_path) if pyvenv_artifact else {}
    include_system = _parse_boolean(pyvenv_values.get("include_system_site_packages"))
    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    outside_locations = sorted(
        {
            str(Path(item["location"]).resolve())
            for item in distributions
            if not _is_within(Path(item["location"]), root)
        }
    )
    executable_lexically_within_root = _is_within(
        _lexical_absolute(sys.executable).parent, root
    )
    prefix_matches = prefix == root
    isolated = (
        prefix_matches
        and pyvenv_artifact is not None
        and include_system is False
        and executable_lexically_within_root
        and not outside_locations
    )
    return {
        "root": str(root),
        "sys_prefix": str(prefix),
        "sys_base_prefix": str(base_prefix),
        "prefix_matches_root": prefix_matches,
        "executable_lexically_within_root": executable_lexically_within_root,
        "pyvenv_cfg": pyvenv_artifact,
        "pyvenv_values": dict(sorted(pyvenv_values.items())),
        "include_system_site_packages": include_system,
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        "user_site_enabled": bool(site.ENABLE_USER_SITE),
        "outside_distribution_locations": outside_locations,
        "scope_classification": (
            "isolated_virtual_environment"
            if isolated
            else "non_isolated_or_mismatched_environment"
        ),
    }


def collect_runtime_state(
    environment_root: Path | str,
    *,
    distribution_provider: Callable[[], Iterable[Any]] | None = None,
    pip_check_result: Mapping[str, Any] | object = _UNSET,
    pytorch_probe: PytorchProbe | object = _UNSET,
    extra_library_candidates: Iterable[tuple[str, Path]] | None = None,
    platform_probe: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Collect the deterministic portion of one runtime manifest.

    Injection points exist for focused tests and offline audits. Production
    callers should use the defaults.
    """
    root = Path(environment_root)
    if not root.is_absolute():
        raise ManifestError("environment_root must be an explicit absolute path")
    root = root.resolve(strict=True)
    provider = distribution_provider or importlib.metadata.distributions
    distributions, distribution_locations = _collect_distributions(provider())
    observed_pip_check = (
        _run_pip_check(sys.executable)
        if pip_check_result is _UNSET
        else dict(pip_check_result)
    )
    observed_pytorch = _probe_pytorch() if pytorch_probe is _UNSET else pytorch_probe
    if not isinstance(observed_pytorch, PytorchProbe):
        raise TypeError("pytorch_probe must be a PytorchProbe")
    candidates = _python_shared_library_candidates()
    candidates.extend(observed_pytorch.library_candidates)
    candidates.extend(_cuda_library_candidates(distribution_locations))
    if extra_library_candidates is not None:
        candidates.extend(extra_library_candidates)
    state = {
        "environment": _environment_state(root, distributions),
        "python": _python_state(),
        "platform_abi": _stable_platform_abi(
            dict((platform_probe or _platform_observation)())
        ),
        "frameworks": {"pytorch": dict(observed_pytorch.metadata)},
        "package_health": {"pip_check": observed_pip_check},
        "installed_distributions": distributions,
        "shared_libraries": _collect_shared_libraries(candidates),
    }
    _canonical_json_bytes(state)
    return state


def _utc_now() -> str:
    """Return a second-resolution RFC 3339 UTC timestamp."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_manifest(
    environment_root: Path | str,
    *,
    collected_at_utc: str | None = None,
    collection_host: str | None = None,
    **collection_options: Any,
) -> dict[str, Any]:
    """Build an integrity-bound manifest without writing it."""
    observed_pytorch = collection_options.get("pytorch_probe", _UNSET)
    if observed_pytorch is _UNSET:
        observed_pytorch = _probe_pytorch()
        collection_options["pytorch_probe"] = observed_pytorch
    if not isinstance(observed_pytorch, PytorchProbe):
        raise TypeError("pytorch_probe must be a PytorchProbe")
    runtime_state = collect_runtime_state(environment_root, **collection_options)
    platform_probe = collection_options.get("platform_probe")
    observed_platform = dict((platform_probe or _platform_observation)())
    generator_path = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "name": Path(__file__).name,
            "version": GENERATOR_VERSION,
            "source": _capture_file(generator_path, role="manifest_generator"),
        },
        "collection_context": {
            "collected_at_utc": collected_at_utc or _utc_now(),
            "frameworks": {
                "pytorch": dict(observed_pytorch.collection_context),
            },
            "host": collection_host if collection_host is not None else platform.node(),
            "observed_platform": observed_platform,
        },
        "runtime_state": runtime_state,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["integrity"] = {
        "algorithm": HASH_ALGORITHM,
        "canonicalization": CANONICALIZATION,
        "runtime_state_sha256": _sha256_bytes(_canonical_json_bytes(runtime_state)),
        "content_sha256": _sha256_bytes(_canonical_json_bytes(payload)),
    }
    return payload


def _strict_json_bytes(payload_bytes: bytes, *, source: str) -> dict[str, Any]:
    """Parse finite UTF-8 JSON bytes while rejecting duplicate object keys."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ManifestError(f"Duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ManifestError(f"Non-finite JSON value: {value}")

    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except ManifestError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"Cannot read strict JSON: {source}") from error

    def require_finite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ManifestError("Manifest contains a non-finite float")
        if isinstance(value, Mapping):
            for nested in value.values():
                require_finite(nested)
        elif isinstance(value, list):
            for nested in value:
                require_finite(nested)

    if not isinstance(payload, dict):
        raise ManifestError("Manifest root must be a JSON object")
    require_finite(payload)
    return payload


def _strict_json_object(path: Path | str) -> dict[str, Any]:
    """Read a finite JSON object while rejecting duplicate object keys."""
    manifest_path = Path(path)
    try:
        payload_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ManifestError(f"Cannot read strict JSON: {manifest_path}") from error
    return _strict_json_bytes(payload_bytes, source=str(manifest_path))


def _read_manifest_once(path: Path) -> tuple[bytes, int]:
    """Read one non-symlink regular manifest and return bytes plus mode."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ManifestError(
            f"Cannot open manifest without symlink traversal: {path}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ManifestError(f"Manifest is not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read(), stat.S_IMODE(metadata.st_mode)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> None:
    """Validate the required top-level structure and digest syntax."""
    required = {
        "schema_version",
        "generator",
        "collection_context",
        "runtime_state",
        "claim_boundary",
        "integrity",
    }
    missing = sorted(required.difference(manifest))
    unknown = sorted(set(manifest).difference(required))
    if missing:
        raise ManifestError(f"Manifest lacks required fields: {missing}")
    if unknown:
        raise ManifestError(f"Manifest has unknown top-level fields: {unknown}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported schema_version: {manifest['schema_version']!r}"
        )
    if not isinstance(manifest["runtime_state"], Mapping):
        raise ManifestError("runtime_state must be an object")
    integrity = manifest["integrity"]
    if not isinstance(integrity, Mapping):
        raise ManifestError("integrity must be an object")
    if integrity.get("algorithm") != HASH_ALGORITHM:
        raise ManifestError("Unsupported integrity algorithm")
    if integrity.get("canonicalization") != CANONICALIZATION:
        raise ManifestError("Unsupported integrity canonicalization")
    for name in ("runtime_state_sha256", "content_sha256"):
        digest = integrity.get(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ManifestError(f"integrity.{name} is not a SHA-256 digest")


def _without_integrity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact content authenticated by ``content_sha256``."""
    return {key: value for key, value in manifest.items() if key != "integrity"}


def _iter_artifacts(value: Any) -> Iterable[Mapping[str, Any]]:
    """Yield every structurally recognizable file-artifact record."""
    if isinstance(value, Mapping):
        required = {"role", "path", "size_bytes", "sha256", "mode"}
        if required.issubset(value):
            yield value
            return
        for nested in value.values():
            yield from _iter_artifacts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_artifacts(nested)


def _verify_artifact(record: Mapping[str, Any]) -> list[str]:
    """Verify one captured file record and return detailed errors."""
    role = record.get("role", "unknown")
    path_value = record.get("path")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        return [f"{role}: artifact path is not absolute"]
    try:
        observed = _capture_file(Path(path_value), role=str(role))
    except ManifestError as error:
        return [f"{role}: {error}"]
    keys = (
        "path",
        "resolved_path",
        "is_symlink",
        "symlink_target",
        "mode",
        "size_bytes",
        "sha256",
    )
    return [
        (
            f"{role}: {key} mismatch: expected={record.get(key)!r}, "
            f"observed={observed.get(key)!r}"
        )
        for key in keys
        if record.get(key) != observed.get(key)
    ]


def _compare_runtime_states(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> list[str]:
    """Report a concise digest-level live-state comparison."""
    expected_digest = _sha256_bytes(_canonical_json_bytes(expected))
    observed_digest = _sha256_bytes(_canonical_json_bytes(observed))
    if expected_digest == observed_digest:
        return []
    changed_sections = sorted(
        key
        for key in set(expected).union(observed)
        if expected.get(key) != observed.get(key)
    )
    return [
        (
            "Live runtime state differs from the locked state: "
            f"expected={expected_digest}, observed={observed_digest}, "
            f"changed_sections={changed_sections}"
        )
    ]


def verify_manifest(
    manifest_path: Path | str,
    *,
    check_live_state: bool = True,
    require_read_only: bool = True,
    expected_file_sha256: str | None = None,
    expected_content_sha256: str | None = None,
    **collection_options: Any,
) -> dict[str, Any]:
    """Verify external anchors, JSON integrity, evidence, and optional live state."""
    path = _lexical_absolute(manifest_path)
    errors: list[str] = []
    observed_file_digest: str | None = None
    observed_mode: int | None = None
    if (
        expected_file_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256) is None
    ):
        return {
            "valid": False,
            "manifest_path": str(path),
            "artifact_count": 0,
            "live_state_checked": False,
            "errors": ["expected_file_sha256 is not a SHA-256 digest"],
        }
    if (
        expected_content_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", expected_content_sha256) is None
    ):
        return {
            "valid": False,
            "manifest_path": str(path),
            "artifact_count": 0,
            "live_state_checked": False,
            "errors": ["expected_content_sha256 is not a SHA-256 digest"],
        }
    try:
        manifest_bytes, observed_mode = _read_manifest_once(path)
        observed_file_digest = _sha256_bytes(manifest_bytes)
    except ManifestError as error:
        return {
            "valid": False,
            "manifest_path": str(path),
            "artifact_count": 0,
            "live_state_checked": False,
            "errors": [str(error)],
        }
    if (
        expected_file_sha256 is not None
        and observed_file_digest != expected_file_sha256
    ):
        errors.append(
            "Manifest file SHA-256 differs from the external anchor: "
            f"expected={expected_file_sha256}, observed={observed_file_digest}"
        )
    try:
        manifest = _strict_json_bytes(manifest_bytes, source=str(path))
        _validate_manifest_shape(manifest)
    except ManifestError as error:
        return {
            "valid": False,
            "manifest_path": str(path),
            "file_sha256": observed_file_digest,
            "artifact_count": 0,
            "live_state_checked": False,
            "errors": errors + [str(error)],
        }

    integrity = manifest["integrity"]
    runtime_state = manifest["runtime_state"]
    observed_content_digest = _sha256_bytes(
        _canonical_json_bytes(_without_integrity(manifest))
    )
    observed_state_digest = _sha256_bytes(_canonical_json_bytes(runtime_state))
    if observed_content_digest != integrity["content_sha256"]:
        errors.append(
            "Manifest content SHA-256 mismatch: "
            f"expected={integrity['content_sha256']}, "
            f"observed={observed_content_digest}"
        )
    if (
        expected_content_sha256 is not None
        and integrity["content_sha256"] != expected_content_sha256
    ):
        errors.append(
            "Manifest content SHA-256 differs from the external anchor: "
            f"expected={expected_content_sha256}, "
            f"observed={integrity['content_sha256']}"
        )
    if observed_state_digest != integrity["runtime_state_sha256"]:
        errors.append(
            "Runtime-state SHA-256 mismatch: "
            f"expected={integrity['runtime_state_sha256']}, "
            f"observed={observed_state_digest}"
        )
    if require_read_only and observed_mode is not None and observed_mode & 0o222:
        errors.append(f"Manifest is writable: path={path}, mode={observed_mode:04o}")

    artifacts = list(_iter_artifacts(manifest))
    for artifact in artifacts:
        errors.extend(_verify_artifact(artifact))

    live_checked = False
    if check_live_state:
        live_checked = True
        environment = runtime_state.get("environment")
        environment_root = (
            environment.get("root") if isinstance(environment, Mapping) else None
        )
        if not isinstance(environment_root, str):
            errors.append("runtime_state.environment.root is missing")
        else:
            try:
                observed_state = collect_runtime_state(
                    environment_root, **collection_options
                )
                errors.extend(_compare_runtime_states(runtime_state, observed_state))
            except (ManifestError, OSError, TypeError) as error:
                errors.append(f"Cannot collect live runtime state: {error}")
    return {
        "valid": not errors,
        "manifest_path": str(path),
        "file_sha256": observed_file_digest,
        "content_sha256": integrity["content_sha256"],
        "runtime_state_sha256": integrity["runtime_state_sha256"],
        "artifact_count": len(artifacts),
        "live_state_checked": live_checked,
        "errors": errors,
    }


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync of a directory after publication."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_manifest(
    output_path: Path | str,
    manifest: Mapping[str, Any],
    *,
    replace: bool = False,
) -> None:
    """Publish a read-only manifest atomically and create-only by default."""
    output = _lexical_absolute(output_path)
    parent = output.parent
    if not parent.is_dir():
        raise ManifestError(f"Output directory does not exist: {parent}")
    serialized = (
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = parent / f".{output.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, READ_ONLY_MODE)
        if replace:
            os.replace(temporary, output)
        else:
            try:
                os.link(temporary, output)
            except FileExistsError as error:
                raise ManifestError(
                    f"Refusing to overwrite existing manifest: {output}"
                ) from error
            temporary.unlink()
        _fsync_directory(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _require_creation_policy(
    manifest: Mapping[str, Any],
    *,
    allow_non_isolated: bool,
    allow_unhealthy_packages: bool,
) -> None:
    """Enforce conservative defaults before publishing a runtime lock."""
    runtime_state = manifest["runtime_state"]
    environment = runtime_state["environment"]
    pip_check = runtime_state["package_health"]["pip_check"]
    if (
        not allow_non_isolated
        and environment["scope_classification"] != "isolated_virtual_environment"
    ):
        raise ManifestError(
            "Environment is not an isolated matching virtual environment; "
            "use --allow-non-isolated only for an explicitly reviewed exception"
        )
    if not allow_unhealthy_packages and pip_check.get("ok") is not True:
        raise ManifestError(
            "pip check did not pass; use --allow-unhealthy-packages only for "
            "an explicitly reviewed exception"
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create", help="Collect and atomically publish a runtime manifest"
    )
    create.add_argument("--environment-root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an existing output; create-only is the default",
    )
    create.add_argument("--allow-non-isolated", action="store_true")
    create.add_argument("--allow-unhealthy-packages", action="store_true")

    verify = subparsers.add_parser("verify", help="Verify an existing runtime manifest")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument(
        "--no-live-state",
        action="store_true",
        help="Verify recorded files and JSON integrity without recollecting state",
    )
    verify.add_argument(
        "--allow-writable",
        action="store_true",
        help="Do not fail solely because the manifest has writable mode bits",
    )
    verify.add_argument(
        "--expected-file-sha256",
        help="Require the exact manifest bytes to match this external SHA-256",
    )
    verify.add_argument(
        "--expected-content-sha256",
        help="Require integrity.content_sha256 to match this external SHA-256",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command == "create":
            manifest = build_manifest(arguments.environment_root)
            _require_creation_policy(
                manifest,
                allow_non_isolated=arguments.allow_non_isolated,
                allow_unhealthy_packages=arguments.allow_unhealthy_packages,
            )
            atomic_write_manifest(
                arguments.output,
                manifest,
                replace=arguments.replace,
            )
            report = verify_manifest(
                arguments.output,
                check_live_state=False,
            )
        else:
            report = verify_manifest(
                arguments.manifest,
                check_live_state=not arguments.no_live_state,
                require_read_only=not arguments.allow_writable,
                expected_file_sha256=arguments.expected_file_sha256,
                expected_content_sha256=arguments.expected_content_sha256,
            )
    except (ManifestError, OSError, TypeError) as error:
        report = {"valid": False, "errors": [str(error)]}
    print(
        json.dumps(
            report,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
