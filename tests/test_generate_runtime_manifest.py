#!/usr/bin/env python3
"""Tests for immutable runtime-manifest creation and verification."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any
from unittest import mock

import generate_runtime_manifest as runtime_manifest
import pytest
from generate_runtime_manifest import (
    ManifestError,
    PytorchProbe,
    _canonical_json_bytes,
    _collect_distributions,
    _require_creation_policy,
    _sha256_bytes,
    _strict_json_object,
    atomic_write_manifest,
    build_manifest,
    verify_manifest,
)


class SyntheticDistribution:
    """Provide the importlib.metadata surface used by the collector."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        location: Path,
        metadata_directory: Path,
    ) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self.location = location
        self._path = metadata_directory

    def locate_file(self, relative: str) -> Path:
        """Resolve a distribution-relative file."""
        return self.location / relative


def _write_distribution(
    root: Path,
    *,
    name: str,
    version: str,
    direct_url: bool = False,
) -> SyntheticDistribution:
    site_packages = root / "lib" / "python3.11" / "site-packages"
    metadata_directory = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
    metadata_directory.mkdir(parents=True)
    (metadata_directory / "RECORD").write_text(
        f"{name}/__init__.py,,\n", encoding="utf-8"
    )
    if direct_url:
        (metadata_directory / "direct_url.json").write_text(
            '{"url":"file:///reviewed/source"}\n', encoding="utf-8"
        )
    return SyntheticDistribution(
        name=name,
        version=version,
        location=site_packages,
        metadata_directory=metadata_directory,
    )


def _synthetic_inputs(
    root: Path,
) -> tuple[list[SyntheticDistribution], PytorchProbe, Path]:
    root.mkdir()
    (root / "pyvenv.cfg").write_text(
        "home = /opt/python\ninclude-system-site-packages = false\nversion = 3.11.9\n",
        encoding="utf-8",
    )
    distributions = [
        _write_distribution(
            root,
            name="Zulu-Package",
            version="2.0",
            direct_url=True,
        ),
        _write_distribution(
            root,
            name="alpha_package",
            version="1.0",
        ),
    ]
    shared_library = root / "lib" / "libsynthetic_nccl.so"
    shared_library.parent.mkdir(exist_ok=True)
    shared_library.write_bytes(b"synthetic shared library\n")
    pytorch = PytorchProbe(
        metadata={
            "available": True,
            "version": "2.13.0+cu130",
            "git_version": "synthetic-git-version",
            "debug_build": False,
            "module": None,
            "native_extension": None,
            "cuda": {
                "build_version": "13.0",
                "compiled_version_integer": 13000,
            },
            "cudnn": {"version_integer": 91002},
            "nccl": {"version_components": [2, 29, 7]},
            "cxx11_abi": True,
            "build_configuration": "synthetic",
        },
        library_candidates=(("nccl_shared_library", shared_library),),
    )
    return distributions, pytorch, shared_library


def _pip_check(ok: bool = True) -> dict[str, Any]:
    return {
        "command": ["/synthetic/python", "-m", "pip", "check"],
        "timeout_seconds": 120,
        "exit_code": 0 if ok else 1,
        "ok": ok,
        "stdout": (
            "No broken requirements found."
            if ok
            else "package-a requires missing-package"
        ),
        "stderr": "",
    }


def _platform() -> dict[str, Any]:
    return {
        "system": "Linux",
        "release": "synthetic-release",
        "version": "synthetic-version",
        "machine": "x86_64",
        "processor": "synthetic",
        "architecture": {"bits": "64bit", "linkage": "ELF"},
        "libc": {"name": "glibc", "version": "2.39"},
    }


def _build_synthetic_manifest(
    root: Path,
) -> tuple[dict[str, Any], list[SyntheticDistribution], PytorchProbe]:
    distributions, pytorch, _ = _synthetic_inputs(root)
    with mock.patch.object(
        runtime_manifest,
        "_python_shared_library_candidates",
        return_value=[],
    ):
        manifest = build_manifest(
            root,
            collected_at_utc="2026-07-25T00:00:00Z",
            collection_host="synthetic-host",
            distribution_provider=lambda: distributions,
            pip_check_result=_pip_check(),
            pytorch_probe=pytorch,
            platform_probe=_platform,
        )
    return manifest, distributions, pytorch


def test_canonical_json_digest_is_order_independent_and_rejects_nan() -> None:
    first = {"b": [2, 3], "a": {"value": 1}}
    second = {"a": {"value": 1}, "b": [2, 3]}
    assert _canonical_json_bytes(first) == _canonical_json_bytes(second)
    assert _sha256_bytes(_canonical_json_bytes(first)) == _sha256_bytes(
        _canonical_json_bytes(second)
    )
    with pytest.raises(ManifestError, match="strict JSON"):
        _canonical_json_bytes({"invalid": float("nan")})


@pytest.mark.parametrize(
    "content, expected",
    [
        ('{"a": 1, "a": 2}\n', "Duplicate JSON key"),
        ('{"a": NaN}\n', "Non-finite JSON value"),
        ('["not", "an", "object"]\n', "root must be a JSON object"),
    ],
)
def test_strict_json_reader_rejects_ambiguous_input(
    tmp_path: Path,
    content: str,
    expected: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ManifestError, match=expected):
        _strict_json_object(path)


def test_distribution_inventory_is_sorted_and_hashes_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    distributions, _, _ = _synthetic_inputs(root)
    inventory, _ = _collect_distributions(distributions)
    assert [item["normalized_name"] for item in inventory] == [
        "alpha-package",
        "zulu-package",
    ]
    assert inventory[0]["evidence"]["record"]["sha256"]
    assert inventory[0]["evidence"]["direct_url"] is None
    assert inventory[1]["evidence"]["record"]["sha256"]
    assert inventory[1]["evidence"]["direct_url"]["sha256"]


def test_manifest_records_versions_scope_libraries_and_claim_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    manifest, _, _ = _build_synthetic_manifest(root)
    state = manifest["runtime_state"]
    pytorch = state["frameworks"]["pytorch"]
    assert pytorch["version"] == "2.13.0+cu130"
    assert pytorch["cuda"]["build_version"] == "13.0"
    assert pytorch["cudnn"]["version_integer"] == 91002
    assert pytorch["nccl"]["version_components"] == [2, 29, 7]
    assert state["environment"]["include_system_site_packages"] is False
    assert state["environment"]["prefix_matches_root"] is False
    assert state["package_health"]["pip_check"]["ok"] is True
    assert state["platform_abi"] == {
        "architecture": {"bits": "64bit", "linkage": "ELF"},
        "kernel_release_bound": False,
        "libc": {"name": "glibc", "version": "2.39"},
        "machine": "x86_64",
        "processor_model_bound": False,
        "system": "Linux",
    }
    assert [item["normalized_name"] for item in state["installed_distributions"]] == [
        "alpha-package",
        "zulu-package",
    ]
    assert any(
        item["role"] == "nccl_shared_library" for item in state["shared_libraries"]
    )
    assert "state-of-the-art" in manifest["claim_boundary"]["does_not_attest"]
    assert manifest["integrity"]["runtime_state_sha256"] == _sha256_bytes(
        _canonical_json_bytes(state)
    )
    unsigned = {key: value for key, value in manifest.items() if key != "integrity"}
    assert manifest["integrity"]["content_sha256"] == _sha256_bytes(
        _canonical_json_bytes(unsigned)
    )


def test_stable_platform_abi_excludes_kernel_and_processor_variation() -> None:
    first = _platform()
    second = {
        **first,
        "release": "different-kernel-release",
        "version": "different-kernel-build",
        "processor": "different-node-processor",
    }

    assert runtime_manifest._stable_platform_abi(first) == (
        runtime_manifest._stable_platform_abi(second)
    )


def test_live_verifier_accepts_node_telemetry_variation(tmp_path: Path) -> None:
    root = tmp_path / "environment"
    manifest, distributions, pytorch = _build_synthetic_manifest(root)
    output = tmp_path / "runtime-manifest.json"
    atomic_write_manifest(output, manifest)
    second_platform = {
        **_platform(),
        "release": "other-kernel",
        "version": "other-kernel-build",
        "processor": "other-processor",
    }

    with mock.patch.object(
        runtime_manifest,
        "_python_shared_library_candidates",
        return_value=[],
    ):
        report = verify_manifest(
            output,
            check_live_state=True,
            distribution_provider=lambda: distributions,
            pip_check_result=_pip_check(),
            pytorch_probe=pytorch,
            platform_probe=lambda: second_platform,
        )

    assert report["valid"] is True, report["errors"]


def test_atomic_write_is_read_only_create_only_and_preserves_existing_bytes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runtime-manifest.json"
    payload = {"strict": True, "value": 1}
    atomic_write_manifest(output, payload)
    original = output.read_bytes()
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(ManifestError, match="Refusing to overwrite"):
        atomic_write_manifest(output, {"strict": True, "value": 2})
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".runtime-manifest.json.tmp-*"))


def test_explicit_replace_is_atomic_and_restores_read_only_mode(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runtime-manifest.json"
    atomic_write_manifest(output, {"generation": 1})
    atomic_write_manifest(output, {"generation": 2}, replace=True)
    assert json.loads(output.read_text(encoding="utf-8"))["generation"] == 2
    assert stat.S_IMODE(output.stat().st_mode) == 0o444


def test_verifier_accepts_untampered_manifest_and_exact_live_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    manifest, distributions, pytorch = _build_synthetic_manifest(root)
    output = tmp_path / "runtime-manifest.json"
    atomic_write_manifest(output, manifest)
    with mock.patch.object(
        runtime_manifest,
        "_python_shared_library_candidates",
        return_value=[],
    ):
        report = verify_manifest(
            output,
            check_live_state=True,
            distribution_provider=lambda: distributions,
            pip_check_result=_pip_check(),
            pytorch_probe=pytorch,
            platform_probe=_platform,
        )
    assert report["valid"] is True, report["errors"]
    assert report["live_state_checked"] is True
    assert report["artifact_count"] >= 5


def test_verifier_detects_recorded_artifact_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    manifest, _, _ = _build_synthetic_manifest(root)
    output = tmp_path / "runtime-manifest.json"
    atomic_write_manifest(output, manifest)
    (root / "pyvenv.cfg").write_text(
        "include-system-site-packages = true\n", encoding="utf-8"
    )
    report = verify_manifest(output, check_live_state=False)
    assert report["valid"] is False
    assert any(
        "pyvenv_cfg" in error and "mismatch" in error for error in report["errors"]
    )


def test_verifier_requires_caller_frozen_external_anchors(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    manifest, _, _ = _build_synthetic_manifest(root)
    output = tmp_path / "runtime-manifest.json"
    atomic_write_manifest(output, manifest)
    file_sha256, _ = runtime_manifest._sha256_file(output)
    content_sha256 = manifest["integrity"]["content_sha256"]

    accepted = verify_manifest(
        output,
        check_live_state=False,
        expected_file_sha256=file_sha256,
        expected_content_sha256=content_sha256,
    )
    assert accepted["valid"] is True, accepted["errors"]
    assert accepted["file_sha256"] == file_sha256

    wrong_file = verify_manifest(
        output,
        check_live_state=False,
        expected_file_sha256="0" * 64,
    )
    assert wrong_file["valid"] is False
    assert any("external anchor" in error for error in wrong_file["errors"])

    wrong_content = verify_manifest(
        output,
        check_live_state=False,
        expected_content_sha256="1" * 64,
    )
    assert wrong_content["valid"] is False
    assert any("external anchor" in error for error in wrong_content["errors"])


@pytest.mark.parametrize(
    "field",
    ("expected_file_sha256", "expected_content_sha256"),
)
def test_verifier_rejects_malformed_external_digest(
    tmp_path: Path,
    field: str,
) -> None:
    root = tmp_path / "environment"
    manifest, _, _ = _build_synthetic_manifest(root)
    output = tmp_path / "runtime-manifest.json"
    atomic_write_manifest(output, manifest)
    report = verify_manifest(
        output,
        check_live_state=False,
        **{field: "../not-a-digest"},
    )
    assert report["valid"] is False
    assert any(field in error for error in report["errors"])


def test_verifier_detects_json_tampering_even_when_file_is_made_read_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    manifest, _, _ = _build_synthetic_manifest(root)
    output = tmp_path / "runtime-manifest.json"
    atomic_write_manifest(output, manifest)
    os.chmod(output, 0o644)
    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["claim_boundary"]["attests"] = "broader unsupported claim"
    output.write_text(
        json.dumps(tampered, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(output, 0o444)
    report = verify_manifest(output, check_live_state=False)
    assert report["valid"] is False
    assert any("content SHA-256 mismatch" in error for error in report["errors"])


def test_verifier_rejects_writable_manifest_by_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    manifest, _, _ = _build_synthetic_manifest(root)
    output = tmp_path / "runtime-manifest.json"
    atomic_write_manifest(output, manifest)
    os.chmod(output, 0o644)
    strict_report = verify_manifest(output, check_live_state=False)
    permissive_report = verify_manifest(
        output,
        check_live_state=False,
        require_read_only=False,
    )
    assert strict_report["valid"] is False
    assert any("Manifest is writable" in error for error in strict_report["errors"])
    assert permissive_report["valid"] is True, permissive_report["errors"]


def test_verifier_rejects_manifest_symlink_even_with_matching_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    manifest, _, _ = _build_synthetic_manifest(root)
    target = tmp_path / "runtime-manifest-target.json"
    link = tmp_path / "runtime-manifest.json"
    atomic_write_manifest(target, manifest)
    link.symlink_to(target.name)
    report = verify_manifest(link, check_live_state=False)
    assert report["valid"] is False
    assert any("symlink traversal" in error for error in report["errors"])


def test_creation_policy_requires_isolation_and_healthy_packages(
    tmp_path: Path,
) -> None:
    root = tmp_path / "environment"
    manifest, _, _ = _build_synthetic_manifest(root)
    with pytest.raises(ManifestError, match="not an isolated"):
        _require_creation_policy(
            manifest,
            allow_non_isolated=False,
            allow_unhealthy_packages=False,
        )
    _require_creation_policy(
        manifest,
        allow_non_isolated=True,
        allow_unhealthy_packages=False,
    )
    manifest["runtime_state"]["package_health"]["pip_check"] = _pip_check(False)
    with pytest.raises(ManifestError, match="pip check did not pass"):
        _require_creation_policy(
            manifest,
            allow_non_isolated=True,
            allow_unhealthy_packages=False,
        )
    _require_creation_policy(
        manifest,
        allow_non_isolated=True,
        allow_unhealthy_packages=True,
    )


def test_symlink_evidence_detects_retargeting(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"same content\n")
    second.write_bytes(b"same content\n")
    link = tmp_path / "runtime.so"
    link.symlink_to(first.name)
    evidence = runtime_manifest._capture_file(link, role="runtime_library")
    assert evidence["is_symlink"] is True
    link.unlink()
    link.symlink_to(second.name)
    errors = runtime_manifest._verify_artifact(evidence)
    assert any("symlink_target mismatch" in error for error in errors)
