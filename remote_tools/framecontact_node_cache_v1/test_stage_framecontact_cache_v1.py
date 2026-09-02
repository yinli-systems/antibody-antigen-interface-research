from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).with_name("stage_framecontact_cache_v1.py")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_cache(root: Path, name: str, payload: bytes) -> tuple[Path, str]:
    cache = root / name
    artifact = cache / "artifacts" / "aa" / "item.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(payload)
    (cache / "auxiliary.txt").write_text("authority\n", encoding="utf-8")
    manifest = cache / "MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "artifact_path": "artifacts/aa/item.bin",
                        "artifact_sha256": digest(artifact),
                        "artifact_size_bytes": len(payload),
                    }
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return cache, digest(manifest)


def command(cache: Path, manifest_sha: str, node_root: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--source-root",
        f"tiny={cache}",
        "--manifest-relative",
        "tiny=MANIFEST.json",
        "--manifest-sha256",
        f"tiny={manifest_sha}",
        "--node-cache-root",
        str(node_root),
    ]


def test_stage_and_reuse(tmp_path: Path) -> None:
    cache, manifest_sha = make_cache(tmp_path, "source", b"payload")
    node_root = tmp_path / "node"
    first = subprocess.run(
        command(cache, manifest_sha, node_root), check=True, capture_output=True, text=True
    )
    first_result = json.loads(first.stdout)
    assert first_result["status"] == "PASS_NODE_LOCAL_CACHE_VERIFIED"
    assert first_result["reused_existing_cache"] is False
    staged = Path(first_result["target_roots"]["tiny"])
    assert (staged / "artifacts/aa/item.bin").read_bytes() == b"payload"

    second = subprocess.run(
        command(cache, manifest_sha, node_root), check=True, capture_output=True, text=True
    )
    second_result = json.loads(second.stdout)
    assert second_result["reused_existing_cache"] is True
    assert second_result["spec_sha256"] == first_result["spec_sha256"]


def test_rejects_bad_source_manifest_authority(tmp_path: Path) -> None:
    cache, manifest_sha = make_cache(tmp_path, "source", b"payload")
    result = subprocess.run(
        command(cache, "0" * 64, tmp_path / "node"), capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "source manifest authority differs" in result.stderr
    assert manifest_sha != "0" * 64


def test_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    cache, manifest_sha = make_cache(tmp_path, "source", b"payload")
    args = command(cache, manifest_sha, tmp_path / "node")
    args[args.index("tiny=MANIFEST.json")] = "tiny=../MANIFEST.json"
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0
    assert "unsafe relative path" in result.stderr


def test_rejects_corrupted_existing_stage(tmp_path: Path) -> None:
    cache, manifest_sha = make_cache(tmp_path, "source", b"payload")
    node_root = tmp_path / "node"
    first = subprocess.run(
        command(cache, manifest_sha, node_root), check=True, capture_output=True, text=True
    )
    staged = Path(json.loads(first.stdout)["target_roots"]["tiny"])
    (staged / "artifacts/aa/item.bin").write_bytes(b"corrupt")

    second = subprocess.run(
        command(cache, manifest_sha, node_root), capture_output=True, text=True
    )
    assert second.returncode != 0
    assert "staged artifact SHA256 differs" in second.stderr
