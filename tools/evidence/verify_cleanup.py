#!/usr/bin/env python3
"""Read-only repository hygiene and preservation checks; no experiment execution."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "archive/2026-10-07"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def validate_bytes(data: bytes, expected: str, label: str) -> None:
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Preserved bytes differ: " + label)


def verify_declared(items: list[dict], path_key: str, tracked: set[str]) -> int:
    for row in items:
        path = row[path_key]
        if path not in tracked:
            raise ValueError("Declared evidence exists only locally, not in Git: " + path)
        validate_bytes((ROOT / path).read_bytes(), row["sha256"], path)
        # Validate staged/committed bytes too, not only an ignored working-tree copy.
        validate_bytes(git("show", ":" + path), row["sha256"], path + " (Git index)")
    return len(items)


def verify(remote: bool = False, require_clean: bool = False) -> dict:
    tracked = set(git("ls-files", "-z").decode().split("\0")) - {""}
    forbidden = []
    for path in tracked:
        parts = Path(path).parts
        name = Path(path).name
        if any(p in {"node_modules", "__pycache__", ".pytest_cache", ".venv"} for p in parts):
            forbidden.append(path)
        if name in {".DS_Store", ".env"} or name.endswith((".pyc", ".tsbuildinfo")):
            forbidden.append(path)
    if forbidden:
        raise ValueError("Transient/private-local files tracked: " + ", ".join(sorted(set(forbidden))))
    if any(p.startswith(".github/workflows/") for p in tracked):
        raise ValueError("Cleanup unexpectedly activated a hosted workflow")
    for path in ("455_check.txt", "abcite/.github/workflows/eval.yml"):
        if path in tracked or (ROOT / path).exists():
            raise ValueError("Superseded root/workflow entry remains: " + path)
    removed = json.loads((ARCHIVE / "REMOVED_CACHES.json").read_text())["files"]
    for row in removed:
        if row["path"] in tracked or (ROOT / row["path"]).exists():
            raise ValueError("Historical cache remains active: " + row["path"])
        validate_bytes(git("show", row["restore_from"] + ":" + row["path"]), row["sha256"], row["path"])
    integration = ROOT / "results/integration_20261007"
    variants = json.loads((integration / "PRESERVED_VARIANTS.json").read_text())["files"]
    logs = json.loads((integration / "SOFTWARE_VALIDATION.json").read_text())["validation_artifacts"]
    nv = verify_declared(variants, "preserved_path", tracked)
    nl = verify_declared(logs, "path", tracked)
    snapshot = json.loads((ARCHIVE / "BRANCH_SNAPSHOTS.json").read_text())
    for row in snapshot["branches"]:
        subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", row["head"], "HEAD"], check=True)
    packed = (ROOT / "results/atlas/FILE_VERSIONS.jsonl.gz").read_bytes()
    if packed[9] != 255:
        raise ValueError("Atlas gzip OS header is not canonical across Python versions")
    result = {"status": "PASS_REPOSITORY_CLEANUP_INTEGRITY", "tracked_files": len(tracked),
              "preserved_variant_entries_tracked_and_hashed": nv,
              "declared_validation_artifacts_tracked_and_hashed": nl,
              "archived_branch_heads_in_history": len(snapshot["branches"]),
              "historical_embedding_caches_removed_from_working_tree": len(removed),
              "tracked_transient_files": 0, "active_hosted_workflows": 0,
              "scope": "Repository hygiene, tracked evidence and ancestry; not security certification or scientific replication."}
    if remote:
        heads = []
        for line in git("ls-remote", "--heads", "origin").decode().splitlines():
            sha, ref = line.split()
            if ref != "refs/heads/main":
                raise ValueError("Non-main active branch remains: " + ref)
            if sha != git("rev-parse", "HEAD").decode().strip():
                raise ValueError("Local and remote main differ")
            heads.append({"name": "main", "sha": sha})
        if len(heads) != 1:
            raise ValueError("Expected exactly one remote main")
        result["remote_heads"] = heads
    if require_clean:
        if git("status", "--porcelain"):
            raise ValueError("Working tree/index not clean")
        result["working_tree_clean"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.remote, args.require_clean), indent=2))
