#!/usr/bin/env python3
"""Freeze a create-only replay intent from an externally issued membership anchor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ANCHOR_FORMAT = "strict_partner_retrieval_membership_anchor_v1"
INTENT_FORMAT = "strict_partner_retrieval_pinned_replay_intent_v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9._:+-]{1,128}$")
REQUIRED_SOURCE_FILES = {
    "scripts/build_train_only_nested_neighborhood_folds.py",
    "scripts/audit_train_only_nested_neighborhood_folds.py",
    "src/interaction_first/nested_joint_milp.py",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_sha(value: str, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def require_id(value: str, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} is not a safe workflow identifier")
    return value


def parse_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pieces = line.split()
        if len(pieces) != 2:
            raise ValueError("Malformed source manifest line")
        digest, relative = pieces
        require_sha(digest, f"source manifest digest for {relative}")
        if relative in rows:
            raise ValueError(f"Duplicate source manifest path: {relative}")
        rows[relative] = digest
    if set(rows) != REQUIRED_SOURCE_FILES:
        raise ValueError(
            f"Source manifest closure differs: {sorted(rows)} != {sorted(REQUIRED_SOURCE_FILES)}"
        )
    return rows


def verify_source(root: Path, manifest_sha256: str) -> dict[str, str]:
    root = root.resolve()
    manifest = root / "SOURCE_MANIFEST.sha256"
    if sha256(manifest) != require_sha(manifest_sha256, "source manifest SHA-256"):
        raise ValueError("Source manifest digest mismatch")
    entries = parse_manifest(manifest)
    observed_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "SOURCE_MANIFEST.sha256"
    }
    if observed_files != REQUIRED_SOURCE_FILES:
        raise ValueError("Source root contains files outside the exact closure")
    for relative, expected in entries.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Source file digest mismatch: {relative}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--train-authority-allowlist", type=Path, required=True)
    parser.add_argument("--expected-train-authority-sha256", required=True)
    parser.add_argument("--train-canonical-manifest", type=Path, required=True)
    parser.add_argument("--expected-train-manifest-sha256", required=True)
    parser.add_argument("--expected-joint-policy-sha256", required=True)
    parser.add_argument("--pinned-replay-id", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--selection-seed", type=int, default=2026072401)
    parser.add_argument("--joint-time-limit-seconds", type=int, default=43200)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    anchor = load(args.anchor)
    if anchor.get("format_version") != ANCHOR_FORMAT:
        raise ValueError("Membership anchor format mismatch")
    if anchor.get("status") != "FROZEN_FOR_SEPARATE_PINNED_REPLAY":
        raise ValueError("Membership anchor is not replay-ready")
    if anchor.get("replay_authorized") is not True:
        raise PermissionError("Membership anchor does not authorize replay")
    for key in (
        "training_authorized",
        "model_selection_authorized",
        "outer_evaluation_authorized",
        "test_access",
        "ood_access",
    ):
        if anchor.get(key) is not False:
            raise ValueError(f"Anchor unexpectedly enables {key}")

    provisional_id = require_id(anchor.get("provisional_attempt_id"), "provisional ID")
    replay_id = require_id(args.pinned_replay_id, "pinned replay ID")
    if replay_id == provisional_id:
        raise ValueError("Replay ID must differ from provisional ID")
    selected_digest = require_sha(
        anchor.get("selected_binary_membership_sha256"), "selected membership"
    )
    policy_digest = require_sha(
        anchor.get("exact_joint_policy_sha256"), "anchor joint policy"
    )
    if policy_digest != require_sha(
        args.expected_joint_policy_sha256, "expected joint policy"
    ):
        raise ValueError("Anchor and expected joint policy differ")

    source_entries = verify_source(
        args.source_root, args.expected_source_manifest_sha256
    )
    if sha256(args.train_authority_allowlist) != require_sha(
        args.expected_train_authority_sha256, "train authority digest"
    ):
        raise ValueError("Train authority allowlist digest mismatch")
    if sha256(args.train_canonical_manifest) != require_sha(
        args.expected_train_manifest_sha256, "train canonical manifest digest"
    ):
        raise ValueError("Train canonical manifest digest mismatch")
    if not args.python_bin.is_absolute() or not args.python_bin.exists():
        raise ValueError("Python trust-root path is missing or non-absolute")
    if args.workers < 1 or args.chunk_size < 1 or args.joint_time_limit_seconds < 1:
        raise ValueError("Invalid replay resource-independent parameters")

    environment = {
        "FRONTIER_SOURCE_ROOT": str(args.source_root.resolve()),
        "FRONTIER_TRAIN_AUTHORITY_ALLOWLIST": str(
            args.train_authority_allowlist.resolve()
        ),
        "FRONTIER_TRAIN_CANONICAL_MANIFEST": str(
            args.train_canonical_manifest.resolve()
        ),
        "FRONTIER_EXPECTED_TRAIN_AUTHORITY_ALLOWLIST_SHA256": require_sha(
            args.expected_train_authority_sha256, "train authority digest"
        ),
        "FRONTIER_EXPECTED_TRAIN_CANONICAL_MANIFEST_SHA256": require_sha(
            args.expected_train_manifest_sha256, "train manifest digest"
        ),
        "FRONTIER_EXPECTED_SOURCE_MANIFEST_SHA256": require_sha(
            args.expected_source_manifest_sha256, "source manifest digest"
        ),
        "FRONTIER_EXPECTED_JOINT_POLICY_SHA256": policy_digest,
        "FRONTIER_EXPECTED_BUILDER_SOURCE_SHA256": source_entries[
            "scripts/build_train_only_nested_neighborhood_folds.py"
        ],
        "FRONTIER_EXPECTED_AUDITOR_SOURCE_SHA256": source_entries[
            "scripts/audit_train_only_nested_neighborhood_folds.py"
        ],
        "FRONTIER_EXPECTED_JOINT_MILP_SOURCE_SHA256": source_entries[
            "src/interaction_first/nested_joint_milp.py"
        ],
        "FRONTIER_JOINT_ASSIGNMENT_MODE": "externally_pinned_replay",
        "FRONTIER_EXPECTED_JOINT_SELECTED_MEMBERSHIP_SHA256": selected_digest,
        "FRONTIER_PINNED_REPLAY_ID": replay_id,
        "FRONTIER_SOURCE_PROVISIONAL_ATTEMPT_ID": provisional_id,
        "FRONTIER_SPLIT_WORKERS": str(args.workers),
        "FRONTIER_SPLIT_CHUNK_SIZE": str(args.chunk_size),
        "FRONTIER_SPLIT_SEED": str(args.selection_seed),
        "FRONTIER_JOINT_TIME_LIMIT_SECONDS": str(args.joint_time_limit_seconds),
        "FRONTIER_SPLIT_PYTHON_BIN": str(args.python_bin.resolve()),
    }
    payload = {
        "format_version": INTENT_FORMAT,
        "status": "FROZEN_REPLAY_INTENT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "anchor": {
            "path": str(args.anchor.resolve()),
            "sha256": sha256(args.anchor),
        },
        "pinned_replay_id": replay_id,
        "source_provisional_attempt_id": provisional_id,
        "selected_binary_membership_sha256": selected_digest,
        "exact_joint_policy_sha256": policy_digest,
        "environment": environment,
        "source_entries": source_entries,
        "replay_authorized": True,
        "training_authorized": False,
        "model_selection_authorized": False,
        "outer_evaluation_authorized": False,
        "contact_labels_read": False,
        "model_scores_read": False,
        "test_access": False,
        "ood_access": False,
        "required_result": "externally_pinned_replay_plus_independent_non_authorizing_audit",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": sha256(args.output),
                "pinned_replay_id": replay_id,
                "selected_binary_membership_sha256": selected_digest,
                "training_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
