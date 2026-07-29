#!/usr/bin/env python3
"""Publish deterministic train-only PAIRScale development subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.development_authority import (  # noqa: E402
    canonical_json_bytes,
    development_policy_from_json,
)
from interaction_first.development_subset import (  # noqa: E402
    build_development_subsets,
    build_subset_receipt,
    parse_train_authorization,
)


def _sha256_file(path: Path) -> str:
    """Hash one regular non-symlink file."""

    path = path.expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: str, label: str) -> str:
    """Validate one lowercase SHA-256 string."""

    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _publish_json(path: Path, payload: Any) -> None:
    """Create one immutable JSON artifact in a private staging directory."""

    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o440)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-inventory", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--expected-builder-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 256])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--requested-at-utc", required=True)
    parser.add_argument("--evaluated-at-utc", required=True)
    return parser.parse_args()


def main() -> None:
    """Verify all anchors and atomically publish one nested subset family."""

    args = _parse_args()
    authority_sha256 = _sha256_file(args.authority_inventory)
    expected_authority_sha256 = _sha256(
        args.expected_authority_sha256,
        "expected_authority_sha256",
    )
    if authority_sha256 != expected_authority_sha256:
        raise ValueError("Authority inventory digest mismatch")
    builder_sha256 = _sha256_file(Path(__file__))
    if builder_sha256 != _sha256(
        args.expected_builder_sha256,
        "expected_builder_sha256",
    ):
        raise ValueError("Builder source digest mismatch")
    policy = development_policy_from_json(args.policy.read_bytes())
    if policy.policy_sha256 != _sha256(
        args.expected_policy_sha256,
        "expected_policy_sha256",
    ):
        raise ValueError("Policy digest mismatch")
    inventory = parse_train_authorization(args.authority_inventory.read_bytes())
    subsets = build_development_subsets(
        inventory=inventory,
        parent_authorization_sha256=authority_sha256,
        policy=policy,
        expected_policy_sha256=args.expected_policy_sha256,
        sizes=args.sizes,
        seed=args.seed,
        requested_at_utc=args.requested_at_utc,
        evaluated_at_utc=args.evaluated_at_utc,
    )
    implementation_source_sha256 = {
        "scripts/build_pairscale_development_subsets.py": builder_sha256,
        "src/interaction_first/development_authority.py": _sha256_file(
            SOURCE_ROOT / "interaction_first" / "development_authority.py"
        ),
        "src/interaction_first/development_subset.py": _sha256_file(
            SOURCE_ROOT / "interaction_first" / "development_subset.py"
        ),
    }
    receipt = build_subset_receipt(
        subsets,
        implementation_source_sha256=implementation_source_sha256,
    )

    output_root = args.output_root.expanduser().absolute()
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("Output root already exists; publication is create-only")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(
        f".{output_root.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    staging.mkdir(mode=0o700)
    try:
        for subset in subsets:
            subset_root = staging / f"n{subset.requested_size}"
            subset_root.mkdir(mode=0o700)
            _publish_json(
                subset_root / "AUTHORIZED_FOLD_INVENTORY.json",
                subset.authorization_payload,
            )
            _publish_json(
                subset_root / "PAIRSCALE_DEVELOPMENT_SUBSET.json",
                subset.subset_payload,
            )
            _publish_json(
                subset_root / "DEVELOPMENT_AUTHORITY_DECISION.json",
                subset.decision.canonical_payload(),
            )
            subset_root.chmod(0o550)
        _publish_json(staging / "SUBSET_BUILD_RECEIPT.json", receipt)
        staging.chmod(0o550)
        os.rename(staging, output_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "receipt_sha256": receipt["receipt_sha256"],
                "subsets": receipt["subsets"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
