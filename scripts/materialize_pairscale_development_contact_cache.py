#!/usr/bin/env python3
"""Publish an isolated PAIRScale contact cache from train-only artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.development_contact_cache import (  # noqa: E402
    materialize_development_contact_cache,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--expected-parent-cache-manifest-sha256", required=True)
    parser.add_argument("--subset-root", type=Path, required=True)
    parser.add_argument("--expected-subset-authorization-sha256", required=True)
    parser.add_argument("--expected-subset-manifest-sha256", required=True)
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--contact-summary", type=Path, required=True)
    parser.add_argument("--expected-contact-summary-sha256", required=True)
    parser.add_argument("--expected-controller-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    controller_sha256 = _sha256_file(Path(__file__))
    if controller_sha256 != args.expected_controller_sha256:
        raise ValueError("Controller source digest mismatch")
    contact_summary_sha256 = _sha256_file(args.contact_summary)
    if contact_summary_sha256 != args.expected_contact_summary_sha256:
        raise ValueError("Contact summary digest mismatch")
    implementation_source_sha256 = {
        "scripts/materialize_pairscale_development_contact_cache.py": (
            controller_sha256
        ),
        "src/interaction_first/contacts.py": _sha256_file(
            SOURCE_ROOT / "interaction_first" / "contacts.py"
        ),
        "src/interaction_first/development_contact_cache.py": _sha256_file(
            SOURCE_ROOT / "interaction_first" / "development_contact_cache.py"
        ),
    }
    subset_root = args.subset_root
    receipt = materialize_development_contact_cache(
        parent_cache_root=args.parent_cache_root,
        parent_cache_manifest_sha256=(
            args.expected_parent_cache_manifest_sha256
        ),
        subset_authorization_bytes=(
            subset_root / "AUTHORIZED_FOLD_INVENTORY.json"
        ).read_bytes(),
        subset_authorization_sha256=(
            args.expected_subset_authorization_sha256
        ),
        subset_manifest_bytes=(
            subset_root / "PAIRSCALE_DEVELOPMENT_SUBSET.json"
        ).read_bytes(),
        subset_manifest_sha256=args.expected_subset_manifest_sha256,
        decision_bytes=(
            subset_root / "DEVELOPMENT_AUTHORITY_DECISION.json"
        ).read_bytes(),
        decision_sha256=args.expected_decision_sha256,
        expected_policy_sha256=args.expected_policy_sha256,
        contact_root=args.contact_root,
        contact_summary_sha256=contact_summary_sha256,
        output_root=args.output_root,
        implementation_source_sha256=implementation_source_sha256,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
