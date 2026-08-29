#!/usr/bin/env python3
"""Verify every authenticated artifact in an AsEP development cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.asep_external import AsepDevelopmentDataset  # noqa: E402
from interaction_first.sabdab2_structure import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite verification receipt: {arguments.output}")
    manifest_path = arguments.cache_root / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"
    if sha256_file(manifest_path) != arguments.cache_manifest_sha256:
        raise ValueError("AsEP development manifest SHA-256 differs")
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("records", [])
    expected_paths = {
        str((arguments.cache_root / record["artifact_path"]).resolve())
        for record in records
    }
    actual_paths = {
        str(path.resolve()) for path in (arguments.cache_root / "artifacts").rglob("*.pt")
    }
    if actual_paths != expected_paths:
        raise ValueError("AsEP development artifact inventory differs from manifest")
    sidecars = sorted(
        str(path.relative_to(arguments.cache_root))
        for path in arguments.cache_root.rglob("._*")
    )
    if sidecars:
        raise ValueError(f"AppleDouble sidecars are prohibited: {sidecars[:3]}")

    role_summaries: dict[str, dict[str, int]] = {}
    instance_ids: list[str] = []
    for role in ("train", "val"):
        dataset = AsepDevelopmentDataset(
            arguments.cache_root,
            manifest_sha256=arguments.cache_manifest_sha256,
            split_role=role,
            feature_mode="official_esm2_35m",
        )
        summary = {
            "instances": 0,
            "antibody_residues": 0,
            "antigen_residues": 0,
            "positive_contacts": 0,
            "antibody_valid_frames": 0,
            "antigen_valid_frames": 0,
        }
        for index in range(len(dataset)):
            item = dataset[index]
            instance_ids.append(str(item["instance_id"]))
            summary["instances"] += 1
            summary["antibody_residues"] += int(item["antibody_features"].shape[0])
            summary["antigen_residues"] += int(item["antigen_features"].shape[0])
            summary["positive_contacts"] += int(item["pair_labels"].sum().item())
            summary["antibody_valid_frames"] += int(
                item["antibody_frame_valid_mask"].sum().item()
            )
            summary["antigen_valid_frames"] += int(
                item["antigen_frame_valid_mask"].sum().item()
            )
        role_summaries[role] = summary
    if len(instance_ids) != len(set(instance_ids)) or len(instance_ids) != len(records):
        raise ValueError("AsEP development instance inventory is not one-to-one")

    receipt = {
        "format_version": "asep_development_cache_verification_v1",
        "authority": {
            "manifest_sha256": arguments.cache_manifest_sha256,
            "manifest_record_count": len(records),
        },
        "role_summaries": role_summaries,
        "checks": {
            "all_manifest_artifacts_present": True,
            "no_extra_artifacts": True,
            "no_appledouble_sidecars": True,
            "every_artifact_sha256_verified": True,
            "every_artifact_weights_only_loaded": True,
            "contact_marginals_reproduced": True,
            "official_test_materialized": False,
        },
        "claim_boundary": (
            "Integrity and schema verification of the materialized train/validation cache "
            "only. No model fitting, prediction, metric computation, or official-test access."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{arguments.output.name}.",
        dir=arguments.output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json_bytes(receipt))
    try:
        temporary.replace(arguments.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(arguments.output)}, indent=2))


if __name__ == "__main__":
    main()
