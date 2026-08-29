#!/usr/bin/env python3
"""Materialize audited heavy-atom distances, CDR masks, and surface masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.sabdab2_pair_geometry import (  # noqa: E402
    materialize_pair_geometry_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    parser.add_argument("--contact-manifest-sha256", required=True)
    parser.add_argument("--cif-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contact-cutoff", type=float, default=4.5)
    parser.add_argument("--distance-cap", type=float, default=20.0)
    parser.add_argument("--cdr-flank", type=int, default=2)
    parser.add_argument("--antigen-relative-sasa-threshold", type=float, default=0.2)
    parser.add_argument(
        "--coordinate-quantization-tolerance",
        type=float,
        default=0.002,
        help=(
            "Maximum cutoff-boundary discrepancy, in Angstrom, that may be "
            "snapped to the source-label-consistent side."
        ),
    )
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--selection-manifest-sha256")
    parser.add_argument(
        "--instance-id",
        action="append",
        dest="instance_ids",
        help=(
            "Materialize an explicit cohort instance; repeat for multiple instances. "
            "Mutually exclusive with --max-instances and --selection-manifest."
        ),
    )
    arguments = parser.parse_args()
    result = materialize_pair_geometry_cache(
        structure_cache_root=arguments.structure_cache_root,
        structure_manifest_sha256=arguments.structure_manifest_sha256,
        contact_cache_root=arguments.contact_cache_root,
        contact_manifest_sha256=arguments.contact_manifest_sha256,
        cif_root=arguments.cif_root,
        metadata_path=arguments.metadata,
        metadata_sha256=arguments.metadata_sha256,
        canonical_manifest_path=arguments.canonical_manifest,
        canonical_manifest_sha256=arguments.canonical_manifest_sha256,
        output_root=arguments.output_root,
        contact_cutoff=arguments.contact_cutoff,
        distance_cap=arguments.distance_cap,
        cdr_flank=arguments.cdr_flank,
        antigen_relative_sasa_threshold=arguments.antigen_relative_sasa_threshold,
        coordinate_quantization_tolerance=(
            arguments.coordinate_quantization_tolerance
        ),
        max_instances=arguments.max_instances,
        instance_ids=arguments.instance_ids,
        selection_manifest_path=arguments.selection_manifest,
        selection_manifest_sha256=arguments.selection_manifest_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
