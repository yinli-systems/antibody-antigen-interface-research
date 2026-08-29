#!/usr/bin/env python3
"""Audit whether CHIMERA feature atom14 can reproduce official contact tuples.

The script is score-independent. It operates on complexes accepted by a prior
adapter audit, verifies materialized artifacts against the pinned manifest, maps
the official contact tuples to feature indices, and compares stored distances
with both feature-atom14 and full-PDB all-atom minima.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load module at {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _minimum(coords_left: np.ndarray, coords_right: np.ndarray) -> float:
    squared = np.sum(
        (coords_left[:, None, :] - coords_right[None, :, :]) ** 2,
        axis=-1,
    )
    return float(np.sqrt(max(float(squared.min()), 0.0)))


def _is_hydrogen(atom_name: str) -> bool:
    normalized = atom_name.lstrip("0123456789").upper()
    return normalized.startswith(("H", "D"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    adapter = _load_module(
        "chimera_external_contact_provenance",
        project_root / "src" / "interaction_first" / "chimera_external.py",
    )
    downloader = _load_module(
        "chimera_external_downloader_contact_provenance",
        project_root / "scripts" / "download_chimera_external_subset.py",
    )

    materialization = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        materialization.get("schema_version")
        != "chimera_external_materialization_manifest_v1"
        or materialization.get("dataset_id") != adapter.DATASET_ID
        or materialization.get("pinned_revision") != adapter.PINNED_REVISION
    ):
        raise SystemExit("materialization manifest authority differs")
    artifacts: dict[str, dict[str, Any]] = {}
    for record in materialization.get("artifacts", []):
        relative = record.get("path")
        if not isinstance(relative, str) or relative in artifacts:
            raise SystemExit("materialization manifest contains malformed paths")
        artifacts[relative] = record

    prior = json.loads(args.adapter_audit.read_text(encoding="utf-8"))
    if prior.get("schema_version") != "chimera_external_adapter_audit_v1":
        raise SystemExit("expected the original strict adapter audit")
    complex_ids = sorted(prior.get("adapted_complexes", {}))
    if not complex_ids:
        raise SystemExit("adapter audit has no accepted complexes")

    totals = {
        "official_contact_pairs": 0,
        "feature_atom14_contacts_recovered": 0,
        "feature_atom14_distances_matching_stored_to_0_01A": 0,
        "pdb_all_atom_distances_matching_stored_to_0_01A": 0,
        "hydrogen_only_contacts_at_4_5A": 0,
    }
    complexes_with_hydrogen_only = 0
    complexes_with_all_feature_contacts = 0
    maximum_pdb_distance_error = 0.0
    per_complex: list[dict[str, Any]] = []

    for ordinal, complex_id in enumerate(complex_ids, start=1):
        pdb_id = complex_id.split("_", 1)[0].lower()
        feature_relative = f"complex_features/{complex_id}.pt"
        pdb_relative = f"structures/{pdb_id}.pdb"
        if feature_relative not in artifacts or pdb_relative not in artifacts:
            raise SystemExit(f"manifest lacks source artifact for {complex_id}")
        for relative in (feature_relative, pdb_relative):
            record = artifacts[relative]
            path = args.root / relative
            if path.is_symlink() or not path.is_file():
                raise SystemExit(f"missing or unsafe source artifact: {relative}")
            observed = downloader.observed_digest(path, record["digest_kind"])
            if observed != record["digest"] or path.stat().st_size != int(record["size"]):
                raise SystemExit(f"source artifact verification failed: {relative}")

        feature_record = artifacts[feature_relative]
        expected_sha = (
            feature_record["digest"]
            if feature_record["digest_kind"] == "sha256"
            else None
        )
        feature = adapter.load_feature_archive(
            args.root / feature_relative, expected_sha256=expected_sha
        )
        chains = adapter.parse_pdb_atom_residues(args.root / pdb_relative)
        _, heavy_chain, light_chain, antigen_chain = complex_id.split("_")
        heavy_sequence = str(feature["heavy_sequence"])
        light_sequence = str(feature["light_sequence"])
        antigen_sequence = str(feature["antigen_sequence"])
        heavy_axis = adapter._coordinate_anchored_axis(
            chains,
            heavy_chain,
            heavy_sequence,
            feature["heavy_ca_coords"],
            feature["heavy_atom14_mask"][:, 1],
            "heavy",
        )
        light_axis = ()
        if light_sequence:
            light_axis = adapter._coordinate_anchored_axis(
                chains,
                light_chain,
                light_sequence,
                feature["light_ca_coords"],
                feature["light_atom14_mask"][:, 1],
                "light",
            )
        antigen_axis = adapter._coordinate_anchored_axis(
            chains,
            antigen_chain,
            antigen_sequence,
            feature["antigen_ca_coords"],
            feature["antigen_atom14_mask"][:, 1],
            "antigen",
        )
        antibody_axis = heavy_axis + light_axis
        mapped_pairs, _ = adapter._map_provided_contact_pairs(
            antibody_axis, antigen_axis, feature["contact_pairs"]
        )

        antibody_coords = np.concatenate(
            (feature["heavy_atom14_coords"], feature["light_atom14_coords"]),
            axis=0,
        )
        antibody_mask = np.concatenate(
            (feature["heavy_atom14_mask"], feature["light_atom14_mask"]), axis=0
        )
        antigen_coords = feature["antigen_atom14_coords"]
        antigen_mask = feature["antigen_atom14_mask"]
        recovered = 0
        native_distance_matches = 0
        pdb_distance_matches = 0
        hydrogen_only = 0
        for item, (antibody_index, antigen_index) in zip(
            feature["contact_pairs"], mapped_pairs
        ):
            stored = float(item[6])
            feature_distance = _minimum(
                antibody_coords[antibody_index, antibody_mask[antibody_index]],
                antigen_coords[antigen_index, antigen_mask[antigen_index]],
            )
            if feature_distance <= 4.5001:
                recovered += 1
            if round(feature_distance, 2) == round(stored, 2):
                native_distance_matches += 1

            left = antibody_axis[antibody_index]
            right = antigen_axis[antigen_index]
            if left is None or right is None:
                raise SystemExit("mapped official contact unexpectedly lacks a PDB residue")
            pdb_distance = adapter._minimum_atom_distance(left, right)
            maximum_pdb_distance_error = max(
                maximum_pdb_distance_error, abs(pdb_distance - stored)
            )
            if round(pdb_distance, 2) == round(stored, 2):
                pdb_distance_matches += 1

            left_heavy = np.asarray(
                [not _is_hydrogen(name) for name in left.atom_names], dtype=np.bool_
            )
            right_heavy = np.asarray(
                [not _is_hydrogen(name) for name in right.atom_names], dtype=np.bool_
            )
            if left_heavy.any() and right_heavy.any():
                heavy_distance = _minimum(
                    left.atoms[left_heavy], right.atoms[right_heavy]
                )
                if pdb_distance <= 4.5001 and heavy_distance > 4.5001:
                    hydrogen_only += 1

        pair_count = len(mapped_pairs)
        totals["official_contact_pairs"] += pair_count
        totals["feature_atom14_contacts_recovered"] += recovered
        totals["feature_atom14_distances_matching_stored_to_0_01A"] += (
            native_distance_matches
        )
        totals["pdb_all_atom_distances_matching_stored_to_0_01A"] += (
            pdb_distance_matches
        )
        totals["hydrogen_only_contacts_at_4_5A"] += hydrogen_only
        complexes_with_hydrogen_only += int(hydrogen_only > 0)
        complexes_with_all_feature_contacts += int(recovered == pair_count)
        per_complex.append(
            {
                "complex_id": complex_id,
                "official_contact_pairs": pair_count,
                "feature_atom14_contacts_recovered": recovered,
                "feature_atom14_distances_matching_stored_to_0_01A": native_distance_matches,
                "pdb_all_atom_distances_matching_stored_to_0_01A": pdb_distance_matches,
                "hydrogen_only_contacts_at_4_5A": hydrogen_only,
            }
        )
        if ordinal % 50 == 0 or ordinal == len(complex_ids):
            print(f"audited {ordinal}/{len(complex_ids)} complexes", flush=True)

    official = totals["official_contact_pairs"]
    output = {
        "schema_version": "chimera_contact_provenance_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Score-independent post-access provenance audit. It diagnoses label "
            "materialization and cannot create a confirmatory model result."
        ),
        "dataset_id": adapter.DATASET_ID,
        "pinned_revision": adapter.PINNED_REVISION,
        "materialization_manifest_sha256": adapter.sha256_file(args.manifest),
        "source_adapter_audit_sha256": adapter.sha256_file(args.adapter_audit),
        "complex_count": len(complex_ids),
        **totals,
        "feature_atom14_contact_recovery_fraction": (
            totals["feature_atom14_contacts_recovered"] / official
        ),
        "feature_atom14_stored_distance_match_fraction": (
            totals["feature_atom14_distances_matching_stored_to_0_01A"] / official
        ),
        "pdb_all_atom_stored_distance_match_fraction": (
            totals["pdb_all_atom_distances_matching_stored_to_0_01A"] / official
        ),
        "complexes_with_hydrogen_only_contact": complexes_with_hydrogen_only,
        "complexes_with_all_feature_atom14_contacts": complexes_with_all_feature_contacts,
        "maximum_pdb_all_atom_stored_distance_error": maximum_pdb_distance_error,
        "model_scores_read": False,
        "metrics_computed": False,
        "per_complex": per_complex,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(f"audit={args.output}")
    print(f"audit_sha256={adapter.sha256_file(args.output)}")


if __name__ == "__main__":
    main()
