#!/usr/bin/env python3
"""Freeze the prospective temporal double-cold v2 protocol.

This program is intentionally label-free.  It binds the already-selected
ABF30B FrameContact checkpoints and writes the protocol that future release
snapshots must satisfy before a single sealed label opening is permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FROZEN_AT_UTC = "2026-08-30T20:37:46Z"
CUTOFF_DATE_UTC = "2026-08-30"
CHECKPOINT_BINDING_SHA256 = (
    "c8be8d83fbedd6cce0809b6df5258f3a9be9d02a653d3285a6eed74ded9621f7"
)
FINAL_MODEL_SHA256 = (
    "c36e0b1d812f97317109f1f8ab085a349beb2a782a6f4092b3074bfc1eaab3ea"
)
SEEDS = (17, 29, 43)
EXPECTED_CHECKPOINTS = {
    "baseline": {
        "17": "793ac962540c325b6c52feec836edfdf39fb2819cb0d130d9c698d05756ff436",
        "29": "bac8583e7a574f2b78ae9fe921fc43cdfcd22291ad3df2eb7f5e9861988aceb3",
        "43": "e5b65270b33969d27c1bbeaa23dd4eb0a5c2826e5c1b67ffa7824d98754ea285",
    },
    "edge_topk2_supervised": {
        "17": "0cb96b35e6ebcb96b53e59e32b341bc08bd8f3220b15a81974b3b91c77ef2ba2",
        "29": "ba7a0b060ced0bf75514ba8a473106a34d934cc366d9c5dea377bc2895139a6f",
        "43": "728b1faa2ca490ec949e1cc8772666f72ea7ded9af36e88d1c8d4c57feacf30e",
    },
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(path: Path, value: Any) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def verify_binding(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint binding is missing or unsafe")
    if sha256_file(path) != CHECKPOINT_BINDING_SHA256:
        raise ValueError("checkpoint binding SHA-256 differs")
    binding = json.loads(path.read_text())
    if (
        binding.get("status") != "PASS_EXACT_ABF30B_DEVELOPMENT_CHECKPOINT_BINDING"
        or binding.get("abf30b_checkpoint_model_sha256") != FINAL_MODEL_SHA256
        or binding.get("seed_matched_task_routing")
        != {
            "contact": "baseline",
            "epitope": "edge_topk2_supervised",
            "paratope": "baseline",
        }
        or binding.get("sable_used_for_checkpoint_selection") is not False
        or binding.get("selection_or_tuning") is not False
        or binding.get("old_sabdab2_test_loaded") is not False
        or binding.get("temporal_holdout_loaded") is not False
    ):
        raise ValueError("checkpoint binding authority differs")
    for arm, seeds in EXPECTED_CHECKPOINTS.items():
        observed = binding.get("runs", {}).get(arm, {})
        if set(observed) != {str(seed) for seed in SEEDS}:
            raise ValueError(f"checkpoint seed binding differs: {arm}")
        for seed, expected in seeds.items():
            if observed[seed].get("checkpoint_sha256") != expected:
                raise ValueError(f"checkpoint SHA-256 differs: {arm}/seed-{seed}")
    return binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-binding", type=Path, required=True)
    parser.add_argument("--metadata-screen-script", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    verify_binding(args.checkpoint_binding)
    if args.metadata_screen_script.is_symlink() or not args.metadata_screen_script.is_file():
        raise ValueError("metadata screen script is missing or unsafe")

    protocol = {
        "format_version": "framecontact_prospective_temporal_double_cold_protocol_v2",
        "status": "PREREGISTERED_BEFORE_FIRST_POST_CUTOFF_CANDIDATE_SCREEN",
        "frozen_at_utc": FROZEN_AT_UTC,
        "scientific_question": (
            "Prospective generalization to newly released antibody-antigen complexes "
            "that are cold on both antibody and antigen sequence axes."
        ),
        "release_boundary": {
            "cutoff_date_utc": CUTOFF_DATE_UTC,
            "eligibility": "PDB initial_release_date is strictly later than cutoff_date_utc",
            "source": "official SAbDab2 all-summary metadata cross-checked against RCSB PDB initial release metadata",
            "scan_cadence": "weekly after the Wednesday 00:00 UTC wwPDB release",
            "accumulation_rule": (
                "append every eligible release in chronological order; use the first immutable "
                "official snapshot that passes both opening minima; no candidate cherry-picking"
            ),
        },
        "candidate_definition": {
            "experimental_structure_only": True,
            "antigen_types": ["protein", "peptide"],
            "resolution_angstrom_maximum": 3.0,
            "cdr_missing_residues_allowed": False,
            "one_record_per_declared_antibody_instance": True,
            "manual_suitability_review": (
                "permitted only by the sealed steward using metadata and mapping validity; "
                "exclusion reasons must be enumerated before labels are computed"
            ),
        },
        "double_cold_definition": {
            "reference_union": [
                "all model training and development antibody-antigen sequences",
                "all previously exposed SAbDab2 cohorts",
                "SABLE v0.2.0 train and analyst-exposed test sequences",
                "failed temporal v1 label-free candidates",
                "all public SAbDab2 antibody-antigen instances released on or before the cutoff",
            ],
            "heavy_cdr3": {
                "distance": "length(HCDR3_a)+length(HCDR3_b)-2*LCS_length",
                "minimum_to_every_reference": 3,
            },
            "variable_domain": {
                "maximum_global_sequence_identity_to_every_reference": 0.90,
                "heavy_and_available_light_variable_domains_are_compared": True,
            },
            "antigen_blast_exclusion": {
                "program": "BLASTP",
                "exclude_if_identity_at_least": 0.30,
                "and_shorter_sequence_coverage_at_least": 0.70,
                "and_e_value_at_most": 1e-5,
                "also_exclude_any_high_identity_hit_at_least": 0.70,
            },
            "both_axes_must_pass": True,
            "thresholds_may_not_be_relaxed": True,
        },
        "label_free_gate": {
            "minimum_eligible_instances": 36,
            "minimum_unique_pdb_ids": 30,
            "both_minima_required": True,
            "opening_before_gate": False,
            "candidate_sequences_and_metadata_may_be_read": True,
            "bound_coordinates_may_be_read_by_model_team": False,
            "cross_chain_distances_or_contact_labels_may_be_computed": False,
        },
        "frozen_model": {
            "name": "ABF-650M OAS-30B FrameContact",
            "model_checkpoint_sha256": FINAL_MODEL_SHA256,
            "checkpoint_binding_sha256": CHECKPOINT_BINDING_SHA256,
            "seeds": list(SEEDS),
            "component_checkpoint_sha256": EXPECTED_CHECKPOINTS,
            "task_routing": {
                "contact": "baseline checkpoint from the same seed",
                "paratope": "baseline checkpoint from the same seed",
                "epitope": "edge_topk2_supervised checkpoint from the same seed",
            },
            "model_or_checkpoint_selection_after_freeze": False,
            "calibration_or_threshold_refit_after_freeze": False,
        },
        "model_input_boundary": {
            "antibody_embedding": "ABF-650M OAS-30B final residue hidden states",
            "antigen_embedding": "pinned ESM-2 t33 650M residue embeddings",
            "geometry": "independent sequence-predicted chain-internal N/CA/C backbone",
            "experimental_bound_coordinates_as_model_input": False,
            "cross_partner_geometry_as_model_input": False,
            "bound_structures": "labels only, held by sealed steward until the one-time opening",
        },
        "label_and_metric_definition": {
            "contact": "minimum inter-residue non-hydrogen atom distance <= 4.5 angstrom",
            "paratope_and_epitope": "strict antibody- and antigen-axis any-contact marginals",
            "average_precision": "sklearn.metrics.average_precision_score (non-interpolated AP)",
            "primary_estimand": (
                "PDB-equal-weighted mean of per-complex AP after averaging the three fixed seeds"
            ),
            "random_reference": "per-complex positive prevalence",
            "primary_report": "point estimate and PDB-block bootstrap 95% confidence interval",
            "bootstrap": {"replicates": 10000, "seed": 20260831},
            "secondary": [
                "pooled AP",
                "pooled AUROC",
                "precision",
                "recall",
                "F1",
                "MCC",
                "Brier score",
                "ECE-15",
            ],
            "zero_positive_policy": (
                "retain all samples in pooled metrics; exclude only zero-positive complexes "
                "from per-complex AP and report every eligible denominator"
            ),
        },
        "opening_and_reporting": {
            "maximum_bound_label_openings": 1,
            "preferred_executor": "independent third-party sealed steward",
            "internal_fallback": "single irreversible opening only after the frozen gate passes",
            "third_party_returns": (
                "signed aggregate metrics, denominators, bootstrap outputs, software/container "
                "hashes, checkpoint hashes, and audit receipt; no tuning interface"
            ),
            "per_instance_labels_returned_before_manuscript_lock": False,
            "results_may_feed_back_to_model_or_protocol": False,
            "all_results_reported_regardless_of_outcome": True,
        },
        "closed_evidence": {
            "sable_v0_2_0_may_be_reused_as_blind_test": False,
            "old_sabdab2_exposed_test_may_be_accessed": False,
            "failed_temporal_v1_may_be_reopened": False,
            "failed_temporal_v1_label_opening_count": 0,
            "failed_temporal_v1_label_free_manifest_sha256": (
                "0440bb2ea3adaf1b6c698826659d23e113c096642c8cf086c65efd3881c7a591"
            ),
        },
        "implementation_authority": {
            "freezer_sha256": sha256_file(Path(__file__)),
            "metadata_screen_script_sha256": sha256_file(args.metadata_screen_script),
            "labels_or_bound_coordinates_read_by_freezer": False,
        },
    }
    protocol_path = args.output_root / "PROSPECTIVE_TEMPORAL_DOUBLE_COLD_PROTOCOL_V2.json"
    protocol_sha = publish(protocol_path, protocol)
    receipt = {
        "format_version": "framecontact_prospective_temporal_double_cold_protocol_receipt_v2",
        "status": protocol["status"],
        "protocol_sha256": protocol_sha,
        "checkpoint_binding_sha256": CHECKPOINT_BINDING_SHA256,
        "frozen_before_first_post_cutoff_candidate_screen": True,
        "bound_coordinates_read": False,
        "task_labels_read": False,
        "opening_count": 0,
        "sable_reuse_as_blind_test": False,
    }
    receipt_sha = publish(
        args.output_root / "PROSPECTIVE_TEMPORAL_DOUBLE_COLD_PROTOCOL_V2_RECEIPT.json",
        receipt,
    )
    print(json.dumps({"protocol_sha256": protocol_sha, "receipt_sha256": receipt_sha}))


if __name__ == "__main__":
    main()
