#!/usr/bin/env python3
"""Freeze three independent high-value study protocols before data access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


FULL_ANALYSIS_SHA256 = "f4368275e38d5134e667e23470702e987bb1a301b1c89aee5e82718d39911824"
FULL_PROTOCOL_SHA256 = "a00a51a4617a581c57a74f854defd6c18d8258180659775b1c5044be10f98c5c"
IDENTITY_MANIFEST_SHA256 = "23e6b9848216cdc57ea3210cba89905a659817b6a49a3de9815d9b5153da5303"
REFERENCE_SEQUENCES_SHA256 = "27e0beb755c3d8e4b9910d4ad2ca979161668a1e1fe7b4d09bf6a60251e07468"
PAIRED_HITS_SHA256 = "edf4c952061ffef3e48b24ddb61a30d303a9cca26a3fa47c5ddcb6ce8f8f8338"
PAIRED_ARCHIVE_SHA256 = "86e89ff7854e1565f343b3377ca713dfc0ff0c79aa28fec99c4b7fecf163f977"
BOUND_UNBOUND_COMMIT = "6891e6e6790c7a898a08471193f494adf6f16b48"
SEEDS = (17, 29, 43)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, path)


def bind_runs(analysis: dict[str, object]) -> dict[str, object]:
    bindings = analysis["run_bindings"]
    result: dict[str, object] = {}
    for arm in ("baseline_current", "edge_topk2_supervised"):
        result[arm] = {}
        for seed in SEEDS:
            binding = dict(bindings[arm][str(seed)])
            root = Path(binding["run_root"])
            checkpoint = root / "FRAMECONTACT_SELECTED.pt"
            run_result = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
            receipt = root / "RUN_RECEIPT.json"
            for path, key, label in (
                (checkpoint, "checkpoint_sha256", "checkpoint"),
                (run_result, "result_sha256", "result"),
                (receipt, "receipt_sha256", "receipt"),
            ):
                require_sha(path, binding[key], f"{arm} seed {seed} {label}")
            result[arm][str(seed)] = {
                **binding,
                "checkpoint": str(checkpoint),
                "result": str(run_result),
                "receipt": str(receipt),
            }
    return result


def write_protocol(root: Path, name: str, payload: dict[str, object]) -> dict[str, object]:
    authority = root / "authority"
    protocol_path = authority / name
    atomic_write(protocol_path, canonical_bytes(payload))
    receipt = {
        "format_version": "frozen_protocol_receipt_v1",
        "status": payload["status"],
        "protocol": name,
        "protocol_sha256": sha256_file(protocol_path),
    }
    receipt_path = authority / name.replace(".json", "_RECEIPT.json")
    atomic_write(receipt_path, canonical_bytes(receipt))
    return {**receipt, "receipt_sha256": sha256_file(receipt_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-analysis", type=Path, required=True)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--reference-sequences", type=Path, required=True)
    parser.add_argument("--paired-hits", type=Path, required=True)
    parser.add_argument("--paired-source-receipt", type=Path, required=True)
    parser.add_argument("--paired-archive", type=Path, required=True)
    parser.add_argument("--legacy-paired-failure-receipt", type=Path, required=True)
    parser.add_argument("--temporal-script", type=Path, required=True)
    parser.add_argument("--bound-unbound-script", type=Path, required=True)
    parser.add_argument("--paired-sharder", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    require_sha(args.full_analysis, FULL_ANALYSIS_SHA256, "full-development analysis")
    require_sha(args.identity_manifest, IDENTITY_MANIFEST_SHA256, "temporal identity manifest")
    require_sha(args.reference_sequences, REFERENCE_SEQUENCES_SHA256, "development sequences")
    require_sha(args.paired_hits, PAIRED_HITS_SHA256, "paired homology hits")
    require_sha(args.paired_archive, PAIRED_ARCHIVE_SHA256, "paired OAS archive")
    analysis = json.loads(args.full_analysis.read_text())
    if (
        analysis.get("status") != "PASS_CURRENT_BEST_FULL_DEVELOPMENT_TASK_ROUTING"
        or analysis.get("protocol_sha256") != FULL_PROTOCOL_SHA256
        or analysis.get("old_test_loaded") is not False
        or analysis.get("temporal_holdout_loaded") is not False
        or analysis.get("full_validation_is_independent_confirmation") is not False
        or analysis.get("task_routing", {}).get("routing_is_fixed_per_task") is not True
        or analysis.get("task_routing", {}).get("mixture_weights_tuned") is not False
    ):
        raise ValueError("full-development authority differs")
    source_receipt = json.loads(args.paired_source_receipt.read_text())
    if source_receipt.get("test_access") is not False:
        raise ValueError("paired source receipt indicates test access")
    legacy_failure = json.loads(args.legacy_paired_failure_receipt.read_text())
    if (
        legacy_failure.get("status") != "FAIL_CLOSED_TIMEOUT"
        or legacy_failure.get("authorization", {}).get("partial_sqlite_reusable") is not False
    ):
        raise ValueError("legacy paired failure receipt differs")
    args.output_root.mkdir(parents=True)
    run_bindings = bind_runs(analysis)

    temporal = {
        "format_version": "framecontact_temporal_double_cold_protocol_v1",
        "status": "FROZEN_BEFORE_STRUCTURE_OR_LABEL_ACCESS",
        "study": "once-only prospective strict-temporal sequence-double-cold evaluation",
        "authorization": {
            "user_explicitly_authorized_this_new_temporal_study": True,
            "authorization_date": "2026-08-27",
            "old_exposed_test_may_be_reopened": False,
            "temporal_opening_count_maximum": 1,
        },
        "identity_cohort": {
            "manifest": str(args.identity_manifest),
            "manifest_sha256": IDENTITY_MANIFEST_SHA256,
            "strict_deposition_after": "2026-03-26",
            "instances": 47,
            "unique_pdbs": 33,
            "old_split_pdb_overlap": 0,
        },
        "label_free_gate": {
            "script_sha256": sha256_file(args.temporal_script),
            "reference_sequences": str(args.reference_sequences),
            "reference_sequences_sha256": REFERENCE_SEQUENCES_SHA256,
            "alignment": "global Needleman-Wunsch; BLOSUM62; gap-open=-10; gap-extend=-0.5",
            "identity_threshold": 0.80,
            "query_and_target_coverage_threshold": 0.80,
            "double_cold_requires_antibody_and_antigen_cold": True,
            "cross_chain_distances_or_labels_allowed_before_gate_freeze": False,
        },
        "frozen_model": {
            "task_routing": analysis["task_routing"],
            "seeds": list(SEEDS),
            "run_bindings": run_bindings,
            "additional_training_on_temporal_data": False,
            "calibration_or_threshold_refit": False,
        },
        "measurement": {
            "primary": "median over seeds of mean-per-eligible-complex Average Precision for each routed task",
            "secondary": ["pooled Average Precision", "pooled AUROC"],
            "uncertainty": "unique-PDB block bootstrap, conditional on the three fixed seeds",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260827,
            "zero_positive_per_complex_policy": "exclude only from per-complex AP denominator; retain in pooled metrics",
        },
        "selection": {
            "allowed": False,
            "model_or_checkpoint_selection_from_temporal": False,
            "hyperparameter_tuning_from_temporal": False,
            "result_feedback_into_development": False,
        },
        "claim_boundary": {
            "prospective_relative_to_training_snapshot": True,
            "independent_of_checkpoint_selection": True,
            "single_opening": True,
            "not_a_claim_of_official_sabdab2_split": True,
        },
    }
    bound_unbound = {
        "format_version": "framecontact_bound_unbound_geometry_protocol_v1",
        "status": "FROZEN_BEFORE_DATA_ACQUISITION",
        "study": "paired bound-to-unbound intra-chain geometry robustness",
        "source": {
            "repository": "Zhaonan99/Antibody-antigen-complex-structure-benchmark-dataset",
            "commit": BOUND_UNBOUND_COMMIT,
            "case_count_expected": 112,
            "subsets": ["87cases", "25cases_Truncated_file"],
            "repository_license_file_observed": False,
            "public_redistribution_by_this_project": False,
        },
        "label_free_gate": {
            "script_sha256": sha256_file(args.bound_unbound_script),
            "minimum_chain_identity": 0.90,
            "minimum_reciprocal_chain_coverage": 0.90,
            "contact_labels_or_cross_chain_distances_allowed": False,
        },
        "frozen_conditions": [
            "bound_antibody_geometry+bound_antigen_geometry",
            "unbound_antibody_geometry+bound_antigen_geometry",
            "bound_antibody_geometry+unbound_antigen_geometry",
            "unbound_antibody_geometry+unbound_antigen_geometry",
        ],
        "frozen_model": {
            "task_routing": analysis["task_routing"],
            "run_bindings": run_bindings,
            "seeds": list(SEEDS),
            "checkpoint_or_threshold_tuning": False,
        },
        "measurement": {
            "paired_unit": "benchmark case",
            "primary": "paired change in per-complex Average Precision from bound/bound",
            "bootstrap": "case-level paired bootstrap, 10000 replicates, conditional on three fixed seeds",
            "bootstrap_seed": 20260828,
        },
        "claim_boundary": {
            "robustness_not_generalization_benchmark": True,
            "training_overlap_will_be_reported_not_used_for_selection": True,
            "sequence_only_comparisons_not_claimed_fair": True,
        },
    }
    paired = {
        "format_version": "abpairfm_esmc_cdr_study_protocol_v1",
        "status": "FROZEN_DATA_GATE_BEFORE_ESMC_MODEL_BINDING",
        "study": "paired-only ESM-C 600M CDR-focused adaptation and fixed FrameContact transfer",
        "literature_anchor": {
            "doi": "10.1038/s44488-026-00010-2",
            "paper": "Preferential CDR masking in paired antibody language models improves binding affinity prediction",
            "paired_only": True,
            "no_unpaired_pretraining": True,
        },
        "data": {
            "archive": str(args.paired_archive),
            "archive_sha256": PAIRED_ARCHIVE_SHA256,
            "source_pairs": source_receipt["counts"]["unique_exact_pairs"],
            "downstream_decontamination_hits": str(args.paired_hits),
            "downstream_decontamination_hits_sha256": PAIRED_HITS_SHA256,
            "provided_test_split_access": False,
            "legacy_partial_sqlite_reused": False,
            "legacy_failure_receipt_sha256": sha256_file(args.legacy_paired_failure_receipt),
        },
        "decontamination": {
            "sharder_sha256": sha256_file(args.paired_sharder),
            "shard_count": 64,
            "minimum_reciprocal_coverage": 0.80,
            "single_chain_exclusion_identity": 0.90,
            "paired_same_development_instance_identity": 0.80,
            "independently_retryable_shards": True,
        },
        "model_binding": {
            "status": "REQUIRED_BEFORE_TRAINING",
            "expected_model_id": "EvolutionaryScale/esmc-600m-2024-12",
            "revision": None,
            "weight_sha256": None,
            "embedding_dimension_expected": 960,
            "license_must_be_accepted_and_recorded": True,
        },
        "paired_input": {
            "format": "VH + dedicated separator token + VL",
            "heavy_light_pairing_preserved": True,
            "cdr_annotation_authority": "OAS IMGT annotations in authenticated paired archive",
        },
        "controlled_arms": {
            "base_esmc": "no antibody adaptation",
            "whole_chain_control": "15% whole-chain masking, BERT 80/10/10 corruption",
            "cdr_focused": "50% masking within CDR1-3 of VH and VL, BERT 80/10/10 corruption",
        },
        "optimization_after_model_binding": {
            "optimizer": "AdamW",
            "learning_rate": 2e-5,
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "precision": "bf16",
            "epochs_maximum": 4,
            "checkpoint_rule": "earliest CDR-recovery optimum on the FR-CDR Pareto frontier",
        },
        "downstream_transfer": {
            "architecture_split_loss_and_seeds_fixed": True,
            "seeds": list(SEEDS),
            "old_exposed_test_access": False,
            "temporal_holdout_access": False,
            "primary": "same-protocol development mean-per-complex Average Precision",
        },
        "claim_boundary": {
            "data_gate_may_run_before_model_weights_arrive": True,
            "training_may_not_start_without_revision_and_weight_sha": True,
            "does_not_claim_pretraining_gain_until_controlled_downstream_transfer": True,
        },
    }
    receipts = {
        "temporal": write_protocol(
            args.output_root / "temporal-double-cold-v1",
            "PROTOCOL_TEMPORAL_DOUBLE_COLD_V1.json",
            temporal,
        ),
        "bound_unbound": write_protocol(
            args.output_root / "bound-unbound-geometry-v1",
            "PROTOCOL_BOUND_UNBOUND_GEOMETRY_V1.json",
            bound_unbound,
        ),
        "paired_esmc_cdr": write_protocol(
            args.output_root / "paired-esmc-cdr-v1",
            "PROTOCOL_PAIRED_ESMC_CDR_V1.json",
            paired,
        ),
    }
    top_receipt = {
        "format_version": "framecontact_three_stream_freeze_receipt_v1",
        "status": "PASS_THREE_STREAM_PROTOCOL_FREEZE",
        "full_development_analysis_sha256": FULL_ANALYSIS_SHA256,
        "protocol_receipts": receipts,
        "old_exposed_test_reopen_authorized": False,
        "temporal_measurements_allowed_before_label_free_gate": False,
        "bound_unbound_labels_allowed_before_mapping_gate": False,
        "esmc_training_allowed_before_weight_binding": False,
    }
    atomic_write(
        args.output_root / "THREE_STREAM_FREEZE_RECEIPT_V1.json",
        canonical_bytes(top_receipt),
    )
    print(json.dumps(top_receipt, sort_keys=True))


if __name__ == "__main__":
    main()
