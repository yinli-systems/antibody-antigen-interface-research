#!/usr/bin/env python3
"""Freeze a label-independent 20-case SAbDab2 top-10 contact study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


FORMAT = "sabdab2_top10_contact_case_study_protocol_v1"
EXPECTED_SEEDS = (17, 29, 43)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "a320a5f60794ea386d355e8ab5ec895130d3b714596bc817e303eee589dbc7e8"
)
EXPECTED_STRUCTURE_MANIFEST_SHA256 = (
    "b13ecb8a3b184f64a0ff509ddefea41a6aa7da226f710d3ddf79c55c84eef4cb"
)
EXPECTED_STRUCTURE_RECEIPT_SHA256 = (
    "e043be4d266ea72833b9c7fc7fa7d19d1f22f46a97555d5bb221416f145ecb69"
)
EXPECTED_STRUCTURE_BINDING_SHA256 = (
    "165690c13a2d75f50dab8262cf48fbf3e1089e4f9cc5895ddfa4b85e3e09aa06"
)
EXPECTED_CONTACT_MANIFEST_SHA256 = (
    "ad7201ecdf339b16a666addbaa5f775a4a1da03fdf0ea5bcdd68e8aa0f6e8e41"
)
EXPECTED_EMBEDDING_MANIFEST_SHA256 = (
    "9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545"
)
EXPECTED_TEST_SPLIT_SHA256 = (
    "84acf74ce3debc3e176c996fd04b75510daf84183060d5d8353fc17758705a46"
)


def canonical_bytes(value: object) -> bytes:
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


def require_sha(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 differs for {path}: {observed}")


def authenticated_development_run(root: Path) -> dict[str, object]:
    result_path = root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = root / "RUN_RECEIPT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("result_sha256") != sha256_file(result_path)
        or receipt.get("selected_checkpoint_sha256") != sha256_file(checkpoint_path)
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or result.get("authority", {}).get("test_loaded") is not False
        or result.get("configuration", {}).get("distance_loss_weight") != 0.0
        or result.get("configuration", {}).get("distance_label_manifest_sha256")
        is not None
    ):
        raise ValueError(f"Development authority differs for {root}")
    return {
        "root": str(root.resolve()),
        "seed": int(result["configuration"]["seed"]),
        "selected_epoch": int(result["selection"]["best_epoch"]),
        "contact_development_mpc_ap": float(
            result["selected_checkpoint_validation"]["contact"][
                "mean_per_complex_auprc"
            ]
        ),
        "result_sha256": receipt["result_sha256"],
        "checkpoint_sha256": receipt["selected_checkpoint_sha256"],
        "receipt_sha256": sha256_file(receipt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-run-root", type=Path, action="append", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--structure-receipt", type=Path, required=True)
    parser.add_argument("--structure-binding", type=Path, required=True)
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--contact-receipt", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-seed", type=int, default=20_260_904)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    if len(args.candidate_run_root) != len(EXPECTED_SEEDS):
        raise ValueError("Exactly three candidate development runs are required")
    if args.sample_size != 20 or args.top_k != 10:
        raise ValueError("The teacher-requested study is fixed at 20 cases and top-10")

    for path, expected in (
        (args.source_manifest, EXPECTED_SOURCE_MANIFEST_SHA256),
        (args.structure_manifest, EXPECTED_STRUCTURE_MANIFEST_SHA256),
        (args.structure_receipt, EXPECTED_STRUCTURE_RECEIPT_SHA256),
        (args.structure_binding, EXPECTED_STRUCTURE_BINDING_SHA256),
        (args.contact_manifest, EXPECTED_CONTACT_MANIFEST_SHA256),
        (args.embedding_manifest, EXPECTED_EMBEDDING_MANIFEST_SHA256),
        (args.test_split, EXPECTED_TEST_SPLIT_SHA256),
    ):
        require_sha(path, expected)

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if Path(source_manifest["source_root"]).resolve() != args.source_root.resolve():
        raise ValueError("Training source root differs")
    structure = json.loads(args.structure_manifest.read_text(encoding="utf-8"))
    structure_receipt = json.loads(args.structure_receipt.read_text(encoding="utf-8"))
    structure_binding = json.loads(args.structure_binding.read_text(encoding="utf-8"))
    if (
        len(structure.get("records", ())) != 549
        or structure.get("authority", {}).get("geometry_source")
        != "independent_sequence_predicted_af3"
        or structure.get("authority", {}).get(
            "experimental_bound_complex_coordinates_read"
        )
        is not False
        or structure.get("authority", {}).get("cross_partner_coordinates_or_distances_used")
        is not False
        or structure_receipt.get("status")
        != "PASS_EXPOSED_TEST_PREDICTED_STRUCTURE_CACHE"
        or structure_binding.get("experimental_bound_coordinates_as_model_input")
        is not False
        or structure_binding.get("cross_partner_geometry") is not False
    ):
        raise ValueError("Predicted-structure test authority differs")
    contact = json.loads(args.contact_manifest.read_text(encoding="utf-8"))
    contact_receipt = json.loads(args.contact_receipt.read_text(encoding="utf-8"))
    if (
        len(contact.get("records", ())) != 549
        or contact.get("authority", {}).get("target_predicted_structure_manifest_sha256")
        != EXPECTED_STRUCTURE_MANIFEST_SHA256
        or contact_receipt.get("manifest_sha256")
        != EXPECTED_CONTACT_MANIFEST_SHA256
        or contact_receipt.get("selection_or_tuning") is not False
    ):
        raise ValueError("Contact-label authority differs")

    runs = sorted(
        (authenticated_development_run(path) for path in args.candidate_run_root),
        key=lambda row: int(row["seed"]),
    )
    if tuple(int(row["seed"]) for row in runs) != EXPECTED_SEEDS:
        raise ValueError("Development seed set differs")
    selected = max(runs, key=lambda row: float(row["contact_development_mpc_ap"]))
    if int(selected["seed"]) != 43:
        raise ValueError("Pre-test development-only checkpoint selection changed")

    structure_ids = sorted(str(row["instance_id"]) for row in structure["records"])
    if len(structure_ids) != len(set(structure_ids)):
        raise ValueError("Structure manifest contains duplicate instances")
    sampled_ids = sorted(
        random.Random(args.sample_seed).sample(structure_ids, args.sample_size)
    )
    sampled_digest = hashlib.sha256("\n".join(sampled_ids).encode("utf-8")).hexdigest()
    protocol = {
        "format_version": FORMAT,
        "status": "FROZEN_BEFORE_TOP10_CONTACT_LABEL_READ",
        "purpose": (
            "Teacher-requested descriptive case study on 20 fixed historical "
            "SAbDab2 test complexes."
        ),
        "claim_boundary": {
            "historical_test_previously_exposed": True,
            "fresh_blind_test": False,
            "performance_estimate": False,
            "case_study_only": True,
            "model_or_checkpoint_selection_from_test": False,
            "calibration_or_threshold_refit": False,
            "case_selection_within_fixed_sample_uses_labels": True,
        },
        "selection": {
            "rule": "maximum development-validation contact MPC-AP across seeds 17,29,43",
            "candidates": runs,
            "selected_seed": 43,
            "selected_run_root": selected["root"],
            "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        },
        "sampling": {
            "population_count": len(structure_ids),
            "sample_seed": args.sample_seed,
            "sample_size": args.sample_size,
            "top_k": args.top_k,
            "method": "random.Random(seed).sample(sorted(test_instance_ids), sample_size)",
            "sampled_instance_ids": sampled_ids,
            "sampled_instance_ids_sha256": sampled_digest,
        },
        "data": {
            "structure_manifest": str(args.structure_manifest.resolve()),
            "structure_manifest_sha256": EXPECTED_STRUCTURE_MANIFEST_SHA256,
            "structure_receipt": str(args.structure_receipt.resolve()),
            "structure_receipt_sha256": EXPECTED_STRUCTURE_RECEIPT_SHA256,
            "structure_binding": str(args.structure_binding.resolve()),
            "structure_binding_sha256": EXPECTED_STRUCTURE_BINDING_SHA256,
            "contact_manifest": str(args.contact_manifest.resolve()),
            "contact_manifest_sha256": EXPECTED_CONTACT_MANIFEST_SHA256,
            "contact_receipt": str(args.contact_receipt.resolve()),
            "contact_receipt_sha256": sha256_file(args.contact_receipt),
            "embedding_manifest": str(args.embedding_manifest.resolve()),
            "embedding_manifest_sha256": EXPECTED_EMBEDDING_MANIFEST_SHA256,
            "test_split": str(args.test_split.resolve()),
            "test_split_sha256": EXPECTED_TEST_SPLIT_SHA256,
        },
        "execution": {
            "source_root": str(args.source_root.resolve()),
            "source_manifest": str(args.source_manifest.resolve()),
            "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "evaluator": str(args.evaluator.resolve()),
            "evaluator_sha256": sha256_file(args.evaluator),
            "gpu": "NVIDIA GeForce RTX 4090",
        },
    }
    authority = args.output_root / "authority"
    authority.mkdir(parents=True)
    protocol_path = authority / "PROTOCOL_SABDAB2_TOP10_CASE_STUDY_V1.json"
    protocol_path.write_bytes(canonical_bytes(protocol))
    receipt = {
        "format_version": FORMAT,
        "status": protocol["status"],
        "protocol_sha256": sha256_file(protocol_path),
        "sampled_instance_ids_sha256": sampled_digest,
        "test_contact_labels_read": False,
        "model_or_checkpoint_selected_from_test": False,
    }
    receipt_path = authority / "PROTOCOL_SABDAB2_TOP10_CASE_STUDY_V1_RECEIPT.json"
    receipt_path.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                **receipt,
                "protocol": str(protocol_path),
                "receipt_sha256": sha256_file(receipt_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
