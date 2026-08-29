#!/usr/bin/env python3
"""Freeze cluster-disjoint selection and confirmation panels before replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


SOURCE_SPLIT_SHA256 = "4ff0965a0f8dde3913a176ac0411b40c9d2db3d1cb2684c393d6b01886b806cd"
PARTNER_PANEL_SHA256 = "c7140727a062d1f42a6e27613aa2708075f3aab65815f3a18c6a0caaf1435c11"
FRAMECONTACT_SHA256 = "a85584f288b32e66e6ab34bcd93b0001ffcd2b173bb1b0fc9e3ab48bccd6171c"
TRAINER_SHA256 = "ef989bbaa8878c285c46440126bdcbfc22e8261c70db9b5c57d48446ed490b2e"
SOURCE_MANIFEST_SHA256 = "f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81"
STRUCTURE_MANIFEST_SHA256 = "edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0"
CONTACT_MANIFEST_SHA256 = "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33"
EMBEDDING_MANIFEST_SHA256 = "9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545"
EVALUATOR_SHA256 = "f3c99990cc9a5f962ed5c289d9d3b2b55adcc5252f0eafffaa544482924b5683"
READOUT_ORDER = [
    "direct_total",
    "paper_mean_total_contact",
    "head_plus_paper_mean_a100",
]
RUNS = {
    "17": {
        "result_sha256": "38bbf1a4ade66b01abd81a925ac89464725c67564bc2c4c352701d6d8a0f8184",
        "checkpoint_sha256": "5c2448c28a335dac0dcfa03d97a63be7331d664c3181c83a38bb563ca61b9a22",
        "receipt_sha256": "f63cf0e8f7cf24f1224bbd0b1890cc259fae0de96e66e47b30d23efa61494623",
    },
    "29": {
        "result_sha256": "5ec2248f51d9690a3d8ce161ea767f3232fa302c3f7dfbb430c4ba97fafb2c86",
        "checkpoint_sha256": "688943d5de706bfc5cb612b94d761fee5ec11064eb8b0c12e082e0d188471e48",
        "receipt_sha256": "fdbdad3813093b30952a0f5624f08fd814ecd7a345c657367291b42e012911f2",
    },
    "43": {
        "result_sha256": "17fdd96cfc1b5c3f15e9736dd356f186fb7ba32574565915fcdcaed3106affc0",
        "checkpoint_sha256": "8baa30bca423044930bf09e8266df144261dcd25edf7b9bc251a18bc0e02539d",
        "receipt_sha256": "7e8012ca213296e80d6eb0756876ceb02bd7eaf3e167507ee145488c6a9dad9a",
    },
}


def canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def write_new(path: Path, payload: object) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.write_bytes(canonical_bytes(payload))
    return sha256_file(path)


def write_csv_new(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return sha256_file(path)


def partition_entries(entries: list[dict]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[str]] = {}
    for entry in entries:
        cluster = str(entry["antigen_cluster_id"])
        groups.setdefault(cluster, []).append(str(entry["instance_id"]))
    ordered = sorted(
        groups,
        key=lambda cluster: hashlib.sha256(
            f"epitope-readout-v1|{cluster}".encode()
        ).hexdigest(),
    )
    assigned: dict[str, list[dict[str, str]]] = {
        "selection": [],
        "confirmation": [],
    }
    cluster_counts = {"selection": 0, "confirmation": 0}
    for cluster in ordered:
        role = min(
            assigned,
            key=lambda name: (
                len(assigned[name]),
                cluster_counts[name],
                0 if name == "selection" else 1,
            ),
        )
        cluster_counts[role] += 1
        assigned[role].extend(
            {
                "instance_id": identifier,
                "antigen_cluster_id": cluster,
            }
            for identifier in sorted(groups[cluster])
        )
    selection_clusters = {
        entry["antigen_cluster_id"] for entry in assigned["selection"]
    }
    confirmation_clusters = {
        entry["antigen_cluster_id"] for entry in assigned["confirmation"]
    }
    if selection_clusters & confirmation_clusters:
        raise RuntimeError("antigen cluster leakage across roles")
    return assigned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-root-template", required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--partner-panel", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    for path, expected, label in (
        (args.source_split, SOURCE_SPLIT_SHA256, "source split"),
        (args.partner_panel, PARTNER_PANEL_SHA256, "partner panel"),
        (args.source_manifest, SOURCE_MANIFEST_SHA256, "source manifest"),
        (args.structure_manifest, STRUCTURE_MANIFEST_SHA256, "structure manifest"),
        (args.contact_manifest, CONTACT_MANIFEST_SHA256, "contact manifest"),
        (args.embedding_manifest, EMBEDDING_MANIFEST_SHA256, "embedding manifest"),
        (args.evaluator, EVALUATOR_SHA256, "readout evaluator"),
        (
            repository_root / "src/interaction_first/framecontact.py",
            FRAMECONTACT_SHA256,
            "FrameContact source",
        ),
        (
            repository_root / "scripts/train_framecontact.py",
            TRAINER_SHA256,
            "trainer source",
        ),
    ):
        authenticate(path, expected, label)
    for seed, expected in RUNS.items():
        root = Path(args.run_root_template.format(seed=seed))
        for filename, field in (
            ("FRAMECONTACT_DEVELOPMENT_RESULT.json", "result_sha256"),
            ("FRAMECONTACT_SELECTED.pt", "checkpoint_sha256"),
            ("RUN_RECEIPT.json", "receipt_sha256"),
        ):
            authenticate(root / filename, expected[field], f"seed-{seed} {filename}")
        result = json.loads((root / "FRAMECONTACT_DEVELOPMENT_RESULT.json").read_text())
        authority = result.get("authority", {})
        if (
            result.get("experiment") != "epiformer_style_same_protocol"
            or int(result.get("configuration", {}).get("seed", -1)) != int(seed)
            or authority.get("test_loaded") is not False
            or authority.get("external_test_loaded") is not False
            or authority.get("split_manifest_sha256") != SOURCE_SPLIT_SHA256
            or authority.get("structure_manifest_sha256") != STRUCTURE_MANIFEST_SHA256
            or authority.get("contact_manifest_sha256") != CONTACT_MANIFEST_SHA256
            or authority.get("embedding_manifest_sha256") != EMBEDDING_MANIFEST_SHA256
        ):
            raise ValueError(f"seed-{seed} source authority differs")

    with args.source_split.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        source_rows = list(reader)
    panel = json.loads(args.partner_panel.read_text(encoding="utf-8"))
    entries = panel.get("entries")
    if not isinstance(entries, list) or len(entries) != 128:
        raise ValueError("expected 128 label-free validation panel entries")
    assigned = partition_entries(entries)
    panel_ids = {str(entry["instance_id"]) for entry in entries}
    validation_rows = {
        row["instance_id"]: row for row in source_rows if row.get("role") == "validation"
    }
    if set(validation_rows) != panel_ids:
        raise ValueError("source validation rows and partner panel differ")

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    authority_root = args.output_root / "authority"
    authority_root.mkdir(parents=True)
    (args.output_root / "logs").mkdir()
    (args.output_root / "seed-results").mkdir()
    role_split_hashes = {}
    for role, role_entries in assigned.items():
        role_ids = {entry["instance_id"] for entry in role_entries}
        rows = []
        for source_row in source_rows:
            row = dict(source_row)
            if row.get("role") == "validation" and row["instance_id"] not in role_ids:
                row["role"] = "train"
            rows.append(row)
        role_split_hashes[role] = write_csv_new(
            authority_root / f"split-{role}.csv", fieldnames, rows
        )
    partition = {
        "format_version": "sabdab2_epitope_readout_cluster_partition_v1",
        "method": "deterministic_label_free_antigen_cluster_greedy_balance",
        "split_view_method": (
            "retain_all_cohort_identifiers; non-panel validation rows are assigned "
            "to train for evaluation-view filtering only"
        ),
        "partition_seed": "epitope-readout-v1",
        "source_partner_panel_sha256": PARTNER_PANEL_SHA256,
        "task_labels_read": False,
        "test_features_or_labels_read": False,
        "roles": assigned,
        "counts": {
            role: {
                "complexes": len(role_entries),
                "antigen_clusters": len(
                    {entry["antigen_cluster_id"] for entry in role_entries}
                ),
            }
            for role, role_entries in assigned.items()
        },
    }
    partition_path = authority_root / "PARTITION_EPITOPE_READOUT_V1.json"
    partition_sha256 = write_new(partition_path, partition)
    protocol = {
        "format_version": "sabdab2_epitope_readout_protocol_v1",
        "phase": "selection",
        "status": "frozen_before_measurement",
        "research_basis": {
            "paper": "EpiFormer arXiv:2606.04154v1 equation 9",
            "official_code_commit": "f420745e28adf759ccdab327d06459da9a9382fe",
            "single_factor": "replace contact marginal readout with mean edge probability",
        },
        "data": {
            "dataset": "SAbDab2",
            "role": "frozen development validation",
            "panel_role": "selection",
            "source_split_sha256": SOURCE_SPLIT_SHA256,
            "role_split_sha256": role_split_hashes["selection"],
            "partition_manifest_sha256": partition_sha256,
            "partner_panel_sha256": PARTNER_PANEL_SHA256,
            "structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
            "contact_manifest_sha256": CONTACT_MANIFEST_SHA256,
            "embedding_manifest_sha256": EMBEDDING_MANIFEST_SHA256,
            "complex_count": partition["counts"]["selection"]["complexes"],
            "antigen_cluster_count": partition["counts"]["selection"][
                "antigen_clusters"
            ],
        },
        "source": {
            "framecontact_sha256": FRAMECONTACT_SHA256,
            "trainer_sha256": TRAINER_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "evaluator_sha256": EVALUATOR_SHA256,
        },
        "runs": RUNS,
        "readout_order": READOUT_ORDER,
        "selection": {
            "seed": 17,
            "baseline": "direct_total",
            "candidates": READOUT_ORDER[1:],
            "minimum_complex_weighted_delta": 0.005,
            "bonferroni_one_sided_lower_quantile": 0.0125,
            "cluster_bootstrap_replicates": 50000,
            "cluster_bootstrap_seed": 20260826,
            "cluster_balanced_delta_strictly_positive": True,
            "tie_preference": [
                "head_plus_paper_mean_a100",
                "paper_mean_total_contact",
            ],
        },
        "authority": {
            "measurement_only": True,
            "model_parameters_updated": False,
            "confirmation_not_yet_authorized": True,
            "test_loaded": False,
            "temporal_holdout_loaded": False,
        },
    }
    protocol_path = authority_root / "PROTOCOL_EPITOPE_READOUT_SELECTION_V1.json"
    protocol_sha256 = write_new(protocol_path, protocol)
    receipt = {
        "format_version": "sabdab2_epitope_readout_protocol_receipt_v1",
        "phase": "selection",
        "protocol_sha256": protocol_sha256,
        "partition_manifest_sha256": partition_sha256,
        "evaluator_sha256": EVALUATOR_SHA256,
        "frozen_before_measurement": True,
        "test_loaded": False,
        "temporal_holdout_loaded": False,
    }
    receipt_sha256 = write_new(
        authority_root / "PROTOCOL_EPITOPE_READOUT_SELECTION_V1_RECEIPT.json",
        receipt,
    )
    print(
        json.dumps(
            {
                "status": "PASS_FROZEN_BEFORE_MEASUREMENT",
                "protocol_sha256": protocol_sha256,
                "protocol_receipt_sha256": receipt_sha256,
                "partition_sha256": partition_sha256,
                "counts": partition["counts"],
                "test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
