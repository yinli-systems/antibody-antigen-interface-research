#!/usr/bin/env python3
"""Freeze and verify paired raw-Brier negative-sample ablation protocols."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ARMS = {
    "with_negative": {
        "specificity_loss_weight": 0.25,
        "wrong_partner_gradient_contribution": True,
    },
    "no_negative": {
        "specificity_loss_weight": 0.0,
        "wrong_partner_gradient_contribution": False,
    },
}
TREE_ROOTS = ("protocols", "scripts", "src")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inventory(source_root: Path) -> tuple[list[dict[str, Any]], str]:
    files: list[dict[str, Any]] = []
    for relative_root in TREE_ROOTS:
        root = source_root / relative_root
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or path.is_symlink()
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
            ):
                continue
            files.append(
                {
                    "bytes": path.stat().st_size,
                    "path": path.relative_to(source_root).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
    encoded = json.dumps(
        files, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return files, hashlib.sha256(encoded).hexdigest()


def publish_json(path: Path, value: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def freeze(
    *,
    dataset: str,
    arm: str,
    base_protocol: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    if arm not in ARMS:
        raise ValueError(arm)
    trainer = source_root / "scripts/train_framecontact.py"
    base_raw = base_protocol.read_bytes()
    base = json.loads(base_raw)
    inventory, inventory_sha = source_inventory(source_root)
    trainer_sha = sha256_file(trainer)
    protocol = copy.deepcopy(base)
    weight = ARMS[arm]["specificity_loss_weight"]
    protocol["schema_version"] = (
        f"framecontact_{dataset}_partner_negative_ablation_protocol_v2"
    )
    protocol["frozen_on"] = "2026-08-21"
    protocol["protocol_status"] = "development_only_raw_brier_v2"
    protocol["claim_boundary"] = (
        f"{dataset} development train/validation comparison of K=1 standard "
        "wrong-partner specificity supervision against an otherwise matched "
        "zero-specificity-loss arm. Test and external-test roles remain sealed."
    )
    objective = protocol.setdefault("training_objective", {})
    objective["loss_weights"] = {
        "contact": 1.0,
        "epitope": 0.25,
        "paratope": 0.25,
        "specificity": weight,
    }
    objective["specificity_margin"] = 0.05
    objective["score_fraction"] = 0.01
    protocol["negative_sample_ablation"] = {
        "arm": arm,
        "base_protocol_file_sha256": hashlib.sha256(base_raw).hexdigest(),
        "base_protocol_path": str(base_protocol),
        "comparison": "paired five-seed K=1 versus zero-weight ablation",
        "specificity_loss_weight": weight,
        "specificity_negative_count": 1,
        "wrong_partner_gradient_contribution": ARMS[arm][
            "wrong_partner_gradient_contribution"
        ],
        "test_loaded": False,
        "external_test_loaded": False,
    }
    protocol["measurement"] = {
        "fixed_raw_logit_contact_metric_path": (
            "validation.contact.fixed_raw_logit_threshold_zero"
        ),
        "probability_temperature": 1.0,
        "probability_threshold": 0.5,
        "trainer_file_sha256": trainer_sha,
    }
    protocol["source_closure"] = {
        "file_count": len(inventory),
        "inventory_sha256": inventory_sha,
        "source_root": str(source_root),
        "trainer_file_sha256": trainer_sha,
    }
    output_root.mkdir(parents=True)
    protocol_path = output_root / f"framecontact_{dataset}_{arm}_v2.json"
    publish_json(protocol_path, protocol)
    receipt = {
        "format_version": "framecontact_partner_negative_ablation_receipt_v2",
        "arm": arm,
        "dataset": dataset,
        "protocol_file_sha256": sha256_file(protocol_path),
        "trainer_file_sha256": trainer_sha,
        "source_inventory_sha256": inventory_sha,
        "source_file_count": len(inventory),
        "base_protocol_file_sha256": hashlib.sha256(base_raw).hexdigest(),
        "base_protocol_path": str(base_protocol),
        "specificity_loss_weight": weight,
        "fixed_raw_logit_contact_metric_path": (
            "validation.contact.fixed_raw_logit_threshold_zero"
        ),
        "test_loaded": False,
        "external_test_loaded": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    publish_json(output_root / "PROTOCOL_RECEIPT.json", receipt)
    return receipt


def verify(
    *, receipt_path: Path, protocol_path: Path, source_root: Path
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("format_version") != (
        "framecontact_partner_negative_ablation_receipt_v2"
    ):
        raise ValueError("Wrong receipt format")
    if receipt.get("test_loaded") is not False or receipt.get(
        "external_test_loaded"
    ) is not False:
        raise ValueError("Test authority is not sealed")
    if sha256_file(protocol_path) != receipt.get("protocol_file_sha256"):
        raise ValueError("Protocol hash differs")
    trainer = source_root / "scripts/train_framecontact.py"
    if sha256_file(trainer) != receipt.get("trainer_file_sha256"):
        raise ValueError("Trainer hash differs")
    inventory, inventory_sha = source_inventory(source_root)
    if inventory_sha != receipt.get("source_inventory_sha256"):
        raise ValueError("Source inventory differs")
    if len(inventory) != receipt.get("source_file_count"):
        raise ValueError("Source file count differs")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--dataset", choices=("asep", "sabdab2"), required=True)
    freeze_parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    freeze_parser.add_argument("--base-protocol", type=Path, required=True)
    freeze_parser.add_argument("--source-root", type=Path, required=True)
    freeze_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--receipt", type=Path, required=True)
    verify_parser.add_argument("--protocol", type=Path, required=True)
    verify_parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        payload = freeze(
            dataset=args.dataset,
            arm=args.arm,
            base_protocol=args.base_protocol,
            source_root=args.source_root,
            output_root=args.output_root,
        )
    else:
        payload = verify(
            receipt_path=args.receipt,
            protocol_path=args.protocol,
            source_root=args.source_root,
        )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
