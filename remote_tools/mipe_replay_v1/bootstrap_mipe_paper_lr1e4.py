#!/usr/bin/env python3
"""Create a fresh, immutable MIPE five-fold run at the paper LR of 1e-4."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

from common import atomic_write_json, sha256_file, verify_file_map, write_exclusive_bytes


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--hardened-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    source_root = arguments.source_root.resolve()
    data_path = arguments.data.resolve()
    hardened_root = arguments.hardened_root.resolve()
    output_root = arguments.output_root.resolve()

    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "mipe-paper-appendix-track-v1":
        raise ValueError("Unexpected track schema")
    if float(config.get("learning_rate", -1)) != 0.0001:
        raise ValueError("Paper appendix learning rate must be 1e-4")
    if int(config.get("target_epochs", -1)) != 800:
        raise ValueError("Paper track must run for 800 epochs")
    if int(config.get("folds", -1)) != 5:
        raise ValueError("Paper track must use five folds")
    if int(config.get("accumulation_steps", -1)) != 16:
        raise ValueError("Released code uses 16-sample gradient accumulation")
    if config.get("training_scope") != "train_and_validation_only":
        raise ValueError("Bootstrap must not access either held-out fold labels or testdata")
    appendix_path = source_root / "Appendix.pdf"
    if sha256_file(appendix_path) != config["paper_protocol"]["appendix_sha256"]:
        raise ValueError("MIPE appendix digest mismatch")
    if sha256_file(data_path) != config["data"]["cvdata.pkl"]:
        raise ValueError("CV dataset digest mismatch")
    source_files = verify_file_map(source_root, config["source_files"])
    hardened_files = {
        name: sha256_file(hardened_root / name)
        for name in config["hardened_source_files"]
    }
    verify_file_map(hardened_root, hardened_files)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    config_digest = sha256_file(config_path)
    write_exclusive_bytes(
        output_root / "track_config.json", config_path.read_bytes(), mode=0o444
    )

    import torch

    code_root = source_root / "code"
    sys.path.insert(0, str(code_root))
    try:
        from model import MIPE
    finally:
        sys.path.pop(0)

    fold_receipts = {}
    for fold in range(5):
        # Reproducible equal initialization across folds; only the data fold differs.
        torch.manual_seed(int(config["seed"]))
        np.random.seed(int(config["seed"]))
        model = MIPE()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["learning_rate"])
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 0,
            "fold": fold,
            "seed": int(config["seed"]),
            "history": [],
            "best_val_joint_auprc": -1.0,
            "source_commit": config["official_source_snapshot"],
            "initialization_policy": "same_seed_per_fold_fresh_model",
        }
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)

        fold_root = output_root / f"fold_{fold}"
        import_root = fold_root / "import"
        state_root = fold_root / "state"
        import_root.mkdir(parents=True)
        (state_root / "epochs").mkdir(parents=True)
        (fold_root / "invocations").mkdir()
        (fold_root / "logs").mkdir()
        checkpoint_path = import_root / "legacy_last.pt"
        write_exclusive_bytes(checkpoint_path, buffer.getvalue(), mode=0o444)
        checkpoint_digest = sha256_file(checkpoint_path)
        receipt = {
            "schema_version": "mipe-fresh-initialization-v1",
            "fold": fold,
            "last_epoch": 0,
            "files": {"legacy_last.pt": checkpoint_digest},
            "track_id": config["track_id"],
            "testdata_accessed": False,
        }
        atomic_write_json(import_root / "receipt.json", receipt)
        os.chmod(import_root / "receipt.json", 0o444)
        atomic_write_json(
            state_root / "BEST.json",
            {
                "schema_version": "mipe-hardened-state-v1",
                "kind": "fresh_initialization",
                "epoch": 0,
                "validation_joint_auprc": -1.0,
                "checkpoint": "../import/legacy_last.pt",
                "checkpoint_sha256": checkpoint_digest,
            },
        )
        fold_receipts[str(fold)] = {
            "last_epoch": 0,
            "checkpoint_sha256": checkpoint_digest,
            "import_receipt_sha256": sha256_file(import_root / "receipt.json"),
        }
        del model, optimizer, checkpoint, buffer

    manifest = {
        "schema_version": "mipe-hardened-run-manifest-v1",
        "status": "ready",
        "track_id": config["track_id"],
        "target_epochs": 800,
        "source_root": str(source_root),
        "source_files": source_files,
        "hardened_source_root": str(hardened_root),
        "hardened_source_files": hardened_files,
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "config_sha256": config_digest,
        "rng_policy": "same_seed_per_fold_then_full_state_restore",
        "evaluation_policy": "validation_only_during_training; official_testdata_not_accessed",
        "paper_protocol": config["paper_protocol"],
        "folds": fold_receipts,
    }
    atomic_write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
