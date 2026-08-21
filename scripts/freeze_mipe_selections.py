#!/usr/bin/env python3
"""Freeze epoch-800 and validation-best MIPE selections before test scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("rb") as handle:
        return json.load(handle)


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def write_new(path: Path, payload: Any) -> str:
    data = json_bytes(payload)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    arguments = parser.parse_args()
    run_root = arguments.run_root.resolve()
    global_path = run_root / "SELECTIONS_FROZEN.json"
    if global_path.exists():
        print(json.dumps(read_json(global_path), indent=2, sort_keys=True))
        return

    config_path = run_root / "track_config.json"
    manifest_path = run_root / "run_manifest.json"
    config = read_json(config_path)
    manifest = read_json(manifest_path)
    config_sha256 = sha256_file(config_path)
    if config_sha256 != manifest["config_sha256"]:
        raise ValueError("Track configuration digest mismatch")
    if int(config["target_epochs"]) != 800 or int(config["folds"]) != 5:
        raise ValueError("Selection requires the complete 800-epoch five-fold track")
    if float(config["learning_rate"]) != 0.0001:
        raise ValueError("Selection is not the paper-appendix 1e-4 track")
    if config["selection"] != {
        "direction": "maximize",
        "metric": "validation_joint_auprc",
        "scope": "validation_only",
    }:
        raise ValueError("Checkpoint selection is not validation-only")

    frozen_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    selection_digests = {}
    for fold in range(5):
        fold_root = run_root / f"fold_{fold}"
        state_root = fold_root / "state"
        complete = read_json(state_root / "COMPLETE.json")
        if int(complete.get("target_epochs", -1)) != 800:
            raise ValueError(f"Fold {fold} is not complete at epoch 800")

        endpoint_root = state_root / "epochs" / "epoch_0800"
        endpoint_checkpoint = endpoint_root / "checkpoint.pt"
        endpoint_history = endpoint_root / "history.json"
        endpoint_receipt_path = endpoint_root / "receipt.json"
        endpoint_receipt = read_json(endpoint_receipt_path)
        if int(endpoint_receipt.get("epoch", -1)) != 800:
            raise ValueError(f"Fold {fold} endpoint receipt is not epoch 800")
        if endpoint_receipt["files"]["checkpoint.pt"] != sha256_file(endpoint_checkpoint):
            raise ValueError(f"Fold {fold} endpoint checkpoint digest mismatch")
        if endpoint_receipt["files"]["history.json"] != sha256_file(endpoint_history):
            raise ValueError(f"Fold {fold} endpoint history digest mismatch")

        best = read_json(state_root / "BEST.json")
        best_epoch = int(best["epoch"])
        best_score = float(best["validation_joint_auprc"])
        if not 1 <= best_epoch <= 800 or not math.isfinite(best_score):
            raise ValueError(f"Fold {fold} validation-best pointer is invalid")
        best_checkpoint = (state_root / best["checkpoint"]).resolve()
        if sha256_file(best_checkpoint) != best["checkpoint_sha256"]:
            raise ValueError(f"Fold {fold} validation-best digest mismatch")

        selection = {
            "schema_version": "mipe-selection-frozen-v1",
            "status": "frozen_before_held_out_performance_scoring",
            "fold": fold,
            "frozen_at": frozen_at,
            "track_id": config["track_id"],
            "config_sha256": config_sha256,
            "data_sha256": manifest["data_sha256"],
            "selection_scope": "validation_only",
            "held_out_performance_scoring_performed": False,
            "provided_test_performance_scoring_performed": False,
            "primary": {
                "kind": "epoch_800_endpoint",
                "epoch": 800,
                "checkpoint": "state/epochs/epoch_0800/checkpoint.pt",
                "checkpoint_sha256": sha256_file(endpoint_checkpoint),
                "history_sha256": sha256_file(endpoint_history),
                "generation_receipt_sha256": sha256_file(endpoint_receipt_path),
            },
            "secondary": {
                "kind": "validation_best",
                "source_kind": best["kind"],
                "epoch": best_epoch,
                "checkpoint": str(best_checkpoint.relative_to(fold_root)),
                "checkpoint_sha256": sha256_file(best_checkpoint),
                "validation_joint_auprc": best_score,
            },
        }
        relative = f"fold_{fold}/state/SELECTION_FROZEN.json"
        selection_digests[relative] = write_new(run_root / relative, selection)

    global_selection = {
        "schema_version": "mipe-selections-frozen-v1",
        "status": "all_fold_selections_frozen_before_held_out_performance_scoring",
        "frozen_at": frozen_at,
        "track_id": config["track_id"],
        "config_sha256": config_sha256,
        "data_sha256": manifest["data_sha256"],
        "primary_kind": "epoch_800_endpoint",
        "secondary_kind": "validation_best",
        "held_out_performance_scoring_performed": False,
        "provided_test_performance_scoring_performed": False,
        "selection_files": selection_digests,
    }
    write_new(global_path, global_selection)
    print(json.dumps(global_selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
