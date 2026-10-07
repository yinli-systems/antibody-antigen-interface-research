#!/usr/bin/env python3
"""Evaluate one frozen ABF-30B component on the exposed historical test."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch


ARMS = ("baseline", "edge_topk2_supervised")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def require(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"authority SHA differs: {path}")


def verify_source(root: Path, manifest_path: Path, expected: str) -> None:
    require(manifest_path, expected)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(manifest["source_root"]).resolve() != root.resolve():
        raise ValueError("source root differs")
    for row in manifest["records"]:
        path = root / row["path"]
        require(path, row["sha256"])
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"source size differs: {path}")


def finite_tree(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values(): finite_tree(child)
    elif isinstance(value, list):
        for child in value: finite_tree(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError("non-finite metric")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--test-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090" or not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("one RTX 4090 is required")
    if args.seed not in (17, 29, 43):
        raise ValueError("seed differs")
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink(): raise FileExistsError(path)
    protocol_path = args.campaign_root / "authority/EXPOSED_TEST_MEASUREMENT_PROTOCOL_V1.json"
    protocol_receipt_path = args.campaign_root / "authority/EXPOSED_TEST_MEASUREMENT_PROTOCOL_V1_RECEIPT.json"
    require(protocol_path, args.protocol_sha256)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_receipt = json.loads(protocol_receipt_path.read_text(encoding="utf-8"))
    claim = protocol.get("claim_boundary", {})
    if (
        protocol.get("status") != "FROZEN_BEFORE_EXPOSED_TEST_LABEL_OPENING"
        or protocol_receipt.get("protocol_sha256") != args.protocol_sha256
        or protocol.get("execution", {}).get("evaluator_sha256") != sha256_file(Path(__file__))
        or claim.get("fresh_blind_test") is not False
        or claim.get("old_test_previously_exposed") is not True
        or claim.get("selection_or_tuning_from_this_measurement") is not False
        or claim.get("cumulative_test_access_ordinal") != 3
    ):
        raise ValueError("measurement protocol differs")
    component = protocol["components"][args.arm][str(args.seed)]
    if Path(component["source_root"]).resolve() != args.source_root.resolve():
        raise ValueError("component source root differs")
    verify_source(args.source_root, args.source_manifest, component["source_manifest_sha256"])
    require(args.source_root / "scripts/train_framecontact.py", component["trainer_sha256"])
    require(args.source_root / "src/interaction_first/framecontact.py", component["framecontact_sha256"])
    require(Path(component["checkpoint"]), component["checkpoint_sha256"])
    structure_manifest = args.structure_root / "STRUCTURE_CACHE_MANIFEST.json"
    contact_manifest = args.contact_root / "CONTACT_CACHE_MANIFEST.json"
    embedding_manifest = args.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
    for path, expected in (
        (structure_manifest, protocol["data"]["structure_manifest_sha256"]),
        (contact_manifest, protocol["data"]["contact_manifest_sha256"]),
        (embedding_manifest, protocol["data"]["embedding_manifest_sha256"]),
        (args.split, protocol["data"]["split_sha256"]),
        (args.test_panel, protocol["data"]["test_panel_sha256"]),
    ):
        require(path, expected)

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root / "src")); sys.path.insert(0, str(source_root / "scripts"))
    import train_framecontact as training
    from interaction_first.sabdab2_contacts import Sabdab2ContactDataset
    from interaction_first.sabdab2_structure import Sabdab2StructureEmbeddingDataset

    training._set_seed(args.seed)
    base = Sabdab2StructureEmbeddingDataset(
        args.structure_root, args.embedding_root, None,
        split_role="test", split_path=args.split,
        split_sha256=protocol["data"]["split_sha256"],
    )
    dataset = Sabdab2ContactDataset(base, args.contact_root)
    panel = json.loads(args.test_panel.read_text(encoding="utf-8"))
    expected_ids = {str(row["instance_id"]) for row in panel["entries"]}
    observed_ids = {str(row["instance_id"]) for row in dataset.records}
    if len(dataset) != 549 or observed_ids != expected_ids:
        raise ValueError("authenticated test cohort differs")
    loader_args = {"batch_size": 8, "workers": 6, "shuffle": False, "seed": args.seed, "pin_memory": True}
    if "persistent_workers" in inspect.signature(training._loader).parameters:
        loader_args["persistent_workers"] = True
    loader = training._loader(dataset, **loader_args)
    configuration = component["configuration"]
    model = training._model_from_configuration(dataset[0], configuration, "epiformer_style_same_protocol")
    device = torch.device("cuda"); model.to(device)
    checkpoint = torch.load(Path(component["checkpoint"]), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    state_before = training._state_sha256(model)
    metrics = training._evaluate(
        model, loader, dataset, panel, device,
        use_bf16=configuration["precision"] == "bf16", evaluation_batch_size=8,
        score_fraction=float(configuration["score_fraction"]), seed=args.seed,
        calibration_max_pairs=int(configuration["calibration_max_pairs"]),
        specificity_score_source=component["specificity_score_source"],
        evaluate_partner_retrieval=True, collect_attention=False,
        frozen_calibration=component["frozen_calibration"],
    )
    state_after = training._state_sha256(model)
    finite_tree(metrics)
    if state_after != state_before:
        raise RuntimeError("model changed during measurement")
    if metrics.get("calibration_source") != "frozen_validation" or metrics["epitope"].get("calibration_source") != "frozen_validation" or metrics["paratope"].get("calibration_source") != "frozen_validation":
        raise ValueError("test calibration was refit")
    payload = {
        "format_version": "abf30b_sabdab2_exposed_test_component_result_v1",
        "status": "PASS_ABF30B_EXPOSED_HISTORICAL_TEST_COMPONENT",
        "arm": args.arm, "seed": args.seed, "metrics": metrics,
        "audit": {
            "protocol_sha256": args.protocol_sha256,
            "checkpoint_sha256": component["checkpoint_sha256"],
            "model_state_sha256": state_after, "test_complexes": len(dataset),
            "calibration_refit_on_test": False, "model_parameters_updated": False,
            "selection_performed_on_test": False, "predicted_geometry_model_input": True,
            "experimental_bound_coordinates_as_model_input": False,
            "old_test_previously_exposed": True, "fresh_blind_test": False,
            "cumulative_test_access_ordinal": 3,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "abf30b_sabdab2_exposed_test_component_receipt_v1",
        "status": payload["status"], "arm": args.arm, "seed": args.seed,
        "result_sha256": sha256_file(args.output), "protocol_sha256": args.protocol_sha256,
        "checkpoint_sha256": component["checkpoint_sha256"],
        "selection_performed_on_test": False, "fresh_blind_test": False,
        "cumulative_test_access_ordinal": 3,
    }
    args.receipt.write_bytes(canonical_bytes(receipt))
    print(json.dumps({"status": payload["status"], "arm": args.arm, "seed": args.seed, "contact_ap": metrics["contact"]["auprc"], "epitope_ap": metrics["epitope"]["auprc"], "paratope_ap": metrics["paratope"]["auprc"], "result_sha256": receipt["result_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
