#!/usr/bin/env python3
"""Evaluate one precommitted component in an exposed historical SAbDab2 batch."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import torch


ARMS = ("baseline_current", "edge_topk2_supervised")


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


def require_sha(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 differs: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--test-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if os.environ.get("SLURM_JOB_PARTITION") != "gpu_4090":
        raise RuntimeError("gpu_4090 is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
        raise RuntimeError("RTX 4090 is required")
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    authority = args.campaign_root / "authority"
    protocol_path = authority / "PROTOCOL_TASK_ROUTED_EXPOSED_TEST_V1.json"
    protocol_receipt_path = (
        authority / "PROTOCOL_TASK_ROUTED_EXPOSED_TEST_V1_RECEIPT.json"
    )
    require_sha(protocol_path, args.protocol_sha256, "protocol")
    protocol = json.loads(protocol_path.read_text())
    protocol_receipt = json.loads(protocol_receipt_path.read_text())
    claim = protocol.get("claim_boundary", {})
    execution = protocol.get("execution", {})
    if (
        protocol.get("status") != "FROZEN_BEFORE_EXPOSED_HISTORICAL_BATCH"
        or protocol_receipt.get("protocol_sha256") != args.protocol_sha256
        or claim.get("fresh_blind_test") is not False
        or claim.get("historical_benchmark_only") is not True
        or claim.get("old_test_previously_exposed") is not True
        or claim.get("selection_performed_on_this_measurement") is not False
        or claim.get("result_may_not_select_checkpoint_or_model") is not True
        or claim.get("cumulative_test_access_ordinal") != 2
        or execution.get("evaluator_sha256") != sha256_file(Path(__file__))
        or execution.get("component_arms") != list(ARMS)
        or execution.get("clean_metrics_only") is not True
    ):
        raise ValueError("exposed historical protocol differs")

    component = protocol["components"][args.arm]
    if args.source_root.resolve() != Path(component["source_root"]).resolve():
        raise ValueError("source root differs from frozen component")
    for path, expected, label in (
        (
            Path(component["source_manifest"]),
            component["source_manifest_sha256"],
            "source manifest",
        ),
        (
            args.source_root / "scripts/train_framecontact.py",
            component["trainer_sha256"],
            "trainer",
        ),
        (
            args.source_root / "src/interaction_first/framecontact.py",
            component["framecontact_sha256"],
            "FrameContact source",
        ),
        (Path(component["checkpoint"]), component["checkpoint_sha256"], "checkpoint"),
        (
            args.structure_manifest,
            protocol["data"]["structure_manifest_sha256"],
            "structure manifest",
        ),
        (
            args.contact_manifest,
            protocol["data"]["contact_manifest_sha256"],
            "contact manifest",
        ),
        (
            args.embedding_manifest,
            protocol["data"]["embedding_manifest_sha256"],
            "embedding manifest",
        ),
        (
            args.split_manifest,
            protocol["data"]["split_manifest_sha256"],
            "split manifest",
        ),
        (
            args.test_panel,
            protocol["data"]["test_partner_panel_sha256"],
            "test partner panel",
        ),
    ):
        require_sha(path, str(expected), label)
    for key in ("development_result", "development_receipt"):
        if key in component:
            require_sha(
                Path(component[key]),
                component[f"{key}_sha256"],
                key.replace("_", " "),
            )

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root / "scripts"))
    import train_framecontact as training  # noqa: PLC0415
    from interaction_first.sabdab2_contacts import Sabdab2ContactDataset  # noqa: PLC0415
    from interaction_first.sabdab2_structure import (  # noqa: PLC0415
        Sabdab2StructureEmbeddingDataset,
    )

    seed = int(protocol["seed"])
    configuration = component["configuration"]
    training._set_seed(seed)
    base = Sabdab2StructureEmbeddingDataset(
        args.structure_cache_root,
        args.embedding_root,
        None,
        split_role="test",
        split_path=args.split_manifest,
        split_sha256=protocol["data"]["split_manifest_sha256"],
    )
    dataset = Sabdab2ContactDataset(base, args.contact_cache_root)
    panel = json.loads(args.test_panel.read_text())
    expected_ids = {str(entry["instance_id"]) for entry in panel["entries"]}
    observed_ids = {str(record["instance_id"]) for record in dataset.records}
    if (
        len(dataset) != int(protocol["data"]["test_complex_count"])
        or observed_ids != expected_ids
    ):
        raise ValueError("authenticated test dataset and panel differ")

    loader_arguments = {
        "batch_size": args.evaluation_batch_size,
        "workers": args.workers,
        "shuffle": False,
        "seed": seed,
        "pin_memory": True,
    }
    if "persistent_workers" in inspect.signature(training._loader).parameters:
        loader_arguments["persistent_workers"] = args.workers > 0
    loader = training._loader(dataset, **loader_arguments)
    model = training._model_from_configuration(
        dataset[0], configuration, "epiformer_style_same_protocol"
    )
    device = torch.device("cuda")
    model.to(device)
    checkpoint = torch.load(
        Path(component["checkpoint"]), map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    state_before = training._state_sha256(model)
    metrics = training._evaluate(
        model,
        loader,
        dataset,
        panel,
        device,
        use_bf16=configuration["precision"] == "bf16",
        evaluation_batch_size=args.evaluation_batch_size,
        score_fraction=float(configuration["score_fraction"]),
        seed=seed,
        calibration_max_pairs=int(configuration["calibration_max_pairs"]),
        specificity_score_source=component["specificity_score_source"],
        evaluate_partner_retrieval=True,
        collect_attention=False,
        frozen_calibration=component["frozen_calibration"],
    )
    state_after = training._state_sha256(model)
    if state_after != state_before:
        raise RuntimeError("model parameters changed during test measurement")
    if (
        metrics.get("calibration_source") != "frozen_validation"
        or metrics["epitope"].get("calibration_source") != "frozen_validation"
        or metrics["paratope"].get("calibration_source") != "frozen_validation"
    ):
        raise ValueError("test calibration was refit")

    payload = {
        "format_version": "sabdab2_task_routed_exposed_component_result_v1",
        "status": "PASS_EXPOSED_HISTORICAL_COMPONENT",
        "arm": args.arm,
        "seed": seed,
        "metrics": metrics,
        "audit": {
            "protocol_sha256": args.protocol_sha256,
            "checkpoint_sha256": component["checkpoint_sha256"],
            "model_state_sha256": state_after,
            "test_complexes": len(dataset),
            "calibration_refit_on_test": False,
            "model_parameters_updated": False,
            "selection_performed_on_this_measurement": False,
            "old_test_previously_exposed": True,
            "fresh_blind_test": False,
            "historical_benchmark_only": True,
            "cumulative_test_access_ordinal": 2,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_bytes(canonical_bytes(payload))
    receipt = {
        "format_version": "sabdab2_task_routed_exposed_component_receipt_v1",
        "status": payload["status"],
        "arm": args.arm,
        "seed": seed,
        "result_sha256": sha256_file(args.output),
        "protocol_sha256": args.protocol_sha256,
        "checkpoint_sha256": component["checkpoint_sha256"],
        "selection_performed_on_this_measurement": False,
        "old_test_previously_exposed": True,
        "fresh_blind_test": False,
        "historical_benchmark_only": True,
        "cumulative_test_access_ordinal": 2,
    }
    args.receipt.write_bytes(canonical_bytes(receipt))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "arm": args.arm,
                "contact_auprc": metrics["contact"]["auprc"],
                "epitope_auprc": metrics["epitope"]["auprc"],
                "paratope_auprc": metrics["paratope"]["auprc"],
                "result_sha256": receipt["result_sha256"],
                "receipt_sha256": sha256_file(args.receipt),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
