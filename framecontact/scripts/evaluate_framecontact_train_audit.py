#!/usr/bin/env python3
"""Audit train-set fit with a validation-selected FrameContact checkpoint.

This is a diagnostic evaluator, not another model-selection path.  It reuses
the checkpoint, calibration temperatures, and operating thresholds already
selected on validation, evaluates the complete authenticated ``train`` role,
and never opens a test or external role.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interaction_first.framecontact_metrics import (  # noqa: E402
    bootstrap_cluster_weighted_margin,
)
from interaction_first.reciprocal_partner_panel import (  # noqa: E402
    RECIPROCAL_PARTNER_PANEL_FORMAT,
)
from interaction_first.sabdab2_contacts import (  # noqa: E402
    Sabdab2ContactDataset,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    Sabdab2StructureEmbeddingDataset,
    canonical_json_bytes,
    sha256_file,
)
from train_framecontact import (  # noqa: E402
    EXPERIMENTS,
    _device,
    _evaluate,
    _loader,
    _model_from_configuration,
)


TRAIN_AUDIT_FORMAT = "framecontact_sabdab2_train_audit_v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    parser.add_argument("--contact-manifest-sha256", required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--embedding-manifest-sha256", required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--training-partner-panel", type=Path, required=True)
    parser.add_argument("--training-partner-panel-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_803)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def _authenticated_run(run_root: Path) -> tuple[dict, dict, dict]:
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or sha256_file(result_path) != receipt.get("result_sha256")
        or sha256_file(checkpoint_path) != receipt.get("selected_checkpoint_sha256")
    ):
        raise ValueError("Development run receipt is not eligible for train audit")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return result, checkpoint, receipt


def _authenticated_training_panel(path: Path, expected_sha256: str) -> dict:
    if sha256_file(path) != expected_sha256:
        raise ValueError("training partner panel SHA-256 differs")
    panel = json.loads(path.read_text(encoding="utf-8"))
    if (
        panel.get("format_version") != RECIPROCAL_PARTNER_PANEL_FORMAT
        or panel.get("role") != "train"
        or panel.get("authority", {}).get("task_label_columns_read") is not False
    ):
        raise ValueError("Training partner panel is not a label-free train authority")
    return panel


def _frozen_validation_calibration(development: dict) -> dict[str, float]:
    validation = development["selected_checkpoint_validation"]
    values = {
        "temperature": float(validation["validation_temperature"]),
        "paratope_temperature": float(
            validation["paratope"]["validation_temperature"]
        ),
        "epitope_temperature": float(
            validation["epitope"]["validation_temperature"]
        ),
        "contact_threshold": float(
            validation["contact"]["threshold_selected_on_validation"]
        ),
        "paratope_threshold": float(
            validation["paratope"]["threshold_selected_on_validation"]
        ),
        "epitope_threshold": float(
            validation["epitope"]["threshold_selected_on_validation"]
        ),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Frozen validation calibration contains non-finite values")
    return values


def main() -> None:
    arguments = _arguments()
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError("Refusing to overwrite train-audit output")
    if arguments.evaluation_batch_size <= 0 or arguments.bootstrap_replicates <= 0:
        raise ValueError("Evaluation batch size and bootstrap replicates must be positive")
    for path, expected, name in (
        (
            arguments.structure_cache_root / "STRUCTURE_CACHE_MANIFEST.json",
            arguments.structure_manifest_sha256,
            "structure manifest",
        ),
        (
            arguments.contact_cache_root / "CONTACT_CACHE_MANIFEST.json",
            arguments.contact_manifest_sha256,
            "contact manifest",
        ),
        (
            arguments.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json",
            arguments.embedding_manifest_sha256,
            "ESM embedding manifest",
        ),
        (arguments.split_manifest, arguments.split_manifest_sha256, "split manifest"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{name} SHA-256 differs")
    panel = _authenticated_training_panel(
        arguments.training_partner_panel,
        arguments.training_partner_panel_sha256,
    )
    development, checkpoint, run_receipt = _authenticated_run(arguments.run_root)
    authority = development.get("authority", {})
    if (
        authority.get("structure_manifest_sha256")
        != arguments.structure_manifest_sha256
        or authority.get("contact_manifest_sha256")
        != arguments.contact_manifest_sha256
        or authority.get("embedding_manifest_sha256")
        != arguments.embedding_manifest_sha256
        or authority.get("split_manifest_sha256")
        != arguments.split_manifest_sha256
        or authority.get("test_loaded") is not False
    ):
        raise ValueError("Development run and train-audit authorities differ")

    device = _device(arguments.device)
    use_bf16 = arguments.precision == "bf16" and device.type == "cuda"
    if arguments.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 requires CUDA")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise ValueError("CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG")
    seed = int(development["configuration"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    base = Sabdab2StructureEmbeddingDataset(
        arguments.structure_cache_root,
        arguments.embedding_root,
        arguments.label_root,
        split_role="train",
        split_path=arguments.split_manifest,
        split_sha256=arguments.split_manifest_sha256,
    )
    dataset = Sabdab2ContactDataset(base, arguments.contact_cache_root)
    loader = _loader(
        dataset,
        batch_size=arguments.evaluation_batch_size,
        workers=arguments.workers,
        shuffle=False,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    configuration = development["configuration"]
    experiment = str(development["experiment"])
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown development experiment")
    model = _model_from_configuration(dataset[0], configuration, experiment)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    frozen_calibration = _frozen_validation_calibration(development)
    metrics = _evaluate(
        model,
        loader,
        dataset,
        panel,
        device,
        use_bf16=use_bf16,
        evaluation_batch_size=arguments.evaluation_batch_size,
        score_fraction=float(configuration["score_fraction"]),
        seed=seed,
        calibration_max_pairs=int(configuration["calibration_max_pairs"]),
        specificity_score_source=str(checkpoint["specificity_score_source"]),
        collect_attention=False,
        frozen_calibration=frozen_calibration,
        compute_calibration_regression=True,
    )
    for offset, direction_metrics in enumerate(
        metrics["partner_retrieval_by_direction"].values()
    ):
        direction_metrics["cluster_bootstrap_ci"] = bootstrap_cluster_weighted_margin(
            direction_metrics["cluster_margins"],
            replicates=arguments.bootstrap_replicates,
            seed=arguments.bootstrap_seed + offset,
        )
    payload = {
        "format_version": TRAIN_AUDIT_FORMAT,
        "claim_boundary": (
            "Post-hoc complete-train diagnostic using a validation-selected checkpoint, "
            "validation-frozen calibration and thresholds. Train metrics never select a "
            "checkpoint; test and external roles are not opened."
        ),
        "experiment": experiment,
        "seed": seed,
        "authority": {
            "development_result_sha256": run_receipt["result_sha256"],
            "selected_checkpoint_sha256": run_receipt[
                "selected_checkpoint_sha256"
            ],
            "structure_manifest_sha256": arguments.structure_manifest_sha256,
            "contact_manifest_sha256": arguments.contact_manifest_sha256,
            "embedding_manifest_sha256": arguments.embedding_manifest_sha256,
            "split_manifest_sha256": arguments.split_manifest_sha256,
            "training_partner_panel_sha256": (
                arguments.training_partner_panel_sha256
            ),
            "validation_calibration_frozen": True,
            "selection_performed_on_train": False,
            "test_loaded": False,
            "external_test_loaded": False,
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
        },
        "train_complexes": len(dataset),
        "metrics": metrics,
    }
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.tmp-",
            dir=arguments.output_root.parent,
        )
    )
    try:
        result_path = temporary / "FRAMECONTACT_TRAIN_AUDIT_RESULT.json"
        result_path.write_bytes(canonical_json_bytes(payload))
        receipt = {
            "format_version": TRAIN_AUDIT_FORMAT,
            "result_sha256": sha256_file(result_path),
            "selection_performed_on_train": False,
            "test_loaded": False,
            "external_test_loaded": False,
        }
        (temporary / "TRAIN_AUDIT_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, arguments.output_root)
        print(
            json.dumps(
                {**receipt, "output_root": str(arguments.output_root)},
                sort_keys=True,
            )
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
