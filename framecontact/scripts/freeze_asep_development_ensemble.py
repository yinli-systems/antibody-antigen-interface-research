#!/usr/bin/env python3
"""Freeze one five-seed AsEP ensemble using development validation only."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from interaction_first.asep_external import AsepDevelopmentDataset  # noqa: E402
from interaction_first.framecontact_ensemble import (  # noqa: E402
    CalibratedProbabilityEnsemble,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)
from train_framecontact import (  # noqa: E402
    EXPERIMENTS,
    RECIPROCAL_PARTNER_PANEL_FORMAT,
    RESULT_FORMAT,
    _authenticated_panel,
    _device,
    _evaluate,
    _loader,
    _model_from_configuration,
)


ENSEMBLE_FORMAT = "framecontact_asep_development_ensemble_v1"
FROZEN_SEEDS = (17, 42, 73, 101, 137)
COMPATIBILITY_KEYS = (
    "dataset_backend",
    "asep_feature_mode",
    "d_model",
    "num_heads",
    "num_self_layers",
    "num_rbf",
    "neighbor_k",
    "neighbor_radius",
    "sequence_window",
    "dropout",
    "score_fraction",
    "calibration_max_pairs",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", type=Path, nargs=5, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--protocol-receipt-sha256", required=True)
    parser.add_argument("--training-partner-panel", type=Path, required=True)
    parser.add_argument("--training-partner-panel-sha256", required=True)
    parser.add_argument("--validation-partner-panel", type=Path, required=True)
    parser.add_argument("--validation-partner-panel-sha256", required=True)
    parser.add_argument("--panel-size", type=int, default=31)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def authenticated_development_run(run_root: Path) -> dict[str, Any]:
    """Authenticate one development-only run and its selected checkpoint."""
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
    for path, name in (
        (result_path, "development result"),
        (checkpoint_path, "selected checkpoint"),
        (receipt_path, "run receipt"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing regular {name}: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("format_version") != RESULT_FORMAT
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or receipt.get("result_sha256") != sha256_file(result_path)
        or receipt.get("selected_checkpoint_sha256") != sha256_file(checkpoint_path)
    ):
        raise ValueError(f"Development run receipt failed: {run_root}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    authority = result.get("authority", {})
    if (
        result.get("format_version") != RESULT_FORMAT
        or authority.get("dataset_backend") != "asep"
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
        or authority.get("structure_manifest_sha256") is not None
        or authority.get("contact_manifest_sha256") is not None
        or authority.get("split_manifest_sha256") is not None
    ):
        raise ValueError(f"Development result is not AsEP-ensemble eligible: {run_root}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    seed = result.get("configuration", {}).get("seed")
    if (
        checkpoint.get("format_version") != RESULT_FORMAT
        or checkpoint.get("seed") != seed
        or receipt.get("seed") != seed
        or checkpoint.get("experiment") != result.get("experiment")
        or receipt.get("selected_epoch") != checkpoint.get("selected_epoch")
    ):
        raise ValueError(f"Checkpoint metadata differs from result: {run_root}")
    return {
        "root": run_root,
        "result": result,
        "checkpoint": checkpoint,
        "result_sha256": receipt["result_sha256"],
        "checkpoint_sha256": receipt["selected_checkpoint_sha256"],
    }


def member_calibration(result: dict[str, Any]) -> dict[str, float]:
    validation = result["selected_checkpoint_validation"]
    calibration = {
        "contact": float(validation["validation_temperature"]),
        "paratope": float(validation["paratope"]["validation_temperature"]),
        "epitope": float(validation["epitope"]["validation_temperature"]),
    }
    if any(not math.isfinite(value) or value <= 0 for value in calibration.values()):
        raise ValueError("Member validation temperatures must be finite and positive")
    return calibration


def _authenticate_protocol(
    protocol_path: Path,
    receipt_path: Path,
    receipt_sha256: str,
    cache_manifest_sha256: str,
) -> dict[str, Any]:
    authenticate_file(receipt_path, receipt_sha256, "AsEP protocol receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    authenticate_file(
        protocol_path,
        str(receipt.get("protocol_file_sha256")),
        "AsEP protocol",
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version")
        != "framecontact_asep_epitope_group_protocol_v1"
        or protocol.get("data", {}).get("dataset_backend") != "asep"
        or protocol.get("data", {}).get("development_cache_manifest_sha256")
        != cache_manifest_sha256
        or tuple(protocol.get("model_selection", {}).get("seeds", []))
        != FROZEN_SEEDS
        or protocol.get("model_selection", {}).get("test_role_forbidden") is not True
    ):
        raise ValueError("Protocol is not eligible for an AsEP development ensemble")
    return protocol


def _validate_members(
    members: list[dict[str, Any]],
    *,
    cache_manifest_sha256: str,
    protocol_sha256: str,
    protocol_receipt_sha256: str,
    training_panel_sha256: str,
    validation_panel_sha256: str,
) -> tuple[str, dict[str, Any], tuple[int, ...]]:
    members.sort(key=lambda item: int(item["checkpoint"]["seed"]))
    seeds = tuple(int(item["checkpoint"]["seed"]) for item in members)
    if seeds != FROZEN_SEEDS:
        raise ValueError("Ensemble requires exactly the five pre-registered seeds")
    experiments = {str(item["result"]["experiment"]) for item in members}
    if len(experiments) != 1:
        raise ValueError("Ensemble members must use one experiment")
    experiment = next(iter(experiments))
    if experiment not in EXPERIMENTS:
        raise ValueError("Ensemble experiment is unknown")
    common_authority = members[0]["result"]["authority"]
    if any(item["result"]["authority"] != common_authority for item in members[1:]):
        raise ValueError("Ensemble member authorities differ")
    expected_authority = {
        "asep_cache_manifest_sha256": cache_manifest_sha256,
        "protocol_file_sha256": protocol_sha256,
        "protocol_receipt_sha256": protocol_receipt_sha256,
        "training_partner_panel_sha256": training_panel_sha256,
        "validation_partner_panel_sha256": validation_panel_sha256,
        "dataset_backend": "asep",
        "test_loaded": False,
        "external_test_loaded": False,
    }
    if any(common_authority.get(key) != value for key, value in expected_authority.items()):
        raise ValueError("Ensemble member and requested authorities differ")
    reference = members[0]["result"]["configuration"]
    for key in COMPATIBILITY_KEYS:
        if key not in reference:
            raise ValueError(f"Missing ensemble compatibility field: {key}")
    for item in members[1:]:
        configuration = item["result"]["configuration"]
        if any(configuration.get(key) != reference[key] for key in COMPATIBILITY_KEYS):
            raise ValueError("Ensemble member model/evaluation configurations differ")
    return experiment, reference, seeds


def main() -> None:
    arguments = _arguments()
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError("Refusing to overwrite a frozen ensemble")
    if arguments.panel_size <= 0 or arguments.evaluation_batch_size <= 0:
        raise ValueError("Panel and batch sizes must be positive")
    manifest_path = arguments.cache_root / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"
    authenticate_file(
        manifest_path,
        arguments.cache_manifest_sha256,
        "AsEP development cache manifest",
    )
    protocol = _authenticate_protocol(
        arguments.protocol,
        arguments.protocol_receipt,
        arguments.protocol_receipt_sha256,
        arguments.cache_manifest_sha256,
    )
    panel_protocol = protocol["data"]["reciprocal_partner_panel"]
    if (
        panel_protocol.get("format_version") != RECIPROCAL_PARTNER_PANEL_FORMAT
        or int(panel_protocol.get("size_per_query", -1)) != arguments.panel_size
        or panel_protocol.get("train_panel_sha256")
        != arguments.training_partner_panel_sha256
        or panel_protocol.get("validation_panel_sha256")
        != arguments.validation_partner_panel_sha256
    ):
        raise ValueError("Requested partner panels differ from the protocol")
    _authenticated_panel(
        arguments.training_partner_panel,
        arguments.training_partner_panel_sha256,
        role="train",
        expected_size=arguments.panel_size,
        expected_format=RECIPROCAL_PARTNER_PANEL_FORMAT,
    )
    validation_panel = _authenticated_panel(
        arguments.validation_partner_panel,
        arguments.validation_partner_panel_sha256,
        role="validation",
        expected_size=arguments.panel_size,
        expected_format=RECIPROCAL_PARTNER_PANEL_FORMAT,
    )
    members = [authenticated_development_run(path) for path in arguments.run_roots]
    experiment, configuration, seeds = _validate_members(
        members,
        cache_manifest_sha256=arguments.cache_manifest_sha256,
        protocol_sha256=sha256_file(arguments.protocol),
        protocol_receipt_sha256=arguments.protocol_receipt_sha256,
        training_panel_sha256=arguments.training_partner_panel_sha256,
        validation_panel_sha256=arguments.validation_partner_panel_sha256,
    )

    device = _device(arguments.device)
    use_bf16 = arguments.precision == "bf16" and device.type == "cuda"
    if arguments.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 requires CUDA")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise ValueError("CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG")
    dataset = AsepDevelopmentDataset(
        arguments.cache_root,
        manifest_sha256=arguments.cache_manifest_sha256,
        split_role="val",
        feature_mode=str(configuration["asep_feature_mode"]),
    )
    loader = _loader(
        dataset,
        batch_size=arguments.evaluation_batch_size,
        workers=arguments.workers,
        shuffle=False,
        seed=20_260_730,
        pin_memory=device.type == "cuda",
    )
    first = dataset[0]
    models = []
    calibrations = []
    for member in members:
        model = _model_from_configuration(first, member["result"]["configuration"], experiment)
        model.load_state_dict(member["checkpoint"]["model_state_dict"])
        model.to(device).eval()
        models.append(model)
        calibrations.append(member_calibration(member["result"]))
    ensemble = CalibratedProbabilityEnsemble(models, calibrations).to(device).eval()
    specificity_sources = {
        str(member["checkpoint"]["specificity_score_source"]) for member in members
    }
    if len(specificity_sources) != 1:
        raise ValueError("Ensemble member specificity score sources differ")
    specificity_source = next(iter(specificity_sources))
    validation = _evaluate(
        ensemble,
        loader,
        dataset,
        validation_panel,
        device,
        use_bf16=use_bf16,
        evaluation_batch_size=arguments.evaluation_batch_size,
        score_fraction=float(configuration["score_fraction"]),
        seed=20_260_730,
        calibration_max_pairs=int(configuration["calibration_max_pairs"]),
        specificity_score_source=specificity_source,
        collect_attention=False,
    )
    payload = {
        "format_version": ENSEMBLE_FORMAT,
        "claim_boundary": (
            "Five-seed ensemble frozen on AsEP development validation only. Official "
            "test tensors and labels were not loaded, and no benchmark claim is authorized."
        ),
        "experiment": experiment,
        "seeds": list(seeds),
        "ensemble_rule": {
            "contact_and_residue": (
                "apply each member's validation temperature, average elementwise "
                "probabilities, convert to logits, then fit one ensemble temperature "
                "and thresholds on development validation"
            ),
            "interaction_only": (
                "average elementwise sigmoid probabilities of raw interaction-only logits"
            ),
            "member_weighting": "uniform 1/5",
            "checkpoint_rule": "each member selected under the pre-registered validation rule",
        },
        "authority": {
            "dataset_backend": "asep",
            "asep_cache_manifest_sha256": arguments.cache_manifest_sha256,
            "protocol_file_sha256": sha256_file(arguments.protocol),
            "protocol_receipt_sha256": arguments.protocol_receipt_sha256,
            "training_partner_panel_sha256": arguments.training_partner_panel_sha256,
            "validation_partner_panel_sha256": arguments.validation_partner_panel_sha256,
            "test_loaded": False,
            "external_test_loaded": False,
            "members": [
                {
                    "seed": int(member["checkpoint"]["seed"]),
                    "development_result_sha256": member["result_sha256"],
                    "selected_checkpoint_sha256": member["checkpoint_sha256"],
                }
                for member in members
            ],
        },
        "configuration": {
            key: configuration[key] for key in COMPATIBILITY_KEYS
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
        },
        "validation_complexes": len(dataset),
        "validation": validation,
    }
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.tmp-",
            dir=arguments.output_root.parent,
        )
    )
    try:
        result_path = temporary / "FRAMECONTACT_ASEP_DEVELOPMENT_ENSEMBLE.json"
        result_path.write_bytes(canonical_json_bytes(payload))
        receipt = {
            "format_version": ENSEMBLE_FORMAT,
            "result_sha256": sha256_file(result_path),
            "test_loaded": False,
            "external_test_loaded": False,
        }
        (temporary / "ENSEMBLE_RECEIPT.json").write_bytes(canonical_json_bytes(receipt))
        os.replace(temporary, arguments.output_root)
        print(json.dumps({**receipt, "output_root": str(arguments.output_root)}, sort_keys=True))
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
