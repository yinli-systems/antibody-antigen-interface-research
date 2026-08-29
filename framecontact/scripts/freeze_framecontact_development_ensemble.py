#!/usr/bin/env python3
"""Freeze a five-seed FrameContact ensemble using validation data only."""

from __future__ import annotations

import argparse
import json
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

from interaction_first.framecontact_ensemble import (  # noqa: E402
    CalibratedProbabilityEnsemble,
)
from interaction_first.sabdab2_contacts import Sabdab2ContactDataset  # noqa: E402
from interaction_first.sabdab2_structure import (  # noqa: E402
    Sabdab2StructureEmbeddingDataset,
    canonical_json_bytes,
    sha256_file,
)
from train_framecontact import (  # noqa: E402
    RECIPROCAL_PARTNER_PANEL_FORMAT,
    RESULT_FORMAT,
    _authenticated_panel,
    _device,
    _evaluate,
    _loader,
    _model_from_configuration,
)


ENSEMBLE_FORMAT = "framecontact_development_ensemble_v1"
FROZEN_SEEDS = (17, 42, 73, 101, 137)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", type=Path, nargs=5, required=True)
    parser.add_argument("--structure-cache-root", type=Path, required=True)
    parser.add_argument("--structure-manifest-sha256", required=True)
    parser.add_argument("--contact-cache-root", type=Path, required=True)
    parser.add_argument("--contact-manifest-sha256", required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--embedding-manifest-sha256", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest-sha256", required=True)
    parser.add_argument("--validation-partner-panel", type=Path, required=True)
    parser.add_argument("--validation-partner-panel-sha256", required=True)
    parser.add_argument("--panel-size", type=int, default=31)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def _authenticated_run(run_root: Path) -> dict[str, Any]:
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
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
    if (
        result.get("format_version") != RESULT_FORMAT
        or result.get("authority", {}).get("test_loaded") is not False
        or result.get("authority", {}).get("external_test_loaded") is not False
    ):
        raise ValueError(f"Development result is not ensemble eligible: {run_root}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("format_version") != RESULT_FORMAT
        or checkpoint.get("seed") != result.get("configuration", {}).get("seed")
        or checkpoint.get("experiment") != result.get("experiment")
    ):
        raise ValueError(f"Checkpoint metadata differs from result: {run_root}")
    return {
        "root": run_root,
        "result": result,
        "checkpoint": checkpoint,
        "result_sha256": receipt["result_sha256"],
        "checkpoint_sha256": receipt["selected_checkpoint_sha256"],
    }


def _member_calibration(result: dict[str, Any]) -> dict[str, float]:
    validation = result["selected_checkpoint_validation"]
    return {
        "contact": float(validation["validation_temperature"]),
        "paratope": float(validation["paratope"]["validation_temperature"]),
        "epitope": float(validation["epitope"]["validation_temperature"]),
    }


def main() -> None:
    arguments = _arguments()
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError("Refusing to overwrite a frozen ensemble")
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
    panel = _authenticated_panel(
        arguments.validation_partner_panel,
        arguments.validation_partner_panel_sha256,
        role="validation",
        expected_size=arguments.panel_size,
        expected_format=RECIPROCAL_PARTNER_PANEL_FORMAT,
    )
    members = [_authenticated_run(path) for path in arguments.run_roots]
    members.sort(key=lambda member: int(member["checkpoint"]["seed"]))
    seeds = tuple(int(member["checkpoint"]["seed"]) for member in members)
    if seeds != FROZEN_SEEDS:
        raise ValueError("Ensemble requires the five pre-registered seeds")
    experiments = {str(member["result"]["experiment"]) for member in members}
    if len(experiments) != 1:
        raise ValueError("Ensemble members must use one experiment")
    experiment = next(iter(experiments))
    common_authority = members[0]["result"]["authority"]
    if any(member["result"]["authority"] != common_authority for member in members[1:]):
        raise ValueError("Ensemble member authorities differ")
    if (
        common_authority.get("structure_manifest_sha256")
        != arguments.structure_manifest_sha256
        or common_authority.get("contact_manifest_sha256")
        != arguments.contact_manifest_sha256
        or common_authority.get("embedding_manifest_sha256")
        != arguments.embedding_manifest_sha256
        or common_authority.get("split_manifest_sha256")
        != arguments.split_manifest_sha256
        or common_authority.get("validation_partner_panel_sha256")
        != arguments.validation_partner_panel_sha256
    ):
        raise ValueError("Ensemble member and requested authorities differ")
    compatibility_keys = (
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
    reference_configuration = members[0]["result"]["configuration"]
    for member in members[1:]:
        if any(
            member["result"]["configuration"][key]
            != reference_configuration[key]
            for key in compatibility_keys
        ):
            raise ValueError("Ensemble member model/evaluation configurations differ")

    device = _device(arguments.device)
    use_bf16 = arguments.precision == "bf16" and device.type == "cuda"
    if arguments.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 requires CUDA")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise ValueError("CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG")
    base = Sabdab2StructureEmbeddingDataset(
        arguments.structure_cache_root,
        arguments.embedding_root,
        None,
        split_role="validation",
        split_path=arguments.split_manifest,
        split_sha256=arguments.split_manifest_sha256,
    )
    dataset = Sabdab2ContactDataset(base, arguments.contact_cache_root)
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
        model = _model_from_configuration(
            first,
            member["result"]["configuration"],
            experiment,
        )
        model.load_state_dict(member["checkpoint"]["model_state_dict"])
        model.to(device).eval()
        models.append(model)
        calibrations.append(_member_calibration(member["result"]))
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
        panel,
        device,
        use_bf16=use_bf16,
        evaluation_batch_size=arguments.evaluation_batch_size,
        score_fraction=float(reference_configuration["score_fraction"]),
        seed=20_260_730,
        calibration_max_pairs=int(reference_configuration["calibration_max_pairs"]),
        specificity_score_source=specificity_source,
        collect_attention=False,
    )
    payload = {
        "format_version": ENSEMBLE_FORMAT,
        "claim_boundary": (
            "Five-seed ensemble frozen on validation only. Test and external roles were "
            "not loaded, and no test-driven selection is authorized."
        ),
        "experiment": experiment,
        "seeds": list(seeds),
        "ensemble_rule": {
            "contact_and_residue": (
                "apply each member's validation temperature, average elementwise "
                "probabilities, convert to logits, then fit one ensemble temperature on validation"
            ),
            "interaction_only": (
                "average elementwise sigmoid of raw joint-interaction logits before top-k scoring"
            ),
            "member_weighting": "uniform 1/5",
            "checkpoint_rule": "each member validation-selected under frozen protocol",
        },
        "authority": {
            "structure_manifest_sha256": arguments.structure_manifest_sha256,
            "contact_manifest_sha256": arguments.contact_manifest_sha256,
            "embedding_manifest_sha256": arguments.embedding_manifest_sha256,
            "split_manifest_sha256": arguments.split_manifest_sha256,
            "validation_partner_panel_sha256": arguments.validation_partner_panel_sha256,
            "protocol_file_sha256": common_authority["protocol_file_sha256"],
            "protocol_receipt_sha256": common_authority["protocol_receipt_sha256"],
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
        result_path = temporary / "FRAMECONTACT_DEVELOPMENT_ENSEMBLE.json"
        result_path.write_bytes(canonical_json_bytes(payload))
        receipt = {
            "format_version": ENSEMBLE_FORMAT,
            "result_sha256": sha256_file(result_path),
            "test_loaded": False,
            "external_test_loaded": False,
        }
        (temporary / "ENSEMBLE_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary, arguments.output_root)
        print(json.dumps({**receipt, "output_root": str(arguments.output_root)}, sort_keys=True))
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
