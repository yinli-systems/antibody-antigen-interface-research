#!/usr/bin/env python3
"""Run causal geometry controls on a validation-selected FrameContact checkpoint.

The checkpoint, temperatures, and thresholds are frozen by the development
run.  This diagnostic evaluates only the authenticated validation role and
cannot select a checkpoint or open test/external data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(REPOSITORY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from evaluate_framecontact_train_audit import (  # noqa: E402
    _authenticated_run,
    _frozen_validation_calibration,
)
from interaction_first.framecontact_interventions import (  # noqa: E402
    corrupt_local_frames,
    degree_matched_graph_rewiring,
    geometry_shuffle,
    independent_monomer_rigid_transform,
    perturb_coordinates,
)
from interaction_first.reciprocal_partner_panel import (  # noqa: E402
    RECIPROCAL_PARTNER_PANEL_FORMAT,
)
from interaction_first.sabdab2_contacts import Sabdab2ContactDataset  # noqa: E402
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


FORMAT = "framecontact_validation_intervention_audit_v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--intervention-seed", type=int, default=20_260_803)
    parser.add_argument(
        "--coordinate-noise",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50),
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def _authenticated_validation_panel(path: Path, expected_sha256: str) -> dict:
    if sha256_file(path) != expected_sha256:
        raise ValueError("validation partner panel SHA-256 differs")
    panel = json.loads(path.read_text(encoding="utf-8"))
    if (
        panel.get("format_version") != RECIPROCAL_PARTNER_PANEL_FORMAT
        or panel.get("role") != "validation"
        or panel.get("authority", {}).get("task_label_columns_read") is not False
    ):
        raise ValueError("Partner panel is not a label-free validation authority")
    return panel


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device="cuda" if device.type == "cuda" else "cpu").manual_seed(seed)


def _metric_snapshot(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {
        "selection_metric": float(metrics["selection_metric"]),
        "contact_auprc": float(metrics["contact"]["auprc"]),
        "contact_mean_per_complex_auprc": float(
            metrics["contact"]["mean_per_complex_auprc"]
        ),
        "contact_precision": float(metrics["contact"]["precision"]),
        "contact_recall": float(metrics["contact"]["recall"]),
        "epitope_auprc": float(metrics["epitope"]["auprc"]),
        "paratope_auprc": float(metrics["paratope"]["auprc"]),
    }
    for direction, values in metrics["partner_retrieval_by_direction"].items():
        result[f"{direction}_margin"] = float(
            values["cluster_weighted_cognate_compatibility_margin"]
        )
        result[f"{direction}_top1"] = float(values["top1_retrieval_accuracy"])
        result[f"{direction}_mrr"] = float(values["mean_reciprocal_rank"])
    return result


def _delta(clean: Mapping[str, Any], changed: Mapping[str, Any]) -> dict[str, float]:
    left = _metric_snapshot(clean)
    right = _metric_snapshot(changed)
    if set(left) != set(right):
        raise ValueError("Clean and intervention metric schemas differ")
    return {key: right[key] - left[key] for key in sorted(left)}


def main() -> None:
    arguments = _arguments()
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError("Refusing to overwrite validation-intervention output")
    if arguments.evaluation_batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    if not arguments.coordinate_noise or any(value < 0 for value in arguments.coordinate_noise):
        raise ValueError("coordinate-noise values must be nonnegative")
    for path, expected, name in (
        (arguments.structure_cache_root / "STRUCTURE_CACHE_MANIFEST.json", arguments.structure_manifest_sha256, "structure"),
        (arguments.contact_cache_root / "CONTACT_CACHE_MANIFEST.json", arguments.contact_manifest_sha256, "contact"),
        (arguments.embedding_root / "SABDAB2_ESM_EMBEDDING_MANIFEST.json", arguments.embedding_manifest_sha256, "embedding"),
        (arguments.split_manifest, arguments.split_manifest_sha256, "split"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{name} manifest SHA-256 differs")

    panel = _authenticated_validation_panel(
        arguments.validation_partner_panel,
        arguments.validation_partner_panel_sha256,
    )
    development, checkpoint, run_receipt = _authenticated_run(arguments.run_root)
    authority = development.get("authority", {})
    if (
        authority.get("structure_manifest_sha256") != arguments.structure_manifest_sha256
        or authority.get("contact_manifest_sha256") != arguments.contact_manifest_sha256
        or authority.get("embedding_manifest_sha256") != arguments.embedding_manifest_sha256
        or authority.get("split_manifest_sha256") != arguments.split_manifest_sha256
        or authority.get("test_loaded") is not False
    ):
        raise ValueError("Development and intervention authorities differ")

    device = _device(arguments.device)
    use_bf16 = arguments.precision == "bf16" and device.type == "cuda"
    if arguments.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 requires CUDA")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
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
    frozen = _frozen_validation_calibration(development)

    def evaluate(batch_intervention: Callable[[dict[str, Any]], dict[str, Any]] | None = None, *, local_seed: int = seed) -> dict:
        return _evaluate(
            model,
            loader,
            dataset,
            panel,
            device,
            use_bf16=use_bf16,
            evaluation_batch_size=arguments.evaluation_batch_size,
            score_fraction=float(configuration["score_fraction"]),
            seed=local_seed,
            calibration_max_pairs=int(configuration["calibration_max_pairs"]),
            specificity_score_source=str(checkpoint["specificity_score_source"]),
            collect_attention=False,
            frozen_calibration=frozen,
            compute_calibration_regression=False,
            batch_intervention=batch_intervention,
        )

    clean = evaluate()
    factories: list[tuple[str, Callable[[torch.Generator], Callable[[dict[str, Any]], dict[str, Any]]]]] = [
        ("independent_monomer_rigid_transform", lambda generator: lambda batch: independent_monomer_rigid_transform(batch, generator=generator)),
        ("geometry_shuffle", lambda generator: lambda batch: geometry_shuffle(batch, generator=generator)),
        ("local_frame_corruption", lambda generator: lambda batch: corrupt_local_frames(batch, generator=generator)),
        ("degree_matched_graph_rewiring", lambda generator: lambda batch: degree_matched_graph_rewiring(batch, generator=generator, neighbor_k=int(configuration["neighbor_k"]), neighbor_radius=float(configuration["neighbor_radius"]))),
    ]
    factories.extend(
        (
            f"coordinate_noise_{standard_deviation:g}_angstrom",
            lambda generator, standard_deviation=standard_deviation: lambda batch: perturb_coordinates(batch, generator=generator, standard_deviation=standard_deviation),
        )
        for standard_deviation in arguments.coordinate_noise
    )
    interventions = {}
    for index, (name, factory) in enumerate(factories):
        local_seed = arguments.intervention_seed + index
        changed = evaluate(factory(_generator(device, local_seed)), local_seed=local_seed)
        interventions[name] = {"metrics": changed, "delta_vs_clean": _delta(clean, changed)}

    payload = {
        "format_version": FORMAT,
        "claim_boundary": "Post-selection causal diagnostic on authenticated validation only; no checkpoint, calibration, or intervention was selected on test data.",
        "experiment": experiment,
        "seed": seed,
        "authority": {
            "development_result_sha256": run_receipt["result_sha256"],
            "selected_checkpoint_sha256": run_receipt["selected_checkpoint_sha256"],
            "validation_partner_panel_sha256": arguments.validation_partner_panel_sha256,
            "validation_calibration_frozen": True,
            "selection_performed_by_this_audit": False,
            "test_loaded": False,
            "external_test_loaded": False,
        },
        "runtime": {"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "torch": torch.__version__},
        "validation_complexes": len(dataset),
        "clean_metrics": clean,
        "interventions": interventions,
    }
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{arguments.output_root.name}.tmp-", dir=arguments.output_root.parent))
    try:
        result_path = temporary / "FRAMECONTACT_VALIDATION_INTERVENTIONS.json"
        result_path.write_bytes(canonical_json_bytes(payload))
        receipt = {
            "format_version": FORMAT,
            "result_sha256": sha256_file(result_path),
            "selection_performed_by_this_audit": False,
            "test_loaded": False,
            "external_test_loaded": False,
        }
        (temporary / "VALIDATION_INTERVENTION_RECEIPT.json").write_bytes(canonical_json_bytes(receipt))
        os.replace(temporary, arguments.output_root)
        print(json.dumps({**receipt, "output_root": str(arguments.output_root)}, sort_keys=True))
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
