#!/usr/bin/env python3
"""Score the once-opened AsEP official test with one validation-frozen winner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from interaction_first.asep_frozen_test import (  # noqa: E402
    FROZEN_TEST_CACHE_FORMAT,
    OPENING_FORMAT,
    AsepFrozenTestDataset,
    _authenticate_official_test_opening,
    asep_evaluation_source_hashes,
    authenticate_asep_development_selection,
)
from interaction_first.framecontact_ensemble import (  # noqa: E402
    CalibratedProbabilityEnsemble,
)
from interaction_first.framecontact_interventions import (  # noqa: E402
    corrupt_local_frames,
    degree_matched_graph_rewiring,
    geometry_shuffle,
    perturb_coordinates,
)
from interaction_first.framecontact_metrics import (  # noqa: E402
    bootstrap_cluster_weighted_margin,
    bootstrap_group_weighted_values,
)
from interaction_first.reciprocal_partner_panel import (  # noqa: E402
    RECIPROCAL_PARTNER_PANEL_FORMAT,
)
from interaction_first.sabdab2_structure import (  # noqa: E402
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)
from evaluate_framecontact_sabdab2_ensemble_test import (  # noqa: E402
    _frozen_calibration,
    _generator,
    _independent_se3_prediction_invariance,
    _intervention_delta,
)
from freeze_asep_development_ensemble import (  # noqa: E402
    ENSEMBLE_FORMAT,
    FROZEN_SEEDS,
    authenticated_development_run,
    member_calibration,
)
from train_framecontact import (  # noqa: E402
    _authenticated_panel,
    _device,
    _evaluate,
    _loader,
    _model_from_configuration,
)


TEST_FORMAT = "framecontact_asep_frozen_ensemble_test_v1"
CONSUMPTION_FORMAT = "framecontact_asep_test_score_consumption_v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opening-receipt", type=Path, required=True)
    parser.add_argument("--opening-receipt-sha256", required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--selection-receipt-sha256", required=True)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--run-roots", type=Path, nargs=5, required=True)
    parser.add_argument("--test-cache-root", type=Path, required=True)
    parser.add_argument("--test-cache-manifest-sha256", required=True)
    parser.add_argument("--test-cache-receipt-sha256", required=True)
    parser.add_argument("--full-test-panel", type=Path, required=True)
    parser.add_argument("--full-test-panel-sha256", required=True)
    parser.add_argument("--clean-test-panel", type=Path, required=True)
    parser.add_argument("--clean-test-panel-sha256", required=True)
    parser.add_argument("--panel-size", type=int, default=31)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot parse {name}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _authenticated_ensemble(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = root / "FRAMECONTACT_ASEP_DEVELOPMENT_ENSEMBLE.json"
    receipt_path = root / "ENSEMBLE_RECEIPT.json"
    if any(path.is_symlink() or not path.is_file() for path in (result_path, receipt_path)):
        raise ValueError("Selected AsEP ensemble files must be regular files")
    receipt = _json(receipt_path, "AsEP ensemble receipt")
    authenticate_file(
        result_path, str(receipt.get("result_sha256", "")), "AsEP ensemble result"
    )
    result = _json(result_path, "AsEP ensemble result")
    if (
        receipt.get("format_version") != ENSEMBLE_FORMAT
        or result.get("format_version") != ENSEMBLE_FORMAT
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or result.get("authority", {}).get("test_loaded") is not False
        or result.get("authority", {}).get("external_test_loaded") is not False
        or tuple(result.get("seeds", ())) != FROZEN_SEEDS
    ):
        raise ValueError("Selected AsEP ensemble is not official-test eligible")
    return result, receipt


def _authenticate_cache(
    root: Path,
    manifest_sha256: str,
    receipt_sha256: str,
    *,
    opening_sha256: str,
    selected_ensemble_sha256: str,
    opening: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = root / "ASEP_FROZEN_TEST_CACHE_MANIFEST.json"
    receipt_path = root / "ASEP_FROZEN_TEST_CACHE_RECEIPT.json"
    authenticate_file(receipt_path, receipt_sha256, "AsEP test cache receipt")
    receipt = _json(receipt_path, "AsEP test cache receipt")
    authenticate_file(manifest_path, manifest_sha256, "AsEP test cache manifest")
    manifest = _json(manifest_path, "AsEP test cache manifest")
    graph_open_token = Path(str(opening.get("test_state_root", ""))) / str(
        opening.get("graph_open_consumption_token_name", "")
    )
    expected_graph_token_sha = manifest.get("authority", {}).get(
        "graph_open_consumption_token_sha256"
    )
    authenticate_file(
        graph_open_token,
        str(expected_graph_token_sha or ""),
        "AsEP graph-open consumption token",
    )
    if (
        receipt.get("format_version") != FROZEN_TEST_CACHE_FORMAT
        or manifest.get("format_version") != FROZEN_TEST_CACHE_FORMAT
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("test_graph_or_label_open_count") != 1
        or receipt.get("model_predictions_or_scores_read") is not False
        or receipt.get("graph_open_consumption_token_sha256")
        != expected_graph_token_sha
        or manifest.get("authority", {}).get("opening_receipt_sha256")
        != opening_sha256
        or manifest.get("authority", {}).get("selected_ensemble_result_sha256")
        != selected_ensemble_sha256
        or manifest.get("checks", {}).get(
            "official_test_graph_and_labels_opened_once"
        )
        is not True
        or manifest.get("checks", {}).get("model_predictions_or_scores_read")
        is not False
    ):
        raise ValueError("AsEP frozen test cache is ineligible")
    return manifest


def _panel_groups(panel: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    entries = panel.get("entries", [])
    return {
        str(entry["instance_id"]): {
            "antigen_cluster_id": str(entry["antigen_cluster_id"]),
            "antibody_cluster_id": str(entry["antibody_cluster_id"]),
        }
        for entry in entries
    }


def _add_bootstrap(
    metrics: dict[str, Any],
    panel: Mapping[str, Any],
    *,
    replicates: int,
    seed: int,
) -> None:
    for index, direction_metrics in enumerate(
        metrics["partner_retrieval_by_direction"].values()
    ):
        direction_metrics["cluster_bootstrap_ci"] = (
            bootstrap_cluster_weighted_margin(
                direction_metrics["cluster_margins"],
                replicates=replicates,
                seed=seed + index,
            )
        )
    groups = _panel_groups(panel)
    for index, (metric_name, group_field) in enumerate(
        (
            ("contact", "antigen_cluster_id"),
            ("epitope", "antigen_cluster_id"),
            ("paratope", "antibody_cluster_id"),
        )
    ):
        group_by_instance = {
            identifier: values[group_field]
            for identifier, values in groups.items()
        }
        metrics[metric_name]["mean_per_complex_auprc_cluster_bootstrap_ci"] = (
            bootstrap_group_weighted_values(
                metrics[metric_name]["per_complex_auprc_by_instance"],
                group_by_instance,
                replicates=replicates,
                seed=seed + 1_000 * (index + 1),
            )
        )
        metrics[metric_name]["mean_per_complex_auprc_cluster_definition"] = (
            f"equal-weight mean over {group_field} means; zero-positive complexes "
            "are excluded only from positive-required per-complex AP"
        )


def _consume_once(
    state_root: Path,
    *,
    opening_sha256: str,
    cache_manifest_sha256: str,
    output_root: Path,
) -> Path:
    """Create an O_EXCL sentinel before any cached label is deserialized."""
    token_path = state_root / "ASEP_TEST_SCORE_CONSUMED.json"
    payload = canonical_json_bytes(
        {
            "format_version": CONSUMPTION_FORMAT,
            "opening_receipt_sha256": opening_sha256,
            "test_cache_manifest_sha256": cache_manifest_sha256,
            "evaluator_source_sha256": sha256_file(Path(__file__)),
            "requested_output_root": str(output_root.resolve()),
            "state": "scoring_started_fail_closed",
            "recovery_boundary": (
                "If scoring crashes, do not delete this token or rerun silently; "
                "publish the failure and freeze a recovery receipt first."
            ),
        }
    )
    descriptor = os.open(
        token_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return token_path


def _evaluate_scope(
    model: torch.nn.Module,
    dataset: Any,
    panel: dict[str, Any],
    *,
    device: torch.device,
    use_bf16: bool,
    evaluation_batch_size: int,
    workers: int,
    configuration: Mapping[str, Any],
    specificity_source: str,
    frozen_calibration: Mapping[str, float],
    seed: int,
    replicates: int,
    batch_intervention: Any = None,
) -> tuple[dict[str, Any], Any]:
    loader = _loader(
        dataset,
        batch_size=evaluation_batch_size,
        workers=workers,
        shuffle=False,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    metrics = _evaluate(
        model,
        loader,
        dataset,
        panel,
        device,
        use_bf16=use_bf16,
        evaluation_batch_size=evaluation_batch_size,
        score_fraction=float(configuration["score_fraction"]),
        seed=seed,
        calibration_max_pairs=int(configuration["calibration_max_pairs"]),
        specificity_score_source=specificity_source,
        collect_attention=False,
        frozen_calibration=frozen_calibration,
        batch_intervention=batch_intervention,
        compute_calibration_regression=batch_intervention is None,
    )
    _add_bootstrap(metrics, panel, replicates=replicates, seed=seed)
    return metrics, loader


def main() -> None:
    arguments = _arguments()
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError("Refusing to overwrite AsEP frozen-test output")
    if arguments.panel_size != 31:
        raise ValueError("AsEP official test requires exactly 31 wrong partners")
    opening = _authenticate_official_test_opening(
        arguments.opening_receipt,
        arguments.opening_receipt_sha256,
    )
    if opening.get("test_evaluator_source_sha256") != sha256_file(Path(__file__)):
        raise ValueError("Opening receipt evaluator source SHA-256 differs")
    if opening.get("evaluation_source_dependencies") != (
        asep_evaluation_source_hashes(REPOSITORY_ROOT)
    ):
        raise ValueError("Opening receipt evaluation dependency hashes differ")
    if (
        opening.get("selection_receipt_sha256")
        != arguments.selection_receipt_sha256
        or opening.get("full_test_panel_sha256")
        != arguments.full_test_panel_sha256
        or opening.get("clean_test_panel_sha256")
        != arguments.clean_test_panel_sha256
        or opening.get("test_partner_panel_size") != arguments.panel_size
    ):
        raise ValueError("Requested evaluator authorities differ from the opening")
    selection = authenticate_asep_development_selection(
        arguments.selection_root,
        selection_receipt_sha256=arguments.selection_receipt_sha256,
    )
    if (
        Path(selection["selected_ensemble_root"]).resolve()
        != arguments.ensemble_root.resolve()
        or selection["selected_ensemble_result_sha256"]
        != opening.get("selected_ensemble_result_sha256")
    ):
        raise ValueError("Requested AsEP ensemble differs from the frozen selection")
    ensemble_result, ensemble_receipt = _authenticated_ensemble(
        arguments.ensemble_root
    )
    if ensemble_receipt.get("result_sha256") != selection[
        "selected_ensemble_result_sha256"
    ]:
        raise ValueError("Selected ensemble result SHA-256 differs")
    members = [authenticated_development_run(path) for path in arguments.run_roots]
    members.sort(key=lambda member: int(member["checkpoint"]["seed"]))
    if tuple(int(member["checkpoint"]["seed"]) for member in members) != FROZEN_SEEDS:
        raise ValueError("Official test requires the five pre-registered seeds")
    expected_members = {
        int(member["seed"]): member
        for member in opening.get("members", [])
    }
    ensemble_members = {
        int(member["seed"]): member
        for member in ensemble_result["authority"]["members"]
    }
    for member in members:
        seed = int(member["checkpoint"]["seed"])
        for expected in (expected_members.get(seed), ensemble_members.get(seed)):
            if (
                expected is None
                or expected.get("development_result_sha256")
                != member["result_sha256"]
                or expected.get("selected_checkpoint_sha256")
                != member["checkpoint_sha256"]
            ):
                raise ValueError("Official-test member differs from frozen authority")

    manifest = _authenticate_cache(
        arguments.test_cache_root,
        arguments.test_cache_manifest_sha256,
        arguments.test_cache_receipt_sha256,
        opening_sha256=arguments.opening_receipt_sha256,
        selected_ensemble_sha256=selection["selected_ensemble_result_sha256"],
        opening=opening,
    )
    full_panel = _authenticated_panel(
        arguments.full_test_panel,
        arguments.full_test_panel_sha256,
        role="test",
        expected_size=arguments.panel_size,
        expected_format=RECIPROCAL_PARTNER_PANEL_FORMAT,
    )
    clean_panel = _authenticated_panel(
        arguments.clean_test_panel,
        arguments.clean_test_panel_sha256,
        role="test",
        expected_size=arguments.panel_size,
        expected_format=RECIPROCAL_PARTNER_PANEL_FORMAT,
    )
    if (
        full_panel.get("authority", {}).get("panel_scope") != "full_official_test"
        or clean_panel.get("authority", {}).get("panel_scope")
        != "sabdab2_decontaminated_primary"
    ):
        raise ValueError("AsEP test panel scopes differ from the opening protocol")
    full_ids = {str(entry["instance_id"]) for entry in full_panel["entries"]}
    clean_ids = {str(entry["instance_id"]) for entry in clean_panel["entries"]}
    if (
        len(full_ids) != int(manifest.get("record_count", -1))
        or len(full_ids) != int(opening.get("official_test_instance_count", -1))
        or len(clean_ids)
        != int(opening.get("decontaminated_primary_instance_count", -1))
        or not clean_ids.issubset(full_ids)
    ):
        raise ValueError("AsEP full or decontaminated test axis differs")

    bootstrap = opening.get("bootstrap", {})
    replicates = int(bootstrap.get("replicates", -1))
    bootstrap_seed = int(bootstrap.get("seed", -1))
    if replicates <= 0 or bootstrap_seed < 0:
        raise ValueError("Opening bootstrap protocol is invalid")
    if replicates != 10_000 or bootstrap_seed != 20_260_730:
        raise ValueError("Opening receipt differs from the frozen bootstrap protocol")
    device = _device(arguments.device)
    use_bf16 = arguments.precision == "bf16" and device.type == "cuda"
    if arguments.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 requires CUDA")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise ValueError("CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG")
    random.seed(bootstrap_seed)
    np.random.seed(bootstrap_seed)
    torch.manual_seed(bootstrap_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(bootstrap_seed)

    # Fail closed before the first artifact (and therefore first label) is read.
    consumption_token = _consume_once(
        Path(str(opening["test_state_root"])),
        opening_sha256=arguments.opening_receipt_sha256,
        cache_manifest_sha256=arguments.test_cache_manifest_sha256,
        output_root=arguments.output_root,
    )
    feature_mode = str(ensemble_result["configuration"]["asep_feature_mode"])
    full_dataset = AsepFrozenTestDataset(
        arguments.test_cache_root,
        manifest_sha256=arguments.test_cache_manifest_sha256,
        feature_mode=feature_mode,
    )
    clean_dataset = AsepFrozenTestDataset(
        arguments.test_cache_root,
        manifest_sha256=arguments.test_cache_manifest_sha256,
        feature_mode=feature_mode,
        instance_ids=clean_ids,
    )
    if {record["instance_id"] for record in full_dataset.records} != full_ids:
        raise ValueError("Full official test panel and cache differ")

    first = full_dataset[0]
    experiment = str(ensemble_result["experiment"])
    models = []
    calibrations = []
    specificity_sources = set()
    member_metrics: dict[str, Any] = {}
    for member in members:
        model = _model_from_configuration(
            first,
            member["result"]["configuration"],
            experiment,
        )
        model.load_state_dict(member["checkpoint"]["model_state_dict"])
        model.to(device).eval()
        source = str(member["checkpoint"]["specificity_score_source"])
        specificity_sources.add(source)
        seed = int(member["checkpoint"]["seed"])
        metrics, _ = _evaluate_scope(
            model,
            full_dataset,
            full_panel,
            device=device,
            use_bf16=use_bf16,
            evaluation_batch_size=arguments.evaluation_batch_size,
            workers=arguments.workers,
            configuration=member["result"]["configuration"],
            specificity_source=source,
            frozen_calibration=_frozen_calibration(
                member["result"]["selected_checkpoint_validation"]
            ),
            seed=bootstrap_seed + seed,
            replicates=replicates,
        )
        member_metrics[str(seed)] = metrics
        models.append(model)
        calibrations.append(member_calibration(member["result"]))
    if specificity_sources != {"interaction_only"}:
        raise ValueError("AsEP official-test members require interaction-only specificity")
    ensemble = CalibratedProbabilityEnsemble(models, calibrations).to(device).eval()
    configuration = ensemble_result["configuration"]
    frozen_ensemble_calibration = _frozen_calibration(ensemble_result["validation"])
    full_metrics, full_loader = _evaluate_scope(
        ensemble,
        full_dataset,
        full_panel,
        device=device,
        use_bf16=use_bf16,
        evaluation_batch_size=arguments.evaluation_batch_size,
        workers=arguments.workers,
        configuration=configuration,
        specificity_source="interaction_only",
        frozen_calibration=frozen_ensemble_calibration,
        seed=bootstrap_seed,
        replicates=replicates,
    )
    clean_metrics, _ = _evaluate_scope(
        ensemble,
        clean_dataset,
        clean_panel,
        device=device,
        use_bf16=use_bf16,
        evaluation_batch_size=arguments.evaluation_batch_size,
        workers=arguments.workers,
        configuration=configuration,
        specificity_source="interaction_only",
        frozen_calibration=frozen_ensemble_calibration,
        seed=bootstrap_seed + 1,
        replicates=replicates,
    )
    invariance = _independent_se3_prediction_invariance(
        ensemble,
        full_loader,
        device,
        use_bf16=use_bf16,
        seed=bootstrap_seed + 60_000,
    )

    interventions: dict[str, Any] = {}
    factories = [
        (
            "geometry_shuffle",
            lambda generator: lambda batch: geometry_shuffle(
                batch, generator=generator
            ),
        ),
        (
            "local_frame_corruption",
            lambda generator: lambda batch: corrupt_local_frames(
                batch, generator=generator
            ),
        ),
        (
            "degree_matched_graph_rewiring",
            lambda generator: lambda batch: degree_matched_graph_rewiring(
                batch,
                generator=generator,
                neighbor_k=int(configuration["neighbor_k"]),
                neighbor_radius=float(configuration["neighbor_radius"]),
            ),
        ),
    ]
    factories.extend(
        (
            f"coordinate_noise_{noise:g}_angstrom",
            lambda generator, noise=noise: lambda batch: perturb_coordinates(
                batch,
                generator=generator,
                standard_deviation=noise,
            ),
        )
        for noise in (0.10, 0.25, 0.50)
    )
    for index, (name, factory) in enumerate(factories):
        local_seed = bootstrap_seed + 10_000 + index
        metrics, _ = _evaluate_scope(
            ensemble,
            full_dataset,
            full_panel,
            device=device,
            use_bf16=use_bf16,
            evaluation_batch_size=arguments.evaluation_batch_size,
            workers=arguments.workers,
            configuration=configuration,
            specificity_source="interaction_only",
            frozen_calibration=frozen_ensemble_calibration,
            seed=local_seed,
            replicates=replicates,
            batch_intervention=factory(_generator(device, local_seed)),
        )
        interventions[name] = {
            "metrics": metrics,
            "delta_vs_clean_full_official_test": _intervention_delta(
                full_metrics, metrics
            ),
        }

    payload = {
        "format_version": TEST_FORMAT,
        "claim_boundary": (
            "One hash-bound AsEP official-test scoring session after unique "
            "validation-only architecture and checkpoint selection. No test fitting, "
            "threshold selection, architecture selection, or contamination exclusion "
            "was performed from predictions or labels."
        ),
        "experiment": experiment,
        "seeds": list(FROZEN_SEEDS),
        "authority": {
            "opening_receipt_sha256": arguments.opening_receipt_sha256,
            "selection_receipt_sha256": arguments.selection_receipt_sha256,
            "selected_ensemble_result_sha256": ensemble_receipt["result_sha256"],
            "test_cache_manifest_sha256": arguments.test_cache_manifest_sha256,
            "test_cache_receipt_sha256": arguments.test_cache_receipt_sha256,
            "full_test_panel_sha256": arguments.full_test_panel_sha256,
            "clean_test_panel_sha256": arguments.clean_test_panel_sha256,
            "consumption_token_sha256": sha256_file(consumption_token),
            "test_graph_or_label_open_count": 1,
            "test_score_session_count": 1,
            "validation_calibration_frozen": True,
            "selection_performed_on_test": False,
            "test_driven_exclusions": False,
            "required_interventions_complete": True,
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
        },
        "test_complexes": {
            "full_official_test": len(full_dataset),
            "sabdab2_decontaminated_primary": len(clean_dataset),
        },
        "per_seed_full_official_test": member_metrics,
        "ensemble": {
            "full_official_test": full_metrics,
            "sabdab2_decontaminated_primary": clean_metrics,
        },
        "independent_monomer_se3_prediction_invariance": invariance,
        "intervention_metrics_full_official_test": interventions,
        "counterfactual_boundary": (
            "Wrong partners are label-free ranking controls and are not asserted to be "
            "experimentally verified non-binders."
        ),
    }
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.tmp-",
            dir=arguments.output_root.parent,
        )
    )
    try:
        result_path = temporary / "FRAMECONTACT_ASEP_FROZEN_TEST_RESULT.json"
        result_path.write_bytes(canonical_json_bytes(payload))
        receipt = {
            "format_version": TEST_FORMAT,
            "result_sha256": sha256_file(result_path),
            "opening_receipt_sha256": arguments.opening_receipt_sha256,
            "test_loaded": True,
            "test_score_session_count": 1,
            "selection_performed_on_test": False,
            "required_interventions_complete": True,
        }
        (temporary / "ASEP_FROZEN_TEST_RECEIPT.json").write_bytes(
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
