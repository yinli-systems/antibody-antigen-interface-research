#!/usr/bin/env python3
"""Score a frozen development checkpoint on same-antigen partner swaps."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
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
    Sabdab2StructureBatchCollator,
    authenticate_file,
    canonical_json_bytes,
    sha256_file,
)
from train_framecontact import (  # noqa: E402
    EXPERIMENTS,
    RESULT_FORMAT,
    _model_from_configuration,
    _wrong_antibody_pair_item,
)
from freeze_asep_development_ensemble import (  # noqa: E402
    ENSEMBLE_FORMAT,
    FROZEN_SEEDS,
    authenticated_development_run,
    member_calibration,
)


CHALLENGE_FORMAT = "asep_same_antigen_different_antibody_challenge_v1"
RESULT_FORMAT_CHALLENGE = "framecontact_asep_same_antigen_challenge_result_v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path)
    source.add_argument("--ensemble-root", type=Path)
    parser.add_argument("--run-roots", type=Path, nargs=5)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--challenge-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_730)
    return parser.parse_args()


def directional_assignment_metrics(
    cognate_probabilities: np.ndarray,
    swapped_probabilities: np.ndarray,
    own_labels: np.ndarray,
    swapped_partner_labels: np.ndarray,
) -> dict[str, float | bool]:
    """Require a partner swap to move scores toward the partner's known epitope."""
    from sklearn.metrics import average_precision_score

    arrays = (
        cognate_probabilities,
        swapped_probabilities,
        own_labels,
        swapped_partner_labels,
    )
    if any(value.ndim != 1 for value in arrays) or len({len(value) for value in arrays}) != 1:
        raise ValueError("Same-antigen challenge arrays must be aligned vectors")
    if len(own_labels) == 0 or not np.isin(own_labels, [0, 1]).all() or not np.isin(
        swapped_partner_labels, [0, 1]
    ).all():
        raise ValueError("Same-antigen challenge labels must be nonempty and binary")
    if any(
        not np.isfinite(value).all()
        for value in (cognate_probabilities, swapped_probabilities)
    ) or any(
        bool(((value < 0) | (value > 1)).any())
        for value in (cognate_probabilities, swapped_probabilities)
    ):
        raise ValueError("Same-antigen challenge probabilities are invalid")

    cognate_own = float(average_precision_score(own_labels, cognate_probabilities))
    swapped_own = float(average_precision_score(own_labels, swapped_probabilities))
    cognate_target = float(
        average_precision_score(swapped_partner_labels, cognate_probabilities)
    )
    swapped_target = float(
        average_precision_score(swapped_partner_labels, swapped_probabilities)
    )
    own_retention = cognate_own - swapped_own
    target_migration = swapped_target - cognate_target
    return {
        "cognate_own_auprc": cognate_own,
        "swapped_input_own_label_auprc": swapped_own,
        "cognate_input_swapped_label_auprc": cognate_target,
        "swapped_input_swapped_label_auprc": swapped_target,
        "own_retention_margin": own_retention,
        "target_migration_margin": target_migration,
        "assignment_margin": 0.5 * (own_retention + target_migration),
        "mean_absolute_prediction_shift": float(
            np.mean(np.abs(cognate_probabilities - swapped_probabilities))
        ),
        "directional_success": own_retention > 0 and target_migration > 0,
    }


def _cluster_bootstrap(
    records: list[dict[str, Any]], *, replicates: int, seed: int
) -> dict[str, float | int]:
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    by_cluster: dict[str, list[float]] = {}
    for record in records:
        by_cluster.setdefault(str(record["antigen_cluster_id"]), []).append(
            float(record["assignment_margin"])
        )
    cluster_means = np.asarray(
        [np.mean(values) for _, values in sorted(by_cluster.items())], dtype=np.float64
    )
    if len(cluster_means) < 2:
        raise ValueError("Same-antigen challenge needs at least two antigen clusters")
    rng = np.random.default_rng(seed)
    samples = cluster_means[
        rng.integers(0, len(cluster_means), size=(replicates, len(cluster_means)))
    ].mean(axis=1)
    return {
        "cluster_count": len(cluster_means),
        "replicates": replicates,
        "mean_cluster_assignment_margin": float(cluster_means.mean()),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
    }


def _authenticated_run(run_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result_path = run_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    checkpoint_path = run_root / "FRAMECONTACT_SELECTED.pt"
    receipt_path = run_root / "RUN_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("format_version") != RESULT_FORMAT
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or sha256_file(result_path) != receipt.get("result_sha256")
        or sha256_file(checkpoint_path) != receipt.get("selected_checkpoint_sha256")
    ):
        raise ValueError("Development run is not eligible for challenge scoring")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        result.get("format_version") != RESULT_FORMAT
        or result.get("authority", {}).get("dataset_backend") != "asep"
        or result.get("authority", {}).get("test_loaded") is not False
        or result.get("authority", {}).get("external_test_loaded") is not False
        or checkpoint.get("format_version") != RESULT_FORMAT
        or checkpoint.get("experiment") != result.get("experiment")
        or checkpoint.get("seed") != result.get("configuration", {}).get("seed")
    ):
        raise ValueError("Development result/checkpoint metadata differs")
    return result, checkpoint, receipt


def _authenticated_ensemble(
    ensemble_root: Path, run_roots: list[Path]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    result_path = ensemble_root / "FRAMECONTACT_ASEP_DEVELOPMENT_ENSEMBLE.json"
    receipt_path = ensemble_root / "ENSEMBLE_RECEIPT.json"
    if any(not path.is_file() or path.is_symlink() for path in (result_path, receipt_path)):
        raise ValueError("Frozen AsEP ensemble result or receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("format_version") != ENSEMBLE_FORMAT
        or receipt.get("test_loaded") is not False
        or receipt.get("external_test_loaded") is not False
        or receipt.get("result_sha256") != sha256_file(result_path)
    ):
        raise ValueError("Frozen AsEP ensemble receipt failed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    authority = result.get("authority", {})
    if (
        result.get("format_version") != ENSEMBLE_FORMAT
        or authority.get("dataset_backend") != "asep"
        or authority.get("test_loaded") is not False
        or authority.get("external_test_loaded") is not False
        or tuple(result.get("seeds", [])) != FROZEN_SEEDS
    ):
        raise ValueError("Frozen AsEP ensemble is not challenge eligible")
    members = [authenticated_development_run(path) for path in run_roots]
    members.sort(key=lambda item: int(item["checkpoint"]["seed"]))
    expected = authority.get("members", [])
    observed = [
        {
            "seed": int(member["checkpoint"]["seed"]),
            "development_result_sha256": member["result_sha256"],
            "selected_checkpoint_sha256": member["checkpoint_sha256"],
        }
        for member in members
    ]
    if expected != observed:
        raise ValueError("Challenge member runs differ from the frozen ensemble")
    if any(
        member["result"].get("experiment") != result.get("experiment")
        for member in members
    ):
        raise ValueError("Challenge member experiment differs from the frozen ensemble")
    return result, receipt, members


def _selection_maps(
    cache_root: Path, manifest: dict[str, Any], identifiers: set[str]
) -> dict[str, dict[int, int]]:
    records = {record["instance_id"]: record for record in manifest["records"]}
    if not identifiers <= set(records):
        raise ValueError("Challenge references an identifier outside the cache")
    maps: dict[str, dict[int, int]] = {}
    for identifier in sorted(identifiers):
        record = records[identifier]
        path = cache_root / record["artifact_path"]
        authenticate_file(path, record["artifact_sha256"], "challenge artifact")
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        selection = artifact["antigen_full_selection"]
        if selection.dtype != torch.bool or selection.ndim != 1:
            raise ValueError("Challenge antigen full selection is malformed")
        maps[identifier] = {
            int(full_index): selected_index
            for selected_index, full_index in enumerate(torch.where(selection)[0].tolist())
        }
    return maps


def main() -> None:
    arguments = _arguments()
    if arguments.output_root.exists() or arguments.output_root.is_symlink():
        raise FileExistsError("Refusing to overwrite challenge result")
    manifest_path = arguments.cache_root / "ASEP_DEVELOPMENT_CACHE_MANIFEST.json"
    authenticate_file(manifest_path, arguments.cache_manifest_sha256, "AsEP cache manifest")
    authenticate_file(arguments.challenge, arguments.challenge_sha256, "challenge panel")
    challenge = json.loads(arguments.challenge.read_text(encoding="utf-8"))
    if (
        challenge.get("format_version") != CHALLENGE_FORMAT
        or challenge.get("role") != "validation"
        or challenge.get("authority", {}).get("cache_manifest_sha256")
        != arguments.cache_manifest_sha256
        or challenge.get("authority", {}).get("official_test_materialized") is not False
        or challenge.get("authority", {}).get("model_predictions_or_scores_read") is not False
        or int(challenge.get("pair_count", -1)) != len(challenge.get("entries", []))
    ):
        raise ValueError("Same-antigen challenge authority is ineligible")
    if arguments.run_root is not None:
        if arguments.run_roots is not None:
            raise ValueError("--run-roots is only valid with --ensemble-root")
        development, checkpoint, run_receipt = _authenticated_run(arguments.run_root)
        members = None
        ensemble_receipt = None
        configuration = development["configuration"]
        model_seed = int(configuration["seed"])
        epitope_temperature = float(
            development["selected_checkpoint_validation"]["epitope"][
                "validation_temperature"
            ]
        )
        model_authority = {
            "mode": "single_seed",
            "development_result_sha256": run_receipt["result_sha256"],
            "selected_checkpoint_sha256": run_receipt[
                "selected_checkpoint_sha256"
            ],
        }
    else:
        if arguments.run_roots is None:
            raise ValueError("--ensemble-root requires exactly five --run-roots")
        development, ensemble_receipt, members = _authenticated_ensemble(
            arguments.ensemble_root, arguments.run_roots
        )
        checkpoint = None
        run_receipt = None
        configuration = members[0]["result"]["configuration"]
        model_seed = 20_260_730
        epitope_temperature = float(
            development["validation"]["epitope"]["validation_temperature"]
        )
        model_authority = {
            "mode": "five_seed_validation_frozen_probability_ensemble",
            "ensemble_result_sha256": ensemble_receipt["result_sha256"],
            "members": development["authority"]["members"],
        }
    if (
        development["authority"].get("asep_cache_manifest_sha256")
        != arguments.cache_manifest_sha256
    ):
        raise ValueError("Development model and challenge cache authorities differ")
    experiment = str(development["experiment"])
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown FrameContact experiment")

    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if arguments.precision == "bf16" and arguments.device != "cuda":
        raise ValueError("BF16 challenge scoring requires CUDA")
    if arguments.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise ValueError("CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG")
    device = torch.device(arguments.device)
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)

    dataset = AsepDevelopmentDataset(
        arguments.cache_root,
        manifest_sha256=arguments.cache_manifest_sha256,
        split_role="val",
        feature_mode=str(configuration["asep_feature_mode"]),
    )
    index_by_id = {
        record["instance_id"]: index for index, record in enumerate(dataset.records)
    }
    identifiers = {
        str(entry[key])
        for entry in challenge["entries"]
        for key in ("first_instance_id", "second_instance_id")
    }
    if not identifiers <= set(index_by_id):
        raise ValueError("Challenge references a non-validation instance")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_positions = _selection_maps(arguments.cache_root, manifest, identifiers)
    first_example = dataset[0]
    if members is None:
        model = _model_from_configuration(first_example, configuration, experiment)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        models = []
        calibrations = []
        for member in members:
            member_model = _model_from_configuration(
                first_example, member["result"]["configuration"], experiment
            )
            member_model.load_state_dict(member["checkpoint"]["model_state_dict"])
            models.append(member_model)
            calibrations.append(member_calibration(member["result"]))
        model = CalibratedProbabilityEnsemble(models, calibrations)
    model.to(device).eval()
    if not math.isfinite(epitope_temperature) or epitope_temperature <= 0:
        raise ValueError("Frozen epitope temperature is invalid")
    collator = Sabdab2StructureBatchCollator()
    directional_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for entry in challenge["entries"]:
            first_id = str(entry["first_instance_id"])
            second_id = str(entry["second_instance_id"])
            first = dataset[index_by_id[first_id]]
            second = dataset[index_by_id[second_id]]
            common = [int(value) for value in entry["common_antigen_full_indices"]]
            positive_by_id = {
                first_id: set(int(value) for value in entry["first_epitope_full_indices"]),
                second_id: set(
                    int(value) for value in entry["second_epitope_full_indices"]
                ),
            }
            for fixed_id, own, partner_id, partner in (
                (first_id, first, second_id, second),
                (second_id, second, first_id, first),
            ):
                swapped = _wrong_antibody_pair_item(own, partner)
                batch = collator([own, swapped])
                batch = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if arguments.precision == "bf16"
                    else nullcontext()
                )
                with context:
                    output = model(batch)
                probabilities = torch.sigmoid(
                    output["epitope_logits"].float() / epitope_temperature
                ).cpu().numpy()
                positions = selected_positions[fixed_id]
                if any(index not in positions for index in common):
                    raise ValueError("Challenge common axis is absent from fixed antigen")
                axis = np.asarray([positions[index] for index in common], dtype=np.int64)
                own_labels = np.asarray(
                    [index in positive_by_id[fixed_id] for index in common], dtype=np.int8
                )
                partner_labels = np.asarray(
                    [index in positive_by_id[partner_id] for index in common], dtype=np.int8
                )
                metrics = directional_assignment_metrics(
                    probabilities[0, axis], probabilities[1, axis], own_labels, partner_labels
                )
                directional_records.append(
                    {
                        "antigen_cluster_id": entry["antigen_cluster_id"],
                        "fixed_antigen_instance_id": fixed_id,
                        "cognate_antibody_instance_id": fixed_id,
                        "swapped_antibody_instance_id": partner_id,
                        "common_residue_count": len(common),
                        **metrics,
                    }
                )
    assignment = np.asarray(
        [float(record["assignment_margin"]) for record in directional_records]
    )
    shift = np.asarray(
        [float(record["mean_absolute_prediction_shift"]) for record in directional_records]
    )
    result = {
        "format_version": RESULT_FORMAT_CHALLENGE,
        "experiment": experiment,
        "seed": int(configuration["seed"]) if members is None else None,
        "authority": {
            **model_authority,
            "cache_manifest_sha256": arguments.cache_manifest_sha256,
            "challenge_sha256": arguments.challenge_sha256,
            "validation_temperature_frozen": epitope_temperature,
            "official_test_loaded": False,
        },
        "summary": {
            "challenge_pair_count": len(challenge["entries"]),
            "directional_case_count": len(directional_records),
            "mean_assignment_margin": float(assignment.mean()),
            "median_assignment_margin": float(np.median(assignment)),
            "mean_absolute_prediction_shift": float(shift.mean()),
            "directional_success_fraction": float(
                np.mean([record["directional_success"] for record in directional_records])
            ),
            "cluster_bootstrap": _cluster_bootstrap(
                directional_records,
                replicates=arguments.bootstrap_replicates,
                seed=arguments.bootstrap_seed,
            ),
        },
        "directional_records": directional_records,
        "interpretation": (
            "Positive assignment margin requires the cognate prediction to prefer its own "
            "epitope and the swapped-antibody prediction to migrate toward the other known "
            "epitope. Nonzero prediction shift alone is not success."
        ),
        "claim_boundary": (
            "Label-aware development-only counterfactual diagnostic. It did not select the "
            "checkpoint, architecture, calibration, or threshold and is not an external result."
        ),
    }
    arguments.output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_root.name}.tmp-",
            dir=arguments.output_root.parent,
        )
    )
    try:
        result_path = temporary_root / "ASEP_SAME_ANTIGEN_CHALLENGE_RESULT.json"
        result_path.write_bytes(canonical_json_bytes(result))
        receipt = {
            "format_version": RESULT_FORMAT_CHALLENGE,
            "result_sha256": sha256_file(result_path),
            "official_test_loaded": False,
            "used_for_model_selection": False,
        }
        (temporary_root / "RUN_RECEIPT.json").write_bytes(
            canonical_json_bytes(receipt)
        )
        os.replace(temporary_root, arguments.output_root)
        print(json.dumps({**receipt, "output_root": str(arguments.output_root)}))
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


if __name__ == "__main__":
    main()
