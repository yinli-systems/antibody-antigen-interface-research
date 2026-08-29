#!/usr/bin/env python3
"""Freeze one precommitted task-routed model for an exposed SAbDab2 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


FULL_PROTOCOL_SHA256 = (
    "a00a51a4617a581c57a74f854defd6c18d8258180659775b1c5044be10f98c5c"
)
OLD_TEST_RESULT_SHA256 = (
    "4946c3c1de03c867b5eb368f0d044a278d429a66cd8ff3da931c9d40aed48de4"
)
FULL_SPLIT_SHA256 = (
    "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
)
TEST_PANEL_SHA256 = (
    "ea70fef770448302fb4ed1ccd7da51a1cc56cba5ea153076e077971735874bd1"
)
STRUCTURE_MANIFEST_SHA256 = (
    "edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0"
)
CONTACT_MANIFEST_SHA256 = (
    "d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33"
)
EMBEDDING_MANIFEST_SHA256 = (
    "9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545"
)
SEED = 17


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


def selected_epoch_validation(log_path: Path, epoch: int) -> dict[str, object]:
    selected: dict[str, object] | None = None
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("epoch") == epoch and isinstance(record.get("validation"), dict):
                selected = record
    if selected is None:
        raise ValueError(f"epoch {epoch} validation is absent from {log_path}")
    validation = selected["validation"]
    assert isinstance(validation, dict)
    return {
        "epoch": epoch,
        "elapsed_seconds": selected.get("elapsed_seconds"),
        "validation_temperature": validation["validation_temperature"],
        "contact": validation["contact"],
        "epitope": validation["epitope"],
        "paratope": validation["paratope"],
    }


def frozen_calibration(validation: dict[str, object]) -> dict[str, float]:
    contact = validation["contact"]
    epitope = validation["epitope"]
    paratope = validation["paratope"]
    assert isinstance(contact, dict)
    assert isinstance(epitope, dict)
    assert isinstance(paratope, dict)
    return {
        "temperature": float(validation["validation_temperature"]),
        "contact_threshold": float(contact["threshold_selected_on_validation"]),
        "epitope_temperature": float(epitope["validation_temperature"]),
        "epitope_threshold": float(epitope["threshold_selected_on_validation"]),
        "paratope_temperature": float(paratope["validation_temperature"]),
        "paratope_threshold": float(paratope["threshold_selected_on_validation"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-campaign-root", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--baseline-source-root", type=Path, required=True)
    parser.add_argument("--baseline-source-manifest", type=Path, required=True)
    parser.add_argument("--candidate-source-root", type=Path, required=True)
    parser.add_argument("--candidate-source-manifest", type=Path, required=True)
    parser.add_argument("--old-test-result", type=Path, required=True)
    parser.add_argument("--old-test-receipt", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--test-panel", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--sbatch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)

    authority_root = args.full_campaign_root / "authority"
    full_protocol_path = authority_root / "PROTOCOL_EDGE_TOPK2_ROUTED_FULL_V1.json"
    full_receipt_path = authority_root / "PROTOCOL_EDGE_TOPK2_ROUTED_FULL_V1_RECEIPT.json"
    require_sha(full_protocol_path, FULL_PROTOCOL_SHA256, "full-development protocol")
    require_sha(args.old_test_result, OLD_TEST_RESULT_SHA256, "old test result")
    for path, expected, label in (
        (args.split_manifest, FULL_SPLIT_SHA256, "split manifest"),
        (args.test_panel, TEST_PANEL_SHA256, "test partner panel"),
        (args.structure_manifest, STRUCTURE_MANIFEST_SHA256, "structure manifest"),
        (args.contact_manifest, CONTACT_MANIFEST_SHA256, "contact manifest"),
        (args.embedding_manifest, EMBEDDING_MANIFEST_SHA256, "embedding manifest"),
    ):
        require_sha(path, expected, label)

    full_protocol = json.loads(full_protocol_path.read_text())
    full_receipt = json.loads(full_receipt_path.read_text())
    old_receipt = json.loads(args.old_test_receipt.read_text())
    if (
        full_protocol.get("task_routing", {}).get("routing_is_fixed_per_task") is not True
        or full_protocol.get("task_routing", {}).get("mixture_weights_tuned") is not False
        or full_receipt.get("protocol_file_sha256") != FULL_PROTOCOL_SHA256
        or old_receipt.get("result_sha256") != OLD_TEST_RESULT_SHA256
        or old_receipt.get("test_opening_count") != 1
        or old_receipt.get("selection_performed_on_test") is not False
    ):
        raise ValueError("predecessor authority differs")

    baseline_bound = full_protocol["baseline_binding"][str(SEED)]
    baseline_root = Path(str(baseline_bound["run_root"]))
    baseline_result_path = baseline_root / "FRAMECONTACT_DEVELOPMENT_RESULT.json"
    baseline_checkpoint_path = baseline_root / "FRAMECONTACT_SELECTED.pt"
    baseline_receipt_path = baseline_root / "RUN_RECEIPT.json"
    for path, expected, label in (
        (baseline_result_path, baseline_bound["result_sha256"], "baseline result"),
        (
            baseline_checkpoint_path,
            baseline_bound["checkpoint_sha256"],
            "baseline checkpoint",
        ),
        (baseline_receipt_path, baseline_bound["receipt_sha256"], "baseline receipt"),
    ):
        require_sha(path, str(expected), label)
    baseline_result = json.loads(baseline_result_path.read_text())
    if (
        baseline_result.get("configuration", {}).get("seed") != SEED
        or baseline_result.get("authority", {}).get("test_loaded") is not False
    ):
        raise ValueError("baseline development authority differs")

    import torch

    source_hash_before = sha256_file(args.candidate_checkpoint)
    checkpoint = torch.load(args.candidate_checkpoint, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("seed") != SEED
        or checkpoint.get("experiment") != "epiformer_style_same_protocol"
        or int(checkpoint.get("selected_epoch", -1)) < 1
    ):
        raise ValueError("candidate checkpoint metadata differs")
    selected_epoch = int(checkpoint["selected_epoch"])
    selected_validation = selected_epoch_validation(args.candidate_log, selected_epoch)
    if (
        str(checkpoint.get("selection_rule"))
        != full_protocol["model_selection"]["checkpoint_rule"]
        or abs(
            float(checkpoint["validation_temperature"])
            - float(selected_validation["validation_temperature"])
        )
        > 1e-12
    ):
        raise ValueError("candidate checkpoint and validation record differ")

    baseline_configuration = dict(baseline_result["configuration"])
    candidate_configuration = dict(baseline_configuration)
    candidate_configuration.update(
        {
            "protocol": str(full_protocol_path),
            "protocol_receipt": str(full_receipt_path),
            "protocol_receipt_sha256": sha256_file(full_receipt_path),
            "source_manifest": str(args.candidate_source_manifest),
            "source_manifest_sha256": sha256_file(args.candidate_source_manifest),
            "seed": SEED,
        }
    )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{args.output_root.name}.", dir=args.output_root.parent)
    )
    try:
        authority = temporary / "authority"
        checkpoints = authority / "checkpoints"
        logs = temporary / "logs"
        components = temporary / "component-results"
        checkpoints.mkdir(parents=True)
        logs.mkdir()
        components.mkdir()
        frozen_candidate = checkpoints / f"edge_topk2_seed{SEED}_validation_best.pt"
        frozen_candidate_final = (
            args.output_root
            / "authority"
            / "checkpoints"
            / f"edge_topk2_seed{SEED}_validation_best.pt"
        )
        shutil.copy2(args.candidate_checkpoint, frozen_candidate)
        source_hash_after = sha256_file(args.candidate_checkpoint)
        frozen_hash = sha256_file(frozen_candidate)
        if source_hash_before != source_hash_after or frozen_hash != source_hash_before:
            raise RuntimeError("candidate checkpoint changed during snapshot")

        components_binding = {
            "baseline_current": {
                "source_root": str(args.baseline_source_root),
                "source_manifest": str(args.baseline_source_manifest),
                "source_manifest_sha256": sha256_file(args.baseline_source_manifest),
                "trainer_sha256": sha256_file(
                    args.baseline_source_root / "scripts/train_framecontact.py"
                ),
                "framecontact_sha256": sha256_file(
                    args.baseline_source_root / "src/interaction_first/framecontact.py"
                ),
                "checkpoint": str(baseline_checkpoint_path),
                "checkpoint_sha256": sha256_file(baseline_checkpoint_path),
                "development_result": str(baseline_result_path),
                "development_result_sha256": sha256_file(baseline_result_path),
                "development_receipt": str(baseline_receipt_path),
                "development_receipt_sha256": sha256_file(baseline_receipt_path),
                "configuration": baseline_configuration,
                "frozen_calibration": frozen_calibration(
                    baseline_result["selected_checkpoint_validation"]
                ),
                "selected_epoch": int(baseline_result["selection"]["best_epoch"]),
                "specificity_score_source": str(
                    baseline_result["selected_checkpoint_validation"][
                        "specificity_score_source"
                    ]
                ),
            },
            "edge_topk2_supervised": {
                "source_root": str(args.candidate_source_root),
                "source_manifest": str(args.candidate_source_manifest),
                "source_manifest_sha256": sha256_file(args.candidate_source_manifest),
                "trainer_sha256": sha256_file(
                    args.candidate_source_root / "scripts/train_framecontact.py"
                ),
                "framecontact_sha256": sha256_file(
                    args.candidate_source_root / "src/interaction_first/framecontact.py"
                ),
                "checkpoint": str(frozen_candidate_final),
                "checkpoint_sha256": frozen_hash,
                "source_checkpoint": str(args.candidate_checkpoint),
                "source_checkpoint_sha256": source_hash_before,
                "configuration": candidate_configuration,
                "frozen_calibration": frozen_calibration(selected_validation),
                "selected_epoch": selected_epoch,
                "selected_checkpoint_validation": selected_validation,
                "specificity_score_source": str(checkpoint["specificity_score_source"]),
            },
        }
        protocol = {
            "format_version": "sabdab2_task_routed_exposed_historical_protocol_v1",
            "status": "FROZEN_BEFORE_EXPOSED_HISTORICAL_BATCH",
            "claim_boundary": {
                "fresh_blind_test": False,
                "historical_benchmark_only": True,
                "old_test_previously_exposed": True,
                "prior_test_feedback_influenced_model_development": True,
                "selection_performed_on_this_measurement": False,
                "result_may_not_select_checkpoint_or_model": True,
                "result_may_not_authorize_temporal_holdout": True,
                "cumulative_test_access_ordinal": 2,
            },
            "seed": SEED,
            "selection_rule": "lowest pre-registered seed; validation-best snapshot before this exposed batch",
            "task_routing": full_protocol["task_routing"],
            "components": components_binding,
            "data": {
                "test_complex_count": 549,
                "candidate_grid": "complete antibody-residue x antigen-residue cartesian product",
                "split_manifest": str(args.split_manifest),
                "split_manifest_sha256": FULL_SPLIT_SHA256,
                "test_partner_panel": str(args.test_panel),
                "test_partner_panel_sha256": TEST_PANEL_SHA256,
                "structure_manifest_sha256": STRUCTURE_MANIFEST_SHA256,
                "contact_manifest_sha256": CONTACT_MANIFEST_SHA256,
                "embedding_manifest_sha256": EMBEDDING_MANIFEST_SHA256,
            },
            "predecessor": {
                "full_development_protocol_sha256": FULL_PROTOCOL_SHA256,
                "old_test_result_sha256": OLD_TEST_RESULT_SHA256,
                "old_test_receipt_sha256": sha256_file(args.old_test_receipt),
            },
            "execution": {
                "evaluator_sha256": sha256_file(args.evaluator),
                "analyzer_sha256": sha256_file(args.analyzer),
                "sbatch_sha256": sha256_file(args.sbatch),
                "component_arms": ["baseline_current", "edge_topk2_supervised"],
                "component_jobs_run_as_one_precommitted_batch": True,
                "clean_metrics_only": True,
                "bootstrap_replicates": 10_000,
                "bootstrap_seed": 20_260_826,
            },
        }
        protocol_path = authority / "PROTOCOL_TASK_ROUTED_EXPOSED_TEST_V1.json"
        protocol_path.write_bytes(canonical_bytes(protocol))
        protocol_sha256 = sha256_file(protocol_path)
        receipt = {
            "format_version": "sabdab2_task_routed_exposed_historical_receipt_v1",
            "status": protocol["status"],
            "protocol_sha256": protocol_sha256,
            "seed": SEED,
            "candidate_checkpoint_sha256": frozen_hash,
            "baseline_checkpoint_sha256": components_binding["baseline_current"][
                "checkpoint_sha256"
            ],
            "old_test_previously_exposed": True,
            "selection_performed_on_this_measurement": False,
            "cumulative_test_access_ordinal": 2,
        }
        receipt_path = authority / "PROTOCOL_TASK_ROUTED_EXPOSED_TEST_V1_RECEIPT.json"
        receipt_path.write_bytes(canonical_bytes(receipt))
        campaign = {
            "format_version": "sabdab2_task_routed_exposed_historical_campaign_v1",
            "status": "FROZEN_BEFORE_EXPOSED_HISTORICAL_BATCH",
            "protocol_sha256": protocol_sha256,
            "protocol_receipt_sha256": sha256_file(receipt_path),
            "seed": SEED,
            "component_results_complete": False,
            "analysis_complete": False,
            "fresh_blind_test": False,
            "historical_benchmark_only": True,
        }
        campaign_path = authority / "CAMPAIGN_TASK_ROUTED_EXPOSED_TEST_V1.json"
        campaign_path.write_bytes(canonical_bytes(campaign))
        protocol_receipt_sha256 = sha256_file(receipt_path)
        campaign_sha256 = sha256_file(campaign_path)
        temporary.replace(args.output_root)
        print(
            json.dumps(
                {
                    "status": campaign["status"],
                    "output_root": str(args.output_root),
                    "protocol_sha256": protocol_sha256,
                    "protocol_receipt_sha256": protocol_receipt_sha256,
                    "campaign_sha256": campaign_sha256,
                    "candidate_checkpoint_sha256": frozen_hash,
                    "candidate_selected_epoch": selected_epoch,
                },
                sort_keys=True,
            )
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


if __name__ == "__main__":
    main()
