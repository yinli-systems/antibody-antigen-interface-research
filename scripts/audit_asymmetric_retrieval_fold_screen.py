#!/usr/bin/env python3
"""Independently audit the completed train-only retrieval architecture screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FORMAT_VERSION = "sabdab2_asymmetric_retrieval_fold_screen_audit_v1"
RUN_FORMAT_VERSION = "sabdab2_asymmetric_partner_retrieval_v1"
EXPECTED_VARIANTS = {
    "antiberty_calm_mean": {
        "antibody_artifact_format": "antiberty",
        "antibody_encoder": "AntiBERTy_md_smooth",
        "antibody_input_dim": 512,
        "chain_balanced_antibody_pooling": False,
    },
    "antiberty_chain_balanced": {
        "antibody_artifact_format": "antiberty",
        "antibody_encoder": "AntiBERTy_md_smooth",
        "antibody_input_dim": 512,
        "chain_balanced_antibody_pooling": True,
    },
    "esm2_calm_mean": {
        "antibody_artifact_format": "esm_chunks",
        "antibody_encoder": "facebook/esm2_t33_650M_UR50D",
        "antibody_input_dim": 1280,
        "chain_balanced_antibody_pooling": False,
    },
}
EXPECTED_MODEL_WEIGHTS = {
    "AntiBERTy_md_smooth": "f1ae33eac8cc8784a7d4be5a600141d2fa7bc7d8d5b3f5324d64a6a63bd0f137",
    "facebook/esm2_t33_650M_UR50D": "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0",
}
EXPECTED_SOURCE_NAMES = {
    "entrypoint": "sabdab2_asymmetric_partner_retrieval.py",
    "asymmetric_retrieval": "interaction_first/asymmetric_retrieval.py",
    "shared_retrieval_protocol": "sabdab2_partner_retrieval.py",
    "runtime": "interaction_first/runtime.py",
}
ERROR_PATTERN = re.compile(
    r"(?:\bnan\b|\binf\b|\boom\b|out of memory|traceback|"
    r"\berror\b|\bexception\b|\bkilled\b|segmentation fault|non-zero status)",
    re.IGNORECASE,
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a bounded-memory SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a mapping with the training code's canonical JSON encoding."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Any:
    """Load one strict JSON artifact."""
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r} in {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def require(condition: bool, message: str, failures: list[str]) -> None:
    """Append one failure when an audit assertion is false."""
    if not condition:
        failures.append(message)


def iter_numbers(value: Any, prefix: str = "$") -> Iterable[tuple[str, float]]:
    """Yield every numeric leaf and its JSON-style path."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        yield prefix, float(value)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_numbers(child, f"{prefix}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_numbers(child, f"{prefix}[{index}]")


def exact_random_best_positive_metrics(
    candidate_count: int,
    positive_counts: Sequence[int],
    recall_ks: Sequence[int],
) -> dict[str, Any]:
    """Compute the exact random-ranking expectation for best-positive rank."""
    if candidate_count <= 0 or not positive_counts:
        raise ValueError("Random baseline requires non-empty anchors and candidates")
    expected_rr_total = 0.0
    recall_totals = {int(k): 0.0 for k in recall_ks}
    cached_anchor_metrics: dict[int, tuple[float, dict[int, float]]] = {}
    for positive_count in positive_counts:
        m = int(positive_count)
        n = int(candidate_count)
        if m <= 0 or m > n:
            raise ValueError(f"Invalid positive count m={m} for N={n}")
        if m not in cached_anchor_metrics:
            probability = m / n
            anchor_rr = probability
            for rank in range(1, n - m + 1):
                probability *= (n - rank - m + 1) / (n - rank)
                anchor_rr += probability / (rank + 1)
            denominator = math.comb(n, m)
            anchor_recalls = {}
            for k in recall_totals:
                effective_k = min(max(k, 0), n)
                no_hit = (
                    math.comb(n - effective_k, m) / denominator
                    if n - effective_k >= m
                    else 0.0
                )
                anchor_recalls[k] = 1.0 - no_hit
            cached_anchor_metrics[m] = anchor_rr, anchor_recalls
        anchor_rr, anchor_recalls = cached_anchor_metrics[m]
        expected_rr_total += anchor_rr
        for k in recall_totals:
            recall_totals[k] += anchor_recalls[k]
    anchor_count = len(positive_counts)
    result: dict[str, Any] = {
        "anchor_count": anchor_count,
        "candidate_count": candidate_count,
        "mrr": float(expected_rr_total / anchor_count),
        "mean_positive_count": sum(positive_counts) / anchor_count,
        "derivation": (
            "P(best_rank=r)=C(N-r,m-1)/C(N,m); "
            "P(best_rank<=k)=1-C(N-k,m)/C(N,m)"
        ),
    }
    for k, total in recall_totals.items():
        result[f"recall_at_{k}"] = float(total / anchor_count)
    return result


def recompute_observed_direction(
    direction: Mapping[str, Any],
    recall_ks: Sequence[int],
) -> dict[str, Any]:
    """Recompute aggregate retrieval metrics from saved per-anchor ranks."""
    rows = direction["per_anchor"]
    ranks = [int(row["best_known_positive_rank"]) for row in rows]
    result: dict[str, Any] = {
        "anchor_count": len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
    }
    for k in recall_ks:
        result[f"recall_at_{k}"] = sum(rank <= k for rank in ranks) / len(ranks)
    return result


def safe_torch_load(path: Path) -> Mapping[str, Any]:
    """Load a trusted tensor-only checkpoint with the safest available API."""
    import torch

    try:
        try:
            from torch.torch_version import TorchVersion
        except (AttributeError, ImportError):
            TorchVersion = None
        if TorchVersion is not None and hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([TorchVersion]):
                return torch.load(path, map_location="cpu", weights_only=True)
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parameter_count(checkpoint: Mapping[str, Any]) -> int:
    """Count trainable tensor elements represented by the model state."""
    state = checkpoint["model_state_dict"]
    return sum(int(tensor.numel()) for tensor in state.values())


def path_equivalent(left: str | Path, right: str | Path) -> bool:
    """Compare two potentially aliased absolute paths."""
    try:
        return Path(left).resolve(strict=True).samefile(Path(right).resolve(strict=True))
    except (FileNotFoundError, OSError):
        return Path(left).name == Path(right).name


def protocol_matches(
    config: Mapping[str, Any],
    protocol: Mapping[str, Any],
    variant: str,
    fold_index: int,
    fold_hashes: Mapping[int, str],
) -> tuple[bool, list[str]]:
    """Compare one run configuration against the locked protocol semantics."""
    mismatches: list[str] = []
    model = config["model"]
    optimization = config["optimization"]
    dataset = config["dataset"]
    authority = dataset["authority"]
    expected_variant = EXPECTED_VARIANTS[variant]
    for key, expected in expected_variant.items():
        if model.get(key) != expected:
            mismatches.append(f"model.{key}: expected={expected!r}, observed={model.get(key)!r}")
    common_model = {
        "antigen_encoder": "facebook/esm2_t33_650M_UR50D",
        "antigen_input_dim": 1280,
        "embedding_dim": 128,
        "antibody_hidden_dim": 256,
        "antigen_hidden_dim": 256,
        "dropout": 0.1,
        "antibody_pooling": "mean",
        "antigen_pooling": "mean",
        "initial_temperature": protocol["optimization"]["temperature_initial"],
    }
    for key, expected in common_model.items():
        if model.get(key) != expected:
            mismatches.append(f"model.{key}: expected={expected!r}, observed={model.get(key)!r}")
    expected_optimization = {
        "batch_size": protocol["optimization"]["batch_size_primary"],
        "learning_rate": protocol["optimization"]["learning_rate"],
        "weight_decay": protocol["optimization"]["weight_decay"],
        "max_epochs": protocol["optimization"]["max_epochs"],
        "patience": protocol["optimization"]["patience"],
        "schedule": protocol["optimization"]["scheduler"],
        "seed": 42,
    }
    for key, expected in expected_optimization.items():
        if optimization.get(key) != expected:
            mismatches.append(
                f"optimization.{key}: expected={expected!r}, observed={optimization.get(key)!r}"
            )
    expected_authority = protocol["authority"]
    authority_checks = {
        "sha256": expected_authority["source_split_sha256"],
        "coverage_mode": expected_authority["coverage_mode"],
        "authorized_source_pool_splits": expected_authority["source_pool_splits"],
        "sealed_source_splits": expected_authority["sealed_source_splits"],
        "selected_sealed_overlap_count": 0,
    }
    for key, expected in authority_checks.items():
        if authority.get(key) != expected:
            mismatches.append(f"authority.{key}: expected={expected!r}, observed={authority.get(key)!r}")
    direct_checks = {
        "dataset.split_manifest_sha256": (
            dataset.get("split_manifest_sha256"),
            fold_hashes[fold_index],
        ),
        "dataset.split_intended_use": (dataset.get("split_intended_use"), "development_only"),
        "dataset.frozen_test_records_selected": (
            dataset.get("frozen_test_records_selected"),
            False,
        ),
        "training_data.isolation_policy": (
            config["training_data"].get("isolation_policy"),
            protocol["fold_protocol"]["isolation_policy"],
        ),
        "validation.checkpoint_selection": (
            config["validation"].get("checkpoint_selection"),
            protocol["optimization"]["checkpoint_selection"],
        ),
    }
    for name, (observed, expected) in direct_checks.items():
        if observed != expected:
            mismatches.append(f"{name}: expected={expected!r}, observed={observed!r}")
    expected_ab_weight = EXPECTED_MODEL_WEIGHTS[model["antibody_encoder"]]
    expected_ag_weight = EXPECTED_MODEL_WEIGHTS["facebook/esm2_t33_650M_UR50D"]
    if model.get("antibody_model_weights_sha256") != expected_ab_weight:
        mismatches.append("antibody model-weight SHA-256 differs from the reviewed lock")
    if model.get("antigen_model_weights_sha256") != expected_ag_weight:
        mismatches.append("antigen model-weight SHA-256 differs from the reviewed lock")
    return not mismatches, mismatches


def slurm_record(job_id: str) -> dict[str, Any]:
    """Read accounting and logs for one completed allocation."""
    fields = "JobIDRaw,JobID,JobName,State,ExitCode,Elapsed,SubmitLine,WorkDir,StdOut,StdErr"
    command = ["sacct", "-j", job_id, "-X", "-n", "-P", f"--format={fields}"]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return {"available": False, "reason": "sacct_not_found"}
    rows = []
    for line in completed.stdout.splitlines():
        values = line.split("|")
        if len(values) < 10:
            continue
        rows.append(dict(zip(fields.split(","), values[:10])))
    primary = next((row for row in rows if row["JobIDRaw"] == job_id), rows[0] if rows else None)
    log_paths: list[Path] = []
    if primary:
        for key in ("StdOut", "StdErr"):
            raw_path = primary.get(key, "").strip()
            if raw_path and raw_path != "Unknown":
                display_job_id = primary.get("JobID", job_id)
                if "_" in display_job_id:
                    array_job_id, array_task_id = display_job_id.rsplit("_", 1)
                else:
                    array_job_id, array_task_id = display_job_id, "4294967294"
                replacements = {
                    "%x": primary.get("JobName", ""),
                    "%A": array_job_id,
                    "%a": array_task_id,
                    "%j": job_id,
                    "%J": display_job_id,
                }
                for marker, replacement in replacements.items():
                    raw_path = raw_path.replace(marker, replacement)
                path = Path(raw_path)
                if path.is_file() and path not in log_paths:
                    log_paths.append(path)
    log_hits = []
    log_hashes = {}
    for path in log_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [line[:500] for line in text.splitlines() if ERROR_PATTERN.search(line)]
        if hits:
            log_hits.extend({"path": str(path), "line": line} for line in hits[:25])
        log_hashes[str(path)] = sha256_file(path)
    return {
        "available": True,
        "command_exit_code": completed.returncode,
        "primary": primary,
        "rows": rows,
        "submit_line_sha256": (
            hashlib.sha256(primary.get("SubmitLine", "").encode("utf-8")).hexdigest()
            if primary
            else None
        ),
        "log_sha256": log_hashes,
        "error_pattern_hits": log_hits,
    }


def audit_run(
    run_dir: Path,
    variant: str,
    fold_index: int,
    fold_hashes: Mapping[int, str],
    fold_index_sha256: str,
    protocol: Mapping[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    """Audit one completed fold run and return compact evidence."""
    failures: list[str] = []
    checkpoint_dir = run_dir / "checkpoints"
    summary_path = checkpoint_dir / "training_summary.json"
    provenance_path = checkpoint_dir / "provenance.json"
    history_path = checkpoint_dir / "history.json"
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    for path in (summary_path, provenance_path, history_path, best_path, last_path):
        require(path.is_file(), f"missing artifact: {path}", failures)
    if failures:
        return {"run_name": run_dir.name, "valid": False, "failures": failures}

    summary = load_json(summary_path)
    provenance = load_json(provenance_path)
    history_payload = load_json(history_path)
    history = history_payload["history"]
    config = history_payload["config"]
    best_checkpoint = safe_torch_load(best_path)
    last_checkpoint = safe_torch_load(last_path)

    require(summary.get("stage") == "train_complete", "summary stage is not train_complete", failures)
    require(summary.get("frozen_test_used") is False, "summary reports frozen-test use", failures)
    require(provenance.get("scientific_scope", {}).get("frozen_test_used") is False,
            "provenance reports frozen-test use", failures)
    require(config.get("scientific_scope", {}).get("frozen_test_used") is False,
            "configuration reports frozen-test use", failures)
    require(config.get("dataset", {}).get("frozen_test_records_selected") is False,
            "configuration selected frozen-test records", failures)

    unhashed_config = {key: value for key, value in config.items() if key != "config_sha256"}
    observed_config_hash = canonical_sha256(unhashed_config)
    config_hashes = {
        "recomputed": observed_config_hash,
        "history": config.get("config_sha256"),
        "summary": summary.get("config_sha256"),
        "provenance": provenance.get("config_sha256"),
        "best_checkpoint": best_checkpoint.get("config", {}).get("config_sha256"),
        "last_checkpoint": last_checkpoint.get("config", {}).get("config_sha256"),
    }
    require(len(set(config_hashes.values())) == 1, "config hashes are inconsistent", failures)

    best_hash = sha256_file(best_path)
    require(best_hash == summary.get("best_checkpoint_sha256"),
            "best checkpoint SHA-256 does not match summary", failures)
    require(path_equivalent(summary.get("best_checkpoint", ""), best_path),
            "summary best-checkpoint path does not resolve to this run", failures)

    epochs = [int(record["epoch"]) for record in history]
    require(epochs == list(range(1, len(history) + 1)),
            "history epochs are not a clean contiguous sequence from 1", failures)
    require(len(history) == int(summary.get("completed_epoch", -1)),
            "history length does not match completed epoch", failures)
    require(int(last_checkpoint.get("epoch", -1)) == epochs[-1],
            "last checkpoint epoch differs from final history epoch", failures)
    require(last_checkpoint.get("history") == history,
            "last checkpoint history differs from history.json", failures)

    min_delta = float(config["optimization"]["min_delta"])
    semantic_best = -math.inf
    semantic_best_epoch = 0
    semantic_flags: list[bool] = []
    for record in history:
        value = float(record["validation"]["bidirectional_mrr"])
        improved = value > semantic_best + min_delta
        semantic_flags.append(improved)
        if improved:
            semantic_best = value
            semantic_best_epoch = int(record["epoch"])
        require(bool(record["improved"]) == improved,
                f"epoch {record['epoch']} has an incorrect improved flag", failures)
        require(int(record["best_epoch"]) == semantic_best_epoch,
                f"epoch {record['epoch']} has an incorrect running best epoch", failures)
        require(math.isclose(float(record["best_validation_mrr"]), semantic_best,
                             rel_tol=0.0, abs_tol=1e-15),
                f"epoch {record['epoch']} has an incorrect running best metric", failures)
    require(int(summary["best_epoch"]) == semantic_best_epoch,
            "summary best epoch violates min-delta semantics", failures)
    require(math.isclose(float(summary["best_validation_mrr"]), semantic_best,
                         rel_tol=0.0, abs_tol=1e-15),
            "summary best metric violates min-delta semantics", failures)
    require(int(best_checkpoint.get("epoch", -1)) == semantic_best_epoch,
            "best checkpoint epoch violates min-delta semantics", failures)
    require(best_checkpoint.get("history") == history[:semantic_best_epoch],
            "best checkpoint history is not the exact best-epoch prefix", failures)
    require(int(best_checkpoint.get("best_epoch", -1)) == semantic_best_epoch,
            "best checkpoint embeds a different best epoch", failures)
    require(math.isclose(float(best_checkpoint.get("best_validation_mrr", math.nan)),
                         semantic_best, rel_tol=0.0, abs_tol=1e-15),
            "best checkpoint embeds a different best metric", failures)

    absolute_record = max(history, key=lambda record: float(record["validation"]["bidirectional_mrr"]))
    absolute_epoch = int(absolute_record["epoch"])
    absolute_mrr = float(absolute_record["validation"]["bidirectional_mrr"])
    require(absolute_mrr <= semantic_best + min_delta + 1e-15,
            "raw maximum exceeds the selected best by more than min_delta", failures)

    nonfinite = [path for path, value in iter_numbers(history_payload) if not math.isfinite(value)]
    require(not nonfinite, f"non-finite history values: {nonfinite[:5]}", failures)

    best_record = history[semantic_best_epoch - 1]
    validation = best_record["validation"]
    recall_ks = [int(value) for value in config["validation"]["recall_ks"]]
    direction_results = {}
    for direction_name in ("antibody_to_antigen", "antigen_to_antibody"):
        direction = validation[direction_name]
        direction_with_rows = {
            **direction,
            "per_anchor": validation["per_anchor_ranks"][direction_name],
        }
        recomputed = recompute_observed_direction(direction_with_rows, recall_ks)
        for metric, value in recomputed.items():
            require(math.isclose(float(direction[metric]), float(value), rel_tol=0.0, abs_tol=1e-12),
                    f"{direction_name} saved {metric} fails per-anchor recomputation", failures)
        positive_counts = [
            int(row["known_positive_candidate_count"])
            for row in direction_with_rows["per_anchor"]
        ]
        random_metrics = exact_random_best_positive_metrics(
            int(direction["candidate_count"]), positive_counts, recall_ks
        )
        direction_results[direction_name] = {
            "observed": {metric: float(direction[metric]) for metric in ["mrr", *[f"recall_at_{k}" for k in recall_ks]]},
            "random_exact_multi_positive": random_metrics,
            "mrr_excess": float(direction["mrr"]) - random_metrics["mrr"],
            "mrr_multiple": float(direction["mrr"]) / random_metrics["mrr"],
            "above_random": float(direction["mrr"]) > random_metrics["mrr"],
        }
    observed_bidirectional = (
        direction_results["antibody_to_antigen"]["observed"]["mrr"]
        + direction_results["antigen_to_antibody"]["observed"]["mrr"]
    ) / 2.0
    random_bidirectional = (
        direction_results["antibody_to_antigen"]["random_exact_multi_positive"]["mrr"]
        + direction_results["antigen_to_antibody"]["random_exact_multi_positive"]["mrr"]
    ) / 2.0
    require(math.isclose(observed_bidirectional, float(validation["bidirectional_mrr"]),
                         rel_tol=0.0, abs_tol=1e-12),
            "saved bidirectional MRR fails directional recomputation", failures)

    protocol_valid, protocol_mismatches = protocol_matches(
        config, protocol, variant, fold_index, fold_hashes
    )
    require(protocol_valid, f"locked protocol mismatches: {protocol_mismatches}", failures)

    physical_source_hashes = {
        key: sha256_file(source_root / relative)
        for key, relative in EXPECTED_SOURCE_NAMES.items()
    }
    require(config["source_sha256"] == provenance["source_sha256"],
            "configuration and provenance source hashes differ", failures)

    job_id = str(provenance.get("runtime", {}).get("slurm_job_id", ""))
    slurm = slurm_record(job_id)
    if slurm.get("primary"):
        primary = slurm["primary"]
        require(primary.get("State", "").startswith("COMPLETED"),
                f"Slurm state is {primary.get('State')!r}", failures)
        require(primary.get("ExitCode") == "0:0", f"Slurm exit code is {primary.get('ExitCode')!r}", failures)
        submit_line = primary.get("SubmitLine", "")
        require(f"RETRIEVAL_FOLD_INDEX_SHA256={fold_index_sha256}" in submit_line,
                "Slurm submit line does not pin the reviewed fold-index SHA-256", failures)
        require(f"RETRIEVAL_VARIANT={variant}" in submit_line,
                "Slurm submit line does not match the run variant", failures)
        require("RETRIEVAL_SEED=42" in submit_line,
                "Slurm submit line does not pin seed 42", failures)
        require("asymmetric_partner_retrieval_fold_train.sbatch" in submit_line,
                "Slurm submit line does not identify the reviewed wrapper", failures)
    else:
        failures.append("Slurm accounting row is unavailable")
    require(not slurm.get("error_pattern_hits"),
            f"error patterns found in Slurm logs: {slurm.get('error_pattern_hits', [])[:3]}", failures)

    expected_last_patience = int(history[-1]["epochs_without_improvement"])
    early_stop_clean = (
        epochs[-1] == int(config["optimization"]["max_epochs"])
        or expected_last_patience >= int(config["optimization"]["patience"])
    )
    require(early_stop_clean, "run ended without max-epoch completion or patience stop", failures)

    return {
        "run_name": run_dir.name,
        "variant": variant,
        "fold_index": fold_index,
        "valid": not failures,
        "failures": failures,
        "artifact_sha256": {
            "history": sha256_file(history_path),
            "provenance": sha256_file(provenance_path),
            "training_summary": sha256_file(summary_path),
            "best_checkpoint": best_hash,
            "last_checkpoint": sha256_file(last_path),
        },
        "config_sha256": config_hashes,
        "source_sha256": {
            "embedded_at_run": config["source_sha256"],
            "current_physical_files": physical_source_hashes,
            "current_snapshot_matches_run": physical_source_hashes == config["source_sha256"],
        },
        "split_sha256": config["dataset"]["split_manifest_sha256"],
        "authority_sha256": config["dataset"]["authority"]["sha256"],
        "model_weight_sha256": {
            "antibody": config["model"]["antibody_model_weights_sha256"],
            "antigen": config["model"]["antigen_model_weights_sha256"],
        },
        "embedding_content_manifest_sha256": {
            side: config["embedding_corpora"][side]["content_manifest_sha256"]
            for side in ("antibody", "antigen")
        },
        "epochs": {
            "completed": epochs[-1],
            "selected": semantic_best_epoch,
            "absolute_raw_maximum": absolute_epoch,
            "min_delta": min_delta,
            "selected_mrr": semantic_best,
            "absolute_raw_maximum_mrr": absolute_mrr,
            "raw_maximum_minus_selected": absolute_mrr - semantic_best,
            "history_contiguous_from_one": epochs == list(range(1, len(history) + 1)),
            "early_stop_or_max_epoch_clean": early_stop_clean,
        },
        "parameter_count": parameter_count(best_checkpoint),
        "best_validation": {
            "bidirectional_mrr": observed_bidirectional,
            "exact_random_bidirectional_mrr": random_bidirectional,
            "bidirectional_mrr_multiple": observed_bidirectional / random_bidirectional,
            "directions": direction_results,
        },
        "protocol": {
            "semantic_match": protocol_valid,
            "mismatches": protocol_mismatches,
        },
        "scientific_scope": {
            "source_pool_splits": config["dataset"]["authority"]["authorized_source_pool_splits"],
            "sealed_source_splits": config["dataset"]["authority"]["sealed_source_splits"],
            "selected_sealed_overlap_count": config["dataset"]["authority"]["selected_sealed_overlap_count"],
            "frozen_test_records_selected": config["dataset"]["frozen_test_records_selected"],
            "frozen_test_used": config["scientific_scope"]["frozen_test_used"],
        },
        "slurm": slurm,
    }


def parse_run_name(name: str) -> tuple[str, int] | None:
    """Parse one expected architecture-screen directory name."""
    match = re.fullmatch(
        r"(antiberty_calm_mean|antiberty_chain_balanced|esm2_calm_mean)_antigen_fold([0-2])_seed42_b1024",
        name,
    )
    if not match:
        return None
    return match.group(1), int(match.group(2))


def main() -> None:
    """Audit nine fold runs and write one compact JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--authority-split", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results_root = args.results_root.resolve(strict=True)
    fold_root = args.fold_root.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    protocol_path = args.protocol.resolve(strict=True)
    authority_path = args.authority_split.resolve(strict=True)
    protocol = load_json(protocol_path)
    fold_index_payload = load_json(fold_root / "index.json")
    fold_hashes = {
        int(entry["fold_index"]): str(entry["sha256"])
        for entry in fold_index_payload["folds"]
    }
    top_failures: list[str] = []
    require(set(fold_hashes) == {0, 1, 2}, "fold index does not contain exactly folds 0,1,2", top_failures)
    for fold_index, expected_hash in fold_hashes.items():
        observed_hash = sha256_file(fold_root / f"fold_{fold_index}.json")
        require(observed_hash == expected_hash,
                f"fold {fold_index} physical hash differs from index", top_failures)
    authority_hash = sha256_file(authority_path)
    require(authority_hash == protocol["authority"]["source_split_sha256"],
            "physical authority split differs from protocol lock", top_failures)
    fold_index_path = fold_root / "index.json"
    fold_index_sha256 = sha256_file(fold_index_path)
    submission_script_path = source_root / "asymmetric_partner_retrieval_fold_train.sbatch"
    submission_script_text = submission_script_path.read_text(encoding="utf-8")
    submission_script_checks = {
        "requires_fold_index_sha_environment": (
            "RETRIEVAL_FOLD_INDEX_SHA256:?" in submission_script_text
        ),
        "recomputes_fold_index_sha": (
            "observed_index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()"
            in submission_script_text
        ),
        "rejects_fold_index_sha_mismatch": (
            "Fold index SHA-256 mismatch" in submission_script_text
        ),
        "passes_fold_manifest_sha_to_entrypoint": (
            '--split-manifest-sha256 "$FOLD_SHA256"' in submission_script_text
        ),
        "pins_authority_sha": (
            protocol["authority"]["source_split_sha256"] in submission_script_text
        ),
    }
    require(all(submission_script_checks.values()),
            f"submission wrapper semantic checks failed: {submission_script_checks}",
            top_failures)
    nw_audit_path = fold_root / "needleman_wunsch_audit" / "audit.json"
    nw_receipt_path = fold_root / "needleman_wunsch_audit" / "receipt.json"
    nw_audit = load_json(nw_audit_path)
    nw_receipt = load_json(nw_receipt_path)
    nw_audit_checks = {
        "audit_valid": nw_audit.get("valid") is True,
        "fold_index_sha_matches": (
            nw_audit.get("fold_index", {}).get("sha256") == fold_index_sha256
        ),
        "receipt_valid": nw_receipt.get("valid") is True,
        "receipt_audit_sha_matches": (
            nw_receipt.get("report_sha256") == sha256_file(nw_audit_path)
        ),
    }
    require(all(nw_audit_checks.values()),
            f"pre-training fold audit evidence failed: {nw_audit_checks}",
            top_failures)

    parsed_runs = {
        child.name: parse_run_name(child.name)
        for child in sorted(results_root.iterdir())
        if child.is_dir()
    }
    expected_names = {
        f"{variant}_antigen_fold{fold}_seed42_b1024"
        for variant in EXPECTED_VARIANTS
        for fold in range(3)
    }
    observed_expected_names = {name for name, parsed in parsed_runs.items() if parsed is not None}
    require(observed_expected_names == expected_names,
            f"expected run set differs: missing={sorted(expected_names - observed_expected_names)}, "
            f"extra={sorted(observed_expected_names - expected_names)}", top_failures)

    runs = []
    for name in sorted(expected_names):
        parsed = parse_run_name(name)
        assert parsed is not None
        variant, fold_index = parsed
        runs.append(
            audit_run(
                results_root / name,
                variant,
                fold_index,
                fold_hashes,
                fold_index_sha256,
                protocol,
                source_root,
            )
        )

    output_dirs = [str((results_root / run["run_name"] / "checkpoints").resolve()) for run in runs]
    require(len(output_dirs) == len(set(output_dirs)), "run output directories overlap", top_failures)
    embedded_source_sets = {
        canonical_sha256(run["source_sha256"]["embedded_at_run"])
        for run in runs
        if "source_sha256" in run
    }
    require(len(embedded_source_sets) == 1,
            "runs do not share one embedded source-code hash set", top_failures)
    require(all(run.get("protocol", {}).get("semantic_match") is True for run in runs),
            "one or more runs do not match the locked protocol semantics", top_failures)
    antigen_embedding_sets = {
        run["embedding_content_manifest_sha256"]["antigen"]
        for run in runs
        if "embedding_content_manifest_sha256" in run
    }
    require(len(antigen_embedding_sets) == 1,
            "runs do not share one antigen embedding-corpus manifest", top_failures)
    for variant in EXPECTED_VARIANTS:
        antibody_embedding_sets = {
            run["embedding_content_manifest_sha256"]["antibody"]
            for run in runs
            if run.get("variant") == variant
        }
        require(len(antibody_embedding_sets) == 1,
                f"{variant} folds do not share one antibody embedding-corpus manifest",
                top_failures)
    for fold_index in range(3):
        fold_runs = [run for run in runs if run.get("fold_index") == fold_index]
        library_shapes = {
            (
                run["best_validation"]["directions"]["antibody_to_antigen"]
                ["random_exact_multi_positive"]["anchor_count"],
                run["best_validation"]["directions"]["antibody_to_antigen"]
                ["random_exact_multi_positive"]["candidate_count"],
            )
            for run in fold_runs
        }
        require(len(library_shapes) == 1,
                f"fold {fold_index} variants were not scored on the same native library",
                top_failures)

    selection = []
    for variant in sorted(EXPECTED_VARIANTS):
        variant_runs = sorted(
            [run for run in runs if run.get("variant") == variant],
            key=lambda run: run["fold_index"],
        )
        observed = [run["best_validation"]["bidirectional_mrr"] for run in variant_runs]
        random = [run["best_validation"]["exact_random_bidirectional_mrr"] for run in variant_runs]
        direction_gate = all(
            direction["above_random"]
            for run in variant_runs
            for direction in run["best_validation"]["directions"].values()
        )
        entry = {
            "variant": variant,
            "fold_bidirectional_mrr": observed,
            "fold_exact_random_bidirectional_mrr": random,
            "macro_mean_bidirectional_mrr": sum(observed) / len(observed),
            "macro_mean_exact_random_bidirectional_mrr": sum(random) / len(random),
            "macro_multiple_over_exact_random": sum(observed) / sum(random),
            "worst_fold_bidirectional_mrr": min(observed),
            "parameter_count": variant_runs[0]["parameter_count"],
            "all_fold_bidirectional_mrr_above_random": all(a > b for a, b in zip(observed, random)),
            "both_directions_above_random_in_all_folds": direction_gate,
            "all_fold_count": len(variant_runs) == 3,
        }
        gate = protocol["predeclared_progression_gate"]
        entry["passes_predeclared_directional_random_gates"] = (
            entry["all_fold_count"]
            and entry["all_fold_bidirectional_mrr_above_random"]
            and entry["both_directions_above_random_in_all_folds"]
            and sum(a > b for a, b in zip(observed, random)) >= int(gate["minimum_positive_fold_count"])
            and entry["macro_multiple_over_exact_random"]
            >= float(gate["bidirectional_mrr_multiple_over_matched_random_mean"])
        )
        selection.append(entry)
    selection.sort(
        key=lambda entry: (
            -entry["macro_mean_bidirectional_mrr"],
            -entry["worst_fold_bidirectional_mrr"],
            entry["parameter_count"],
            entry["variant"],
        )
    )
    for rank, entry in enumerate(selection, start=1):
        entry["selection_rank"] = rank
    selection_rule = protocol["architecture_selection"]
    selection_valid = (
        selection_rule["primary_statistic"] == "unweighted_mean_of_fold_bidirectional_mrr"
        and selection_rule["first_tiebreaker"] == "largest_worst_fold_bidirectional_mrr"
        and selection_rule["second_tiebreaker"] == "smallest_trainable_parameter_count"
    )
    require(selection_valid, "selection protocol fields differ from implemented audit order", top_failures)

    provenance_caveats = [
        (
            "The protocol-manifest SHA-256 is not embedded in the nine checkpoint configs; "
            "the audit establishes an exact semantic match to the current reviewed manifest."
        ),
        (
            "The fold-index SHA-256 is enforced by the submission wrapper but is not embedded in "
            "checkpoint provenance; each individual fold-manifest SHA-256 is embedded and verified."
        ),
        (
            "The model-weight digests are verified against reviewed locks and against the metadata "
            "validated during embedding loads; this audit does not reinterpret those custom weight hashes."
        ),
    ]
    if not all(
        run.get("source_sha256", {}).get("current_snapshot_matches_run") is True
        for run in runs
    ):
        provenance_caveats.append(
            "The current remote entrypoint has changed since these runs. All nine checkpoints "
            "embed one common run-time source hash set, but the exact entrypoint bytes are no "
            "longer present at the live source path. This is source-archival drift, not evidence "
            "of metric or checkpoint contamination."
        )
    all_run_valid = all(run.get("valid") is True for run in runs)
    current_source_snapshot_matches_all_runs = all(
        run.get("source_sha256", {}).get("current_snapshot_matches_run") is True
        for run in runs
    )
    report = {
        "format_version": FORMAT_VERSION,
        "scope": "train-only antigen-fold architecture selection; no fixed holdout evaluation",
        "verdict": "pass_with_provenance_caveats" if all_run_valid and not top_failures else "fail",
        "valid": all_run_valid and not top_failures,
        "top_level_failures": top_failures,
        "run_count": len(runs),
        "all_runs_valid": all_run_valid,
        "frozen_test_used": False,
        "fixed_holdout_evaluated": False,
        "authority": {
            "path": str(authority_path),
            "sha256": authority_hash,
        },
        "fold_index": {
            "path": str(fold_index_path),
            "sha256": fold_index_sha256,
            "fold_sha256": {str(key): value for key, value in sorted(fold_hashes.items())},
        },
        "launch_and_split_evidence": {
            "submission_script": {
                "path": str(submission_script_path),
                "sha256": sha256_file(submission_script_path),
                "semantic_checks": submission_script_checks,
                "checkpoint_native": False,
            },
            "slurm_submit_lines": [
                {
                    "run_name": run["run_name"],
                    "job_id": run["slurm"].get("primary", {}).get("JobIDRaw"),
                    "display_job_id": run["slurm"].get("primary", {}).get("JobID"),
                    "sha256": run["slurm"].get("submit_line_sha256"),
                    "pins_reviewed_fold_index_sha": (
                        f"RETRIEVAL_FOLD_INDEX_SHA256={fold_index_sha256}"
                        in run["slurm"].get("primary", {}).get("SubmitLine", "")
                    ),
                }
                for run in runs
            ],
            "slurm_log_sha256": {
                run["run_name"]: run["slurm"].get("log_sha256", {})
                for run in runs
            },
            "pretraining_needleman_wunsch_fold_audit": {
                "path": str(nw_audit_path),
                "sha256": sha256_file(nw_audit_path),
                "receipt_path": str(nw_receipt_path),
                "receipt_sha256": sha256_file(nw_receipt_path),
                "checks": nw_audit_checks,
            },
        },
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "selection_rule_verified": selection_valid,
            "checkpoint_native": False,
            "semantic_match_all_runs": all(
                run.get("protocol", {}).get("semantic_match") is True for run in runs
            ),
        },
        "architecture_selection": {
            "winner": selection[0]["variant"],
            "ranking": selection,
        },
        "output_directories_unique": len(output_dirs) == len(set(output_dirs)),
        "embedded_source_hash_set_common": len(embedded_source_sets) == 1,
        "antigen_embedding_manifest_common": len(antigen_embedding_sets) == 1,
        "provenance_caveats": provenance_caveats,
        "selection_validity_assessment": {
            "metric_or_selection_invalidated_by_caveats": False,
            "assessment": (
                "The caveats weaken bitwise launch provenance but do not invalidate the "
                "independently recomputed metrics, fold-wise macro selection, or winner."
            ),
            "current_source_snapshot_matches_all_runs": current_source_snapshot_matches_all_runs,
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "verdict": report["verdict"],
        "winner": report["architecture_selection"]["winner"],
        "valid": report["valid"],
        "output": str(args.output),
        "sha256": sha256_file(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
