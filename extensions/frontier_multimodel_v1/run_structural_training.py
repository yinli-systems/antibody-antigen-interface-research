#!/usr/bin/env python3
"""Run one frozen SaProt/ProSST downstream training task."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

CURRENT_OUTPUT: Path | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def inventory(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
        count += 1
        total += stat.st_size
    return {
        "path": str(root.resolve()), "file_count": count, "total_bytes": total,
        "path_size_mtime_inventory_sha256": digest.hexdigest(), "content_digest": False,
    }


def verify_plan(plan_path: Path, external_sha256: str) -> dict[str, Any]:
    if sha256(plan_path) != external_sha256:
        raise RuntimeError("external plan digest mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_root = Path(plan["source_root"])
    for relative, expected in plan["source_sha256"].items():
        actual = sha256(source_root / relative)
        if actual != expected:
            raise RuntimeError(f"source changed: {relative}: {actual} != {expected}")
    for raw_path, expected in plan["input_sha256"].items():
        path = Path(raw_path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"input changed: {path}: {actual} != {expected}")
    for label in ("contact_dir", "embedding_dir"):
        root = Path(plan["large_base_store_inventory"][label]["path"])
        current = inventory(root)
        frozen = plan["large_base_store_inventory"][label]
        if current != frozen:
            raise RuntimeError(f"large base-store inventory changed: {label}")
    if plan.get("test_authorized") or plan.get("ood_authorized"):
        raise RuntimeError("plan unexpectedly authorizes test/OOD access")
    return plan


def evaluation_negative_count(smoke: bool) -> int:
    """Small smoke subsets use one distractor; formal 217-case runs retain nine."""
    return 1 if smoke else 9


def representation_args(task: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    args = ["--representation-manifest", plan["paths"]["canonical_manifest"]]
    for spec in task["specs"]:
        args.extend(["--generic-representation-source", spec])
    if task["random_projection_control"]:
        args.extend([
            "--random-projection-control",
            "--random-projection-seed", str(plan["training"]["random_projection_seed"]),
        ])
    return args


def run_checked(command: list[str], cwd: Path, commands_path: Path) -> None:
    with commands_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(command) + "\n")
    print("[command]", " ".join(command), flush=True)
    environment = os.environ.copy()
    source_python = str((cwd / "src").resolve())
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_python if not previous else f"{source_python}:{previous}"
    subprocess.run(command, cwd=cwd, check=True, env=environment)


def read_evaluation(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "evaluate" or payload.get("split") != "validation":
        raise RuntimeError(f"invalid evaluation receipt: {path}")
    if payload.get("instance_count", 0) <= 0:
        raise RuntimeError(f"empty evaluation receipt: {path}")
    pair = payload.get("pair", {}).get("macro", {})
    if not isinstance(pair.get("auprc"), (int, float)):
        raise RuntimeError(f"missing primary metric: {path}")
    return payload


def main() -> None:
    global CURRENT_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    began = time.monotonic()
    plan = verify_plan(args.plan.resolve(), args.plan_sha256)
    tasks = {int(task["task_id"]): task for task in plan["tasks"]}
    if args.task_id not in tasks:
        raise ValueError(f"unknown task {args.task_id}")
    task = tasks[args.task_id]
    if args.smoke and args.task_id != int(plan["smoke"]["task_id"]):
        raise ValueError("smoke task does not match frozen plan")

    task_name = f"smoke_task_{args.task_id}" if args.smoke else f"task_{args.task_id:02d}"
    output = args.attempt_root.resolve() / "runs" / task_name
    output.mkdir(parents=True, exist_ok=False)
    CURRENT_OUTPUT = output
    exclusive_json(output / "START.json", {
        "status": "STARTED", "task": task, "smoke": args.smoke,
        "plan_sha256": args.plan_sha256, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "test_authorized": False, "ood_authorized": False,
    })

    python = plan["paths"]["python_bin"]
    source_root = Path(plan["source_root"])
    driver = str(source_root / "scripts/sabdab2_interaction_first.py")
    data = plan["paths"]
    training = plan["training"]
    max_epochs = plan["smoke"]["max_epochs"] if args.smoke else training["max_epochs"]

    common = [
        "--prepare-dir", data["prepare_dir"],
        "--contact-dir", data["contact_dir"],
        "--embedding-dir", data["embedding_dir"],
        "--split-manifest", data["split_manifest"],
    ]
    representation = representation_args(task, plan)
    train = [
        python, "-u", driver, "train", *common, *representation,
        "--output-dir", str(output / "train"),
        "--seed", str(task["seed"]), "--device", "cuda", "--no-resume",
        "--input-dim", "1280",
        "--d-model", str(training["d_model"]),
        "--pair-dim", str(training["pair_dim"]),
        "--num-heads", str(training["num_heads"]),
        "--num-layers", str(training["num_layers"]),
        "--dropout", str(training["dropout"]),
        "--marginal-pooling", training["marginal_pooling"],
        "--score-pooling", training["score_pooling"],
        "--negative-count", str(training["negative_count"]),
        "--negative-directions", training["negative_directions"],
        "--hard-pool-size", str(training["hard_pool_size"]),
        "--partner-rank-weight", str(training["partner_rank_weight"]),
        "--learning-rate", str(training["learning_rate"]),
        "--minimum-learning-rate", str(training["minimum_learning_rate"]),
        "--weight-decay", str(training["weight_decay"]),
        "--gradient-clip", str(training["gradient_clip"]),
        "--max-epochs", str(max_epochs),
        "--patience", str(1 if args.smoke else training["patience"]),
        "--scheduler-factor", str(training["scheduler_factor"]),
        "--scheduler-patience", str(training["scheduler_patience"]),
        "--selection-metric", training["selection_metric"],
        "--pair-decoder", training["pair_decoder"],
        "--embedding-cache-gib", "12", "--log-every", "12" if args.smoke else "100",
    ]
    if args.smoke:
        train.extend([
            "--max-train-instances", str(plan["smoke"]["max_train_instances"]),
            "--max-validation-instances", str(plan["smoke"]["max_validation_instances"]),
        ])
    commands_path = output / "COMMANDS.jsonl"
    run_checked(train, source_root, commands_path)

    evaluations = {}
    for checkpoint_name, directory in (("best.pt", "validation_best"), ("last.pt", "validation_last")):
        checkpoint = output / "train" / checkpoint_name
        if not checkpoint.is_file():
            raise RuntimeError(f"training did not write {checkpoint}")
        evaluate = [
            python, "-u", driver, "evaluate", *common, *representation,
            "--checkpoint", str(checkpoint),
            "--output-dir", str(output / directory),
            "--split", "validation", "--device", "cuda",
            "--negative-count", str(evaluation_negative_count(args.smoke)),
            "--seed", str(task["seed"]), "--embedding-cache-gib", "12",
            "--log-every", "12" if args.smoke else "100",
        ]
        if args.smoke:
            evaluate.extend(["--max-instances", str(plan["smoke"]["max_validation_instances"])])
        run_checked(evaluate, source_root, commands_path)
        evaluations[directory] = read_evaluation(output / directory / "evaluation.json")

    config = json.loads((output / "train" / "history.json").read_text(encoding="utf-8"))["config"]
    dim = 1280 if task["family"] == "saprot" else 768
    expected_width = 1280 + dim + 1
    if int(config["antibody_input_dim"]) != expected_width or int(config["antigen_input_dim"]) != expected_width:
        raise RuntimeError(
            f"input width mismatch: {config['antibody_input_dim']}/{config['antigen_input_dim']} != {expected_width}"
        )
    sources = config.get("representations", {}).get("sources", [])
    if len(sources) != 2:
        raise RuntimeError(f"expected separate antibody/antigen sources, got {sources}")
    if any(int(source["representation_dim"]) != dim for source in sources):
        raise RuntimeError("representation dimension mismatch")
    if task["random_projection_control"]:
        if any(source.get("kind") != "random_projection_of_base" for source in sources):
            raise RuntimeError("control task loaded stored features instead of random projections")
    else:
        expected = {plan["generic_source_metadata"][spec]["name"] for spec in task["specs"]}
        observed = {source.get("name") for source in sources}
        if observed != expected:
            raise RuntimeError(f"generic source identity mismatch: {observed} != {expected}")
        for source in sources:
            metadata = next(
                plan["generic_source_metadata"][spec]
                for spec in task["specs"]
                if plan["generic_source_metadata"][spec]["name"] == source["name"]
            )
            if source.get("expected_model_name") != metadata["model_name"]:
                raise RuntimeError("model-name authority mismatch")
            if source.get("expected_model_revision") != metadata["model_revision"]:
                raise RuntimeError("model-revision authority mismatch")

    # Verify no selected source/input changed while training.
    verify_plan(args.plan.resolve(), args.plan_sha256)
    receipt = {
        "status": "PASS_ENGINEERING_SMOKE" if args.smoke else "COMPLETE_DEVELOPMENT_TRAINING",
        "task": task, "smoke": args.smoke, "plan_sha256": args.plan_sha256,
        "source_commit": plan["source_commit"], "expected_per_side_width": expected_width,
        "observed_antibody_input_dim": int(config["antibody_input_dim"]),
        "observed_antigen_input_dim": int(config["antigen_input_dim"]),
        "parameter_count": int(config["parameter_count"]),
        "best_checkpoint_sha256": sha256(output / "train" / "best.pt"),
        "last_checkpoint_sha256": sha256(output / "train" / "last.pt"),
        "best_primary_pair_ap": evaluations["validation_best"]["pair"]["macro"]["auprc"],
        "last_primary_pair_ap": evaluations["validation_last"]["pair"]["macro"]["auprc"],
        "best_instance_count": evaluations["validation_best"]["instance_count"],
        "last_instance_count": evaluations["validation_last"]["instance_count"],
        "wall_seconds": time.monotonic() - began,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "training_performed": True, "test_labels_read": False, "ood_labels_read": False,
        "scientific_result": not args.smoke, "blind_claim_authorized": False,
        "sota_claim_authorized": False,
    }
    exclusive_json(output / "COMPLETE.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if CURRENT_OUTPUT is not None and CURRENT_OUTPUT.exists():
            failure = CURRENT_OUTPUT / "FAILURE.json"
            if not failure.exists():
                exclusive_json(failure, {
                    "status": "FAILED", "error_type": type(error).__name__,
                    "error": str(error), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "test_labels_read": False, "ood_labels_read": False,
                    "sota_claim_authorized": False,
                })
        print(f"[fatal] {type(error).__name__}: {error}", flush=True)
        raise
