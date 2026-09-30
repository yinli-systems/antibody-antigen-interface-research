#!/usr/bin/env python3
"""Hash-bound, matched multi-seed EPIC reviewer-ablation trainer.

This runner deliberately reuses the frozen post-submission official-455 AF3
substitution authority. It never opens campaign test labels and never claims to
reproduce the unavailable submitted OpenDDE checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_parent_protocol_module(parent: Path) -> Any:
    module_path = parent / "runner" / "protocol.py"
    spec = importlib.util.spec_from_file_location("epic_parent_protocol", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load parent protocol helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-protocol", required=True, type=Path)
    parser.add_argument("--campaign-protocol-sha256", required=True)
    parser.add_argument("--authority-receipt", required=True, type=Path)
    parser.add_argument("--authority-receipt-sha256", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--smoke-receipt-sha256")
    return parser.parse_args()


def validate_authorities(arguments: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path, Any]:
    campaign_path = arguments.campaign_protocol.resolve()
    require(sha256_file(campaign_path) == arguments.campaign_protocol_sha256,
            "Campaign protocol hash differs")
    campaign = read_json(campaign_path)
    require(campaign.get("status") == "PREREGISTERED_BEFORE_GPU_EXECUTION",
            "Campaign is not preregistered")
    require(sha256_file(Path(__file__).resolve()) == campaign["implementation"]["train_condition.py"],
            "Campaign trainer differs from preregistered implementation")
    conditions = {item["name"]: item for item in campaign["conditions"]}
    require(arguments.condition in conditions, "Condition was not preregistered")
    require(arguments.seed in campaign["seeds"], "Seed was not preregistered")

    receipt_path = arguments.authority_receipt.resolve()
    require(sha256_file(receipt_path) == arguments.authority_receipt_sha256,
            "Authority receipt hash differs")
    authority_receipt = read_json(receipt_path)
    require(authority_receipt.get("status") == "PASS_FULL_PARENT_BUNDLE_VERIFICATION",
            "Parent authority verification did not pass")
    require(authority_receipt.get("campaign_protocol_sha256") == arguments.campaign_protocol_sha256,
            "Authority receipt belongs to another campaign")

    parent = Path(campaign["parent_authority"]["root"]).resolve()
    require(parent.is_dir() and not parent.is_symlink(), "Parent authority root is unavailable")
    require(sha256_file(parent / "BUNDLE.json") == campaign["parent_authority"]["bundle_json_sha256"],
            "Parent BUNDLE.json changed")
    require(authority_receipt.get("parent_bundle_json_sha256") == campaign["parent_authority"]["bundle_json_sha256"],
            "Authority receipt references another parent bundle")
    parent_protocol = load_parent_protocol_module(parent)

    if not arguments.smoke:
        require(arguments.smoke_receipt is not None and arguments.smoke_receipt_sha256 is not None,
                "Main training requires an exact-condition smoke receipt")
        smoke_path = arguments.smoke_receipt.resolve()
        require(sha256_file(smoke_path) == arguments.smoke_receipt_sha256,
                "Smoke receipt hash differs")
        smoke = read_json(smoke_path)
        require(smoke.get("status") == "PASS_CONDITION_SMOKE", "Smoke did not pass")
        require(smoke.get("condition") == arguments.condition, "Smoke condition differs")
        require(smoke.get("campaign_protocol_sha256") == arguments.campaign_protocol_sha256,
                "Smoke belongs to another campaign")
    return campaign, conditions[arguments.condition], parent, parent_protocol


def state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def freeze_geometry(model: Any) -> None:
    assert model.interleaved_self_encoders is not None
    for encoder in model.interleaved_self_encoders:
        for stream in (encoder.antibody_encoder, encoder.antigen_encoder):
            with __import__("torch").no_grad():
                stream.geometry_residual_gate.zero_()
            stream.geometry_residual_gate.requires_grad_(False)


def freeze_cross_attention(model: Any) -> None:
    assert model.interleaved_cross_attentions is not None
    torch = __import__("torch")
    for attention in model.interleaved_cross_attentions:
        for direction in (attention.antibody_from_antigen, attention.antigen_from_antibody):
            with torch.no_grad():
                direction.residual_gate.zero_()
            direction.residual_gate.requires_grad_(False)


def remove_pair_interaction(model: Any) -> None:
    torch = __import__("torch")
    directional = getattr(model.contact_head, "directional", None)
    require(directional is not None, "Unary pair ablation requires symmetric directional head")
    for layer in (directional.antibody_query, directional.antigen_key):
        with torch.no_grad():
            layer.weight.zero_()
        layer.weight.requires_grad_(False)


def remove_residue_contact_readout(model: Any) -> None:
    torch = __import__("torch")
    # The frozen EPIC-style forward adds contact marginals directly. Switching
    # to the gated branch and fixing both gates at zero removes only this readout
    # path; pair decoding and contact-gated cross-attention remain intact.
    model.adapter_mode = "framecontact"
    for parameter in (model.paratope_contact_gate, model.epitope_contact_gate):
        with torch.no_grad():
            parameter.zero_()
        parameter.requires_grad_(False)


def condition_settings(name: str, campaign: Mapping[str, Any]) -> dict[str, Any]:
    objective = dict(campaign["base_objective"])
    settings = {
        "contact_gate_mode": "learned",
        "pair_rank": int(campaign["base_model"]["pair_rank"]),
        "freeze_geometry": False,
        "freeze_cross": False,
        "unary_pair": False,
        "unary_residue": False,
        "contact_weight": float(objective["contact_weight"]),
        "paratope_weight": float(objective["paratope_weight"]),
        "epitope_weight": float(objective["epitope_weight"]),
        "epitope_dice_weight": float(objective["epitope_dice_weight"]),
        "epitope_count_weight": float(objective["epitope_proportion_weight"]),
    }
    if name == "no_contact_gate":
        settings["contact_gate_mode"] = "content_only"
    elif name == "no_cross_attention":
        settings["freeze_cross"] = True
    elif name == "no_geometry_branch":
        settings["freeze_geometry"] = True
    elif name == "unary_pair_decoder":
        settings["unary_pair"] = True
    elif name == "rank1_pair_decoder":
        settings["pair_rank"] = 1
    elif name == "unary_residue_readout":
        settings["unary_residue"] = True
    elif name == "contact_only":
        settings.update(paratope_weight=0.0, epitope_weight=0.0,
                        epitope_dice_weight=0.0, epitope_count_weight=0.0)
    elif name == "residue_only":
        settings["contact_weight"] = 0.0
    elif name == "no_epitope_aux":
        settings.update(epitope_dice_weight=0.0, epitope_count_weight=0.0)
    elif name == "dice_only":
        settings["epitope_count_weight"] = 0.0
    elif name == "proportion_only":
        settings["epitope_dice_weight"] = 0.0
    elif name != "full_learned_gate":
        raise ValueError(f"Unhandled condition: {name}")
    return settings


def build_model(first: Mapping[str, Any], campaign: Mapping[str, Any], settings: Mapping[str, Any]) -> Any:
    from interaction_first.framecontact import FrameContactModel
    base = campaign["base_model"]
    model = FrameContactModel(
        antibody_input_dim=int(first["antibody_features"].shape[-1]),
        antigen_input_dim=int(first["antigen_features"].shape[-1]),
        d_model=int(base["d_model"]),
        num_heads=int(base["num_heads"]),
        num_self_layers=int(base["num_self_layers"]),
        geometry_feature_mode=str(base["geometry_feature_mode"]),
        geometry_bias_mode="pair_mlp",
        num_rbf=int(base["num_rbf"]),
        neighbor_k=int(base["neighbor_k"]),
        neighbor_radius=float(base["neighbor_radius_angstrom"]),
        sequence_window=int(base["sequence_window"]),
        pair_dim=int(settings["pair_rank"]),
        dropout=float(base["dropout"]),
        fusion_mode="interleaved",
        cross_attention_contact_gate_mode=str(settings["contact_gate_mode"]),
        adapter_mode="epiformer_style_same_protocol",
        distance_bin_count=int(base["distance_bin_count"]),
        contrastive_dim=64,
    )
    if settings["freeze_geometry"]:
        freeze_geometry(model)
    if settings["freeze_cross"]:
        freeze_cross_attention(model)
    if settings["unary_pair"]:
        remove_pair_interaction(model)
    if settings["unary_residue"]:
        remove_residue_contact_readout(model)
    return model


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def selection_key(metrics: Mapping[str, Any], mode: str) -> tuple[float, ...]:
    if mode == "epitope_then_contact":
        return (float(metrics["epitope"]["mpc_ap"]), float(metrics["contact"]["mpc_ap"]))
    if mode == "contact_only":
        return (float(metrics["contact"]["mpc_ap"]),)
    if mode == "epitope_then_paratope":
        return (float(metrics["epitope"]["mpc_ap"]), float(metrics["paratope"]["mpc_ap"]))
    raise ValueError(f"Unknown selection mode: {mode}")


def selection_improves(key: tuple[float, ...], best: tuple[float, ...] | None, tolerance: float) -> bool:
    if best is None:
        return True
    if key[0] > best[0] + tolerance:
        return True
    if abs(key[0] - best[0]) <= tolerance:
        return key[1:] > best[1:]
    return False


def evaluate(dataset: Any, indices: Sequence[int], model: Any, collator: Any, device: Any,
             training_module: Any, *, detailed: bool) -> dict[str, Any]:
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    task_spec = {
        "contact": ("pair_labels", "pair_candidate_mask"),
        "paratope": ("antibody_labels", "antibody_mask"),
        "epitope": ("antigen_labels", "antigen_mask"),
    }
    accum: dict[str, dict[str, Any]] = {
        name: {
            "ap": [], "auroc": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "brier_sum": 0.0, "count": 0,
            "ece_count": np.zeros(20, dtype=np.int64),
            "ece_prob": np.zeros(20, dtype=np.float64),
            "ece_label": np.zeros(20, dtype=np.float64),
            "per_complex": [],
        }
        for name in task_spec
    }
    rank_names = ("10", "20", "50", "100", "lmin", "top_1pct", "true_count")
    rank_values = {name: {"precision": [], "recall": [], "enrichment": []} for name in rank_names}

    model.eval()
    with torch.inference_mode():
        for dataset_index in indices:
            item = dataset[dataset_index]
            batch = training_module._move_batch(collator([item]), device)
            inputs = {key: value for key, value in batch.items() if "label" not in key}
            with training_module._autocast(device, True):
                outputs = model(inputs, return_attention=False)
            instance_id = str(item["instance_id"])
            for task, (label_key, mask_key) in task_spec.items():
                mask = outputs[mask_key][0]
                labels = batch[label_key][0][mask].detach().cpu().numpy().astype(np.int8)
                logits = outputs[f"{task}_logits"][0][mask].detach().float().cpu().numpy()
                require(labels.size > 0 and np.isfinite(logits).all(), "Invalid evaluation vector")
                probabilities = sigmoid(logits)
                predicted = probabilities >= 0.5
                local = accum[task]
                positives = int(labels.sum())
                negatives = int(labels.size - positives)
                ap = float(average_precision_score(labels, logits)) if positives else None
                auroc = float(roc_auc_score(labels, logits)) if positives and negatives else None
                if ap is not None:
                    local["ap"].append(ap)
                if auroc is not None:
                    local["auroc"].append(auroc)
                tp = int(((labels == 1) & predicted).sum())
                fp = int(((labels == 0) & predicted).sum())
                tn = int(((labels == 0) & ~predicted).sum())
                fn = int(((labels == 1) & ~predicted).sum())
                for key, value in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn)):
                    local[key] += value
                local["brier_sum"] += float(np.square(probabilities - labels).sum())
                local["count"] += int(labels.size)
                bins = np.minimum((probabilities * 20).astype(np.int64), 19)
                local["ece_count"] += np.bincount(bins, minlength=20)
                local["ece_prob"] += np.bincount(bins, weights=probabilities, minlength=20)
                local["ece_label"] += np.bincount(bins, weights=labels, minlength=20)
                if detailed:
                    local["per_complex"].append({
                        "instance_id": instance_id,
                        "candidate_count": int(labels.size),
                        "positive_count": positives,
                        "prevalence": float(positives / labels.size),
                        "average_precision": ap,
                        "auroc": auroc,
                        "precision_at_05": float(tp / (tp + fp)) if tp + fp else 0.0,
                        "recall_at_05": float(tp / (tp + fn)) if tp + fn else 0.0,
                    })
                if detailed and task == "contact" and positives:
                    antibody_length = int(outputs["antibody_mask"][0].sum().item())
                    antigen_length = int(outputs["antigen_mask"][0].sum().item())
                    cutoff_map = {
                        "10": 10, "20": 20, "50": 50, "100": 100,
                        "lmin": min(antibody_length, antigen_length),
                        "top_1pct": max(1, math.ceil(0.01 * labels.size)),
                        "true_count": positives,
                    }
                    prevalence = positives / labels.size
                    for cutoff_name, requested_k in cutoff_map.items():
                        k = min(int(requested_k), int(labels.size))
                        selected = np.argpartition(logits, -k)[-k:]
                        hits = int(labels[selected].sum())
                        precision = hits / k
                        recall = hits / positives
                        rank_values[cutoff_name]["precision"].append(precision)
                        rank_values[cutoff_name]["recall"].append(recall)
                        rank_values[cutoff_name]["enrichment"].append(precision / prevalence)
            del outputs, batch

    result: dict[str, Any] = {}
    for task, local in accum.items():
        total = int(local["count"])
        ece = 0.0
        for count, probability_sum, label_sum in zip(
            local["ece_count"], local["ece_prob"], local["ece_label"], strict=True
        ):
            if count:
                ece += (count / total) * abs(probability_sum / count - label_sum / count)
        tp, fp, tn, fn = (int(local[key]) for key in ("tp", "fp", "tn", "fn"))
        result[task] = {
            "mpc_ap": float(np.mean(local["ap"])) if local["ap"] else None,
            "mpc_auroc": float(np.mean(local["auroc"])) if local["auroc"] else None,
            "mpc_ap_complex_count": len(local["ap"]),
            "mpc_auroc_complex_count": len(local["auroc"]),
            "fixed_05_counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "fixed_05_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "fixed_05_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "brier": float(local["brier_sum"] / total),
            "ece_20bin": float(ece),
        }
        if detailed:
            result[task]["per_complex"] = local["per_complex"]
    if detailed:
        result["contact_ranked_cutoffs"] = {
            name: {
                "mean_precision": float(np.mean(values["precision"])) if values["precision"] else None,
                "mean_recall": float(np.mean(values["recall"])) if values["recall"] else None,
                "mean_enrichment_over_prevalence": float(np.mean(values["enrichment"])) if values["enrichment"] else None,
                "positive_complex_count": len(values["precision"]),
            }
            for name, values in rank_values.items()
        }
    return result


def main() -> None:
    arguments = parse_arguments()
    campaign, condition, parent, parent_protocol = validate_authorities(arguments)
    require(os.environ.get("SLURM_JOB_PARTITION") == "gpu_4090",
            "This campaign requires a gpu_4090 allocation")
    require(not arguments.output.exists() and not arguments.output.is_symlink(),
            "Output path already exists")
    arguments.output.mkdir(parents=True)

    # Load only the frozen parent source after authority validation.
    sys.path[:0] = [str(parent / "source" / "src"), str(parent / "source" / "scripts")]
    import torch
    from interaction_first.framecontact_objective import FrameContactLossWeights, FrameContactObjective
    from interaction_first.sabdab2_contacts import Sabdab2ContactBatchCollator, Sabdab2ContactDataset
    from interaction_first.sabdab2_structure import Sabdab2StructureEmbeddingDataset
    import train_framecontact as training_module

    require(Path(training_module.__file__).resolve() == parent / "source" / "scripts" / "train_framecontact.py",
            "Wrong frozen training source imported")
    require(torch.cuda.is_available(), "CUDA is required")
    torch.set_num_threads(1)
    training_module._set_seed(arguments.seed)
    device = torch.device("cuda")

    with (parent / "authority" / "FULL_SPLIT.csv").open(newline="") as handle:
        full_rows = list(csv.DictReader(handle))
    parent_protocol.validate_split(full_rows)
    with (parent / "authority" / "DEVELOPMENT_SPLIT.csv").open(newline="") as handle:
        development_rows = list(csv.DictReader(handle))
    require(development_rows == [row for row in full_rows if row["role"] != "test"],
            "Development split differs from frozen full split")

    parent_training_protocol = read_json(parent / "PROTOCOL.json")
    datasets: dict[str, Any] = {}
    for role in ("train", "validation"):
        base = Sabdab2StructureEmbeddingDataset(
            parent / "development" / "structure",
            Path(parent_training_protocol["embedding_root"]),
            None,
            split_role=role,
            split_path=parent / "authority" / "DEVELOPMENT_SPLIT.csv",
            split_sha256=sha256_file(parent / "authority" / "DEVELOPMENT_SPLIT.csv"),
        )
        dataset = Sabdab2ContactDataset(base, parent / "development" / "contact")
        expected_ids = {row["instance_id"] for row in development_rows if row["role"] == role}
        require({row["instance_id"] for row in dataset.records} == expected_ids,
                f"{role} dataset membership differs")
        datasets[role] = dataset
    require(len(datasets["train"]) == campaign["data_counts"]["train"], "Wrong train count")
    require(len(datasets["validation"]) == campaign["data_counts"]["validation"], "Wrong validation count")

    settings = condition_settings(arguments.condition, campaign)
    model = build_model(datasets["train"][0], campaign, settings)
    initial_state = state_sha256(model)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model.to(device)
    collator = Sabdab2ContactBatchCollator()

    weights = FrameContactLossWeights(
        contact=settings["contact_weight"],
        paratope=settings["paratope_weight"],
        epitope=settings["epitope_weight"],
        specificity=0.0,
        distance=0.0,
        contrastive=0.0,
    )
    objective = FrameContactObjective(
        weights=weights,
        specificity_margin=0.05,
        score_fraction=0.01,
        specificity_score_source="total_contact",
        contact_focal_gamma=float(campaign["base_objective"]["contact_focal_gamma"]),
        epitope_dice_weight=settings["epitope_dice_weight"],
        epitope_count_weight=settings["epitope_count_weight"],
        contrastive_temperature=0.1,
        contrastive_maximum_residues=256,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(campaign["training"]["learning_rate"]),
        weight_decay=float(campaign["training"]["weight_decay"]),
    )
    rng = np.random.default_rng(arguments.seed)

    started_payload = {
        "format_version": campaign["format_version"],
        "campaign_protocol_sha256": arguments.campaign_protocol_sha256,
        "authority_receipt_sha256": arguments.authority_receipt_sha256,
        "condition": arguments.condition,
        "condition_description": condition["description"],
        "condition_settings": settings,
        "selection": condition["selection"],
        "seed": arguments.seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "initial_state_sha256": initial_state,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "test_label_artifacts_opened": False,
        "smoke": arguments.smoke,
    }
    write_json_exclusive(arguments.output / "STARTED.json", started_payload)

    def forward_item(item: Mapping[str, Any], *, training_mode: bool) -> tuple[Any, Any, Any | None]:
        batch = training_module._move_batch(collator([item]), device)
        inputs = {key: value for key, value in batch.items() if "label" not in key}
        with training_module._autocast(device, True), training_module._saved_tensor_context(
            bool(parent_training_protocol["configuration"]["offload_saved_tensors"]) and training_mode
        ):
            outputs = model(inputs, return_attention=False)
            loss = (
                objective(training_module._cognate_training_view(outputs), batch,
                          wrong_partner_outputs=None)["loss"]
                if training_mode else None
            )
        return batch, outputs, loss

    maximum_epochs = 1 if arguments.smoke else int(campaign["training"]["maximum_epochs"])
    patience = int(campaign["training"]["early_stopping_patience"])
    tolerance = float(campaign["training"]["selection_tolerance"])
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_checkpoint: Path | None = None
    trace: list[dict[str, Any]] = []
    run_started = time.monotonic()

    for epoch in range(1, maximum_epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        schedule = rng.permutation(len(datasets["train"])).tolist()
        if arguments.smoke:
            by_size = sorted(range(len(datasets["train"])),
                             key=lambda index: datasets["train"].records[index]["candidate_pair_count"])
            schedule = [by_size[0], by_size[len(by_size) // 2], by_size[-1]]
        loss_sum = 0.0
        for step, dataset_index in enumerate(schedule, 1):
            optimizer.zero_grad(set_to_none=True)
            _, outputs, loss = forward_item(datasets["train"][dataset_index], training_mode=True)
            require(loss is not None and bool(torch.isfinite(loss).item()), "Nonfinite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(campaign["training"]["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            if arguments.smoke or step % 250 == 0 or step == len(schedule):
                print(json.dumps({
                    "condition": arguments.condition, "seed": arguments.seed,
                    "epoch": epoch, "step": step, "steps": len(schedule),
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "elapsed_seconds": time.monotonic() - epoch_started,
                }, sort_keys=True), flush=True)
            del outputs, loss

        validation_indices = (
            [0, len(datasets["validation"]) - 1]
            if arguments.smoke else list(range(len(datasets["validation"])))
        )
        state_before_evaluation = state_sha256(model)
        validation = evaluate(datasets["validation"], validation_indices, model, collator,
                              device, training_module, detailed=False)
        require(state_sha256(model) == state_before_evaluation, "Validation changed model state")
        key = selection_key(validation, condition["selection"])
        improved = selection_improves(key, best_key, tolerance)
        if improved:
            best_key = key
            best_epoch = epoch
            best_checkpoint = arguments.output / f"checkpoint-epoch-{epoch:03d}.pt"
            torch.save({
                "model_state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                "epoch": epoch,
                "seed": arguments.seed,
                "condition": arguments.condition,
                "campaign_protocol_sha256": arguments.campaign_protocol_sha256,
                "initial_state_sha256": initial_state,
                "validation": validation,
                "selection_key": key,
            }, best_checkpoint)
        epoch_payload = {
            "epoch": epoch,
            "mean_train_loss": loss_sum / len(schedule),
            "validation": validation,
            "selection_key": key,
            "selected": improved,
            "best_epoch": best_epoch,
            "seconds": time.monotonic() - epoch_started,
            "schedule_instance_ids": [datasets["train"].records[index]["instance_id"] for index in schedule],
        }
        write_json_exclusive(arguments.output / f"epoch-{epoch:03d}.json", epoch_payload)
        trace.append({key: value for key, value in epoch_payload.items() if key != "schedule_instance_ids"})
        print(json.dumps({
            "condition": arguments.condition, "seed": arguments.seed,
            "epoch": epoch, "validation": validation, "best_epoch": best_epoch,
        }, sort_keys=True), flush=True)
        if not arguments.smoke and epoch - best_epoch >= patience:
            break

    require(best_checkpoint is not None and best_checkpoint.is_file(), "No checkpoint was frozen")
    require(state_sha256(model) != initial_state, "No parameter update occurred")
    checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    frozen_state = state_sha256(model)
    detailed_validation = evaluate(
        datasets["validation"],
        [0, len(datasets["validation"]) - 1] if arguments.smoke else list(range(len(datasets["validation"])),),
        model, collator, device, training_module, detailed=not arguments.smoke,
    )
    validation_path = arguments.output / "VALIDATION_EVIDENCE.json"
    write_json_exclusive(validation_path, {
        "condition": arguments.condition,
        "seed": arguments.seed,
        "best_epoch": best_epoch,
        "checkpoint_sha256": sha256_file(best_checkpoint),
        "frozen_state_sha256": frozen_state,
        "metrics": detailed_validation,
        "test_label_artifacts_opened": False,
    })

    status = "PASS_CONDITION_SMOKE" if arguments.smoke else "PASS_TRAINING_COMPLETE_CHECKPOINT_FROZEN"
    result_payload = {
        "status": status,
        "format_version": campaign["format_version"],
        "evidence_class": campaign["evidence_class"],
        "campaign_protocol_sha256": arguments.campaign_protocol_sha256,
        "authority_receipt_sha256": arguments.authority_receipt_sha256,
        "condition": arguments.condition,
        "condition_settings": settings,
        "selection": condition["selection"],
        "seed": arguments.seed,
        "best_epoch": best_epoch,
        "selection_key": best_key,
        "checkpoint": best_checkpoint.name,
        "checkpoint_sha256": sha256_file(best_checkpoint),
        "frozen_state_sha256": frozen_state,
        "validation_evidence": validation_path.name,
        "validation_evidence_sha256": sha256_file(validation_path),
        "trace": trace,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "initial_state_sha256": initial_state,
        "old_checkpoint_loaded": False,
        "test_label_artifacts_opened": False,
        "test_selection_performed": False,
        "wall_seconds": time.monotonic() - run_started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    write_json_exclusive(arguments.output / "RESULT.json", result_payload)
    print("COMPLETE", arguments.output / "RESULT.json",
          sha256_file(arguments.output / "RESULT.json"), flush=True)


if __name__ == "__main__":
    main()
