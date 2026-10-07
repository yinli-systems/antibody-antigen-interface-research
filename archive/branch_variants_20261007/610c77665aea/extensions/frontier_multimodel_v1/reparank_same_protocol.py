#!/usr/bin/env python3
"""Audit and evaluate official RePaRank checkpoints on the exact contact grid."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MODEL_ARGUMENTS = (
    "dims", "num_heads", "hidden_dims", "num_sequence_layers",
    "num_complex_layers", "ax_dims", "ax_num", "pos_num", "activation",
    "dropout", "skip_position_encoding", "feature_dims",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sequence_key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def canonical_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"INSTANCE", "vh_seq", "vl_seq", "ag_seq"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"canonical manifest lacks {sorted(required)}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            instance = str(row["INSTANCE"])
            if instance in rows:
                raise ValueError(f"duplicate canonical instance {instance}")
            heavy = (row["vh_seq"] or "").strip()
            light = (row["vl_seq"] or "").strip()
            antigen = (row["ag_seq"] or "").strip()
            if not heavy or not antigen:
                raise ValueError(f"missing heavy/antigen sequence for {instance}")
            rows[instance] = {"heavy": heavy, "light": light, "antigen": antigen}
    return rows


def structure_records(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != "frontier_structure_units_v2":
        raise ValueError("unsupported structure authority")
    if payload.get("contact_labels_loaded") is not False or payload.get("test_loaded") is not False:
        raise ValueError("structure authority violates the contact/test firewall")
    records: dict[str, dict[str, Any]] = {}
    for row in payload["records"]:
        key = str(row["sequence_sha256"])
        if key in records:
            raise ValueError(f"duplicate structure sequence {key}")
        if sequence_key(str(row["sequence"])) != key:
            raise ValueError(f"structure sequence digest mismatch {key}")
        records[key] = row
    return payload, records


AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}


def parse_backbone_pdb(path: Path, expected_sequence: str) -> np.ndarray:
    """Read exact N/CA/C/O coordinates from one structure-unit PDB."""
    residues: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    names: dict[tuple[str, str, str], str] = {}
    order: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom = line[12:16].strip()
        if atom not in {"N", "CA", "C", "O"}:
            continue
        altloc = line[16:17]
        if altloc not in {" ", "A"}:
            continue
        chain = line[21:22]
        resid = line[22:26].strip()
        insertion = line[26:27]
        key = (chain, resid, insertion)
        if key not in residues:
            residues[key] = {}
            names[key] = line[17:20].strip()
            order.append(key)
        coordinate = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float32,
        )
        residues[key].setdefault(atom, coordinate)
    sequence = []
    coordinates = []
    for key in order:
        atoms = residues[key]
        if not {"N", "CA", "C", "O"} <= set(atoms):
            raise ValueError(f"incomplete backbone residue {key} in {path}")
        residue_name = names[key]
        if residue_name not in AA3_TO_1:
            raise ValueError(f"unknown residue {residue_name} in {path}")
        sequence.append(AA3_TO_1[residue_name])
        coordinates.append([atoms[name] for name in ("N", "CA", "C", "O")])
    observed = "".join(sequence)
    if observed != expected_sequence:
        raise ValueError(
            f"PDB sequence mismatch for {path}: observed {len(observed)}, expected {len(expected_sequence)}"
        )
    array = np.asarray(coordinates, dtype=np.float32)
    if array.shape != (len(expected_sequence), 4, 3) or not np.isfinite(array).all():
        raise ValueError(f"invalid backbone array {array.shape} for {path}")
    return array


def checkpoint_hyperparameters(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    raw = checkpoint.get("hyper_parameters")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint has no hyper_parameters mapping")
    candidates = [raw]
    for key in ("model", "model_config", "architecture"):
        value = raw.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    for candidate in candidates:
        if all(key in candidate for key in ("dims", "num_heads", "hidden_dims", "num_sequence_layers", "num_complex_layers")):
            defaults = {
                "ax_dims": 32, "ax_num": 3, "pos_num": 1,
                "activation": "gelu", "dropout": 0.0,
                "skip_position_encoding": False, "feature_dims": None,
            }
            output = {}
            for key in MODEL_ARGUMENTS:
                if key in candidate:
                    output[key] = candidate[key]
                elif key in defaults:
                    output[key] = defaults[key]
                else:
                    raise ValueError(f"checkpoint is missing required model hyperparameter {key}")
            return output
    raise ValueError(f"could not locate model hyperparameters among keys {sorted(raw)}")


def model_state(checkpoint: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = checkpoint.get("state_dict")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint has no state_dict mapping")
    prefixes = ("_model.", "model.", "module._model.", "module.model.")
    for prefix in prefixes:
        selected = {
            str(key)[len(prefix):]: value
            for key, value in raw.items()
            if str(key).startswith(prefix)
        }
        if selected:
            return prefix, selected
    # Direct model state dictionaries are accepted only when every key is plain.
    if raw and not any(str(key).startswith("_") for key in raw):
        return "", dict(raw)
    raise ValueError("could not isolate RePaRank model parameters")


def load_official_model(checkpoint_path: Path, source_root: Path, device: str):
    import torch

    source = str(source_root.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from reparank.models.reparank import RePaRank  # type: ignore

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyperparameters = checkpoint_hyperparameters(checkpoint)
    model = RePaRank(**hyperparameters)
    prefix, state = model_state(checkpoint)
    incompatibility = model.load_state_dict(state, strict=False)
    missing = list(incompatibility.missing_keys)
    unexpected = list(incompatibility.unexpected_keys)
    if missing or unexpected:
        raise ValueError(f"checkpoint model-state mismatch: missing={missing}, unexpected={unexpected}")
    model.to(torch.device(device)).eval()
    return model, hyperparameters, prefix, checkpoint


def tie_aware_first_positive(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if not labels.any():
        raise ValueError("ranking metric requires at least one positive")
    best_positive = float(scores[labels].max())
    before = int(np.sum(scores > best_positive))
    tied = scores == best_positive
    tied_count = int(tied.sum())
    tied_positive = int(labels[tied].sum())
    probabilities = [
        math.comb(tied_count - position, tied_positive - 1) / math.comb(tied_count, tied_positive)
        for position in range(1, tied_count - tied_positive + 2)
    ]
    mrr = sum(probability / (before + position) for position, probability in enumerate(probabilities, 1))
    recall_at_1 = tied_positive / tied_count if before == 0 else 0.0
    return {"mrr": float(mrr), "recall_at_1": float(recall_at_1)}


def per_complex_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = labels.astype(bool)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("metrics require matching vectors")
    if not labels.any() or labels.all():
        raise ValueError("complex must contain positive and negative candidate pairs")
    ranking = tie_aware_first_positive(scores, labels)
    positive_count = int(labels.sum())
    order = np.argsort(-scores, kind="mergesort")
    cutoff = float(scores[order[min(positive_count - 1, len(order) - 1)]])
    above = scores > cutoff
    tied = scores == cutoff
    slots = positive_count - int(above.sum())
    expected_hits = int(labels[above].sum()) + slots * float(labels[tied].mean())
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "precision_at_true_contact_count": float(expected_hits / positive_count),
        "mrr": ranking["mrr"],
        "recall_at_1": ranking["recall_at_1"],
        "prevalence": float(labels.mean()),
        "pair_count": int(labels.size),
        "positive_count": positive_count,
    }


def equal_component_bootstrap(
    values: Mapping[str, float], components: Mapping[str, str], *, repeats: int = 10000, seed: int = 20260930
) -> dict[str, Any]:
    if set(values) != set(components):
        raise ValueError("component map must cover measured complexes exactly")
    grouped: dict[str, list[float]] = defaultdict(list)
    for instance, value in values.items():
        grouped[components[instance]].append(float(value))
    names = sorted(grouped)
    component_values = np.array([np.mean(grouped[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        samples[index] = float(component_values[rng.integers(len(names), size=len(names))].mean())
    return {
        "estimand": "equal_authority_component_mean_per_complex_AP",
        "estimate": float(component_values.mean()),
        "complex_weighted_estimate": float(np.mean(list(values.values()))),
        "component_bootstrap_ci95": [float(v) for v in np.quantile(samples, [0.025, 0.975])],
        "component_count": len(names),
        "complex_count": len(values),
        "bootstrap_repeats": repeats,
    }


def command_audit(args: argparse.Namespace) -> None:
    import torch

    model, hyperparameters, prefix, checkpoint = load_official_model(
        args.checkpoint.resolve(), args.reparank_source.resolve(), "cpu"
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    payload = {
        "format_version": "reparank_checkpoint_audit_v1",
        "status": "PASS_CHECKPOINT_AUDIT",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "official_source": str(args.reparank_source.resolve()),
        "official_source_commit": args.source_commit,
        "torch_version": torch.__version__,
        "hyperparameters": hyperparameters,
        "state_prefix": prefix,
        "model_parameter_count": int(parameter_count),
        "checkpoint_top_level_keys": sorted(map(str, checkpoint.keys())),
        "mode": (
            "sequence" if hyperparameters.get("skip_position_encoding") and not hyperparameters.get("feature_dims")
            else "sequence_plus_structure"
        ),
        "requires_positions": not bool(hyperparameters.get("skip_position_encoding")),
        "requires_additional_features": bool(hyperparameters.get("feature_dims")),
        "model_state_loaded_strictly": True,
        "contact_labels_read": False,
        "test_labels_read": False,
    }
    write_exclusive_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def command_evaluate(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    repo_root = args.project_source.resolve()
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))
    from interaction_first.contacts import load_contact_map

    split_path = args.split_manifest.resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    validation = [str(value) for value in split["splits"]["validation"]]
    if set(validation) & (set(map(str, split["splits"]["train"])) | set(map(str, split["splits"]["test"]))):
        raise ValueError("split firewall overlap")
    if args.max_instances is not None:
        validation = validation[: args.max_instances]
    canonical = canonical_rows(args.canonical_manifest.resolve())
    if not set(validation) <= set(canonical):
        raise ValueError("canonical manifest does not cover validation")
    components = {
        instance: str(split["component_id_by_instance"][instance]) for instance in validation
    }

    model, hyperparameters, prefix, _checkpoint = load_official_model(
        args.checkpoint.resolve(), args.reparank_source.resolve(), args.device
    )
    requires_positions = not bool(hyperparameters.get("skip_position_encoding"))
    if hyperparameters.get("feature_dims"):
        raise RuntimeError(
            "checkpoint requires additional features; this authority supplies only sequence and predicted backbone positions"
        )
    structure_payload = None
    structure_by_key: dict[str, dict[str, Any]] = {}
    if requires_positions:
        if args.structure_manifest is None:
            raise ValueError("structure checkpoint requires --structure-manifest")
        structure_payload, structure_by_key = structure_records(args.structure_manifest.resolve())

    state_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    device = torch.device(args.device)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    pair_output_dir = output_dir / "pair_predictions"
    if args.write_pair_predictions:
        pair_output_dir.mkdir()
    rows = []
    pooled_labels: list[np.ndarray] = []
    pooled_scores: list[np.ndarray] = []
    primary: dict[str, float] = {}
    began = time.monotonic()

    from reparank.models.reparank import RePaRankInput  # type: ignore

    with torch.inference_mode():
        for index, instance in enumerate(validation, 1):
            sequence = canonical[instance]
            sequences: dict[str, tuple[str, ...]] = {"heavy": (sequence["heavy"],)}
            if sequence["light"]:
                sequences["light"] = (sequence["light"],)
            sequences["antigen"] = (sequence["antigen"],)
            positions: dict[str, torch.Tensor] = {}
            structure_receipts = {}
            if requires_positions:
                for name, chain_sequence in sequence.items():
                    if name == "light" and not chain_sequence:
                        continue
                    key = sequence_key(chain_sequence)
                    if key not in structure_by_key:
                        raise ValueError(f"missing predicted-chain structure {name}:{key}")
                    record = structure_by_key[key]
                    pdb_path = args.structure_manifest.resolve().parent / record["pdb_path"]
                    if sha256(pdb_path) != record["pdb_sha256"]:
                        raise ValueError(f"predicted PDB hash mismatch {pdb_path}")
                    backbone = parse_backbone_pdb(pdb_path, chain_sequence)
                    pos_num = int(hyperparameters.get("pos_num") or 1)
                    if pos_num == 1:
                        backbone = backbone[:, 1:2, :]
                    elif pos_num != 4:
                        raise ValueError(f"unsupported checkpoint pos_num={pos_num}")
                    positions[name] = torch.as_tensor(backbone, device=device).unsqueeze(0)
                    structure_receipts[name] = {
                        "sequence_sha256": key,
                        "pdb_path": str(pdb_path),
                        "pdb_sha256": record["pdb_sha256"],
                    }
            output = model(RePaRankInput(sequences=sequences, positions=positions))
            antibody_embeddings = [output.complex_embeddings["heavy"]]
            if "light" in output.complex_embeddings:
                antibody_embeddings.append(output.complex_embeddings["light"])
            antibody = torch.cat(antibody_embeddings, dim=1)[0]
            antigen = output.complex_embeddings["antigen"][0]

            contact = load_contact_map(args.contact_dir.resolve() / f"{instance}.npz", instance)
            antibody_indices = torch.as_tensor(contact.antibody_indices, dtype=torch.long, device=device)
            antigen_indices = torch.as_tensor(contact.antigen_indices, dtype=torch.long, device=device)
            if int(antibody.shape[0]) != len(sequence["heavy"]) + len(sequence["light"]):
                raise ValueError(f"antibody embedding length mismatch {instance}")
            if int(antigen.shape[0]) != len(sequence["antigen"]):
                raise ValueError(f"antigen embedding length mismatch {instance}")
            selected_ab = antibody.index_select(0, antibody_indices)
            selected_ag = antigen.index_select(0, antigen_indices)
            score_matrix = -torch.cdist(selected_ag, selected_ab).cpu().numpy()
            labels, mask = contact.dense()
            if score_matrix.shape != labels.shape or mask.shape != labels.shape:
                raise ValueError(f"candidate-grid mismatch {instance}")
            labels_vector = labels[mask].astype(np.uint8)
            scores_vector = score_matrix[mask].astype(np.float64)
            metrics = per_complex_metrics(labels_vector, scores_vector)
            primary[instance] = metrics["average_precision"]
            pooled_labels.append(labels_vector.copy())
            pooled_scores.append(scores_vector.copy())

            epitope_labels, paratope_labels = contact.node_labels()
            epitope_score = score_matrix.max(axis=1)
            paratope_score = score_matrix.max(axis=0)
            metrics["epitope_average_precision"] = float(
                average_precision_score(epitope_labels, epitope_score)
            )
            metrics["paratope_average_precision"] = float(
                average_precision_score(paratope_labels, paratope_score)
            )
            rows.append({
                "instance_id": instance,
                "component_id": components[instance],
                **metrics,
                "structure_receipts": structure_receipts,
            })
            if args.write_pair_predictions:
                antigen_local, antibody_local = np.nonzero(mask)
                np.savez_compressed(
                    pair_output_dir / f"{instance}.npz",
                    instance_id=np.asarray(instance),
                    antigen_residue_index=contact.antigen_indices[antigen_local].astype(np.int64),
                    antibody_residue_index=contact.antibody_indices[antibody_local].astype(np.int64),
                    label=labels_vector,
                    score=scores_vector.astype(np.float32),
                    score_definition=np.asarray("negative_euclidean_distance"),
                )
            if index % args.log_every == 0 or index == len(validation):
                print(f"evaluated {index}/{len(validation)}", flush=True)

    for name, value in model.state_dict().items():
        if not torch.equal(value.cpu(), state_before[name]):
            raise RuntimeError("inference mutated model state")

    pooled_label = np.concatenate(pooled_labels)
    pooled_score = np.concatenate(pooled_scores)
    pooled = {
        "average_precision": float(average_precision_score(pooled_label, pooled_score)),
        "auroc": float(roc_auc_score(pooled_label, pooled_score)),
        "pair_count": int(pooled_label.size),
        "positive_count": int(pooled_label.sum()),
    }
    pd.DataFrame([{key: value for key, value in row.items() if key != "structure_receipts"} for row in rows]).to_csv(
        output_dir / "per_complex.csv", index=False
    )
    payload = {
        "format_version": "reparank_same_protocol_evaluation_v1",
        "status": "PASS_ENGINEERING_CANARY" if args.max_instances is not None else "COMPLETE_EXPOSED_PRETRAINED_VALIDATION_DIAGNOSTIC",
        "mode": "sequence_plus_structure" if requires_positions else "sequence",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "official_source_commit": args.source_commit,
        "state_prefix": prefix,
        "hyperparameters": hyperparameters,
        "split_manifest_sha256": sha256(split_path),
        "canonical_manifest_sha256": sha256(args.canonical_manifest.resolve()),
        "structure_manifest_sha256": (
            sha256(args.structure_manifest.resolve()) if args.structure_manifest else None
        ),
        "instance_count": len(rows),
        "primary_macro_mean_per_complex_AP": float(np.mean(list(primary.values()))),
        "primary_equal_component_analysis": equal_component_bootstrap(primary, components),
        "epitope_macro_AP": float(np.mean([row["epitope_average_precision"] for row in rows])),
        "paratope_macro_AP": float(np.mean([row["paratope_average_precision"] for row in rows])),
        "pooled_pair_metrics": pooled,
        "raw_pair_prediction_directory": str(pair_output_dir) if args.write_pair_predictions else None,
        "raw_pair_prediction_file_count": len(rows) if args.write_pair_predictions else 0,
        "per_complex": rows,
        "score_definition": "negative Euclidean distance between official RePaRank complex embeddings",
        "candidate_grid": "exact existing complete antibody-by-antigen residue grid",
        "wall_seconds": time.monotonic() - began,
        "model_state_unchanged": True,
        "checkpoint_training_overlap_audited": False,
        "test_labels_read": False,
        "ood_labels_read": False,
        "blind_claim_authorized": False,
        "sota_claim_authorized": False,
    }
    write_exclusive_json(output_dir / "evaluation.json", payload)
    print(json.dumps({
        "status": payload["status"], "mode": payload["mode"],
        "instance_count": payload["instance_count"],
        "macro_AP": payload["primary_macro_mean_per_complex_AP"],
        "output": str(output_dir),
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-checkpoint")
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--reparank-source", type=Path, required=True)
    audit.add_argument("--source-commit", required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(function=command_audit)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--reparank-source", type=Path, required=True)
    evaluate.add_argument("--source-commit", required=True)
    evaluate.add_argument("--project-source", type=Path, required=True)
    evaluate.add_argument("--canonical-manifest", type=Path, required=True)
    evaluate.add_argument("--split-manifest", type=Path, required=True)
    evaluate.add_argument("--contact-dir", type=Path, required=True)
    evaluate.add_argument("--structure-manifest", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate.add_argument("--max-instances", type=int)
    evaluate.add_argument("--write-pair-predictions", action="store_true")
    evaluate.add_argument("--log-every", type=int, default=25)
    evaluate.set_defaults(function=command_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
