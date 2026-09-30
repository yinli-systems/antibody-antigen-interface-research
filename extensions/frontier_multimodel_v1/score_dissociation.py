#!/usr/bin/env python3
"""Full unique-sequence all-vs-all decomposition of the partner-score mismatch."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts"), str(REPO)]

from extensions.correspondence_v2.core import chain_permutation, transform_extra_columns

SCORE_COLUMNS = (
    "score_current_centered_lme",
    "score_uncentered_lme",
    "score_centered_top_fraction",
    "interaction_centered_std",
    "interaction_centered_rms",
    "interaction_centered_max",
    "interaction_centered_top16",
    "interaction_uncentered_mean",
    "interaction_uncentered_std",
    "interaction_uncentered_rms",
    "interaction_uncentered_max",
    "interaction_uncentered_top16",
    "pair_probability_mean",
    "pair_probability_max",
    "pair_probability_top16",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def expected_first_positive_reciprocal_rank(scores: Sequence[float], positives: Sequence[bool]) -> float:
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives, dtype=bool)
    if scores.ndim != 1 or scores.shape != positives.shape or not positives.any():
        raise ValueError("matching finite scores with at least one known positive required")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    best = float(scores[positives].max())
    before = int(np.sum(scores > best))
    tied = scores == best
    tie_size = int(tied.sum())
    tie_positives = int(positives[tied].sum())
    weights = [
        math.comb(tie_size - position, tie_positives - 1) / math.comb(tie_size, tie_positives)
        for position in range(1, tie_size - tie_positives + 2)
    ]
    return float(sum(weight / (before + position) for position, weight in enumerate(weights, 1)))


def expected_recall_at_k(scores: Sequence[float], positives: Sequence[bool], k: int) -> float:
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives, dtype=bool)
    if k <= 0 or scores.shape != positives.shape or not positives.any():
        raise ValueError("positive k and at least one known positive required")
    best = float(scores[positives].max())
    before = int(np.sum(scores > best))
    if before >= k:
        return 0.0
    tied = scores == best
    tie_size = int(tied.sum())
    tie_positives = int(positives[tied].sum())
    slots = min(tie_size, k - before)
    if slots >= tie_size or tie_size - tie_positives < slots:
        return 1.0
    miss = math.comb(tie_size - tie_positives, slots) / math.comb(tie_size, slots)
    return float(1.0 - miss)


def top_mean(values: Any, count: int) -> float:
    import torch
    flat = values.reshape(-1)
    return float(torch.topk(flat, min(count, int(flat.numel())), sorted=False).values.mean())


def tensor_stats(values: Any) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.float().std(unbiased=False)),
        "rms": float(values.float().square().mean().sqrt()),
        "max": float(values.max()),
        "top16": top_mean(values, 16),
    }


def pad_batch(tensors: Sequence[Any], device: Any) -> tuple[Any, Any]:
    import torch
    widths = {int(tensor.shape[-1]) for tensor in tensors}
    if len(widths) != 1:
        raise ValueError("feature widths differ")
    length = max(int(tensor.shape[0]) for tensor in tensors)
    output = torch.zeros(len(tensors), length, next(iter(widths)), dtype=tensors[0].dtype, device=device)
    mask = torch.zeros(len(tensors), length, dtype=torch.bool, device=device)
    for index, tensor in enumerate(tensors):
        output[index, : len(tensor)] = tensor
        mask[index, : len(tensor)] = True
    return output, mask


def dynamic_groups(keys: Sequence[str], lengths: dict[str, int], fixed_length: int,
                   max_batch: int, pair_budget: int) -> Iterable[list[str]]:
    group: list[str] = []
    longest = 0
    for key in sorted(keys, key=lambda item: (lengths[item], item)):
        proposed_longest = max(longest, lengths[key])
        proposed_size = len(group) + 1
        if group and (proposed_size > max_batch or proposed_longest * fixed_length * proposed_size > pair_budget):
            yield group
            group, longest = [], 0
        group.append(key)
        longest = max(longest, lengths[key])
    if group:
        yield group


def manifest_metadata(path: Path) -> tuple[dict[tuple[str, str], tuple[str, ...]], set[tuple[str, str]]]:
    parts: dict[tuple[str, str], tuple[str, ...]] = {}
    pairs: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            heavy, light, antigen = ((row.get(name) or "").strip() for name in ("vh_seq", "vl_seq", "ag_seq"))
            if not heavy or not antigen:
                continue
            antibody = heavy + light
            ab_key = hashlib.sha256(antibody.encode("ascii")).hexdigest()
            ag_key = hashlib.sha256(antigen.encode("ascii")).hexdigest()
            ab_parts = tuple(hashlib.sha256(value.encode("ascii")).hexdigest() for value in (heavy, light) if value)
            parts[("antibody", ab_key)] = ab_parts
            parts[("antigen", ag_key)] = (ag_key,)
            pairs.add((ab_key, ag_key))
    return parts, pairs


def read_ca(path: Path) -> np.ndarray:
    coordinates = []
    for line in path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    values = np.asarray(coordinates, dtype=np.float32)
    if values.ndim != 2 or values.shape[1:] != (3,) or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"invalid CA coordinates: {path}")
    return values


def chain_local_packing(parts: Sequence[str], structure_records: dict[tuple[str, str], dict], side: str,
                        structure_root: Path) -> dict[str, float]:
    all_counts = []
    for key in parts:
        record = structure_records.get((side, key))
        if record is None:
            raise ValueError(f"no structure authority for {side}:{key}")
        xyz = read_ca(structure_root / record["pdb_path"])
        if len(xyz) != int(record["length"]):
            raise ValueError(f"CA count differs from sequence length: {key}")
        distances = np.sqrt(np.square(xyz[:, None, :] - xyz[None, :, :]).sum(axis=2))
        counts = ((distances <= 10.0) & (distances > 0.0)).sum(axis=1)
        all_counts.extend(int(value) for value in counts)
    array = np.asarray(all_counts, dtype=float)
    return {
        "packing_mean_neighbors_10a": float(array.mean()),
        "packing_median_neighbors_10a": float(np.median(array)),
        "packing_surface_fraction_le12_10a": float(np.mean(array <= 12)),
    }


def ranking_metrics(scores: Sequence[float], positives: Sequence[bool]) -> dict[str, float]:
    from sklearn.metrics import average_precision_score
    labels = np.asarray(positives, dtype=np.uint8)
    values = np.asarray(scores, dtype=float)
    return {
        "mrr_tie_expected": expected_first_positive_reciprocal_rank(values, labels),
        "recall_at_1_tie_expected": expected_recall_at_k(values, labels, 1),
        "recall_at_5_tie_expected": expected_recall_at_k(values, labels, 5),
        "recall_at_10_tie_expected": expected_recall_at_k(values, labels, 10),
        "known_positive_ap_unknown_as_zero": float(average_precision_score(labels, values)),
    }


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    from scipy.stats import spearmanr
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 3 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def main() -> None:
    arguments = parser().parse_args()
    import pandas as pd
    import torch
    from interaction_first.runtime import EmbeddingStore, safe_torch_load, set_reproducible_seed
    from interaction_first.model import PartnerSpecificInteractionModel, merge_chunk_embeddings
    from interaction_first.composite_embeddings import CompositeEmbeddingStore, build_sources
    from interaction_first.scoring import masked_double_center, masked_logmeanexp_score, masked_top_fraction_score
    from sabdab2_interaction_first import load_split_assignments

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    split_path = arguments.split_manifest.resolve()
    assignments = load_split_assignments(split_path)
    split_payload = json.loads(split_path.read_text())
    frame_all = pd.read_parquet(arguments.prepare_dir / "instances.parquet").set_index("instance_id", verify_integrity=True)
    if set(frame_all.index.astype(str)) != set(assignments):
        raise ValueError("prepared instances and split authority differ")
    validation_ids = sorted(identifier for identifier, role in assignments.items() if role == "validation")
    validation = frame_all.loc[validation_ids]
    ab_keys = sorted(set(validation.ab_sequence_sha256.astype(str)))
    ag_keys = sorted(set(validation.ag_sequence_sha256.astype(str)))
    ab_lengths = {key: int(validation.loc[validation.ab_sequence_sha256.astype(str) == key, "ab_length"].iloc[0]) for key in ab_keys}
    ag_lengths = {key: int(validation.loc[validation.ag_sequence_sha256.astype(str) == key, "ag_length"].iloc[0]) for key in ag_keys}
    if len(ab_keys) != 122 or len(ag_keys) != 119:
        raise ValueError(f"candidate authority changed: {len(ab_keys)} antibodies, {len(ag_keys)} antigens")

    parts, known_pairs = manifest_metadata(arguments.canonical_manifest.resolve())
    structure_manifest = json.loads(arguments.structure_manifest.read_text())
    if structure_manifest.get("test_loaded") or structure_manifest.get("contact_labels_loaded"):
        raise ValueError("structure authority firewall failed")
    structure_records = {(record["side"], record["sequence_sha256"]): record for record in structure_manifest["records"]}
    structure_root = arguments.structure_manifest.parent
    static: dict[tuple[str, str], dict[str, float]] = {}
    for side, keys, lengths in (("antibody", ab_keys, ab_lengths), ("antigen", ag_keys, ag_lengths)):
        for key in keys:
            static[(side, key)] = {"length": lengths[key], **chain_local_packing(parts[(side, key)], structure_records, side, structure_root)}

    checkpoint_path = arguments.checkpoint.resolve()
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if int(config["seed"]) != arguments.training_seed or config["split_manifest_sha256"] != sha256_file(split_path):
        raise ValueError("checkpoint seed or split mismatch")
    base = EmbeddingStore(arguments.embedding_dir.resolve(), arguments.embedding_cache_gib)
    sources = build_sources(arguments.canonical_manifest.resolve(), esm_if1_root=arguments.esm_if1_root.resolve(), esm_if1_dim=512)
    store = CompositeEmbeddingStore(base, sources)
    if store.describe() != config.get("representations"):
        raise ValueError("feature source/provenance differs from checkpoint")

    device = torch.device("cuda")
    set_reproducible_seed(20260930, deterministic=True)
    kwargs = {name: config[name] for name in inspect.signature(PartnerSpecificInteractionModel).parameters if name in config}
    kwargs["antigen_input_dim"] = int(config.get("antigen_input_dim", config["input_dim"]))
    kwargs["antibody_input_dim"] = int(config.get("antibody_input_dim", config["input_dim"]))
    model = PartnerSpecificInteractionModel(**kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    original_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    mode, severity = arguments.condition.split(":", 1)
    severity = int(severity)
    if (mode, severity) not in {("identity", 0), ("chain_shift", 8), ("global_shuffle", 0)}:
        raise ValueError("condition was not registered")
    raw_cache: dict[str, Any] = {}
    condition_cache: dict[tuple[str, str], Any] = {}

    def raw_feature(key: str) -> Any:
        if key not in raw_cache:
            chunks, starts, length = store.load(key, device)
            raw_cache[key] = merge_chunk_embeddings(chunks, starts, length, device)
        return raw_cache[key]

    def feature(side: str, key: str) -> Any:
        cache_key = (side, key)
        if cache_key not in condition_cache:
            tensor = raw_feature(key)
            order, _ = chain_permutation(parts[cache_key], mode, severity, 20260930, f"{side}:{key}")
            condition_cache[cache_key] = transform_extra_columns(tensor, int(config["input_dim"]), 512, order)
        return condition_cache[cache_key]

    rows: list[dict[str, Any]] = []
    began = time.monotonic()
    with torch.inference_mode():
        for position, ab_key in enumerate(ab_keys, 1):
            antibody = feature("antibody", ab_key)
            for group in dynamic_groups(ag_keys, ag_lengths, len(antibody), arguments.max_batch_size, arguments.pair_budget):
                antigens, antigen_mask = pad_batch([feature("antigen", key) for key in group], device)
                antibody_batch = antibody.unsqueeze(0).expand(len(group), -1, -1)
                antibody_mask = torch.ones(antibody_batch.shape[:2], dtype=torch.bool, device=device)
                result = model(antigens, antibody_batch, antigen_mask, antibody_mask)
                components = model.decoder.components(result["antigen_embeddings"], result["antibody_embeddings"])
                pair_mask = result["pair_mask"]
                interaction = result["interaction_logits"]
                centered = masked_double_center(interaction, pair_mask, iterations=16)
                uncentered_lme = masked_logmeanexp_score(interaction, pair_mask, temperature=float(config["score_temperature"]))
                centered_top = masked_top_fraction_score(centered, pair_mask, fraction=float(config["score_top_fraction"]), min_count=int(config["score_top_k"]))
                for index, ag_key in enumerate(group):
                    valid = pair_mask[index]
                    inter = interaction[index][valid]
                    centre = centered[index][valid]
                    probability = result["pair_probabilities"][index][valid]
                    inter_stats = tensor_stats(inter)
                    centre_stats = tensor_stats(centre)
                    probability_stats = tensor_stats(probability)
                    epi = result["epitope_probability"][index][antigen_mask[index]]
                    para = result["paratope_probability"][index][antibody_mask[index]]
                    ag_bias = components["antigen_bias"][index, : len(epi), 0]
                    ab_bias = components["antibody_bias"][index, 0, : len(para)]
                    row = {
                        "training_seed": arguments.training_seed,
                        "condition": arguments.condition,
                        "ab_sequence_sha256": ab_key,
                        "ag_sequence_sha256": ag_key,
                        "known_complex": (ab_key, ag_key) in known_pairs,
                        "ab_length": ab_lengths[ab_key],
                        "ag_length": ag_lengths[ag_key],
                        "pair_count": int(valid.sum()),
                        **{f"ab_{key}": value for key, value in static[("antibody", ab_key)].items() if key != "length"},
                        **{f"ag_{key}": value for key, value in static[("antigen", ag_key)].items() if key != "length"},
                        "score_current_centered_lme": float(result["complex_score"][index]),
                        "score_uncentered_lme": float(uncentered_lme[index]),
                        "score_centered_top_fraction": float(centered_top[index]),
                        "interaction_centered_std": centre_stats["std"],
                        "interaction_centered_rms": centre_stats["rms"],
                        "interaction_centered_max": centre_stats["max"],
                        "interaction_centered_top16": centre_stats["top16"],
                        "interaction_uncentered_mean": inter_stats["mean"],
                        "interaction_uncentered_std": inter_stats["std"],
                        "interaction_uncentered_rms": inter_stats["rms"],
                        "interaction_uncentered_max": inter_stats["max"],
                        "interaction_uncentered_top16": inter_stats["top16"],
                        "pair_probability_mean": probability_stats["mean"],
                        "pair_probability_max": probability_stats["max"],
                        "pair_probability_top16": probability_stats["top16"],
                        "epitope_head_mean": float(epi.mean()),
                        "epitope_head_max": float(epi.max()),
                        "epitope_head_top4": top_mean(epi, 4),
                        "paratope_head_mean": float(para.mean()),
                        "paratope_head_max": float(para.max()),
                        "paratope_head_top4": top_mean(para, 4),
                        "antigen_decoder_bias_mean": float(ag_bias.mean()),
                        "antigen_decoder_bias_std": float(ag_bias.float().std(unbiased=False)),
                        "antibody_decoder_bias_mean": float(ab_bias.mean()),
                        "antibody_decoder_bias_std": float(ab_bias.float().std(unbiased=False)),
                        "global_decoder_bias": float(components["global_bias"]),
                    }
                    rows.append(row)
            if position % 10 == 0 or position == len(ab_keys):
                print(f"grid seed={arguments.training_seed} condition={arguments.condition} {position}/{len(ab_keys)}", flush=True)

    grid = pd.DataFrame(rows)
    expected_pairs = len(ab_keys) * len(ag_keys)
    if len(grid) != expected_pairs or grid[["ab_sequence_sha256", "ag_sequence_sha256"]].duplicated().any():
        raise RuntimeError("all-vs-all grid is incomplete or duplicated")
    grid_path = output / "PAIR_GRID.parquet"
    grid.to_parquet(grid_path, index=False)

    component_by_instance = {str(key): str(value) for key, value in split_payload.get("component_id_by_instance", {}).items()}
    ab_components: dict[str, set[str]] = {key: set() for key in ab_keys}
    ag_components: dict[str, set[str]] = {key: set() for key in ag_keys}
    for instance_id, row in validation.iterrows():
        component = component_by_instance[str(instance_id)]
        ab_components[str(row.ab_sequence_sha256)].add(component)
        ag_components[str(row.ag_sequence_sha256)].add(component)

    anchor_rows = []
    directions = (
        ("rank_antigen_for_antibody", "ab_sequence_sha256", "ag_sequence_sha256", "ag_length", "ag_packing_surface_fraction_le12_10a", "epitope_head_mean"),
        ("rank_antibody_for_antigen", "ag_sequence_sha256", "ab_sequence_sha256", "ab_length", "ab_packing_surface_fraction_le12_10a", "paratope_head_mean"),
    )
    for direction, anchor_column, candidate_column, length_column, surface_column, prior_column in directions:
        component_map = ab_components if anchor_column.startswith("ab_") else ag_components
        for anchor, panel in grid.groupby(anchor_column, sort=True):
            labels = panel.known_complex.to_numpy(dtype=bool)
            if not labels.any():
                raise RuntimeError(f"anchor has no known positive: {anchor}")
            for score_column in SCORE_COLUMNS:
                scores = panel[score_column].to_numpy(dtype=float)
                summary = ranking_metrics(scores, labels)
                anchor_rows.append({
                    "training_seed": arguments.training_seed,
                    "condition": arguments.condition,
                    "direction": direction,
                    "anchor_sequence_sha256": anchor,
                    "anchor_components": "|".join(sorted(component_map[anchor])),
                    "candidate_count": len(panel),
                    "known_positive_count": int(labels.sum()),
                    "score_name": score_column,
                    **summary,
                    "spearman_score_vs_candidate_length": safe_spearman(scores, panel[length_column]),
                    "spearman_score_vs_candidate_surface_proxy": safe_spearman(scores, panel[surface_column]),
                    "spearman_score_vs_candidate_node_prior": safe_spearman(scores, panel[prior_column]),
                    "spearman_score_vs_interaction_std": safe_spearman(scores, panel["interaction_uncentered_std"]),
                })
    anchor_frame = pd.DataFrame(anchor_rows)
    anchor_path = output / "ANCHOR_METRICS.parquet"
    anchor_frame.to_parquet(anchor_path, index=False)

    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor.cpu(), original_state[name]):
            raise RuntimeError("model state changed during inference")
    receipt = {
        "status": "PASS_FULL_UNIQUE_SEQUENCE_ALL_VS_ALL_DECOMPOSITION",
        "format_version": "score_dissociation_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_seed": arguments.training_seed,
        "condition": arguments.condition,
        "validation_instance_count": len(validation_ids),
        "unique_antibody_count": len(ab_keys),
        "unique_antigen_count": len(ag_keys),
        "unique_pair_count": len(grid),
        "anchor_metric_rows": len(anchor_frame),
        "known_positive_pair_count_in_grid": int(grid.known_complex.sum()),
        "candidate_semantics": "project-observed sequence pairs are known positives; all others are unlabelled distractors, not non-binders",
        "score_columns": list(SCORE_COLUMNS),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_manifest_sha256": sha256_file(split_path),
        "structure_manifest_sha256": sha256_file(arguments.structure_manifest),
        "protocol_sha256": sha256_file(REPO / "extensions/frontier_multimodel_v1/SCORE_DISSOCIATION_PROTOCOL.md"),
        "source_sha256": sha256_file(Path(__file__)),
        "artifacts": {
            "pair_grid": str(grid_path.name),
            "pair_grid_sha256": sha256_file(grid_path),
            "anchor_metrics": str(anchor_path.name),
            "anchor_metrics_sha256": sha256_file(anchor_path),
        },
        "wall_seconds": time.monotonic() - began,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "contact_artifacts_loaded": False,
        "test_loaded": False,
        "training_performed": False,
        "limitations": [
            "validation-only diagnosis; alternative scores cannot be promoted",
            "unobserved sequence pairs are unlabelled, not confirmed non-binders",
            "chain-local CA packing is a proxy, not solvent-accessible surface area",
            "three fixed training seeds and pointwise exploratory analyses",
        ],
    }
    exclusive_json(output / "RECEIPT.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "training_seed", "condition", "unique_pair_count", "wall_seconds", "peak_cuda_bytes")}))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--esm-if1-root", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--pair-budget", type=int, default=1_500_000)
    parser.add_argument("--embedding-cache-gib", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    main()
