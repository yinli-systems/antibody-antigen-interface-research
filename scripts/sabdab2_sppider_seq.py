#!/usr/bin/env python3
"""Train and evaluate SPPIDER-seq on the frozen SAbDab2 benchmark.

The adapter exposes five resumable stages:

``prepare``
    Stream authoritative candidate rows into paratope and epitope residue
    labels and create leakage-aware role-specific validation splits.
``embed``
    Deduplicate sequences and precompute the released ESM-2 chunk embeddings.
``train``
    Train one released SPPIDER-seq cross-attention head with the paper's BCE,
    50-epoch cap, and ten-epoch validation early stopping protocol.
``infer``
    Run a trained or official checkpoint over a deterministic test shard.
``assemble``
    Validate all shard artifacts and emit separate benchmark paratope or
    epitope projections plus native residue-level metrics.

The public v2.1.0 repository does not release a training script and the paper
does not disclose optimizer, learning rate, batch size, or random seed.  Those
implementation choices are therefore explicit CLI options and are recorded in
every checkpoint and provenance file rather than described as official values.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sppider_seq_core import (
    EMBED_DIM,
    ESM_MODEL_NAME,
    MAX_TOKENS,
    NUM_HEADS,
    STRIDE,
    ChunkwiseInteractionModel,
    atomic_torch_save,
    embed_sequence,
    embedding_path,
    extract_model_state,
    merge_chunk_logits_to_full,
    predict_residue_scores,
    require_torch,
    safe_torch_load,
    sequence_sha256,
    set_reproducible_seed,
    validate_embedding_artifact,
)


FORMAT_VERSION = "sabdab2_sppider_seq_v1"
EXPECTED_MANIFEST_SHA256 = "15f7819193d79f637849744b7cc8160b34b81b5bc2f50671c0cd07560b524377"
ROLES = ("paratope", "epitope")

PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("instance_id", pa.string(), nullable=False),
        pa.field("ab_res_idx", pa.int64(), nullable=False),
        pa.field("ag_res_idx", pa.int64(), nullable=False),
        pa.field("score", pa.float32(), nullable=False),
    ]
)

RESIDUE_SCHEMA = pa.schema(
    [
        pa.field("instance_id", pa.string(), nullable=False),
        pa.field("role", pa.string(), nullable=False),
        pa.field("res_idx", pa.int64(), nullable=False),
        pa.field("label", pa.bool_(), nullable=False),
        pa.field("score", pa.float32(), nullable=False),
    ]
)


@dataclass(frozen=True)
class BenchmarkLayout:
    """Resolved paths for one materialized SAbDab2 benchmark delivery."""

    root: Path
    frozen: Path

    @classmethod
    def from_root(cls, root: Path) -> "BenchmarkLayout":
        root = root.expanduser().resolve()
        frozen = root / "data/nfs/yangwuyu/baseline-eval/frozen/sabdab2"
        return cls(root=root, frozen=frozen)

    def validate(self, verify_manifest_hash: bool = True) -> None:
        """Enforce the five-file frozen input contract and optional manifest hash."""
        required = [
            self.frozen / "pairs.parquet",
            self.frozen / "instances.parquet",
            self.frozen / "splits.json",
            self.frozen / "contamination.json",
            self.frozen / "MANIFEST.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing frozen benchmark files: {missing}")
        if verify_manifest_hash:
            digest = sha256_file(self.frozen / "MANIFEST.json")
            if digest != EXPECTED_MANIFEST_SHA256:
                raise ValueError(
                    "Frozen MANIFEST.json digest mismatch: "
                    f"expected={EXPECTED_MANIFEST_SHA256}, observed={digest}"
                )


@dataclass(frozen=True)
class PreparedInstance:
    """Compact metadata retained after authoritative label projection."""

    instance_id: str
    ab_sequence_sha256: str
    ag_sequence_sha256: str
    ab_length: int
    ag_length: int
    paratope_positive_count: int
    epitope_positive_count: int
    candidate_pair_count: int
    positive_pair_count: int
    frozen_split: str
    paratope_split: str
    epitope_split: str


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a bounded-memory SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict JSON through an adjacent temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed NumPy artifact atomically without suffix rewriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_write_parquet(table: pa.Table, path: Path) -> None:
    """Write a Parquet table through an adjacent temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def load_frozen_splits(layout: BenchmarkLayout) -> dict[str, list[str]]:
    """Load and validate the fixed train and test instance memberships."""
    payload = json.loads(
        (layout.frozen / "splits.json").read_text(encoding="utf-8-sig")
    )
    try:
        raw_splits = payload["sabdab2"]
        splits = {
            "train": [str(value) for value in raw_splits["train"]],
            "test": [str(value) for value in raw_splits["test"]],
        }
    except KeyError as error:
        raise KeyError("splits.json does not contain sabdab2 train/test lists") from error
    for name, values in splits.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Frozen split {name!r} contains duplicate IDs")
    overlap = set(splits["train"]) & set(splits["test"])
    if overlap:
        raise ValueError(f"Frozen train/test overlap detected: {sorted(overlap)[:5]}")
    return splits


def load_instances(layout: BenchmarkLayout) -> pd.DataFrame:
    """Load sequence metadata and enforce one complete row per instance."""
    columns = ["instance_id", "ab_seq", "ag_seq"]
    instances = pd.read_parquet(layout.frozen / "instances.parquet", columns=columns)
    if instances["instance_id"].isna().any() or instances["instance_id"].duplicated().any():
        raise ValueError("instances.parquet has null or duplicate instance IDs")
    for column in ("ab_seq", "ag_seq"):
        if instances[column].isna().any():
            raise ValueError(f"instances.parquet contains null {column} values")
        instances[column] = instances[column].astype(str).str.strip().str.upper()
        if (instances[column].str.len() == 0).any():
            raise ValueError(f"instances.parquet contains empty {column} values")
    return instances.set_index("instance_id", verify_integrity=True)


def deterministic_group_validation_ids(
    train_instances: pd.DataFrame,
    sequence_column: str,
    validation_fraction: float,
    seed: int,
) -> set[str]:
    """Select a deterministic validation subset without repeated-sequence leakage."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    groups: dict[str, list[str]] = defaultdict(list)
    for instance_id, sequence in train_instances[sequence_column].items():
        groups[sequence_sha256(str(sequence))].append(str(instance_id))
    if len(groups) < 2:
        raise ValueError(f"Need at least two unique {sequence_column} groups")

    ordered = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("ascii")).hexdigest(),
    )
    target = max(1, round(len(train_instances) * validation_fraction))
    validation: set[str] = set()
    for group_key in ordered[:-1]:
        if len(validation) >= target:
            break
        validation.update(groups[group_key])
    if not validation or len(validation) == len(train_instances):
        raise ValueError("Unable to construct a non-empty train/validation split")
    return validation


def deterministic_component_validation_ids(
    train_instances: pd.DataFrame,
    sequence_column: str,
    validation_fraction: float,
    seed: int,
    additional_group_by_instance: Mapping[str, str],
) -> set[str]:
    """Select validation components joined by query sequence or external group."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    identifiers = [str(value) for value in train_instances.index]
    if set(identifiers) != set(additional_group_by_instance):
        raise ValueError("Additional validation groups do not match frozen train IDs")

    parent = {identifier: identifier for identifier in identifiers}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    first_by_sequence: dict[str, str] = {}
    first_by_group: dict[str, str] = {}
    for identifier in identifiers:
        sequence_key = sequence_sha256(str(train_instances.at[identifier, sequence_column]))
        group_key = str(additional_group_by_instance[identifier])
        if group_key == "" or group_key.lower() == "nan":
            raise ValueError(f"Empty additional validation group for {identifier}")
        sequence_first = first_by_sequence.setdefault(sequence_key, identifier)
        group_first = first_by_group.setdefault(group_key, identifier)
        union(identifier, sequence_first)
        union(identifier, group_first)

    components: dict[str, list[str]] = defaultdict(list)
    for identifier in identifiers:
        components[find(identifier)].append(identifier)
    if len(components) < 2:
        raise ValueError("Need at least two connected validation components")

    target = max(1, round(len(identifiers) * validation_fraction))
    target_group_count = max(
        1,
        round(len(set(additional_group_by_instance.values())) * validation_fraction),
    )
    ordered_components = sorted(
        (sorted(component) for component in components.values()),
        key=lambda component: hashlib.sha256(
            f"{seed}:{','.join(component)}".encode("utf-8")
        ).hexdigest(),
    )
    component_records = [
        (
            component,
            len(
                {
                    str(additional_group_by_instance[identifier])
                    for identifier in component
                }
            ),
        )
        for component in ordered_components
    ]

    def objective(instance_count: int, group_count: int) -> float:
        return (
            abs(instance_count - target) / target
            + abs(group_count - target_group_count) / target_group_count
        )

    maximum_instance_sum = min(
        len(identifiers) - 1,
        max(target * 2, target + max(len(component) for component, _ in component_records)),
    )
    maximum_group_sum = min(
        len(set(additional_group_by_instance.values())) - 1,
        max(
            target_group_count * 2,
            target_group_count + max(group_count for _, group_count in component_records),
        ),
    )
    states: dict[tuple[int, int], int] = {(0, 0): 0}
    for component_index, (component, group_count) in enumerate(component_records):
        updates: dict[tuple[int, int], int] = {}
        for (instance_sum, group_sum), selection_mask in list(states.items()):
            new_instance_sum = instance_sum + len(component)
            new_group_sum = group_sum + group_count
            if (
                new_instance_sum > maximum_instance_sum
                or new_group_sum > maximum_group_sum
            ):
                continue
            key = (new_instance_sum, new_group_sum)
            updates.setdefault(key, selection_mask | (1 << component_index))
        for key, selection_mask in updates.items():
            states.setdefault(key, selection_mask)
    candidate_states = [key for key in states if key != (0, 0)]
    if not candidate_states:
        raise ValueError("Unable to enumerate a component validation split")
    best_state = min(
        candidate_states,
        key=lambda key: (
            objective(key[0], key[1]),
            abs(key[0] - target),
            abs(key[1] - target_group_count),
            key,
        ),
    )
    selection_mask = states[best_state]
    validation = {
        identifier
        for component_index, (component, _) in enumerate(component_records)
        if selection_mask & (1 << component_index)
        for identifier in component
    }
    if not validation or len(validation) == len(identifiers):
        raise ValueError("Unable to construct a non-empty component validation split")
    return validation


def build_role_split_maps(
    instances: pd.DataFrame,
    splits: Mapping[str, Sequence[str]],
    validation_fraction: float,
    seed: int,
    validation_group_by_instance: Mapping[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """Create role-specific train/validation assignments on the frozen train set."""
    train_ids = [str(value) for value in splits["train"]]
    test_ids = [str(value) for value in splits["test"]]
    missing = (set(train_ids) | set(test_ids)) - set(instances.index)
    if missing:
        raise ValueError(f"Frozen splits reference missing instances: {sorted(missing)[:5]}")
    train_instances = instances.loc[train_ids]
    role_to_column = {"paratope": "ab_seq", "epitope": "ag_seq"}
    output: dict[str, dict[str, str]] = {}
    for role, sequence_column in role_to_column.items():
        if validation_group_by_instance is None:
            validation = deterministic_group_validation_ids(
                train_instances,
                sequence_column,
                validation_fraction,
                seed,
            )
        else:
            train_groups = {
                instance_id: str(validation_group_by_instance[instance_id])
                for instance_id in train_ids
            }
            validation = deterministic_component_validation_ids(
                train_instances,
                sequence_column,
                validation_fraction,
                seed,
                train_groups,
            )
        assignments = {
            instance_id: ("validation" if instance_id in validation else "train")
            for instance_id in train_ids
        }
        assignments.update({instance_id: "test" for instance_id in test_ids})

        train_sequences = {
            sequence_sha256(str(instances.at[instance_id, sequence_column]))
            for instance_id, split_name in assignments.items()
            if split_name == "train"
        }
        validation_sequences = {
            sequence_sha256(str(instances.at[instance_id, sequence_column]))
            for instance_id, split_name in assignments.items()
            if split_name == "validation"
        }
        if train_sequences & validation_sequences:
            raise AssertionError(f"Repeated {role} query sequence leaked into validation")
        if validation_group_by_instance is not None:
            train_groups = {
                str(validation_group_by_instance[instance_id])
                for instance_id, split_name in assignments.items()
                if split_name == "train"
            }
            validation_groups = {
                str(validation_group_by_instance[instance_id])
                for instance_id, split_name in assignments.items()
                if split_name == "validation"
            }
            if train_groups & validation_groups:
                raise AssertionError(f"Canonical group leaked into {role} validation")
        output[role] = assignments
    return output


def load_validation_groups(
    canonical_manifest: Path,
    splits: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Load canonical complex-cluster groups and verify frozen split agreement."""
    frame = pd.read_csv(
        canonical_manifest.expanduser().resolve(),
        usecols=["INSTANCE", "split", "ab_ag_cluster"],
    )
    if frame["INSTANCE"].isna().any() or frame["INSTANCE"].duplicated().any():
        raise ValueError("Canonical manifest contains null or duplicate instance IDs")
    frame["INSTANCE"] = frame["INSTANCE"].astype(str)
    frame = frame.set_index("INSTANCE", verify_integrity=True)
    expected_ids = set(splits["train"]) | set(splits["test"])
    if set(frame.index) != expected_ids:
        raise ValueError("Canonical manifest and frozen instance memberships differ")
    for split_name in ("train", "test"):
        observed = set(frame.index[frame["split"].astype(str) == split_name])
        if observed != set(splits[split_name]):
            raise ValueError(f"Canonical and frozen {split_name} memberships differ")
    if frame["ab_ag_cluster"].isna().any():
        raise ValueError("Canonical manifest contains null ab_ag_cluster values")
    return frame["ab_ag_cluster"].astype(str).to_dict()


def validate_label_artifact(
    path: Path,
    expected_ab_length: int,
    expected_ag_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one prepared label artifact and enforce its complete residue axes."""
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != {"ab_labels", "ag_labels"}:
            raise ValueError(f"Unexpected label keys in {path}: {artifact.files}")
        ab_labels = np.asarray(artifact["ab_labels"], dtype=np.uint8)
        ag_labels = np.asarray(artifact["ag_labels"], dtype=np.uint8)
    if ab_labels.shape != (expected_ab_length,):
        raise ValueError(f"Antibody label shape mismatch in {path}")
    if ag_labels.shape != (expected_ag_length,):
        raise ValueError(f"Antigen label shape mismatch in {path}")
    if not np.isin(ab_labels, (0, 1)).all() or not np.isin(ag_labels, (0, 1)).all():
        raise ValueError(f"Non-binary residue labels found in {path}")
    return ab_labels, ag_labels


def command_prepare(args: argparse.Namespace) -> None:
    """Project candidate contacts into authoritative marginal residue labels."""
    layout = BenchmarkLayout.from_root(args.benchmark_root)
    layout.validate(verify_manifest_hash=not args.skip_manifest_hash)
    output_dir = args.output_dir.expanduser().resolve()
    label_dir = output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    splits = load_frozen_splits(layout)
    instances = load_instances(layout)
    expected_ids = set(splits["train"]) | set(splits["test"])
    if set(instances.index) != expected_ids:
        extra = set(instances.index) - expected_ids
        missing = expected_ids - set(instances.index)
        raise ValueError(
            f"Instance/split membership mismatch: extra={len(extra)}, missing={len(missing)}"
        )
    validation_groups = (
        load_validation_groups(args.canonical_manifest, splits)
        if args.canonical_manifest is not None
        else None
    )
    role_splits = build_role_split_maps(
        instances,
        splits,
        args.validation_fraction,
        args.seed,
        validation_group_by_instance=validation_groups,
    )
    frozen_split = {
        **{instance_id: "train" for instance_id in splits["train"]},
        **{instance_id: "test" for instance_id in splits["test"]},
    }

    pair_file = pq.ParquetFile(layout.frozen / "pairs.parquet")
    if pair_file.num_row_groups != len(instances):
        raise ValueError(
            f"Expected one pair row group per instance; got {pair_file.num_row_groups} "
            f"for {len(instances)} instances"
        )

    prepared: list[PreparedInstance] = []
    observed_ids: set[str] = set()
    for row_group in range(pair_file.num_row_groups):
        table = pair_file.read_row_group(
            row_group,
            columns=["instance_id", "ab_res_idx", "ag_res_idx", "label"],
        )
        frame = table.to_pandas()
        unique_ids = frame["instance_id"].unique()
        if len(unique_ids) != 1:
            raise ValueError(f"Pair row group {row_group} spans {len(unique_ids)} instances")
        instance_id = str(unique_ids[0])
        if instance_id in observed_ids or instance_id not in instances.index:
            raise ValueError(f"Duplicate or unknown pair row group ID: {instance_id}")
        observed_ids.add(instance_id)
        ab_sequence = str(instances.at[instance_id, "ab_seq"])
        ag_sequence = str(instances.at[instance_id, "ag_seq"])
        ab_indices = frame["ab_res_idx"].to_numpy(dtype=np.int64, copy=False)
        ag_indices = frame["ag_res_idx"].to_numpy(dtype=np.int64, copy=False)
        pair_labels = frame["label"].to_numpy(dtype=np.bool_, copy=False)
        if len(frame) == 0:
            raise ValueError(f"Instance {instance_id} has no finite candidate pairs")
        if ab_indices.min() < 0 or ab_indices.max() >= len(ab_sequence):
            raise ValueError(f"Antibody candidate index out of range for {instance_id}")
        if ag_indices.min() < 0 or ag_indices.max() >= len(ag_sequence):
            raise ValueError(f"Antigen candidate index out of range for {instance_id}")

        ab_labels = np.zeros(len(ab_sequence), dtype=np.uint8)
        ag_labels = np.zeros(len(ag_sequence), dtype=np.uint8)
        ab_labels[ab_indices[pair_labels]] = 1
        ag_labels[ag_indices[pair_labels]] = 1
        label_path = label_dir / f"{instance_id}.npz"
        if label_path.exists() and args.resume:
            old_ab, old_ag = validate_label_artifact(
                label_path,
                len(ab_sequence),
                len(ag_sequence),
            )
            if not np.array_equal(old_ab, ab_labels) or not np.array_equal(old_ag, ag_labels):
                raise ValueError(f"Existing label artifact disagrees for {instance_id}")
        else:
            atomic_write_npz(label_path, ab_labels=ab_labels, ag_labels=ag_labels)

        prepared.append(
            PreparedInstance(
                instance_id=instance_id,
                ab_sequence_sha256=sequence_sha256(ab_sequence),
                ag_sequence_sha256=sequence_sha256(ag_sequence),
                ab_length=len(ab_sequence),
                ag_length=len(ag_sequence),
                paratope_positive_count=int(ab_labels.sum()),
                epitope_positive_count=int(ag_labels.sum()),
                candidate_pair_count=len(frame),
                positive_pair_count=int(pair_labels.sum()),
                frozen_split=frozen_split[instance_id],
                paratope_split=role_splits["paratope"][instance_id],
                epitope_split=role_splits["epitope"][instance_id],
            )
        )
        if (row_group + 1) % 100 == 0 or row_group + 1 == pair_file.num_row_groups:
            print(f"Prepared {row_group + 1}/{pair_file.num_row_groups} instances", flush=True)

    if observed_ids != expected_ids:
        raise ValueError(f"Missing pair row groups for {len(expected_ids - observed_ids)} instances")
    prepared_frame = pd.DataFrame([asdict(value) for value in prepared]).sort_values(
        "instance_id",
        kind="stable",
    )
    atomic_write_parquet(pa.Table.from_pandas(prepared_frame, preserve_index=False), output_dir / "instances.parquet")

    summary = {
        "format_version": FORMAT_VERSION,
        "created_at_unix": time.time(),
        "benchmark_root": str(layout.root),
        "manifest_sha256": sha256_file(layout.frozen / "MANIFEST.json"),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "validation_grouping": (
            "canonical_cluster_and_focal_sequence_connected_components"
            if validation_groups is not None
            else "focal_sequence"
        ),
        "canonical_manifest": (
            str(args.canonical_manifest.expanduser().resolve())
            if args.canonical_manifest is not None
            else None
        ),
        "instance_count": len(prepared_frame),
        "train_instance_count": len(splits["train"]),
        "test_instance_count": len(splits["test"]),
        "candidate_pair_count": int(prepared_frame["candidate_pair_count"].sum()),
        "positive_pair_count": int(prepared_frame["positive_pair_count"].sum()),
        "unique_antibody_sequence_count": int(prepared_frame["ab_sequence_sha256"].nunique()),
        "unique_antigen_sequence_count": int(prepared_frame["ag_sequence_sha256"].nunique()),
        "role_split_counts": {
            role: prepared_frame[f"{role}_split"].value_counts().sort_index().to_dict()
            for role in ROLES
        },
        "role_actual_validation_fraction": {
            role: float(
                (prepared_frame[f"{role}_split"] == "validation").sum()
                / len(splits["train"])
            )
            for role in ROLES
        },
        "role_validation_group_count": (
            {
                role: len(
                    {
                        str(validation_groups[instance_id])
                        for instance_id in prepared_frame.loc[
                            prepared_frame[f"{role}_split"] == "validation",
                            "instance_id",
                        ]
                    }
                )
                for role in ROLES
            }
            if validation_groups is not None
            else None
        ),
        "label_definition": {
            "paratope": "any positive candidate pair for an antibody residue",
            "epitope": "any positive candidate pair for an antigen residue",
            "authoritative_source": "pairs.parquet.label",
        },
    }
    atomic_write_json(output_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_prepared_instances(prepare_dir: Path) -> pd.DataFrame:
    """Load the prepared manifest and enforce its unique instance key."""
    path = prepare_dir / "instances.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared instance manifest is missing: {path}")
    frame = pd.read_parquet(path)
    required = {
        "instance_id",
        "ab_sequence_sha256",
        "ag_sequence_sha256",
        "ab_length",
        "ag_length",
        "paratope_split",
        "epitope_split",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prepared instance manifest is missing columns: {sorted(missing)}")
    if frame["instance_id"].isna().any() or frame["instance_id"].duplicated().any():
        raise ValueError("Prepared instance manifest has null or duplicate IDs")
    return frame.set_index("instance_id", verify_integrity=True)


def collect_unique_sequences(
    layout: BenchmarkLayout,
    prepared: pd.DataFrame,
) -> dict[str, str]:
    """Recover and validate the exact sequences referenced by prepared digests."""
    instances = load_instances(layout)
    if set(prepared.index) != set(instances.index):
        raise ValueError("Prepared and frozen instance memberships differ")
    sequences: dict[str, str] = {}
    for instance_id, row in instances.iterrows():
        for side, column in (("ab", "ab_seq"), ("ag", "ag_seq")):
            sequence = str(row[column])
            key = sequence_sha256(sequence)
            expected = str(prepared.at[instance_id, f"{side}_sequence_sha256"])
            if key != expected:
                raise ValueError(f"Prepared {side} sequence digest mismatch for {instance_id}")
            previous = sequences.setdefault(key, sequence)
            if previous != sequence:
                raise AssertionError("SHA-256 collision detected between protein sequences")
    return sequences


def shard_for_key(key: str, shard_count: int) -> int:
    """Assign a hexadecimal SHA-256 key to a deterministic shard."""
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    return int(key[:16], 16) % shard_count


def command_embed(args: argparse.Namespace) -> None:
    """Precompute resumable released-style ESM-2 chunks for unique sequences."""
    require_torch()
    import torch
    from transformers import EsmModel, EsmTokenizer

    layout = BenchmarkLayout.from_root(args.benchmark_root)
    layout.validate(verify_manifest_hash=not args.skip_manifest_hash)
    prepare_dir = args.prepare_dir.expanduser().resolve()
    embedding_dir = args.embedding_dir.expanduser().resolve()
    prepared = load_prepared_instances(prepare_dir)
    sequences = collect_unique_sequences(layout, prepared)
    selected = sorted(
        key
        for key in sequences
        if shard_for_key(key, args.shard_count) == args.shard_index
    )
    if args.max_sequences is not None:
        selected = selected[: args.max_sequences]
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard_index must fall within [0, shard_count)")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model_source = Path(args.esm_model_name).expanduser()
    local_weight_path = model_source / "model.safetensors"
    model_weight_sha256 = (
        sha256_file(local_weight_path) if local_weight_path.is_file() else None
    )
    set_reproducible_seed(args.seed, deterministic=True)
    print(f"Loading {args.esm_model_name} on {device}", flush=True)
    tokenizer = EsmTokenizer.from_pretrained(
        args.esm_model_name,
        cache_dir=str(args.hf_cache) if args.hf_cache else None,
        local_files_only=args.local_files_only,
    )
    esm_model = EsmModel.from_pretrained(
        args.esm_model_name,
        cache_dir=str(args.hf_cache) if args.hf_cache else None,
        local_files_only=args.local_files_only,
    ).to(device)
    if int(esm_model.config.hidden_size) != EMBED_DIM:
        raise ValueError(
            f"Expected ESM hidden size {EMBED_DIM}, got {esm_model.config.hidden_size}"
        )
    for parameter in esm_model.parameters():
        parameter.requires_grad = False
    esm_model.eval()

    failures: list[dict[str, str]] = []
    written = 0
    skipped = 0
    started = time.time()
    for position, key in enumerate(selected, 1):
        path = embedding_path(embedding_dir, key)
        try:
            if path.is_file() and args.resume:
                artifact = safe_torch_load(path, map_location="cpu")
                validate_embedding_artifact(artifact, expected_key=key)
                if int(artifact["sequence_length"]) != len(sequences[key]):
                    raise ValueError("Existing embedding sequence length mismatch")
                if (
                    model_weight_sha256 is not None
                    and artifact.get("model_weight_sha256") != model_weight_sha256
                ):
                    raise ValueError("Existing embedding model weight digest mismatch")
                skipped += 1
            else:
                artifact = embed_sequence(
                    sequences[key],
                    tokenizer,
                    esm_model,
                    device,
                    max_tokens=args.max_tokens,
                    stride=args.stride,
                )
                if artifact["sequence_sha256"] != key:
                    raise AssertionError("New embedding digest mismatch")
                if model_weight_sha256 is not None:
                    artifact["model_weight_sha256"] = model_weight_sha256
                atomic_torch_save(artifact, path)
                written += 1
        except Exception as error:
            failures.append(
                {
                    "sequence_sha256": key,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        if position % 20 == 0 or position == len(selected):
            print(
                f"Embedding shard {args.shard_index}: {position}/{len(selected)} "
                f"written={written} skipped={skipped} failures={len(failures)}",
                flush=True,
            )
        if position % 20 == 0:
            release_device_cache(device)

    summary = {
        "format_version": FORMAT_VERSION,
        "stage": "embed",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected_sequence_count": len(selected),
        "written_count": written,
        "skipped_count": skipped,
        "failure_count": len(failures),
        "elapsed_seconds": time.time() - started,
        "model_name": args.esm_model_name,
        "model_weight_sha256": model_weight_sha256,
        "max_tokens": args.max_tokens,
        "stride": args.stride,
        "storage_dtype": "float32",
        "device": str(device),
        "torch_version": str(torch.__version__),
    }
    shard_report = embedding_dir / "reports" / f"shard_{args.shard_index:04d}.json"
    atomic_write_json(shard_report, {**summary, "failures": failures})
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"Embedding shard failed for {len(failures)} sequences")


def role_columns(role: str) -> tuple[str, str, str, str, str]:
    """Return prepared columns for query/context keys, lengths, labels, and split."""
    if role == "paratope":
        return (
            "ab_sequence_sha256",
            "ag_sequence_sha256",
            "ab_length",
            "ab_labels",
            "paratope_split",
        )
    if role == "epitope":
        return (
            "ag_sequence_sha256",
            "ab_sequence_sha256",
            "ag_length",
            "ag_labels",
            "epitope_split",
        )
    raise ValueError(f"Unsupported role: {role!r}")


class EmbeddingStore:
    """Bounded CPU LRU cache for validated embedding artifacts."""

    def __init__(self, root: Path, max_cache_gib: float) -> None:
        if max_cache_gib < 0:
            raise ValueError("max_cache_gib cannot be negative")
        self.root = root
        self.max_cache_bytes = int(max_cache_gib * 1024**3)
        self.cache: OrderedDict[str, tuple[Mapping[str, Any], int]] = OrderedDict()
        self.cache_bytes = 0
        self.hits = 0
        self.misses = 0

    @staticmethod
    def artifact_bytes(artifact: Mapping[str, Any]) -> int:
        """Return the exact tensor payload size retained by an embedding artifact."""
        return sum(int(chunk.numel() * chunk.element_size()) for chunk in artifact["chunks"])

    def load(self, sequence_key: str, device: Any) -> tuple[list[Any], list[int], int]:
        """Load a validated artifact, update the LRU, and transfer chunks to a device."""
        import torch

        if sequence_key in self.cache:
            artifact, artifact_size = self.cache.pop(sequence_key)
            self.cache[sequence_key] = (artifact, artifact_size)
            self.hits += 1
        else:
            artifact = safe_torch_load(
                embedding_path(self.root, sequence_key),
                map_location="cpu",
            )
            validate_embedding_artifact(artifact, expected_key=sequence_key)
            artifact_size = self.artifact_bytes(artifact)
            self.misses += 1
            if artifact_size <= self.max_cache_bytes:
                while self.cache and self.cache_bytes + artifact_size > self.max_cache_bytes:
                    _, (_, evicted_size) = self.cache.popitem(last=False)
                    self.cache_bytes -= evicted_size
                self.cache[sequence_key] = (artifact, artifact_size)
                self.cache_bytes += artifact_size
        chunks = [chunk.to(device=device, dtype=torch.float32) for chunk in artifact["chunks"]]
        starts = [int(value) for value in artifact["starts"]]
        return chunks, starts, int(artifact["sequence_length"])

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe cache usage and hit statistics."""
        return {
            "max_cache_bytes": self.max_cache_bytes,
            "cache_bytes": self.cache_bytes,
            "cached_sequence_count": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
        }


def load_training_example(
    row: pd.Series,
    role: str,
    prepare_dir: Path,
    embedding_store: EmbeddingStore,
    device: Any,
    context_key_override: str | None = None,
) -> tuple[list[Any], list[int], int, list[Any], Any]:
    """Load one role-specific query/context embedding pair and its labels."""
    import torch

    query_key_column, context_key_column, length_column, label_key, _ = role_columns(role)
    query_key = str(row[query_key_column])
    context_key = (
        str(context_key_override)
        if context_key_override is not None
        else str(row[context_key_column])
    )
    query_chunks, query_starts, query_length = embedding_store.load(query_key, device)
    context_chunks, _, _ = embedding_store.load(context_key, device)
    expected_length = int(row[length_column])
    if query_length != expected_length:
        raise ValueError(
            f"Embedding length {query_length} disagrees with prepared length {expected_length}"
        )
    label_path = prepare_dir / "labels" / f"{row.name}.npz"
    with np.load(label_path, allow_pickle=False) as artifact:
        labels = np.asarray(artifact[label_key], dtype=np.float32)
    if labels.shape != (query_length,) or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError(f"Invalid {role} labels for {row.name}")
    label_tensor = torch.from_numpy(labels).to(device=device, dtype=torch.float32)
    return query_chunks, query_starts, query_length, context_chunks, label_tensor


def release_device_cache(device: Any) -> None:
    """Collect released tensors and return cached GPU blocks to the allocator."""
    gc.collect()
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.empty_cache()


def evaluate_validation_loss(
    model: Any,
    frame: pd.DataFrame,
    role: str,
    prepare_dir: Path,
    embedding_store: EmbeddingStore,
    device: Any,
    criterion: Any,
    max_instances: int | None = None,
) -> float:
    """Compute mean per-instance BCE over the complete role-specific validation set."""
    import torch

    _, _, _, _, split_column = role_columns(role)
    validation = frame.loc[frame[split_column] == "validation"].sort_index()
    if max_instances is not None:
        validation = validation.iloc[:max_instances]
    if validation.empty:
        raise ValueError(f"No validation instances are available for {role}")
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for _, row in validation.iterrows():
            example = load_training_example(
                row,
                role,
                prepare_dir,
                embedding_store,
                device,
            )
            query_chunks, query_starts, query_length, context_chunks, labels = example
            pooled = model(query_chunks, context_chunks)
            logits = merge_chunk_logits_to_full(
                pooled,
                query_starts,
                query_length,
                device,
            )
            total += float(criterion(logits, labels).item())
            del pooled, logits, labels, query_chunks, context_chunks, example
    return total / len(validation)


def command_train(args: argparse.Namespace) -> None:
    """Train one role-specific released SPPIDER-seq head with resumable checkpoints."""
    require_torch()
    import torch

    set_reproducible_seed(args.seed, deterministic=not args.allow_nondeterministic)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.max_epochs > 50:
        raise ValueError("The paper limits training to a maximum of 50 epochs")
    if args.patience != 10:
        print(
            "WARNING: patience differs from the paper's ten-epoch setting",
            file=sys.stderr,
            flush=True,
        )

    prepare_dir = args.prepare_dir.expanduser().resolve()
    embedding_dir = args.embedding_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_prepared_instances(prepare_dir)
    embedding_store = EmbeddingStore(embedding_dir, args.embedding_cache_gib)
    _, _, _, _, split_column = role_columns(args.role)
    train_frame = frame.loc[frame[split_column] == "train"].sort_index()
    if args.max_instances is not None:
        train_frame = train_frame.iloc[: args.max_instances]
    if train_frame.empty:
        raise ValueError(f"No training instances are available for {args.role}")

    model = ChunkwiseInteractionModel(EMBED_DIM, NUM_HEADS).to(device)
    if args.initial_checkpoint is not None:
        initial_payload = safe_torch_load(args.initial_checkpoint, map_location=device)
        model.load_state_dict(extract_model_state(initial_payload), strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.json"

    start_epoch = 1
    best_validation_loss = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    if args.resume and last_path.is_file():
        checkpoint = safe_torch_load(last_path, map_location=device)
        if checkpoint.get("format_version") != FORMAT_VERSION:
            raise ValueError("Existing training checkpoint format is incompatible")
        if checkpoint.get("role") != args.role:
            raise ValueError("Existing training checkpoint role is incompatible")
        previous_config = checkpoint.get("config", {})
        expected_resume_values = {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "optimizer": "Adam",
            "batch_size": 1,
        }
        for key, expected in expected_resume_values.items():
            if previous_config.get(key) != expected:
                raise ValueError(
                    f"Resume configuration mismatch for {key}: "
                    f"checkpoint={previous_config.get(key)!r}, requested={expected!r}"
                )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history = list(checkpoint.get("history", []))
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state["torch_cpu"].to(device="cpu"))
            if device.type == "cuda" and rng_state.get("torch_cuda"):
                torch.cuda.set_rng_state_all(
                    [state.to(device="cpu") for state in rng_state["torch_cuda"]]
                )
        else:
            torch.manual_seed(args.seed + start_epoch)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + start_epoch)

    config = {
        "architecture": "released SPPIDER-seq v2.1.0 ChunkwiseInteractionModel",
        "role": args.role,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "dropout": 0.10,
        "loss": "BCEWithLogitsLoss(mean), no class weighting",
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": 1,
        "embedding_cache_gib": args.embedding_cache_gib,
        "seed": args.seed,
        "initial_checkpoint": str(args.initial_checkpoint) if args.initial_checkpoint else None,
        "paper_disclosed": [
            "frozen ESM-2 esm2_t33_650M_UR50D embeddings",
            "16 cross-attention heads",
            "binary cross-entropy loss",
            "maximum 50 epochs",
            "early stopping patience 10",
            "best validation-loss checkpoint selection",
            "1022-residue windows with stride 512 and max overlap pooling",
        ],
        "paper_not_disclosed": [
            "optimizer",
            "learning rate",
            "weight decay",
            "batch size",
            "random seed",
        ],
    }
    if start_epoch > args.max_epochs or epochs_without_improvement >= args.patience:
        print("Training is already complete according to the last checkpoint")
        return

    stop_epoch = args.max_epochs
    if args.max_epochs_this_run is not None:
        stop_epoch = min(stop_epoch, start_epoch + args.max_epochs_this_run - 1)
    for epoch in range(start_epoch, stop_epoch + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_frame))
        training_loss = 0.0
        epoch_started = time.time()
        for position, row_index in enumerate(order, 1):
            row = train_frame.iloc[int(row_index)]
            example = load_training_example(
                row,
                args.role,
                prepare_dir,
                embedding_store,
                device,
            )
            query_chunks, query_starts, query_length, context_chunks, labels = example
            optimizer.zero_grad(set_to_none=True)
            pooled = model(query_chunks, context_chunks)
            logits = merge_chunk_logits_to_full(
                pooled,
                query_starts,
                query_length,
                device,
            )
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss for {row.name}")
            loss.backward()
            optimizer.step()
            training_loss += float(loss.item())
            del loss, pooled, logits, labels, query_chunks, context_chunks, example
            if position % args.log_every == 0 or position == len(train_frame):
                print(
                    f"epoch={epoch} train={position}/{len(train_frame)} "
                    f"mean_loss={training_loss / position:.8f}",
                    flush=True,
                )
        release_device_cache(device)

        validation_loss = evaluate_validation_loss(
            model,
            frame,
            args.role,
            prepare_dir,
            embedding_store,
            device,
            criterion,
            max_instances=args.max_validation_instances,
        )
        release_device_cache(device)
        mean_training_loss = training_loss / len(train_frame)
        improved = validation_loss < best_validation_loss - args.min_delta
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch,
            "training_loss": mean_training_loss,
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "elapsed_seconds": time.time() - epoch_started,
        }
        history.append(record)
        payload = {
            "format_version": FORMAT_VERSION,
            "role": args.role,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "config": config,
            "embedding_cache": embedding_store.summary(),
            "rng_state": {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else [],
            },
            "torch_version": str(torch.__version__),
            "device": str(device),
        }
        atomic_torch_save(payload, last_path)
        if improved:
            atomic_torch_save(payload, best_path)
        atomic_write_json(
            history_path,
            {
                "format_version": FORMAT_VERSION,
                "role": args.role,
                "config": config,
                "history": history,
            },
        )
        print(json.dumps(record, sort_keys=True), flush=True)
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break


def instance_shard(instance_id: str, shard_count: int) -> int:
    """Assign an instance to a deterministic inference shard."""
    digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % shard_count


def build_partner_context_mapping(
    frame: pd.DataFrame,
    role: str,
    mode: str,
    seed: int,
) -> dict[str, str]:
    """Build a deterministic matched or sequence-deranged partner assignment."""
    if frame.empty:
        raise ValueError("Partner assignment requires at least one instance")
    _, context_key_column, _, _, _ = role_columns(role)
    ordered = frame.sort_index()
    instance_ids = [str(value) for value in ordered.index]
    context_keys = ordered[context_key_column].astype(str).to_numpy()
    if mode == "matched":
        return dict(zip(instance_ids, context_keys, strict=True))
    if mode != "partner_permuted":
        raise ValueError(f"Unsupported partner assignment mode: {mode!r}")
    if len(instance_ids) < 2:
        raise ValueError("Partner permutation requires at least two instances")

    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(seed)
    same_sequence_cost = (context_keys[:, None] == context_keys[None, :]).astype(np.float64)
    tie_breaker = rng.random(same_sequence_cost.shape) * 1e-6
    row_indices, column_indices = linear_sum_assignment(same_sequence_cost + tie_breaker)
    if not np.array_equal(row_indices, np.arange(len(instance_ids))):
        raise AssertionError("Unexpected row ordering from partner assignment solver")
    candidate = context_keys[column_indices]
    if np.any(candidate == context_keys):
        raise ValueError(
            "Could not construct a partner permutation without an unchanged context sequence"
        )
    return dict(zip(instance_ids, candidate.tolist(), strict=True))


def mapping_sha256(mapping: Mapping[str, str]) -> str:
    """Return a canonical digest for an instance-to-context assignment."""
    payload = json.dumps(dict(sorted(mapping.items())), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_inference_artifact(
    path: Path,
    instance_id: str,
    role: str,
    expected_length: int,
    checkpoint_sha256: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load one inference artifact and enforce its identity, role, and score axis."""
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "format_version",
            "instance_id",
            "role",
            "checkpoint_sha256",
            "scores",
            "labels",
        }
        if set(artifact.files) != required:
            raise ValueError(f"Unexpected inference artifact keys in {path}")
        format_version = str(artifact["format_version"].item())
        observed_instance = str(artifact["instance_id"].item())
        observed_role = str(artifact["role"].item())
        observed_checkpoint = str(artifact["checkpoint_sha256"].item())
        scores = np.asarray(artifact["scores"], dtype=np.float32)
        labels = np.asarray(artifact["labels"], dtype=np.uint8)
    if format_version != FORMAT_VERSION:
        raise ValueError(f"Unsupported inference artifact format in {path}")
    if observed_instance != instance_id or observed_role != role:
        raise ValueError(f"Inference artifact identity mismatch in {path}")
    if checkpoint_sha256 is not None and observed_checkpoint != checkpoint_sha256:
        raise ValueError(f"Inference checkpoint digest mismatch in {path}")
    if scores.shape != (expected_length,) or labels.shape != (expected_length,):
        raise ValueError(f"Inference residue axis mismatch in {path}")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError(f"Invalid probability scores in {path}")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"Invalid binary labels in {path}")
    return scores, labels, observed_checkpoint


def command_ensemble_inference(args: argparse.Namespace) -> None:
    """Average aligned per-instance inference artifacts with fixed weights."""
    prepare_dir = args.prepare_dir.expanduser().resolve()
    input_dirs = [path.expanduser().resolve() for path in args.input_dirs]
    output_dir = args.output_dir.expanduser().resolve()
    artifact_dir = output_dir / "instances"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if len(input_dirs) < 2:
        raise ValueError("An inference ensemble requires at least two input directories")
    if args.weights is None:
        weights = np.full(len(input_dirs), 1.0 / len(input_dirs), dtype=np.float64)
    else:
        weights = np.asarray(args.weights, dtype=np.float64)
        if weights.shape != (len(input_dirs),):
            raise ValueError("Inference ensemble weight count does not match input count")
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Inference ensemble weights must be finite and non-negative")
        weights = weights / weights.sum()

    prepared = load_prepared_instances(prepare_dir)
    _, _, length_column, _, split_column = role_columns(args.role)
    selected = prepared.loc[prepared[split_column] == "test"].sort_index()
    component_checkpoint_sets = [set() for _ in input_dirs]
    ensemble_payload = json.dumps(
        {
            "role": args.role,
            "inputs": [str(path) for path in input_dirs],
            "weights": weights.tolist(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    ensemble_digest = hashlib.sha256(ensemble_payload.encode("utf-8")).hexdigest()
    for position, (instance_id, row) in enumerate(selected.iterrows(), 1):
        expected_length = int(row[length_column])
        component_scores = []
        reference_labels = None
        for component_index, input_dir in enumerate(input_dirs):
            scores, labels, checkpoint_digest = validate_inference_artifact(
                input_dir / "instances" / f"{instance_id}.npz",
                str(instance_id),
                args.role,
                expected_length,
            )
            component_checkpoint_sets[component_index].add(checkpoint_digest)
            if reference_labels is None:
                reference_labels = labels
            elif not np.array_equal(reference_labels, labels):
                raise ValueError(f"Inference ensemble labels disagree for {instance_id}")
            component_scores.append(scores.astype(np.float64, copy=False))
        ensemble_scores = np.average(
            np.stack(component_scores, axis=0),
            axis=0,
            weights=weights,
        ).astype(np.float32)
        output_path = artifact_dir / f"{instance_id}.npz"
        atomic_write_npz(
            output_path,
            format_version=np.asarray(FORMAT_VERSION),
            instance_id=np.asarray(str(instance_id)),
            role=np.asarray(args.role),
            checkpoint_sha256=np.asarray(ensemble_digest),
            scores=ensemble_scores,
            labels=reference_labels.astype(np.uint8, copy=False),
        )
        validate_inference_artifact(
            output_path,
            str(instance_id),
            args.role,
            expected_length,
            checkpoint_sha256=ensemble_digest,
        )
        if position % 100 == 0 or position == len(selected):
            print(f"Ensembled {position}/{len(selected)} instances", flush=True)
    for component_index, checkpoint_set in enumerate(component_checkpoint_sets):
        if len(checkpoint_set) != 1:
            raise ValueError(
                f"Input inference directory {input_dirs[component_index]} contains "
                "multiple checkpoint digests"
            )
    result = {
        "format_version": FORMAT_VERSION,
        "stage": "ensemble_inference",
        "role": args.role,
        "instance_count": len(selected),
        "ensemble_digest": ensemble_digest,
        "inputs": [
            {
                "path": str(path),
                "weight": float(weight),
                "checkpoint_sha256": next(iter(checkpoint_set)),
            }
            for path, weight, checkpoint_set in zip(
                input_dirs,
                weights,
                component_checkpoint_sets,
                strict=True,
            )
        ],
    }
    atomic_write_json(output_dir / "ensemble_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def command_infer(args: argparse.Namespace) -> None:
    """Run one deterministic test shard with a trained or official checkpoint."""
    require_torch()
    import torch

    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard_index must fall within [0, shard_count)")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_reproducible_seed(args.seed, deterministic=not args.allow_nondeterministic)

    prepare_dir = args.prepare_dir.expanduser().resolve()
    embedding_dir = args.embedding_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    artifact_dir = output_dir / "instances"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frame = load_prepared_instances(prepare_dir)
    embedding_store = EmbeddingStore(embedding_dir, args.embedding_cache_gib)
    _, _, length_column, label_key, split_column = role_columns(args.role)
    test_frame = frame.loc[frame[split_column] == "test"].sort_index()
    selected = test_frame.loc[
        [
            instance_shard(instance_id, args.shard_count) == args.shard_index
            for instance_id in test_frame.index
        ]
    ]
    if args.max_instances is not None:
        selected = selected.iloc[: args.max_instances]
    if selected.empty:
        raise ValueError("Selected inference shard is empty")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_digest = sha256_file(checkpoint_path)
    payload = safe_torch_load(checkpoint_path, map_location=device)
    model = ChunkwiseInteractionModel(EMBED_DIM, NUM_HEADS).to(device)
    model.load_state_dict(extract_model_state(payload), strict=True)
    model.eval()

    failures: list[dict[str, str]] = []
    written = 0
    skipped = 0
    started = time.time()
    for position, (instance_id, row) in enumerate(selected.iterrows(), 1):
        path = artifact_dir / f"{instance_id}.npz"
        expected_length = int(row[length_column])
        try:
            if path.is_file() and args.resume:
                validate_inference_artifact(
                    path,
                    instance_id,
                    args.role,
                    expected_length,
                    checkpoint_sha256=checkpoint_digest,
                )
                skipped += 1
            else:
                example = load_training_example(
                    row,
                    args.role,
                    prepare_dir,
                    embedding_store,
                    device,
                )
                query_chunks, query_starts, query_length, context_chunks, labels = example
                scores = predict_residue_scores(
                    model,
                    query_chunks,
                    query_starts,
                    query_length,
                    context_chunks,
                    device,
                )
                label_array = labels.to(device="cpu", dtype=torch.uint8).numpy()
                atomic_write_npz(
                    path,
                    format_version=np.asarray(FORMAT_VERSION),
                    instance_id=np.asarray(instance_id),
                    role=np.asarray(args.role),
                    checkpoint_sha256=np.asarray(checkpoint_digest),
                    scores=scores.astype(np.float32, copy=False),
                    labels=label_array.astype(np.uint8, copy=False),
                )
                validate_inference_artifact(
                    path,
                    instance_id,
                    args.role,
                    expected_length,
                    checkpoint_sha256=checkpoint_digest,
                )
                written += 1
                del scores, label_array, labels, query_chunks, context_chunks, example
        except Exception as error:
            failures.append(
                {
                    "instance_id": instance_id,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        if position % 20 == 0 or position == len(selected):
            print(
                f"Inference shard {args.shard_index}: {position}/{len(selected)} "
                f"written={written} skipped={skipped} failures={len(failures)}",
                flush=True,
            )
            release_device_cache(device)

    summary = {
        "format_version": FORMAT_VERSION,
        "stage": "infer",
        "role": args.role,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected_instance_count": len(selected),
        "written_count": written,
        "skipped_count": skipped,
        "failure_count": len(failures),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "embedding_cache": embedding_store.summary(),
    }
    report_path = output_dir / "reports" / f"shard_{args.shard_index:04d}.json"
    atomic_write_json(report_path, {**summary, "failures": failures})
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"Inference shard failed for {len(failures)} instances")


def command_evaluate(args: argparse.Namespace) -> None:
    """Evaluate one checkpoint on a prepared split with a controlled partner assignment."""
    require_torch()
    import torch

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    set_reproducible_seed(args.seed, deterministic=not args.allow_nondeterministic)

    prepare_dir = args.prepare_dir.expanduser().resolve()
    embedding_dir = args.embedding_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    frame = load_prepared_instances(prepare_dir)
    _, _, _, _, split_column = role_columns(args.role)
    selected = frame.loc[frame[split_column] == args.split].sort_index()
    if args.max_instances is not None:
        selected = selected.iloc[: args.max_instances]
    if selected.empty:
        raise ValueError(f"No {args.role} instances are available for split {args.split!r}")
    context_mapping = build_partner_context_mapping(
        selected,
        args.role,
        args.context_mode,
        args.context_seed,
    )

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_digest = sha256_file(checkpoint_path)
    payload = safe_torch_load(checkpoint_path, map_location=device)
    model = ChunkwiseInteractionModel(EMBED_DIM, NUM_HEADS).to(device)
    model.load_state_dict(extract_model_state(payload), strict=True)
    model.eval()
    embedding_store = EmbeddingStore(embedding_dir, args.embedding_cache_gib)

    labels_per_instance: list[np.ndarray] = []
    scores_per_instance: list[np.ndarray] = []
    prediction_frames: list[pd.DataFrame] = []
    macro_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.time()
    for position, (instance_id, row) in enumerate(selected.iterrows(), 1):
        try:
            example = load_training_example(
                row,
                args.role,
                prepare_dir,
                embedding_store,
                device,
                context_key_override=context_mapping[str(instance_id)],
            )
            query_chunks, query_starts, query_length, context_chunks, labels = example
            scores = predict_residue_scores(
                model,
                query_chunks,
                query_starts,
                query_length,
                context_chunks,
                device,
            )
            label_array = labels.to(device="cpu", dtype=torch.uint8).numpy()
            labels_per_instance.append(label_array)
            scores_per_instance.append(scores)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "instance_id": str(instance_id),
                        "role": args.role,
                        "res_idx": np.arange(len(label_array), dtype=np.int64),
                        "label": label_array.astype(np.uint8, copy=False),
                        "score": scores.astype(np.float32, copy=False),
                    }
                )
            )
            macro_records.append(binary_metrics(label_array, scores, args.threshold))
            del scores, label_array, labels, query_chunks, context_chunks, example
        except Exception as error:
            failures.append(
                {
                    "instance_id": str(instance_id),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        if position % 20 == 0 or position == len(selected):
            print(
                f"Evaluation {args.context_mode}: {position}/{len(selected)} "
                f"failures={len(failures)}",
                flush=True,
            )
            release_device_cache(device)
    if failures:
        atomic_write_json(
            output_path,
            {
                "format_version": FORMAT_VERSION,
                "stage": "evaluate",
                "role": args.role,
                "split": args.split,
                "context_mode": args.context_mode,
                "failures": failures,
            },
        )
        raise RuntimeError(f"Checkpoint evaluation failed for {len(failures)} instances")

    metric_names = ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "mcc")
    prediction_path = (
        args.residue_predictions.expanduser().resolve()
        if args.residue_predictions is not None
        else None
    )
    prediction_digest = None
    if prediction_path is not None:
        prediction_frame = pd.concat(prediction_frames, ignore_index=True)
        atomic_write_parquet(
            pa.Table.from_pandas(prediction_frame, preserve_index=False),
            prediction_path,
        )
        prediction_digest = sha256_file(prediction_path)
    result = {
        "format_version": FORMAT_VERSION,
        "stage": "evaluate",
        "role": args.role,
        "split": args.split,
        "context_mode": args.context_mode,
        "context_seed": args.context_seed,
        "context_mapping_sha256": mapping_sha256(context_mapping),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "instance_count": len(selected),
        "threshold": args.threshold,
        "residue_predictions": str(prediction_path) if prediction_path is not None else None,
        "residue_predictions_sha256": prediction_digest,
        "residue_micro": binary_metrics(
            np.concatenate(labels_per_instance),
            np.concatenate(scores_per_instance),
            args.threshold,
        ),
        "residue_macro_mean": {
            name: mean_optional(record[name] for record in macro_records)
            for name in metric_names
        },
        "embedding_cache": embedding_store.summary(),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "failures": [],
    }
    atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute strict JSON-safe binary metrics for one residue or projection axis."""
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Metric labels and scores must be aligned one-dimensional arrays")
    if len(labels) == 0 or not np.isfinite(scores).all():
        raise ValueError("Metrics require non-empty finite scores")
    predictions = scores >= threshold
    both_classes = np.unique(labels).size == 2
    return {
        "count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "positive_fraction": float(labels.mean()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "auprc": float(average_precision_score(labels, scores)) if both_classes else None,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)) if both_classes else None,
    }


def mean_optional(values: Iterable[float | None]) -> float | None:
    """Return the finite mean after omitting undefined single-class metrics."""
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def pair_row_group_instance_id(pair_file: pq.ParquetFile, row_group: int) -> str:
    """Resolve one row group's single instance ID using statistics when available."""
    metadata = pair_file.metadata.row_group(row_group).column(0)
    statistics = metadata.statistics
    if statistics is not None and statistics.has_min_max and statistics.min == statistics.max:
        value = statistics.min
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    column = pair_file.read_row_group(row_group, columns=["instance_id"])["instance_id"]
    values = column.unique().to_pylist()
    if len(values) != 1:
        raise ValueError(f"Pair row group {row_group} spans multiple instances")
    return str(values[0])


def verify_prediction_file(
    source: pq.ParquetFile,
    prediction_path: Path,
    test_ids: set[str],
) -> dict[str, Any]:
    """Independently compare every emitted key against its authoritative source row."""
    predictions = pq.ParquetFile(prediction_path)
    if predictions.schema_arrow != PREDICTION_SCHEMA:
        raise ValueError(f"Prediction schema mismatch: {predictions.schema_arrow}")
    expected_groups = [
        row_group
        for row_group in range(source.num_row_groups)
        if pair_row_group_instance_id(source, row_group) in test_ids
    ]
    if predictions.num_row_groups != len(expected_groups):
        raise ValueError("Prediction row-group count does not match the frozen test instances")
    total_rows = 0
    for prediction_group, source_group in enumerate(expected_groups):
        expected = source.read_row_group(
            source_group,
            columns=["instance_id", "ab_res_idx", "ag_res_idx"],
        )
        observed = predictions.read_row_group(
            prediction_group,
            columns=["instance_id", "ab_res_idx", "ag_res_idx", "score"],
        )
        for column in ("instance_id", "ab_res_idx", "ag_res_idx"):
            if not expected[column].equals(observed[column]):
                raise ValueError(
                    f"Prediction key mismatch in row group {prediction_group}, column {column}"
                )
        score_array = observed["score"].to_numpy(zero_copy_only=False)
        if not np.isfinite(score_array).all():
            raise ValueError(f"Non-finite scores in prediction row group {prediction_group}")
        total_rows += len(observed)
    return {
        "row_count": total_rows,
        "row_group_count": predictions.num_row_groups,
        "schema": str(predictions.schema_arrow),
        "key_equality": True,
        "null_score_count": 0,
        "nonfinite_score_count": 0,
        "prediction_sha256": sha256_file(prediction_path),
    }


def command_assemble(args: argparse.Namespace) -> None:
    """Assemble complete marginal projections and native residue evaluation outputs."""
    layout = BenchmarkLayout.from_root(args.benchmark_root)
    layout.validate(verify_manifest_hash=not args.skip_manifest_hash)
    prepare_dir = args.prepare_dir.expanduser().resolve()
    inference_dir = args.inference_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = load_prepared_instances(prepare_dir)
    _, _, length_column, _, split_column = role_columns(args.role)
    test_frame = prepared.loc[prepared[split_column] == "test"].sort_index()
    test_ids = set(test_frame.index)
    if len(test_ids) != 455 and not args.allow_nonstandard_test_count:
        raise ValueError(f"Expected 455 frozen test instances, observed {len(test_ids)}")

    pair_file = pq.ParquetFile(layout.frozen / "pairs.parquet")
    prediction_path = output_dir / "predictions.parquet"
    residue_path = output_dir / "residue_predictions.parquet"
    prediction_temporary = prediction_path.with_name(f".{prediction_path.name}.{os.getpid()}.tmp")
    residue_temporary = residue_path.with_name(f".{residue_path.name}.{os.getpid()}.tmp")
    prediction_writer = pq.ParquetWriter(prediction_temporary, PREDICTION_SCHEMA, compression="zstd")
    residue_writer = pq.ParquetWriter(residue_temporary, RESIDUE_SCHEMA, compression="zstd")

    projection_labels: list[np.ndarray] = []
    projection_scores: list[np.ndarray] = []
    residue_labels: list[np.ndarray] = []
    residue_scores: list[np.ndarray] = []
    macro_projection: list[dict[str, Any]] = []
    macro_residue: list[dict[str, Any]] = []
    checkpoint_digest: str | None = None
    observed_ids: set[str] = set()
    try:
        for row_group in range(pair_file.num_row_groups):
            instance_id = pair_row_group_instance_id(pair_file, row_group)
            if instance_id not in test_ids:
                continue
            row = test_frame.loc[instance_id]
            scores, labels, observed_checkpoint = validate_inference_artifact(
                inference_dir / "instances" / f"{instance_id}.npz",
                instance_id,
                args.role,
                int(row[length_column]),
                checkpoint_sha256=checkpoint_digest,
            )
            checkpoint_digest = observed_checkpoint
            prepared_labels_path = prepare_dir / "labels" / f"{instance_id}.npz"
            with np.load(prepared_labels_path, allow_pickle=False) as artifact:
                expected_labels = np.asarray(
                    artifact["ab_labels" if args.role == "paratope" else "ag_labels"],
                    dtype=np.uint8,
                )
            if not np.array_equal(labels, expected_labels):
                raise ValueError(f"Inference labels disagree with preparation for {instance_id}")

            pair_table = pair_file.read_row_group(
                row_group,
                columns=["instance_id", "ab_res_idx", "ag_res_idx", "label"],
            )
            score_indices = pair_table[
                "ab_res_idx" if args.role == "paratope" else "ag_res_idx"
            ].to_numpy(zero_copy_only=False)
            candidate_scores = scores[score_indices].astype(np.float32, copy=False)
            candidate_labels = pair_table["label"].to_numpy(zero_copy_only=False).astype(
                np.bool_,
                copy=False,
            )
            output_table = pa.Table.from_arrays(
                [
                    pair_table["instance_id"],
                    pair_table["ab_res_idx"],
                    pair_table["ag_res_idx"],
                    pa.array(candidate_scores, type=pa.float32()),
                ],
                schema=PREDICTION_SCHEMA,
            )
            prediction_writer.write_table(output_table, row_group_size=len(output_table))
            residue_table = pa.Table.from_arrays(
                [
                    pa.array([instance_id] * len(scores), type=pa.string()),
                    pa.array([args.role] * len(scores), type=pa.string()),
                    pa.array(np.arange(len(scores), dtype=np.int64), type=pa.int64()),
                    pa.array(labels.astype(np.bool_, copy=False), type=pa.bool_()),
                    pa.array(scores, type=pa.float32()),
                ],
                schema=RESIDUE_SCHEMA,
            )
            residue_writer.write_table(residue_table, row_group_size=len(residue_table))

            projection_labels.append(candidate_labels)
            projection_scores.append(candidate_scores)
            residue_labels.append(labels.astype(np.bool_, copy=False))
            residue_scores.append(scores)
            macro_projection.append(
                binary_metrics(candidate_labels, candidate_scores, args.threshold)
            )
            macro_residue.append(binary_metrics(labels, scores, args.threshold))
            observed_ids.add(instance_id)
            if len(observed_ids) % 25 == 0 or len(observed_ids) == len(test_ids):
                print(f"Assembled {len(observed_ids)}/{len(test_ids)} instances", flush=True)
        prediction_writer.close()
        residue_writer.close()
        if observed_ids != test_ids:
            raise ValueError(f"Missing inference artifacts for {len(test_ids - observed_ids)} instances")
        os.replace(prediction_temporary, prediction_path)
        os.replace(residue_temporary, residue_path)
    except Exception:
        try:
            prediction_writer.close()
            residue_writer.close()
        finally:
            prediction_temporary.unlink(missing_ok=True)
            residue_temporary.unlink(missing_ok=True)
        raise

    projection_micro = binary_metrics(
        np.concatenate(projection_labels),
        np.concatenate(projection_scores),
        args.threshold,
    )
    residue_micro = binary_metrics(
        np.concatenate(residue_labels),
        np.concatenate(residue_scores),
        args.threshold,
    )
    metric_names = ("auroc", "auprc", "accuracy", "precision", "recall", "f1", "mcc")
    projection_macro = {
        name: mean_optional(record[name] for record in macro_projection)
        for name in metric_names
    }
    residue_macro = {
        name: mean_optional(record[name] for record in macro_residue)
        for name in metric_names
    }
    score = {
        "format_version": FORMAT_VERSION,
        "role": args.role,
        "projection": args.role,
        "candidate_pair_score_definition": (
            "paratope_score[ab_res_idx]"
            if args.role == "paratope"
            else "epitope_score[ag_res_idx]"
        ),
        "native_pair_interaction_score": False,
        "threshold": args.threshold,
        "projection_micro": projection_micro,
        "projection_macro_mean": projection_macro,
        "residue_micro": residue_micro,
        "residue_macro_mean": residue_macro,
        "test_instance_count": len(test_ids),
    }
    verification = verify_prediction_file(pair_file, prediction_path, test_ids)
    provenance = {
        "format_version": FORMAT_VERSION,
        "created_at_unix": time.time(),
        "benchmark_root": str(layout.root),
        "benchmark_manifest_sha256": sha256_file(layout.frozen / "MANIFEST.json"),
        "role": args.role,
        "checkpoint_sha256": checkpoint_digest,
        "source_release": "SPPIDER-seq v2.1.0",
        "source_commit": "241d08e1dcee6d5f4ca444468db96a4707ec6f10",
        "architecture": "released ChunkwiseInteractionModel",
        "esm_model": ESM_MODEL_NAME,
        "projection_warning": (
            "SPPIDER-seq emits marginal residue probabilities, not a native "
            "antibody-antigen pair interaction matrix."
        ),
        "python_version": platform.python_version(),
        "command": " ".join(sys.argv),
    }
    atomic_write_json(output_dir / "score.json", score)
    atomic_write_json(output_dir / "verification.json", verification)
    atomic_write_json(output_dir / "provenance.json", provenance)
    print(json.dumps({"score": score, "verification": verification}, indent=2, sort_keys=True))


def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared frozen-delivery arguments to a command parser."""
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        required=True,
        help="Materialized SAbDab2 delivery root.",
    )
    parser.add_argument(
        "--skip-manifest-hash",
        action="store_true",
        help="Skip the fixed MANIFEST.json digest check for synthetic tests only.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the staged SPPIDER-seq SAbDab2 command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build authoritative residue labels.")
    add_benchmark_arguments(prepare)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--canonical-manifest", type=Path)
    prepare.add_argument("--validation-fraction", type=float, default=0.10)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    prepare.set_defaults(handler=command_prepare)

    embed = subparsers.add_parser("embed", help="Precompute unique ESM-2 embeddings.")
    add_benchmark_arguments(embed)
    embed.add_argument("--prepare-dir", type=Path, required=True)
    embed.add_argument("--embedding-dir", type=Path, required=True)
    embed.add_argument("--shard-index", type=int, required=True)
    embed.add_argument("--shard-count", type=int, required=True)
    embed.add_argument("--max-sequences", type=int)
    embed.add_argument("--esm-model-name", default=ESM_MODEL_NAME)
    embed.add_argument("--hf-cache", type=Path)
    embed.add_argument("--local-files-only", action="store_true")
    embed.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    embed.add_argument("--stride", type=int, default=STRIDE)
    embed.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    embed.add_argument("--seed", type=int, default=42)
    embed.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    embed.set_defaults(handler=command_embed)

    train = subparsers.add_parser("train", help="Train one role-specific model.")
    train.add_argument("--prepare-dir", type=Path, required=True)
    train.add_argument("--embedding-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--role", choices=ROLES, required=True)
    train.add_argument("--initial-checkpoint", type=Path)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--max-epochs", type=int, default=50)
    train.add_argument("--max-epochs-this-run", type=int)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--min-delta", type=float, default=0.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--allow-nondeterministic", action="store_true")
    train.add_argument("--max-instances", type=int)
    train.add_argument("--max-validation-instances", type=int)
    train.add_argument("--embedding-cache-gib", type=float, default=24.0)
    train.add_argument("--log-every", type=int, default=25)
    train.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.set_defaults(handler=command_train)

    infer = subparsers.add_parser("infer", help="Run one test inference shard.")
    infer.add_argument("--prepare-dir", type=Path, required=True)
    infer.add_argument("--embedding-dir", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--role", choices=ROLES, required=True)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--shard-index", type=int, required=True)
    infer.add_argument("--shard-count", type=int, required=True)
    infer.add_argument("--max-instances", type=int)
    infer.add_argument("--embedding-cache-gib", type=float, default=24.0)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    infer.add_argument("--allow-nondeterministic", action="store_true")
    infer.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    infer.set_defaults(handler=command_infer)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a checkpoint with matched or permuted partner contexts.",
    )
    evaluate.add_argument("--prepare-dir", type=Path, required=True)
    evaluate.add_argument("--embedding-dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--residue-predictions", type=Path)
    evaluate.add_argument("--role", choices=ROLES, required=True)
    evaluate.add_argument("--split", choices=("train", "validation", "test"), required=True)
    evaluate.add_argument(
        "--context-mode",
        choices=("matched", "partner_permuted"),
        default="matched",
    )
    evaluate.add_argument("--context-seed", type=int, default=20260722)
    evaluate.add_argument("--threshold", type=float, default=0.50)
    evaluate.add_argument("--max-instances", type=int)
    evaluate.add_argument("--embedding-cache-gib", type=float, default=24.0)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate.add_argument("--allow-nondeterministic", action="store_true")
    evaluate.set_defaults(handler=command_evaluate)

    ensemble = subparsers.add_parser(
        "ensemble-inference",
        help="Average complete per-instance inference artifacts.",
    )
    ensemble.add_argument("--prepare-dir", type=Path, required=True)
    ensemble.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    ensemble.add_argument("--weights", type=float, nargs="+")
    ensemble.add_argument("--output-dir", type=Path, required=True)
    ensemble.add_argument("--role", choices=ROLES, required=True)
    ensemble.set_defaults(handler=command_ensemble_inference)

    assemble = subparsers.add_parser(
        "assemble",
        help="Assemble and independently verify a complete marginal projection.",
    )
    add_benchmark_arguments(assemble)
    assemble.add_argument("--prepare-dir", type=Path, required=True)
    assemble.add_argument("--inference-dir", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--role", choices=ROLES, required=True)
    assemble.add_argument("--threshold", type=float, default=0.50)
    assemble.add_argument("--allow-nonstandard-test-count", action="store_true")
    assemble.set_defaults(handler=command_assemble)
    return parser


def main() -> None:
    """Dispatch one staged adapter command."""
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
