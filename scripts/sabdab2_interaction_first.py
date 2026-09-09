#!/usr/bin/env python3
"""Prepare, split, train, and smoke-test an interaction-first SAbDab2 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from interaction_first.contacts import (
    SparseContactMap,
    load_contact_map,
    save_contact_map,
)
from interaction_first.partners import PartnerRecord, PartnerSampler
from interaction_first.splits import (
    SimilarityEdge,
    audit_component_split,
    build_components,
    deterministic_component_split,
)


FORMAT_VERSION = "sabdab2_interaction_first_v1"
MODEL_FORMAT_VERSION = "sabdab2_interaction_first_v2"
EXPECTED_MANIFEST_SHA256 = "15f7819193d79f637849744b7cc8160b34b81b5bc2f50671c0cd07560b524377"


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write strict JSON through an adjacent temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_dataframe_parquet(frame: Any, path: Path) -> None:
    """Write a pandas frame as compressed Parquet through an adjacent file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def directory_fingerprint(path: Path) -> dict[str, Any]:
    """Identify a data directory without hashing gigabytes of it.

    Embedding and contact stores run to gigabytes across thousands of shards,
    so a content digest is not affordable on every run. The resolved path and
    the file count are affordable, and together they distinguish the stores
    this pipeline can be pointed at.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        return {"path": str(resolved), "exists": False}
    return {
        "path": str(resolved),
        "exists": True,
        "file_count": sum(1 for entry in resolved.rglob("*") if entry.is_file()),
    }


def input_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Record which data a run actually read.

    The receipts named every hyperparameter and the split manifest digest but
    no input directory, so a finished run did not say what it was trained or
    evaluated on. Recovering that for the seed replication meant matching the
    split file by digest and the embedding store by tensor width -- and the
    store that width identified was not the one the directory name suggested.
    """
    prepare = args.prepare_dir.expanduser().resolve()
    instances = prepare / "instances.parquet"
    manifest = args.split_manifest.expanduser().resolve()
    return {
        "coverage": (
            "paths and cheap identifiers only; the contact and embedding "
            "stores are identified by path and file count, not by content"
        ),
        "prepare_dir": {
            "path": str(prepare),
            "instances_parquet_sha256": (
                sha256_file(instances) if instances.is_file() else None
            ),
        },
        "contact_dir": directory_fingerprint(args.contact_dir),
        "embedding_dir": directory_fingerprint(args.embedding_dir),
        "split_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    """Return a stable identifier for a normalized protein sequence."""
    normalized = str(sequence).strip().upper()
    if not normalized:
        raise ValueError("Protein sequences cannot be empty")
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def frozen_root(benchmark_root: Path, verify_hash: bool = True) -> Path:
    """Resolve and validate the frozen SAbDab2 delivery."""
    root = benchmark_root.expanduser().resolve()
    frozen = root / "data/nfs/yangwuyu/baseline-eval/frozen/sabdab2"
    required = (
        frozen / "pairs.parquet",
        frozen / "instances.parquet",
        frozen / "splits.json",
        frozen / "contamination.json",
        frozen / "MANIFEST.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen benchmark files: {missing}")
    if verify_hash:
        observed = sha256_file(frozen / "MANIFEST.json")
        if observed != EXPECTED_MANIFEST_SHA256:
            raise ValueError(
                "Frozen MANIFEST.json digest mismatch: "
                f"expected={EXPECTED_MANIFEST_SHA256}, observed={observed}"
            )
    return frozen


def _normalize_fraction(value: float) -> float:
    """Accept either fractional or percentage-form identity values."""
    value = float(value)
    if value > 1.0:
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Similarity fraction is outside [0, 1]: {value}")
    return value


def command_prepare_contacts(args: argparse.Namespace) -> None:
    """Preserve authoritative candidate-pair labels as validated sparse maps."""
    import pandas as pd
    import pyarrow.parquet as pq

    started = time.time()
    frozen = frozen_root(args.benchmark_root, verify_hash=not args.skip_manifest_hash)
    instances = pd.read_parquet(
        frozen / "instances.parquet", columns=["instance_id", "ab_seq", "ag_seq"]
    ).set_index("instance_id", verify_integrity=True)
    output_dir = args.output_dir.expanduser().resolve()
    contact_dir = output_dir / "contacts"
    contact_dir.mkdir(parents=True, exist_ok=True)

    pair_file = pq.ParquetFile(frozen / "pairs.parquet")
    if pair_file.num_row_groups != len(instances):
        raise ValueError("pairs.parquet must contain exactly one row group per instance")
    records = []
    observed_ids: set[str] = set()
    for row_group in range(pair_file.num_row_groups):
        frame = pair_file.read_row_group(
            row_group,
            columns=["instance_id", "ab_res_idx", "ag_res_idx", "label"],
        ).to_pandas()
        unique_ids = frame["instance_id"].astype(str).unique()
        if len(unique_ids) != 1:
            raise ValueError(f"Pair row group {row_group} spans multiple instances")
        instance_id = str(unique_ids[0])
        if instance_id in observed_ids or instance_id not in instances.index:
            raise ValueError(f"Duplicate or unknown contact-map instance: {instance_id}")
        observed_ids.add(instance_id)
        contact_map = SparseContactMap.from_rows(
            instance_id=instance_id,
            antibody_length=len(str(instances.at[instance_id, "ab_seq"])),
            antigen_length=len(str(instances.at[instance_id, "ag_seq"])),
            antibody_indices=frame["ab_res_idx"],
            antigen_indices=frame["ag_res_idx"],
            labels=frame["label"],
        )
        path = contact_dir / f"{instance_id}.npz"
        if path.exists() and args.resume:
            existing = load_contact_map(path, expected_instance_id=instance_id)
            if existing.candidate_count != contact_map.candidate_count:
                raise ValueError(f"Existing candidate count disagrees for {instance_id}")
            if existing.positive_count != contact_map.positive_count:
                raise ValueError(f"Existing positive count disagrees for {instance_id}")
        else:
            save_contact_map(contact_map, path)
        records.append(
            {
                "instance_id": instance_id,
                "antibody_candidate_residues": len(contact_map.antibody_indices),
                "antigen_candidate_residues": len(contact_map.antigen_indices),
                "candidate_pair_count": contact_map.candidate_count,
                "positive_pair_count": contact_map.positive_count,
            }
        )
        if (row_group + 1) % 100 == 0 or row_group + 1 == pair_file.num_row_groups:
            print(f"Prepared contacts {row_group + 1}/{pair_file.num_row_groups}", flush=True)

    if observed_ids != set(instances.index.astype(str)):
        raise ValueError("Contact row groups do not match the instance manifest")
    summary = {
        "format_version": FORMAT_VERSION,
        "stage": "prepare_contacts",
        "benchmark_manifest_sha256": sha256_file(frozen / "MANIFEST.json"),
        "instance_count": len(records),
        "candidate_pair_count": sum(record["candidate_pair_count"] for record in records),
        "positive_pair_count": sum(record["positive_pair_count"] for record in records),
        "contact_definition": "authoritative pairs.parquet label",
        "candidate_policy": "loss is evaluated only where pairs.parquet contains a candidate row",
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output_dir / "prepare_contacts_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def write_unique_fasta(path: Path, sequences: Iterable[str]) -> dict[str, str]:
    """Write one digest-keyed FASTA record per unique normalized sequence."""
    normalized = sorted({str(sequence).strip().upper() for sequence in sequences})
    if not normalized or any(not sequence for sequence in normalized):
        raise ValueError("FASTA export requires non-empty sequences")
    mapping = {sequence_sha256(sequence): sequence for sequence in normalized}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="ascii") as handle:
        for identifier, sequence in sorted(mapping.items()):
            handle.write(f">{identifier}\n{sequence}\n")
    os.replace(temporary, path)
    return mapping


def run_all_vs_all_mmseqs(
    mmseqs: Path,
    fasta: Path,
    result_tsv: Path,
    temporary_dir: Path,
    identity_threshold: float,
    coverage_threshold: float,
    threads: int,
    max_hits: int,
    sensitivity: float,
) -> list[tuple[str, str, float, float, float]]:
    """Run one all-vs-all search and return threshold-qualified sequence edges."""
    command = [
        str(mmseqs),
        "easy-search",
        str(fasta),
        str(fasta),
        str(result_tsv),
        str(temporary_dir),
        "--format-output",
        "query,target,fident,qcov,tcov",
        "--min-seq-id",
        str(identity_threshold),
        "-c",
        str(coverage_threshold),
        "--cov-mode",
        "0",
        "--max-seqs",
        str(max_hits),
        "-s",
        str(sensitivity),
        "--threads",
        str(threads),
    ]
    subprocess.run(command, check=True)
    edges = []
    if not result_tsv.is_file() or result_tsv.stat().st_size == 0:
        return edges
    with result_tsv.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"Malformed MMseqs row {line_number} in {result_tsv}")
            query, target = fields[:2]
            identity, query_coverage, target_coverage = map(_normalize_fraction, fields[2:])
            if query == target:
                continue
            if (
                identity >= identity_threshold
                and query_coverage >= coverage_threshold
                and target_coverage >= coverage_threshold
            ):
                edges.append((query, target, identity, query_coverage, target_coverage))
    return edges


def expand_sequence_edges(
    sequence_by_instance: Mapping[str, str],
    raw_edges: Iterable[tuple[str, str, float, float, float]],
    source: str,
) -> tuple[list[SimilarityEdge], list[list[str]]]:
    """Map digest-level MMseqs results back to minimal instance-level constraints."""
    instances_by_sequence: dict[str, list[str]] = defaultdict(list)
    for instance_id, sequence in sequence_by_instance.items():
        instances_by_sequence[sequence_sha256(sequence)].append(str(instance_id))
    equivalence_groups = [sorted(group) for group in instances_by_sequence.values() if len(group) > 1]
    edges = []
    seen: set[tuple[str, str]] = set()
    for query, target, identity, query_coverage, target_coverage in raw_edges:
        if query not in instances_by_sequence or target not in instances_by_sequence:
            raise ValueError("MMseqs results reference an unknown sequence digest")
        left = min(instances_by_sequence[query])
        right = min(instances_by_sequence[target])
        key = tuple(sorted((left, right)))
        if left == right or key in seen:
            continue
        seen.add(key)
        edges.append(
            SimilarityEdge(
                left=left,
                right=right,
                source=source,
                identity=identity,
                query_coverage=query_coverage,
                target_coverage=target_coverage,
            )
        )
    return edges, equivalence_groups


def _cdrh3_identity(left: str, right: str) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def cdrh3_similarity_edges(
    metadata_path: Path,
    instance_ids: set[str],
    threshold: float,
) -> list[SimilarityEdge]:
    """Build antibody-aware edges within V-gene and CDRH3-length buckets."""
    import pandas as pd

    frame = pd.read_csv(metadata_path)
    required = {"instance_id", "cdrh3"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CDRH3 metadata is missing columns: {sorted(missing)}")
    frame["instance_id"] = frame["instance_id"].astype(str)
    if frame["instance_id"].duplicated().any() or set(frame["instance_id"]) != instance_ids:
        raise ValueError("CDRH3 metadata must contain exactly one row per instance")
    gene_columns = [column for column in ("vh_gene", "vl_gene") if column in frame.columns]
    for column in ("cdrh3", *gene_columns):
        if frame[column].isna().any():
            raise ValueError(f"CDRH3 metadata contains null {column} values")
        frame[column] = frame[column].astype(str).str.strip().str.upper()
    frame["cdrh3_length"] = frame["cdrh3"].str.len()
    edges = []
    bucket_columns = [*gene_columns, "cdrh3_length"]
    for _, bucket in frame.groupby(bucket_columns, sort=True):
        records = list(bucket[["instance_id", "cdrh3"]].itertuples(index=False, name=None))
        for left_index, (left_id, left_sequence) in enumerate(records):
            for right_id, right_sequence in records[left_index + 1 :]:
                identity = _cdrh3_identity(left_sequence, right_sequence)
                if identity >= threshold:
                    edges.append(
                        SimilarityEdge(
                            left=left_id,
                            right=right_id,
                            source=(
                                "cdrh3_vh_vl" if len(gene_columns) == 2 else "cdrh3_length"
                            ),
                            identity=identity,
                            query_coverage=1.0,
                            target_coverage=1.0,
                        )
                    )
    return edges


def command_prepare_antibody_metadata(args: argparse.Namespace) -> None:
    """Extract CDRH3 sequences from the delivered canonical IMGT mappings."""
    import pandas as pd

    frozen = frozen_root(args.benchmark_root, verify_hash=not args.skip_manifest_hash)
    instances = pd.read_parquet(
        frozen / "instances.parquet", columns=["instance_id", "ab_seq"]
    ).set_index("instance_id", verify_integrity=True)
    mapping_dir = args.mapping_dir.expanduser().resolve()
    records = []
    for instance_id, row in instances.sort_index().iterrows():
        path = mapping_dir / f"{instance_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Canonical antibody mapping is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        positions = list(payload.get("vh_imgt_positions", []))
        vl_positions = list(payload.get("vl_imgt_positions", []))
        antibody_sequence = str(row["ab_seq"]).strip().upper()
        if len(positions) + len(vl_positions) != len(antibody_sequence):
            raise ValueError(f"VH/VL mapping lengths disagree for {instance_id}")
        cdrh3_indices = []
        for index, position in enumerate(positions):
            match = re.match(r"^(\d+)", str(position))
            if match is None:
                raise ValueError(f"Invalid IMGT position {position!r} for {instance_id}")
            if 105 <= int(match.group(1)) <= 117:
                cdrh3_indices.append(index)
        if not cdrh3_indices:
            raise ValueError(f"IMGT mapping contains no CDRH3 residues for {instance_id}")
        cdrh3 = "".join(antibody_sequence[index] for index in cdrh3_indices)
        records.append(
            {
                "instance_id": str(instance_id),
                "cdrh3": cdrh3,
                "cdrh3_length": len(cdrh3),
                "vh_length": len(positions),
                "vl_length": len(vl_positions),
            }
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    os.replace(temporary, output)
    summary = {
        "format_version": FORMAT_VERSION,
        "stage": "prepare_antibody_metadata",
        "instance_count": len(records),
        "cdrh3_identity_policy": (
            "length-matched CDRH3 identity without germline grouping; this is stricter "
            "than same-gene-only clustering"
        ),
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def groups_from_manifest(
    path: Path,
    instance_ids: set[str],
    id_column: str,
    group_columns: Sequence[str],
) -> list[list[str]]:
    """Load one or more external equivalence relationships from a CSV file."""
    import pandas as pd

    frame = pd.read_csv(path)
    required = {id_column, *group_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Group manifest is missing columns: {sorted(missing)}")
    frame[id_column] = frame[id_column].astype(str)
    if frame[id_column].duplicated().any() or set(frame[id_column]) != instance_ids:
        raise ValueError("Group manifest must contain exactly one row per instance")
    groups = []
    for column in group_columns:
        if frame[column].isna().any():
            raise ValueError(f"Group manifest contains null {column} values")
        for _, group in frame.groupby(frame[column].astype(str), sort=True):
            members = sorted(group[id_column].astype(str))
            if len(members) > 1:
                groups.append(members)
    return groups


def command_build_split(args: argparse.Namespace) -> None:
    """Create a double-remote component split from all configured relationships."""
    import pandas as pd

    started = time.time()
    frozen = frozen_root(args.benchmark_root, verify_hash=not args.skip_manifest_hash)
    instances = pd.read_parquet(
        frozen / "instances.parquet", columns=["instance_id", "ab_seq", "ag_seq"]
    ).set_index("instance_id", verify_integrity=True)
    instance_ids = set(instances.index.astype(str))
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    mmseqs = args.mmseqs.expanduser().resolve()
    if not mmseqs.is_file():
        raise FileNotFoundError(f"MMseqs2 binary is missing: {mmseqs}")
    version = subprocess.run(
        [str(mmseqs), "version"], check=True, capture_output=True, text=True
    ).stdout.strip()

    similarity_edges: list[SimilarityEdge] = []
    equivalence_groups: list[list[str]] = []
    role_config = (
        ("antibody_full", "ab_seq", args.antibody_identity, args.coverage),
        ("antigen_full", "ag_seq", args.antigen_identity, args.coverage),
    )
    mmseqs_edge_counts = {}
    for role, column, identity_threshold, coverage_threshold in role_config:
        fasta = work_dir / f"{role}.fasta"
        write_unique_fasta(fasta, instances[column])
        raw_edges = run_all_vs_all_mmseqs(
            mmseqs=mmseqs,
            fasta=fasta,
            result_tsv=work_dir / f"{role}_all_vs_all.tsv",
            temporary_dir=work_dir / f"{role}_tmp",
            identity_threshold=identity_threshold,
            coverage_threshold=coverage_threshold,
            threads=args.threads,
            max_hits=args.max_hits,
            sensitivity=args.sensitivity,
        )
        expanded, exact_groups = expand_sequence_edges(
            instances[column].astype(str).to_dict(), raw_edges, role
        )
        similarity_edges.extend(expanded)
        equivalence_groups.extend(exact_groups)
        mmseqs_edge_counts[role] = len(expanded)

    if args.cdrh3_metadata is not None:
        similarity_edges.extend(
            cdrh3_similarity_edges(
                args.cdrh3_metadata.expanduser().resolve(),
                instance_ids,
                args.cdrh3_identity,
            )
        )
    if args.group_manifest is not None:
        equivalence_groups.extend(
            groups_from_manifest(
                args.group_manifest.expanduser().resolve(),
                instance_ids,
                args.group_id_column,
                args.group_columns,
            )
        )

    components = build_components(
        instance_ids,
        similarity_edges=similarity_edges,
        equivalence_groups=equivalence_groups,
    )
    assignments = deterministic_component_split(
        components,
        fractions={
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "test": args.test_fraction,
        },
        seed=args.seed,
    )
    audit = audit_component_split(components, assignments)
    if not audit["valid"]:
        raise AssertionError("Generated split failed component-isolation validation")
    split_lists = {
        name: sorted(identifier for identifier, split in assignments.items() if split == name)
        for name in ("train", "validation", "test")
    }
    component_id_by_instance = {}
    component_records = []
    for component_index, component in enumerate(components):
        component_id = f"cc_{component_index:05d}"
        split_name = assignments[component[0]]
        component_records.append(
            {"component_id": component_id, "split": split_name, "members": list(component)}
        )
        for identifier in component:
            component_id_by_instance[identifier] = component_id
    source_counts = Counter(edge.source for edge in similarity_edges)
    output = {
        "format_version": FORMAT_VERSION,
        "stage": "build_double_remote_split",
        "intended_use": "development_only",
        "seed": args.seed,
        "thresholds": {
            "antibody_full_identity": args.antibody_identity,
            "antigen_full_identity": args.antigen_identity,
            "bidirectional_coverage": args.coverage,
            "cdrh3_identity_with_same_vh_vl_and_length": (
                args.cdrh3_identity if args.cdrh3_metadata is not None else None
            ),
        },
        "splits": split_lists,
        "component_id_by_instance": component_id_by_instance,
        "components": component_records,
        "audit": audit,
        "edge_source_counts": dict(sorted(source_counts.items())),
        "mmseqs_edge_counts": mmseqs_edge_counts,
        "mmseqs_version": version,
        "group_manifest": str(args.group_manifest) if args.group_manifest else None,
        "group_columns": args.group_columns if args.group_manifest else [],
        "limitations": [
            "A split rebuilt from previously inspected benchmark instances is suitable for development but is not a new prospective test set.",
            "A prospective claim requires structures released after a locked temporal cutoff.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(args.output.expanduser().resolve(), output)
    print(json.dumps({key: output[key] for key in ("stage", "thresholds", "audit")}, indent=2))


def load_split_assignments(path: Path) -> dict[str, str]:
    """Load a strict component split and verify unique membership."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Split manifest has an incompatible format version")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Split manifest does not contain split lists")
    assignments = {}
    for split_name in ("train", "validation", "test"):
        values = [str(value) for value in splits.get(split_name, [])]
        if not values:
            raise ValueError(f"Split {split_name!r} is empty")
        for identifier in values:
            if identifier in assignments:
                raise ValueError(f"Instance {identifier} occurs in multiple splits")
            assignments[identifier] = split_name
    return assignments


def _load_group_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    import pandas as pd

    frame = pd.read_csv(path)
    if "instance_id" not in frame.columns or frame["instance_id"].duplicated().any():
        raise ValueError("Partner metadata requires unique instance_id values")
    result = {}
    for row in frame.fillna("").to_dict(orient="records"):
        identifier = str(row.pop("instance_id"))
        result[identifier] = {str(key): str(value) for key, value in row.items()}
    return result


def _representation_summary(embedding_store: Any) -> dict[str, Any] | None:
    """Return the concatenation summary, or None when nothing was concatenated.

    Called where the receipt is written rather than where the store is built:
    the zero-fill counters are zero until the data has actually been read, so a
    snapshot taken at construction reports a count that was never measured.
    """
    summary = getattr(embedding_store, "summary", None)
    if summary is None or not hasattr(embedding_store, "describe"):
        return None
    return summary()


def _add_representation_arguments(parser: argparse.ArgumentParser) -> None:
    """Extra frozen encoders to concatenate onto the ESM-2 input.

    Train and evaluate must be given the same set: the checkpoint records the
    per-side widths it was built with, and a mismatch fails the state-dict load
    rather than producing a quietly different model.
    """
    parser.add_argument(
        "--representation-manifest",
        type=Path,
        default=None,
        help="canonical manifest; supplies the chain keys AntiBERTy is stored under",
    )
    parser.add_argument("--antiberty-root", type=Path, default=None)
    parser.add_argument("--antiberty-dim", type=int, default=512)
    parser.add_argument("--esm-if1-root", type=Path, default=None)
    parser.add_argument("--esm-if1-dim", type=int, default=512)
    parser.add_argument(
        "--random-projection-control",
        action="store_true",
        help="append fixed random projections of the base embedding instead of real "
        "encoders, matching the encoder arm's widths and parameter count exactly",
    )
    parser.add_argument("--random-projection-seed", type=int, default=20260910)


def _wrap_with_representations(
    embedding_store: Any, args: argparse.Namespace, frame: Any, base_dim: int
) -> tuple[Any, int, int, dict[str, Any] | None]:
    """Return the store to use and the input width of each side.

    ``base_dim`` is the width of the store being wrapped. Train takes it from
    ``--input-dim``; evaluate has no such flag and takes it from the checkpoint,
    which is the only place it is recorded once a run is over.

    With no extra representation named this is the identity, and the widths are
    ``base_dim`` on both sides. With one named, the store is
    wrapped and the two sides diverge -- AntiBERTy covers antibodies only -- so
    the widths are computed rather than assumed, and checked to be constant
    across the cohort. A width that varied by instance would build a model that
    cannot accept some of its own training data.
    """
    roots = (args.antiberty_root, args.esm_if1_root)
    if all(root is None for root in roots) and not args.random_projection_control:
        return embedding_store, base_dim, base_dim, None
    if args.representation_manifest is None:
        raise ValueError(
            "--representation-manifest is required with --antiberty-root/--esm-if1-root: "
            "the chain keys AntiBERTy is stored under are derived from it"
        )

    from interaction_first.composite_embeddings import (
        CompositeEmbeddingStore,
        build_random_control,
        build_sources,
    )

    if args.random_projection_control:
        if any(root is not None for root in roots):
            raise ValueError(
                "--random-projection-control is the width-matched control for the "
                "encoder arm and cannot be combined with it"
            )
        sources = build_random_control(
            args.representation_manifest.expanduser().resolve(),
            dim=args.antiberty_dim,
            seed=args.random_projection_seed,
        )
        store = CompositeEmbeddingStore(embedding_store, sources)
        widths = {"antibody": set(), "antigen": set()}
        for side, column in (
            ("antibody", "ab_sequence_sha256"),
            ("antigen", "ag_sequence_sha256"),
        ):
            for key in frame[column].astype(str).unique():
                widths[side].add(store.width_for(key, base_dim))
        for side, observed in widths.items():
            if len(observed) != 1:
                raise ValueError(
                    f"{side} input width is not constant across the cohort: {sorted(observed)}"
                )
        return store, widths["antibody"].pop(), widths["antigen"].pop(), store.describe()

    sources = build_sources(
        args.representation_manifest.expanduser().resolve(),
        antiberty_root=(
            args.antiberty_root.expanduser().resolve() if args.antiberty_root else None
        ),
        antiberty_dim=args.antiberty_dim,
        esm_if1_root=(
            args.esm_if1_root.expanduser().resolve() if args.esm_if1_root else None
        ),
        esm_if1_dim=args.esm_if1_dim,
    )
    store = CompositeEmbeddingStore(embedding_store, sources)

    widths = {"antibody": set(), "antigen": set()}
    for side, column in (("antibody", "ab_sequence_sha256"), ("antigen", "ag_sequence_sha256")):
        for key in frame[column].astype(str).unique():
            widths[side].add(store.width_for(key, base_dim))
    for side, observed in widths.items():
        if len(observed) != 1:
            raise ValueError(
                f"{side} input width is not constant across the cohort: {sorted(observed)}"
            )
    return store, widths["antibody"].pop(), widths["antigen"].pop(), store.describe()


def _selected_embedding(
    embedding_store: Any,
    sequence_key: str,
    residue_indices: Any,
    device: Any,
) -> Any:
    import torch

    from interaction_first.model import merge_chunk_embeddings

    chunks, starts, full_length = embedding_store.load(sequence_key, device)
    full = merge_chunk_embeddings(chunks, starts, full_length, device)
    indices = torch.as_tensor(residue_indices, dtype=torch.long, device=device)
    return full.index_select(0, indices).unsqueeze(0)


def _example_tensors(
    contact: SparseContactMap,
    row: Any,
    embedding_store: Any,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    import torch

    antigen = _selected_embedding(
        embedding_store, str(row["ag_sequence_sha256"]), contact.antigen_indices, device
    )
    antibody = _selected_embedding(
        embedding_store, str(row["ab_sequence_sha256"]), contact.antibody_indices, device
    )
    pair_labels, pair_mask = contact.dense()
    epitope_labels, paratope_labels = contact.node_labels()
    pair_labels = torch.as_tensor(pair_labels, dtype=torch.float32, device=device).unsqueeze(0)
    pair_mask = torch.as_tensor(pair_mask, dtype=torch.bool, device=device).unsqueeze(0)
    epitope_labels = torch.as_tensor(
        epitope_labels, dtype=torch.float32, device=device
    ).unsqueeze(0)
    paratope_labels = torch.as_tensor(
        paratope_labels, dtype=torch.float32, device=device
    ).unsqueeze(0)
    antigen_mask = torch.ones(antigen.shape[:2], dtype=torch.bool, device=device)
    antibody_mask = torch.ones(antibody.shape[:2], dtype=torch.bool, device=device)
    return (
        antigen,
        antibody,
        pair_labels,
        pair_mask,
        epitope_labels,
        paratope_labels,
        antigen_mask,
        antibody_mask,
    )


def _class_weights(frame: Any, contact_dir: Path) -> dict[str, float]:
    counts = {"edge_positive": 0, "edge_total": 0, "epi_positive": 0, "epi_total": 0, "para_positive": 0, "para_total": 0}
    for instance_id in frame.index:
        contact = load_contact_map(contact_dir / f"{instance_id}.npz", str(instance_id))
        epitope, paratope = contact.node_labels()
        counts["edge_positive"] += contact.positive_count
        counts["edge_total"] += contact.candidate_count
        counts["epi_positive"] += int(epitope.sum())
        counts["epi_total"] += len(epitope)
        counts["para_positive"] += int(paratope.sum())
        counts["para_total"] += len(paratope)

    def ratio(prefix: str) -> float:
        positive = counts[f"{prefix}_positive"]
        total = counts[f"{prefix}_total"]
        if positive <= 0 or positive >= total:
            raise ValueError(f"Cannot derive {prefix} class weight from single-class labels")
        return (total - positive) / positive

    return {
        "edge": ratio("edge"),
        "epitope": ratio("epi"),
        "paratope": ratio("para"),
        **counts,
    }


def _partner_records(frame: Any, metadata: Mapping[str, Mapping[str, str]]) -> list[PartnerRecord]:
    records = []
    for instance_id, row in frame.iterrows():
        groups = metadata.get(str(instance_id), {})
        records.append(
            PartnerRecord(
                instance_id=str(instance_id),
                antibody_key=str(row["ab_sequence_sha256"]),
                antigen_key=str(row["ag_sequence_sha256"]),
                antibody_length=int(row["ab_length"]),
                antigen_length=int(row["ag_length"]),
                antibody_group=groups.get("antibody_group") or None,
                antigen_group=groups.get("antigen_group") or None,
                epitope_group=groups.get("epitope_group") or None,
            )
        )
    return records


def _run_epoch(
    model: Any,
    frame: Any,
    contact_dir: Path,
    embedding_store: Any,
    sampler: PartnerSampler,
    optimizer: Any | None,
    device: Any,
    epoch: int,
    args: argparse.Namespace,
    class_weights: Mapping[str, float],
) -> dict[str, float]:
    import numpy as np
    import torch

    from interaction_first.losses import InteractionLossWeights, interaction_first_loss

    training = optimizer is not None
    model.train(training)
    identifiers = list(frame.index.astype(str))
    if training:
        order = np.random.default_rng(args.seed + epoch).permutation(len(identifiers))
        identifiers = [identifiers[int(index)] for index in order]
    totals: dict[str, float] = defaultdict(float)
    processed = 0
    context = torch.enable_grad() if training else torch.no_grad()
    weights = InteractionLossWeights(
        edge=args.edge_loss_weight,
        epitope_bce=args.epitope_bce_weight,
        paratope_bce=args.paratope_bce_weight,
        epitope_dice=args.epitope_dice_weight,
        paratope_dice=args.paratope_dice_weight,
        edge_count=args.edge_count_weight,
        epitope_count=args.epitope_count_weight,
        paratope_count=args.paratope_count_weight,
        epitope_consistency=args.epitope_consistency_weight,
        paratope_consistency=args.paratope_consistency_weight,
        partner_rank=args.partner_rank_weight,
    )
    with context:
        for instance_id in identifiers:
            row = frame.loc[instance_id]
            contact = load_contact_map(contact_dir / f"{instance_id}.npz", instance_id)
            (
                antigen,
                antibody,
                pair_labels,
                pair_mask,
                epitope_labels,
                paratope_labels,
                antigen_mask,
                antibody_mask,
            ) = _example_tensors(contact, row, embedding_store, device)
            matched = model(
                antigen,
                antibody,
                antigen_mask,
                antibody_mask,
                candidate_pair_mask=pair_mask,
            )
            negative_scores = []
            direction_counts = {"antibody": args.negative_count}
            if args.negative_directions == "both":
                direction_counts = {
                    "antibody": (args.negative_count + 1) // 2,
                    "antigen": args.negative_count // 2,
                }
            for swap, negative_count in direction_counts.items():
                if negative_count == 0:
                    continue
                negatives = sampler.sample(
                    instance_id,
                    count=negative_count,
                    epoch=epoch,
                    swap=swap,
                    hard_pool_size=args.hard_pool_size,
                )
                for negative in negatives:
                    negative_contact = load_contact_map(
                        contact_dir / f"{negative.instance_id}.npz", negative.instance_id
                    )
                    negative_row = frame.loc[negative.instance_id]
                    if swap == "antibody":
                        negative_antibody = _selected_embedding(
                            embedding_store,
                            str(negative_row["ab_sequence_sha256"]),
                            negative_contact.antibody_indices,
                            device,
                        )
                        negative_antibody_mask = torch.ones(
                            negative_antibody.shape[:2], dtype=torch.bool, device=device
                        )
                        negative_output = model(
                            antigen,
                            negative_antibody,
                            antigen_mask,
                            negative_antibody_mask,
                        )
                    else:
                        negative_antigen = _selected_embedding(
                            embedding_store,
                            str(negative_row["ag_sequence_sha256"]),
                            negative_contact.antigen_indices,
                            device,
                        )
                        negative_antigen_mask = torch.ones(
                            negative_antigen.shape[:2], dtype=torch.bool, device=device
                        )
                        negative_output = model(
                            negative_antigen,
                            antibody,
                            negative_antigen_mask,
                            antibody_mask,
                        )
                    negative_scores.append(negative_output["complex_score"])
            stacked_negative_scores = torch.stack(negative_scores, dim=1)
            loss, components = interaction_first_loss(
                matched,
                pair_targets=pair_labels,
                epitope_targets=epitope_labels,
                paratope_targets=paratope_labels,
                antigen_mask=antigen_mask,
                antibody_mask=antibody_mask,
                weights=weights,
                edge_positive_weight=class_weights["edge"],
                epitope_positive_weight=class_weights["epitope"],
                paratope_positive_weight=class_weights["paratope"],
                negative_scores=stacked_negative_scores,
                partner_temperature=args.partner_temperature,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {instance_id}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
            totals["total"] += float(loss.detach().item())
            for name, value in components.items():
                totals[name] += float(value.detach().item())
            processed += 1
            if training and (processed % args.log_every == 0 or processed == len(identifiers)):
                print(
                    f"epoch={epoch} train={processed}/{len(identifiers)} "
                    f"mean_loss={totals['total'] / processed:.8f}",
                    flush=True,
                )
            del antigen, antibody, matched, stacked_negative_scores, loss
    if processed == 0:
        raise ValueError("Epoch received no instances")
    return {name: value / processed for name, value in sorted(totals.items())}


def command_train(args: argparse.Namespace) -> None:
    """Train the sequence interaction model with contact and partner supervision."""
    import pandas as pd
    import torch

    from interaction_first.model import PartnerSpecificInteractionModel, model_parameter_count
    from interaction_first.runtime import (
        EmbeddingStore,
        atomic_torch_save,
        safe_torch_load,
        set_reproducible_seed,
    )

    if args.negative_count <= 0 or args.hard_pool_size < args.negative_count:
        raise ValueError("Negative sampling counts are invalid")
    if args.max_positive_weight <= 0.0:
        raise ValueError("max_positive_weight must be positive")
    if not 0.0 < args.minimum_learning_rate <= args.learning_rate:
        raise ValueError("Learning-rate bounds are invalid")
    if not 0.0 < args.scheduler_factor < 1.0 or args.scheduler_patience < 0:
        raise ValueError("Scheduler settings are invalid")
    if args.max_epochs <= 0 or args.patience <= 0 or args.gradient_clip <= 0.0:
        raise ValueError("Training limits must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    set_reproducible_seed(args.seed, deterministic=not args.allow_nondeterministic)
    prepare_dir = args.prepare_dir.expanduser().resolve()
    contact_dir = args.contact_dir.expanduser().resolve()
    frame = pd.read_parquet(prepare_dir / "instances.parquet").set_index(
        "instance_id", verify_integrity=True
    )
    assignments = load_split_assignments(args.split_manifest.expanduser().resolve())
    if set(frame.index.astype(str)) != set(assignments):
        raise ValueError("Prepared instances and strict split memberships differ")
    frame["interaction_split"] = [assignments[str(identifier)] for identifier in frame.index]
    train_frame = frame.loc[frame["interaction_split"] == "train"].sort_index()
    validation_frame = frame.loc[frame["interaction_split"] == "validation"].sort_index()
    if args.max_train_instances is not None:
        train_frame = train_frame.sample(
            n=min(args.max_train_instances, len(train_frame)),
            random_state=args.seed,
        ).sort_index()
    if args.max_validation_instances is not None:
        validation_frame = validation_frame.sample(
            n=min(args.max_validation_instances, len(validation_frame)),
            random_state=args.seed,
        ).sort_index()
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Strict train and validation splits must both be non-empty")
    if len(train_frame) <= args.negative_count or len(validation_frame) <= args.negative_count:
        raise ValueError(
            "Train and validation subsets must each be larger than negative_count"
        )
    metadata = _load_group_metadata(args.partner_metadata)
    known_positive_pairs = {
        (str(row["ab_sequence_sha256"]), str(row["ag_sequence_sha256"]))
        for _, row in frame.iterrows()
    }
    train_sampler = PartnerSampler(
        _partner_records(train_frame, metadata),
        known_positive_pairs=known_positive_pairs,
        seed=args.seed,
    )
    validation_sampler = PartnerSampler(
        _partner_records(validation_frame, metadata),
        known_positive_pairs=known_positive_pairs,
        seed=args.seed,
    )
    embedding_store = EmbeddingStore(args.embedding_dir.expanduser().resolve(), args.embedding_cache_gib)
    embedding_store, antibody_input_dim, antigen_input_dim, representation_summary = (
        _wrap_with_representations(embedding_store, args, train_frame, args.input_dim)
    )
    raw_class_weights = _class_weights(train_frame, contact_dir)
    class_weights = {
        **raw_class_weights,
        "edge": min(raw_class_weights["edge"], args.max_positive_weight),
        "epitope": min(raw_class_weights["epitope"], args.max_positive_weight),
        "paratope": min(raw_class_weights["paratope"], args.max_positive_weight),
    }

    model = PartnerSpecificInteractionModel(
        antigen_input_dim=antigen_input_dim,
        antibody_input_dim=antibody_input_dim,
        d_model=args.d_model,
        pair_dim=args.pair_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        marginal_pooling=args.marginal_pooling,
        marginal_top_k=args.marginal_top_k,
        marginal_temperature=args.marginal_temperature,
        score_top_k=args.score_top_k,
        score_pooling=args.score_pooling,
        score_temperature=args.score_temperature,
        score_top_fraction=args.score_top_fraction,
        center_partner_score=not args.disable_score_centering,
        cross_attention_gate_initial_logit=args.cross_attention_gate_initial_logit,
        pair_decoder=args.pair_decoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.minimum_learning_rate,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.json"
    split_digest = sha256_file(args.split_manifest.expanduser().resolve())
    config = {
        "format_version": MODEL_FORMAT_VERSION,
        "architecture": "PartnerSpecificInteractionModelV2",
        "input_dim": args.input_dim,
        "antibody_input_dim": antibody_input_dim,
        "antigen_input_dim": antigen_input_dim,
        "representations": representation_summary,
        "d_model": args.d_model,
        "pair_decoder": args.pair_decoder,
        "pair_dim": args.pair_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "marginal_pooling": args.marginal_pooling,
        "marginal_top_k": args.marginal_top_k,
        "marginal_temperature": args.marginal_temperature,
        "score_top_k": args.score_top_k,
        "score_pooling": args.score_pooling,
        "score_temperature": args.score_temperature,
        "score_top_fraction": args.score_top_fraction,
        "center_partner_score": not args.disable_score_centering,
        "cross_attention_gate_initial_logit": args.cross_attention_gate_initial_logit,
        "negative_count": args.negative_count,
        "negative_directions": args.negative_directions,
        "hard_pool_size": args.hard_pool_size,
        "partner_temperature": args.partner_temperature,
        "loss_weights": {
            "edge": args.edge_loss_weight,
            "epitope_bce": args.epitope_bce_weight,
            "paratope_bce": args.paratope_bce_weight,
            "epitope_dice": args.epitope_dice_weight,
            "paratope_dice": args.paratope_dice_weight,
            "edge_count": args.edge_count_weight,
            "epitope_count": args.epitope_count_weight,
            "paratope_count": args.paratope_count_weight,
            "epitope_consistency": args.epitope_consistency_weight,
            "paratope_consistency": args.paratope_consistency_weight,
            "partner_rank": args.partner_rank_weight,
        },
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "weight_decay": args.weight_decay,
        "selection_metric": args.selection_metric,
        "seed": args.seed,
        "split_manifest_sha256": split_digest,
        "inputs": input_provenance(args),
        "class_weights": class_weights,
        "max_positive_weight": args.max_positive_weight,
        "train_instance_count": len(train_frame),
        "validation_instance_count": len(validation_frame),
        "parameter_count": model_parameter_count(model),
    }
    start_epoch = 1
    best_validation = math.inf
    epochs_without_improvement = 0
    history = []
    if args.resume and last_path.is_file():
        checkpoint = safe_torch_load(last_path, map_location=device)
        if checkpoint.get("format_version") != MODEL_FORMAT_VERSION:
            raise ValueError("Checkpoint format is incompatible")
        if checkpoint.get("config", {}).get("split_manifest_sha256") != split_digest:
            raise ValueError("Checkpoint was trained with a different split manifest")
        previous_config = checkpoint["config"]
        resume_keys = (
            "input_dim",
            "d_model",
            "pair_dim",
            "num_heads",
            "num_layers",
            "dropout",
            "marginal_pooling",
            "marginal_top_k",
            "marginal_temperature",
            "score_top_k",
            "score_pooling",
            "score_temperature",
            "score_top_fraction",
            "center_partner_score",
            "cross_attention_gate_initial_logit",
            "pair_decoder",
            "negative_count",
            "negative_directions",
            "hard_pool_size",
            "partner_temperature",
            "loss_weights",
            "class_weights",
            "max_positive_weight",
            "learning_rate",
            "minimum_learning_rate",
            "scheduler_factor",
            "scheduler_patience",
            "weight_decay",
            "selection_metric",
            "seed",
        )
        for key in resume_keys:
            if previous_config.get(key) != config.get(key):
                raise ValueError(f"Checkpoint configuration mismatch for {key}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation_metric"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history = list(checkpoint.get("history", []))

    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_started = time.time()
        train_metrics = _run_epoch(
            model,
            train_frame,
            contact_dir,
            embedding_store,
            train_sampler,
            optimizer,
            device,
            epoch,
            args,
            class_weights,
        )
        validation_metrics = _run_epoch(
            model,
            validation_frame,
            contact_dir,
            embedding_store,
            validation_sampler,
            None,
            device,
            0,
            args,
            class_weights,
        )
        validation_metric = validation_metrics[args.selection_metric]
        scheduler.step(validation_metric)
        improved = validation_metric < best_validation - args.min_delta
        if improved:
            best_validation = validation_metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_metric": args.selection_metric,
            "selection_value": validation_metric,
            "best_validation_metric": best_validation,
            "improved": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - epoch_started,
        }
        history.append(record)
        payload = {
            "format_version": MODEL_FORMAT_VERSION,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_metric": best_validation,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "config": config,
            "embedding_cache": embedding_store.summary(),
        }
        atomic_torch_save(payload, last_path)
        if improved:
            atomic_torch_save(payload, best_path)
        atomic_write_json(history_path, {"config": config, "history": history})
        print(json.dumps(record, sort_keys=True), flush=True)
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break


def command_evaluate(args: argparse.Namespace) -> None:
    """Evaluate pair contacts, marginals, and partner retrieval on a locked split."""
    import numpy as np
    import pandas as pd
    import torch

    from interaction_first.metrics import (
        average_precision,
        binary_metrics,
        cluster_bootstrap_mean_ci,
        contact_map_metrics,
        macro_metric_summary,
        partner_ranking_metrics,
    )
    from interaction_first.model import PartnerSpecificInteractionModel
    from interaction_first.runtime import EmbeddingStore, safe_torch_load, set_reproducible_seed

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    set_reproducible_seed(args.seed, deterministic=not args.allow_nondeterministic)
    split_path = args.split_manifest.expanduser().resolve()
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = load_split_assignments(split_path)
    component_by_instance = {
        str(identifier): str(component)
        for identifier, component in split_payload.get("component_id_by_instance", {}).items()
    }
    prepare_dir = args.prepare_dir.expanduser().resolve()
    contact_dir = args.contact_dir.expanduser().resolve()
    frame = pd.read_parquet(prepare_dir / "instances.parquet").set_index(
        "instance_id", verify_integrity=True
    )
    if set(frame.index.astype(str)) != set(assignments):
        raise ValueError("Prepared instances and strict split memberships differ")
    selected = frame.loc[
        [identifier for identifier, split in assignments.items() if split == args.split]
    ].sort_index()
    if args.max_instances is not None:
        selected = selected.iloc[: args.max_instances]
    if len(selected) <= args.negative_count:
        raise ValueError("Evaluation split is too small for the requested negative count")
    metadata = _load_group_metadata(args.partner_metadata)
    known_positive_pairs = {
        (str(row["ab_sequence_sha256"]), str(row["ag_sequence_sha256"]))
        for _, row in frame.iterrows()
    }
    sampler = PartnerSampler(
        _partner_records(selected, metadata),
        known_positive_pairs=known_positive_pairs,
        seed=args.seed,
    )
    embedding_store = EmbeddingStore(
        args.embedding_dir.expanduser().resolve(), args.embedding_cache_gib
    )
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    if checkpoint.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Checkpoint format is incompatible")
    config = checkpoint["config"]
    expected_split_digest = sha256_file(split_path)
    if config.get("split_manifest_sha256") != expected_split_digest:
        raise ValueError("Checkpoint and evaluation split manifests differ")
    # After the checkpoint, because the base width lives in its config: evaluate
    # has no --input-dim, and a finished run records it nowhere else.
    embedding_store, _, _, _ = _wrap_with_representations(
        embedding_store, args, selected, int(config["input_dim"])
    )
    # Checkpoints written before the sides could differ record only input_dim.
    model = PartnerSpecificInteractionModel(
        antigen_input_dim=int(config.get("antigen_input_dim", config["input_dim"])),
        antibody_input_dim=int(config.get("antibody_input_dim", config["input_dim"])),
        d_model=int(config["d_model"]),
        pair_dim=int(config["pair_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        marginal_pooling=str(config["marginal_pooling"]),
        marginal_top_k=int(config["marginal_top_k"]),
        marginal_temperature=float(config["marginal_temperature"]),
        score_top_k=int(config["score_top_k"]),
        score_pooling=str(config["score_pooling"]),
        score_temperature=float(config["score_temperature"]),
        score_top_fraction=float(config["score_top_fraction"]),
        center_partner_score=bool(config["center_partner_score"]),
        cross_attention_gate_initial_logit=float(
            config["cross_attention_gate_initial_logit"]
        ),
        # Absent from checkpoints written before the decoder became selectable,
        # which were all bilinear.
        pair_decoder=str(config.get("pair_decoder", "bilinear")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    pair_records = []
    node_records = []
    partner_records = []
    per_instance_pair = []
    per_instance_epitope = []
    per_instance_paratope = []
    matched_scores = []
    all_negative_scores = []
    epitope_delta_by_instance = {}
    with torch.no_grad():
        for position, instance_id in enumerate(selected.index.astype(str), 1):
            row = selected.loc[instance_id]
            contact = load_contact_map(contact_dir / f"{instance_id}.npz", instance_id)
            (
                antigen,
                antibody,
                pair_labels,
                pair_mask,
                epitope_labels,
                paratope_labels,
                antigen_mask,
                antibody_mask,
            ) = _example_tensors(contact, row, embedding_store, device)
            matched = model(
                antigen,
                antibody,
                antigen_mask,
                antibody_mask,
                candidate_pair_mask=pair_mask,
            )
            pair_probability = matched["pair_probabilities"].squeeze(0).cpu().numpy()
            epitope_probability = matched["epitope_probability"].squeeze(0).cpu().numpy()
            paratope_probability = matched["paratope_probability"].squeeze(0).cpu().numpy()
            pair_truth = pair_labels.squeeze(0).cpu().numpy()
            pair_valid = pair_mask.squeeze(0).cpu().numpy()
            epitope_truth = epitope_labels.squeeze(0).cpu().numpy()
            paratope_truth = paratope_labels.squeeze(0).cpu().numpy()
            instance_pair_metrics = contact_map_metrics(
                pair_truth[pair_valid], pair_probability[pair_valid], args.edge_threshold
            )
            instance_epitope_metrics = binary_metrics(
                epitope_truth, epitope_probability, args.epitope_threshold
            )
            instance_paratope_metrics = binary_metrics(
                paratope_truth, paratope_probability, args.paratope_threshold
            )
            per_instance_pair.append(instance_pair_metrics)
            per_instance_epitope.append(instance_epitope_metrics)
            per_instance_paratope.append(instance_paratope_metrics)

            for ag_local, ab_local, label in zip(
                contact.pair_antigen_local,
                contact.pair_antibody_local,
                contact.pair_labels,
            ):
                pair_records.append(
                    {
                        "instance_id": instance_id,
                        "ag_res_idx": int(contact.antigen_indices[ag_local]),
                        "ab_res_idx": int(contact.antibody_indices[ab_local]),
                        "label": bool(label),
                        "score": float(pair_probability[ag_local, ab_local]),
                    }
                )
            for role, indices, truth, probability in (
                ("epitope", contact.antigen_indices, epitope_truth, epitope_probability),
                ("paratope", contact.antibody_indices, paratope_truth, paratope_probability),
            ):
                for residue_index, label, score in zip(indices, truth, probability):
                    node_records.append(
                        {
                            "instance_id": instance_id,
                            "role": role,
                            "res_idx": int(residue_index),
                            "label": bool(label),
                            "score": float(score),
                        }
                    )

            negatives = sampler.sample(
                instance_id,
                count=args.negative_count,
                epoch=0,
                swap="antibody",
                hard_pool_size=args.hard_pool_size,
            )
            negative_scores = []
            negative_epitope_probabilities = []
            for negative in negatives:
                negative_contact = load_contact_map(
                    contact_dir / f"{negative.instance_id}.npz", negative.instance_id
                )
                negative_row = selected.loc[negative.instance_id]
                negative_antibody = _selected_embedding(
                    embedding_store,
                    str(negative_row["ab_sequence_sha256"]),
                    negative_contact.antibody_indices,
                    device,
                )
                negative_antibody_mask = torch.ones(
                    negative_antibody.shape[:2], dtype=torch.bool, device=device
                )
                negative_output = model(
                    antigen,
                    negative_antibody,
                    antigen_mask,
                    negative_antibody_mask,
                )
                score = float(negative_output["complex_score"].item())
                negative_scores.append(score)
                negative_epitope_probabilities.append(
                    negative_output["epitope_probability"].squeeze(0).cpu().numpy()
                )
                partner_records.append(
                    {
                        "anchor_instance_id": instance_id,
                        "candidate_instance_id": negative.instance_id,
                        "matched": False,
                        "score": score,
                    }
                )
            matched_score = float(matched["complex_score"].item())
            matched_scores.append(matched_score)
            all_negative_scores.append(negative_scores)
            partner_records.append(
                {
                    "anchor_instance_id": instance_id,
                    "candidate_instance_id": instance_id,
                    "matched": True,
                    "score": matched_score,
                }
            )
            mean_wrong_epitope = np.mean(negative_epitope_probabilities, axis=0)
            matched_ap = average_precision(epitope_truth, epitope_probability)
            wrong_ap = average_precision(epitope_truth, mean_wrong_epitope)
            if matched_ap is not None and wrong_ap is not None:
                epitope_delta_by_instance[instance_id] = float(matched_ap - wrong_ap)
            if position % args.log_every == 0 or position == len(selected):
                print(f"Evaluated {position}/{len(selected)} instances", flush=True)

    pair_frame = pd.DataFrame(pair_records)
    node_frame = pd.DataFrame(node_records)
    partner_frame = pd.DataFrame(partner_records)
    output_dir = args.output_dir.expanduser().resolve()
    atomic_write_dataframe_parquet(pair_frame, output_dir / "pair_predictions.parquet")
    atomic_write_dataframe_parquet(node_frame, output_dir / "node_predictions.parquet")
    atomic_write_dataframe_parquet(partner_frame, output_dir / "partner_scores.parquet")
    pair_micro = contact_map_metrics(
        pair_frame["label"].to_numpy(dtype=np.uint8),
        pair_frame["score"].to_numpy(dtype=np.float64),
        args.edge_threshold,
    )
    node_metrics = {}
    for role, threshold in (
        ("epitope", args.epitope_threshold),
        ("paratope", args.paratope_threshold),
    ):
        role_frame = node_frame.loc[node_frame["role"] == role]
        node_metrics[role] = binary_metrics(
            role_frame["label"].to_numpy(dtype=np.uint8),
            role_frame["score"].to_numpy(dtype=np.float64),
            threshold,
        )
    sensitivity_ci = None
    sensitivity_ci_skipped = None
    if epitope_delta_by_instance and component_by_instance:
        # component_by_instance covers every instance in the split file, while
        # the deltas cover only the evaluated ones, so the mapping has to be
        # restricted before the bootstrap, which requires an exact match. Left
        # unrestricted this raised and failed the whole evaluation after the
        # predictions had already been written.
        evaluated = set(epitope_delta_by_instance)
        missing = sorted(evaluated - set(component_by_instance))
        if missing:
            sensitivity_ci_skipped = (
                f"{len(missing)} evaluated instance(s) have no component assignment"
            )
        else:
            sensitivity_ci = cluster_bootstrap_mean_ci(
                epitope_delta_by_instance,
                {
                    instance_id: component_by_instance[instance_id]
                    for instance_id in evaluated
                },
                iterations=args.bootstrap_iterations,
                seed=args.seed,
            )
    summary = {
        "format_version": MODEL_FORMAT_VERSION,
        "stage": "evaluate",
        "split": args.split,
        "instance_count": len(selected),
        "sensitivity_ci_skipped": sensitivity_ci_skipped,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_manifest_sha256": expected_split_digest,
        "inputs": input_provenance(args),
        "representations": _representation_summary(embedding_store),
        "threshold_policy": "fixed CLI thresholds; no evaluation-split threshold selection",
        "pair": {
            "micro": pair_micro,
            "macro": macro_metric_summary(per_instance_pair),
        },
        "nodes": {
            "epitope": {
                "micro": node_metrics["epitope"],
                "macro": macro_metric_summary(per_instance_epitope),
            },
            "paratope": {
                "micro": node_metrics["paratope"],
                "macro": macro_metric_summary(per_instance_paratope),
            },
        },
        "partner_retrieval": partner_ranking_metrics(matched_scores, all_negative_scores),
        "epitope_matched_minus_mean_wrong_auprc": sensitivity_ci,
        "negative_policy": {
            "count": args.negative_count,
            "hard_pool_size": args.hard_pool_size,
            "unknown pairs are ranked but never assigned all-zero contact labels": True,
        },
        "artifacts": {
            "pair_predictions": str(output_dir / "pair_predictions.parquet"),
            "node_predictions": str(output_dir / "node_predictions.parquet"),
            "partner_scores": str(output_dir / "partner_scores.parquet"),
        },
    }
    atomic_write_json(output_dir / "evaluation.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_smoke(args: argparse.Namespace) -> None:
    """Run a synthetic forward/backward pass through every new objective."""
    import torch

    from interaction_first.losses import interaction_first_loss
    from interaction_first.model import PartnerSpecificInteractionModel, model_parameter_count

    torch.manual_seed(args.seed)
    model = PartnerSpecificInteractionModel(
        antigen_input_dim=32,
        antibody_input_dim=32,
        d_model=64,
        pair_dim=32,
        num_heads=8,
        num_layers=2,
        dropout=0.0,
    )
    antigen = torch.randn(2, 11, 32)
    antibody = torch.randn(2, 7, 32)
    antigen_mask = torch.tensor([[1] * 11, [1] * 9 + [0] * 2], dtype=torch.bool)
    antibody_mask = torch.tensor([[1] * 7, [1] * 6 + [0]], dtype=torch.bool)
    candidate_mask = antigen_mask.unsqueeze(2) & antibody_mask.unsqueeze(1)
    pair_targets = torch.zeros(2, 11, 7)
    pair_targets[0, 2, 3] = 1.0
    pair_targets[0, 4, 1] = 1.0
    pair_targets[1, 1, 2] = 1.0
    epitope_targets = (pair_targets.sum(dim=2) > 0).float()
    paratope_targets = (pair_targets.sum(dim=1) > 0).float()
    matched = model(antigen, antibody, antigen_mask, antibody_mask, candidate_mask)
    negative = model(antigen, antibody.flip(1), antigen_mask, antibody_mask)
    loss, components = interaction_first_loss(
        matched,
        pair_targets,
        epitope_targets,
        paratope_targets,
        antigen_mask,
        antibody_mask,
        edge_positive_weight=20.0,
        epitope_positive_weight=5.0,
        paratope_positive_weight=3.0,
        negative_scores=negative["complex_score"].unsqueeze(1),
    )
    loss.backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.detach().square().sum()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    result = {
        "format_version": MODEL_FORMAT_VERSION,
        "stage": "synthetic_smoke",
        "parameter_count": model_parameter_count(model),
        "loss": float(loss.detach().item()),
        "gradient_norm": float(gradient_norm.item()),
        "component_losses": {
            name: float(value.detach().item()) for name, value in components.items()
        },
        "pair_shape": list(matched["pair_logits"].shape),
        "epitope_shape": list(matched["epitope_probability"].shape),
        "paratope_shape": list(matched["paratope_probability"].shape),
    }
    if not math.isfinite(result["loss"]) or not math.isfinite(result["gradient_norm"]):
        raise FloatingPointError("Synthetic smoke test produced non-finite values")
    if args.output is not None:
        atomic_write_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--skip-manifest-hash", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """Build the staged interaction-first command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-contacts")
    add_benchmark_arguments(prepare)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    prepare.set_defaults(handler=command_prepare_contacts)

    antibody_metadata = subparsers.add_parser("prepare-antibody-metadata")
    add_benchmark_arguments(antibody_metadata)
    antibody_metadata.add_argument("--mapping-dir", type=Path, required=True)
    antibody_metadata.add_argument("--output", type=Path, required=True)
    antibody_metadata.set_defaults(handler=command_prepare_antibody_metadata)

    split = subparsers.add_parser("build-split")
    add_benchmark_arguments(split)
    split.add_argument("--mmseqs", type=Path, required=True)
    split.add_argument("--work-dir", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--cdrh3-metadata", type=Path)
    split.add_argument("--group-manifest", type=Path)
    split.add_argument("--group-id-column", default="INSTANCE")
    split.add_argument("--group-columns", nargs="+", default=["ab_ag_cluster"])
    split.add_argument("--antibody-identity", type=float, default=0.80)
    split.add_argument("--antigen-identity", type=float, default=0.70)
    split.add_argument("--cdrh3-identity", type=float, default=0.70)
    split.add_argument("--coverage", type=float, default=0.80)
    split.add_argument("--train-fraction", type=float, default=0.80)
    split.add_argument("--validation-fraction", type=float, default=0.10)
    split.add_argument("--test-fraction", type=float, default=0.10)
    split.add_argument("--threads", type=int, default=8)
    split.add_argument("--max-hits", type=int, default=10000)
    split.add_argument("--sensitivity", type=float, default=7.5)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(handler=command_build_split)

    train = subparsers.add_parser("train")
    train.add_argument("--prepare-dir", type=Path, required=True)
    train.add_argument("--contact-dir", type=Path, required=True)
    train.add_argument("--embedding-dir", type=Path, required=True)
    _add_representation_arguments(train)
    train.add_argument("--split-manifest", type=Path, required=True)
    train.add_argument("--partner-metadata", type=Path)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--input-dim", type=int, default=1280)
    train.add_argument("--d-model", type=int, default=128)
    train.add_argument("--pair-dim", type=int, default=64)
    train.add_argument("--num-heads", type=int, default=8)
    train.add_argument("--num-layers", type=int, default=1)
    train.add_argument("--dropout", type=float, default=0.20)
    train.add_argument(
        "--marginal-pooling",
        choices=("mean", "topk", "noisy_or", "logmeanexp"),
        default="logmeanexp",
    )
    train.add_argument("--marginal-top-k", type=int, default=3)
    train.add_argument("--marginal-temperature", type=float, default=1.0)
    train.add_argument("--score-top-k", type=int, default=16)
    train.add_argument(
        "--score-pooling",
        choices=("logmeanexp", "top_fraction"),
        default="logmeanexp",
    )
    train.add_argument("--score-temperature", type=float, default=1.0)
    train.add_argument("--score-top-fraction", type=float, default=0.01)
    train.add_argument("--disable-score-centering", action="store_true")
    train.add_argument("--cross-attention-gate-initial-logit", type=float, default=-2.0)
    train.add_argument("--negative-count", type=int, default=4)
    train.add_argument("--negative-directions", choices=("antibody", "both"), default="both")
    train.add_argument("--hard-pool-size", type=int, default=64)
    train.add_argument("--partner-temperature", type=float, default=0.10)
    train.add_argument("--edge-loss-weight", type=float, default=1.0)
    train.add_argument("--epitope-bce-weight", type=float, default=0.25)
    train.add_argument("--paratope-bce-weight", type=float, default=0.25)
    train.add_argument("--epitope-dice-weight", type=float, default=0.25)
    train.add_argument("--paratope-dice-weight", type=float, default=0.25)
    train.add_argument("--edge-count-weight", type=float, default=0.0)
    train.add_argument("--epitope-count-weight", type=float, default=0.0)
    train.add_argument("--paratope-count-weight", type=float, default=0.0)
    train.add_argument("--epitope-consistency-weight", type=float, default=0.0)
    train.add_argument("--paratope-consistency-weight", type=float, default=0.0)
    train.add_argument("--partner-rank-weight", type=float, default=0.5)
    train.add_argument("--max-positive-weight", type=float, default=100.0)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--minimum-learning-rate", type=float, default=1e-7)
    train.add_argument("--scheduler-factor", type=float, default=0.5)
    train.add_argument("--scheduler-patience", type=int, default=3)
    train.add_argument("--weight-decay", type=float, default=5e-4)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--max-epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--min-delta", type=float, default=0.0)
    train.add_argument(
        "--selection-metric",
        choices=("partner_rank", "total"),
        default="partner_rank",
    )
    train.add_argument(
        "--pair-decoder",
        choices=("bilinear", "hadamard_mlp"),
        default="bilinear",
        help="pair decoder head; the default reproduces every earlier run",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--allow-nondeterministic", action="store_true")
    train.add_argument("--embedding-cache-gib", type=float, default=24.0)
    train.add_argument("--max-train-instances", type=int)
    train.add_argument("--max-validation-instances", type=int)
    train.add_argument("--log-every", type=int, default=25)
    train.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    train.set_defaults(handler=command_train)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument("--output", type=Path)
    smoke.set_defaults(handler=command_smoke)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prepare-dir", type=Path, required=True)
    evaluate.add_argument("--contact-dir", type=Path, required=True)
    evaluate.add_argument("--embedding-dir", type=Path, required=True)
    _add_representation_arguments(evaluate)
    evaluate.add_argument("--split-manifest", type=Path, required=True)
    evaluate.add_argument("--partner-metadata", type=Path)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate.add_argument("--negative-count", type=int, default=9)
    evaluate.add_argument("--hard-pool-size", type=int, default=64)
    evaluate.add_argument("--edge-threshold", type=float, default=0.5)
    evaluate.add_argument("--epitope-threshold", type=float, default=0.5)
    evaluate.add_argument("--paratope-threshold", type=float, default=0.5)
    evaluate.add_argument("--bootstrap-iterations", type=int, default=2000)
    evaluate.add_argument("--embedding-cache-gib", type=float, default=24.0)
    evaluate.add_argument("--max-instances", type=int)
    evaluate.add_argument("--log-every", type=int, default=25)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate.add_argument("--allow-nondeterministic", action="store_true")
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main() -> None:
    """Dispatch one interaction-first pipeline stage."""
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
