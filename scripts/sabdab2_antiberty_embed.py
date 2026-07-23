#!/usr/bin/env python3
"""Create provenance-locked AntiBERTy embeddings for SAbDab2 antibodies.

The default execution boundary is deliberately limited to the development
train and validation splits. Frozen test sequences are never parsed unless the
caller both selects ``test`` and passes the explicit test-unlock flag.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_FORMAT_VERSION = "sabdab2_antiberty_embedding_v1"
CONTRACT_FORMAT_VERSION = "sabdab2_antiberty_run_contract_v1"
MANIFEST_FORMAT_VERSION = "sabdab2_antiberty_content_manifest_v1"
PROVENANCE_FORMAT_VERSION = "sabdab2_antiberty_provenance_v1"
MODEL_NAME = "AntiBERTy_md_smooth"
EXPECTED_EMBEDDING_DIM = 512
DEFAULT_SPLITS = ("train", "validation")
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_KEYS = frozenset(
    {
        "format_version",
        "sequence_key",
        "source_sequence_sha256",
        "embedding",
        "mask",
        "vh_length",
        "vl_length",
        "embedding_dim",
        "model_name",
        "model_weights_sha256",
    }
)


@dataclass(frozen=True)
class AntibodyRecord:
    """One canonical antibody sequence associated with one benchmark row."""

    instance_id: str
    split: str
    vh_sequence: str
    vl_sequence: str
    sequence_key: str

    @property
    def vh_length(self) -> int:
        """Return the heavy-chain residue count."""
        return len(self.vh_sequence)

    @property
    def vl_length(self) -> int:
        """Return the light-chain residue count."""
        return len(self.vl_sequence)

    @property
    def total_length(self) -> int:
        """Return the concatenated heavy/light residue count."""
        return self.vh_length + self.vl_length


@dataclass(frozen=True)
class ModelInfo:
    """Immutable identity of the installed AntiBERTy model and runtime."""

    model_name: str
    model_weights_path: str
    model_weights_sha256: str
    model_config_path: str | None
    model_config_sha256: str | None
    tokenizer_vocab_path: str
    tokenizer_vocab_sha256: str
    antiberty_package_path: str
    package_versions: dict[str, str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line interface without importing ML dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--instances-parquet", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Named split memberships to embed; defaults to train validation.",
    )
    parser.add_argument(
        "--allow-frozen-test",
        action="store_true",
        help="Explicitly unlock frozen test sequence processing.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--expected-model-weights-sha256",
        type=str,
        default=None,
        help="Optional pinned digest for the installed AntiBERTy weights.",
    )
    parser.add_argument(
        "--debug-max-antibodies",
        type=int,
        default=None,
        help="Embed a deterministic prefix for smoke testing only.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.debug_max_antibodies is not None and args.debug_max_antibodies <= 0:
        parser.error("--debug-max-antibodies must be positive")
    if args.expected_model_weights_sha256 is not None:
        digest = args.expected_model_weights_sha256.lower()
        if not SHA256_PATTERN.fullmatch(digest):
            parser.error("--expected-model-weights-sha256 must be a SHA-256 digest")
        args.expected_model_weights_sha256 = digest
    return args


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest for a byte string."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a bounded-memory SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize JSON deterministically for identity comparisons."""
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Create bytes atomically without ever replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a canonical JSON object atomically."""
    atomic_write_bytes(path, canonical_json_bytes(payload))


def write_new_or_validate_identical(path: Path, payload: bytes) -> None:
    """Create an immutable metadata file or verify its existing contents."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing immutable metadata differs: {path}")
        return
    atomic_write_bytes(path, payload)


def normalize_sequence(value: str, *, chain: str, allow_empty: bool) -> str:
    """Normalize and validate one canonical amino-acid sequence."""
    sequence = str(value).strip().upper()
    if not sequence and not allow_empty:
        raise ValueError(f"Canonical {chain} sequence is empty")
    invalid = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if invalid:
        raise ValueError(f"Canonical {chain} sequence has invalid residue symbols: {invalid}")
    return sequence


def sequence_sha256(vh_sequence: str, vl_sequence: str) -> str:
    """Match the benchmark antibody digest over heavy then light residues."""
    return sha256_bytes((vh_sequence + vl_sequence).encode("ascii"))


def load_split_selection(
    path: Path,
    requested_splits: Sequence[str],
    allow_frozen_test: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load disjoint memberships and enforce the frozen-test safety boundary."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    split_payload = payload.get("splits")
    if not isinstance(split_payload, dict):
        raise ValueError("Split manifest does not contain a splits object")

    names = [str(value) for value in requested_splits]
    if not names or len(names) != len(set(names)):
        raise ValueError("Requested split names must be non-empty and unique")
    unknown = sorted(set(names) - set(split_payload))
    if unknown:
        raise ValueError(f"Requested splits are not present in the manifest: {unknown}")
    if "test" in names and not allow_frozen_test:
        raise PermissionError(
            "Frozen test processing is locked; remove test or pass --allow-frozen-test"
        )

    membership: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    for split_name, raw_values in split_payload.items():
        if not isinstance(raw_values, list):
            raise ValueError(f"Split {split_name!r} is not a list")
        values = [str(value) for value in raw_values]
        if len(values) != len(set(values)):
            raise ValueError(f"Split {split_name!r} contains duplicate IDs")
        split_counts[str(split_name)] = len(values)
        for instance_id in values:
            if instance_id in membership:
                raise ValueError("Split manifest contains overlapping memberships")
            membership[instance_id] = str(split_name)

    selected = {
        instance_id: split_name
        for instance_id, split_name in membership.items()
        if split_name in names
    }
    if not selected:
        raise ValueError("Requested splits select no instances")
    summary = {
        "intended_use": payload.get("intended_use"),
        "selected_splits": names,
        "selected_instance_count": len(selected),
        "all_split_counts": split_counts,
        "frozen_test_unlocked": bool("test" in names and allow_frozen_test),
    }
    return selected, summary


def _parse_first_unquoted_csv_field(line: str) -> str:
    """Extract the leading identifier without parsing later sequence fields."""
    prefix = line.partition(",")[0].strip()
    if prefix.startswith('"'):
        parsed = next(csv.reader([prefix]))
        if len(parsed) != 1:
            raise ValueError("Unable to parse the leading canonical-manifest field")
        prefix = parsed[0]
    return prefix


def load_selected_antibodies(
    path: Path,
    selected_membership: Mapping[str, str],
) -> list[AntibodyRecord]:
    """Parse sequence columns only after a row passes the selected-ID filter."""
    for instance_id in selected_membership:
        if not instance_id or any(symbol in instance_id for symbol in (",", "\n", "\r")):
            raise ValueError("Selected instance IDs must be non-empty single CSV fields")

    records: dict[str, AntibodyRecord] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError("Canonical manifest is empty")
        header = next(csv.reader([header_line]))
        required = {"INSTANCE", "vh_seq", "vl_seq"}
        missing_columns = sorted(required - set(header))
        if missing_columns:
            raise ValueError(f"Canonical manifest is missing columns: {missing_columns}")
        if header[0] != "INSTANCE":
            raise ValueError("INSTANCE must be the first canonical-manifest column")
        index = {name: header.index(name) for name in required}

        for line_number, raw_line in enumerate(handle, start=2):
            instance_id = _parse_first_unquoted_csv_field(raw_line)
            if instance_id not in selected_membership:
                continue
            fields = next(csv.reader([raw_line]))
            if len(fields) != len(header):
                raise ValueError(
                    f"Malformed selected canonical row at line {line_number}: "
                    f"expected {len(header)} fields, observed {len(fields)}"
                )
            if instance_id in records:
                raise ValueError(f"Duplicate selected canonical row for {instance_id}")
            vh_sequence = normalize_sequence(
                fields[index["vh_seq"]], chain="VH", allow_empty=False
            )
            vl_sequence = normalize_sequence(
                fields[index["vl_seq"]], chain="VL", allow_empty=True
            )
            records[instance_id] = AntibodyRecord(
                instance_id=instance_id,
                split=selected_membership[instance_id],
                vh_sequence=vh_sequence,
                vl_sequence=vl_sequence,
                sequence_key=sequence_sha256(vh_sequence, vl_sequence),
            )

    missing_ids = sorted(set(selected_membership) - set(records))
    if missing_ids:
        raise ValueError(
            f"Canonical manifest is missing {len(missing_ids)} selected IDs; "
            f"first={missing_ids[:5]}"
        )
    return [records[instance_id] for instance_id in sorted(records)]


def validate_instance_metadata_rows(
    records: Sequence[AntibodyRecord],
    metadata_by_id: Mapping[str, tuple[str, int]],
) -> None:
    """Cross-check canonical sequences against prepared digest and length metadata."""
    expected_ids = {record.instance_id for record in records}
    if set(metadata_by_id) != expected_ids:
        missing = sorted(expected_ids - set(metadata_by_id))
        extra = sorted(set(metadata_by_id) - expected_ids)
        raise ValueError(
            "Prepared instance metadata does not match selected IDs: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for record in records:
        digest, length = metadata_by_id[record.instance_id]
        if digest != record.sequence_key:
            raise ValueError(
                f"Antibody digest mismatch for selected instance {record.instance_id}"
            )
        if int(length) != record.total_length:
            raise ValueError(
                f"Antibody length mismatch for selected instance {record.instance_id}: "
                f"prepared={length}, canonical={record.total_length}"
            )


def load_and_validate_instance_metadata(
    path: Path,
    records: Sequence[AntibodyRecord],
) -> None:
    """Read only selected immutable metadata columns from prepared Parquet."""
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("pandas and pyarrow are required to read instances.parquet") from error

    selected_ids = sorted(record.instance_id for record in records)
    columns = ["instance_id", "ab_sequence_sha256", "ab_length"]
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[("instance_id", "in", selected_ids)],
    )
    if frame["instance_id"].isna().any() or frame["instance_id"].duplicated().any():
        raise ValueError("Prepared metadata has null or duplicate selected instance IDs")
    metadata = {
        str(row.instance_id): (str(row.ab_sequence_sha256), int(row.ab_length))
        for row in frame.itertuples(index=False)
    }
    validate_instance_metadata_rows(records, metadata)


def deduplicate_antibodies(
    records: Sequence[AntibodyRecord],
) -> tuple[list[AntibodyRecord], dict[str, dict[str, int]]]:
    """Deduplicate exact antibodies while preserving chain-boundary consistency."""
    representative: dict[str, AntibodyRecord] = {}
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        previous = representative.get(record.sequence_key)
        if previous is not None and (
            previous.vh_sequence != record.vh_sequence
            or previous.vl_sequence != record.vl_sequence
        ):
            raise ValueError(
                "One concatenated antibody digest maps to inconsistent VH/VL boundaries"
            )
        representative.setdefault(record.sequence_key, record)
        split_counts = counts.setdefault(record.sequence_key, {})
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
    ordered = [representative[key] for key in sorted(representative)]
    return ordered, counts


def package_version(distribution: str) -> str:
    """Return an installed distribution version without assuming package metadata."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def resolve_model_info() -> tuple[ModelInfo, Any, Any]:
    """Resolve the installed AntiBERTy package and return lazy runtime classes."""
    try:
        import antiberty
        import torch
        from antiberty import AntiBERTyRunner
    except ImportError as error:
        raise RuntimeError("AntiBERTy and PyTorch must be installed for embedding") from error

    package_root = Path(antiberty.__file__).resolve().parent
    model_root = package_root / "trained_models" / MODEL_NAME
    weights_path = model_root / "pytorch_model.bin"
    config_path = model_root / "config.json"
    vocab_path = package_root / "trained_models" / "vocab.txt"
    for required_path in (weights_path, vocab_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing installed AntiBERTy asset: {required_path}")

    info = ModelInfo(
        model_name=MODEL_NAME,
        model_weights_path=str(weights_path),
        model_weights_sha256=sha256_file(weights_path),
        model_config_path=str(config_path) if config_path.is_file() else None,
        model_config_sha256=sha256_file(config_path) if config_path.is_file() else None,
        tokenizer_vocab_path=str(vocab_path),
        tokenizer_vocab_sha256=sha256_file(vocab_path),
        antiberty_package_path=str(package_root),
        package_versions={
            "antiberty": package_version("antiberty"),
            "torch": str(torch.__version__),
            "transformers": package_version("transformers"),
        },
    )
    return info, AntiBERTyRunner, torch


def requested_runtime_device(runner: Any, requested: str, torch: Any) -> str:
    """Move the installed runner when needed and validate the requested device."""
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if requested == "cpu" and runner.device.type != "cpu":
        runner.model = runner.model.to(torch.device("cpu"))
        runner.device = torch.device("cpu")
    if requested == "cuda" and runner.device.type != "cuda":
        runner.model = runner.model.to(torch.device("cuda"))
        runner.device = torch.device("cuda")
    return str(runner.device)


def artifact_path(output_root: Path, sequence_key: str) -> Path:
    """Return the deterministic two-hex sharded artifact path."""
    if not SHA256_PATTERN.fullmatch(sequence_key):
        raise ValueError(f"Invalid sequence digest: {sequence_key!r}")
    return output_root / sequence_key[:2] / f"{sequence_key}.pt"


def validate_embedding_artifact(
    payload: Any,
    record: AntibodyRecord,
    model_info: ModelInfo,
    torch: Any,
) -> None:
    """Enforce the complete tensor and provenance contract for one artifact."""
    if not isinstance(payload, dict):
        raise TypeError("AntiBERTy artifact must be a dictionary")
    observed_keys = frozenset(payload)
    if observed_keys != ARTIFACT_KEYS:
        missing = sorted(ARTIFACT_KEYS - observed_keys)
        extra = sorted(observed_keys - ARTIFACT_KEYS)
        raise ValueError(f"Artifact keys differ: missing={missing}, extra={extra}")
    expected_scalars = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "sequence_key": record.sequence_key,
        "source_sequence_sha256": record.sequence_key,
        "vh_length": record.vh_length,
        "vl_length": record.vl_length,
        "embedding_dim": EXPECTED_EMBEDDING_DIM,
        "model_name": model_info.model_name,
        "model_weights_sha256": model_info.model_weights_sha256,
    }
    for key, expected in expected_scalars.items():
        if payload[key] != expected:
            raise ValueError(
                f"Artifact field {key!r} differs: expected={expected!r}, "
                f"observed={payload[key]!r}"
            )

    embedding = payload["embedding"]
    mask = payload["mask"]
    if not isinstance(embedding, torch.Tensor) or embedding.dtype != torch.float32:
        raise TypeError("Artifact embedding must be a float32 torch.Tensor")
    if embedding.ndim != 2 or tuple(embedding.shape) != (
        record.total_length,
        EXPECTED_EMBEDDING_DIM,
    ):
        raise ValueError(
            f"Artifact embedding shape mismatch: {tuple(embedding.shape)}"
        )
    if embedding.device.type != "cpu" or not embedding.is_contiguous():
        raise ValueError("Artifact embedding must be contiguous CPU storage")
    if not bool(torch.isfinite(embedding).all().item()):
        raise ValueError("Artifact embedding contains non-finite values")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("Artifact mask must be a boolean torch.Tensor")
    if mask.device.type != "cpu" or mask.ndim != 1 or len(mask) != record.total_length:
        raise ValueError("Artifact mask shape or device is invalid")
    if not mask.is_contiguous() or not bool(mask.all().item()):
        raise ValueError("Artifact mask must be contiguous and entirely true")


def load_and_validate_artifact(
    path: Path,
    record: AntibodyRecord,
    model_info: ModelInfo,
    torch: Any,
) -> None:
    """Load one existing artifact safely and apply the strict contract."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    validate_embedding_artifact(payload, record, model_info, torch)


def atomic_torch_save(payload: Mapping[str, Any], path: Path, torch: Any) -> None:
    """Save a PyTorch artifact atomically without replacing existing data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def strip_special_tokens(
    embedding: Any,
    expected_length: int,
    chain: str,
    torch: Any,
) -> Any:
    """Remove CLS/SEP tokens and validate residue-aligned AntiBERTy output."""
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise TypeError(f"AntiBERTy returned an invalid {chain} embedding tensor")
    if embedding.shape[0] != expected_length + 2:
        raise ValueError(
            f"AntiBERTy {chain} token count mismatch: "
            f"expected={expected_length + 2}, observed={embedding.shape[0]}"
        )
    residue_embedding = embedding[1:-1].detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    if residue_embedding.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"AntiBERTy embedding dimension mismatch: "
            f"expected={EXPECTED_EMBEDDING_DIM}, observed={residue_embedding.shape[1]}"
        )
    if not bool(torch.isfinite(residue_embedding).all().item()):
        raise ValueError(f"AntiBERTy returned non-finite {chain} embeddings")
    return residue_embedding


def embed_batch(
    runner: Any,
    records: Sequence[AntibodyRecord],
    model_info: ModelInfo,
    torch: Any,
) -> list[dict[str, Any]]:
    """Embed heavy and light chains separately, then concatenate residues."""
    heavy_raw = runner.embed([record.vh_sequence for record in records])
    if len(heavy_raw) != len(records):
        raise ValueError("AntiBERTy returned the wrong number of heavy-chain tensors")

    light_indices = [index for index, record in enumerate(records) if record.vl_sequence]
    light_raw = (
        runner.embed([records[index].vl_sequence for index in light_indices])
        if light_indices
        else []
    )
    if len(light_raw) != len(light_indices):
        raise ValueError("AntiBERTy returned the wrong number of light-chain tensors")
    light_by_index = dict(zip(light_indices, light_raw))

    artifacts: list[dict[str, Any]] = []
    for index, (record, raw_heavy) in enumerate(zip(records, heavy_raw)):
        heavy = strip_special_tokens(raw_heavy, record.vh_length, "VH", torch)
        if record.vl_length:
            light = strip_special_tokens(
                light_by_index[index], record.vl_length, "VL", torch
            )
        else:
            light = torch.empty((0, EXPECTED_EMBEDDING_DIM), dtype=torch.float32)
        embedding = torch.cat((heavy, light), dim=0).contiguous()
        artifact = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "sequence_key": record.sequence_key,
            "source_sequence_sha256": record.sequence_key,
            "embedding": embedding,
            "mask": torch.ones(record.total_length, dtype=torch.bool),
            "vh_length": record.vh_length,
            "vl_length": record.vl_length,
            "embedding_dim": EXPECTED_EMBEDDING_DIM,
            "model_name": model_info.model_name,
            "model_weights_sha256": model_info.model_weights_sha256,
        }
        validate_embedding_artifact(artifact, record, model_info, torch)
        artifacts.append(artifact)
    return artifacts


def stable_membership_sha256(values: Iterable[str]) -> str:
    """Hash a sorted identifier collection without exposing it in provenance."""
    encoded = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return sha256_bytes(encoded)


def build_run_contract(
    args: argparse.Namespace,
    records: Sequence[AntibodyRecord],
    unique_records: Sequence[AntibodyRecord],
    split_summary: Mapping[str, Any],
    model_info: ModelInfo,
) -> dict[str, Any]:
    """Create immutable input, selection, model, and code provenance."""
    source_paths = {
        "canonical_manifest": args.canonical_manifest,
        "instances_parquet": args.instances_parquet,
        "split_manifest": args.split_manifest,
        "embedding_script": Path(__file__).resolve(),
    }
    source_files = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path.resolve()),
            "size_bytes": path.resolve().stat().st_size,
        }
        for name, path in source_paths.items()
    }
    debug_limit = args.debug_max_antibodies
    return {
        "format_version": CONTRACT_FORMAT_VERSION,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "selection": {
            **dict(split_summary),
            "selected_instance_membership_sha256": stable_membership_sha256(
                record.instance_id for record in records
            ),
            "unique_antibody_count_before_debug_limit": len(unique_records),
            "emitted_antibody_count": min(len(unique_records), debug_limit)
            if debug_limit is not None
            else len(unique_records),
            "emitted_antibody_membership_sha256": stable_membership_sha256(
                record.sequence_key
                for record in (
                    unique_records[:debug_limit] if debug_limit is not None else unique_records
                )
            ),
            "debug_max_antibodies": debug_limit,
            "intended_use": "smoke_only" if debug_limit is not None else split_summary.get("intended_use"),
        },
        "model": asdict(model_info),
        "source_files": source_files,
        "embedding": {
            "chain_processing": "embed_vh_and_vl_separately_then_concatenate",
            "hidden_layer": -1,
            "special_tokens_removed": True,
            "dtype": "float32",
            "embedding_dim": EXPECTED_EMBEDDING_DIM,
            "sharding": "sha256_prefix_2",
        },
        "runtime_request": {
            "batch_size": args.batch_size,
            "device": args.device,
        },
    }


def initialize_output_root(
    output_root: Path,
    contract: Mapping[str, Any],
    resume: bool,
) -> str:
    """Create or validate the immutable run contract before artifact access."""
    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = output_root / "run_contract.json"
    if contract_path.exists():
        if not resume:
            raise FileExistsError(
                f"Output contract already exists; pass --resume to validate it: {contract_path}"
            )
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != contract:
            raise ValueError("Existing run contract differs from the requested run")
    else:
        existing = [path for path in output_root.iterdir()]
        if existing:
            raise FileExistsError(
                "Output root contains data without an immutable run contract"
            )
        atomic_write_json(contract_path, contract)
    return sha256_bytes(canonical_json_bytes(contract))


def build_content_manifest(
    output_root: Path,
    records: Sequence[AntibodyRecord],
    split_counts: Mapping[str, Mapping[str, int]],
) -> bytes:
    """Build a deterministic JSONL inventory with artifact content hashes."""
    lines: list[str] = []
    for record in records:
        path = artifact_path(output_root, record.sequence_key)
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed AntiBERTy artifact: {path}")
        entry = {
            "format_version": MANIFEST_FORMAT_VERSION,
            "sequence_key": record.sequence_key,
            "source_sequence_sha256": record.sequence_key,
            "relative_path": str(path.relative_to(output_root)),
            "artifact_sha256": sha256_file(path),
            "artifact_size_bytes": path.stat().st_size,
            "vh_length": record.vh_length,
            "vl_length": record.vl_length,
            "total_length": record.total_length,
            "embedding_dim": EXPECTED_EMBEDDING_DIM,
            "selected_instance_count": sum(split_counts[record.sequence_key].values()),
            "selected_split_counts": dict(sorted(split_counts[record.sequence_key].items())),
        }
        lines.append(json.dumps(entry, sort_keys=True, allow_nan=False, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode("utf-8")


def validate_completed_provenance(
    path: Path,
    contract_sha256: str,
    content_manifest_sha256: str,
    artifact_count: int,
) -> None:
    """Validate a prior completion marker without rewriting it."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "run_contract_sha256": contract_sha256,
        "content_manifest_sha256": content_manifest_sha256,
        "artifact_count": artifact_count,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Existing provenance field {key!r} differs")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the isolated, resumable AntiBERTy embedding workflow."""
    started = time.time()
    args.canonical_manifest = args.canonical_manifest.expanduser().resolve()
    args.instances_parquet = args.instances_parquet.expanduser().resolve()
    args.split_manifest = args.split_manifest.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    for source in (
        args.canonical_manifest,
        args.instances_parquet,
        args.split_manifest,
    ):
        if not source.is_file():
            raise FileNotFoundError(f"Missing required input: {source}")

    selected_membership, split_summary = load_split_selection(
        args.split_manifest,
        args.splits,
        args.allow_frozen_test,
    )
    records = load_selected_antibodies(args.canonical_manifest, selected_membership)
    load_and_validate_instance_metadata(args.instances_parquet, records)
    unique_records, split_counts = deduplicate_antibodies(records)
    emitted_records = (
        unique_records[: args.debug_max_antibodies]
        if args.debug_max_antibodies is not None
        else unique_records
    )

    model_info, runner_class, torch = resolve_model_info()
    if (
        args.expected_model_weights_sha256 is not None
        and model_info.model_weights_sha256 != args.expected_model_weights_sha256
    ):
        raise ValueError(
            "Installed AntiBERTy weight digest does not match the pinned digest"
        )
    contract = build_run_contract(
        args,
        records,
        unique_records,
        split_summary,
        model_info,
    )
    contract_sha256 = initialize_output_root(args.output_root, contract, args.resume)

    pending: list[AntibodyRecord] = []
    resumed_count = 0
    for record in emitted_records:
        path = artifact_path(args.output_root, record.sequence_key)
        if path.exists():
            if not args.resume:
                raise FileExistsError(f"Refusing to reuse existing artifact: {path}")
            load_and_validate_artifact(path, record, model_info, torch)
            resumed_count += 1
        else:
            pending.append(record)

    generated_count = 0
    runtime_device = "not_initialized"
    if pending:
        runner = runner_class()
        runtime_device = requested_runtime_device(runner, args.device, torch)
        runner.model.eval()
        torch.set_grad_enabled(False)
        print(
            f"selected_instances={len(records)} unique_antibodies={len(unique_records)} "
            f"emitted={len(emitted_records)} pending={len(pending)} device={runtime_device}",
            flush=True,
        )
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            artifacts = embed_batch(runner, batch, model_info, torch)
            for record, artifact in zip(batch, artifacts):
                path = artifact_path(args.output_root, record.sequence_key)
                atomic_torch_save(artifact, path, torch)
                generated_count += 1
            completed = resumed_count + generated_count
            if completed % args.progress_every < len(batch) or completed == len(emitted_records):
                print(
                    f"completed={completed}/{len(emitted_records)} "
                    f"elapsed_seconds={time.time() - started:.1f}",
                    flush=True,
                )
        if runtime_device.startswith("cuda"):
            torch.cuda.synchronize()

    for record in emitted_records:
        load_and_validate_artifact(
            artifact_path(args.output_root, record.sequence_key),
            record,
            model_info,
            torch,
        )

    content_manifest = build_content_manifest(
        args.output_root,
        emitted_records,
        split_counts,
    )
    content_manifest_path = args.output_root / "content_manifest.jsonl"
    write_new_or_validate_identical(content_manifest_path, content_manifest)
    content_manifest_sha256 = sha256_bytes(content_manifest)

    provenance_path = args.output_root / "provenance.json"
    if provenance_path.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite provenance: {provenance_path}")
        validate_completed_provenance(
            provenance_path,
            contract_sha256,
            content_manifest_sha256,
            len(emitted_records),
        )
        status = json.loads(provenance_path.read_text(encoding="utf-8"))
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return status

    status = {
        "format_version": PROVENANCE_FORMAT_VERSION,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_contract_sha256": contract_sha256,
        "content_manifest_path": str(content_manifest_path),
        "content_manifest_sha256": content_manifest_sha256,
        "artifact_count": len(emitted_records),
        "selected_instance_count": len(records),
        "unique_antibody_count_before_debug_limit": len(unique_records),
        "generated_artifact_count": generated_count,
        "resumed_artifact_count": resumed_count,
        "runtime_device": runtime_device,
        "elapsed_seconds": time.time() - started,
        "host": platform.node(),
        "python_version": platform.python_version(),
        "frozen_test_unlocked": split_summary["frozen_test_unlocked"],
    }
    atomic_write_json(provenance_path, status)
    print(json.dumps(status, indent=2, sort_keys=True), flush=True)
    return status


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command and report failures without printing biological sequences."""
    args = parse_args(argv)
    try:
        run(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
