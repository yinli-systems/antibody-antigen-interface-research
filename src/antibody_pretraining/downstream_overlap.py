"""Audit exact antibody-chain overlap before downstream representation transfer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .production_corpus import ProductionCorpus, ProductionCorpusError
from .tokenizer import ID_TO_CHAIN_TYPE, decode_residues


FORMAT_VERSION = "oas_asep_development_exact_overlap_audit_v1"


class DownstreamOverlapError(RuntimeError):
    """Report malformed references or an unauthenticated corpus search."""


@dataclass(frozen=True)
class DevelopmentChainReference:
    """Bind one label-free AsEP development chain to an exact digest."""

    chain: str
    instance_id: str
    role: str
    sequence_length: int
    sequence_sha256: str

    def validate(self) -> None:
        """Reject malformed or test-role references before corpus access."""
        if self.chain not in {"heavy", "light"}:
            raise DownstreamOverlapError("Development chain role is invalid")
        if self.role not in {"train", "val"}:
            raise DownstreamOverlapError("Development data role is invalid")
        if not self.instance_id:
            raise DownstreamOverlapError("Development instance identifier is empty")
        if not 1 <= self.sequence_length <= 192:
            raise DownstreamOverlapError("Development chain length is invalid")
        _digest_bytes(self.sequence_sha256)


def sequence_sha256(sequence: str) -> str:
    """Return the canonical exact-sequence digest used by the OAS corpus."""
    if not isinstance(sequence, str):
        raise DownstreamOverlapError("Development sequence must be text")
    canonical = sequence.strip().upper()
    if not canonical or len(canonical) > 192:
        raise DownstreamOverlapError("Development sequence length is invalid")
    if set(canonical) - set("ACDEFGHIKLMNPQRSTVWYX"):
        raise DownstreamOverlapError("Development sequence has invalid residues")
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _digest_bytes(value: str) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DownstreamOverlapError("Sequence digest is not a lowercase SHA-256")
    return bytes.fromhex(value)


def _array(path: Path, *, dtype: np.dtype[Any], dimensions: int) -> np.ndarray:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise DownstreamOverlapError(f"Cannot open corpus array: {path.name}") from error
    if value.dtype != dtype or value.ndim != dimensions or value.flags.writeable:
        raise DownstreamOverlapError(f"Corpus array schema differs: {path.name}")
    return value


def _digest_positions(shard: Any, digest: bytes) -> np.ndarray:
    """Find one digest in a manifest-authenticated, digest-sorted shard."""
    reference = shard.arrays.get("sequence_sha256.npy")
    if reference is None:
        raise DownstreamOverlapError("Corpus shard lacks sequence digests")
    values = _array(reference.path, dtype=np.dtype("uint8"), dimensions=2)
    if values.shape != (shard.record_count, 32):
        raise DownstreamOverlapError("Corpus sequence-digest shape differs")
    if values.size:
        if int(values[0, 0]) != shard.bucket or int(values[-1, 0]) != shard.bucket:
            raise DownstreamOverlapError("Corpus digest lies in the wrong bucket")
    packed = values.view(np.dtype("V32")).reshape(-1)
    needle = np.frombuffer(digest, dtype=np.dtype("V32"), count=1)[0]
    left = int(np.searchsorted(packed, needle, side="left"))
    right = int(np.searchsorted(packed, needle, side="right"))
    positions = np.arange(left, right, dtype=np.int64)
    if positions.size and not np.all(values[positions] == np.frombuffer(digest, dtype=np.uint8)):
        raise DownstreamOverlapError("Corpus digest binary search is inconsistent")
    return positions


def _match_records(shard: Any, digest: bytes, positions: np.ndarray) -> list[dict[str, Any]]:
    """Reconstruct every match and re-prove its digest from authenticated tokens."""
    if not positions.size:
        return []
    required = {
        "chain_types.npy": (np.dtype("uint8"), 1),
        "lengths.npy": (np.dtype("uint16"), 1),
        "tokens.npy": (np.dtype("uint8"), 2),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, (dtype, dimensions) in required.items():
        reference = shard.arrays.get(name)
        if reference is None:
            raise DownstreamOverlapError(f"Corpus shard lacks {name}")
        arrays[name] = _array(reference.path, dtype=dtype, dimensions=dimensions)
        if arrays[name].shape[0] != shard.record_count:
            raise DownstreamOverlapError(f"Corpus array row count differs: {name}")
    output: list[dict[str, Any]] = []
    for position in positions.tolist():
        encoded_length = int(arrays["lengths.npy"][position])
        tokens = arrays["tokens.npy"][position, :encoded_length]
        try:
            sequence = decode_residues(tokens.tolist())
            chain_type = ID_TO_CHAIN_TYPE[int(arrays["chain_types.npy"][position])]
        except (KeyError, ValueError) as error:
            raise DownstreamOverlapError("Matched corpus row cannot be decoded") from error
        observed = hashlib.sha256(sequence.encode("ascii")).digest()
        if observed != digest or len(sequence) + 2 != encoded_length:
            raise DownstreamOverlapError("Matched corpus row does not reproduce its digest")
        output.append(
            {
                "bucket": int(shard.bucket),
                "chain_type": chain_type,
                "local_index": int(position),
                "sequence_length": len(sequence),
            }
        )
    return output


def find_exact_matches(
    corpus: ProductionCorpus,
    references: Iterable[DevelopmentChainReference],
) -> dict[str, list[dict[str, Any]]]:
    """Return exact matches keyed by reference digest for one corpus split."""
    if corpus.contract.array_file_hashes_verified is not True:
        raise DownstreamOverlapError("Corpus array hashes were not verified")
    by_bucket = {int(shard.bucket): shard for shard in corpus.shards}
    if len(by_bucket) != len(corpus.shards):
        raise DownstreamOverlapError("Corpus repeats a digest bucket")
    digests = sorted({reference.sequence_sha256 for reference in references})
    output: dict[str, list[dict[str, Any]]] = {}
    for value in digests:
        digest = _digest_bytes(value)
        bucket = digest[0]
        shard = by_bucket.get(bucket)
        if shard is None:
            raise DownstreamOverlapError("Corpus lacks a required digest bucket")
        positions = _digest_positions(shard, digest)
        output[value] = _match_records(shard, digest, positions)
    return output


def audit_exact_development_overlap(
    *,
    references: Iterable[DevelopmentChainReference],
    train_corpus: ProductionCorpus,
    validation_corpus: ProductionCorpus,
) -> dict[str, Any]:
    """Audit exact OAS overlap and fail closed for full-development transfer."""
    items = tuple(references)
    if not items:
        raise DownstreamOverlapError("Development reference set is empty")
    for item in items:
        item.validate()
    identities = [(item.instance_id, item.chain) for item in items]
    if len(identities) != len(set(identities)):
        raise DownstreamOverlapError("Development chain reference is duplicated")
    try:
        train_matches = find_exact_matches(train_corpus, items)
        validation_matches = find_exact_matches(validation_corpus, items)
    except ProductionCorpusError as error:
        raise DownstreamOverlapError("Production corpus rejected overlap audit") from error

    records: list[dict[str, Any]] = []
    contaminated_instances: set[str] = set()
    matched_unique_digests: set[str] = set()
    for item in sorted(items, key=lambda value: (value.instance_id, value.chain)):
        train = train_matches[item.sequence_sha256]
        validation = validation_matches[item.sequence_sha256]
        matched = bool(train or validation)
        if matched:
            contaminated_instances.add(item.instance_id)
            matched_unique_digests.add(item.sequence_sha256)
        records.append(
            {
                "chain": item.chain,
                "exact_overlap": matched,
                "instance_id": item.instance_id,
                "oas_train_matches": train,
                "oas_validation_matches": validation,
                "role": item.role,
                "sequence_length": item.sequence_length,
                "sequence_sha256": item.sequence_sha256,
            }
        )

    role_counts = Counter(item.role for item in items)
    chain_counts = Counter(item.chain for item in items)
    clean_instances = sorted({item.instance_id for item in items} - contaminated_instances)
    return {
        "format_version": FORMAT_VERSION,
        "counts": {
            "development_chain_references": len(items),
            "development_instances": len({item.instance_id for item in items}),
            "exactly_overlapping_chain_references": sum(
                bool(record["exact_overlap"]) for record in records
            ),
            "exactly_overlapping_instances": len(contaminated_instances),
            "matched_unique_sequence_sha256": len(matched_unique_digests),
            "unique_reference_sequence_sha256": len(
                {item.sequence_sha256 for item in items}
            ),
        },
        "reference_counts_by_chain": dict(sorted(chain_counts.items())),
        "reference_counts_by_role": dict(sorted(role_counts.items())),
        "records": records,
        "decontaminated_development_instance_ids": clean_instances,
        "exactly_overlapping_instance_ids": sorted(contaminated_instances),
        "transfer_gate": {
            "full_development_exact_decontamination_passed": not contaminated_instances,
            "decontaminated_subset_required": bool(contaminated_instances),
            "near_identity_cluster_audit_completed": False,
            "downstream_transfer_authorized": False,
        },
        "claim_boundary": (
            "Exact antibody-chain overlap audit only. A zero exact-match count does "
            "not establish sequence-cluster decontamination, downstream benefit, "
            "official-test validity, or partner-specific pretraining."
        ),
    }


__all__ = [
    "DevelopmentChainReference",
    "DownstreamOverlapError",
    "FORMAT_VERSION",
    "audit_exact_development_overlap",
    "find_exact_matches",
    "sequence_sha256",
]
