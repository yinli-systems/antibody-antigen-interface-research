"""Test fail-closed OAS-to-AsEP exact-overlap auditing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from antibody_pretraining.downstream_overlap import (
    DevelopmentChainReference,
    DownstreamOverlapError,
    audit_exact_development_overlap,
    sequence_sha256,
)
from antibody_pretraining.production_corpus import ProductionCorpusContract
from antibody_pretraining.tokenizer import (
    CHAIN_TYPE_TO_ID,
    encode_unpaired,
    pad_encoded,
)


@dataclass(frozen=True)
class _Reference:
    path: Path


@dataclass(frozen=True)
class _Shard:
    arrays: dict[str, _Reference]
    bucket: int
    record_count: int


def _corpus(
    tmp_path: Path,
    name: str,
    sequences: list[tuple[str, str]],
    *,
    required_sequences: tuple[str, ...] = (),
):
    root = tmp_path / name
    root.mkdir()
    grouped: dict[int, list[tuple[bytes, str, str]]] = {}
    for sequence, chain_type in sequences:
        digest = hashlib.sha256(sequence.encode("ascii")).digest()
        grouped.setdefault(digest[0], []).append((digest, sequence, chain_type))
    required_buckets = {
        hashlib.sha256(sequence.encode("ascii")).digest()[0]
        for sequence in required_sequences
    }
    shards = []
    for bucket in sorted(set(grouped) | required_buckets):
        rows = grouped.get(bucket, [])
        rows.sort(key=lambda value: value[0])
        encoded = [encode_unpaired(sequence, chain_type) for _, sequence, chain_type in rows]
        inputs = pad_encoded(encoded)[0] if encoded else []
        arrays = {
            "sequence_sha256.npy": np.asarray(
                [list(digest) for digest, _, _ in rows], dtype=np.uint8
            ).reshape(-1, 32),
            "chain_types.npy": np.asarray(
                [CHAIN_TYPE_TO_ID[chain_type] for _, _, chain_type in rows],
                dtype=np.uint8,
            ),
            "lengths.npy": np.asarray(
                [len(item.input_ids) for item in encoded], dtype=np.uint16
            ),
            "tokens.npy": (
                np.asarray(inputs, dtype=np.uint8)
                if inputs
                else np.empty((0, 194), dtype=np.uint8)
            ),
        }
        references = {}
        for field, value in arrays.items():
            path = root / f"{bucket}-{field}"
            np.save(path, value, allow_pickle=False)
            references[field] = _Reference(path)
        shards.append(_Shard(references, bucket, len(rows)))
    return SimpleNamespace(
        contract=ProductionCorpusContract(
            independent_audit_file_sha256="a" * 64,
            independent_audit_payload_sha256="b" * 64,
            manifest_file_sha256="c" * 64,
            manifest_payload_sha256="d" * 64,
            protocol_file_sha256="e" * 64,
            split=name,
            array_file_hashes_verified=True,
        ),
        shards=tuple(shards),
    )


def _reference(instance: str, chain: str, role: str, sequence: str):
    return DevelopmentChainReference(
        chain=chain,
        instance_id=instance,
        role=role,
        sequence_length=len(sequence),
        sequence_sha256=sequence_sha256(sequence),
    )


def test_exact_overlap_reconstructs_tokens_and_fails_full_transfer(tmp_path: Path) -> None:
    heavy = "A" * 80
    light = "C" * 80
    train = _corpus(
        tmp_path,
        "train",
        [(heavy, "heavy")],
        required_sequences=(heavy, light),
    )
    validation = _corpus(
        tmp_path,
        "validation",
        [(light, "kappa")],
        required_sequences=(heavy, light),
    )
    result = audit_exact_development_overlap(
        references=(
            _reference("complex-1", "heavy", "train", heavy),
            _reference("complex-1", "light", "train", light),
        ),
        train_corpus=train,
        validation_corpus=validation,
    )
    assert result["counts"]["exactly_overlapping_chain_references"] == 2
    assert result["exactly_overlapping_instance_ids"] == ["complex-1"]
    assert result["decontaminated_development_instance_ids"] == []
    assert result["transfer_gate"] == {
        "full_development_exact_decontamination_passed": False,
        "decontaminated_subset_required": True,
        "near_identity_cluster_audit_completed": False,
        "downstream_transfer_authorized": False,
    }
    assert result["records"][0]["oas_train_matches"][0]["chain_type"] == "heavy"
    assert result["records"][1]["oas_validation_matches"][0]["chain_type"] == "kappa"


def test_nonoverlap_keeps_exact_gate_but_does_not_authorize_transfer(tmp_path: Path) -> None:
    reference_sequence = "A" * 80
    train = _corpus(
        tmp_path,
        "train",
        [("C" * 80, "heavy")],
        required_sequences=(reference_sequence,),
    )
    validation = _corpus(
        tmp_path,
        "validation",
        [("D" * 80, "lambda")],
        required_sequences=(reference_sequence,),
    )
    result = audit_exact_development_overlap(
        references=(_reference("complex-2", "heavy", "val", reference_sequence),),
        train_corpus=train,
        validation_corpus=validation,
    )
    gate = result["transfer_gate"]
    assert gate["full_development_exact_decontamination_passed"] is True
    assert gate["near_identity_cluster_audit_completed"] is False
    assert gate["downstream_transfer_authorized"] is False


def test_unverified_corpus_and_test_role_are_rejected(tmp_path: Path) -> None:
    sequence = "A" * 80
    corpus = _corpus(tmp_path, "train", [(sequence, "heavy")])
    corpus.contract = SimpleNamespace(array_file_hashes_verified=False)
    with pytest.raises(DownstreamOverlapError, match="array hashes"):
        audit_exact_development_overlap(
            references=(_reference("complex", "heavy", "train", sequence),),
            train_corpus=corpus,
            validation_corpus=corpus,
        )
    reference = DevelopmentChainReference(
        chain="heavy",
        instance_id="official-test",
        role="test",
        sequence_length=len(sequence),
        sequence_sha256=sequence_sha256(sequence),
    )
    with pytest.raises(DownstreamOverlapError, match="data role"):
        reference.validate()


def test_match_digest_is_reproved_from_tokens(tmp_path: Path) -> None:
    sequence = "A" * 80
    corpus = _corpus(tmp_path, "train", [(sequence, "heavy")])
    token_path = corpus.shards[0].arrays["tokens.npy"].path
    tokens = np.load(token_path, allow_pickle=False)
    tokens[0, 1] = encode_unpaired("C" * 80, "heavy").input_ids[1]
    np.save(token_path, tokens, allow_pickle=False)
    with pytest.raises(DownstreamOverlapError, match="does not reproduce"):
        audit_exact_development_overlap(
            references=(_reference("complex", "heavy", "train", sequence),),
            train_corpus=corpus,
            validation_corpus=corpus,
        )
