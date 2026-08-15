"""Test authenticated development-query and native OAS FASTA export."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from antibody_pretraining.downstream_overlap import (
    DevelopmentChainReference,
    sequence_sha256,
)
from antibody_pretraining.near_identity_export import (
    TargetShardExport,
    write_development_query_fasta,
    write_target_inventory,
)
from antibody_pretraining.near_identity_controls import (
    audit_sensitivity_hits,
    build_sensitivity_controls,
)
from antibody_pretraining.near_identity_overlap import NearIdentityOverlapError
from antibody_pretraining.tokenizer import encode_unpaired, pad_encoded


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/export_oas_production_fasta.cpp"
SMOKE_SLURM = ROOT / "slurm/oas_asep_near_identity_mmseqs_smoke.sbatch"


def _compiler() -> str:
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    return compiler


def _arrays(root: Path, sequences: tuple[str, ...]) -> tuple[Path, Path]:
    encoded = tuple(encode_unpaired(sequence, "heavy") for sequence in sequences)
    tokens = np.asarray(pad_encoded(encoded, pad_to_length=194)[0], dtype=np.uint8)
    lengths = np.asarray([len(item.input_ids) for item in encoded], dtype=np.uint16)
    tokens_path = root / "tokens.npy"
    lengths_path = root / "lengths.npy"
    np.save(tokens_path, tokens, allow_pickle=False)
    np.save(lengths_path, lengths, allow_pickle=False)
    return tokens_path, lengths_path


def test_native_exporter_streams_exact_sequences_and_split_ids(tmp_path: Path) -> None:
    first = "ACDEFGHIKLMNPQRSTVWY" * 4
    second = "YXWVTSRQPNMLKIHGFEDC" * 4
    train_root = tmp_path / "train"
    validation_root = tmp_path / "validation"
    train_root.mkdir()
    validation_root.mkdir()
    train_tokens, train_lengths = _arrays(train_root, (first, second))
    validation_tokens, validation_lengths = _arrays(validation_root, (second,))
    inventory_path = tmp_path / "inventory.tsv"
    receipt = write_target_inventory(
        inventory_path,
        (
            TargetShardExport("train", 0, 0, 2, train_tokens, train_lengths),
            TargetShardExport(
                "validation", 0, 0, 1, validation_tokens, validation_lengths
            ),
        ),
    )
    assert receipt["record_counts"] == {"train": 2, "validation": 1}
    binary = tmp_path / "exporter"
    subprocess.run(
        [_compiler(), "-std=c++17", "-O2", str(SOURCE), "-o", str(binary)],
        check=True,
    )
    fasta = tmp_path / "target.fasta"
    result = subprocess.run(
        [str(binary), str(inventory_path), str(fasta)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        "train_records=2",
        "validation_records=1",
        "train_residues=160",
        "validation_residues=80",
    ]
    assert fasta.read_text(encoding="ascii") == (
        f">t0\n{first}\n>t1\n{second}\n>v0\n{second}\n"
    )
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [str(binary), str(inventory_path), str(fasta)],
            check=True,
            capture_output=True,
        )


def test_native_exporter_rejects_corrupt_boundary_token(tmp_path: Path) -> None:
    root = tmp_path / "shard"
    root.mkdir()
    tokens_path, lengths_path = _arrays(root, ("A" * 80,))
    tokens = np.load(tokens_path, allow_pickle=False)
    tokens[0, 0] = 5
    np.save(tokens_path, tokens, allow_pickle=False)
    inventory = tmp_path / "inventory.tsv"
    write_target_inventory(
        inventory,
        (TargetShardExport("train", 0, 0, 1, tokens_path, lengths_path),),
    )
    binary = tmp_path / "exporter"
    subprocess.run(
        [_compiler(), "-std=c++17", "-O2", str(SOURCE), "-o", str(binary)],
        check=True,
    )
    result = subprocess.run(
        [str(binary), str(inventory), str(tmp_path / "target.fasta")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "boundary token differs" in result.stderr


def test_development_query_export_is_digest_bound_and_label_free(tmp_path: Path) -> None:
    sequence = "A" * 80
    reference = DevelopmentChainReference(
        chain="heavy",
        instance_id="complex-1",
        role="val",
        sequence_length=len(sequence),
        sequence_sha256=sequence_sha256(sequence),
    )
    manifest = write_development_query_fasta(
        fasta_path=tmp_path / "query.fasta",
        manifest_path=tmp_path / "query.json",
        sequences=((reference, sequence),),
    )
    persisted = json.loads((tmp_path / "query.json").read_text(encoding="ascii"))
    assert persisted == manifest
    assert persisted["query_count"] == 1
    assert persisted["test_access"] == {
        "development_sequence_fields_accessed": True,
        "official_frozen_external_test_accessed": False,
        "task_labels_used": False,
    }
    assert (tmp_path / "query.fasta").read_text(encoding="ascii").endswith(
        f"\n{sequence}\n"
    )
    with pytest.raises(NearIdentityOverlapError, match="binding differs"):
        write_development_query_fasta(
            fasta_path=tmp_path / "bad.fasta",
            manifest_path=tmp_path / "bad.json",
            sequences=((reference, "C" * 80),),
        )


def test_sensitivity_controls_require_exact_substitution_and_both_indels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "controls"
    manifest = build_sensitivity_controls(root)
    expected = manifest["expected_pairs"]
    lines = [
        f"{row['query']}\t{row['target']}\t99\t0.99\t99\n" for row in expected
    ]
    result = audit_sensitivity_hits(lines, expected_manifest=manifest)
    assert result["gate_passed"] is True
    assert all(result["controls"].values())
    missing_indel = audit_sensitivity_hits(lines[:-1], expected_manifest=manifest)
    assert missing_indel["gate_passed"] is False
    assert missing_indel["controls"]["terminal_indel_above_threshold"] is False


def test_mmseqs_smoke_script_freezes_gpu_and_alignment_semantics() -> None:
    subprocess.run(["bash", "-n", str(SMOKE_SLURM)], check=True)
    source = SMOKE_SLURM.read_text(encoding="ascii")
    assert "#SBATCH --partition=gpu_5090" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "--gpu 1" in source
    assert "--min-seq-id 0.9" in source
    assert "--cov-mode 0" in source
    assert "--seq-id-mode 0" in source
    assert "--format-output query,target,fident,qcov,tcov" in source
    assert "official_test_accessed=false" in source
