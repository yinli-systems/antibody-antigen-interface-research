from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "extensions/strict_partner_retrieval_v1/stage_tensor_cache.py"
spec = importlib.util.spec_from_file_location("stage_tensor_cache", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def key(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def generic(torch, sequence: str, value: float) -> dict:
    return {
        "format_version": "pretrained_representation_v1",
        "sequence_sha256": key(sequence),
        "sequence_length": len(sequence),
        "model_name": "antiberty",
        "model_revision": "antiberty-pypi",
        "representation_dim": 512,
        "starts": [0],
        "ends": [len(sequence)],
        "chunks": [torch.full((len(sequence), 512), value)],
    }


def esm2(torch, sequence: str, value: float) -> dict:
    return {
        "format_version": "sppider_seq_embedding_v1",
        "sequence_sha256": key(sequence),
        "sequence_length": len(sequence),
        "model_name": "facebook/esm2_t33_650M_UR50D",
        "model_weight_sha256": "e" * 64,
        "starts": [0],
        "ends": [len(sequence)],
        "chunks": [torch.full((len(sequence), 1280), value)],
    }


def test_chainwise_antibody_concat_and_esm2_adapter(tmp_path: Path) -> None:
    import torch

    anti = tmp_path / "antiberty"
    antigen = tmp_path / "esm2"
    vh, vl, ag = "ACDE", "FGH", "IKLMN"
    entries = []
    for role, sequence, value in (("vh", vh, 1.0), ("vl", vl, 2.0)):
        k = key(sequence)
        path = anti / k[:2] / f"{k}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(generic(torch, sequence, value), path)
        entries.append({"bank": "antiberty", "sequence_sha256": k, "file_sha256": module.sha256(path), "length": len(sequence), "dim": 512, "authority_role": role})
    k = key(ag)
    path = antigen / k[:2] / f"{k}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(esm2(torch, ag, 3.0), path)
    entries.append({"bank": "esm2", "sequence_sha256": k, "file_sha256": module.sha256(path), "length": len(ag), "dim": 1280, "authority_role": "antigen"})
    index = module.bank_index({"entries": entries})
    x, metadata = module.load_antibody(torch, anti, index, vh, vl, "antiberty-pypi", "a" * 64)
    assert tuple(x.shape) == (7, 512)
    assert bool((x[:4] == 1).all()) and bool((x[4:] == 2).all())
    assert [chain["role"] for chain in metadata["chains"]] == ["vh", "vl"]
    y, metadata = module.load_antigen(torch, path, index, k, ag, "facebook/esm2_t33_650M_UR50D", "e" * 64)
    assert tuple(y.shape) == (5, 1280)
    assert bool((y == 3).all())
    assert metadata["source_format"] == "sppider_seq_embedding_v1"


def test_scoped_digest_change_fails(tmp_path: Path) -> None:
    import torch

    sequence = "ACDE"
    k = key(sequence)
    root = tmp_path / "anti"
    path = root / k[:2] / f"{k}.pt"
    path.parent.mkdir(parents=True)
    torch.save(generic(torch, sequence, 1.0), path)
    index = module.bank_index({"entries": [{"bank": "antiberty", "sequence_sha256": k, "file_sha256": "0" * 64, "length": len(sequence), "dim": 512}]})
    try:
        module.load_antibody(torch, root, index, sequence, "", "antiberty-pypi", "a" * 64)
    except ValueError as error:
        assert "digest" in str(error).lower()
    else:
        raise AssertionError("changed source artifact was accepted")
