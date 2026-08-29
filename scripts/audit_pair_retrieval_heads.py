#!/usr/bin/env python3
"""Audit the pinned official MINT and PPLM PPI scoring heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch
from torch import Tensor


MINT_HEAD_SHA256 = "702849af78e245c7596d8c032391da52ff6657e9173af380e29a8ea3858fdab9"
PPLM_HEAD_SHA256 = "2695b886ef3ba2c1acacb74cd4f8c8173323addfc548b6d482143fb2b7070d8f"
MINT_REVISION = "75b1602c34e9df1a89089a1245f000a51ee3c746"
PPLM_REVISION = "1267db4592703095b889cc22da6cf1796ce038b5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mint_expected_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "project.0.weight": (2560, 2560),
        "project.0.bias": (2560,),
        "project.3.weight": (1, 2560),
        "project.3.bias": (1,),
    }


def pplm_expected_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "linear_intra.weight": (660, 660),
        "linear_intra.bias": (660,),
        "linear_inter.weight": (660, 660),
        "linear_inter.bias": (660,),
        "linear_embed.weight": (660, 1280),
        "linear_embed.bias": (660,),
        "mlp.0.weight": (1024, 3300),
        "mlp.0.bias": (1024,),
        "mlp.1.weight": (1024,),
        "mlp.1.bias": (1024,),
        "mlp.3.weight": (512, 1024),
        "mlp.3.bias": (512,),
        "mlp.4.weight": (512,),
        "mlp.4.bias": (512,),
        "mlp.6.weight": (256, 512),
        "mlp.6.bias": (256,),
        "mlp.7.weight": (256,),
        "mlp.7.bias": (256,),
        "mlp.9.weight": (128, 256),
        "mlp.9.bias": (128,),
        "mlp.10.weight": (128,),
        "mlp.10.bias": (128,),
        "mlp.12.weight": (1, 128),
        "mlp.12.bias": (1,),
    }


def audit_state_shapes(
    state: Mapping[str, Tensor], expected: Mapping[str, tuple[int, ...]]
) -> dict[str, object]:
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatches = {
        key: {"expected": list(expected[key]), "observed": list(state[key].shape)}
        for key in sorted(set(expected) & set(state))
        if isinstance(state[key], Tensor) and tuple(state[key].shape) != expected[key]
    }
    non_tensor = sorted(key for key, value in state.items() if not isinstance(value, Tensor))
    wrong_dtype = sorted(
        key for key, value in state.items() if isinstance(value, Tensor) and value.dtype != torch.float32
    )
    nonfinite = sorted(
        key
        for key, value in state.items()
        if isinstance(value, Tensor) and not bool(torch.isfinite(value).all())
    )
    compatible = not any(
        (missing, unexpected, shape_mismatches, non_tensor, wrong_dtype, nonfinite)
    )
    return {
        "compatible": compatible,
        "missing_keys": missing,
        "non_tensor_keys": non_tensor,
        "nonfinite_keys": nonfinite,
        "parameter_elements": sum(
            value.numel() for value in state.values() if isinstance(value, Tensor)
        ),
        "shape_mismatches": shape_mismatches,
        "unexpected_keys": unexpected,
        "wrong_dtype_keys": wrong_dtype,
    }


def state_digest(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_materialization(
    receipt_path: Path,
    *,
    expected_file: str,
    expected_revision: str,
    expected_sha256: str,
) -> str:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    file_record = receipt.get("files", {}).get(expected_file, {})
    if (
        receipt.get("format_version") != "abpairfm_public_materialization_v1"
        or receipt.get("status") != "MATERIALIZED_NOT_NORMALIZED"
        or receipt.get("pinned_revision") != expected_revision
        or receipt.get("test_access") is not False
        or file_record.get("sha256") != expected_sha256
    ):
        raise ValueError(f"Materialization receipt differs: {expected_file}")
    return sha256_file(receipt_path)


def publish_create_only(path: Path, value: object) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite retrieval-head audit: {path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mint-head", type=Path, required=True)
    parser.add_argument("--mint-materialization", type=Path, required=True)
    parser.add_argument("--pplm-head", type=Path, required=True)
    parser.add_argument("--pplm-materialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if sha256_file(arguments.mint_head) != MINT_HEAD_SHA256:
        raise ValueError("MINT head SHA-256 differs")
    if sha256_file(arguments.pplm_head) != PPLM_HEAD_SHA256:
        raise ValueError("PPLM head SHA-256 differs")
    mint_materialization_sha256 = verify_materialization(
        arguments.mint_materialization,
        expected_file="bernett_mlp.pth",
        expected_revision=MINT_REVISION,
        expected_sha256=MINT_HEAD_SHA256,
    )
    pplm_materialization_sha256 = verify_materialization(
        arguments.pplm_materialization,
        expected_file="ppi_models.pkl",
        expected_revision=PPLM_REVISION,
        expected_sha256=PPLM_HEAD_SHA256,
    )
    mint_state = torch.load(
        arguments.mint_head, map_location="cpu", mmap=True, weights_only=True
    )
    if not isinstance(mint_state, Mapping):
        raise ValueError("MINT head is not a state mapping")
    mint_audit = audit_state_shapes(mint_state, mint_expected_shapes())
    pplm_payload = torch.load(
        arguments.pplm_head, map_location="cpu", mmap=True, weights_only=True
    )
    if not isinstance(pplm_payload, Mapping) or set(pplm_payload) != {"mean", "max"}:
        raise ValueError("PPLM PPI head top-level schema differs")
    pplm_audits: dict[str, list[dict[str, object]]] = {}
    member_digests: list[str] = []
    for reduction in ("mean", "max"):
        members = pplm_payload[reduction]
        if not isinstance(members, list) or len(members) != 5:
            raise ValueError(f"PPLM {reduction} ensemble membership differs")
        pplm_audits[reduction] = []
        for state in members:
            if not isinstance(state, Mapping):
                raise ValueError("PPLM ensemble member is not a state mapping")
            audit = audit_state_shapes(state, pplm_expected_shapes())
            digest = state_digest(state)
            member_digests.append(digest)
            pplm_audits[reduction].append({**audit, "state_sha256": digest})
    pplm_compatible = (
        all(member["compatible"] for values in pplm_audits.values() for member in values)
        and len(set(member_digests)) == 10
    )
    compatible = bool(mint_audit["compatible"] and pplm_compatible)
    result = {
        "authorization": {
            "frozen_development_pair_retrieval": compatible,
            "frozen_development_contact": False,
            "production_training": False,
            "test_access": False,
        },
        "claim_boundary": "Safe state-schema audit of the official MINT Bernett PPI MLP and PPLM-PPI ten-member ensemble. Compatibility authorizes development-only ranking on the already frozen cognate/decoy panel; it does not establish antibody-domain calibration, contact compatibility, test performance, or SOTA.",
        "format_version": "abpairfm_pair_retrieval_head_audit_v1",
        "materialization_receipt_sha256": {
            "mint": mint_materialization_sha256,
            "pplm": pplm_materialization_sha256,
        },
        "mint": {
            "audit": mint_audit,
            "checkpoint_sha256": MINT_HEAD_SHA256,
            "official_ensemble_members": 1,
            "source_revision": MINT_REVISION,
        },
        "pplm": {
            "checkpoint_sha256": PPLM_HEAD_SHA256,
            "ensemble": pplm_audits,
            "ensemble_member_count": len(member_digests),
            "member_state_digests_unique": len(set(member_digests)) == 10,
            "source_revision": PPLM_REVISION,
        },
        "status": "PAIR_RETRIEVAL_HEADS_COMPATIBLE" if compatible else "INCOMPATIBLE_FAIL_CLOSED",
        "test_access": False,
    }
    publish_create_only(arguments.output, result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
