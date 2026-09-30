#!/usr/bin/env python3
"""Freeze, audit, and spec-bind the full SaProt/ProSST feature programme."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path[:0] = [str(REPO / "src"), str(REPO)]

from interaction_first.pretrained_representations import (
    representation_path,
    validate_representation_artifact,
)
from extensions.frontier_multimodel_v1.structural_plm import (
    PROSST_OFFICIAL_ENVIRONMENT,
    SAPROT_ALLOWED_MISSING_KEYS,
    load_units,
    selected,
    sha256_file,
)

LAYERS = {"saprot": (0, 8, 16, 24, 33), "prosst": (0, 3, 6, 9, 12)}
SHARDS = {"saprot_extract": 16, "prosst_quant": 32, "prosst_extract": 16}
MODEL = {
    "saprot": {
        "revision": "d9b9ad00ef61c0990e611b2b43f2231c7de24b38",
        "source_commit": "e91e4858b55944523f1f8d385f7b96a0d3d34c1d",
        "dim": 1280,
        "weight_sha256": "513c39088804b74309d961174d5fc7aeabfb9b79d2814cdc7e465d4fe4b9f216",
        "tokenizer_sha256": "d4289b3ff55d50b1aa33a3141ec344fc2a69cb3dca43aeefe9fccfee371ffd28",
    },
    "prosst": {
        "revision": "e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8",
        "source_commit": "13bd56495641a84cd5475eaef7aa98b14fdc6ed5",
        "dim": 768,
        "weight_sha256": "aa6742cd2fd4c0ad3b908f19358136c66614f07093cdf9a995120723c6d54368",
        "tokenizer_sha256": "89b5c54e41dff9efbc23323f0750ab329585cb01ccb77144dd8838adddaf358c",
    },
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    os.close(handle)
    try:
        Path(temporary).write_text(data)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def tree_sha256(root: Path, *, exclude_names: set[str] | None = None) -> tuple[str, int, int]:
    exclude_names = exclude_names or set()
    digest = hashlib.sha256()
    count = 0
    total = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in exclude_names):
        relative = path.relative_to(root).as_posix()
        content = sha256_file(path)
        size = path.stat().st_size
        digest.update(f"{relative}\t{size}\t{content}\n".encode())
        count += 1
        total += size
    return digest.hexdigest(), count, total


def freeze(arguments: argparse.Namespace) -> None:
    structure = arguments.structure_units.resolve()
    saprot_tokens = arguments.saprot_tokens.resolve()
    model_authority = arguments.model_authority.resolve()
    authority, rows = load_units(structure)
    counts = {side: sum(row["side"] == side for row in rows) for side in ("antibody", "antigen")}
    if counts != {"antibody": 2174, "antigen": 1171}:
        raise ValueError(f"structure authority changed: {counts}")
    saprot_receipt = json.loads((saprot_tokens / "RECEIPT.json").read_text())
    if saprot_receipt.get("status") != "PASS" or saprot_receipt.get("token_count") != len(rows):
        raise ValueError("SaProt token authority is incomplete")
    if saprot_receipt.get("structure_units_sha256") != sha256_file(structure):
        raise ValueError("SaProt token/structure authority mismatch")
    saprot_tree, saprot_files, saprot_bytes = tree_sha256(saprot_tokens)
    models = json.loads(model_authority.read_text())
    by_name = {item["name"]: item for item in models["models"]}
    if by_name["saprot650"]["hf_revision"] != MODEL["saprot"]["revision"]:
        raise ValueError("SaProt revision drift")
    if by_name["prosst2048"]["hf_revision"] != MODEL["prosst"]["revision"]:
        raise ValueError("ProSST revision drift")
    inputs = [
        structure,
        saprot_tokens / "RECEIPT.json",
        model_authority,
        REPO / "extensions/frontier_multimodel_v1/structural_plm.py",
        REPO / "extensions/frontier_multimodel_v1/structural_plm_full.py",
        REPO / "extensions/frontier_multimodel_v1/STRUCTURAL_PLM_LAYER_SWEEP_PROTOCOL.md",
    ]
    plan = {
        "format_version": "structural_plm_full_plan_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": arguments.commit,
        "structure_units": str(structure),
        "structure_units_sha256": sha256_file(structure),
        "record_count": len(rows),
        "counts_by_side": counts,
        "saprot_tokens": str(saprot_tokens),
        "saprot_token_tree_sha256": saprot_tree,
        "saprot_token_tree_files": saprot_files,
        "saprot_token_tree_bytes": saprot_bytes,
        "model_authority": str(model_authority),
        "models": MODEL,
        "layers": {key: list(value) for key, value in LAYERS.items()},
        "shards": SHARDS,
        "prosst_official_environment": PROSST_OFFICIAL_ENVIRONMENT,
        "saprot_allowed_missing_keys": sorted(SAPROT_ALLOWED_MISSING_KEYS),
        "input_sha256": {str(path): sha256_file(path) for path in inputs},
        "output": str(arguments.output_root.resolve()),
        "test_authorized": False,
        "contact_labels_authorized": False,
        "training_authorized": False,
        "evidence_role": "development-only frozen structural feature extraction",
    }
    if arguments.plan.exists():
        raise FileExistsError(arguments.plan)
    atomic_json(arguments.plan, plan)
    print(json.dumps({"status": "PASS_FULL_EXTRACTION_PLAN_FROZEN", "plan_sha256": sha256_file(arguments.plan), "records": len(rows), "layers": plan["layers"]}))


def expected_by_side(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        side: {row["sequence_sha256"]: row for row in rows if row["side"] == side}
        for side in ("antibody", "antigen")
    }


def verify_receipt_coverage(receipts: Iterable[Path], expected: set[str], *, kind: str,
                            model: str | None = None, layers: tuple[int, ...] | None = None) -> list[dict]:
    seen: set[str] = set()
    payloads = []
    for path in sorted(receipts):
        payload = json.loads(path.read_text())
        if payload.get("status") != "PASS" or payload.get("test_loaded") or payload.get("contact_labels_loaded"):
            raise ValueError(f"bad receipt firewall/status: {path}")
        if kind == "extract":
            if payload.get("model_kind") != model or tuple(payload.get("layers", [])) != tuple(layers or ()):
                raise ValueError(f"extract receipt model/layer mismatch: {path}")
            keys = [row["sequence_sha256"] for row in payload.get("records", [])]
            if payload.get("record_count") != len(keys):
                raise ValueError(f"extract receipt count mismatch: {path}")
        else:
            # Token receipts record a digest and count, but not every key. Rebuild
            # the shard authority using the same deterministic selector.
            side = payload["side"]
            shard = int(payload["shard_index"])
            shard_count = int(payload["shard_count"])
            keys = sorted(
                key for key in expected
                if int(key[:16], 16) % shard_count == shard
            )
            if payload.get("requested") != len(keys) or payload.get("written") + payload.get("skipped_exact_existing") != len(keys):
                raise ValueError(f"token receipt count mismatch: {path}")
            digest = hashlib.sha256(("\n".join(keys) + "\n").encode()).hexdigest()
            if payload.get("keys_sha256") != digest:
                raise ValueError(f"token shard key digest mismatch: {path}")
        overlap = seen.intersection(keys)
        if overlap:
            raise ValueError(f"duplicate receipt coverage {path}: {len(overlap)}")
        seen.update(keys)
        payloads.append(payload)
    if seen != expected:
        raise ValueError(f"{kind} receipt coverage mismatch: missing={len(expected-seen)} extra={len(seen-expected)}")
    return payloads


def audit(arguments: argparse.Namespace) -> None:
    import torch
    plan = json.loads(arguments.plan.read_text())
    if sha256_file(arguments.plan) != arguments.plan_sha256:
        raise ValueError("plan digest mismatch")
    for path, digest in plan["input_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"frozen input changed: {path}")
    saprot_tree, saprot_files, saprot_bytes = tree_sha256(Path(plan["saprot_tokens"]))
    if (saprot_tree != plan["saprot_token_tree_sha256"]
            or saprot_files != plan["saprot_token_tree_files"]
            or saprot_bytes != plan["saprot_token_tree_bytes"]):
        raise ValueError("SaProt structural token tree changed after freeze")
    authority, rows = load_units(Path(plan["structure_units"]))
    expected = expected_by_side(rows)
    root = Path(plan["output"])
    token_root = root / "prosst_tokens"
    feature_root = root / "features"
    audit_rows = []
    loading = {}
    for side in ("antibody", "antigen"):
        side_expected = set(expected[side])
        quant_receipts = list(token_root.glob(f"RECEIPT_{side}_*_of_*.json"))
        quant_payloads = verify_receipt_coverage(quant_receipts, side_expected, kind="token")
        for payload in quant_payloads:
            if payload.get("quantizer_environment") != PROSST_OFFICIAL_ENVIRONMENT:
                raise ValueError("ProSST official environment not recorded")
        for model in ("saprot", "prosst"):
            receipt_paths = list(feature_root.glob(f"RECEIPT_{model}_{side}_*_of_*.json"))
            payloads = verify_receipt_coverage(receipt_paths, side_expected, kind="extract", model=model, layers=LAYERS[model])
            loading[(model, side)] = [payload.get("model_loading_info") for payload in payloads]
            for payload in payloads:
                info = payload.get("model_loading_info", {})
                missing = set(info.get("missing_keys", []))
                if model == "saprot":
                    if missing != SAPROT_ALLOWED_MISSING_KEYS or info.get("unexpected_keys") or info.get("mismatched_keys"):
                        raise ValueError("SaProt loading info not exact whitelist")
                elif missing or info.get("unexpected_keys") or info.get("mismatched_keys"):
                    raise ValueError("ProSST loading info is not clean")
            for key, record in expected[side].items():
                token_path = (
                    Path(plan["saprot_tokens"]) / side / key[:2] / f"{key}.json"
                    if model == "saprot"
                    else token_root / side / key[:2] / f"{key}.json"
                )
                token_payload = json.loads(token_path.read_text())
                if token_payload.get("sequence_sha256") != key or token_payload.get("sequence") != record["sequence"] or token_payload.get("pdb_sha256") != record["pdb_sha256"]:
                    raise ValueError(f"token authority mismatch: {model}:{side}:{key}")
                token_sha = sha256_file(token_path)
                for layer in LAYERS[model]:
                    artifact_path = representation_path(feature_root / f"{model}_layer_{layer:02d}" / side, key)
                    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
                    validate_representation_artifact(artifact, expected_key=key, expected_dim=MODEL[model]["dim"])
                    if artifact.get("model_name") != f"{model}_{MODEL[model]['revision']}_layer_{layer}":
                        raise ValueError(f"model name mismatch: {artifact_path}")
                    if artifact.get("model_revision") != MODEL[model]["revision"] or artifact.get("source_pdb_sha256") != record["pdb_sha256"]:
                        raise ValueError(f"model/source provenance mismatch: {artifact_path}")
                    if artifact.get("structural_token_sha256") != token_sha or artifact.get("layer_index") != layer:
                        raise ValueError(f"token/layer provenance mismatch: {artifact_path}")
                    audit_rows.append({
                        "model": model,
                        "side": side,
                        "layer": layer,
                        "sequence_sha256": key,
                        "artifact_path": str(artifact_path),
                        "artifact_sha256": sha256_file(artifact_path),
                        "sequence_length": int(artifact["sequence_length"]),
                    })
    expected_artifacts = len(rows) * sum(len(value) for value in LAYERS.values())
    if len(audit_rows) != expected_artifacts:
        raise ValueError(f"artifact count mismatch: {len(audit_rows)} vs {expected_artifacts}")
    specs = []
    specs_root = root / "source_specs"
    for model, layers in LAYERS.items():
        for layer in layers:
            for side in ("antibody", "antigen"):
                specification = {
                    "format_version": "generic_representation_source_v1",
                    "name": f"{model}_l{layer:02d}_{'ab' if side=='antibody' else 'ag'}",
                    "root": str(feature_root / f"{model}_layer_{layer:02d}" / side),
                    "dim": MODEL[model]["dim"],
                    "sides": [side],
                    "antibody_storage": "parts",
                    "model_name": f"{model}_{MODEL[model]['revision']}_layer_{layer}",
                    "model_revision": MODEL[model]["revision"],
                    "evidence_role": "development-only frozen structural PLM layer sweep",
                    "source_commit": MODEL[model]["source_commit"],
                    "model_sha256": MODEL[model]["weight_sha256"],
                    "tokenizer_sha256": MODEL[model]["tokenizer_sha256"],
                    "structure_authority_sha256": plan["structure_units_sha256"],
                }
                target = specs_root / f"{model}_layer_{layer:02d}_{side}.json"
                atomic_json(target, specification)
                specs.append({"path": str(target), "sha256": sha256_file(target), **specification})
    artifact_digest = hashlib.sha256()
    for row in sorted(audit_rows, key=lambda item: (item["model"], item["side"], item["layer"], item["sequence_sha256"])):
        artifact_digest.update(f"{row['model']}\t{row['side']}\t{row['layer']}\t{row['sequence_sha256']}\t{row['artifact_sha256']}\n".encode())
    result = {
        "status": "PASS_COMPLETE_STRUCTURAL_PLM_FEATURE_AUTHORITY",
        "format_version": "structural_plm_full_audit_v1",
        "plan_sha256": arguments.plan_sha256,
        "structure_record_count": len(rows),
        "feature_artifact_count": len(audit_rows),
        "feature_artifacts_expected": expected_artifacts,
        "artifact_manifest_sha256": artifact_digest.hexdigest(),
        "layers": {key: list(value) for key, value in LAYERS.items()},
        "source_specifications": specs,
        "test_loaded": False,
        "contact_labels_loaded": False,
        "training_performed": False,
        "scope": "complete development-only frozen feature authority; not downstream performance",
    }
    atomic_json(arguments.output, result)
    print(json.dumps({key: result[key] for key in ("status", "structure_record_count", "feature_artifact_count", "artifact_manifest_sha256")}))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--structure-units", type=Path, required=True)
    freeze_parser.add_argument("--saprot-tokens", type=Path, required=True)
    freeze_parser.add_argument("--model-authority", type=Path, required=True)
    freeze_parser.add_argument("--output-root", type=Path, required=True)
    freeze_parser.add_argument("--commit", required=True)
    freeze_parser.add_argument("--plan", type=Path, required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--plan", type=Path, required=True)
    audit_parser.add_argument("--plan-sha256", required=True)
    audit_parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = parser().parse_args()
    {"freeze": freeze, "audit": audit}[arguments.command](arguments)
