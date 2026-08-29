#!/usr/bin/env python3
"""Download the frozen public AbCDR-ESMC checkpoint and emit a SHA receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


EXPECTED_BINDING_SHA256 = "bb2605b211b4d4a247a786450e492392d1e9a532e56110ee68d83d07590ca18f"
EXPECTED_MODEL_SHA256 = "0490932ca515c6fe1738d070ede8b3ade28745d990a5e1325f286f3624c1d4e6"
REVISION = "d6e08c7a771f6c7b399e9f94cd3b9b36d0ac12b8"
URL = f"https://huggingface.co/NOC-Lab/AbCDR-ESMC/resolve/{REVISION}/model.safetensors?download=true"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def download_resumable(destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "FrameContact-authority-audit/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(URL, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    partial.unlink()
                    offset = 0
                mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as handle:
                    while True:
                        block = response.read(8 << 20)
                        if not block:
                            break
                        handle.write(block)
            os.replace(partial, destination)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 16))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.binding) != EXPECTED_BINDING_SHA256:
        raise RuntimeError("model-binding SHA differs")
    binding = json.loads(args.binding.read_text())
    published = binding.get("published_cdr_model", {})
    if (
        published.get("revision") != REVISION
        or published.get("expected_sha256") != EXPECTED_MODEL_SHA256
        or binding.get("gates", {}).get("downstream_training_authorized") is not False
    ):
        raise RuntimeError("model-binding content differs")
    final = args.output_root / "models" / "NOC-Lab--AbCDR-ESMC" / REVISION / "model.safetensors"
    receipt_path = args.output_root / "ABCDR_ESMC_CHECKPOINT_DOWNLOAD_RECEIPT_V1.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(receipt_path)
    if not final.exists():
        download_resumable(final)
    observed = sha256_file(final)
    if observed != EXPECTED_MODEL_SHA256:
        raise RuntimeError(f"checkpoint SHA differs: {observed}")
    receipt = {
        "format_version": "abcdresmc_checkpoint_download_receipt_v1",
        "status": "PASS_PUBLIC_CHECKPOINT_SHA_VERIFIED",
        "binding_sha256": EXPECTED_BINDING_SHA256,
        "repository": "NOC-Lab/AbCDR-ESMC",
        "revision": REVISION,
        "filename": str(final),
        "size_bytes": final.stat().st_size,
        "sha256": observed,
        "model_loaded": False,
        "embedding_materialized": False,
        "downstream_training_authorized": False,
        "adaptation_training_authorized": False,
        "old_exposed_test_access": False,
        "temporal_holdout_access": False,
    }
    receipt_path.write_bytes(canonical_bytes(receipt))
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(receipt_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
