#!/usr/bin/env python3
"""Compare continuous and DCP-resumed antibody training receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    continuous = _load(args.continuous / "receipt.json")
    resumed = _load(args.resumed / "receipt.json")
    continuous_state = _load(args.continuous / "trainer_state.json")
    resumed_state = _load(args.resumed / "trainer_state.json")
    equal = {
        "model": continuous["final_model_sha256"] == resumed["final_model_sha256"],
        "optimizer": continuous["final_optimizer_sha256"] == resumed["final_optimizer_sha256"],
        "trainer_state": continuous_state == resumed_state,
    }
    if not all(equal.values()):
        raise RuntimeError(f"DCP recovery differs: {equal}")
    payload = {
        "claim_boundary": "single-GPU engineering recovery only",
        "equal": equal,
        "format_version": "antibody_dcp_recovery_v1",
        "final_model_sha256": continuous["final_model_sha256"],
        "final_optimizer_sha256": continuous["final_optimizer_sha256"],
        "test_access": {"downstream_labels_used": False, "sabdab2_heldout_accessed": False},
    }
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    main()
